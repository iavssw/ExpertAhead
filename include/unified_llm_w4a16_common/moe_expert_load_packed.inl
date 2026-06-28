// Body of MixtureOfExpertsImpl::load_experts_weights_packed — include inside member function.
// Uses `this->` and unified_llm_w4a16_common::read_bin_tensor_pread is not used here.

{
    static constexpr uint32_t kExpkMagicLe = 0x4B505845u;  // "EXPK"
    static constexpr int kExpkNumTensors = 9;
    static constexpr size_t kExpkDescSize = 48;
    static constexpr size_t kExpkHeaderSize = 8 + kExpkNumTensors * kExpkDescSize;

    struct ExpkDesc {
        char name[32];
        uint64_t offset;
        uint64_t size;
    };

    unified_llm_w4a16_common::log_odirect_expert_io_once();
    unified_llm_w4a16_common::log_sequential_expert_io_once();

    int64_t max_slot_idx = -1;
    for (const auto& se : slots_and_experts) {
#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
        max_slot_idx = std::max(max_slot_idx, se.slot_idx);
#else
        max_slot_idx = std::max(max_slot_idx, se.first);
#endif
    }
    
    if (max_slot_idx >= 0) {
        if (static_cast<size_t>(max_slot_idx) >= gate_up_q_pinned_.size()) {
            gate_up_q_pinned_.resize(max_slot_idx + 1);
            gate_up_s_pinned_.resize(max_slot_idx + 1);
            gate_up_z_pinned_.resize(max_slot_idx + 1);
            down_q_pinned_.resize(max_slot_idx + 1);
            down_s_pinned_.resize(max_slot_idx + 1);
            down_z_pinned_.resize(max_slot_idx + 1);
        }
    }

    auto load_expert_worker = [this](int64_t slot_idx, int64_t expert_idx, uint64_t load_id, const std::string& path) {
#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
        if (this->slot_load_id_[slot_idx].load(std::memory_order_relaxed) != load_id) return;
#endif
        const int fd = unified_llm_w4a16_common::open_odirect_or_throw(path);
        
        struct Finalizer {
            int fd_;
            ~Finalizer() { if (fd_ != -1) { posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED); close(fd_); } }
        } finalizer{fd};

        alignas(512) char header_buf[512];
        if (pread(fd, header_buf, 512, 0) < static_cast<ssize_t>(kExpkHeaderSize)) {
            throw std::runtime_error("Short header read for packed expert: " + path);
        }
        if (*reinterpret_cast<const uint32_t*>(header_buf) != kExpkMagicLe) {
            throw std::runtime_error("Bad EXPK magic in: " + path);
        }

        const uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(header_buf + 4);
        const ExpkDesc* descs = reinterpret_cast<const ExpkDesc*>(header_buf + 8);

        auto expk_find = [&](const char* tensor_name) -> const ExpkDesc* {
            for (uint32_t i = 0; i < num_tensors; ++i) {
                if (std::strncmp(descs[i].name, tensor_name, 32) == 0) {
                    return &descs[i];
                }
            }
            return nullptr;
        };

        auto require = [&](const char* name) -> const ExpkDesc& {
            const ExpkDesc* d = expk_find(name);
            if (!d) throw std::runtime_error(std::string("Missing tensor '") + name + "' in: " + path);
            return *d;
        };

        ExpkDesc gate_qw = require("gate.qweight");
        ExpkDesc gate_sc = require("gate.scales");
        ExpkDesc gate_zr = require("gate.zeros");
        ExpkDesc up_qw = require("up.qweight");
        ExpkDesc up_sc = require("up.scales");
        ExpkDesc up_zr = require("up.zeros");
        ExpkDesc down_qw = require("down.qweight");
        ExpkDesc down_sc = require("down.scales");
        ExpkDesc down_zr = require("down.zeros");

        auto ensure_pinned_buffer = [&](std::vector<torch::Tensor>& bufs, int64_t slot, const std::vector<int64_t>& shape, torch::ScalarType dtype) {
            // Because bufs are sized to max_cached_experts_ at initialization, slot is always < bufs.size()
            if (!bufs[slot].defined()) {
                bufs[slot] = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
            } else if (bufs[slot].sizes() != shape) {
                bufs[slot].resize_(shape);
            }
            return bufs[slot];
        };

        const int64_t out_feat = intermediate_size_;
        const int64_t packed_in = (hidden_size_ + 1) / 2;
        const int64_t gs_grps = static_cast<int64_t>(gate_sc.size / 2) / out_feat;
        const std::vector<int64_t> s_shape = (gs_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gs_grps};
        const int64_t gz_grps = static_cast<int64_t>(gate_zr.size) / out_feat;
        const std::vector<int64_t> z_shape = (gz_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gz_grps};

        torch::Tensor dest_q = ensure_pinned_buffer(gate_up_q_pinned_, slot_idx, {out_feat * 2, packed_in}, torch::kUInt8);
        torch::Tensor dest_s = ensure_pinned_buffer(gate_up_s_pinned_, slot_idx, {s_shape[0] * 2, s_shape.size() > 1 ? s_shape[1] : 1}, torch::kBFloat16);
        torch::Tensor dest_z = ensure_pinned_buffer(gate_up_z_pinned_, slot_idx, {z_shape[0] * 2, z_shape.size() > 1 ? z_shape[1] : 1}, torch::kInt8);
        char* ptr_q = static_cast<char*>(dest_q.data_ptr());
        char* ptr_s = static_cast<char*>(dest_s.data_ptr());
        char* ptr_z = static_cast<char*>(dest_z.data_ptr());

        auto& down_layer = down_experts[slot_idx];
        const int64_t d_out_feat = down_layer->out_features();
        const int64_t d_packed_in = (down_layer->in_features() + 1) / 2;
        const int64_t ds_grps = static_cast<int64_t>(down_sc.size / 2) / d_out_feat;
        const std::vector<int64_t> ds_shape = (ds_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, ds_grps};
        const int64_t dz_grps = static_cast<int64_t>(down_zr.size) / d_out_feat;
        const std::vector<int64_t> dz_shape = (dz_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, dz_grps};

        torch::Tensor dest_dq = ensure_pinned_buffer(down_q_pinned_, slot_idx, {d_out_feat, d_packed_in}, torch::kUInt8);
        torch::Tensor dest_ds = ensure_pinned_buffer(down_s_pinned_, slot_idx, ds_shape, torch::kBFloat16);
        torch::Tensor dest_dz = ensure_pinned_buffer(down_z_pinned_, slot_idx, dz_shape, torch::kInt8);
        char* ptr_dq = static_cast<char*>(dest_dq.data_ptr());
        char* ptr_ds = static_cast<char*>(dest_ds.data_ptr());
        char* ptr_dz = static_cast<char*>(dest_dz.data_ptr());

        struct IovMapping {
            uint64_t offset;
            char* dest;
            size_t size;
        };
        std::vector<IovMapping> mappings = {
            {gate_qw.offset, ptr_q, gate_qw.size},
            {up_qw.offset, ptr_q + gate_qw.size, up_qw.size},
            {gate_sc.offset, ptr_s, gate_sc.size},
            {up_sc.offset, ptr_s + gate_sc.size, up_sc.size},
            {gate_zr.offset, ptr_z, gate_zr.size},
            {up_zr.offset, ptr_z + gate_zr.size, up_zr.size},
            {down_qw.offset, ptr_dq, down_qw.size},
            {down_sc.offset, ptr_ds, down_sc.size},
            {down_zr.offset, ptr_dz, down_zr.size}
        };

        std::sort(mappings.begin(), mappings.end(), [](const IovMapping& a, const IovMapping& b) {
            return a.offset < b.offset;
        });

        struct iovec iov[9];
        uint64_t current_offset = 512;
        for (int i = 0; i < 9; ++i) {
            if (mappings[i].offset != current_offset) {
                throw std::runtime_error("Tensors are not contiguous on disk in: " + path);
            }
            iov[i].iov_base = mappings[i].dest;
            iov[i].iov_len = mappings[i].size;
            current_offset += mappings[i].size;
        }

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
        if (this->slot_load_id_[slot_idx].load(std::memory_order_relaxed) != load_id) return;
#endif

        if (preadv(fd, iov, 9, 512) < 0) {
            throw std::runtime_error("preadv failed for " + path + ": " + strerror(errno));
        }

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
        std::lock_guard<std::mutex> lock(expert_slots_mutex_);
        if (slot_load_id_[slot_idx].load(std::memory_order_relaxed) == load_id) {
            gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
            down_experts[slot_idx]->set_unpacked_params(dest_dq, dest_ds, dest_dz);
        }
#else
        gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
        down_experts[slot_idx]->set_unpacked_params(dest_dq, dest_ds, dest_dz);
#endif
    };

    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        for (const auto& se : slots_and_experts) {
#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
            int64_t slot_idx = se.slot_idx;
            int64_t expert_idx = se.expert_idx;
            uint64_t load_id = se.load_id;
#else
            int64_t slot_idx = se.first;
            int64_t expert_idx = se.second;
            uint64_t load_id = 0;
#endif
            std::string path = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx) + ".bin";
            load_expert_worker(slot_idx, expert_idx, load_id, path);
        }
    } else {
        std::vector<std::future<void>> futures;
        auto& io_pool = unified_llm_w4a16_common::IOThreadPool::get_instance();
        
        for (const auto& se : slots_and_experts) {
#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
            int64_t slot_idx = se.slot_idx;
            int64_t expert_idx = se.expert_idx;
            uint64_t load_id = se.load_id;
#else
            int64_t slot_idx = se.first;
            int64_t expert_idx = se.second;
            uint64_t load_id = 0;
#endif
            std::string path = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx) + ".bin";
            futures.push_back(io_pool.enqueue(load_expert_worker, slot_idx, expert_idx, load_id, path));
        }

        for (auto& f : futures) {
            f.get();
        }
    }
}

