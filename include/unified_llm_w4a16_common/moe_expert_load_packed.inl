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

    struct ExpertLoadState {
        int64_t slot_idx;
        int64_t expert_idx;
        std::string path;
        int fd;
        ExpkDesc gate_qw, gate_sc, gate_zr;
        ExpkDesc up_qw, up_sc, up_zr;
        ExpkDesc down_qw, down_sc, down_zr;

        torch::Tensor dest_q, dest_s, dest_z;
        char* ptr_q; char* ptr_s; char* ptr_z;
        torch::Tensor dest_dq, dest_ds, dest_dz;
        char* ptr_dq; char* ptr_ds; char* ptr_dz;
    };

    std::vector<ExpertLoadState> states;
    states.reserve(slots_and_experts.size());

    for (const auto& se : slots_and_experts) {
        int64_t slot_idx = se.first;
        int64_t expert_idx = se.second;
        std::string path = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx) + ".bin";

        const int fd = unified_llm_w4a16_common::open_odirect_or_throw(path);

        alignas(512) char header_buf[512];
        if (pread(fd, header_buf, 512, 0) < static_cast<ssize_t>(kExpkHeaderSize)) {
            close(fd);
            throw std::runtime_error("Short header read for packed expert: " + path);
        }
        if (*reinterpret_cast<const uint32_t*>(header_buf) != kExpkMagicLe) {
            close(fd);
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
            if (!d) {
                close(fd);
                throw std::runtime_error(std::string("Missing tensor '") + name + "' in: " + path);
            }
            return *d;
        };

        ExpertLoadState state;
        state.slot_idx = slot_idx;
        state.expert_idx = expert_idx;
        state.path = path;
        state.fd = fd;
        state.gate_qw = require("gate.qweight");
        state.gate_sc = require("gate.scales");
        state.gate_zr = require("gate.zeros");
        state.up_qw = require("up.qweight");
        state.up_sc = require("up.scales");
        state.up_zr = require("up.zeros");
        state.down_qw = require("down.qweight");
        state.down_sc = require("down.scales");
        state.down_zr = require("down.zeros");

        unified_llm_w4a16_common::require_odirect_region(path + " gate.qweight", state.gate_qw.offset, state.gate_qw.size);
        unified_llm_w4a16_common::require_odirect_region(path + " gate.scales", state.gate_sc.offset, state.gate_sc.size);
        unified_llm_w4a16_common::require_odirect_region(path + " gate.zeros", state.gate_zr.offset, state.gate_zr.size);
        unified_llm_w4a16_common::require_odirect_region(path + " up.qweight", state.up_qw.offset, state.up_qw.size);
        unified_llm_w4a16_common::require_odirect_region(path + " up.scales", state.up_sc.offset, state.up_sc.size);
        unified_llm_w4a16_common::require_odirect_region(path + " up.zeros", state.up_zr.offset, state.up_zr.size);
        unified_llm_w4a16_common::require_odirect_region(path + " down.qweight", state.down_qw.offset, state.down_qw.size);
        unified_llm_w4a16_common::require_odirect_region(path + " down.scales", state.down_sc.offset, state.down_sc.size);
        unified_llm_w4a16_common::require_odirect_region(path + " down.zeros", state.down_zr.offset, state.down_zr.size);

        auto ensure_pinned_buffer = [&](std::vector<torch::Tensor>& bufs, int64_t slot, const std::vector<int64_t>& shape,
                                      torch::ScalarType dtype) {
            if (static_cast<size_t>(slot) >= bufs.size()) {
                bufs.resize(max_cached_experts_);
            }
            if (!bufs[slot].defined()) {
                bufs[slot] = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
            } else if (bufs[slot].sizes() != shape) {
                bufs[slot].resize_(shape);
            }
            return bufs[slot];
        };

        const int64_t out_feat = intermediate_size_;
        const int64_t packed_in = (hidden_size_ + 1) / 2;
        const int64_t gs_grps = static_cast<int64_t>(state.gate_sc.size / 2) / out_feat;
        const std::vector<int64_t> s_shape =
            (gs_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gs_grps};
        const int64_t gz_grps = static_cast<int64_t>(state.gate_zr.size) / out_feat;
        const std::vector<int64_t> z_shape =
            (gz_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gz_grps};

        state.dest_q = ensure_pinned_buffer(gate_up_q_pinned_, slot_idx, {out_feat * 2, packed_in}, torch::kUInt8);
        state.dest_s = ensure_pinned_buffer(
            gate_up_s_pinned_, slot_idx, {s_shape[0] * 2, s_shape.size() > 1 ? s_shape[1] : 1}, torch::kBFloat16);
        state.dest_z = ensure_pinned_buffer(
            gate_up_z_pinned_, slot_idx, {z_shape[0] * 2, z_shape.size() > 1 ? z_shape[1] : 1}, torch::kInt8);
        state.ptr_q = static_cast<char*>(state.dest_q.data_ptr());
        state.ptr_s = static_cast<char*>(state.dest_s.data_ptr());
        state.ptr_z = static_cast<char*>(state.dest_z.data_ptr());

        auto& down_layer = down_experts[slot_idx];
        const int64_t d_out_feat = down_layer->out_features();
        const int64_t d_packed_in = (down_layer->in_features() + 1) / 2;
        const int64_t ds_grps = static_cast<int64_t>(state.down_sc.size / 2) / d_out_feat;
        const std::vector<int64_t> ds_shape =
            (ds_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, ds_grps};
        const int64_t dz_grps = static_cast<int64_t>(state.down_zr.size) / d_out_feat;
        const std::vector<int64_t> dz_shape =
            (dz_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, dz_grps};

        state.dest_dq = ensure_pinned_buffer(down_q_pinned_, slot_idx, {d_out_feat, d_packed_in}, torch::kUInt8);
        state.dest_ds = ensure_pinned_buffer(down_s_pinned_, slot_idx, ds_shape, torch::kBFloat16);
        state.dest_dz = ensure_pinned_buffer(down_z_pinned_, slot_idx, dz_shape, torch::kInt8);
        state.ptr_dq = static_cast<char*>(state.dest_dq.data_ptr());
        state.ptr_ds = static_cast<char*>(state.dest_ds.data_ptr());
        state.ptr_dz = static_cast<char*>(state.dest_dz.data_ptr());

        states.push_back(std::move(state));
    }

    auto pread_exact = [](int fd_, char* dest, size_t size, off_t offset, const std::string& path) {
        unified_llm_w4a16_common::pread_odirect_region(fd_, dest, size, offset, path);
    };

    unified_llm_w4a16_common::log_odirect_expert_io_once();
    unified_llm_w4a16_common::log_sequential_expert_io_once();
    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        for (const auto& state : states) {
            pread_exact(state.fd, state.ptr_q, state.gate_qw.size, static_cast<off_t>(state.gate_qw.offset), state.path);
            pread_exact(state.fd, state.ptr_q + state.gate_qw.size, state.up_qw.size, static_cast<off_t>(state.up_qw.offset), state.path);
            pread_exact(state.fd, state.ptr_s, state.gate_sc.size, static_cast<off_t>(state.gate_sc.offset), state.path);
            pread_exact(state.fd, state.ptr_s + state.gate_sc.size, state.up_sc.size, static_cast<off_t>(state.up_sc.offset), state.path);
            pread_exact(state.fd, state.ptr_z, state.gate_zr.size, static_cast<off_t>(state.gate_zr.offset), state.path);
            pread_exact(state.fd, state.ptr_z + state.gate_zr.size, state.up_zr.size, static_cast<off_t>(state.up_zr.offset), state.path);
            pread_exact(state.fd, state.ptr_dq, state.down_qw.size, static_cast<off_t>(state.down_qw.offset), state.path);
            pread_exact(state.fd, state.ptr_ds, state.down_sc.size, static_cast<off_t>(state.down_sc.offset), state.path);
            pread_exact(state.fd, state.ptr_dz, state.down_zr.size, static_cast<off_t>(state.down_zr.offset), state.path);
        }
    } else {
        std::vector<std::future<void>> futures;
        for (const auto& state : states) {
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_q, state.gate_qw.size, static_cast<off_t>(state.gate_qw.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_q + state.gate_qw.size, state.up_qw.size, static_cast<off_t>(state.up_qw.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_s, state.gate_sc.size, static_cast<off_t>(state.gate_sc.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_s + state.gate_sc.size, state.up_sc.size, static_cast<off_t>(state.up_sc.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_z, state.gate_zr.size, static_cast<off_t>(state.gate_zr.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_z + state.gate_zr.size, state.up_zr.size, static_cast<off_t>(state.up_zr.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_dq, state.down_qw.size, static_cast<off_t>(state.down_qw.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_ds, state.down_sc.size, static_cast<off_t>(state.down_sc.offset), state.path));
            futures.push_back(std::async(std::launch::async, pread_exact, state.fd, state.ptr_dz, state.down_zr.size, static_cast<off_t>(state.down_zr.offset), state.path));
        }

        try {
            for (auto& f : futures) {
                f.get();
            }
        } catch (...) {
            for (const auto& state : states) {
                posix_fadvise(state.fd, 0, 0, POSIX_FADV_DONTNEED);
                close(state.fd);
            }
            throw;
        }
    }
    
    for (const auto& state : states) {
        posix_fadvise(state.fd, 0, 0, POSIX_FADV_DONTNEED);
        close(state.fd);
        gate_up_experts[state.slot_idx]->set_unpacked_params(state.dest_q, state.dest_s, state.dest_z);
        down_experts[state.slot_idx]->set_unpacked_params(state.dest_dq, state.dest_ds, state.dest_dz);
    }
}
