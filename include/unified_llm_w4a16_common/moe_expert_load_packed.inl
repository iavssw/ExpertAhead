// Body of MixtureOfExpertsImpl::load_expert_weights_packed — include inside member function.
// Uses `this->` and unified_llm_w4a16_common::read_bin_tensor_pread is not used here.

{
    std::string path = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx) + ".bin";

    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd == -1) {
        // Fallback to normal open if O_DIRECT is not supported or if the file format is old/unaligned
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd == -1) {
        throw std::runtime_error("Cannot open packed expert: " + path + " (" + strerror(errno) + ")");
    }

    static constexpr uint32_t kExpkMagicLe = 0x4B505845u;  // "EXPK"
    static constexpr int kExpkNumTensors = 9;
    static constexpr size_t kExpkDescSize = 48;
    static constexpr size_t kExpkHeaderSize = 8 + kExpkNumTensors * kExpkDescSize;

    struct ExpkDesc {
        char name[32];
        uint64_t offset;
        uint64_t size;
    };

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

    const ExpkDesc& gate_qw = require("gate.qweight");
    const ExpkDesc& gate_sc = require("gate.scales");
    const ExpkDesc& gate_zr = require("gate.zeros");
    const ExpkDesc& up_qw = require("up.qweight");
    const ExpkDesc& up_sc = require("up.scales");
    const ExpkDesc& up_zr = require("up.zeros");
    const ExpkDesc& down_qw = require("down.qweight");
    const ExpkDesc& down_sc = require("down.scales");
    const ExpkDesc& down_zr = require("down.zeros");

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
    const int64_t gs_grps = static_cast<int64_t>(gate_sc.size / 2) / out_feat;
    const std::vector<int64_t> s_shape =
        (gs_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gs_grps};
    const int64_t gz_grps = static_cast<int64_t>(gate_zr.size) / out_feat;
    const std::vector<int64_t> z_shape =
        (gz_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gz_grps};

    auto dest_q = ensure_pinned_buffer(gate_up_q_pinned_, slot_idx, {out_feat * 2, packed_in}, torch::kUInt8);
    auto dest_s = ensure_pinned_buffer(
        gate_up_s_pinned_, slot_idx, {s_shape[0] * 2, s_shape.size() > 1 ? s_shape[1] : 1}, torch::kBFloat16);
    auto dest_z = ensure_pinned_buffer(
        gate_up_z_pinned_, slot_idx, {z_shape[0] * 2, z_shape.size() > 1 ? z_shape[1] : 1}, torch::kInt8);
    char* ptr_q = static_cast<char*>(dest_q.data_ptr());
    char* ptr_s = static_cast<char*>(dest_s.data_ptr());
    char* ptr_z = static_cast<char*>(dest_z.data_ptr());

    auto& down_layer = down_experts[slot_idx];
    const int64_t d_out_feat = down_layer->out_features();
    const int64_t d_packed_in = (down_layer->in_features() + 1) / 2;
    const int64_t ds_grps = static_cast<int64_t>(down_sc.size / 2) / d_out_feat;
    const std::vector<int64_t> ds_shape =
        (ds_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, ds_grps};
    const int64_t dz_grps = static_cast<int64_t>(down_zr.size) / d_out_feat;
    const std::vector<int64_t> dz_shape =
        (dz_grps <= 1) ? std::vector<int64_t>{d_out_feat} : std::vector<int64_t>{d_out_feat, dz_grps};

    auto dest_dq = ensure_pinned_buffer(down_q_pinned_, slot_idx, {d_out_feat, d_packed_in}, torch::kUInt8);
    auto dest_ds = ensure_pinned_buffer(down_s_pinned_, slot_idx, ds_shape, torch::kBFloat16);
    auto dest_dz = ensure_pinned_buffer(down_z_pinned_, slot_idx, dz_shape, torch::kInt8);
    char* ptr_dq = static_cast<char*>(dest_dq.data_ptr());
    char* ptr_ds = static_cast<char*>(dest_ds.data_ptr());
    char* ptr_dz = static_cast<char*>(dest_dz.data_ptr());

    auto pread_exact = [&path](int fd_, char* dest, size_t size, off_t offset) {
        size_t done = 0;
        while (done < size) {
            const ssize_t ret = pread(fd_, dest + done, size - done, offset + static_cast<off_t>(done));
            if (ret <= 0) {
                throw std::runtime_error("pread failed reading packed expert: " + path);
            }
            done += static_cast<size_t>(ret);
        }
    };

    unified_llm_w4a16_common::log_sequential_expert_io_once();
    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        pread_exact(fd, ptr_q, gate_qw.size, static_cast<off_t>(gate_qw.offset));
        pread_exact(fd, ptr_q + gate_qw.size, up_qw.size, static_cast<off_t>(up_qw.offset));
        pread_exact(fd, ptr_s, gate_sc.size, static_cast<off_t>(gate_sc.offset));
        pread_exact(fd, ptr_s + gate_sc.size, up_sc.size, static_cast<off_t>(up_sc.offset));
        pread_exact(fd, ptr_z, gate_zr.size, static_cast<off_t>(gate_zr.offset));
        pread_exact(fd, ptr_z + gate_zr.size, up_zr.size, static_cast<off_t>(up_zr.offset));
        pread_exact(fd, ptr_dq, down_qw.size, static_cast<off_t>(down_qw.offset));
        pread_exact(fd, ptr_ds, down_sc.size, static_cast<off_t>(down_sc.offset));
        pread_exact(fd, ptr_dz, down_zr.size, static_cast<off_t>(down_zr.offset));
    } else {
        std::vector<std::future<void>> futures;
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_q, gate_qw.size, static_cast<off_t>(gate_qw.offset)));
        futures.push_back(std::async(std::launch::async, pread_exact, fd, ptr_q + gate_qw.size, up_qw.size,
                                     static_cast<off_t>(up_qw.offset)));
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_s, gate_sc.size, static_cast<off_t>(gate_sc.offset)));
        futures.push_back(std::async(std::launch::async, pread_exact, fd, ptr_s + gate_sc.size, up_sc.size,
                                     static_cast<off_t>(up_sc.offset)));
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_z, gate_zr.size, static_cast<off_t>(gate_zr.offset)));
        futures.push_back(std::async(std::launch::async, pread_exact, fd, ptr_z + gate_zr.size, up_zr.size,
                                     static_cast<off_t>(up_zr.offset)));
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_dq, down_qw.size, static_cast<off_t>(down_qw.offset)));
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_ds, down_sc.size, static_cast<off_t>(down_sc.offset)));
        futures.push_back(
            std::async(std::launch::async, pread_exact, fd, ptr_dz, down_zr.size, static_cast<off_t>(down_zr.offset)));

        try {
            for (auto& f : futures) {
                f.get();
            }
        } catch (...) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
            throw;
        }
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);

    gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
    down_layer->set_unpacked_params(dest_dq, dest_ds, dest_dz);
}
