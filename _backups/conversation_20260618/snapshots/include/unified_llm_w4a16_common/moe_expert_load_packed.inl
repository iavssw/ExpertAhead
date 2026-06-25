// Body of MixtureOfExpertsImpl::load_expert_weights_packed — include inside member function.
// Uses `this->` and unified_llm_w4a16_common helpers from moe_expert_io.inl.

{
    std::string path = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx) + ".bin";

    static constexpr uint32_t kExpkMagicLe = 0x4B505845u;  // "EXPK"
    static constexpr size_t kOdirectAlign = 512;
    static constexpr size_t kExpkDescSize = 48;
    static constexpr size_t kExpkHeaderSize = 8 + 9 * kExpkDescSize;

    struct ExpkDesc {
        char name[32];
        uint64_t offset;
        uint64_t size;
    };

    struct stat sb {};
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open packed expert: " + path + " (" + strerror(errno) + ")");
    }
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("fstat failed for packed expert: " + path);
    }
    close(fd);

    const size_t file_size = static_cast<size_t>(sb.st_size);
    const size_t read_size = unified_llm_w4a16_common::align_up(file_size, kOdirectAlign);

    void* staging_raw = nullptr;
    if (posix_memalign(&staging_raw, kOdirectAlign, read_size) != 0) {
        throw std::runtime_error("posix_memalign failed for packed expert staging: " + path);
    }
    char* staging = static_cast<char*>(staging_raw);
    std::unique_ptr<void, decltype(&free)> staging_guard(staging_raw, free);

    fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    const bool used_odirect = (fd != -1);
    if (!used_odirect) {
        if (unified_llm_w4a16_common::strict_ssd_io()) {
            throw std::runtime_error("O_DIRECT open failed for packed expert: " + path + " (" +
                                     strerror(errno) + ")");
        }
        fd = open(path.c_str(), O_RDONLY);
    }
    if (fd == -1) {
        throw std::runtime_error("Cannot open packed expert: " + path + " (" + strerror(errno) + ")");
    }

    unified_llm_w4a16_common::log_strict_ssd_io_once();

  {
    auto read_exact = [&](int read_fd, char* dst, size_t nbytes, off_t offset) {
      size_t done = 0;
      while (done < nbytes) {
        const ssize_t ret = pread(read_fd, dst + done, nbytes - done, offset + static_cast<off_t>(done));
        if (ret < 0) {
          throw std::runtime_error("pread failed for packed expert: " + path + " (" + strerror(errno) + ")");
        }
        if (ret == 0) {
          throw std::runtime_error("pread EOF for packed expert: " + path + " (wanted " +
                                   std::to_string(nbytes) + " bytes at offset " + std::to_string(offset) +
                                   ", got " + std::to_string(done) + ")");
        }
        done += static_cast<size_t>(ret);
      }
    };

    if (used_odirect) {
      const size_t odirect_bytes = (file_size / kOdirectAlign) * kOdirectAlign;
      read_exact(fd, staging, odirect_bytes, 0);
      if (file_size > odirect_bytes) {
        const size_t tail = file_size - odirect_bytes;
        // EXPK files end unaligned (e.g. 440 bytes past last 512 boundary). O_DIRECT cannot
        // read fewer than a full block at EOF — use a short buffered pread for the tail only.
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        fd = -1;
        const int fd_tail = open(path.c_str(), O_RDONLY);
        if (fd_tail == -1) {
          throw std::runtime_error("Cannot open packed expert tail: " + path + " (" + strerror(errno) + ")");
        }
        read_exact(fd_tail, staging + odirect_bytes, tail, static_cast<off_t>(odirect_bytes));
        posix_fadvise(fd_tail, 0, 0, POSIX_FADV_DONTNEED);
        close(fd_tail);
      } else {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        fd = -1;
      }
    } else {
      read_exact(fd, staging, file_size, 0);
      close(fd);
      fd = -1;
    }
  }

    if (*reinterpret_cast<const uint32_t*>(staging) != kExpkMagicLe) {
        throw std::runtime_error("Bad EXPK magic in: " + path);
    }

    const uint32_t num_tensors = *reinterpret_cast<const uint32_t*>(staging + 4);
    const ExpkDesc* descs = reinterpret_cast<const ExpkDesc*>(staging + 8);

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

    struct CopyJob {
        char* dest;
        const ExpkDesc* desc;
    };

    const CopyJob jobs[] = {
        {ptr_q, &gate_qw},
        {ptr_q + gate_qw.size, &up_qw},
        {ptr_s, &gate_sc},
        {ptr_s + gate_sc.size, &up_sc},
        {ptr_z, &gate_zr},
        {ptr_z + gate_zr.size, &up_zr},
        {ptr_dq, &down_qw},
        {ptr_ds, &down_sc},
        {ptr_dz, &down_zr},
    };

    auto copy_job = [&](const CopyJob& job) {
        std::memcpy(job.dest, staging + job.desc->offset, job.desc->size);
    };

    unified_llm_w4a16_common::log_sequential_expert_io_once();
    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        for (const auto& job : jobs) {
            copy_job(job);
        }
    } else {
        std::vector<std::future<void>> futures;
        futures.reserve(sizeof(jobs) / sizeof(jobs[0]));
        for (const auto& job : jobs) {
            futures.push_back(std::async(std::launch::async, copy_job, job));
        }
        for (auto& f : futures) {
            f.get();
        }
    }

    gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
    down_layer->set_unpacked_params(dest_dq, dest_ds, dest_dz);
}
