// Unpacked expert load body — include inside load_expert_weights() after format detection.
// Expects: expert_prefix, gate_prefix, up_prefix, down_prefix, slot_idx, weights_dir.

{
#ifndef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
    uint64_t load_id = 0;
#endif

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

  {
    const int64_t out_feat = intermediate_size_;
    const int64_t in_feat = hidden_size_;
    const int64_t packed_in = (in_feat + 1) / 2;

    const std::string gq = weights_dir + "/" + gate_prefix + ".qweight.bin";
    const std::string gs = weights_dir + "/" + gate_prefix + ".scales.bin";
    const std::string gz = weights_dir + "/" + gate_prefix + ".zeros.bin";
    const std::string uq = weights_dir + "/" + up_prefix + ".qweight.bin";
    const std::string us = weights_dir + "/" + up_prefix + ".scales.bin";
    const std::string uz = weights_dir + "/" + up_prefix + ".zeros.bin";

    const size_t gs_bytes = std::filesystem::file_size(gs);
    const int64_t gs_numel = static_cast<int64_t>(gs_bytes / 2);
    const int64_t gs_grps = gs_numel / out_feat;
    const std::vector<int64_t> s_shape =
        (gs_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gs_grps};
    const size_t gz_bytes = std::filesystem::file_size(gz);
    const int64_t gz_numel = static_cast<int64_t>(gz_bytes);
    const int64_t gz_grps = gz_numel / out_feat;
    const std::vector<int64_t> z_shape =
        (gz_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, gz_grps};

    auto dest_q = ensure_pinned_buffer(gate_up_q_pinned_, slot_idx, {out_feat * 2, packed_in}, torch::kUInt8);
    auto dest_s = ensure_pinned_buffer(
        gate_up_s_pinned_, slot_idx, {s_shape[0] * 2, s_shape.size() > 1 ? s_shape[1] : 1}, torch::kBFloat16);
    auto dest_z = ensure_pinned_buffer(
        gate_up_z_pinned_, slot_idx, {z_shape[0] * 2, z_shape.size() > 1 ? z_shape[1] : 1}, torch::kInt8);

    const size_t expected_q = static_cast<size_t>(out_feat * packed_in * sizeof(uint8_t));
    const size_t expected_s = static_cast<size_t>(s_shape[0] * (s_shape.size() > 1 ? s_shape[1] : 1) * sizeof(uint16_t));
    const size_t expected_z = static_cast<size_t>(z_shape[0] * (z_shape.size() > 1 ? z_shape[1] : 1) * sizeof(int8_t));

    char* ptr_q = static_cast<char*>(dest_q.data_ptr());
    char* ptr_s = static_cast<char*>(dest_s.data_ptr());
    char* ptr_z = static_cast<char*>(dest_z.data_ptr());

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
    auto pread_abortable = [this, slot_idx, load_id](const std::string& path, void* dest_ptr, size_t copy_size) {
        if (this->slot_load_id_[slot_idx].load(std::memory_order_relaxed) != load_id) return;
        unified_llm_w4a16_common::read_bin_tensor_pread(path, dest_ptr, copy_size);
    };
#else
    auto pread_abortable = [slot_idx, load_id](const std::string& path, void* dest_ptr, size_t copy_size) {
        unified_llm_w4a16_common::read_bin_tensor_pread(path, dest_ptr, copy_size);
    };
#endif

    unified_llm_w4a16_common::log_sequential_expert_io_once();
    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        pread_abortable(gq, ptr_q, expected_q);
        pread_abortable(uq, ptr_q + expected_q, expected_q);
        pread_abortable(gs, ptr_s, expected_s);
        pread_abortable(us, ptr_s + expected_s, expected_s);
        pread_abortable(gz, ptr_z, expected_z);
        pread_abortable(uz, ptr_z + expected_z, expected_z);
    } else {
        std::vector<std::future<void>> futures;
        futures.push_back(
            std::async(std::launch::async, pread_abortable, gq, ptr_q, expected_q));
        futures.push_back(std::async(std::launch::async, pread_abortable, uq,
                                     ptr_q + expected_q, expected_q));
        futures.push_back(
            std::async(std::launch::async, pread_abortable, gs, ptr_s, expected_s));
        futures.push_back(std::async(std::launch::async, pread_abortable, us,
                                     ptr_s + expected_s, expected_s));
        futures.push_back(
            std::async(std::launch::async, pread_abortable, gz, ptr_z, expected_z));
        futures.push_back(std::async(std::launch::async, pread_abortable, uz,
                                     ptr_z + expected_z, expected_z));

        for (auto& f : futures) {
            f.get();
        }
    }

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
    {
        std::lock_guard<std::mutex> lock(expert_slots_mutex_);
        if (slot_load_id_[slot_idx].load(std::memory_order_relaxed) == load_id) {
            gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
        }
    }
#else
    gate_up_experts[slot_idx]->set_unpacked_params(dest_q, dest_s, dest_z);
#endif
  }

  {
    auto& down_layer = down_experts[slot_idx];
    const int64_t out_feat = down_layer->out_features();
    const int64_t in_feat = down_layer->in_features();
    const int64_t packed_in = (in_feat + 1) / 2;

    const std::string dq = weights_dir + "/" + down_prefix + ".qweight.bin";
    const std::string ds = weights_dir + "/" + down_prefix + ".scales.bin";
    const std::string dz = weights_dir + "/" + down_prefix + ".zeros.bin";

    const size_t ds_bytes = std::filesystem::file_size(ds);
    const int64_t ds_numel = static_cast<int64_t>(ds_bytes / 2);
    const int64_t ds_grps = ds_numel / out_feat;
    const std::vector<int64_t> s_shape =
        (ds_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, ds_grps};
    const size_t dz_bytes = std::filesystem::file_size(dz);
    const int64_t dz_numel = static_cast<int64_t>(dz_bytes);
    const int64_t dz_grps = dz_numel / out_feat;
    const std::vector<int64_t> z_shape =
        (dz_grps <= 1) ? std::vector<int64_t>{out_feat} : std::vector<int64_t>{out_feat, dz_grps};

    auto dest_q = ensure_pinned_buffer(down_q_pinned_, slot_idx, {out_feat, packed_in}, torch::kUInt8);
    auto dest_s = ensure_pinned_buffer(down_s_pinned_, slot_idx, s_shape, torch::kBFloat16);
    auto dest_z = ensure_pinned_buffer(down_z_pinned_, slot_idx, z_shape, torch::kInt8);

    const size_t expected_q = static_cast<size_t>(out_feat * packed_in * sizeof(uint8_t));
    const size_t expected_s = static_cast<size_t>(s_shape[0] * (s_shape.size() > 1 ? s_shape[1] : 1) * sizeof(uint16_t));
    const size_t expected_z = static_cast<size_t>(z_shape[0] * (z_shape.size() > 1 ? z_shape[1] : 1) * sizeof(int8_t));

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
    auto pread_abortable_down = [this, slot_idx, load_id](const std::string& path, void* dest_ptr, size_t copy_size) {
        if (this->slot_load_id_[slot_idx].load(std::memory_order_relaxed) != load_id) return;
        unified_llm_w4a16_common::read_bin_tensor_pread(path, dest_ptr, copy_size);
    };
#else
    auto pread_abortable_down = [slot_idx, load_id](const std::string& path, void* dest_ptr, size_t copy_size) {
        unified_llm_w4a16_common::read_bin_tensor_pread(path, dest_ptr, copy_size);
    };
#endif

    if (unified_llm_w4a16_common::sequential_expert_io_loads()) {
        pread_abortable_down(dq, dest_q.data_ptr(), expected_q);
        pread_abortable_down(ds, dest_s.data_ptr(), expected_s);
        pread_abortable_down(dz, dest_z.data_ptr(), expected_z);
    } else {
        std::vector<std::future<void>> futures;
        futures.push_back(std::async(std::launch::async, pread_abortable_down, dq,
                                     dest_q.data_ptr(), expected_q));
        futures.push_back(std::async(std::launch::async, pread_abortable_down, ds,
                                     dest_s.data_ptr(), expected_s));
        futures.push_back(std::async(std::launch::async, pread_abortable_down, dz,
                                     dest_z.data_ptr(), expected_z));

        for (auto& f : futures) {
            f.get();
        }
    }

#ifdef HETEROPREDICT_SUPPORT_LOGICAL_ABORT
    {
        std::lock_guard<std::mutex> lock(expert_slots_mutex_);
        if (slot_load_id_[slot_idx].load(std::memory_order_relaxed) == load_id) {
            down_layer->set_unpacked_params(dest_q, dest_s, dest_z);
        }
    }
#else
    down_layer->set_unpacked_params(dest_q, dest_s, dest_z);
#endif
  }
}
