#include "unified_llm_w4a16_cached/helper.hpp"
#include "unified_llm_w4a16_cached/npuSetup.hpp"
#include "unified_llm_w4a16_cached/unified_llm_w4a16.hpp"
#include "unified_llm_w4a16_common/io_thread_pool.hpp"
#include "unified_llm_w4a16_common/moe_timing_stats.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <omp.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/uio.h>
#include <future>
#include <vector>

// Helper function to sample from logits
int64_t sample_token(const torch::Tensor &logits, float temperature, float top_p, int64_t top_k) {
    auto dtype = logits.dtype();
    auto device = logits.device();
    torch::Tensor probs = logits.to(torch::kFloat32);

    if (temperature != 1.0f) {
        probs = probs / temperature;
    }

    if (top_k > 0 && top_k < probs.size(0)) {
        auto topk_result = torch::topk(probs, top_k);
        torch::Tensor topk_values = std::get<0>(topk_result);
        torch::Tensor topk_indices = std::get<1>(topk_result);
        torch::Tensor mask = torch::full({probs.size(0)}, -std::numeric_limits<float>::infinity(),
                                         torch::TensorOptions().dtype(torch::kFloat32).device(device));
        mask.scatter_(0, topk_indices, topk_values);
        probs = mask;
    }

    if (top_p < 1.0f) {
        auto sorted_result = torch::sort(probs, -1, true);
        torch::Tensor sorted_probs = std::get<0>(sorted_result);
        torch::Tensor sorted_indices = std::get<1>(sorted_result);
        torch::Tensor cumulative_probs = torch::cumsum(torch::softmax(sorted_probs, -1), -1);

        auto sorted_indices_to_remove = cumulative_probs > top_p;
        // Avoid partial overlap: index_put_ from a view of the same tensor triggers assert_no_partial_overlap.
        auto shifted = sorted_indices_to_remove.index({torch::indexing::Slice(0, -1)}).clone();
        sorted_indices_to_remove.index_put_({torch::indexing::Slice(1, torch::indexing::None)}, shifted);
        sorted_indices_to_remove.index_put_({0}, false);

        torch::Tensor indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove).to(torch::kBool);
        probs.masked_fill_(indices_to_remove, -std::numeric_limits<float>::infinity());
    }

    probs = torch::softmax(probs, -1);
    torch::Tensor next_token = torch::multinomial(probs, 1);
    return next_token.item<int64_t>();
}

// Parse safetensors header and extract tensor metadata
std::map<std::string, SafetensorsTensorInfo> parse_safetensors_header_from_map(const char *data_ptr, uint64_t file_size,
                                                                               uint64_t &header_size) {
    std::map<std::string, SafetensorsTensorInfo> tensor_map;

    if (file_size < 8)
        return tensor_map;

    // Read header size (first 8 bytes)
    uint64_t header_size_le = *reinterpret_cast<const uint64_t *>(data_ptr);
    header_size = header_size_le;

    if (file_size < 8 + header_size)
        return tensor_map;

    // Parse JSON header
    std::string header(data_ptr + 8, header_size);

    // Simple JSON parsing for safetensors format
    size_t pos = 0;
    while (pos < header.size()) {
        // Find next tensor name
        size_t name_start = header.find('"', pos);
        if (name_start == std::string::npos)
            break;
        size_t name_end = header.find('"', name_start + 1);
        if (name_end == std::string::npos)
            break;
        std::string tensor_name = header.substr(name_start + 1, name_end - name_start - 1);

        // Find the object for this tensor
        size_t obj_start = header.find('{', name_end);
        if (obj_start == std::string::npos)
            break;
        size_t obj_end = obj_start + 1;
        int depth = 1;
        while (obj_end < header.size() && depth > 0) {
            if (header[obj_end] == '{')
                depth++;
            else if (header[obj_end] == '}')
                depth--;
            obj_end++;
        }

        std::string obj = header.substr(obj_start, obj_end - obj_start);
        SafetensorsTensorInfo info;

        // Parse dtype
        size_t dtype_pos = obj.find("\"dtype\"");
        if (dtype_pos != std::string::npos) {
            size_t dtype_start = obj.find('"', dtype_pos + 7);
            size_t dtype_end = obj.find('"', dtype_start + 1);
            if (dtype_end != std::string::npos) {
                info.dtype = obj.substr(dtype_start + 1, dtype_end - dtype_start - 1);
            }
        }

        // Parse shape
        size_t shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            size_t lb = obj.find('[', shape_pos);
            size_t rb = obj.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string shape_str = obj.substr(lb + 1, rb - lb - 1);
                std::istringstream iss(shape_str);
                int64_t val;
                while (iss >> val) {
                    info.shape.push_back(val);
                    if (iss.peek() == ',')
                        iss.ignore();
                }
            }
        }

        // Parse data_offsets
        size_t offset_pos = obj.find("\"data_offsets\"");
        if (offset_pos != std::string::npos) {
            size_t lb = obj.find('[', offset_pos);
            size_t rb = obj.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string offset_str = obj.substr(lb + 1, rb - lb - 1);
                std::istringstream iss(offset_str);
                uint64_t offset1, offset2;
                if (iss >> offset1 && iss.peek() == ',' && (iss.ignore(), iss >> offset2)) {
                    info.offset_begin = offset1;
                    info.offset_end = offset2;
                }
            }
        }

        if (!info.dtype.empty() && !info.shape.empty() && info.offset_end > info.offset_begin) {
            info.valid = true;
            tensor_map[tensor_name] = info;
        }

        pos = obj_end;
    }

    return tensor_map;
}

// Load a tensor from memory mapped pointer
torch::Tensor load_tensor_from_ptr(const char *data_ptr, const SafetensorsTensorInfo &info, uint64_t header_size) {
    // Calculate absolute offset in mmap
    uint64_t data_start = 8 + header_size;
    const char *tensor_data = data_ptr + data_start + info.offset_begin;
    uint64_t bytes = info.offset_end - info.offset_begin;

    // Determine dtype
    torch::ScalarType torch_dtype = torch::kFloat32;
    if (info.dtype == "BF16")
        torch_dtype = torch::kBFloat16;
    else if (info.dtype == "F16")
        torch_dtype = torch::kFloat16;
    else if (info.dtype == "F32")
        torch_dtype = torch::kFloat32;
    else if (info.dtype == "I64")
        torch_dtype = torch::kInt64;
    else if (info.dtype == "I32")
        torch_dtype = torch::kInt32;
    else if (info.dtype == "U8" || info.dtype == "UINT8")
        torch_dtype = torch::kUInt8;
    else {
        std::cerr << "Unsupported dtype: " << info.dtype << std::endl;
        return torch::Tensor();
    }

    // Create tensor from blob (no copy) - unsafe generally, but safe here because we manage mmap
    // Note: We deliberately do NOT clone here. The clone happens when we copy_ to the model parameter later.
    // This tensor is just a view into the file.
    torch::Tensor tensor = torch::from_blob((void *)tensor_data, info.shape, torch_dtype);
    return tensor;
}

// Helper to unpack AWQ-packed int32 tensor to 4-bit values (as uint8)
// Applies the {0, 4, 1, 5, 2, 6, 3, 7} permutation to transform ZigZag order to Contiguous order.
// Input: Tensor of int32 (ZigZag packed)
// Output: Tensor of uint8 with shape [..., 8] (Contiguous logical order)
torch::Tensor unpack_awq_zigzag_to_contiguous(torch::Tensor packed_int32) {
    auto device = packed_int32.device();
    std::vector<torch::Tensor> unpacked_parts;

    // AWQ permutation: [0, 4, 1, 5, 2, 6, 3, 7]
    // This maps output index k to the shift amount (permutation[k] * 4)
    // This ensures that unpacked_parts[k] corresponds to the k-th logical element.
    const int permutation[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    for (int k = 0; k < 8; ++k) {
        int shift_amount = permutation[k] * 4;
        // Use scalar operations to avoid tensor allocation
        torch::Tensor part;
        if (shift_amount > 0) {
            part = torch::bitwise_right_shift(packed_int32, shift_amount);
        } else {
            part = packed_int32;
        }
        part = torch::bitwise_and(part, 0x0F).to(torch::kUInt8);
        unpacked_parts.push_back(part);
    }

    // Stack along the last dimension
    return torch::stack(unpacked_parts, -1);
}

// Helper to unpack AWQ int32 qweight to 4-bit (stored as uint8)
// Input: [in_features, out_features/8] int32
// Output: [in_features, out_features] uint8 (Unpacked [In, Out])
torch::Tensor unpack_awq_qweight(torch::Tensor qweight) {
    auto device = qweight.device();
    // Transpose to [Out/8, In]
    qweight = qweight.t().contiguous();

    int64_t out_features_div_8 = qweight.size(0);
    int64_t in_features = qweight.size(1);
    int64_t out_features = out_features_div_8 * 8;

    // Unpack: [Out/8, In] -> [Out/8, In, 8]
    // This step converts ZigZag packing to Contiguous unpacking
    auto unpacked = unpack_awq_zigzag_to_contiguous(qweight);

    // Permute to [Out/8, 8, In]
    unpacked = unpacked.permute({0, 2, 1}).contiguous();

    // Flatten: [Out, In]
    unpacked = unpacked.view({out_features, in_features});

    // Return unpacked [In, Out] uint8 (Transpose here)
    return unpacked.t().contiguous().to(torch::kUInt8);
}

// Helper to unpack AWQ qzeros
// Input: [n_groups, out_features/8] int32
// Output: [n_groups, out_features] int8 (Unpacked [Groups, Out])
torch::Tensor unpack_awq_qzeros(torch::Tensor qzeros) {
    auto device = qzeros.device();
    // qzeros is [Groups, Out/8] (int32)

    // Unpack: [Groups, Out/8] -> [Groups, Out/8, 8]
    // Convert ZigZag to Contiguous
    auto unpacked = unpack_awq_zigzag_to_contiguous(qzeros);

    // Flatten last two dims: [Groups, Out]
    int64_t n_groups = qzeros.size(0);
    int64_t out_features = qzeros.size(1) * 8;
    unpacked = unpacked.view({n_groups, out_features});

    // Return [Groups, Out] int8 (No Transpose)
    return unpacked.contiguous().to(torch::kInt8);
}

// Load quantized weights from safetensors file with mmap
void UnifiedLLMW4A16Impl::load_quantized_weights_from_safetensors(const std::string &filename) {
    if (debug_verbosity >= 1) {
        std::cout << "Loading weights from safetensors (mmap): " << filename << std::endl;
    }

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::perror("open");
        return;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::perror("fstat");
        close(fd);
        return;
    }

    size_t file_size = sb.st_size;
    const char *map = (const char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd); // Can close fd after mmap

    if (map == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    // Prefetch pages
    if (madvise((void *)map, file_size, MADV_WILLNEED) != 0) {
        std::perror("madvise");
    }

    // Parse header
    uint64_t header_size = 0;
    auto tensor_map = parse_safetensors_header_from_map(map, file_size, header_size);

    if (tensor_map.empty()) {
        std::cerr << "Failed to parse safetensors header" << std::endl;
        munmap((void *)map, file_size);
        return;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Found " << tensor_map.size() << " tensors in safetensors file" << std::endl;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    size_t loaded = 0;
    size_t skipped = 0;

    // Helper loading lambda using mmap ptr
    auto load_tensor = [&](const SafetensorsTensorInfo &info) { return load_tensor_from_ptr(map, info, header_size); };

    // Load token embedding (not quantized)
    auto it_embed = tensor_map.find("model.embed_tokens.weight");
    if (it_embed != tensor_map.end() && it_embed->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_embed->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            token_embedding->weight.set_requires_grad(false);
            token_embedding->weight.copy_(loaded_tensor.to(token_embedding->weight.dtype()).to(token_embedding->weight.device()));
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: model.embed_tokens.weight  Shape: " << loaded_tensor.sizes()
                          << "  Dtype: " << it_embed->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: token_embedding.weight  Shape: " << token_embedding->weight.sizes()
                          << "  Dtype: " << token_embedding->weight.dtype() << std::endl;
            }
            if (debug_verbosity >= 1) {
                std::cout << "  -> Loaded directly from model.embed_tokens.weight" << std::endl;
            }
        }
    } else {
        std::cerr << "Warning: model.embed_tokens.weight not found!" << std::endl;
        skipped++;
    }

    // Load quantized linear layers
    // Optimize: Find which layers are present in this safetensors file first
    std::set<int64_t> present_layers;
    std::string prefix = "model.layers.";
    for (const auto &pair : tensor_map) {
        const std::string &name = pair.first;
        if (name.compare(0, prefix.length(), prefix) == 0) {
            size_t end_pos = name.find('.', prefix.length());
            if (end_pos != std::string::npos) {
                std::string layer_num_str = name.substr(prefix.length(), end_pos - prefix.length());
                try {
                    int64_t layer_idx = std::stoll(layer_num_str);
                    if (layer_idx >= 0 && layer_idx < num_hidden_layers_) {
                        present_layers.insert(layer_idx);
                    }
                } catch (...) {
                }
            }
        }
    }

    // Load quantized linear layers
    // Optimize: Find which layers are present in this safetensors file first
    auto load_linear_layer = [&](QuantizedLinear &layer, const std::string &base_name) {
        auto it_qweight = tensor_map.find(base_name + ".qweight");
        auto it_scales = tensor_map.find(base_name + ".scales");
        auto it_qzeros = tensor_map.find(base_name + ".qzeros");
        auto it_bias = tensor_map.find(base_name + ".bias");

        torch::Tensor qweight, scales, qzeros;

        if (it_qweight != tensor_map.end() && it_qweight->second.valid && it_scales != tensor_map.end() && it_scales->second.valid &&
            it_qzeros != tensor_map.end() && it_qzeros->second.valid) {
            qweight = load_tensor(it_qweight->second);
            scales = load_tensor(it_scales->second);
            qzeros = load_tensor(it_qzeros->second);
        } else {
            std::cerr << "Warning: Missing quantized weights for " << base_name << std::endl;
            skipped++;
            return;
        }

        if (qweight.defined() && scales.defined()) {
            auto device = layer->get_quantized_weights().device();

            std::string qweight_dtype = it_qweight->second.dtype;
            std::string scales_dtype = it_scales->second.dtype;
            std::string qzeros_dtype = it_qzeros->second.dtype;

            std::string qweight_name_str = base_name + ".qweight";
            std::string scales_name_str = base_name + ".scales";
            std::string qzeros_name_str = base_name + ".qzeros";

            int64_t in_feat = layer->in_features();
            int64_t out_feat = layer->out_features();
            if (!(qweight.size(0) == in_feat && qweight.size(1) == out_feat / 8)) {
                std::cerr << "Warning: Weight shape " << qweight.sizes() << " does not match AWQ layout In=" << in_feat
                          << " Out=" << out_feat << std::endl;
                skipped++;
                return;
            }

            if (debug_verbosity >= 1) {
                std::cout << "  -> Detected AWQ layout [In, Out/8], using AWQ unpack" << std::endl;
            }
            qweight = unpack_awq_qweight(qweight.to(device));

            scales = scales.to(torch::kBFloat16).to(device).contiguous();

            torch::Tensor qzeros_dev = qzeros.to(device);
            qzeros = unpack_awq_qzeros(qzeros_dev);

            if (debug_verbosity >= 1) {
                std::cout << base_name << std::endl;
                std::cout << "  -> qweight shape: " << qweight.sizes() << ", dtype: " << qweight.dtype() << std::endl;
                std::cout << "  -> scales shape: " << scales.sizes() << ", dtype: " << scales.dtype() << std::endl;
                std::cout << "  -> qzeros shape: " << qzeros.sizes() << ", dtype: " << qzeros.dtype() << std::endl;
            }

            layer->set_quantized_weights(qweight, scales, qzeros);
            loaded++;

            // Bias
            if (it_bias != tensor_map.end() && it_bias->second.valid) {
                torch::Tensor bias_tensor = load_tensor(it_bias->second);
                if (bias_tensor.defined()) {
                    for (auto &param : layer->named_parameters()) {
                        if (param.key() == "bias") {
                            param.value().set_requires_grad(false);
                            param.value().copy_(bias_tensor.to(param.value().dtype()).to(device));
                            if (debug_verbosity >= 1) {
                                std::cout << "  -> Loaded bias for " << base_name << std::endl;
                            }
                            break;
                        }
                    }
                }
            }

            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: " << qweight_name_str << "  Shape: " << qweight.sizes() << "  Dtype: " << qweight_dtype << " ("
                          << qweight.dtype() << ")" << std::endl;
                std::cout << "Safetensors: " << scales_name_str << "  Shape: " << scales.sizes() << "  Dtype: " << scales_dtype << " ("
                          << scales.dtype() << ")" << std::endl;
                std::cout << "Safetensors: " << qzeros_name_str << "  Shape: " << qzeros.sizes() << "  Dtype: " << qzeros_dtype << " ("
                          << qzeros.dtype() << ")" << std::endl;
                std::cout << "Model param: " << base_name << " (quantized)" << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        } else {
            std::cerr << "Warning: Failed to load (undefined tensors) for " << base_name << std::endl;
            skipped++;
        }
    };

    auto has_quantized = [&](const std::string &base_name) {
        auto it_qweight = tensor_map.find(base_name + ".qweight");
        auto it_scales = tensor_map.find(base_name + ".scales");
        auto it_qzeros = tensor_map.find(base_name + ".qzeros");
        return it_qweight != tensor_map.end() && it_qweight->second.valid && it_scales != tensor_map.end() && it_scales->second.valid &&
               it_qzeros != tensor_map.end() && it_qzeros->second.valid;
    };

    auto slice_out_dim = [&](torch::Tensor t, int64_t out_offset, int64_t out_features) {
        if (!t.defined()) {
            return t;
        }
        if (t.dim() == 1 && t.size(0) >= out_offset + out_features) {
            return t.slice(0, out_offset, out_offset + out_features);
        }
        if (t.dim() == 2) {
            if (t.size(0) >= out_offset + out_features) {
                return t.slice(0, out_offset, out_offset + out_features);
            }
            if (t.size(1) >= out_offset + out_features) {
                return t.slice(1, out_offset, out_offset + out_features);
            }
        }
        return t;
    };

    auto load_linear_layer_from_combined = [&](QuantizedLinear &layer, const std::string &base_name, int64_t expert_idx,
                                               int64_t out_offset) {
        auto it_qweight = tensor_map.find(base_name + ".qweight");
        auto it_scales = tensor_map.find(base_name + ".scales");
        auto it_qzeros = tensor_map.find(base_name + ".qzeros");

        if (it_qweight == tensor_map.end() || it_scales == tensor_map.end() || it_qzeros == tensor_map.end() || !it_qweight->second.valid ||
            !it_scales->second.valid || !it_qzeros->second.valid) {
            std::cerr << "Warning: Missing quantized weights for " << base_name << std::endl;
            skipped++;
            return;
        }

        torch::Tensor qweight = load_tensor(it_qweight->second);
        torch::Tensor scales = load_tensor(it_scales->second);
        torch::Tensor qzeros = load_tensor(it_qzeros->second);

        if (qweight.dim() == 3) {
            qweight = qweight.select(0, expert_idx);
        }
        if (scales.dim() == 3) {
            scales = scales.select(0, expert_idx);
        }
        if (qzeros.dim() == 3) {
            qzeros = qzeros.select(0, expert_idx);
        }

        int64_t in_feat = layer->in_features();
        int64_t out_feat = layer->out_features();
        int64_t start_block = out_offset / 8;
        int64_t end_block = (out_offset + out_feat) / 8;

        if (qweight.dim() != 2 || qweight.size(0) != in_feat || qweight.size(1) < end_block) {
            std::cerr << "Warning: Weight shape " << qweight.sizes() << " does not match AWQ layout for " << base_name << std::endl;
            skipped++;
            return;
        }

        qweight = qweight.slice(1, start_block, end_block).contiguous();
        scales = slice_out_dim(scales, out_offset, out_feat).contiguous();

        if (qzeros.dim() == 2 && qzeros.size(1) >= end_block) {
            qzeros = qzeros.slice(1, start_block, end_block).contiguous();
        } else {
            qzeros = qzeros.contiguous();
        }

        if (debug_verbosity >= 1) {
            std::cout << "  -> Detected AWQ layout [In, Out/8], using AWQ unpack (combined) for " << base_name << std::endl;
        }

        auto device = layer->get_quantized_weights().device();
        qweight = unpack_awq_qweight(qweight.to(device));
        scales = scales.to(torch::kBFloat16).to(device).contiguous();
        qzeros = unpack_awq_qzeros(qzeros.to(device));

        layer->set_quantized_weights(qweight, scales, qzeros);
        loaded++;
    };

    auto set_gate_up_from_separate = [&](QuantizedLinear &gate_up, QuantizedLinear &gate, QuantizedLinear &up) {
        auto q = torch::cat({gate->get_quantized_weights(), up->get_quantized_weights()}, 0).contiguous();
        auto s = torch::cat({gate->get_scales(), up->get_scales()}, 0).contiguous();
        auto z = torch::cat({gate->get_zeros(), up->get_zeros()}, 0).contiguous();
        gate_up->set_unpacked_params(q, s, z);
    };

    std::vector<int64_t> layers_vec(present_layers.begin(), present_layers.end());

#pragma omp parallel for
    for (size_t idx = 0; idx < layers_vec.size(); ++idx) {
        int64_t i = layers_vec[idx];
        std::string layer_prefix = "model.layers." + std::to_string(i);

        // Attention layers: q_proj, k_proj, v_proj, o_proj
        std::vector<std::pair<QuantizedLinear, std::string>> attn_layers = {{q_layers[i], "self_attn.q_proj"},
                                                                            {k_layers[i], "self_attn.k_proj"},
                                                                            {v_layers[i], "self_attn.v_proj"},
                                                                            {o_layers[i], "self_attn.o_proj"}};

        for (auto &[layer, suffix] : attn_layers) {
            load_linear_layer(layer, layer_prefix + "." + suffix);
        }

        if (arch_type_ == ArchitectureType::MIXTRAL) {
            for (int64_t e = 0; e < num_experts_; ++e) {
                std::string expert_prefix = layer_prefix + ".block_sparse_moe.experts." + std::to_string(e);
                if (has_quantized(expert_prefix + ".w1")) {
                    QuantizedLinear tmp_gate(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_gate");
                    QuantizedLinear tmp_up(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_up");
                    load_linear_layer(tmp_gate, expert_prefix + ".w1");
                    load_linear_layer(tmp_up, expert_prefix + ".w3");
                    set_gate_up_from_separate(moe_layers[i]->gate_up_experts[e], tmp_gate, tmp_up);
                    load_linear_layer(moe_layers[i]->down_experts[e], expert_prefix + ".w2");
                } else if (has_quantized(expert_prefix + ".gate_proj")) {
                    QuantizedLinear tmp_gate(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_gate");
                    QuantizedLinear tmp_up(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_up");
                    load_linear_layer(tmp_gate, expert_prefix + ".gate_proj");
                    load_linear_layer(tmp_up, expert_prefix + ".up_proj");
                    set_gate_up_from_separate(moe_layers[i]->gate_up_experts[e], tmp_gate, tmp_up);
                    load_linear_layer(moe_layers[i]->down_experts[e], expert_prefix + ".down_proj");
                } else if (has_quantized(layer_prefix + ".block_sparse_moe.experts.gate_up_proj")) {
                    load_linear_layer_from_combined(moe_layers[i]->gate_up_experts[e],
                                                    layer_prefix + ".block_sparse_moe.experts.gate_up_proj", e, 0);
                    load_linear_layer_from_combined(moe_layers[i]->down_experts[e], layer_prefix + ".block_sparse_moe.experts.down_proj", e,
                                                    0);
                } else {
                    std::cerr << "Warning: Missing Mixtral expert weights for " << expert_prefix << std::endl;
                    skipped++;
                }
            }
        } else if (arch_type_ == ArchitectureType::QWEN) {
            for (int64_t e = 0; e < num_experts_; ++e) {
                std::string expert_prefix = layer_prefix + ".mlp.experts." + std::to_string(e);
                QuantizedLinear tmp_gate(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_gate");
                QuantizedLinear tmp_up(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_up");
                load_linear_layer(tmp_gate, expert_prefix + ".gate_proj");
                load_linear_layer(tmp_up, expert_prefix + ".up_proj");
                set_gate_up_from_separate(moe_layers[i]->gate_up_experts[e], tmp_gate, tmp_up);
                load_linear_layer(moe_layers[i]->down_experts[e], expert_prefix + ".down_proj");
            }
        }

        // RMSNorm layers (not quantized)
        auto it_input_norm = tensor_map.find(layer_prefix + ".input_layernorm.weight");
        if (it_input_norm != tensor_map.end() && it_input_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_input_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                input_norms[i]->set_weight(loaded_tensor);
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "Safetensors: " << layer_prefix << ".input_layernorm.weight  Shape: " << loaded_tensor.sizes()
                              << "  Dtype: " << it_input_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                    std::cout << "Model param: input_norms[" << i << "].weight" << std::endl;
                    std::cout << "  -> Loaded" << std::endl;
                }
            }
        }

        auto it_post_norm = tensor_map.find(layer_prefix + ".post_attention_layernorm.weight");
        if (it_post_norm != tensor_map.end() && it_post_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_post_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                post_attn_norms[i]->set_weight(loaded_tensor);
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "Safetensors: " << layer_prefix << ".post_attention_layernorm.weight  Shape: " << loaded_tensor.sizes()
                              << "  Dtype: " << it_post_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                    std::cout << "Model param: post_attn_norms[" << i << "].weight" << std::endl;
                    std::cout << "  -> Loaded" << std::endl;
                }
            }
        }

        if (arch_type_ == ArchitectureType::QWEN) {
            auto it_q_norm = tensor_map.find(layer_prefix + ".self_attn.q_norm.weight");
            if (it_q_norm != tensor_map.end() && it_q_norm->second.valid) {
                torch::Tensor loaded_tensor = load_tensor(it_q_norm->second);
                if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                    q_norms[i]->set_weight(loaded_tensor);
                    loaded++;
                }
            }

            auto it_k_norm = tensor_map.find(layer_prefix + ".self_attn.k_norm.weight");
            if (it_k_norm != tensor_map.end() && it_k_norm->second.valid) {
                torch::Tensor loaded_tensor = load_tensor(it_k_norm->second);
                if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                    k_norms[i]->set_weight(loaded_tensor);
                    loaded++;
                }
            }
        }

        if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
            auto it_router = tensor_map.find(layer_prefix + ".block_sparse_moe.gate.weight");
            if (it_router == tensor_map.end() || !it_router->second.valid) {
                it_router = tensor_map.find(layer_prefix + ".block_sparse_moe.router.weight");
            }
            if ((it_router == tensor_map.end() || !it_router->second.valid) && arch_type_ == ArchitectureType::QWEN) {
                it_router = tensor_map.find(layer_prefix + ".mlp.gate.weight");
            }
            if (it_router != tensor_map.end() && it_router->second.valid) {
                torch::Tensor loaded_tensor = load_tensor(it_router->second);
                if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                    moe_layers[i]->router->weight.set_requires_grad(false);
                    moe_layers[i]->router->weight.copy_(
                        loaded_tensor.to(moe_layers[i]->router->weight.dtype()).to(moe_layers[i]->router->weight.device()));
                    loaded++;
                }
            }
        }
    }

    // Final norm
    auto it_final_norm = tensor_map.find("model.norm.weight");
    if (it_final_norm != tensor_map.end() && it_final_norm->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_final_norm->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            final_norm->set_weight(loaded_tensor);
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: model.norm.weight  Shape: " << loaded_tensor.sizes()
                          << "  Dtype: " << it_final_norm->second.dtype << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: final_norm.weight" << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        }
    }

    // LM head (unquantized)
    auto it_lm_weight = tensor_map.find("lm_head.weight");
    if (it_lm_weight != tensor_map.end() && it_lm_weight->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_lm_weight->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(loaded_tensor.to(lm_head->weight.dtype()).to(lm_head->weight.device()));
            loaded++;
            if (debug_verbosity >= 1) {
                std::cout << "Safetensors: lm_head.weight  Shape: " << loaded_tensor.sizes() << "  Dtype: " << it_lm_weight->second.dtype
                          << " (" << loaded_tensor.dtype() << ")" << std::endl;
                std::cout << "Model param: lm_head.weight  Shape: " << lm_head->weight.sizes() << "  Dtype: " << lm_head->weight.dtype()
                          << std::endl;
                std::cout << "  -> Loaded" << std::endl;
            }
        }
    } else {
        // Check if we should tie weights (Gemma/Qwen often tie weights)
        // Llama usually doesn't, but if safetensors is missing lm_head, it likely means tied.
        // We check if token_embedding has been loaded (it's loaded at the start).
        // Note: We don't strictly check arch_type_ here because if the file is missing lm_head,
        // it's the best guess anyway.

        // Check if token_embedding matches lm_head shape (vocab_size, hidden_size)
        if (token_embedding->weight.size(0) == lm_head->weight.size(0) && token_embedding->weight.size(1) == lm_head->weight.size(1)) {

            std::cout << "Model param: lm_head.weight" << std::endl;
            std::cout << "  -> Tied to token_embedding.weight" << std::endl;

            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(token_embedding->weight);
            loaded++;
        } else {
            std::cerr << "Warning: Missing lm_head.weight and could not tie to token_embedding" << std::endl;
            skipped++;
        }
    }

    std::cout << "Loaded " << loaded << " parameters, skipped " << skipped << std::endl;

    munmap((void *)map, file_size);
}

void UnifiedLLMW4A16Impl::load_non_quantized_weights_from_safetensors(const std::string &filename) {
    if (debug_verbosity >= 1) {
        std::cout << "Loading non-quantized weights from safetensors (mmap): " << filename << std::endl;
    }

    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::perror("open");
        return;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::perror("fstat");
        close(fd);
        return;
    }

    size_t file_size = sb.st_size;
    const char *map = (const char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (map == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    if (madvise((void *)map, file_size, MADV_WILLNEED) != 0) {
        std::perror("madvise");
    }

    uint64_t header_size = 0;
    auto tensor_map = parse_safetensors_header_from_map(map, file_size, header_size);
    if (tensor_map.empty()) {
        std::cerr << "Failed to parse safetensors header" << std::endl;
        munmap((void *)map, file_size);
        return;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    size_t loaded = 0;
    size_t skipped = 0;

    auto load_tensor = [&](const SafetensorsTensorInfo &info) { return load_tensor_from_ptr(map, info, header_size); };

    // Token embedding
    auto it_embed = tensor_map.find("model.embed_tokens.weight");
    if (it_embed != tensor_map.end() && it_embed->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_embed->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            token_embedding->weight.set_requires_grad(false);
            token_embedding->weight.copy_(loaded_tensor.to(token_embedding->weight.dtype()).to(token_embedding->weight.device()));
            loaded++;
        }
    } else {
        std::cerr << "Warning: model.embed_tokens.weight not found!" << std::endl;
        skipped++;
    }

    // Per-layer norms
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i);

        auto it_input_norm = tensor_map.find(layer_prefix + ".input_layernorm.weight");
        if (it_input_norm != tensor_map.end() && it_input_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_input_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                input_norms[i]->set_weight(loaded_tensor);
                loaded++;
            }
        }

        auto it_post_norm = tensor_map.find(layer_prefix + ".post_attention_layernorm.weight");
        if (it_post_norm != tensor_map.end() && it_post_norm->second.valid) {
            torch::Tensor loaded_tensor = load_tensor(it_post_norm->second);
            if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                post_attn_norms[i]->set_weight(loaded_tensor);
                loaded++;
            }
        }
    }

    if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
        for (int64_t i = 0; i < num_hidden_layers_; ++i) {
            std::string layer_prefix = "model.layers." + std::to_string(i);
            auto it_router = tensor_map.find(layer_prefix + ".block_sparse_moe.gate.weight");
            if (it_router == tensor_map.end() || !it_router->second.valid) {
                it_router = tensor_map.find(layer_prefix + ".block_sparse_moe.router.weight");
            }
            if ((it_router == tensor_map.end() || !it_router->second.valid) && arch_type_ == ArchitectureType::QWEN) {
                it_router = tensor_map.find(layer_prefix + ".mlp.gate.weight");
            }
            if (it_router != tensor_map.end() && it_router->second.valid) {
                torch::Tensor loaded_tensor = load_tensor(it_router->second);
                if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                    moe_layers[i]->router->weight.set_requires_grad(false);
                    moe_layers[i]->router->weight.copy_(
                        loaded_tensor.to(moe_layers[i]->router->weight.dtype()).to(moe_layers[i]->router->weight.device()));
                    loaded++;
                }
            }

            if (arch_type_ == ArchitectureType::QWEN) {
                auto it_q_norm = tensor_map.find(layer_prefix + ".self_attn.q_norm.weight");
                if (it_q_norm != tensor_map.end() && it_q_norm->second.valid) {
                    torch::Tensor loaded_tensor = load_tensor(it_q_norm->second);
                    if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                        q_norms[i]->set_weight(loaded_tensor);
                        loaded++;
                    }
                }

                auto it_k_norm = tensor_map.find(layer_prefix + ".self_attn.k_norm.weight");
                if (it_k_norm != tensor_map.end() && it_k_norm->second.valid) {
                    torch::Tensor loaded_tensor = load_tensor(it_k_norm->second);
                    if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
                        k_norms[i]->set_weight(loaded_tensor);
                        loaded++;
                    }
                }
            }
        }
    }

    // Quantized layer biases (q/k/v/o and MLP) are stored as regular tensors in some models (e.g., Qwen).
    // When using pre-saved quantized bins, load these biases here to match the full safetensors path.
    auto load_bias = [&](QuantizedLinear &layer, const std::string &base_name) {
        auto it_bias = tensor_map.find(base_name + ".bias");
        if (it_bias == tensor_map.end() || !it_bias->second.valid) {
            return;
        }

        torch::Tensor bias_tensor = load_tensor(it_bias->second);
        if (!bias_tensor.defined() || bias_tensor.numel() == 0) {
            return;
        }

        auto device = layer->get_quantized_weights().device();
        for (auto &param : layer->named_parameters()) {
            if (param.key() == "bias") {
                param.value().set_requires_grad(false);
                param.value().copy_(bias_tensor.to(param.value().dtype()).to(device));
                loaded++;
                if (debug_verbosity >= 1) {
                    std::cout << "  -> Loaded bias for " << base_name << std::endl;
                }
                break;
            }
        }
    };

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i);

        // Attention layers
        load_bias(q_layers[i], layer_prefix + ".self_attn.q_proj");
        load_bias(k_layers[i], layer_prefix + ".self_attn.k_proj");
        load_bias(v_layers[i], layer_prefix + ".self_attn.v_proj");
        load_bias(o_layers[i], layer_prefix + ".self_attn.o_proj");
    }

    // Final norm
    auto it_final_norm = tensor_map.find("model.norm.weight");
    if (it_final_norm != tensor_map.end() && it_final_norm->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_final_norm->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            final_norm->set_weight(loaded_tensor);
            loaded++;
        }
    }

    // LM head
    auto it_lm_weight = tensor_map.find("lm_head.weight");
    if (it_lm_weight != tensor_map.end() && it_lm_weight->second.valid) {
        torch::Tensor loaded_tensor = load_tensor(it_lm_weight->second);
        if (loaded_tensor.defined() && loaded_tensor.numel() > 0) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(loaded_tensor.to(lm_head->weight.dtype()).to(lm_head->weight.device()));
            loaded++;
        }
    } else {
        if (token_embedding->weight.size(0) == lm_head->weight.size(0) && token_embedding->weight.size(1) == lm_head->weight.size(1)) {
            lm_head->weight.set_requires_grad(false);
            lm_head->weight.copy_(token_embedding->weight);
            loaded++;
        } else {
            std::cerr << "Warning: Missing lm_head.weight and could not tie to token_embedding" << std::endl;
            skipped++;
        }
    }

    if (debug_verbosity >= 1) {
        std::cout << "Loaded " << loaded << " non-quantized parameters, skipped " << skipped << std::endl;
    }

    munmap((void *)map, file_size);
}

// Fast binary tensor loader using mmap + MADV_SEQUENTIAL.
// Avoids kernel-buffered ifstream overhead; the OS DMA-reads directly into
// the mapped pages which we then memcpy into a pinned CPU tensor for fast H2D.
static torch::Tensor read_bin_tensor(const std::string &path, torch::ScalarType dtype, const std::vector<int64_t> &shape) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Could not open file: " + path + " (" + strerror(errno) + ")");
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("fstat failed for: " + path);
    }
    size_t file_size = static_cast<size_t>(sb.st_size);

    // Allocate pinned (page-locked) CPU tensor so H2D DMA is ~2x faster.
    auto tensor = torch::empty(shape, torch::TensorOptions().dtype(dtype).device(torch::kCPU).pinned_memory(true));
    size_t expected_bytes = static_cast<size_t>(tensor.numel()) * tensor.element_size();

    if (file_size != expected_bytes) {
        close(fd);
        throw std::runtime_error("File size mismatch for " + path + " (expected " + std::to_string(expected_bytes) +
                                 ", got " + std::to_string(file_size) + ")");
    }

    void *map = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        throw std::runtime_error("mmap failed for: " + path);
    }
    // Hint to the kernel: read sequentially, prefetch aggressively.
    madvise(map, file_size, MADV_SEQUENTIAL);

    std::memcpy(tensor.data_ptr(), map, file_size);
    munmap(map, file_size);

    return tensor;
}

#include "unified_llm_w4a16_common/moe_expert_io.inl"

void UnifiedLLMW4A16Impl::load_quantized_weights_from_bins(const std::string &weights_dir,
                                                            const std::string &expert_weights_dir) {
    const std::string &moe_dir = expert_weights_dir.empty() ? weights_dir : expert_weights_dir;
    if (debug_verbosity >= 1) {
        std::cout << "Loading quantized weights from bins: " << weights_dir << std::endl;
    }

    this->eval();
    torch::NoGradGuard no_grad;

    auto load_layer = [&](QuantizedLinear &layer, const std::string &prefix) {
        try {
            int64_t out_features = layer->out_features();
            int64_t in_features = layer->in_features();
            int64_t packed_in = (in_features + 1) / 2;

            std::string q_path = weights_dir + "/" + prefix + ".qweight.bin";
            std::string s_path = weights_dir + "/" + prefix + ".scales.bin";
            std::string z_path = weights_dir + "/" + prefix + ".zeros.bin";

            auto q = read_bin_tensor(q_path, torch::kUInt8, {out_features, packed_in});

            size_t s_bytes = std::filesystem::file_size(s_path);
            int64_t s_numel = static_cast<int64_t>(s_bytes / 2); // bf16
            int64_t s_groups = s_numel / out_features;
            if (s_groups * out_features != s_numel) {
                throw std::runtime_error("Invalid scales shape for " + s_path);
            }
            std::vector<int64_t> s_shape =
                (s_groups <= 1) ? std::vector<int64_t>{out_features} : std::vector<int64_t>{out_features, s_groups};
            auto s = read_bin_tensor(s_path, torch::kBFloat16, s_shape);

            size_t z_bytes = std::filesystem::file_size(z_path);
            int64_t z_numel = static_cast<int64_t>(z_bytes);
            int64_t z_groups = z_numel / out_features;
            if (z_groups * out_features != z_numel) {
                throw std::runtime_error("Invalid zeros shape for " + z_path);
            }
            std::vector<int64_t> z_shape =
                (z_groups <= 1) ? std::vector<int64_t>{out_features} : std::vector<int64_t>{out_features, z_groups};
            auto z = read_bin_tensor(z_path, torch::kInt8, z_shape);

            layer->set_unpacked_params(q, s, z);
        } catch (const std::exception &e) {
            std::cerr << "Error loading layer " << prefix << ": " << e.what() << std::endl;
            throw;
        }
    };

    auto load_gate_up_from_bins = [&](QuantizedLinear &gate_up, const std::string &gate_prefix, const std::string &up_prefix) {
        QuantizedLinear tmp_gate(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_gate");
        QuantizedLinear tmp_up(hidden_size_, intermediate_size_, false, max_seq_len_, "tmp_moe_up");
        load_layer(tmp_gate, gate_prefix);
        load_layer(tmp_up, up_prefix);
        auto q = torch::cat({tmp_gate->get_quantized_weights(), tmp_up->get_quantized_weights()}, 0).contiguous();
        auto s = torch::cat({tmp_gate->get_scales(), tmp_up->get_scales()}, 0).contiguous();
        auto z = torch::cat({tmp_gate->get_zeros(), tmp_up->get_zeros()}, 0).contiguous();
        gate_up->set_unpacked_params(q, s, z);
    };

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        load_layer(q_layers[i], "layer_" + std::to_string(i) + "_q");
        load_layer(k_layers[i], "layer_" + std::to_string(i) + "_k");
        load_layer(v_layers[i], "layer_" + std::to_string(i) + "_v");
        load_layer(o_layers[i], "layer_" + std::to_string(i) + "_o");
        if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
             // For cached backend, we rely on on-demand loading.
             // For cached backend, we rely on on-demand loading.
             // We just set the directory for each MoE layer.
             for (int64_t lay = 0; lay < num_hidden_layers_; ++lay) {
                  moe_layers[lay]->set_weights_dir(moe_dir);
             }
        }
    }
}

void UnifiedLLMW4A16Impl::prewarm_experts(int64_t num_to_warm, bool verbose) {
    if (verbose || debug_verbosity >= 1) {
        std::cout << "Pre-warming " << num_to_warm << " experts for all layers..." << std::endl;
    }
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        if (moe_layers[i]) {
            moe_layers[i]->prewarm_experts(num_to_warm);
            if (verbose && (i + 1) % 4 == 0) {
                 std::cout << "Prewarmed experts for layer " << (i + 1) << "/" << num_hidden_layers_ << std::endl;
            }
        }
    }
}

void UnifiedLLMW4A16Impl::initialize_dummy_weights(int seed) {
    torch::NoGradGuard no_grad;
    torch::manual_seed(seed);
    if (debug_verbosity >= 1) {
        std::cout << "Initializing dummy weights (seed=" << seed << ")..." << std::endl;
    }

    auto device = token_embedding->weight.device();

    // Initialize embeddings and heads
    token_embedding->weight.uniform_(-0.1, 0.1);
    final_norm->set_weight(torch::ones({hidden_size_}).to(device));
    lm_head->weight.uniform_(-0.1, 0.1);

    for (int i = 0; i < num_hidden_layers_; ++i) {
        // Initialize norms
        auto layer_device = q_layers[i]->get_quantized_weights().device();
        input_norms[i]->set_weight(torch::ones({hidden_size_}).to(layer_device));
        post_attn_norms[i]->set_weight(torch::ones({hidden_size_}).to(layer_device));
        if (arch_type_ == ArchitectureType::QWEN) {
            q_norms[i]->set_weight(torch::ones({head_dim_}).to(layer_device));
            k_norms[i]->set_weight(torch::ones({head_dim_}).to(layer_device));
        }

        // Helper to init quantized linear
        auto init_layer = [&](QuantizedLinear &layer) {
            int64_t in_feat = layer->in_features();
            int64_t out_feat = layer->out_features();

            auto layer_device = layer->get_quantized_weights().device();
            // qweight: [In, Out] uint8 (values 0-15)
            auto qweight = torch::randint(0, 16, {in_feat, out_feat}, torch::TensorOptions().dtype(torch::kUInt8).device(layer_device));

            // scales: [Groups, Out] bf16
            int64_t groups = in_feat / groupsize_;
            if (groups < 1)
                groups = 1;

            auto scales = torch::rand({groups, out_feat}, torch::TensorOptions().dtype(torch::kBFloat16).device(layer_device));

            // qzeros: [Groups, Out] int8 (values 0-15, usually around 8)
            auto qzeros = torch::full({groups, out_feat}, 8, torch::TensorOptions().dtype(torch::kInt8).device(layer_device));

            layer->set_quantized_weights(qweight, scales, qzeros);
        };

        // Attention
        init_layer(q_layers[i]);
        init_layer(k_layers[i]);
        init_layer(v_layers[i]);
        init_layer(o_layers[i]);

        if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
            moe_layers[i]->router->weight.uniform_(-0.1, 0.1);
            size_t num_slots = moe_layers[i]->gate_up_experts.size();
            for (size_t e = 0; e < num_slots; ++e) {
                init_layer(moe_layers[i]->gate_up_experts[e]);
                init_layer(moe_layers[i]->down_experts[e]);
            }
            moe_layers[i]->set_weights_dir("DUMMY");
            moe_layers[i]->prefill_cache_for_testing();
        }
    }

    if (debug_verbosity >= 1) {
        std::cout << "Dummy weights initialized." << std::endl;
    }
}



void MixtureOfExpertsImpl::prefill_cache_for_testing() {
    for (size_t i = 0; i < expert_slots_indices.size(); ++i) {
        expert_slots_indices[i] = i; // Map slot i to expert i
    }
}

void MixtureOfExpertsImpl::prewarm_experts(int64_t num_to_warm) {
    if (num_to_warm > max_cached_experts_) {
        num_to_warm = max_cached_experts_;
    }
    if (num_to_warm > num_experts_) {
        num_to_warm = num_experts_;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Layer " << layer_idx_ << ": Pre-warming " << num_to_warm << " experts." << std::endl;
    }

    for (int64_t i = 0; i < num_to_warm; ++i) {
        // Load global expert i into slot i
        load_expert_weights(i, i, weights_dir_);
        expert_slots_indices[i] = i;
        
        // Ensure slot i is initialized properly in metadata
        ++access_clock_;
        slot_meta_[i].expert_id = i;
        slot_meta_[i].access_count = 1;
        slot_meta_[i].last_access = access_clock_;
        slot_meta_[i].clock_bit = 1;

        if (expert_cache_bitmask_.size() == num_experts_) {
            expert_cache_bitmask_[i] = 1;
        }
    }
}

void MixtureOfExpertsImpl::load_experts_weights_packed(const std::vector<std::pair<int64_t, int64_t>>& slots_and_experts,
                                                       const std::string& weights_dir, bool is_prefetch) {
#include "unified_llm_w4a16_common/moe_expert_load_packed.inl"
}

void MixtureOfExpertsImpl::load_expert_weights_packed(int64_t slot_idx, int64_t expert_idx,
                                                      const std::string& weights_dir) {
    load_experts_weights_packed({{slot_idx, expert_idx}}, weights_dir);
}

void MixtureOfExpertsImpl::load_experts_weights(const std::vector<std::pair<int64_t, int64_t>>& slots_and_experts, const std::string& weights_dir, bool is_prefetch) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (weights_dir.empty()) {
        throw std::runtime_error("Weights directory not set for MoE layer " + std::to_string(layer_idx_));
    }

    if (weights_dir == "DUMMY") {
        if (debug_verbosity >= 2)
            std::cout << "DUMMY load for " << slots_and_experts.size() << " experts." << std::endl;
        return;
    }

    // Auto-detect format on first real load: packed wins if layer_0_expert_0.bin exists.
    if (expert_format_ == ExpertFormat::UNKNOWN) {
        std::string probe = weights_dir + "/layer_" + std::to_string(layer_idx_) + "_expert_0.bin";
        expert_format_ = std::filesystem::exists(probe) ? ExpertFormat::PACKED : ExpertFormat::UNPACKED;
        if (debug_verbosity >= 1)
            std::cout << "Layer " << layer_idx_ << ": using "
                      << (expert_format_ == ExpertFormat::PACKED ? "packed" : "unpacked")
                      << " expert format." << std::endl;
    }

    if (expert_format_ == ExpertFormat::PACKED) {
        load_experts_weights_packed(slots_and_experts, weights_dir, is_prefetch);
        auto end_time = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
        total_expert_load_time_ms_ += ms;
        return;
    }

    for (const auto& se : slots_and_experts) {
        int64_t slot_idx = se.first;
        int64_t expert_idx = se.second;
        const std::string expert_prefix = "layer_" + std::to_string(layer_idx_) + "_expert_" + std::to_string(expert_idx);
        const std::string gate_prefix = expert_prefix + "_gate";
        const std::string up_prefix = expert_prefix + "_up";
        const std::string down_prefix = expert_prefix + "_down";

#include "unified_llm_w4a16_common/moe_expert_load_unpacked.inl"
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    
    total_expert_load_time_ms_ += ms;
}

void MixtureOfExpertsImpl::load_expert_weights(int64_t slot_idx, int64_t expert_idx, const std::string& weights_dir) {
    load_experts_weights({{slot_idx, expert_idx}}, weights_dir);
}
