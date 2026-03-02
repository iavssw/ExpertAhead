#include "unified_llm_w4a16_base/unified_llm_w4a16.hpp"
#include "hipkernels/embedding.hpp"
#include "hipkernels/flash_attn_decode.hpp"
#include "hipkernels/lm_head.hpp"
#include "hipkernels/rmsnorm.hpp"
#include "hipkernels/rope.hpp"
#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "unified_llm_w4a16_base/helper.hpp"
#include "unified_llm_w4a16_base/npuSetup.hpp"
#include <c10/hip/HIPFunctions.h>
#include <c10/hip/HIPStream.h>
#include <chrono>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>
#include <algorithm>

namespace {

struct GpuVramInfo {
    int index = -1;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

struct GpuSelectionInfo {
    std::vector<GpuVramInfo> ranked;
    std::vector<int> selected;
    bool used_fallback = false;
    std::string fallback_reason;
};

double bytes_to_gib(size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

std::string join_gpu_indices(const std::vector<int> &gpu_indices) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < gpu_indices.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << gpu_indices[i];
    }
    oss << "]";
    return oss.str();
}

GpuSelectionInfo select_gpus_by_free_vram(int requested_gpu_count, int available_gpus) {
    GpuSelectionInfo selection;
    if (requested_gpu_count <= 0 || available_gpus <= 0) {
        return selection;
    }

    int original_device = 0;
    bool has_original_device = (hipGetDevice(&original_device) == hipSuccess);

    for (int gpu_idx = 0; gpu_idx < available_gpus; ++gpu_idx) {
        hipError_t set_err = hipSetDevice(gpu_idx);
        if (set_err != hipSuccess) {
            selection.used_fallback = true;
            selection.fallback_reason = "hipSetDevice(" + std::to_string(gpu_idx) + ") failed: " + hipGetErrorString(set_err);
            break;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        hipError_t mem_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (mem_err != hipSuccess) {
            selection.used_fallback = true;
            selection.fallback_reason = "hipMemGetInfo failed on gpu " + std::to_string(gpu_idx) + ": " + hipGetErrorString(mem_err);
            break;
        }

        selection.ranked.push_back({gpu_idx, free_bytes, total_bytes});
    }

    if (has_original_device) {
        (void)hipSetDevice(original_device);
    }

    if (!selection.used_fallback && static_cast<int>(selection.ranked.size()) == available_gpus) {
        std::sort(selection.ranked.begin(), selection.ranked.end(), [](const GpuVramInfo &a, const GpuVramInfo &b) {
            if (a.free_bytes != b.free_bytes) {
                return a.free_bytes > b.free_bytes;
            }
            return a.index < b.index;
        });

        for (const auto &info : selection.ranked) {
            if (static_cast<int>(selection.selected.size()) >= requested_gpu_count) {
                break;
            }
            selection.selected.push_back(info.index);
        }
        return selection;
    }

    selection.used_fallback = true;
    selection.ranked.clear();
    selection.selected.clear();
    for (int gpu_idx = 0; gpu_idx < available_gpus && static_cast<int>(selection.selected.size()) < requested_gpu_count; ++gpu_idx) {
        selection.selected.push_back(gpu_idx);
    }
    return selection;
}

} // namespace

template <typename Func> void time_op(const std::string &name, Func func) {
    auto sync0 = hipDeviceSynchronize();
    if (sync0 != hipSuccess) {
        std::cerr << "HIP sync error before " << name << ": " << hipGetErrorString(sync0) << std::endl;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    func();
    auto sync1 = hipDeviceSynchronize();
    if (sync1 != hipSuccess) {
        std::cerr << "HIP sync error after " << name << ": " << hipGetErrorString(sync1) << std::endl;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    if (ms > 0.001) {
        std::cout << "Node: " << std::left << std::setw(20) << name << " | time = " << std::fixed << std::setprecision(3) << std::setw(8)
                  << ms << " ms" << std::endl;
    }
}

torch::Tensor repeat_kv(const torch::Tensor &x, int64_t n_rep) {
    if (n_rep == 1) {
        return x;
    }
    auto sizes = x.sizes();
    int64_t batch = sizes[0];
    int64_t num_kv_heads = sizes[1];
    int64_t seq_len = sizes[2];
    int64_t head_dim = sizes[3];

    // Assuming [batch, num_kv_heads, seq_len, head_dim] layout
    // We want to repeat the heads (dim 1)
    auto expanded = x.unsqueeze(2).expand({batch, num_kv_heads, n_rep, seq_len, head_dim});
    return expanded.reshape({batch, num_kv_heads * n_rep, seq_len, head_dim});
}

int64_t positive_mod(int64_t value, int64_t mod) {
    if (mod <= 0) {
        return 0;
    }
    int64_t r = value % mod;
    return (r < 0) ? (r + mod) : r;
}

void write_kv_ring(torch::Tensor cache_slice, const torch::Tensor &kv, int64_t start_pos, int64_t window_size) {
    if (window_size <= 0 || kv.numel() == 0) {
        return;
    }

    int64_t seq_len = kv.size(2);
    int64_t tokens_to_write = std::min<int64_t>(seq_len, window_size);
    int64_t src_start = seq_len - tokens_to_write;
    int64_t write_head = positive_mod(start_pos + src_start, window_size);

    auto ring_cache = cache_slice.narrow(2, 0, window_size);
    int64_t first_chunk = std::min<int64_t>(tokens_to_write, window_size - write_head);
    ring_cache.narrow(2, write_head, first_chunk).copy_(kv.narrow(2, src_start, first_chunk));

    int64_t remaining = tokens_to_write - first_chunk;
    if (remaining > 0) {
        ring_cache.narrow(2, 0, remaining).copy_(kv.narrow(2, src_start + first_chunk, remaining));
    }
}

torch::Tensor read_kv_window(const torch::Tensor &cache_slice, int64_t kv_len, int64_t oldest_pos, int64_t window_size) {
    auto ring_cache = cache_slice.narrow(2, 0, window_size);
    if (kv_len <= 0) {
        return ring_cache.narrow(2, 0, 0);
    }

    int64_t oldest_idx = positive_mod(oldest_pos, window_size);
    if (oldest_idx + kv_len <= window_size) {
        return ring_cache.narrow(2, oldest_idx, kv_len);
    }

    int64_t first_chunk = window_size - oldest_idx;
    int64_t second_chunk = kv_len - first_chunk;
    return torch::cat({ring_cache.narrow(2, oldest_idx, first_chunk), ring_cache.narrow(2, 0, second_chunk)}, 2);
}

// Tile sizes
constexpr int LARGE_TILE_SIZE_ROW = 128;
constexpr int LARGE_TILE_SIZE_COL = 64;
constexpr int SMALL_TILE_SIZE = 8;

// Helper to unpack int4 to int8
int8_t unpack_nibble(uint8_t packed, bool high) { return high ? (int8_t)((packed >> 4) & 0x0F) : (int8_t)(packed & 0x0F); }

// RMSNormImpl Implementation
RMSNormImpl::RMSNormImpl(int64_t dim, float eps) : eps_(eps) { weight_ = register_parameter("weight", torch::ones({dim})); }

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    if (x.device().is_cuda()) {
        auto output = torch::empty_like(x);
        launch_rmsnorm(output, x, weight_, eps_);
        return output;
    }

    auto rms = torch::sqrt(x.pow(2).mean(-1, true) + eps_);
    auto normalized = x / rms;

    return normalized * weight_.to(x.dtype());
}

void RMSNormImpl::set_weight(torch::Tensor weight) {
    weight_.set_requires_grad(false);
    weight_.copy_(weight.to(weight_.dtype()).to(weight_.device()));
}

void RMSNormImpl::forward_out(torch::Tensor output, torch::Tensor x) {
    if (x.device().is_cuda()) {
        launch_rmsnorm(output, x, weight_, eps_);
        return;
    }

    // Compute variance and normalize
    auto var = torch::mean(torch::square(x), -1, true);
    auto normed = x * torch::rsqrt(var + eps_);

    // Apply weight
    normed = normed * weight_.to(x.dtype());

    // Copy result to output buffer
    output.copy_(normed);
}

// Helper to get tile indices
std::tuple<torch::Tensor, int64_t, int64_t> get_tile_indices(int64_t rows, int64_t cols) {
    int64_t num_tiles_row = (rows + LARGE_TILE_SIZE_ROW - 1) / LARGE_TILE_SIZE_ROW;
    int64_t num_tiles_col = (cols + LARGE_TILE_SIZE_COL - 1) / LARGE_TILE_SIZE_COL;

    std::vector<int64_t> tile_indices;
    tile_indices.reserve(num_tiles_row * num_tiles_col);

    // The specific ordering: col % 8, then col, then row
    for (int col_mod = 0; col_mod < 8; col_mod++) {
        for (int c = col_mod; c < num_tiles_col; c += 8) {
            for (int r = 0; r < num_tiles_row; r++) {
                tile_indices.push_back(r * num_tiles_col + c);
            }
        }
    }

    return std::make_tuple(torch::tensor(tile_indices, torch::kLong), num_tiles_row, num_tiles_col);
}

// HipEmbeddingImpl Implementation
HipEmbeddingImpl::HipEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim, int64_t max_batch_size, int64_t max_seq_len) {
    weight = register_parameter("weight", torch::empty({num_embeddings, embedding_dim}, torch::kBFloat16));
    // Pre-allocate buffer [max_bsz, max_seq, hidden] (BF16)
    output_buffer = register_buffer(
        "output_buffer", torch::empty({max_batch_size, max_seq_len, embedding_dim}, torch::TensorOptions().dtype(torch::kBFloat16)));
}

torch::Tensor HipEmbeddingImpl::forward(torch::Tensor input) {
    if (use_hip_) {
        int64_t bsz = input.size(0);
        int64_t seq_len = input.size(1);
        int64_t total_tokens = bsz * seq_len;

        if (total_tokens == 1) {
            auto output = output_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

            hipkernels::launch_embedding_forward(weight, input, output, c10::hip::getCurrentHIPStream().stream());
            return output;
        }
    }
    return torch::nn::functional::embedding(input, weight);
}

// LinearMatmulImpl Implementation
LinearMatmulImpl::LinearMatmulImpl(int64_t in_features, int64_t out_features, bool bias) {
    weight = register_parameter("weight", torch::empty({out_features, in_features}, torch::kBFloat16));
    if (bias) {
        this->bias = register_parameter("bias", torch::empty({out_features}, torch::kBFloat16));
    } else {
        this->register_parameter("bias", torch::Tensor(), false);
    }
}

torch::Tensor LinearMatmulImpl::forward(torch::Tensor input) {
    auto res = torch::matmul(input, weight.t());
    if (bias.defined()) {
        res += bias;
    }
    return res;
}

// LmHeadLinearImpl Implementation
LmHeadLinearImpl::LmHeadLinearImpl(int64_t in_features, int64_t out_features, int64_t max_batch_size, int64_t max_seq_len) {
    // Initialize weight as a parameter [Out, In] with BF16
    weight = register_parameter("weight", torch::empty({out_features, in_features}, torch::kBFloat16));
    // Pre-allocate logits buffer [max_bsz, max_seq, vocab] using BF16
    // This will match the model dtype (BF16) when .to() is called.
    logits_buffer = register_buffer(
        "logits_buffer", torch::empty({max_batch_size, max_seq_len, out_features}, torch::TensorOptions().dtype(torch::kBFloat16)));
}

torch::Tensor LmHeadLinearImpl::forward(torch::Tensor input) {
    if (use_hip_) {
        int64_t bsz = input.size(0);
        int64_t seq_len = input.size(1);
        int64_t total_tokens = bsz * seq_len;

        if (total_tokens == 1) {
            int64_t vocab_size = weight.size(0);
            int64_t hidden_size = weight.size(1);

            auto logits = logits_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

            hipkernels::launch_lm_head_forward((hip_bfloat16 *)logits.data_ptr<at::BFloat16>(),
                                               (const hip_bfloat16 *)input.data_ptr<at::BFloat16>(),
                                               (const hip_bfloat16 *)weight.data_ptr<at::BFloat16>(), 1, hidden_size, vocab_size,
                                               c10::hip::getCurrentHIPStream().stream());
            return logits.to(torch::kFloat32);
        }
    }
    return torch::nn::functional::linear(input, weight).to(torch::kFloat32);
}

torch::Tensor QuantizedLinearImpl::dequantize_weights() {
    torch::NoGradGuard no_grad;
    auto device = quantized_weight_.device();

    const int64_t out_features = out_features_;
    const int64_t in_features = in_features_;
    const int64_t packed_cols = quantized_weight_.size(1);

    // Unpack 4-bit weights without creating temporary stacks.
    auto unpack_pairs = torch::empty({out_features, packed_cols, 2}, torch::dtype(torch::kUInt8).device(device));

    auto low_bits = unpack_pairs.select(-1, 0);
    auto high_bits = unpack_pairs.select(-1, 1);

    torch::bitwise_and_out(low_bits, quantized_weight_, 0x0F);
    torch::bitwise_right_shift_out(high_bits, quantized_weight_, 4);

    auto unpacked_uint8 = unpack_pairs.view({out_features, packed_cols * 2});
    if (unpacked_uint8.size(1) != in_features) {
        unpacked_uint8 = unpacked_uint8.slice(1, 0, in_features);
    }
    auto unpacked = unpacked_uint8.to(torch::kInt8);

    if (scale_.dim() == 2) {
        auto apply_group = [&](const torch::Tensor &scales, const torch::Tensor &zeros) {
            int64_t n_groups = scales.size(1);
            TORCH_CHECK(in_features % n_groups == 0, "Group size mismatch");
            int64_t group_size = in_features / n_groups;

            auto w_view = unpacked.view({out_features, n_groups, group_size});
            auto s_view = scales.view({out_features, n_groups, 1});
            auto z_view = zeros.view({out_features, n_groups, 1}); // Int8

            auto w_sub = w_view.sub(z_view);
            return w_sub.to(torch::kBFloat16).mul_(s_view).view({out_features, in_features});
        };

        return apply_group(scale_, zero_point_);
    } else {
        auto s_view = scale_.view({out_features, 1});
        auto z_view = zero_point_.view({out_features, 1}); // Int8
        // Perform subtraction in Int8, then convert to BFloat16 and apply scale
        return unpacked.sub(z_view).to(torch::kBFloat16).mul_(s_view);
    }
}

// QuantizedLinearImpl Implementation
QuantizedLinearImpl::QuantizedLinearImpl(int64_t in_features, int64_t out_features, bool bias, int64_t max_seq_len, std::string layer_type)
    : in_features_(in_features), out_features_(out_features), max_seq_len_(max_seq_len) {
    // Unpacked buffers only
    (void)layer_type;
    int64_t packed_size = (in_features + 1) / 2;
    quantized_weight_ = register_buffer("quantized_weight", torch::zeros({out_features, packed_size}, torch::kUInt8));
    scale_ = register_buffer("scale", torch::ones({out_features}, torch::kBFloat16));
    zero_point_ = register_buffer("zero_point", torch::zeros({out_features}, torch::kInt8));

    if (bias) {
        bias_ = register_parameter("bias", torch::zeros({out_features}, torch::kBFloat16));
    }
}

void QuantizedLinearImpl::forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type) {
    // we don't care about the future for now
    if (debug_verbosity >= 2) {
        std::cout << "Forward " << layer_type << " (Target: " << hw_target << ")" << std::endl;
        std::cout << "  input.device=" << input.device() << " output.device=" << output_buffer.device()
                  << " qweight.device=" << quantized_weight_.device() << " scale.device=" << scale_.device()
                  << " zeros.device=" << zero_point_.device() << std::endl;
        std::cout << "  input.shape=" << input.sizes() << " qweight.shape=" << quantized_weight_.sizes()
                  << " scale.shape=" << scale_.sizes() << " zeros.shape=" << zero_point_.sizes() << std::endl;
    }

    int64_t M = input.numel() / in_features_;
    int64_t group_size = in_features_;
    if (scale_.dim() == 2) {
        int64_t n_groups = scale_.size(1);
        if (n_groups > 0)
            group_size = in_features_ / n_groups;
    }
    auto input_2d = input.contiguous().view({-1, in_features_});
    auto output_2d = output_buffer.view({-1, out_features_});

    if (!input.is_cuda()) {
        // Fallback to dequantize + matmul for CPU
        auto dequant = dequantize_weights();
        auto result = torch::matmul(input_2d.to(torch::kBFloat16), dequant.t());
        output_2d.copy_(result);

        if (bias_.defined()) {
            if (output_buffer.size(-1) != bias_.size(-1)) {
                throw std::runtime_error("Bias size mismatch with output buffer.");
            }
            output_buffer.add_(bias_);
        }
        return;
    }

    // Compute the GEMV or GEMM on Fused Hip Kernels
    // This does not check input padding correctly for 128 tiles sizes
    if (M == 1) {
        if (debug_verbosity >= 2) {
            std::cout << "GPU Unpacked GEMV" << std::endl;
        }
        hipkernels::w4a16_gemv_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_, out_features_,
                                              group_size);
    } else {
        if (debug_verbosity >= 2) {
            std::cout << "GPU Unpacked GEMM" << std::endl;
        }
        hipkernels::w4a16_gemm_unpacked_fused(output_2d, input_2d, quantized_weight_, scale_, zero_point_, in_features_, out_features_,
                                              group_size);
    }

    if (bias_.defined()) {
        if (output_buffer.size(-1) != bias_.size(-1)) {
            throw std::runtime_error("Bias size mismatch with output buffer.");
        }
        output_buffer.add_(bias_);
    }
}

torch::Tensor QuantizedLinearImpl::forward(torch::Tensor input, std::string layer_type) {
    int64_t M = input.numel() / in_features_;
    int64_t group_size = in_features_;
    if (scale_.dim() == 2) {
        int64_t n_groups = scale_.size(1);
        if (n_groups > 0)
            group_size = in_features_ / n_groups;
    }

    if (debug_verbosity >= 2) {
        std::cout << "Forward (Allocating) " << layer_type << " (Target: " << hw_target << ")" << std::endl;
        std::cout << "  input.device=" << input.device() << " qweight.device=" << quantized_weight_.device()
                  << " scale.device=" << scale_.device() << " zeros.device=" << zero_point_.device() << std::endl;
        std::cout << "  input.shape=" << input.sizes() << " qweight.shape=" << quantized_weight_.sizes()
                  << " scale.shape=" << scale_.sizes() << " zeros.shape=" << zero_point_.sizes() << std::endl;
    }

    if (!input.is_cuda()) {
        // Fallback to dequantize + matmul for CPU
        auto dequant = dequantize_weights();
        auto input_2d = input.contiguous().view({-1, in_features_});
        auto result = torch::matmul(input_2d.to(torch::kBFloat16), dequant.t());

        if (bias_.defined()) {
            result.add_(bias_);
        }
        return result;
    }

    // HIP Path
    auto input_2d = input.contiguous().view({-1, in_features_});
    torch::Tensor output;
    if (M == 1) {
        if (debug_verbosity >= 2) {
            std::cout << "GPU Unpacked GEMV (Alloc)" << std::endl;
        }
        output = torch::empty({1, out_features_}, torch::TensorOptions().dtype(torch::kBFloat16).device(input.device()));
        hipkernels::w4a16_gemv_unpacked_fused(output, input_2d, quantized_weight_, scale_, zero_point_, in_features_, out_features_,
                                              group_size);
    } else {
        if (debug_verbosity >= 2) {
            std::cout << "GPU Unpacked GEMM (Alloc)" << std::endl;
        }

        int64_t rows = input_2d.size(0);
        int64_t padded_rows = rows;
        if (rows > 1) {
            const int64_t tile = 128;
            padded_rows = ((rows + tile - 1) / tile) * tile;
        }

        torch::Tensor input_padded = input_2d;
        if (padded_rows != rows) {
            input_padded = torch::zeros({padded_rows, in_features_}, input_2d.options());
            input_padded.narrow(0, 0, rows).copy_(input_2d);
        }

        output = hipkernels::w4a16_gemm_unpacked_alloc_and_compute(input_padded, quantized_weight_, scale_, zero_point_, in_features_,
                                                                   out_features_, group_size);
        if (padded_rows != rows) {
            output = output.narrow(0, 0, rows).contiguous();
        }
    }

    if (bias_.defined()) {
        output.add_(bias_);
    }

    return output;
}

void QuantizedLinearImpl::set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx) {

    auto device = qweight.device();

    // 1. Prepare unpacked int8 weights [out_features, in_features]
    // (Normalized to [In, Out] Column Major)

    torch::Tensor unpacked_qweight; // Declare unpacked_qweight here

    if (qweight.size(0) == in_features_ && qweight.size(1) == out_features_) {
        // [In, Out] -> Use directly
        // Prioritize this check so that square matrices (In==Out) are treated as [In, Out]
        if (debug_verbosity >= 2) {
            std::cout << "Using qweight INT8 COLUMN major [In, Out]" << std::endl;
        }
        unpacked_qweight = qweight.contiguous().to(torch::kInt8);
    } else if (qweight.size(0) == out_features_ && qweight.size(1) == in_features_) {
        // [Out, In] -> Transpose to [In, Out] (NPU Expects [In, Out])
        if (debug_verbosity >= 2) {
            std::cout << "Using qweight INT8 ROW major [Out, In] -> Transposing to [In, Out]" << std::endl;
        }
        unpacked_qweight = qweight.to(torch::kInt8).t().contiguous();
    } else if (qweight.size(0) == out_features_ && qweight.size(1) == (in_features_ + 1) / 2) {

        // [out, in/2] packed -> unpack
        int64_t out_features = qweight.size(0);
        int64_t in_features = qweight.size(1) * 2;

        auto q_view = qweight.view({out_features, in_features / 2, 1});
        auto w_low = torch::bitwise_and(q_view, 0x0F).to(torch::kInt8);
        auto w_high = torch::bitwise_right_shift(q_view, 4).to(torch::kInt8);

        std::vector<torch::Tensor> cat_tensors;
        cat_tensors.push_back(w_low);
        cat_tensors.push_back(w_high);
        // Result is [Out, In] -> Transpose to [In, Out]
        unpacked_qweight = torch::cat(cat_tensors, 2).view({out_features, in_features}).t().contiguous();
    } else {
        std::cerr << "Error: qweight shape " << qweight.sizes() << " not supported in set_quantized_weights" << std::endl;
        return;
    }

    // Manual / unpacked mode only
    if (debug_verbosity >= 2) {
        std::cout << "Setting weights (Unpacked Mode) - repacking to [Out, In/2]" << std::endl;
    }

    // Pack weights: [In, Out] -> [Out, In] -> [Out, In/2]
    auto w_out_in = unpacked_qweight.t().contiguous(); // [Out, In]
    auto w_view = w_out_in.view({out_features_, in_features_ / 2, 2});
    auto w_low = w_view.select(-1, 0).to(torch::kUInt8);
    auto w_high = w_view.select(-1, 1).to(torch::kUInt8);
    auto packed_w = torch::bitwise_or(torch::bitwise_and(w_low, 0x0F), torch::bitwise_left_shift(torch::bitwise_and(w_high, 0x0F), 4));

    quantized_weight_ = packed_w;

    // Handle Scales and Zeros
    if (scale.size(0) == out_features_ && scale.dim() == 1) {
        scale_ = scale.to(device).to(torch::kBFloat16);
        zero_point_ = zero_point.to(device).to(torch::kInt8);
    } else {
        int64_t num_scales = scale.numel();
        int64_t n_groups = num_scales / out_features_;

        if (scale.size(1) == out_features_) {
            scale_ = scale.t().contiguous().to(device).to(torch::kBFloat16);
            zero_point_ = zero_point.t().contiguous().to(device).to(torch::kInt8);
        } else {
            scale_ = scale.reshape({out_features_, n_groups}).to(device).to(torch::kBFloat16);
            zero_point_ = zero_point.reshape({out_features_, n_groups}).to(device).to(torch::kInt8);
        }
    }
}

void QuantizedLinearImpl::set_unpacked_params(torch::Tensor qweight_packed, torch::Tensor scale, torch::Tensor zero_point) {
    auto device = quantized_weight_.device();

    quantized_weight_ = qweight_packed.to(torch::kUInt8).contiguous().to(device);
    scale_ = scale.to(torch::kBFloat16).contiguous().to(device);
    zero_point_ = zero_point.to(torch::kInt8).contiguous().to(device);
}

// MixtureOfExpertsImpl Implementation
MixtureOfExpertsImpl::MixtureOfExpertsImpl(int64_t hidden_size, int64_t intermediate_size, int64_t num_experts, int64_t num_experts_per_tok,
                                           int64_t max_seq_len, bool use_softmax_before_topk, bool normalize_topk_prob)
    : hidden_size_(hidden_size), intermediate_size_(intermediate_size), num_experts_(num_experts),
      num_experts_per_tok_(num_experts_per_tok), use_softmax_before_topk_(use_softmax_before_topk),
      normalize_topk_prob_(normalize_topk_prob) {
    router = register_module("router", LinearMatmul(hidden_size_, num_experts_, false));

    gate_up_experts.reserve(num_experts_);
    down_experts.reserve(num_experts_);

    for (int64_t e = 0; e < num_experts_; ++e) {
        gate_up_experts.push_back(register_module(
            "gate_up_" + std::to_string(e), QuantizedLinear(hidden_size_, 2 * intermediate_size_, false, max_seq_len, "moe_gate_up")));
        down_experts.push_back(register_module("down_" + std::to_string(e),
                                               QuantizedLinear(intermediate_size_, hidden_size_, false, max_seq_len, "moe_down")));
    }
}

torch::Tensor MixtureOfExpertsImpl::forward_cpu(const torch::Tensor &x_flat, const torch::Tensor &topk_vals, const torch::Tensor &topk_idx,
                                                torch::Tensor &output) {
    for (int64_t t = 0; t < x_flat.size(0); ++t) {
        auto token_input = x_flat.narrow(0, t, 1);
        for (int64_t k = 0; k < num_experts_per_tok_; ++k) {
            int64_t e = topk_idx[t][k].item<int64_t>();
            auto weight = topk_vals[t][k].to(output.dtype());

            auto gate_up = gate_up_experts[e]->forward(token_input, "moe_gate_up");
            auto gate_buf = gate_up.narrow(1, 0, intermediate_size_);
            auto up_buf = gate_up.narrow(1, intermediate_size_, intermediate_size_);
            torch::silu_(gate_buf);
            gate_buf.mul_(up_buf);

            auto down_out = down_experts[e]->forward(gate_buf, "moe_down");
            down_out.mul_(weight);
            output.narrow(0, t, 1).add_(down_out);
        }
    }
    return output;
}

// Old Simple Version: Do not delete for reference
// torch::Tensor MixtureOfExpertsImpl::forward_generation(const torch::Tensor &x_flat, const torch::Tensor &topk_vals,
//                                                        const torch::Tensor &topk_idx, torch::Tensor &output) {
//     std::vector<torch::Tensor> down_buffers;
//     std::vector<torch::Tensor> expert_weights;
//     down_buffers.reserve(num_experts_per_tok_);
//     expert_weights.reserve(num_experts_per_tok_);

//     // Dispatch independent expert paths first.
//     for (int64_t k = 0; k < num_experts_per_tok_; ++k) {
//         int64_t e = topk_idx[0][k].item<int64_t>();
//         expert_weights.push_back(topk_vals[0][k].to(output.dtype()));
//         auto gate_up = gate_up_experts[e]->forward(x_flat, "moe_gate_up");
//         auto gate_buf = gate_up.narrow(1, 0, intermediate_size_);
//         auto up_buf = gate_up.narrow(1, intermediate_size_, intermediate_size_);
//         torch::silu_(gate_buf);
//         gate_buf.mul_(up_buf);
//         down_buffers.push_back(down_experts[e]->forward(gate_buf, "moe_down"));
//     }

//     // Separate reduction pass to aggregate expert outputs.
//     for (int64_t k = 0; k < num_experts_per_tok_; ++k) {
//         down_buffers[k].mul_(expert_weights[k]);
//         output.add_(down_buffers[k]);
//     }
//     return output;
// }

torch::Tensor MixtureOfExpertsImpl::forward_generation(const torch::Tensor &x_flat, const torch::Tensor &topk_vals,
                                                       const torch::Tensor &topk_idx, torch::Tensor &output) {
    auto opts = x_flat.options();
    const int64_t active_experts = num_experts_per_tok_;

    // Build expert pointer arrays once, then run both MoE projections batched in parallel.
    std::vector<int64_t> gate_up_qw_ptrs(active_experts), gate_up_s_ptrs(active_experts), gate_up_z_ptrs(active_experts);
    std::vector<int64_t> down_qw_ptrs(active_experts), down_s_ptrs(active_experts), down_z_ptrs(active_experts);
    std::vector<int64_t> expert_ids(active_experts);

    for (int64_t k = 0; k < active_experts; ++k) {
        int64_t e = topk_idx[0][k].item<int64_t>();
        expert_ids[k] = e;
        gate_up_qw_ptrs[k] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_quantized_weights().data_ptr<uint8_t>());
        gate_up_s_ptrs[k] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_scales().data_ptr<at::BFloat16>());
        gate_up_z_ptrs[k] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_zeros().data_ptr<int8_t>());
        down_qw_ptrs[k] = reinterpret_cast<int64_t>(down_experts[e]->get_quantized_weights().data_ptr<uint8_t>());
        down_s_ptrs[k] = reinterpret_cast<int64_t>(down_experts[e]->get_scales().data_ptr<at::BFloat16>());
        down_z_ptrs[k] = reinterpret_cast<int64_t>(down_experts[e]->get_zeros().data_ptr<int8_t>());
    }

    int64_t group_size = hidden_size_;
    if (gate_up_experts[expert_ids[0]]->get_scales().dim() == 2) {
        int64_t n_groups = gate_up_experts[expert_ids[0]]->get_scales().size(1);
        if (n_groups > 0)
            group_size = hidden_size_ / n_groups;
    }
    int64_t down_group_size = intermediate_size_;
    if (down_experts[expert_ids[0]]->get_scales().dim() == 2) {
        int64_t n_groups = down_experts[expert_ids[0]]->get_scales().size(1);
        if (n_groups > 0)
            down_group_size = intermediate_size_ / n_groups;
    }

    auto input_batched = x_flat.expand({active_experts, hidden_size_}).contiguous();
    auto gate_up_batched = torch::empty({active_experts, 2 * intermediate_size_}, opts);
    auto down_batched = torch::empty({active_experts, hidden_size_}, opts);

    hipkernels::w4a16_gemv_unpacked_fused_3d(gate_up_batched, input_batched, gate_up_qw_ptrs, gate_up_s_ptrs, gate_up_z_ptrs, hidden_size_,
                                             2 * intermediate_size_, group_size, active_experts);

    auto gate_buf = gate_up_batched.narrow(1, 0, intermediate_size_);
    auto up_buf = gate_up_batched.narrow(1, intermediate_size_, intermediate_size_);
    torch::silu_(gate_buf);
    gate_buf.mul_(up_buf);

    hipkernels::w4a16_gemv_unpacked_fused_3d(down_batched, gate_buf.contiguous(), down_qw_ptrs, down_s_ptrs, down_z_ptrs,
                                             intermediate_size_, hidden_size_, down_group_size, active_experts);

    auto weights = topk_vals[0].to(output.dtype()).view({active_experts, 1});
    down_batched.mul_(weights);
    output.add_(down_batched.sum(0, true));
    return output;
}

torch::Tensor MixtureOfExpertsImpl::forward_prefill(const torch::Tensor &x_flat, const torch::Tensor &topk_vals,
                                                    const torch::Tensor &topk_idx, torch::Tensor &output) {
    auto opts = x_flat.options();

    torch::Tensor expert_mask = torch::one_hot(topk_idx, num_experts_).to(torch::kBool);
    expert_mask = expert_mask.permute({2, 1, 0}); // [experts, top_k, tokens]
    torch::Tensor expert_hit = torch::nonzero(expert_mask.sum({1, 2}) > 0).squeeze();

    if (expert_hit.numel() == 0) {
        return output;
    }

    int64_t num_active = expert_hit.dim() == 0 ? 1 : expert_hit.size(0);
    const int64_t tile = 128;

    // Single pass: build dispatches, find max rows, gather pointers
    std::vector<int64_t> expert_ids;
    std::vector<torch::Tensor> token_indices;
    std::vector<torch::Tensor> top_k_positions;
    std::vector<int64_t> row_counts;
    expert_ids.reserve(num_active);
    token_indices.reserve(num_active);
    top_k_positions.reserve(num_active);
    row_counts.reserve(num_active);

    int64_t max_rows = 0;
    for (int64_t hit = 0; hit < num_active; ++hit) {
        int64_t e = (expert_hit.dim() == 0) ? expert_hit.item<int64_t>() : expert_hit[hit].item<int64_t>();
        auto where = torch::where(expert_mask[e]);
        if (where[1].numel() == 0)
            continue;

        expert_ids.push_back(e);
        top_k_positions.push_back(where[0]);
        token_indices.push_back(where[1]);
        int64_t rows = where[1].size(0);
        row_counts.push_back(rows);
        max_rows = std::max(max_rows, rows);
    }

    if (expert_ids.empty())
        return output;

    int64_t actual_num_experts = expert_ids.size();

    // Stable path for long prefill: avoid very large 3D MoE GEMM launches.
    const int64_t stable_prefill_row_limit = 0;
    if (max_rows > stable_prefill_row_limit) {
        for (size_t i = 0; i < actual_num_experts; ++i) {
            int64_t e = expert_ids[i];
            auto tok_idx = token_indices[i];
            auto expert_in = x_flat.index_select(0, tok_idx).contiguous();
            auto weights = topk_vals.index({tok_idx, top_k_positions[i]}).to(opts.dtype());

            auto gate_up = gate_up_experts[e]->forward(expert_in, "moe_gate_up");
            auto gate_buf = gate_up.narrow(1, 0, intermediate_size_);
            auto up_buf = gate_up.narrow(1, intermediate_size_, intermediate_size_);
            torch::silu_(gate_buf);
            gate_buf.mul_(up_buf);

            auto down_out = down_experts[e]->forward(gate_buf.contiguous(), "moe_down");
            down_out.mul_(weights.unsqueeze(-1));
            output.index_add_(0, tok_idx, down_out);
        }
        return output;
    }

    int64_t padded_M = ((max_rows + tile - 1) / tile) * tile;

    // Get group sizes once
    int64_t group_size = hidden_size_;
    if (gate_up_experts[expert_ids[0]]->get_scales().dim() == 2) {
        int64_t n_groups = gate_up_experts[expert_ids[0]]->get_scales().size(1);
        if (n_groups > 0)
            group_size = hidden_size_ / n_groups;
    }
    int64_t down_group_size = intermediate_size_;
    if (down_experts[expert_ids[0]]->get_scales().dim() == 2) {
        int64_t n_groups = down_experts[expert_ids[0]]->get_scales().size(1);
        if (n_groups > 0)
            down_group_size = intermediate_size_ / n_groups;
    }

    // Allocate batched input once. GEMM temporaries are chunked by rows to avoid
    // very large single launches in long-prefill scenarios.
    auto input_batched = torch::zeros({actual_num_experts, padded_M, hidden_size_}, opts);

    // Build pointer arrays and gather inputs in single loop
    std::vector<int64_t> gate_up_qw_ptrs(actual_num_experts), gate_up_s_ptrs(actual_num_experts), gate_up_z_ptrs(actual_num_experts);
    std::vector<int64_t> down_qw_ptrs(actual_num_experts), down_s_ptrs(actual_num_experts), down_z_ptrs(actual_num_experts);

    // For vectorized scatter: collect all token indices and weights
    std::vector<torch::Tensor> all_token_idx, all_weights;
    all_token_idx.reserve(actual_num_experts);
    all_weights.reserve(actual_num_experts);

    for (size_t i = 0; i < actual_num_experts; ++i) {
        int64_t e = expert_ids[i];
        int64_t rows = row_counts[i];

        input_batched[i].narrow(0, 0, rows).copy_(x_flat.index_select(0, token_indices[i]));

        gate_up_qw_ptrs[i] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_quantized_weights().data_ptr<uint8_t>());
        gate_up_s_ptrs[i] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_scales().data_ptr<at::BFloat16>());
        gate_up_z_ptrs[i] = reinterpret_cast<int64_t>(gate_up_experts[e]->get_zeros().data_ptr<int8_t>());
        down_qw_ptrs[i] = reinterpret_cast<int64_t>(down_experts[e]->get_quantized_weights().data_ptr<uint8_t>());
        down_s_ptrs[i] = reinterpret_cast<int64_t>(down_experts[e]->get_scales().data_ptr<at::BFloat16>());
        down_z_ptrs[i] = reinterpret_cast<int64_t>(down_experts[e]->get_zeros().data_ptr<int8_t>());

        all_token_idx.push_back(token_indices[i]);
        all_weights.push_back(topk_vals.index({token_indices[i], top_k_positions[i]}).to(opts.dtype()));
    }

    // Chunk rows to keep each 3D GEMM launch in a stable region for long prompts.
    const int64_t gemm_row_chunk = 4096; // must be a multiple of tile (128)
    for (int64_t row_start = 0; row_start < padded_M; row_start += gemm_row_chunk) {
        int64_t rows_this = std::min(gemm_row_chunk, padded_M - row_start);

        auto input_chunk = input_batched.narrow(1, row_start, rows_this).contiguous();
        auto gate_up_chunk = torch::empty({actual_num_experts, rows_this, intermediate_size_ * 2}, opts);
        auto down_chunk = torch::zeros({actual_num_experts, rows_this, hidden_size_}, opts);

        // Gate-up GEMM
        hipkernels::w4a16_gemm_unpacked_fused_3d(gate_up_chunk, input_chunk, gate_up_qw_ptrs, gate_up_s_ptrs, gate_up_z_ptrs, hidden_size_,
                                                 intermediate_size_ * 2, group_size, actual_num_experts);

        // Fused SiLU + mul
        auto gate_buf = gate_up_chunk.narrow(2, 0, intermediate_size_);
        auto up_buf = gate_up_chunk.narrow(2, intermediate_size_, intermediate_size_);
        torch::silu_(gate_buf);
        gate_buf.mul_(up_buf);

        // Down GEMM
        hipkernels::w4a16_gemm_unpacked_fused_3d(down_chunk, gate_buf.contiguous(), down_qw_ptrs, down_s_ptrs, down_z_ptrs,
                                                 intermediate_size_, hidden_size_, down_group_size, actual_num_experts);

        // Apply top-k weights + scatter only for real (unpadded) rows.
        for (size_t i = 0; i < actual_num_experts; ++i) {
            int64_t rows = row_counts[i];
            if (row_start >= rows) {
                continue;
            }
            int64_t local_rows = std::min(rows_this, rows - row_start);
            auto down_slice = down_chunk[i].narrow(0, 0, local_rows);
            auto local_weights = all_weights[i].narrow(0, row_start, local_rows);
            auto local_tokens = all_token_idx[i].narrow(0, row_start, local_rows);
            down_slice.mul_(local_weights.unsqueeze(-1));
            output.index_add_(0, local_tokens, down_slice);
        }
    }

    return output;
}

torch::Tensor MixtureOfExpertsImpl::forward(const torch::Tensor &x) {
    auto x_flat = x.view({-1, hidden_size_});
    auto opts = x.options();

    torch::Tensor router_out = router->forward(x_flat);

    torch::Tensor router_scores = router_out;
    if (use_softmax_before_topk_) {
        router_scores = torch::softmax(router_scores.to(torch::kFloat32), -1).to(router_out.dtype());
    }

    auto topk = router_scores.topk(num_experts_per_tok_, -1);
    torch::Tensor topk_vals = std::get<0>(topk);
    torch::Tensor topk_idx = std::get<1>(topk);
    if (use_softmax_before_topk_) {
        if (normalize_topk_prob_) {
            auto denom = topk_vals.sum(-1, true).clamp_min(1e-9);
            topk_vals = (topk_vals / denom).to(router_out.dtype());
        } else {
            topk_vals = topk_vals.to(router_out.dtype());
        }
    } else {
        topk_vals = torch::softmax(topk_vals.to(torch::kFloat32), -1).to(router_out.dtype());
    }

    auto output = torch::zeros({x_flat.size(0), hidden_size_}, opts);

    if (!x_flat.is_cuda()) {
        forward_cpu(x_flat, topk_vals, topk_idx, output);
        return output.view({x.size(0), x.size(1), hidden_size_});
    }

    if (x_flat.size(0) == 1) {
        forward_generation(x_flat, topk_vals, topk_idx, output);
    } else {
        forward_prefill(x_flat, topk_vals, topk_idx, output);
    }

    return output.view({x.size(0), x.size(1), hidden_size_});
}

// Prefetch removed
// void QuantizedLinearImpl::prefetch_cpu_weights() {}
UnifiedLLMW4A16Impl::UnifiedLLMW4A16Impl(ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                                         int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads,
                                         int64_t head_dim, float rms_norm_eps, float rope_theta, const NPUGlobalConfig &npu_config,
                                         int64_t max_seq_len, int64_t max_batch_size, int64_t groupsize, int64_t num_experts,
                                         int64_t num_experts_per_tok, torch::Device device)
    : arch_type_(arch_type), vocab_size_(vocab_size), hidden_size_(hidden_size), intermediate_size_(intermediate_size),
      num_hidden_layers_(num_hidden_layers), num_attention_heads_(num_attention_heads), num_key_value_heads_(num_key_value_heads),
      head_dim_(head_dim), rms_norm_eps_(rms_norm_eps), rope_theta_(rope_theta), max_seq_len_(max_seq_len), max_batch_size_(max_batch_size),
      groupsize_(groupsize), GQA_head_ratio_(num_attention_heads / num_key_value_heads), npu_config_(npu_config) {

    std::cout << "Initializing UnifiedLLMW4A16Impl_base" << std::endl;
    std::cout << "Using NPU config struct." << std::endl;
    configure_npu(npu_config_);
    warmup_ = warmup_enabled;

    // Set attention mode and HIP kernel usage based on heterogeneity
    bool is_cpu = (npu_config_.heterogeneity == "cpu");
    attention_mode_ = is_cpu ? 1 : ATTENTION_BACKEND;
    if (is_cpu) {
        device = torch::kCPU;
    }
    if (arch_type_ == ArchitectureType::MIXTRAL) {
        num_experts_ = (num_experts > 0) ? num_experts : 8;
        num_experts_per_tok_ = (num_experts_per_tok > 0) ? num_experts_per_tok : 2;
    } else if (arch_type_ == ArchitectureType::QWEN) {
        num_experts_ = (num_experts > 0) ? num_experts : 128;
        num_experts_per_tok_ = (num_experts_per_tok > 0) ? num_experts_per_tok : 8;
    } else {
        throw std::runtime_error("Unsupported architecture.");
    }

    sliding_window_enabled_ = (arch_type_ == ArchitectureType::MIXTRAL);
    sliding_window_size_ = std::min<int64_t>(4096, max_seq_len_);
    cache_filled_ = 0;
    if (sliding_window_size_ <= 0) {
        sliding_window_size_ = max_seq_len_;
    }
    prefill_chunk_size_ = sliding_window_enabled_ ? sliding_window_size_ : max_seq_len_;
    if (prefill_chunk_size_ <= 0) {
        prefill_chunk_size_ = 1;
    }
    if (debug_verbosity >= 1 && sliding_window_enabled_) {
        std::cout << "Sliding window enabled (size=" << sliding_window_size_ << ", prefill_chunk_size=" << prefill_chunk_size_ << ")"
                  << std::endl;
    }

    // GPU placement configuration (GPU mode only): auto-select by free VRAM.
    gpu_count_ = std::max(1, npu_config_.gpu_count);
    embedding_device_ = device;
    output_device_ = device;
    layer_devices_.assign(num_hidden_layers_, device);

    if (!is_cpu && device.is_cuda()) {
        int available_gpus = 0;
        if (hipGetDeviceCount(&available_gpus) != hipSuccess) {
            available_gpus = 0;
        }
        if (available_gpus <= 0) {
            throw std::runtime_error("Config requested GPU execution, but no GPU is available.");
        }
        if (available_gpus < gpu_count_) {
            throw std::runtime_error("Config requested gpu-count=" + std::to_string(gpu_count_) + " but only " +
                                     std::to_string(available_gpus) + " GPU(s) are available.");
        }

        auto selection = select_gpus_by_free_vram(gpu_count_, available_gpus);
        if (static_cast<int>(selection.selected.size()) < gpu_count_) {
            throw std::runtime_error("Failed to select enough GPUs for gpu-count=" + std::to_string(gpu_count_) + ".");
        }

        multi_gpu_enabled_ = (gpu_count_ > 1);
        embedding_device_ = torch::Device(torch::kCUDA, selection.selected.front());
        for (int64_t i = 0; i < num_hidden_layers_; ++i) {
            int selected_slot = static_cast<int>((i * gpu_count_) / num_hidden_layers_);
            int gpu_idx = selection.selected[selected_slot];
            layer_devices_[i] = torch::Device(torch::kCUDA, gpu_idx);
        }
        output_device_ = layer_devices_.back();
        device = embedding_device_;

        if (debug_verbosity >= 1) {
            if (selection.used_fallback) {
                std::cout << "[GPU SELECT] VRAM query failed, using fallback gpu ordering 0..N-1";
                if (!selection.fallback_reason.empty()) {
                    std::cout << " (" << selection.fallback_reason << ")";
                }
                std::cout << std::endl;
            } else {
                std::cout << "[GPU SELECT] Free VRAM ranking: ";
                for (size_t i = 0; i < selection.ranked.size(); ++i) {
                    if (i > 0) {
                        std::cout << ", ";
                    }
                    const auto &info = selection.ranked[i];
                    std::ostringstream vram_ss;
                    vram_ss << std::fixed << std::setprecision(2) << bytes_to_gib(info.free_bytes) << "/" << bytes_to_gib(info.total_bytes)
                            << " GiB";
                    std::cout << "gpu" << info.index << "=" << vram_ss.str();
                }
                std::cout << std::endl;
            }

            std::cout << "[GPU SELECT] Selected GPUs for gpu-count=" << gpu_count_ << ": " << join_gpu_indices(selection.selected)
                      << std::endl;

            if (multi_gpu_enabled_) {
                std::ostringstream layer_map_ss;
                layer_map_ss << "[GPU MAP] Layer ranges: ";
                bool first = true;
                for (int slot = 0; slot < gpu_count_; ++slot) {
                    const int64_t layer_start = (slot * num_hidden_layers_) / gpu_count_;
                    const int64_t layer_end = (((slot + 1) * num_hidden_layers_) / gpu_count_) - 1;
                    if (layer_start > layer_end) {
                        continue;
                    }
                    if (!first) {
                        layer_map_ss << "; ";
                    }
                    first = false;
                    layer_map_ss << "gpu" << selection.selected[slot] << ": layers " << layer_start << "-" << layer_end;
                }
                std::cout << layer_map_ss.str() << std::endl;
            } else {
                std::cout << "[GPU MAP] Single-GPU placement on gpu" << static_cast<int>(embedding_device_.index()) << std::endl;
            }
        }
    } else {
        gpu_count_ = 1;
    }

    // Token embedding - initialize on device
    token_embedding = register_module("token_embedding", HipEmbedding(vocab_size_, hidden_size_, max_batch_size_, max_seq_len_));
    token_embedding->set_use_hip(!is_cpu);
    token_embedding->to(embedding_device_);
    token_embedding->to(torch::kBFloat16);

    bool use_qkv_bias = false;

    // Initialize quantized layers for each transformer block
    // Initialize quantized layers for each transformer block
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        // Attention layers (quantized)
        q_layers.push_back(register_module(
            "q_" + std::to_string(i), QuantizedLinear(hidden_size_, num_attention_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "q")));
        k_layers.push_back(register_module(
            "k_" + std::to_string(i), QuantizedLinear(hidden_size_, num_key_value_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "k")));
        v_layers.push_back(register_module(
            "v_" + std::to_string(i), QuantizedLinear(hidden_size_, num_key_value_heads_ * head_dim_, use_qkv_bias, max_seq_len_, "v")));
        o_layers.push_back(register_module("o_" + std::to_string(i),
                                           QuantizedLinear(num_attention_heads_ * head_dim_, hidden_size_, false, max_seq_len_, "o")));

        bool use_qwen_router = (arch_type_ == ArchitectureType::QWEN);
        if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
            moe_layers.push_back(register_module("moe_" + std::to_string(i),
                                                 MixtureOfExperts(hidden_size_, intermediate_size_, num_experts_, num_experts_per_tok_,
                                                                  max_seq_len_, use_qwen_router, use_qwen_router)));
        }
        if (arch_type_ == ArchitectureType::QWEN) {
            q_norms.push_back(register_module("q_norm_" + std::to_string(i), RMSNorm(head_dim_, rms_norm_eps_)));
            k_norms.push_back(register_module("k_norm_" + std::to_string(i), RMSNorm(head_dim_, rms_norm_eps_)));
        }

        // Normalization layers (not quantized)
        input_norms.push_back(register_module("input_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_)));
        post_attn_norms.push_back(register_module("post_attn_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_)));

        // KV caches - initialize on device with bf16
        caches_k.push_back(register_buffer("cache_k_" + std::to_string(i),
                                           torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                        torch::TensorOptions().device(layer_devices_[i]).dtype(torch::kBFloat16))));
        caches_v.push_back(register_buffer("cache_v_" + std::to_string(i),
                                           torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                        torch::TensorOptions().device(layer_devices_[i]).dtype(torch::kBFloat16))));
    }

    // Final norm and output head
    final_norm = register_module("final_norm", RMSNorm(hidden_size_, rms_norm_eps_));
    final_norm->to(output_device_);
    final_norm->to(torch::kBFloat16);

    lm_head = register_module("lm_head", LmHeadLinear(hidden_size_, vocab_size_, max_batch_size_, max_seq_len_));
    lm_head->set_use_hip(!is_cpu);

    // Move all layers to device
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        auto layer_device = layer_devices_[i];
        q_layers[i]->to(layer_device);
        k_layers[i]->to(layer_device);
        v_layers[i]->to(layer_device);
        o_layers[i]->to(layer_device);
        if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
            moe_layers[i]->to(layer_device);
            moe_layers[i]->router->to(torch::kBFloat16);
        }
        if (arch_type_ == ArchitectureType::QWEN) {
            q_norms[i]->to(layer_device);
            q_norms[i]->to(torch::kBFloat16);
            k_norms[i]->to(layer_device);
            k_norms[i]->to(torch::kBFloat16);
        }
        input_norms[i]->to(layer_device);
        input_norms[i]->to(torch::kBFloat16);
        post_attn_norms[i]->to(layer_device);
        post_attn_norms[i]->to(torch::kBFloat16);
    }
    lm_head->to(output_device_);
    lm_head->to(torch::kBFloat16);

    // Register scratch buffers

    x_buffer = register_buffer("x_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                        torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    gate_buffer = register_buffer("gate_buffer", torch::zeros({max_batch_size_, max_seq_len_, intermediate_size_},
                                                              torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    up_buffer = register_buffer("up_buffer", torch::zeros({max_batch_size_, max_seq_len_, intermediate_size_},
                                                          torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    output_buffer =
        register_buffer("output_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                      torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));

    hidden_states_buffer =
        register_buffer("hidden_states_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                             torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    queries_buffer =
        register_buffer("queries_buffer", torch::zeros({max_batch_size_, max_seq_len_, num_attention_heads_ * head_dim_},
                                                       torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    keys_buffer = register_buffer("keys_buffer", torch::zeros({max_batch_size_, max_seq_len_, num_key_value_heads_ * head_dim_},
                                                              torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    values_buffer =
        register_buffer("values_buffer", torch::zeros({max_batch_size_, max_seq_len_, num_key_value_heads_ * head_dim_},
                                                      torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    // Decode output buffer uses q_len=1 to keep contiguous [B, H, 1, D] layout.
    attn_output_heads_buffer =
        register_buffer("attn_output_heads_buffer", torch::zeros({max_batch_size_, num_attention_heads_, 1, head_dim_},
                                                                 torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    attn_output_buffer =
        register_buffer("attn_output_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                           torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));
    attn_output_proj_buffer =
        register_buffer("attn_output_proj_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                                torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));

    norm_buffer = register_buffer("norm_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_},
                                                              torch::TensorOptions().device(embedding_device_).dtype(torch::kBFloat16)));

    // Keep explicit per-layer placement in model-parallel mode.
    if (!multi_gpu_enabled_) {
        this->to(device);
    }

    // Preload MoE kernels to reduce cold-start spikes (Mixtral only)
    preload_moe_kernels();

    if (debug_verbosity >= 1) {
        if (multi_gpu_enabled_) {
            std::cout << "Layer-split model parallel enabled across " << gpu_count_ << " GPUs." << std::endl;
        }
        std::cout << "UnifiedLLMW4A16 initialized on device: " << device << " with w4a16 quantization" << std::endl;
    }
}

UnifiedLLMW4A16Impl &UnifiedLLMW4A16Impl::to(torch::Device device) {
    if (multi_gpu_enabled_) {
        if (debug_verbosity >= 1) {
            std::cout << "Ignoring to(" << device << ") in multi-GPU layer-split mode." << std::endl;
        }
        return *this;
    }

    torch::nn::Module::to(device);

    for (auto &cache : caches_k) {
        cache = cache.to(device);
    }
    for (auto &cache : caches_v) {
        cache = cache.to(device);
    }

    // Explicitly move scratch buffers
    x_buffer = x_buffer.to(device);
    gate_buffer = gate_buffer.to(device);
    up_buffer = up_buffer.to(device);
    output_buffer = output_buffer.to(device);
    hidden_states_buffer = hidden_states_buffer.to(device);
    queries_buffer = queries_buffer.to(device);
    keys_buffer = keys_buffer.to(device);
    values_buffer = values_buffer.to(device);
    attn_output_heads_buffer = attn_output_heads_buffer.to(device);
    attn_output_buffer = attn_output_buffer.to(device);
    attn_output_proj_buffer = attn_output_proj_buffer.to(device);
    norm_buffer = norm_buffer.to(device);

    if (arch_type_ == ArchitectureType::MIXTRAL || arch_type_ == ArchitectureType::QWEN) {
        for (int64_t i = 0; i < num_hidden_layers_; ++i) {
            moe_layers[i]->to(device);
        }
    }
    if (arch_type_ == ArchitectureType::QWEN) {
        for (int64_t i = 0; i < num_hidden_layers_; ++i) {
            q_norms[i]->to(device);
            k_norms[i]->to(device);
        }
    }

    if (debug_verbosity >= 1) {
        std::cout << "UnifiedLLMW4A16 moved to device: " << device << std::endl;
    }
    return *this;
}

torch::Tensor UnifiedLLMW4A16Impl::compute_rope_freqs(int64_t seq_len, int64_t start_pos) {
    auto arange = torch::arange(0, head_dim_, 2, torch::kFloat32).slice(0, 0, head_dim_ / 2);
    auto freqs = 1.0 / torch::pow(rope_theta_, arange / head_dim_);

    if (rope_scaling_enabled && rope_scaling_type == "llama3" && rope_scaling_factor > 0.0f &&
        rope_scaling_original_max_position_embeddings > 0.0f && rope_scaling_high_freq_factor != rope_scaling_low_freq_factor) {
        const float kPi = 3.14159265358979323846f;
        const float low_wavelen = rope_scaling_original_max_position_embeddings / rope_scaling_low_freq_factor;
        const float high_wavelen = rope_scaling_original_max_position_embeddings / rope_scaling_high_freq_factor;

        auto wavelen = (2.0f * kPi) / freqs;
        auto smooth = (rope_scaling_original_max_position_embeddings / wavelen - rope_scaling_low_freq_factor) /
                      (rope_scaling_high_freq_factor - rope_scaling_low_freq_factor);

        auto ones = torch::ones_like(freqs);
        auto factor = torch::full_like(freqs, rope_scaling_factor);
        auto mid = 1.0f / ((1.0f - smooth) / rope_scaling_factor + smooth);

        auto rope_factors = torch::where(wavelen < high_wavelen, ones, torch::where(wavelen > low_wavelen, factor, mid));
        freqs = freqs / rope_factors;
    }

    auto t = torch::arange(start_pos, start_pos + seq_len, torch::kFloat32);
    auto freqs_matrix = torch::outer(t, freqs);
    return torch::polar(torch::ones_like(freqs_matrix), freqs_matrix);
}

std::pair<torch::Tensor, torch::Tensor> UnifiedLLMW4A16Impl::apply_rotary_emb(const torch::Tensor &xq, const torch::Tensor &xk,
                                                                              const torch::Tensor &freqs_cis) {
    auto cos = torch::real(freqs_cis);
    auto sin = torch::imag(freqs_cis);

    // Concatenate to get full head_dim
    std::vector<torch::Tensor> cos_chunks = {cos, cos};
    std::vector<torch::Tensor> sin_chunks = {sin, sin};
    cos = torch::cat(cos_chunks, -1);
    sin = torch::cat(sin_chunks, -1);

    // Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.device()).to(xq.dtype());
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.device()).to(xq.dtype());

    // rotate_half helper
    auto rotate_half = [](const torch::Tensor &x) -> torch::Tensor {
        int64_t head_dim = x.size(-1);
        auto x1 = x.slice(-1, 0, head_dim / 2);
        auto x2 = x.slice(-1, head_dim / 2);
        return torch::cat({-x2, x1}, -1);
    };

    auto xq_out = (xq * cos) + (rotate_half(xq) * sin);
    auto xk_out = (xk * cos) + (rotate_half(xk) * sin);

    return std::make_pair(xq_out, xk_out);
}

torch::Tensor UnifiedLLMW4A16Impl::silu(const torch::Tensor &x) { return torch::silu(x); }
torch::Tensor UnifiedLLMW4A16Impl::gelu(const torch::Tensor &x) { return torch::gelu(x); }
torch::Tensor UnifiedLLMW4A16Impl::swiglu(const torch::Tensor &gate, const torch::Tensor &up) { return silu(gate) * up; }

void UnifiedLLMW4A16Impl::preload_moe_kernels() {
    if ((arch_type_ != ArchitectureType::MIXTRAL && arch_type_ != ArchitectureType::QWEN) || !preload_moe_kernels_enabled) {
        return;
    }
    if (!x_buffer.is_cuda()) {
        return;
    }

    if (debug_verbosity >= 1) {
        std::cout << "Preloading MoE kernels..." << std::endl;
    }

    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().device(x_buffer.device()).dtype(torch::kBFloat16);

    // Minimal shapes to trigger topk/softmax/one_hot/nonzero/index_add kernels.
    auto router_out = torch::rand({1, num_experts_}, opts);
    auto topk = router_out.topk(num_experts_per_tok_, -1);
    auto topk_vals = std::get<0>(topk);
    auto topk_idx = std::get<1>(topk);
    topk_vals = torch::softmax(topk_vals.to(torch::kFloat32), -1).to(router_out.dtype());

    auto expert_mask = torch::one_hot(topk_idx, num_experts_).to(torch::kBool);
    expert_mask = expert_mask.permute({2, 1, 0}); // [experts, top_k, tokens]
    auto expert_hit = torch::nonzero(expert_mask.sum({1, 2}) > 0).squeeze();

    if (expert_hit.numel() > 0) {
        auto token_idx = torch::zeros({1}, torch::TensorOptions().device(router_out.device()).dtype(torch::kLong));
        auto down_slice = torch::zeros({1, hidden_size_}, opts);
        auto topk_pos = torch::zeros({1}, torch::TensorOptions().device(router_out.device()).dtype(torch::kLong));
        auto weights = topk_vals.index({token_idx, topk_pos}).to(down_slice.dtype());

        down_slice.mul_(weights.unsqueeze(-1));
        auto output = torch::zeros({1, hidden_size_}, opts);
        output.index_add_(0, token_idx, down_slice);
    }

    // Warmup the MoE layer as well
    if (moe_layers.size() > 0) {
        auto router_input = torch::randn({1, 2, hidden_size_}, opts);
        moe_layers[0]->forward(router_input);
    }

    if (debug_verbosity >= 1) {
        std::cout << "MoE kernels preloaded." << std::endl;
    }
}

torch::Tensor UnifiedLLMW4A16Impl::forward(torch::Tensor x, int64_t start_pos) {
    if (arch_type_ == ArchitectureType::MIXTRAL) {
        if (multi_gpu_enabled_) {
            return forward_mixtral_multi_gpu(x, start_pos);
        }
        return forward_mixtral(x, start_pos);
    }
    if (arch_type_ == ArchitectureType::QWEN) {
        if (multi_gpu_enabled_) {
            return forward_qwen_multi_gpu(x, start_pos);
        }
        return forward_qwen(x, start_pos);
    }
    throw std::runtime_error("Unsupported architecture.");
}

torch::Tensor UnifiedLLMW4A16Impl::forward_mixtral_multi_gpu(torch::Tensor x, int64_t start_pos) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);

    if (embedding_device_.is_cuda() && embedding_device_.index() >= 0) {
        c10::hip::set_device(static_cast<c10::DeviceIndex>(embedding_device_.index()));
    }
    if (x.device() != embedding_device_) {
        x = x.to(embedding_device_);
    }
    x = token_embedding->forward(x);

    torch::Tensor mask;
    if (seq_len > 1) {
        mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask = torch::triu(mask, 1);
        mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
        mask = mask.to(x.dtype());
    }

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        auto layer_device = layer_devices_[i];
        if (layer_device.is_cuda() && layer_device.index() >= 0) {
            c10::hip::set_device(static_cast<c10::DeviceIndex>(layer_device.index()));
        }
        if (x.device() != layer_device) {
            x = x.to(layer_device);
        }

        auto opts = torch::TensorOptions().device(layer_device).dtype(torch::kBFloat16);
        auto normed = torch::empty({bsz, seq_len, hidden_size_}, opts);
        input_norms[i]->forward_out(normed, x);

        auto normed_2d = normed.view({-1, hidden_size_});
        auto q = q_layers[i]->forward(normed_2d, "q").view({bsz, seq_len, num_attention_heads_, head_dim_});
        auto k = k_layers[i]->forward(normed_2d, "k").view({bsz, seq_len, num_key_value_heads_, head_dim_});
        auto v = v_layers[i]->forward(normed_2d, "v").view({bsz, seq_len, num_key_value_heads_, head_dim_});
        if (debug_verbosity >= 2) {
            std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": qkv done" << std::endl;
        }

        if (!q.is_cuda() || (rope_scaling_enabled && rope_scaling_type == "llama3")) {
            if (debug_verbosity >= 2) {
                std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": rope (torch) begin" << std::endl;
            }
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            if (debug_verbosity >= 2) {
                std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": rope (hip) begin" << std::endl;
            }
            launch_rope(q, k, start_pos, rope_theta_);
        }
        if (debug_verbosity >= 2) {
            std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": rope done" << std::endl;
        }

        auto cache_k = caches_k[i];
        auto cache_v = caches_v[i];
        cache_k = cache_k.narrow(2, start_pos, seq_len).copy_(k.transpose(1, 2));
        cache_v = cache_v.narrow(2, start_pos, seq_len).copy_(v.transpose(1, 2));
        if (debug_verbosity >= 2) {
            std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": cache write done" << std::endl;
        }

        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

        if (attention_mode_ >= 1) {
            if (start_pos == 0) {
                k = repeat_kv(k, GQA_head_ratio_);
                v = repeat_kv(v, GQA_head_ratio_);
            }
        } else {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }

        q = q.transpose(1, 2);

        torch::Tensor attn_output;
        torch::Tensor layer_mask;
        if (mask.defined() && mask.numel() > 0) {
            layer_mask = (mask.device() == layer_device) ? mask : mask.to(layer_device);
        }

        if (debug_verbosity >= 2) {
            const char *attn_backend = (attention_mode_ == 1) ? "SDPA" : (attention_mode_ == 2) ? "HIP_FA_KERNEL" : "EAGER";
            std::cout << "[MIXTRAL-MULTIGPU] Layer " << i << ": attention backend(mode) = " << attn_backend << std::endl;
        }

        if (attention_mode_ == 1) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                c10::optional<torch::Tensor> opt_mask;
                if (layer_mask.defined() && layer_mask.numel() > 0) {
                    opt_mask = layer_mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
                }
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
            }

        } else if (attention_mode_ == 2) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                attn_output = torch::empty_like(q);
                int element_size = q.element_size();

                launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                             (layer_mask.defined() && layer_mask.numel() > 0) ? layer_mask.data_ptr() : nullptr,
                                             attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale,
                                             q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                                             k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size,
                                             v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                                             q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
            }

        } else {
            auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
            if (layer_mask.defined() && layer_mask.numel() > 0) {
                att = att + layer_mask.to(q.dtype());
            }
            auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
            attn_output = torch::matmul(attn_weights, v);
        }

        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});
        auto attn_proj_buf = o_layers[i]->forward(attn_output.view({-1, hidden_size_}), "o").view({bsz, seq_len, hidden_size_});
        x = x + attn_proj_buf;

        auto post_normed = torch::empty({bsz, seq_len, hidden_size_}, opts);
        post_attn_norms[i]->forward_out(post_normed, x);
        auto moe_out = moe_layers[i]->forward(post_normed);
        x = x + moe_out;
    }

    if (x.device() != output_device_) {
        x = x.to(output_device_);
    }
    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_mixtral(torch::Tensor x, int64_t start_pos) {
    if (embedding_device_.is_cuda() && embedding_device_.index() >= 0) {
        c10::hip::set_device(static_cast<c10::DeviceIndex>(embedding_device_.index()));
    }
    if (x.device() != embedding_device_) {
        x = x.to(embedding_device_);
    }

    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t active_window_size = sliding_window_enabled_ ? std::min<int64_t>(sliding_window_size_, max_seq_len_) : max_seq_len_;
    if (active_window_size <= 0) {
        active_window_size = max_seq_len_;
    }
    int64_t abs_end_pos = start_pos + seq_len;
    int64_t kv_seq_len = std::min<int64_t>(active_window_size, abs_end_pos);
    int64_t oldest_pos = abs_end_pos - kv_seq_len;
    if (start_pos == 0) {
        cache_filled_ = 0;
    }
    cache_filled_ = kv_seq_len;

    // Embedding
    x = token_embedding->forward(x);

    // Create causal mask aligned with retained KV window [oldest_pos, abs_end_pos).
    torch::Tensor mask;
    if (seq_len > 1 && kv_seq_len > 0) {
        auto pos_opts = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
        auto q_pos = torch::arange(start_pos, start_pos + seq_len, pos_opts).unsqueeze(1);
        auto k_pos = torch::arange(oldest_pos, oldest_pos + kv_seq_len, pos_opts).unsqueeze(0);
        mask = torch::zeros({seq_len, kv_seq_len}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask.masked_fill_(k_pos > q_pos, -std::numeric_limits<float>::infinity());
        mask = mask.to(x.dtype());
    }

    // Slice buffers for current batch size and sequence length
    auto q_buf = queries_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto normed = norm_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        // Pre-attention norm
        input_norms[i]->forward_out(normed, x);

        // Synchronous calls for Q, K, V (since future removed)
        torch::Tensor q, k, v;

        q_layers[i]->forward(q_buf, normed, "q");
        k_layers[i]->forward(k_buf, normed, "k");
        v_layers[i]->forward(v_buf, normed, "v");

        // Slice padded buffers to valid dimensions before usage
        q = q_buf.slice(-1, 0, num_attention_heads_ * head_dim_);
        k = k_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        // Reshape
        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});
        // Apply RoPE:
        // - CPU path uses PyTorch implementation.
        // - CUDA path uses HIP kernel unless llama3 scaling is enabled.
        if (!q.is_cuda() || (rope_scaling_enabled && rope_scaling_type == "llama3")) {
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            launch_rope(q, k, start_pos, rope_theta_);
        }

        v = v_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        // Cache K/V in ring-buffer layout and read back in chronological order.
        auto k_for_cache = k.transpose(1, 2).contiguous();
        auto v_for_cache = v.transpose(1, 2).contiguous();
        auto cache_k = caches_k[i].narrow(0, 0, bsz);
        auto cache_v = caches_v[i].narrow(0, 0, bsz);
        write_kv_ring(cache_k, k_for_cache, start_pos, active_window_size);
        write_kv_ring(cache_v, v_for_cache, start_pos, active_window_size);

        k = read_kv_window(cache_k, kv_seq_len, oldest_pos, active_window_size);
        v = read_kv_window(cache_v, kv_seq_len, oldest_pos, active_window_size);

        // Expand KV heads for GQA (8 KV heads -> 32 Q heads)
        // For SDPA: Skip repeat during decoding (start_pos > 0) to use native GQA.
        // For Prompt (start_pos == 0) or non-SDPA: Apply repeat (workaround for potential GQA+Causal issues).
        if (attention_mode_ >= 1) {
            if (start_pos == 0) {
                k = repeat_kv(k, GQA_head_ratio_);
                v = repeat_kv(v, GQA_head_ratio_);
            }
        } else {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }

        // Transpose for attention
        q = q.transpose(1, 2);

        torch::Tensor attn_output;
        if (debug_verbosity >= 2) {
            const char *attn_backend = (attention_mode_ == 1) ? "SDPA" : (attention_mode_ == 2) ? "HIP_FA_KERNEL" : "EAGER";
            std::cout << "[MIXTRAL] Layer " << i << ": attention backend(mode) = " << attn_backend << std::endl;
        }
        if (attention_mode_ == 1) {
            // Mode 1: PyTorch SDPA
            if (start_pos == 0 && seq_len > 1 && kv_seq_len == seq_len) {
                // First prompt pass: use native causal optimization
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                // Decoding or subsequent chunks: use explicit mask if defined
                c10::optional<torch::Tensor> opt_mask;
                if (mask.defined() && mask.numel() > 0) {
                    // Expand mask from [seq_len, kv_seq_len] to [1, 1, seq_len, kv_seq_len]
                    opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
                }
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
            }

        } else if (attention_mode_ == 2) {
            // Mode 2: Custom HIP Kernel
            if (seq_len > 1) {
                // Prefill chunks use SDPA; HIP decode kernel path remains for single-token decoding.
                c10::optional<torch::Tensor> opt_mask;
                if (mask.defined() && mask.numel() > 0) {
                    opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
                }
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
            } else {
                // Decoding phase: use Custom HIP Kernel
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                // Output tensor: slice from preallocated heads buffer when q_len == 1
                if (q.size(2) == 1) {
                    attn_output = attn_output_heads_buffer.narrow(0, 0, batch_size);
                } else {
                    attn_output = torch::empty_like(q);
                }

                int element_size = q.element_size(); // Should be 2 for BF16

                launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                             (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr, // Basic mask support check
                                             attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale,
                                             q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                                             k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size,
                                             v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size,
                                             0, // stride_mask_seq (not fully supported yet in this call site, assuming basic usage)
                                             q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
            }

        } else {
            // Mode 0: Manual Matmul
            auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));

            if (mask.defined() && mask.numel() > 0) {
                att = att + mask.to(q.dtype());
            }

            auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
            attn_output = torch::matmul(attn_weights, v);
        }

        // Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});

        // Output projection
        auto attn_out_buf = attn_output_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);

        auto attn_out_buf_proj = attn_output_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        o_layers[i]->forward(attn_proj_buf, attn_out_buf_proj.slice(-1, 0, hidden_size_), "o");

        // Residual connection
        x = x + attn_proj_buf.slice(-1, 0, hidden_size_);

        // Post-attention norm
        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));

        // MoE block
        auto moe_out = moe_layers[i]->forward(normed.slice(-1, 0, hidden_size_));

        // Residual connection
        x = x + moe_out;
    }

    // Final norm and output
    x = final_norm->forward(x);
    x = lm_head->forward(x);

    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen_multi_gpu(torch::Tensor x, int64_t start_pos) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);
    const int64_t q_proj_size = num_attention_heads_ * head_dim_;

    if (embedding_device_.is_cuda() && embedding_device_.index() >= 0) {
        c10::hip::set_device(static_cast<c10::DeviceIndex>(embedding_device_.index()));
    }
    if (x.device() != embedding_device_) {
        x = x.to(embedding_device_);
    }
    x = token_embedding->forward(x);

    torch::Tensor mask;
    if (seq_len > 1) {
        mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask = torch::triu(mask, 1);
        mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
        mask = mask.to(x.dtype());
    }

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        auto layer_device = layer_devices_[i];
        if (layer_device.is_cuda() && layer_device.index() >= 0) {
            c10::hip::set_device(static_cast<c10::DeviceIndex>(layer_device.index()));
        }
        if (x.device() != layer_device) {
            x = x.to(layer_device);
        }

        auto opts = torch::TensorOptions().device(layer_device).dtype(torch::kBFloat16);
        auto normed = torch::empty({bsz, seq_len, hidden_size_}, opts);
        input_norms[i]->forward_out(normed, x);

        auto normed_2d = normed.view({-1, hidden_size_});
        auto q = q_layers[i]->forward(normed_2d, "q").view({bsz, seq_len, num_attention_heads_, head_dim_});
        auto k = k_layers[i]->forward(normed_2d, "k").view({bsz, seq_len, num_key_value_heads_, head_dim_});
        auto v = v_layers[i]->forward(normed_2d, "v").view({bsz, seq_len, num_key_value_heads_, head_dim_});

        q = q_norms[i]->forward(q);
        k = k_norms[i]->forward(k);

        if (!q.is_cuda() || (rope_scaling_enabled && rope_scaling_type == "llama3")) {
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            launch_rope(q, k, start_pos, rope_theta_);
        }

        auto cache_k = caches_k[i];
        auto cache_v = caches_v[i];
        cache_k = cache_k.narrow(2, start_pos, seq_len).copy_(k.transpose(1, 2));
        cache_v = cache_v.narrow(2, start_pos, seq_len).copy_(v.transpose(1, 2));

        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

        if (attention_mode_ >= 1) {
            if (start_pos == 0) {
                k = repeat_kv(k, GQA_head_ratio_);
                v = repeat_kv(v, GQA_head_ratio_);
            }
        } else {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }

        q = q.transpose(1, 2);

        torch::Tensor attn_output;
        torch::Tensor layer_mask;
        if (mask.defined() && mask.numel() > 0) {
            layer_mask = (mask.device() == layer_device) ? mask : mask.to(layer_device);
        }

        if (debug_verbosity >= 2) {
            const char *attn_backend = (attention_mode_ == 1) ? "SDPA" : (attention_mode_ == 2) ? "HIP_FA_KERNEL" : "EAGER";
            std::cout << "[QWEN-MULTIGPU] Layer " << i << ": attention backend(mode) = " << attn_backend << std::endl;
        }

        if (attention_mode_ == 1) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                c10::optional<torch::Tensor> opt_mask;
                if (layer_mask.defined() && layer_mask.numel() > 0) {
                    opt_mask = layer_mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
                }
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
            }
        } else if (attention_mode_ == 2) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
                attn_output = torch::empty_like(q);
                int element_size = q.element_size();

                launch_flash_attn_decode_hip(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                                             (layer_mask.defined() && layer_mask.numel() > 0) ? layer_mask.data_ptr() : nullptr,
                                             attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale,
                                             q.stride(2) * element_size, q.stride(1) * element_size, q.stride(0) * element_size,
                                             k.stride(2) * element_size, k.stride(1) * element_size, k.stride(0) * element_size,
                                             v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                                             q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
            }
        } else {
            auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
            if (layer_mask.defined() && layer_mask.numel() > 0) {
                att = att + layer_mask.to(q.dtype());
            }
            auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
            attn_output = torch::matmul(attn_weights, v);
        }

        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, q_proj_size});
        auto attn_proj_buf = o_layers[i]->forward(attn_output.view({-1, q_proj_size}), "o").view({bsz, seq_len, hidden_size_});

        // Residual connection
        x = x + attn_proj_buf;

        // Post-attention norm
        auto post_normed = torch::empty({bsz, seq_len, hidden_size_}, opts);

        // Post-attention norm
        post_attn_norms[i]->forward_out(post_normed, x);

        // MoE block
        auto moe_out = moe_layers[i]->forward(post_normed);

        // Residual connection
        x = x + moe_out;
    }

    if (x.device() != output_device_) {
        x = x.to(output_device_);
    }
    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::forward_qwen(torch::Tensor x, int64_t start_pos) {
    if (embedding_device_.is_cuda() && embedding_device_.index() >= 0) {
        c10::hip::set_device(static_cast<c10::DeviceIndex>(embedding_device_.index()));
    }
    if (x.device() != embedding_device_) {
        x = x.to(embedding_device_);
    }

    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);
    const int64_t q_proj_size = num_attention_heads_ * head_dim_;

    x = token_embedding->forward(x);

    torch::Tensor mask;
    if (seq_len > 1) {
        mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                           torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
        mask = torch::triu(mask, 1);
        mask = torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
        mask = mask.to(x.dtype());
    }

    auto q_buf = queries_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        input_norms[i]->forward_out(normed, x);

        torch::Tensor q, k, v;

        q_layers[i]->forward(q_buf, normed, "q");
        k_layers[i]->forward(k_buf, normed, "k");
        v_layers[i]->forward(v_buf, normed, "v");

        q = q_buf.slice(-1, 0, num_attention_heads_ * head_dim_);
        k = k_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);

        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        q = q_norms[i]->forward(q);
        k = k_norms[i]->forward(k);

        if (!q.is_cuda() || (rope_scaling_enabled && rope_scaling_type == "llama3")) {
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            launch_rope(q, k, start_pos, rope_theta_);
        }

        v = v_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        auto cache_k = caches_k[i];
        auto cache_v = caches_v[i];
        cache_k = cache_k.narrow(2, start_pos, seq_len).copy_(k.transpose(1, 2));
        cache_v = cache_v.narrow(2, start_pos, seq_len).copy_(v.transpose(1, 2));

        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

        if (attention_mode_ >= 1) {
            if (start_pos == 0) {
                k = repeat_kv(k, GQA_head_ratio_);
                v = repeat_kv(v, GQA_head_ratio_);
            }
        } else {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }

        q = q.transpose(1, 2);

        torch::Tensor attn_output;
        if (debug_verbosity >= 2) {
            const char *attn_backend = (attention_mode_ == 1) ? "SDPA" : (attention_mode_ == 2) ? "HIP_FA_KERNEL" : "EAGER";
            std::cout << "[QWEN] Layer " << i << ": attention backend(mode) = " << attn_backend << std::endl;
        }
        if (attention_mode_ == 1) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                c10::optional<torch::Tensor> opt_mask;
                if (mask.defined() && mask.numel() > 0) {
                    opt_mask = mask.unsqueeze(0).unsqueeze(0).to(q.dtype());
                }
                attn_output = torch::scaled_dot_product_attention(q, k, v, opt_mask, 0.0, false, std::nullopt, true);
            }
        } else if (attention_mode_ == 2) {
            if (start_pos == 0 && seq_len > 1) {
                attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
            } else {
                int batch_size = q.size(0);
                int n_heads_Q = q.size(1);
                int n_heads_KV = k.size(1);
                int head_dim = q.size(3);
                int seq_len_kv = k.size(2);
                float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

                if (q.size(2) == 1) {
                    attn_output = attn_output_heads_buffer.narrow(0, 0, batch_size);
                } else {
                    attn_output = torch::empty_like(q);
                }

                int element_size = q.element_size();
                launch_flash_attn_decode_hip(
                    q.data_ptr(), k.data_ptr(), v.data_ptr(), (mask.defined() && mask.numel() > 0) ? mask.data_ptr() : nullptr,
                    attn_output.data_ptr(), batch_size, n_heads_Q, n_heads_KV, head_dim, seq_len_kv, scale, q.stride(2) * element_size,
                    q.stride(1) * element_size, q.stride(0) * element_size, k.stride(2) * element_size, k.stride(1) * element_size,
                    k.stride(0) * element_size, v.stride(2) * element_size, v.stride(1) * element_size, v.stride(0) * element_size, 0,
                    q.dtype() == torch::kBFloat16, c10::hip::getCurrentHIPStream().stream());
            }
        } else {
            auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));
            if (mask.defined() && mask.numel() > 0) {
                att = att + mask.to(q.dtype());
            }
            auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
            attn_output = torch::matmul(attn_weights, v);
        }
        // Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, q_proj_size});

        // Output projection
        auto attn_out_buf = q_buf;
        attn_out_buf.slice(-1, 0, q_proj_size).copy_(attn_output);

        o_layers[i]->forward(attn_proj_buf, attn_out_buf.slice(-1, 0, q_proj_size), "o");

        // Residual connection
        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));
        // Post-attention norm
        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));

        // MoE block
        auto moe_out = moe_layers[i]->forward(normed.slice(-1, 0, hidden_size_));

        // Residual connection
        x.add_(moe_out);
    }

    x = final_norm->forward(x);
    x = lm_head->forward(x);
    return x;
}

torch::Tensor UnifiedLLMW4A16Impl::generate(torch::Tensor input_ids, int64_t max_new_tokens, float temperature, float top_p, int64_t top_k,
                                            int64_t eos_token_id) {
    std::cout << "Libtorch input_ids shape: " << input_ids.sizes() << std::endl;

    int64_t batch_size = input_ids.size(0);
    int64_t prompt_len = input_ids.size(1);
    int64_t start_pos = 0;
    torch::Tensor output;
    torch::Tensor next_token;
    torch::Tensor last_token;

    // Ensure model is in eval mode (no dropout, etc.)
    this->eval();
    torch::NoGradGuard no_grad;
    const bool should_sync_device = input_ids.is_cuda();
    const int64_t prefill_chunk_size = std::max<int64_t>(1, prefill_chunk_size_);

    auto run_prefill_chunks = [&](const torch::Tensor &prompt_tokens) {
        torch::Tensor local_output;
        int64_t total_len = prompt_tokens.size(1);
        for (int64_t chunk_start = 0; chunk_start < total_len; chunk_start += prefill_chunk_size) {
            int64_t chunk_len = std::min<int64_t>(prefill_chunk_size, total_len - chunk_start);
            local_output = forward(prompt_tokens.narrow(1, chunk_start, chunk_len), chunk_start);
        }
        return local_output;
    };

    // Warmup cycle
    if (warmup_) {
        std::cout << "Running warmup..." << std::endl;
        int warm_up = 1;
        // Prefill warmup
        for (int i = 0; i < warm_up; i++) {
            run_prefill_chunks(input_ids);
        }

        // M=1 Warmup (Single token generation after prefill)
        std::cout << "Running M=1 Warmup (Hetero Path)..." << std::endl;

        auto dummy_token = torch::zeros({1, 1}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        int64_t warmup_start_pos = input_ids.size(1); // Position after prefill
        for (int i = 0; i < 1; i++) {
            forward(dummy_token, warmup_start_pos + i);
        }
    } else {
        if (debug_verbosity >= 1)
            std::cout << "Skipping warmup." << std::endl;
    }

    // Prefill phase: process initial prompt
    if (debug_verbosity >= 2) {
        std::cout << "Prefill phase" << std::endl;
    }
    if (should_sync_device) {
        torch::cuda::synchronize();
    }
    auto start_prefill = std::chrono::high_resolution_clock::now();
    output = run_prefill_chunks(input_ids);

    if (should_sync_device) {
        torch::cuda::synchronize();
    }
    auto end_prefill = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_prefill = end_prefill - start_prefill;
    // std::cout << "Prefill time: " << elapsed_prefill.count() << " seconds" << std::endl;

    // Get next token: implementation when temperature is 0
    last_token = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
    if (temperature < 0.01f) {
        // Greedy decoding: argmax
        next_token = torch::argmax(last_token, -1, true);
    } else {
        // Sampling
        int64_t next_token_id = sample_token(last_token.squeeze(0), temperature, top_p, top_k);
        next_token = torch::tensor({{next_token_id}}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
    }

    // Pre-allocate output buffer for generation to avoid per-token cat
    auto input_tensor = torch::zeros({batch_size, prompt_len + max_new_tokens}, input_ids.options());
    input_tensor.narrow(1, 0, prompt_len).copy_(input_ids);
    input_tensor.narrow(1, prompt_len, 1).copy_(next_token);

    // Check for EOS token
    int64_t next_token_id = next_token.item<int64_t>();
    int64_t token_len = prompt_len + 1;
    if (eos_token_id >= 0 && next_token_id == eos_token_id) {
        return input_tensor.narrow(1, 0, token_len);
    }

    start_pos = token_len - 1;

    // Generate remaining tokens
    if (should_sync_device) {
        torch::cuda::synchronize();
    }
    auto start_gen = std::chrono::high_resolution_clock::now();
    int64_t actual_generated = 0;

    // Generation loop
    std::cout << "Generation phase" << std::endl;
    while (token_len < max_new_tokens + prompt_len) {
        output = forward(next_token, start_pos);
        last_token = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

        if (temperature < 0.01f) {
            next_token = torch::argmax(last_token, -1, true);
        } else {
            int64_t next_token_id = sample_token(last_token.squeeze(0), temperature, top_p, top_k);
            next_token = torch::tensor({{next_token_id}}, torch::TensorOptions().dtype(torch::kInt64).device(input_ids.device()));
        }

        next_token_id = next_token.item<int64_t>();
        if (eos_token_id >= 0 && next_token_id == eos_token_id) {
            break;
        }

        input_tensor.narrow(1, token_len, 1).copy_(next_token);
        actual_generated++;
        token_len++;
        start_pos++;
    }

    if (should_sync_device) {
        torch::cuda::synchronize();
    }
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> generation_time = end_gen - start_gen;

    // print to terminal
    std::cout << "Prefill time: " << elapsed_prefill.count() << " seconds" << std::endl;
    std::cout << "Total Generation Time: " << generation_time.count() << " seconds" << std::endl;
    if (actual_generated > 0) {
        double time_per_token = generation_time.count() / actual_generated;
        std::cout << "Average Time per Token: " << time_per_token << " seconds" << std::endl;
    }

    return input_tensor.narrow(1, 0, token_len);
}
