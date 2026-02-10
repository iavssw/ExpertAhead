#include "unified_llm_w4a16_predict/unified_llm_w4a16.hpp"
#include "hipkernels/embedding.hpp"
#include "hipkernels/flash_attn_decode.hpp"
#include "hipkernels/lm_head.hpp"
#include "hipkernels/rmsnorm.hpp"
#include "hipkernels/rope.hpp"
#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include "unified_llm_w4a16_predict/helper.hpp"
#include "unified_llm_w4a16_predict/npuSetup.hpp"
#include <c10/hip/HIPStream.h>
#include <chrono>
#include <fstream>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

template <typename Func> void time_op(const std::string &name, Func func) {
    torch::cuda::synchronize();
    auto start = std::chrono::high_resolution_clock::now();
    func();
    torch::cuda::synchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << name << ": " << ms.count() << " ms" << std::endl;
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
#if USE_HIP_EMBEDDING
    int64_t bsz = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t total_tokens = bsz * seq_len;

    if (total_tokens == 1) {
        auto output = output_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

        hipkernels::launch_embedding_forward(weight, input, output, c10::hip::getCurrentHIPStream().stream());
        return output;
    }
#endif
    return torch::nn::functional::embedding(input, weight);
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
#if USE_HIP_LM_HEAD
    int64_t bsz = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t total_tokens = bsz * seq_len;

    if (total_tokens == 1) {
        int64_t vocab_size = weight.size(0);
        int64_t hidden_size = weight.size(1);

        auto logits = logits_buffer.slice(0, 0, bsz).slice(1, 0, seq_len);

        hipkernels::launch_lm_head_forward(
            (hip_bfloat16 *)logits.data_ptr<at::BFloat16>(), (const hip_bfloat16 *)input.data_ptr<at::BFloat16>(),
            (const hip_bfloat16 *)weight.data_ptr<at::BFloat16>(), 1, hidden_size, vocab_size, c10::hip::getCurrentHIPStream().stream());
        return logits.to(torch::kFloat32);
    }
#endif
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

std::future<int> QuantizedLinearImpl::forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type) {
    // we don't care about the future for now
    std::future<int> fut;

    if (debug_verbosity >= 2) {
        std::cout << "Forward " << layer_type << " (Target: " << hw_target << ")" << std::endl;
    }

    int64_t M = input.numel() / in_features_;
    int64_t group_size = in_features_;
    if (scale_.dim() == 2) {
        int64_t n_groups = scale_.size(1);
        if (n_groups > 0)
            group_size = in_features_ / n_groups;
    }
    auto input_2d = input.view({-1, in_features_});
    auto output_2d = output_buffer.view({-1, out_features_});

    // Compute the GEMV or GEMM on Fused Hip Kernels
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
        // If bias is defined, we must wait for GEMM to finish before adding bias
        if (fut.valid()) {
            fut.wait();
        }

        if (output_buffer.size(-1) != bias_.size(-1)) {
            throw std::runtime_error("Bias size mismatch with output buffer.");
        }
        output_buffer.add_(bias_);
        return std::future<int>(); // Return invalid/empty future as we are done
    }
    return fut;
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

void QuantizedLinearImpl::import_weights_to_xdna() {
    // Import buffers to XDNA to allow CPU access to GPU memory
    // This is crucial for the CPU path in hetero mode
    if (quantized_weight_.defined()) {
        import_dma_buf_to_xdna(quantized_weight_.data_ptr(), quantized_weight_.numel(), 1); // uint8
    }
    if (scale_.defined()) {
        import_dma_buf_to_xdna(scale_.data_ptr(), scale_.numel(), 2); // bf16
    }
    if (zero_point_.defined()) {
        import_dma_buf_to_xdna(zero_point_.data_ptr(), zero_point_.numel(), 1); // int8
    }

    if (bias_.defined()) {
        import_dma_buf_to_xdna(bias_.data_ptr(), bias_.numel(), bias_.element_size());
    }
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
      groupsize_(groupsize), GQA_head_ratio_(num_attention_heads / num_key_value_heads), num_experts_(num_experts),
      num_experts_per_tok_(num_experts_per_tok), npu_config_(npu_config) {

    std::cout << "Initializing UnifiedLLMW4A16Impl_Predict" << std::endl;

    // Read NPU config early to set debug verbosity
    std::cout << "Using NPU config struct." << std::endl;
    // Initialize NPU (reads config internally)
    if (this->initialize_npu() != 0) {
        throw std::runtime_error("Failed to initialize NPU context");
    }

    // Use global warmup setting from npuSetup
    warmup_ = warmup_enabled;
    std::cout << "Warmup: " << (warmup_ ? "Enabled" : "Disabled") << std::endl;

    // Token embedding - initialize on device
    token_embedding = register_module("token_embedding", HipEmbedding(vocab_size_, hidden_size_, max_batch_size_, max_seq_len_));
    token_embedding->to(device);
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

        // MLP layers (quantized)
        gate_layers.push_back(
            register_module("gate_" + std::to_string(i), QuantizedLinear(hidden_size_, intermediate_size_, false, max_seq_len_, "gate")));
        up_layers.push_back(
            register_module("up_" + std::to_string(i), QuantizedLinear(hidden_size_, intermediate_size_, false, max_seq_len_, "up")));
        down_layers.push_back(
            register_module("down_" + std::to_string(i), QuantizedLinear(intermediate_size_, hidden_size_, false, max_seq_len_, "down")));

        // Normalization layers (not quantized)
        input_norms.push_back(register_module("input_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_)));
        post_attn_norms.push_back(register_module("post_attn_norm_" + std::to_string(i), RMSNorm(hidden_size_, rms_norm_eps_)));

        // KV caches - initialize on device with bf16
        caches_k.push_back(
            register_buffer("cache_k_" + std::to_string(i), torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                                         torch::TensorOptions().device(device).dtype(torch::kBFloat16))));
        caches_v.push_back(
            register_buffer("cache_v_" + std::to_string(i), torch::zeros({max_batch_size_, num_key_value_heads_, max_seq_len_, head_dim_},
                                                                         torch::TensorOptions().device(device).dtype(torch::kBFloat16))));
    }

    // Final norm and output head
    final_norm = register_module("final_norm", RMSNorm(hidden_size_, rms_norm_eps_));
    final_norm->to(device);
    final_norm->to(torch::kBFloat16);

    lm_head = register_module("lm_head", LmHeadLinear(hidden_size_, vocab_size_, max_batch_size_, max_seq_len_));

    // Move all layers to device
    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        q_layers[i]->to(device);
        k_layers[i]->to(device);
        v_layers[i]->to(device);
        o_layers[i]->to(device);
        gate_layers[i]->to(device);
        up_layers[i]->to(device);
        down_layers[i]->to(device);
        input_norms[i]->to(device);
        input_norms[i]->to(torch::kBFloat16);
        post_attn_norms[i]->to(device);
        post_attn_norms[i]->to(torch::kBFloat16);
    }
    lm_head->to(device);
    lm_head->to(torch::kBFloat16);

    // Register scratch buffers

    x_buffer = register_buffer("x_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));
    gate_buffer = register_buffer("gate_buffer", torch::zeros({max_batch_size_, max_seq_len_, intermediate_size_}, torch::kBFloat16));
    up_buffer = register_buffer("up_buffer", torch::zeros({max_batch_size_, max_seq_len_, intermediate_size_}, torch::kBFloat16));
    output_buffer = register_buffer("output_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));

    hidden_states_buffer =
        register_buffer("hidden_states_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));
    queries_buffer = register_buffer("queries_buffer",
                                     torch::zeros({max_batch_size_, max_seq_len_, num_attention_heads_ * head_dim_}, torch::kBFloat16));
    keys_buffer =
        register_buffer("keys_buffer", torch::zeros({max_batch_size_, max_seq_len_, num_key_value_heads_ * head_dim_}, torch::kBFloat16));
    values_buffer =
        register_buffer("values_buffer", torch::zeros({max_batch_size_, max_seq_len_, num_key_value_heads_ * head_dim_}, torch::kBFloat16));
    // Decode output buffer uses q_len=1 to keep contiguous [B, H, 1, D] layout.
    attn_output_heads_buffer =
        register_buffer("attn_output_heads_buffer", torch::zeros({max_batch_size_, num_attention_heads_, 1, head_dim_}, torch::kBFloat16));
    attn_output_buffer =
        register_buffer("attn_output_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));
    attn_output_proj_buffer =
        register_buffer("attn_output_proj_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));

    norm_buffer = register_buffer("norm_buffer", torch::zeros({max_batch_size_, max_seq_len_, hidden_size_}, torch::kBFloat16));

    // Move entire module (including all buffers and parameters) to device
    this->to(device);

    if (debug_verbosity >= 1) {
        std::cout << "UnifiedLLMW4A16 initialized on device: " << device << " with w4a16 quantization" << std::endl;
    }
}

UnifiedLLMW4A16Impl &UnifiedLLMW4A16Impl::to(torch::Device device) {
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

torch::Tensor UnifiedLLMW4A16Impl::forward(torch::Tensor x, int64_t start_pos) {
    if (arch_type_ != ArchitectureType::LLAMA3) {
        throw std::runtime_error("Only LLAMA3 is supported.");
    }
    return forward_llama3(x, start_pos);
}

torch::Tensor UnifiedLLMW4A16Impl::forward_llama3(torch::Tensor x, int64_t start_pos) {
    int64_t bsz = x.size(0);
    int64_t seq_len = x.size(1);

    // Embedding
    x = token_embedding->forward(x);

    // Create causal mask
    torch::Tensor mask;
    if (seq_len > 1) {
        if (seq_len > 1) {
            mask = torch::full({seq_len, seq_len}, -std::numeric_limits<float>::infinity(),
                               torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
            mask = torch::triu(mask, 1);
            mask =
                torch::hstack({torch::zeros({seq_len, start_pos}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())), mask});
            mask = mask.to(x.dtype());
        }
    }

    // Slice buffers for current batch size and sequence length
    auto q_buf = queries_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto k_buf = keys_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto v_buf = values_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto attn_proj_buf = attn_output_proj_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    auto gate_buf = gate_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto up_buf = up_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto out_buf = output_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
    auto normed = norm_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

    for (int64_t i = 0; i < num_hidden_layers_; ++i) {
        // Pre-attention norm
        input_norms[i]->forward_out(normed, x);

        // Async calls for Q, K, V
        torch::Tensor q, k, v;

        auto f_q = q_layers[i]->forward(q_buf, normed, "q");
        auto f_k = k_layers[i]->forward(k_buf, normed, "k");
        auto f_v = v_layers[i]->forward(v_buf, normed, "v");

        if (f_q.valid())
            f_q.wait();
        if (f_k.valid())
            f_k.wait();

        // Slice padded buffers to valid dimensions before usage
        q = q_buf.slice(-1, 0, num_attention_heads_ * head_dim_);
        k = k_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);

        // Reshape
        q = q.view({bsz, seq_len, num_attention_heads_, head_dim_});
        k = k.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        // Apply RoPE (Llama3 scaling if configured)
        if (rope_scaling_enabled && rope_scaling_type == "llama3") {
            auto freqs_cis = compute_rope_freqs(seq_len, start_pos);
            auto rope_result = apply_rotary_emb(q, k, freqs_cis);
            q = rope_result.first;
            k = rope_result.second;
        } else {
            launch_rope(q, k, start_pos, rope_theta_);
        }

        if (f_v.valid())
            f_v.wait();

        v = v_buf.slice(-1, 0, num_key_value_heads_ * head_dim_);
        v = v.view({bsz, seq_len, num_key_value_heads_, head_dim_});

        // Reshape K/V to [B, H, S, D] for storage
        auto k_transposed = k.transpose(1, 2);
        auto v_transposed = v.transpose(1, 2);

        caches_k[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(k_transposed);
        caches_v[i].narrow(0, 0, bsz).narrow(2, start_pos, seq_len).copy_(v_transposed);

        // Retrieve full cache in [B, H, S, D]
        k = caches_k[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);
        v = caches_v[i].narrow(0, 0, bsz).narrow(2, 0, start_pos + seq_len);

        // Expand KV heads for GQA (8 KV heads -> 32 Q heads)
        // For SDPA: Skip repeat during decoding (start_pos > 0) to use native GQA.
        // For Prompt (start_pos == 0) or non-SDPA: Apply repeat (workaround for potential GQA+Causal issues).
#if LLAMA_USE_SCALED_ATTENTION >= 1
        if (start_pos == 0) {
            k = repeat_kv(k, GQA_head_ratio_);
            v = repeat_kv(v, GQA_head_ratio_);
        }
#else
        k = repeat_kv(k, GQA_head_ratio_);
        v = repeat_kv(v, GQA_head_ratio_);
#endif

        // Transpose for attention
        q = q.transpose(1, 2);

        torch::Tensor attn_output;
#if LLAMA_USE_SCALED_ATTENTION == 1
        // Mode 1: PyTorch SDPA
        if (start_pos == 0 && seq_len > 1) {
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

#elif LLAMA_USE_SCALED_ATTENTION == 2
        // Mode 2: Custom HIP Kernel
        if (start_pos == 0 && seq_len > 1) {
            // First prompt pass: use native causal optimization (SDPA fallback for prefill)
            attn_output = torch::scaled_dot_product_attention(q, k, v, c10::nullopt, 0.0, true, std::nullopt, false);
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

#else
        // Mode 0: Manual Matmul
        auto att = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_dim_));

        if (mask.defined() && mask.numel() > 0) {
            att = att + mask.to(q.dtype());
        }

        auto attn_weights = torch::softmax(att.to(torch::kFloat32), -1).to(q.dtype());
        attn_output = torch::matmul(attn_weights, v);
#endif

        // Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view({bsz, seq_len, hidden_size_});

        // Copy to attn_output_buffer to ensure contiguous memory and registered handle
        auto attn_out_buf = attn_output_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);
        attn_out_buf.slice(-1, 0, hidden_size_).copy_(attn_output);

        auto attn_out_buf_proj = attn_output_buffer.narrow(0, 0, bsz).narrow(1, 0, seq_len);

        auto f_o = o_layers[i]->forward(attn_proj_buf, attn_out_buf_proj.slice(-1, 0, hidden_size_), "o");
        if (f_o.valid())
            f_o.wait();

        // Residual connection: x = x + attn_output
        x.add_(attn_proj_buf.slice(-1, 0, hidden_size_));

        // MLP (GeGLU)
        post_attn_norms[i]->forward_out(normed.slice(-1, 0, hidden_size_), x.slice(-1, 0, hidden_size_));

        auto normed_sliced = normed.slice(-1, 0, hidden_size_);

        // MLP projections with buffers
        auto f_gate = gate_layers[i]->forward(gate_buf, normed_sliced, "gate");
        auto f_up = up_layers[i]->forward(up_buf, normed_sliced, "up");
        if (f_gate.valid())
            f_gate.wait();

        // In-place GELU and multiplication
        torch::silu_(gate_buf); // Llama3 uses SiLU (Swish), not GELU

        if (f_up.valid())
            f_up.wait();

        gate_buf.mul_(up_buf);

        auto f_down = down_layers[i]->forward(out_buf, gate_buf.slice(-1, 0, intermediate_size_), "down");
        if (f_down.valid())
            f_down.wait();

        // Residual connection: x = x + mlp_output
        x.add_(out_buf.slice(-1, 0, hidden_size_));
    }

    // Final norm and output
    x = final_norm->forward(x);

    // Optimized LM Head (Ported from Llama.cpp mmvf)
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

    // Warmup cycle
    if (warmup_) {
        std::cout << "Running warmup..." << std::endl;
        int warm_up = 1;
        // Prefill warmup
        for (int i = 0; i < warm_up; i++) {
            forward(input_ids, start_pos);
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
    torch::cuda::synchronize();
    auto start_prefill = std::chrono::high_resolution_clock::now();

    output = forward(input_ids, start_pos);

    torch::cuda::synchronize();
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

    // Check for EOS token
    int64_t next_token_id = next_token.item<int64_t>();
    if (eos_token_id >= 0 && next_token_id == eos_token_id) {
        return input_ids;
    }

    torch::Tensor input_tensor = torch::cat({input_ids, next_token}, 1);
    int64_t token_len = input_tensor.size(1);
    start_pos = token_len - 1;

    // Generate remaining tokens
    torch::cuda::synchronize();
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

        input_tensor = torch::cat({input_tensor, next_token}, 1);
        actual_generated++;
        token_len++;
        start_pos++;
    }

    torch::cuda::synchronize();
    auto end_gen = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> generation_time = end_gen - start_gen;

    // print to terminal
    std::cout << "Prefill time: " << elapsed_prefill.count() << " seconds" << std::endl;
    std::cout << "Total Generation Time: " << generation_time.count() << " seconds" << std::endl;
    if (actual_generated > 0) {
        double time_per_token = generation_time.count() / actual_generated;
        std::cout << "Average Time per Token: " << time_per_token << " seconds" << std::endl;
    }

    return input_tensor;
}

int UnifiedLLMW4A16Impl::initialize_npu() {
    // Read config first to set debug_verbosity (already called in constructor, but safe to call again)
    // Configure NPU variables from struct
    configure_npu(npu_config_);

    if (debug_verbosity >= 1)
        std::cout << "Initializing NPU..." << std::endl;

    // Hardcoded driver path
    const char *drv_path = "/dev/accel/accel0";

    // Open XDNA driver (using global xdna_drv_fd defined in npuSetup.cpp)
    if (initialize_xdna_driver(drv_path) != 0) {
        return -1;
    }

    if (debug_verbosity >= 1)
        std::cout << "Verbosity: " << debug_verbosity << std::endl;

    // Import all weights to XDNA using this class instance
    if (debug_verbosity >= 1)
        std::cout << "Loading NPU kernels..." << std::endl;
    init_npu(); // Initialize NPU context arrays with max capacity

    load_npu_kernels(xdna_drv_fd, npu_config_);

    return 0;
}

void UnifiedLLMW4A16Impl::import_weights() {
    if (debug_verbosity >= 1)
        std::cout << "Importing weights to XDNA..." << std::endl;
    // Import all weights and buffers to XDNA
    import_all_weights_to_xdna(*this);
    if (debug_verbosity >= 1)
        std::cout << "Weights imported." << std::endl;
}
