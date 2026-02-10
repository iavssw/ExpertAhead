#pragma once

#include <cstdint>
#include <future>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>

// Attention mechanism selection:
// 0 = Manual matmul, 1 = PyTorch SDPA, 2 = Custom HIP kernel
#define LLAMA_USE_SCALED_ATTENTION 2
// Use optimized HIP kernel for LM Head and Embedding
#define USE_HIP_EMBEDDING 1
#define USE_HIP_LM_HEAD 1

// Architecture type enum
enum class ArchitectureType { LLAMA3 };

// Quantized Linear Layer for w4a16 (4-bit weights, 16-bit activations)
// Weights are stored as 4-bit packed in uint8, with scales for dequantization
class QuantizedLinearImpl : public torch::nn::Module {
  public:
    QuantizedLinearImpl(int64_t in_features, int64_t out_features, bool bias = false, int64_t max_seq_len = 2048,
                        std::string layer_type = "");

    // Dequantize 4-bit weights to bf16 (Standard/Unpacked)
    torch::Tensor dequantize_weights();

    // Forward pass: dequantize weights, then perform linear operation
    // Takes an optional output buffer for in-place operation
    // Returns a future if async execution is possible and no bias addition is needed immediately
    std::future<int> forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type);

    // Set quantized weights (for loading from state dict)
    void set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx = torch::Tensor());
    // Directly set preprocessed (packed int4) weights
    void set_unpacked_params(torch::Tensor qweight_packed, torch::Tensor scale, torch::Tensor zero_point);

    // Explicitly import weights to XDNA (for benchmark/testing)
    void import_weights_to_xdna();

    // Accessors
    int64_t in_features() const { return in_features_; }
    int64_t out_features() const { return out_features_; }
    torch::Tensor get_quantized_weights() const { return quantized_weight_; }
    torch::Tensor get_scales() const { return scale_; }
    torch::Tensor get_zeros() const { return zero_point_; }

  private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t max_seq_len_;            // Max sequence length for buffer pre-allocation
    torch::Tensor quantized_weight_; // Packed int4 weights [Out, In/2]
    torch::Tensor scale_;            // Scales (per-channel or grouped)
    torch::Tensor zero_point_;       // Zero points (per-channel or grouped)
    torch::Tensor bias_;             // Optional bias [out_features]

    // Persistent buffers for decode execution to avoid dynamic re-allocation
};
TORCH_MODULE(QuantizedLinear);

// Custom LM Head Layer (BF16)
// optimized for high-bandwidth generation using custom HIP kernels
class LmHeadLinearImpl : public torch::nn::Module {
  public:
    LmHeadLinearImpl(int64_t in_features, int64_t out_features, int64_t max_batch_size = 1, int64_t max_seq_len = 8192);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor weight;

  private:
    torch::Tensor logits_buffer;
};
TORCH_MODULE(LmHeadLinear);

// RMSNorm implementation
class RMSNormImpl : public torch::nn::Module {
  public:
    RMSNormImpl(int64_t dim, float eps);

    torch::Tensor forward(torch::Tensor x);
    void forward_out(torch::Tensor output, torch::Tensor x);

    // Setter for weight loading
    void set_weight(torch::Tensor weight);

  private:
    float eps_;
    torch::Tensor weight_;
};
TORCH_MODULE(RMSNorm);

// Unified LLM Model Implementation for w4a16 quantization
// Custom HIP Embedding Layer
// optimized to skip unnecessary tensor copies
class HipEmbeddingImpl : public torch::nn::Module {
  public:
    HipEmbeddingImpl(int64_t num_embeddings, int64_t embedding_dim, int64_t max_batch_size = 1, int64_t max_seq_len = 8192);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor weight;

  private:
    torch::Tensor output_buffer;
};
TORCH_MODULE(HipEmbedding);

#include "unified_llm_w4a16_predict/npuSetup.hpp"

class UnifiedLLMW4A16Impl : public torch::nn::Module {
  public:
    UnifiedLLMW4A16Impl(ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                        int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads, int64_t head_dim,
                        float rms_norm_eps, float rope_theta, const NPUGlobalConfig &npu_config, int64_t max_seq_len = 8192,
                        int64_t max_batch_size = 1, int64_t groupsize = 128, int64_t num_experts = 0, int64_t num_experts_per_tok = 0,
                        torch::Device device = torch::kCPU);

    // Forward pass: takes token IDs and returns logits
    torch::Tensor forward(torch::Tensor input_ids, int64_t start_pos = 0);

    // Generate tokens: takes prompt, generates max_new_tokens, returns all token IDs
    torch::Tensor generate(torch::Tensor input_ids, int64_t max_new_tokens, float temperature = 1.0f, float top_p = 0.9f,
                           int64_t top_k = 50, int64_t eos_token_id = -1);

    // Load quantized weights from safetensors file
    void load_quantized_weights_from_safetensors(const std::string &filename);
    // Load non-quantized weights only (embeddings, norms, lm_head) from safetensors
    void load_non_quantized_weights_from_safetensors(const std::string &filename);
    // Load quantized weights from preprocessed bin directory
    void load_quantized_weights_from_bins(const std::string &weights_dir);

    // Move model to device
    // Move model to device
    UnifiedLLMW4A16Impl &to(torch::Device device);

    // Initialize NPU configuration and resources
    int initialize_npu();

    // Import weights to NPU (must be called after loading weights)
    void import_weights();

    // Initialize all weights with dummy values (random) for testing without loading files
    void initialize_dummy_weights(int seed = 42);

    // NPU Helper functions
    // We declare them as friends or static/global if they are not members
    // But they are global in npuSetup.cpp.
    // So we should declare them outside the class or as static members if we moved them.
    // Since they are global in npuSetup.cpp, we declare them here as global functions.
  private:
    ArchitectureType arch_type_;
    int64_t vocab_size_;
    int64_t hidden_size_;
    int64_t intermediate_size_;
    int64_t num_hidden_layers_;
    int64_t num_attention_heads_;
    int64_t num_key_value_heads_;
    int64_t head_dim_;
    float rms_norm_eps_;
    float rope_theta_;
    int64_t max_seq_len_;
    int64_t max_batch_size_;
    int64_t groupsize_;
    int64_t GQA_head_ratio_;
    int64_t num_experts_;
    int64_t num_experts_per_tok_;
    NPUGlobalConfig npu_config_;
    bool warmup_;

    // Common components
    HipEmbedding token_embedding{nullptr};
    std::vector<QuantizedLinear> q_layers;
    std::vector<QuantizedLinear> k_layers;
    std::vector<QuantizedLinear> v_layers;
    std::vector<QuantizedLinear> o_layers;

    // MLP layers
    std::vector<QuantizedLinear> gate_layers;
    std::vector<QuantizedLinear> up_layers;
    std::vector<QuantizedLinear> down_layers;

    // LLaMA3 specific MLP layers (unified approach might not need separate vectors if logic handles it,
    // but UnifiedLLM.cpp used separate vectors for LLaMA3.
    // However, looking at UnifiedLLM.cpp, LLaMA3 uses gate/up/down just like others.
    // I can reuse gate_layers/up_layers/down_layers for all architectures if I map them correctly during init and
    // weight loading.) Let's reuse the existing vectors to keep it simple, but we need to know which is which.

    // Normalization layers
    std::vector<RMSNorm> input_norms;
    std::vector<RMSNorm> post_attn_norms;
    RMSNorm final_norm{nullptr};

    // Output layer (can be quantized or regular)
    LmHeadLinear lm_head{nullptr};

    // KV caches
    std::vector<torch::Tensor> caches_k;
    std::vector<torch::Tensor> caches_v;

    // Scratch buffers
    torch::Tensor x_buffer;
    torch::Tensor gate_buffer;
    torch::Tensor up_buffer;
    torch::Tensor output_buffer;
    torch::Tensor hidden_states_buffer;
    torch::Tensor queries_buffer;
    torch::Tensor keys_buffer;
    torch::Tensor values_buffer;
    torch::Tensor attn_output_heads_buffer;
    torch::Tensor attn_output_buffer;
    torch::Tensor attn_output_proj_buffer;
    torch::Tensor norm_buffer;

    // RoPE embedding
    torch::Tensor compute_rope_freqs(int64_t seq_len, int64_t start_pos);
    std::pair<torch::Tensor, torch::Tensor> apply_rotary_emb(const torch::Tensor &xq, const torch::Tensor &xk,
                                                             const torch::Tensor &freqs_cis);

    // Architecture-specific forward methods
    torch::Tensor forward_llama3(torch::Tensor x, int64_t start_pos);

    // Activation functions
    torch::Tensor silu(const torch::Tensor &x);
    torch::Tensor gelu(const torch::Tensor &x);
    torch::Tensor swiglu(const torch::Tensor &gate, const torch::Tensor &up);
};

// Global NPU functions
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes);
std::pair<int, int> get_npu_context(int M, int K, int N);
int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   uint32_t output_xdna_handle, uint32_t input_xdna_handle, uint32_t weight_xdna_handle, void *hip_event);

// Reference Implementation Functions
torch::Tensor reference_dequantize_weights(torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                                           int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

torch::Tensor reference_gemm(torch::Tensor input, torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                             int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

TORCH_MODULE(UnifiedLLMW4A16);
