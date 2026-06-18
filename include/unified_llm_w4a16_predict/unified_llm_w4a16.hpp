#pragma once

#include <cstdint>
#include <future>
#include <string>
#include <torch/torch.h>
#include <utility>
#include <vector>
#include <deque>
#include <mutex>
#include <future>
#include <unordered_set>
#include <unified_llm_w4a16_predict/expert_predictor.h>

// Attention mechanism default (can be overridden at runtime by heterogeneity config):
// 0 = Manual matmul, 1 = PyTorch SDPA, 2 = Custom HIP kernel
#define ATTENTION_BACKEND 2

// Architecture type enum
enum class ArchitectureType { MIXTRAL, QWEN };

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
    void forward(torch::Tensor output_buffer, torch::Tensor input, std::string layer_type);

    // Forward pass with internal allocation
    torch::Tensor forward(torch::Tensor input, std::string layer_type);

    // Set quantized weights (for loading from state dict)
    void set_quantized_weights(torch::Tensor qweight, torch::Tensor scale, torch::Tensor zero_point, torch::Tensor g_idx = torch::Tensor());
    // Directly set preprocessed (packed int4) weights
    void set_unpacked_params(torch::Tensor qweight_packed, torch::Tensor scale, torch::Tensor zero_point);

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
    void set_use_hip(bool use_hip) { use_hip_ = use_hip; }
    torch::Tensor weight;

  private:
    torch::Tensor logits_buffer;
    bool use_hip_ = true;
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
    void set_use_hip(bool use_hip) { use_hip_ = use_hip; }
    torch::Tensor weight;

  private:
    torch::Tensor output_buffer;
    bool use_hip_ = true;
};
TORCH_MODULE(HipEmbedding);

// Linear Matmul Layer (replacement for torch::nn::Linear for MoE router)
class LinearMatmulImpl : public torch::nn::Module {
  public:
    LinearMatmulImpl(int64_t in_features, int64_t out_features, bool bias = true);
    torch::Tensor forward(torch::Tensor input);
    torch::Tensor weight;
    torch::Tensor bias;
};
TORCH_MODULE(LinearMatmul);

// Mixtral MoE layer (router + experts)
class MixtureOfExpertsImpl : public torch::nn::Module {
  public:

    MixtureOfExpertsImpl(int64_t hidden_size, int64_t intermediate_size, int64_t num_experts, int64_t num_experts_per_tok,
                         int64_t max_cached_experts, int64_t layer_idx, 
                         int64_t max_seq_len = 8192, bool use_softmax_before_topk = false, bool normalize_topk_prob = false,
                         double lambda = 0.0, const std::string& predictor_model_path = "", torch::Device predictor_device = torch::kCPU,
                         int64_t prefetch_experts_count = 1);

    torch::Tensor forward(const torch::Tensor &x, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt);
    void set_weights_dir(const std::string& dir) { weights_dir_ = dir; }
    void prefill_cache_for_testing();
    void prewarm_experts(int64_t num_to_warm);
    
    // Lambda parameter control (router logit biasing)
    void set_lambda(double lambda) { 
        if (lambda < 0.0 || lambda > 100.0) {
            throw std::invalid_argument("Lambda must be in range [0, 100], got: " + std::to_string(lambda));
        }
        lambda_ = lambda; 
    }
    double get_lambda() const { return lambda_; }

    // Forced top-N guarantee: how many unbiased top experts are always kept in the bias mask.
    // Default 1 matches original behavior. Set to num_experts_per_tok to never override routing.
    void set_forced_top_n(int64_t n) { forced_top_n_ = std::max(int64_t(0), n); }
    int64_t get_forced_top_n() const { return forced_top_n_; }

    // Forced top-P: force the minimum set of experts whose softmax probabilities sum to >= p.
    // Adapts to routing confidence: peaked distributions force fewer experts than flat ones.
    // Set to -1.0 to disable (default). Mutually composable with forced_top_n_.
    void set_forced_top_p(double p) { forced_top_p_ = p; }
    double get_forced_top_p() const { return forced_top_p_; }

    // Per-token probability-mass prefix threshold p: smallest sorted-prob prefix with mass >= p is
    // OR'd into the cache-conditional bias mask (with forced_top_n / forced_top_p). Experts in the
    // mask get +lambda*delta_avg; remaining top-k slots come from biased top-k (same path as FN).
    // Requires lambda>0 for effect. Set to -1.0 to disable.
    void set_mass_threshold_substitution_p(double p) { mass_threshold_substitution_p_ = p; }
    double get_mass_threshold_substitution_p() const { return mass_threshold_substitution_p_; }

    enum class CachePolicy { LRU, MRU, LFU, MFU, CLOCK, RANDOM, LFRU, PREFILL };
    void set_cache_policy(CachePolicy policy) { cache_policy_ = policy; }
    
    // Number of top experts to lock into cache during prefill.
    void set_prefill_top_n(int64_t n) { prefill_top_n_ = std::max(int64_t(0), n); }
    int64_t get_prefill_top_n() const { return prefill_top_n_; }

    // Experiment mode: keep top forced_top_n_ correct experts, fill remaining slots with random experts.
    // Used to measure perplexity impact of substituting lower-ranked active experts.
    void set_random_fill_mode(bool on) { random_fill_mode_ = on; }
    
    // Correlation-based expert tracking
    // We still keep correlation constant API around if something calls it but ignore it, or remove it. Let's remove it.
    torch::Tensor get_last_router_logits() const { return last_router_logits_; }
    
    // Cache stats
    void print_cache_stats() const;
    std::pair<int64_t, int64_t> get_cache_stats() const { return {cache_hits_, cache_misses_}; }
    void reset_cache_stats();

    // Predictor hit-rate stats (generation only)
    std::tuple<int64_t, int64_t> get_predictor_stats() const {
        return {pred_hits_no_bias_, pred_total_};
    }
    void reset_predictor_stats() {
        pred_hits_no_bias_ = 0;
        pred_total_ = 0;
        pred_hits_routed_forced_n_ = 0;
        pred_total_routed_forced_n_ = 0;
        pred_hits_routed_topk_ = 0;
        pred_total_routed_topk_ = 0;
        pred_requested_hits_forced_n_ = 0;
        pred_requested_total_forced_n_ = 0;
        pred_requested_hits_topk_ = 0;
        pred_requested_total_topk_ = 0;
        pred_hits_window_recall_ = 0;
        pred_total_window_recall_ = 0;
        pred_hits_window_precision_ = 0;
        pred_total_window_precision_ = 0;
        last_true_top1_expert_ = -1;  // Reset so first token doesn't count
        decode_token_count_ = 0;      // Reset stride counter for new generation
        decode_step_counter_ = 0;
        pred_results_ready_.store(false);
        {
            std::lock_guard<std::mutex> lock(pred_results_mutex_);
            last_pred_no_bias_.clear();
        }
        {
            std::lock_guard<std::mutex> lock(pending_predictions_mutex_);
            pending_predictions_.clear();
        }
    }

    // Lookahead stride: only invoke the predictor every N tokens (N = lookahead depth).
    // Default 1 = run every token (original behaviour).
    void set_lookahead_stride(int64_t stride) { lookahead_stride_ = std::max(int64_t(1), stride); }
    int64_t get_lookahead_stride() const { return lookahead_stride_; }

    // Sequential top1 caching stats (generation only)
    std::tuple<int64_t, int64_t> get_sequential_top1_stats() const {
        return {sequential_top1_hits_, sequential_top1_total_};
    }
    void reset_sequential_top1_stats() {
        sequential_top1_hits_ = 0;
        sequential_top1_total_ = 0;
        last_top1_expert_ = -1;
    }

    // Prediction & Speculative Loading
    void set_context_token_ids(const std::vector<int64_t>& token_ids);
    void trigger_speculative_loading(const torch::Tensor& embedding, int64_t source_decode_step,
                                     c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt);
    /// One-shot predictor + prefetch after prefill (uses last / prev-prefill-token routing; does not use decode prev_token state).
    void run_predictor_prefill_warmup(const torch::Tensor& embedding, const torch::Tensor& prefill_dist_row,
                                    const torch::Tensor& prev_expert_mh_row, c10::optional<torch::Tensor> prev_layers_feat);
    void load_predicted_experts(const std::vector<int64_t>& predicted_expert_ids);

    bool has_predictor() const { return predictor_ != nullptr; }
    void set_suppress_predictor_stats(bool v) { suppress_predictor_stats_ = v; }
    void wait_for_speculative_idle() {
        if (speculative_load_future_.valid()) {
            speculative_load_future_.wait();
        }
    }
    torch::Tensor get_prefill_expert_counts() const { return prefill_expert_counts_; }

    torch::Tensor routing_mh_current_token_cpu() const { return routing_mh_current_token_; }
    torch::Tensor prefill_last_token_mh_cpu() const { return prefill_last_token_mh_; }
    torch::Tensor prefill_prev_token_mh_cpu() const { return prefill_prev_token_mh_; }

    // Exposed for weight loading
    LinearMatmul router{nullptr};
    
    // Changed: These now represent the *slots* in the cache, not the logical experts.
    // Size will be max_cached_experts.
    std::vector<QuantizedLinear> gate_up_experts;
    std::vector<QuantizedLinear> down_experts;

  private:
    // Pinned memory buffers for fast expert weight loading (one per slot)
    std::vector<torch::Tensor> gate_up_q_pinned_;
    std::vector<torch::Tensor> gate_up_s_pinned_;
    std::vector<torch::Tensor> gate_up_z_pinned_;
    std::vector<torch::Tensor> down_q_pinned_;
    std::vector<torch::Tensor> down_s_pinned_;
    std::vector<torch::Tensor> down_z_pinned_;

    int64_t hidden_size_;
    int64_t intermediate_size_;
    int64_t num_experts_;
    int64_t num_experts_per_tok_;
    int64_t max_cached_experts_;
    int64_t layer_idx_;
    int64_t prefetch_experts_count_;
    bool use_softmax_before_topk_;
    bool normalize_topk_prob_;

    std::string weights_dir_;

    // Router logit biasing (lambda parameter)
    double lambda_ = 0.0;                        // Bias parameter [0, 1]
    double delta_avg_ = 0.0;                     // Running average of logit ranges
    std::vector<int64_t> expert_cache_bitmask_;  // Binary mask of cached experts
    int64_t forced_top_n_ = 1;                   // How many unbiased top-k experts are forced into the mask
    double  forced_top_p_ = -1.0;               // Cumulative prob mass threshold for forced experts (-1 = disabled)
    double  mass_threshold_substitution_p_ = -1.0; // Alternate routing threshold (-1 = disabled)
    int64_t prefill_top_n_ = 0;                  // How many top experts from prefill to lock in cache
    bool random_fill_mode_ = false;               // Experiment: substitute non-top-N slots with random experts

    CachePolicy cache_policy_ = CachePolicy::LRU;
    std::vector<int64_t> locked_experts_;        // Experts locked by the PREFILL policy
    std::vector<int64_t> currently_selected_experts_; // Experts currently selected to prevent their eviction

    // Prefill distribution tracking
    torch::Tensor prefill_expert_counts_;

    // Cache State
    std::vector<int64_t> expert_slots_indices; // Maps Slot ID [0..max_cached] -> Global Expert ID. -1 if empty.

    struct ExpertSlotMeta {
        int64_t expert_id = -1;    // global expert in this slot (-1 = empty)
        uint64_t access_count = 0; // for LFU/MFU/LFRU
        uint64_t last_access = 0;  // for LRU/MRU/LFRU
        uint8_t clock_bit = 0;     // for CLOCK algorithm
    };

    std::vector<ExpertSlotMeta> slot_meta_; // Per-slot metadata (matches cached backend)
    uint64_t access_clock_ = 0;             // global logical clock for recency
    size_t clock_hand_ = 0;                 // clock hand for CLOCK eviction
    
    int64_t cache_hits_ = 0;
    int64_t cache_misses_ = 0;
    std::atomic<double> total_expert_load_time_ms_{0.0};
    
    // Tracking loads per step
    std::atomic<int64_t> experts_loaded_this_step_{0};
    int64_t total_steps_0_loaded_ = 0;
    int64_t total_steps_1_loaded_ = 0;
    int64_t total_steps_gt1_loaded_ = 0;

    int64_t stall_loads_ = 0;
    /** SSD reads started on the speculative prefetch path (issued, not necessarily hidden). */
    int64_t prefetch_loads_ = 0;
    /** Router access: expert was prefetched and slot was ready (no wait) — maps to M_prefetch hides. */
    int64_t prefetch_hits_ready_ = 0;
    /** Router access: expert was prefetched but decode waited on in-flight load — overlap/M_cap limited. */
    int64_t prefetch_hits_wait_ = 0;
    /** Predictor tick skipped because previous async prefetch batch still running. */
    int64_t prefetch_ticks_skipped_ = 0;
    // Per-slot provenance: 0=unknown/prewarm, 1=prefetch, 2=stall (main-thread miss).
    std::vector<uint8_t> slot_load_origin_;

    // Tracker for how many of the prefetched experts were correctly in the actual Top-K chosen
    int64_t pred_match_0_ = 0;
    int64_t pred_match_1_ = 0;
    int64_t pred_match_2_ = 0;

    // Predictor accuracy counters (generation only)
    // Definition:
    //   denominator = number of routed experts evaluated per token
    //                 (forced_top_n_ if >0, otherwise full routed top-k size).
    //   numerator   = among those routed experts, how many are present in the
    //                 predictor's prefetched set at the configured horizon.
    int64_t pred_hits_no_bias_   = 0;  // legacy aggregate (kept for compatibility)
    int64_t pred_total_          = 0;  // legacy aggregate (kept for compatibility)
    int64_t pred_hits_routed_forced_n_ = 0;
    int64_t pred_total_routed_forced_n_ = 0;
    int64_t pred_hits_routed_topk_ = 0;
    int64_t pred_total_routed_topk_ = 0;
    // Precision-style metric: of predicted experts, how many were requested by router.
    int64_t pred_requested_hits_forced_n_ = 0;
    int64_t pred_requested_total_forced_n_ = 0;
    int64_t pred_requested_hits_topk_ = 0;
    int64_t pred_requested_total_topk_ = 0;
    // Window-aware precision/recall: evaluate predicted set against the UNION of
    // actual routed experts across all tokens in the lookahead window [t+1, t+N].
    int64_t pred_hits_window_recall_    = 0;  // |pred_set ∩ actual_window|
    int64_t pred_total_window_recall_   = 0;  // |actual_window|
    int64_t pred_hits_window_precision_ = 0;  // |pred_set ∩ actual_window|
    int64_t pred_total_window_precision_= 0;  // |pred_set|
    int64_t last_true_top1_expert_ = -1; // unbiased top-1 from previous token (ground truth for predictor stat)

    // Pending prediction queue entry: accumulates the actual expert union over the
    // lookahead window so precision/recall can be evaluated against the full window.
    struct PendingPrediction {
        int64_t target_step;                        // decode step at which to finalize (= source + lookahead_stride_)
        std::vector<int64_t> pred_set;              // predicted expert IDs (top-B)
        std::unordered_set<int64_t> actual_union;   // union of actual routed experts accumulated so far
    };
    std::deque<PendingPrediction> pending_predictions_;
    std::mutex pending_predictions_mutex_;
    
    // Sequential top1 tracking
    int64_t sequential_top1_hits_ = 0;
    int64_t sequential_top1_total_ = 0;
    int64_t last_top1_expert_ = -1;
    
    // Ranked predictions from the previous token (set in async lambda, read next token)
    std::vector<int64_t> last_pred_no_bias_;
    std::mutex pred_results_mutex_;
    std::atomic<bool> pred_results_ready_{false};
    // (replaced by PendingPrediction deque above)
    
    // Training data collection
    mutable torch::Tensor last_router_logits_;  // Store last router logits for training data collection
    
    // Unbiased routing features (float32, CPU) for predictor inputs — matches training multi-hot + normalize.
    torch::Tensor routing_mh_current_token_;  // current forward row (decode: one token; prefill: last row)
    torch::Tensor prev_token_routing_mh_;      // previous token (for decode predictor "prev" branch)
    torch::Tensor prefill_last_token_mh_;      // last prefill position, per layer
    torch::Tensor prefill_prev_token_mh_;      // second-to-last prefill position (prompt_len >= 2)
    bool suppress_predictor_stats_ = false;    // prefill-end warmup: do not count predictor hits / prefetch loads in stats
    int64_t lookahead_stride_ = 1;             // invoke predictor every N decode tokens (N = lookahead depth)
    int64_t decode_token_count_ = 0;           // counts decode tokens since last predictor call
    int64_t decode_step_counter_ = 0;          // absolute decode step index within current generation

    // Prediction & Speculative Loading
    std::unique_ptr<IExpertPredictor> predictor_;
    std::vector<int64_t> recent_token_ids_;
    std::deque<torch::Tensor> recent_embeddings_;
    std::future<void> speculative_load_future_;
    bool in_generation_mode_ = false;
    std::mutex expert_slots_mutex_;  // For thread safety during loading
    std::vector<bool> expert_slot_ready_; // For condition variable, size max_cached_experts_
    std::condition_variable expert_slots_cv_; // To wait for background loading
    
    // Expert file format (auto-detected on first load)
    enum class ExpertFormat { UNKNOWN, UNPACKED, PACKED };

    // Pinned staging buffer reused across packed loads (avoids per-call allocation)
    torch::Tensor expert_staging_buf_;
    ExpertFormat  expert_format_ = ExpertFormat::UNKNOWN;

    void load_expert_weights(int64_t slot_idx, int64_t expert_idx, const std::string& weights_dir);
    void load_expert_weights_packed(int64_t slot_idx, int64_t expert_idx, const std::string& weights_dir);
    int64_t ensure_expert_cached(int64_t global_expert_idx, bool update_stats = true);
    size_t pick_victim_ready();
    size_t pick_lru_ready();
    size_t pick_mru_ready();
    size_t pick_lfu_ready();
    size_t pick_mfu_ready();
    size_t pick_clock_ready();
    size_t pick_random_ready();
    size_t pick_lfru_ready();
    void update_cache_bitmask_locked();

    torch::Tensor forward_cpu(const torch::Tensor &x_flat, const torch::Tensor &topk_vals, const torch::Tensor &topk_idx,
                              torch::Tensor &output);
    torch::Tensor forward_generation(const torch::Tensor &x_flat, const torch::Tensor &topk_vals, const torch::Tensor &topk_idx,
                                     torch::Tensor &output, int64_t current_true_top1,
                                     c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt);
    torch::Tensor forward_prefill(const torch::Tensor &x_flat, const torch::Tensor &topk_vals, const torch::Tensor &topk_idx,
                                  torch::Tensor &output);
};
TORCH_MODULE(MixtureOfExperts);

#include "unified_llm_w4a16_predict/npuSetup.hpp"

class UnifiedLLMW4A16Impl : public torch::nn::Module {
  public:
    UnifiedLLMW4A16Impl(ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                        int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads, int64_t head_dim,
                        float rms_norm_eps, float rope_theta, const NPUGlobalConfig &npu_config, int64_t max_seq_len = 8192,
                        int64_t max_batch_size = 1, int64_t groupsize = 128, int64_t num_experts = 0, int64_t num_experts_per_tok = 0,
                        torch::Device device = torch::kCPU, int64_t max_cached_experts_per_layer = 0,
                        const std::string& predictor_model_path = "", int64_t prefetch_experts_count = 1,
                        const std::vector<int>& predict_layers = {},
                        const std::vector<int64_t>& per_layer_cache_sizes = {},
                        const std::vector<int64_t>& per_layer_prefetch_counts = {});

    // Forward pass: takes token IDs and returns logits
    torch::Tensor forward(torch::Tensor input_ids, int64_t start_pos = 0);

    // Generate tokens: takes prompt, generates max_new_tokens, returns all token IDs
    torch::Tensor generate(torch::Tensor input_ids, int64_t max_new_tokens, float temperature = 1.0f, float top_p = 0.9f,
                           int64_t top_k = 50, int64_t eos_token_id = -1);

    // Calculate generation perplexity (NLL) by simulating sequential token generation (step-by-step)
    double calculate_generation_perplexity(torch::Tensor input_ids);

    // Load quantized weights from safetensors file
    void load_quantized_weights_from_safetensors(const std::string &filename);
    // Load non-quantized weights only (embeddings, norms, lm_head) from safetensors
    void load_non_quantized_weights_from_safetensors(const std::string &filename);
    // Load quantized weights from preprocessed bin directory.
    // If expert_weights_dir is non-empty it overrides weights_dir for MoE expert
    // files (supports both packed and unpacked formats — auto-detected at runtime).
    void load_quantized_weights_from_bins(const std::string &weights_dir,
                                          const std::string &expert_weights_dir = "");

    // Pre-warm expert cache
    void prewarm_experts(int64_t num_to_warm, bool verbose = true);
    
    // Lambda parameter control for router logit biasing
    void set_lambda(double lambda, int64_t layer_idx = -1);
    double get_lambda(int64_t layer_idx = 0) const;

    // Forced top-N / top-P and random-fill experiment controls (applied to all layers)
    void set_forced_top_n(int64_t n);
    void set_forced_top_p(double p);
    void set_mass_threshold_substitution_p(double p);
    void set_random_fill_mode(bool on);
    void set_prefill_top_n(int64_t n);
    void set_cache_policy(const std::string& policy_name, int64_t layer_idx = -1);

    // Move model to device
    // Move model to device
    UnifiedLLMW4A16Impl &to(torch::Device device);

    // Initialize all weights with dummy values (random) for testing without loading files
    void initialize_dummy_weights(int seed = 42);

    void print_cache_stats() const;
    void reset_cache_stats();
    std::pair<int64_t, int64_t> get_cache_stats() const;  // Returns (total_hits, total_misses)

    // Predictor hit-rate stats across all MoE layers
    // Each element: (hits, total) for that layer
    std::vector<std::tuple<int64_t, int64_t>> get_predictor_stats() const;
    void reset_predictor_stats();
    void set_suppress_predictor_stats(bool v);

    // Set how often the predictor fires: every `stride` decode tokens.
    // Call this after construction with stride = lookahead depth (fN from predictor path).
    void set_predictor_lookahead(int64_t stride);

    std::vector<std::tuple<int64_t, int64_t>> get_sequential_top1_stats() const;
    void reset_sequential_top1_stats();
    
    // Training data collection
    void enable_training_data_collection() { collect_training_data_ = true; }
    void disable_training_data_collection() { collect_training_data_ = false; }
    std::vector<std::pair<torch::Tensor, torch::Tensor>> get_training_data(); // Returns [(embeddings, router_logits), ...]
    void clear_training_data();

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
    
    // Training data collection
    bool collect_training_data_ = false;
    bool measurement_suppressed_globally_ = false;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> training_data_;  // [(post_attn_norm_embeddings, router_logits), ...]
    int attention_mode_; // 0=manual matmul, 1=PyTorch SDPA, 2=Custom HIP kernel
    bool multi_gpu_enabled_ = false;
    int gpu_count_ = 1;
    torch::Device embedding_device_ = torch::kCPU;
    torch::Device output_device_ = torch::kCPU;
    std::vector<torch::Device> layer_devices_;

    // Common components
    HipEmbedding token_embedding{nullptr};
    std::vector<QuantizedLinear> q_layers;
    std::vector<QuantizedLinear> k_layers;
    std::vector<QuantizedLinear> v_layers;
    std::vector<QuantizedLinear> o_layers;
    std::vector<RMSNorm> q_norms;
    std::vector<RMSNorm> k_norms;

    // MLP layers

    // Mixtral MoE layers
    std::vector<MixtureOfExperts> moe_layers;

    // Note: gate_layers/up_layers/down_layers have been removed. Mixtral uses MoE layers instead.
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

    // Preload MoE kernels to avoid cold-start spikes
    void preload_moe_kernels();

    // Architecture-specific forward methods

    torch::Tensor forward_mixtral_multi_gpu(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_mixtral(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen_multi_gpu(torch::Tensor x, int64_t start_pos);
    torch::Tensor forward_qwen(torch::Tensor x, int64_t start_pos);

    /// After a multi-token prefill, run each layer predictor once to prefetch experts for the first decode step.
    void warm_predictor_caches_after_prefill(int64_t prompt_len);

    // Last-token MoE inputs from the most recent forward (for prefill warmup); [hidden_size] per layer, CPU.
    std::vector<torch::Tensor> prefill_last_moe_inputs_cpu_;

    // Activation functions
    torch::Tensor silu(const torch::Tensor &x);
    torch::Tensor gelu(const torch::Tensor &x);
    torch::Tensor swiglu(const torch::Tensor &gate, const torch::Tensor &up);
};

// Reference Implementation Functions
torch::Tensor reference_dequantize_weights(torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                                           int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

torch::Tensor reference_gemm(torch::Tensor input, torch::Tensor quantized_weight, torch::Tensor scale, torch::Tensor zero_point,
                             int64_t in_features, int64_t out_features, torch::Tensor g_idx = torch::Tensor());

TORCH_MODULE(UnifiedLLMW4A16);
