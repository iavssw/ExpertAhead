#pragma once

#include <string>
#include <vector>

// Global hardware target setting from config ("gpu", "npu", or "hetero")
extern std::string hw_target;

// Global debug verbosity level from config (1=minimal, 2=moderate, 3=verbose)
extern int debug_verbosity;

// Global dummy weights flag
extern bool dummy_weights_enabled;

// Global warmup flag
extern bool warmup_enabled;

// Preload MoE kernels (topk/one_hot/index_add) for cold-start performance
extern bool preload_moe_kernels_enabled;

// Global minimal PDI flag
extern bool minimal_pdi;

// Global RoPE scaling settings (e.g., Llama3)
extern bool rope_scaling_enabled;
extern std::string rope_scaling_type;
extern float rope_scaling_factor;
extern float rope_scaling_low_freq_factor;
extern float rope_scaling_high_freq_factor;
extern float rope_scaling_original_max_position_embeddings;

struct NPUKernelConfig {
    bool use = true;
    int npuM = 0;
    int npuK = 0;
    int npuN = 0;
    int forM = 0;
    int forK = 0;
    int forN = 0;
    std::string layer_type;
    int config = -1;
    int num_tiles = 1;

    // Path options
    std::string xclbin;
    std::string inst;

    // Dynamic path construction
    std::string fw_path;
    std::string tile_size;
    std::string col;
    std::string dtype;
};

struct NPURopeScalingConfig {
    bool enabled = false;
    std::string type;
    float factor = 1.0f;
    float low_freq_factor = 1.0f;
    float high_freq_factor = 1.0f;
    float original_max_position_embeddings = 0.0f;
};

struct NPUGlobalConfig {
    std::string heterogeneity = "gpu";
    int gpu_count = 1;
    int debug_verbosity = 1;
    bool dummy_weights = false;
    bool warmup = true;
    bool preload_moe_kernels = false;
    bool minimal_pdi = false;
    // -2: use model default, -1: disable replacement, >=0: replace ranks >= this index
    int random_replace_rank_start_idx = -2;

    NPURopeScalingConfig rope_scaling;
    std::vector<NPUKernelConfig> kernels;
};

// Configure NPU globals from struct
void configure_npu(const NPUGlobalConfig &config);

// Placeholder for loading PDI/instructions (no-op in base backend)
void load_npu_kernels(int drv_fd, const NPUGlobalConfig &config);
