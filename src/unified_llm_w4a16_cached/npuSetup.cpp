#include "unified_llm_w4a16_cached/npuSetup.hpp"

#include <iostream>
#include <string>

std::string hw_target = "gpu";
int debug_verbosity = 1;
bool dummy_weights_enabled = false;
bool warmup_enabled = true;
bool preload_moe_kernels_enabled = false;
bool minimal_pdi = false;

bool rope_scaling_enabled = false;
std::string rope_scaling_type;
float rope_scaling_factor = 1.0f;
float rope_scaling_low_freq_factor = 1.0f;
float rope_scaling_high_freq_factor = 1.0f;
float rope_scaling_original_max_position_embeddings = 0.0f;

// Configure NPU globals from struct
void configure_npu(const NPUGlobalConfig &config) {
    hw_target = config.heterogeneity;
    debug_verbosity = config.debug_verbosity;
    dummy_weights_enabled = config.dummy_weights;
    warmup_enabled = config.warmup;
    minimal_pdi = config.minimal_pdi;
    preload_moe_kernels_enabled = config.preload_moe_kernels;

    if (debug_verbosity >= 1) {
        std::cout << "Configuring NPU (Base):" << std::endl;
        std::cout << "  Heterogeneity: " << hw_target << std::endl;
        std::cout << "  GPU Count: " << config.gpu_count << std::endl;
        std::cout << "  Debug Verbosity: " << debug_verbosity << std::endl;
        std::cout << "  Dummy Weights: " << (dummy_weights_enabled ? "true" : "false") << std::endl;
        std::cout << "  Warmup: " << (warmup_enabled ? "true" : "false") << std::endl;
        std::cout << "  Preload MoE: " << (preload_moe_kernels_enabled ? "true" : "false") << std::endl;
        std::cout << "  Minimal PDI: " << (minimal_pdi ? "true" : "false") << std::endl;
    }

    // RoPE config
    rope_scaling_enabled = config.rope_scaling.enabled;
    if (rope_scaling_enabled) {
        rope_scaling_type = config.rope_scaling.type;
        rope_scaling_factor = config.rope_scaling.factor;
        rope_scaling_low_freq_factor = config.rope_scaling.low_freq_factor;
        rope_scaling_high_freq_factor = config.rope_scaling.high_freq_factor;
        rope_scaling_original_max_position_embeddings = config.rope_scaling.original_max_position_embeddings;

        if (debug_verbosity >= 1) {
            std::cout << "  RoPE Scaling: Enabled (" << rope_scaling_type << ")" << std::endl;
        }
    } else {
        rope_scaling_enabled = false;
        rope_scaling_type.clear();
        rope_scaling_factor = 1.0f;
        rope_scaling_low_freq_factor = 1.0f;
        rope_scaling_high_freq_factor = 1.0f;
        rope_scaling_original_max_position_embeddings = 0.0f;
    }
}

void read_npu_config(const std::string &config_path) {
    std::cerr << "Warning: read_npu_config(string) is deprecated. Use configure_npu(NPUGlobalConfig)" << std::endl;
}

void load_npu_kernels(int drv_fd, const std::string &config_path) {
    std::cerr << "Warning: load_npu_kernels(int, string) is deprecated." << std::endl;
}

void load_npu_kernels(int drv_fd, const NPUGlobalConfig &config) {
    (void)drv_fd;
    (void)config;
    if (debug_verbosity >= 1) {
        std::cout << "Skipping PDI/instruction loading for base backend." << std::endl;
    }
}
