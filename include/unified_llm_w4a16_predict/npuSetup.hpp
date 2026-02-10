#pragma once

#include <array>
#include <cstdint>
// #include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

#include <map>
#include <mutex>
#include <string>
#include <torch/torch.h>
#include <tuple>
#include <vector>

// Use standard types in header, convert to __u32/__u64 in implementation
// This avoids conflicts with system headers that may define these types differently

// XDNA driver file descriptor
extern int xdna_drv_fd;
// (global variable accessible to all functions)

// Global map to cache imported handles: ptr -> handle
extern std::map<void *, uint32_t> ptr_to_handle_map;

// Global hardware target setting from kernels.json ("gpu", "npu", or "hetero")
extern std::string hw_target;

// Global debug verbosity level from kernels.json (1=minimal, 2=moderate, 3=verbose)
extern int debug_verbosity;

// Global dummy weights flag
extern bool dummy_weights_enabled;

// Global warmup flag
extern bool warmup_enabled;

// Global minimal PDI flag
extern bool minimal_pdi;

// Packed/CPU decode flags removed (GPU-only path)
// Global RoPE scaling settings (e.g., Llama3)
extern bool rope_scaling_enabled;
extern std::string rope_scaling_type;
extern float rope_scaling_factor;
extern float rope_scaling_low_freq_factor;
extern float rope_scaling_high_freq_factor;
extern float rope_scaling_original_max_position_embeddings;

// Global NPU mutex for synchronization
extern std::mutex npu_mutex;

// Maximum number of NPU contexts
#define MAX_NPU_HW_CTX 6
#define MAX_NPU_INST_CTX 32

// Initialize NPU context arrays
void init_npu();

// Orchestration function to initialize everything
int initialize_npu();

// Initialize XDNA driver (opens /dev/accel/accel0)
// Returns 0 on success, -1 on failure
int initialize_xdna_driver(const char *drv_path = "/dev/accel/accel0");

// Function to export DMA-BUF and import to xdna
// Returns handle (uint32_t) or 0 on error
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes);

// Function to import all weights and parameters of a PyTorch module to XDNA
void import_all_weights_to_xdna(torch::nn::Module &module);

#include "npu_matmul_func/npu_matmul_func.hpp"

// Key: {forM, forK, forN, layer_id} for lookup
using NPUKey = std::array<int, 4>;

// Value: hardware/instruction context indices, actual kernel dimensions, and config
struct NPUValue {
    int hw_idx;
    int inst_idx;
    int npuM;
    int npuK;
    int npuN;
    int cpuN;
    int cpuThreads;
    int config;
};

extern std::vector<hwctxt> hwctxt_array;
extern std::vector<instctxt> instctxt_array;
extern std::map<NPUKey, NPUValue> config_map;

// Helper to convert layer string to ID
int get_layer_id(const std::string &layer_type);

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
    int debug_verbosity = 1;
    bool dummy_weights = false;
    bool warmup = true;
    bool preload_moe_kernels = false;
    bool minimal_pdi = false;

    NPURopeScalingConfig rope_scaling;
    std::vector<NPUKernelConfig> kernels;
};

// Configure NPU globals from struct
void configure_npu(const NPUGlobalConfig &config);

// Read NPU configuration (debug verbosity, heterogeneity) from JSON (DEPRECATED)
void read_npu_config(const std::string &config_path = "");

// Load NPU kernels from JSON configuration (DEPRECATED)
void load_npu_kernels(int drv_fd, const std::string &config_path = "");

// Load NPU kernels from struct
void load_npu_kernels(int drv_fd, const NPUGlobalConfig &config);

// Get NPU context indices for given dimensions
// Returns {hw_idx, inst_idx} or {-1, -1} if not found
std::pair<int, int> get_npu_context(int M, int K, int N, const std::string &layer_type = "");

int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   uint32_t output_xdna_handle, uint32_t input_xdna_handle, uint32_t weight_xdna_handle, hipEvent_t hip_event = nullptr);
