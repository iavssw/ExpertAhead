#include "unified_llm_w4a16_predict/npuSetup.hpp"
#include <algorithm>

#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <libdrm/drm.h>
#include <map>
#include <mutex>
#include <stdint.h>
#include <sys/ioctl.h>
#include <torch/torch.h>
#include <unistd.h>

// Include actual amdxdna_accel.h header
#ifdef __KERNEL__
#include <drm/drm.h>
#else
#include <libdrm/drm.h>
#endif
#include "amdxdna_accel.h"

// Define IOMMU_STRIDE
#define IOMMU_STRIDE 1024

// Define myBfloat type (bfloat16 as uint16_t)
typedef uint16_t myBfloat;

// XDNA driver file descriptor
// Initialize to -1 (invalid)
int xdna_drv_fd = -1;

// Global map to cache imported handles: ptr -> handle
std::map<void *, uint32_t> ptr_to_handle_map;

// Global hardware target setting from kernels.json
std::string hw_target = "gpu";

// Global debug verbosity level from kernels.json
int debug_verbosity = 1;

// Global dummy weights flag
bool dummy_weights_enabled = false;

// Global warmup flag
bool warmup_enabled = true;

// Global minimal PDI flag
bool minimal_pdi = false;

// Packed weights and split-K flags removed (GPU-only path)

// Global RoPE scaling settings (e.g., Llama3)
bool rope_scaling_enabled = false;
std::string rope_scaling_type;
float rope_scaling_factor = 1.0f;
float rope_scaling_low_freq_factor = 1.0f;
float rope_scaling_high_freq_factor = 1.0f;
float rope_scaling_original_max_position_embeddings = 0.0f;

// Global NPU contexts
std::vector<hwctxt> hwctxt_array;
std::vector<instctxt> instctxt_array;
std::map<NPUKey, NPUValue> config_map;

// Configure NPU globals from struct
void configure_npu(const NPUGlobalConfig &config) {
    hw_target = config.heterogeneity;
    debug_verbosity = config.debug_verbosity;
    dummy_weights_enabled = config.dummy_weights;
    warmup_enabled = config.warmup;
    minimal_pdi = config.minimal_pdi;

    // Set debug verbosity for this module
    set_npu_debug_verbosity(debug_verbosity);

    if (debug_verbosity >= 1) {
        std::cout << "Configuring NPU:" << std::endl;
        std::cout << "  Heterogeneity: " << hw_target << std::endl;
        std::cout << "  Debug Verbosity: " << debug_verbosity << std::endl;
        std::cout << "  Dummy Weights: " << (dummy_weights_enabled ? "true" : "false") << std::endl;
        std::cout << "  Warmup: " << (warmup_enabled ? "true" : "false") << std::endl;
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
    // Root dir for xclbin/inst paths
    const char *env_root_ptr = std::getenv("HETEROMOSAIC_ROOT");
    std::string root_dir = env_root_ptr ? env_root_ptr : "/home/greg/Desktop/heteroMosaic";

    // Track the current index for HW and Inst contexts
    int hw_ctx_count = 0;
    int inst_ctx_count = 0;

    // Cache for PDI reuse: key is {npuK, tile_size, col, dtype} -> hw_idx
    std::map<std::tuple<int, std::string, std::string, std::string>, int> pdi_cache;

    for (const auto &k : config.kernels) {
        if (!k.use) {
            if (debug_verbosity >= 1)
                std::cout << "Skipping kernel (use=false)" << std::endl;
            continue;
        }

        // Skip if npuM is 0 (GPU-only mode)
        if (k.npuM == 0) {
            if (debug_verbosity >= 1) {
                std::cout << "Skipping kernel (npuM=0, GPU-only mode)" << std::endl;
                std::cout << "  forM=" << k.forM << " forK=" << k.forK << " forN=" << k.forN << " layer=" << k.layer_type << std::endl;
            }
            continue;
        }

        bool skip_pdi = false;
        int hw_idx = -1;
        int inst_idx = -1;

        if (!skip_pdi) {
            std::string xclbin = k.xclbin;
            std::string inst = k.inst;
            bool reuse_pdi = false;
            int reused_hw_idx = -1;

            // Props for caching
            std::tuple<int, std::string, std::string, std::string> cache_key;
            bool can_cache = false;

            // If paths are empty, try to construct them from dynamic parts
            if (xclbin.empty() || inst.empty()) {
                if (!k.fw_path.empty() && !k.tile_size.empty() && !k.col.empty() && !k.dtype.empty()) {
                    // Check for PDI reuse
                    if (minimal_pdi) {
                        cache_key = std::make_tuple(k.npuK, k.tile_size, k.col, k.dtype);
                        can_cache = true;
                        if (pdi_cache.find(cache_key) != pdi_cache.end()) {
                            reuse_pdi = true;
                            reused_hw_idx = pdi_cache[cache_key];
                            if (debug_verbosity >= 1) {
                                std::cout << "Reusing PDI for " << k.layer_type << " (hw_idx=" << reused_hw_idx << ")" << std::endl;
                            }
                        }
                    }

                    std::string fw_path = k.fw_path;
                    if (fw_path.back() != '/') {
                        fw_path += "/";
                    }

                    std::string dims_str = std::to_string(k.npuM) + "x" + std::to_string(k.npuK) + "x" + std::to_string(k.npuN);
                    inst = fw_path + "insts_" + dims_str + "_" + k.tile_size + "_" + k.col + "_" + k.dtype + ".txt";

                    if (!reuse_pdi) {
                        xclbin = fw_path + "final_" + dims_str + "_" + k.tile_size + "_" + k.col + "_" + k.dtype + ".pdi";
                    }
                } else {
                    std::cerr << "Kernel definition missing path info" << std::endl;
                    continue;
                }
            }

            std::string inst_path = root_dir + "/" + inst;

            // Load HW Context (PDI)
            if (reuse_pdi) {
                hw_idx = reused_hw_idx;
            } else {
                std::string xclbin_path = root_dir + "/" + xclbin;
                hw_idx = hw_ctx_count;
                if (debug_verbosity >= 1)
                    std::cout << "Loading PDI: " << xclbin_path << " with num_tiles=" << k.num_tiles << std::endl;
                if (createHWctxt(drv_fd, hwctxt_array[hw_idx], xclbin_path.c_str(), k.num_tiles) != 0) {
                    std::cerr << "Failed to create HW context for " << xclbin_path << std::endl;
                    continue;
                }
                FlushCpuCache((const void *)hwctxt_array[hw_idx].pdi_vaddr, 0, hwctxt_array[hw_idx].pdi_size);
                hw_ctx_count++;

                if (minimal_pdi && can_cache) {
                    pdi_cache[cache_key] = hw_idx;
                }
            }

            if (hw_ctx_count >= MAX_NPU_HW_CTX) {
                std::cerr << "Warning: Reached maximum HW contexts (" << MAX_NPU_HW_CTX << ")" << std::endl;
            }

            // Load Instruction Context
            inst_idx = inst_ctx_count;
            if (debug_verbosity >= 1)
                std::cout << "Loading Instructions: " << inst_path << std::endl;
            if (createInstctxt(drv_fd, instctxt_array[inst_idx], inst_path.c_str(), true) != 0) {
                std::cerr << "Failed to create Inst context for " << inst_path << std::endl;
                continue;
            }
            FlushCpuCache((const void *)instctxt_array[inst_idx].dpu_0_vaddr, 0,
                          instctxt_array[inst_idx].num_dpu_0_insts * sizeof(uint32_t));
            inst_ctx_count++;

            if (inst_ctx_count >= MAX_NPU_INST_CTX) {
                std::cerr << "Warning: Reached maximum Inst contexts (" << MAX_NPU_INST_CTX << ")" << std::endl;
            }
        } // End of PDI loading block

        // Update config map with all values
        int layer_id = get_layer_id(k.layer_type);
        NPUKey key = {k.forM, k.forK, k.forN, layer_id};
        config_map[key] = {hw_idx, inst_idx, k.npuM, k.npuK, k.npuN, 0, 1, k.config}; // cpuN=0, cpuThreads=1 for GEMM

        if (debug_verbosity >= 1) {
            std::cout << "Mapped kernel " << k.forM << "x" << k.forK << "x" << k.forN << " layer=" << k.layer_type << "(" << layer_id
                      << ") -> npuM=" << k.npuM << " npuK=" << k.npuK << " npuN=" << k.npuN << " config=" << k.config << " to HW[" << hw_idx
                      << "] Inst[" << inst_idx << "]" << std::endl;
        }
    } // End of for loop

    // Resize arrays to actual size used
    hwctxt_array.resize(hw_ctx_count);
    instctxt_array.resize(inst_ctx_count);

    if (debug_verbosity >= 1)
        std::cout << "Loaded " << hw_ctx_count << " HW contexts and " << inst_ctx_count << " Inst contexts" << std::endl;
}

std::pair<int, int> get_npu_context(int M, int K, int N, const std::string &layer_type) {
    int layer_id = get_layer_id(layer_type);
    NPUKey key = {M, K, N, layer_id};
    auto it = config_map.find(key);
    if (it != config_map.end()) {
        return {it->second.hw_idx, it->second.inst_idx};
    }
    return {-1, -1};
}

// Initialize XDNA driver
int initialize_xdna_driver(const char *drv_path) {
    if (xdna_drv_fd >= 0) {
        // Already initialized
        return 0;
    }
    xdna_drv_fd = open(drv_path, O_RDWR);
    if (xdna_drv_fd < 0) {
        std::cerr << "Failed to open XDNA driver at " << drv_path << ": " << strerror(errno) << std::endl;
        return -1;
    }
    if (debug_verbosity >= 1) {
        std::cout << "XDNA driver opened successfully: " << drv_path << " (fd: " << xdna_drv_fd << ")" << std::endl;
    }

    // Allocate device heap
    if (debug_verbosity >= 1)
        std::cout << "Allocating device heap..." << std::endl;
    allocate_heap_and_error(xdna_drv_fd);

    return 0;
}

int npuMatmul_zero(int hwctx_numb, int instctx_numb, void *output_pointer, void *input_pointer, void *weight_pointer,
                   __u32 output_xdna_handle, __u32 input_xdna_handle, __u32 weight_xdna_handle, hipEvent_t hip_event) {

    struct amdxdna_drm_exec_cmd exec_cmd;
    // Pass all handles (Instruction, Input, Weight, Output) to ensure residency
    uint32_t bo_args[4] = {instctxt_array[instctx_numb].dpu_0_handle, input_xdna_handle, weight_xdna_handle, output_xdna_handle};

    int ret = create_cmd_packet(xdna_drv_fd, hwctxt_array[hwctx_numb].pdi_handle, instctxt_array[instctx_numb].dpu_0_sram_vaddr,
                                instctxt_array[instctx_numb].dpu_0_handle, instctxt_array[instctx_numb].num_dpu_0_insts,
                                (__u64)input_pointer, (__u64)weight_pointer, (__u64)output_pointer, input_xdna_handle, weight_xdna_handle,
                                output_xdna_handle, hwctxt_array[hwctx_numb].hw_ctx, exec_cmd, bo_args, 4);

    if (ret != 0) {
        perror("Failed to create command packet chain");
        return -1;
    }

    // Wait for input GPU buffers to be ready (Polling loop as requested)
    // The previous atomic check allowed a race; separate polling is safe here because:
    // 1. We wait for THIS thread's input data to be ready on GPU.
    // 2. THEN we lock the NPU to ensure exclusive access for the execution.
    if (hip_event != nullptr) {
        while (hipEventQuery(hip_event) != hipSuccess) {
            // Spin-wait / Poll until GPU is ready
            // (User requested avoiding heavy hipEventSynchronize)
        }
        if (debug_verbosity >= 3) {
            std::cout << "Hip Event Sync Success" << std::endl;
        }
    }

    // Lock the NPU for execution
    // EXPLANATION: std::lock_guard acquires 'npu_mutex'.
    // If another thread holds it, this thread will BLOCK (sleep/wait) until it is released.
    // It does NOT strictly spin-wait (burn CPU) unless the mutex implementation decides to spin briefly.
    std::lock_guard<std::mutex> lock(npu_mutex);

    if (debug_verbosity >= 2)
        std::cout << "Executing command" << std::endl;

    ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_EXEC_CMD, &exec_cmd);
    // Execute the command
    if (ret != 0) {
        perror("Failed to submit work");
        return -1;
    }

    // Wait for the command to complete
    struct amdxdna_drm_wait_cmd wait_cmd = {
        .ctx = hwctxt_array[hwctx_numb].hw_ctx.handle,
        .timeout = 500, // 50ms timeout
        .seq = exec_cmd.seq,
    };

    ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wait_cmd);
    if (ret != 0) {
        perror("Failed to wait");
        return -1;
    }

    // lock_guard destructor is called HERE automatically, releasing npu_mutex.
    return 0;
}

// Import DMA-BUF to XDNA
uint32_t import_dma_buf_to_xdna(void *hip_managed_ptr, size_t size, int dataTypeinBytes) {
    // Check if XDNA driver is initialized
    if (xdna_drv_fd < 0) {
        std::cerr << "XDNA driver not initialized. Call initialize_xdna_driver() first." << std::endl;
        return 0;
    }

    // Check cache first
    if (ptr_to_handle_map.find(hip_managed_ptr) != ptr_to_handle_map.end()) {
        if (debug_verbosity >= 1)
            std::cout << "Using cached handle for ptr: " << hip_managed_ptr << " handle: " << ptr_to_handle_map[hip_managed_ptr]
                      << std::endl;
        return ptr_to_handle_map[hip_managed_ptr];
    }

    // std::cout << "Importing pointer to XDNA: " << hip_managed_ptr << std::endl;

    int dmabuf_fd;
    uint64_t offset;
    hsa_status_t status = hsa_amd_portable_export_dmabuf(hip_managed_ptr, size * dataTypeinBytes, &dmabuf_fd, &offset);
    if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Failed to export DMA-BUF. Status: " << status << std::endl;
        return 0;
    }

    // std::cout << "DMA-BUF export successful. FD: " << dmabuf_fd << ", Offset: " << offset << " "
    // << hip_managed_ptr << std::endl;

    if (dmabuf_fd < 0) {
        if (debug_verbosity >= 1)
            std::cout << "Invalid DMA-BUF FD: " << dmabuf_fd << std::endl;
        return 0;
    }

    drm_prime_handle prime_params;
    prime_params.handle = 0;
    prime_params.flags = 0;
    prime_params.fd = dmabuf_fd;

    if (ioctl(xdna_drv_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime_params) < 0) {
        std::cerr << "Failed to import DMA-BUF: " << strerror(errno) << " (errno=" << errno << ")" << std::endl;
        std::cerr << "xdna_drv_fd=" << xdna_drv_fd << ", dmabuf_fd=" << dmabuf_fd << std::endl;
        close(dmabuf_fd); // Close the DMA-BUF FD on error
        return 0;
    }

    // std::cout << "Successfully imported DMA-BUF. Handle: " << prime_params.handle << std::endl;

    // Dummy operation to prevent compiler optimizations
    if (dataTypeinBytes == 4) {
        volatile float dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<float *>(hip_managed_ptr)[i];
        }
    } else if (dataTypeinBytes == 2) {
        volatile myBfloat dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<myBfloat *>(hip_managed_ptr)[i];
        }
    } else if (dataTypeinBytes == 1) {
        volatile uint8_t dummy_buffer = 0;
        for (int i = 0; i < size; i += IOMMU_STRIDE) {
            dummy_buffer += reinterpret_cast<uint8_t *>(hip_managed_ptr)[i];
        }
    } else {
        std::cerr << "Invalid data type size: " << dataTypeinBytes << std::endl;
        return 0;
    }

    // mmap the buffer in XDNA
    struct amdxdna_drm_get_bo_info get_bo_info = {.handle = prime_params.handle};
    int ret = ioctl(xdna_drv_fd, DRM_IOCTL_AMDXDNA_GET_BO_INFO, &get_bo_info);
    if (ret != 0) {
        perror("Failed to get BO info: ");
        return -2;
    }

    // Cache the handle
    ptr_to_handle_map[hip_managed_ptr] = prime_params.handle;
    // std::cout << "Cached handle " << prime_params.handle << " for ptr: " << hip_managed_ptr << " (map size: " <<
    // ptr_to_handle_map.size() << ")" << std::endl;

    // Return the imported handle
    return prime_params.handle;
}

// Import all module params/buffers to XDNA
void import_all_weights_to_xdna(torch::nn::Module &module) {
    if (debug_verbosity >= 1)
        std::cout << "Starting to import all weights and parameters to XDNA..." << std::endl;

    // Iterate weights (including sub-modules)
    auto named_parameters = module.named_parameters(true);
    for (const auto &param : named_parameters) {
        const std::string &name = param.key();
        const torch::Tensor &tensor = param.value();

        // Skip if tensor is empty
        if (tensor.numel() == 0) {
            if (debug_verbosity >= 1)
                std::cout << "Skipping empty parameter: " << name << std::endl;
            continue;
        }

        // Check if tensor is on HIP/CUDA device (required for DMA-BUF export)
        if (!tensor.device().is_cuda() && !tensor.device().is_hip()) {
            if (debug_verbosity >= 1) {
                std::cout << "Skipping parameter not on HIP/CUDA device: " << name << " (device: " << tensor.device() << ")" << std::endl;
            }
            continue;
        }

        // Get tensor data pointer
        void *data_ptr = tensor.data_ptr();
        size_t numel = tensor.numel();

        // Determine data type size in bytes
        int dtype_bytes = 0;
        if (tensor.dtype() == torch::kFloat32) {
            dtype_bytes = 4;
        } else if (tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat16) {
            dtype_bytes = 2;
        } else if (tensor.dtype() == torch::kUInt8 || tensor.dtype() == torch::kInt8) {
            dtype_bytes = 1;
        } else {
            std::cerr << "Unsupported dtype for parameter: " << name << " (dtype: " << tensor.dtype() << ")" << std::endl;
            continue;
        }

        // std::cout << "Importing parameter: " << name << " (size: " << numel
        //           << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;

        // Import to XDNA
        uint32_t handle = import_dma_buf_to_xdna(data_ptr, numel, dtype_bytes);
        if (handle == 0 || handle == static_cast<uint32_t>(-2)) {
            std::cerr << "Failed to import parameter to XDNA: " << name << std::endl;
        } else {
            if (debug_verbosity >= 1) {
                std::cout << "Successfully imported parameter to XDNA: " << name << " (handle: " << handle << ", numel: " << numel
                          << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;
            }
        }
    }

    // Iterate buffers
    if (debug_verbosity >= 1)
        std::cout << "\nStarting to import all buffers to XDNA..." << std::endl;
    auto named_buffers = module.named_buffers(true);
    for (const auto &buf : named_buffers) {
        const std::string &name = buf.key();
        const torch::Tensor &tensor = buf.value();

        // Skip if tensor is empty
        if (tensor.numel() == 0) {
            if (debug_verbosity >= 1)
                std::cout << "Skipping empty buffer: " << name << std::endl;
            continue;
        }

        // Check if tensor is on HIP/CUDA device (required for DMA-BUF export)
        if (!tensor.device().is_cuda() && !tensor.device().is_hip()) {
            if (debug_verbosity >= 1) {
                std::cout << "Skipping buffer not on HIP/CUDA device: " << name << " (device: " << tensor.device() << ")" << std::endl;
            }
            continue;
        }

        // Get tensor data pointer
        void *data_ptr = tensor.data_ptr();
        size_t numel = tensor.numel();

        // Determine data type size in bytes
        int dtype_bytes = 0;
        if (tensor.dtype() == torch::kFloat32) {
            dtype_bytes = 4;
        } else if (tensor.dtype() == torch::kBFloat16 || tensor.dtype() == torch::kFloat16) {
            dtype_bytes = 2;
        } else if (tensor.dtype() == torch::kUInt8 || tensor.dtype() == torch::kInt8) {
            dtype_bytes = 1;
        } else {
            std::cerr << "Unsupported dtype for buffer: " << name << " (dtype: " << tensor.dtype() << ")" << std::endl;
            continue;
        }

        // Import to XDNA
        uint32_t handle = import_dma_buf_to_xdna(data_ptr, numel, dtype_bytes);
        if (handle == 0 || handle == static_cast<uint32_t>(-2)) {
            std::cerr << "Failed to import buffer to XDNA: " << name << std::endl;
        } else {
            if (debug_verbosity >= 1) {
                std::cout << "Successfully imported buffer to XDNA: " << name << " (handle: " << handle << ", numel: " << numel
                          << ", dtype_bytes: " << dtype_bytes << ")" << std::endl;
            }
        }
    }

    if (debug_verbosity >= 1)
        std::cout << "Finished importing all weights and parameters to XDNA." << std::endl;
}
