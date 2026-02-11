/**
 * @file w4a16_gemm_unpacked.cu
 * @brief Fused W4A16 GEMM kernel for Unpacked (Manual) weights layout using rocWMMA
 */

#undef __HIP_NO_HALF_CONVERSIONS__
#include <hip/hip_fp16.h>

#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

// Use hip_bfloat16 type for AMD
using bfloat16_t = hip_bfloat16;

#include <cstdlib>
#include <hipblas/hipblas.h>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

// Macro Function to handle HIP errors
#define HIP_CHECK(call)                                                                                                                    \
    {                                                                                                                                      \
        hipError_t err = (call);                                                                                                           \
        if (err != hipSuccess) {                                                                                                           \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << " (" << err << ")"            \
                      << std::endl;                                                                                                        \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    }

namespace hipkernels {

__device__ __forceinline__ float bf16_to_float(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16(float val) { return static_cast<bfloat16_t>(val); }

namespace {
struct DevicePtrCache {
    uint64_t *d_qweights = nullptr;
    uint64_t *d_scales = nullptr;
    uint64_t *d_zeros = nullptr;
    int64_t size = 0;
    const int64_t *host_q = nullptr;
    const int64_t *host_s = nullptr;
    const int64_t *host_z = nullptr;
};

void ensure_device_ptrs(DevicePtrCache &cache, const std::vector<int64_t> &qweights, const std::vector<int64_t> &scales,
                        const std::vector<int64_t> &zeros) {
    TORCH_CHECK(qweights.size() == scales.size() && qweights.size() == zeros.size(), "Pointer array size mismatch");
    int64_t n = static_cast<int64_t>(qweights.size());
    bool need_alloc = (!cache.d_qweights || cache.size != n);
    bool need_copy = (need_alloc || cache.host_q != qweights.data() || cache.host_s != scales.data() || cache.host_z != zeros.data());

    if (need_alloc) {
        if (cache.d_qweights)
            HIP_CHECK(hipFree(cache.d_qweights));
        if (cache.d_scales)
            HIP_CHECK(hipFree(cache.d_scales));
        if (cache.d_zeros)
            HIP_CHECK(hipFree(cache.d_zeros));

        HIP_CHECK(hipMalloc(&cache.d_qweights, n * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&cache.d_scales, n * sizeof(uint64_t)));
        HIP_CHECK(hipMalloc(&cache.d_zeros, n * sizeof(uint64_t)));
        cache.size = n;
    }

    if (need_copy) {
        HIP_CHECK(hipMemcpy(cache.d_qweights, qweights.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cache.d_scales, scales.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cache.d_zeros, zeros.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
        cache.host_q = qweights.data();
        cache.host_s = scales.data();
        cache.host_z = zeros.data();
    }
}

DevicePtrCache &gemm_ptr_cache_for_device(int device) {
    static std::unordered_map<int, DevicePtrCache> caches;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    return caches[device];
}
} // namespace

// rocWMMA Kernel with F32 Accumulator for Unpacked Layout
using FragA = fragment<matrix_a, 16, 16, 16, bfloat16_t, row_major>;
using FragB_Col = fragment<matrix_b, 16, 16, 16, bfloat16_t, col_major>; // W^T col-major [K,N]
using FragC = fragment<accumulator, 16, 16, 16, float>;

#define PAD 16 // SMEM padding for 16-byte alignment

#if defined(CDNA_DEVICE) && CDNA_DEVICE == 1
constexpr int GEMM_BLOCK_SIZE = 512;
constexpr int WAVE_SIZE = 64;
#else
constexpr int GEMM_BLOCK_SIZE = 256;
constexpr int WAVE_SIZE = 32;
#endif

// Kernel assumes Block M=128, N=128, K_step=128
// Grid dimensions: (N / 128), (M / 128)
__global__ void __launch_bounds__(GEMM_BLOCK_SIZE, 1)
    w4a16_gemm_rocwmma_unpacked(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                                const uint8_t *__restrict__ qweights,  // [N, K/2]
                                const bfloat16_t *__restrict__ scales, // [N, Groups]
                                const uint8_t *__restrict__ zeros,     // [N, Groups]
                                const int M, const int K, const int N, const int group_size) {
    __shared__ __align__(16) bfloat16_t smem_B[128][128 + PAD]; // [N,K] tile

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // Wave ID
    const int wave_id = tid / WAVE_SIZE;
    const int wave_y = wave_id / 2; // 0..3 (Same for Wave32/256thr and Wave64/512thr)
    const int wave_x = wave_id % 2; // 0..1

    // Grid: bx maps to N, by maps to M (match packed kernel layout)
    const int global_n_start = bx * 128;
    const int global_m_start = by * 128;

    // Accumulators
    FragC acc[2][4];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            fill_fragment(acc[i][j], 0.0f);

    const int groups = K / group_size;

    for (int k_outer = 0; k_outer < K; k_outer += 128) {
        int group_idx = k_outer / group_size;

        // Load weights: 2 threads per N-row, each loads 64 K-values
        int w_row = tid >> 1; // 0..127 (N-index)
        int w_blk = tid & 1;  // 0..1 (64 K-values each)

        if (w_row < 128) {
            int gn = global_n_start + w_row;
            int k_base = w_blk * 64;
            if (gn < N) {
                float scale = 0.0f;
                int zero = 0;
                if (w_blk == 0) {
                    int idx = gn * groups + group_idx;
                    scale = bf16_to_float(scales[idx]);
                    zero = (int8_t)zeros[idx];
                }
                float scale_peer = __shfl_xor(scale, 1);
                int zero_peer = __shfl_xor(zero, 1);
                if (w_blk == 1) {
                    scale = scale_peer;
                    zero = zero_peer;
                }

                const uint8_t *w_src_base = &qweights[gn * (K / 2) + (k_outer / 2) + w_blk * 32];
                const uint4 *src_u4 = reinterpret_cast<const uint4 *>(w_src_base);
                uint4 v0 = src_u4[0];
                uint4 v1 = src_u4[1];

                float sz = scale * (float)zero;

#define DEQUANT_STORE_RAW(PACKED, OFF)                                                                                                     \
    do {                                                                                                                                   \
        uint32_t packed = (PACKED);                                                                                                        \
        union {                                                                                                                            \
            bfloat16_t res[8];                                                                                                             \
            float packed_res[4];                                                                                                           \
        } u;                                                                                                                               \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                                                                                    \
            uint8_t p = (packed >> (b * 8)) & 0xFF;                                                                                        \
            int q0 = p & 0x0F;                                                                                                             \
            int q1 = (p >> 4) & 0x0F;                                                                                                      \
            float val0 = (float)q0 * scale - sz;                                                                                           \
            float val1 = (float)q1 * scale - sz;                                                                                           \
            float2 f2;                                                                                                                     \
            f2.x = val0;                                                                                                                   \
            f2.y = val1;                                                                                                                   \
            __bf16_2 bf2 = __float22bfloat162_rn(f2);                                                                                      \
            union {                                                                                                                        \
                __bf16_2 bf;                                                                                                               \
                float f;                                                                                                                   \
            } converter;                                                                                                                   \
            converter.bf = bf2;                                                                                                            \
            u.packed_res[b] = converter.f;                                                                                                 \
        }                                                                                                                                  \
        *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + (OFF)]) = *reinterpret_cast<uint4 *>(u.res);                                    \
    } while (0)

                DEQUANT_STORE_RAW(v0.x, 0);
                DEQUANT_STORE_RAW(v0.y, 8);
                DEQUANT_STORE_RAW(v0.z, 16);
                DEQUANT_STORE_RAW(v0.w, 24);
                DEQUANT_STORE_RAW(v1.x, 32);
                DEQUANT_STORE_RAW(v1.y, 40);
                DEQUANT_STORE_RAW(v1.z, 48);
                DEQUANT_STORE_RAW(v1.w, 56);

#undef DEQUANT_STORE_RAW
            } else {
                const uint4 zero4 = {0, 0, 0, 0};
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + x * 8]) = zero4;
                }
            }
        }
        __syncthreads();

        FragA fA;
        FragB_Col fB;

#pragma unroll
        for (int ki = 0; ki < 128; ki += 16) {
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int r_offset = global_m_start + wave_y * 32 + i * 16;
                load_matrix_sync(fA, input + r_offset * K + (k_outer + ki), K);

#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int c_offset = wave_x * 64 + j * 16;
                    load_matrix_sync(fB, &smem_B[c_offset][ki], 128 + PAD);
                    mma_sync(acc[i][j], fA, fB, acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store: serialize waves through smem for F32->BF16 conversion
    float *smem_wb = (float *)&smem_B[0][0];

    for (int w = 0; w < 8; ++w) {
        __syncthreads(); // Wait for SMEM to be free
        if (wave_id == w) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    // Tile inside the wave (32x64)
                    int tr = i * 16;
                    int tc = j * 16;
                    store_matrix_sync(&smem_wb[tr * 64 + tc], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads(); // Wait for store to complete

        // All threads help write 32x64 tile from SMEM to Global (converting F32->BF16)
        // 2048 elements. 256 threads. 8 per thread.

        int target_wy = w / 2;
        int target_wx = w % 2;
        int r_base = global_m_start + target_wy * 32;
        int c_base = global_n_start + target_wx * 64;

        int tid_offset = tid * 8;
        if (tid_offset < 2048) {
            for (int k = 0; k < 8; ++k) {
                int idx = tid_offset + k;
                int r = idx / 64; // 0..31
                int c = idx % 64; // 0..63

                float val = smem_wb[r * 64 + c];

                int gr = r_base + r;
                int gc = c_base + c;

                if (gr < M && gc < N) {
                    output[gr * N + gc] = float_to_bf16(val);
                }
            }
        }
    }
}

__global__ void __launch_bounds__(GEMM_BLOCK_SIZE, 1)
    w4a16_gemm_rocwmma_unpacked_3d(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                                   const uint64_t *__restrict__ qweights_ptrs, const uint64_t *__restrict__ scales_ptrs,
                                   const uint64_t *__restrict__ zeros_ptrs, const int M, const int K, const int N, const int group_size) {
    const int expert = blockIdx.z;
    const bfloat16_t *__restrict__ input_e = input + static_cast<size_t>(expert) * static_cast<size_t>(M) * static_cast<size_t>(K);
    bfloat16_t *__restrict__ output_e = output + static_cast<size_t>(expert) * static_cast<size_t>(M) * static_cast<size_t>(N);
    const uint8_t *__restrict__ qweights = reinterpret_cast<const uint8_t *>(qweights_ptrs[expert]);
    const bfloat16_t *__restrict__ scales = reinterpret_cast<const bfloat16_t *>(scales_ptrs[expert]);
    const uint8_t *__restrict__ zeros = reinterpret_cast<const uint8_t *>(zeros_ptrs[expert]);

    __shared__ __align__(16) bfloat16_t smem_B[128][128 + PAD];

    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int wave_id = tid / WAVE_SIZE;
    const int wave_y = wave_id / 2;
    const int wave_x = wave_id % 2;

    const int global_n_start = bx * 128;
    const int global_m_start = by * 128;

    FragC acc[2][4];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            fill_fragment(acc[i][j], 0.0f);

    const int groups = K / group_size;

    int w_row = tid >> 1;
    int w_blk = tid & 1;
    int gn = global_n_start + w_row;

    // Pre-calculate invariant offsets
    int q_base_offset = 0;
    int s_base_offset = 0;
    bool active_n = (gn < N);
    const uint8_t *__restrict__ w_ptr = nullptr;

    if (active_n) {
        q_base_offset = gn * (K / 2) + w_blk * 32;
        s_base_offset = gn * groups;
        w_ptr = qweights + q_base_offset;
    }

    for (int k_outer = 0; k_outer < K; k_outer += 256) {
        // --- Unroll 0 ---
        {
            int k_curr = k_outer;
            int group_idx = k_curr / group_size;
            int k_base = w_blk * 64;

            if (active_n) {
                float scale = 0.0f;
                int zero = 0;
                if (w_blk == 0) {
                    int idx = s_base_offset + group_idx;
                    scale = bf16_to_float(scales[idx]);
                    zero = (int8_t)zeros[idx];
                }
                float scale_peer = __shfl_xor(scale, 1);
                int zero_peer = __shfl_xor(zero, 1);
                if (w_blk == 1) {
                    scale = scale_peer;
                    zero = zero_peer;
                }

                const uint4 *src_u4 = reinterpret_cast<const uint4 *>(w_ptr);
                w_ptr += 64;

                uint4 v0 = src_u4[0];
                uint4 v1 = src_u4[1];

                float sz = scale * (float)zero;

#define DEQUANT_STORE_RAW_3D(PACKED, OFF)                                                                                                  \
    do {                                                                                                                                   \
        uint32_t packed = (PACKED);                                                                                                        \
        union {                                                                                                                            \
            bfloat16_t res[8];                                                                                                             \
            float packed_res[4];                                                                                                           \
        } u;                                                                                                                               \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                                                                                    \
            uint8_t p = (packed >> (b * 8)) & 0xFF;                                                                                        \
            int q0 = p & 0x0F;                                                                                                             \
            int q1 = (p >> 4) & 0x0F;                                                                                                      \
            float val0 = (float)q0 * scale - sz;                                                                                           \
            float val1 = (float)q1 * scale - sz;                                                                                           \
            float2 f2;                                                                                                                     \
            f2.x = val0;                                                                                                                   \
            f2.y = val1;                                                                                                                   \
            __bf16_2 bf2 = __float22bfloat162_rn(f2);                                                                                      \
            union {                                                                                                                        \
                __bf16_2 bf;                                                                                                               \
                float f;                                                                                                                   \
            } converter;                                                                                                                   \
            converter.bf = bf2;                                                                                                            \
            u.packed_res[b] = converter.f;                                                                                                 \
        }                                                                                                                                  \
        *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + (OFF)]) = *reinterpret_cast<uint4 *>(u.res);                                    \
    } while (0)

                DEQUANT_STORE_RAW_3D(v0.x, 0);
                DEQUANT_STORE_RAW_3D(v0.y, 8);
                DEQUANT_STORE_RAW_3D(v0.z, 16);
                DEQUANT_STORE_RAW_3D(v0.w, 24);
                DEQUANT_STORE_RAW_3D(v1.x, 32);
                DEQUANT_STORE_RAW_3D(v1.y, 40);
                DEQUANT_STORE_RAW_3D(v1.z, 48);
                DEQUANT_STORE_RAW_3D(v1.w, 56);

#undef DEQUANT_STORE_RAW_3D
            } else {
                const uint4 zero4 = {0, 0, 0, 0};
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + x * 8]) = zero4;
                }
            }
            __syncthreads();

            FragA fA;
            FragB_Col fB;

#pragma unroll
            for (int ki = 0; ki < 128; ki += 16) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r_offset = global_m_start + wave_y * 32 + i * 16;
                    load_matrix_sync(fA, input_e + r_offset * K + (k_curr + ki), K);

#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int c_offset = wave_x * 64 + j * 16;
                        load_matrix_sync(fB, &smem_B[c_offset][ki], 128 + PAD);
                        mma_sync(acc[i][j], fA, fB, acc[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // --- Unroll 1 ---
        if (k_outer + 128 < K) {
            int k_curr = k_outer + 128;
            int group_idx = k_curr / group_size;
            int k_base = w_blk * 64;

            if (active_n) {
                float scale = 0.0f;
                int zero = 0;
                if (w_blk == 0) {
                    int idx = s_base_offset + group_idx;
                    scale = bf16_to_float(scales[idx]);
                    zero = (int8_t)zeros[idx];
                }
                float scale_peer = __shfl_xor(scale, 1);
                int zero_peer = __shfl_xor(zero, 1);
                if (w_blk == 1) {
                    scale = scale_peer;
                    zero = zero_peer;
                }

                const uint4 *src_u4 = reinterpret_cast<const uint4 *>(w_ptr);
                w_ptr += 64;

                uint4 v0 = src_u4[0];
                uint4 v1 = src_u4[1];

                float sz = scale * (float)zero;

#define DEQUANT_STORE_RAW_3D(PACKED, OFF)                                                                                                  \
    do {                                                                                                                                   \
        uint32_t packed = (PACKED);                                                                                                        \
        union {                                                                                                                            \
            bfloat16_t res[8];                                                                                                             \
            float packed_res[4];                                                                                                           \
        } u;                                                                                                                               \
        _Pragma("unroll") for (int b = 0; b < 4; ++b) {                                                                                    \
            uint8_t p = (packed >> (b * 8)) & 0xFF;                                                                                        \
            int q0 = p & 0x0F;                                                                                                             \
            int q1 = (p >> 4) & 0x0F;                                                                                                      \
            float val0 = (float)q0 * scale - sz;                                                                                           \
            float val1 = (float)q1 * scale - sz;                                                                                           \
            float2 f2;                                                                                                                     \
            f2.x = val0;                                                                                                                   \
            f2.y = val1;                                                                                                                   \
            __bf16_2 bf2 = __float22bfloat162_rn(f2);                                                                                      \
            union {                                                                                                                        \
                __bf16_2 bf;                                                                                                               \
                float f;                                                                                                                   \
            } converter;                                                                                                                   \
            converter.bf = bf2;                                                                                                            \
            u.packed_res[b] = converter.f;                                                                                                 \
        }                                                                                                                                  \
        *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + (OFF)]) = *reinterpret_cast<uint4 *>(u.res);                                    \
    } while (0)

                DEQUANT_STORE_RAW_3D(v0.x, 0);
                DEQUANT_STORE_RAW_3D(v0.y, 8);
                DEQUANT_STORE_RAW_3D(v0.z, 16);
                DEQUANT_STORE_RAW_3D(v0.w, 24);
                DEQUANT_STORE_RAW_3D(v1.x, 32);
                DEQUANT_STORE_RAW_3D(v1.y, 40);
                DEQUANT_STORE_RAW_3D(v1.z, 48);
                DEQUANT_STORE_RAW_3D(v1.w, 56);

#undef DEQUANT_STORE_RAW_3D
            } else {
                const uint4 zero4 = {0, 0, 0, 0};
#pragma unroll
                for (int x = 0; x < 8; ++x) {
                    *reinterpret_cast<uint4 *>(&smem_B[w_row][k_base + x * 8]) = zero4;
                }
            }
            __syncthreads();

            FragA fA;
            FragB_Col fB;

#pragma unroll
            for (int ki = 0; ki < 128; ki += 16) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r_offset = global_m_start + wave_y * 32 + i * 16;
                    load_matrix_sync(fA, input_e + r_offset * K + (k_curr + ki), K);

#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int c_offset = wave_x * 64 + j * 16;
                        load_matrix_sync(fB, &smem_B[c_offset][ki], 128 + PAD);
                        mma_sync(acc[i][j], fA, fB, acc[i][j]);
                    }
                }
            }
            __syncthreads();
        }
    }

    float *smem_wb = (float *)&smem_B[0][0];

    for (int w = 0; w < 8; ++w) {
        __syncthreads();
        if (wave_id == w) {
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    int tr = i * 16;
                    int tc = j * 16;
                    store_matrix_sync(&smem_wb[tr * 64 + tc], acc[i][j], 64, mem_row_major);
                }
            }
        }
        __syncthreads();

        int target_wy = w / 2;
        int target_wx = w % 2;
        int r_base = global_m_start + target_wy * 32;
        int c_base = global_n_start + target_wx * 64;

        int tid_offset = tid * 8;
        if (tid_offset < 2048) {
            for (int k = 0; k < 8; ++k) {
                int idx = tid_offset + k;
                int r = idx / 64;
                int c = idx % 64;

                float val = smem_wb[r * 64 + c];

                int gr = r_base + r;
                int gc = c_base + c;

                if (gr < M && gc < N) {
                    output_e[gr * N + gc] = float_to_bf16(val);
                }
            }
        }
    }
}

void w4a16_gemm_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size) {
    const int M = input.numel() / in_features;
    const int K = in_features;
    const int N = out_features;

    // Ensure grid covers M, N with 128x128 blocks (grid.x = N, grid.y = M)
    // Ensure grid covers M, N with 128x128 blocks (grid.x = N, grid.y = M)
    dim3 block(GEMM_BLOCK_SIZE);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    static bool cache_configured = false;
    if (!cache_configured) {
        hipError_t cache_err = hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_unpacked, hipFuncCachePreferL1);
        if (cache_err != hipSuccess) {
            throw std::runtime_error(std::string("HIP GEMM unpacked cache config error: ") + hipGetErrorString(cache_err));
        }
        cache_configured = true;
    }

    hipLaunchKernelGGL(w4a16_gemm_rocwmma_unpacked, grid, block, 0, 0, (bfloat16_t *)output.data_ptr(),
                       (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(), (const bfloat16_t *)scales.data_ptr(),
                       (const uint8_t *)zeros.data_ptr(), M, K, N, group_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel error: ") + hipGetErrorString(err));
    }
}

void w4a16_gemm_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights_ptrs,
                                  const torch::Tensor &scales_ptrs, const torch::Tensor &zeros_ptrs, int64_t in_features,
                                  int64_t out_features, int64_t group_size, int64_t num_experts) {
    TORCH_CHECK(input.dim() == 3, "input must be 3D [E, M, K]");
    TORCH_CHECK(output.dim() == 3, "output must be 3D [E, M, N]");
    TORCH_CHECK(qweights_ptrs.scalar_type() == torch::kInt64, "qweights_ptrs must be int64");
    TORCH_CHECK(scales_ptrs.scalar_type() == torch::kInt64, "scales_ptrs must be int64");
    TORCH_CHECK(zeros_ptrs.scalar_type() == torch::kInt64, "zeros_ptrs must be int64");
    TORCH_CHECK(qweights_ptrs.is_cuda() && scales_ptrs.is_cuda() && zeros_ptrs.is_cuda(), "pointer arrays must be on CUDA");

    const int M = static_cast<int>(input.size(1));
    const int K = static_cast<int>(in_features);
    const int N = static_cast<int>(out_features);

    dim3 block(GEMM_BLOCK_SIZE);
    dim3 grid((N + 127) / 128, (M + 127) / 128, static_cast<uint32_t>(num_experts));

    static bool cache_configured_3d = false;
    if (!cache_configured_3d) {
        hipError_t cache_err = hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_unpacked_3d, hipFuncCachePreferL1);
        if (cache_err != hipSuccess) {
            throw std::runtime_error(std::string("HIP GEMM unpacked 3D cache config error: ") + hipGetErrorString(cache_err));
        }
        cache_configured_3d = true;
    }

    hipLaunchKernelGGL(w4a16_gemm_rocwmma_unpacked_3d, grid, block, 0, 0, (bfloat16_t *)output.data_ptr(),
                       (const bfloat16_t *)input.data_ptr(), (const uint64_t *)qweights_ptrs.data_ptr<int64_t>(),
                       (const uint64_t *)scales_ptrs.data_ptr<int64_t>(), (const uint64_t *)zeros_ptrs.data_ptr<int64_t>(), M, K, N,
                       group_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel 3D error: ") + hipGetErrorString(err));
    }
}

void w4a16_gemm_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const std::vector<int64_t> &qweights_ptrs,
                                  const std::vector<int64_t> &scales_ptrs, const std::vector<int64_t> &zeros_ptrs, int64_t in_features,
                                  int64_t out_features, int64_t group_size, int64_t num_experts) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    const int target_device = input.get_device();
    int current_device = 0;
    HIP_CHECK(hipGetDevice(&current_device));
    if (current_device != target_device) {
        HIP_CHECK(hipSetDevice(target_device));
    }

    auto &cache = gemm_ptr_cache_for_device(target_device);
    ensure_device_ptrs(cache, qweights_ptrs, scales_ptrs, zeros_ptrs);

    const int M = static_cast<int>(input.size(1));
    const int K = static_cast<int>(in_features);
    const int N = static_cast<int>(out_features);

    dim3 block(GEMM_BLOCK_SIZE);
    dim3 grid((N + 127) / 128, (M + 127) / 128, static_cast<uint32_t>(num_experts));

    static bool cache_configured_3d = false;
    if (!cache_configured_3d) {
        hipError_t cache_err = hipFuncSetCacheConfig((const void *)w4a16_gemm_rocwmma_unpacked_3d, hipFuncCachePreferL1);
        if (cache_err != hipSuccess) {
            throw std::runtime_error(std::string("HIP GEMM unpacked 3D cache config error: ") + hipGetErrorString(cache_err));
        }
        cache_configured_3d = true;
    }

    hipLaunchKernelGGL(w4a16_gemm_rocwmma_unpacked_3d, grid, block, 0, 0, (bfloat16_t *)output.data_ptr(),
                       (const bfloat16_t *)input.data_ptr(), cache.d_qweights, cache.d_scales, cache.d_zeros, M, K, N, group_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP kernel 3D error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels

torch::Tensor hipkernels::w4a16_gemm_unpacked_alloc_and_compute(const torch::Tensor &input, const torch::Tensor &qweights,
                                                                const torch::Tensor &scales, const torch::Tensor &zeros,
                                                                int64_t in_features, int64_t out_features, int64_t group_size) {
    int64_t M = input.numel() / in_features;
    auto output = torch::empty({M, out_features}, torch::TensorOptions().dtype(torch::kBFloat16).device(input.device()));
    w4a16_gemm_unpacked_fused(output, input, qweights, scales, zeros, in_features, out_features, group_size);
    return output;
}
