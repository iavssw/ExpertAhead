#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include <c10/hip/HIPStream.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#define HIP_CHECK(call)                                                                                                                    \
    do {                                                                                                                                   \
        hipError_t err = (call);                                                                                                           \
        if (err != hipSuccess) {                                                                                                           \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << " (" << err << ")"            \
                      << std::endl;                                                                                                        \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    } while (0)

using bfloat16_t = hip_bfloat16;

namespace hipkernels {

__device__ __forceinline__ float bf16_to_float_dev(bfloat16_t val) { return static_cast<float>(val); }
__device__ __forceinline__ bfloat16_t float_to_bf16_dev(float val) { return static_cast<bfloat16_t>(val); }

namespace {
struct DevicePtrCache {
    uint64_t *d_qweights = nullptr;
    uint64_t *d_scales = nullptr;
    uint64_t *d_zeros = nullptr;
    int64_t size = 0;
};

void ensure_device_ptrs(DevicePtrCache &cache, const std::vector<int64_t> &qweights, const std::vector<int64_t> &scales,
                        const std::vector<int64_t> &zeros) {
    TORCH_CHECK(qweights.size() == scales.size() && qweights.size() == zeros.size(), "Pointer array size mismatch");
    int64_t n = static_cast<int64_t>(qweights.size());
    bool need_alloc = (!cache.d_qweights || cache.size != n);

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

    // Pointer vectors are frequently rebuilt per call; host addresses can repeat
    // while contents change, so refresh device pointer arrays every invocation.
    HIP_CHECK(hipMemcpy(cache.d_qweights, qweights.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cache.d_scales, scales.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cache.d_zeros, zeros.data(), n * sizeof(uint64_t), hipMemcpyHostToDevice));
}

uint64_t make_cache_key(int device, hipStream_t stream) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(device)) << 32) ^
           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream));
}

DevicePtrCache &gemv_ptr_cache_for_device_stream(int device, hipStream_t stream) {
    static std::unordered_map<uint64_t, DevicePtrCache> caches;
    static std::mutex cache_mutex;
    std::lock_guard<std::mutex> lock(cache_mutex);
    return caches[make_cache_key(device, stream)];
}
} // namespace

#if defined(CDNA_DEVICE) && CDNA_DEVICE == 1
constexpr int WAVE_SIZE = 64;
#else
constexpr int WAVE_SIZE = 32;
#endif

// Kernel: One Wave per Output (N). Outputs per Block = 256 / WAVE_SIZE.
// Optimization: v23 (Hybrid Scale Caching + NegZS + Branch).
template <int MAX_GROUPS>
__global__ void __launch_bounds__(256)
    w4a16_gemv_unpacked_kernel(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                               const uint8_t *__restrict__ qweights,  // [N, K/2]
                               const bfloat16_t *__restrict__ scales, // [Groups, N_total] or [N_total, Groups]
                               const uint8_t *__restrict__ zeros,     // [Groups, N_total] or [N_total, Groups]
                               const int M,                           // 1
                               const int K, const int N, const int group_shift, const int stride_groups, const int stride_n) {
    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % WAVE_SIZE;
    const int warp_id = tid / WAVE_SIZE;

    const int waves_per_block = 256 / WAVE_SIZE;
    const int global_n = bx * waves_per_block + warp_id;
    bool active = (global_n < N);

    __shared__ __half2 s_input[16 * 257];
    __shared__ __half2 s_scales[MAX_GROUPS * (256 / WAVE_SIZE)];
    __shared__ __half2 s_neg_zs[MAX_GROUPS * (256 / WAVE_SIZE)];

    const int stride = 257;

    const uint4 *input_vecs = reinterpret_cast<const uint4 *>(input);
    const uint4 *w_row_vec = nullptr;
    if (active) {
        const uint8_t *w_base = &qweights[global_n * (K / 2)];
        w_row_vec = reinterpret_cast<const uint4 *>(w_base);
    }

    // Pre-load Scales/Zeros & Pre-compute NegZS
    const int num_groups = K >> group_shift;
    const int groups_to_cache = (num_groups < MAX_GROUPS) ? num_groups : MAX_GROUPS;

    if (active) {
        // Load scales/zeros for all groups (up to MAX_GROUPS cached)
        if constexpr (MAX_GROUPS == 128) {
            for (int g_idx = lane_id; g_idx < groups_to_cache; g_idx += WAVE_SIZE) {
                int idx_s = g_idx * stride_groups + global_n * stride_n;
                unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                uint8_t z_raw = __ldg(&zeros[idx_s]);
                float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                float z_f = (float)((int8_t)z_raw);

                __half2 s2 = __half2half2(__float2half(s_f));
                __half2 z2 = __half2half2(__float2half(z_f));

                s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
            }
        } else {
            for (int g_idx = lane_id; g_idx < groups_to_cache; g_idx += WAVE_SIZE) {
                int idx_s = g_idx * stride_groups + global_n * stride_n;
                unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                uint8_t z_raw = __ldg(&zeros[idx_s]);
                float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                float z_f = (float)((int8_t)z_raw);

                __half2 s2 = __half2half2(__float2half(s_f));
                __half2 z2 = __half2half2(__float2half(z_f));

                s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
            }
        }
    }
    __syncthreads();

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);

    int current_group = -1;
    __half2 s2 = __float2half2_rn(1.0f);
    __half2 neg_zs = __float2half2_rn(0.0f);

    const int TILE_SIZE = 8192;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    int stagger_idx = (global_n & 31);
    int stagger_offset = (stagger_idx * 8);

    for (int t = 0; t < num_tiles; ++t) {
        int k_tile_base = t * TILE_SIZE;

        // 1. Vectorized Input Load (v18)
        int vec_base_k = k_tile_base / 8;
        int max_vec_k = K / 8;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int vec_offset = i * 256 + tid;
            int vec_idx = vec_base_k + vec_offset;
            if (vec_idx < max_vec_k) {
                uint4 val4 = input_vecs[vec_idx];
                int *vals = reinterpret_cast<int *>(&val4);

                int base_pair = vec_offset * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int pair_offset_rel = base_pair + j;
                    int val = vals[j];
                    bfloat16_t *v = reinterpret_cast<bfloat16_t *>(&val);
                    float f0 = bf16_to_float_dev(v[0]);
                    float f1 = bf16_to_float_dev(v[1]);
                    __half2 h2 = __float22half2_rn(make_float2(f0, f1));
                    int r_s = pair_offset_rel >> 4;
                    int c_s = pair_offset_rel & 15;
                    s_input[c_s * stride + r_s] = h2;
                }
            }
        }
        __syncthreads();

        if (active) {

            uint4 w_vals[8];
            int chunk_ids[8];

            // Prefetch Weights (with bounds check for smaller K)
            int max_vecs = (K / 2) / 16; // Total uint4 vectors per row

            // Number of chunks processed per iteration depends on WAVE_SIZE.
            // Wave32: 8 iters * 32 lanes = 256 chunks.
            // Wave64: 4 iters * 64 lanes = 256 chunks.
            constexpr int ITERS_PER_LOOP = 256 / WAVE_SIZE;

#pragma unroll
            for (int i = 0; i < ITERS_PER_LOOP; ++i) {
                int chunk_id_raw = i * WAVE_SIZE + lane_id;
                int chunk_id = (chunk_id_raw + stagger_offset) & 255;
                chunk_ids[i] = chunk_id;
                int vec_idx = t * 256 + chunk_id;
                if (vec_idx < max_vecs) {
                    w_vals[i] = w_row_vec[vec_idx];
                } else {
                    w_vals[i] = {0, 0, 0, 0}; // Zero out-of-bounds
                }
            }

// Math Loop
#pragma unroll
            for (int i = 0; i < ITERS_PER_LOOP; ++i) {
                int chunk_id = chunk_ids[i];
                uint4 w_val = w_vals[i];

                // Register Cache
                __half2 in_reg[16];
                int r = chunk_id;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    in_reg[k] = s_input[k * stride + r];
                }

                int k_abs_start = k_tile_base + chunk_id * 32;

                if (k_abs_start < K) {
                    int g = k_abs_start >> group_shift;
                    // Branched Read: Only update on group change
                    // Reduces SMEM Conflicts
                    if (g != current_group) {
                        current_group = g;
                        if (g < MAX_GROUPS) {
                            s2 = s_scales[warp_id * MAX_GROUPS + g];
                            neg_zs = s_neg_zs[warp_id * MAX_GROUPS + g];
                        } else {
                            // Fallback: load directly from global if groups exceed cache
                            int idx_s = g * stride_groups + global_n * stride_n;
                            unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                            uint8_t z_raw = __ldg(&zeros[idx_s]);
                            float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                            float z_f = (float)((int8_t)z_raw);
                            __half2 s2_tmp = __half2half2(__float2half(s_f));
                            __half2 z2_tmp = __half2half2(__float2half(z_f));
                            s2 = s2_tmp;
                            neg_zs = __hneg2(__hmul2(z2_tmp, s2_tmp));
                        }
                    }

#define PROCESS_PAIR(w_byte, c_idx, acc)                                                                                                   \
    {                                                                                                                                      \
        uint8_t p = w_byte;                                                                                                                \
        int w0_i = p & 0x0F;                                                                                                               \
        int w1_i = (p >> 4) & 0x0F;                                                                                                        \
        __half w0 = __float2half((float)w0_i);                                                                                             \
        __half w1 = __float2half((float)w1_i);                                                                                             \
        __half2 w_pair = __halves2half2(w0, w1);                                                                                           \
        __half2 term = __hfma2(w_pair, s2, neg_zs);                                                                                        \
        acc = __hfma2(term, in_reg[c_idx], acc);                                                                                           \
    }

                    uint32_t wx = w_val.x;
                    PROCESS_PAIR((wx & 0xFF), 0, acc0);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 1, acc1);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 2, acc2);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 3, acc3);

                    uint32_t wy = w_val.y;
                    PROCESS_PAIR((wy & 0xFF), 4, acc0);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 5, acc1);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 6, acc2);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 7, acc3);

                    uint32_t wz = w_val.z;
                    PROCESS_PAIR((wz & 0xFF), 8, acc0);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 9, acc1);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 10, acc2);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 11, acc3);

                    uint32_t ww = w_val.w;
                    PROCESS_PAIR((ww & 0xFF), 12, acc0);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 13, acc1);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 14, acc2);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 15, acc3);

#undef PROCESS_PAIR
                }
            }
        }
        __syncthreads();
    }

    acc0 = __hadd2(acc0, acc1);
    acc2 = __hadd2(acc2, acc3);
    acc0 = __hadd2(acc0, acc2);
    __half acc_sum = __hadd(acc0.x, acc0.y);
    float final_acc = __half2float(acc_sum);

#pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2)
        final_acc += __shfl_xor(final_acc, offset);

    if (active && lane_id == 0) {
        output[global_n] = float_to_bf16_dev(final_acc);
    }
}

template <int MAX_GROUPS>
__global__ void __launch_bounds__(256)
    w4a16_gemv_unpacked_kernel_3d(bfloat16_t *__restrict__ output, const bfloat16_t *__restrict__ input,
                                  const uint64_t *__restrict__ qweights_ptrs, const uint64_t *__restrict__ scales_ptrs,
                                  const uint64_t *__restrict__ zeros_ptrs, const int M, const int K, const int N, const int group_shift,
                                  const int stride_groups, const int stride_n) {
    const int expert = blockIdx.z;
    const bfloat16_t *input_e = input + static_cast<size_t>(expert) * static_cast<size_t>(K);
    bfloat16_t *output_e = output + static_cast<size_t>(expert) * static_cast<size_t>(N);
    const uint8_t *qweights = reinterpret_cast<const uint8_t *>(qweights_ptrs[expert]);
    const bfloat16_t *scales = reinterpret_cast<const bfloat16_t *>(scales_ptrs[expert]);
    const uint8_t *zeros = reinterpret_cast<const uint8_t *>(zeros_ptrs[expert]);

    const int bx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % WAVE_SIZE;
    const int warp_id = tid / WAVE_SIZE;

    const int waves_per_block = blockDim.x / WAVE_SIZE;
    const int global_n = bx * waves_per_block + warp_id;
    bool active = (global_n < N);

    __shared__ __half2 s_input[16 * 257];
    __shared__ __half2 s_scales[MAX_GROUPS * (256 / WAVE_SIZE)];
    __shared__ __half2 s_neg_zs[MAX_GROUPS * (256 / WAVE_SIZE)];

    const int stride = 257;

    const uint4 *__restrict__ input_vecs = reinterpret_cast<const uint4 *>(input_e);
    const uint4 *__restrict__ w_row_vec = nullptr;
    if (active) {
        const uint8_t *w_base = &qweights[global_n * (K / 2)];
        w_row_vec = reinterpret_cast<const uint4 *>(w_base);
    }

    const int num_groups = K >> group_shift;
    const int groups_to_cache = (num_groups < MAX_GROUPS) ? num_groups : MAX_GROUPS;

    if (active) {
        if constexpr (MAX_GROUPS == 128) {
            for (int g_idx = lane_id; g_idx < groups_to_cache; g_idx += WAVE_SIZE) {
                int idx_s = g_idx * stride_groups + global_n * stride_n;
                unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                uint8_t z_raw = __ldg(&zeros[idx_s]);
                float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                float z_f = (float)((int8_t)z_raw);

                __half2 s2 = __half2half2(__float2half(s_f));
                __half2 z2 = __half2half2(__float2half(z_f));

                s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
            }
        } else {
            for (int g_idx = lane_id; g_idx < groups_to_cache; g_idx += WAVE_SIZE) {
                int idx_s = g_idx * stride_groups + global_n * stride_n;
                unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                uint8_t z_raw = __ldg(&zeros[idx_s]);
                float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                float z_f = (float)((int8_t)z_raw);

                __half2 s2 = __half2half2(__float2half(s_f));
                __half2 z2 = __half2half2(__float2half(z_f));

                s_scales[warp_id * MAX_GROUPS + g_idx] = s2;
                s_neg_zs[warp_id * MAX_GROUPS + g_idx] = __hneg2(__hmul2(z2, s2));
            }
        }
    }
    __syncthreads();

    __half2 acc0 = __float2half2_rn(0.0f);
    __half2 acc1 = __float2half2_rn(0.0f);
    __half2 acc2 = __float2half2_rn(0.0f);
    __half2 acc3 = __float2half2_rn(0.0f);

    int current_group = -1;
    __half2 s2 = __float2half2_rn(1.0f);
    __half2 neg_zs = __float2half2_rn(0.0f);

    const int TILE_SIZE = 8192;
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    int stagger_idx = (global_n & 31);
    int stagger_offset = (stagger_idx * 8);

    // Pre-calculate chunks (Strength Reduction)
    constexpr int ITERS_PER_LOOP = 256 / WAVE_SIZE;
    int chunk_ids[ITERS_PER_LOOP];
    const uint4 *w_ptrs[ITERS_PER_LOOP];
    if (active) {
#pragma unroll
        for (int i = 0; i < ITERS_PER_LOOP; ++i) {
            int chunk_id_raw = i * WAVE_SIZE + lane_id;
            int chunk_id = (chunk_id_raw + stagger_offset) & 255;
            chunk_ids[i] = chunk_id;
            w_ptrs[i] = w_row_vec + chunk_id;
        }
    }

    const int max_vecs = (K / 2) / 16;
    int k_tile_base = 0;

    for (int t = 0; t < num_tiles; ++t, k_tile_base += TILE_SIZE) {

        int vec_base_k = k_tile_base / 8;
        int max_vec_k = K / 8;

        // Load Input (Block 256: 4 iters required to load 1024 uint4s)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int vec_offset = i * 256 + tid;
            int vec_idx = vec_base_k + vec_offset;
            if (vec_idx < max_vec_k) {
                uint4 val4 = input_vecs[vec_idx];
                int *vals = reinterpret_cast<int *>(&val4);

                int base_pair = vec_offset * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int pair_offset_rel = base_pair + j;
                    int val = vals[j];
                    bfloat16_t *v = reinterpret_cast<bfloat16_t *>(&val);
                    float f0 = bf16_to_float_dev(v[0]);
                    float f1 = bf16_to_float_dev(v[1]);
                    __half2 h2 = __float22half2_rn(make_float2(f0, f1));
                    int r_s = pair_offset_rel >> 4;
                    int c_s = pair_offset_rel & 15;
                    s_input[c_s * stride + r_s] = h2;
                }
            }
        }
        __syncthreads();

        if (active) {
            uint4 w_vals[ITERS_PER_LOOP];

            // Load Loads
#pragma unroll
            for (int i = 0; i < ITERS_PER_LOOP; ++i) {
                int vec_idx = t * 256 + chunk_ids[i];
                if (vec_idx < max_vecs) {
                    // Use non-temporal load to bypass L2 cache (streaming weights)
                    // Cast to __int128_t (16 bytes) which is a native type for the builtin
                    const __int128_t *ptr_128 = reinterpret_cast<const __int128_t *>(w_ptrs[i]);
                    __int128_t val_128 = __builtin_nontemporal_load(ptr_128);
                    w_vals[i] = *reinterpret_cast<uint4 *>(&val_128);
                } else {
                    w_vals[i] = {0, 0, 0, 0};
                }
                w_ptrs[i] += 256;
            }

#pragma unroll
            for (int i = 0; i < ITERS_PER_LOOP; ++i) {
                int chunk_id = chunk_ids[i];
                uint4 w_val = w_vals[i];

                __half2 in_reg[16];
                int r = chunk_id;
#pragma unroll
                for (int k = 0; k < 16; ++k) {
                    in_reg[k] = s_input[k * stride + r];
                }

                int k_abs_start = k_tile_base + chunk_id * 32;

                if (k_abs_start < K) {
                    int g = k_abs_start >> group_shift;
                    if (g != current_group) {
                        current_group = g;
                        if (g < MAX_GROUPS) {
                            s2 = s_scales[warp_id * MAX_GROUPS + g];
                            neg_zs = s_neg_zs[warp_id * MAX_GROUPS + g];
                        } else {
                            int idx_s = g * stride_groups + global_n * stride_n;
                            unsigned short s_raw = __ldg((const unsigned short *)&scales[idx_s]);
                            uint8_t z_raw = __ldg(&zeros[idx_s]);
                            float s_f = bf16_to_float_dev(*reinterpret_cast<bfloat16_t *>(&s_raw));
                            float z_f = (float)((int8_t)z_raw);
                            __half2 s2_tmp = __half2half2(__float2half(s_f));
                            __half2 z2_tmp = __half2half2(__float2half(z_f));
                            s2 = s2_tmp;
                            neg_zs = __hneg2(__hmul2(z2_tmp, s2_tmp));
                        }
                    }

#define PROCESS_PAIR(w_byte, c_idx, acc)                                                                                                   \
    {                                                                                                                                      \
        uint8_t p = w_byte;                                                                                                                \
        int w0_i = p & 0x0F;                                                                                                               \
        int w1_i = (p >> 4) & 0x0F;                                                                                                        \
        __half w0 = __float2half((float)w0_i);                                                                                             \
        __half w1 = __float2half((float)w1_i);                                                                                             \
        __half2 w_pair = __halves2half2(w0, w1);                                                                                           \
        __half2 term = __hfma2(w_pair, s2, neg_zs);                                                                                        \
        acc = __hfma2(term, in_reg[c_idx], acc);                                                                                           \
    }

                    uint32_t wx = w_val.x;
                    PROCESS_PAIR((wx & 0xFF), 0, acc0);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 1, acc1);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 2, acc2);
                    wx >>= 8;
                    PROCESS_PAIR((wx & 0xFF), 3, acc3);

                    uint32_t wy = w_val.y;
                    PROCESS_PAIR((wy & 0xFF), 4, acc0);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 5, acc1);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 6, acc2);
                    wy >>= 8;
                    PROCESS_PAIR((wy & 0xFF), 7, acc3);

                    uint32_t wz = w_val.z;
                    PROCESS_PAIR((wz & 0xFF), 8, acc0);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 9, acc1);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 10, acc2);
                    wz >>= 8;
                    PROCESS_PAIR((wz & 0xFF), 11, acc3);

                    uint32_t ww = w_val.w;
                    PROCESS_PAIR((ww & 0xFF), 12, acc0);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 13, acc1);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 14, acc2);
                    ww >>= 8;
                    PROCESS_PAIR((ww & 0xFF), 15, acc3);

#undef PROCESS_PAIR
                }
            }
        }
        __syncthreads();
    }

    acc0 = __hadd2(acc0, acc1);
    acc2 = __hadd2(acc2, acc3);
    acc0 = __hadd2(acc0, acc2);
    __half acc_sum = __hadd(acc0.x, acc0.y);
    float final_acc = __half2float(acc_sum);

#pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2)
        final_acc += __shfl_xor(final_acc, offset);

    if (active && lane_id == 0) {
        output_e[global_n] = float_to_bf16_dev(final_acc);
    }
}

void w4a16_gemv_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size) {
    const int K = in_features;
    const int N = out_features;
    const int M = 1;

    int stride_groups, stride_n;
    if (scales.size(0) == out_features) {
        stride_n = scales.stride(0);
        stride_groups = scales.stride(1);
    } else {
        stride_groups = scales.stride(0);
        stride_n = scales.stride(1);
    }

    dim3 block(256);
    int waves_per_block = 256 / WAVE_SIZE;
    dim3 grid((N + waves_per_block - 1) / waves_per_block);
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    int group_shift = 0;
    int gs = group_size;
    while (gs > 1) {
        gs >>= 1;
        group_shift++;
    }

    int num_groups = K >> group_shift;
    if (num_groups <= 128) {
        w4a16_gemv_unpacked_kernel<128><<<grid, block, 0, stream>>>(
            (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(),
            (const bfloat16_t *)scales.data_ptr(), (const uint8_t *)zeros.data_ptr(), M, K, N, group_shift, stride_groups, stride_n);
    } else {
        w4a16_gemv_unpacked_kernel<256><<<grid, block, 0, stream>>>(
            (bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(), qweights.data_ptr<uint8_t>(),
            (const bfloat16_t *)scales.data_ptr(), (const uint8_t *)zeros.data_ptr(), M, K, N, group_shift, stride_groups, stride_n);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP GEMV Unpacked error: ") + hipGetErrorString(err));
    }
}

void w4a16_gemv_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights_ptrs,
                                  const torch::Tensor &scales_ptrs, const torch::Tensor &zeros_ptrs, int64_t in_features,
                                  int64_t out_features, int64_t group_size, int64_t num_experts) {
    TORCH_CHECK(qweights_ptrs.scalar_type() == torch::kInt64, "qweights_ptrs must be int64");
    TORCH_CHECK(scales_ptrs.scalar_type() == torch::kInt64, "scales_ptrs must be int64");
    TORCH_CHECK(zeros_ptrs.scalar_type() == torch::kInt64, "zeros_ptrs must be int64");
    TORCH_CHECK(qweights_ptrs.is_cuda() && scales_ptrs.is_cuda() && zeros_ptrs.is_cuda(), "pointer arrays must be on CUDA");

    const int K = static_cast<int>(in_features);
    const int N = static_cast<int>(out_features);
    const int M = 1;

    int stride_groups = 1;
    int stride_n = K / group_size;

    dim3 block(256);
    int waves_per_block = 256 / WAVE_SIZE;
    dim3 grid((N + waves_per_block - 1) / waves_per_block, 1, static_cast<uint32_t>(num_experts));
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    int group_shift = 0;
    int gs = group_size;
    while (gs > 1) {
        gs >>= 1;
        group_shift++;
    }

    int num_groups = K >> group_shift;
    if (num_groups <= 128) {
        w4a16_gemv_unpacked_kernel_3d<128>
            <<<grid, block, 0, stream>>>((bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                                    (const uint64_t *)qweights_ptrs.data_ptr<int64_t>(), (const uint64_t *)scales_ptrs.data_ptr<int64_t>(),
                                    (const uint64_t *)zeros_ptrs.data_ptr<int64_t>(), M, K, N, group_shift, stride_groups, stride_n);
    } else {
        w4a16_gemv_unpacked_kernel_3d<256>
            <<<grid, block, 0, stream>>>((bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                                    (const uint64_t *)qweights_ptrs.data_ptr<int64_t>(), (const uint64_t *)scales_ptrs.data_ptr<int64_t>(),
                                    (const uint64_t *)zeros_ptrs.data_ptr<int64_t>(), M, K, N, group_shift, stride_groups, stride_n);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP GEMV Unpacked 3D error: ") + hipGetErrorString(err));
    }
}

void w4a16_gemv_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const std::vector<int64_t> &qweights_ptrs,
                                  const std::vector<int64_t> &scales_ptrs, const std::vector<int64_t> &zeros_ptrs, int64_t in_features,
                                  int64_t out_features, int64_t group_size, int64_t num_experts) {
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    const int target_device = input.get_device();
    int current_device = 0;
    HIP_CHECK(hipGetDevice(&current_device));
    if (current_device != target_device) {
        HIP_CHECK(hipSetDevice(target_device));
    }
    hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

    auto &cache = gemv_ptr_cache_for_device_stream(target_device, stream);
    ensure_device_ptrs(cache, qweights_ptrs, scales_ptrs, zeros_ptrs);

    const int K = static_cast<int>(in_features);
    const int N = static_cast<int>(out_features);
    const int M = 1;

    int stride_groups = 1;
    int stride_n = K / group_size;

    dim3 block(256);
    int waves_per_block = 256 / WAVE_SIZE;
    dim3 grid((N + waves_per_block - 1) / waves_per_block, 1, static_cast<uint32_t>(num_experts));

    int group_shift = 0;
    int gs = group_size;
    while (gs > 1) {
        gs >>= 1;
        group_shift++;
    }

    int num_groups = K >> group_shift;
    if (num_groups <= 128) {
        w4a16_gemv_unpacked_kernel_3d<128><<<grid, block, 0, stream>>>((bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                                                                  cache.d_qweights, cache.d_scales, cache.d_zeros, M, K, N, group_shift,
                                                                  stride_groups, stride_n);
    } else {
        w4a16_gemv_unpacked_kernel_3d<256><<<grid, block, 0, stream>>>((bfloat16_t *)output.data_ptr(), (const bfloat16_t *)input.data_ptr(),
                                                                  cache.d_qweights, cache.d_scales, cache.d_zeros, M, K, N, group_shift,
                                                                  stride_groups, stride_n);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        throw std::runtime_error(std::string("HIP GEMV Unpacked 3D error: ") + hipGetErrorString(err));
    }
}

} // namespace hipkernels
