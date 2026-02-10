#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// =============================================================================
// HIP Helpers
// =============================================================================

#define HIP_CHECK(call)                                                                                                                    \
    do {                                                                                                                                   \
        hipError_t _e = (call);                                                                                                            \
        if (_e != hipSuccess) {                                                                                                            \
            std::cerr << "HIP error: " << hipGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
            std::abort();                                                                                                                  \
        }                                                                                                                                  \
    } while (0)

namespace {
struct FlashAttnDecodeWorkspace {
    void *partials = nullptr;
    void *meta = nullptr;
    size_t partials_bytes = 0;
    size_t meta_bytes = 0;
    int device = -1;
};

std::mutex g_flash_attn_decode_mutex;
FlashAttnDecodeWorkspace g_flash_attn_decode_ws;

FlashAttnDecodeWorkspace acquire_flash_attn_decode_workspace(int device, size_t partials_bytes, size_t meta_bytes) {
    std::lock_guard<std::mutex> lock(g_flash_attn_decode_mutex);

    if (g_flash_attn_decode_ws.device != device) {
        if (g_flash_attn_decode_ws.partials) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.partials));
        }
        if (g_flash_attn_decode_ws.meta) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.meta));
        }
        g_flash_attn_decode_ws = FlashAttnDecodeWorkspace{};
        g_flash_attn_decode_ws.device = device;
    }

    if (partials_bytes > g_flash_attn_decode_ws.partials_bytes) {
        if (g_flash_attn_decode_ws.partials) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.partials));
        }
        HIP_CHECK(hipMalloc(&g_flash_attn_decode_ws.partials, partials_bytes));
        g_flash_attn_decode_ws.partials_bytes = partials_bytes;
    }

    if (meta_bytes > g_flash_attn_decode_ws.meta_bytes) {
        if (g_flash_attn_decode_ws.meta) {
            HIP_CHECK(hipFree(g_flash_attn_decode_ws.meta));
        }
        HIP_CHECK(hipMalloc(&g_flash_attn_decode_ws.meta, meta_bytes));
        g_flash_attn_decode_ws.meta_bytes = meta_bytes;
    }

    return g_flash_attn_decode_ws;
}
} // namespace

// =============================================================================
// Constants - llama.cpp fattn-vec style
// =============================================================================

#if defined(CDNA_DEVICE) && CDNA_DEVICE == 1
#define FLASH_ATTN_CDNA_MODE 1
#define WARP_SIZE 64
#define NWARPS 2
#define FLASH_ATTN_USE_F16_SPECIALIZED 0
#else
#define FLASH_ATTN_CDNA_MODE 0
#define WARP_SIZE 32
#define NWARPS 4
#define FLASH_ATTN_USE_F16_SPECIALIZED 1
#endif

#define NTHREADS (WARP_SIZE * NWARPS) // 128 threads total
#define D_HEAD 128

// RDNA optimization: 2 threads collaborate on each KQ dot product
#define NTHREADS_KQ 2
// Tokens processed per iteration (each thread pair handles one token)
#define TOKENS_PER_ITER (NTHREADS / NTHREADS_KQ) // 64
// Decode kernel processes two tokens per thread-pair each iteration to amortize syncs.
#define TOKENS_PER_ITER_DECODE (TOKENS_PER_ITER * 2) // 128

#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)

// =============================================================================
// Helper Functions
// =============================================================================

// RDNA3 Intrinsic for dot2_f32_f16
static __device__ __forceinline__ float dot2_f32_f16(half2 a, half2 b, float c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__)
    // v_dot2_f32_f16 accumulates a * b into c
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(c) : "v"(a), "v"(b));
    return c;
#else
    float2 af = __half22float2(a);
    float2 bf = __half22float2(b);
    return c + af.x * bf.x + af.y * bf.y;
#endif
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE> static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, width));
    }
    return x;
}

// Sum lanes that share the same parity (lane_in_pair) inside a warp/wave.
static __device__ __forceinline__ float pair_reduce_sum(float x) {
#pragma unroll
    for (int offset = 2; offset < WARP_SIZE; offset <<= 1) {
        x += __shfl_xor(x, offset, WARP_SIZE);
    }
    return x;
}

static __device__ __forceinline__ half2 pair_reduce_sum_half2(half2 x) {
    union Half2Pack {
        int i;
        half2 h2;
    };
    Half2Pack acc;
    acc.h2 = x;
#pragma unroll
    for (int offset = 2; offset < WARP_SIZE; offset <<= 1) {
        Half2Pack remote;
        remote.i = __shfl_xor(acc.i, offset, WARP_SIZE);
        acc.h2 = __hadd2(acc.h2, remote.h2);
    }
    return acc.h2;
}

template <typename T> __device__ __forceinline__ T convert_out(float v);

template <> __device__ __forceinline__ half convert_out<half>(float v) { return __float2half(v); }

template <> __device__ __forceinline__ __hip_bfloat16 convert_out<__hip_bfloat16>(float v) { return __float2bfloat16(v); }

// Load 4 elements as float
template <typename T> __device__ __forceinline__ void load_vec4_as_float(const void *ptr, float *out);

template <> __device__ __forceinline__ void load_vec4_as_float<half>(const void *ptr, float *out) {
    const half2 *h2 = (const half2 *)ptr;
    float2 f0 = __half22float2(h2[0]);
    float2 f1 = __half22float2(h2[1]);
    out[0] = f0.x;
    out[1] = f0.y;
    out[2] = f1.x;
    out[3] = f1.y;
}

template <> __device__ __forceinline__ void load_vec4_as_float<__hip_bfloat16>(const void *ptr, float *out) {
    const __hip_bfloat16 *bf = (const __hip_bfloat16 *)ptr;
    out[0] = __bfloat162float(bf[0]);
    out[1] = __bfloat162float(bf[1]);
    out[2] = __bfloat162float(bf[2]);
    out[3] = __bfloat162float(bf[3]);
}

template <typename T> __device__ __forceinline__ void store_float_as_vec4(void *ptr, const float *in);

template <> __device__ __forceinline__ void store_float_as_vec4<half>(void *ptr, const float *in) {
    half2 *h2 = (half2 *)ptr;
    h2[0] = __float22half2_rn(make_float2(in[0], in[1]));
    h2[1] = __float22half2_rn(make_float2(in[2], in[3]));
}

template <> __device__ __forceinline__ void store_float_as_vec4<__hip_bfloat16>(void *ptr, const float *in) {
    __hip_bfloat16 *bf = (__hip_bfloat16 *)ptr;
    bf[0] = __float2bfloat16(in[0]);
    bf[1] = __float2bfloat16(in[1]);
    bf[2] = __float2bfloat16(in[2]);
    bf[3] = __float2bfloat16(in[3]);
}

// =============================================================================
// Flash Attention Decode Kernel - half2 optimized with 128-bit loads
// =============================================================================

// Specialized for FP16 to use half2 intrinsics
template <bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_vec_f16(const char *__restrict__ Q_base, const char *__restrict__ K_base,
                                                                  const char *__restrict__ V_base, half *__restrict__ dst,
                                                                  float *__restrict__ dst_partials, float2 *__restrict__ dst_meta,
                                                                  const float scale,
                                                                  const int ne11,                                 // seq_len_kv
                                                                  const int ne02,                                 // n_heads_Q
                                                                  const int ne12,                                 // n_heads_KV
                                                                  const int nb01, const int nb02, const int nb03, // Q strides
                                                                  const int nb11, const int nb12, const int nb13, // K strides
                                                                  const int nb21, const int nb22, const int nb23, // V strides
                                                                  const int batch_size) {

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing
    const int token_idx = tid / NTHREADS_KQ;    // 0-63
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;

    // =========================================================================
    // Load Q as half2 - using 128-bit loads (4x half2 per load)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64 floats
    constexpr int H2_PER_THREAD = D_PER_THREAD / 2;    // 32 half2s

    half2 Q_h2[H2_PER_THREAD];
    {
        const int4 *Q_I4 = (const int4 *)Q_ptr;
        // Each int4 = 16 bytes = 4x half2
        // H2_PER_THREAD = 32 half2s = 8x int4s
        const int base_i4 = lane_in_pair * 8;

        half2 scale_h2 = __float2half2_rn(scale);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int4 val_i4 = Q_I4[base_i4 + i];
            // Unpack int4 -> 4x half2
            half2 *h2_ptr = (half2 *)&val_i4;

            Q_h2[i * 4 + 0] = __hmul2(h2_ptr[0], scale_h2);
            Q_h2[i * 4 + 1] = __hmul2(h2_ptr[1], scale_h2);
            Q_h2[i * 4 + 2] = __hmul2(h2_ptr[2], scale_h2);
            Q_h2[i * 4 + 3] = __hmul2(h2_ptr[3], scale_h2);
        }
    }

    // =========================================================================
    // VKQ accumulator - half2
    // =========================================================================

    half2 VKQ[H2_PER_THREAD];
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i)
        VKQ[i] = __float2half2_rn(0.0f); // Init to 0

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER_DECODE]; // 128
    __shared__ float reduce_smem[NWARPS];      // 4

    // =========================================================================
    // Main loop
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_DECODE; k0 < ne11; k0 += gridDim.y * TOKENS_PER_ITER_DECODE) {
        const int k_idx0 = k0 + token_idx;
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER;

        // ---------------------------------------------------------------------
        // 1. Compute Q*K dot product (Mixed Precision)
        // ---------------------------------------------------------------------
        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < ne11);
        const bool in_bounds1 = (k_idx1 < ne11);

        if (in_bounds0) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx0 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                // Process 4 half2s
                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx1 * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                // Process 4 half2s
                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }
        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Find tile max
        // ---------------------------------------------------------------------
        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_DECODE; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        // ---------------------------------------------------------------------
        // 3. Rescale previous accumulator
        // ---------------------------------------------------------------------
        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

        half2 scale_h2 = __float2half2_rn(scale_prev);
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ[i] = __hmul2(VKQ[i], scale_h2);
        }

        // ---------------------------------------------------------------------
        // 4. Compute softmax and accumulate V
        // ---------------------------------------------------------------------
        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        // Sum reduction (only lane 0 of each pair contributes)
        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        // Accumulate V (FP16 accumulation) using 128-bit loads
        if (in_bounds0) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx0 * nb21);
            const int base_i4 = lane_in_pair * 8;

            half2 prob_h2 = __float2half2_rn(KQ_exp0);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
        if (in_bounds1) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx1 * nb21);
            const int base_i4 = lane_in_pair * 8;

            half2 prob_h2 = __float2half2_rn(KQ_exp1);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
        __syncthreads();
    }

// =========================================================================
// Final: Reduce VKQ across token pairs using warp shuffles
// =========================================================================

// Step 1: Sum within warp (pair reduction)
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i) {
        VKQ[i] = pair_reduce_sum_half2(VKQ[i]);
    }

    // Step 2: Sum across warps
    __shared__ half2 VKQ_smem_h2[NWARPS][D_HEAD / 2]; // D_HEAD/2 because half2

    if (lane_id < 2) {
        const int base = lane_id * H2_PER_THREAD;
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ_smem_h2[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    // Thread 0 sums across warps
    if (tid < D_HEAD / 2) {
        half2 val = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val = __hadd2(val, VKQ_smem_h2[w][tid]);
        }

        if constexpr (WRITE_PARTIALS) {
            const float2 val_f = __half22float2(val);
            const int base = (blockIdx.z * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid * 2 + 0] = val_f.x;
            dst_partials[base + tid * 2 + 1] = val_f.y;
        } else {
            float inv_sum = 1.0f / KQ_sum;
            half2 inv_sum_h2 = __float2half2_rn(inv_sum);
            val = __hmul2(val, inv_sum_h2);

            // Write output
            ((half2 *)(dst + blockIdx.z * D_HEAD))[tid] = val;
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[blockIdx.z * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

// Generic fallback for BF16 or other types
template <typename T, bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_vec_generic(const char *__restrict__ Q_base, const char *__restrict__ K_base,
                                                                      const char *__restrict__ V_base, T *__restrict__ dst,
                                                                      float *__restrict__ dst_partials, float2 *__restrict__ dst_meta,
                                                                      const float scale,
                                                                      const int ne11,                                 // seq_len_kv
                                                                      const int ne02,                                 // n_heads_Q
                                                                      const int ne12,                                 // n_heads_KV
                                                                      const int nb01, const int nb02, const int nb03, // Q strides
                                                                      const int nb11, const int nb12, const int nb13, // K strides
                                                                      const int nb21, const int nb22, const int nb23, // V strides
                                                                      const int batch_size) {

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing: threads 0-1 handle token 0, threads 2-3 handle token 1, etc.
    const int token_idx = tid / NTHREADS_KQ;    // 0-63: which token in tile
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1: which half of dimensions

    // Sequence and head indexing
    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    // Pointers
    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;

    // =========================================================================
    // Load Q - each thread loads D_HEAD/2 = 64 elements (16 vec4s)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64

    float Q_f[D_PER_THREAD];
    {
        const T *Q_T = (const T *)Q_ptr;
        const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q_T[base + i], &Q_f[i]);
        }

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q_f[i] *= scale;
        }
    }

    // =========================================================================
    // VKQ accumulator - each thread accumulates D_HEAD/2 dimensions
    // =========================================================================

    float VKQ[D_PER_THREAD] = {0.0f};

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    // Shared memory for KQ scores and cross-warp reduction
    __shared__ float KQ_smem[TOKENS_PER_ITER_DECODE]; // 128
    __shared__ float reduce_smem[NWARPS];      // 4

    // =========================================================================
    // Main loop over KV cache tiles
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER_DECODE; k0 < ne11; k0 += gridDim.y * TOKENS_PER_ITER_DECODE) {
        const int k_idx0 = k0 + token_idx;           // Global token index (first)
        const int k_idx1 = k_idx0 + TOKENS_PER_ITER; // Global token index (second)

        // ---------------------------------------------------------------------
        // 1. Compute Q*K dot product
        // ---------------------------------------------------------------------
        float KQ_val0 = -FLT_MAX;
        float KQ_val1 = -FLT_MAX;
        const bool in_bounds0 = (k_idx0 < ne11);
        const bool in_bounds1 = (k_idx1 < ne11);

        if (in_bounds0) {
            const T *K_row = (const T *)(K_ptr + k_idx0 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val0 = dot;
        }

        if (in_bounds1) {
            const T *K_row = (const T *)(K_ptr + k_idx1 * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            // Reduce across pair (lane 0 and 1)
            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val1 = dot;
        }
        // Store to shared memory (only lane 0 of each pair)
        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds0 ? KQ_val0 : -FLT_MAX;
            KQ_smem[token_idx + TOKENS_PER_ITER] = in_bounds1 ? KQ_val1 : -FLT_MAX;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // 2. Find tile max (parallel reduction)
        // ---------------------------------------------------------------------
        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER_DECODE; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        // ---------------------------------------------------------------------
        // 3. Rescale previous accumulator
        // ---------------------------------------------------------------------
        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ[i] *= scale_prev;
        }

        // ---------------------------------------------------------------------
        // 4. Compute softmax and accumulate V
        // ---------------------------------------------------------------------
        float my_score0 = KQ_smem[token_idx];
        float my_score1 = KQ_smem[token_idx + TOKENS_PER_ITER];
        float KQ_exp0 = in_bounds0 ? __expf(my_score0 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;
        float KQ_exp1 = in_bounds1 ? __expf(my_score1 + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        // Sum reduction (only lane 0 of each pair contributes)
        float tile_sum = (lane_in_pair == 0) ? (KQ_exp0 + KQ_exp1) : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        // Accumulate V (each thread accumulates its D_PER_THREAD dimensions)
        if (in_bounds0) {
            const T *V_row = (const T *)(V_ptr + k_idx0 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp0;
                VKQ[i + 1] += v_f[1] * KQ_exp0;
                VKQ[i + 2] += v_f[2] * KQ_exp0;
                VKQ[i + 3] += v_f[3] * KQ_exp0;
            }
        }
        if (in_bounds1) {
            const T *V_row = (const T *)(V_ptr + k_idx1 * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp1;
                VKQ[i + 1] += v_f[1] * KQ_exp1;
                VKQ[i + 2] += v_f[2] * KQ_exp1;
                VKQ[i + 3] += v_f[3] * KQ_exp1;
            }
        }
        __syncthreads();
    }

// =========================================================================
// Final: Reduce VKQ across token pairs using warp shuffles
// =========================================================================

// Step 1: Sum within warp (pair reduction)
#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        VKQ[i] = pair_reduce_sum(VKQ[i]);
    }

    // Step 2: Sum across warps using shared memory
    __shared__ float VKQ_smem[NWARPS][D_HEAD];

    // Thread 0 and 1 from each warp write to smem
    if (lane_id < 2) { // lane 0 = lower half, lane 1 = upper half
        const int base = lane_id * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ_smem[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    // Thread 0 sums across warps and writes output
    if (tid < D_HEAD) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val += VKQ_smem[w][tid];
        }

        if constexpr (WRITE_PARTIALS) {
            const int base = (blockIdx.z * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid] = val;
        } else {
            // Normalize
            float inv_sum = 1.0f / KQ_sum;

            // Write output
            T *dst_head = dst + blockIdx.z * D_HEAD;
            if constexpr (sizeof(T) == 2) {
                if (tid % 4 == 0) {
                    float out_f[4];
                    out_f[0] = val * inv_sum;
                    for (int j = 1; j < 4 && tid + j < D_HEAD; ++j) {
                        out_f[j] = 0.0f;
                        for (int w = 0; w < NWARPS; ++w) {
                            out_f[j] += VKQ_smem[w][tid + j];
                        }
                        out_f[j] *= inv_sum;
                    }
                    store_float_as_vec4<T>(&dst_head[tid], out_f);
                }
            }
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[blockIdx.z * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

// =============================================================================
// Flash Attention Prefill Kernels - Q length > 1 (no split-K by default)
// =============================================================================

template <bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_prefill_vec_f16(
    const char *__restrict__ Q_base, const char *__restrict__ K_base, const char *__restrict__ V_base, half *__restrict__ dst,
    float *__restrict__ dst_partials, float2 *__restrict__ dst_meta, const float scale, const int ne11, // seq_len_kv
    const int ne02,                                     // n_heads_Q
    const int ne12,                                     // n_heads_KV
    const int nb01, const int nb02, const int nb03,     // Q strides
    const int nb11, const int nb12, const int nb13,     // K strides
    const int nb21, const int nb22, const int nb23,     // V strides
    const int nbO1, const int nbO2, const int nbO3,     // O strides
    const int batch_size) {

    const int q_idx = blockIdx.x;
    const int effective_seq_len = min(ne11, q_idx + 1);

    // Thread indexing
    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    // Token pair indexing
    const int token_idx = tid / NTHREADS_KQ;    // 0-63
    const int lane_in_pair = tid % NTHREADS_KQ; // 0 or 1

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head + (int64_t)nb01 * q_idx;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;
    char *O_ptr = (char *)dst + (int64_t)nbO3 * sequence + (int64_t)nbO2 * head + (int64_t)nbO1 * q_idx;

    // =========================================================================
    // Load Q as half2 - using 128-bit loads (4x half2 per load)
    // =========================================================================

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ; // 64 floats
    constexpr int H2_PER_THREAD = D_PER_THREAD / 2;    // 32 half2s

    half2 Q_h2[H2_PER_THREAD];
    {
        const int4 *Q_I4 = (const int4 *)Q_ptr;
        const int base_i4 = lane_in_pair * 8;

        half2 scale_h2 = __float2half2_rn(scale);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int4 val_i4 = Q_I4[base_i4 + i];
            half2 *h2_ptr = (half2 *)&val_i4;

            Q_h2[i * 4 + 0] = __hmul2(h2_ptr[0], scale_h2);
            Q_h2[i * 4 + 1] = __hmul2(h2_ptr[1], scale_h2);
            Q_h2[i * 4 + 2] = __hmul2(h2_ptr[2], scale_h2);
            Q_h2[i * 4 + 3] = __hmul2(h2_ptr[3], scale_h2);
        }
    }

    // =========================================================================
    // VKQ accumulator - half2
    // =========================================================================

    half2 VKQ[H2_PER_THREAD];
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i)
        VKQ[i] = __float2half2_rn(0.0f);

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER];
    __shared__ float reduce_smem[NWARPS];

    // =========================================================================
    // Main loop
    // =========================================================================

    for (int k0 = blockIdx.y * TOKENS_PER_ITER; k0 < effective_seq_len; k0 += gridDim.y * TOKENS_PER_ITER) {
        const int k_idx = k0 + token_idx;

        float KQ_val = -FLT_MAX;
        const bool in_bounds = (k_idx < effective_seq_len);

        if (in_bounds) {
            const int4 *K_I4_row = (const int4 *)(K_ptr + k_idx * nb11);
            const int base_i4 = lane_in_pair * 8;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 k_val_i4 = K_I4_row[base_i4 + i];
                half2 *k_h2 = (half2 *)&k_val_i4;

                dot = dot2_f32_f16(Q_h2[i * 4 + 0], k_h2[0], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 1], k_h2[1], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 2], k_h2[2], dot);
                dot = dot2_f32_f16(Q_h2[i * 4 + 3], k_h2[3], dot);
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val = dot;
        }

        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds ? KQ_val : -FLT_MAX;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

        half2 scale_h2 = __float2half2_rn(scale_prev);
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ[i] = __hmul2(VKQ[i], scale_h2);
        }

        float my_score = KQ_smem[token_idx];
        float KQ_exp = in_bounds ? __expf(my_score + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        float tile_sum = (lane_in_pair == 0) ? KQ_exp : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        if (in_bounds) {
            const int4 *V_I4_row = (const int4 *)(V_ptr + k_idx * nb21);
            const int base_i4 = lane_in_pair * 8;

            half2 prob_h2 = __float2half2_rn(KQ_exp);

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int4 v_val_i4 = V_I4_row[base_i4 + i];
                half2 *v_h2 = (half2 *)&v_val_i4;

                VKQ[i * 4 + 0] = __hfma2(v_h2[0], prob_h2, VKQ[i * 4 + 0]);
                VKQ[i * 4 + 1] = __hfma2(v_h2[1], prob_h2, VKQ[i * 4 + 1]);
                VKQ[i * 4 + 2] = __hfma2(v_h2[2], prob_h2, VKQ[i * 4 + 2]);
                VKQ[i * 4 + 3] = __hfma2(v_h2[3], prob_h2, VKQ[i * 4 + 3]);
            }
        }
        __syncthreads();
    }

    // Final reduction
#pragma unroll
    for (int i = 0; i < H2_PER_THREAD; ++i) {
        VKQ[i] = pair_reduce_sum_half2(VKQ[i]);
    }

    __shared__ half2 VKQ_smem_h2[NWARPS][D_HEAD / 2];

    if (lane_id < 2) {
        const int base = lane_id * H2_PER_THREAD;
#pragma unroll
        for (int i = 0; i < H2_PER_THREAD; ++i) {
            VKQ_smem_h2[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    if (tid < D_HEAD / 2) {
        half2 val = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val = __hadd2(val, VKQ_smem_h2[w][tid]);
        }

        if constexpr (WRITE_PARTIALS) {
            const float2 val_f = __half22float2(val);
            const int base = ((blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid * 2 + 0] = val_f.x;
            dst_partials[base + tid * 2 + 1] = val_f.y;
        } else {
            float inv_sum = 1.0f / KQ_sum;
            half2 inv_sum_h2 = __float2half2_rn(inv_sum);
            val = __hmul2(val, inv_sum_h2);

            ((half2 *)O_ptr)[tid] = val;
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[(blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

template <typename T, bool WRITE_PARTIALS>
__global__ __launch_bounds__(NTHREADS, 1) void flash_attn_prefill_vec_generic(
    const char *__restrict__ Q_base, const char *__restrict__ K_base, const char *__restrict__ V_base, T *__restrict__ dst,
    float *__restrict__ dst_partials, float2 *__restrict__ dst_meta, const float scale, const int ne11, // seq_len_kv
    const int ne02,                                     // n_heads_Q
    const int ne12,                                     // n_heads_KV
    const int nb01, const int nb02, const int nb03,     // Q strides
    const int nb11, const int nb12, const int nb13,     // K strides
    const int nb21, const int nb22, const int nb23,     // V strides
    const int nbO1, const int nbO2, const int nbO3,     // O strides
    const int batch_size) {

    const int q_idx = blockIdx.x;
    const int effective_seq_len = min(ne11, q_idx + 1);

    const int tid = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int token_idx = tid / NTHREADS_KQ;
    const int lane_in_pair = tid % NTHREADS_KQ;

    const int sequence = blockIdx.z / ne02;
    const int head = blockIdx.z % ne02;
    if (sequence >= batch_size)
        return;

    const int gqa_ratio = ne02 / ne12;
    const int head_kv = head / gqa_ratio;

    const char *Q_ptr = Q_base + (int64_t)nb03 * sequence + (int64_t)nb02 * head + (int64_t)nb01 * q_idx;
    const char *K_ptr = K_base + (int64_t)nb13 * sequence + (int64_t)nb12 * head_kv;
    const char *V_ptr = V_base + (int64_t)nb23 * sequence + (int64_t)nb22 * head_kv;
    char *O_ptr = (char *)dst + (int64_t)nbO3 * sequence + (int64_t)nbO2 * head + (int64_t)nbO1 * q_idx;

    constexpr int D_PER_THREAD = D_HEAD / NTHREADS_KQ;

    float Q_f[D_PER_THREAD];
    {
        const T *Q_T = (const T *)Q_ptr;
        const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; i += 4) {
            load_vec4_as_float<T>(&Q_T[base + i], &Q_f[i]);
        }

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            Q_f[i] *= scale;
        }
    }

    float VKQ[D_PER_THREAD] = {0.0f};

    float KQ_max = -FLT_MAX / 2.0f;
    float KQ_sum = 0.0f;

    __shared__ float KQ_smem[TOKENS_PER_ITER];
    __shared__ float reduce_smem[NWARPS];

    for (int k0 = blockIdx.y * TOKENS_PER_ITER; k0 < effective_seq_len; k0 += gridDim.y * TOKENS_PER_ITER) {
        const int k_idx = k0 + token_idx;

        float KQ_val = -FLT_MAX;
        const bool in_bounds = (k_idx < effective_seq_len);

        if (in_bounds) {
            const T *K_row = (const T *)(K_ptr + k_idx * nb11);
            const int base = lane_in_pair * D_PER_THREAD;

            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float k_f[4];
                load_vec4_as_float<T>(&K_row[base + i], k_f);
                dot += Q_f[i + 0] * k_f[0];
                dot += Q_f[i + 1] * k_f[1];
                dot += Q_f[i + 2] * k_f[2];
                dot += Q_f[i + 3] * k_f[3];
            }

            dot += __shfl_xor(dot, 1, WARP_SIZE);
            KQ_val = dot;
        }

        if (lane_in_pair == 0) {
            KQ_smem[token_idx] = in_bounds ? KQ_val : -FLT_MAX;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int i = tid; i < TOKENS_PER_ITER; i += NTHREADS) {
            tile_max = fmaxf(tile_max, KQ_smem[i]);
        }
        tile_max = warp_reduce_max(tile_max);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_max;
        __syncthreads();

        if (tid == 0) {
            float m = reduce_smem[0];
            for (int w = 1; w < NWARPS; ++w)
                m = fmaxf(m, reduce_smem[w]);
            reduce_smem[0] = m + FATTN_KQ_MAX_OFFSET;
        }
        __syncthreads();
        float block_max = reduce_smem[0];

        float scale_prev = __expf(KQ_max - block_max);
        KQ_max = block_max;
        KQ_sum *= scale_prev;

#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ[i] *= scale_prev;
        }

        float my_score = KQ_smem[token_idx];
        float KQ_exp = in_bounds ? __expf(my_score + FATTN_KQ_MAX_OFFSET - KQ_max) : 0.0f;

        float tile_sum = (lane_in_pair == 0) ? KQ_exp : 0.0f;
        tile_sum = warp_reduce_sum(tile_sum);

        if (lane_id == 0)
            reduce_smem[warp_id] = tile_sum;
        __syncthreads();

        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < NWARPS; ++w)
                s += reduce_smem[w];
            reduce_smem[0] = s;
        }
        __syncthreads();
        KQ_sum += reduce_smem[0];

        if (in_bounds) {
            const T *V_row = (const T *)(V_ptr + k_idx * nb21);
            const int base = lane_in_pair * D_PER_THREAD;

#pragma unroll
            for (int i = 0; i < D_PER_THREAD; i += 4) {
                float v_f[4];
                load_vec4_as_float<T>(&V_row[base + i], v_f);
                VKQ[i + 0] += v_f[0] * KQ_exp;
                VKQ[i + 1] += v_f[1] * KQ_exp;
                VKQ[i + 2] += v_f[2] * KQ_exp;
                VKQ[i + 3] += v_f[3] * KQ_exp;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < D_PER_THREAD; ++i) {
        VKQ[i] = pair_reduce_sum(VKQ[i]);
    }

    __shared__ float VKQ_smem[NWARPS][D_HEAD];

    if (lane_id < 2) {
        const int base = lane_id * D_PER_THREAD;
#pragma unroll
        for (int i = 0; i < D_PER_THREAD; ++i) {
            VKQ_smem[warp_id][base + i] = VKQ[i];
        }
    }
    __syncthreads();

    if (tid < D_HEAD) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            val += VKQ_smem[w][tid];
        }

        if constexpr (WRITE_PARTIALS) {
            const int base = ((blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y) * D_HEAD;
            dst_partials[base + tid] = val;
        } else {
            float inv_sum = 1.0f / KQ_sum;

            T *dst_head = (T *)O_ptr;
            if constexpr (sizeof(T) == 2) {
                if (tid % 4 == 0) {
                    float out_f[4];
                    out_f[0] = val * inv_sum;
                    for (int j = 1; j < 4 && tid + j < D_HEAD; ++j) {
                        out_f[j] = 0.0f;
                        for (int w = 0; w < NWARPS; ++w) {
                            out_f[j] += VKQ_smem[w][tid + j];
                        }
                        out_f[j] *= inv_sum;
                    }
                    store_float_as_vec4<T>(&dst_head[tid], out_f);
                }
            }
        }
    }

    if constexpr (WRITE_PARTIALS) {
        if (tid == 0) {
            dst_meta[(blockIdx.z * gridDim.x + q_idx) * gridDim.y + blockIdx.y] = make_float2(KQ_max, KQ_sum);
        }
    }
}

template <typename T>
__global__ void flash_attn_decode_combine(const float *__restrict__ partials, const float2 *__restrict__ meta, T *__restrict__ dst,
                                          int parallel_blocks) {
    const int tid = threadIdx.x;
    if (tid >= D_HEAD)
        return;

    const int head_idx = blockIdx.x;
    const int base = head_idx * parallel_blocks;

    float kqmax = meta[base].x;
    for (int i = 1; i < parallel_blocks; ++i) {
        kqmax = fmaxf(kqmax, meta[base + i].x);
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int i = 0; i < parallel_blocks; ++i) {
        const float2 m = meta[base + i];
        const float scale = __expf(m.x - kqmax);
        numerator += scale * partials[(base + i) * D_HEAD + tid];
        denominator += scale * m.y;
    }

    const float out = denominator > 0.0f ? (numerator / denominator) : 0.0f;
    dst[head_idx * D_HEAD + tid] = convert_out<T>(out);
}

// =============================================================================
// Launcher
// =============================================================================

extern "C" void launch_flash_attn_decode_hip(const void *Q, const void *K, const void *V, const void *mask, void *dst, int batch_size,
                                             int n_heads_Q, int n_heads_KV, int head_dim, int seq_len_kv, float scale, int stride_Q_seq,
                                             int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                             int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_mask_seq, bool is_bf16,
                                             hipStream_t stream) {
    dim3 block(WARP_SIZE, NWARPS); // 2D launch; always 128 threads total
    size_t smem_size = 0;          // kernel uses static shared memory only

    const int num_k_tiles = (seq_len_kv + TOKENS_PER_ITER_DECODE - 1) / TOKENS_PER_ITER_DECODE;
    int parallel_blocks = 1;

    if (num_k_tiles > 1) {
        int max_blocks_per_sm = 1;
        if (is_bf16) {
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, flash_attn_vec_generic<__hip_bfloat16, true>, block.x * block.y * block.z, smem_size));
        } else {
#if FLASH_ATTN_USE_F16_SPECIALIZED
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, flash_attn_vec_f16<true>, block.x * block.y * block.z, smem_size));
#else
            HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm, flash_attn_vec_generic<half, true>, block.x * block.y * block.z, smem_size));
#endif
        }

        int device = 0;
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDevice(&device));
        HIP_CHECK(hipGetDeviceProperties(&props, device));

        const int blocks_per_head = batch_size * n_heads_Q;
        const int target_blocks = props.multiProcessorCount * max_blocks_per_sm;
        const int desired_parallel = std::max(1, (target_blocks + blocks_per_head - 1) / blocks_per_head);

        int min_parallel = 1;
        if (num_k_tiles >= 32) {
            min_parallel = 8;
        } else if (num_k_tiles >= 16) {
            min_parallel = 4;
        } else if (num_k_tiles >= 8) {
            min_parallel = 2;
        }

        const int kMaxParallelBlocks = 32;
        parallel_blocks = std::min(num_k_tiles, std::min(std::max(desired_parallel, min_parallel), kMaxParallelBlocks));
    }

    if (is_bf16) {
        // For BF16, small sequences run best without split-K.
        // For long sequences, a light split-K (2) improves throughput.
        const int bf16_split_k_threshold = 32 * TOKENS_PER_ITER; // 2048 tokens
        if (seq_len_kv >= bf16_split_k_threshold) {
            parallel_blocks = std::min(parallel_blocks, 2);
            if (parallel_blocks < 2) {
                parallel_blocks = 2;
            }
        } else {
            parallel_blocks = 1;
        }
    }

#if FLASH_ATTN_CDNA_MODE
    // Prefer deterministic single-block accumulation on CDNA.
    parallel_blocks = 1;
#endif

    const char *parallel_override = std::getenv("FLASH_ATTN_DECODE_PARALLEL_BLOCKS");
    if (parallel_override && std::atoi(parallel_override) > 0) {
        const int override_val = std::atoi(parallel_override);
        parallel_blocks = std::min(num_k_tiles, override_val);
    }

    const char *disable_parallel = std::getenv("FLASH_ATTN_DECODE_DISABLE_PARALLEL");
    if (disable_parallel && std::atoi(disable_parallel) != 0) {
        parallel_blocks = 1;
    }

    if (parallel_blocks <= 1) {
        dim3 grid(1, 1, batch_size * n_heads_Q);
        if (is_bf16) {
            flash_attn_vec_generic<__hip_bfloat16, false>
                <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, nullptr,
                                                     nullptr, scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head,
                                                     stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq,
                                                     stride_V_head, stride_V_batch, batch_size);
        } else {
#if FLASH_ATTN_USE_F16_SPECIALIZED
            // Use optimized FP16 kernel on wave32 platforms.
            flash_attn_vec_f16<false><<<grid, block, smem_size, stream>>>(
                (const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr, scale, seq_len_kv, n_heads_Q, n_heads_KV,
                stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
                stride_V_batch, batch_size);
#else
            // CDNA correctness path: generic FP16 kernel.
            flash_attn_vec_generic<half, false>
                <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr,
                                                     scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch,
                                                     stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch,
                                                     batch_size);
#endif
        }
        return;
    }

    // Parallel blocks path for long KV: compute partials + meta, then combine.
    const size_t partials_bytes = static_cast<size_t>(batch_size) * n_heads_Q * parallel_blocks * D_HEAD * sizeof(float);
    const size_t meta_bytes = static_cast<size_t>(batch_size) * n_heads_Q * parallel_blocks * sizeof(float2);

    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    FlashAttnDecodeWorkspace ws = acquire_flash_attn_decode_workspace(device, partials_bytes, meta_bytes);
    float *partials = static_cast<float *>(ws.partials);
    float2 *meta = static_cast<float2 *>(ws.meta);

    dim3 grid(1, parallel_blocks, batch_size * n_heads_Q);
    if (is_bf16) {
        flash_attn_vec_generic<__hip_bfloat16, true>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, partials, meta,
                                                 scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch,
                                                 stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch,
                                                 batch_size);
        dim3 combine_grid(batch_size * n_heads_Q, 1, 1);
        dim3 combine_block(D_HEAD, 1, 1);
        flash_attn_decode_combine<__hip_bfloat16><<<combine_grid, combine_block, 0, stream>>>(partials, meta,
                                                                                             (__hip_bfloat16 *)dst, parallel_blocks);
    } else {
#if FLASH_ATTN_USE_F16_SPECIALIZED
        flash_attn_vec_f16<true><<<grid, block, smem_size, stream>>>(
            (const char *)Q, (const char *)K, (const char *)V, (half *)dst, partials, meta, scale, seq_len_kv, n_heads_Q, n_heads_KV,
            stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head,
            stride_V_batch, batch_size);
#else
        flash_attn_vec_generic<half, true>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, partials, meta, scale,
                                                 seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq,
                                                 stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch, batch_size);
#endif
        dim3 combine_grid(batch_size * n_heads_Q, 1, 1);
        dim3 combine_block(D_HEAD, 1, 1);
        flash_attn_decode_combine<half><<<combine_grid, combine_block, 0, stream>>>(partials, meta, (half *)dst, parallel_blocks);
    }
}

extern "C" void launch_flash_attn_prefill_hip(const void *Q, const void *K, const void *V, void *dst, int batch_size, int n_heads_Q,
                                              int n_heads_KV, int head_dim, int seq_len_q, int seq_len_kv, float scale, int stride_Q_seq,
                                              int stride_Q_head, int stride_Q_batch, int stride_K_seq, int stride_K_head, int stride_K_batch,
                                              int stride_V_seq, int stride_V_head, int stride_V_batch, int stride_O_seq, int stride_O_head,
                                              int stride_O_batch, bool is_bf16, hipStream_t stream) {
    dim3 block(WARP_SIZE, NWARPS);
    size_t smem_size = 0;

    if (head_dim != D_HEAD) {
        std::cerr << "FlashAttn prefill only supports head_dim=" << D_HEAD << std::endl;
        return;
    }

    dim3 grid(seq_len_q, 1, batch_size * n_heads_Q);

    if (is_bf16) {
        flash_attn_prefill_vec_generic<__hip_bfloat16, false>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (__hip_bfloat16 *)dst, nullptr, nullptr,
                                                 scale, seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch,
                                                 stride_K_seq, stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch,
                                                 stride_O_seq, stride_O_head, stride_O_batch, batch_size);
    } else {
#if FLASH_ATTN_USE_F16_SPECIALIZED
        flash_attn_prefill_vec_f16<false>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr, scale,
                                                 seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq,
                                                 stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch, stride_O_seq,
                                                 stride_O_head, stride_O_batch, batch_size);
#else
        flash_attn_prefill_vec_generic<half, false>
            <<<grid, block, smem_size, stream>>>((const char *)Q, (const char *)K, (const char *)V, (half *)dst, nullptr, nullptr, scale,
                                                 seq_len_kv, n_heads_Q, n_heads_KV, stride_Q_seq, stride_Q_head, stride_Q_batch, stride_K_seq,
                                                 stride_K_head, stride_K_batch, stride_V_seq, stride_V_head, stride_V_batch, stride_O_seq,
                                                 stride_O_head, stride_O_batch, batch_size);
#endif
    }
}
