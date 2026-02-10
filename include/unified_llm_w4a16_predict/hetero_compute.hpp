#pragma once

#include <future>
#include <hip/hip_runtime.h>
#include <torch/torch.h>

// Custom GEMV wrapper (M = 1) - Unpacked
// Custom GEMV wrapper (M = 1) - Unpacked
std::future<int> hetero_matmul_out_gemv_unpacked_M(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                   const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                   int64_t out_features, std::string layer_type);

// Custom GEMV wrapper (M = 1) - Unpacked - Explicit Config
std::future<int> hetero_matmul_out_gemv_unpacked(torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &qweights,
                                                 const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features,
                                                 int64_t out_features, int cpuN, int cpuThreads, std::string layer_type,
                                                 hipEvent_t hip_event = nullptr);

// Initialize GEMM resources (HIP events, etc.)
void init_gemm_resources();
