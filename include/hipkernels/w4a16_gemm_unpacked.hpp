#pragma once

#include <torch/torch.h>
#include <vector>

namespace hipkernels {

void w4a16_gemm_unpacked_fused(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights,
                               const torch::Tensor &scales, const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                               int64_t group_size);

void w4a16_gemm_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const torch::Tensor &qweights_ptrs,
                                  const torch::Tensor &scales_ptrs, const torch::Tensor &zeros_ptrs, int64_t in_features,
                                  int64_t out_features, int64_t group_size, int64_t num_experts);

void w4a16_gemm_unpacked_fused_3d(torch::Tensor &output, const torch::Tensor &input, const std::vector<int64_t> &qweights_ptrs,
                                  const std::vector<int64_t> &scales_ptrs, const std::vector<int64_t> &zeros_ptrs,
                                  int64_t in_features, int64_t out_features, int64_t group_size, int64_t num_experts);

torch::Tensor w4a16_gemm_unpacked_alloc_and_compute(const torch::Tensor &input, const torch::Tensor &qweights, const torch::Tensor &scales,
                                                    const torch::Tensor &zeros, int64_t in_features, int64_t out_features,
                                                    int64_t group_size);

} // namespace hipkernels
