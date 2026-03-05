#include "hipkernels/w4a16_gemm_unpacked.hpp"
#include "hipkernels/w4a16_gemv_unpacked.hpp"
#include <chrono>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <torch/torch.h>
#include <vector>

#define HIP_CHECK(status)                                                                                                                  \
    if (status != hipSuccess) {                                                                                                            \
        std::cerr << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl;                     \
        std::exit(1);                                                                                                                      \
    }

void hip_synchronize() { HIP_CHECK(hipDeviceSynchronize()); }

int main() {
    constexpr int64_t kNumExperts = 8;
    constexpr int64_t kExpertsPerToken = 2;
    constexpr int64_t kPrefillPromptLen = 16384;
    constexpr int64_t kM = (kPrefillPromptLen * kExpertsPerToken) / kNumExperts; // ~tokens routed to each expert during prefill
    constexpr int64_t kK = 4096;
    constexpr int64_t kN = 14336;
    constexpr int64_t kNumExpertsGemv = kExpertsPerToken; // generation activates top-2 experts/token
    constexpr int64_t kGroupSize = 128;
    constexpr int64_t kWarmup = 8;
    constexpr int64_t kIters = 20;

    std::cout << "test_layout_hip_3d (hardcoded)\n";
    std::cout << "Experts=" << kNumExperts << " ExpertsPerToken=" << kExpertsPerToken << " PromptTokens=" << kPrefillPromptLen
              << " M(per-expert)=" << kM << " K=" << kK << " N=" << kN << " group_size=" << kGroupSize << std::endl;

    if (!torch::cuda::is_available()) {
        std::cerr << "HIP available check failed" << std::endl;
        return 1;
    }

    auto device = torch::kCUDA;
    const int64_t num_groups = kK / kGroupSize;

    torch::manual_seed(42);

    auto input_gemm_3d = torch::rand({kNumExperts, kM, kK}, torch::kBFloat16).to(device) * 0.1f;
    auto output_gemm_3d = torch::zeros({kNumExperts, kM, kN}, torch::kBFloat16).to(device);

    auto input_gemv_3d = torch::rand({kNumExpertsGemv, kK}, torch::kBFloat16).to(device) * 0.1f;
    auto output_gemv_3d = torch::zeros({kNumExpertsGemv, kN}, torch::kBFloat16).to(device);

    std::vector<torch::Tensor> qweights_all;
    std::vector<torch::Tensor> scales_all;
    std::vector<torch::Tensor> zeros_all;
    qweights_all.reserve(kNumExperts);
    scales_all.reserve(kNumExperts);
    zeros_all.reserve(kNumExperts);

    for (int64_t e = 0; e < kNumExperts; ++e) {
        auto qweight_raw = torch::randint(0, 16, {kK, kN}, torch::kUInt8).to(device);
        auto scales_raw = (torch::rand({num_groups, kN}, torch::kFloat32).to(torch::kBFloat16) * 0.1f).to(device);
        auto zeros_raw = torch::randint(0, 16, {num_groups, kN}, torch::kInt8).to(device);

        auto w_nk = qweight_raw.t().contiguous(); // [N, K]
        auto w_pairs = w_nk.view({kN, kK / 2, 2});
        auto w_low = w_pairs.select(2, 0);
        auto w_high = w_pairs.select(2, 1);
        auto qweights_gpu = (w_low & 0x0F) | torch::bitwise_left_shift(w_high & 0x0F, 4);
        qweights_gpu = qweights_gpu.to(torch::kUInt8).contiguous();

        auto scales_gpu = scales_raw.t().contiguous();                // [N, groups]
        auto zeros_gpu = zeros_raw.t().contiguous().to(torch::kInt8); // [N, groups]

        qweights_all.push_back(qweights_gpu);
        scales_all.push_back(scales_gpu);
        zeros_all.push_back(zeros_gpu);
    }

    std::vector<int64_t> qweight_ptrs(kNumExperts);
    std::vector<int64_t> scales_ptrs(kNumExperts);
    std::vector<int64_t> zeros_ptrs(kNumExperts);
    for (int64_t e = 0; e < kNumExperts; ++e) {
        qweight_ptrs[e] = reinterpret_cast<int64_t>(qweights_all[e].data_ptr<uint8_t>());
        scales_ptrs[e] = reinterpret_cast<int64_t>(scales_all[e].data_ptr<at::BFloat16>());
        zeros_ptrs[e] = reinterpret_cast<int64_t>(zeros_all[e].data_ptr<int8_t>());
    }

    std::cout << "Pointer arrays ready for " << kNumExperts << " experts" << std::endl;

    std::cout << "================ GEMM =================" << std::endl;

    // GEMM correctness for all experts
    {
        hipkernels::w4a16_gemm_unpacked_fused_3d(output_gemm_3d, input_gemm_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExperts);
        hip_synchronize();

        int64_t failures = 0;
        for (int64_t e = 0; e < kNumExperts; ++e) {
            auto qweights = qweights_all[e];
            auto scales = scales_all[e];
            auto zeros = zeros_all[e];

            auto w_low = qweights & 0x0F;
            auto w_high = torch::bitwise_right_shift(qweights, 4) & 0x0F;
            auto w_pairs = torch::stack({w_low, w_high}, -1); // [N, K/2, 2]
            auto w_int = w_pairs.view({kN, kK}).to(torch::kBFloat16);
            auto s_exp = scales.repeat_interleave(kGroupSize, 1);
            auto z_exp = zeros.to(torch::kBFloat16).repeat_interleave(kGroupSize, 1);
            auto w_ref = (w_int - z_exp) * s_exp;

            auto y_ref = torch::matmul(input_gemm_3d[e], w_ref.t());
            auto y_out = output_gemm_3d[e];

            std::cout << "Expert " << e << " - First 8 values of y_ref_gemm: ";
            {
                auto y_ref_cpu = y_ref.cpu();
                std::cout << std::fixed << std::setprecision(2);
                for (int i = 0; i < 8; ++i)
                    std::cout << y_ref_cpu[0][i].item<float>() << " ";
                std::cout << std::endl;
            }

            std::cout << "Expert " << e << " - First 8 values of output_gemm: ";
            {
                auto y_out_cpu = y_out.cpu();
                std::cout << std::fixed << std::setprecision(2);
                for (int i = 0; i < 8; ++i)
                    std::cout << y_out_cpu[0][i].item<float>() << " ";
                std::cout << std::endl;
            }

            bool match = torch::allclose(y_ref, y_out, 0.01, 0.1);
            if (!match) {
                failures++;
                std::cout << "GEMM correctness FAIL (expert " << e << ")" << std::endl;
                auto diff = (y_ref - y_out).abs();
                auto max_val = diff.max();
                auto max_idx = diff.argmax();
                auto N_dim = y_ref.size(1);
                long flat_idx = max_idx.item<long>();
                long row = flat_idx / N_dim;
                long col = flat_idx % N_dim;
                std::cout << "Max Diff: " << max_val.item<float>() << " at [" << row << ", " << col << "]" << std::endl;
                std::cout << "Ref: " << y_ref[row][col].item<float>() << " Kernel: " << y_out[row][col].item<float>() << std::endl;
            }
        }
        if (failures == 0) {
            std::cout << "GEMM correctness: PASS (all experts)" << std::endl;
        }
    }

    // Benchmark GEMM 3D
    for (int w = 0; w < kWarmup; ++w) {
        hipkernels::w4a16_gemm_unpacked_fused_3d(output_gemm_3d, input_gemm_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExperts);
    }
    hip_synchronize();

    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIters; ++it) {
        hipkernels::w4a16_gemm_unpacked_fused_3d(output_gemm_3d, input_gemm_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExperts);
    }
    hip_synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double avg_time = elapsed.count() / static_cast<double>(kIters);
    double ops = 2.0 * static_cast<double>(kNumExperts) * static_cast<double>(kM) * static_cast<double>(kK) * static_cast<double>(kN);
    double tops = ops / (avg_time * 1e12);

    std::cout << "GEMM 3D Avg Time: " << (avg_time * 1000.0) << " ms\n";
    std::cout << "GEMM 3D Performance: " << tops << " TOPS" << std::endl;

    std::cout << "================ GEMV =================" << std::endl;

    // GEMV correctness for all experts
    {
        hipkernels::w4a16_gemv_unpacked_fused_3d(output_gemv_3d, input_gemv_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExpertsGemv);
        hip_synchronize();

        int64_t failures = 0;
        for (int64_t e = 0; e < kNumExpertsGemv; ++e) {
            auto qweights = qweights_all[e];
            auto scales = scales_all[e];
            auto zeros = zeros_all[e];

            auto w_low = qweights & 0x0F;
            auto w_high = torch::bitwise_right_shift(qweights, 4) & 0x0F;
            auto w_pairs = torch::stack({w_low, w_high}, -1); // [N, K/2, 2]
            auto w_int = w_pairs.view({kN, kK}).to(torch::kBFloat16);
            auto s_exp = scales.repeat_interleave(kGroupSize, 1);
            auto z_exp = zeros.to(torch::kBFloat16).repeat_interleave(kGroupSize, 1);
            auto w_ref = (w_int - z_exp) * s_exp;

            auto y_ref = torch::matmul(input_gemv_3d[e].unsqueeze(0), w_ref.t());
            auto y_ref_1d = y_ref.squeeze(0);
            auto y_out = output_gemv_3d[e];

            std::cout << "Expert " << e << " - First 8 values of y_ref_gemv: ";
            {
                auto y_ref_cpu = y_ref_1d.cpu();
                std::cout << std::fixed << std::setprecision(2);
                for (int i = 0; i < 8; ++i)
                    std::cout << y_ref_cpu[i].item<float>() << " ";
                std::cout << std::endl;
            }

            std::cout << "Expert " << e << " - First 8 values of output_gemv: ";
            {
                auto y_out_cpu = y_out.cpu();
                std::cout << std::fixed << std::setprecision(2);
                for (int i = 0; i < 8; ++i)
                    std::cout << y_out_cpu[i].item<float>() << " ";
                std::cout << std::endl;
            }

            bool match = torch::allclose(y_ref_1d, y_out, 0.01, 0.1);
            if (!match) {
                failures++;
                std::cout << "GEMV correctness FAIL (expert " << e << ")" << std::endl;
                auto diff = (y_ref_1d - y_out).abs();
                auto max_val = diff.max();
                auto max_idx = diff.argmax();
                long col = max_idx.item<long>();
                std::cout << "Max Diff: " << max_val.item<float>() << " at [0, " << col << "]" << std::endl;
                std::cout << "Ref: " << y_ref_1d[col].item<float>() << " Kernel: " << y_out[col].item<float>() << std::endl;
            }
        }
        if (failures == 0) {
            std::cout << "GEMV correctness: PASS (all " << kNumExpertsGemv << " experts)" << std::endl;
        }
    }

    // Benchmark GEMV 3D
    for (int w = 0; w < kWarmup; ++w) {
        hipkernels::w4a16_gemv_unpacked_fused_3d(output_gemv_3d, input_gemv_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExpertsGemv);
    }
    hip_synchronize();

    start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < kIters; ++it) {
        hipkernels::w4a16_gemv_unpacked_fused_3d(output_gemv_3d, input_gemv_3d, qweight_ptrs, scales_ptrs, zeros_ptrs, kK, kN, kGroupSize,
                                                 kNumExpertsGemv);
    }
    hip_synchronize();
    end = std::chrono::high_resolution_clock::now();

    elapsed = end - start;
    avg_time = elapsed.count() / static_cast<double>(kIters);
    ops = 2.0 * static_cast<double>(kNumExpertsGemv) * static_cast<double>(kK) * static_cast<double>(kN);
    tops = ops / (avg_time * 1e12);

    std::cout << "GEMV 3D Avg Time: " << (avg_time * 1000.0) << " ms\n";
    std::cout << "GEMV 3D Performance: " << tops << " TOPS" << std::endl;

    return 0;
}
