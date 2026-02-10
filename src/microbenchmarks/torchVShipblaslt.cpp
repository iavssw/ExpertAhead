#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <torch/cuda.h>
#include <torch/torch.h>

namespace {
constexpr int64_t kM = 4096;
constexpr int64_t kK = 14336;
constexpr int64_t kN = 4096;
constexpr int kIters = 8;
constexpr int kTuneIters = 6;
constexpr int kAlgoCount = 20;
constexpr int kCheckSamples = 8192;
constexpr int kSeed = 0;
constexpr int kWorkspaceMB = 64;

void check_hip(hipError_t err, const char *msg) {
    if (err != hipSuccess) {
        std::cerr << "HIP error: " << msg << ": " << hipGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void check_hipblaslt(hipblasStatus_t status, const char *msg) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "hipBLASLt error: " << msg << " (status " << status << ")" << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct TimingStats {
    std::vector<double> ms;
    double avg_ms = 0.0;
    double avg_tops = 0.0;
};

TimingStats summarize(const std::vector<double> &ms, double ops) {
    TimingStats stats;
    stats.ms = ms;
    double sum = 0.0;
    for (double v : ms) {
        sum += v;
    }
    stats.avg_ms = sum / static_cast<double>(ms.size());
    stats.avg_tops = (ops / (stats.avg_ms / 1000.0)) / 1.0e12;
    return stats;
}

struct HipMatmulConfig {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t A = nullptr;
    hipblasLtMatrixLayout_t B = nullptr;
    hipblasLtMatrixLayout_t C = nullptr;
    hipblasLtMatrixLayout_t D = nullptr;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    hipblasLtMatmulAlgo_t best_algo{};
    uint64_t best_workspace = 0;
};

HipMatmulConfig make_config(hipblasLtHandle_t handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int64_t A_rows, int64_t A_cols,
                            int64_t B_rows, int64_t B_cols,
                            int64_t C_rows, int64_t C_cols,
                            uint64_t max_workspace_size) {
    HipMatmulConfig cfg;

    check_hipblaslt(hipblasLtMatmulDescCreate(&cfg.desc, HIPBLAS_COMPUTE_32F_FAST_16BF, HIP_R_32F),
                    "hipblasLtMatmulDescCreate");

    check_hipblaslt(hipblasLtMatmulDescSetAttribute(cfg.desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                                    &transA, sizeof(transA)),
                    "set TRANSA");
    check_hipblaslt(hipblasLtMatmulDescSetAttribute(cfg.desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                                    &transB, sizeof(transB)),
                    "set TRANSB");

    check_hipblaslt(hipblasLtMatrixLayoutCreate(&cfg.A, HIP_R_16BF, A_rows, A_cols, A_rows),
                    "A layout create");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&cfg.B, HIP_R_16BF, B_rows, B_cols, B_rows),
                    "B layout create");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&cfg.C, HIP_R_16BF, C_rows, C_cols, C_rows),
                    "C layout create");
    check_hipblaslt(hipblasLtMatrixLayoutCreate(&cfg.D, HIP_R_16BF, C_rows, C_cols, C_rows),
                    "D layout create");

    hipblasLtMatmulPreference_t pref;
    check_hipblaslt(hipblasLtMatmulPreferenceCreate(&pref), "hipblasLtMatmulPreferenceCreate");
    check_hipblaslt(hipblasLtMatmulPreferenceSetAttribute(
                        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                        &max_workspace_size, sizeof(max_workspace_size)),
                    "hipblasLtMatmulPreferenceSetAttribute");

    const int request_solutions = kAlgoCount;
    cfg.heuristics.resize(request_solutions);
    int returned_algo_count = 0;
    check_hipblaslt(hipblasLtMatmulAlgoGetHeuristic(
                        handle, cfg.desc, cfg.A, cfg.B, cfg.C, cfg.D, pref,
                        request_solutions, cfg.heuristics.data(), &returned_algo_count),
                    "hipblasLtMatmulAlgoGetHeuristic");

    if (returned_algo_count == 0) {
        std::cerr << "No hipBLASLt heuristic solution found." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cfg.heuristics.resize(returned_algo_count);

    return cfg;
}

void tune_best_algo(const char *label,
                    hipblasLtHandle_t handle,
                    HipMatmulConfig &cfg,
                    void *A_ptr,
                    void *B_ptr,
                    void *C_add,
                    void *D_ptr,
                    void *workspace,
                    uint64_t workspace_size,
                    hipStream_t stream) {
    std::cout << "\nTuning " << label << " hipBLASLt (" << cfg.heuristics.size()
              << " algos, " << kTuneIters << " iters each)" << std::endl;

    float alpha = 1.0f;
    float beta = 0.0f;
    double best_ms = 1e30;
    size_t best_idx = 0;

    for (size_t i = 0; i < cfg.heuristics.size(); ++i) {
        const auto &heur = cfg.heuristics[i];
        if (heur.workspaceSize > workspace_size) {
            continue;
        }
        double sum_ms = 0.0;
        for (int iter = 0; iter < kTuneIters; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            check_hipblaslt(hipblasLtMatmul(
                                handle,
                                cfg.desc,
                                &alpha,
                                A_ptr,
                                cfg.A,
                                B_ptr,
                                cfg.B,
                                &beta,
                                C_add,
                                cfg.C,
                                D_ptr,
                                cfg.D,
                                &heur.algo,
                                workspace,
                                heur.workspaceSize,
                                stream),
                            "hipblasLtMatmul");
            check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize tune");
            auto end = std::chrono::high_resolution_clock::now();
            sum_ms += std::chrono::duration<double, std::milli>(end - start).count();
        }
        double avg_ms = sum_ms / static_cast<double>(kTuneIters);
        if (avg_ms < best_ms) {
            best_ms = avg_ms;
            best_idx = i;
        }
    }

    cfg.best_algo = cfg.heuristics[best_idx].algo;
    cfg.best_workspace = cfg.heuristics[best_idx].workspaceSize;
    std::cout << "Tuning complete: best avg " << best_ms << " ms, workspace "
              << cfg.best_workspace / (1024.0 * 1024.0) << " MB" << std::endl;
}

TimingStats run_hip_case(const char *label,
                         hipblasLtHandle_t handle,
                         const HipMatmulConfig &cfg,
                         void *A_ptr,
                         void *B_ptr,
                         void *C_add,
                         void *D_ptr,
                         void *workspace,
                         hipStream_t stream,
                         double ops) {
    std::cout << "\n" << label << " hipBLASLt per-iter" << std::endl;
    std::vector<double> ms;
    ms.reserve(kIters);

    float alpha = 1.0f;
    float beta = 0.0f;

    for (int i = 0; i < kIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        check_hipblaslt(hipblasLtMatmul(
                            handle,
                            cfg.desc,
                            &alpha,
                            A_ptr,
                            cfg.A,
                            B_ptr,
                            cfg.B,
                            &beta,
                            C_add,
                            cfg.C,
                            D_ptr,
                            cfg.D,
                            &cfg.best_algo,
                            workspace,
                            cfg.best_workspace,
                            stream),
                        "hipblasLtMatmul");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize hip");
        auto end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double tops = (ops / (iter_ms / 1000.0)) / 1.0e12;
        ms.push_back(iter_ms);
        std::cout << "iter " << i << ": " << iter_ms << " ms, " << tops << " TOPS" << std::endl;
    }

    auto stats = summarize(ms, ops);
    std::cout << label << " hipBLASLt avg: " << stats.avg_ms << " ms, " << stats.avg_tops << " TOPS" << std::endl;
    return stats;
}

TimingStats run_torch_case(const char *label,
                           const std::function<torch::Tensor()> &fn,
                           torch::Tensor &out,
                           double ops) {
    std::cout << "\n" << label << " Torch per-iter" << std::endl;
    std::vector<double> ms;
    ms.reserve(kIters);

    for (int i = 0; i < kIters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        out = fn();
        torch::cuda::synchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double iter_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double tops = (ops / (iter_ms / 1000.0)) / 1.0e12;
        ms.push_back(iter_ms);
        std::cout << "iter " << i << ": " << iter_ms << " ms, " << tops << " TOPS" << std::endl;
    }

    auto stats = summarize(ms, ops);
    std::cout << label << " Torch avg: " << stats.avg_ms << " ms, " << stats.avg_tops << " TOPS" << std::endl;
    return stats;
}

void check_correctness(const char *label,
                       const std::vector<at::BFloat16> &hip_out,
                       const torch::Tensor &torch_out) {
    auto torch_cpu = torch_out.to(torch::kCPU).contiguous();
    const at::BFloat16 *torch_ptr = reinterpret_cast<const at::BFloat16 *>(torch_cpu.data_ptr());

    std::mt19937 rng(kSeed);
    std::uniform_int_distribution<int64_t> row_dist(0, kM - 1);
    std::uniform_int_distribution<int64_t> col_dist(0, kN - 1);

    const float rtol = 2e-2f;
    const float atol = 2e-2f;
    int mismatches = 0;

    for (int i = 0; i < kCheckSamples; ++i) {
        int64_t r = row_dist(rng);
        int64_t c = col_dist(rng);
        size_t idx = static_cast<size_t>(r) * static_cast<size_t>(kN) + static_cast<size_t>(c);
        float hip_val = static_cast<float>(hip_out[idx]);
        float torch_val = static_cast<float>(torch_ptr[idx]);
        float diff = std::fabs(hip_val - torch_val);
        float allowed = atol + rtol * std::fabs(torch_val);
        if (diff > allowed) {
            if (mismatches < 10) {
                std::cout << label << " mismatch [" << r << "," << c << "] hip=" << hip_val
                          << " torch=" << torch_val << " diff=" << diff
                          << " allowed=" << allowed << std::endl;
            }
            ++mismatches;
        }
    }

    if (mismatches == 0) {
        std::cout << label << " correctness: PASS (" << kCheckSamples << " samples)" << std::endl;
    } else {
        std::cout << label << " correctness: FAIL with " << mismatches << " mismatches ("
                  << kCheckSamples << " samples)" << std::endl;
    }
}

} // namespace

int main() {
    if (!torch::cuda::is_available()) {
        std::cerr << "Torch CUDA/ROCm backend is not available." << std::endl;
        return 1;
    }

    std::cout << "M = " << kM << "\n";
    std::cout << "K = " << kK << "\n";
    std::cout << "N = " << kN << "\n";
    std::cout << "iters = " << kIters << "\n";
    std::cout << "dtype = bf16\n";

    torch::manual_seed(kSeed);

    const size_t elemsA = static_cast<size_t>(kM) * static_cast<size_t>(kK);
    const size_t elemsB = static_cast<size_t>(kK) * static_cast<size_t>(kN);
    const size_t elemsC = static_cast<size_t>(kM) * static_cast<size_t>(kN);
    const size_t bytesA = elemsA * sizeof(at::BFloat16);
    const size_t bytesB = elemsB * sizeof(at::BFloat16);
    const size_t bytesC = elemsC * sizeof(at::BFloat16);

    auto cpu_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
    auto A_cpu = torch::rand({kM, kK}, cpu_opts).contiguous();
    auto B_cpu = torch::rand({kK, kN}, cpu_opts).contiguous();
    auto B_T_cpu = B_cpu.t().contiguous();

    auto gpu_opts = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA);
    auto A_gpu = A_cpu.to(gpu_opts);
    auto B_gpu = B_cpu.to(gpu_opts);
    auto B_T_gpu = B_T_cpu.to(gpu_opts);

    void *dA = nullptr;
    void *dB = nullptr;
    void *dB_T = nullptr;
    void *dC_add = nullptr;
    void *dD = nullptr;

    check_hip(hipMalloc(&dA, bytesA), "hipMalloc dA");
    check_hip(hipMalloc(&dB, bytesB), "hipMalloc dB");
    check_hip(hipMalloc(&dB_T, bytesB), "hipMalloc dB_T");
    check_hip(hipMalloc(&dC_add, bytesC), "hipMalloc dC_add");
    check_hip(hipMalloc(&dD, bytesC), "hipMalloc dD");

    check_hip(hipMemcpy(dA, A_cpu.data_ptr(), bytesA, hipMemcpyHostToDevice), "hipMemcpy dA");
    check_hip(hipMemcpy(dB, B_cpu.data_ptr(), bytesB, hipMemcpyHostToDevice), "hipMemcpy dB");
    check_hip(hipMemcpy(dB_T, B_T_cpu.data_ptr(), bytesB, hipMemcpyHostToDevice), "hipMemcpy dB_T");
    check_hip(hipMemset(dC_add, 0, bytesC), "hipMemset dC_add");
    check_hip(hipMemset(dD, 0, bytesC), "hipMemset dD");

    hipblasLtHandle_t blas_handle;
    check_hipblaslt(hipblasLtCreate(&blas_handle), "hipblasLtCreate");

    const uint64_t max_workspace_size = static_cast<uint64_t>(kWorkspaceMB) * 1024ULL * 1024ULL;

    // Standard weights: B is KxN row-major.
    HipMatmulConfig cfg_standard = make_config(
        blas_handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        kN, kK,  // A (column-major) uses B_row interpreted as N x K
        kK, kM,  // B (column-major) uses A_row interpreted as K x M
        kN, kM,  // C/D are N x M in column-major
        max_workspace_size);

    // Transposed weights: stored as N x K row-major, use TRANSA to interpret as N x K.
    HipMatmulConfig cfg_transposed = make_config(
        blas_handle,
        HIPBLAS_OP_T,
        HIPBLAS_OP_N,
        kK, kN,  // A (column-major) uses B_T_row interpreted as K x N, then transpose
        kK, kM,  // B (column-major) uses A_row interpreted as K x M
        kN, kM,  // C/D are N x M in column-major
        max_workspace_size);

    uint64_t workspace_size = 0;
    for (const auto &heur : cfg_standard.heuristics) {
        if (heur.workspaceSize > workspace_size) {
            workspace_size = heur.workspaceSize;
        }
    }
    for (const auto &heur : cfg_transposed.heuristics) {
        if (heur.workspaceSize > workspace_size) {
            workspace_size = heur.workspaceSize;
        }
    }

    void *workspace = nullptr;
    if (workspace_size > 0) {
        check_hip(hipMalloc(&workspace, workspace_size), "hipMalloc workspace");
    }

    hipStream_t stream;
    check_hip(hipStreamCreate(&stream), "hipStreamCreate");

    const double ops = 2.0 * static_cast<double>(kM) * static_cast<double>(kN) * static_cast<double>(kK);

    // Case 1: standard weights
    tune_best_algo("Standard", blas_handle, cfg_standard, dB, dA, dC_add, dD,
                   workspace, workspace_size, stream);
    run_hip_case("Standard", blas_handle, cfg_standard, dB, dA, dC_add, dD,
                 workspace, stream, ops);

    torch::Tensor torch_out_standard;
    run_torch_case("Standard", [&]() { return torch::matmul(A_gpu, B_gpu); },
                   torch_out_standard, ops);

    std::vector<at::BFloat16> hip_out_standard(elemsC);
    check_hip(hipMemcpy(hip_out_standard.data(), dD, bytesC, hipMemcpyDeviceToHost), "hipMemcpy dD->host standard");
    check_correctness("Standard", hip_out_standard, torch_out_standard);

    // Case 2: weights stored transposed, torch uses weights.t()
    tune_best_algo("Transposed weights", blas_handle, cfg_transposed, dB_T, dA, dC_add, dD,
                   workspace, workspace_size, stream);
    run_hip_case("Transposed weights", blas_handle, cfg_transposed, dB_T, dA, dC_add, dD,
                 workspace, stream, ops);

    torch::Tensor torch_out_transposed;
    run_torch_case("Transposed weights", [&]() { return torch::matmul(A_gpu, B_T_gpu.t()); },
                   torch_out_transposed, ops);

    std::vector<at::BFloat16> hip_out_transposed(elemsC);
    check_hip(hipMemcpy(hip_out_transposed.data(), dD, bytesC, hipMemcpyDeviceToHost), "hipMemcpy dD->host transposed");
    check_correctness("Transposed weights", hip_out_transposed, torch_out_transposed);

    // NOTE: Some ROCm builds report an invalid free during teardown after a successful run.
    // This is a short-lived benchmark binary, so we skip explicit teardown and exit immediately.
    std::cout << std::flush;
    std::_Exit(0);
}
