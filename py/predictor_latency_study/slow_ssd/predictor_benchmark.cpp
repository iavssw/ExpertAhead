#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <string>
#include <fstream>
#include <iomanip>

// Include the header for TorchScriptPredictor.
// Adjust the relative include path based on compiler -I flags.
#include "unified_llm_w4a16_predict/expert_predictor.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_best_jit.pt> [num_iterations]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int num_iterations = (argc >= 3) ? std::stoi(argv[2]) : 1000;
    int warmup_iterations = 10;

    std::cout << "Loading model: " << model_path << std::endl;

    // Instantiate on CPU (can be modified to CUDA/NPU if needed)
    TorchScriptPredictor predictor(model_path, /*layer_idx=*/0, torch::kCPU, /*prefetch_threshold=*/0.0f);

    // Dummy shapes as confirmed by the user
    int64_t hidden_size = 2048;
    int64_t num_experts = 128;

    auto embedding = torch::randn({1, hidden_size}, torch::kFloat32);
    auto prefill_dist = torch::randn({1, num_experts}, torch::kFloat32);
    auto prev_expert_onehot = torch::zeros({1, num_experts}, torch::kFloat32);
    // Let's set some random expert as active in the onehot
    prev_expert_onehot[0][0] = 1.0f;

    std::cout << "Warming up for " << warmup_iterations << " iterations..." << std::endl;
    for (int i = 0; i < warmup_iterations; ++i) {
        predictor.predict_sync(embedding, prefill_dist, prev_expert_onehot, c10::nullopt, -1);
    }

    std::cout << "Running benchmark for " << num_iterations << " iterations..." << std::endl;
    std::vector<double> latencies;
    latencies.reserve(num_iterations);

    for (int i = 0; i < num_iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        predictor.predict_sync(embedding, prefill_dist, prev_expert_onehot, c10::nullopt, -1);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        latencies.push_back(duration.count());
    }

    // Compute stats
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / latencies.size();
    
    double min_lat = *std::min_element(latencies.begin(), latencies.end());
    double max_lat = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << "Avg Latency: " << avg << " ms" << std::endl;
    std::cout << "Min Latency: " << min_lat << " ms" << std::endl;
    std::cout << "Max Latency: " << max_lat << " ms" << std::endl;
    std::cout << "Total Latency for 60 layers: " << avg * 60.0 << " ms" << std::endl;

    // Write latencies to CSV for python plotting
    std::ofstream out("predictor_latency.csv");
    out << "iteration,latency_ms\n";
    for (size_t i = 0; i < latencies.size(); ++i) {
        out << i << "," << latencies[i] << "\n";
    }
    out.close();

    std::cout << "Saved raw latencies to predictor_latency.csv" << std::endl;

    return 0;
}
