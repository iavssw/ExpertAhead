#pragma once

#include <atomic>
#include <vector>
#include <cstdint>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <torch/script.h>

// ============================================================================
// Shared Memory Structure for Expert Prediction (for future NPU)
// ============================================================================

struct ExpertPredictionRequest {
    std::atomic<bool> ready{false};
    std::atomic<bool> processed{false};
    
    int64_t token_id;
    int64_t context_length;
    int64_t context_token_ids[32];
    
    void reset() {
        ready.store(false);
        processed.store(false);
        token_id = -1;
        context_length = 0;
        memset(context_token_ids, 0, sizeof(context_token_ids));
    }
};

struct ExpertPredictionResponse {
    std::atomic<bool> ready{false};
    int64_t expert_ids[32];
    double prediction_time_ms;
    
    void reset() {
        ready.store(false);
        memset(expert_ids, -1, sizeof(expert_ids));
        prediction_time_ms = 0.0;
    }
};

// ============================================================================
// Abstract Expert Predictor Interface
// ============================================================================

class IExpertPredictor {
public:
    virtual ~IExpertPredictor() = default;
    
    virtual void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) = 0;
    virtual std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) = 0;
    virtual bool is_ready() = 0;
    virtual std::vector<int64_t> get_prediction() = 0;
    virtual std::vector<int64_t> try_get_prediction() = 0;
    virtual double get_prediction_time_ms() = 0;
    virtual int get_history_length() { return 1; }
    /// When true, callers should load every expert ID returned (full lookahead union), not top-B.
    virtual bool prefetch_full_union() const { return false; }
};

// ============================================================================
// Shared Memory Predictor (for future NPU)
// ============================================================================

class SharedMemoryPredictor : public IExpertPredictor {
public:
    SharedMemoryPredictor(int layer_idx, const std::string& shm_name = "/expert_predictor_shm") 
        : layer_idx_(layer_idx), shm_name_(shm_name) {
        
        shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to create shared memory");
        }
        
        size_t shm_size = sizeof(ExpertPredictionRequest) + sizeof(ExpertPredictionResponse);
        if (ftruncate(shm_fd_, shm_size) == -1) {
            throw std::runtime_error("Failed to set shared memory size");
        }
        
        void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
        if (ptr == MAP_FAILED) {
            throw std::runtime_error("Failed to map shared memory");
        }
        
        request_ = static_cast<ExpertPredictionRequest*>(ptr);
        response_ = reinterpret_cast<ExpertPredictionResponse*>(
            static_cast<char*>(ptr) + sizeof(ExpertPredictionRequest));
        
        request_->reset();
        response_->reset();
        
        std::cout << "[SharedMemoryPredictor Layer " << layer_idx_ 
                  << "] Initialized: " << shm_name_ << std::endl;
    }
    
    ~SharedMemoryPredictor() {
        if (request_) {
            size_t shm_size = sizeof(ExpertPredictionRequest) + sizeof(ExpertPredictionResponse);
            munmap(request_, shm_size);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }
        shm_unlink(shm_name_.c_str());
    }
    
    void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        response_->reset();
        
        request_->token_id = 0; // Deprecated
        request_->context_length = 0;
        
        request_->processed.store(false);
        request_->ready.store(true);
    }
    
    std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        predict_async(embedding, prefill_dist, prev_expert_onehot, prev_layers_feat, source_decode_step);
        return get_prediction();
    }

    bool is_ready() override {
        return response_->ready.load();
    }
    
    std::vector<int64_t> get_prediction() override {
        while (!response_->ready.load()) {
            std::this_thread::yield();
        }
        
        if (layer_idx_ < 0 || layer_idx_ >= 32) {
            return {};
        }
        
        return {response_->expert_ids[layer_idx_]};
    }
    
    std::vector<int64_t> try_get_prediction() override {
        if (!response_->ready.load()) {
            return {};
        }
        
        if (layer_idx_ < 0 || layer_idx_ >= 32) {
            return {};
        }
        
        return {response_->expert_ids[layer_idx_]};
    }
    
    double get_prediction_time_ms() override {
        return response_->prediction_time_ms;
    }
    
private:
    int layer_idx_;
    std::string shm_name_;
    int shm_fd_ = -1;
    ExpertPredictionRequest* request_ = nullptr;
    ExpertPredictionResponse* response_ = nullptr;
};

// ============================================================================
// Single-Model TorchScript Predictor with Dedicated Thread
// ============================================================================


class ThreadedTorchScriptPredictor : public IExpertPredictor {
public:
    struct PredictionJob {
        torch::Tensor embedding;
        c10::optional<torch::Tensor> prefill_dist;
        c10::optional<torch::Tensor> prev_expert_onehot;
        c10::optional<torch::Tensor> prev_layers_feat;
        int64_t source_decode_step;
        int64_t job_id;
    };
    
    ThreadedTorchScriptPredictor(const std::string& model_path, int layer_idx = -1, torch::Device device = torch::kCPU) 
        : layer_idx_(layer_idx), device_(device), running_(true), model_loaded_(false) {
        
        // Load the model
        try {
            // Attempt to read best.json for history length.
            // The model path is like ".../layer_X/best_jit.pt", but the config is
            // ".../layer_X/best.json" — so look in the same directory.
            {
                std::string dir_path = model_path;
                size_t last_sep = dir_path.find_last_of("/\\");
                if (last_sep != std::string::npos) {
                    dir_path = dir_path.substr(0, last_sep);
                } else {
                    dir_path = ".";
                }
                // Try best.json first, then training_metrics.json
                std::vector<std::string> json_candidates = {
                    dir_path + "/best.json",
                    dir_path + "/training_metrics.json",
                };
                for (const auto& json_path : json_candidates) {
                    std::ifstream ifs(json_path);
                    if (!ifs.is_open()) continue;
                    std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
                    size_t pos = content.find("\"history\"");
                    if (pos != std::string::npos) {
                        pos = content.find(":", pos);
                        if (pos != std::string::npos) {
                            pos++;
                            while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
                            size_t end_pos = pos;
                            while (end_pos < content.length() && std::isdigit(content[end_pos])) end_pos++;
                            if (end_pos > pos) {
                                history_length_ = std::stoi(content.substr(pos, end_pos - pos));
                                std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                                          << "] Detected history_length=" << history_length_ 
                                          << " from " << json_path << std::endl;
                                break;
                            }
                        }
                    }
                }
            }

            // Attempt to load on the specified device
            if (device_.type() != torch::kCPU) {
                 std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                          << "] Loading model on device: " << device_ << " (NPU/GPU)" << std::endl;
            } else {
                 std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                          << "] Loading model on CPU" << std::endl;
            }

            model_ = torch::jit::load(model_path, device_);
            model_.eval();
            model_loaded_ = true;

            // As of the new PredictorJITWrapper, all models take exactly 3 inputs + self.
            // We bypass the flaky TorchScript schema reflection which can throw on traced modules.
            model_accepts_prefill_dist_ = true;
            model_accepts_prev_expert_ = true;
            model_accepts_prev_layers_ = true;
            std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_
                      << "] Assuming prefill_dist, prev_expert_onehot, and prev_layers_feat support (v3 API)." << std::endl;
            std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_
                      << "] prefill_dist support: " << (model_accepts_prefill_dist_ ? "yes" : "no") 
                      << ", prev_expert_onehot support: " << (model_accepts_prev_expert_ ? "yes" : "no") 
                      << ", prev_layers_feat support: " << (model_accepts_prev_layers_ ? "yes" : "no") << std::endl;
            std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                      << "] Loaded model: " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                      << "] Failed to load model on " << device_ << ": " << e.what() << std::endl;
            // Fallback to CPU? Maybe not if user explicitly requested NPU.
            model_loaded_ = false;
        }
        
        // Start worker thread
        worker_thread_ = std::thread(&ThreadedTorchScriptPredictor::worker_loop, this);
        
        std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                  << "] Worker thread started" << std::endl;
    }
    
    ~ThreadedTorchScriptPredictor() {
        // Signal thread to stop
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            running_ = false;
        }
        queue_cv_.notify_one();
        
        // Wait for thread to finish
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        
        std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
                  << "] Shutdown complete" << std::endl;
    }
    
    void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        PredictionJob job;
        job.embedding = embedding;
        job.prefill_dist = prefill_dist;
        job.prev_expert_onehot = prev_expert_onehot;
        job.prev_layers_feat = prev_layers_feat;
        job.source_decode_step = source_decode_step;
        job.job_id = next_job_id_++;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            job_queue_.push(job);
        }
        queue_cv_.notify_one();
    }

    std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        return internal_predict(embedding, prefill_dist, prev_expert_onehot, prev_layers_feat);
    }
    
    bool is_ready() override {
        return prediction_ready_.load();
    }
    
    std::vector<int64_t> get_prediction() override {
        // Wait for prediction to be ready
        std::unique_lock<std::mutex> lock(result_mutex_);
        result_cv_.wait(lock, [this] { return prediction_ready_.load(); });
        // if (!predicted_experts_.empty()) {
        //     std::cout << "[ThreadedTorchScriptPredictor Layer " << layer_idx_ 
        //               << "] Prediction ready, top value: " << predicted_experts_[0] << std::endl;
        // }
        return predicted_experts_;
    }
    
    std::vector<int64_t> try_get_prediction() override {
        if (!prediction_ready_.load()) {
            return {};
        }
        
        std::lock_guard<std::mutex> lock(result_mutex_);
        return predicted_experts_;
    }
    
    double get_prediction_time_ms() override {
        return prediction_time_ms_;
    }

    int get_history_length() override {
        return history_length_;
    }
    
private:
    void worker_loop() {
        while (running_) {
            PredictionJob job;
            
            // Wait for a job
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { 
                    return !job_queue_.empty() || !running_; 
                });
                
                if (!running_ && job_queue_.empty()) {
                    break;
                }
                
                if (!job_queue_.empty()) {
                    job = job_queue_.front();
                    job_queue_.pop();
                }
            }
            
            // Process the job
            process_prediction(job);
        }
    }
    
    void process_prediction(const PredictionJob& job) {
        auto result = internal_predict(job.embedding, job.prefill_dist, job.prev_expert_onehot, job.prev_layers_feat);
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            predicted_experts_ = result;
        }
        prediction_ready_.store(true);
        result_cv_.notify_all();
    }

    std::vector<int64_t> internal_predict(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist, c10::optional<torch::Tensor> prev_expert_onehot, c10::optional<torch::Tensor> prev_layers_feat) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!model_loaded_) {
            return {};
        }
        
        try {
            torch::Tensor input = embedding.to(torch::kFloat32).to(device_);
            if (input.dim() == 1) {
                input = input.unsqueeze(0); 
            }
            
            if (history_length_ > 1) {
                emb_history_.push_back(input);
                while (emb_history_.size() > history_length_) {
                    emb_history_.pop_front();
                }
                
                std::vector<torch::Tensor> to_concat;
                int pad_count = history_length_ - emb_history_.size();
                for (int i = 0; i < pad_count; ++i) {
                    to_concat.push_back(emb_history_.front());
                }
                for (const auto& t : emb_history_) {
                    to_concat.push_back(t);
                }
                input = torch::cat(to_concat, 1);
            }
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            if (model_accepts_prefill_dist_) {
                if (prefill_dist.has_value()) {
                    torch::Tensor pdist = prefill_dist.value().to(torch::kFloat32).to(device_);
                    if (pdist.dim() == 1) pdist = pdist.unsqueeze(0);
                    inputs.push_back(pdist);
                }
            }
            if (model_accepts_prev_expert_) {
                if (prev_expert_onehot.has_value()) {
                    torch::Tensor prev_exp = prev_expert_onehot.value().to(torch::kFloat32).to(device_);
                    if (prev_exp.dim() == 1) prev_exp = prev_exp.unsqueeze(0);
                    // Do NOT add history buffering here. The model expects [batch, experts], not [batch, history, experts]
                    inputs.push_back(prev_exp);
                }
            }
            
            torch::NoGradGuard no_grad;
            torch::jit::IValue output_ivalue;

            if (model_accepts_prev_layers_ && prev_layers_feat.has_value()) {
                torch::Tensor pl_feat = prev_layers_feat.value().to(torch::kFloat32).to(device_);
                if (pl_feat.dim() == 1) pl_feat = pl_feat.unsqueeze(0);
                std::vector<torch::jit::IValue> inputs_4 = inputs;
                inputs_4.push_back(pl_feat);
                
                try {
                    output_ivalue = model_.forward(inputs_4);
                } catch (const c10::Error& e) {
                    // Fallback to 3 args if the model doesn't support the 4th feature tensor
                    model_accepts_prev_layers_ = false;
                    output_ivalue = model_.forward(inputs);
                }
            } else {
                output_ivalue = model_.forward(inputs);
            }
           
            auto output = output_ivalue.toTensor();
            
            //std::cout << "predictor model output: " << output << std::endl;
            // Extract prediction
            auto indices = output.argsort(-1, true).to(torch::kCPU, torch::kInt64);
            
            std::vector<int64_t> predicted_experts;
            if (indices.dim() == 2) {
                auto acc = indices.accessor<int64_t, 2>();
                for (int i = 0; i < indices.size(1); ++i) {
                    predicted_experts.push_back(acc[0][i]);
                }
            } else if (indices.dim() == 1) {
                auto acc = indices.accessor<int64_t, 1>();
                for (int i = 0; i < indices.size(0); ++i) {
                    predicted_experts.push_back(acc[i]);
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            prediction_time_ms_ = duration.count();
            
            return predicted_experts;
            
        } catch (const std::exception& e) {
            std::cerr << "[Predictor Layer " << layer_idx_ << "] Sync prediction error: " 
                      << e.what() << std::endl;
            return {};
        }
    }
    
    int layer_idx_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    bool model_accepts_prefill_dist_ = false;  // Detected from the model's forward schema at load time
    bool model_accepts_prev_expert_ = false;   // Detected from the model's forward schema at load time
    bool model_accepts_prev_layers_ = false;   // Fallback detected dynamically on first call
    
    std::atomic<bool> running_;
    std::thread worker_thread_;
    
    std::queue<PredictionJob> job_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    std::atomic<bool> prediction_ready_{false};
    std::vector<int64_t> predicted_experts_;
    double prediction_time_ms_ = 0.0;
    std::mutex result_mutex_;
    std::condition_variable result_cv_;
    
    double delta_avg_ = 0.0;
    int history_length_ = 1;
    
    std::deque<torch::Tensor> emb_history_;
    std::deque<torch::Tensor> prev_expert_history_;
    
    std::atomic<int64_t> next_job_id_{0};
};

// ============================================================================
// Oracle Trace Predictor
// ============================================================================

class OracleTracePredictor : public IExpertPredictor {
public:
    OracleTracePredictor(const std::string& trace_path, int layer_idx, int lookahead, int budget,
                         bool full_union = false)
        : layer_idx_(layer_idx), lookahead_(lookahead), budget_(budget), full_union_(full_union) {
        
        std::ifstream ifs(trace_path);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open oracle trace file: " << trace_path << std::endl;
            return;
        }

        std::string line;
        bool in_trace = false;
        while (std::getline(ifs, line)) {
            if (line.find("EXPERT TRACE") != std::string::npos) {
                in_trace = true;
                std::getline(ifs, line); // Skip the '===' line
                continue;
            }
            if (!in_trace) continue;

            if (line.find("Token") == 0) {
                size_t colon = line.find(':');
                if (colon == std::string::npos) continue;

                std::string experts_str = line.substr(colon + 1);
                
                int current_layer = 0;
                size_t pos = 0;
                while ((pos = experts_str.find('[', pos)) != std::string::npos) {
                    size_t end_pos = experts_str.find(']', pos);
                    if (current_layer == layer_idx_) {
                        std::string layer_experts = experts_str.substr(pos + 1, end_pos - pos - 1);
                        std::vector<int64_t> experts;
                        std::stringstream ss(layer_experts);
                        std::string token;
                        while (std::getline(ss, token, ',')) {
                            experts.push_back(std::stoll(token));
                        }
                        token_experts_.push_back(experts);
                        break;
                    }
                    current_layer++;
                    pos = end_pos;
                }
            }
        }
        std::cout << "[OracleTracePredictor Layer " << layer_idx_ << "] Loaded " << token_experts_.size() << " tokens from " << trace_path << std::endl;
    }

    void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        // Not used
    }

    std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        (void)embedding;
        (void)prefill_dist;
        (void)prev_expert_onehot;
        (void)prev_layers_feat;

        int token_idx = 0;
        if (source_decode_step >= 0) {
            token_idx = static_cast<int>(source_decode_step);
            current_token_idx_ = token_idx;
        }

        const int horizon = std::max(lookahead_, 1);
        // Prefill-end warmup: decode step 0 is next; during decode at step t prefetch t+1..t+horizon.
        const int start_i = (source_decode_step < 0) ? 0 : 1;

        std::unordered_map<int64_t, int> expert_counts;
        for (int i = start_i; i <= horizon; ++i) {
            const int future_idx = token_idx + i;
            if (future_idx < 0 || future_idx >= static_cast<int>(token_experts_.size())) {
                continue;
            }
            for (int64_t expert : token_experts_[static_cast<size_t>(future_idx)]) {
                expert_counts[expert]++;
            }
        }

        std::vector<std::pair<int64_t, int>> sorted_experts(expert_counts.begin(), expert_counts.end());
        std::sort(sorted_experts.begin(), sorted_experts.end(),
                  [](const std::pair<int64_t, int>& a, const std::pair<int64_t, int>& b) {
                      if (a.second != b.second) {
                          return a.second > b.second;
                      }
                      return a.first < b.first;
                  });

        std::vector<int64_t> predicted_experts;
        predicted_experts.reserve(sorted_experts.size());
        for (const auto& pair : sorted_experts) {
            predicted_experts.push_back(pair.first);
        }

        prediction_time_ms_ = 0.0;
        return predicted_experts;
    }

    bool prefetch_full_union() const override { return full_union_; }

    bool is_ready() override { return true; }
    std::vector<int64_t> get_prediction() override { return {}; }
    std::vector<int64_t> try_get_prediction() override { return {}; }
    double get_prediction_time_ms() override { return prediction_time_ms_; }

private:
    int layer_idx_;
    int lookahead_;
    int budget_;
    bool full_union_;
    std::vector<std::vector<int64_t>> token_experts_;
    std::atomic<int> current_token_idx_{0};
    double prediction_time_ms_ = 0.0;
};
