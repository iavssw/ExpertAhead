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
    
    virtual void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> expert_bias = c10::nullopt) = 0;
    virtual bool is_ready() = 0;
    virtual std::vector<int64_t> get_prediction() = 0;
    virtual std::vector<int64_t> try_get_prediction() = 0;
    virtual double get_prediction_time_ms() = 0;
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
    
    void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> expert_bias = c10::nullopt) override {
        response_->reset();
        
        request_->token_id = 0; // Deprecated
        request_->context_length = 0;
        
        request_->processed.store(false);
        request_->ready.store(true);
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
        c10::optional<torch::Tensor> expert_bias;
        int64_t job_id;
    };
    
    ThreadedTorchScriptPredictor(const std::string& model_path, int layer_idx = -1, torch::Device device = torch::kCPU) 
        : layer_idx_(layer_idx), device_(device), running_(true), model_loaded_(false) {
        
        // Load the model
        try {
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
    
    void predict_async(torch::Tensor embedding, c10::optional<torch::Tensor> expert_bias = c10::nullopt) override {
        PredictionJob job;
        job.embedding = embedding;
        job.expert_bias = expert_bias;
        job.job_id = next_job_id_++;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            job_queue_.push(job);
        }
        queue_cv_.notify_one();
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
        auto start = std::chrono::high_resolution_clock::now();
        
        prediction_ready_.store(false);
        
        if (!model_loaded_) {
            std::lock_guard<std::mutex> lock(result_mutex_);
            predicted_experts_.clear();
            prediction_ready_.store(true);
            result_cv_.notify_all();
            return;
        }
        
        try {
            // Prepare input
            // The embedding needs to be passed to the TorchScript model.
            // Predictor models accept [1, hidden_dim] tensors, likely in float32.
            torch::Tensor input = job.embedding.to(torch::kFloat32).to(device_);
            if (input.dim() == 1) {
                input = input.unsqueeze(0); // Ensure [1, hidden_dim]
            }
            
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            
            torch::NoGradGuard no_grad;
            auto output = model_.forward(inputs).toTensor();
            
            if (job.expert_bias.has_value()) {
                auto bias = job.expert_bias.value().to(torch::kFloat32).to(device_);
                output = output + bias;
            }
            
            // Extract prediction (get fully ranked list of experts)
            // Perform argsort on the dedicated stream/device before safely copying to CPU
            auto indices = output.argsort(-1, true).to(torch::kCPU, torch::kInt64);
            
            std::vector<int64_t> predicted_experts;
            if (indices.dim() == 2) {
                // Batch dimension exists
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
            
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                predicted_experts_ = predicted_experts;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[Worker Layer " << layer_idx_ << "] Prediction error: " 
                      << e.what() << std::endl;
            std::lock_guard<std::mutex> lock(result_mutex_);
            predicted_experts_.clear();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        prediction_time_ms_ = duration.count();
        
        prediction_ready_.store(true);
        result_cv_.notify_all();
    }
    
    int layer_idx_;
    torch::Device device_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    
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
    
    std::atomic<int64_t> next_job_id_{0};
};

class RandomExpertPredictor : public IExpertPredictor {
public:
    RandomExpertPredictor(int64_t num_experts) 
        : num_experts_(num_experts),
          gen_(std::random_device{}()) {}

    void predict_async(torch::Tensor embedding) override {
        // Generate a random ranking of all experts
        std::vector<int64_t> experts(num_experts_);
        std::iota(experts.begin(), experts.end(), 0);
        std::shuffle(experts.begin(), experts.end(), gen_);
        
        std::lock_guard<std::mutex> lock(mutex_);
        predicted_experts_ = experts;
        ready_ = true;
    }

    bool is_ready() override {
        return ready_.load();
    }

    std::vector<int64_t> get_prediction() override {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = false; // Reset for next prediction
        return predicted_experts_;
    }

    std::vector<int64_t> try_get_prediction() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ready_) return {};
        // ready_ = false;
        return predicted_experts_;
    }

    double get_prediction_time_ms() override {
        return 0.0; // Negligible time
    }

private:
    int64_t num_experts_;
    std::mt19937 gen_;
    
    std::mutex mutex_;
    std::vector<int64_t> predicted_experts_;
    std::atomic<bool> ready_{false};
};