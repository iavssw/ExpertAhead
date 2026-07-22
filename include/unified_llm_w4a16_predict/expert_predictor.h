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
#include <unordered_set>
#include <functional>
#include <chrono>
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
    
    virtual std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) = 0;
    virtual double get_prediction_time_ms() = 0;
    virtual int get_history_length() { return 1; }
    /// When true, callers should load every expert ID returned (full lookahead union), not top-B.
    virtual bool prefetch_full_union() const { return false; }
    /// When true, caller should bypass budget limitation because predictor has already filtered by threshold.
    virtual bool has_prefetch_threshold() const { return false; }
    /// Paper Section 3.2 gating-heuristic prefetch (next-layer router on previous hidden state).
    virtual bool is_gating_heuristic() const { return false; }
};

// ============================================================================
// Gating-Heuristic Predictor (arXiv:2312.17238 Section 3.2)
// ============================================================================

class GatingHeuristicPredictor : public IExpertPredictor {
public:
    using RouterFn = std::function<torch::Tensor(torch::Tensor)>;

    GatingHeuristicPredictor(RouterFn router_fn, int layer_idx, int prefetch_count,
                             bool use_softmax_before_topk, int num_experts_per_tok,
                             float score_percentile = 0.0f)
        : router_fn_(std::move(router_fn)),
          layer_idx_(layer_idx),
          prefetch_count_(prefetch_count),
          use_softmax_before_topk_(use_softmax_before_topk),
          num_experts_per_tok_(num_experts_per_tok),
          score_percentile_(score_percentile) {
        std::cout << "[GatingHeuristicPredictor Layer " << layer_idx_ << "] ";
        if (score_percentile_ > 0.0f) {
            std::cout << "score_percentile=" << score_percentile_
                      << " (Zhu et al. score-based prefetch)";
        } else {
            std::cout << "prefetch_count=" << prefetch_count_
                      << " softmax_before_topk=" << (use_softmax_before_topk_ ? "yes" : "no");
        }
        std::cout << std::endl;
    }

    std::vector<int64_t> predict_sync(torch::Tensor hidden_state,
                                      c10::optional<torch::Tensor> prefill_dist = c10::nullopt,
                                      c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt,
                                      c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt,
                                      int64_t source_decode_step = -1) override {
        (void)prefill_dist;
        (void)prev_expert_onehot;
        (void)prev_layers_feat;
        (void)source_decode_step;

        auto start = std::chrono::high_resolution_clock::now();
        if (!router_fn_) {
            return {};
        }

        try {
            torch::Tensor h = hidden_state;
            if (h.dim() == 1) {
                h = h.unsqueeze(0);
            } else if (h.dim() == 3) {
                // Decode path passes [batch, seq, hidden]; router expects [tokens, hidden].
                h = h.reshape({-1, h.size(-1)});
            }
            torch::Tensor router_out = router_fn_(h);
            if (router_out.dim() == 1) {
                router_out = router_out.unsqueeze(0);
            }

            torch::Tensor scores = router_out;
            if (use_softmax_before_topk_) {
                scores = torch::softmax(scores.to(torch::kFloat32), -1).to(router_out.dtype());
            }

            std::vector<int64_t> predicted_experts;
            if (score_percentile_ > 0.0f) {
                torch::Tensor probs = scores;
                if (!use_softmax_before_topk_) {
                    probs = torch::softmax(scores.to(torch::kFloat32), -1);
                }
                torch::Tensor row = probs.flatten().to(torch::kFloat32);
                const float qval = torch::quantile(row, score_percentile_).item<float>();
                torch::Tensor mask = row >= qval;
                torch::Tensor hit_idx = mask.nonzero().flatten();
                if (hit_idx.numel() == 0) {
                    auto top1 = row.argmax();
                    hit_idx = top1.reshape({1});
                }
                torch::Tensor hit_scores = row.index_select(0, hit_idx);
                auto order = hit_scores.argsort(/*dim=*/0, /*descending=*/true);
                hit_idx = hit_idx.index_select(0, order).to(torch::kCPU, torch::kInt64);
                predicted_experts.reserve(static_cast<size_t>(hit_idx.numel()));
                auto acc = hit_idx.accessor<int64_t, 1>();
                for (int i = 0; i < hit_idx.size(0); ++i) {
                    predicted_experts.push_back(acc[i]);
                }
                if (static_cast<int>(predicted_experts.size()) > num_experts_per_tok_) {
                    predicted_experts.resize(static_cast<size_t>(num_experts_per_tok_));
                }
            } else {
                const int topk = std::max(
                    1, std::min({prefetch_count_, num_experts_per_tok_,
                                 static_cast<int>(scores.size(-1))}));
                auto topk_result = scores.topk(topk, -1);
                auto topk_idx = std::get<1>(topk_result).to(torch::kCPU, torch::kInt64);
                predicted_experts.reserve(topk);
                if (topk_idx.dim() == 2) {
                    auto acc = topk_idx.accessor<int64_t, 2>();
                    for (int i = 0; i < topk_idx.size(1); ++i) {
                        predicted_experts.push_back(acc[0][i]);
                    }
                } else if (topk_idx.dim() == 1) {
                    auto acc = topk_idx.accessor<int64_t, 1>();
                    for (int i = 0; i < topk_idx.size(0); ++i) {
                        predicted_experts.push_back(acc[i]);
                    }
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            prediction_time_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
            return predicted_experts;
        } catch (const std::exception& e) {
            std::cerr << "[GatingHeuristicPredictor Layer " << layer_idx_
                      << "] prediction error: " << e.what() << std::endl;
            return {};
        }
    }

    bool is_gating_heuristic() const override { return true; }

    bool has_prefetch_threshold() const override { return score_percentile_ > 0.0f; }

    double get_prediction_time_ms() override { return prediction_time_ms_; }

private:
    RouterFn router_fn_;
    int layer_idx_;
    int prefetch_count_;
    bool use_softmax_before_topk_;
    int num_experts_per_tok_;
    float score_percentile_;
    double prediction_time_ms_ = 0.0;
};

// ============================================================================
// Shared Memory Predictor (for future NPU)
// ============================================================================

class SharedMemoryPredictor : public IExpertPredictor {
public:
    SharedMemoryPredictor(int layer_idx, const std::string& shm_name = "/expert_predictor_shm") 
        : layer_idx_(layer_idx), shm_name_(shm_name) {
        
        bool created = false;
        shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
        if (shm_fd_ == -1) {
            shm_fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
            created = true;
        }
        if (shm_fd_ == -1) {
            throw std::runtime_error("Failed to create shared memory");
        }
        
        size_t shm_size = sizeof(ExpertPredictionRequest) + sizeof(ExpertPredictionResponse);
        if (created) {
            if (ftruncate(shm_fd_, shm_size) == -1) {
                throw std::runtime_error("Failed to set shared memory size");
            }
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
    
    std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        response_->reset();
        request_->token_id = 0; // Deprecated
        request_->context_length = 0;
        request_->processed.store(false);
        request_->ready.store(true);
        
        int spin_count = 0;
        while (!response_->ready.load(std::memory_order_acquire)) {
            if (spin_count < 1000) {
                // hot spin
            } else if (spin_count < 10000) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            spin_count++;
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


class TorchScriptPredictor : public IExpertPredictor {
public:
    TorchScriptPredictor(const std::string& model_path, int layer_idx = -1, torch::Device device = torch::kCPU, float prefetch_threshold = 0.0f) 
        : layer_idx_(layer_idx), device_(device), prefetch_threshold_(prefetch_threshold), model_loaded_(false) {
        
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
                 std::cout << "[TorchScriptPredictor Layer " << layer_idx_ 
                          << "] Loading model on device: " << device_ << " (NPU/GPU)" << std::endl;
            } else {
                 std::cout << "[TorchScriptPredictor Layer " << layer_idx_ 
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
            std::cout << "[TorchScriptPredictor Layer " << layer_idx_
                      << "] Assuming prefill_dist, prev_expert_onehot, and prev_layers_feat support (v3 API)." << std::endl;
            std::cout << "[TorchScriptPredictor Layer " << layer_idx_
                      << "] prefill_dist support: " << (model_accepts_prefill_dist_ ? "yes" : "no") 
                      << ", prev_expert_onehot support: " << (model_accepts_prev_expert_ ? "yes" : "no") 
                      << ", prev_layers_feat support: " << (model_accepts_prev_layers_ ? "yes" : "no") << std::endl;
            std::cout << "[TorchScriptPredictor Layer " << layer_idx_ 
                      << "] Loaded model: " << model_path << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "[TorchScriptPredictor Layer " << layer_idx_ 
                      << "] Failed to load model on " << device_ << ": " << e.what() << std::endl;
            // Fallback to CPU? Maybe not if user explicitly requested NPU.
            model_loaded_ = false;
        }
    }
    
    ~TorchScriptPredictor() {
        std::cout << "[TorchScriptPredictor Layer " << layer_idx_ 
                  << "] Shutdown complete" << std::endl;
    }

    std::vector<int64_t> predict_sync(torch::Tensor embedding, c10::optional<torch::Tensor> prefill_dist = c10::nullopt, c10::optional<torch::Tensor> prev_expert_onehot = c10::nullopt, c10::optional<torch::Tensor> prev_layers_feat = c10::nullopt, int64_t source_decode_step = -1) override {
        (void)source_decode_step;
        
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
            
            if (prefetch_threshold_ > 0.0f) {
                output = torch::sigmoid(output);
            }

            // Extract prediction
            auto indices = output.argsort(-1, true).to(torch::kCPU, torch::kInt64);
            auto values = output.to(torch::kCPU, torch::kFloat32);
            
            std::vector<int64_t> predicted_experts;
            if (indices.dim() == 2) {
                auto idx_acc = indices.accessor<int64_t, 2>();
                auto val_acc = values.accessor<float, 2>();
                for (int i = 0; i < indices.size(1); ++i) {
                    int64_t exp_id = idx_acc[0][i];
                    if (prefetch_threshold_ <= 0.0f || val_acc[0][exp_id] >= prefetch_threshold_) {
                        predicted_experts.push_back(exp_id);
                    }
                }
            } else if (indices.dim() == 1) {
                auto idx_acc = indices.accessor<int64_t, 1>();
                auto val_acc = values.accessor<float, 1>();
                for (int i = 0; i < indices.size(0); ++i) {
                    int64_t exp_id = idx_acc[i];
                    if (prefetch_threshold_ <= 0.0f || val_acc[exp_id] >= prefetch_threshold_) {
                        predicted_experts.push_back(exp_id);
                    }
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
    
    bool has_prefetch_threshold() const override {
        return prefetch_threshold_ > 0.0f;
    }

    double get_prediction_time_ms() override {
        return prediction_time_ms_;
    }
    
private:
    int layer_idx_;
    torch::Device device_;
    float prefetch_threshold_;
    torch::jit::script::Module model_;
    bool model_loaded_;
    bool model_accepts_prefill_dist_ = false;  // Detected from the model's forward schema at load time
    bool model_accepts_prev_expert_ = false;   // Detected from the model's forward schema at load time
    bool model_accepts_prev_layers_ = false;   // Fallback detected dynamically on first call
    
    double prediction_time_ms_ = 0.0;
    int history_length_ = 1;
    
    std::deque<torch::Tensor> emb_history_;
    std::deque<torch::Tensor> prev_expert_history_;
};

// ============================================================================
// Oracle Trace Predictor
// ============================================================================

class OracleTracePredictor : public IExpertPredictor {
public:
    OracleTracePredictor(const std::string& trace_path, int layer_idx, int lookahead, int budget,
                         bool full_union = false, float routing_agreement = 1.0f,
                         int64_t num_experts = 128, uint64_t noise_seed = 42)
        : layer_idx_(layer_idx), lookahead_(lookahead), budget_(budget), full_union_(full_union),
          routing_agreement_(routing_agreement), num_experts_(num_experts), noise_seed_(noise_seed) {
        
        std::ifstream ifs(trace_path);
        if (!ifs.is_open()) {
            std::cerr << "Failed to open oracle trace file: " << trace_path << std::endl;
            std::exit(EXIT_FAILURE);
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
        std::cout << "[OracleTracePredictor Layer " << layer_idx_ << "] Loaded " << token_experts_.size()
                  << " tokens from " << trace_path;
        if (routing_agreement_ < 1.0f - 1e-6f) {
            std::cout << " (noisy oracle: routing_agreement=" << routing_agreement_
                      << ", noise_seed=" << noise_seed_ << ")";
        }
        std::cout << std::endl;
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
        } else {
            token_idx = current_token_idx_++;
        }

        const int horizon = std::max(lookahead_, 1);
        // Prefill-end warmup: decode step 0 is next; during decode at step t prefetch t+1..t+horizon.
        const int start_i = (source_decode_step < 0) ? 0 : 1;

        std::unordered_map<int64_t, int> expert_counts;
        // Track earliest token index at which each expert is needed (for tie-breaking)
        std::unordered_map<int64_t, int> expert_earliest_need;

        const bool use_noisy_oracle = routing_agreement_ < 1.0f - 1e-6f;
        std::mt19937_64 rng;
        std::uniform_int_distribution<int64_t> expert_dist;
        if (use_noisy_oracle) {
            rng.seed(noise_seed_ ^ (static_cast<uint64_t>(layer_idx_) << 32) ^
                     (static_cast<uint64_t>(token_idx) * 0x9E3779B97F4A7C15ULL));
            expert_dist = std::uniform_int_distribution<int64_t>(0, std::max<int64_t>(num_experts_ - 1, 0));
        }

        for (int i = start_i; i <= horizon; ++i) {
            const int future_idx = token_idx + i;
            if (future_idx < 0 || future_idx >= static_cast<int>(token_experts_.size())) {
                continue;
            }
            const std::vector<int64_t>& true_experts = token_experts_[static_cast<size_t>(future_idx)];
            if (true_experts.empty()) {
                continue;
            }

            std::unordered_set<int64_t> predicted_for_token;
            if (!use_noisy_oracle) {
                for (int64_t expert : true_experts) {
                    predicted_for_token.insert(expert);
                }
            } else {
                const int B = static_cast<int>(true_experts.size());
                int n_keep = std::max(1, static_cast<int>(std::lround(routing_agreement_ * B)));
                n_keep = std::min(B, n_keep);

                std::vector<int> pick_idx(B);
                std::iota(pick_idx.begin(), pick_idx.end(), 0);
                std::shuffle(pick_idx.begin(), pick_idx.end(), rng);

                std::unordered_set<int64_t> true_set(true_experts.begin(), true_experts.end());
                for (int k = 0; k < n_keep; ++k) {
                    predicted_for_token.insert(true_experts[static_cast<size_t>(pick_idx[k])]);
                }

                int attempts = 0;
                while (static_cast<int>(predicted_for_token.size()) < B && attempts < num_experts_ * 4) {
                    ++attempts;
                    const int64_t cand = expert_dist(rng);
                    if (true_set.count(cand) || predicted_for_token.count(cand)) {
                        continue;
                    }
                    predicted_for_token.insert(cand);
                }
            }

            for (int64_t expert : predicted_for_token) {
                expert_counts[expert]++;
                auto it = expert_earliest_need.find(expert);
                if (it == expert_earliest_need.end()) {
                    expert_earliest_need[expert] = future_idx;
                }
            }
        }

        std::vector<std::pair<int64_t, int>> sorted_experts(expert_counts.begin(), expert_counts.end());
        std::sort(sorted_experts.begin(), sorted_experts.end(),
                  [&](const std::pair<int64_t, int>& a, const std::pair<int64_t, int>& b) {
                      if (a.second != b.second) {
                          return a.second > b.second; // Higher frequency first
                      }
                      // Tie-break: expert needed sooner (smaller future_idx) comes first
                      int ea = expert_earliest_need.count(a.first) ? expert_earliest_need.at(a.first) : INT_MAX;
                      int eb = expert_earliest_need.count(b.first) ? expert_earliest_need.at(b.first) : INT_MAX;
                      if (ea != eb) {
                          return ea < eb; // Sooner-needed expert first
                      }
                      return a.first < b.first; // Final tie-break by ID (stable)
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

    bool has_prefetch_threshold() const override {
        return false;
    }

    double get_prediction_time_ms() override { return prediction_time_ms_; }

private:
    int layer_idx_;
    int lookahead_;
    int budget_;
    bool full_union_;
    float routing_agreement_;
    int64_t num_experts_;
    uint64_t noise_seed_;
    std::vector<std::vector<int64_t>> token_experts_;
    std::atomic<int> current_token_idx_{0};
    double prediction_time_ms_ = 0.0;
};
