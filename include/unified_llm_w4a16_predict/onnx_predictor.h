#pragma once

// ============================================================================
// OnnxPredictor — IExpertPredictor backed by ONNX Runtime
//
// Enabled only when compiled with -DHETEROPREDICT_ONNXRUNTIME=1.
// Without that flag this header is a no-op; nothing else changes.
//
// Execution providers (ep_name):
//   "cpu"     — CPUExecutionProvider   (always available)
//   "cuda"    — CUDA EP                (requires onnxruntime-gpu)
//   "vitisai" — VitisAIExecutionProvider (requires Ryzen AI stack)
//
// Expected ONNX model I/O (confirmed from export_onnx.py):
//   Inputs : "embedding"   [1, hidden_size]    float32
//            "prefill_dist" [1, num_experts]   float32   (may be absent)
//            "prev_expert"  [1, num_experts]   float32   (may be absent)
//            "prev_layers"  [1, layer*experts] float32   (optional, some models)
//   Output : "prediction"  [1, num_experts]    float32
// ============================================================================

#ifdef HETEROPREDICT_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "unified_llm_w4a16_predict/expert_predictor.h"

class OnnxPredictor : public IExpertPredictor {
public:
    // -----------------------------------------------------------------------
    // Constructor
    //   model_path       : path to best_model.onnx for this layer
    //   layer_idx        : MoE layer index (logging only)
    //   ep_name          : "cpu", "cuda", or "vitisai"
    //   vaip_config_path : path to vaip_config.json (required for vitisai EP;
    //                      empty string → skip provider option)
    //   prefetch_threshold : if > 0, apply sigmoid + threshold filtering
    // -----------------------------------------------------------------------
    OnnxPredictor(const std::string& model_path,
                  int                layer_idx,
                  const std::string& ep_name          = "cpu",
                  const std::string& vaip_config_path = "",
                  float              prefetch_threshold = 0.0f)
        : layer_idx_(layer_idx),
          ep_name_(ep_name),
          prefetch_threshold_(prefetch_threshold),
          env_(ORT_LOGGING_LEVEL_WARNING, "OnnxPredictor"),
          model_loaded_(false)
    {
        try {
            Ort::SessionOptions session_opts;
            session_opts.SetIntraOpNumThreads(1);
            session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // ── Execution provider selection ────────────────────────────────
            if (ep_name_ == "vitisai") {
                const std::unordered_map<std::string, std::string> vaip_opts = {
                    {"config_file", vaip_config_path.empty() ? "vaip_config.json" : vaip_config_path}
                };
#if ORT_API_VERSION >= 17
                session_opts.AppendExecutionProvider("VitisAI", vaip_opts);
#else
                // Older ORT API — pass as C string pairs
                std::vector<const char*> keys   = {"config_file"};
                const std::string&       cv     = vaip_opts.at("config_file");
                std::vector<const char*> values = {cv.c_str()};
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_VitisAI(
                    session_opts, keys.data(), values.data(), keys.size()));
#endif
                std::cout << "[OnnxPredictor Layer " << layer_idx_
                          << "] Using VitisAI EP"
                          << (vaip_config_path.empty() ? "" : " (config: " + vaip_config_path + ")")
                          << std::endl;
            } else if (ep_name_ == "cuda") {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0;
                session_opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[OnnxPredictor Layer " << layer_idx_
                          << "] Using CUDA EP" << std::endl;
            } else {
                // CPU is always available and always appended as fallback
                std::cout << "[OnnxPredictor Layer " << layer_idx_
                          << "] Using CPU EP" << std::endl;
            }

            // ── Load model ──────────────────────────────────────────────────
            session_ = std::make_unique<Ort::Session>(
                env_, model_path.c_str(), session_opts);

            // Introspect input names and shapes
            Ort::AllocatorWithDefaultOptions alloc;
            const size_t num_inputs = session_->GetInputCount();
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name_ptr = session_->GetInputNameAllocated(i, alloc);
                input_names_str_.emplace_back(name_ptr.get());
            }
            const size_t num_outputs = session_->GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name_ptr = session_->GetOutputNameAllocated(i, alloc);
                output_names_str_.emplace_back(name_ptr.get());
            }

            // Build raw char* name lists (required by Run())
            for (const auto& n : input_names_str_)  input_names_cstr_.push_back(n.c_str());
            for (const auto& n : output_names_str_) output_names_cstr_.push_back(n.c_str());

            has_prefill_input_    = has_input("prefill_dist");
            has_prev_exp_input_   = has_input("prev_expert");
            has_prev_layers_input_= has_input("prev_layers");

            model_loaded_ = true;
            std::cout << "[OnnxPredictor Layer " << layer_idx_
                      << "] Loaded: " << model_path
                      << "  inputs=" << num_inputs
                      << " (prefill=" << has_prefill_input_
                      << " prev_exp=" << has_prev_exp_input_
                      << " prev_layers=" << has_prev_layers_input_ << ")"
                      << std::endl;

        } catch (const Ort::Exception& e) {
            std::cerr << "[OnnxPredictor Layer " << layer_idx_
                      << "] Failed to load " << model_path
                      << ": " << e.what() << std::endl;
        }
    }

    ~OnnxPredictor() override {
        std::cout << "[OnnxPredictor Layer " << layer_idx_ << "] Shutdown" << std::endl;
    }

    // -----------------------------------------------------------------------
    // predict_sync — mirrors TorchScriptPredictor's interface exactly
    // -----------------------------------------------------------------------
    std::vector<int64_t> predict_sync(
        torch::Tensor                    embedding,
        c10::optional<torch::Tensor>     prefill_dist       = c10::nullopt,
        c10::optional<torch::Tensor>     prev_expert_onehot = c10::nullopt,
        c10::optional<torch::Tensor>     prev_layers_feat   = c10::nullopt,
        int64_t                          /*source_decode_step*/ = -1) override
    {
        if (!model_loaded_) return {};

        auto t_start = std::chrono::high_resolution_clock::now();

        try {
            // All inputs must be float32 contiguous CPU
            auto to_f32_cpu = [](torch::Tensor t) -> torch::Tensor {
                return t.to(torch::kFloat32).cpu().contiguous();
            };

            // ── embedding ────────────────────────────────────────────────────
            torch::Tensor emb = to_f32_cpu(embedding);
            if (emb.dim() == 1) emb = emb.unsqueeze(0);  // [1, H]

            // ── build input list ─────────────────────────────────────────────
            std::vector<Ort::Value> ort_inputs;
            ort_inputs.reserve(4);

            ort_inputs.push_back(tensor_to_ort_value(emb));

            if (has_prefill_input_) {
                torch::Tensor pf;
                if (prefill_dist.has_value()) {
                    pf = to_f32_cpu(prefill_dist.value());
                    if (pf.dim() == 1) pf = pf.unsqueeze(0);
                } else {
                    // Zero-fill with shape [1, num_experts] inferred from model
                    pf = zeros_like_input(1);  // index 1 = prefill_dist slot
                }
                ort_inputs.push_back(tensor_to_ort_value(pf));
            }

            if (has_prev_exp_input_) {
                torch::Tensor pe;
                if (prev_expert_onehot.has_value()) {
                    pe = to_f32_cpu(prev_expert_onehot.value());
                    if (pe.dim() == 1) pe = pe.unsqueeze(0);
                } else {
                    pe = zeros_like_input(2);  // index 2 = prev_expert slot
                }
                ort_inputs.push_back(tensor_to_ort_value(pe));
            }

            if (has_prev_layers_input_) {
                torch::Tensor pl;
                if (prev_layers_feat.has_value()) {
                    pl = to_f32_cpu(prev_layers_feat.value());
                    if (pl.dim() == 1) pl = pl.unsqueeze(0);
                } else {
                    pl = zeros_like_input(3);  // index 3 = prev_layers slot
                }
                ort_inputs.push_back(tensor_to_ort_value(pl));
            }

            // Trim to actual number of inputs the model accepts
            const size_t n_model_inputs = session_->GetInputCount();
            if (ort_inputs.size() > n_model_inputs) {
                ort_inputs.resize(n_model_inputs);
            }

            // ── run inference ────────────────────────────────────────────────
            auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr_.data(),
                ort_inputs.data(),
                ort_inputs.size(),
                output_names_cstr_.data(),
                output_names_cstr_.size());

            // ── extract scores → ranked expert list ──────────────────────────
            auto& out_val       = ort_outputs[0];
            auto  tensor_info   = out_val.GetTensorTypeAndShapeInfo();
            auto  shape         = tensor_info.GetShape();
            const size_t n_exp  = (shape.size() >= 2) ? static_cast<size_t>(shape[1])
                                                       : static_cast<size_t>(shape[0]);
            const float* scores = out_val.GetTensorData<float>();

            std::vector<int64_t> predicted_experts;
            predicted_experts.reserve(n_exp);

            if (prefetch_threshold_ > 0.0f) {
                // Threshold mode: sigmoid(score) >= threshold
                for (size_t i = 0; i < n_exp; ++i) {
                    float s = 1.0f / (1.0f + std::exp(-scores[i]));
                    if (s >= prefetch_threshold_) {
                        predicted_experts.push_back(static_cast<int64_t>(i));
                    }
                }
                // Sort by sigmoid score descending
                std::sort(predicted_experts.begin(), predicted_experts.end(),
                    [&](int64_t a, int64_t b) {
                        float sa = 1.0f / (1.0f + std::exp(-scores[a]));
                        float sb = 1.0f / (1.0f + std::exp(-scores[b]));
                        return sa > sb;
                    });
            } else {
                // Standard mode: sort all experts by score descending
                std::vector<size_t> order(n_exp);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                    [&](size_t a, size_t b) { return scores[a] > scores[b]; });
                for (size_t i : order) {
                    predicted_experts.push_back(static_cast<int64_t>(i));
                }
            }

            auto t_end = std::chrono::high_resolution_clock::now();
            prediction_time_ms_ = std::chrono::duration<double, std::milli>(t_end - t_start).count();

            return predicted_experts;

        } catch (const Ort::Exception& e) {
            std::cerr << "[OnnxPredictor Layer " << layer_idx_
                      << "] predict_sync error: " << e.what() << std::endl;
            return {};
        }
    }

    bool has_prefetch_threshold() const override { return prefetch_threshold_ > 0.0f; }
    double get_prediction_time_ms() override { return prediction_time_ms_; }

private:
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    bool has_input(const std::string& name) const {
        for (const auto& n : input_names_str_) {
            if (n == name) return true;
        }
        return false;
    }

    // Create a zero tensor matching the shape of model input at `slot_idx`
    torch::Tensor zeros_like_input(size_t slot_idx) const {
        if (slot_idx >= session_->GetInputCount()) {
            return torch::zeros({1, 1});
        }
        auto type_info = session_->GetInputTypeInfo(slot_idx);
        auto shape_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        std::vector<int64_t> sz;
        sz.reserve(shape.size());
        for (auto d : shape) {
            sz.push_back(d > 0 ? d : 1);  // replace symbolic dims with 1
        }
        return torch::zeros(sz, torch::kFloat32);
    }

    // Wrap a contiguous CPU float32 tensor as a non-owning ORT Value
    static Ort::Value tensor_to_ort_value(const torch::Tensor& t) {
        TORCH_CHECK(t.is_cpu() && t.dtype() == torch::kFloat32 && t.is_contiguous(),
                    "OnnxPredictor: tensor must be CPU float32 contiguous");
        auto  mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto  shape    = t.sizes();
        std::vector<int64_t> ort_shape(shape.begin(), shape.end());
        return Ort::Value::CreateTensor<float>(
            mem_info,
            const_cast<float*>(t.data_ptr<float>()),
            static_cast<size_t>(t.numel()),
            ort_shape.data(),
            ort_shape.size());
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    int         layer_idx_;
    std::string ep_name_;
    float       prefetch_threshold_;

    Ort::Env                         env_;
    std::unique_ptr<Ort::Session>    session_;
    bool                             model_loaded_;

    std::vector<std::string>  input_names_str_;
    std::vector<std::string>  output_names_str_;
    std::vector<const char*>  input_names_cstr_;
    std::vector<const char*>  output_names_cstr_;

    bool has_prefill_input_     = false;
    bool has_prev_exp_input_    = false;
    bool has_prev_layers_input_ = false;

    double prediction_time_ms_ = 0.0;
};

#endif  // HETEROPREDICT_ONNXRUNTIME
