#include "unified_llm_w4a16_predict/npuSetup.hpp"
#include "unified_llm_w4a16_predict/unified_llm_w4a16.hpp"
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(unified_llm_w4a16_predict_libtorch, m) {
    m.doc() = "Pybind11 bindings for Unified LLM W4A16 Quantized LibTorch backend";

    py::enum_<ArchitectureType>(m, "ArchitectureType")
        .value("MIXTRAL", ArchitectureType::MIXTRAL)
        .value("QWEN", ArchitectureType::QWEN)
        .export_values();

    py::class_<UnifiedLLMW4A16Impl, std::shared_ptr<UnifiedLLMW4A16Impl>>(m, "UnifiedLLMW4A16")
        .def(
            py::init([](ArchitectureType arch_type, int64_t vocab_size, int64_t hidden_size, int64_t intermediate_size,
                        int64_t num_hidden_layers, int64_t num_attention_heads, int64_t num_key_value_heads, int64_t head_dim,
                        float rms_norm_eps, float rope_theta, int64_t max_seq_len, int64_t max_batch_size, int64_t groupsize,
                        int64_t num_experts, int64_t num_experts_per_tok, const std::string &device_str,
                        int64_t max_cached_experts_per_layer, const std::string predictor_model_path, const std::string &config_path, int64_t prefetch_experts_count, std::vector<int> predict_layers, std::vector<int64_t> per_layer_cache_sizes, std::vector<int64_t> per_layer_prefetch_counts, const std::string& oracle_trace_path, int64_t oracle_lookahead, bool oracle_full_union, float prefetch_threshold, const std::string& predictor_type, int64_t gating_lookahead, bool prefetch_non_evicting, float gating_score_percentile, float speculative_cache_fraction, float oracle_routing_agreement, uint64_t oracle_noise_seed) {
                torch::Device device = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;

                NPUGlobalConfig config;
                if (!config_path.empty()) {
                    try {
                        py::module json = py::module::import("json");
                        py::module re = py::module::import("re");
                        py::module builtins = py::module::import("builtins");
                        py::object open_func = builtins.attr("open");
                        py::object file_obj = open_func(config_path, "r");
                        py::object content = file_obj.attr("read")();
                        file_obj.attr("close")();
                        // Strip // and /* */ comments for JSON5 compatibility
                        content = re.attr("sub")(py::str("//.*"), py::str(""), content);
                        content =
                            re.attr("sub")(py::str("/\\*.*?\\*/"), py::str(""), content, py::arg("flags") = py::int_(16)); // re.DOTALL=16
                        py::object data = json.attr("loads")(content);

                        if (data.contains("heterogeneity"))
                            config.heterogeneity = data["heterogeneity"].cast<std::string>();
                        if (data.contains("debug_verbosity"))
                            config.debug_verbosity = data["debug_verbosity"].cast<int>();
                        if (data.contains("dummy_weights"))
                            config.dummy_weights = data["dummy_weights"].cast<bool>();
                        if (data.contains("warmup"))
                            config.warmup = data["warmup"].cast<bool>();
                        if (data.contains("preload_moe_kernels"))
                            config.preload_moe_kernels = data["preload_moe_kernels"].cast<bool>();
                        if (data.contains("minimal_pdi"))
                            config.minimal_pdi = data["minimal_pdi"].cast<bool>();
                        if (data.contains("predictor_device"))
                            config.predictor_device = data["predictor_device"].cast<std::string>();

                        if (data.contains("rope_scaling")) {
                            py::dict rs = data["rope_scaling"].cast<py::dict>();
                            config.rope_scaling.enabled = true;
                            if (rs.contains("type"))
                                config.rope_scaling.type = rs["type"].cast<std::string>();
                            if (rs.contains("factor"))
                                config.rope_scaling.factor = rs["factor"].cast<float>();
                            if (rs.contains("low_freq_factor"))
                                config.rope_scaling.low_freq_factor = rs["low_freq_factor"].cast<float>();
                            if (rs.contains("high_freq_factor"))
                                config.rope_scaling.high_freq_factor = rs["high_freq_factor"].cast<float>();
                            if (rs.contains("original_max_position_embeddings"))
                                config.rope_scaling.original_max_position_embeddings = rs["original_max_position_embeddings"].cast<float>();
                        }

                        if (data.contains("kernels")) {
                            py::list kernels = data["kernels"].cast<py::list>();
                            for (auto item : kernels) {
                                py::dict k = item.cast<py::dict>();
                                NPUKernelConfig kc;
                                if (k.contains("use"))
                                    kc.use = k["use"].cast<bool>();
                                if (k.contains("npuM"))
                                    kc.npuM = k["npuM"].cast<int>();
                                if (k.contains("npuK"))
                                    kc.npuK = k["npuK"].cast<int>();
                                if (k.contains("npuN"))
                                    kc.npuN = k["npuN"].cast<int>();
                                if (k.contains("forM"))
                                    kc.forM = k["forM"].cast<int>();
                                if (k.contains("forK"))
                                    kc.forK = k["forK"].cast<int>();
                                if (k.contains("forN"))
                                    kc.forN = k["forN"].cast<int>();
                                if (k.contains("layer_type"))
                                    kc.layer_type = k["layer_type"].cast<std::string>();
                                if (k.contains("config"))
                                    kc.config = k["config"].cast<int>();
                                if (k.contains("num_tiles"))
                                    kc.num_tiles = k["num_tiles"].cast<int>();
                                if (k.contains("xclbin"))
                                    kc.xclbin = k["xclbin"].cast<std::string>();
                                if (k.contains("inst"))
                                    kc.inst = k["inst"].cast<std::string>();
                                if (k.contains("fw_path"))
                                    kc.fw_path = k["fw_path"].cast<std::string>();
                                if (k.contains("tile_size"))
                                    kc.tile_size = k["tile_size"].cast<std::string>();
                                if (k.contains("col"))
                                    kc.col = k["col"].cast<std::string>();
                                if (k.contains("dtype"))
                                    kc.dtype = k["dtype"].cast<std::string>();
                                config.kernels.push_back(kc);
                            }
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "Error parsing config file in Python binding: " << e.what() << std::endl;
                        throw;
                    }
                }

                return std::make_shared<UnifiedLLMW4A16Impl>(arch_type, vocab_size, hidden_size, intermediate_size, num_hidden_layers,
                                                             num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, rope_theta,
                                                             config, max_seq_len, max_batch_size, groupsize, num_experts,
                                                             num_experts_per_tok, device, max_cached_experts_per_layer, predictor_model_path, prefetch_experts_count, predict_layers, per_layer_cache_sizes, per_layer_prefetch_counts, oracle_trace_path, oracle_lookahead, oracle_full_union, prefetch_threshold, predictor_type, gating_lookahead, prefetch_non_evicting, gating_score_percentile, speculative_cache_fraction, oracle_routing_agreement, oracle_noise_seed);
            }),
            py::arg("arch_type"), py::arg("vocab_size"), py::arg("hidden_size"), py::arg("intermediate_size"), py::arg("num_hidden_layers"),
            py::arg("num_attention_heads"), py::arg("num_key_value_heads"), py::arg("head_dim"), py::arg("rms_norm_eps"),
            py::arg("rope_theta"), py::arg("max_seq_len") = 8192, py::arg("max_batch_size") = 1, py::arg("groupsize") = 128,
            py::arg("num_experts") = 0, py::arg("num_experts_per_tok") = 0, py::arg("device") = "cpu",
            py::arg("max_cached_experts_per_layer") = 0, py::arg("predictor_model_path") = "", py::arg("config_path") = "",
            py::arg("prefetch_experts_count") = 1, py::arg("predict_layers") = std::vector<int>(),
            py::arg("per_layer_cache_sizes") = std::vector<int64_t>(),
            py::arg("per_layer_prefetch_counts") = std::vector<int64_t>(),
            py::arg("oracle_trace_path") = "", py::arg("oracle_lookahead") = 0, py::arg("oracle_full_union") = false,
            py::arg("prefetch_threshold") = 0.0f, py::arg("predictor_type") = "torchscript",
            py::arg("gating_lookahead") = 1, py::arg("prefetch_non_evicting") = false,
            py::arg("gating_score_percentile") = 0.0f, py::arg("speculative_cache_fraction") = 0.0f,
            py::arg("oracle_routing_agreement") = 1.0f, py::arg("oracle_noise_seed") = 42)
        .def(
            "forward",
            [](UnifiedLLMW4A16Impl &self, torch::Tensor input_ids, int64_t start_pos) -> torch::Tensor {
                // Ensure input is contiguous
                if (!input_ids.is_contiguous())
                    input_ids = input_ids.contiguous();
                return self.forward(input_ids, start_pos);
            },
            py::arg("input_ids"), py::arg("start_pos") = 0)
        .def(
            "generate",
            [](UnifiedLLMW4A16Impl &self, torch::Tensor input_ids, int64_t max_new_tokens, float temperature, float top_p, int64_t top_k,
               int64_t eos_token_id) -> torch::Tensor {
                // Ensure input is contiguous
                if (!input_ids.is_contiguous())
                    input_ids = input_ids.contiguous();
                return self.generate(input_ids, max_new_tokens, temperature, top_p, top_k, eos_token_id);
            },
            py::arg("input_ids"), py::arg("max_new_tokens"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f, py::arg("top_k") = 50,
            py::arg("eos_token_id") = -1)
        .def("to", &UnifiedLLMW4A16Impl::to, py::arg("device"))
        .def("load_quantized_weights_from_safetensors", &UnifiedLLMW4A16Impl::load_quantized_weights_from_safetensors, py::arg("filename"))
        .def("load_non_quantized_weights_from_safetensors", &UnifiedLLMW4A16Impl::load_non_quantized_weights_from_safetensors,
             py::arg("filename"))
        .def("load_quantized_weights_from_bins", &UnifiedLLMW4A16Impl::load_quantized_weights_from_bins,
             py::arg("weights_dir"), py::arg("expert_weights_dir") = "",
             "Load bins from weights_dir (attention) and optionally expert_weights_dir (MoE experts). "
             "Expert dir auto-detects packed (layer_L_expert_E.bin) vs unpacked format.")
        .def("set_lambda", &UnifiedLLMW4A16Impl::set_lambda, py::arg("lambda"), py::arg("layer_idx") = -1)
        .def("get_lambda", &UnifiedLLMW4A16Impl::get_lambda, py::arg("layer_idx") = 0)
        .def("set_forced_top_n", &UnifiedLLMW4A16Impl::set_forced_top_n, py::arg("n"))
        .def("set_forced_top_p", &UnifiedLLMW4A16Impl::set_forced_top_p, py::arg("p"),
             "Force the minimum set of experts whose cumulative softmax probability >= p into the cache mask. "
             "Set to -1.0 to disable (default).")
        .def("set_mass_threshold_substitution_p", &UnifiedLLMW4A16Impl::set_mass_threshold_substitution_p, py::arg("p"),
             "Alternate routing mode: keep the smallest top-k prefix with cumulative probability >= p, "
             "then substitute the remaining routed slots.")
        .def("set_prefill_top_n", &UnifiedLLMW4A16Impl::set_prefill_top_n, py::arg("n"))
        .def("set_cache_policy", &UnifiedLLMW4A16Impl::set_cache_policy, py::arg("policy_name"), py::arg("layer_idx") = -1)
        .def("set_random_fill_mode", &UnifiedLLMW4A16Impl::set_random_fill_mode, py::arg("on"))
        .def("calculate_generation_perplexity", &UnifiedLLMW4A16Impl::calculate_generation_perplexity, py::arg("input_ids"))
        .def("print_cache_stats", &UnifiedLLMW4A16Impl::print_cache_stats)
        .def("reset_cache_stats", &UnifiedLLMW4A16Impl::reset_cache_stats)
        .def("get_cache_stats", &UnifiedLLMW4A16Impl::get_cache_stats)
        .def("get_predictor_stats", &UnifiedLLMW4A16Impl::get_predictor_stats)
        .def("reset_predictor_stats", &UnifiedLLMW4A16Impl::reset_predictor_stats)
        .def("set_suppress_predictor_stats", &UnifiedLLMW4A16Impl::set_suppress_predictor_stats, py::arg("v"))
        .def("set_predictor_lookahead", &UnifiedLLMW4A16Impl::set_predictor_lookahead, py::arg("stride"),
             "Fire the predictor every `stride` decode tokens (set to lookahead depth fN).")
        .def("set_prefetch_non_evicting", &UnifiedLLMW4A16Impl::set_prefetch_non_evicting, py::arg("enabled"),
             "When true, speculative prefetch only uses empty cache slots (paper non-evicting mode).")
        .def("prewarm_experts", &UnifiedLLMW4A16Impl::prewarm_experts, py::arg("num_to_warm"), py::arg("verbose") = true)

        .def("begin_oracle_trace_capture", &UnifiedLLMW4A16Impl::begin_oracle_trace_capture,
             "Begin recording routed expert IDs on each decode forward in generate().")
        .def("cancel_oracle_trace_capture", &UnifiedLLMW4A16Impl::cancel_oracle_trace_capture)
        .def(
            "write_oracle_trace_file",
            [](UnifiedLLMW4A16Impl& self, const std::string& path, torch::Tensor prompt_ids, torch::Tensor output_ids,
               const std::string& prompt_text, const std::string& generated_text, const std::string& model_name) {
                return self.write_oracle_trace_file(path, prompt_ids, output_ids, prompt_text, generated_text, model_name);
            },
            py::arg("path"), py::arg("prompt_ids"), py::arg("output_ids"), py::arg("prompt_text") = "",
            py::arg("generated_text") = "", py::arg("model_name") = "qwen3_30b")

        .def("initialize_dummy_weights", &UnifiedLLMW4A16Impl::initialize_dummy_weights, py::arg("seed") = 42,
             "Initialize dummy weights for testing");
}
