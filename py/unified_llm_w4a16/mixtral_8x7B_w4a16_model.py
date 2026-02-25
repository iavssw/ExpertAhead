"""
Mixtral 8x7B v0.1 AWQ w4a16 Python frontend.
Handles tokenization and interfaces with the C++ base backend.
"""

import os
import sys
import json
import time
import re
import subprocess
from pathlib import Path
from typing import Optional, Union, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_script_dir = Path(__file__).parent.resolve()
_project_root = _script_dir.parent.parent
_build_dir = _project_root / "build" / "py" / "unified_llm_w4a16"

if _build_dir.exists():
    sys.path.insert(0, str(_build_dir))
else:
    _alt_build = _project_root / "build" / "py"
    if _alt_build.exists():
        sys.path.insert(0, str(_alt_build))
    _local_build = Path("build") / "py" / "unified_llm_w4a16"
    if _local_build.exists():
        sys.path.insert(0, str(_local_build.resolve()))

ArchitectureType = None


def load_config_with_comments(path: str) -> dict:
    """Load JSON/JSON5-like config with // and /* */ comments stripped."""
    try:
        with open(path, "r") as f:
            content = f.read()
        content = re.sub(r"//.*", "", content)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        return json.loads(content)
    except Exception as e:
        print(f"Error loading config with comment stripping: {e}")
        with open(path, "r") as f:
            return json.load(f)


def _model_name_from_path(model_path: str) -> str:
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    return model_name.replace("/", "_").replace("\\", "_")


def _write_tensor_raw(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.contiguous().cpu()
    t_bytes = t.view(torch.uint8)
    t_bytes.numpy().tofile(str(path))


def _unpack_awq_zigzag_to_contiguous(packed_int32: torch.Tensor) -> torch.Tensor:
    device = packed_int32.device
    permutation = [0, 4, 1, 5, 2, 6, 3, 7]
    parts = []
    for k in range(8):
        shift_amount = permutation[k] * 4
        if shift_amount > 0:
            part = torch.bitwise_right_shift(packed_int32, shift_amount)
        else:
            part = packed_int32
        part = torch.bitwise_and(part, 0x0F).to(torch.uint8)
        parts.append(part)
    return torch.stack(parts, dim=-1).to(device)


def _unpack_awq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    # Input: [In, Out/8] int32 -> Output: [In, Out] uint8
    qweight = qweight.t().contiguous()
    out_features_div_8 = qweight.size(0)
    in_features = qweight.size(1)
    out_features = out_features_div_8 * 8
    unpacked = _unpack_awq_zigzag_to_contiguous(qweight)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    unpacked = unpacked.view(out_features, in_features)
    return unpacked.t().contiguous().to(torch.uint8)


def _unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    # Input: [G, Out/8] int32 -> Output: [G, Out] int8
    unpacked = _unpack_awq_zigzag_to_contiguous(qzeros)
    n_groups = qzeros.size(0)
    out_features = qzeros.size(1) * 8
    unpacked = unpacked.view(n_groups, out_features)
    return unpacked.contiguous().to(torch.int8)


def _pack_qweight_out_in2(unpacked_qweight: torch.Tensor) -> torch.Tensor:
    # Input: [In, Out] int8 -> Output: [Out, In/2] uint8
    w_out_in = unpacked_qweight.t().contiguous()
    out_features = w_out_in.size(0)
    in_features = w_out_in.size(1)
    w_view = w_out_in.view(out_features, in_features // 2, 2)
    w_low = w_view.select(-1, 0).to(torch.uint8)
    w_high = w_view.select(-1, 1).to(torch.uint8)
    packed = torch.bitwise_or(torch.bitwise_and(w_low, 0x0F), torch.bitwise_left_shift(torch.bitwise_and(w_high, 0x0F), 4))
    return packed.to(torch.uint8)


def _normalize_group_tensor(t: torch.Tensor, out_feat: int) -> torch.Tensor:
    if t is None:
        return t
    if t.dim() == 1:
        if t.numel() == out_feat:
            return t
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
        return t
    if t.dim() == 2:
        if t.size(0) == out_feat:
            return t
        if t.size(1) == out_feat:
            return t.t().contiguous()
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
        return t
    if t.numel() % out_feat == 0:
        return t.view(out_feat, -1)
    return t


def _align_zeros_to_scales(zeros: torch.Tensor, scales_out: torch.Tensor, out_feat: int) -> torch.Tensor:
    zeros_out = _normalize_group_tensor(zeros, out_feat)
    if scales_out is None:
        return zeros_out
    if scales_out.dim() == 2:
        target_groups = scales_out.size(1)
        if zeros_out.dim() == 2:
            if zeros_out.size(0) == target_groups and zeros_out.size(1) == out_feat:
                zeros_out = zeros_out.t().contiguous()
            elif zeros_out.size(0) == out_feat and zeros_out.size(1) != target_groups and zeros_out.numel() == out_feat * target_groups:
                zeros_out = zeros_out.reshape(out_feat, target_groups)
        elif zeros_out.numel() == out_feat * target_groups:
            zeros_out = zeros_out.view(out_feat, target_groups)
    return zeros_out


def _get_quantized_tensors(state_dict, base_name: str):
    qweight = state_dict.get(base_name + ".qweight")
    scales = state_dict.get(base_name + ".scales")
    qzeros = state_dict.get(base_name + ".qzeros")
    g_idx = state_dict.get(base_name + ".g_idx")
    if qweight is None or scales is None or qzeros is None:
        missing = [k for k in (base_name + ".qweight", base_name + ".scales", base_name + ".qzeros") if k not in state_dict]
        raise KeyError(f"Missing quantized tensors for {base_name}: {missing}")
    return qweight, scales, qzeros, g_idx


class Mixtral8x7BW4A16Model:
    """Mixtral 8x7B v0.1 AWQ w4a16 quantized model wrapper."""

    def __init__(
        self,
        model_path: Optional[str] = "TheBloke/mixtral-8x7b-v0.1-AWQ",
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seq_len: int = 8192,
        max_batch_size: int = 1,
        groupsize: int = 128,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        device: str = "cuda",
        backend: str = "base",
        config_path: Optional[str] = None,
        max_cached_experts_per_layer: int = 8,
        use_cached_moe: bool = False,
        predictor_models_dir: str = "/home/michael/mixtral_project/expert_prediction_full/embedding_only_predictors",
        weights_dir: str = "",
        predictor_device: str = "gpu",
        prefetch_experts_count: int = 1,
        predict_layers: Optional[List[int]] = None,
    ):
        """
        Initialize Mixtral 8x7B v0.1 AWQ w4a16 quantized model.

        predictor_device: where to run the TorchScript expert predictor.
            "gpu"  -> same GPU as the MoE layer (default — use this to benchmark/validate
                      speculative-loading speedup without NPU hardware)
            "cpu"  -> CPU, which routes through the Ryzen AI NPU backend on Strix
            "auto" -> inferred from heterogeneity in config (gpu->GPU, hetero/npu->CPU/NPU)
        """

        if backend not in ["base", "predict", "cached"]:
            raise ValueError(f"Invalid backend: {backend}. Choose from: base, predict, cached")

        try:
            if backend == "base":
                import unified_llm_w4a16_base_libtorch as backend_module
            elif backend == "predict":
                import unified_llm_w4a16_predict_libtorch as backend_module
            elif backend == "cached":
                import unified_llm_w4a16_cached_libtorch as backend_module
        except ImportError as e:
            raise ImportError(f"Could not import {backend} backend: {e}")

        global ArchitectureType
        ArchitectureType = backend_module.ArchitectureType

        self.device = device
        self.model_path = model_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.groupsize = groupsize
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        constructor_args = [
            ArchitectureType.MIXTRAL,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            max_seq_len,
            max_batch_size,
            groupsize,
            num_experts,
            num_experts_per_tok,
        ]

        if backend == "cached":
             # Cached backend specific args: max_cached, device, config
             constructor_args.append(max_cached_experts_per_layer)
             constructor_args.append(device)
        else:
             # Base/Predict backend args (assuming predict uses extra args)
             constructor_args.append(device)
             # predict backend usage would need to be checked, for now keeping compatibility with previous style if backend=predict?
             if backend == "predict":
                 constructor_args.append(max_cached_experts_per_layer) # Predict might use this too?
                 constructor_args.append(predictor_models_dir)

        if config_path is None:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/configs_strixH_mixtral7x8B.json5"))

        # If predictor_device override is requested, write a temp config with the field injected.
        # The C++ constructor reads predictor_device from the JSON on construction.
        _temp_config_path = None
        if predictor_device != "auto" and backend == "predict":
            import tempfile, json
            base_cfg = load_config_with_comments(config_path) if config_path else {}
            base_cfg["predictor_device"] = predictor_device
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(base_cfg, tmp)
            tmp.close()
            _temp_config_path = tmp.name
            print(f"[predictor_device={predictor_device}] Using temp config: {_temp_config_path}")
            constructor_args.append(_temp_config_path)
        else:
            constructor_args.append(config_path)
        if backend == "predict":
            constructor_args.append(prefetch_experts_count)
            constructor_args.append(predict_layers if predict_layers is not None else [])
            
        self.model = backend_module.UnifiedLLMW4A16(*constructor_args)

        # Clean up temp config after C++ has read it
        if _temp_config_path:
            import os as _os
            try:
                _os.unlink(_temp_config_path)
            except OSError:
                pass

        self.config = {}
        self.use_pre_saved_weights = False
        self.debug_verbosity = 0

        use_dummy = False
        if config_path:
            try:
                self.config = load_config_with_comments(config_path)
                self.use_pre_saved_weights = bool(self.config.get("usePreSavedWeights", False))
                self.debug_verbosity = int(self.config.get("debug_verbosity", 0))
                if self.config.get("heterogeneity", "gpu") == "cpu":
                    self.device = "cpu"
                if self.config.get("dummy_weights", False):
                    use_dummy = True
                    print("Dummy weights enabled in config.")
            except Exception as e:
                print(f"Error reading config: {e}")

        if use_dummy:
            print("Initializing dummy weights...")
            self.model.initialize_dummy_weights()
        elif model_path:
            self._load_quantized_weights(model_path, weights_folder="model_weights")

        if backend in ["cached", "predict"]:
            print(f"Pre-warming expert cache with {max_cached_experts_per_layer} experts...")
            self.model.prewarm_experts(max_cached_experts_per_layer)

        tokenizer_path = tokenizer_path or model_path
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

    def _load_quantized_weights(self, model_path: str, weights_folder: str = "model_weights"):
        """Load quantized weights from safetensors and pass to the C++ backend."""
        print(f"Loading quantized weights from {model_path}...")
        try:
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file, save_file
            import glob
            import shutil

            script_dir = Path(__file__).parent
            weights_dir = script_dir / weights_folder
            os.makedirs(weights_dir, exist_ok=True)

            model_name = _model_name_from_path(model_path)
            saved_safetensors = weights_dir / f"{model_name}.safetensors"

            if saved_safetensors.exists():
                print(f"Using saved safetensors: {saved_safetensors}")
                safetensors_files = [str(saved_safetensors)]
            else:
                if not os.path.exists(model_path):
                    print(f"Model path {model_path} not found locally, downloading from Hub...")
                    model_path = snapshot_download(repo_id=model_path)
                    print(f"Model downloaded to {model_path}")

                safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
                if not safetensors_files:
                    raise FileNotFoundError(f"No safetensors files found in {model_path}")

                if len(safetensors_files) == 1:
                    print(f"Saving safetensors to {saved_safetensors}...")
                    shutil.copy2(safetensors_files[0], saved_safetensors)
                    print(f"Saved safetensors file: {saved_safetensors}")
                else:
                    print(f"Found {len(safetensors_files)} safetensors files, merging...")
                    merged_state_dict = {}
                    for st_file in safetensors_files:
                        print(f"  Loading {st_file}...")
                        merged_state_dict.update(load_file(st_file))
                    print(f"Saving merged safetensors to {saved_safetensors}...")
                    save_file(merged_state_dict, str(saved_safetensors))
                    safetensors_files = [str(saved_safetensors)]

            use_presaved = self.use_pre_saved_weights

            if use_presaved:
                presaved_dir = weights_dir / f"{model_name}_unpacked"
                self._prepare_presaved_weights(saved_safetensors, presaved_dir)

                t0 = time.time()
                self.model.load_non_quantized_weights_from_safetensors(str(saved_safetensors))
                self.model.load_quantized_weights_from_bins(str(presaved_dir))
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded from pre-saved bins in {self.load_time:.2f} seconds")
            else:
                t0 = time.time()
                self.model.load_quantized_weights_from_safetensors(str(saved_safetensors))
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded in {self.load_time:.2f} seconds")

        except Exception as e:
            print(f"Error loading quantized weights: {e}")
            import traceback
            traceback.print_exc()
            print("\nNote: Falling back to randomly initialized weights.")
            print("The model will not produce meaningful output without proper weights.")

    def _prepare_presaved_weights(self, saved_safetensors: Path, presaved_dir: Path) -> None:
        manifest_path = presaved_dir / "manifest.json"
        model_name = saved_safetensors.stem
        hetero_mode = str(self.config.get("heterogeneity", "")).lower() if isinstance(self.config, dict) else ""

        expected_manifest = {
            "format_version": 1,
            "model_name": model_name,
            "heterogeneity": hetero_mode,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "groupsize": self.groupsize,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
        }
        expected_manifest["unpacked_layout"] = "out_groups_v2"

        def _bins_exist() -> bool:
            if not presaved_dir.exists():
                return False
            for layer_idx in range(self.num_hidden_layers):
                for short_name in ["q", "k", "v", "o"]:
                    if not (presaved_dir / f"layer_{layer_idx}_{short_name}.qweight.bin").exists():
                        return False
                    if not (presaved_dir / f"layer_{layer_idx}_{short_name}.scales.bin").exists():
                        return False
                    if not (presaved_dir / f"layer_{layer_idx}_{short_name}.zeros.bin").exists():
                        return False
                for e in range(self.num_experts):
                    for short_name in ["gate", "up", "down"]:
                        if not (presaved_dir / f"layer_{layer_idx}_expert_{e}_{short_name}.qweight.bin").exists():
                            return False
                        if not (presaved_dir / f"layer_{layer_idx}_expert_{e}_{short_name}.scales.bin").exists():
                            return False
                        if not (presaved_dir / f"layer_{layer_idx}_expert_{e}_{short_name}.zeros.bin").exists():
                            return False
            return True

        if _bins_exist():
            if self.debug_verbosity >= 1:
                print(f"Using existing pre-saved weights in {presaved_dir}")
            return

        print(f"Preprocessing weights into bin files under {presaved_dir}...")
        from safetensors.torch import load_file

        state_dict = load_file(str(saved_safetensors))

        def _write_quantized(base_name: str, out_name: str, in_feat: int, out_feat: int):
            qweight, scales, qzeros, _g_idx = _get_quantized_tensors(state_dict, base_name)

            qweight = qweight.cpu()
            scales = scales.cpu()
            qzeros = qzeros.cpu()

            is_awq = qweight.size(0) == in_feat and qweight.size(1) == out_feat // 8
            if not is_awq:
                raise ValueError(f"Unsupported quantization format for {base_name}: expected AWQ qweight layout.")

            qweight_unpacked = _unpack_awq_qweight(qweight.to(torch.int32))
            scales = scales.to(torch.bfloat16)
            zeros = _unpack_awq_qzeros(qzeros.to(torch.int32))

            qweight_packed = _pack_qweight_out_in2(qweight_unpacked.to(torch.int8))

            scales_out = _normalize_group_tensor(scales, out_feat)
            zeros_out = _align_zeros_to_scales(zeros, scales_out, out_feat)

            scales_out = scales_out.to(torch.bfloat16)
            zeros_out = zeros_out.to(torch.int8)

            _write_tensor_raw(presaved_dir / f"{out_name}.qweight.bin", qweight_packed.to(torch.uint8))
            _write_tensor_raw(presaved_dir / f"{out_name}.scales.bin", scales_out)
            _write_tensor_raw(presaved_dir / f"{out_name}.zeros.bin", zeros_out)

        layer_specs = [
            ("self_attn.q_proj", "q", self.hidden_size, self.num_attention_heads * self.head_dim),
            ("self_attn.k_proj", "k", self.hidden_size, self.num_key_value_heads * self.head_dim),
            ("self_attn.v_proj", "v", self.hidden_size, self.num_key_value_heads * self.head_dim),
            ("self_attn.o_proj", "o", self.num_attention_heads * self.head_dim, self.hidden_size),
        ]

        for layer_idx in range(self.num_hidden_layers):
            for suffix, short_name, in_feat, out_feat in layer_specs:
                base_name = f"model.layers.{layer_idx}.{suffix}"
                _write_quantized(base_name, f"layer_{layer_idx}_{short_name}", in_feat, out_feat)

            for expert_idx in range(self.num_experts):
                expert_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
                if expert_prefix + ".w1.qweight" in state_dict:
                    gate_base = expert_prefix + ".w1"
                    up_base = expert_prefix + ".w3"
                    down_base = expert_prefix + ".w2"
                elif expert_prefix + ".gate_proj.qweight" in state_dict:
                    gate_base = expert_prefix + ".gate_proj"
                    up_base = expert_prefix + ".up_proj"
                    down_base = expert_prefix + ".down_proj"
                else:
                    raise KeyError(f"Missing Mixtral expert weights for layer {layer_idx}, expert {expert_idx}")

                _write_quantized(gate_base, f"layer_{layer_idx}_expert_{expert_idx}_gate", self.hidden_size, self.intermediate_size)
                _write_quantized(up_base, f"layer_{layer_idx}_expert_{expert_idx}_up", self.hidden_size, self.intermediate_size)
                _write_quantized(down_base, f"layer_{layer_idx}_expert_{expert_idx}_down", self.intermediate_size, self.hidden_size)

        presaved_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(expected_manifest, f, indent=2)
        print(f"Pre-saved weights written to {presaved_dir}")

    def tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available.")
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        return encoded["input_ids"].to(self.device)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
        start_pos: int = 0
    ) -> torch.Tensor:
        """Generate tokens from input using C++ backend."""
        eos_token_id = -1
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            eos_token_id = self.tokenizer.eos_token_id

        print(" ========================== Starting generation ==========================")

        return self.model.generate(
            input_ids,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            eos_token_id
        )

    def __call__(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Forward pass."""
        if isinstance(input_ids, str):
            input_ids = self.tokenize(input_ids)
        return self.model.forward(input_ids, start_pos)

    def set_lambda(self, lambda_value: float, layer_idx: int = -1):
        """
        Set the router logit bias parameter (lambda).
        
        Args:
            lambda_value: Bias strength in range [0, 1].
                         0.0: no bias (standard routing)
                         0.0 < λ ≤ 1.0: bias toward cached experts
                         Higher values = stronger bias toward cache
            layer_idx: Layer index to modify, or -1 for all layers
        
        Raises:
            ValueError: If lambda_value is not in [0, 1]
        """
        if lambda_value < 0.0 or lambda_value > 1.0:
            raise ValueError(f"Lambda must be in range [0, 1], got: {lambda_value}")
        self.model.set_lambda(lambda_value, layer_idx)

    def get_lambda(self, layer_idx: int = 0) -> float:
        """Get lambda parameter for specified layer."""
        return self.model.get_lambda(layer_idx)

    def set_layer_correlation_constants(self, constants: List[float]):
        """Set the correlation constant for each layer."""
        if hasattr(self.model, "set_layer_correlation_constants"):
            self.model.set_layer_correlation_constants(constants)

    def calculate_generation_perplexity(self, text: str) -> float:
        """
        Calculate generation-time perplexity using the optimized C++ backend loop.
        This provides a fair evaluation of cache performance during generation.
        """
        if isinstance(text, str):
            input_ids = self.tokenize(text)
            # Ensure input_ids on correct device is handled by tokenize/backend
        elif isinstance(text, torch.Tensor):
            input_ids = text
        else:
            raise ValueError("Text must be string or tensor")
            
        # Call C++ backend method directly
        if hasattr(self.model, "calculate_generation_perplexity"):
             return self.model.calculate_generation_perplexity(input_ids)
        else:
             print("Error: Backend does not support calculate_generation_perplexity")
             return 0.0

    def print_cache_stats(self):
        """Print cache hits and misses."""
        if hasattr(self.model, "print_cache_stats"):
            self.model.print_cache_stats()

    def reset_cache_stats(self):
        """Reset cache hits and misses."""
        if hasattr(self.model, "reset_cache_stats"):
            self.model.reset_cache_stats()
    
    def get_cache_stats(self):
        """Get cache statistics as (total_hits, total_misses) tuple."""
        if hasattr(self.model, "get_cache_stats"):
            return self.model.get_cache_stats()
        return (0, 0)

    def get_predictor_stats(self):
        """
        Get per-layer predictor hit-rate stats.

        Returns a list of (no_bias_hits, with_bias_hits, total) tuples,
        one per MoE layer. 'total' is the number of generation tokens for
        which a prior prediction existed and was evaluated.
        """
        if hasattr(self.model, "get_predictor_stats"):
            return self.model.get_predictor_stats()
        return []

    def reset_predictor_stats(self):
        """Reset predictor hit-rate counters across all layers."""
        if hasattr(self.model, "reset_predictor_stats"):
            self.model.reset_predictor_stats()


def run_prompt_test(target_tokens, model_path=None, tokenizer_path=None, device="cuda", backend="base",
                    max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, generate=True, config_path=None):
    """
    Run prompt test case: load single long prompt from prompts.txt,
    concatenate base prompt, truncate to requested token count, and generate output.
    """
    from pathlib import Path

    script_dir = Path(__file__).parent
    prompts_file = script_dir.parent / "prompts.txt"

    if not prompts_file.exists():
        print(f"Error: Prompts file not found at {prompts_file}")
        return 1

    print("=" * 60)
    print(f"PROMPT TEST: Target token count = {target_tokens}")
    print("=" * 60 + "\n")

    print(f"Reading long prompt from: {prompts_file}")
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_content = f.read()

    long_prompt = prompts_content.replace("<|begin_of_text|>", "").strip()

    print("Initializing tokenizer...")
    try:
        from transformers import AutoTokenizer
        if tokenizer_path is None:
            tokenizer_path = "mistralai/Mixtral-8x7B-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return 1

    base_prompt = """Please provide a comprehensive summary of the following document. The summary should capture the main points, key developments, and important themes discussed in the text."""

    base_prompt_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
    base_prompt_token_count = len(base_prompt_tokens)

    long_prompt_tokens = tokenizer.encode(long_prompt, add_special_tokens=False)
    long_prompt_token_count = len(long_prompt_tokens)

    print("Token counts:")
    print(f"  Base prompt: {base_prompt_token_count} tokens")
    print(f"  Long prompt (doc + Summary:): {long_prompt_token_count} tokens")
    print(f"  Combined (before truncation): {base_prompt_token_count + long_prompt_token_count} tokens")
    print(f"  Target: {target_tokens} tokens\n")

    full_prompt_tokens = base_prompt_tokens + long_prompt_tokens
    full_token_count = len(full_prompt_tokens)

    if full_token_count > target_tokens:
        truncated_tokens = full_prompt_tokens[:target_tokens]
        actual_token_count = len(truncated_tokens)
        print(f"Truncated from {full_token_count} to {actual_token_count} tokens")
    elif full_token_count < target_tokens:
        tokens_needed = target_tokens - full_token_count
        long_prompt_token_count = len(long_prompt_tokens)

        if long_prompt_token_count > 0:
            additional_repeats = (tokens_needed + long_prompt_token_count - 1) // long_prompt_token_count
            if additional_repeats == 0:
                additional_repeats = 1

            extended_tokens = full_prompt_tokens + (long_prompt_tokens * additional_repeats)
            truncated_tokens = extended_tokens[:target_tokens]
            actual_token_count = len(truncated_tokens)
            print(f"Extended from {full_token_count} to {actual_token_count} tokens (target: {target_tokens}, added {additional_repeats} more document copies)")
        else:
            truncated_tokens = full_prompt_tokens
            actual_token_count = full_token_count
            print(f"Warning: Cannot extend prompt (long_prompt is empty). Using {actual_token_count} tokens")
    else:
        truncated_tokens = full_prompt_tokens
        actual_token_count = full_token_count
        print(f"Prompt is exactly {actual_token_count} tokens (target: {target_tokens})")

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL:")
    print("=" * 60 + "\n")

    if model_path is None:
        model_path = "TheBloke/mixtral-8x7b-v0.1-AWQ"

    print("Initializing Mixtral 8x7B v0.1 AWQ w4a16 quantized model...")
    try:
        model = Mixtral8x7BW4A16Model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            backend=backend,
            config_path=config_path
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    if generate:
        print("GENERATING OUTPUT:")
    else:
        print("FORWARD PASS (NO GENERATION):")
    print("=" * 60 + "\n")

    try:
        input_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)

        if not generate:
            print("Running forward pass only...")
            with torch.no_grad():
                start_time = time.time()
                logits = model(input_ids)
                end_time = time.time()
            print(f"Prefill time: {end_time - start_time:.4f} seconds")

            print(f"Logits shape: {logits.shape}")
            print(f"First 4 logits (last token in batch 0): {logits[0, -1, :4].tolist()}")
            return 0

        print(f"Input prompt length: {actual_token_count} tokens")
        print(f"Generating up to {max_new_tokens} new tokens...")
        print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}\n")
        print(f"Input token IDs shape: {input_ids.shape}")

        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )

        if model.tokenizer is not None:
            prompt_len = input_ids.size(1)
            generated_tokens = generated[0, prompt_len:].tolist()
            decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)

            print(f"\n{'='*60}")
            print("GENERATED TEXT:")
            print(f"{'='*60}")
            print(decoded_generated)
            print(f"{'='*60}")
            print(f"\nGenerated {len(generated_tokens)} tokens")
        else:
            print(f"\nGenerated token IDs: {generated}")
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n{'=' * 60}")
    if hasattr(model, "load_time"):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print("Done!")
    print(f"{'=' * 60}\n")
    return 0


def main():
    """Example usage of Mixtral8x7BW4A16Model when run as a script."""
    import argparse

    parser = argparse.ArgumentParser(description="Mixtral 8x7B v0.1 AWQ W4A16 Quantized Model - Unified LibTorch Backend")
    parser.add_argument(
        "--text",
        type=str,
        default="What is the meaning of life the universe and everything?",
        help="Input text to process (default: 'What is the meaning of life?')"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer or HuggingFace model name"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="TheBloke/mixtral-8x7b-v0.1-AWQ",
        help="Path to quantized model (default: TheBloke/mixtral-8x7b-v0.1-AWQ)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cuda)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="base",
        choices=["base", "predict", "cached"],
        help="Backend to use: base (all experts), predict (heterogeneous), cached (selective loading)"
    )
    parser.add_argument(
        "--expert-cache",
        type=int,
        default=2,
        help="Maximum number of cached experts per layer (for cached backend)"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/configs_strixH_mixtral7x8B.json5")),
        help="Path to NPU config JSON"
    )
    parser.add_argument(
        "--predictor-model",
        type=str,
        default="/home/michael/mixtral_project/expert_prediction_full/embedding_only_predictors",
        help="Path to predictor model directory (containing layer_X subdirs) for 'predict' backend"
    )
    parser.add_argument(
        "--generate",
        dest="generate",
        action="store_true",
        default=True,
        help="Generate text instead of just getting logits (default: True)"
    )
    parser.add_argument(
        "--no-generate",
        dest="generate",
        action="store_false",
        help="Disable generation, just get logits"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=40,
        help="Maximum number of tokens to generate (if --generate is used, default: 16)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy decoding)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (0.0-1.0)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter: only considers the top k most likely tokens."
    )
    parser.add_argument(
        "--prompt-test",
        type=int,
        default=None,
        help="Run prompt test case with specified token count."
    )
    parser.add_argument(
        "--lambda-val",
        type=float,
        default=0.0,
        help="Lambda value for router logit biasing (range [0, 1])"
    )
    parser.add_argument(
        "--expert-correlation-csv",
        type=str,
        default=None,
        help="Path to CSV containing layer correlation multipliers."
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        default=False,
        help="Calculate perplexity instead of generating text"
    )

    parser.add_argument(
        "--generation-perplexity",
        action="store_true",
        default=False,
        help="Calculate generation-time perplexity (slower, token-by-token)"
    )


    parser.add_argument(
        "--benchmark-prompts",
        type=str,
        default=None,
        help="Path to a file containing prompts for benchmarking. If set, runs benchmark mode."
    )
    
    parser.add_argument(
        "--predictor-device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu", "auto"],
        help=(
            "Device for expert predictor TorchScript inference (predict backend only). "
            "'gpu' = same GPU as MoE (default — validates GPU-based speculative-loading speedup), "
            "'cpu' = CPU, routes through Ryzen AI NPU on Strix, "
            "'auto' = inferred from heterogeneity config."
        )
    )
    
    parser.add_argument(
        "--predict-layers",
        type=int,
        nargs="+",
        default=None,
        help="List of layer indices (e.g., 0 1 2) to enable the predictor. If not specified, runs on all layers."
    )
    
    parser.add_argument(
        "--prefetch-experts-count",
        type=int,
        default=1,
        help="Number of top experts for the predictor engine to proactively prefetch."
    )
    
    parser.add_argument(
        "--sweep-prompts-file",
        type=str,
        default=None,
        help="Path to a JSON file containing a list of prompts. Runs all prompts sequentially without reloading."
    )
    
    args = parser.parse_args()

    if args.prompt_test is not None:
        return run_prompt_test(
            args.prompt_test,
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            generate=args.generate,
            config_path=args.config_path
        )

    print("=" * 60)
    print("Initializing Mixtral 8x7B v0.1 AWQ w4a16 quantized model...")
    print("=" * 60)

    try:
        model = Mixtral8x7BW4A16Model(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path,
            max_cached_experts_per_layer=args.expert_cache,
            predictor_models_dir=args.predictor_model,
            predictor_device=args.predictor_device,
            prefetch_experts_count=args.prefetch_experts_count,
            predict_layers=args.predict_layers,
        )

        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure you have:")
        print("  1. Built the C++ backend (run 'make' in build directory)")
        print("  2. The unified_llm_w4a16_{predict,base}_libtorch module is in your Python path")
        print("  3. Model weights are loaded (if required)")
        return 1

    # Set lambda if specified
    if args.lambda_val != 0.0:
        print(f"Setting lambda to {args.lambda_val}")
        model.set_lambda(args.lambda_val)

    if args.expert_correlation_csv:
        print(f"Loading correlations from {args.expert_correlation_csv}")
        import pandas as pd
        df = pd.read_csv(args.expert_correlation_csv)
        df = df.sort_values(by="layer")
        correlations = df['correlation'].tolist()
        model.set_layer_correlation_constants(correlations)

    # Benchmark Mode
    if args.benchmark_prompts:
        print(f"\nRunning benchmark using prompts from: {args.benchmark_prompts}")
        if not os.path.exists(args.benchmark_prompts):
            print(f"Error: Prompts file {args.benchmark_prompts} not found.")
            return 1
            
        with open(args.benchmark_prompts, 'r') as f:
            content = f.read()
            
        # Parse prompts (simple splitting by newline or custom separator if needed)
        # Using the logic from previous script attempt:
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        prompts = []
        current_prompt = ""
        for line in lines:
            if line.startswith("<|begin_of_text|>"):
                 if current_prompt: prompts.append(current_prompt)
                 current_prompt = line.replace("<|begin_of_text|>", "")
            else:
                 current_prompt += " " + line
                 
            if len(current_prompt) > 200: 
                 prompts.append(current_prompt)
                 current_prompt = ""
        if current_prompt: prompts.append(current_prompt)
        
        # Limit to 20 prompts for reasonable runtime
        prompts = [p for p in prompts if len(p) > 20][:20]
        print(f"Loaded {len(prompts)} prompts for benchmarking.")
        
        tps_values = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing Prompt {i+1}/{len(prompts)}...")
            try:
                # Reset stats
                model.reset_cache_stats()
                
                # Tokenize
                input_ids = model.tokenize(prompt)
                
                # Warmup / Forward pass measure
                start_time = time.time()
                
                # Generate
                generated = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                # Calculate TPS (generation only, usually)
                # But here elapsed includes prefill. 
                # Ideally we want generation TPS.
                # The generate() function returns, we assume prefill dominates short prompts?
                # Actually for long prompts prefill dominates.
                # User asked for "tokens per second".
                
                # Let's count generated tokens
                num_generated = generated.size(1) - input_ids.size(1)
                
                # We should subtract prefill time? exact prefill time is harder to get from wrappers 
                # unless we instrument generate().
                # But simple Total Time / Generated Tokens is "End-to-End TPS"
                
                # However, usually benchmarks exclude prefill.
                # For now, let's use Total Time / Tokens and note it. 
                # Or better: check if model has internal TPS tracking.
                # The C++ backend prints "Average Time per Token" which is generation only.
                # We can capture that from stdout if we were capturing it, but here we are IN python.
                
                # Let's rely on wall clock for now as a rough metric, or look for C++ output.
                # But wait, we want to return the metric to the user.
                
                if num_generated > 0 and elapsed > 0:
                    tps = num_generated / elapsed
                    tps_values.append(tps)
                    print(f"  Generated {num_generated} tokens in {elapsed:.4f}s")
                    print(f"  End-to-End TPS: {tps:.2f}")
                
            except Exception as e:
                print(f"  Error on prompt {i+1}: {e}")
                
        if tps_values:
            avg_tps = sum(tps_values) / len(tps_values)
            import statistics
            std_tps = statistics.stdev(tps_values) if len(tps_values) > 1 else 0
            print(f"\nBenchmark Complete.")
            print(f"Average TPS: {avg_tps:.2f} +/- {std_tps:.2f}")
            # Identify special output for parsing
            print(f"BENCHMARK_RESULT_TPS: {avg_tps:.4f}")
            print(f"BENCHMARK_RESULT_STD: {std_tps:.4f}")
        else:
            print("No valid benchmark results.")
            
        return 0

    if args.sweep_prompts_file:
        print(f"\nRunning sweep using prompts from JSON: {args.sweep_prompts_file}")
        if not os.path.exists(args.sweep_prompts_file):
            print(f"Error: Prompts file {args.sweep_prompts_file} not found.")
            return 1
            
        with open(args.sweep_prompts_file, 'r') as f:
            prompts = json.load(f)
            
        print(f"Loaded {len(prompts)} prompts for sweeping.")
        
        total_ppl = 0.0
        valid_ppl_count = 0
        total_time = 0.0
        total_generated_tokens = 0
        
        # Reset stats globally before starting sweep
        model.reset_cache_stats()

        for i, prompt in enumerate(prompts):
            print(f"\nProcessing Prompt {i+1}/{len(prompts)}...")
            try:
                if args.generation_perplexity:
                    gppl = model.calculate_generation_perplexity(prompt)
                    total_ppl += gppl
                    valid_ppl_count += 1
                
                if args.generate:
                    input_ids = model.tokenize(prompt)
                    start_time = time.time()
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k
                    )
                    end_time = time.time()
                    elapsed = end_time - start_time
                    num_generated = generated.size(1) - input_ids.size(1)
                    
                    if num_generated > 0:
                        total_time += elapsed
                        total_generated_tokens += num_generated

            except Exception as e:
                print(f"  Error on prompt {i+1}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 60}")
        print("Sweep Complete.")
        
        if args.generation_perplexity and valid_ppl_count > 0:
            avg_ppl = total_ppl / valid_ppl_count
            print(f"Generation Perplexity: {avg_ppl:.4f}")
            
        if args.generate and total_generated_tokens > 0:
            avg_tps = total_generated_tokens / total_time
            print(f"Average Time per Token: {1.0 / avg_tps:.6f}") # Output inverse since parser expects time per token
            print(f"End-to-End TPS: {avg_tps:.4f}")

        # Get and print cache stats for entire sweep
        hits, misses = model.get_cache_stats()
        total = hits + misses
        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        print(f"Cache Stats: Hits={hits}, Misses={misses}, HitRate={hit_rate:.2f}%")
        
        model.print_cache_stats()
        return 0

    # Normal execution path (Single Text)
    print(f"Processing text: '{args.text}'")

    if args.generation_perplexity:
        print("Calculating generation-time perplexity (using C++ backend loop)...")
        # Reset stats before PPL so we capture only PPL phases
        model.reset_cache_stats()
        
        gppl = model.calculate_generation_perplexity(args.text)
             
        print(f"Generation Perplexity: {gppl:.4f}")
        
        # Get and print cache stats
        hits, misses = model.get_cache_stats()
        total = hits + misses
        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        print(f"Cache Stats: Hits={hits}, Misses={misses}, HitRate={hit_rate:.2f}%")
        
        model.print_cache_stats()

    if args.generate:
        print(f"Generating {args.max_new_tokens} tokens...\n")
        # Reset stats before generation so we only capture generation stats
        model.reset_cache_stats()
        try:
            input_ids = model.tokenize(args.text)

            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )

            if model.tokenizer is not None:
                decoded_full = model.tokenizer.decode(generated[0].tolist(), skip_special_tokens=False)

                prompt_len = input_ids.size(1)
                generated_tokens = generated[0, prompt_len:].tolist()
                decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)

                num_generated_tokens = len(generated_tokens)

                print(f"\n{'='*60}")
                print("Full output (prompt + generated):")
                print(f"{'='*60}")
                print(decoded_full)
                print(f"{'='*60}")
                print("Generated text only:")
                print(f"{'='*60}")
                print(decoded_generated)
                print(f"{'='*60}")


                # Get and print cache stats
                hits, misses = model.get_cache_stats()
                total = hits + misses
                hit_rate = (hits / total * 100.0) if total > 0 else 0.0
                print(f"Cache Stats: Hits={hits}, Misses={misses}, HitRate={hit_rate:.2f}%")
                
                model.print_cache_stats()
            else:
                print(f"\nGenerated token IDs: {generated}")
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\nRunning forward pass (getting logits)...")
        try:
            start_time = time.time()
            logits = model(args.text)
            end_time = time.time()
            print(f"Prefill time: {end_time - start_time:.4f} seconds")

            print(f"Logits shape: {logits.shape}")
            print(f"Logits dtype: {logits.dtype}")
            print(f"Logits device: {logits.device}")

            print("\nLogits statistics:")
            print(f"  Min: {logits.min().item():.4f}")
            print(f"  Max: {logits.max().item():.4f}")
            print(f"  Mean: {logits.mean().item():.4f}")
            print(f"  Std: {logits.std().item():.4f}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return 1

    print(f"\n{'=' * 60}")
    if hasattr(model, "load_time"):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    print("Done!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    exit_code = main()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    exit(exit_code)
