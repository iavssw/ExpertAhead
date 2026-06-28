"""
Qwen3 30B-A3B AWQ w4a16 Python frontend.
Handles tokenization and interfaces with the C++ base backend.
"""

import os
import sys
import json
import time
import re
import math
import subprocess
from pathlib import Path
from typing import Optional, Union, List, Any
from urllib.request import urlopen

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_script_dir = Path(__file__).parent.resolve()
DEFAULT_EXPERT_WEIGHTS_DIR = str(_script_dir / "model_weights" / "Qwen3-30B-A3B-AWQ_packed")
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


def _apply_routing_and_cache_cli(model: Any, run_config: dict) -> None:
    """Apply lambda / forced routing / cache policy from config."""
    lambda_val = run_config.get("lambda_val", 0.0)
    if lambda_val != 0.0 and hasattr(model, "set_lambda"):
        print(f"Setting lambda to {lambda_val}")
        model.set_lambda(lambda_val)

    forced_top_n = run_config.get("forced_top_n", 0)
    if forced_top_n > 0 and hasattr(model, "set_forced_top_n"):
        model.set_forced_top_n(forced_top_n)
        print(f"Forced top-{forced_top_n} experts into cache mask.")

    forced_top_p = run_config.get("forced_top_p", -1.0)
    if forced_top_p >= 0.0 and hasattr(model, "set_forced_top_p"):
        model.set_forced_top_p(forced_top_p)
        print(f"Forced top-p={forced_top_p} experts into cache mask.")

    mass_p = run_config.get("mass_threshold_substitution_p", -1.0)
    if mass_p >= 0.0 and hasattr(model, "set_mass_threshold_substitution_p"):
        model.set_mass_threshold_substitution_p(mass_p)
        print(
            f"Probability-mass prefix p={mass_p} OR'd into cache mask "
            f"(same λ-biased top-k as forced_top_n; use --lambda-val e.g. 1.0 for cache-conditional routing)."
        )
        if lambda_val == 0.0:
            print("Warning: mass_threshold_substitution_p has no effect while lambda_val is 0.")

    prefill_top_n = run_config.get("prefill_top_n", 0)
    if prefill_top_n > 0 and hasattr(model, "set_prefill_top_n"):
        model.set_prefill_top_n(prefill_top_n)
        print(f"Set prefill top-{prefill_top_n} locked experts.")

    cache_policy = run_config.get("cache_policy")
    if cache_policy and hasattr(model, "set_cache_policy"):
        model.set_cache_policy(cache_policy)
        print(f"Set expert cache policy to {cache_policy}.")

    if run_config.get("suppress_predictor_stats", False):
        if hasattr(model, "set_suppress_predictor_stats"):
            model.set_suppress_predictor_stats(True)
            print("Suppressing predictor stats.")
        else:
            print("Warning: suppress_predictor_stats ignored (backend has no set_suppress_predictor_stats).")


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


QWEN3_30B_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 2048,
    "intermediate_size": 768,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_seq_len": 8192,
    "max_batch_size": 1,
    "groupsize": 128,
    "num_experts": 128,
    "num_experts_per_tok": 8,
}

class Qwen3_30BA3BW4A16Model:
    """Qwen3 30B-A3B AWQ w4a16 quantized model wrapper."""

    def __init__(
        self,
        run_config: dict,
    ):
        """
        Initialize Qwen3 30B-A3B AWQ w4a16 quantized model using a single JSON config.
        """
        
        # Extract core config with defaults
        model_path = run_config.get("model_path", "QuixiAI/Qwen3-30B-A3B-AWQ")
        tokenizer_path = run_config.get("tokenizer_path", model_path)
        device = run_config.get("device", "cuda")
        backend = run_config.get("backend", "base")
        config_path = run_config.get("config_path", None)
        expert_weights_dir = run_config.get("expert_weights_dir", DEFAULT_EXPERT_WEIGHTS_DIR)

        if backend not in ["base", "predict", "cached"]:
            raise ValueError(f"Invalid backend: {backend}. Choose from: base, predict, cached")

        if not expert_weights_dir:
            expert_weights_dir = None

        predictor_lookahead = run_config.get("predictor_lookahead", 1)
        oracle_trace_path = run_config.get("oracle_trace_path", "")
        oracle_lookahead = run_config.get("oracle_lookahead", 0)
        
        if oracle_trace_path and oracle_lookahead <= 0:
            oracle_lookahead = predictor_lookahead if predictor_lookahead > 0 else 1

        try:
            old_flags = sys.getdlopenflags()
            sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_LAZY)
            if backend == "base":
                import unified_llm_w4a16_base_libtorch as backend_module
            elif backend == "predict":
                import unified_llm_w4a16_predict_libtorch as backend_module
            elif backend == "cached":
                import unified_llm_w4a16_cached_libtorch as backend_module
            sys.setdlopenflags(old_flags)
        except ImportError as e:
            sys.setdlopenflags(old_flags)
            raise ImportError(f"Could not import {backend} backend: {e}")

        global ArchitectureType
        ArchitectureType = backend_module.ArchitectureType

        self.device = device
        self.model_path = model_path
        
        # Apply the static architecture config
        for k, v in QWEN3_30B_CONFIG.items():
            setattr(self, k, v)

        constructor_args = [
            ArchitectureType.QWEN,
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.rms_norm_eps,
            self.rope_theta,
            self.max_seq_len,
            self.max_batch_size,
            self.groupsize,
            self.num_experts,
            self.num_experts_per_tok,
        ]

        max_cached_experts_per_layer = run_config.get("max_cached_experts", 0)
        per_layer_cache_sizes = run_config.get("per_layer_cache_sizes", None)

        if backend == "cached":
            constructor_args.append(max_cached_experts_per_layer)

        if backend in ["cached", "predict"]:
            constructor_args.append(device)
        else:
            constructor_args.append(device)

        if config_path is None:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/configs_strixH_qwen3_30B_A3B.json5"))

        if backend == "predict":
            # predict backend arg order: device, max_cached, predictor_path, config, prefetch, predict_layers, per_layer_cache_sizes
            import tempfile, json as _json
            _temp_config_path = None
            
            predictor_device = run_config.get("predictor_device", "gpu")
            predictor_models_dir = run_config.get("predictor_model", "")
            expert_reuse_csv = run_config.get("expert_reuse_csv", None)
            
            if predictor_device != "auto" and predictor_models_dir:
                base_cfg = load_config_with_comments(config_path) if config_path else {}
                base_cfg["predictor_device"] = predictor_device
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                _json.dump(base_cfg, tmp)
                tmp.close()
                _temp_config_path = tmp.name
                config_to_pass = _temp_config_path
            else:
                config_to_pass = config_path
            
            per_layer_cache_sizes = run_config.get("per_layer_cache_sizes", None)
            if expert_reuse_csv and predictor_models_dir:
                per_layer_cache_sizes, per_layer_prefetch_counts = self._calibrate_from_csv(expert_reuse_csv, predictor_models_dir)
                print(f"[Calibration] Loaded per-layer counts from {expert_reuse_csv}")
            else:
                per_layer_prefetch_counts = []

            constructor_args.append(max_cached_experts_per_layer)
            constructor_args.append(predictor_models_dir)
            constructor_args.append(config_to_pass)
            constructor_args.append(run_config.get("prefetch_experts_count", 1))
            constructor_args.append(run_config.get("predict_layers", []) or [])
            constructor_args.append(per_layer_cache_sizes if per_layer_cache_sizes is not None else [])
            constructor_args.append(per_layer_prefetch_counts)
            constructor_args.append(oracle_trace_path)
            constructor_args.append(oracle_lookahead)
            constructor_args.append(run_config.get("oracle_full_union", False))
            constructor_args.append(run_config.get("prefetch_threshold", 0.0))
        else:
            constructor_args.append(config_path)

        self.model = backend_module.UnifiedLLMW4A16(*constructor_args)

        if backend == "predict" and hasattr(self.model, "set_predictor_lookahead"):
            la = oracle_lookahead if oracle_trace_path else predictor_lookahead
            if la > 0:
                self.model.set_predictor_lookahead(la)

        # Clean up temp config
        if backend == "predict" and 'tmp' in dir() and hasattr(tmp, 'name'):
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        self.config = {}
        self.use_pre_saved_weights = False
        self.debug_verbosity = 0
        self.expert_weights_dir = expert_weights_dir

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
            self._load_quantized_weights(model_path, weights_folder="model_weights",
                                         expert_weights_dir=expert_weights_dir)

        if backend in ["cached", "predict"]:
            num_to_warm = max(per_layer_cache_sizes) if per_layer_cache_sizes else max_cached_experts_per_layer
            if num_to_warm > 0:
                print(f"Pre-warming expert cache with {num_to_warm} experts...")
                self.model.prewarm_experts(num_to_warm)

        tokenizer_path = tokenizer_path or model_path
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = None

    def _calibrate_from_csv(self, csv_path: str, predictor_path: str):
        """
        Calibrate global 'fair' cache and prefetch counts from an expert reuse CSV file.
        Matches window_size in CSV with 'fN' in predictor_path.
        Formula:
          cache_size = ceil(1.2 * max(unique_experts_per_layer))
          prefetch_budget = round(avg(unique_experts_per_layer))
        """
        import csv
        import math

        # 1. Detect window size (fN) from predictor_path
        window_size = 1
        match = re.search(r"f(\d+)", predictor_path)
        if match:
            window_size = int(match.group(1))
            print(f"[Calibration] Detected window_size {window_size} from predictor path")
        else:
            print(f"[Calibration] Warning: Could not detect window_size from path '{predictor_path}', defaulting to f1")

        raw_counts = None
        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['window_size']) == window_size:
                        raw_counts = []
                        for k, v in row.items():
                            if k.startswith('layer_'):
                                raw_counts.append(float(v))
                        break
        except Exception as e:
            print(f"[Calibration] Error reading CSV {csv_path}: {e}")

        if not raw_counts:
            print(f"[Calibration] Warning: window_size {window_size} not found in CSV, using fallback defaults")
            # Default to 8 experts per layer if CSV fails
            return [8] * self.num_hidden_layers, [8] * self.num_hidden_layers

        # Formula:
        # 1. Cache Size: 1.2 * max(unique experts)
        fair_cache = int(math.ceil(1.2 * max(raw_counts)))
        # 2. Prefetch Budget: avg(unique experts)
        fair_prefetch = int(round(sum(raw_counts) / len(raw_counts)))
        
        print(f"[Calibration] Calculated Fair Metrics: Cache={fair_cache}, Prefetch={fair_prefetch}")

        # Return flat lists to satisfy the predict backend's per-layer requirement
        cache_list = [fair_cache] * self.num_hidden_layers
        prefetch_list = [fair_prefetch] * self.num_hidden_layers
        
        return cache_list, prefetch_list

    def _load_quantized_weights(self, model_path: str, weights_folder: str = "model_weights",
                               expert_weights_dir: Optional[str] = None):
        """Load quantized weights from safetensors and pass to the C++ backend.

        Args:
            model_path: HuggingFace repo ID or local path to the quantized model.
            weights_folder: Sub-folder under the script directory used to cache
                downloaded safetensors and pre-saved bins.
            expert_weights_dir: Optional path to a directory containing either
                unpacked expert bins (``layer_{L}_expert_{E}_gate.qweight.bin``
                etc.) or packed expert bins (``layer_{L}_expert_{E}.bin``).
                When supplied the C++ backend will point each MoE layer at this
                directory for on-demand expert loading, while the attention
                weights continue to be served from ``--bin-dir`` / safetensors.
                The backend auto-detects packed vs unpacked by probing for the
                packed filename.  When omitted the standard presaved-bins path
                (``{model_name}_unpacked``) is used as before.
        """
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

            if expert_weights_dir is not None:
                # Explicit expert directory supplied (packed or unpacked).
                # Detect format by probing for a packed file.
                expert_path = Path(expert_weights_dir)
                probe = expert_path / f"layer_0_expert_0.bin"
                fmt = "packed" if probe.exists() else "unpacked"
                print(f"Expert weights dir: {expert_path}  (format: {fmt})")

                # Load non-MoE weights from safetensors, then set the expert
                # directory so the C++ backend can load experts on demand.
                presaved_dir = weights_dir / f"{model_name}_unpacked"
                self._prepare_presaved_weights(saved_safetensors, presaved_dir, attention_only=True)

                t0 = time.time()
                self.model.load_non_quantized_weights_from_safetensors(str(saved_safetensors))
                # Load attention weights from the unpacked presaved dir.
                # Expert weights are served from expert_weights_dir (auto-detect packed/unpacked in C++).
                self.model.load_quantized_weights_from_bins(str(presaved_dir),
                                                            str(expert_path))
                t1 = time.time()
                self.load_time = t1 - t0
                print(f"Weights loaded (expert dir override) in {self.load_time:.2f} seconds")

            elif use_presaved:
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

    def _prepare_presaved_weights(
        self, saved_safetensors: Path, presaved_dir: Path, *, attention_only: bool = False
    ) -> None:
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

        def _attention_bins_exist() -> bool:
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
            return True

        def _bins_exist() -> bool:
            if not _attention_bins_exist():
                return False
            if attention_only:
                return True
            for layer_idx in range(self.num_hidden_layers):
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
            label = "attention" if attention_only else "pre-saved"
            if self.debug_verbosity >= 1:
                print(f"Using existing {label} weights in {presaved_dir}")
            return

        if attention_only:
            raise FileNotFoundError(
                f"Missing attention weight bins in {presaved_dir}. "
                f"When using --expert-weights-dir (packed experts), the unpacked dir must "
                f"contain layer_*_{{q,k,v,o}}.{{qweight,scales,zeros}}.bin for all layers."
            )

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
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                if expert_prefix + ".gate_proj.qweight" in state_dict:
                    gate_base = expert_prefix + ".gate_proj"
                    up_base = expert_prefix + ".up_proj"
                    down_base = expert_prefix + ".down_proj"
                else:
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
                        raise KeyError(f"Missing MoE expert weights for layer {layer_idx}, expert {expert_idx}")

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

    def set_layer_correlation_constants(self, constants: List[float]):
        """Set the correlation constant (prefill bias alpha) for each layer."""
        if hasattr(self.model, "set_layer_correlation_constants"):
            self.model.set_layer_correlation_constants(constants)

    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        if hasattr(self.model, "reset_cache_stats"):
            self.model.reset_cache_stats()

    def get_cache_stats(self):
        """Return (total_hits, total_misses) across all MoE layers."""
        if hasattr(self.model, "get_cache_stats"):
            return self.model.get_cache_stats()
        return (0, 0)

    def print_cache_stats(self):
        if hasattr(self.model, "print_cache_stats"):
            self.model.print_cache_stats()

    def get_predictor_stats(self):
        """Return per-layer (hits, total) tuples for unbiased top-1 predictor accuracy."""
        if hasattr(self.model, "get_predictor_stats"):
            return self.model.get_predictor_stats()
        return []

    def reset_predictor_stats(self):
        if hasattr(self.model, "reset_predictor_stats"):
            self.model.reset_predictor_stats()

    def set_cache_policy(self, policy: str):
        """Set the expert cache eviction policy."""
        if hasattr(self.model, "set_cache_policy"):
            self.model.set_cache_policy(policy)

    def set_lambda(self, lambda_value: float, layer_idx: int = -1):
        """Set the cache-conditional routing bias strength (λ).

        When λ > 0 the router logits are biased toward currently-cached experts,
        increasing cache hit rate without a predictor.

        Args:
            lambda_value: Bias strength in range [0, 100].
            layer_idx: Layer to target (-1 = all layers).
        """
        if lambda_value < 0.0 or lambda_value > 100.0:
            raise ValueError(f"Lambda must be in range [0, 100], got: {lambda_value}")
        if hasattr(self.model, "set_lambda"):
            self.model.set_lambda(lambda_value, layer_idx)

    def set_forced_top_n(self, n: int):
        """Set how many unbiased top-K experts are forced into the lambda bias mask."""
        if hasattr(self.model, "set_forced_top_n"):
            self.model.set_forced_top_n(n)

    def set_forced_top_p(self, p: float):
        """Force the minimum set of experts whose cumulative softmax probability >= p.

        Unlike forced_top_n (fixed count), this adapts to routing confidence: a peaked
        distribution forces fewer experts than a flat one. Set to -1.0 to disable.
        """
        if hasattr(self.model, "set_forced_top_p"):
            self.model.set_forced_top_p(p)

    def set_mass_threshold_substitution_p(self, p: float):
        """Enable probability-mass routing with tail substitution.

        Keeps the smallest top-k prefix whose cumulative router probability >= p,
        then substitutes the remaining routed slots (instead of dropping tail experts).
        Set to -1.0 to disable.
        """
        if hasattr(self.model, "set_mass_threshold_substitution_p"):
            self.model.set_mass_threshold_substitution_p(p)

    def set_prefill_top_n(self, n: int):
        """Lock the top n most used experts from prefill into the cache under PREFILL policy."""
        if hasattr(self.model, "set_prefill_top_n"):
            self.model.set_prefill_top_n(n)

    def set_random_fill_mode(self, on: bool):
        """Experiment mode: keep top forced_top_n correct experts; fill remaining with random experts."""
        if hasattr(self.model, "set_random_fill_mode"):
            self.model.set_random_fill_mode(on)

    def set_suppress_predictor_stats(self, v: bool) -> None:
        """Disable predictor/prefetch measurement (predict backend only)."""
        if hasattr(self.model, "set_suppress_predictor_stats"):
            self.model.set_suppress_predictor_stats(v)

def run_prompt_test(run_config: dict):
    """
    Run prompt test case: load single long prompt from prompts.txt,
    concatenate base prompt, truncate to requested token count, and generate output.
    """
    target_tokens = run_config.get("prompt_test", 256)
    model_path = run_config.get("model_path", "QuixiAI/Qwen3-30B-A3B-AWQ")
    tokenizer_path = run_config.get("tokenizer_path", model_path)
    device = run_config.get("device", "cuda")
    max_new_tokens = run_config.get("max_new_tokens", 512)
    temperature = run_config.get("temperature", 0.7)
    top_p = run_config.get("top_p", 0.9)
    top_k = run_config.get("top_k", 50)
    generate = run_config.get("generate", True)
    perplexity = run_config.get("perplexity", False)
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
            tokenizer_path = "QuixiAI/Qwen3-30B-A3B-AWQ"
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
        model_path = "QuixiAI/Qwen3-30B-A3B-AWQ"

    print("Initializing Qwen3 30B-A3B AWQ w4a16 quantized model...")
    try:
        model = Qwen3_30BA3BW4A16Model(run_config)
        _apply_routing_and_cache_cli(model, run_config)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    if perplexity:
        print("PERPLEXITY EVALUATION:")
    elif generate:
        print("GENERATING OUTPUT:")
    else:
        print("FORWARD PASS (NO GENERATION):")
    print("=" * 60 + "\n")

    try:
        input_ids = torch.tensor([truncated_tokens], dtype=torch.long, device=device)

        if perplexity:
            print("Running perplexity evaluation...")
            start_time = time.time()
            metrics = model.perplexity(input_ids)
            end_time = time.time()
            print(f"Eval time: {end_time - start_time:.4f} seconds")
            print(f"Tokens evaluated: {metrics['num_tokens']}")
            print(f"Cross-entropy loss: {metrics['loss']:.6f}")
            print(f"Perplexity: {metrics['perplexity']:.6f}")
            return 0

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


def _load_wikitext103_raw_text(model_weights_dir: Path, split: str = "test") -> str:
    """
    Load WikiText-103 raw split and cache the plain text under model_weights_dir.
    Tries Hugging Face datasets first.
    """
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    text_cache_path = model_weights_dir / f"wikitext-103-raw-v1_{split}.txt"

    if text_cache_path.exists():
        print(f"Using cached WikiText-103 text: {text_cache_path}")
        return text_cache_path.read_text(encoding="utf-8")

    text = None
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        lines = [line for line in ds["text"] if line and line.strip()]
        text = "\n\n".join(lines)
        print(f"Downloaded WikiText-103 via datasets ({split} split).")
    except Exception as e:
        print(f"Could not load WikiText-103 via datasets ({e}).")
        raise RuntimeError("Failed to load WikiText-103 dataset.")

    text_cache_path.write_text(text, encoding="utf-8")
    print(f"Saved WikiText-103 text cache: {text_cache_path}")
    return text


def run_wikitext103_perplexity(run_config: dict):
    """
    Evaluate perplexity on WikiText-103 with sliding-window evaluation.
    Saves fetched text and tokenized IDs under model_weights.
    """
    model_path = run_config.get("model_path", "QuixiAI/Qwen3-30B-A3B-AWQ")
    tokenizer_path = run_config.get("tokenizer_path", model_path)
    device = run_config.get("device", "cuda")
    split = run_config.get("wikitext103_split", "test")
    max_length = run_config.get("wikitext103_max_length", 2048)
    stride = run_config.get("wikitext103_stride", 2048)
    max_windows = run_config.get("wikitext103_max_windows", 0)

    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    script_dir = Path(__file__).parent
    model_weights_dir = script_dir / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"WIKITEXT-103 PERPLEXITY ({split} split)")
    print("=" * 60 + "\n")

    text = _load_wikitext103_raw_text(model_weights_dir, split=split)

    print("Initializing Qwen3 30B-A3B AWQ w4a16 quantized model...")
    try:
        model = Qwen3_30BA3BW4A16Model(run_config)
        _apply_routing_and_cache_cli(model, run_config)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if model.tokenizer is None:
        print("Error: tokenizer is required for WikiText-2 perplexity.")
        return 1

    print("Tokenizing WikiText-2 corpus...")
    encoded = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids_full = encoded["input_ids"]
    if input_ids_full.size(1) < 2:
        print("Error: tokenized WikiText-2 corpus is too short.")
        return 1

    token_cache_path = model_weights_dir / f"wikitext-103-raw-v1_{split}_tokens.pt"
    torch.save(input_ids_full.cpu(), token_cache_path)
    print(f"Saved tokenized WikiText-2 tensor: {token_cache_path}")
    print(f"Total tokens: {input_ids_full.size(1)}")
    print(f"Eval max_length: {max_length}, stride: {stride}")
    if max_windows > 0:
        print(f"Quick test: evaluating at most {max_windows} sliding window(s).")
    backend_prefill_chunk = None
    if hasattr(model, "model") and hasattr(model.model, "get_prefill_chunk_size"):
        try:
            backend_prefill_chunk = int(model.model.get_prefill_chunk_size())
        except Exception:
            backend_prefill_chunk = None
    if backend_prefill_chunk is None or backend_prefill_chunk <= 0:
        backend_prefill_chunk = min(int(getattr(model, "max_seq_len", 4096)), 4096)
    forward_chunk_size = max(1, min(backend_prefill_chunk, max_length))
    print(f"Internal forward chunk size: {forward_chunk_size}")

    total_nll = 0.0
    total_tokens = 0
    prev_end_loc = 0
    seq_len = input_ids_full.size(1)
    start_time = time.time()
    window_starts = list(range(0, seq_len, stride))
    total_windows = len(window_starts)

    for window_idx, begin_loc in enumerate(window_starts):
        if max_windows > 0 and window_idx >= max_windows:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids_window = input_ids_full[:, begin_loc:end_loc]
        window_len = input_ids_window.size(1)
        input_ids_window_dev = input_ids_window.to(device)
        tokens_to_ignore = max(0, (window_len - 1) - trg_len)
        window_loss_sum = 0.0
        window_valid_tokens = 0

        try:
            with torch.no_grad():
                if window_len <= forward_chunk_size:
                    chunk_ranges = [(0, window_len)]
                else:
                    chunk_ranges = [(i, min(i + forward_chunk_size, window_len)) for i in range(0, window_len, forward_chunk_size)]

                for chunk_begin, chunk_end in chunk_ranges:
                    chunk_input = input_ids_window_dev[:, chunk_begin:chunk_end]
                    if chunk_begin == 0:
                        chunk_logits = model(chunk_input)
                    else:
                        chunk_logits = model(chunk_input, start_pos=chunk_begin)

                    target_begin = chunk_begin + 1
                    target_end = min(chunk_end + 1, window_len)
                    if target_begin >= target_end:
                        continue

                    labels = input_ids_window_dev[:, target_begin:target_end].to(chunk_logits.device).contiguous()
                    chunk_token_count = labels.size(1)
                    logits_for_loss = chunk_logits[:, :chunk_token_count, :].float().contiguous()
                    vocab_size = logits_for_loss.size(-1)

                    ignore_prefix = max(0, tokens_to_ignore - chunk_begin)
                    if ignore_prefix >= chunk_token_count:
                        continue
                    if ignore_prefix > 0:
                        labels = labels.clone()
                        labels[:, :ignore_prefix] = -100

                    loss_sum = F.cross_entropy(
                        logits_for_loss.view(-1, vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    valid_tokens_chunk = int((labels != -100).sum().item())
                    window_loss_sum += float(loss_sum.item())
                    window_valid_tokens += valid_tokens_chunk
        except Exception as e:
            print(
                f"Error during forward pass at window begin={begin_loc}, end={end_loc}, "
                f"window_len={end_loc - begin_loc}, stride={stride}: {e}"
            )
            return 1

        total_nll += window_loss_sum
        total_tokens += window_valid_tokens

        progress_ratio = float(window_idx + 1) / float(total_windows)
        bar_width = 28
        filled = int(progress_ratio * bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = time.time() - start_time
        running_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
        print(
            f"\rProgress [{bar}] {window_idx + 1}/{total_windows} "
            f"({progress_ratio * 100.0:5.1f}%) | begin={begin_loc}, end={end_loc}, "
            f"tokens_evaluated={total_tokens}, running_ppl={running_ppl:.4f}, elapsed={elapsed:.1f}s",
            end="",
            flush=True,
        )

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    print()

    if total_tokens <= 0:
        print("Error: no valid tokens were evaluated.")
        return 1

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    end_time = time.time()

    print("\n" + "=" * 60)
    print("WIKITEXT-2 PERPLEXITY RESULT")
    print("=" * 60)
    print(f"Split: {split}")
    print(f"Tokens evaluated: {total_tokens}")
    print(f"Average NLL (loss): {avg_nll:.6f}")
    print(f"Perplexity: {ppl:.6f}")
    print(f"Eval time: {end_time - start_time:.2f} seconds")
    print("=" * 60 + "\n")
    return 0


def main():
    """Example usage of Qwen3_30BA3BW4A16Model when run as a script."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3 30B-A3B AWQ W4A16 Quantized Model")
    parser.add_argument(
        "--run-config",
        type=str,
        required=True,
        help="Path to JSON run configuration file"
    )
    cli_args = parser.parse_args()
    
    with open(cli_args.run_config, 'r') as f:
        run_config = json.load(f)
        
    class Args:
        pass
    args = Args()
    for k, v in run_config.items():
        setattr(args, k, v)
        
    args.text = getattr(args, "text", "What is the meaning of life the universe and everything?")
    args.tokenizer_path = getattr(args, "tokenizer_path", None)
    args.model_path = getattr(args, "model_path", "QuixiAI/Qwen3-30B-A3B-AWQ")
    args.device = getattr(args, "device", "cuda")
    args.backend = getattr(args, "backend", "base")
    args.config_path = getattr(args, "config_path", None)
    args.generate = getattr(args, "generate", True)
    args.max_new_tokens = getattr(args, "max_new_tokens", 16)
    args.temperature = getattr(args, "temperature", 0.0)
    args.top_p = getattr(args, "top_p", 0.9)
    args.top_k = getattr(args, "top_k", 50)
    args.prompt_test = getattr(args, "prompt_test", None)
    args.max_cached_experts = getattr(args, "max_cached_experts", 8)
    args.prefetch_experts_count = getattr(args, "prefetch_experts_count", 1)
    args.prefetch_threshold = getattr(args, "prefetch_threshold", 0.0)
    args.perplexity = getattr(args, "perplexity", False)
    args.wikitext103_perplexity = getattr(args, "wikitext103_perplexity", False)
    args.wikitext103_split = getattr(args, "wikitext103_split", "test")
    args.wikitext103_max_length = getattr(args, "wikitext103_max_length", 2048)
    args.wikitext103_stride = getattr(args, "wikitext103_stride", 2048)
    args.wikitext103_max_windows = getattr(args, "wikitext103_max_windows", 0)
    args.sweep_prompts_file = getattr(args, "sweep_prompts_file", None)
    args.generation_perplexity = getattr(args, "generation_perplexity", False)
    args.lambda_val = getattr(args, "lambda_val", 0.0)
    args.predict_layers = getattr(args, "predict_layers", None)
    args.predictor_model = getattr(args, "predictor_model", "")
    args.predictor_device = getattr(args, "predictor_device", "gpu")
    args.predictor_lookahead = getattr(args, "predictor_lookahead", 1)
    args.expert_reuse_csv = getattr(args, "expert_reuse_csv", None)
    args.forced_top_n = getattr(args, "forced_top_n", 0)
    args.forced_top_p = getattr(args, "forced_top_p", -1.0)
    args.mass_threshold_substitution_p = getattr(args, "mass_threshold_substitution_p", -1.0)
    args.expert_weights_dir = getattr(args, "expert_weights_dir", DEFAULT_EXPERT_WEIGHTS_DIR)
    args.cache_policy = getattr(args, "cache_policy", "LRU")
    args.prefill_top_n = getattr(args, "prefill_top_n", 0)
    args.suppress_predictor_stats = getattr(args, "suppress_predictor_stats", False)
    args.oracle_trace = getattr(args, "oracle_trace", "")
    args.oracle_lookahead = getattr(args, "oracle_lookahead", 0)
    args.oracle_full_union = getattr(args, "oracle_full_union", False)
    args.capture_oracle_trace = getattr(args, "capture_oracle_trace", "")
    args.oracle_prompt_token_ids = None

    if args.wikitext103_perplexity:
        return run_wikitext103_perplexity(run_config)

    if args.prompt_test is not None:
        return run_prompt_test(run_config)

    print("=" * 60)
    print("Initializing Qwen3 30B-A3B AWQ w4a16 quantized model...")
    print("=" * 60)

    try:
        model = Qwen3_30BA3BW4A16Model(run_config)

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

    _apply_routing_and_cache_cli(model, run_config)

    if args.capture_oracle_trace:
        oracle_bundle = None
        if args.oracle_trace and os.path.exists(args.oracle_trace):
            utils_dir = _script_dir.parent / "utils"
            if str(utils_dir) not in sys.path:
                sys.path.insert(0, str(utils_dir))
            from oracle_trace import load_oracle_trace

            oracle_bundle = load_oracle_trace(args.oracle_trace)
            if oracle_bundle.prompt_token_ids:
                input_ids = torch.tensor([oracle_bundle.prompt_token_ids], dtype=torch.long, device=args.device)
                print(f"Capture prompt: {len(oracle_bundle.prompt_token_ids)} token IDs from {args.oracle_trace}")
            elif oracle_bundle.prompt_text:
                args.text = oracle_bundle.prompt_text
                input_ids = model.tokenize(args.text)
                print(f"Capture prompt: tokenized text from {args.oracle_trace} ({input_ids.size(1)} tokens)")
            else:
                input_ids = model.tokenize(args.text)
        else:
            input_ids = model.tokenize(args.text)

        if not hasattr(model.model, "begin_oracle_trace_capture"):
            print("Error: backend lacks oracle trace capture (rebuild predict libtorch).")
            return 1

        print(f"Capturing oracle trace -> {args.capture_oracle_trace}")
        model.model.begin_oracle_trace_capture()
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        prompt_len = input_ids.size(1)
        gen_ids = output[0, prompt_len:].tolist()
        prompt_text = args.text
        if oracle_bundle and oracle_bundle.prompt_text:
            prompt_text = oracle_bundle.prompt_text
        generated_text = ""
        if model.tokenizer is not None:
            generated_text = model.tokenizer.decode(gen_ids, skip_special_tokens=False)
        ok = model.model.write_oracle_trace_file(
            args.capture_oracle_trace,
            input_ids,
            output,
            prompt_text,
            generated_text,
            "qwen3_30b",
        )
        model.model.cancel_oracle_trace_capture()
        return 0 if ok else 1

    if args.sweep_prompts_file:
        print(f"\nRunning sweep using prompts from JSON: {args.sweep_prompts_file}")
        if not os.path.exists(args.sweep_prompts_file):
            print(f"Error: Prompts file {args.sweep_prompts_file} not found.")
            return 1

        with open(args.sweep_prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        print(f"Loaded {len(prompts)} prompts for sweeping.")
        total_nll_sum = 0.0
        total_gen_toks_for_ppl = 0
        prompts_with_gen = 0
        total_time = 0.0
        total_generated_tokens = 0
        prompt_tps_list = []
        prompt_ppl_list = []
        
        model.reset_cache_stats()

        for i, prompt in enumerate(prompts):
            print(f"\nProcessing Prompt {i+1}/{len(prompts)}...")
            try:
                if isinstance(prompt, dict):
                    if "token_ids" in prompt and prompt["token_ids"]:
                        input_ids = torch.tensor([prompt["token_ids"]], dtype=torch.long, device=args.device)
                    else:
                        input_ids = model.tokenize(prompt.get("text", ""))
                else:
                    input_ids = model.tokenize(prompt)

                if args.generation_perplexity:
                    start_time = time.time()
                    full_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    elapsed = time.time() - start_time
                    num_generated = full_ids.size(1) - input_ids.size(1)
                    if num_generated > 0:
                        gp = model.generation_perplexity(input_ids, full_ids)
                        total_nll_sum += gp["sum_nll"]
                        total_gen_toks_for_ppl += gp["num_gen_tokens"]
                        prompts_with_gen += 1
                        prompt_ppl = math.exp(gp["sum_nll"] / gp["num_gen_tokens"])
                        prompt_ppl_list.append(prompt_ppl)
                        if args.generate:
                            total_time += elapsed
                            total_generated_tokens += num_generated
                            prompt_tps_list.append(num_generated / elapsed)
                            
                            if model.tokenizer is not None:
                                prompt_len = input_ids.size(1)
                                generated_tokens = full_ids[0, prompt_len:].tolist()
                                decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                                print(f"\n{'='*60}")
                                print("Generated text only:")
                                print(f"{'='*60}")
                                print(decoded_generated)
                                print(f"{'='*60}")
                elif args.generate:
                    start_time = time.time()
                    generated = model.generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                    )
                    elapsed = time.time() - start_time
                    num_generated = generated.size(1) - input_ids.size(1)
                    if num_generated > 0:
                        total_time += elapsed
                        total_generated_tokens += num_generated
                        prompt_tps_list.append(num_generated / elapsed)
                        
                        if model.tokenizer is not None:
                            prompt_len = input_ids.size(1)
                            generated_tokens = generated[0, prompt_len:].tolist()
                            decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                            print(f"\n{'='*60}")
                            print("Generated text only:")
                            print(f"{'='*60}")
                            print(decoded_generated)
                            print(f"{'='*60}")

            except Exception as e:
                print(f"  Error on prompt {i+1}: {e}")

        print("\n" + "=" * 60)
        print("Sweep Complete.")
        
        if args.generation_perplexity and total_gen_toks_for_ppl > 0:
            avg_nll = total_nll_sum / total_gen_toks_for_ppl
            avg_ppl = math.exp(avg_nll)
            print(f"Generation Perplexity: {avg_ppl:.4f}")
            print(
                f"(token-weighted over {total_gen_toks_for_ppl} generated tokens, "
                f"{prompts_with_gen}/{len(prompts)} prompts with ≥1 new token)"
            )
            if len(prompt_ppl_list) > 1:
                import statistics
                ppl_std = statistics.stdev(prompt_ppl_list)
                print(f"Generation Perplexity StdDev: {ppl_std:.4f}")
            
        if args.generate and total_generated_tokens > 0:
            avg_tps = total_generated_tokens / total_time
            print(f"Average Time per Token: {1.0 / avg_tps:.6f}") # Output inverse since parser expects time per token
            print(f"End-to-End TPS: {avg_tps:.4f}")
            if len(prompt_tps_list) > 1:
                import statistics
                tps_std = statistics.stdev(prompt_tps_list)
                print(f"TPS StdDev: {tps_std:.4f}")

        # Get and print cache stats for entire sweep
        hits, misses = model.get_cache_stats()
        total = hits + misses
        hit_rate = (hits / total * 100.0) if total > 0 else 0.0
        print(f"Cache Stats: Hits={hits}, Misses={misses}, HitRate={hit_rate:.2f}%")
        
        pred_stats = model.get_predictor_stats()
        if pred_stats:
            pred_hits = sum(s[0] for s in pred_stats)
            pred_total = sum(s[1] for s in pred_stats)
            pred_rate = (pred_hits / pred_total * 100.0) if pred_total > 0 else 0.0
            print(f"Predictor Stats: Hits={pred_hits}, Total={pred_total}, HitRate={pred_rate:.2f}%")

        if hasattr(model, "print_cache_stats"):
            model.print_cache_stats()
        return 0

    if args.oracle_trace and os.path.exists(args.oracle_trace):
        try:
            utils_dir = _script_dir.parent / "utils"
            if str(utils_dir) not in sys.path:
                sys.path.insert(0, str(utils_dir))
            from oracle_trace import load_oracle_trace

            oracle_bundle = load_oracle_trace(args.oracle_trace)
            if oracle_bundle.prompt_token_ids:
                args.oracle_prompt_token_ids = oracle_bundle.prompt_token_ids
                print(
                    f"Loaded {len(oracle_bundle.prompt_token_ids)} prompt token IDs from oracle trace "
                    f"({args.oracle_trace})"
                )
            elif oracle_bundle.prompt_text:
                args.text = oracle_bundle.prompt_text.strip()
                print(f"Loaded prompt text from oracle trace ({len(args.text)} chars, no token IDs — re-capture recommended)")
        except Exception as e:
            print(f"Warning: Failed to load prompt from oracle trace: {e}")

    if getattr(args, "oracle_prompt_token_ids", None):
        print("Replay uses exact PROMPT TOKEN IDS from oracle trace.")
    else:
        print(f"Processing text: '{args.text}'")

    if args.perplexity:
        print("\nRunning perplexity evaluation...")
        try:
            input_ids = model.tokenize(args.text)
            start_time = time.time()
            metrics = model.perplexity(input_ids)
            end_time = time.time()
            print(f"Eval time: {end_time - start_time:.4f} seconds")
            print(f"Tokens evaluated: {metrics['num_tokens']}")
            print(f"Cross-entropy loss: {metrics['loss']:.6f}")
            print(f"Perplexity: {metrics['perplexity']:.6f}")
            return 0
        except Exception as e:
            print(f"Error during perplexity evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.generate:
        print(f"Generating {args.max_new_tokens} tokens...\n")
        try:
            if getattr(args, "oracle_prompt_token_ids", None):
                input_ids = torch.tensor([args.oracle_prompt_token_ids], dtype=torch.long, device=args.device)
            else:
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

                print(f"\n{'='*60}")
                print("Full output (prompt + generated):")
                print(f"{'='*60}")
                print(decoded_full)
                print(f"{'='*60}")
                print("Generated text only:")
                print(f"{'='*60}")
                print(decoded_generated)
                print(f"{'='*60}")
            else:
                print(f"\nGenerated token IDs: {generated}")
            if hasattr(model, "print_cache_stats"):
                model.print_cache_stats()
            return 0
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    exit(main())

    # WikiText-103 (full): --wikitext103-perplexity --wikitext103-split test --wikitext103-max-length 4096 --wikitext103-stride 2048
    # Examples of limited runs:
    #   --wikitext103-perplexity --wikitext103-max-windows 2 --wikitext103-max-length 512 --wikitext103-stride 512
