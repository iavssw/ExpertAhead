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


def _apply_routing_and_cache_cli(model: Any, args: Any) -> None:
    """Apply lambda / forced routing / cache policy from argparse (shared by main and WikiText eval)."""
    if args.lambda_val != 0.0 and hasattr(model, "set_lambda"):
        print(f"Setting lambda to {args.lambda_val}")
        model.set_lambda(args.lambda_val)

    if args.forced_top_n > 0 and hasattr(model, "set_forced_top_n"):
        model.set_forced_top_n(args.forced_top_n)
        print(f"Forced top-{args.forced_top_n} experts into cache mask.")

    if args.forced_top_p >= 0.0 and hasattr(model, "set_forced_top_p"):
        model.set_forced_top_p(args.forced_top_p)
        print(f"Forced top-p={args.forced_top_p} experts into cache mask.")

    if args.mass_threshold_substitution_p >= 0.0 and hasattr(model, "set_mass_threshold_substitution_p"):
        model.set_mass_threshold_substitution_p(args.mass_threshold_substitution_p)
        print(
            f"Probability-mass prefix p={args.mass_threshold_substitution_p} OR'd into cache mask "
            f"(same λ-biased top-k as forced_top_n; use --lambda-val e.g. 1.0 for cache-conditional routing)."
        )
        if args.lambda_val == 0.0:
            print("Warning: --mass-threshold-substitution-p has no effect while --lambda-val is 0.")

    if args.prefill_top_n > 0 and hasattr(model, "set_prefill_top_n"):
        model.set_prefill_top_n(args.prefill_top_n)
        print(f"Set prefill top-{args.prefill_top_n} locked experts.")

    if hasattr(args, "cache_policy") and hasattr(model, "set_cache_policy"):
        model.set_cache_policy(args.cache_policy)
        print(f"Set expert cache policy to {args.cache_policy}.")


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


class Qwen3_30BA3BW4A16Model:
    """Qwen3 30B-A3B AWQ w4a16 quantized model wrapper."""

    def __init__(
        self,
        model_path: Optional[str] = "QuixiAI/Qwen3-30B-A3B-AWQ",
        tokenizer_path: Optional[str] = None,
        vocab_size: int = 151936,
        hidden_size: int = 2048,
        intermediate_size: int = 768,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        max_seq_len: int = 8192,
        max_batch_size: int = 1,
        groupsize: int = 128,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        device: str = "cuda",
        backend: str = "base",
        max_cached_experts_per_layer: int = 0,
        config_path: Optional[str] = None,
        predictor_models_dir: str = "",
        predictor_device: str = "gpu",
        prefetch_experts_count: int = 1,
        predict_layers: Optional[List[int]] = None,
        per_layer_cache_sizes: Optional[List[int]] = None,
        expert_reuse_csv: Optional[str] = None,
        predictor_lookahead: int = 1,
        expert_weights_dir: Optional[str] = None,
    ):
        """
        Initialize Qwen3 30B-A3B AWQ w4a16 quantized model.
        """

        if backend not in ["base", "predict", "cached"]:
            raise ValueError(f"Invalid backend: {backend}. Choose from: base, predict, cached")

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
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.groupsize = groupsize
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        constructor_args = [
            ArchitectureType.QWEN,
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
            
            if expert_reuse_csv and predictor_models_dir:
                per_layer_cache_sizes, per_layer_prefetch_counts = self._calibrate_from_csv(expert_reuse_csv, predictor_models_dir)
                print(f"[Calibration] Loaded per-layer counts from {expert_reuse_csv}")
            else:
                per_layer_prefetch_counts = []

            constructor_args.append(max_cached_experts_per_layer)
            constructor_args.append(predictor_models_dir)
            constructor_args.append(config_to_pass)
            constructor_args.append(prefetch_experts_count)
            constructor_args.append(predict_layers if predict_layers is not None else [])
            constructor_args.append(per_layer_cache_sizes if per_layer_cache_sizes is not None else [])
            constructor_args.append(per_layer_prefetch_counts)
        else:
            constructor_args.append(config_path)

        self.model = backend_module.UnifiedLLMW4A16(*constructor_args)

        if backend == "predict" and predictor_lookahead > 1 and hasattr(self.model, "set_predictor_lookahead"):
            self.model.set_predictor_lookahead(predictor_lookahead)

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
                self._prepare_presaved_weights(saved_safetensors, presaved_dir)

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
    def perplexity(self, input_ids: Union[str, torch.Tensor]) -> dict:
        """
        Compute causal-LM perplexity for the provided sequence(s).
        Returns: loss, perplexity, num_tokens.
        """
        if isinstance(input_ids, str):
            input_ids = self.tokenize(input_ids)

        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids with shape [batch, seq_len], got {tuple(input_ids.shape)}")
        if input_ids.size(1) < 2:
            raise ValueError("Need at least 2 tokens to compute perplexity.")

        with torch.no_grad():
            logits = self.model.forward(input_ids, 0)

        shift_logits = logits[:, :-1, :].float().contiguous()
        shift_labels = input_ids[:, 1:].to(shift_logits.device).contiguous()
        vocab_size = shift_logits.size(-1)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        if pad_token_id is not None:
            valid_mask = shift_labels.ne(pad_token_id)
            num_tokens = int(valid_mask.sum().item())
            if num_tokens == 0:
                raise ValueError("No non-pad tokens available for perplexity computation.")
            labels_for_loss = shift_labels.masked_fill(~valid_mask, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                labels_for_loss.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        else:
            num_tokens = int(shift_labels.numel())
            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="mean",
            )

        ppl = torch.exp(loss)
        return {
            "loss": float(loss.item()),
            "perplexity": float(ppl.item()),
            "num_tokens": num_tokens,
        }

    def generation_perplexity(self, prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> dict:
        """Perplexity of the generated continuation only (not the prompt).

        For each generated position t in [P, T-1], computes -log p(full_ids[t] | full_ids[:t])
        using a single teacher-forced forward on ``full_ids`` (same routing as standard ``forward``).

        Pool multiple prompts with token weighting: sum_nll / sum(num_gen_tokens), then exp.

        Returns keys: perplexity, mean_nll, sum_nll, num_gen_tokens.
        """
        if prompt_ids.dim() != 2 or full_ids.dim() != 2:
            raise ValueError("Expected prompt_ids and full_ids shaped [batch, seq].")
        if prompt_ids.size(0) != 1 or full_ids.size(0) != 1:
            raise ValueError("generation_perplexity currently supports batch size 1.")
        P = int(prompt_ids.size(1))
        T = int(full_ids.size(1))
        if T <= P:
            return {
                "perplexity": float("inf"),
                "mean_nll": float("inf"),
                "sum_nll": 0.0,
                "num_gen_tokens": 0,
            }

        with torch.no_grad():
            logits = self.model.forward(full_ids, 0)

        logits = logits[:, :-1, :].float().contiguous()
        vocab_size = logits.size(-1)
        # logits[:, i, :] predicts token at index i+1
        gen_logits = logits[0, P - 1 : T - 1, :]
        gen_targets = full_ids[0, P:T].to(logits.device).long()
        loss_vec = F.cross_entropy(gen_logits, gen_targets, reduction="none")
        sum_nll = float(loss_vec.sum().item())
        n_gen = int(gen_targets.numel())
        mean_nll = sum_nll / max(n_gen, 1)
        return {
            "perplexity": float(math.exp(mean_nll)),
            "mean_nll": float(mean_nll),
            "sum_nll": sum_nll,
            "num_gen_tokens": n_gen,
        }


def run_prompt_test(target_tokens, model_path=None, tokenizer_path=None, device="cuda", backend="base",
                    max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50,
                    generate=True, perplexity=False, config_path=None):
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
        model = Qwen3_30BA3BW4A16Model(
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


def _load_wikitext2_raw_text(model_weights_dir: Path, split: str = "test") -> str:
    """
    Load WikiText-2 raw split and cache the plain text under model_weights_dir.
    Tries Hugging Face datasets first, then falls back to raw text URL.
    """
    model_weights_dir.mkdir(parents=True, exist_ok=True)
    text_cache_path = model_weights_dir / f"wikitext-2-raw-v1_{split}.txt"

    if text_cache_path.exists():
        print(f"Using cached WikiText-2 text: {text_cache_path}")
        return text_cache_path.read_text(encoding="utf-8")

    text = None
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        lines = [line for line in ds["text"] if line and line.strip()]
        text = "\n\n".join(lines)
        print(f"Downloaded WikiText-2 via datasets ({split} split).")
    except Exception as e:
        print(f"Could not load WikiText-2 via datasets ({e}). Falling back to raw text URL.")
        fallback_urls = {
            "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
            "validation": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
            "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
            "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        }
        if split not in fallback_urls:
            raise ValueError(f"Unsupported WikiText-2 split '{split}'. Use one of train/valid/validation/test.")
        with urlopen(fallback_urls[split]) as resp:
            text = resp.read().decode("utf-8")
        text = "\n\n".join([line for line in text.splitlines() if line.strip()])
        print(f"Downloaded WikiText-2 from fallback URL ({split} split).")

    text_cache_path.write_text(text, encoding="utf-8")
    print(f"Saved WikiText-2 text cache: {text_cache_path}")
    return text


def run_wikitext2_perplexity(
    model_path=None,
    tokenizer_path=None,
    device="cuda",
    backend="base",
    config_path=None,
    split: str = "test",
    max_length: int = 2048,
    stride: int = 2048,
    max_windows: int = 0,
    cli_args: Any = None,
):
    """
    Evaluate perplexity on WikiText-2 with sliding-window evaluation.
    Saves fetched text and tokenized IDs under model_weights.

    If *cli_args* is set (typically ``argparse.Namespace`` from ``main``), the model is
    constructed like a normal CLI run (cache size, predict backend, etc.) and routing
    flags (lambda, forced top-n / mass, cache policy) are applied.

    *max_windows*: if > 0, stop after that many windows (quick smoke test).
    """
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    script_dir = Path(__file__).parent
    model_weights_dir = script_dir / "model_weights"
    model_weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"WIKITEXT-2 PERPLEXITY ({split} split)")
    print("=" * 60 + "\n")

    text = _load_wikitext2_raw_text(model_weights_dir, split=split)

    if model_path is None:
        model_path = "QuixiAI/Qwen3-30B-A3B-AWQ"

    print("Initializing Qwen3 30B-A3B AWQ w4a16 quantized model...")
    try:
        if cli_args is not None:
            model = Qwen3_30BA3BW4A16Model(
                model_path=cli_args.model_path or model_path,
                tokenizer_path=cli_args.tokenizer_path or tokenizer_path,
                device=cli_args.device,
                backend=cli_args.backend,
                config_path=cli_args.config_path or config_path,
                max_cached_experts_per_layer=cli_args.max_cached_experts,
                prefetch_experts_count=cli_args.prefetch_experts_count,
                predictor_models_dir=cli_args.predictor_model,
                predictor_device=cli_args.predictor_device,
                predict_layers=cli_args.predict_layers,
                expert_reuse_csv=cli_args.expert_reuse_csv,
                predictor_lookahead=cli_args.predictor_lookahead,
                expert_weights_dir=cli_args.expert_weights_dir,
            )
            _apply_routing_and_cache_cli(model, cli_args)
        else:
            model = Qwen3_30BA3BW4A16Model(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                device=device,
                backend=backend,
                config_path=config_path,
            )
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

    token_cache_path = model_weights_dir / f"wikitext-2-raw-v1_{split}_tokens.pt"
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

    parser = argparse.ArgumentParser(description="Qwen3 30B-A3B AWQ W4A16 Quantized Model - Unified LibTorch Backend")
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
        default="QuixiAI/Qwen3-30B-A3B-AWQ",
        help="Path to quantized model (default: QuixiAI/Qwen3-30B-A3B-AWQ)"
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
        "--config-path",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "configs/configs_strixH_qwen3_30B_A3B.json5")),
        help="Path to NPU config JSON"
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
        default=16,
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
        "--max-cached-experts",
        type=int,
        default=8,
        help="Maximum number of experts to cache per layer (cached/predict backends only, default: 8)"
    )
    parser.add_argument(
        "--prefetch-experts-count",
        type=int,
        default=1,
        help="Number of experts to speculatively prefetch (predict backend only, default: 1)"
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Compute perplexity for the input text (or prompt-test sequence) instead of generation."
    )
    parser.add_argument(
        "--wikitext2-perplexity",
        action="store_true",
        help="Compute perplexity on WikiText-2 and save downloaded/tokenized files under model_weights."
    )
    parser.add_argument(
        "--wikitext2-split",
        type=str,
        default="test",
        choices=["train", "valid", "validation", "test"],
        help="WikiText-2 split to evaluate."
    )
    parser.add_argument(
        "--wikitext2-max-length",
        type=int,
        default=2048,
        help="Max context length per evaluation window for WikiText-2 perplexity."
    )
    parser.add_argument(
        "--wikitext2-stride",
        type=int,
        default=2048,
        help="Stride for sliding-window WikiText-2 perplexity."
    )
    parser.add_argument(
        "--wikitext2-max-windows",
        type=int,
        default=0,
        help="If > 0, only evaluate this many sliding windows (smoke test / cheap run). 0 = full split.",
    )
    parser.add_argument("--sweep-prompts-file", type=str, default=None)
    parser.add_argument(
        "--generation-perplexity",
        action="store_true",
        default=False,
        help="Score -log p(generated token | prefix) on model-generated continuations (requires generation).",
    )
    parser.add_argument("--lambda-val", type=float, default=0.0)
    parser.add_argument("--predict-layers", type=int, nargs="+", default=None)
    parser.add_argument("--predictor-model", type=str, default="")
    parser.add_argument("--predictor-device", type=str, default="gpu")
    parser.add_argument(
        "--predictor-lookahead", type=int, default=1,
        help="Invoke the predictor every N decode tokens where N = lookahead depth (fN from model path). "
             "Default 1 = every token (original behaviour)."
    )
    parser.add_argument("--expert-reuse-csv", type=str, default=None,
                        help="Path to expert_reuse.csv to calibrate per-layer cache and prefetch counts.")
    parser.add_argument(
        "--forced-top-n", type=int, default=0,
        help="(cached backend) Force the top-N unbiased experts into the cache mask every step. "
             "0 = disabled (standard routing)."
    )
    parser.add_argument(
        "--forced-top-p", type=float, default=-1.0,
        help="(cached backend) Force the minimum set of experts whose cumulative softmax "
             "probability >= p into the cache mask every step. Adapts to routing confidence. "
             "-1.0 = disabled (default)."
    )
    parser.add_argument(
        "--mass-threshold-substitution-p", type=float, default=-1.0,
        help="(cached/predict backends) Alternate routing: keep smallest top-k prefix "
             "with cumulative mass >= p and substitute remaining routed slots. "
             "-1.0 = disabled (default)."
    )
    parser.add_argument(
        "--expert-weights-dir", type=str, default=None,
        help="Optional directory for MoE expert weight files. Supports both unpacked "
             "(9 files/expert) and packed (1 file/expert, EXPK format) layouts — "
             "auto-detected at runtime. When omitted, the standard presaved-bins path is used."
    )
    parser.add_argument(
        "--cache-policy",
        type=str,
        default="LRU",
        choices=["LRU", "MRU", "LFU", "MFU", "CLOCK", "RANDOM", "LFRU", "PREFILL", "lru", "mru", "lfu", "mfu", "clock", "random", "lfru", "prefill"],
        help="Cache eviction policy for experts"
    )
    parser.add_argument(
        "--prefill-top-n", type=int, default=0,
        help="(cached backend) Under PREFILL policy, lock the top-N experts from prefill into the cache."
    )

    args = parser.parse_args()

    if args.wikitext2_perplexity:
        return run_wikitext2_perplexity(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path,
            split=args.wikitext2_split,
            max_length=args.wikitext2_max_length,
            stride=args.wikitext2_stride,
            max_windows=args.wikitext2_max_windows,
            cli_args=args,
        )

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
            perplexity=args.perplexity,
            config_path=args.config_path
        )

    print("=" * 60)
    print("Initializing Qwen3 30B-A3B AWQ w4a16 quantized model...")
    print("=" * 60)

    try:
        model = Qwen3_30BA3BW4A16Model(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
            backend=args.backend,
            config_path=args.config_path,
            max_cached_experts_per_layer=args.max_cached_experts,
            prefetch_experts_count=args.prefetch_experts_count,
            predictor_models_dir=args.predictor_model,
            predictor_device=args.predictor_device,
            predict_layers=args.predict_layers,
            expert_reuse_csv=args.expert_reuse_csv,
            predictor_lookahead=args.predictor_lookahead,
            expert_weights_dir=args.expert_weights_dir,
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

    _apply_routing_and_cache_cli(model, args)

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
        
        model.reset_cache_stats()

        for i, prompt in enumerate(prompts):
            print(f"\nProcessing Prompt {i+1}/{len(prompts)}...")
            try:
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
                        if args.generate:
                            total_time += elapsed
                            total_generated_tokens += num_generated
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
            
        if args.generate and total_generated_tokens > 0:
            avg_tps = total_generated_tokens / total_time
            print(f"Average Time per Token: {1.0 / avg_tps:.6f}") # Output inverse since parser expects time per token
            print(f"End-to-End TPS: {avg_tps:.4f}")

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
        except Exception as e:
            print(f"Error during perplexity evaluation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    elif args.generate:
        print(f"Generating {args.max_new_tokens} tokens...\n")
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
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print(f"Running {len(prompts)} prompt(s) from {prompts_file if prompts_file.exists() else '--text'}...\n")

    for prompt_idx, prompt_text in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"PROMPT {prompt_idx + 1}/{len(prompts)}: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")
        print(f"{'='*60}")

        if args.generate:
            try:
                input_ids = model.tokenize(prompt_text)

                generated = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k
                )

                if model.tokenizer is not None:
                    prompt_len = input_ids.size(1)
                    generated_tokens = generated[0, prompt_len:].tolist()
                    decoded_generated = model.tokenizer.decode(generated_tokens, skip_special_tokens=False)

                    print("Generated text:")
                    print(f"{'='*60}")
                    print(decoded_generated)
                    print(f"{'='*60}")
                else:
                    print(f"Generated token IDs: {generated}")
            except Exception as e:
                print(f"Error during generation for prompt {prompt_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
        else:
            try:
                start_time = time.time()
                logits = model(prompt_text)
                end_time = time.time()
                print(f"Prefill time: {end_time - start_time:.4f} seconds")
                print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
                print(f"Logits stats — min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, "
                      f"mean: {logits.mean().item():.4f}")
            except Exception as e:
                print(f"Error during forward pass for prompt {prompt_idx + 1}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'=' * 60}")
    if hasattr(model, "load_time"):
        print(f"Weight loading time: {model.load_time:.2f} seconds")
    if hasattr(model, "print_cache_stats"):
        model.print_cache_stats()
    print("Done!")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    exit(main())

    # WikiText-2 (full): --wikitext2-perplexity --wikitext2-split test --wikitext2-max-length 4096 --wikitext2-stride 2048
    # Quick cached smoke: --backend cached --max-cached-experts 36 --lambda-val 1.0 --forced-top-n 4 \
    #   --wikitext2-perplexity --wikitext2-max-windows 2 --wikitext2-max-length 512 --wikitext2-stride 512
