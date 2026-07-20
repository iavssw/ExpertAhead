"""
Llama 4 Scout 17B-16E AWQ w4a16 — Python frontend (download + AWQ unpack).

C++ backend support for ArchitectureType.LLAMA4 is not wired yet. This module focuses on:
  - Downloading / merging Hugging Face safetensors shards
  - Stripping the ``language_model.`` prefix so keys match ``model.layers.*`` / ``lm_head.*``
  - Unpacking AWQ tensors into pre-saved ``.bin`` files (same layout as Qwen3: q/k/v + experts)

fp16 tensors (o_proj, router, norms, embeddings, lm_head) stay in the merged safetensors file for
``load_non_quantized_weights_from_safetensors`` once the C++ loader understands Llama 4 key paths.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

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

# Default HF repo (AWQ); requires HF auth if the repo is gated.
DEFAULT_LLAMA4_SCOUT_AWQ_REPO = "meta-llama/Llama-4-Scout-17B-16E-Instruct-AWQ"


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
    qweight = qweight.t().contiguous()
    out_features_div_8 = qweight.size(0)
    in_features = qweight.size(1)
    out_features = out_features_div_8 * 8
    unpacked = _unpack_awq_zigzag_to_contiguous(qweight)
    unpacked = unpacked.permute(0, 2, 1).contiguous()
    unpacked = unpacked.view(out_features, in_features)
    return unpacked.t().contiguous().to(torch.uint8)


def _unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    unpacked = _unpack_awq_zigzag_to_contiguous(qzeros)
    n_groups = qzeros.size(0)
    out_features = qzeros.size(1) * 8
    unpacked = unpacked.view(n_groups, out_features)
    return unpacked.contiguous().to(torch.int8)


def _pack_qweight_out_in2(unpacked_qweight: torch.Tensor) -> torch.Tensor:
    w_out_in = unpacked_qweight.t().contiguous()
    out_features = w_out_in.size(0)
    in_features = w_out_in.size(1)
    w_view = w_out_in.view(out_features, in_features // 2, 2)
    w_low = w_view.select(-1, 0).to(torch.uint8)
    w_high = w_view.select(-1, 1).to(torch.uint8)
    packed = torch.bitwise_or(
        torch.bitwise_and(w_low, 0x0F),
        torch.bitwise_left_shift(torch.bitwise_and(w_high, 0x0F), 4),
    )
    return packed.to(torch.uint8)


def _normalize_group_tensor(t: torch.Tensor, out_feat: int) -> torch.Tensor:
    if t is None:
        return t
    if t.dim() == 1:
        if t.numel() == out_feat:
            return t
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
    if t.dim() == 2:
        if t.size(0) == out_feat:
            return t
        if t.size(1) == out_feat:
            return t.t().contiguous()
        if t.numel() % out_feat == 0:
            return t.view(out_feat, -1)
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


def _get_quantized_tensors(state_dict: Dict[str, torch.Tensor], base_name: str):
    qweight = state_dict.get(base_name + ".qweight")
    scales = state_dict.get(base_name + ".scales")
    qzeros = state_dict.get(base_name + ".qzeros")
    g_idx = state_dict.get(base_name + ".g_idx")
    if qweight is None or scales is None or qzeros is None:
        missing = [k for k in (base_name + ".qweight", base_name + ".scales", base_name + ".qzeros") if k not in state_dict]
        raise KeyError(f"Missing quantized tensors for {base_name}: {missing}")
    return qweight, scales, qzeros, g_idx


def strip_language_model_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    HF Llama 4 checkpoints wrap weights under ``language_model.``. Strip it so keys look like
    ``model.layers.*``, ``lm_head.weight``, ``model.norm.weight``, etc.
    """
    prefix = "language_model."
    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


@dataclass
class Llama4ScoutShapeConfig:
    """Tensor layout for Llama 4 Scout 17B-16E (text); align with config.json when available."""

    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    groupsize: int = 128
    num_experts: int = 16
    num_experts_per_tok: int = 1


def download_or_merge_safetensors(
    model_path: str,
    weights_dir: Union[str, Path],
    *,
    force_redownload: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    If ``model_path`` is a local directory, merge all ``*.safetensors`` into one file under ``weights_dir``.
    If it is not local, call ``snapshot_download`` then merge.

    Returns:
        (path to merged ``{{name}}.safetensors`` with keys normalized (no ``language_model.``),
         directory that contains the original ``config.json`` — for shape loading — or None if using a cached merge only)
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    model_name = _model_name_from_path(model_path)
    saved_safetensors = weights_dir / f"{model_name}.safetensors"
    meta_dir: Optional[Path] = None

    if saved_safetensors.exists() and not force_redownload:
        print(f"Using existing merged safetensors: {saved_safetensors}")
        cfg_in_weights = weights_dir / "config.json"
        if cfg_in_weights.is_file():
            meta_dir = weights_dir
        return saved_safetensors, meta_dir

    if os.path.isdir(model_path):
        local_dir = os.path.abspath(model_path)
    else:
        if not os.path.exists(model_path):
            print(f"Model path {model_path} not found locally, downloading from Hub...")
            local_dir = snapshot_download(repo_id=model_path)
            print(f"Model downloaded to {local_dir}")
        else:
            local_dir = os.path.abspath(model_path)

    meta_dir = Path(local_dir)
    safetensors_files = sorted(glob.glob(os.path.join(local_dir, "*.safetensors")))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {local_dir}")

    merged_state_dict: Dict[str, torch.Tensor] = {}
    for st_file in safetensors_files:
        print(f"  Loading {st_file}...")
        merged_state_dict.update(load_file(st_file))

    merged_state_dict = strip_language_model_prefix(merged_state_dict)

    print(f"Writing merged + key-normalized safetensors to {saved_safetensors}...")
    save_file(merged_state_dict, str(saved_safetensors))
    print(f"Saved: {saved_safetensors} ({len(merged_state_dict)} tensors)")

    cfg_src = meta_dir / "config.json"
    if cfg_src.is_file():
        shutil.copy2(cfg_src, weights_dir / "config.json")
        meta_dir = weights_dir

    return saved_safetensors, meta_dir


def _bins_exist_llama4(presaved_dir: Path, cfg: Llama4ScoutShapeConfig) -> bool:
    if not presaved_dir.exists():
        return False
    for layer_idx in range(cfg.num_hidden_layers):
        for short_name in ["q", "k", "v"]:
            if not (presaved_dir / f"layer_{layer_idx}_{short_name}.qweight.bin").exists():
                return False
            if not (presaved_dir / f"layer_{layer_idx}_{short_name}.scales.bin").exists():
                return False
            if not (presaved_dir / f"layer_{layer_idx}_{short_name}.zeros.bin").exists():
                return False
        for e in range(cfg.num_experts):
            for short_name in ["gate", "up", "down"]:
                stem = f"layer_{layer_idx}_expert_{e}_{short_name}"
                for suffix in (".qweight.bin", ".scales.bin", ".zeros.bin"):
                    if not (presaved_dir / f"{stem}{suffix}").exists():
                        return False
    return True


def unpack_awq_to_bins(
    state_dict: Dict[str, torch.Tensor],
    presaved_dir: Path,
    cfg: Llama4ScoutShapeConfig,
    *,
    skip_if_present: bool = True,
) -> None:
    """
    Write AWQ-packed layers to flat binaries compatible with ``load_quantized_weights_from_bins``.

    Quantized: q_proj, k_proj, v_proj, and each expert gate/up/down.
    Not written here (fp16, load from merged safetensors in C++): o_proj, router, norms, embed, lm_head.
    """
    manifest_path = presaved_dir / "manifest.json"
    if skip_if_present and _bins_exist_llama4(presaved_dir, cfg):
        print(f"Re-using existing unpacked bins under {presaved_dir}")
        return

    presaved_dir.mkdir(parents=True, exist_ok=True)

    def _write_quantized(base_name: str, out_name: str, in_feat: int, out_feat: int) -> None:
        qweight, scales, qzeros, _g_idx = _get_quantized_tensors(state_dict, base_name)

        qweight = qweight.cpu()
        scales = scales.cpu()
        qzeros = qzeros.cpu()

        is_awq = qweight.size(0) == in_feat and qweight.size(1) == out_feat // 8
        if not is_awq:
            raise ValueError(
                f"Unsupported quantization format for {base_name}: expected AWQ qweight [in, out/8], "
                f"got {tuple(qweight.shape)} (in={in_feat}, out={out_feat})"
            )

        qweight_unpacked = _unpack_awq_qweight(qweight.to(torch.int32))
        scales_bf = scales.to(torch.bfloat16)
        zeros = _unpack_awq_qzeros(qzeros.to(torch.int32))

        qweight_packed = _pack_qweight_out_in2(qweight_unpacked.to(torch.int8))

        scales_out = _normalize_group_tensor(scales_bf, out_feat)
        zeros_out = _align_zeros_to_scales(zeros, scales_out, out_feat)

        scales_out = scales_out.to(torch.bfloat16)
        zeros_out = zeros_out.to(torch.int8)

        _write_tensor_raw(presaved_dir / f"{out_name}.qweight.bin", qweight_packed.to(torch.uint8))
        _write_tensor_raw(presaved_dir / f"{out_name}.scales.bin", scales_out)
        _write_tensor_raw(presaved_dir / f"{out_name}.zeros.bin", zeros_out)

    # Attention: q, k, v AWQ; o_proj is fp16 in Scout AWQ — not unpacked to bins here.
    attn_specs: List[Tuple[str, str, int, int]] = [
        ("self_attn.q_proj", "q", cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
        ("self_attn.k_proj", "k", cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
        ("self_attn.v_proj", "v", cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
    ]

    for layer_idx in range(cfg.num_hidden_layers):
        layer_prefix = f"model.layers.{layer_idx}"
        for suffix, short_name, in_feat, out_feat in attn_specs:
            base_name = f"{layer_prefix}.{suffix}"
            _write_quantized(base_name, f"layer_{layer_idx}_{short_name}", in_feat, out_feat)

        expert_root = f"{layer_prefix}.feed_forward.experts"
        for expert_idx in range(cfg.num_experts):
            gate_base = f"{expert_root}.{expert_idx}.gate_proj"
            up_base = f"{expert_root}.{expert_idx}.up_proj"
            down_base = f"{expert_root}.{expert_idx}.down_proj"
            if gate_base + ".qweight" not in state_dict:
                raise KeyError(
                    f"Expected AWQ expert weights at {gate_base}.* — check checkpoint naming "
                    f"(feed_forward.experts.*.gate_proj)."
                )
            _write_quantized(
                gate_base,
                f"layer_{layer_idx}_expert_{expert_idx}_gate",
                cfg.hidden_size,
                cfg.intermediate_size,
            )
            _write_quantized(
                up_base,
                f"layer_{layer_idx}_expert_{expert_idx}_up",
                cfg.hidden_size,
                cfg.intermediate_size,
            )
            _write_quantized(
                down_base,
                f"layer_{layer_idx}_expert_{expert_idx}_down",
                cfg.intermediate_size,
                cfg.hidden_size,
            )

    expected_manifest = {
        "format_version": 1,
        "architecture": "llama4_scout_w4a16",
        "unpacked_layout": "out_groups_v2",
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "groupsize": cfg.groupsize,
        "num_experts": cfg.num_experts,
        "num_experts_per_tok": cfg.num_experts_per_tok,
        "quantized_in_bins": "q,k,v + expert gate,up,down",
        "fp16_in_safetensors": "o_proj, router, norms, q_norm, k_norm, embed, lm_head, model.norm",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(expected_manifest, f, indent=2)
    print(f"Unpacked AWQ bins + manifest written to {presaved_dir}")


def load_shape_config_from_hf_folder(folder: Union[str, Path]) -> Llama4ScoutShapeConfig:
    """Override defaults using ``config.json`` next to downloaded weights (optional)."""
    folder = Path(folder)
    cfg_path = folder / "config.json"
    if not cfg_path.is_file():
        return Llama4ScoutShapeConfig()
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    text_cfg = raw.get("text_config") or raw
    return Llama4ScoutShapeConfig(
        vocab_size=int(text_cfg.get("vocab_size", Llama4ScoutShapeConfig().vocab_size)),
        hidden_size=int(text_cfg.get("hidden_size", Llama4ScoutShapeConfig().hidden_size)),
        intermediate_size=int(text_cfg.get("intermediate_size", Llama4ScoutShapeConfig().intermediate_size)),
        num_hidden_layers=int(text_cfg.get("num_hidden_layers", Llama4ScoutShapeConfig().num_hidden_layers)),
        num_attention_heads=int(text_cfg.get("num_attention_heads", Llama4ScoutShapeConfig().num_attention_heads)),
        num_key_value_heads=int(text_cfg.get("num_key_value_heads", Llama4ScoutShapeConfig().num_key_value_heads)),
        head_dim=int(text_cfg.get("head_dim", Llama4ScoutShapeConfig().head_dim)),
        groupsize=int(raw.get("quantization_config", {}).get("group_size", Llama4ScoutShapeConfig().groupsize)),
        num_experts=int(text_cfg.get("num_local_experts", Llama4ScoutShapeConfig().num_experts)),
        num_experts_per_tok=int(text_cfg.get("num_experts_per_tok", Llama4ScoutShapeConfig().num_experts_per_tok)),
    )


class Llama4ScoutW4A16Model:
    """
    Thin wrapper: download merged safetensors and optionally unpack AWQ to bins.

    The LibTorch C++ backend does not expose ``ArchitectureType.LLAMA4`` yet — do not pass
    ``backend='base'`` until that exists. Use this class for weight pipeline only.
    """

    def __init__(
        self,
        model_path: Optional[str] = DEFAULT_LLAMA4_SCOUT_AWQ_REPO,
        weights_folder: str = "model_weights",
        shape_cfg: Optional[Llama4ScoutShapeConfig] = None,
        download: bool = True,
        unpack_bins: bool = False,
    ):
        self.model_path = model_path
        self.script_dir = Path(__file__).parent
        self.weights_dir = self.script_dir / weights_folder
        self.shape_cfg = shape_cfg or Llama4ScoutShapeConfig()

        self.merged_safetensors_path: Optional[Path] = None
        self.presaved_dir: Optional[Path] = None
        self.load_time: float = 0.0

        if download and model_path:
            t0 = time.time()
            self.merged_safetensors_path, meta_dir = download_or_merge_safetensors(model_path, self.weights_dir)
            if meta_dir is not None:
                self.shape_cfg = load_shape_config_from_hf_folder(meta_dir)
            elif os.path.isdir(model_path):
                self.shape_cfg = load_shape_config_from_hf_folder(model_path)
            t1 = time.time()
            self.load_time = t1 - t0
            print(f"Merged safetensors ready in {self.load_time:.2f}s: {self.merged_safetensors_path}")

        if unpack_bins and self.merged_safetensors_path:
            from safetensors.torch import load_file

            self.presaved_dir = self.weights_dir / f"{_model_name_from_path(model_path or 'llama4_scout')}_unpacked"
            if (self.weights_dir / "config.json").is_file():
                self.shape_cfg = load_shape_config_from_hf_folder(self.weights_dir)

            t0 = time.time()
            state_dict = load_file(str(self.merged_safetensors_path))
            if any(k.startswith("language_model.") for k in state_dict.keys()):
                state_dict = strip_language_model_prefix(state_dict)
            unpack_awq_to_bins(state_dict, self.presaved_dir, self.shape_cfg)
            t1 = time.time()
            print(f"Unpack finished in {t1 - t0:.2f}s -> {self.presaved_dir}")


def _cli() -> int:
    p = argparse.ArgumentParser(description="Llama 4 Scout AWQ: download / merge safetensors and unpack AWQ bins.")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="Download (if needed) and merge safetensors with normalized keys.")
    d.add_argument("--repo", type=str, default=DEFAULT_LLAMA4_SCOUT_AWQ_REPO, help="HF repo id or local model directory")
    d.add_argument(
        "--weights-dir",
        type=str,
        default=str(_script_dir / "model_weights"),
        help="Directory for merged {name}.safetensors",
    )
    d.add_argument("--force", action="store_true", help="Re-merge even if merged file exists")

    u = sub.add_parser("unpack", help="Unpack AWQ from an already merged .safetensors file to bin tensors.")
    u.add_argument("--safetensors", type=str, required=True, help="Path to merged, key-normalized .safetensors")
    u.add_argument("--out-dir", type=str, required=True, help="Output directory for .bin files + manifest.json")
    u.add_argument("--config", type=str, default="", help="Optional config.json for shapes (text_config)")

    args = p.parse_args()

    if args.cmd == "download":
        download_or_merge_safetensors(args.repo, args.weights_dir, force_redownload=args.force)
        return 0

    if args.cmd == "unpack":
        from safetensors.torch import load_file

        cfg = Llama4ScoutShapeConfig()
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                raw = json.load(f)
            text_cfg = raw.get("text_config") or raw
            cfg = Llama4ScoutShapeConfig(
                vocab_size=int(text_cfg.get("vocab_size", cfg.vocab_size)),
                hidden_size=int(text_cfg.get("hidden_size", cfg.hidden_size)),
                intermediate_size=int(text_cfg.get("intermediate_size", cfg.intermediate_size)),
                num_hidden_layers=int(text_cfg.get("num_hidden_layers", cfg.num_hidden_layers)),
                num_attention_heads=int(text_cfg.get("num_attention_heads", cfg.num_attention_heads)),
                num_key_value_heads=int(text_cfg.get("num_key_value_heads", cfg.num_key_value_heads)),
                head_dim=int(text_cfg.get("head_dim", cfg.head_dim)),
                groupsize=int(raw.get("quantization_config", {}).get("group_size", cfg.groupsize)),
                num_experts=int(text_cfg.get("num_local_experts", cfg.num_experts)),
                num_experts_per_tok=int(text_cfg.get("num_experts_per_tok", cfg.num_experts_per_tok)),
            )
        state_dict = load_file(args.safetensors)
        if any(k.startswith("language_model.") for k in state_dict.keys()):
            state_dict = strip_language_model_prefix(state_dict)
        unpack_awq_to_bins(state_dict, Path(args.out_dir), cfg, skip_if_present=False)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
