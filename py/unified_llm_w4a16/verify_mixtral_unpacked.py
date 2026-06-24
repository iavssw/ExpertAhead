#!/usr/bin/env python3
"""Verify Mixtral unpacked weights against full AWQ safetensors reference."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "model_weights"
MODEL_NAME = "mixtral-8x7b-v0.1-AWQ"
STRIPPED_ST = WEIGHTS_DIR / f"{MODEL_NAME}.safetensors"
FULL_ST = WEIGHTS_DIR / f"{MODEL_NAME}_FULL.safetensors"
UNPACKED_DIR = WEIGHTS_DIR / f"{MODEL_NAME}_unpacked"
DEFAULT_CONFIG = SCRIPT_DIR / "configs/configs_strixH_mixtral7x8B.json5"


def parse_safetensors_key_count(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    keys = [k for k in header if k != "__metadata__"]
    return len(keys), sum(1 for k in keys if "qweight" in k)


def ensure_full_safetensors(force: bool = False) -> Path:
    if FULL_ST.exists() and not force:
        _, qweight_count = parse_safetensors_key_count(FULL_ST)
        if qweight_count > 0:
            print(f"Using existing full safetensors: {FULL_ST}")
            return FULL_ST
        print(f"Removing incomplete full safetensors: {FULL_ST}")
        FULL_ST.unlink()

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file, save_file

    print("Downloading full AWQ model from HuggingFace (3 shards)...")
    model_path = snapshot_download("TheBloke/mixtral-8x7b-v0.1-AWQ")
    shard_files = sorted(str(p) for p in Path(model_path).glob("model-*-of-*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors shards found under {model_path}")

    print(f"Merging {len(shard_files)} shards into {FULL_ST} ...")
    merged: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        print(f"  loading {shard}")
        merged.update(load_file(shard))
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(FULL_ST))
    key_count, qweight_count = parse_safetensors_key_count(FULL_ST)
    print(f"Saved full safetensors: keys={key_count}, qweight={qweight_count}, size={FULL_ST.stat().st_size/1e9:.2f} GB")
    return FULL_ST


def compare_layer0_bins(full_st: Path, unpacked_dir: Path) -> bool:
    sys.path.insert(0, str(SCRIPT_DIR))
    from mixtral_8x7B_w4a16_model import (
        _align_zeros_to_scales,
        _get_quantized_tensors,
        _normalize_group_tensor,
        _pack_qweight_out_in2,
        _unpack_awq_qweight,
        _unpack_awq_qzeros,
    )
    from safetensors.torch import load_file

    state = load_file(str(full_st))

    checks = [
        ("model.layers.0.self_attn.q_proj", "layer_0_q", 4096, 4096),
        ("model.layers.0.block_sparse_moe.experts.0.w1", "layer_0_expert_0_gate", 4096, 14336),
        ("model.layers.0.block_sparse_moe.experts.0.w3", "layer_0_expert_0_up", 4096, 14336),
        ("model.layers.0.block_sparse_moe.experts.0.w2", "layer_0_expert_0_down", 14336, 4096),
    ]

    all_ok = True
    for base, short, in_feat, out_feat in checks:
        qweight, scales, qzeros, _ = _get_quantized_tensors(state, base)
        qweight_unpacked = _unpack_awq_qweight(qweight.to(torch.int32))
        zeros = _unpack_awq_qzeros(qzeros.to(torch.int32))
        qweight_packed = _pack_qweight_out_in2(qweight_unpacked.to(torch.int8))
        scales_out = _normalize_group_tensor(scales.cpu().to(torch.bfloat16), out_feat)
        zeros_out = _align_zeros_to_scales(zeros, scales_out, out_feat)

        for suffix, fresh in [
            ("qweight.bin", qweight_packed.to(torch.uint8)),
            ("scales.bin", scales_out),
            ("zeros.bin", zeros_out.to(torch.int8)),
        ]:
            path = unpacked_dir / f"{short}.{suffix}"
            on_disk = torch.frombuffer(bytearray(path.read_bytes()), dtype=fresh.dtype).clone()
            fresh_flat = fresh.flatten().cpu()
            if on_disk.numel() != fresh_flat.numel():
                print(f"FAIL {path.name}: size mismatch disk={on_disk.numel()} fresh={fresh_flat.numel()}")
                all_ok = False
                continue
            match = torch.equal(on_disk, fresh_flat)
            max_diff = (on_disk.float() - fresh_flat.float()).abs().max().item() if not match else 0.0
            status = "OK" if match else f"FAIL max_diff={max_diff}"
            print(f"{status:12} {path.name}")
            all_ok = all_ok and match
    return all_ok


def make_temp_config(base_config: Path, use_presaved: bool) -> str:
    text = base_config.read_text()
    # json5 -> json-ish: strip comments crudely for temp config
    lines = []
    for line in text.splitlines():
        stripped = line.split("//", 1)[0].rstrip()
        if stripped:
            lines.append(stripped)
    cfg = json.loads("\n".join(lines))
    cfg["usePreSavedWeights"] = use_presaved
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _swap_stripped_for_full(full_st: Path):
    """Hide stripped safetensors so loader uses the full merged file."""
    backup = STRIPPED_ST.with_suffix(".safetensors.stripped_bak")
    if STRIPPED_ST.exists():
        if backup.exists():
            backup.unlink()
        STRIPPED_ST.rename(backup)
    link_target = WEIGHTS_DIR / f"{MODEL_NAME}.safetensors"
    if link_target.exists() or link_target.is_symlink():
        link_target.unlink()
    link_target.symlink_to(full_st.resolve())
    return backup


def _restore_stripped(backup: Path | None) -> None:
    link_target = WEIGHTS_DIR / f"{MODEL_NAME}.safetensors"
    if link_target.is_symlink():
        link_target.unlink()
    elif link_target.exists() and link_target.resolve() == FULL_ST.resolve():
        link_target.unlink()
    if backup is not None and backup.exists():
        backup.rename(STRIPPED_ST)


def run_generation(label: str, use_presaved: bool, expert_weights_dir: str | None, use_full_st: bool) -> str:
    from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model

    config_path = make_temp_config(DEFAULT_CONFIG, use_presaved=use_presaved)
    backup = _swap_stripped_for_full(FULL_ST) if use_full_st and not use_presaved else None
    try:
        kwargs = dict(
            model_path="TheBloke/mixtral-8x7b-v0.1-AWQ",
            backend="cached",
            max_cached_experts_per_layer=2,
            prewarm_experts=False,
            config_path=config_path,
            expert_weights_dir=expert_weights_dir,
        )
        print(f"\n=== {label} ===")
        print(f"usePreSavedWeights={use_presaved}, expert_weights_dir={expert_weights_dir}, use_full_st={use_full_st}")
        t0 = time.time()
        model = Mixtral8x7BW4A16Model(**kwargs)
        input_ids = model.tokenize("What is the meaning of life the universe and everything?")
        output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.0)
        gen_only = model.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)
        elapsed = time.time() - t0
        print(f"elapsed={elapsed:.1f}s")
        print(f"generated: {gen_only!r}")
        return gen_only
    finally:
        if backup is not None:
            _restore_stripped(backup)
        os.unlink(config_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Mixtral unpacked weights")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Only compare/run with existing FULL safetensors")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    stripped_keys, stripped_q = parse_safetensors_key_count(STRIPPED_ST)
    print(f"Local stripped safetensors: {STRIPPED_ST}")
    print(f"  keys={stripped_keys}, qweight={stripped_q}")

    if args.skip_download and not FULL_ST.exists():
        print(f"ERROR: {FULL_ST} not found; run without --skip-download first")
        return 1

    full_st = FULL_ST if args.skip_download else ensure_full_safetensors(force=args.force_download)

    print(f"\nComparing layer-0 bins in {UNPACKED_DIR} against freshly packed reference...")
    bins_ok = compare_layer0_bins(full_st, UNPACKED_DIR)
    print(f"Layer-0 bin parity: {'PASS' if bins_ok else 'FAIL'}")

    if args.skip_inference:
        return 0 if bins_ok else 1

    ref_out = run_generation(
        "Reference: direct safetensors load",
        use_presaved=False,
        expert_weights_dir=None,
        use_full_st=True,
    )
    unpacked_out = run_generation(
        "Unpacked bins + stripped safetensors",
        use_presaved=True,
        expert_weights_dir=str(UNPACKED_DIR),
        use_full_st=False,
    )

    print("\n=== Summary ===")
    print(f"bin parity: {'PASS' if bins_ok else 'FAIL'}")
    print(f"reference : {ref_out!r}")
    print(f"unpacked  : {unpacked_out!r}")
    print(f"match     : {ref_out == unpacked_out}")
    return 0 if bins_ok and ref_out == unpacked_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
