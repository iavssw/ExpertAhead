"""
pack_experts.py — Pack 9 per-expert .bin files into one combined binary per expert.

Packed file format  (layer_{L}_expert_{E}.bin):
─────────────────────────────────────────────────────────────────
 Offset  Size  Description
────────────────────────────────────────────────────────────────
  0      4     Magic: b"EXPK"
  4      4     uint32  num_tensors  (always 9)
  8      9×48  Tensor descriptor table (9 entries × 48 bytes):
               - 32 bytes  name  (ASCII, null-padded)
               -  8 bytes  uint64  offset into file (from byte 0)
               -  8 bytes  uint64  byte length
─────────────────────────────────────────────────────────────────
  440+   raw tensor data; each tensor starts at a 512-byte-aligned offset
         (padding bytes between tensors are ignored by readers)
─────────────────────────────────────────────────────────────────

Tensor order (fixed):
  0  gate.qweight    uint8    [out, in/2]
  1  gate.scales     bfloat16 [out, groups]
  2  gate.zeros      int8     [out, groups]
  3  up.qweight      uint8    [out, in/2]
  4  up.scales       bfloat16 [out, groups]
  5  up.zeros        int8     [out, groups]
  6  down.qweight    uint8    [out, in/2]
  7  down.scales     bfloat16 [out, groups]
  8  down.zeros      int8     [out, groups]
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MAGIC = b"EXPK"
NUM_TENSORS = 9
DESC_ENTRY_SIZE = 48       # 32 (name) + 8 (offset) + 8 (size)
TABLE_SIZE = 8 + NUM_TENSORS * DESC_ENTRY_SIZE
HEADER_SIZE = 512   # Pad to 512 for O_DIRECT alignment
TENSOR_ALIGN = 512  # Each tensor body starts on a 512-byte boundary (O_DIRECT)


def align_up(n: int, align: int = TENSOR_ALIGN) -> int:
    return ((n + align - 1) // align) * align

# Fixed order of the 9 tensors inside the packed file
TENSOR_NAMES = [
    "gate.qweight",
    "gate.scales",
    "gate.zeros",
    "up.qweight",
    "up.scales",
    "up.zeros",
    "down.qweight",
    "down.scales",
    "down.zeros",
]

# Map name -> source filename suffix
def _src_name(tensor_name: str, layer: int, expert: int) -> str:
    proj, kind = tensor_name.split(".")
    return f"layer_{layer}_expert_{expert}_{proj}.{kind}.bin"


def pack_expert(src_dir: Path, dst_dir: Path, layer: int, expert: int) -> None:
    """Pack the 9 files for one expert into a single file."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_path = dst_dir / f"layer_{layer}_expert_{expert}.bin"

    # Collect raw bytes for each tensor
    blobs: list[bytes] = []
    for name in TENSOR_NAMES:
        src = src_dir / _src_name(name, layer, expert)
        blobs.append(src.read_bytes())

    # Build descriptor table — align each tensor start for O_DIRECT pread
    offsets: list[int] = []
    cursor = HEADER_SIZE
    for blob in blobs:
        cursor = align_up(cursor)
        offsets.append(cursor)
        cursor += len(blob)

    # Encode header
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", NUM_TENSORS)
    for name, blob, offset in zip(TENSOR_NAMES, blobs, offsets):
        name_bytes = name.encode("ascii")
        name_bytes = name_bytes[:32].ljust(32, b"\x00")
        header += name_bytes
        header += struct.pack("<QQ", offset, len(blob))

    header += b"\x00" * (HEADER_SIZE - len(header))
    assert len(header) == HEADER_SIZE, f"Header size mismatch"

    with out_path.open("wb") as f:
        f.write(header)
        cursor = HEADER_SIZE
        for blob in blobs:
            cursor = align_up(cursor)
            if f.tell() < cursor:
                f.write(b"\x00" * (cursor - f.tell()))
            f.write(blob)
            cursor += len(blob)


def pack_all(
    src_dir: Path,
    dst_dir: Path,
    num_layers: int,
    num_experts: int,
    num_workers: int = 8,
) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = num_layers * num_experts
    done = 0

    tasks = [
        (layer, expert)
        for layer in range(num_layers)
        for expert in range(num_experts)
    ]

    print(f"Packing {total} experts ({num_layers} layers × {num_experts} experts) "
          f"from {src_dir} → {dst_dir}")
    print(f"Using {num_workers} worker threads.")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(pack_expert, src_dir, dst_dir, layer, expert): (layer, expert)
            for layer, expert in tasks
        }
        for fut in as_completed(futures):
            layer, expert = futures[fut]
            fut.result()   # re-raise any exception
            done += 1
            if done % 500 == 0 or done == total:
                pct = 100.0 * done / total
                print(f"  [{done}/{total}] {pct:.1f}%", flush=True)

    print(f"Done. {total} packed files written to {dst_dir}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Pack 9 per-expert .bin files into one combined binary per expert."
    )
    p.add_argument(
        "--src",
        type=str,
        default="model_weights/Qwen3-30B-A3B-AWQ_unpacked",
        help="Source directory containing the 9 individual .bin files per expert.",
    )
    p.add_argument(
        "--dst",
        type=str,
        default="model_weights/Qwen3-30B-A3B-AWQ_packed",
        help="Destination directory for packed expert files.",
    )
    p.add_argument("--num-layers", type=int, default=48)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel worker threads for I/O.",
    )
    args = p.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)

    if not src_dir.exists():
        print(f"ERROR: source directory not found: {src_dir}", file=sys.stderr)
        return 1

    pack_all(src_dir, dst_dir, args.num_layers, args.num_experts, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
