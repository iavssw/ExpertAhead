"""
Shared I/O helpers for the expert-loading microbenchmark study.

Mirrors the C++ load path in:
  - include/unified_llm_w4a16_common/moe_expert_load_packed.inl
  - include/unified_llm_w4a16_common/moe_expert_load_unpacked.inl
  - include/unified_llm_w4a16_common/moe_expert_io.inl
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

EXPK_MAGIC = 0x4B505845  # "EXPK" little-endian
EXPK_NUM_TENSORS = 9
EXPK_DESC_SIZE = 48
EXPK_HEADER_SIZE = 8 + EXPK_NUM_TENSORS * EXPK_DESC_SIZE
ALIGN = 512

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
_LIBC.pread.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int64]
_LIBC.pread.restype = ctypes.c_ssize_t


@dataclass(frozen=True)
class ExpkTensor:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class ExpertPaths:
    layer: int
    expert: int
    format: str  # "packed" | "unpacked"
    packed_path: Optional[Path]
    unpacked_paths: Tuple[Path, ...]


@dataclass
class AlignedBuffer:
    ptr: ctypes.c_void_p
    alloc: int

    def pread_into(self, fd: int, size: int, offset: int = 0) -> None:
        done = 0
        while done < size:
            n = _LIBC.pread(fd, ctypes.c_void_p(self.ptr.value + done), size - done, offset + done)
            if n < 0:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
            if n == 0:
                raise OSError(f"pread short read at offset {offset + done}")
            done += n

    def as_bytes(self, size: int) -> bytes:
        return ctypes.string_at(self.ptr, size)


_ALIVE_BUFFERS: List[AlignedBuffer] = []


def is_packed_dir(bin_dir: str | Path) -> bool:
    return (Path(bin_dir) / "layer_0_expert_0.bin").exists()


def expert_paths(bin_dir: str | Path, layer: int, expert: int) -> ExpertPaths:
    root = Path(bin_dir)
    prefix = f"layer_{layer}_expert_{expert}"
    packed = root / f"{prefix}.bin"
    if packed.exists():
        return ExpertPaths(layer, expert, "packed", packed, ())
    names = (
        "gate.qweight", "gate.scales", "gate.zeros",
        "up.qweight", "up.scales", "up.zeros",
        "down.qweight", "down.scales", "down.zeros",
    )
    paths = tuple(root / f"{prefix}_{n}.bin" for n in names)
    return ExpertPaths(layer, expert, "unpacked", None, paths)


def expert_total_bytes(bin_dir: str | Path, layer: int, expert: int) -> int:
    ep = expert_paths(bin_dir, layer, expert)
    if ep.format == "packed":
        return ep.packed_path.stat().st_size
    return sum(p.stat().st_size for p in ep.unpacked_paths)


def align_up(n: int, align: int = ALIGN) -> int:
    return ((n + align - 1) // align) * align


def alloc_aligned(size: int, align: int = ALIGN) -> AlignedBuffer:
    ptr = ctypes.c_void_p()
    alloc = align_up(size, align)
    if _LIBC.posix_memalign(ctypes.byref(ptr), align, alloc) != 0:
        raise MemoryError("posix_memalign failed")
    buf = AlignedBuffer(ptr=ptr, alloc=alloc)
    _ALIVE_BUFFERS.append(buf)
    return buf


def _strict_ssd_io() -> bool:
    v = os.environ.get("HETEROPREDICT_STRICT_SSD_IO", "")
    return v in ("1", "y", "Y", "yes", "YES")


def _open_odirect(path: Path) -> tuple[int, bool]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
    try:
        return os.open(str(path), flags), True
    except OSError as e:
        if _strict_ssd_io():
            raise OSError(f"O_DIRECT open failed for {path}: {e}") from e
        return os.open(str(path), os.O_RDONLY), False


def _strip_odirect(fd: int) -> None:
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    if fl & os.O_DIRECT:
        fcntl.fcntl(fd, fcntl.F_SETFL, fl & ~os.O_DIRECT)


def pread_exact(fd: int, buf: AlignedBuffer, size: int, offset: int = 0, *, use_odirect: bool = True) -> None:
    try:
        buf.pread_into(fd, size, offset)
    except OSError:
        if use_odirect:
            if _strict_ssd_io():
                raise
            _strip_odirect(fd)
            buf.pread_into(fd, size, offset)
        else:
            raise


def read_file_odirect(path: Path) -> int:
    size = path.stat().st_size
    fd, is_direct = _open_odirect(path)
    try:
        if is_direct:
            read_size = align_up(size)
            buf = alloc_aligned(read_size)
            odirect_bytes = (size // ALIGN) * ALIGN
            pread_exact(fd, buf, odirect_bytes, 0, use_odirect=True)
            if size > odirect_bytes:
                pread_exact(fd, buf, ALIGN, odirect_bytes, use_odirect=True)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        else:
            buf = alloc_aligned(align_up(size))
            pread_exact(fd, buf, size, use_odirect=False)
    finally:
        os.close(fd)
    return size


def parse_expk_header(header: bytes) -> List[ExpkTensor]:
    magic, num_tensors = struct.unpack_from("<II", header, 0)
    if magic != EXPK_MAGIC:
        raise ValueError(f"bad EXPK magic: {magic:#x}")
    tensors: List[ExpkTensor] = []
    for i in range(num_tensors):
        base = 8 + i * EXPK_DESC_SIZE
        name = header[base : base + 32].split(b"\x00", 1)[0].decode()
        offset, size = struct.unpack_from("<QQ", header, base + 32)
        tensors.append(ExpkTensor(name, offset, size))
    return tensors


def _packed_tensor_order(tensors: Sequence[ExpkTensor]) -> List[ExpkTensor]:
    order = (
        "gate.qweight", "up.qweight", "gate.scales", "up.scales",
        "gate.zeros", "up.zeros", "down.qweight", "down.scales", "down.zeros",
    )
    by_name = {t.name: t for t in tensors}
    return [by_name[n] for n in order]


def load_packed_expert(
    path: Path,
    *,
    parallel: bool,
    max_workers: int = 9,
) -> int:
    """Read all tensor regions from one packed .bin; return bytes transferred."""
    fd, is_direct = _open_odirect(path)
    try:
        header_buf = alloc_aligned(ALIGN)
        pread_exact(fd, header_buf, ALIGN, 0, use_odirect=is_direct)
        tensors = _packed_tensor_order(
            parse_expk_header(header_buf.as_bytes(EXPK_HEADER_SIZE))
        )

        def read_one(t: ExpkTensor) -> int:
            read_size = align_up(t.size) if is_direct else t.size
            buf = alloc_aligned(read_size)
            pread_exact(fd, buf, read_size, t.offset, use_odirect=is_direct)
            return t.size

        if parallel:
            total = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [pool.submit(read_one, t) for t in tensors]
                for f in as_completed(futs):
                    total += f.result()
            return total

        return sum(read_one(t) for t in tensors)
    finally:
        os.close(fd)


def load_unpacked_expert(
    paths: Sequence[Path],
    *,
    parallel: bool,
    max_workers: int = 9,
) -> int:
    """Read all 9 unpacked .bin files; return bytes transferred."""

    def read_one(p: Path) -> int:
        return read_file_odirect(p)

    if parallel:
        total = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(read_one, p) for p in paths]
            for f in as_completed(futs):
                total += f.result()
        return total

    return sum(read_one(p) for p in paths)


def load_one_expert(
    bin_dir: str | Path,
    layer: int,
    expert: int,
    *,
    parallel_intra: bool,
) -> int:
    ep = expert_paths(bin_dir, layer, expert)
    if ep.format == "packed":
        return load_packed_expert(ep.packed_path, parallel=parallel_intra)
    return load_unpacked_expert(ep.unpacked_paths, parallel=parallel_intra)


def load_experts(
    bin_dir: str | Path,
    specs: Iterable[Tuple[int, int]],
    *,
    parallel_intra: bool,
    parallel_inter: bool,
    max_workers: int = 8,
) -> int:
    """
    Load multiple (layer, expert) pairs.

    parallel_intra: parallel pread within each expert (C++ default).
    parallel_inter: load multiple experts concurrently (NOT current C++ behavior).
    """
    items = list(specs)

    def load_item(le: Tuple[int, int]) -> int:
        layer, expert = le
        return load_one_expert(bin_dir, layer, expert, parallel_intra=parallel_intra)

    if parallel_inter:
        total = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(load_item, le) for le in items]
            for f in as_completed(futs):
                total += f.result()
        return total

    return sum(load_item(le) for le in items)


def drop_page_cache() -> bool:
    """Drop Linux page cache (requires write access to /proc/sys/vm/drop_caches)."""
    try:
        os.sync()
        with open("/proc/sys/vm/drop_caches", "w", encoding="utf-8") as f:
            f.write("3\n")
        print("[io_common] Dropped OS page cache.", flush=True)
        return True
    except OSError as e:
        print(f"[io_common] Could not drop page cache: {e}", flush=True)
        return False


def apply_strict_ssd_env(env: dict[str, str], *, enabled: bool = True) -> dict[str, str]:
    """Enable O_DIRECT-only expert loads in the C++ backend."""
    if enabled:
        env["HETEROPREDICT_STRICT_SSD_IO"] = "1"
    else:
        env.pop("HETEROPREDICT_STRICT_SSD_IO", None)
    return env


def apply_legacy_expert_io_env(
    env: dict[str, str],
    *,
    enabled: bool = True,
    sequential_inter: bool = True,
    sequential_intra: bool = False,
) -> dict[str, str]:
    """
    Jun-17 pre-batch expert I/O (per-expert decode loads + sequential prefetch thread).

    Umbrella: HETEROPREDICT_LEGACY_EXPERT_IO=1
    Granular: HETEROPREDICT_LEGACY_PER_EXPERT_CACHE, HETEROPREDICT_LEGACY_PREFETCH_LOADS

    sequential_intra=False (default): parallel pread within each expert load (original;
    overrides setup.sh default of SEQUENTIAL_EXPERT_IO=1).
    """
    keys = (
        "HETEROPREDICT_LEGACY_EXPERT_IO",
        "HETEROPREDICT_LEGACY_PER_EXPERT_CACHE",
        "HETEROPREDICT_LEGACY_PREFETCH_LOADS",
    )
    if enabled:
        env["HETEROPREDICT_LEGACY_EXPERT_IO"] = "1"
        if sequential_inter:
            env["HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO"] = "1"
        else:
            env.pop("HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO", None)
        if sequential_intra:
            env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "1"
        else:
            env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "0"
    else:
        for k in keys:
            env.pop(k, None)
    return env


def apply_modern_expert_io_env(
    env: dict[str, str],
    *,
    sequential_inter: bool = False,
    sequential_intra: bool = False,
) -> dict[str, str]:
    """Default post-batch I/O: batched loads + parallel inter-expert (unless overridden)."""
    for k in (
        "HETEROPREDICT_LEGACY_EXPERT_IO",
        "HETEROPREDICT_LEGACY_PER_EXPERT_CACHE",
        "HETEROPREDICT_LEGACY_PREFETCH_LOADS",
    ):
        env.pop(k, None)
    if sequential_inter:
        env["HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO"] = "1"
    else:
        env.pop("HETEROPREDICT_SEQUENTIAL_INTER_EXPERT_IO", None)
    if sequential_intra:
        env["HETEROPREDICT_SEQUENTIAL_EXPERT_IO"] = "1"
    else:
        env.pop("HETEROPREDICT_SEQUENTIAL_EXPERT_IO", None)
    return env


def strict_ssd_sweep_flags(*, enabled: bool = True) -> list[str]:
    """CLI flags for sweep_predict_cached_cache_metrics.py to minimize page-cache carryover."""
    if not enabled:
        return []
    return [
        "--drop-page-cache-before-first-run",
        "--drop-page-cache-between-runs",
    ]


def drop_page_cache_for_paths(paths: Iterable[Path]) -> None:
    """Best-effort per-file fadvise (complements global drop_caches)."""
    for p in paths:
        try:
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def timed_load(fn: Callable[[], int]) -> Tuple[float, int]:
    import time

    t0 = time.perf_counter()
    nbytes = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms, nbytes
