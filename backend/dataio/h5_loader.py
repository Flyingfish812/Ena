# backend/dataio/h5_loader.py

"""HDF5 loader for data/2D_rdb_NA_NA.h5.

Dataset structure (as in scripts/analyze_new_datasets.py):
  root/
    "0000".."0999" (groups)
      data: float32 [101, 128, 128, 1]

For feasibility we do *sampling*:
  - select K groups starting at `h5_rdb_group_start` with step `h5_rdb_group_step`
  - for each group, pick `h5_rdb_frames_per_group` frames using linspace indices

Returns X_thwc float32 [T,H,W,C], where T = K * frames_per_group.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np

from backend.config.schemas import DataConfig
from backend.dataio.io_utils import ensure_dir


def _resolve_h5_path(cfg: DataConfig) -> Path:
    if cfg.path is not None:
        return Path(cfg.path)
    if cfg.nc_path is not None:
        # allow legacy field usage
        return Path(cfg.nc_path)
    raise ValueError("h5_rdb loader requires DataConfig.path (or legacy nc_path)")


def _cache_key(cfg: DataConfig) -> str:
    # Make a stable filename component for sampled result.
    return (
        f"rdb_h5_"
        f"g{int(cfg.h5_rdb_group_start)}_"
        f"step{int(cfg.h5_rdb_group_step)}_"
        f"K{int(cfg.h5_rdb_group_count)}_"
        f"F{int(cfg.h5_rdb_frames_per_group)}_"
        f"{str(cfg.h5_rdb_frame_sampling)}"
    )


def _try_load_cache(cfg: DataConfig) -> np.ndarray | None:
    if cfg.cache_dir is None:
        return None

    cache_dir = Path(cfg.cache_dir)
    ensure_dir(cache_dir)

    stem = _cache_key(cfg)
    mmap_path = cache_dir / f"{stem}.float32.mmap"
    meta_path = cache_dir / f"{stem}.meta.json"

    if not (mmap_path.exists() and meta_path.exists()):
        return None

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    shape = tuple(int(x) for x in meta["shape"])
    dtype = str(meta.get("dtype", "float32"))

    mm = np.memmap(mmap_path, mode="r", dtype=dtype, shape=shape)
    # Return an ndarray view (still backed by memmap; caller can .copy() if needed)
    return np.asarray(mm)


def _save_cache(cfg: DataConfig, X: np.ndarray) -> None:
    if cfg.cache_dir is None:
        return

    cache_dir = Path(cfg.cache_dir)
    ensure_dir(cache_dir)

    stem = _cache_key(cfg)
    mmap_path = cache_dir / f"{stem}.float32.mmap"
    meta_path = cache_dir / f"{stem}.meta.json"

    # Write memmap
    mm = np.memmap(mmap_path, mode="w+", dtype="float32", shape=X.shape)
    mm[:] = X.astype(np.float32, copy=False)
    mm.flush()

    meta: Dict[str, Any] = {
        "shape": list(X.shape),
        "dtype": "float32",
        "data_cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _select_groups(all_group_names: list[str], *, start: int, step: int, count: int) -> list[str]:
    # Prefer numeric sort if names are digits.
    def _to_int(name: str) -> int:
        try:
            return int(name)
        except Exception:
            return 10**18

    names_sorted = sorted(all_group_names, key=_to_int)

    if count <= 0:
        raise ValueError(f"h5_rdb_group_count must be positive, got {count}")
    if step <= 0:
        raise ValueError(f"h5_rdb_group_step must be positive, got {step}")
    if start < 0:
        raise ValueError(f"h5_rdb_group_start must be >=0, got {start}")

    picked = []
    idx = int(start)
    while idx < len(names_sorted) and len(picked) < int(count):
        picked.append(names_sorted[idx])
        idx += int(step)

    if len(picked) < int(count):
        raise ValueError(
            f"Not enough groups in H5: requested {count} but found only {len(picked)} (total={len(names_sorted)})."
        )

    return picked


def _select_frame_indices(n_frames: int, frames_per_group: int, sampling: str) -> np.ndarray:
    if frames_per_group <= 0:
        raise ValueError(f"h5_rdb_frames_per_group must be positive, got {frames_per_group}")

    if sampling != "linspace":
        raise ValueError(f"Unsupported h5_rdb_frame_sampling={sampling!r} (only 'linspace' supported)")

    if n_frames <= 0:
        raise ValueError(f"Invalid frame count {n_frames}")

    if frames_per_group >= n_frames:
        return np.arange(n_frames, dtype=np.int64)

    # Equally spaced indices, inclusive ends.
    idx = np.linspace(0, n_frames - 1, frames_per_group, dtype=np.int64)
    # Guarantee unique + sorted
    idx = np.unique(idx)
    if idx.size != frames_per_group:
        # If duplicates happen due to small n_frames, pad by adding missing indices.
        missing = [i for i in range(n_frames) if i not in set(idx.tolist())]
        need = frames_per_group - int(idx.size)
        idx = np.concatenate([idx, np.array(missing[:need], dtype=np.int64)], axis=0)
        idx = np.sort(idx)
    return idx


def load_raw_h5_rdb(cfg: DataConfig) -> np.ndarray:
    cached = _try_load_cache(cfg)
    if cached is not None:
        return cached

    h5_path = _resolve_h5_path(cfg)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise ImportError("h5py is required to load .h5 datasets") from e

    dataset_key = str(cfg.h5_rdb_dataset_key)

    with h5py.File(str(h5_path), "r") as f:
        all_groups = [k for k in f.keys()]
        groups = _select_groups(
            all_groups,
            start=int(cfg.h5_rdb_group_start),
            step=int(cfg.h5_rdb_group_step),
            count=int(cfg.h5_rdb_group_count),
        )

        # Inspect first group to infer shape
        first = f[groups[0]][dataset_key]
        arr0 = np.asarray(first)
        if arr0.ndim != 4:
            raise ValueError(f"Expected dataset '{dataset_key}' to be 4D [T,H,W,C], got {arr0.shape}")

        n_frames, H, W, C = (int(arr0.shape[0]), int(arr0.shape[1]), int(arr0.shape[2]), int(arr0.shape[3]))
        frame_idx = _select_frame_indices(n_frames, int(cfg.h5_rdb_frames_per_group), str(cfg.h5_rdb_frame_sampling))

        T_out = int(len(groups)) * int(frame_idx.size)
        X = np.empty((T_out, H, W, C), dtype=np.float32)

        out_i = 0
        for gname in groups:
            dset = f[gname][dataset_key]
            # Read selected frames only
            block = np.asarray(dset[frame_idx, ...], dtype=np.float32)
            if block.shape != (frame_idx.size, H, W, C):
                raise ValueError(f"Unexpected block shape from group {gname}: {block.shape}")

            X[out_i : out_i + block.shape[0]] = block
            out_i += block.shape[0]

        if out_i != T_out:
            raise RuntimeError(f"Internal error: filled {out_i} frames, expected {T_out}")

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values detected in sampled H5 data: {h5_path}")

    _save_cache(cfg, X)
    return X
