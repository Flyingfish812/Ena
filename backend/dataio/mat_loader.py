# backend/dataio/mat_loader.py

"""MAT v7.3 (HDF5) loader for data/sst_weekly.mat.

We normalize to float32 [T,H,W,C].

Assumptions (as in scripts/analyze_new_datasets.py):
  - key name is cfg.mat_key (default "sst")
  - each frame is a vector of length 64800 (= 180*360)
  - spatial transform is cfg.sst_reshape == "360x180_rot90":
      img = vec.reshape(360,180)
      img = np.rot90(img, k=1)  # -> (180,360)
  - NaNs are filled per-frame using finite-value mean (cfg.sst_fill_nan)

This keeps the same H/W convention as your analysis script.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import json
import numpy as np

from backend.config.schemas import DataConfig
from backend.dataio.io_utils import ensure_dir


_VEC_LEN = 180 * 360


def _resolve_mat_path(cfg: DataConfig) -> Path:
    if cfg.path is not None:
        return Path(cfg.path)
    if cfg.nc_path is not None:
        return Path(cfg.nc_path)
    raise ValueError("mat_sst loader requires DataConfig.path (or legacy nc_path)")


def _cache_key(cfg: DataConfig) -> str:
    maxf = "NA" if cfg.sst_max_frames is None else str(int(cfg.sst_max_frames))
    return f"sst_mat_{cfg.mat_key}_fill{cfg.sst_fill_nan}_reshape{cfg.sst_reshape}_max{maxf}"


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
    mm = np.memmap(mmap_path, mode="r", dtype="float32", shape=shape)
    return np.asarray(mm)


def _save_cache(cfg: DataConfig, X: np.ndarray) -> None:
    if cfg.cache_dir is None:
        return

    cache_dir = Path(cfg.cache_dir)
    ensure_dir(cache_dir)

    stem = _cache_key(cfg)
    mmap_path = cache_dir / f"{stem}.float32.mmap"
    meta_path = cache_dir / f"{stem}.meta.json"

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


def _as_time_by_vec(arr: np.ndarray) -> np.ndarray:
    """Convert various HDF5-loaded shapes into [T, V] where V=64800."""
    if arr.ndim == 1:
        if arr.size != _VEC_LEN:
            raise ValueError(f"Unexpected 1D sst size {arr.size}, expected {_VEC_LEN}")
        return arr.reshape(1, _VEC_LEN)

    if arr.ndim == 2:
        if arr.shape[0] == _VEC_LEN:
            return arr.T
        if arr.shape[1] == _VEC_LEN:
            return arr
        raise ValueError(f"Unexpected 2D sst shape {arr.shape}, expected one dim == {_VEC_LEN}")

    if arr.ndim == 3:
        # Try interpret as (A,B,T) with A*B=64800
        a, b, t = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        if a * b == _VEC_LEN:
            flat = arr.reshape(a * b, t).T
            return flat
        # Or (T,A,B)
        t, a, b = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        if a * b == _VEC_LEN:
            flat = arr.reshape(t, a * b)
            return flat

    raise ValueError(f"Unsupported sst array ndim={arr.ndim}, shape={arr.shape}")


def _fill_nan(vecs: np.ndarray, strategy: str) -> np.ndarray:
    vecs = vecs.astype(np.float32, copy=False)
    if strategy == "zero":
        out = vecs.copy()
        out[~np.isfinite(out)] = 0.0
        return out

    if strategy == "global_mean":
        finite = vecs[np.isfinite(vecs)]
        mean = float(np.mean(finite)) if finite.size else 0.0
        out = vecs.copy()
        out[~np.isfinite(out)] = mean
        return out

    if strategy == "per_frame_mean":
        out = vecs.copy()
        for i in range(out.shape[0]):
            row = out[i]
            mask = np.isfinite(row)
            mean = float(np.mean(row[mask])) if np.any(mask) else 0.0
            row[~mask] = mean
        return out

    raise ValueError(f"Unknown sst_fill_nan strategy: {strategy!r}")


def _vec_to_hw(vec: np.ndarray, reshape_mode: str) -> np.ndarray:
    if vec.size != _VEC_LEN:
        raise ValueError(f"Expected vec length {_VEC_LEN}, got {vec.size}")

    if reshape_mode != "360x180_rot90":
        raise ValueError(f"Unsupported sst_reshape={reshape_mode!r} (only '360x180_rot90' supported)")

    img = vec.reshape(360, 180)
    img = np.rot90(img, k=1)
    # -> (180,360)
    return img


def load_raw_mat_sst(cfg: DataConfig) -> np.ndarray:
    cached = _try_load_cache(cfg)
    if cached is not None:
        return cached

    mat_path = _resolve_mat_path(cfg)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    try:
        import h5py  # type: ignore
    except Exception as e:
        raise ImportError("h5py is required to load MAT v7.3 (.mat) datasets") from e

    key = str(cfg.mat_key)

    def _read_time_by_vec(dset, *, max_frames: int | None) -> np.ndarray:
        """Read dataset into [T, V] with V=64800, using slicing (no full-load)."""
        shape = tuple(int(x) for x in dset.shape)

        if len(shape) == 1:
            if int(shape[0]) != _VEC_LEN:
                raise ValueError(f"Unexpected 1D sst size {shape[0]}, expected {_VEC_LEN}")
            arr = np.asarray(dset[:], dtype=np.float32)
            return arr.reshape(1, _VEC_LEN)

        if len(shape) == 2:
            a, b = int(shape[0]), int(shape[1])
            if b == _VEC_LEN:
                T = a
                T_lim = T if max_frames is None else min(T, int(max_frames))
                # Chunked read along time axis
                out = np.empty((T_lim, _VEC_LEN), dtype=np.float32)
                chunk = 64
                for i in range(0, T_lim, chunk):
                    j = min(T_lim, i + chunk)
                    out[i:j] = np.asarray(dset[i:j, :], dtype=np.float32)
                return out
            if a == _VEC_LEN:
                T = b
                T_lim = T if max_frames is None else min(T, int(max_frames))
                out = np.empty((T_lim, _VEC_LEN), dtype=np.float32)
                chunk = 64
                for i in range(0, T_lim, chunk):
                    j = min(T_lim, i + chunk)
                    # dset[:, i:j] -> (V, chunk) then transpose
                    out[i:j] = np.asarray(dset[:, i:j], dtype=np.float32).T
                return out
            raise ValueError(f"Unexpected 2D sst shape {shape}, expected one dim == {_VEC_LEN}")

        # Fallback: read as ndarray then convert (may be expensive)
        raw = np.asarray(dset, dtype=np.float32)
        vecs = _as_time_by_vec(raw)
        if max_frames is not None:
            vecs = vecs[: int(max_frames)]
        return vecs

    with h5py.File(str(mat_path), "r") as f:
        if key not in f:
            raise KeyError(f"Key {key!r} not found in MAT file: {mat_path}")
        dset = f[key]
        vecs = _read_time_by_vec(dset, max_frames=cfg.sst_max_frames)

    vecs = _fill_nan(vecs, str(cfg.sst_fill_nan))

    T = int(vecs.shape[0])
    H, W = 180, 360
    X = np.empty((T, H, W, 1), dtype=np.float32)

    reshape_mode = str(cfg.sst_reshape)
    for t in range(T):
        X[t, :, :, 0] = _vec_to_hw(vecs[t], reshape_mode)

    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite values remain after fill_nan={cfg.sst_fill_nan}: {mat_path}")

    _save_cache(cfg, X)
    return X
