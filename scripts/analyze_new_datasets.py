#!/usr/bin/env python3
"""Compact pre-analysis for newly added datasets (standalone, no backend imports).

What it does (minimal output):
1) Fully reads the two target payloads and prints:
   - shape/dtype
   - whether NaNs exist
2) Randomly samples and saves a few visualizations:
   - 2D_rdb_NA_NA.h5: pick (group i, time j) -> (128,128,1) slice, plot
   - sst_weekly.mat: pick time i -> length-64800 vector, fill NaNs with mean,
     reshape to (180,360), flip y-axis, plot

Notes:
- The HDF5 file is ~6GB; to "fully read" without requiring huge RAM, this script
  streams the data into a numpy memmap cache file under artifacts/.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LoadResult:
    name: str
    shape: tuple[int, ...]
    dtype: str
    has_nan: bool
    cache_path: str | None = None


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}" if u != "B" else f"{int(x)} {u}"
        x /= 1024.0
    return f"{n} B"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_2d_rdb_h5_to_memmap(
    h5_path: str,
    *,
    cache_dir: str,
    force: bool,
    chunk_groups: int = 4,
    chunk_time: int = 10,
) -> tuple[LoadResult, np.memmap]:
    """Fully read 2D_rdb_NA_NA.h5 into a memmap cache (disk-backed array).

    Expected logical payload: (1000,101,128,128,1) float32.
    The file stores 1000 groups like '0000'..'0999', each with dataset 'data'.
    """

    import h5py

    _ensure_dir(cache_dir)
    cache_path = os.path.join(cache_dir, "2D_rdb_NA_NA.float32.mmap")
    meta_path = os.path.join(cache_dir, "2D_rdb_NA_NA.meta.json")
    shape = (1000, 101, 128, 128, 1)
    dtype = np.float32

    cache_exists = os.path.exists(cache_path)
    meta_exists = os.path.exists(meta_path)

    # If a cache file exists without a meta file, it may be an incomplete/old cache.
    # In that case, rebuild automatically for correctness.
    if cache_exists and not force and not meta_exists:
        force = True

    mm = np.memmap(cache_path, mode="w+" if force or not cache_exists else "r+", dtype=dtype, shape=shape)

    has_nan = False

    # If cache exists and not force, we still treat it as already loaded.
    # We prefer reading a small meta file instead of scanning the full payload.
    if os.path.exists(cache_path) and not force:
        meta = _read_json(meta_path)
        if not bool(meta.get("complete", False)):
            force = True
        else:
            has_nan = bool(meta.get("has_nan", False))
            return LoadResult("2D_rdb_NA_NA.h5:data", shape, str(dtype), has_nan, cache_path=cache_path), mm

    with h5py.File(h5_path, "r") as f:
        keys = sorted(list(f.keys()))
        if len(keys) != 1000:
            raise ValueError(f"Expected 1000 groups in {h5_path}, got {len(keys)}")

        for g0 in range(0, 1000, chunk_groups):
            g1 = min(1000, g0 + chunk_groups)
            for gi in range(g0, g1):
                k = f"{gi:04d}"
                dset = f[k]["data"]
                if tuple(int(x) for x in dset.shape) != (101, 128, 128, 1):
                    raise ValueError(f"Unexpected shape for {k}/data: {dset.shape}")

                # stream along time axis
                for t0 in range(0, 101, chunk_time):
                    t1 = min(101, t0 + chunk_time)
                    block = np.asarray(dset[t0:t1, ...], dtype=dtype)
                    if not has_nan and np.isnan(block).any():
                        has_nan = True
                    mm[gi, t0:t1, ...] = block

        mm.flush()

    _write_json(
        meta_path,
        {
            "name": "2D_rdb_NA_NA.h5:data",
            "shape": list(shape),
            "dtype": "float32",
            "has_nan": has_nan,
            "complete": True,
            "source_h5": os.path.abspath(h5_path),
        },
    )

    return LoadResult("2D_rdb_NA_NA.h5:data", shape, str(dtype), has_nan, cache_path=cache_path), mm


def load_sst_weekly_mat(path: str) -> tuple[LoadResult, np.ndarray]:
    """Fully read sst from MATLAB v7.3 .mat (HDF5) file."""

    import h5py

    with h5py.File(path, "r") as f:
        if "sst" not in f:
            raise KeyError("Expected dataset 'sst' in sst_weekly.mat")
        sst = np.asarray(f["sst"], dtype=np.float32)

    has_nan = bool(np.isnan(sst).any())
    return LoadResult("sst_weekly.mat:sst", tuple(int(x) for x in sst.shape), str(sst.dtype), has_nan), sst


def _plot_2d_field(arr_128_128_1: np.ndarray, *, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.asarray(arr_128_128_1)
    if a.shape != (128, 128, 1):
        raise ValueError(f"Expected (128,128,1), got {a.shape}")
    img = a[:, :, 0]

    plt.figure(figsize=(4.5, 4.0), dpi=140)
    plt.imshow(img, origin="lower", aspect="equal", cmap="RdBu_r", interpolation="bilinear")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_sst_vector(vec_64800: np.ndarray, *, title: str, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = np.asarray(vec_64800, dtype=np.float32)
    if v.size != 64800:
        raise ValueError(f"Expected length 64800, got {v.size}")

    finite = v[np.isfinite(v)]
    fill = float(finite.mean()) if finite.size else 0.0
    v2 = v.copy()
    v2[~np.isfinite(v2)] = fill

    # 64800 = 180 * 360.
    # The stored vector layout matches reshape (360,180) but needs an extra
    # 90° CCW rotation to display as a (180,360) world-map grid:
    #   horizontal=360 (lon), vertical=180 (lat).
    img = v2.reshape(360, 180)
    img = np.rot90(img, k=1)

    plt.figure(figsize=(6.0, 3.5), dpi=140)
    plt.imshow(img, aspect="auto", cmap="RdBu_r", interpolation="bilinear")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Data directory relative to repo root")
    ap.add_argument("--out-dir", default="artifacts/preanalysis_figs", help="Where to save figures")
    ap.add_argument("--cache-dir", default="artifacts/preanalysis_cache", help="Where to save memmap cache")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--num-samples", type=int, default=6, help="How many random samples to plot for each dataset")
    ap.add_argument("--force-reload", action="store_true", help="Rebuild cache and re-read large HDF5 payload")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, args.data_dir)
    out_dir = os.path.join(repo_root, args.out_dir)
    cache_dir = os.path.join(repo_root, args.cache_dir)

    _ensure_dir(out_dir)
    _ensure_dir(cache_dir)

    h5_path = os.path.join(data_dir, "2D_rdb_NA_NA.h5")
    mat_path = os.path.join(data_dir, "sst_weekly.mat")

    rng = np.random.default_rng(args.seed)

    print("== Preanalysis (compact) ==")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)

    # 1) Full read + shape + NaN existence
    print(f"2D_rdb_NA_NA.h5 file_size={_fmt_bytes(os.path.getsize(h5_path))}")
    rdb_info, rdb_mm = load_2d_rdb_h5_to_memmap(h5_path, cache_dir=cache_dir, force=args.force_reload)
    print(
        f"  payload={rdb_info.name} shape={rdb_info.shape} dtype={rdb_info.dtype} has_nan={rdb_info.has_nan}"
    )
    if rdb_info.cache_path:
        print(f"  cache_memmap={rdb_info.cache_path}")

    print(f"sst_weekly.mat file_size={_fmt_bytes(os.path.getsize(mat_path))}")
    sst_info, sst = load_sst_weekly_mat(mat_path)
    print(
        f"  payload={sst_info.name} shape={sst_info.shape} dtype={sst_info.dtype} has_nan={sst_info.has_nan}"
    )

    # 2) Random sample plots
    n = max(1, int(args.num_samples))
    print(f"Saving figures -> {out_dir}")

    for s in range(n):
        gi = int(rng.integers(0, 1000))
        tj = int(rng.integers(0, 101))
        out_path = os.path.join(out_dir, f"rdb_g{gi:04d}_t{tj:03d}.png")
        _plot_2d_field(rdb_mm[gi, tj, :, :, :], title=f"2D_rdb: group={gi}, t={tj}", out_path=out_path)

    # sst rows are time-like index 0..1913
    for s in range(n):
        ti = int(rng.integers(0, sst.shape[0]))
        out_path = os.path.join(out_dir, f"sst_t{ti:04d}.png")
        _plot_sst_vector(sst[ti], title=f"SST weekly: t={ti} (NaNs filled with mean, rot90 CCW)", out_path=out_path)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
