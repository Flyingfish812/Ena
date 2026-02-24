# backend/dataio/loader.py

"""Unified raw-data loader.

All raw datasets are normalized to:
  X_thwc: np.ndarray, float32, shape [T, H, W, C]

This is the contract expected by Level-1 POD builder and downstream L2/L3 pipelines.

We keep the legacy netCDF loader (`load_raw_nc`) unchanged and route based on
`DataConfig.source`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np

from backend.config.schemas import DataConfig
from backend.dataio.nc_loader import load_raw_nc


def _resolve_path(cfg: DataConfig) -> Path:
    if cfg.path is not None:
        return Path(cfg.path)
    if cfg.nc_path is not None:
        return Path(cfg.nc_path)
    raise ValueError("DataConfig missing both 'path' and 'nc_path'")


def describe_source(cfg: DataConfig) -> str:
    src = str(getattr(cfg, "source", "netcdf") or "netcdf")
    p = _resolve_path(cfg)
    return f"{src}:{p}"


def load_raw(cfg: DataConfig) -> np.ndarray:
    """Load raw data and return float32 [T,H,W,C]."""
    source = str(getattr(cfg, "source", "netcdf") or "netcdf").lower().strip()

    if source in ("netcdf", "nc"):
        # Keep legacy field usage inside nc_loader.
        p = _resolve_path(cfg)
        nc_cfg = cfg if cfg.nc_path is not None else replace(cfg, nc_path=p)
        X = load_raw_nc(nc_cfg)
    elif source in ("h5_rdb", "rdb_h5", "h5"):
        from backend.dataio.h5_loader import load_raw_h5_rdb

        X = load_raw_h5_rdb(cfg)
    elif source in ("mat_sst", "sst_mat", "mat"):
        from backend.dataio.mat_loader import load_raw_mat_sst

        X = load_raw_mat_sst(cfg)
    else:
        raise ValueError(f"Unknown DataConfig.source={source!r}. Supported: netcdf, h5_rdb, mat_sst")

    X = np.asarray(X)
    if X.ndim != 4:
        raise ValueError(f"Expected raw data with 4 dims [T,H,W,C], got shape {X.shape}")

    X = X.astype(np.float32, copy=False)

    if not np.isfinite(X).all():
        bad = np.size(X) - int(np.isfinite(X).sum())
        raise ValueError(f"Raw data contains non-finite values ({bad} entries): {describe_source(cfg)}")

    return X
