# backend/metrics/fourier_metrics.py
from __future__ import annotations

from typing import Dict, Any, Sequence, Tuple, Optional, List
import numpy as np

from ..fourier.filters import (
    FourierGrid2D,
    make_wavenumber_grid,
    fft2_field,
    radial_bin_spectrum,
)


def _infer_grid_from_meta(
    H: int,
    W: int,
    grid_meta: Optional[Dict[str, Any]] = None,
) -> FourierGrid2D:
    """grid_meta supports:
      - dx, dy
      - or Lx, Ly (then dx=Lx/W, dy=Ly/H)
      - angular (default True)
    """
    grid_meta = grid_meta or {}
    angular = bool(grid_meta.get("angular", True))

    if "dx" in grid_meta and "dy" in grid_meta:
        dx = float(grid_meta["dx"])
        dy = float(grid_meta["dy"])
    elif "Lx" in grid_meta and "Ly" in grid_meta:
        Lx = float(grid_meta["Lx"])
        Ly = float(grid_meta["Ly"])
        dx = Lx / float(W)
        dy = Ly / float(H)
    else:
        dx = float(grid_meta.get("dx", 1.0))
        dy = float(grid_meta.get("dy", 1.0))

    return make_wavenumber_grid(H=H, W=W, dx=dx, dy=dy, angular=angular)


def _fft_for_metric(
    x: np.ndarray,
    mean_mode: str = "none",
) -> np.ndarray:
    # 统一走 filters.fft2_field（支持 HW/CHW/HWC）
    return fft2_field(x, mean_mode=mean_mode)


def _infer_hw(x: np.ndarray) -> Tuple[int, int]:
    x = np.asarray(x)
    if x.ndim == 2:
        return int(x.shape[0]), int(x.shape[1])
    if x.ndim == 3:
        # allow (H,W,C) or (C,H,W)
        if x.shape[0] <= 8 and x.shape[1] > 8 and x.shape[2] > 8:
            return int(x.shape[1]), int(x.shape[2])  # (C,H,W)
        return int(x.shape[0]), int(x.shape[1])      # (H,W,C)
    raise ValueError(f"Unsupported x shape: {x.shape}")


def energy_spectrum(
    x_true: np.ndarray,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    grid_meta: Optional[Dict[str, Any]] = None,
    mean_mode: str = "none",
    *,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_zero_bin: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_centers, E_k, k_edges).

    Notes:
      - Unused.
      - This function no longer silently drops k≈0 for plotting.
        Use drop_zero_bin=True if the caller wants log-friendly output.
      - binning="log" enables geometric radial bins (requires k_min>0; if None a grid-based default is used).
    """
    x_true = np.asarray(x_true)
    H, W = _infer_hw(x_true)

    grid = _infer_grid_from_meta(H, W, grid_meta)
    F = _fft_for_metric(x_true, mean_mode=mean_mode)

    k_centers, E_k, k_edges = radial_bin_spectrum(
        F,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_zero_bin=drop_zero_bin,
        return_edges=True,
    )

    return np.asarray(k_centers), np.asarray(E_k), np.asarray(k_edges)


def fourier_radial_nrmse_curve(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    grid_meta: Optional[Dict[str, Any]] = None,
    mean_mode: str = "none",
    eps: float = 1e-12,
    *,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_zero_bin: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial NRMSE(k) curve based on energy in each k-bin.

    Returns:
      k_centers, E_true_k, E_err_k, nrmse_k
      where nrmse_k = sqrt(E_err_k / (E_true_k + eps)).

    Notes:
      - binning="log" produces better multi-scale interpretability on log axes.
      - drop_zero_bin is optional; keep False for scientifically faithful low-k coverage.
    """
    x_hat = np.asarray(x_hat)
    x_true = np.asarray(x_true)
    H, W = _infer_hw(x_true)

    grid = _infer_grid_from_meta(H, W, grid_meta)

    F_true = _fft_for_metric(x_true, mean_mode=mean_mode)
    F_err = _fft_for_metric(x_hat - x_true, mean_mode="none")  # error already mean-free notionally

    k_centers, E_true_k, _ = radial_bin_spectrum(
        F_true,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_zero_bin=drop_zero_bin,
        return_edges=True,
    )
    _, E_err_k, _ = radial_bin_spectrum(
        F_err,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_zero_bin=drop_zero_bin,
        return_edges=True,
    )

    E_true_k = np.asarray(E_true_k, dtype=np.float64)
    E_err_k = np.asarray(E_err_k, dtype=np.float64)
    nrmse_k = np.sqrt(E_err_k / (E_true_k + float(eps)))
    return np.asarray(k_centers), E_true_k, E_err_k, nrmse_k


def fourier_band_nrmse(
    *,
    k_centers: np.ndarray,
    E_true_k: np.ndarray,
    E_err_k: np.ndarray,
    full_edges: Sequence[float],
    band_names: Optional[Sequence[str]] = None,
    eps: float = 1e-12,
    monotone_enforce: bool = True,
) -> Dict[str, float]:
    """
    NEW SCHEMA ONLY.

    Compute NRMSE per radial-k band using already-binned spectra.

    Inputs:
      - k_centers: (B,) radial bin centers
      - E_true_k:  (B,) radial energy spectrum of true field
      - E_err_k:   (B,) radial energy spectrum of error field (x_hat - x_true)
      - full_edges: band edges INCLUDING endpoints, length = n_bands + 1
          e.g. [0.0, k1, k2, ..., kN]  (monotone increasing)
        Note: membership uses [edge_i, edge_{i+1}) bins.

      - band_names: optional names per band, length must equal n_bands
      - eps: numerical stabilizer
      - monotone_enforce: if True, enforce nondecreasing envelope across bands (low->high k)

    Returns:
      dict {band_name: nrmse_band}
    """
    k_centers = np.asarray(k_centers, dtype=np.float64)
    E_true_k = np.asarray(E_true_k, dtype=np.float64)
    E_err_k = np.asarray(E_err_k, dtype=np.float64)

    if k_centers.ndim != 1 or E_true_k.ndim != 1 or E_err_k.ndim != 1:
        raise ValueError("k_centers, E_true_k, E_err_k must be 1D arrays.")
    if not (k_centers.size == E_true_k.size == E_err_k.size):
        raise ValueError("k_centers, E_true_k, E_err_k must have the same length.")
    if k_centers.size == 0:
        return {}

    edges = [float(v) for v in full_edges]
    if len(edges) < 2:
        raise ValueError("full_edges must have length >= 2.")
    if any(not np.isfinite(v) for v in edges):
        raise ValueError("full_edges contains non-finite values.")
    if any(edges[i + 1] <= edges[i] for i in range(len(edges) - 1)):
        raise ValueError("full_edges must be strictly increasing.")

    n_bands = len(edges) - 1
    if band_names is None:
        band_names = [f"band_{i}" for i in range(n_bands)]
    else:
        band_names = list(band_names)
        if len(band_names) != n_bands:
            raise ValueError(f"band_names length must be {n_bands}, got {len(band_names)}")

    out_vals: List[float] = []
    for i in range(n_bands):
        lo, hi = edges[i], edges[i + 1]
        sel = (k_centers >= lo) & (k_centers < hi)
        num = float(np.sum(E_err_k[sel]))
        den = float(np.sum(E_true_k[sel]))
        out_vals.append(float(np.sqrt(num / (den + float(eps)))))

    if monotone_enforce and len(out_vals) > 0:
        out_vals = np.maximum.accumulate(np.asarray(out_vals, dtype=np.float64)).tolist()

    return {band_names[i]: float(out_vals[i]) for i in range(n_bands)}


def kstar_from_radial_curve(
    k_centers: np.ndarray,
    nrmse_k: np.ndarray,
    threshold: float = 1.0,
    monotone_enforce: bool = True,
    *,
    method: str = "last_ok",
) -> float:
    """Compute k* as a cutoff wavenumber based on NRMSE(k) <= threshold.

    method:
      - "last_ok": (legacy) return the largest sampled k with NRMSE<=thr
      - "crossing_interp": estimate the last crossing point with linear interpolation

    monotone_enforce:
      - True: enforce nondecreasing NRMSE envelope via cumulative max,
              to avoid non-physical dips producing fake larger k*.
    """
    k_centers = np.asarray(k_centers, dtype=np.float64)
    nrmse_k = np.asarray(nrmse_k, dtype=np.float64)
    if k_centers.ndim != 1 or nrmse_k.ndim != 1 or k_centers.size != nrmse_k.size:
        raise ValueError("k_centers and nrmse_k must be 1D arrays of same length.")
    if k_centers.size == 0:
        return 0.0

    thr = float(threshold)
    curve = nrmse_k
    if monotone_enforce:
        curve = np.maximum.accumulate(curve)

    method = str(method).lower().strip()
    if method not in ("last_ok", "crossing_interp"):
        raise ValueError(f"method must be 'last_ok' or 'crossing_interp', got {method}")

    ok = curve <= thr
    if not np.any(ok):
        return 0.0

    if method == "last_ok":
        idx = int(np.where(ok)[0][-1])
        return float(k_centers[idx])

    # crossing_interp:
    # find the last index where ok is True; then interpolate to the next point where curve>thr (if any).
    i_last_ok = int(np.where(ok)[0][-1])
    if i_last_ok == (k_centers.size - 1):
        return float(k_centers[i_last_ok])

    # We assume curve is nondecreasing if monotone_enforce=True; if False, this is still a reasonable
    # local interpolation near the last ok point.
    k0 = float(k_centers[i_last_ok])
    y0 = float(curve[i_last_ok])
    k1 = float(k_centers[i_last_ok + 1])
    y1 = float(curve[i_last_ok + 1])

    if not (np.isfinite(k0) and np.isfinite(k1) and np.isfinite(y0) and np.isfinite(y1)):
        return float(k0)

    # if y1 is still <= thr (rare with nondecreasing), fall back to scanning forward
    if y1 <= thr:
        j = i_last_ok + 1
        while j < k_centers.size and float(curve[j]) <= thr:
            j += 1
        if j >= k_centers.size:
            return float(k_centers[-1])
        k0 = float(k_centers[j - 1])
        y0 = float(curve[j - 1])
        k1 = float(k_centers[j])
        y1 = float(curve[j])

    # linear interpolation to y=thr
    if np.isclose(y1, y0):
        return float(k0)

    t = (thr - y0) / (y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))
    return float(k0 + t * (k1 - k0))
