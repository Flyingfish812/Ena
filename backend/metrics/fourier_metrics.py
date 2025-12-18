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


def energy_spectrum(
    x_true: np.ndarray,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    grid_meta: Optional[Dict[str, Any]] = None,
    mean_mode: str = "none",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (k_centers, E_k, k_edges)."""
    x_true = np.asarray(x_true)
    if x_true.ndim == 2:
        H, W = x_true.shape
    elif x_true.ndim == 3:
        # allow (H,W,C) or (C,H,W); grid sizes are last two spatial dims
        if x_true.shape[0] <= 8 and x_true.shape[1] > 8 and x_true.shape[2] > 8:
            H, W = x_true.shape[1], x_true.shape[2]  # (C,H,W)
        else:
            H, W = x_true.shape[0], x_true.shape[1]  # (H,W,C)
    else:
        raise ValueError(f"Unsupported x_true shape: {x_true.shape}")

    grid = _infer_grid_from_meta(H, W, grid_meta)
    F = _fft_for_metric(x_true, mean_mode=mean_mode)
    k_centers, E_k, k_edges = radial_bin_spectrum(F, grid, num_bins=num_bins, k_max=k_max, return_edges=True)

    # remove k=0 bin for log-scale friendly plots
    k_centers = np.asarray(k_centers)
    E_k = np.asarray(E_k)
    k_edges = np.asarray(k_edges)

    if k_centers.size > 0 and np.isclose(k_centers[0], 0.0):
        # drop the first bin (k=0) to make loglog plots stable
        k_centers = k_centers[1:]
        E_k = E_k[1:]
        # edges length = bins+1; drop the first edge as well
        if k_edges.size == (k_centers.size + 2):  # original edges
            k_edges = k_edges[1:]
        elif k_edges.size == (k_centers.size + 1):
            # already consistent, no-op
            pass

    return k_centers, E_k, k_edges


def fourier_radial_nrmse_curve(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    grid_meta: Optional[Dict[str, Any]] = None,
    mean_mode: str = "none",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial NRMSE(k) curve based on energy in each k-bin.

    Returns:
      k_centers, E_true_k, E_err_k, nrmse_k
      where nrmse_k = sqrt(E_err_k / (E_true_k + eps)).
    """
    x_hat = np.asarray(x_hat)
    x_true = np.asarray(x_true)

    # infer spatial size
    if x_true.ndim == 2:
        H, W = x_true.shape
    elif x_true.ndim == 3:
        if x_true.shape[0] <= 8 and x_true.shape[1] > 8 and x_true.shape[2] > 8:
            H, W = x_true.shape[1], x_true.shape[2]
        else:
            H, W = x_true.shape[0], x_true.shape[1]
    else:
        raise ValueError(f"Unsupported x_true shape: {x_true.shape}")

    grid = _infer_grid_from_meta(H, W, grid_meta)

    F_true = _fft_for_metric(x_true, mean_mode=mean_mode)
    F_err = _fft_for_metric(x_hat - x_true, mean_mode="none")  # error already mean-free notionally

    k_centers, E_true_k, _ = radial_bin_spectrum(F_true, grid, num_bins=num_bins, k_max=k_max, return_edges=True)
    _, E_err_k, _ = radial_bin_spectrum(F_err, grid, num_bins=num_bins, k_max=k_max, return_edges=True)

    nrmse_k = np.sqrt(E_err_k / (E_true_k + float(eps)))
    return k_centers, E_true_k, E_err_k, nrmse_k


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
) -> float:
    """Compute k* as the largest k such that NRMSE(k) <= threshold.

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
        curve = np.maximum.accumulate(curve)  # make it nondecreasing vs k

    ok = curve <= thr
    if not np.any(ok):
        return 0.0

    idx = int(np.where(ok)[0][-1])
    return float(k_centers[idx])
