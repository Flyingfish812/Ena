from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Literal
import numpy as np

from ..fourier.filters import FourierGrid2D, make_wavenumber_grid, fft2_field


# ----------------------------
# Grid / bin helpers (no dependency on fourier_metrics.py to avoid circular imports)
# ----------------------------

def infer_grid_from_meta(
    H: int,
    W: int,
    grid_meta: Optional[Dict[str, Any]] = None,
) -> FourierGrid2D:
    """
    grid_meta supports:
      - dx, dy
      - or Lx, Ly (then dx=Lx/W, dy=Ly/H)
      - angular (default True to match existing fourier_metrics.py behavior)
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


def _infer_default_k_min_from_grid(grid: FourierGrid2D) -> float:
    # A safe positive minimum for log bins: smallest nonzero |kx| or |ky|
    kx = np.asarray(grid.kx, dtype=np.float64)
    ky = np.asarray(grid.ky, dtype=np.float64)
    vals = []
    if kx.size:
        v = np.min(np.abs(kx[np.abs(kx) > 0]))
        if np.isfinite(v) and v > 0:
            vals.append(float(v))
    if ky.size:
        v = np.min(np.abs(ky[np.abs(ky) > 0]))
        if np.isfinite(v) and v > 0:
            vals.append(float(v))
    if len(vals) == 0:
        return 1e-6
    return float(min(vals))


def _make_k_edges(
    grid: FourierGrid2D,
    *,
    num_bins: int,
    k_max: Optional[float],
    binning: str,
    k_min: Optional[float],
) -> np.ndarray:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be >0, got {num_bins}")
    k = np.asarray(grid.k, dtype=np.float64)
    if k_max is None:
        k_max = float(np.max(k))
    k_max = float(k_max)
    if not np.isfinite(k_max) or k_max <= 0:
        raise ValueError(f"Invalid k_max={k_max}")

    binning = str(binning).lower().strip()
    if binning == "linear":
        edges = np.linspace(0.0, k_max, num_bins + 1, dtype=np.float64)
    elif binning == "log":
        if k_min is None:
            k_min = _infer_default_k_min_from_grid(grid)
        k_min = float(k_min)
        if not np.isfinite(k_min) or k_min <= 0:
            raise ValueError(f"log-binning requires k_min>0, got k_min={k_min}")
        if k_min >= k_max:
            raise ValueError(f"log-binning requires k_min<k_max, got k_min={k_min}, k_max={k_max}")
        edges = np.geomspace(k_min, k_max, num_bins + 1).astype(np.float64)
    else:
        raise ValueError(f"binning must be 'linear' or 'log', got {binning}")

    return edges


def radial_profile_from_map(
    Z_hw: np.ndarray,
    grid: FourierGrid2D,
    *,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_first_bin: bool = False,
    agg: Literal["mean", "median"] = "mean",
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generic radial profile from any 2D scalar map Z(kx,ky).

    Returns:
      k_centers: (B,)
      z_k:       (B,)
      k_edges:   (B+1,)
      count_k:   (B,)
    """
    Z = np.asarray(Z_hw, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z_hw must be (H,W), got {Z.shape}")

    H, W = int(Z.shape[0]), int(Z.shape[1])
    k = np.asarray(grid.k, dtype=np.float64)
    if k.shape != (H, W):
        raise ValueError(f"grid.k shape {k.shape} != Z shape {(H, W)}")

    edges = _make_k_edges(grid, num_bins=int(num_bins), k_max=k_max, binning=binning, k_min=k_min)
    centers = 0.5 * (edges[:-1] + edges[1:])

    k_flat = k.reshape(-1)
    z_flat = Z.reshape(-1)
    valid = np.isfinite(k_flat) & np.isfinite(z_flat)
    k_flat = k_flat[valid]
    z_flat = z_flat[valid]

    idx = np.digitize(k_flat, edges, right=False) - 1
    B = int(num_bins)

    if agg == "mean":
        count = np.zeros((B,), dtype=np.int64)
        s = np.zeros((B,), dtype=np.float64)
        ok = (idx >= 0) & (idx < B)
        np.add.at(count, idx[ok], 1)
        np.add.at(s, idx[ok], z_flat[ok])
        z_k = s / np.maximum(count.astype(np.float64), float(eps))
    elif agg == "median":
        buckets: list[list[float]] = [[] for _ in range(B)]
        ok = (idx >= 0) & (idx < B)
        for i, z in zip(idx[ok], z_flat[ok], strict=False):
            buckets[int(i)].append(float(z))
        count = np.zeros((B,), dtype=np.int64)
        z_k = np.full((B,), np.nan, dtype=np.float64)
        for i in range(B):
            count[i] = len(buckets[i])
            if count[i] > 0:
                z_k[i] = float(np.median(np.asarray(buckets[i], dtype=np.float64)))
    else:
        raise ValueError(f"Unknown agg={agg}, expected 'mean' or 'median'")

    if drop_first_bin:
        if B <= 1:
            return centers[:0], z_k[:0], edges[:1], count[:0]
        centers = centers[1:]
        z_k = z_k[1:]
        count = count[1:]
        edges = edges[1:]

    # empty bin => NaN (not 0)
    empty = (count <= 0)
    if np.any(empty):
        z_k = np.asarray(z_k, dtype=np.float64)
        z_k[empty] = np.nan

    return centers, z_k, edges, count.astype(np.float64)


# ----------------------------
# Coherence / SNR radial profiles
# ----------------------------

def coherence_radial_profile(
    coh2d: np.ndarray,
    grid: FourierGrid2D,
    *,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_first_bin: bool = False,
    agg: Literal["mean", "median"] = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    coh2d: (H,W) coherence map in [0,1]
    Returns: k_centers, coh_k, k_edges, count_k
    """
    return radial_profile_from_map(
        coh2d,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_first_bin=drop_first_bin,
        agg=agg,
    )


def snr_radial_profile(
    P_true2d: np.ndarray,
    P_err2d: np.ndarray,
    grid: FourierGrid2D,
    *,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_first_bin: bool = False,
    agg: Literal["mean", "median"] = "mean",
    eps: float = 1e-12,
    log10: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Frequency-domain SNR profile.

    Define pointwise SNR2D = P_true2d / (P_err2d + eps), then radial-aggregate.
    If log10=True, output log10(SNR_k + eps).
    """
    P_true = np.asarray(P_true2d, dtype=np.float64)
    P_err = np.asarray(P_err2d, dtype=np.float64)
    if P_true.shape != P_err.shape or P_true.ndim != 2:
        raise ValueError(f"P_true2d and P_err2d must be same (H,W), got {P_true.shape} vs {P_err.shape}")

    snr2d = P_true / (P_err + float(eps))
    k_centers, snr_k, k_edges, count_k = radial_profile_from_map(
        snr2d,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_first_bin=drop_first_bin,
        agg=agg,
        eps=eps,
    )
    if log10:
        snr_k = np.log10(np.asarray(snr_k, dtype=np.float64) + float(eps))
    return k_centers, snr_k, k_edges, count_k


# ----------------------------
# k_eff extraction from a 1D profile
# ----------------------------

def _weighted_moving_average_1d(y: np.ndarray, w: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    B = int(y.size)
    if B == 0:
        return y.copy()
    window = int(window)
    if window <= 1:
        return y.copy()
    half = window // 2
    out = np.empty_like(y, dtype=np.float64)

    finite = np.isfinite(y) & np.isfinite(w) & (w > 0)
    for i in range(B):
        lo = max(0, i - half)
        hi = min(B, i + half + 1)
        m = finite[lo:hi]
        if not np.any(m):
            out[i] = np.nan
            continue
        yy = y[lo:hi][m]
        ww = w[lo:hi][m]
        s = float(np.sum(ww))
        out[i] = float(np.sum(yy * ww) / s) if s > 0 else np.nan
    return out


@dataclass
class KEffDebug:
    rule: str
    threshold: float
    found: bool
    k_eff: float
    idx: Optional[int]
    reason: Optional[str]
    y_used: np.ndarray


def k_eff_from_profile(
    k_centers: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    rule: Literal["last_above", "last_below", "crossing"] = "last_above",
    k_min_eval: Optional[float] = None,
    smooth_window: int = 1,
    weights: Optional[np.ndarray] = None,
    monotone_enforce: bool = False,
    direction: Literal["decreasing", "increasing", "auto"] = "auto",
    eps: float = 1e-12,
) -> Tuple[float, KEffDebug]:
    """
    从任意“尺度响应函数” y(k) 提取有效截止 k_eff。

    Common cases:
      - coherence(k): 下降型（高k变差），希望找 “最后一个 >= thr” => rule="last_above"
      - SNR(k): 下降型，thr=1 或 3；若你输出 log10(SNR) 则 thr=0 或 log10(3)
      - error-like curve: 上升型（高k更差），希望找 “最后一个 <= thr” => rule="last_below"

    Args:
      k_min_eval:
        only consider bins with k >= k_min_eval (avoid low-k instability / empty bins).
      smooth_window:
        weighted moving average window (odd recommended). 1 means no smoothing.
      weights:
        (B,) weights (e.g., count per bin). If None, uniform weights are used.
      monotone_enforce:
        If True, enforce monotonic envelope before thresholding.
        - For decreasing curves: enforce non-increasing via cumulative minimum.
        - For increasing curves: enforce non-decreasing via cumulative maximum.
      direction:
        "auto" tries to infer sign by comparing early vs late finite points.

    Returns:
      k_eff and debug struct.
    """
    k = np.asarray(k_centers, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)

    if k.ndim != 1 or yy.ndim != 1 or k.size != yy.size:
        raise ValueError("k_centers and y must be 1D arrays of same length.")
    B = int(k.size)
    if B == 0:
        dbg = KEffDebug(rule=str(rule), threshold=float(threshold), found=False, k_eff=0.0, idx=None, reason="empty", y_used=yy.copy())
        return 0.0, dbg

    if weights is None:
        w = np.ones((B,), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != B:
            raise ValueError("weights must be 1D array of same length as k_centers.")
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # valid region
    m = np.isfinite(k) & np.isfinite(yy) & (w > 0)
    if k_min_eval is not None:
        m &= (k >= float(k_min_eval))

    y_used = yy.copy()
    y_used[~m] = np.nan

    # smoothing
    if int(smooth_window) > 1:
        y_used = _weighted_moving_average_1d(y_used, w, int(smooth_window))

    # infer direction if auto
    dir_used = str(direction)
    if dir_used == "auto":
        finite_idx = np.where(np.isfinite(y_used))[0]
        if finite_idx.size >= 2:
            i0 = int(finite_idx[0])
            i1 = int(finite_idx[-1])
            dir_used = "decreasing" if float(y_used[i1]) < float(y_used[i0]) else "increasing"
        else:
            dir_used = "decreasing"

    # monotone envelope (optional)
    if monotone_enforce:
        vals = y_used.copy()
        finite_idx = np.where(np.isfinite(vals))[0]
        if finite_idx.size > 0:
            if dir_used == "decreasing":
                # non-increasing envelope: cumulative minimum
                last = None
                for i in finite_idx:
                    v = float(vals[i])
                    if last is None:
                        last = v
                    else:
                        last = float(min(last, v))
                        vals[i] = last
            else:
                # non-decreasing envelope: cumulative maximum
                last = None
                for i in finite_idx:
                    v = float(vals[i])
                    if last is None:
                        last = v
                    else:
                        last = float(max(last, v))
                        vals[i] = last
        y_used = vals

    thr = float(threshold)
    rule = str(rule)

    # thresholding mask
    if rule == "last_above":
        ok = np.isfinite(y_used) & (y_used >= thr)
    elif rule == "last_below":
        ok = np.isfinite(y_used) & (y_used <= thr)
    elif rule == "crossing":
        # find last crossing (linear interp) around threshold
        ok = np.isfinite(y_used)
    else:
        raise ValueError(f"Unknown rule={rule}")

    if rule in ("last_above", "last_below"):
        if not np.any(ok):
            dbg = KEffDebug(
                rule=rule,
                threshold=thr,
                found=False,
                k_eff=0.0,
                idx=None,
                reason="no_bin_satisfies_threshold",
                y_used=y_used,
            )
            return 0.0, dbg
        idx = int(np.where(ok)[0][-1])
        k_eff = float(k[idx])
        dbg = KEffDebug(rule=rule, threshold=thr, found=True, k_eff=k_eff, idx=idx, reason=None, y_used=y_used)
        return k_eff, dbg

    # crossing mode
    finite_idx = np.where(np.isfinite(y_used))[0]
    if finite_idx.size < 2:
        dbg = KEffDebug(rule=rule, threshold=thr, found=False, k_eff=0.0, idx=None, reason="insufficient_points", y_used=y_used)
        return 0.0, dbg

    # Determine crossing condition based on direction:
    # decreasing: want last k where y >= thr, crossing goes from >=thr to <thr
    # increasing: want last k where y <= thr, crossing goes from <=thr to >thr
    if dir_used == "decreasing":
        above = (y_used >= thr)
        # last index where above True
        if not np.any(above & np.isfinite(y_used)):
            dbg = KEffDebug(rule=rule, threshold=thr, found=False, k_eff=0.0, idx=None, reason="never_above_threshold", y_used=y_used)
            return 0.0, dbg
        i0 = int(np.where(above & np.isfinite(y_used))[0][-1])
        if i0 >= B - 1 or not np.isfinite(y_used[i0 + 1]):
            k_eff = float(k[i0])
            dbg = KEffDebug(rule=rule, threshold=thr, found=True, k_eff=k_eff, idx=i0, reason="edge_no_interp", y_used=y_used)
            return k_eff, dbg
        y0, y1 = float(y_used[i0]), float(y_used[i0 + 1])
        k0, k1 = float(k[i0]), float(k[i0 + 1])
        if np.isclose(y1, y0):
            k_eff = k0
        else:
            t = (thr - y0) / (y1 - y0)
            t = float(np.clip(t, 0.0, 1.0))
            k_eff = float(k0 + t * (k1 - k0))
        dbg = KEffDebug(rule=rule, threshold=thr, found=True, k_eff=k_eff, idx=i0, reason=None, y_used=y_used)
        return k_eff, dbg

    else:
        below = (y_used <= thr)
        if not np.any(below & np.isfinite(y_used)):
            dbg = KEffDebug(rule=rule, threshold=thr, found=False, k_eff=0.0, idx=None, reason="never_below_threshold", y_used=y_used)
            return 0.0, dbg
        i0 = int(np.where(below & np.isfinite(y_used))[0][-1])
        if i0 >= B - 1 or not np.isfinite(y_used[i0 + 1]):
            k_eff = float(k[i0])
            dbg = KEffDebug(rule=rule, threshold=thr, found=True, k_eff=k_eff, idx=i0, reason="edge_no_interp", y_used=y_used)
            return k_eff, dbg
        y0, y1 = float(y_used[i0]), float(y_used[i0 + 1])
        k0, k1 = float(k[i0]), float(k[i0 + 1])
        if np.isclose(y1, y0):
            k_eff = k0
        else:
            t = (thr - y0) / (y1 - y0)
            t = float(np.clip(t, 0.0, 1.0))
            k_eff = float(k0 + t * (k1 - k0))
        dbg = KEffDebug(rule=rule, threshold=thr, found=True, k_eff=k_eff, idx=i0, reason=None, y_used=y_used)
        return k_eff, dbg


# ----------------------------
# Optional: convenience helpers from fields / from L3 arrays
# ----------------------------

def coherence2d_from_cross_spectra(
    C_tp2d: np.ndarray,
    P_true2d: np.ndarray,
    P_pred2d: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    coh = |C_tp|^2 / (P_true * P_pred + eps)
    """
    C = np.asarray(C_tp2d)
    Pt = np.asarray(P_true2d, dtype=np.float64)
    Pp = np.asarray(P_pred2d, dtype=np.float64)
    return (np.abs(C) ** 2) / (Pt * Pp + float(eps))


def power2d_from_field(
    x: np.ndarray,
    *,
    mean_mode: str = "none",
) -> np.ndarray:
    """
    Convenience: compute per-frequency power |F|^2 (summed over channels if any).
    Output: (H,W) float64
    """
    F = fft2_field(x, mean_mode=mean_mode)
    if F.ndim == 2:
        return (np.abs(F) ** 2).astype(np.float64, copy=False)
    if F.ndim == 3:
        # (C,H,W)
        return np.sum(np.abs(F) ** 2, axis=0).astype(np.float64, copy=False)
    raise ValueError(f"Unsupported FFT shape: {F.shape}")
