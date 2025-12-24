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


def _weighted_moving_average_1d(
    y: np.ndarray,
    w: np.ndarray,
    window: int,
) -> np.ndarray:
    """Weighted moving average with finite-mask handling.

    - y: (B,) may contain non-finite values
    - w: (B,) nonnegative weights (e.g., count per bin)
    - window: odd integer recommended; effective half-width = window//2
    """
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


def smooth_nrmse_from_energy(
    E_true_k: np.ndarray,
    E_err_k: np.ndarray,
    count_k: Optional[np.ndarray] = None,
    *,
    eps: float = 1e-12,
    smooth_window: int = 7,
    empty_to_nan: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth Et/Ee first, then compute NRMSE.

    Returns:
      Et_smooth, Ee_smooth, nrmse_smooth
    """
    Et = np.asarray(E_true_k, dtype=np.float64)
    Ee = np.asarray(E_err_k, dtype=np.float64)

    if count_k is None:
        w = np.ones_like(Et, dtype=np.float64)
    else:
        w = np.asarray(count_k, dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    # mark empty bins as NaN before smoothing (so they don't drag averages down)
    if empty_to_nan and count_k is not None:
        empty = (np.asarray(count_k) <= 0)
        Et = Et.copy()
        Ee = Ee.copy()
        Et[empty] = np.nan
        Ee[empty] = np.nan

    Et_s = _weighted_moving_average_1d(Et, w, smooth_window)
    Ee_s = _weighted_moving_average_1d(Ee, w, smooth_window)

    nrmse_s = np.sqrt(Ee_s / (Et_s + float(eps)))
    return Et_s, Ee_s, nrmse_s


def fourier_cumulative_nrmse_curve_from_energy(
    k_centers: np.ndarray,
    E_true_k: np.ndarray,
    E_err_k: np.ndarray,
    count_k: Optional[np.ndarray] = None,
    *,
    k_min_eval: float = 1.0,
    eps: float = 1e-12,
    monotone_enforce: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute cumulative low-pass NRMSE_{<=K} curve from already-binned energies.

    This corresponds to the project definition:
      NRMSE_{<=K} = sqrt( sum_{k<=K} Ee(k) / (sum_{k<=K} Et(k) + eps) )

    Inputs:
      - k_centers: (B,)
      - E_true_k:  (B,)
      - E_err_k:   (B,)
      - count_k:   (B,) optional; used to mask empty bins (<=0)

    Args:
      k_min_eval:
        Only bins with k >= k_min_eval are considered as the start of the cumulative evaluation.
        This avoids the low-k "background / near-zero-energy" instability.
      eps:
        Numerical stabilizer for denominator.
      monotone_enforce:
        If True, enforce a non-increasing envelope on cumulative NRMSE curve
        (since adding more high-k content should not magically improve cumulative error in a stable way,
         but small numerical / binning noise can cause slight bumps).

    Returns:
      - k_eval: (B,) same length as k_centers; values below k_min_eval are kept but output as NaN in nrmse_cum
      - nrmse_cum: (B,) cumulative curve aligned with k_eval
      - meta: dict with debug fields (start index, cumulative energies, etc.)
    """
    k = np.asarray(k_centers, dtype=np.float64)
    Et = np.asarray(E_true_k, dtype=np.float64)
    Ee = np.asarray(E_err_k, dtype=np.float64)

    if k.ndim != 1 or Et.ndim != 1 or Ee.ndim != 1 or not (k.size == Et.size == Ee.size):
        raise ValueError("k_centers, E_true_k, E_err_k must be 1D arrays of same length.")
    B = int(k.size)
    if B == 0:
        return k.copy(), np.asarray([], dtype=np.float64), {
            "k_min_eval": float(k_min_eval),
            "start_idx": None,
            "cum_Et": np.asarray([], dtype=np.float64),
            "cum_Ee": np.asarray([], dtype=np.float64),
        }

    if count_k is None:
        valid_bin = np.isfinite(k) & np.isfinite(Et) & np.isfinite(Ee)
    else:
        c = np.asarray(count_k)
        if c.ndim != 1 or c.size != B:
            raise ValueError("count_k must be 1D array of same length as k_centers.")
        valid_bin = np.isfinite(k) & np.isfinite(Et) & np.isfinite(Ee) & (c > 0)

    # We compute cumulative only on bins with k>=k_min_eval and valid_bin
    k_min_eval = float(k_min_eval)
    start_mask = (k >= k_min_eval) & valid_bin

    # locate first eligible index
    idxs = np.where(start_mask)[0]
    start_idx = int(idxs[0]) if idxs.size > 0 else None

    nrmse_cum = np.full((B,), np.nan, dtype=np.float64)
    cum_Et = np.zeros((B,), dtype=np.float64)
    cum_Ee = np.zeros((B,), dtype=np.float64)

    if start_idx is None:
        # No valid region to evaluate
        meta = {
            "k_min_eval": float(k_min_eval),
            "start_idx": None,
            "cum_Et": cum_Et,
            "cum_Ee": cum_Ee,
        }
        return k.copy(), nrmse_cum, meta

    running_Et = 0.0
    running_Ee = 0.0

    for i in range(B):
        if i < start_idx:
            cum_Et[i] = 0.0
            cum_Ee[i] = 0.0
            continue

        if valid_bin[i]:
            # guard negatives (shouldn't happen, but don't let it poison sums)
            et_i = float(Et[i]) if np.isfinite(Et[i]) else 0.0
            ee_i = float(Ee[i]) if np.isfinite(Ee[i]) else 0.0
            if et_i < 0:
                et_i = 0.0
            if ee_i < 0:
                ee_i = 0.0
            running_Et += et_i
            running_Ee += ee_i

        cum_Et[i] = running_Et
        cum_Ee[i] = running_Ee

        if running_Et > 0:
            nrmse_cum[i] = float(np.sqrt(running_Ee / (running_Et + float(eps))))
        else:
            nrmse_cum[i] = np.nan

    if monotone_enforce:
        # cumulative NRMSE should be roughly non-increasing as K grows;
        # enforce envelope to remove small bumps.
        finite = np.isfinite(nrmse_cum)
        if np.any(finite):
            # apply cumulative minimum over finite region
            vals = nrmse_cum.copy()
            # keep NaN as-is; build envelope over indices with finite vals
            last = None
            for i in range(B):
                if not np.isfinite(vals[i]):
                    continue
                if last is None:
                    last = float(vals[i])
                else:
                    last = float(min(last, float(vals[i])))
                    vals[i] = last
            nrmse_cum = vals

    meta = {
        "k_min_eval": float(k_min_eval),
        "start_idx": int(start_idx),
        "cum_Et": cum_Et,
        "cum_Ee": cum_Ee,
    }
    return k.copy(), nrmse_cum, meta


def kstar_from_cumulative_curve(
    k_centers: np.ndarray,
    nrmse_cum: np.ndarray,
    *,
    eps_plateau: float = 1e-3,
    m_plateau: int = 4,
    k_min_eval: float = 1.0,
    prefer_last_if_not_found: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """Pick k* as the onset of plateau on the cumulative low-pass NRMSE curve.

    Criterion (discrete):
      Let Δ_i = NRMSE_{<=k_{i-1}} - NRMSE_{<=k_i}  (improvement; ideally >=0)
      Find the first index i such that:
        Δ_{i+1}, Δ_{i+2}, ..., Δ_{i+m_plateau}  are all < eps_plateau
      Then k* = k_i.

    Notes:
      - nrmse_cum is expected to be (roughly) non-increasing. If not, negative Δ are treated as 0 improvement.
      - We only search on bins with k >= k_min_eval and finite nrmse_cum.

    Returns:
      - k_star (float): plateau onset k*. If not found:
          * if prefer_last_if_not_found=True: return last finite k in the valid region
          * else: return 0.0
      - debug dict: indices, found flag, deltas, etc.
    """
    k = np.asarray(k_centers, dtype=np.float64)
    y = np.asarray(nrmse_cum, dtype=np.float64)

    if k.ndim != 1 or y.ndim != 1 or k.size != y.size:
        raise ValueError("k_centers and nrmse_cum must be 1D arrays of same length.")
    B = int(k.size)
    if B == 0:
        return 0.0, {
            "found": False,
            "reason": "empty",
            "k_min_eval": float(k_min_eval),
            "eps_plateau": float(eps_plateau),
            "m_plateau": int(m_plateau),
        }

    eps_plateau = float(eps_plateau)
    m_plateau = int(m_plateau)
    if m_plateau < 1:
        raise ValueError(f"m_plateau must be >= 1, got {m_plateau}")
    k_min_eval = float(k_min_eval)

    valid = (k >= k_min_eval) & np.isfinite(k) & np.isfinite(y)
    idxs = np.where(valid)[0]
    if idxs.size < (m_plateau + 2):
        # Not enough points to detect plateau robustly
        if prefer_last_if_not_found and idxs.size > 0:
            k_last = float(k[idxs[-1]])
            return k_last, {
                "found": False,
                "reason": "insufficient_points",
                "k_star_idx": int(idxs[-1]),
                "k_min_eval": float(k_min_eval),
                "eps_plateau": float(eps_plateau),
                "m_plateau": int(m_plateau),
            }
        return 0.0, {
            "found": False,
            "reason": "insufficient_points",
            "k_min_eval": float(k_min_eval),
            "eps_plateau": float(eps_plateau),
            "m_plateau": int(m_plateau),
        }

    # compute improvement deltas on the valid subsequence, but keep original indexing
    # Δ_i is defined for i>=1: Δ_i = y_{i-1} - y_i
    deltas = np.full((B,), np.nan, dtype=np.float64)
    for i in range(1, B):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]):
            d = float(y[i - 1] - y[i])
            # if curve bumps up, treat as 0 improvement for plateau detection
            deltas[i] = float(max(d, 0.0))

    # search: plateau onset index i such that next m_plateau deltas are all < eps_plateau
    # use only indices in idxs (valid region)
    idx_set = set(int(v) for v in idxs.tolist())

    found_idx = None
    # start from the first valid index + 1 because deltas[i] uses i-1
    for i in idxs[1:]:
        # require i and i+m_plateau within bounds and valid
        ok = True
        for j in range(1, m_plateau + 1):
            ii = int(i + j)
            if ii >= B or ii not in idx_set:
                ok = False
                break
            dj = float(deltas[ii]) if np.isfinite(deltas[ii]) else np.inf
            if not (dj < eps_plateau):
                ok = False
                break
        if ok:
            found_idx = int(i)
            break

    if found_idx is not None:
        return float(k[found_idx]), {
            "found": True,
            "k_star_idx": int(found_idx),
            "k_min_eval": float(k_min_eval),
            "eps_plateau": float(eps_plateau),
            "m_plateau": int(m_plateau),
            "deltas": deltas,
        }

    # not found
    if prefer_last_if_not_found:
        k_last = float(k[idxs[-1]])
        return k_last, {
            "found": False,
            "reason": "no_plateau_detected",
            "k_star_idx": int(idxs[-1]),
            "k_min_eval": float(k_min_eval),
            "eps_plateau": float(eps_plateau),
            "m_plateau": int(m_plateau),
            "deltas": deltas,
        }

    return 0.0, {
        "found": False,
        "reason": "no_plateau_detected",
        "k_min_eval": float(k_min_eval),
        "eps_plateau": float(eps_plateau),
        "m_plateau": int(m_plateau),
        "deltas": deltas,
    }


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

    k_centers, E_k, k_edges, _count_k = radial_bin_spectrum(
        F,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_first_bin=drop_zero_bin,
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
    drop_first_bin: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial NRMSE(k) curve based on energy in each k-bin.

    Always returns:
      k_centers, E_true_k, E_err_k, nrmse_k, count_k

    Note:
      - nrmse_k is raw. Empty bins (count_k==0) are set to NaN here.
        (This prevents the "fake zero spikes" problem by construction.)
    """
    x_hat = np.asarray(x_hat)
    x_true = np.asarray(x_true)
    H, W = _infer_hw(x_true)

    grid = _infer_grid_from_meta(H, W, grid_meta)

    F_true = _fft_for_metric(x_true, mean_mode=mean_mode)
    F_err = _fft_for_metric(x_hat - x_true, mean_mode="none")

    k_centers, E_true_k, _edges, count_k = radial_bin_spectrum(
        F_true,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_first_bin=drop_first_bin,
    )
    _, E_err_k, _edges2, _count2 = radial_bin_spectrum(
        F_err,
        grid,
        num_bins=num_bins,
        k_max=k_max,
        binning=binning,
        k_min=k_min,
        drop_first_bin=drop_first_bin,
    )

    E_true_k = np.asarray(E_true_k, dtype=np.float64)
    E_err_k = np.asarray(E_err_k, dtype=np.float64)
    count_k = np.asarray(count_k, dtype=np.int64)

    nrmse_k = np.sqrt(E_err_k / (E_true_k + float(eps)))

    # empty bin => NaN (not 0)
    empty = (count_k <= 0)
    if np.any(empty):
        nrmse_k = np.asarray(nrmse_k, dtype=np.float64)
        nrmse_k[empty] = np.nan

    return np.asarray(k_centers), E_true_k, E_err_k, nrmse_k, count_k


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
