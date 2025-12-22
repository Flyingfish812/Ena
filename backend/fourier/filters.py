# backend/fourier/filters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional, Any, List
import numpy as np


@dataclass(frozen=True)
class FourierGrid2D:
    """2D Fourier grid meta and k-grid.

    Notes on units:
      - If angular=True, kx/ky/k are angular wavenumbers (rad / length).
      - If angular=False, kx/ky/k are cycles per length.
    """
    H: int
    W: int
    dx: float
    dy: float
    angular: bool
    kx: np.ndarray  # shape (H, W)
    ky: np.ndarray  # shape (H, W)
    k: np.ndarray   # shape (H, W)


def _as_hw_or_chw(x: np.ndarray) -> Tuple[np.ndarray, str]:
    """Normalize x into either (H, W) or (C, H, W).

    Returns:
        (x_norm, policy) where policy in {"HW","CHW"}.
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False), "HW"
    if x.ndim == 3:
        # Heuristic: treat first dim as C if small, else last dim as C
        # Common in your project: (H,W,C) or (C,H,W).
        if x.shape[0] <= 8 and x.shape[1] > 8 and x.shape[2] > 8:
            # (C,H,W)
            return x.astype(np.float64, copy=False), "CHW"
        if x.shape[2] <= 8 and x.shape[0] > 8 and x.shape[1] > 8:
            # (H,W,C) -> (C,H,W)
            return np.moveaxis(x, -1, 0).astype(np.float64, copy=False), "CHW"
    raise ValueError(f"Unsupported x shape: {x.shape}. Expect (H,W) or (H,W,C) or (C,H,W).")


def make_wavenumber_grid(
    H: int,
    W: int,
    dx: float = 1.0,
    dy: float = 1.0,
    angular: bool = True,
) -> FourierGrid2D:
    """Build (kx, ky, k) grids matched to numpy.fft.fft2 indexing."""
    # fftfreq gives cycles per unit; multiply 2π if using angular wavenumber.
    fx = np.fft.fftfreq(W, d=dx)  # (W,)
    fy = np.fft.fftfreq(H, d=dy)  # (H,)
    if angular:
        fx = 2.0 * np.pi * fx
        fy = 2.0 * np.pi * fy

    kx_1d = fx
    ky_1d = fy
    kx, ky = np.meshgrid(kx_1d, ky_1d)  # both (H,W)
    k = np.sqrt(kx**2 + ky**2)
    return FourierGrid2D(H=H, W=W, dx=float(dx), dy=float(dy), angular=bool(angular), kx=kx, ky=ky, k=k)


def fft2_field(
    x: np.ndarray,
    mean_mode: str = "none",
) -> np.ndarray:
    """2D FFT for x in (H,W) or (C,H,W)/(H,W,C).

    mean_mode:
      - "none": do nothing
      - "global": subtract global mean per-channel
      - "per_row"/"per_col": optional hooks (rarely needed; kept for completeness)
    """
    x_norm, policy = _as_hw_or_chw(x)

    def _demean(a: np.ndarray) -> np.ndarray:
        if mean_mode == "none":
            return a
        if mean_mode == "global":
            return a - np.mean(a, axis=(-2, -1), keepdims=True)
        if mean_mode == "per_row":
            return a - np.mean(a, axis=-1, keepdims=True)
        if mean_mode == "per_col":
            return a - np.mean(a, axis=-2, keepdims=True)
        raise ValueError(f"Unknown mean_mode={mean_mode}")

    if policy == "HW":
        xx = _demean(x_norm)
        return np.fft.fft2(xx)
    else:
        xx = _demean(x_norm)
        return np.fft.fft2(xx, axes=(-2, -1))


def ifft2_field(F: np.ndarray, real_output: bool = True) -> np.ndarray:
    """2D inverse FFT for (H,W) or (C,H,W)."""
    F = np.asarray(F)
    if F.ndim == 2:
        x = np.fft.ifft2(F)
        return np.real(x) if real_output else x
    if F.ndim == 3:
        x = np.fft.ifft2(F, axes=(-2, -1))
        return np.real(x) if real_output else x
    raise ValueError(f"Unsupported F shape: {F.shape}. Expect (H,W) or (C,H,W).")


def parseval_energy_from_fft(F: np.ndarray) -> float:
    """Energy in spatial domain equals (1/N) sum |F|^2 under numpy fft convention."""
    F = np.asarray(F)
    if F.ndim == 2:
        H, W = F.shape
        N = H * W
        return float(np.sum(np.abs(F) ** 2) / N)
    if F.ndim == 3:
        C, H, W = F.shape
        N = H * W
        return float(np.sum(np.abs(F) ** 2) / N)
    raise ValueError(f"Unsupported F shape: {F.shape}")


def _infer_default_k_min_from_grid(grid: FourierGrid2D) -> float:
    """Heuristic: use the smallest *positive* frequency step among axes.

    For angular=False: units are cycles/length, so:
      dkx = 1/(W*dx), dky = 1/(H*dy)
    For angular=True: multiplied by 2π.

    This is a safe-ish lower bound when user doesn't provide k_min for log binning.
    """
    dkx = 1.0 / (float(grid.W) * float(grid.dx))
    dky = 1.0 / (float(grid.H) * float(grid.dy))
    dk = min(dkx, dky)
    if grid.angular:
        dk = 2.0 * np.pi * dk
    # avoid pathological tiny / zero
    return float(max(dk, 1e-12))


def radial_bin_spectrum(
    F: np.ndarray,
    grid: FourierGrid2D,
    num_bins: int = 64,
    k_max: Optional[float] = None,
    *,
    binning: str = "log",
    k_min: Optional[float] = None,
    drop_zero_bin: bool = False,
    return_edges: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial binned energy spectrum E(k) from FFT F.

    Binning strategies:
      - binning="linear": uniform edges in k
      - binning="log": geometric edges in k (requires k_min>0)

    Returns:
        k_centers: (B,)
        E_k: (B,)   where sum(E_k) ~= spatial energy (Parseval), up to binning resolution.
        (optional) k_edges: (B+1,)
    """
    if num_bins <= 0:
        raise ValueError(f"num_bins must be >0, got {num_bins}")

    k = np.asarray(grid.k, dtype=np.float64)
    if k_max is None:
        k_max = float(np.max(k))
    k_max = float(k_max)
    if not np.isfinite(k_max) or k_max <= 0:
        raise ValueError(f"Invalid k_max={k_max}")

    binning = str(binning).lower().strip()
    if binning not in ("linear", "log"):
        raise ValueError(f"binning must be 'linear' or 'log', got {binning}")

    if binning == "linear":
        edges = np.linspace(0.0, k_max, num_bins + 1, dtype=np.float64)
    else:
        if k_min is None:
            k_min = _infer_default_k_min_from_grid(grid)
        k_min = float(k_min)
        if not np.isfinite(k_min) or k_min <= 0:
            raise ValueError(f"log-binning requires k_min>0, got k_min={k_min}")
        if k_min >= k_max:
            raise ValueError(f"log-binning requires k_min<k_max, got k_min={k_min}, k_max={k_max}")
        edges = np.geomspace(k_min, k_max, num_bins + 1).astype(np.float64)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # energy density per frequency sample (Parseval): |F|^2 / N
    F = np.asarray(F)
    if F.ndim == 2:
        H, W = F.shape
        N = H * W
        E_samples = (np.abs(F) ** 2) / N
    elif F.ndim == 3:
        C, H, W = F.shape
        N = H * W
        E_samples = (np.abs(F) ** 2) / N
        E_samples = np.sum(E_samples, axis=0)  # sum across channels -> (H,W)
    else:
        raise ValueError(f"Unsupported F shape: {F.shape}")

    # binning
    k_flat = k.reshape(-1)
    e_flat = np.asarray(E_samples, dtype=np.float64).reshape(-1)

    # np.digitize returns 1..B (or B+1), we map to 0..B-1
    idx = np.digitize(k_flat, edges, right=False) - 1
    B = int(num_bins)
    E_k = np.zeros((B,), dtype=np.float64)
    valid = (idx >= 0) & (idx < B)
    np.add.at(E_k, idx[valid], e_flat[valid])

    if drop_zero_bin:
        # drop the first bin if it's effectively k≈0 (linear binning),
        # or if centers[0] is extremely small (log binning with tiny k_min).
        if centers.size > 0 and (np.isclose(centers[0], 0.0) or centers[0] <= 0.0):
            centers = centers[1:]
            E_k = E_k[1:]
            edges = edges[1:]  # keep alignment: edges length = centers+1

    if return_edges:
        return centers, E_k, edges
    return centers, E_k


def make_band_masks_from_edges(
    grid: FourierGrid2D,
    k_edges: Sequence[float],
    include_lowest: bool = True,
) -> Dict[str, np.ndarray]:
    """Hard (binary) band masks by radial k edges.

    Example k_edges: [0, k1, k2, k_max]
      -> bands: b0:[0,k1), b1:[k1,k2), b2:[k2,kmax]
    """
    edges = np.asarray(list(k_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("k_edges must be 1D with >=2 elements.")
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("k_edges must be nondecreasing.")

    k = grid.k
    masks: Dict[str, np.ndarray] = {}
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0 and include_lowest:
            m = (k >= lo) & (k < hi)
        else:
            m = (k >= lo) & (k < hi)
        masks[f"band_{i}"] = m
    # last edge inclusive (to catch max)
    masks[f"band_{edges.size-2}"] |= (k == edges[-1])
    return masks


def make_soft_band_weights_from_edges(
    grid: FourierGrid2D,
    k_edges: Sequence[float],
    transition: float,
) -> Dict[str, np.ndarray]:
    """Weak split (soft weights) between bands using cosine ramps.

    transition: width of the transition region in k-units (same unit as grid.k).
      - transition=0 -> nearly hard split
    """
    edges = np.asarray(list(k_edges), dtype=np.float64)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("k_edges must be 1D with >=2 elements.")
    if transition < 0:
        raise ValueError("transition must be >= 0.")
    k = grid.k.astype(np.float64, copy=False)

    def smooth_step(a: np.ndarray) -> np.ndarray:
        # clamp to [0,1], then cosine
        aa = np.clip(a, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * aa)

    weights: Dict[str, np.ndarray] = {}
    for i in range(edges.size - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if transition <= 0:
            w = ((k >= lo) & (k < hi)).astype(np.float64)
        else:
            # core region
            core_lo = lo + transition
            core_hi = hi - transition

            w = np.zeros_like(k, dtype=np.float64)

            # rising ramp: [lo, lo+transition]
            if core_lo > lo:
                a = (k - lo) / max(transition, 1e-12)
                w += smooth_step(a) * (k >= lo) * (k < core_lo)

            # flat core: [lo+transition, hi-transition]
            w += 1.0 * (k >= core_lo) * (k < core_hi)

            # falling ramp: [hi-transition, hi]
            if hi > core_hi:
                a = (hi - k) / max(transition, 1e-12)
                w += smooth_step(a) * (k >= core_hi) * (k < hi)

        weights[f"band_{i}"] = w

    # Normalize per-pixel sum to 1 (avoid overlaps causing >1)
    stack = np.stack(list(weights.values()), axis=0)  # (B,H,W)
    s = np.sum(stack, axis=0, keepdims=False)
    s_safe = np.where(s > 0, s, 1.0)
    stack = stack / s_safe

    out: Dict[str, np.ndarray] = {}
    for j, key in enumerate(weights.keys()):
        out[key] = stack[j]
    return out


def apply_band_mask_in_fourier(F: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a (H,W) mask to F which is (H,W) or (C,H,W)."""
    F = np.asarray(F)
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be (H,W).")
    if F.ndim == 2:
        return F * mask
    if F.ndim == 3:
        return F * mask[None, :, :]
    raise ValueError(f"Unsupported F shape: {F.shape}")


def auto_pick_k_edges_from_energy(
    k_centers: np.ndarray,
    E_k: np.ndarray,
    quantiles: Sequence[float] = (0.8, 0.95),
) -> List[float]:
    """Pick k edges based on cumulative energy quantiles.

    Returns edges: [0, k_q1, k_q2, k_max]
    """
    k_centers = np.asarray(k_centers, dtype=np.float64)
    E_k = np.asarray(E_k, dtype=np.float64)
    if k_centers.ndim != 1 or E_k.ndim != 1 or k_centers.size != E_k.size:
        raise ValueError("k_centers and E_k must be 1D of same length.")
    if np.any(E_k < 0):
        raise ValueError("E_k must be nonnegative.")

    total = float(np.sum(E_k))
    if total <= 0:
        # degenerate
        k_max = float(np.max(k_centers)) if k_centers.size > 0 else 0.0
        return [0.0, 0.0, 0.0, k_max]

    cdf = np.cumsum(E_k) / total
    edges = [0.0]
    for q in quantiles:
        q = float(q)
        q = np.clip(q, 0.0, 1.0)
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), k_centers.size - 1)
        edges.append(float(k_centers[idx]))
    edges.append(float(np.max(k_centers)))
    # ensure nondecreasing
    edges = np.maximum.accumulate(np.asarray(edges, dtype=np.float64)).tolist()
    return edges
