# backend/pipelines/eval_mods/scale_io.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Sequence

import numpy as np

from backend.fourier.filters import make_wavenumber_grid
from backend.pipelines.eval.utils import load_npz, write_json


# -----------------------------
# Schema helpers
# -----------------------------

@dataclass(frozen=True)
class Scale2DKeys:
    # canonical names we prefer
    P_true2d: str = "P_true2d"
    P_pred2d: str = "P_pred2d"
    P_err2d: str = "P_err2d"
    C_tp2d: str = "C_tp2d"      # cross power (true*conj(pred)) or similar
    coh2d: str = "coh2d"        # coherence magnitude-squared or coherence (caller-defined)

@dataclass(frozen=True)
class ScaleL3Entry:
    model_type: str
    mask_rate: float
    noise_sigma: float
    l3_fft_path: str

_1D_KEY_ALIASES: Dict[str, List[str]] = {
    "k_centers": ["k_centers", "k_center", "k"],
    "rho_k": ["rho_k", "rho", "rho_profile"],
    "coh_k": ["coh_k", "coherence_k", "coherence"],
    "snr_k": ["snr_k", "SNR_k", "log10SNR_k"],
}

# Accept a few common aliases to be tolerant to earlier experiments.
_2D_KEY_ALIASES: Dict[str, List[str]] = {
    "P_true2d": ["fft2_2d_P_true", "P_true2d", "P_true_2d", "P_true2", "P_true"],
    "P_pred2d": ["fft2_2d_P_pred", "P_pred2d", "P_pred_2d", "P_pred2", "P_pred"],
    "P_err2d":  ["fft2_2d_P_err",  "P_err2d",  "P_err_2d",  "P_err2",  "P_err"],
    "C_tp2d":   ["fft2_2d_C_tp",   "C_tp2d", "C_tp_2d", "Ctp2d", "C_tp", "cross2d", "E_cross2d"],
    "coh2d":    ["fft2_2d_coh",    "coh2d", "coh_2d", "coherence2d", "coh2", "coh"],
}

# also accept L3 meta axes if present in npz (v2.1)
_2D_META_ALIASES: Dict[str, List[str]] = {
    "hw": ["fft2_2d_hw", "hw", "H_W", "shape_hw"],
    "dx": ["fft2_2d_dx", "dx"],
    "dy": ["fft2_2d_dy", "dy"],
    "angular": ["fft2_2d_angular", "angular"],
    "kx": ["fft2_2d_kx", "kx"],
    "ky": ["fft2_2d_ky", "ky"],
}



def _pick_first_present(z: Dict[str, Any], names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in z:
            return n
    return None


def parse_l3_scale_entries(l3_index: Dict[str, Any]) -> List[ScaleL3Entry]:
    out: List[ScaleL3Entry] = []
    for e in (l3_index.get("entries", []) or []):
        try:
            out.append(
                ScaleL3Entry(
                    model_type=str(e["model_type"]),
                    mask_rate=float(e["mask_rate"]),
                    noise_sigma=float(e["noise_sigma"]),
                    l3_fft_path=str(e["l3_fft_path"]),
                )
            )
        except Exception:
            continue
    return out


def resolve_scale_2d_keys(z: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canon, aliases in _2D_KEY_ALIASES.items():
        hit = _pick_first_present(z, aliases)
        if hit is not None:
            out[canon] = hit
    return out


def require_fft_grid_from_l3_meta(l3_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expect l3_meta like:
      meta["fft_settings"] = {"dx":..., "dy":..., "angular":..., ...}
    """
    fs = l3_meta.get("fft_settings", None)
    if not isinstance(fs, dict):
        raise ValueError("L3 meta.json missing fft_settings; cannot build k-grid for scale metrics.")
    for k in ("dx", "dy", "angular"):
        if k not in fs:
            raise ValueError(f"L3 meta.json fft_settings missing key '{k}'")
    return fs


def build_kgrid_for_2d(*, H: int, W: int, l3_meta: Dict[str, Any]):
    fs = require_fft_grid_from_l3_meta(l3_meta)
    grid = make_wavenumber_grid(
        H=int(H),
        W=int(W),
        dx=float(fs["dx"]),
        dy=float(fs["dy"]),
        angular=bool(fs["angular"]),
    )
    return grid


# -----------------------------
# Loaders
# -----------------------------

def load_l3_npz(path: str | Path) -> Dict[str, Any]:
    return load_npz(Path(path), allow_pickle=False)


def load_scale_2d_from_l3_npz(
    z: Dict[str, Any],
    *,
    require: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "P_true2d": ndarray|None,
        "P_pred2d": ndarray|None,
        "P_err2d": ndarray|None,
        "C_tp2d": ndarray|None,
        "coh2d": ndarray|None,

        # optional meta (if present in npz)
        "H": int|None,
        "W": int|None,
        "dx": float|None,
        "dy": float|None,
        "angular": bool|None,
        "kx": ndarray|None,
        "ky": ndarray|None,

        "_keys_used": {canon->actual},
        "_meta_keys_used": {canon->actual},
      }
    """
    keys_used = resolve_scale_2d_keys(z)

    def _get(canon: str) -> Optional[np.ndarray]:
        k = keys_used.get(canon, None)
        if k is None:
            return None
        return np.asarray(z[k])

    # ---- main 2D fields ----
    P_true2d = _get("P_true2d")
    P_pred2d = _get("P_pred2d")
    P_err2d  = _get("P_err2d")
    C_tp2d   = _get("C_tp2d")
    coh2d    = _get("coh2d")

    missing_required: List[str] = []
    for name, arr in (("P_true2d", P_true2d), ("P_pred2d", P_pred2d), ("P_err2d", P_err2d)):
        if arr is None:
            missing_required.append(name)

    if require and missing_required:
        # print what keys exist to help debug quickly
        present = sorted([k for k in z.keys() if not str(k).startswith("_")])
        raise KeyError(
            "L3 npz missing required 2D keys for scale metrics: "
            f"{missing_required}. "
            "Expected v2.1 keys like: fft2_2d_P_true/fft2_2d_P_pred/fft2_2d_P_err (and optionally fft2_2d_C_tp/fft2_2d_coh). "
            "Hint: set eval.fourier.save_fft2_2d_stats=true and include needed tokens in eval.fourier.fft2_2d_stats_what, then rerun Level-3. "
            f"Available keys: {present}"
        )

    # ---- optional meta (prefer reading from npz; most robust) ----
    meta_used: Dict[str, str] = {}

    def _pick_meta(canon: str) -> Optional[Any]:
        names = _2D_META_ALIASES.get(canon, [])
        for n in names:
            if n in z:
                meta_used[canon] = n
                v = z[n]
                # unwrap scalar arrays
                if isinstance(v, np.ndarray) and v.shape == ():
                    return v.item()
                return v
        return None

    hw = _pick_meta("hw")
    H = W = None
    if hw is not None:
        arr = np.asarray(hw).reshape(-1)
        if arr.size >= 2:
            H = int(arr[0])
            W = int(arr[1])
    dx = _pick_meta("dx")
    dy = _pick_meta("dy")
    angular = _pick_meta("angular")
    kx = _pick_meta("kx")
    ky = _pick_meta("ky")

    # normalize meta dtypes
    dx = None if dx is None else float(dx)
    dy = None if dy is None else float(dy)
    angular = None if angular is None else bool(angular)
    if kx is not None:
        kx = np.asarray(kx)
    if ky is not None:
        ky = np.asarray(ky)

    return {
        "P_true2d": P_true2d,
        "P_pred2d": P_pred2d,
        "P_err2d": P_err2d,
        "C_tp2d": C_tp2d,
        "coh2d": coh2d,
        "H": H,
        "W": W,
        "dx": dx,
        "dy": dy,
        "angular": angular,
        "kx": kx,
        "ky": ky,
        "_keys_used": keys_used,
        "_meta_keys_used": meta_used,
    }


def snr_from_l3_1d_energy(
    z: Dict[str, Any],
    *,
    eps: float = 1e-12,
    log10: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback SNR(k) from existing 1D energy:
      snr_k = E_true_k / (E_err_k + eps)
    """
    if "k_centers" not in z:
        raise KeyError("L3 npz missing k_centers; cannot compute SNR fallback.")
    if "E_true_k" not in z or "E_err_k" not in z:
        raise KeyError("L3 npz missing E_true_k/E_err_k; cannot compute SNR fallback.")

    k = np.asarray(z["k_centers"], dtype=float)
    Et = np.asarray(z["E_true_k"], dtype=float)
    Ee = np.asarray(z["E_err_k"], dtype=float)

    snr = Et / (Ee + float(eps))
    if log10:
        snr = np.log10(np.maximum(snr, float(eps)))
    return k, snr


def rho_from_l3_1d(
    z: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rho_k from L3 if present:
      - returns (k_centers, rho_k)
    """
    if "k_centers" not in z:
        raise KeyError("L3 npz missing k_centers; cannot load rho_k.")
    if "rho_k" not in z:
        raise KeyError("L3 npz missing rho_k; cannot load rho_k.")
    k = np.asarray(z["k_centers"], dtype=float).reshape(-1)
    rho = np.asarray(z["rho_k"], dtype=float).reshape(-1)
    return k, rho


def score_from_l3_1d_energy(
    z,
    *,
    eps: float = 1e-12,
    clip_rho_to_01: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (k, rho(k), gain(k), score(k)) from existing L3 1D spectra.

    Required keys:
      - k_centers
      - E_true_k, E_pred_k, E_cross_k
    Optional:
      - rho_k (if present, use it; otherwise compute)

    Definitions:
      rho(k)  = E_cross / sqrt(E_true * E_pred)     (clipped to [-1,1])
      gain(k) = sqrt(E_pred / E_true)               (>=0)
      score(k)= clip(rho,0,1) * exp(-abs(log(gain)))  in [0,1]
    """
    if "k_centers" not in z:
        raise KeyError("L3 npz missing k_centers; cannot compute score.")
    for key in ("E_true_k", "E_pred_k", "E_cross_k"):
        if key not in z:
            raise KeyError(f"L3 npz missing {key}; cannot compute score.")

    k = np.asarray(z["k_centers"], dtype=float).reshape(-1)
    Et = np.asarray(z["E_true_k"], dtype=float).reshape(-1)
    Ep = np.asarray(z["E_pred_k"], dtype=float).reshape(-1)
    Ec = np.asarray(z["E_cross_k"], dtype=float).reshape(-1)

    if not (k.size == Et.size == Ep.size == Ec.size):
        raise ValueError("L3 1D arrays shape mismatch; cannot compute score.")

    # rho
    if "rho_k" in z:
        rho = np.asarray(z["rho_k"], dtype=float).reshape(-1)
        if rho.size != k.size:
            raise ValueError("rho_k shape mismatch with k_centers.")
    else:
        denom = np.sqrt(np.maximum(Et, eps) * np.maximum(Ep, eps))
        rho = Ec / np.maximum(denom, eps)

    rho = np.clip(rho, -1.0, 1.0)

    # gain: amplitude ratio (symmetric on log-scale)
    gain = np.sqrt(np.maximum(Ep, eps) / np.maximum(Et, eps))

    # score
    rho01 = np.clip(rho, 0.0, 1.0) if clip_rho_to_01 else rho
    score = rho01 * np.exp(-np.abs(np.log(np.maximum(gain, eps))))
    score = np.clip(score, 0.0, 1.0)

    return k, rho, gain, score


def _as_real_power(x: np.ndarray) -> np.ndarray:
    """
    L3 may store power-like arrays as complex (complex64) by accident or by design.
    For power spectra, we want a non-negative real magnitude.
    Strategy:
      - if complex: use real part if imag is negligible; else use abs.
      - if real: keep.
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        im = np.abs(np.imag(a))
        re = np.abs(np.real(a))
        # heuristic: if imag is tiny relative to real, treat as numerical noise
        if np.nanmax(im) <= 1e-6 * max(1.0, float(np.nanmax(re))):
            a = np.real(a)
        else:
            a = np.abs(a)
    return np.asarray(a, dtype=float)


def derive_rho2d_from_stats(
    *,
    P_true2d: np.ndarray,
    P_pred2d: np.ndarray,
    C_tp2d: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    rho2d = Re(C_tp) / sqrt(P_true * P_pred)
    """
    Pt = _as_real_power(P_true2d)
    Pp = _as_real_power(P_pred2d)
    C = np.asarray(C_tp2d)
    num = np.real(C).astype(float)
    den = np.sqrt(np.maximum(Pt, 0.0) * np.maximum(Pp, 0.0)) + float(eps)
    rho2d = num / den
    # clip to [-1,1] to avoid tiny numerical overshoots
    return np.clip(rho2d, -1.0, 1.0)


def derive_gain2d_from_stats(
    *,
    P_true2d: np.ndarray,
    C_tp2d: np.ndarray,
    eps: float = 1e-12,
    mode: str = "mag_over_true",  # or "re_over_true"
) -> np.ndarray:
    """
    A simple transfer-amplitude diagnostic.
    - mag_over_true: gain2d = |C_tp| / (P_true + eps)
    - re_over_true : gain2d = Re(C_tp) / (P_true + eps)  (signed)
    """
    Pt = _as_real_power(P_true2d)
    C = np.asarray(C_tp2d)
    if mode == "mag_over_true":
        num = np.abs(C).astype(float)
    elif mode == "re_over_true":
        num = np.real(C).astype(float)
    else:
        raise ValueError("mode must be 'mag_over_true' or 're_over_true'")
    den = Pt + float(eps)
    return num / den