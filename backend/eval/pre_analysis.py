# backend/pre_analysis.py
# v2.0: Level-3 artifacts (FFT packages) computed from stored Level-2 raw predictions.

"""Level-3 (FFT) pipeline.

This module is intentionally *storage-driven*:
  - It does NOT depend on in-memory results from training/reconstruction.
  - It reads Level-2 artifacts from exp_dir and produces Level-3 FFT packages.

Design notes
  - Level-3 stores *base data packages* that Level-4 tools can reuse for plots / composite analyses.
  - By default we store **radial-binned** spectra (k-centers) because it's compact and stable.
  - We also provide an explicit API to compute **2D FFT (kx,ky)** on-demand from a spatial field,
    so Level-4 can still do deep dives without having to re-run training.

File layout (recommended, short paths)
  exp_dir/
    config_used.yaml            # saved by Level-2 after training finishes
    L2/                         # or L2_rebuild/ (legacy)
      meta.json
      linear/*.npz
      mlp/*.npz
    L3_fft/
      meta.json
      linear/*.npz
      mlp/*.npz
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np

from backend.config.schemas import EvalConfig
from backend.config.yaml_io import load_experiment_yaml
from backend.dataio.io_utils import ensure_dir, load_json, save_json
from backend.dataio.io_utils import load_numpy  # used by POD artifacts
from backend.dataio.nc_loader import load_raw_nc
from backend.pod.project import reconstruct_from_pod
from backend.fourier.filters import make_wavenumber_grid, fft2_field
from backend.fourier.filters import radial_bin_spectrum
from backend.metrics.fourier_metrics import (
    fourier_cumulative_nrmse_curve_from_energy,
    kstar_from_cumulative_curve,
)

# -----------------------------
# Level-3 npz key specification
# and counter tools
# -----------------------------

def _encode_p_s_codes(mask_rate: float, noise_sigma: float, *, scale: int = 10000) -> Tuple[int, int]:
    # Encode floats to fixed-point integer codes (same convention as Level-2)
    p_code = int(np.round(float(mask_rate) * float(scale)))
    s_code = int(np.round(float(noise_sigma) * float(scale)))
    return p_code, s_code

def _tqdm(iterable, *, total: Optional[int] = None, desc: str = "", disable: bool = False):
    # Optional progress bar wrapper (no hard dependency on tqdm)
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc, disable=disable)
    except Exception:
        return iterable

# -----------------------------
# Public entrypoints
# -----------------------------

def run_pre_analysis_from_storage(
    *,
    exp_dir: str | Path,
    eval_cfg: EvalConfig,
    model_types: Sequence[str] = ("linear",),
    verbose: bool = True,
) -> Dict[str, Any]:
    # Level-3 FFT pipeline from disk.
    exp_dir = Path(exp_dir)
    ensure_dir(exp_dir)

    # Load the config used by L2, so we can locate nc_path / pod_dir.
    cfg_path = _pick_config_used_yaml(exp_dir)
    if cfg_path is None:
        raise FileNotFoundError(
            "Cannot find config_used.yaml under exp_dir. "
            "Expected one of: config_used.yaml / config.yaml / experiment.yaml"
        )
    data_cfg, pod_cfg, _, _ = load_experiment_yaml(cfg_path)

    # Load POD artifacts (Level-1) and raw data once.
    Ur, mean_flat, pod_meta = _load_pod_artifacts(pod_cfg.save_dir)
    X_thwc = load_raw_nc(data_cfg)  # [T,H,W,C]
    T, H, W, C = (int(X_thwc.shape[0]), int(X_thwc.shape[1]), int(X_thwc.shape[2]), int(X_thwc.shape[3]))
    D = H * W * C
    if Ur.shape[0] != D:
        raise ValueError(f"POD Ur first dim {Ur.shape[0]} != H*W*C={D} from data")

    fs = _get_fft_settings(eval_cfg, H=H, W=W)
    if not fs["enabled"]:
        if verbose:
            print("[L3] Fourier disabled in eval_cfg.fourier.enabled=False, skip.")
        return {
            "root": str((exp_dir / "L3_fft").resolve()),
            "meta": {"enabled": False},
            "entries": [],
            "model_types": list(model_types),
        }

    # Locate Level-2 root and discover prediction files.
    l2_root = _pick_l2_root(exp_dir)
    if l2_root is None:
        raise FileNotFoundError("Cannot find Level-2 root under exp_dir (expected L2/ or L2_rebuild/).")

    l3_root = exp_dir / "L3_fft"
    ensure_dir(l3_root)

    # meta.json: record schema + npz keys for quick inspection
    l3_npz_keys = _level3_npz_key_spec()
    meta = {
        "schema_version": "v2.0-L3",
        "enabled": True,
        "source": {
            "exp_dir": str(exp_dir),
            "l2_root": str(l2_root),
            "config_used": str(cfg_path),
        },
        "fft_settings": fs,
        "npz_keys": l3_npz_keys,
    }
    save_json(l3_root / "meta.json", meta)

    entries_out: List[Dict[str, Any]] = []

    # Shared grids for efficiency
    grid = make_wavenumber_grid(H=H, W=W, dx=float(fs["dx"]), dy=float(fs["dy"]), angular=bool(fs["angular"]))
    k_max_eval = float(_k_nyquist_axis_min(grid))  # min(Nyquist_x, Nyquist_y)

    # Discover all Level-2 pred files for requested model_types.
    l2_items = _discover_level2_pred_files(l2_root=l2_root, model_types=model_types, verbose=verbose)

    if verbose:
        print(f"[L3] Found {len(l2_items)} Level-2 pred files under {l2_root}")

    for item in l2_items:
        model_type = str(item["model_type"])
        pred_path = Path(item["pred_path"])
        mask_rate = float(item["mask_rate"])
        noise_sigma = float(item["noise_sigma"])

        if verbose:
            print(f"[L3] FFT for {model_type}: p={mask_rate:g}, s={noise_sigma:g} <- {pred_path.name}")

        # Load A_hat_all + minimal meta from L2 npz.
        pred_npz = np.load(pred_path, allow_pickle=False)
        A_hat_all, l2_key_hint = _extract_A_hat_all(pred_npz, pred_path=pred_path)
        r_eff = int(A_hat_all.shape[1])

        # Use only first r_eff columns of POD basis.
        Ur_eff = Ur[:, :r_eff]

        # Select sampled frames
        t_samples = _select_sample_frames(T, int(fs["sample_frames"]))
        if len(t_samples) == 0:
            if verbose:
                print("  -> skip: no sample frames")
            continue

        # Accumulate radial spectra across sampled frames.
        B = int(fs["num_bins"])
        Et_sum = np.zeros((B,), dtype=np.float64)
        Ep_sum = np.zeros((B,), dtype=np.float64)
        Ee_sum = np.zeros((B,), dtype=np.float64)
        Ec_sum = np.zeros((B,), dtype=np.float64)
        Ct_sum = np.zeros((B,), dtype=np.float64)
        k_centers_ref = None
        k_edges_ref = None

        it = _tqdm(
            t_samples,
            total=len(t_samples),
            desc=f"[L3:{model_type}] p={mask_rate:g}, s={noise_sigma:g} FFT",
            disable=not verbose,
        )

        for t in it:
            x_true = X_thwc[t]  # [H,W,C]

            a_hat = A_hat_all[t]  # [r_eff]
            x_pred = reconstruct_from_pod(a_hat, Ur_eff, mean_flat).reshape(H, W, C)

            # FFT (supports HWC internally)
            F_true = fft2_field(x_true, mean_mode=str(fs["mean_mode_true"]))
            F_pred = fft2_field(x_pred, mean_mode=str(fs["mean_mode_true"]))
            F_err = F_pred - F_true

            # Radial spectra
            k_centers, Et, k_edges, count_k = radial_bin_spectrum(
                F_true,
                grid,
                num_bins=B,
                k_max=k_max_eval,
                binning=str(fs["binning"]),
                k_min=fs.get("k_min", None),
                drop_first_bin=bool(fs.get("drop_first_bin", False)),
            )
            _, Ep, _, _ = radial_bin_spectrum(
                F_pred,
                grid,
                num_bins=B,
                k_max=k_max_eval,
                binning=str(fs["binning"]),
                k_min=fs.get("k_min", None),
                drop_first_bin=bool(fs.get("drop_first_bin", False)),
            )
            _, Ee, _, _ = radial_bin_spectrum(
                F_err,
                grid,
                num_bins=B,
                k_max=k_max_eval,
                binning=str(fs["binning"]),
                k_min=fs.get("k_min", None),
                drop_first_bin=bool(fs.get("drop_first_bin", False)),
            )
            Ec = _radial_bin_cross_energy(
                F_pred=F_pred,
                F_true=F_true,
                grid=grid,
                num_bins=B,
                k_max=k_max_eval,
                binning=str(fs["binning"]),
                k_min=fs.get("k_min", None),
                drop_first_bin=bool(fs.get("drop_first_bin", False)),
            )[1]

            if k_centers_ref is None:
                k_centers_ref = np.asarray(k_centers, dtype=np.float64)
                k_edges_ref = np.asarray(k_edges, dtype=np.float64)

            Et_sum += np.asarray(Et, dtype=np.float64)
            Ep_sum += np.asarray(Ep, dtype=np.float64)
            Ee_sum += np.asarray(Ee, dtype=np.float64)
            Ec_sum += np.asarray(Ec, dtype=np.float64)
            Ct_sum += np.asarray(count_k, dtype=np.float64)

        if k_centers_ref is None or k_edges_ref is None:
            if verbose:
                print("  -> skip: k bins not initialized")
            continue

        # Derived radial packages (compact base data)
        eps = 1e-12
        empty = Ct_sum <= 0
        nrmse_k = np.sqrt(Ee_sum / (Et_sum + eps))
        nrmse_k = np.asarray(nrmse_k, dtype=np.float64)
        nrmse_k[empty] = np.nan

        # Spectral correlation (per radial bin): rho = Re<Fp,Ft> / sqrt(Ep*Et)
        rho_k = Ec_sum / np.sqrt((Ep_sum + eps) * (Et_sum + eps))
        rho_k = np.asarray(rho_k, dtype=np.float64)
        rho_k[empty] = np.nan

        # Cumulative low-pass curve (use true/error energies)
        k_eval, nrmse_cum, cum_meta = fourier_cumulative_nrmse_curve_from_energy(
            k_centers=k_centers_ref,
            E_true_k=Et_sum,
            E_err_k=Ee_sum,
            count_k=Ct_sum,
            k_min_eval=float(fs.get("k_min_eval", 0.25)),
            eps=eps,
            monotone_enforce=bool(fs.get("kstar_cum_monotone", True)),
        )

        # Optional: k* from cumulative plateau (still “base data” enough to keep)
        k_star, kstar_dbg = kstar_from_cumulative_curve(
            np.asarray(k_eval, dtype=np.float64),
            np.asarray(nrmse_cum, dtype=np.float64),
            eps_plateau=float(fs.get("kstar_plateau_eps", 1e-3)),
            m_plateau=int(fs.get("kstar_plateau_m", 4)),
            k_min_eval=float(fs.get("k_min_eval", 0.25)),
            prefer_last_if_not_found=bool(fs.get("kstar_prefer_last", True)),
        )

                # Save Level-3 npz (flat layout, same style as Level-2)
        p_code, s_code = _encode_p_s_codes(mask_rate, noise_sigma, scale=10000)
        out_name = f"{model_type}_p{p_code:04d}_s{s_code:04d}.npz"
        out_path = l3_root / out_name

        np.savez_compressed(
            out_path,
            # --- axes / binning ---
            k_centers=np.asarray(k_centers_ref, dtype=np.float64),
            k_edges=np.asarray(k_edges_ref, dtype=np.float64),
            count_k=np.asarray(Ct_sum, dtype=np.float64),
            # --- base energies ---
            E_true_k=np.asarray(Et_sum, dtype=np.float64),
            E_pred_k=np.asarray(Ep_sum, dtype=np.float64),
            E_err_k=np.asarray(Ee_sum, dtype=np.float64),
            E_cross_k=np.asarray(Ec_sum, dtype=np.float64),
            # --- derived compact metrics ---
            nrmse_k=np.asarray(nrmse_k, dtype=np.float64),
            rho_k=np.asarray(rho_k, dtype=np.float64),
            nrmse_cum=np.asarray(nrmse_cum, dtype=np.float64),
            k_eval=np.asarray(k_eval, dtype=np.float64),
            # --- scalar hints ---
            k_star=float(k_star),
        )

        entries_out.append(
            {
                "model_type": model_type,
                "mask_rate": mask_rate,
                "noise_sigma": noise_sigma,
                "l2_pred_path": str(pred_path),
                "l2_key_hint": l2_key_hint,
                "l3_fft_path": str(out_path),
                "r_eff": int(r_eff),
                "frames": list(t_samples),
                "k_star": float(k_star),
                "kstar_debug": {
                    "found": bool(kstar_dbg.get("found", False)),
                    "reason": kstar_dbg.get("reason", None),
                    "k_star_idx": kstar_dbg.get("k_star_idx", None),
                },
                "cum_meta": {
                    "k_min_eval": cum_meta.get("k_min_eval", None),
                    "start_idx": int(cum_meta.get("start_idx")) if cum_meta.get("start_idx", None) is not None else None,
                },
            }
        )

    index = {
        "root": str(l3_root),
        "meta_path": str(l3_root / "meta.json"),
        "entries": entries_out,
        "model_types": list(model_types),
    }
    save_json(l3_root / "index.json", index)

    if verbose:
        print(f"[L3] Saved index.json with {len(entries_out)} entries -> {l3_root / 'index.json'}")

    return index


def compute_fft2_for_field(
    x_hw_or_hwc: np.ndarray,
    *,
    grid_meta: Dict[str, Any],
    mean_mode: str = "none",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # On-demand 2D FFT interface for deep dives (Level-4 tools can call this).
    H, W = _infer_hw_from_field(x_hw_or_hwc)
    dx, dy, angular = _infer_dxdy_angular(grid_meta=grid_meta, H=H, W=W)
    grid = make_wavenumber_grid(H=H, W=W, dx=dx, dy=dy, angular=angular)
    F = fft2_field(x_hw_or_hwc, mean_mode=str(mean_mode))
    return F, {"kx": grid.kx, "ky": grid.ky, "k": grid.k}


def read_level3_fft_npz(npz_path: str | Path) -> Dict[str, Any]:
    # Quick reader for ipynb inspection / plotting.
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["_path"] = str(npz_path)
    out["_keys"] = list(out.keys())
    return out


# -----------------------------
# Internal helpers (no docstrings; use # comments only)
# -----------------------------

def _pick_config_used_yaml(exp_dir: Path) -> Optional[Path]:
    cand = [
        exp_dir / "config_used.yaml",
        exp_dir / "config.yaml",
        exp_dir / "experiment.yaml",
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def _pick_l2_root(exp_dir: Path) -> Optional[Path]:
    cand = [
        exp_dir / "L2",
        exp_dir / "L2_rebuild",
    ]
    for p in cand:
        if p.exists():
            return p
    return None


def _load_pod_artifacts(pod_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pod_dir = Path(pod_dir)
    Ur = load_numpy(pod_dir / "Ur.npy")
    mean_flat = load_numpy(pod_dir / "mean_flat.npy")
    meta = load_json(pod_dir / "pod_meta.json")
    return Ur, mean_flat, meta


def _infer_hw_from_field(x: np.ndarray) -> Tuple[int, int]:
    x = np.asarray(x)
    if x.ndim == 2:
        return int(x.shape[0]), int(x.shape[1])
    if x.ndim == 3:
        # (H,W,C) or (C,H,W)
        if x.shape[0] <= 8 and x.shape[1] > 8 and x.shape[2] > 8:
            return int(x.shape[1]), int(x.shape[2])
        return int(x.shape[0]), int(x.shape[1])
    raise ValueError(f"Unsupported x shape: {x.shape}")


def _infer_dxdy_angular(*, grid_meta: Dict[str, Any], H: int, W: int) -> Tuple[float, float, bool]:
    grid_meta = dict(grid_meta or {})
    angular = bool(grid_meta.get("angular", False))

    if "dx" in grid_meta and "dy" in grid_meta:
        dx = float(grid_meta["dx"])
        dy = float(grid_meta["dy"])
        return dx, dy, angular

    if "Lx" in grid_meta and "Ly" in grid_meta:
        Lx = float(grid_meta["Lx"])
        Ly = float(grid_meta["Ly"])
        dx = Lx / float(W) if W > 0 else 1.0
        dy = Ly / float(H) if H > 0 else 1.0
        return float(dx), float(dy), angular

    dx = float(grid_meta.get("dx", 1.0))
    dy = float(grid_meta.get("dy", 1.0))
    return dx, dy, angular


def _get_fft_settings(eval_cfg: EvalConfig, *, H: int, W: int) -> Dict[str, Any]:
    f = getattr(eval_cfg, "fourier", None)
    if f is None:
        return {"enabled": False}

    enabled = bool(getattr(f, "enabled", False))
    grid_meta = dict(getattr(f, "grid_meta", {}) or {})
    dx, dy, angular = _infer_dxdy_angular(grid_meta=grid_meta, H=H, W=W)

    binning = str(getattr(f, "binning", "log")).strip().lower()
    num_bins = int(getattr(f, "num_bins", 64))
    k_min_eval = float(getattr(f, "k_min_eval", 0.25))
    sample_frames = int(getattr(f, "sample_frames", 8))
    mean_mode_true = str(getattr(f, "mean_mode_true", "global"))

    # plateau params (if present in schema)
    eps_plateau = float(getattr(f, "kstar_plateau_eps", 1e-3))
    m_plateau = int(getattr(f, "kstar_plateau_m", 4))
    cum_monotone = bool(getattr(f, "kstar_cum_monotone", True))
    prefer_last = bool(getattr(f, "kstar_prefer_last", True))

    # optional bin controls (not always in schema; safe defaults)
    k_min = getattr(f, "k_min", None)
    drop_first_bin = bool(getattr(f, "drop_first_bin", False))

    return {
        "enabled": enabled,
        "dx": float(dx),
        "dy": float(dy),
        "angular": bool(angular),
        "binning": str(binning),
        "num_bins": int(num_bins),
        "k_min_eval": float(k_min_eval),
        "sample_frames": int(sample_frames),
        "mean_mode_true": str(mean_mode_true),
        "kstar_plateau_eps": float(eps_plateau),
        "kstar_plateau_m": int(m_plateau),
        "kstar_cum_monotone": bool(cum_monotone),
        "kstar_prefer_last": bool(prefer_last),
        "k_min": None if k_min is None else float(k_min),
        "drop_first_bin": bool(drop_first_bin),
    }


def _select_sample_frames(T: int, n: int) -> List[int]:
    # n = -1  -> use all frames [0..T-1]
    # n <= 0  -> use none
    # otherwise -> uniform sampling of n frames across [0..T-1]
    if T <= 0:
        return []
    n = int(n)
    if n == -1:
        return list(range(int(T)))
    if n <= 0:
        return []
    n = min(n, int(T))
    return [int(x) for x in np.linspace(0, int(T) - 1, num=n, dtype=int)]


def _k_nyquist_axis_min(grid) -> float:
    # Nyquist along x: 1/(2*dx) (cycles/len) or pi/dx (rad/len)
    if bool(getattr(grid, "angular", False)):
        kNx = float(np.pi) / float(grid.dx)
        kNy = float(np.pi) / float(grid.dy)
        return float(min(kNx, kNy))
    kNx = 1.0 / (2.0 * float(grid.dx))
    kNy = 1.0 / (2.0 * float(grid.dy))
    return float(min(kNx, kNy))


def _discover_level2_pred_files(
    *,
    l2_root: Path,
    model_types: Sequence[str],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    l2_root = Path(l2_root)

    # 扫描策略：
    # - 不依赖 meta.json 的 npz_keys / npz_key_map（因为 L2/meta.json 里目前写的是 entry_npz_schema）
    # - 直接在 l2_root 下递归找 *.npz
    # - 从文件名（优先）或父目录推断 model_type / p / s
    # - 最终按 (model_type, p, s) 去重，优先选“路径更短”的文件（扁平布局优先）

    npz_paths = sorted(l2_root.rglob("*.npz"))

    for p in npz_paths:
        if p.name.startswith("_"):
            continue

        mt, mr, ns = _parse_l2_filename_or_path(p)
        if mt is None or mr is None or ns is None:
            continue

        # 只保留指定 model_types
        if str(mt) not in set(str(x) for x in model_types):
            continue

        items.append(
            {
                "model_type": str(mt),
                "mask_rate": float(mr),
                "noise_sigma": float(ns),
                "pred_path": str(p),
            }
        )

    if verbose and len(items) == 0:
        print(f"[L3] WARNING: no Level-2 .npz found under {l2_root} for model_types={list(model_types)}")

    # Deduplicate by (model_type, p, s) keeping the shortest path (prefer flat layout)
    best: Dict[Tuple[str, float, float], Dict[str, Any]] = {}
    for it in items:
        k = (it["model_type"], float(it["mask_rate"]), float(it["noise_sigma"]))
        if k not in best:
            best[k] = it
        else:
            if len(str(it["pred_path"])) < len(str(best[k]["pred_path"])):
                best[k] = it

    return list(best.values())


def _parse_l2_filename_or_path(p: Path) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    # 只支持扁平化命名：<model>_pXXXX_sXXXX(.npz)
    # 其中 XXXX 为定点整数编码（默认 scale=10000），例如 p0004 => 0.0004
    stem = p.stem

    if "_p" not in stem or "_s" not in stem:
        return None, None, None

    try:
        mt = stem.split("_p", 1)[0]
        rest = stem.split("_p", 1)[1]
        p_code_str, s_part = rest.split("_s", 1)

        # 允许后缀（例如 linear_p0004_s0000_v2.npz）
        s_code_str = s_part
        for sep in ["_", "-"]:
            if sep in s_code_str:
                s_code_str = s_code_str.split(sep, 1)[0]

        p_code = int(p_code_str)
        s_code = int(s_code_str)

        scale = 10000.0
        mr = float(p_code) / scale
        ns = float(s_code) / scale
        return str(mt), mr, ns
    except Exception:
        return None, None, None


def _extract_A_hat_all(npz: np.lib.npyio.NpzFile, *, pred_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Try a list of common keys used in your codebase.
    # We DO NOT invent upstream writer names; we only accept what exists in file.
    candidates = [
        "A_hat_all",
        "a_hat_all",
        "A_hat",
        "a_hat",
        "pred_coeffs",
        "coeffs",
    ]

    found_key = None
    for k in candidates:
        if k in npz.files:
            found_key = k
            break

    if found_key is None:
        raise KeyError(
            f"Cannot find A_hat_all in {pred_path}. "
            f"Available keys: {list(npz.files)}. "
            "Expected one of: " + ", ".join(candidates)
        )

    A = np.asarray(npz[found_key])
    if A.ndim != 2:
        raise ValueError(f"A_hat_all must be 2D [T,r], got shape {A.shape} from key='{found_key}'")

    return A.astype(np.float64, copy=False), {"A_hat_key": found_key, "available": list(npz.files)}


def _level3_npz_key_spec() -> Dict[str, Any]:
    return {
        "k_centers": "(B,) float64",
        "k_edges": "(B+1,) float64",
        "count_k": "(B,) float64",
        "E_true_k": "(B,) float64",
        "E_pred_k": "(B,) float64",
        "E_err_k": "(B,) float64",
        "E_cross_k": "(B,) float64 (Re<Fp,Ft>/N)",
        "nrmse_k": "(B,) float64",
        "rho_k": "(B,) float64",
        "k_eval": "(B,) float64",
        "nrmse_cum": "(B,) float64",
        "k_star": "float",
    }


def _radial_bin_cross_energy(
    *,
    F_pred: np.ndarray,
    F_true: np.ndarray,
    grid,
    num_bins: int,
    k_max: float,
    binning: str,
    k_min: Optional[float],
    drop_first_bin: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Radial-binned cross-energy: Re(sum_c Fp_c * conj(Ft_c))/N.
    Fp = np.asarray(F_pred)
    Ft = np.asarray(F_true)

    if Fp.shape != Ft.shape:
        raise ValueError(f"F_pred shape {Fp.shape} != F_true shape {Ft.shape}")

    if Fp.ndim == 2:
        H, W = Fp.shape
        N = H * W
        cross_samples = (Fp * np.conj(Ft)).real / float(N)
    elif Fp.ndim == 3:
        C, H, W = Fp.shape
        N = H * W
        cross_samples = np.sum((Fp * np.conj(Ft)).real, axis=0) / float(N)
    else:
        raise ValueError(f"Unsupported FFT shape: {Fp.shape}")

    k = np.asarray(grid.k, dtype=np.float64)
    if k_max is None:
        k_max = float(np.max(k))
    k_max = float(k_max)

    binning = str(binning).lower().strip()
    if binning == "linear":
        edges = np.linspace(0.0, k_max, int(num_bins) + 1, dtype=np.float64)
    elif binning == "log":
        if k_min is None:
            # Conservative: pick smallest positive step from grid (similar to filters._infer_default_k_min_from_grid)
            dkx = 1.0 / (float(grid.W) * float(grid.dx))
            dky = 1.0 / (float(grid.H) * float(grid.dy))
            dk = min(dkx, dky)
            if bool(getattr(grid, "angular", False)):
                dk = 2.0 * np.pi * dk
            k_min = float(max(dk, 1e-12))
        edges = np.geomspace(float(k_min), k_max, int(num_bins) + 1).astype(np.float64)
    else:
        raise ValueError(f"binning must be 'linear' or 'log', got {binning}")

    centers = 0.5 * (edges[:-1] + edges[1:])

    k_flat = k.reshape(-1)
    c_flat = np.asarray(cross_samples, dtype=np.float64).reshape(-1)

    idx = np.digitize(k_flat, edges, right=False) - 1
    B = int(num_bins)
    Ck = np.zeros((B,), dtype=np.float64)
    count_k = np.zeros((B,), dtype=np.int64)
    valid = (idx >= 0) & (idx < B)
    np.add.at(Ck, idx[valid], c_flat[valid])
    np.add.at(count_k, idx[valid], 1)

    if drop_first_bin:
        if B <= 1:
            return centers[:0], Ck[:0], edges[:1], count_k[:0]
        centers = centers[1:]
        Ck = Ck[1:]
        count_k = count_k[1:]
        edges = edges[1:]

    return centers, Ck, edges, count_k

"""
Level-3 entry loader / summarizer / plotter
"""
def load_l3_npz(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.shape == ():
            out[k] = v.item()
        else:
            out[k] = v
    out["_path"] = str(path)
    out["_keys"] = list(data.files)
    return out


def summarize_l3_npz(path: str | Path, *, max_array_elems: int = 8) -> Dict[str, Any]:
    path = Path(path)
    d = load_l3_npz(path)

    keys = [k for k in d.keys() if not k.startswith("_")]
    summary: Dict[str, Any] = {"path": str(path), "keys": keys, "items": {}}

    for k in keys:
        v = d[k]
        if isinstance(v, np.ndarray):
            info = {"type": "ndarray", "dtype": str(v.dtype), "shape": tuple(v.shape)}
            if v.size <= max_array_elems:
                info["value"] = v
            summary["items"][k] = info
        else:
            summary["items"][k] = {"type": type(v).__name__, "value": v}

    # Basic schema sanity checks (non-throwing; just a report)
    report = _l3_schema_sanity_report(d)
    summary["sanity"] = report
    return summary


def plot_l3_fft_preview(
    path: str | Path,
    *,
    show_energy: bool = True,
    show_nrmse: bool = True,
    show_rho: bool = True,
    show_cum: bool = True,
    xscale: str = "log",
    max_k: Optional[float] = None,
):
    import matplotlib.pyplot as plt

    d = load_l3_npz(path)

    k = d.get("k_centers", None)
    if k is None:
        raise KeyError("Missing key 'k_centers' in L3 npz.")
    k = np.asarray(k, dtype=float)

    # Optional max_k clip for display
    mask = np.ones_like(k, dtype=bool)
    if max_k is not None:
        mask = mask & (k <= float(max_k))
    k_plot = k[mask]

    # Prepare figures
    figs = {}

    # 1) Energy spectra (true/pred/err)
    if show_energy:
        Et = d.get("E_true_k", None)
        Ep = d.get("E_pred_k", None)
        Ee = d.get("E_err_k", None)

        if Et is not None or Ep is not None or Ee is not None:
            fig = plt.figure()
            if Et is not None:
                plt.plot(k_plot, np.asarray(Et, dtype=float)[mask], label="E_true_k")
            if Ep is not None:
                plt.plot(k_plot, np.asarray(Ep, dtype=float)[mask], label="E_pred_k")
            if Ee is not None:
                plt.plot(k_plot, np.asarray(Ee, dtype=float)[mask], label="E_err_k")
            plt.xlabel("k")
            plt.ylabel("radial energy")
            plt.title(f"[L3] Energy spectra: {Path(path).name}")
            if xscale:
                plt.xscale(xscale)
            plt.legend()
            plt.tight_layout()
            figs["energy"] = fig

    # 2) NRMSE(k) and rho(k)
    if show_nrmse or show_rho:
        y_any = False
        fig = plt.figure()
        if show_nrmse and "nrmse_k" in d:
            plt.plot(k_plot, np.asarray(d["nrmse_k"], dtype=float)[mask], label="nrmse_k")
            y_any = True
        if show_rho and "rho_k" in d:
            plt.plot(k_plot, np.asarray(d["rho_k"], dtype=float)[mask], label="rho_k")
            y_any = True
        if y_any:
            plt.xlabel("k")
            plt.ylabel("metric")
            plt.title(f"[L3] Radial metrics: {Path(path).name}")
            if xscale:
                plt.xscale(xscale)
            plt.legend()
            plt.tight_layout()
            figs["metrics"] = fig

    # 3) Cumulative curve and k*
    if show_cum and ("k_eval" in d) and ("nrmse_cum" in d):
        k_eval = np.asarray(d["k_eval"], dtype=float)
        nrmse_cum = np.asarray(d["nrmse_cum"], dtype=float)

        mask2 = np.ones_like(k_eval, dtype=bool)
        if max_k is not None:
            mask2 = mask2 & (k_eval <= float(max_k))

        fig = plt.figure()
        plt.plot(k_eval[mask2], nrmse_cum[mask2], label="nrmse_cum")
        if "k_star" in d:
            k_star = float(d["k_star"])
            if np.isfinite(k_star) and k_star > 0:
                plt.axvline(k_star, linestyle="--", label=f"k*={k_star:.3g}")
        plt.xlabel("k")
        plt.ylabel("cumulative nrmse")
        plt.title(f"[L3] Cumulative curve: {Path(path).name}")
        if xscale:
            plt.xscale(xscale)
        plt.legend()
        plt.tight_layout()
        figs["cumulative"] = fig

    # Return a dict of figures to make ipynb usage convenient
    return figs


def _l3_schema_sanity_report(d: Dict[str, Any]) -> Dict[str, Any]:
    # Non-throwing sanity checks. Returns a compact report.
    required = ["k_centers", "count_k", "E_true_k", "E_pred_k", "E_err_k", "E_cross_k"]
    optional = ["nrmse_k", "rho_k", "k_eval", "nrmse_cum", "k_star", "k_edges"]

    present = set(k for k in d.keys() if not k.startswith("_"))
    missing_required = [k for k in required if k not in present]
    missing_optional = [k for k in optional if k not in present]

    shapes = {}
    for k in ["k_centers", "count_k", "E_true_k", "E_pred_k", "E_err_k", "E_cross_k", "nrmse_k", "rho_k"]:
        if k in d and isinstance(d[k], np.ndarray):
            shapes[k] = tuple(d[k].shape)

    # Check shape consistency among radial arrays
    baseK = shapes.get("k_centers", None)
    inconsistent = []
    for k, sh in shapes.items():
        if k == "k_centers":
            continue
        if baseK is not None and sh != baseK:
            inconsistent.append((k, sh, baseK))

    # Quick runtime plausibility heuristics
    heuristics = {}
    if "count_k" in d and isinstance(d["count_k"], np.ndarray):
        Ct = np.asarray(d["count_k"], dtype=float)
        heuristics["count_k_nonzero_bins"] = int(np.sum(Ct > 0))
        heuristics["count_k_min"] = float(np.nanmin(Ct)) if Ct.size else None
        heuristics["count_k_max"] = float(np.nanmax(Ct)) if Ct.size else None

    if "E_true_k" in d and isinstance(d["E_true_k"], np.ndarray):
        Et = np.asarray(d["E_true_k"], dtype=float)
        heuristics["E_true_k_sum"] = float(np.nansum(Et))

    if "E_err_k" in d and isinstance(d["E_err_k"], np.ndarray):
        Ee = np.asarray(d["E_err_k"], dtype=float)
        heuristics["E_err_k_sum"] = float(np.nansum(Ee))

    if "k_star" in d:
        try:
            heuristics["k_star"] = float(d["k_star"])
        except Exception:
            heuristics["k_star"] = None

    return {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "shape_inconsistency": inconsistent,
        "heuristics": heuristics,
    }
