# backend/pipelines/eval_mods/scale_mods.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval.utils import write_json
from backend.pipelines.eval_mods.scale_io import (
    build_kgrid_for_2d,
    load_l3_npz,
    load_scale_2d_from_l3_npz,
    snr_from_l3_1d_energy,
    parse_l3_scale_entries,
)
from backend.viz.scale_plots import plot_profile_curves, plot_cutoff_heatmap

# -----------------------------
# tiny numeric helpers (pure; no plotting)
# -----------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, *, dpi: int = 160) -> None:
    if fig is None:
        return
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)


def radial_bin_mean(
    field2d: np.ndarray,
    k2d: np.ndarray,
    k_edges: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Bin-average of a 2D field over radial shells defined by k_edges.
    Returns y_k length nbins = len(k_edges)-1.
    """
    y = np.asarray(field2d, dtype=float)
    k = np.asarray(k2d, dtype=float)
    edges = np.asarray(k_edges, dtype=float)
    nb = int(edges.shape[0] - 1)
    out = np.full((nb,), np.nan, dtype=float)

    # Flatten once
    yv = y.reshape(-1)
    kv = k.reshape(-1)

    for i in range(nb):
        lo = edges[i]
        hi = edges[i + 1]
        m = (kv >= lo) & (kv < hi)
        if not np.any(m):
            continue
        out[i] = float(np.nanmean(yv[m]))
    return out


def k_eff_from_profile(
    k_centers: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float,
    rule: str = "last_above",   # "last_above" or "first_below"
    eps: float = 1e-12,
) -> float:
    """
    Turn a monotone-ish profile into a single cutoff scale.
    - last_above: largest k where y >= threshold
    - first_below: first k where y < threshold (returns previous bin's k)
    """
    k = np.asarray(k_centers, dtype=float)
    yy = np.asarray(y, dtype=float)
    ok = np.isfinite(k) & np.isfinite(yy)
    k = k[ok]
    yy = yy[ok]
    if k.size == 0:
        return float("nan")

    if rule == "last_above":
        idx = np.where(yy >= float(threshold))[0]
        if idx.size == 0:
            return float("nan")
        return float(k[int(idx[-1])])
    elif rule == "first_below":
        idx = np.where(yy < float(threshold))[0]
        if idx.size == 0:
            return float(k[-1])
        j = int(idx[0]) - 1
        j = max(j, 0)
        return float(k[j])
    else:
        raise ValueError(f"Unknown rule='{rule}' (expected 'last_above' or 'first_below').")


# -----------------------------
# internal cache loader
# -----------------------------

def _load_l3_cache(ctx) -> Dict[str, Any]:
    # cached in ctx._caches similar to fourier_mods.py style
    caches = getattr(ctx, "_caches", None)
    if caches is None or not isinstance(caches, dict):
        caches = {}
        setattr(ctx, "_caches", caches)

    if "l3_entries" not in caches:
        l3_index = ctx.l3_index()
        entries = parse_l3_scale_entries(l3_index)
        caches["l3_entries"] = entries
    if "l3_meta" not in caches:
        caches["l3_meta"] = ctx.l3_meta()
    return caches


# -----------------------------
# mods
# -----------------------------

def mod_scale_coherence_curves(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    For each L3 entry:
      - load coh2d
      - radial-bin -> coh(k)
      - dump json + save curve png (batch5)
    """
    pack = _load_l3_cache(ctx)
    entries = pack["l3_entries"]
    l3_meta = pack["l3_meta"]

    out_dir = Path(ctx.paths.l4_root) / "scale" / "coherence"  # type: ignore[attr-defined]
    fig_dir = out_dir / "figs"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    model_types = kwargs.get("model_types", None)
    if model_types is not None:
        model_types = set([str(x) for x in model_types])

    require_2d = bool(kwargs.get("require_2d", True))
    threshold = float(kwargs.get("threshold", 0.5))
    rule = str(kwargs.get("rule", "last_above"))
    xlog = bool(kwargs.get("xlog", True))
    smooth = int(kwargs.get("smooth_window", 1))  # optional: simple smoothing by moving avg (not mandatory)
    show_nyquist = bool(kwargs.get("show_nyquist", True))

    # grid meta for Nyquist / secondary axis (from l3_meta.fft_settings)
    fs = l3_meta.get("fft_settings", {}) if isinstance(l3_meta, dict) else {}
    grid_meta = {
        "dx": float(fs.get("dx", 1.0)),
        "dy": float(fs.get("dy", 1.0)),
        "angular": bool(fs.get("angular", False)),
    }

    rows: List[Dict[str, Any]] = []
    fig_paths: List[str] = []

    for e in entries:
        mt = str(getattr(e, "model_type"))
        if model_types is not None and mt not in model_types:
            continue

        npz_path = Path(e.l3_fft_path)
        z = load_l3_npz(npz_path)

        if "k_edges" not in z or "k_centers" not in z:
            raise KeyError(f"{npz_path} missing k_edges/k_centers; cannot bin coherence.")

        k_edges = np.asarray(z["k_edges"], dtype=float)
        k_centers = np.asarray(z["k_centers"], dtype=float)

        s2d = load_scale_2d_from_l3_npz(z, require=require_2d)
        coh2d = s2d.get("coh2d", None)
        if coh2d is None:
            if require_2d:
                raise KeyError(
                    f"{npz_path} missing coh2d; coherence has no safe fallback. "
                    "Enable save_2d and store coh2d in L3."
                )
            else:
                continue

        coh2d = np.asarray(coh2d, dtype=float)
        H, W = int(coh2d.shape[0]), int(coh2d.shape[1])
        grid = build_kgrid_for_2d(H=H, W=W, l3_meta=l3_meta)
        coh_k = radial_bin_mean(coh2d, grid.k, k_edges)

        # optional smoothing (very light)
        if smooth > 1 and np.isfinite(coh_k).sum() > smooth:
            kk = coh_k.copy()
            for i in range(len(kk)):
                lo = max(0, i - smooth // 2)
                hi = min(len(kk), i + smooth // 2 + 1)
                seg = kk[lo:hi]
                coh_k[i] = float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan

        k_eff = k_eff_from_profile(k_centers, coh_k, threshold=threshold, rule=rule)

        payload = {
            "model_type": mt,
            "mask_rate": float(getattr(e, "mask_rate")),
            "noise_sigma": float(getattr(e, "noise_sigma")),
            "source_npz": str(npz_path),
            "threshold": threshold,
            "rule": rule,
            "k_centers": k_centers.tolist(),
            "coh_k": np.asarray(coh_k, dtype=float).tolist(),
            "k_eff": float(k_eff),
        }
        out_name = f"{mt}_p{float(getattr(e,'mask_rate')):.6g}_s{float(getattr(e,'noise_sigma')):.6g}".replace(".", "_")
        json_path = out_dir / f"{out_name}.json"
        write_json(json_path, payload)

        # --- FIG (batch5) ---
        fig = plot_profile_curves(
            [
                {
                    "k": k_centers,
                    "y": coh_k,
                    "label": "coherence(k)",
                    "k_eff": k_eff if np.isfinite(k_eff) else None,
                    "k_eff_label": rf"$k_{{eff}}$={k_eff:.3g} (coh≥{threshold:g})" if np.isfinite(k_eff) else None,
                    "style": {"linewidth": 2.0},
                }
            ],
            title=f"coherence(k) | {mt} | p={payload['mask_rate']:.3g} σ={payload['noise_sigma']:.3g}",
            xlabel="wavenumber k",
            ylabel="coherence",
            xlog=xlog,
            ylog=False,
            ylim=(0.0, 1.02),
            grid_meta=grid_meta,
            show_nyquist=show_nyquist,
            show_secondary_lambda=True,
            secondary_label="ℓ = 1/k (length unit)",
        )
        fig_path = fig_dir / f"{out_name}.png"
        _save_fig(fig, fig_path)
        fig_paths.append(str(fig_path))

        rows.append({
            "model_type": mt,
            "mask_rate": payload["mask_rate"],
            "noise_sigma": payload["noise_sigma"],
            "k_eff": float(k_eff),
            "json": str(json_path),
            "fig": str(fig_path),
        })

    write_json(out_dir / "index.json", {"items": rows, "threshold": threshold, "rule": rule})

    return {
        "out_dir": str(out_dir),
        "count": int(len(rows)),
        "threshold": threshold,
        "rule": rule,
        "fig_paths": fig_paths,
    }


def mod_scale_snr_curves(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    SNR(k) curves:
      - fallback to 1D E_true_k/E_err_k if 2D isn't stored (safe)
      - dump json + save curve png (batch5)
    """
    pack = _load_l3_cache(ctx)
    entries = pack["l3_entries"]
    l3_meta = pack["l3_meta"]

    out_dir = Path(ctx.paths.l4_root) / "scale" / "snr"  # type: ignore[attr-defined]
    fig_dir = out_dir / "figs"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    model_types = kwargs.get("model_types", None)
    if model_types is not None:
        model_types = set([str(x) for x in model_types])

    eps = float(kwargs.get("eps", 1e-12))
    log10 = bool(kwargs.get("log10", True))
    # If log10(SNR): threshold 0 means SNR>=1; if linear SNR: threshold 1 means SNR>=1
    threshold = float(kwargs.get("threshold", 0.0 if log10 else 1.0))
    rule = str(kwargs.get("rule", "last_above"))
    xlog = bool(kwargs.get("xlog", True))
    ylog = bool(kwargs.get("ylog", False))  # usually False since we already log10
    show_nyquist = bool(kwargs.get("show_nyquist", True))

    fs = l3_meta.get("fft_settings", {}) if isinstance(l3_meta, dict) else {}
    grid_meta = {
        "dx": float(fs.get("dx", 1.0)),
        "dy": float(fs.get("dy", 1.0)),
        "angular": bool(fs.get("angular", False)),
    }

    rows: List[Dict[str, Any]] = []
    fig_paths: List[str] = []

    for e in entries:
        mt = str(getattr(e, "model_type"))
        if model_types is not None and mt not in model_types:
            continue

        npz_path = Path(e.l3_fft_path)
        z = load_l3_npz(npz_path)

        k_centers, snr_k = snr_from_l3_1d_energy(z, eps=eps, log10=log10)
        if log10:
            snr_k = np.asarray(snr_k, dtype=float)
            k_centers = np.asarray(k_centers, dtype=float)

            floor = -12.0
            tol = 1e-9  # guard for float noise

            m = np.isfinite(k_centers) & np.isfinite(snr_k) & (snr_k > floor + tol)
            # IMPORTANT: actually drop these points (not set to NaN), so matplotlib connects.
            k_centers = k_centers[m]
            snr_k = snr_k[m]
        else:
            # non-log10: keep only finite and positive
            snr_k = np.asarray(snr_k, dtype=float)
            k_centers = np.asarray(k_centers, dtype=float)
            m = np.isfinite(k_centers) & np.isfinite(snr_k) & (snr_k > 0)
            k_centers = k_centers[m]
            snr_k = snr_k[m]
        k_eff = k_eff_from_profile(k_centers, snr_k, threshold=threshold, rule=rule)

        payload = {
            "model_type": mt,
            "mask_rate": float(getattr(e, "mask_rate")),
            "noise_sigma": float(getattr(e, "noise_sigma")),
            "source_npz": str(npz_path),
            "eps": eps,
            "log10": log10,
            "threshold": threshold,
            "rule": rule,
            "k_centers": np.asarray(k_centers, dtype=float).tolist(),
            "snr_k": np.asarray(snr_k, dtype=float).tolist(),
            "k_eff": float(k_eff),
        }
        out_name = f"{mt}_p{float(getattr(e,'mask_rate')):.6g}_s{float(getattr(e,'noise_sigma')):.6g}".replace(".", "_")
        json_path = out_dir / f"{out_name}.json"
        write_json(json_path, payload)

        # --- FIG (batch5) ---
        ylab = "log10(SNR(k))" if log10 else "SNR(k)"
        fig = plot_profile_curves(
            [
                {
                    "k": k_centers,
                    "y": snr_k,
                    "label": ylab,
                    "k_eff": k_eff if np.isfinite(k_eff) else None,
                    "k_eff_label": rf"$k_{{eff}}$={k_eff:.3g} (SNR≥{(10**threshold if log10 else threshold):g})" if np.isfinite(k_eff) else None,
                    "style": {"linewidth": 2.0},
                }
            ],
            title=f"{ylab} | {mt} | p={payload['mask_rate']:.3g} σ={payload['noise_sigma']:.3g}",
            xlabel="wavenumber k",
            ylabel=ylab,
            xlog=xlog,
            ylog=ylog,
            ylim=None,
            grid_meta=grid_meta,
            show_nyquist=show_nyquist,
            show_secondary_lambda=True,
            secondary_label="ℓ = 1/k (length unit)",
        )
        fig_path = fig_dir / f"{out_name}.png"
        _save_fig(fig, fig_path)
        fig_paths.append(str(fig_path))

        rows.append({
            "model_type": mt,
            "mask_rate": payload["mask_rate"],
            "noise_sigma": payload["noise_sigma"],
            "k_eff": float(k_eff),
            "json": str(json_path),
            "fig": str(fig_path),
        })

    write_json(out_dir / "index.json", {"items": rows, "threshold": threshold, "rule": rule, "log10": log10})

    return {
        "out_dir": str(out_dir),
        "count": int(len(rows)),
        "threshold": threshold,
        "rule": rule,
        "log10": log10,
        "fig_paths": fig_paths,
    }


def mod_scale_cutoff_heatmap(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build k_eff heatmap matrix data AND save heatmap figure (batch5).
    """
    metric = str(kwargs.get("metric", "coherence"))  # "coherence" or "snr"
    model_type = str(kwargs.get("model_type", "linear"))

    base_dir = Path(ctx.paths.l4_root) / "scale"  # type: ignore[attr-defined]
    if metric == "coherence":
        idx_path = base_dir / "coherence" / "index.json"
    elif metric == "snr":
        idx_path = base_dir / "snr" / "index.json"
    else:
        raise ValueError("metric must be 'coherence' or 'snr'")

    if not idx_path.exists():
        raise FileNotFoundError(
            f"Missing {idx_path}. Run mod 'scale.{metric}_curves' first "
            "(or include it earlier in assemble)."
        )

    import json
    with idx_path.open("r", encoding="utf-8") as f:
        idx = json.load(f)

    items = idx.get("items", [])
    items = [it for it in items if str(it.get("model_type")) == model_type]

    mrs = sorted(set([float(it["mask_rate"]) for it in items]))
    nss = sorted(set([float(it["noise_sigma"]) for it in items]))

    mat = np.full((len(nss), len(mrs)), np.nan, dtype=float)
    for it in items:
        mr = float(it["mask_rate"])
        ns = float(it["noise_sigma"])
        val = float(it.get("k_eff", float("nan")))
        yi = nss.index(ns)
        xi = mrs.index(mr)
        mat[yi, xi] = val

    out_dir = Path(ctx.paths.l4_root) / "scale" / "heatmaps"  # type: ignore[attr-defined]
    fig_dir = out_dir / "figs"
    _ensure_dir(out_dir)
    _ensure_dir(fig_dir)

    out = {
        "metric": metric,
        "model_type": model_type,
        "mask_rates": mrs,
        "noise_sigmas": nss,
        "k_eff_matrix": mat.tolist(),
        "source_index": str(idx_path),
        "threshold": idx.get("threshold", None),
        "rule": idx.get("rule", None),
        "log10": idx.get("log10", None),
    }
    out_path = out_dir / f"{metric}_{model_type}_k_eff.json"
    write_json(out_path, out)

    # --- FIG (batch5) ---
    l3_meta = _load_l3_cache(ctx).get("l3_meta", {})
    fs = l3_meta.get("fft_settings", {}) if isinstance(l3_meta, dict) else {}
    grid_meta = {
        "dx": float(fs.get("dx", 1.0)),
        "dy": float(fs.get("dy", 1.0)),
        "angular": bool(fs.get("angular", False)),
    }

    info_lines = [
        f"metric={metric}",
        f"model={model_type}",
        f"threshold={out.get('threshold')}",
        f"rule={out.get('rule')}",
    ]
    if metric == "snr":
        info_lines.append(f"log10={out.get('log10')}")

    fig = plot_cutoff_heatmap(
        mask_rates=mrs,
        noise_sigmas=nss,
        k_eff_matrix=mat,
        title=f"k_eff heatmap ({metric}) | {model_type}",
        grid_meta=grid_meta,
        show_numbers=bool(kwargs.get("show_numbers", True)),
        number_fmt_k=str(kwargs.get("number_fmt_k", "{:.3g}")),
        number_fmt_ell=str(kwargs.get("number_fmt_ell", "{:.3g}")),
        use_log10_ell=bool(kwargs.get("use_log10_ell", False)),
        info_lines=info_lines,
    )
    fig_path = fig_dir / f"{metric}_{model_type}_k_eff.png"
    _save_fig(fig, fig_path)

    return {
        "out_path": str(out_path),
        "metric": metric,
        "model_type": model_type,
        "fig_paths": [str(fig_path)],
    }


def register_scale_mods() -> None:
    register_mod(
        EvalMod(
            name="scale.coherence_curves",
            run=mod_scale_coherence_curves,
            description="Compute coherence radial curves coh(k) and dump per-entry json/figs.",
        )
    )
    register_mod(
        EvalMod(
            name="scale.snr_curves",
            run=mod_scale_snr_curves,
            description="Compute SNR radial curves snr(k) (fallback to 1D energies) and dump per-entry json/figs.",
        )
    )
    register_mod(
        EvalMod(
            name="scale.cutoff_heatmap",
            run=mod_scale_cutoff_heatmap,
            description="Build k_eff heatmap matrix data (and figs) from scale curves outputs.",
        )
    )
