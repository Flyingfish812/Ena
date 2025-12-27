# backend/pipelines/eval_mods/fourier_mods.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .core import EvalContext, EvalMod, ModRegistry
from .fourier_io import (
    read_l3_index,
    read_l3_meta,
    parse_l3_entries,
    build_fourier_df_from_l3,
    pick_representative_l3_npz,
    lambda_edges_to_k_edges_interior,
)

from backend.viz.fourier_plots import (
    plot_kstar_heatmap,
    plot_fourier_band_nrmse_curves,
    plot_energy_spectrum_with_band_edges,
    plot_kstar_curve_from_entry,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _load_l3_cache(ctx: EvalContext) -> Dict[str, Any]:
    # cache key: "L3"
    if "L3" in ctx.caches:
        return ctx.caches["L3"]

    l3_index = read_l3_index(ctx.exp_dir)
    l3_meta = read_l3_meta(ctx.exp_dir)
    entries = parse_l3_entries(l3_index)

    pack = {
        "index": l3_index,
        "meta": l3_meta,
        "entries": entries,
    }
    ctx.caches["L3"] = pack
    return pack


def _resolve_fourier_cfg(ctx: EvalContext) -> Any:
    eval_cfg = ctx.cfg.get("eval_cfg", None)
    if eval_cfg is None:
        raise ValueError("ctx.cfg.eval_cfg missing")
    f = getattr(eval_cfg, "fourier", None)
    if f is None:
        raise ValueError("eval_cfg.fourier missing")
    return f


# ----------------------------
# Mod: k* heatmap
# ----------------------------

def mod_fourier_kstar_heatmap(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    fourier_cfg = _resolve_fourier_cfg(ctx)

    model_types: Sequence[str] = kwargs.get("model_types", None) or tuple(
        sorted(set([e.model_type for e in entries]))
    )

    out_dir = Path(ctx.out_dir) / "fourier"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    mention: Dict[str, Any] = {}

    for mt in model_types:
        df = build_fourier_df_from_l3(entries=entries, eval_cfg_fourier=fourier_cfg, model_type=str(mt))
        if df is None or len(df) == 0:
            ctx.log(f"[L4:fourier.kstar_heatmap] skip {mt}: empty df")
            continue

        fig = plot_kstar_heatmap(
            df_lin=df,
            df_mlp=None,
            title=f"k* heatmap ({mt})",
            model=str(mt),
            grid_meta=dict(getattr(fourier_cfg, "grid_meta", {}) or {}),
            show_numbers=bool(kwargs.get("show_numbers", True)),
            use_log10_ell=bool(kwargs.get("use_log10_ell", False)),
        )
        if fig is None:
            continue

        png = out_dir / f"kstar_heatmap_{mt}.png"
        _save_fig(fig, png, dpi=int(kwargs.get("dpi", 180)))
        fig_paths.append(str(png))
        mention[str(mt)] = {"png": str(png)}

    return {"fig_paths": fig_paths, "by_model": mention}


# ----------------------------
# Mod: Fourier band NRMSE curves
# ----------------------------

def mod_fourier_band_nrmse_curves(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    fourier_cfg = _resolve_fourier_cfg(ctx)

    model_types: Sequence[str] = kwargs.get("model_types", None) or tuple(
        sorted(set([e.model_type for e in entries]))
    )

    band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H")))

    out_dir = Path(ctx.out_dir) / "fourier"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    for mt in model_types:
        df = build_fourier_df_from_l3(entries=entries, eval_cfg_fourier=fourier_cfg, model_type=str(mt))
        if df is None or len(df) == 0:
            continue

        figs = plot_fourier_band_nrmse_curves(df_lin=df, df_mlp=None, band_names=band_names)
        # figs: dict[str, fig]
        for key, fig in (figs or {}).items():
            if fig is None:
                continue
            png = out_dir / f"{key}_{mt}.png"
            _save_fig(fig, png, dpi=int(kwargs.get("dpi", 180)))
            fig_paths.append(str(png))

    return {"fig_paths": fig_paths}


# ----------------------------
# Mod: E(k) + band edges (legend / explanation)
# ----------------------------

def mod_fourier_energy_spectrum_legend(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    if not entries:
        return {"fig_paths": []}

    fourier_cfg = _resolve_fourier_cfg(ctx)
    band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H")))
    k_edges_interior = lambda_edges_to_k_edges_interior(fourier_cfg)

    pick = pick_representative_l3_npz(entries=entries, prefer_model=str(kwargs.get("prefer_model", "linear")))
    if pick is None:
        return {"fig_paths": []}

    with np.load(pick, allow_pickle=False) as z:
        k_centers = np.asarray(z["k_centers"], dtype=float)
        E_true_k = np.asarray(z["E_true_k"], dtype=float)

    fig = plot_energy_spectrum_with_band_edges(
        k_centers=k_centers,
        energy_k=E_true_k,
        k_edges=k_edges_interior,
        band_names=band_names,
        grid_meta=dict(getattr(fourier_cfg, "grid_meta", {}) or {}),
        title=str(kwargs.get("title", "Energy spectrum E(k) with band edges")),
    )
    if fig is None:
        return {"fig_paths": []}

    out_dir = Path(ctx.out_dir) / "fourier"
    _ensure_dir(out_dir)
    png = out_dir / "energy_spectrum_with_band_edges.png"
    _save_fig(fig, png, dpi=int(kwargs.get("dpi", 180)))

    return {"fig_paths": [str(png)], "source_npz": str(pick)}


# ----------------------------
# Mod: per-cfg k* curve plots
# ----------------------------

def mod_fourier_kstar_curves(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    if not entries:
        return {"fig_paths": []}

    fourier_cfg = _resolve_fourier_cfg(ctx)
    band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H")))
    k_edges_interior = lambda_edges_to_k_edges_interior(fourier_cfg)
    grid_meta = dict(getattr(fourier_cfg, "grid_meta", {}) or {})

    # 限制数量避免一次性爆炸（默认全画，但你可以传 max_plots）
    max_plots = kwargs.get("max_plots", None)
    max_plots = None if max_plots is None else int(max_plots)

    # 排序：更“像基准”的配置优先（噪声小、观测多）
    entries_sorted = sorted(entries, key=lambda e: (e.model_type, e.noise_sigma, -e.mask_rate))

    out_dir = Path(ctx.out_dir) / "fourier" / "kstar_curves"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    n = 0

    for e in entries_sorted:
        if max_plots is not None and n >= max_plots:
            break

        npz_path = Path(e.l3_fft_path)
        if not npz_path.exists():
            continue

        # 组装成旧版 plot_kstar_curve_from_entry 兼容的 entry dict
        with np.load(npz_path, allow_pickle=False) as z:
            curve = {
                "k_centers": z["k_centers"].tolist(),
                "k_edges": z["k_edges"].tolist(),
                "count_k": z["count_k"].tolist(),
                "E_true_k": z["E_true_k"].tolist(),
                "E_pred_k": z["E_pred_k"].tolist(),
                "E_err_k": z["E_err_k"].tolist(),
                "E_cross_k": z["E_cross_k"].tolist(),
                "nrmse_k": z["nrmse_k"].tolist(),
                "rho_k": z["rho_k"].tolist(),
                "nrmse_cum": z["nrmse_cum"].tolist(),
                "k_eval": z["k_eval"].tolist(),
                "k_star": float(z["k_star"]),
            }

        entry = {
            "model_type": e.model_type,
            "mask_rate": e.mask_rate,
            "noise_sigma": e.noise_sigma,
            "fourier_curve": curve,
            "fourier_meta": {
                "grid_meta": grid_meta,
                "band_names": list(band_names),
                "lambda_edges": list(getattr(fourier_cfg, "lambda_edges", (1.0, 0.25))),
            },
        }

        fig = plot_kstar_curve_from_entry(
            entry,
            title=f"k* curve | {e.model_type} | p={e.mask_rate:.4g}, σ={e.noise_sigma:.4g}",
            k_edges=k_edges_interior,
            band_names=band_names,
            grid_meta=grid_meta,
            show_local_curve=bool(kwargs.get("show_local_curve", True)),
        )
        if fig is None:
            continue

        p_code = int(round(float(e.mask_rate) * 10000))
        s_code = int(round(float(e.noise_sigma) * 10000))
        png = out_dir / f"kstar_curve_{e.model_type}_p{p_code:04d}_s{s_code:04d}.png"
        _save_fig(fig, png, dpi=int(kwargs.get("dpi", 180)))
        fig_paths.append(str(png))
        n += 1

    return {"fig_paths": fig_paths, "count": n}


def register_fourier_mods(registry: ModRegistry) -> None:
    registry.register(
        EvalMod(
            name="fourier.kstar_heatmap",
            fn=mod_fourier_kstar_heatmap,
            description="k* heatmap per model_type from L3_fft",
        )
    )
    registry.register(
        EvalMod(
            name="fourier.band_nrmse_curves",
            fn=mod_fourier_band_nrmse_curves,
            description="Fourier band NRMSE curves (vs p / vs sigma) from L3_fft energies",
        )
    )
    registry.register(
        EvalMod(
            name="fourier.energy_spectrum_legend",
            fn=mod_fourier_energy_spectrum_legend,
            description="E(k) with band edges explanation plot",
        )
    )
    registry.register(
        EvalMod(
            name="fourier.kstar_curves",
            fn=mod_fourier_kstar_curves,
            description="Per-(p,σ) k* curve plots from L3_fft npz",
        )
    )
