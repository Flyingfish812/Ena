# backend/pipelines/eval_mods/fourier_mods.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from backend.pipelines.eval.registry import EvalMod, register_mod

from backend.pipelines.eval_mods.fourier_io import (
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
    plot_energy_spectra_with_band_edges,
    plot_kstar_curve_from_entry,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, *, dpi: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def _ensure_ctx_caches(ctx: Any) -> Dict[str, Any]:
    # 新引擎应当提供 ctx.caches；若你还没加，先在这里兜底挂上
    if not hasattr(ctx, "caches") or getattr(ctx, "caches") is None:
        setattr(ctx, "caches", {})
    caches: Dict[str, Any] = getattr(ctx, "caches")
    return caches


def _load_l3_cache(ctx: Any) -> Dict[str, Any]:
    """
    统一缓存键：'L3'
    读取仍复用 fourier_io.read_l3_index/read_l3_meta（它们按 exp_dir/L3_fft 查找）。
    """
    caches = _ensure_ctx_caches(ctx)
    if "L3" in caches:
        return caches["L3"]

    assert ctx.paths is not None, "EvalContext must be resolved (ctx.paths is None)."
    exp_dir = ctx.paths.exp_dir

    l3_index = read_l3_index(exp_dir)
    l3_meta = read_l3_meta(exp_dir)
    entries = parse_l3_entries(l3_index)

    pack = {"index": l3_index, "meta": l3_meta, "entries": entries}
    caches["L3"] = pack
    return pack


def _resolve_fourier_cfg(ctx: Any) -> Any:
    """
    新引擎统一入口：ctx.eval_cfg.fourier
    """
    eval_cfg = getattr(ctx, "eval_cfg", None)
    if eval_cfg is None:
        raise ValueError("ctx.eval_cfg missing (new L4 unified schema expects eval_cfg on context).")
    f = getattr(eval_cfg, "fourier", None)
    if f is None:
        raise ValueError("eval_cfg.fourier missing")
    return f


# ----------------------------
# Mod: k* heatmap
# ----------------------------

def mod_fourier_kstar_heatmap(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    fourier_cfg = _resolve_fourier_cfg(ctx)

    model_types: Sequence[str] = kwargs.get("model_types", None) or tuple(
        sorted(set([e.model_type for e in entries]))
    )

    assert ctx.paths is not None
    out_dir = Path(ctx.paths.l4_root) / "fourier"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    mention: Dict[str, Any] = {}

    for mt in model_types:
        df = build_fourier_df_from_l3(entries=entries, eval_cfg_fourier=fourier_cfg, model_type=str(mt))
        if df is None or len(df) == 0:
            # 新引擎不强制 ctx.log；这里用 print 保持最小依赖
            print(f"[L4:fourier.kstar_heatmap] skip {mt}: empty df")
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

def mod_fourier_band_nrmse_curves(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    fourier_cfg = _resolve_fourier_cfg(ctx)

    model_types: Sequence[str] = kwargs.get("model_types", None) or tuple(
        sorted(set([e.model_type for e in entries]))
    )

    band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H")))

    assert ctx.paths is not None
    out_dir = Path(ctx.paths.l4_root) / "fourier"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    df_lin = build_fourier_df_from_l3(entries=entries, eval_cfg_fourier=fourier_cfg, model_type="linear")
    df_mlp = build_fourier_df_from_l3(entries=entries, eval_cfg_fourier=fourier_cfg, model_type="mlp")
    figs = plot_fourier_band_nrmse_curves(df_lin=df_lin, df_mlp=df_mlp, band_names=band_names)
    for key, fig in (figs or {}).items():
        if fig is None:
            print(f"[L4:fourier.band_nrmse_curves] Warning: skip {key}: no figure")
            continue
        png = out_dir / f"{key}.png"
        _save_fig(fig, png, dpi=int(kwargs.get("dpi", 180)))
        fig_paths.append(str(png))

    return {"fig_paths": fig_paths}


# ----------------------------
# Mod: E(k) + band edges (legend / explanation)
# ----------------------------

def mod_fourier_energy_spectrum_legend(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
        E_pred_k = np.asarray(z["E_pred_k"], dtype=float) if "E_pred_k" in z.files else None
        E_err_k  = np.asarray(z["E_err_k"], dtype=float)  if "E_err_k"  in z.files else None
        E_cross_k = np.asarray(z["E_cross_k"], dtype=float) if "E_cross_k" in z.files else None

    grid_meta = dict(getattr(fourier_cfg, "grid_meta", {}) or {})

    # ---- Fig A: keep the original "true spectrum + edges" ----
    fig_true = plot_energy_spectrum_with_band_edges(
        k_centers=k_centers,
        energy_k=E_true_k,
        k_edges=k_edges_interior,
        band_names=band_names,
        grid_meta=grid_meta,
        title=str(kwargs.get("title_true", "True energy spectrum E_true(k) with band edges")),
    )

    # ---- Fig B: comparison plot (true/pred/err/cross) ----
    curves = {"E_true_k": E_true_k}
    if E_pred_k is not None:
        curves["E_pred_k"] = E_pred_k
    if E_err_k is not None:
        curves["E_err_k"] = E_err_k
    if E_cross_k is not None:
        curves["E_cross_k"] = E_cross_k  # function will plot abs if needed on log axis

    fig_cmp = plot_energy_spectra_with_band_edges(
        k_centers=k_centers,
        curves=curves,
        k_edges=k_edges_interior,
        band_names=band_names,
        grid_meta=grid_meta,
        title=str(kwargs.get("title_cmp", "Energy spectra comparison with band edges")),
    )

    assert ctx.paths is not None
    out_dir = Path(ctx.paths.l4_root) / "fourier"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []

    if fig_true is not None:
        png_true = out_dir / "energy_spectrum_true_with_band_edges.png"
        _save_fig(fig_true, png_true, dpi=int(kwargs.get("dpi", 180)))
        fig_paths.append(str(png_true))

    if fig_cmp is not None:
        png_cmp = out_dir / "energy_spectrum_compare_with_band_edges.png"
        _save_fig(fig_cmp, png_cmp, dpi=int(kwargs.get("dpi", 180)))
        fig_paths.append(str(png_cmp))

    return {"fig_paths": fig_paths, "source_npz": str(pick)}


# ----------------------------
# Mod: per-cfg k* curve plots
# ----------------------------

def mod_fourier_kstar_curves(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    pack = _load_l3_cache(ctx)
    entries = pack["entries"]
    if not entries:
        return {"fig_paths": []}

    fourier_cfg = _resolve_fourier_cfg(ctx)
    band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H")))
    k_edges_interior = lambda_edges_to_k_edges_interior(fourier_cfg)
    grid_meta = dict(getattr(fourier_cfg, "grid_meta", {}) or {})

    max_plots = kwargs.get("max_plots", None)
    max_plots = None if max_plots is None else int(max_plots)

    entries_sorted = sorted(entries, key=lambda e: (e.model_type, e.noise_sigma, -e.mask_rate))

    assert ctx.paths is not None
    out_dir = Path(ctx.paths.l4_root) / "fourier" / "kstar_curves"
    _ensure_dir(out_dir)

    fig_paths: List[str] = []
    n = 0

    for e in entries_sorted:
        if max_plots is not None and n >= max_plots:
            break

        npz_path = Path(e.l3_fft_path)
        if not npz_path.exists():
            continue

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


def register_fourier_mods() -> None:
    """
    新引擎统一注册入口：直接注册到 backend.pipelines.eval.registry 的全局注册表
    """
    register_mod(
        EvalMod(
            name="fourier.kstar_heatmap",
            requires=(),
            description="k* heatmap per model_type from L3_fft",
            run=mod_fourier_kstar_heatmap,
        )
    )
    register_mod(
        EvalMod(
            name="fourier.band_nrmse_curves",
            requires=(),
            description="Fourier band NRMSE curves (vs p / vs sigma) from L3_fft energies",
            run=mod_fourier_band_nrmse_curves,
        )
    )
    register_mod(
        EvalMod(
            name="fourier.energy_spectrum_legend",
            requires=(),
            description="E(k) with band edges explanation plot",
            run=mod_fourier_energy_spectrum_legend,
        )
    )
    register_mod(
        EvalMod(
            name="fourier.kstar_curves",
            requires=(),
            description="Per-(p,σ) k* curve plots from L3_fft npz",
            run=mod_fourier_kstar_curves,
        )
    )
