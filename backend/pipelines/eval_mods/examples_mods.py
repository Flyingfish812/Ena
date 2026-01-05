# backend/pipelines/eval_mods/examples_mods.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval_mods.examples_io import reconstruct_fields_from_l2, ensure_subdir
from backend.viz.field_plots import plot_recon_triptych
from backend.viz.fourier_plots import (
    plot_fft2_spectrum_triptych,
    plot_spatial_fourier_band_decomposition,
)


def _save_fig(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _k_edges_from_lambda_edges(lambda_edges: List[float]) -> List[float]:
    """
    k = 1/lambda
    返回“分带边界（不含 0 与 inf）”
    """
    le = [float(x) for x in lambda_edges]
    out: List[float] = []
    for x in le:
        if x <= 0:
            raise ValueError(f"lambda_edges must be positive, got {lambda_edges}")
        out.append(1.0 / x)
    return out


def _require_fourier_grid_meta(ctx: Any) -> Dict[str, Any]:
    fourier_cfg = getattr(ctx.eval_cfg, "fourier", None)
    if fourier_cfg is None:
        raise ValueError("eval_cfg.fourier missing (required for this examples mod)")

    grid_meta = getattr(fourier_cfg, "grid_meta", None)
    if grid_meta is None:
        raise ValueError("eval_cfg.fourier.grid_meta missing (required; unified schema)")

    if not isinstance(grid_meta, dict):
        # 允许是 dataclass/namespace 风格，但最终要转 dict 使用
        grid_meta = {k: getattr(grid_meta, k) for k in ("dx", "dy") if hasattr(grid_meta, k)}
    return grid_meta


def mod_examples_triptych(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """空间域：pred/true/err 三联图（每组 cfg 抽若干帧）"""
    assert ctx.paths is not None

    out_dir = ensure_subdir(ctx.paths.l4_root, "examples_triptych")
    fig_paths: List[str] = []

    sample_frames = int(kwargs.get("sample_frames", 8))
    channel = int(kwargs.get("channel", 0))
    seed = int(kwargs.get("seed", 0))
    dpi = int(kwargs.get("dpi", 180))
    show_mask = bool(kwargs.get("show_mask", True))

    for mt in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            pack = reconstruct_fields_from_l2(
                ctx,
                model_type=str(mt),
                mask_rate=float(p),
                noise_sigma=float(s),
                channel=channel,
                sample_frames=sample_frames,
                seed=seed,
            )
            frames = pack["frames"]
            x_pred = pack["x_pred"]
            x_true = pack["x_true"]
            mask_hw = pack["mask_hw"]

            for i, t in enumerate(frames):
                fig = plot_recon_triptych(
                    x_pred[i],
                    x_true[i],
                    title=f"triptych | {mt} | p={float(p):.4g}, σ={float(s):.4g} | t={t}",
                    mask_hw=mask_hw,
                    show_mask=show_mask,
                )
                p_code = int(round(float(p) * 10000))
                s_code = int(round(float(s) * 10000))
                png = out_dir / f"triptych_{mt}_p{p_code:04d}_s{s_code:04d}_t{int(t):04d}.png"
                _save_fig(fig, png, dpi=dpi)
                fig_paths.append(str(png))

    return {"fig_paths": fig_paths, "count": len(fig_paths), "out_dir": str(out_dir)}


def mod_examples_fourier_band_decomp(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """空间域：傅里叶分带 iFFT 联图（pred/true/err per band）"""
    assert ctx.paths is not None

    out_dir = ensure_subdir(ctx.paths.l4_root, "examples_band_decomp")
    fig_paths: List[str] = []

    sample_frames = int(kwargs.get("sample_frames", 4))
    channel = int(kwargs.get("channel", 0))
    seed = int(kwargs.get("seed", 0))
    dpi = int(kwargs.get("dpi", 180))

    fourier_cfg = getattr(ctx.eval_cfg, "fourier", None)
    if fourier_cfg is None:
        raise ValueError("eval_cfg.fourier missing (required for examples.fourier_band_decomp)")

    band_names = list(getattr(fourier_cfg, "band_names", ["L", "M", "H"]))
    lambda_edges = list(getattr(fourier_cfg, "lambda_edges", []))
    if len(lambda_edges) == 0:
        raise ValueError("eval_cfg.fourier.lambda_edges missing/empty (required; unified schema)")
    k_edges = _k_edges_from_lambda_edges(lambda_edges)

    grid_meta = _require_fourier_grid_meta(ctx)
    dx = float(grid_meta.get("dx", 1.0))
    dy = float(grid_meta.get("dy", 1.0))

    for mt in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            pack = reconstruct_fields_from_l2(
                ctx,
                model_type=str(mt),
                mask_rate=float(p),
                noise_sigma=float(s),
                channel=channel,
                sample_frames=sample_frames,
                seed=seed,
            )
            frames = pack["frames"]
            x_pred = pack["x_pred"]
            x_true = pack["x_true"]

            for i, t in enumerate(frames):
                fig = plot_spatial_fourier_band_decomposition(
                    x_true_hw=x_true[i],
                    x_pred_hw=x_pred[i],
                    k_edges=k_edges,
                    band_names=band_names,
                    dx=dx,
                    dy=dy,
                    channel=0,
                )
                if fig is None:
                    continue
                p_code = int(round(float(p) * 10000))
                s_code = int(round(float(s) * 10000))
                png = out_dir / f"band_decomp_{mt}_p{p_code:04d}_s{s_code:04d}_t{int(t):04d}.png"
                _save_fig(fig, png, dpi=dpi)
                fig_paths.append(str(png))

    return {"fig_paths": fig_paths, "count": len(fig_paths), "out_dir": str(out_dir)}


def mod_examples_fft2_triptych(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """FFT2 空间频域联图（pred vs true）"""
    assert ctx.paths is not None

    out_dir = ensure_subdir(ctx.paths.l4_root, "examples_fft2_triptych")
    fig_paths: List[str] = []

    sample_frames = int(kwargs.get("sample_frames", 4))
    channel = int(kwargs.get("channel", 0))
    seed = int(kwargs.get("seed", 0))
    dpi = int(kwargs.get("dpi", 180))
    log_scale = bool(kwargs.get("log_scale", True))

    grid_meta = _require_fourier_grid_meta(ctx)
    dx = float(grid_meta.get("dx", 1.0))
    dy = float(grid_meta.get("dy", 1.0))

    for mt in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            pack = reconstruct_fields_from_l2(
                ctx,
                model_type=str(mt),
                mask_rate=float(p),
                noise_sigma=float(s),
                channel=channel,
                sample_frames=sample_frames,
                seed=seed,
            )
            frames = pack["frames"]
            x_pred = pack["x_pred"]
            x_true = pack["x_true"]

            for i, t in enumerate(frames):
                fig = plot_fft2_spectrum_triptych(
                    x_pred[i],
                    x_true[i],
                    title=f"FFT2 | {mt} | p={float(p):.4g}, σ={float(s):.4g} | t={t}",
                    dx=dx,
                    dy=dy,
                    log_scale=log_scale,
                )
                p_code = int(round(float(p) * 10000))
                s_code = int(round(float(s) * 10000))
                png = out_dir / f"fft2_{mt}_p{p_code:04d}_s{s_code:04d}_t{int(t):04d}.png"
                _save_fig(fig, png, dpi=dpi)
                fig_paths.append(str(png))

    return {"fig_paths": fig_paths, "count": len(fig_paths), "out_dir": str(out_dir)}


def register_example_mods() -> None:
    """
    统一引擎注册入口：
    - 不再接收 registry 参数
    - 直接注册到 backend.pipelines.eval.registry 的全局注册表
    """
    register_mod(
        EvalMod(
            name="examples.triptych",
            requires=(),
            description="Spatial-domain triptych (pred/true/err) from L2 coeffs + POD basis",
            run=mod_examples_triptych,
        )
    )
    register_mod(
        EvalMod(
            name="examples.fourier_band_decomp",
            requires=(),
            description="Spatial Fourier band iFFT decomposition (pred/true/err) per frame",
            run=mod_examples_fourier_band_decomp,
        )
    )
    register_mod(
        EvalMod(
            name="examples.fft2_triptych",
            requires=(),
            description="2D FFT magnitude triptych (pred/true), shared params from grid_meta",
            run=mod_examples_fft2_triptych,
        )
    )
