# backend/pipelines/eval_mods/examples_mods.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backend.pipelines.eval_mods.examples_io import reconstruct_fields_from_l2, ensure_subdir
from backend.viz.field_plots import plot_recon_triptych
from backend.viz.fourier_plots import plot_fft2_spectrum_triptych, plot_spatial_fourier_band_decomposition
import matplotlib.pyplot as plt


def _save_fig(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _k_edges_from_lambda_edges(lambda_edges: List[float]) -> List[float]:
    """
    你的工程里 k = 1/lambda（与之前的实现一致）。
    lambda_edges: [1.0, 0.25] => k_edges: [1.0, 4.0]
    """
    le = [float(x) for x in lambda_edges]
    out = []
    for x in le:
        if x <= 0:
            raise ValueError(f"lambda_edges must be positive, got {lambda_edges}")
        out.append(1.0 / x)
    # 这里返回“分带边界（不含 0 与 inf）”，交给 plot_spatial_fourier_band_decomposition 自己处理
    return out


def mod_examples_triptych(ctx: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    空间域：pred/true/err 三联图（每组 cfg 抽若干帧）
    """
    assert ctx.paths is not None

    out_dir = ensure_subdir(ctx.paths.l4_root, "examples_triptych")
    fig_paths: List[str] = []

    sample_frames = int(kwargs.get("sample_frames", 8))
    channel = int(kwargs.get("channel", 0))
    seed = int(kwargs.get("seed", 0))
    dpi = int(kwargs.get("dpi", 180))
    show_mask = bool(kwargs.get("show_mask", False))

    # 遍历：model_types x (mask_rate, noise_sigma)
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
            mask_hw = pack.get("mask_hw", None)

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
    """
    空间域：傅里叶分带 iFFT 联图（pred/true/err，每个 band 一行/一列，取决于你旧函数实现）
    """
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
    lambda_edges = list(getattr(fourier_cfg, "lambda_edges", [1.0, 0.25]))
    k_edges = _k_edges_from_lambda_edges(lambda_edges)

    # grid: dx/dy 优先用 eval_cfg.fourier.grid（你之前就是这么分的）
    grid = getattr(fourier_cfg, "grid", None)
    dx = float(getattr(grid, "dx", 1.0)) if grid is not None else 1.0
    dy = float(getattr(grid, "dy", 1.0)) if grid is not None else 1.0

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
                # 旧函数一般签名：plot_spatial_fourier_band_decomposition(x_true_hw, x_pred_hw, k_edges, band_names, dx, dy, ...)
                fig = plot_spatial_fourier_band_decomposition(
                    x_true_hw=x_true[i],
                    x_pred_hw=x_pred[i],
                    k_edges=k_edges,
                    band_names=band_names,
                    dx=dx,
                    dy=dy,
                    channel=0,  # 这里已经是单通道 HW，旧函数若还要 channel 就给 0
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
    """
    新增：FFT2 空间频域联图
    """
    assert ctx.paths is not None
    out_dir = ensure_subdir(ctx.paths.l4_root, "examples_fft2_triptych")
    fig_paths: List[str] = []

    sample_frames = int(kwargs.get("sample_frames", 4))
    channel = int(kwargs.get("channel", 0))
    seed = int(kwargs.get("seed", 0))
    dpi = int(kwargs.get("dpi", 180))
    log_scale = bool(kwargs.get("log_scale", True))

    fourier_cfg = getattr(ctx.eval_cfg, "fourier", None)
    grid = getattr(fourier_cfg, "grid", None) if fourier_cfg is not None else None
    dx = float(getattr(grid, "dx", 1.0)) if grid is not None else 1.0
    dy = float(getattr(grid, "dy", 1.0)) if grid is not None else 1.0

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


def register_example_mods(registry: Any) -> None:
    """
    挂到你现有的 ModRegistry（pipelines_code.md 里那个）
    """
    from backend.pipelines.eval_mods import EvalMod  # 你 __init__ 里应当 export 了

    registry.register(
        EvalMod(
            name="examples.triptych",
            fn=mod_examples_triptych,
            description="Spatial-domain triptych (pred/true/err) from L2 coeffs + POD basis",
        )
    )
    registry.register(
        EvalMod(
            name="examples.fourier_band_decomp",
            fn=mod_examples_fourier_band_decomp,
            description="Spatial Fourier band iFFT decomposition (pred/true/err) per frame",
        )
    )
    registry.register(
        EvalMod(
            name="examples.fft2_triptych",
            fn=mod_examples_fft2_triptych,
            description="2D FFT magnitude triptych (pred/true/err), shared colorbar",
        )
    )
