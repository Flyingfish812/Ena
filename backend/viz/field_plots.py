# backend/viz/field_plots.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ..dataio.io_utils import ensure_dir
from typing import Dict

def _to_2d(field: np.ndarray) -> np.ndarray:
    """
    将 [H,W] 或 [H,W,C] 的场裁剪为 [H,W]，默认取第 0 个通道。
    """
    f = np.asarray(field)
    if f.ndim == 2:
        return f
    elif f.ndim == 3:
        return f[..., 0]
    else:
        raise ValueError(f"Expected [H,W] or [H,W,C], got {f.shape}")


def plot_field_comparison(
    x_true_hw: np.ndarray,
    x_lin_hw: np.ndarray | None = None,
    x_nn_hw: np.ndarray | None = None,
    title_prefix: str = "",
    names: tuple[str, ...] | None = None,
) -> plt.Figure:
    """
    对比单个样本的空间场：

    - x_true
    - x_lin 线性基线（可选）
    - x_nn  MLP（可选）

    设计要点：
    - 所有子图统一使用 RdBu_r 色图；
    - 第一行子图共享一个水平 colorbar；
    - 若提供 names，用其覆盖默认标签。
    """
    # 收集字段
    fields: list[tuple[str, np.ndarray]] = [("True", _to_2d(x_true_hw))]
    if x_lin_hw is not None:
        fields.append(("Linear", _to_2d(x_lin_hw)))
    if x_nn_hw is not None:
        fields.append(("MLP", _to_2d(x_nn_hw)))

    # 覆盖默认名称
    if names is not None:
        if len(names) != len(fields):
            raise ValueError(f"names 有 {len(names)} 个，但实际字段有 {len(fields)} 个")
        fields = list(zip(names, [f for _, f in fields], strict=False))

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    # 统一色标范围
    vmin = min(f.min() for _, f in fields)
    vmax = max(f.max() for _, f in fields)

    ims: list[plt.Axes] = []
    for ax, (name, f) in zip(axes, fields, strict=False):
        im = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(f"{title_prefix}{name}")
        ax.set_xticks([])
        ax.set_yticks([])

    # 共享一个水平 colorbar
    fig.colorbar(
        ims[0],
        ax=axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.10,
    )

    fig.tight_layout()
    return fig


def plot_error_map(
    x_true_hw: np.ndarray,
    x_hat_hw: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """
    绘制误差热力图 |x_hat - x_true|。

    为了与其他场图风格统一，这里也使用 RdBu_r 色图。
    """
    t = _to_2d(x_true_hw)
    h = _to_2d(x_hat_hw)
    err = np.abs(h - t)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    im = ax.imshow(err, origin="lower", cmap="RdBu_r")
    ax.set_title(title or "Absolute error")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig

def plot_recon_quadruple(
    x_input_hw,
    x_output_hw,
    x_target_hw,
    x_error_hw=None,
    *,
    mask_hw: np.ndarray | None = None,
    title: str | None = None,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """绘制 input / output / target / error 的四联图，并在 input 上标采样点。"""

    x_in = _to_2d(x_input_hw)
    x_out = _to_2d(x_output_hw)
    x_tg = _to_2d(x_target_hw)

    if x_error_hw is None:
        x_err = x_out - x_tg
    else:
        x_err = _to_2d(x_error_hw)

    fields = [
        ("Input (observed)", x_in),
        ("Output (reconstructed)", x_out),
        ("Target (reference)", x_tg),
        ("Error (output - target)", x_err),
    ]

    # 统一主场色标
    vals = np.concatenate([f.ravel() for _, f in fields[:3]], axis=0)
    vmin_main = float(vals.min())
    vmax_main = float(vals.max())

    # 误差用对称色标
    err_abs = float(np.max(np.abs(x_err))) or 1.0
    vmin_err, vmax_err = -err_abs, err_abs

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    ims = []

    for ax, (name, field) in zip(axes, fields, strict=False):
        if name.startswith("Error"):
            im = ax.imshow(
                field,
                origin="lower",
                cmap=cmap,
                vmin=vmin_err,
                vmax=vmax_err,
            )
        else:
            im = ax.imshow(
                field,
                origin="lower",
                cmap=cmap,
                vmin=vmin_main,
                vmax=vmax_main,
            )
        ims.append(im)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # === 在 input 子图上标出采样点 ===
    if mask_hw is not None:
        mask_hw = np.asarray(mask_hw, dtype=bool)
        yy, xx = np.where(mask_hw)      # 注意：yy 是行，xx 是列
        axes[0].scatter(
            xx,
            yy,
            s=6,
            facecolors="none",
            edgecolors="k",
            linewidths=0.4,
            zorder=2,
        )

    # 统一 colorbar（主场一个，误差一个）
    cbar_main = fig.colorbar(
        ims[0],
        ax=axes[:3],
        orientation="horizontal",
        fraction=0.046,
        pad=0.10,
    )
    cbar_main.ax.set_xlabel("Field value", fontsize=8)

    cbar_err = fig.colorbar(
        ims[3],
        ax=axes[3],
        orientation="horizontal",
        fraction=0.046,
        pad=0.10,
    )
    cbar_err.ax.set_xlabel("Error", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        fig.tight_layout()

    return fig

def plot_example_from_npz(
    npz_path: str | Path,
    *,
    model_name: str | None = None,
    title_prefix: str | None = None,
) -> plt.Figure:
    """
    从保存的 example npz 文件中恢复一张四联图。

    兼容 v1.08 的 npz 格式：
        - x_true:   [H,W,C]
        - x_hat:    [H,W,C]
        - x_interp: [H,W,C]
      可选:
        - mask_hw:      [H,W] bool
        - mask_rate:    float
        - noise_sigma:  float
        - frame_idx:    int
        - model_type:   str
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    x_true = np.asarray(data["x_true"])
    x_hat = np.asarray(data["x_hat"])
    x_interp = np.asarray(data["x_interp"])
    mask_hw = np.asarray(data["mask_hw"]) if "mask_hw" in data.files else None
    mask_rate = float(data["mask_rate"]) if "mask_rate" in data.files else None
    noise_sigma = float(data["noise_sigma"]) if "noise_sigma" in data.files else None
    frame_idx = int(data["frame_idx"]) if "frame_idx" in data.files else None

    if model_name is None and "model_type" in data.files:
        model_name = str(data["model_type"])
    elif model_name is None:
        model_name = "model"

    # 自动生成标题
    if title_prefix is None:
        parts = [model_name]
        if frame_idx is not None:
            parts.append(f"frame={frame_idx}")
        if mask_rate is not None:
            parts.append(f"p={mask_rate:.3g}")
        if noise_sigma is not None:
            parts.append(f"σ={noise_sigma:.3g}")
        title_prefix = " | ".join(parts)

    # 这里调用的是你现在的四联图函数：注意参数名对应！
    fig = plot_recon_quadruple(
        x_input_hw=x_interp,
        x_output_hw=x_hat,
        x_target_hw=x_true,
        mask_hw=mask_hw,
        title=title_prefix,
    )
    return fig