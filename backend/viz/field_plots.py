# backend/viz/field_plots.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    err_abs = float(np.max(np.abs(err))) or 1.0
    vmin_err, vmax_err = -err_abs, err_abs

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    im = ax.imshow(err, origin="lower", cmap="RdBu_r", vmin=vmin_err, vmax=vmax_err)
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
    """
    绘制 input / output / target / error 的四联图，并在 input 上标采样点。

    - 对 input/output/target 做「减 target 空间均值」的中心化，只用于可视化，
      让背景接近 0（白色），突出扰动结构；
    - 主场色标按 target 扰动的分布设定，对称于 0；
    - 误差色标不再按“自己最大值”算，而是与主场共用同一尺度（同一帧内），
      这样同一帧上不同模型的误差可直接拿来比较。
    """

    # ---------- 数据预处理 ----------
    x_in = _to_2d(x_input_hw)
    x_out = _to_2d(x_output_hw)
    x_tg = _to_2d(x_target_hw)

    if x_error_hw is None:
        x_err = x_out - x_tg
    else:
        x_err = _to_2d(x_error_hw)

    # 只用于可视化的中心化：围绕 target 均值的扰动
    mu_tg = float(np.mean(x_tg))
    x_in_c = x_in - mu_tg
    x_out_c = x_out - mu_tg
    x_tg_c = x_tg - mu_tg

    fields = [
        ("Input (observed)", x_in_c),
        ("Output (reconstructed)", x_out_c),
        ("Target (reference)", x_tg_c),
        ("Error (output - target)", x_err),
    ]

    # ---------- 主场色标：target 扰动 + 对称于 0 ----------
    vals_tg = x_tg_c.ravel()
    # 稍微用个稳健分位数，防止极端点把色标拉爆
    max_abs_main = float(np.percentile(np.abs(vals_tg), 99.5))
    if max_abs_main == 0.0:
        max_abs_main = 1.0
    vmin_main, vmax_main = -max_abs_main, max_abs_main

    # ---------- 误差色标：跟主场用同一尺度（单帧统一标尺） ----------
    # 你也可以改成 factor * max_abs_main，比如 1.5 * max_abs_main
    err_scale = 1.0
    max_abs_err = max_abs_main * err_scale
    vmin_err, vmax_err = -max_abs_err, max_abs_err

    # ---------- 上面一行：四个等高子图 ----------
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.4))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.78, bottom=0.32, wspace=0.25)

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

    # ---------- 在 input 子图上标采样点 ----------
    if mask_hw is not None:
        mask_hw = np.asarray(mask_hw, dtype=bool)
        yy, xx = np.where(mask_hw)
        axes[0].scatter(
            xx,
            yy,
            s=6,
            facecolors="none",
            edgecolors="k",
            linewidths=0.4,
            zorder=2,
        )

    # ---------- 底部两条短而细的 colorbar ----------
    # 主场 colorbar：挂在第 2 个子图下面
    pos_main = axes[1].get_position()
    cbar_width = pos_main.width * 0.75
    cbar_height = pos_main.height * 0.25
    cbar_left = pos_main.x0 + (pos_main.width - cbar_width) / 2
    cbar_bottom = pos_main.y0 - 0.75 * pos_main.height

    cax_main = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar_main = fig.colorbar(ims[1], cax=cax_main, orientation="horizontal")

    # 误差 colorbar：挂在第 4 个子图下面，使用「统一误差尺度」
    pos_err = axes[3].get_position()
    cbar_width_err = pos_err.width * 0.75
    cbar_height_err = pos_err.height * 0.25
    cbar_left_err = pos_err.x0 + (pos_err.width - cbar_width_err) / 2
    cbar_bottom_err = pos_err.y0 - 0.75 * pos_err.height

    cax_err = fig.add_axes([cbar_left_err, cbar_bottom_err, cbar_width_err, cbar_height_err])
    cbar_err = fig.colorbar(ims[3], cax=cax_err, orientation="horizontal")

    # ---------- 标题 ----------
    if title:
        fig.suptitle(title, fontsize=12, y=0.96)

    return fig

def plot_example_from_npz_data(
    data: "np.lib.npyio.NpzFile | dict[str, object]",
    *,
    model_name: str | None = None,
    title_prefix: str | None = None,
) -> plt.Figure:
    """
    从已加载的 example npz 数据对象中恢复一张 POD 四联图（不做任何 IO）。

    预期包含（至少）以下 key：
        - x_true:      (H, W) 或 (C, H, W) 或 (H, W, C)
        - x_hat:       同上
        - x_interp:    同上
    可选 key：
        - mask_hw:     (H, W) 的 0/1 采样掩码
        - mask_rate:   float
        - noise_sigma: float
        - frame_idx:   int
        - model_type:  str
    """
    files = getattr(data, "files", None)
    has = (lambda k: (k in files) if files is not None else (k in data))
    get = (lambda k: data[k])  # NpzFile / dict 都支持 __getitem__

    x_true = np.asarray(get("x_true"))
    x_hat = np.asarray(get("x_hat"))
    x_interp = np.asarray(get("x_interp"))
    mask_hw = np.asarray(get("mask_hw")) if has("mask_hw") else None
    mask_rate = float(get("mask_rate")) if has("mask_rate") else None
    noise_sigma = float(get("noise_sigma")) if has("noise_sigma") else None
    frame_idx = int(get("frame_idx")) if has("frame_idx") else None

    if model_name is None and has("model_type"):
        model_name = str(get("model_type"))
    elif model_name is None:
        model_name = "model"

    if title_prefix is None:
        parts = [model_name]
        if frame_idx is not None:
            parts.append(f"frame={frame_idx}")
        if mask_rate is not None:
            parts.append(f"p={mask_rate:.3g}")
        if noise_sigma is not None:
            parts.append(f"σ={noise_sigma:.3g}")
        title_prefix = " | ".join(parts)

    fig = plot_recon_quadruple(
        x_input_hw=x_interp,
        x_output_hw=x_hat,
        x_target_hw=x_true,
        mask_hw=mask_hw,
        title=title_prefix,
    )
    return fig

def plot_example_from_npz(
    npz_path: str | Path,
    *,
    model_name: str | None = None,
    title_prefix: str | None = None,
) -> plt.Figure:
    """
    从保存的 example npz 文件中恢复一张 POD 四联图。

    这是一个薄封装：负责 IO，实际绘制逻辑在 plot_example_from_npz_data() 内。
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        return plot_example_from_npz_data(
            data,
            model_name=model_name,
            title_prefix=title_prefix,
        )
    
def plot_recon_triptych(
    x_pred_hw: np.ndarray,
    x_true_hw: np.ndarray,
    *,
    title: str = "reconstruction (pred/true/err)",
    mask_hw: np.ndarray | None = None,
    show_mask: bool = False,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    空间域三联图：pred / true / err，共用 colorbar。
    （不画 input，因为你这次明确说 input 没意义）
    """
    x_pred_hw = np.asarray(x_pred_hw)
    x_true_hw = np.asarray(x_true_hw)
    x_err_hw = x_pred_hw - x_true_hw

    # color range：pred & true 用同一套；err 单独范围也行，但你要共用 colorbar => 统一
    vmin = float(np.nanmin([x_pred_hw.min(), x_true_hw.min(), x_err_hw.min()]))
    vmax = float(np.nanmax([x_pred_hw.max(), x_true_hw.max(), x_err_hw.max()]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ims = []
    ims.append(axes[0].imshow(x_pred_hw, vmin=vmin, vmax=vmax, cmap=cmap))
    axes[0].set_title("pred")
    ims.append(axes[1].imshow(x_true_hw, vmin=vmin, vmax=vmax, cmap=cmap))
    axes[1].set_title("true")
    ims.append(axes[2].imshow(x_err_hw, vmin=vmin, vmax=vmax, cmap=cmap))
    axes[2].set_title("err (pred-true)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_mask and mask_hw is not None:
        mhw = np.asarray(mask_hw)
        # 用半透明叠一层：观测点更亮
        for ax in axes:
            ax.imshow(mhw, alpha=0.15)

    # shared colorbar
    cbar = fig.colorbar(ims[-1], ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("value")

    fig.suptitle(title)
    return fig