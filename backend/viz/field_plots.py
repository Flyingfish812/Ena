# backend/viz/field_plots.py

import numpy as np
import matplotlib.pyplot as plt


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
