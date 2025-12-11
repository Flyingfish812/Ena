# backend/viz/pod_plots.py

"""
POD 能量谱相关可视化 + 模态自身可视化。
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_energy_spectrum(
    singular_values: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    绘制 POD 奇异值谱（半对数）。

    - y 轴用 log-scale
    - 截断到 1e-4，避免尾部噪声撑爆纵轴
    """
    S = np.asarray(singular_values)
    k = np.arange(1, len(S) + 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.semilogy(
        k,
        S,
        marker="o",
        markersize=2,
        linewidth=0.5,
    )
    ax.set_xlabel("Mode index k")
    ax.set_ylabel("Singular value σ_k")
    ax.set_title("POD Singular Value Spectrum")
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.set_ylim(1e-4, float(S.max()) * 1.1)

    return ax


def plot_cumulative_energy(
    cum_energy: np.ndarray,
    ax: plt.Axes | None = None,
    K_zoom: int = 200,
) -> plt.Axes:
    """
    绘制剩余能量 (1 - cum(k)) 的对数图（只看前 K_zoom 个模态）。

    这比直接画 cum(k) 更能展示能量衰减速度。
    """
    cum = np.asarray(cum_energy)
    k = np.arange(1, len(cum) + 1)
    K_zoom = min(K_zoom, len(cum))

    residual = 1.0 - cum

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.semilogy(
        k[:K_zoom],
        residual[:K_zoom],
        marker="o",
        markersize=2,
        linewidth=0.5,
    )

    ax.set_xlabel("Mode index k (zoomed)")
    ax.set_ylabel("Remaining energy 1 - cum(k)")
    ax.set_title(f"Log Remaining Energy (first {K_zoom} modes)")

    for thr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        ax.axhline(thr, ls="--", lw=0.6, label=f"{thr:.0e}")

    ymin = residual[:K_zoom].min()
    ax.set_ylim(max(ymin * 0.8, 1e-12), 1.0)
    ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax.legend(fontsize=7, loc="upper right")

    return ax


def plot_pod_modes_grid(
    Ur: np.ndarray,
    H: int,
    W: int,
    C: int,
    *,
    max_modes: int = 48,
    modes_per_fig: int = 16,
    channel: int = 0,
    cmap: str = "RdBu_r",
) -> list[plt.Figure]:
    """
    将前 r 个模态中的前 max_modes 个，按每 fig modes_per_fig 个排成网格。

    用途：
    - 展示“低频 / 中频 / 高频”模态的大致形态
    - 论文里可以直接贴图唬人

    参数
    ----
    Ur:
        POD 基底矩阵 [D,r]，与构建时保存的 Ur.npy 一致。
    H, W, C:
        空间尺寸与通道数。
    max_modes:
        最多可视化的模态数（从 k=1 开始）。
    modes_per_fig:
        每个 Figure 包含的模态数。
    channel:
        若 C>1，仅取该通道进行可视化。
    cmap:
        matplotlib colormap 名称。

    返回
    ----
    figs:
        一系列 matplotlib Figure 对象。
    """
    Ur = np.asarray(Ur)
    D, r = Ur.shape

    if D != H * W * C:
        raise ValueError(f"Ur shape {Ur.shape} not compatible with HWC=({H},{W},{C})")

    n_modes = min(max_modes, r)
    n_figs = math.ceil(n_modes / modes_per_fig)
    figs: list[plt.Figure] = []

    mode_indices = np.arange(n_modes)  # 0-based

    for fi in range(n_figs):
        start = fi * modes_per_fig
        end = min((fi + 1) * modes_per_fig, n_modes)
        idx_slice = mode_indices[start:end]
        count = len(idx_slice)

        # 布局：尽量接近正方形
        n_cols = math.ceil(math.sqrt(modes_per_fig))
        n_rows = math.ceil(count / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows))
        axes = np.array(axes).reshape(-1)  # 扁平化处理

        # 先取这一组模态的最大振幅，用于统一色标
        fields = []
        for k in idx_slice:
            vec = Ur[:, k]  # [D]
            field = vec.reshape(H, W, C)[..., channel]
            fields.append(field)
        max_abs = max(float(np.max(np.abs(f))) for f in fields) or 1.0

        for ax, k, field in zip(axes, idx_slice, fields):
            im = ax.imshow(
                field,
                origin="lower",
                cmap=cmap,
                vmin=-max_abs,
                vmax=max_abs,
            )
            ax.set_title(f"Mode {k+1}")
            ax.set_xticks([])
            ax.set_yticks([])

        # 多余的格子关掉
        for ax in axes[count:]:
            ax.axis("off")

        fig.suptitle(f"POD modes {idx_slice[0]+1}–{idx_slice[-1]+1}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        figs.append(fig)

    return figs

def plot_pod_mode_groups(
    Ur: np.ndarray,
    H: int,
    W: int,
    C: int,
    *,
    max_modes: int = 64,
    group_size: int = 16,
    channel: int = 0,
    cmap: str = "RdBu_r",
    robust_percentile: float = 99.5,  # 新增：稳健缩放的分位数（0~100）
) -> plt.Figure:
    """
    将前 max_modes 个 POD 模态按 group_size 分组叠加可视化。

    例如：
    - max_modes=64, group_size=16，则画出 4 个子图：
        1: q1 + ... + q16
        2: q17 + ... + q32
        3: q33 + ... + q48
        4: q49 + ... + q64

    这里的 qk 是第 k 个 POD 空间模态（Ur 的第 k 列，reshape 成 H×W×C 后取指定 channel）。

    参数
    ----
    robust_percentile:
        用于确定全局色标上限的绝对值分位数（例如 99.5 表示 |值| 的 99.5% 分位数），
        用来抑制少量极端 outlier 对色标的“拉爆”效应。
        若不想用稳健缩放，可以传 100.0 退化为全局 max_abs。

    返回
    ----
    fig:
        单个 Figure，其中每个子图是一组模态的叠加场。
    """
    Ur = np.asarray(Ur)
    D, r = Ur.shape

    if D != H * W * C:
        raise ValueError(f"Ur shape {Ur.shape} not compatible with HWC=({H},{W},{C})")

    K = min(max_modes, r)
    if K <= 0:
        raise ValueError(f"max_modes={max_modes} 与 r={r} 导致 K={K}，无可视化模态")

    # 组数
    n_groups = math.ceil(K / group_size)

    # 每组一个叠加场
    group_fields: list[np.ndarray] = []
    group_ranges: list[tuple[int, int]] = []

    for gi in range(n_groups):
        start = gi * group_size      # 0-based index
        end = min((gi + 1) * group_size, K)
        group_ranges.append((start, end))  # [start, end)

        # 对该组的列求和：sum_{k=start}^{end-1} q_k
        vec_group = Ur[:, start:end].sum(axis=1)  # [D]
        field = vec_group.reshape(H, W, C)[..., channel]
        group_fields.append(field)

    # ========= 统一色标（改成稳健缩放） =========
    # 把所有 panel 拼在一起，按绝对值分位数定 vmax
    all_vals = np.stack(group_fields, axis=0)  # [n_groups, H, W]
    abs_vals = np.abs(all_vals)

    # 允许通过 robust_percentile 调节：比如 99, 99.5, 99.9 ...
    p = float(robust_percentile)
    if not (0 < p <= 100):
        raise ValueError("robust_percentile 必须在 (0, 100] 内")
    max_abs = float(np.percentile(abs_vals, p))
    if max_abs == 0.0:
        max_abs = 1.0  # 全 0 的退路，防止除以 0

    vmin, vmax = -max_abs, max_abs

    # ========= 布局：每行最多 4 个 panel =========
    max_cols = 4
    n_cols = min(n_groups, max_cols)
    n_rows = (n_groups + max_cols - 1) // max_cols  # 画图的行数（不含 colorbar）

    # 每行不用太高，长条数据更适合扁一点
    row_height = 1.8          # 每行子图大概 1.8 inch 高
    extra_bottom = 0.8        # 给 colorbar 和 x 轴留一点空间
    fig = plt.figure(figsize=(4 * n_cols, row_height * n_rows + extra_bottom))

    # 上面 n_rows 行放子图，最下面一行放一个细 colorbar
    cbar_height = 0.08        # colorbar 相当于 0.08 行的高度，很薄
    gs = gridspec.GridSpec(
        n_rows + 1,
        n_cols,
        height_ratios=[1] * n_rows + [cbar_height],
        hspace=0.4,           # 行间距，适当留一点但不要太大
    )

    # ---- 子图 axes ----
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

    axes = np.array(axes)
    axes_used = axes[:n_groups]

    last_im = None
    for ax, field, (start, end) in zip(axes_used, group_fields, group_ranges, strict=False):
        im = ax.imshow(
            field,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        last_im = im
        k1, k2 = start + 1, end
        ax.set_title(f"Modes {k1}–{k2} sum")
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余子图
    for ax in axes[n_groups:]:
        ax.set_visible(False)

    # ---- colorbar：细 + 不占满整行宽度 ----
    if last_im is not None:
        # n_cols >= 3 时，用中间几列，让 colorbar 稍微短一点
        if n_cols >= 3:
            cax = fig.add_subplot(gs[-1, 1:-1])   # 去掉最左最右一列
        else:
            cax = fig.add_subplot(gs[-1, :])      # 列数少时就全宽

        cbar = fig.colorbar(last_im, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)

    # ---- 标题与整体排版 ----
    fig.suptitle(
        f"POD mode groups (sum of {group_size} modes per panel, up to {K})",
        fontsize=20,
        y=0.98,   # 明确指定标题位置，避免被后面调整挤乱
    )

    # 用 subplots_adjust 控整体边距，比 tight_layout 更可控
    fig.subplots_adjust(
        top=0.90,
        bottom=0.18,   # 给 colorbar 和标签留出空间
        left=0.05,
        right=0.98,
    )

    return fig