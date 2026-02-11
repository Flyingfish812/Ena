# backend/viz/cumulate_plots.py
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _as_1d_float(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        a = a.reshape(-1)
    return a


def _finite_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isfinite(y)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_close(fig, path: Path, *, dpi: int = 200) -> None:
    if fig is None:
        return
    _ensure_parent_dir(path)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_nrmse_family_vs_r_curves(
    curves: Sequence[Dict[str, Any]],
    *,
    title: str = "NRMSE family vs r",
    xlabel: str = "r (number of modes)",
    ylabel: str = "NRMSE (relative error)",
    nrmse_kinds: Sequence[str] = ("nrmse_full", "nrmse_prefix", "nrmse_tail"),
    legend_outside: bool = True,
):
    """
    画 “NRMSE_{full/prefix/tail} vs r”（线性坐标，不拟合）：

    - 同一条配置(label)的多种 nrmse：同色，不同线型
        nrmse_full   : 实线，alpha=0.9
        nrmse_prefix : 虚线，alpha=0.5
        nrmse_tail   : 点虚线，alpha=0.5
    - 标点：每隔 20 个 r 打一个点，但不标注 r=...
    - marker 颜色与对应线条颜色一致
    - legend 放到图外（双 legend）
    """
    if curves is None or len(curves) == 0:
        return None

    want = [str(e).strip() for e in list(nrmse_kinds)]
    want = [e for e in want if e in ("nrmse_full", "nrmse_prefix", "nrmse_tail")]
    if not want:
        want = ["nrmse_full", "nrmse_prefix", "nrmse_tail"]

    style = {
        "nrmse_full": "-",
        "nrmse_prefix": "--",
        "nrmse_tail": "-.",
    }
    alpha_map = {
        "nrmse_full": 0.9,
        "nrmse_prefix": 0.5,
        "nrmse_tail": 0.5,
    }

    step = 20  # 每隔 20 个 r 打一个点

    fig, ax = plt.subplots(figsize=(10.8, 6.2))

    config_handles, config_labels = [], []
    type_handles, type_labels = [], []
    any_plotted = False

    for c in curves:
        label = str(c.get("label", ""))
        x = _as_1d_float(c.get("x", None))
        nrmse = c.get("nrmse", {}) or {}

        if x is None or len(x) <= 1:
            continue

        def _get_series(k: str):
            y = nrmse.get(k, None)
            if y is None:
                return None
            y = _as_1d_float(y)
            if y is None:
                return None
            n = min(len(x), len(y))
            if n <= 1:
                return None
            xx = x[:n]
            yy = y[:n]
            m = _finite_mask(xx, yy)
            xx = xx[m]
            yy = yy[m]
            if len(xx) <= 1:
                return None
            return xx, yy

        base_color = None
        first_key = None

        # ---- 先画第一条曲线，用来“占位”颜色 ----
        for k in want:
            got = _get_series(k)
            if got is None:
                continue
            xx, yy = got
            ln, = ax.plot(
                xx, yy,
                linestyle=style[k],
                linewidth=2.2,
                alpha=alpha_map[k],
                label="_nolegend_",
            )
            base_color = ln.get_color()
            first_key = k
            any_plotted = True

            # 每隔 step 个点打 marker（不标文字）
            ax.scatter(
                xx[::step],
                yy[::step],
                s=20,
                color=base_color,
                zorder=3,
            )
            break

        if base_color is None:
            continue

        # 配置 legend（颜色）
        config_handles.append(
            plt.Line2D([0], [0], color=base_color, linestyle="-", linewidth=2.6)
        )
        config_labels.append(label)

        # ---- 画同一配置的其它 nrmse 曲线（同色，不同线型/透明度） ----
        for k in want:
            if k == first_key:
                continue
            got = _get_series(k)
            if got is None:
                continue
            xx, yy = got
            ax.plot(
                xx, yy,
                color=base_color,
                linestyle=style[k],
                linewidth=2.2,
                alpha=alpha_map[k],
                label="_nolegend_",
            )
            any_plotted = True

            # 每隔 step 个点打 marker（不标文字）
            ax.scatter(
                xx[::step],
                yy[::step],
                s=20,
                color=base_color,
                zorder=3,
            )

    if not any_plotted:
        plt.close(fig)
        return None

    # ---- nrmse 类型 legend（线型） ----
    for k in want:
        type_handles.append(
            plt.Line2D(
                [0], [0],
                color="black",
                linestyle=style[k],
                linewidth=2.2,
                alpha=alpha_map[k],
                marker="o",
                markersize=4,
            )
        )
        type_labels.append(k)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.25)

    if legend_outside:
        right = 0.78 if len(config_labels) > 6 else 0.82
        fig.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)

        leg1 = fig.legend(
            config_handles, config_labels,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.92),
            bbox_transform=fig.transFigure,
            frameon=True,
            borderaxespad=0.0,
            title="curves",
        )
        fig.add_artist(leg1)

        fig.legend(
            type_handles, type_labels,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.08),
            bbox_transform=fig.transFigure,
            frameon=True,
            borderaxespad=0.0,
            title="nrmse kinds",
        )
    else:
        handles = config_handles + type_handles
        labels = config_labels + type_labels
        ax.legend(handles, labels, loc="best")
        fig.tight_layout()

    return fig


def render_save_nrmse_family_vs_r(
    curves: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    dpi: int = 200,
    **plot_kwargs: Any,
) -> Optional[str]:
    fig = plot_nrmse_family_vs_r_curves(curves, **plot_kwargs)
    if fig is None:
        return None
    _save_close(fig, Path(out_png), dpi=int(dpi))
    return str(out_png)


def plot_dual_scale_nrmse_xy_vs_r(
    *,
    r_grid: np.ndarray,
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    curves_right: Sequence[Dict[str, Any]],
    title: str,
    xlabel: str = "r (number of modes)",
    ylabel_left_x: str = "ℓ_x(r)",
    ylabel_left_y: str = "ℓ_y(r)",
    ylabel_right: str = "NRMSE(r)",
    legend_outside: bool = True,
):
    """
    单图双纵轴（x/y 尺度同图）：
      - 左纵轴：ℓ_x, ℓ_y scatter（两种固定颜色，与配置无关）
      - 右纵轴：NRMSE 实线（多配置，颜色区分）
    约束：
      - ℓ_* 仅画一组散点（不随配置变化）
      - NRMSE 用彩色实线
      - 常数坐标轴（linear）
      - 共用一个 legend（放在底部）
    """
    fig, ax_l = plt.subplots(1, 1, figsize=(11.8, 6.2))
    ax_r = ax_l.twinx()

    # 固定使用颜色轮盘之外的颜色
    color_ell_x = "#222222"   # 深灰（x 尺度）
    color_ell_y = "#7A3E9D"   # 紫色（y 尺度）

    # ---- 左轴：尺度 scatter ----
    ax_l.scatter(
        r_grid, ell_x,
        s=12, c=color_ell_x, alpha=0.5, label="ell_x"
    )
    ax_l.scatter(
        r_grid, ell_y,
        s=12, c=color_ell_y, alpha=0.5, label="ell_y"
    )
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel("scale ℓ(r)")
    ax_l.set_yscale("linear")
    ax_l.grid(True, alpha=0.25)

    # ---- 右轴：NRMSE 曲线 ----
    for c in curves_right:
        label_lower = c.get("label", "").lower()
        if "mlp" in label_lower:
            linestyle = "--"
        else:
            linestyle = "-"   # linear / 其它默认

        ax_r.plot(
            r_grid,
            c["y"],
            linewidth=2.2,
            linestyle=linestyle,
            color=c["color"],
            label=c["label"],
        )

    ax_r.set_ylabel(ylabel_right)
    ax_r.set_yscale("log")

    fig.suptitle(title)

    # ---- shared legend (bottom) ----
    handles_scale = [
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   color=color_ell_x, label="ell_x"),
        plt.Line2D([0], [0], marker="o", linestyle="None",
                   color=color_ell_y, label="ell_y"),
    ]
    handles_err = [
        plt.Line2D([0], [0], color=c["color"], linewidth=2.2, label=c["label"])
        for c in curves_right
    ]
    handles = handles_scale + handles_err
    labels = [h.get_label() for h in handles]

    if legend_outside:
        fig.subplots_adjust(bottom=0.22, top=0.90, left=0.10, right=0.90)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(labels)),
            frameon=True,
        )
    else:
        ax_l.legend(handles, labels, loc="best")

    return fig


def render_save_dual_xy_vs_r(
    *,
    r_grid: np.ndarray,
    ell_x: np.ndarray,
    ell_y: np.ndarray,
    curves_right: Sequence[Dict[str, Any]],
    out_png: Path,
    dpi: int = 200,
    **plot_kwargs: Any,
) -> Optional[str]:
    fig = plot_dual_scale_nrmse_xy_vs_r(
        r_grid=r_grid,
        ell_x=ell_x,
        ell_y=ell_y,
        curves_right=curves_right,
        **plot_kwargs,
    )
    if fig is None:
        return None
    _save_close(fig, out_png, dpi=dpi)
    return str(out_png)