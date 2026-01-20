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


def _annotate_r(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    every: int = 20,
    fontsize: int = 9,
    prefix: str = "r=",
    digits: int = 0,
) -> None:
    """
    在曲线上每隔 every 个点标注一次 r。
    改动点：不再用 ri=i+1，而是用 x[i]（通常 r_grid 就是 1..R）。
    digits=0 表示 r 取整显示；如果 x 不是整数可提高 digits。
    """
    if every is None or int(every) <= 0:
        return
    n = int(len(x))
    if n <= 0:
        return

    step = int(every)
    idxs = list(range(0, n, step))
    if (n - 1) not in idxs:
        idxs.append(n - 1)

    for i in idxs:
        xi = float(x[i])
        yi = float(y[i])
        ax.plot([xi], [yi], marker="o", markersize=5, linestyle="None")
        if digits <= 0:
            tag = f"{prefix}{int(round(xi))}"
        else:
            tag = f"{prefix}{xi:.{int(digits)}f}"
        ax.text(xi, yi, tag, fontsize=fontsize, ha="left", va="bottom", alpha=0.9)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_close(fig, path: Path, *, dpi: int = 200) -> None:
    if fig is None:
        return
    _ensure_parent_dir(path)
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def plot_nrmse_vs_r_curves(
    curves: Sequence[Dict[str, Any]],
    *,
    title: str = "NRMSE(r) vs r",
    xlabel: str = "r (number of modes)",
    ylabel: str = "NRMSE(r)",
    plot_mode: str = "line",            # "line" | "scatter"
    linewidth: float = 2.2,
    alpha: float = 0.9,
    yscale: str = "log",                # 默认 log 更符合误差曲线
    ymin: Optional[float] = None,
    y_eps: float = 1e-12,
    annotate_every: int = 20,
    annotate_fontsize: int = 9,
    annotate_digits: int = 0,
    legend_mode: str = "short",         # "short"|"none"
    legend_outside: bool = True,
    make_zoom: bool = False,
    zoom_ymax: float = 0.05,
    return_extra_figs: bool = False,
):
    if curves is None or len(curves) == 0:
        return (None, {}) if return_extra_figs else None

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    extra_figs: Dict[str, Any] = {}
    any_plotted = False

    yscale = str(yscale).lower().strip()
    if yscale not in ("linear", "log"):
        yscale = "log"

    for c in curves:
        label = str(c.get("label", ""))
        x = _as_1d_float(c.get("x", None))
        y = _as_1d_float(c.get("y", None))
        if x is None or y is None:
            continue

        n = min(len(x), len(y))
        if n <= 1:
            continue
        x = x[:n]
        y = y[:n]

        m = _finite_mask(x, y)
        x = x[m]
        y = y[m]
        if len(x) <= 1:
            continue

        leg = (label if legend_mode != "none" else "_nolegend_")

        if plot_mode == "scatter":
            ax.scatter(x, y, s=18, alpha=0.55, label=leg)
        else:
            ax.plot(x, y, linewidth=float(linewidth), alpha=float(alpha), label=leg)

        if annotate_every and int(annotate_every) > 0:
            _annotate_r(
                ax, x, y,
                every=int(annotate_every),
                fontsize=int(annotate_fontsize),
                prefix="r=",
                digits=int(annotate_digits),
            )

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return (None, {}) if return_extra_figs else None

    if yscale == "log":
        floor = float(y_eps if y_eps is not None else 1e-12)
        if ymin is not None:
            floor = max(floor, float(ymin))
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
    else:
        if ymin is not None:
            ax.set_ylim(bottom=float(ymin))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, which="both" if yscale == "log" else "major")

    if legend_mode != "none":
        handles, labels = ax.get_legend_handles_labels()
        if legend_outside and len(labels) > 0:
            n_legend = len(labels)
            right = 0.86 if n_legend <= 2 else (0.80 if n_legend <= 5 else 0.74)
            fig.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)
            fig.legend(
                handles, labels,
                loc="center right",
                bbox_to_anchor=(0.985, 0.5),
                bbox_transform=fig.transFigure,
                frameon=True,
                borderaxespad=0.0,
            )
        else:
            ax.legend(loc="best")
            fig.tight_layout()
    else:
        fig.tight_layout()

    # zoom figure (linear y)
    if make_zoom:
        figz, axz = plt.subplots(figsize=(10.5, 6.0))
        for c in curves:
            label = str(c.get("label", ""))
            x = _as_1d_float(c.get("x", None))
            y = _as_1d_float(c.get("y", None))
            if x is None or y is None:
                continue
            n = min(len(x), len(y))
            if n <= 1:
                continue
            x = x[:n]
            y = y[:n]
            m = _finite_mask(x, y)
            x = x[m]
            y = y[m]
            if len(x) <= 1:
                continue

            leg = (label if legend_mode != "none" else "_nolegend_")
            if plot_mode == "scatter":
                axz.scatter(x, y, s=18, alpha=0.55, label=leg)
            else:
                axz.plot(x, y, linewidth=float(linewidth), alpha=float(alpha), label=leg)

        axz.set_title(title + f" (zoom y≤{zoom_ymax:g})")
        axz.set_xlabel(xlabel)
        axz.set_ylabel(ylabel)
        axz.set_ylim(0.0, float(zoom_ymax))
        axz.grid(True, alpha=0.25)

        if legend_mode != "none":
            handles, labels = axz.get_legend_handles_labels()
            if legend_outside and len(labels) > 0:
                n_legend = len(labels)
                right = 0.86 if n_legend <= 2 else (0.80 if n_legend <= 5 else 0.74)
                figz.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)
                figz.legend(
                    handles, labels,
                    loc="center right",
                    bbox_to_anchor=(0.985, 0.5),
                    bbox_transform=figz.transFigure,
                    frameon=True,
                    borderaxespad=0.0,
                )
            else:
                axz.legend(loc="best")
                figz.tight_layout()
        else:
            figz.tight_layout()

        extra_figs["zoom"] = figz

    return (fig, extra_figs) if return_extra_figs else fig


def plot_dual_scale_nrmse_vs_r(
    curves: Sequence[Dict[str, Any]],
    *,
    title: str = "Scale (left) + NRMSE (right) vs r",
    xlabel: str = "r (number of modes)",
    ylabel_left: str = "ℓ(r)  (smaller=finer)",
    ylabel_right: str = "NRMSE(r)",
    left_yscale: str = "log",
    left_ymin: Optional[float] = None,
    left_eps: float = 1e-12,
    invert_left: bool = True,   # 默认更直观：越细越高
    right_yscale: str = "log",
    right_ymin: Optional[float] = None,
    right_eps: float = 1e-12,
    linewidth_left: float = 2.4,
    linewidth_right: float = 2.0,
    alpha_left: float = 0.9,
    alpha_right: float = 0.75,
    annotate_every: int = 20,
    annotate_fontsize: int = 9,
    annotate_digits: int = 0,
    annotate_on: str = "right",   # "left"|"right"|"none"
    legend_outside: bool = True,
):
    if curves is None or len(curves) == 0:
        return None

    fig, axL = plt.subplots(figsize=(11.2, 6.2))
    axR = axL.twinx()

    left_yscale = str(left_yscale).lower().strip()
    if left_yscale not in ("linear", "log"):
        left_yscale = "log"
    right_yscale = str(right_yscale).lower().strip()
    if right_yscale not in ("linear", "log"):
        right_yscale = "log"

    any_plotted = False

    for c in curves:
        label = str(c.get("label", ""))
        x = _as_1d_float(c.get("x", None))
        yl = _as_1d_float(c.get("y_left", None))
        yr = _as_1d_float(c.get("y_right", None))
        if x is None or yl is None or yr is None:
            continue

        n = min(len(x), len(yl), len(yr))
        if n <= 1:
            continue
        x = x[:n]
        yl = yl[:n]
        yr = yr[:n]

        m = np.isfinite(x) & np.isfinite(yl) & np.isfinite(yr)
        x = x[m]
        yl = yl[m]
        yr = yr[m]
        if len(x) <= 1:
            continue

        axL.plot(x, yl, linewidth=float(linewidth_left), alpha=float(alpha_left), label=f"{label} | scale")
        axR.plot(x, yr, linewidth=float(linewidth_right), alpha=float(alpha_right), linestyle="--",
                 label=f"{label} | nrmse")

        if annotate_on == "left":
            _annotate_r(axL, x, yl, every=int(annotate_every), fontsize=int(annotate_fontsize),
                        prefix="r=", digits=int(annotate_digits))
        elif annotate_on == "right":
            _annotate_r(axR, x, yr, every=int(annotate_every), fontsize=int(annotate_fontsize),
                        prefix="r=", digits=int(annotate_digits))

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return None

    if left_yscale == "log":
        floor = float(left_eps if left_eps is not None else 1e-12)
        if left_ymin is not None:
            floor = max(floor, float(left_ymin))
        axL.set_yscale("log")
        axL.set_ylim(bottom=floor)
    else:
        if left_ymin is not None:
            axL.set_ylim(bottom=float(left_ymin))

    if invert_left:
        axL.invert_yaxis()

    if right_yscale == "log":
        floor = float(right_eps if right_eps is not None else 1e-12)
        if right_ymin is not None:
            floor = max(floor, float(right_ymin))
        axR.set_yscale("log")
        axR.set_ylim(bottom=floor)
    else:
        if right_ymin is not None:
            axR.set_ylim(bottom=float(right_ymin))

    axL.set_title(title)
    axL.set_xlabel(xlabel)
    axL.set_ylabel(ylabel_left)
    axR.set_ylabel(ylabel_right)

    axL.grid(True, alpha=0.25, which="both" if (left_yscale == "log" or right_yscale == "log") else "major")

    # legend combine
    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    handles = hL + hR
    labels = lL + lR

    if legend_outside and len(labels) > 0:
        n_legend = len(labels)
        right = 0.86 if n_legend <= 4 else (0.80 if n_legend <= 8 else 0.74)
        fig.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)
        fig.legend(
            handles, labels,
            loc="center right",
            bbox_to_anchor=(0.985, 0.5),
            bbox_transform=fig.transFigure,
            frameon=True,
            borderaxespad=0.0,
        )
    else:
        axL.legend(handles, labels, loc="best")
        fig.tight_layout()

    return fig


def render_save_nrmse_vs_r(
    curves: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    out_png_zoom: Optional[Path] = None,
    dpi: int = 200,
    **plot_kwargs: Any,
) -> Dict[str, Optional[str]]:
    fig, extra = plot_nrmse_vs_r_curves(curves, return_extra_figs=True, **plot_kwargs)
    written = {"main": None, "zoom": None}
    if fig is None:
        return written

    _save_close(fig, Path(out_png), dpi=int(dpi))
    written["main"] = str(out_png)

    if out_png_zoom is not None and extra and extra.get("zoom", None) is not None:
        _save_close(extra["zoom"], Path(out_png_zoom), dpi=int(dpi))
        written["zoom"] = str(out_png_zoom)

    return written


def render_save_dual_vs_r(
    curves: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    dpi: int = 200,
    **plot_kwargs: Any,
) -> Optional[str]:
    fig = plot_dual_scale_nrmse_vs_r(curves, **plot_kwargs)
    if fig is None:
        return None
    _save_close(fig, Path(out_png), dpi=int(dpi))
    return str(out_png)
