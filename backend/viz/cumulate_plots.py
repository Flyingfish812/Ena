# backend/viz/cumulate_plots.py
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, List

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


def _fit_poly(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int = 2,
    num: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """普通坐标多项式拟合：y = poly(x)"""
    degree = int(max(1, degree))
    if len(x) <= degree:
        raise ValueError(f"not enough points for polyfit: n={len(x)} degree={degree}")

    coef = np.polyfit(x, y, degree)
    poly = np.poly1d(coef)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("invalid x range for fit")

    x_fit = np.linspace(x_min, x_max, int(num))
    y_fit = poly(x_fit)

    meta = {"kind": "poly", "degree": degree, "coef": [float(c) for c in coef.tolist()]}
    return x_fit, y_fit, meta


def _fit_log(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num: int = 200,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """对数拟合：y = a*log(x) + b（要求 x>0）"""
    m = x > eps
    if np.sum(m) < 2:
        raise ValueError("log fit requires x>0 with at least 2 points")

    xx = x[m]
    yy = y[m]
    lx = np.log(xx)

    a, b = np.polyfit(lx, yy, 1)
    x_fit = np.linspace(float(np.min(xx)), float(np.max(xx)), int(num))
    y_fit = a * np.log(x_fit) + b

    meta = {"kind": "log", "a": float(a), "b": float(b)}
    return x_fit, y_fit, meta


def _fit_exp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num: int = 200,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """指数拟合：y = a*exp(b*x)（要求 y>0）"""
    m = y > eps
    if np.sum(m) < 2:
        raise ValueError("exp fit requires y>0 with at least 2 points")

    xx = x[m]
    yy = y[m]
    ly = np.log(yy)

    b, loga = np.polyfit(xx, ly, 1)  # ly = b*x + log(a)
    a = np.exp(loga)

    x_fit = np.linspace(float(np.min(xx)), float(np.max(xx)), int(num))
    y_fit = a * np.exp(b * x_fit)

    meta = {"kind": "exp", "a": float(a), "b": float(b)}
    return x_fit, y_fit, meta


def _fit_powerlaw(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num: int = 200,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """幂律拟合：y = a*x^b（要求 x>0, y>0）"""
    m = (x > eps) & (y > eps)
    if np.sum(m) < 2:
        raise ValueError("power fit requires x>0 and y>0 with at least 2 points")

    xx = x[m]
    yy = y[m]
    lx = np.log(xx)
    ly = np.log(yy)

    b, loga = np.polyfit(lx, ly, 1)  # ly = b*lx + log(a)
    a = np.exp(loga)

    x_fit = np.linspace(float(np.min(xx)), float(np.max(xx)), int(num))
    y_fit = a * (x_fit ** b)

    meta = {"kind": "power", "a": float(a), "b": float(b)}
    return x_fit, y_fit, meta


def _annotate_r(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    every: int = 20,
    fontsize: int = 9,
    prefix: str = "r=",
) -> None:
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
        ri = i + 1
        ax.plot([xi], [yi], marker="o", markersize=5, linestyle="None")
        ax.text(xi, yi, f"{prefix}{ri}", fontsize=fontsize, ha="left", va="bottom", alpha=0.9)


def _format_fit_label(
    base_label: str,
    meta: Dict[str, Any],
    *,
    digits: int = 3,
) -> str:
    """
    生成 legend 用的“拟合形式 + 参数”标签（短式、可读、可比较）。
    例如：
      exp: y=a*exp(bx), a=0.018, b=-4.271
      log: y=a*log(x)+b, a=-0.006, b=0.021
      power: y=a*x^b, a=0.002, b=1.530
      poly2: y=c2 x^2 + c1 x + c0, c2=..., c1=..., c0=...
    """
    d = int(max(0, digits))

    kind = str(meta.get("kind", ""))
    if kind == "exp":
        a = float(meta.get("a", float("nan")))
        b = float(meta.get("b", float("nan")))
        return f"{base_label} | exp: y=a*exp(bx), a={a:.{d}f}, b={b:.{d}f}"

    if kind == "log":
        a = float(meta.get("a", float("nan")))
        b = float(meta.get("b", float("nan")))
        return f"{base_label} | log: y=a*log(x)+b, a={a:.{d}f}, b={b:.{d}f}"

    if kind == "power":
        a = float(meta.get("a", float("nan")))
        b = float(meta.get("b", float("nan")))
        return f"{base_label} | pow: y=a*x^b, a={a:.{d}f}, b={b:.{d}f}"

    if kind == "poly":
        deg = int(meta.get("degree", 1))
        coef = meta.get("coef", [])
        # coef 是 np.polyfit 的降幂顺序：c_deg ... c0
        # legend 里不拼长公式，给一个简短系数表足够比较
        if isinstance(coef, (list, tuple)) and len(coef) >= 1:
            parts = []
            for i, c in enumerate(coef):
                p = deg - i
                if p >= 0:
                    parts.append(f"c{p}={float(c):.{d}f}")
            return f"{base_label} | poly{deg}: " + ", ".join(parts)
        return f"{base_label} | poly{deg}"

    # fallback
    return f"{base_label} | fit:{kind}"


def plot_nrmse_leff_curves(
    curves: Sequence[Dict[str, Any]],
    *,
    title: str = "NRMSE vs ℓ_eff(r)",
    xlabel: str = "ℓ_eff(r)",
    ylabel: str = "NRMSE(r)",
    which_leff: str = "agg",
    sort_by_leff: bool = False,
    # plotting
    plot_mode: str = "scatter_fit",     # "scatter_fit" | "scatter_only" | "line"
    fit_kind: str = "poly",             # "poly"|"log"|"exp"|"power"
    fit_degree: int = 2,                # only for poly
    fit_points: int = 200,
    fit_eps: float = 1e-12,
    # annotations
    annotate_every: int = 20,
    annotate_mode: str = "r",           # "r" | "none"
    annotate_fontsize: int = 9,
    # legend & layout
    legend_mode: str = "fit",           # "fit"|"short"|"none"
    label_digits: int = 3,
    legend_outside: bool = True,
    # NEW: axis scaling & zoom
    yscale: str = "linear",             # "linear"|"log"
    ymin: Optional[float] = None,       # optional manual lower bound
    make_zoom: bool = False,            # output an extra zoom figure
    zoom_ymax: float = 0.05,            # ymax for zoom linear plot
    return_fit_summaries: bool = False,
    return_extra_figs: bool = False,
):
    """
    Return:
      - fig or (fig, fit_summaries) or (fig, fit_summaries, extra)
      where extra is dict like {"zoom": fig_zoom}
    """
    if curves is None or len(curves) == 0:
        if return_fit_summaries and return_extra_figs:
            return None, [], {}
        if return_fit_summaries:
            return None, []
        return None

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    any_plotted = False
    fit_summaries: List[Dict[str, Any]] = []
    extra_figs: Dict[str, Any] = {}

    # collect global y range for scaling decisions
    y_all = []

    for c in curves:
        base_label = str(c.get("label", ""))

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

        if sort_by_leff:
            order = np.argsort(x)
            x = x[order]
            y = y[order]

        y_all.append(y)

        # scatter always low alpha (do not enter legend)
        if plot_mode != "line":
            ax.scatter(x, y, s=18, alpha=0.25, label="_nolegend_")

        # fit
        fit_meta: Optional[Dict[str, Any]] = None
        if plot_mode == "scatter_fit":
            try:
                if fit_kind == "poly":
                    x_fit, y_fit, fit_meta = _fit_poly(x, y, degree=fit_degree, num=fit_points)
                elif fit_kind == "log":
                    x_fit, y_fit, fit_meta = _fit_log(x, y, num=fit_points, eps=fit_eps)
                elif fit_kind == "exp":
                    x_fit, y_fit, fit_meta = _fit_exp(x, y, num=fit_points, eps=fit_eps)
                elif fit_kind == "power":
                    x_fit, y_fit, fit_meta = _fit_powerlaw(x, y, num=fit_points, eps=fit_eps)
                else:
                    raise NotImplementedError(f"fit_kind='{fit_kind}' not implemented")

                if legend_mode == "none":
                    fit_label = "_nolegend_"
                elif legend_mode == "short":
                    fit_label = base_label
                else:
                    fit_label = _format_fit_label(base_label, fit_meta, digits=int(label_digits))

                ax.plot(x_fit, y_fit, linestyle="--", linewidth=2.0, alpha=0.9, label=fit_label)

            except Exception as e:
                ax.text(
                    0.02, 0.02,
                    f"[fit skipped: {type(e).__name__}]",
                    transform=ax.transAxes,
                    fontsize=9,
                    alpha=0.6,
                )
                fit_meta = None

        elif plot_mode == "line":
            if legend_mode == "none":
                ax.plot(x, y, linewidth=2.0, alpha=0.9, label="_nolegend_")
            else:
                ax.plot(x, y, linewidth=2.0, alpha=0.9, label=base_label)

        # annotate
        if annotate_mode == "r":
            _annotate_r(ax, x, y, every=annotate_every, fontsize=annotate_fontsize)

        # collect fit summary
        if fit_meta is not None:
            sm = {"label": base_label, "fit_kind": str(fit_meta.get("kind", fit_kind)), "fit_meta": fit_meta}
            if isinstance(c.get("meta", None), dict):
                sm.update(c["meta"])
            fit_summaries.append(sm)

        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        if return_fit_summaries and return_extra_figs:
            return None, [], {}
        if return_fit_summaries:
            return None, []
        return None

    # ---- axis scaling ----
    yscale = str(yscale).lower().strip()
    if yscale not in ("linear", "log"):
        yscale = "linear"

    if len(y_all) > 0:
        y_stack = np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in y_all], axis=0)
        y_stack = y_stack[np.isfinite(y_stack)]
    else:
        y_stack = np.asarray([], dtype=float)

    if yscale == "log":
        # log requires positive y; use eps floor
        floor = float(fit_eps if fit_eps is not None else 1e-12)
        if ymin is not None:
            floor = max(floor, float(ymin))
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
    else:
        if ymin is not None:
            ax.set_ylim(bottom=float(ymin))

    ax.set_title(title + (f" (y-{yscale})" if yscale == "log" else ""))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, which="both" if yscale == "log" else "major")

    # ---- legend placement (figure legend, no clipping) ----
    if legend_mode != "none":
        handles, labels = ax.get_legend_handles_labels()
        n_legend = len(labels)

        if legend_outside and n_legend > 0:
            if n_legend <= 2:
                right = 0.86
            elif n_legend <= 5:
                right = 0.80
            else:
                right = 0.74

            fig.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)

            fig.legend(
                handles,
                labels,
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

    # ---- optional zoom figure (linear y, small range) ----
    if make_zoom:
        figz, axz = plt.subplots(figsize=(10.5, 6.0))

        # re-plot only scatter and fit lines (no annotation by default)
        for c in curves:
            base_label = str(c.get("label", ""))
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
            if sort_by_leff:
                order = np.argsort(x)
                x = x[order]
                y = y[order]

            axz.scatter(x, y, s=18, alpha=0.25, label="_nolegend_")

            if plot_mode == "scatter_fit":
                try:
                    if fit_kind == "poly":
                        x_fit, y_fit, meta = _fit_poly(x, y, degree=fit_degree, num=fit_points)
                    elif fit_kind == "log":
                        x_fit, y_fit, meta = _fit_log(x, y, num=fit_points, eps=fit_eps)
                    elif fit_kind == "exp":
                        x_fit, y_fit, meta = _fit_exp(x, y, num=fit_points, eps=fit_eps)
                    elif fit_kind == "power":
                        x_fit, y_fit, meta = _fit_powerlaw(x, y, num=fit_points, eps=fit_eps)
                    else:
                        meta = None

                    if meta is not None:
                        if legend_mode == "none":
                            fit_label = "_nolegend_"
                        elif legend_mode == "short":
                            fit_label = base_label
                        else:
                            fit_label = _format_fit_label(base_label, meta, digits=int(label_digits))
                        axz.plot(x_fit, y_fit, linestyle="--", linewidth=2.0, alpha=0.9, label=fit_label)
                except Exception:
                    pass

        axz.set_title(title + f" (zoom y≤{zoom_ymax:g})")
        axz.set_xlabel(xlabel)
        axz.set_ylabel(ylabel)
        axz.set_ylim(0.0, float(zoom_ymax))
        axz.grid(True, alpha=0.25)

        if legend_mode != "none":
            handles, labels = axz.get_legend_handles_labels()
            n_legend = len(labels)
            if legend_outside and n_legend > 0:
                if n_legend <= 2:
                    right = 0.86
                elif n_legend <= 5:
                    right = 0.80
                else:
                    right = 0.74
                figz.subplots_adjust(left=0.10, right=right, top=0.92, bottom=0.12)
                figz.legend(
                    handles,
                    labels,
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

    if return_fit_summaries and return_extra_figs:
        return fig, fit_summaries, extra_figs
    if return_fit_summaries:
        return fig, fit_summaries
    if return_extra_figs:
        return fig, extra_figs
    return fig
