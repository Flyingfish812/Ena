# backend/viz/curves.py

"""
NMSE/NMAE/PSNR 随 mask_rate / noise_sigma 变化的曲线。
"""

from typing import Dict, Any, Sequence
import numpy as np

import matplotlib.pyplot as plt
try:
    import pandas as pd
except Exception:
    pd = None


def _filter_entries(
    results: Dict[str, Any],
    *,
    mask_rate: float | None = None,
    noise_sigma: float | None = None,
):
    """
    从 results["entries"] 里按条件筛选记录。
    """
    entries = results.get("entries", [])
    out = []
    for e in entries:
        if mask_rate is not None and abs(e["mask_rate"] - mask_rate) > 1e-12:
            continue
        if noise_sigma is not None and abs(e["noise_sigma"] - noise_sigma) > 1e-12:
            continue
        out.append(e)
    return out


def plot_nmse_vs_mask_rate(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 mask_rate 下的 NMSE 曲线。

    约定：
    - 若 results 中包含多种 noise_sigma，则默认选取其中最小的 noise_sigma；
    - x 轴: mask_rate，y 轴: nmse_mean。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    mask_rates = sorted(set(float(m) for m in results.get("mask_rates", [])))
    noise_sigmas = sorted(set(float(s) for s in results.get("noise_sigmas", [])))

    if not mask_rates or not noise_sigmas:
        raise ValueError("results 中缺少 mask_rates 或 noise_sigmas 信息")

    sigma_ref = noise_sigmas[0]  # 默认用噪声最小的那一组
    entries = _filter_entries(results, noise_sigma=sigma_ref)

    x = []
    y = []
    for mr in mask_rates:
        # 找到该 mask_rate 下的条目
        e_list = [e for e in entries if abs(e["mask_rate"] - mr) < 1e-12]
        if not e_list:
            continue
        e = e_list[0]
        x.append(mr)
        y.append(e["nmse_mean"])

    model_label = label or results.get("model_type", "model")
    ax.plot(x, y, marker="o", label=f"{model_label} (σ={sigma_ref:.3g})")
    ax.set_xlabel("mask_rate")
    ax.set_ylabel("NMSE (mean)")
    ax.set_title("NMSE vs mask_rate")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return ax


def plot_nmse_vs_noise(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 noise_sigma 下的 NMSE 曲线。

    约定：
    - 若 results 中包含多种 mask_rate，则默认选取其中最小的 mask_rate；
    - x 轴: noise_sigma，y 轴: nmse_mean。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    mask_rates = sorted(set(float(m) for m in results.get("mask_rates", [])))
    noise_sigmas = sorted(set(float(s) for s in results.get("noise_sigmas", [])))

    if not mask_rates or not noise_sigmas:
        raise ValueError("results 中缺少 mask_rates 或 noise_sigmas 信息")

    mr_ref = mask_rates[0]  # 默认用采样率最小的那一组
    entries = _filter_entries(results, mask_rate=mr_ref)

    x = []
    y = []
    for sigma in noise_sigmas:
        e_list = [e for e in entries if abs(e["noise_sigma"] - sigma) < 1e-12]
        if not e_list:
            continue
        e = e_list[0]
        x.append(sigma)
        y.append(e["nmse_mean"])

    model_label = label or results.get("model_type", "model")
    ax.plot(x, y, marker="o", label=f"{model_label} (p={mr_ref:.3g})")
    ax.set_xlabel("noise_sigma")
    ax.set_ylabel("NMSE (mean)")
    ax.set_title("NMSE vs noise_sigma")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return ax


def _infer_eval_columns(
    df,
    *,
    mask_col: str | None = None,
    noise_col: str | None = None,
    metric_col: str | None = None,
) -> tuple[str, str, str]:
    """从 eval DataFrame 中推断列名。

    默认约定：
    - 采样率列名包含 "mask" 且包含 "rate"；
    - 噪声列名包含 "noise" 或 "sigma"；
    - 指标列名优先找 "nmse" "nrmse"，否则回退为 "metric" 或第一列。
    """
    cols: Sequence[str] = list(getattr(df, "columns", []))

    def _guess(predicates) -> str:
        for c in cols:
            name = c.lower()
            if all(p(name) for p in predicates):
                return c
        return ""

    if mask_col is None:
        mask_col = _guess([
            lambda s: "mask" in s,
            lambda s: "rate" in s or "p" in s,
        ]) or "mask_rate"

    if noise_col is None:
        noise_col = _guess([
            lambda s: "noise" in s or "sigma" in s,
        ]) or "noise_sigma"

    if metric_col is None:
        metric_col = _guess([
            lambda s: "nmse" in s or "nrmse" in s,
        ]) or "nmse_mean"

    return mask_col, noise_col, metric_col


def plot_eval_nmse_curves(
    df_linear,
    df_mlp=None,
    *,
    mask_col: str | None = None,
    noise_col: str | None = None,
    metric_col: str | None = None,
) -> dict[str, plt.Figure]:
    """从 eval DataFrame 构造 4 张全局 NMSE 曲线图。

    - 线性基线：NMSE vs mask_rate（固定最小 noise），NMSE vs noise_sigma（固定最小 mask）
    - MLP：同上（若提供 df_mlp）
    - 坐标：x 轴、y 轴均使用 log-scale，便于展示跨量级对比。
    """
    if df_linear is None:
        raise ValueError("df_linear 不可为 None")

    mask_col, noise_col, metric_col = _infer_eval_columns(
        df_linear,
        mask_col=mask_col,
        noise_col=noise_col,
        metric_col=metric_col,
    )

    def _unique_sorted(values):
        arr = np.asarray(values, dtype=float)
        return np.unique(arr[~np.isnan(arr)])

    figs: dict[str, plt.Figure] = {}

    # ---------- Helper: 单模型多曲线 ----------
    def _plot_pair(df, model_name: str) -> tuple[plt.Figure, plt.Figure]:
        # 1) metric vs mask_rate，全 noise 曲线，x/y 均 log
        noise_values = _unique_sorted(df[noise_col])
        if noise_values.size == 0:
            raise ValueError(f"DataFrame 中缺少 {noise_col} 信息")

        fig1, ax1 = plt.subplots(1, 1, figsize=(4, 3))
        for sigma in noise_values:
            df_slice = df[df[noise_col] == sigma]
            xs = _unique_sorted(df_slice[mask_col])
            ys = []
            for mr in xs:
                val = df_slice.loc[df_slice[mask_col] == mr, metric_col].mean()
                ys.append(val)
            ax1.plot(xs, ys, marker="o", label=f"σ={sigma:g}")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(mask_col)
        ax1.set_ylabel(metric_col)
        ax1.set_title(f"{model_name}: {metric_col} vs {mask_col}")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax1.legend(fontsize=7)
        fig1.tight_layout()

        # 2) metric vs noise_sigma，全 mask 曲线，x/y 均 log
        mask_values = _unique_sorted(df[mask_col])
        if mask_values.size == 0:
            raise ValueError(f"DataFrame 中缺少 {mask_col} 信息")

        fig2, ax2 = plt.subplots(1, 1, figsize=(4, 3))
        for p in mask_values:
            df_slice = df[df[mask_col] == p]
            xs = _unique_sorted(df_slice[noise_col])
            ys = []
            for sigma in xs:
                val = df_slice.loc[df_slice[noise_col] == sigma, metric_col].mean()
                ys.append(val)
            ax2.plot(xs, ys, marker="o", label=f"p={p:g}")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel(noise_col)
        ax2.set_ylabel(metric_col)
        ax2.set_title(f"{model_name}: {metric_col} vs {noise_col}")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax2.legend(fontsize=7)
        fig2.tight_layout()

        return fig1, fig2

    fig_lin_mask, fig_lin_noise = _plot_pair(df_linear, "linear")
    figs["fig_nmse_vs_mask_linear"] = fig_lin_mask
    figs["fig_nmse_vs_noise_linear"] = fig_lin_noise

    if df_mlp is not None:
        fig_mlp_mask, fig_mlp_noise = _plot_pair(df_mlp, "mlp")
        figs["fig_nmse_vs_mask_mlp"] = fig_mlp_mask
        figs["fig_nmse_vs_noise_mlp"] = fig_mlp_noise

    return figs
