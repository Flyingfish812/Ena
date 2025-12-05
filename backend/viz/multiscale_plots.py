# backend/viz/multiscale_plots.py

"""
多尺度 POD 误差与能量谱的可视化工具。

包含：
- plot_multiscale_bar: 单模型的 POD band 系数 RMSE 柱状图
- plot_multiscale_summary: 线性 vs MLP 的多尺度四合一对比图
"""

from typing import Dict, Mapping, Sequence, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_multiscale_bar(
    band_errors: Dict[str, float],
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """
    绘制一个组合的 POD band 误差柱状图。

    band_errors 形如 {"L": 0.01, "M": 0.02, "H": 0.05}。
    y 轴用线性坐标，数值是系数 RMSE（越低越好）。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    names = list(band_errors.keys())
    values = [float(band_errors[k]) for k in names]

    ax.bar(names, values)
    ax.set_xlabel("POD band")
    ax.set_ylabel("Coefficient RMSE")
    ax.set_title(title or "POD band-wise coefficient error")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    return ax


def _sorted_keys_union(*dicts: Mapping[str, float]) -> Sequence[str]:
    keys: set[str] = set()
    for d in dicts:
        keys.update(d.keys())
    # 简单按字符串排序，"S1" < "S1+S2" < "S1+S2+S3" 这种顺序是符合直觉的
    return sorted(keys)


def plot_multiscale_summary(
    entry: Mapping[str, object],
    *,
    energy_cum: Optional[Sequence[float]] = None,
    title_prefix: str = "",
    model_label: str = "model",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """
    为“单个模型在一组 (p,σ) 上”的结果画出四种多尺度曲线：

    1. Per-band NRMSE vs band index  （条形图）
    2. Per-mode NRMSE vs mode index  （折线图）
    3. 部分重建误差 E_(n,*)          （折线图）
    4. 能量比例 E^(n)                 （累计能量曲线，只跟 POD 有关）

    参数
    ----
    entry:
        run_linear_baseline_experiment / run_mlp_experiment 中某个 (p,σ) 的 entry：
            - "band_nrmse": Dict[str,float]
            - "coeff_nrmse_per_mode": List[float]
            - "field_nmse_partial": Dict[str,float]

    energy_cum:
        POD 模态的累计能量比例数组，长度为 r_eff。

    model_label:
        图例/标题中用于标记该模型的名字，例如 "linear" 或 "mlp"。

    返回
    ----
    fig, (ax_band, ax_mode, ax_partial, ax_energy)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax_band, ax_mode, ax_partial, ax_energy = axes.ravel()

    # ---------------- 1) Per-band NRMSE 条形图 ----------------
    band_nrmse = dict(entry.get("band_nrmse", {})) if entry else {}
    band_names = sorted(band_nrmse.keys())

    if band_names:
        x = np.arange(len(band_names))
        y = [float(band_nrmse[k]) for k in band_names]

        ax_band.bar(x, y, width=0.6)
        ax_band.set_xticks(x)
        ax_band.set_xticklabels(band_names)
        ax_band.set_xlabel("POD band")
        ax_band.set_ylabel("Per-band NRMSE")
        ax_band.set_title(f"Per-band NRMSE ({model_label})")
        ax_band.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax_band.text(
            0.5,
            0.5,
            "No band_nrmse available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_band.transAxes,
        )
        ax_band.set_axis_off()

    # ---------------- 2) Per-mode NRMSE 谱线 ----------------
    coeff_nrmse = entry.get("coeff_nrmse_per_mode", None)
    if coeff_nrmse is not None:
        coeff = np.asarray(coeff_nrmse, dtype=float)
        modes = np.arange(1, coeff.shape[0] + 1)
        ax_mode.plot(modes, coeff, linewidth=1.0)
        ax_mode.set_xlabel("Mode index k")
        ax_mode.set_ylabel("Per-mode NRMSE")
        ax_mode.set_title(f"Per-mode NRMSE vs mode index ({model_label})")
        ax_mode.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax_mode.text(
            0.5,
            0.5,
            "No per-mode NRMSE available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_mode.transAxes,
        )
        ax_mode.set_axis_off()

    # ---------------- 3) 部分重建误差 E_(n,*) ----------------
    field_partial = dict(entry.get("field_nmse_partial", {})) if entry else {}
    part_names = sorted(field_partial.keys())

    if part_names:
        x_idx = np.arange(1, len(part_names) + 1)
        y_vals = [float(field_partial[k]) for k in part_names]
        ax_partial.plot(
            x_idx,
            y_vals,
            marker="o",
            linestyle="-",
            linewidth=1.0,
        )
        ax_partial.set_xticks(x_idx)
        ax_partial.set_xticklabels(part_names, rotation=45, ha="right", fontsize=8)
        ax_partial.set_xlabel("Cumulative band index")
        ax_partial.set_ylabel(r"$E_{(n),\star}$ (NMSE)")
        ax_partial.set_title(f"Partial reconstruction NMSE ({model_label})")
        ax_partial.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax_partial.text(
            0.5,
            0.5,
            "No partial NMSE available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_partial.transAxes,
        )
        ax_partial.set_axis_off()

    # ---------------- 4) 能量比例累积曲线 ----------------
    if energy_cum is not None:
        e = np.asarray(energy_cum, dtype=float)
        n_modes = e.shape[0]
        x_e = np.arange(1, n_modes + 1)
        ax_energy.plot(x_e, e, linewidth=1.0)
        ax_energy.set_xlabel("Mode index n")
        ax_energy.set_ylabel(r"Cumulative energy $\mathcal{E}^{(n)}$")
        ax_energy.set_title("Cumulative POD energy ratio")
        ax_energy.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_energy.set_ylim(0.0, 1.05)
    else:
        ax_energy.text(
            0.5,
            0.5,
            "No energy_cum available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_energy.transAxes,
        )
        ax_energy.set_axis_off()

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, (ax_band, ax_mode, ax_partial, ax_energy)
