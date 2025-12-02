# backend/viz/multiscale_plots.py

"""
多尺度 POD band 误差的柱状图可视化。
"""

from typing import Dict

import matplotlib.pyplot as plt


def plot_multiscale_bar(
    band_errors: Dict[str, float],
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """
    绘制一个组合的 POD band 误差柱状图。

    band_errors 形如 {"L": 0.01, "M": 0.02, "S": 0.05}。
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
