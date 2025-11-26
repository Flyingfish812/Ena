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
    """
    raise NotImplementedError
