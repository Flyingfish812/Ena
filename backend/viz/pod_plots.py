# backend/viz/pod_plots.py

"""
POD 能量谱相关可视化。
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_energy_spectrum(
    singular_values: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    绘制 POD 奇异值谱（能量谱）。

    返回用于后续调整的 Axes。
    """
    raise NotImplementedError


def plot_cumulative_energy(
    cum_energy: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    绘制 POD 累计能量曲线。
    """
    raise NotImplementedError
