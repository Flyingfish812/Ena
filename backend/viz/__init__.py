# backend/viz/__init__.py

"""
各类可视化函数，使用 matplotlib 绘制。

注意：这里只负责“画图本身”，
不负责数据预处理或实验逻辑。
"""

from .pod_plots import plot_energy_spectrum, plot_cumulative_energy
from .field_plots import plot_field_comparison, plot_error_map
from .curves import plot_nmse_vs_mask_rate, plot_nmse_vs_noise
from .multiscale_plots import plot_multiscale_bar

__all__ = [
    "plot_energy_spectrum",
    "plot_cumulative_energy",
    "plot_field_comparison",
    "plot_error_map",
    "plot_nmse_vs_mask_rate",
    "plot_nmse_vs_noise",
    "plot_multiscale_bar",
]
