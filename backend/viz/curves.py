# backend/viz/curves.py

"""
NMSE/NMAE/PSNR 随 mask_rate / noise_sigma 变化的曲线。
"""

from typing import Dict, Any

import matplotlib.pyplot as plt


def plot_nmse_vs_mask_rate(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 mask_rate 下的 NMSE 曲线。

    results 的格式应与 eval 模块的输出保持一致。
    """
    raise NotImplementedError


def plot_nmse_vs_noise(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 noise_sigma 下的 NMSE 曲线。
    """
    raise NotImplementedError
