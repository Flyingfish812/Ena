# backend/viz/field_plots.py

"""
单个样本的场可视化：真值 / 重建 / 误差图。
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_field_comparison(
    x_true_hw: np.ndarray,
    x_lin_hw: np.ndarray | None = None,
    x_nn_hw: np.ndarray | None = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    对比单个样本的空间场：

    - x_true
    - x_lin 线性基线
    - x_nn  MLP

    返回包含多个子图的 Figure，方便直接保存为论文插图。
    """
    raise NotImplementedError


def plot_error_map(
    x_true_hw: np.ndarray,
    x_hat_hw: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """
    绘制误差热力图 |x_hat - x_true|。
    """
    raise NotImplementedError
