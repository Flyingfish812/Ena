# backend/metrics/errors.py

"""
基础误差指标：NMSE / NMAE / PSNR。
"""

import numpy as np


def nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    计算归一化均方误差 (Normalized MSE)。

        NMSE = ||x_hat - x_true||^2 / ||x_true||^2
    """
    raise NotImplementedError


def nmae(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    计算归一化平均绝对误差 (Normalized MAE)。

        NMAE = mean(|x_hat - x_true|) / mean(|x_true|)
    """
    raise NotImplementedError


def psnr(x_hat: np.ndarray, x_true: np.ndarray, data_range: float | None = None) -> float:
    """
    计算峰值信噪比 (PSNR)。

    参数
    ----
    data_range:
        可选，数据值范围（max - min），若不提供则自动从 x_true 推断。
    """
    raise NotImplementedError
