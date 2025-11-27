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
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)

    diff = x_hat - x_true
    num = np.sum(diff ** 2)
    denom = np.sum(x_true ** 2)
    if denom == 0:
        return float("nan")
    return float(num / denom)


def nmae(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    计算归一化平均绝对误差 (Normalized MAE)。

        NMAE = mean(|x_hat - x_true|) / mean(|x_true|)
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)

    diff = np.abs(x_hat - x_true)
    num = np.mean(diff)
    denom = np.mean(np.abs(x_true))
    if denom == 0:
        return float("nan")
    return float(num / denom)


def psnr(x_hat: np.ndarray, x_true: np.ndarray, data_range: float | None = None) -> float:
    """
    计算峰值信噪比 (PSNR)。

    参数
    ----
    data_range:
        可选，数据值范围（max - min），若不提供则自动从 x_true 推断。
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)

    if data_range is None:
        data_range = float(x_true.max() - x_true.min())
        if data_range == 0:
            return float("nan")

    mse = np.mean((x_hat - x_true) ** 2)
    if mse == 0:
        return float("inf")

    return 10.0 * float(np.log10((data_range ** 2) / mse))
