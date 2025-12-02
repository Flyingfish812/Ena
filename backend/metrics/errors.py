# backend/metrics/errors.py

"""
基础误差指标与常用变体：

- mse    : Mean Squared Error
- mae    : Mean Absolute Error
- rmse   : Root Mean Squared Error
- linf   : L∞ 绝对误差（max |x_hat - x_true|）

- nmse   : Normalized MSE = ||e||² / ||x_true||²
- nmae   : Normalized MAE = mean|e| / mean|x_true|
- psnr   : Peak Signal-to-Noise Ratio（基于 MSE）

另外提供一个工具函数：
- compute_basic_errors(...) 一次性返回上述指标的字典，
  方便在评估循环里调用并写入 logs / DataFrame。
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np


# ---------------------------------------------------------------------
# 内部工具：统一转成 float64 数组
# ---------------------------------------------------------------------


def _to_float_arrays(
    x_hat: np.ndarray, x_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)
    return x_hat, x_true


# ---------------------------------------------------------------------
# 原始（未归一化）误差：MSE / MAE / RMSE / L∞
# ---------------------------------------------------------------------


def mse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    Mean Squared Error:

        MSE = mean( (x_hat - x_true)^2 )
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)
    diff = x_hat - x_true
    return float(np.mean(diff ** 2))


def mae(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    Mean Absolute Error:

        MAE = mean( |x_hat - x_true| )
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)
    diff = np.abs(x_hat - x_true)
    return float(np.mean(diff))


def rmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    Root Mean Squared Error:

        RMSE = sqrt( MSE )
    """
    return float(np.sqrt(mse(x_hat, x_true)))


def linf(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    L∞ 绝对误差：

        L∞ = max( |x_hat - x_true| )
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)
    diff = np.abs(x_hat - x_true)
    return float(np.max(diff))


# ---------------------------------------------------------------------
# 归一化误差：NMSE / NMAE
# ---------------------------------------------------------------------


def nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    归一化均方误差 (Normalized MSE)。

        NMSE = ||x_hat - x_true||^2 / ||x_true||^2

    等价于：NMSE = (L2(e)^2 / L2(x_true)^2)
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)

    diff = x_hat - x_true
    num = np.sum(diff ** 2)
    denom = np.sum(x_true ** 2)
    if denom == 0:
        return float("nan")
    return float(num / denom)


def nmae(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    归一化平均绝对误差 (Normalized MAE)。

        NMAE = mean(|x_hat - x_true|) / mean(|x_true|)

    注意：若 mean(|x_true|) = 0，则返回 NaN。
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)

    diff = np.abs(x_hat - x_true)
    num = np.mean(diff)
    denom = np.mean(np.abs(x_true))
    if denom == 0:
        return float("nan")
    return float(num / denom)


# ---------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------


def psnr(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    data_range: float | None = None,
) -> float:
    """
    峰值信噪比 (PSNR)。

    参数
    ----
    data_range:
        可选，数据值范围（max - min），若不提供则自动从 x_true 推断。

    定义
    ----
        PSNR = 10 * log10( data_range^2 / MSE )
    """
    x_hat, x_true = _to_float_arrays(x_hat, x_true)

    if data_range is None:
        data_range = float(x_true.max() - x_true.min())
        if data_range == 0:
            return float("nan")

    _mse = mse(x_hat, x_true)
    if _mse == 0:
        return float("inf")

    return 10.0 * float(np.log10((data_range ** 2) / _mse))


# ---------------------------------------------------------------------
# 便捷汇总函数：一把梭返回常用误差
# ---------------------------------------------------------------------


def compute_basic_errors(
    x_hat: np.ndarray,
    x_true: np.ndarray,
    data_range: float | None = None,
) -> Dict[str, Any]:
    """
    一次性计算一组基础误差指标，返回字典：

        {
            "mse": ...,
            "mae": ...,
            "rmse": ...,
            "linf": ...,
            "nmse": ...,
            "nmae": ...,
            "psnr": ...,
        }

    方便在评估循环内调用，然后直接写入结果表 / JSON。
    """
    return {
        "mse": mse(x_hat, x_true),
        "mae": mae(x_hat, x_true),
        "rmse": rmse(x_hat, x_true),
        "linf": linf(x_hat, x_true),
        "nmse": nmse(x_hat, x_true),
        "nmae": nmae(x_hat, x_true),
        "psnr": psnr(x_hat, x_true, data_range=data_range),
    }
