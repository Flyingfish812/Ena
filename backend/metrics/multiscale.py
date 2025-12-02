# backend/metrics/multiscale.py

"""
基于 POD 模态的多尺度误差评估。
"""

from typing import Dict, Tuple

import numpy as np


def compute_pod_band_errors(
    a_hat: np.ndarray,
    a_true: np.ndarray,
    bands: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    """
    按给定的 POD 模态区间（band）计算每个 band 的系数 RMSE。

    参数
    ----
    a_hat:
        预测的 POD 系数，形状 [N, r] 或 [r]。
    a_true:
        真实的 POD 系数，形状 [N, r] 或 [r]。
    bands:
        例如 {"L": (0,10), "M": (10,40), "S": (40,128)}，
        下标区间为半开区间 [start, end)。

    返回
    ----
    band_errors:
        例如 {"L": 0.01, "M": 0.02, "S": 0.05}，
        这里的数值是该 band 内所有样本、所有模态的 **系数 RMSE**：
            RMSE_band = sqrt( mean( (a_hat - a_true)^2 ) )
    """
    a_hat = np.asarray(a_hat, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)

    if a_hat.shape != a_true.shape:
        raise ValueError(f"a_hat shape {a_hat.shape} != a_true shape {a_true.shape}")

    # 统一成 [N, r]
    if a_hat.ndim == 1:
        a_hat = a_hat[None, :]
        a_true = a_true[None, :]
    elif a_hat.ndim != 2:
        raise ValueError(f"a_hat must be 1D or 2D, got {a_hat.shape}")

    N, r = a_hat.shape

    band_errors: Dict[str, float] = {}
    for name, (start, end) in bands.items():
        if not (0 <= start < end <= r):
            raise ValueError(
                f"Band '{name}' with range [{start},{end}) is invalid for r={r}"
            )

        diff = a_hat[:, start:end] - a_true[:, start:end]  # [N, r_band]
        mse = float(np.mean(diff ** 2))  # 所有样本+该段模态一起平均
        rmse = float(np.sqrt(mse))
        band_errors[name] = rmse

    return band_errors
