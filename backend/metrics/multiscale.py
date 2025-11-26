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
        预测的 POD 系数，形状 [N, r]。
    a_true:
        真实的 POD 系数，形状 [N, r]。
    bands:
        例如 {"L": (0,10), "M": (10,40), "S": (40,128)}，
        下标区间为半开区间 [start, end)。

    返回
    ----
    band_errors:
        例如 {"L": 0.01, "M": 0.02, "S": 0.05}
    """
    raise NotImplementedError
