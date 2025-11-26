# backend/models/linear_baseline.py

"""
POD 子空间中的线性最小二乘基线解法。

用于从稀疏观测 y 估计 POD 系数 a_lin。
"""

from typing import Tuple

import numpy as np


def solve_pod_coeffs_least_squares(
    y: np.ndarray,
    Ur_masked: np.ndarray,
) -> np.ndarray:
    """
    对于给定观测向量 y 和掩膜后的 POD 基底列 Ur_masked，解最小二乘问题：

        min_a || Ur_masked @ a - y ||^2

    参数
    ----
    y:
        观测向量，形状为 [M] 或 [N,M]。
    Ur_masked:
        观测点处的 POD 基底列，形状为 [M, r]。

    返回
    ----
    a_lin:
        线性估计的 POD 系数，形状为 [r] 或 [N,r]。
    """
    raise NotImplementedError
