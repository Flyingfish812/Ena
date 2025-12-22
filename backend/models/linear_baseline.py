# backend/models/linear_baseline.py

"""
POD 子空间中的线性最小二乘基线解法。

用于从稀疏观测 y 估计 POD 系数 a_lin。
"""

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
    Y = np.asarray(y, dtype=np.float64)
    A = np.asarray(Ur_masked, dtype=np.float64)  # [M, r]

    if A.ndim != 2:
        raise ValueError(f"Ur_masked must be 2D [M,r], got {A.shape}")

    M, r = A.shape

    if Y.ndim == 1:
        if Y.shape[0] != M:
            raise ValueError(f"Dimension mismatch: y[{Y.shape}] vs Ur_masked[{A.shape}]")
        coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)  # [r]
        return coeffs.astype(np.float32)

    if Y.ndim == 2:
        if Y.shape[1] != M:
            raise ValueError(f"Dimension mismatch: y[{Y.shape}] vs Ur_masked[{A.shape}]")

        # 关键优化：一次 lstsq 解多个 RHS
        # A: [M, r], Y.T: [M, N]  ->  coeffs: [r, N]
        coeffs, *_ = np.linalg.lstsq(A, Y.T, rcond=None)
        return coeffs.T.astype(np.float32)  # [N, r]

    raise ValueError(f"y must be 1D or 2D, got {Y.shape}")