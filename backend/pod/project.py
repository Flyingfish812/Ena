# backend/pod/project.py

"""
在 POD 基底上进行投影与重建的工具函数。
"""

from typing import Tuple

import numpy as np


def _as_2d(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    将输入转为 [N, D] 形式，记录是否为单样本。
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :], True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")


def project_to_pod(
    x_flat: np.ndarray,
    Ur: np.ndarray,
    mean_flat: np.ndarray | None = None,
) -> np.ndarray:
    """
    将一个或一批空间场展平后的向量投影到 POD 系数空间。

    参数
    ----
    x_flat:
        形状为 [N, D] 或 [D] 的数组（展平后的场）。
    Ur:
        POD 基底矩阵，形状为 [D, r]，列为正交基。
    mean_flat:
        可选的均值场展平向量，若给定则先做 x_flat - mean_flat。

    返回
    ----
    a:
        POD 系数，形状为 [N, r] 或 [r]。
    """
    X, is_single = _as_2d(x_flat)
    Ur = np.asarray(Ur)
    if mean_flat is not None:
        m = np.asarray(mean_flat).reshape(1, -1)
        Xc = X - m
    else:
        Xc = X

    # 正交基下的投影： a = Xc @ Ur
    a = Xc @ Ur  # [N,D] @ [D,r] -> [N,r]

    return a[0] if is_single else a


def reconstruct_from_pod(
    a: np.ndarray,
    Ur: np.ndarray,
    mean_flat: np.ndarray | None = None,
) -> np.ndarray:
    """
    根据 POD 系数 a 重建空间场展平向量。

    参数
    ----
    a:
        形状为 [N, r] 或 [r]。
    Ur:
        POD 基底矩阵，形状为 [D, r]。
    mean_flat:
        可选的均值场展平向量，若给定则在重建后加回。

    返回
    ----
    x_flat:
        形状为 [N, D] 或 [D] 的重建向量。
    """
    A, is_single = _as_2d(a)
    Ur = np.asarray(Ur)

    Xc = A @ Ur.T  # [N,r] @ [r,D] -> [N,D]

    if mean_flat is not None:
        m = np.asarray(mean_flat).reshape(1, -1)
        X = Xc + m
    else:
        X = Xc

    return X[0] if is_single else X
