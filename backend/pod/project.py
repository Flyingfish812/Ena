# backend/pod/project.py

"""
在 POD 基底上进行投影与重建的工具函数。
"""

from typing import Tuple

import numpy as np


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
        形状为 [N, D] 或 [D] 的数组。
    Ur:
        POD 基底矩阵，形状为 [D, r]。
    mean_flat:
        可选的均值场展平向量，若给定则先做 x_flat - mean_flat。

    返回
    ----
    a:
        POD 系数，形状为 [N, r] 或 [r]。
    """
    raise NotImplementedError


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
    raise NotImplementedError
