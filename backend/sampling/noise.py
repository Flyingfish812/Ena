# backend/sampling/noise.py

"""
对观测向量加入噪声。
"""

import numpy as np


def add_gaussian_noise(
    y: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    在观测向量上加入零均值高斯噪声。

    参数
    ----
    y:
        原始观测值，形状 [M] 或 [N,M]。
    sigma:
        噪声标准差。
    seed:
        随机种子，便于复现。

    返回
    ----
    y_noisy:
        加噪后的观测向量，形状与 y 相同。
    """
    raise NotImplementedError
