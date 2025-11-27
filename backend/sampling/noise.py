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
        噪声标准差。若 sigma <= 0，则直接返回原值。
    seed:
        随机种子，便于复现。

    返回
    ----
    y_noisy:
        加噪后的观测向量，形状与 y 相同。
    """
    y = np.asarray(y, dtype=np.float32)

    if sigma <= 0:
        return y

    rng = np.random.RandomState(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=y.shape).astype(np.float32)
    return y + noise
