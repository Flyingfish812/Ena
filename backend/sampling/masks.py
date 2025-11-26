# backend/sampling/masks.py

"""
生成稀疏空间观测的 mask，并在展平向量上应用。
"""

from typing import Tuple

import numpy as np


def generate_random_mask_hw(
    H: int,
    W: int,
    mask_rate: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    在 H×W 网格上生成随机均匀采样的观测 mask。

    返回
    ----
    mask:
        形状为 [H, W] 的 bool 数组，其中 True 表示被观测。
    """
    raise NotImplementedError


def flatten_mask(mask_hw: np.ndarray, C: int) -> np.ndarray:
    """
    将 H×W 的空间 mask 扩展到包含通道维度后展平为长度 D 的向量。

    例如：
    - 输入 mask_hw 形状 [H,W]
    - 输出 mask_flat 形状 [H*W*C]
    """
    raise NotImplementedError


def apply_mask_flat(
    x_flat: np.ndarray,
    mask_flat: np.ndarray,
) -> np.ndarray:
    """
    在展平向量上应用 mask，只保留被观测的元素。

    参数
    ----
    x_flat:
        形状为 [D] 或 [N,D] 的数组。
    mask_flat:
        形状为 [D] 的 bool 数组。

    返回
    ----
    y:
        观测值向量：
        - 若输入为 [D]，则输出为 [M]
        - 若输入为 [N,D]，则输出为 [N,M]
        其中 M 为 mask 中 True 的个数。
    """
    raise NotImplementedError
