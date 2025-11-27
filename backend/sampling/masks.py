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
    if not (0 < mask_rate <= 1.0):
        raise ValueError(f"mask_rate must be in (0,1], got {mask_rate}")

    rng = np.random.RandomState(seed)
    num_points = H * W
    num_obs = max(1, int(round(num_points * mask_rate)))

    flat_mask = np.zeros(num_points, dtype=bool)
    idx = rng.choice(num_points, size=num_obs, replace=False)
    flat_mask[idx] = True

    return flat_mask.reshape(H, W)


def flatten_mask(mask_hw: np.ndarray, C: int) -> np.ndarray:
    """
    将 H×W 的空间 mask 扩展到包含通道维度后展平为长度 D 的向量。

    例如：
    - 输入 mask_hw 形状 [H,W]
    - 输出 mask_flat 形状 [H*W*C]
    """
    mask_hw = np.asarray(mask_hw, dtype=bool)
    if mask_hw.ndim != 2:
        raise ValueError(f"mask_hw must be 2D [H,W], got {mask_hw.shape}")

    H, W = mask_hw.shape
    # [H,W] -> [H,W,1] -> [H,W,C]
    mask_hwc = np.repeat(mask_hw[:, :, None], C, axis=2)
    mask_flat = mask_hwc.reshape(-1)  # [H*W*C]

    return mask_flat


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
    x = np.asarray(x_flat)
    mask = np.asarray(mask_flat, dtype=bool)

    if x.ndim == 1:
        if x.shape[0] != mask.shape[0]:
            raise ValueError(f"Dimension mismatch: x[{x.shape}] vs mask[{mask.shape}]")
        return x[mask]
    elif x.ndim == 2:
        if x.shape[1] != mask.shape[0]:
            raise ValueError(f"Dimension mismatch: x[{x.shape}] vs mask[{mask.shape}]")
        return x[:, mask]
    else:
        raise ValueError(f"x_flat must be 1D or 2D, got {x.shape}")
