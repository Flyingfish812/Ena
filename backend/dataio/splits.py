# backend/dataio/splits.py

"""
提供 train/val/test 划分工具。
"""

from typing import Tuple, Dict

import numpy as np


def train_val_test_split(
    num_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    按比例将 [0, num_samples) 的索引划分为 train/val/test 三个集合。

    返回一个字典：
        {
            "train": np.ndarray[int],
            "val":   np.ndarray[int],
            "test":  np.ndarray[int],
        }
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be in (0,1).")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.")

    rng = np.random.RandomState(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    n_test = num_samples - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
