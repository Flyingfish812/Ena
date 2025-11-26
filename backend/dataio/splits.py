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
    raise NotImplementedError
