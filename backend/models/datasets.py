# backend/models/datasets.py

"""
用于 MLP 训练的 PyTorch Dataset。

Dataset 的职责：
- 提供 (观测向量 y_noisy, 真值 POD 系数 a_true) 对
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ObservationDataset(Dataset):
    """
    MLP 训练数据集。

    每个样本包含：
    - y_noisy: 加噪后的观测向量
    - a_true: 真实 POD 系数
    """

    def __init__(
        self,
        y: np.ndarray,
        a_true: np.ndarray,
    ) -> None:
        """
        参数
        ----
        y:
            形状为 [N, M] 的观测向量数组。
        a_true:
            形状为 [N, r] 的 POD 系数数组。
        """
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回单个样本 (y_noisy, a_true)，均为 float32 Tensor。
        """
        raise NotImplementedError
