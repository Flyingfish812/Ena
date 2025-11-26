# backend/models/train_mlp.py

"""
MLP 训练主循环。

负责：
- 构建 ObservationDataset
- 创建 DataLoader
- 训练 / 验证 MLP，保存最优模型
"""

from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config.schemas import DataConfig, PodConfig, TrainConfig
from ..pod.project import project_to_pod
from ..sampling.masks import flatten_mask, apply_mask_flat
from ..sampling.noise import add_gaussian_noise
from .mlp import build_mlp
from .datasets import ObservationDataset


def prepare_observation_datasets(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: TrainConfig,
) -> Tuple[ObservationDataset, ObservationDataset]:
    """
    从原始数据与 POD 基底构造训练集与验证集。

    步骤（实现时）：
    1. 读取 POD 基底与均值。
    2. 加载 train/val 对应的 snapshot。
    3. 投影到 POD 系数 a_true。
    4. 根据 mask_rate 生成固定 mask，并提取观测向量 y。
    5. 对 y 加噪，得到 y_noisy。
    6. 构造 ObservationDataset(train) 与 ObservationDataset(val)。

    返回
    ----
    train_ds, val_ds
    """
    raise NotImplementedError


def train_mlp(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    使用给定配置训练一个 MLP 模型。

    返回
    ----
    result:
        {
            "best_model_path": str,
            "train_losses": list[float],
            "val_losses": list[float],
        }
    """
    raise NotImplementedError
