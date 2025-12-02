# backend/dataio/observation_dataset.py

"""
ObservationDataset:
将时序场数据 + POD 基底 + 固定 mask 组合成 y -> a 的监督数据集，
用于训练 MLP 基线。
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..pod.project import project_to_pod
from ..sampling.masks import apply_mask_flat
from ..sampling.noise import add_gaussian_noise


class ObservationDataset(Dataset):
    """
    观测数据集（固定 mask 版本）：

    每个样本:
        输入:  带噪观测向量 y_noisy ∈ R^M
        输出:  POD 真系数 a_true ∈ R^r

    约定:
    - 所有样本共享同一个 mask_flat，M = mask_flat.sum()
    - X_flat_all: 所有 snapshot 展平后的场 [N, D]
    - Ur_eff: POD 基底 [D, r_eff]
    - mean_flat: 均值场 [D]
    """

    def __init__(
        self,
        X_flat_all: np.ndarray,
        Ur_eff: np.ndarray,
        mean_flat: np.ndarray,
        mask_flat: np.ndarray,
        noise_sigma: float = 0.0,
    ) -> None:
        super().__init__()

        X_flat_all = np.asarray(X_flat_all, dtype=np.float32)  # [N,D]
        Ur_eff = np.asarray(Ur_eff, dtype=np.float32)          # [D,r]
        mean_flat = np.asarray(mean_flat, dtype=np.float32)    # [D]
        mask_flat = np.asarray(mask_flat, dtype=bool)          # [D]

        if X_flat_all.ndim != 2:
            raise ValueError(f"X_flat_all must be [N,D], got {X_flat_all.shape}")
        if Ur_eff.ndim != 2:
            raise ValueError(f"Ur_eff must be [D,r], got {Ur_eff.shape}")
        if mean_flat.ndim != 1:
            raise ValueError(f"mean_flat must be [D], got {mean_flat.shape}")
        if mask_flat.ndim != 1:
            raise ValueError(f"mask_flat must be [D], got {mask_flat.shape}")
        if X_flat_all.shape[1] != Ur_eff.shape[0] or X_flat_all.shape[1] != mask_flat.shape[0]:
            raise ValueError(
                f"Dimension mismatch: "
                f"X_flat_all.shape={X_flat_all.shape}, "
                f"Ur_eff.shape={Ur_eff.shape}, "
                f"mask_flat.shape={mask_flat.shape}"
            )

        self.X_flat_all = X_flat_all          # [N,D]
        self.Ur_eff = Ur_eff                  # [D,r]
        self.mean_flat = mean_flat            # [D]
        self.mask_flat = mask_flat            # [D]
        self.noise_sigma = float(noise_sigma)

        # 预先计算所有 snapshot 的 POD 真系数 a_true_all: [N,r]
        self.a_true_all = project_to_pod(
            X_flat_all,
            Ur_eff,
            mean_flat,
        ).astype(np.float32)

        # 观测维度 M
        self.M = int(mask_flat.sum())
        self.r_eff = Ur_eff.shape[1]

    def __len__(self) -> int:
        return self.X_flat_all.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 展平场
        x_flat = self.X_flat_all[idx]            # [D]
        a_true = self.a_true_all[idx]            # [r]

        # 应用 mask 得到观测 y
        y = apply_mask_flat(x_flat, self.mask_flat)  # [M]

        # 加噪
        y_noisy = add_gaussian_noise(y, sigma=self.noise_sigma)

        # 转 torch tensor
        y_tensor = torch.from_numpy(y_noisy.astype(np.float32))
        a_tensor = torch.from_numpy(a_true.astype(np.float32))

        return y_tensor, a_tensor
