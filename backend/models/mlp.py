# backend/models/mlp.py

"""
POD 系数回归用的简单 MLP 模型。
"""

from typing import Iterable, List

import torch
import torch.nn as nn


class PodMLP(nn.Module):
    """
    一个简单的全连接 MLP，用于从观测向量 y ∈ R^M 预测 POD 系数 a ∈ R^r。

    结构示例:
        M -> 256 -> 256 -> r
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Iterable[int] = (256, 256),
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        dims: List[int] = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:
                layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [batch, in_dim]
        返回: [batch, out_dim]
        """
        return self.net(y)
