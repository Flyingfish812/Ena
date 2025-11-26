# backend/models/mlp.py

"""
用于预测 POD 系数的简单 MLP 模型。
"""

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    多层感知机，用于从观测向量预测 POD 系数。

    结构：
        input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> output_dim
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数
        ----
        x:
            形状为 [N, input_dim] 的张量。

        返回
        ----
        out:
            形状为 [N, output_dim] 的预测 POD 系数。
        """
        raise NotImplementedError


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "relu",
) -> MLP:
    """
    构建一个 MLP 模型实例。

    封装一层，便于在其它模块中统一创建模型。
    """
    raise NotImplementedError
