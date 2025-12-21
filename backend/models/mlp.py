# backend/models/mlp.py
"""
POD 系数回归用的简单 MLP 模型（优化版）：
- 可选 LayerNorm / Dropout
- 更稳的激活（默认 SiLU）
- 合理的权重初始化
- 可选输入/输出标准化封装（NormalizedPodMLP）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class PodMLP(nn.Module):
    """
    一个简单的全连接 MLP，用于从观测向量 y ∈ R^M 预测 POD 系数 a ∈ R^r。

    结构示例:
        M -> 256 -> 256 -> r
    可选:
        - LayerNorm（更稳）
        - Dropout（一般回归不一定需要，默认 0）
        - 更平滑激活函数 SiLU/GELU
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Iterable[int] = (256, 256),
        *,
        activation: str = "silu",
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError(f"in_dim/out_dim must be positive, got {in_dim}/{out_dim}")

        act = _get_activation(activation)
        dims: List[int] = [in_dim] + list(hidden_dims) + [out_dim]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            lin = nn.Linear(in_d, out_d)
            layers.append(lin)

            is_last = (i == len(dims) - 2)
            if not is_last:
                if use_layernorm:
                    layers.append(nn.LayerNorm(out_d))
                layers.append(act)
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(p=float(dropout)))

        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 对回归 MLP 来说，一个稳妥的初始化足够
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [batch, in_dim]
        返回: [batch, out_dim]
        """
        return self.net(y)


@dataclass
class NormStats:
    """
    保存标准化统计量（训练集上估计）。
    使用 buffer 存进模型后，推理可复现。
    """
    y_mean: torch.Tensor  # [M]
    y_std: torch.Tensor   # [M]
    a_mean: torch.Tensor  # [r]
    a_std: torch.Tensor   # [r]


class NormalizedPodMLP(nn.Module):
    """
    在 PodMLP 外包一层输入/输出标准化：
        y_norm = (y - μy) / σy
        a_norm = f(y_norm)
        a = a_norm * σa + μa

    这样：
    - 训练更快更稳（学习率更好调）
    - 输出直接是“原始尺度”的 POD 系数，外部调用不需要改
    """

    def __init__(self, core: PodMLP, stats: NormStats, eps: float = 1e-8) -> None:
        super().__init__()
        self.core = core
        self.eps = float(eps)

        # register_buffer: 跟随 .to(device)，并保存到 state_dict
        self.register_buffer("y_mean", stats.y_mean.clone())
        self.register_buffer("y_std", stats.y_std.clone())
        self.register_buffer("a_mean", stats.a_mean.clone())
        self.register_buffer("a_std", stats.a_std.clone())

    @torch.no_grad()
    def normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / (self.y_std + self.eps)

    @torch.no_grad()
    def denormalize_a(self, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm * (self.a_std + self.eps) + self.a_mean

    @torch.no_grad()
    def normalize_a(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / (self.a_std + self.eps)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y_norm = (y - self.y_mean) / (self.y_std + self.eps)
        a_norm = self.core(y_norm)
        a = a_norm * (self.a_std + self.eps) + self.a_mean
        return a
