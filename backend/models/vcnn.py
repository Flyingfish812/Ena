"""
VCNN-style fully convolutional baseline for sparse field reconstruction.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _weighted_error(error: torch.Tensor, obs_mask: Optional[torch.Tensor], obs_weight: float = 1.0) -> torch.Tensor:
    if obs_mask is None:
        return error
    weight = obs_mask * float(obs_weight) - (obs_mask - 1.0)
    return error * weight


class FieldL1Loss(nn.Module):
    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = torch.abs(target - pred)
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class FieldL2Loss(nn.Module):
    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = (target - pred) ** 2
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class LogCoshLoss(nn.Module):
    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = torch.log(torch.cosh(target - pred + 1e-12))
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class XTanhLoss(nn.Module):
    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = target - pred
        error = error * torch.tanh(error)
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class XSigmoidLoss(nn.Module):
    def __init__(self, obs_weight: float = 1.0) -> None:
        super().__init__()
        self.obs_weight = float(obs_weight)

    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = target - pred
        error = 2.0 * error / (1.0 + torch.exp(-error)) - error
        error = _weighted_error(error, obs_mask, self.obs_weight)
        return torch.mean(error)


class RelativeL2Loss(nn.Module):
    def forward(self, target: torch.Tensor, pred: torch.Tensor, obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = target - pred
        if obs_mask is not None:
            diff = diff * obs_mask
            target = target * obs_mask
        num = torch.sum(diff ** 2, dim=(1, 2, 3))
        den = torch.sum(target ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        return torch.mean(torch.sqrt(num / den))


def get_field_loss(loss_type: str = "mae", obs_weight: float = 1.0) -> nn.Module:
    name = str(loss_type or "mae").strip().lower()
    if name == "mae":
        return FieldL1Loss(obs_weight)
    if name == "mse":
        return FieldL2Loss(obs_weight)
    if name == "l2norm":
        return RelativeL2Loss()
    if name == "logcosh":
        return LogCoshLoss(obs_weight)
    if name == "xtanh":
        return XTanhLoss(obs_weight)
    if name == "xsigmoid":
        return XSigmoidLoss(obs_weight)
    raise ValueError(
        f"Unsupported loss_type='{loss_type}'. Supported: mae, mse, l2norm, logcosh, xtanh, xsigmoid"
    )


class VCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 8,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"num_layers must be >=2, got {num_layers}")
        padding = int(kernel_size) // 2

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        for _ in range(int(num_layers) - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)