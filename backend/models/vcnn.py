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


class _EarlyExitHead(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class _EarlyExitCoeffHead(nn.Module):
    def __init__(self, hidden_channels: int, out_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class VCNNMultiScale(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 48,
        num_layers: int = 8,
        kernel_size: int = 7,
        output_mode: str = "field",
        coeff_dims: tuple[int, int, int] = (16, 48, 128),
    ) -> None:
        super().__init__()
        if num_layers < 5:
            raise ValueError(f"num_layers must be >=5 for multi-scale exits, got {num_layers}")

        padding = int(kernel_size) // 2
        hidden_blocks: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
                nn.ReLU(inplace=True),
            )
        ]
        for _ in range(int(num_layers) - 2):
            hidden_blocks.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                )
            )

        mode = str(output_mode).strip().lower()
        if mode not in ("field", "coeff"):
            raise ValueError(f"Unsupported output_mode='{output_mode}'. Expected 'field' or 'coeff'.")
        coeff_dims_eff = tuple(int(v) for v in coeff_dims)
        if len(coeff_dims_eff) != 3 or any(v <= 0 for v in coeff_dims_eff):
            raise ValueError(f"coeff_dims must contain 3 positive integers, got {coeff_dims}")

        self.hidden_blocks = nn.ModuleList(hidden_blocks)
        self.output_mode = mode
        self.coeff_dims = coeff_dims_eff
        if self.output_mode == "field":
            self.exit1_head = _EarlyExitHead(hidden_channels=hidden_channels, out_channels=out_channels)
            self.exit2_head = _EarlyExitHead(hidden_channels=hidden_channels, out_channels=out_channels)
            self.final_head = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        else:
            self.exit1_head = _EarlyExitCoeffHead(hidden_channels=hidden_channels, out_dim=self.coeff_dims[0])
            self.exit2_head = _EarlyExitCoeffHead(hidden_channels=hidden_channels, out_dim=self.coeff_dims[1])
            self.final_head = _EarlyExitCoeffHead(hidden_channels=hidden_channels, out_dim=self.coeff_dims[2])

    def forward(self, x: torch.Tensor, exit_level: int | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if exit_level is not None and int(exit_level) not in (1, 2, 3):
            raise ValueError(f"exit_level must be one of 1, 2, 3, got {exit_level}")

        h = x
        y_exit1: torch.Tensor | None = None
        y_exit2: torch.Tensor | None = None

        for block_idx, block in enumerate(self.hidden_blocks):
            h = block(h)
            if block_idx == 1:
                y_exit1 = self.exit1_head(h)
                if exit_level == 1:
                    return y_exit1
            if block_idx == 3:
                y_exit2 = self.exit2_head(h)
                if exit_level == 2:
                    return y_exit2

        y_exit3 = self.final_head(h)
        if exit_level == 3:
            return y_exit3

        if y_exit1 is None or y_exit2 is None:
            raise RuntimeError("Internal error: early-exit features were not constructed as expected.")
        return y_exit1, y_exit2, y_exit3