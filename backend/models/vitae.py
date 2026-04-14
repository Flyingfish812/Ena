"""
Lightweight ViTAE-style model for sparse field reconstruction.

This implementation follows the external repository's broad design:
patch embedding -> transformer encoder -> CNN decoder, and returns
both a main decoder prediction and an auxiliary encoder-side prediction.
"""

from __future__ import annotations

from math import pi
from typing import Sequence

import torch
import torch.nn as nn

from .vcnn import get_field_loss


def _build_2d_sincos_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"dim must be divisible by 4 for 2D sincos embedding, got {dim}")

    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, (half // 2 - 1))))

    yy, xx = torch.meshgrid(
        torch.arange(grid_h, dtype=torch.float32),
        torch.arange(grid_w, dtype=torch.float32),
        indexing="ij",
    )
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    out_y = yy * omega.view(1, -1) * 2.0 * pi
    out_x = xx * omega.view(1, -1) * 2.0 * pi
    emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return emb


class CNNDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.02, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ViTAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        patch_size: int,
        enc_channels: int = 32,
        enc_dim: int = 64,
        enc_depth: int = 8,
        enc_num_heads: int = 8,
        enc_mlp_ratio: float = 4.0,
        dec_dims: Sequence[int] = (32, 32, 32, 32, 32),
    ) -> None:
        super().__init__()
        h, w = int(input_size[0]), int(input_size[1])
        if h % int(patch_size) != 0 or w % int(patch_size) != 0:
            raise ValueError(
                f"input_size must be divisible by patch_size, got input_size={input_size}, patch_size={patch_size}"
            )

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.enc_channels = int(enc_channels)
        self.patch_size = int(patch_size)
        self.grid_size = (h // self.patch_size, w // self.patch_size)

        self.patch_embed = nn.Conv2d(
            self.in_channels,
            enc_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
        )
        pos_embed = _build_2d_sincos_pos_embed(self.grid_size[0], self.grid_size[1], enc_dim)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=enc_dim,
            nhead=enc_num_heads,
            dim_feedforward=int(enc_dim * float(enc_mlp_ratio)),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(enc_depth))
        self.encoder_norm = nn.LayerNorm(enc_dim)

        self.decoder_embed = nn.Linear(enc_dim, self.patch_size * self.patch_size * self.enc_channels)
        self.encoder_out = nn.Conv2d(self.enc_channels, self.out_channels, kernel_size=1)

        decoder_dims = [self.enc_channels] + [int(v) for v in dec_dims]
        blocks = [CNNDecoderBlock(decoder_dims[i], decoder_dims[i + 1]) for i in range(len(decoder_dims) - 1)]
        self.decoder_cnn = nn.Sequential(*blocks)
        self.decoder_out = nn.Conv2d(decoder_dims[-1], self.out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_patches, _ = x.shape
        gh, gw = self.grid_size
        ph = pw = self.patch_size
        if num_patches != gh * gw:
            raise ValueError(f"Unexpected num_patches={num_patches}, expected {gh * gw}")
        x = x.view(batch, gh, gw, self.enc_channels, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch, self.enc_channels, gh * ph, gw * pw)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1], :].to(dtype=x.dtype, device=x.device)
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward_decoder(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.decoder_embed(latent)
        x = self._unpatchify(x)
        pred_enc = self.encoder_out(x)
        pred_dec = self.decoder_out(self.decoder_cnn(x))
        return pred_dec, pred_enc

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.forward_encoder(x)
        return self.forward_decoder(latent)


def build_vitae_model(
    *,
    input_size: tuple[int, int],
    in_channels: int,
    out_channels: int,
    patch_size: int,
    variant: str = "base",
) -> ViTAutoEncoder:
    name = str(variant or "base").strip().lower()
    if name == "lite":
        return ViTAutoEncoder(input_size, in_channels, out_channels, patch_size, enc_channels=16, enc_dim=32, dec_dims=(16, 16, 16, 16, 16))
    if name == "base":
        return ViTAutoEncoder(input_size, in_channels, out_channels, patch_size, enc_channels=32, enc_dim=64, dec_dims=(32, 32, 32, 32, 32))
    if name == "large":
        return ViTAutoEncoder(input_size, in_channels, out_channels, patch_size, enc_channels=64, enc_dim=128, dec_dims=(64, 64, 64, 64, 64))
    raise ValueError("variant must be one of: lite, base, large")


def build_vitae_loss(loss_type: str = "mae", obs_weight: float = 1.0) -> nn.Module:
    return get_field_loss(loss_type=loss_type, obs_weight=obs_weight)