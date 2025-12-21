# backend/train/mlp_trainer.py
"""
在 ObservationDataset 上训练 PodMLP 的工具函数（优化版）：

优化点：
- 标准化 y/a（训练更快更稳）
- AdamW + OneCycleLR（更少 epoch 收敛）
- AMP 混合精度（CUDA 时加速）
- Early Stopping（不浪费 epoch）
- 更快 DataLoader（workers/pin_memory/non_blocking）
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..dataio.observation_dataset import ObservationDataset
from ..models.mlp import PodMLP, NormalizedPodMLP, NormStats


def _choose_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _estimate_norm_stats(
    loader: DataLoader,
    device: str,
    *,
    eps: float = 1e-8,
    max_batches: Optional[int] = None,
) -> NormStats:
    """
    用训练集估计 y/a 的均值与标准差（逐批累计，避免一次性吃满内存）。
    """
    # Welford 在线算法
    y_mean = None
    y_M2 = None
    y_count = 0

    a_mean = None
    a_M2 = None
    a_count = 0

    for bi, (y, a) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        y = y.to(device, non_blocking=True).float()
        a = a.to(device, non_blocking=True).float()

        # flatten batch dim
        # y: [B,M], a: [B,r]
        if y_mean is None:
            y_mean = torch.zeros(y.shape[1], device=device)
            y_M2 = torch.zeros(y.shape[1], device=device)
            a_mean = torch.zeros(a.shape[1], device=device)
            a_M2 = torch.zeros(a.shape[1], device=device)

        # 更新 y
        y_count_batch = y.shape[0]
        y_count_new = y_count + y_count_batch
        y_delta = y.mean(dim=0) - y_mean
        y_mean = y_mean + y_delta * (y_count_batch / max(1, y_count_new))
        # 近似：用 batch 内方差 + 均值差修正（足够用于标准化）
        y_var_batch = y.var(dim=0, unbiased=False)
        y_M2 = y_M2 + y_var_batch * y_count_batch + (y_delta ** 2) * (y_count * y_count_batch / max(1, y_count_new))
        y_count = y_count_new

        # 更新 a
        a_count_batch = a.shape[0]
        a_count_new = a_count + a_count_batch
        a_delta = a.mean(dim=0) - a_mean
        a_mean = a_mean + a_delta * (a_count_batch / max(1, a_count_new))
        a_var_batch = a.var(dim=0, unbiased=False)
        a_M2 = a_M2 + a_var_batch * a_count_batch + (a_delta ** 2) * (a_count * a_count_batch / max(1, a_count_new))
        a_count = a_count_new

    if y_mean is None or a_mean is None:
        raise RuntimeError("Empty loader while estimating normalization stats.")

    y_var = y_M2 / max(1, y_count)
    a_var = a_M2 / max(1, a_count)

    y_std = torch.sqrt(torch.clamp(y_var, min=0.0)) + eps
    a_std = torch.sqrt(torch.clamp(a_var, min=0.0)) + eps

    # 回 CPU 保存更轻
    return NormStats(
        y_mean=y_mean.detach().cpu(),
        y_std=y_std.detach().cpu(),
        a_mean=a_mean.detach().cpu(),
        a_std=a_std.detach().cpu(),
    )


def train_mlp_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    # --- dataloader / split ---
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    # --- training ---
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    use_onecycle: bool = True,
    max_lr: float | None = None,
    amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    # --- early stop ---
    early_stop: bool = True,
    patience: int = 5,
    min_delta: float = 0.0,
    # --- model ---
    hidden_dims: tuple[int, ...] = (256, 256),
    activation: str = "silu",
    use_layernorm: bool = True,
    dropout: float = 0.0,
    # --- normalization ---
    normalize_io: bool = True,
    norm_max_batches: int | None = None,
    # --- misc ---
    device: str | None = None,
    seed: int | None = 0,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    在给定的场数据 + POD 基底 + mask 上训练一个 MLP:

        y (masked noisy observation) -> a_true (POD 系数)

    返回:
      model: 训练好的模型（若 normalize_io=True，则返回 NormalizedPodMLP；否则 PodMLP）
      info:  训练曲线/配置/统计量等信息
    """
    device = _choose_device(device)

    if pin_memory is None:
        pin_memory = (device == "cuda")

    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        if device == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    # 构建数据集
    dataset = ObservationDataset(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=noise_sigma,
    )

    N = len(dataset)
    n_val = max(1, int(N * val_ratio))
    n_train = N - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    if verbose:
        print(f"[train_mlp] Dataset size: N={N}, train={n_train}, val={n_val}")
        print(f"[train_mlp] Obs dim M={dataset.M}, coeff dim r={dataset.r_eff}, device={device}")

    # DataLoader 加速参数
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    # --- 估计标准化统计量（仅用训练集）---
    stats = None
    if normalize_io:
        # 用一个“无 shuffle 的 loader”估计均值方差更稳定
        stat_loader = DataLoader(
            train_ds,
            batch_size=max(256, batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,  # 统计不需要多进程
            pin_memory=pin_memory,
        )
        stats = _estimate_norm_stats(
            stat_loader,
            device=device,
            max_batches=norm_max_batches,
        )
        if verbose:
            print("[train_mlp] Normalization enabled (y/a standardized on train split).")

    # --- 构建模型 ---
    core = PodMLP(
        in_dim=dataset.M,
        out_dim=dataset.r_eff,
        hidden_dims=hidden_dims,
        activation=activation,
        use_layernorm=use_layernorm,
        dropout=dropout,
    ).to(device)

    if normalize_io:
        assert stats is not None
        model: nn.Module = NormalizedPodMLP(core=core, stats=stats).to(device)
    else:
        model = core

    # --- 损失与优化器 ---
    criterion = nn.MSELoss()

    # AdamW 更稳（尤其配合 OneCycle）
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR：让你用更少 epoch 达到更好收敛
    scheduler = None
    if use_onecycle:
        if max_lr is None:
            # 如果启用了标准化，一般可以更大胆一点
            max_lr = 3e-3 if lr <= 1e-3 else lr
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=max(1.0, max_lr / max(lr, 1e-12)),
            final_div_factor=1e3,
        )
        if verbose:
            print(f"[train_mlp] OneCycleLR enabled: lr={lr:g}, max_lr={max_lr:g}")

    # AMP
    use_amp = (amp and device == "cuda")
    if device == "cuda":
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda", enabled=use_amp)
    else:
        scaler = None  # CPU 不用 AMP

    best_val_loss = float("inf")
    best_state_dict = None
    train_losses: list[float] = []
    val_losses: list[float] = []
    lr_hist: list[float] = []

    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for y_batch, a_batch in train_loader:
            y_batch = y_batch.to(device, non_blocking=True).float()
            a_batch = a_batch.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            if device == "cuda":
                with autocast("cuda", enabled=use_amp):
                    a_pred = model(y_batch)
                    loss = criterion(a_pred, a_batch)
            else:
                a_pred = model(y_batch)
                loss = criterion(a_pred, a_batch)

            loss.backward() if (device != "cuda" or not use_amp) else scaler.scale(loss).backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                if device == "cuda" and use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

            did_step = True
            if device == "cuda" and use_amp:
                # 记录 scale，判断这一步是否被跳过
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)   # 可能会跳过 optimizer.step
                scaler.update()
                did_step = (scaler.get_scale() == prev_scale)  # scale 没下降 => 没发生 inf/nan => step 执行了
            else:
                optimizer.step()

            if scheduler is not None and did_step:
                scheduler.step()

            total_train_loss += float(loss.item())
            n_train_batches += 1

            # 记录 lr（取第一个 param group）
            lr_hist.append(float(optimizer.param_groups[0]["lr"]))

        avg_train_loss = total_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # ---- val ----
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for y_batch, a_batch in val_loader:
                y_batch = y_batch.to(device, non_blocking=True).float()
                a_batch = a_batch.to(device, non_blocking=True).float()
                if device == "cuda":
                    with autocast("cuda", enabled=use_amp):
                        a_pred = model(y_batch)
                        loss = criterion(a_pred, a_batch)
                else:
                    a_pred = model(y_batch)
                    loss = criterion(a_pred, a_batch)
                total_val_loss += float(loss.item())
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)

        if verbose:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train_mlp] Epoch {epoch:03d}/{num_epochs:03d} "
                f"lr={cur_lr:.2e} train={avg_train_loss:.4e} val={avg_val_loss:.4e}"
            )

        # ---- best / early stop ----
        improved = (best_val_loss - avg_val_loss) > float(min_delta)
        if improved:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if early_stop and bad_epochs >= patience:
            if verbose:
                print(f"[train_mlp] Early stopping at epoch {epoch} (patience={patience}).")
            break

    # 恢复最佳参数
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info: Dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "lr_hist": lr_hist,
        "best_val_loss": best_val_loss,
        "num_epochs_requested": num_epochs,
        "num_epochs_ran": len(train_losses),
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "noise_sigma": noise_sigma,
        "mask_obs_dim": dataset.M,
        "r_eff": dataset.r_eff,
        "device": device,
        "use_onecycle": use_onecycle,
        "max_lr": max_lr,
        "amp": use_amp,
        "grad_clip_norm": grad_clip_norm,
        "early_stop": early_stop,
        "patience": patience,
        "min_delta": min_delta,
        "hidden_dims": hidden_dims,
        "activation": activation,
        "use_layernorm": use_layernorm,
        "dropout": dropout,
        "normalize_io": normalize_io,
    }

    if normalize_io and stats is not None:
        # 保存统计量，便于复现实验/推理
        info["norm_stats"] = {
            "y_mean": stats.y_mean.numpy(),
            "y_std": stats.y_std.numpy(),
            "a_mean": stats.a_mean.numpy(),
            "a_std": stats.a_std.numpy(),
        }

    return model, info
