# backend/train/mlp_trainer.py

"""
在 ObservationDataset 上训练 PodMLP 的工具函数。
"""

from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..dataio.observation_dataset import ObservationDataset
from ..models.mlp import PodMLP


def train_mlp_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
) -> Tuple[PodMLP, Dict[str, Any]]:
    """
    在给定的场数据 + POD 基底 + mask 上训练一个 MLP:

        y (masked noisy observation) -> a_true (POD 系数)

    参数
    ----
    X_flat_all: [N,D] 所有 snapshot 展平后的场。
    Ur_eff:     [D,r_eff] POD 基底（可能已截断）。
    mean_flat:  [D] 均值场。
    mask_flat:  [D] bool mask 向量。
    noise_sigma:
        训练时观测噪声标准差。
    batch_size, num_epochs, lr, weight_decay:
        训练超参数。
    val_ratio:
        验证集占比。
    device:
        "cuda" / "cpu"，若为 None 则自动检测。
    verbose:
        是否打印每个 epoch 的损失。

    返回
    ----
    model:
        训练好的 PodMLP（加载了 best val loss 权重）。
    info:
        包含训练曲线等信息的字典。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # 构建模型
    model = PodMLP(
        in_dim=dataset.M,
        out_dim=dataset.r_eff,
        hidden_dims=(256, 256),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state_dict = None
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for y_batch, a_batch in train_loader:
            y_batch = y_batch.to(device)
            a_batch = a_batch.to(device)

            optimizer.zero_grad()
            a_pred = model(y_batch)
            loss = criterion(a_pred, a_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = total_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for y_batch, a_batch in val_loader:
                y_batch = y_batch.to(device)
                a_batch = a_batch.to(device)
                a_pred = model(y_batch)
                loss = criterion(a_pred, a_batch)
                total_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = total_val_loss / max(1, n_val_batches)
        val_losses.append(avg_val_loss)

        if verbose:
            print(
                f"[train_mlp] Epoch {epoch:03d}/{num_epochs:03d} "
                f"train_loss={avg_train_loss:.4e}, val_loss={avg_val_loss:.4e}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最佳参数
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info: Dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "noise_sigma": noise_sigma,
        "mask_obs_dim": dataset.M,
        "r_eff": dataset.r_eff,
    }
    return model, info
