from typing import Dict, Any, Tuple, Optional
from pathlib import Path

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
    # --- NEW: tiny monitoring helpers (opt-in, default keeps old behavior) ---
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    # --- NEW: early stopping (simple & optional) ---
    early_stop: bool = True,            # 开关：默认开（你想要的大 max epoch 场景更友好）
    early_patience: int = 20,           # 连续多少个 epoch 没明显变好就停
    early_min_delta: float = 0.0,       # 至少改善多少才算“变好”（MSE 场景可先用 0 或 1e-6）
    early_warmup: int = 5,              # 前几个 epoch 不做 early stop（避免刚开始抖动误停）
) -> Tuple[PodMLP, Dict[str, Any]]:
    """
    在给定的场数据 + POD 基底 + mask 上训练一个 MLP:
        y (masked noisy observation) -> a_true (POD 系数)

    返回:
      - model: best val loss 权重
      - info: 训练曲线 + 收敛诊断
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # ---- NEW: early stopping state ----
    patience_ctr = 0
    stopped_early = False
    stop_epoch = None

    def _print_epoch_line(epoch: int, avg_train_loss: float, avg_val_loss: float) -> None:
        msg = (
            f"[train_mlp] Epoch {epoch:03d}/{num_epochs:03d} "
            f"train_loss={avg_train_loss:.4e}, val_loss={avg_val_loss:.4e}"
        )
        if not verbose:
            return
        if live_line:
            print("\r" + msg, end="", flush=True)
        else:
            print(msg)

    for epoch in range(1, num_epochs + 1):
        # ---- train ----
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

        # ---- val ----
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

        # print
        if verbose and (epoch == 1 or epoch == num_epochs or (epoch % max(1, int(live_every)) == 0)):
            _print_epoch_line(epoch, avg_train_loss, avg_val_loss)

        # best checkpoint
        improved = (best_val_loss - avg_val_loss) > float(early_min_delta)
        if improved:
            best_val_loss = avg_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            # ---- NEW: early stopping update ----
            if early_stop and epoch >= int(early_warmup):
                patience_ctr += 1
                if patience_ctr >= int(early_patience):
                    stopped_early = True
                    stop_epoch = epoch
                    if verbose:
                        # 换行避免 live_line 把信息挤在一行尾巴
                        if live_line:
                            print("")
                        print(
                            f"[train_mlp] Early stopping at epoch {epoch} "
                            f"(no val improvement > {early_min_delta:g} for {early_patience} epochs)."
                        )
                    break

    # ensure newline after live-line
    if verbose and live_line:
        print("")

    # restore best
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # ---- convergence diagnostic: slope on log10(val_loss) ----
    slope_log10_val = None
    plateau_like = None
    w = int(max(5, conv_window))
    if len(val_losses) >= w:
        y = np.asarray(val_losses[-w:], dtype=float)
        y = np.maximum(y, 1e-30)
        x = np.arange(len(y), dtype=float)
        a, b = np.polyfit(x, np.log10(y), deg=1)
        slope_log10_val = float(a)
        plateau_like = bool(slope_log10_val > float(conv_slope_thresh))

    if verbose and slope_log10_val is not None:
        verdict = "PLATEAU-ish" if plateau_like else "still improving"
        print(
            f"[train_mlp] Convergence check (last {w} epochs): "
            f"slope(log10(val_loss))={slope_log10_val:.3e} -> {verdict}"
        )

    # ---- optional plot ----
    saved_plot = None
    if plot_loss:
        try:
            import matplotlib.pyplot as plt

            if plot_path is None:
                plot_path = Path("mlp_train_loss.png")
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            fig = plt.figure()
            ax = plt.gca()
            ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
            ax.plot(np.arange(1, len(val_losses) + 1), val_losses, label="val")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MSE loss (log scale)")
            ax.set_title("MLP training curve")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            saved_plot = str(plot_path)
            if verbose:
                print(f"[train_mlp] Saved loss curve: {saved_plot}")
        except Exception as e:
            if verbose:
                print(f"[train_mlp] Warning: failed to plot loss curve: {type(e).__name__}: {e}")

    info: Dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "num_epochs": int(num_epochs),
        "epochs_ran": int(stop_epoch) if stopped_early else int(num_epochs),
        "stopped_early": bool(stopped_early),
        "early_patience": int(early_patience) if early_stop else None,
        "early_min_delta": float(early_min_delta) if early_stop else None,
        "early_warmup": int(early_warmup) if early_stop else None,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "noise_sigma": float(noise_sigma),
        "mask_obs_dim": int(dataset.M),
        "r_eff": int(dataset.r_eff),
        "conv_window": int(w) if slope_log10_val is not None else None,
        "slope_log10_val": slope_log10_val,
        "plateau_like": plateau_like,
        "loss_plot_path": saved_plot,
    }
    return model, info
