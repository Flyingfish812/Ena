from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..dataio.observation_dataset import ObservationDataset
from ..sampling.noise import add_gaussian_noise
from .mlp import (
    LatentGuidedBudgetExpertMLP,
    LatentGuidedBudgetLoss,
    PodMLP,
    ProgressiveModalResidualHead,
    ProgressiveStageLoss,
    SharedStemBudgetExpertMLP,
    SharedStemBudgetLoss,
    build_mlp_loss,
)
from .vcnn import VCNN, RelativeL2Loss, get_field_loss
from .vitae import build_vitae_loss, build_vitae_model


def _set_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed_value = int(seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def _split_train_val_dataset(
    dataset: Dataset,
    *,
    val_ratio: float,
    seed: int | None = None,
) -> tuple[Dataset, Dataset, int, int]:
    total = int(len(dataset))
    if total < 2:
        raise ValueError(f"Dataset must contain at least 2 samples, got {total}")

    n_val = min(max(1, int(round(float(total) * float(val_ratio)))), total - 1)
    n_train = total - n_val
    split_generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)
    return train_ds, val_ds, int(n_train), int(n_val)


def _subset_indices(subset: Dataset, fallback_size: int) -> np.ndarray:
    indices = getattr(subset, "indices", None)
    if indices is None:
        return np.arange(int(fallback_size), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _set_module_trainable(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _pmrh_monitor_from_metrics(metrics: Dict[str, torch.Tensor], active_stage: str) -> torch.Tensor:
    stage_name = str(active_stage).strip().lower()
    if stage_name in ("stage1", "coarse"):
        return metrics["loss_stage1"]
    return metrics.get("loss_total", metrics["loss_stage3"])


def _compute_channelwise_mean_std(x_thwc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_thwc, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected X_thwc [T,H,W,C], got {x.shape}")
    mean = np.mean(x, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.std(x, axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.asarray(1e-6, dtype=np.float32))
    return mean, std


def _normalize_field_hwc(x_hwc: np.ndarray, mean_c: np.ndarray, std_c: np.ndarray) -> np.ndarray:
    return ((np.asarray(x_hwc, dtype=np.float32) - mean_c[None, None, :]) / std_c[None, None, :]).astype(np.float32, copy=False)


def _denormalize_field_nchw(x_nchw: np.ndarray, mean_c: np.ndarray, std_c: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean_c, dtype=np.float32)[None, :, None, None]
    std = np.asarray(std_c, dtype=np.float32)[None, :, None, None]
    return (np.asarray(x_nchw, dtype=np.float32) * std + mean).astype(np.float32, copy=False)


def _build_optimizer_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    norm_modules = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if name.endswith("bias") or isinstance(module, norm_modules):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    progress: float,
    base_lr: float,
    min_lr: float,
    total_epochs: int,
    warmup_epochs: int,
) -> float:
    total = max(1, int(total_epochs))
    warmup = max(0, int(warmup_epochs))
    current = float(progress)
    if warmup > 0 and current < warmup:
        lr = float(base_lr) * current / float(warmup)
    else:
        cosine_total = max(1.0, float(total - warmup))
        cosine_progress = min(max((current - warmup) / cosine_total, 0.0), 1.0)
        lr = float(min_lr) + (float(base_lr) - float(min_lr)) * 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return float(lr)


def _resolve_torch_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    requested = str(device).strip().lower()
    if requested in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}' but CUDA is not available.")
    return str(device)


def _to_nchw(x_thwc: np.ndarray) -> np.ndarray:
    x = np.asarray(x_thwc, dtype=np.float32)
    return np.transpose(x, (0, 3, 1, 2))


def _to_thwc(x_nchw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_nchw, dtype=np.float32)
    return np.transpose(x, (0, 2, 3, 1))


def _compute_pad(width: int, patch_size: int) -> tuple[int, int]:
    if patch_size <= 1 or width % patch_size == 0:
        return (0, 0)
    total = ((width + patch_size - 1) // patch_size) * patch_size - width
    left = total // 2
    right = total - left
    return (left, right)


def _pad_chw(arr: np.ndarray, pad_hw: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    (top, bottom), (left, right) = pad_hw
    if top == bottom == left == right == 0:
        return arr.astype(np.float32, copy=False)
    return np.pad(arr, ((0, 0), (top, bottom), (left, right)), mode="constant").astype(np.float32, copy=False)


def _crop_chw(arr: np.ndarray, pad_hw: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    (top, bottom), (left, right) = pad_hw
    h_slice = slice(top, None if bottom == 0 else -bottom)
    w_slice = slice(left, None if right == 0 else -right)
    return arr[:, h_slice, w_slice].astype(np.float32, copy=False)


def _mask_to_channel(mask_hw: np.ndarray) -> np.ndarray:
    return np.asarray(mask_hw, dtype=np.float32)[None, :, :]


def _build_nearest_seed_index(mask_hw: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask_hw, dtype=bool)
    obs_coords = np.argwhere(mask)
    if obs_coords.size == 0:
        raise ValueError("mask_hw must contain at least one observed point.")
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    obs_y = obs_coords[:, 0].astype(np.float32)
    obs_x = obs_coords[:, 1].astype(np.float32)
    dist2 = (yy[:, :, None] - obs_y[None, None, :]) ** 2 + (xx[:, :, None] - obs_x[None, None, :]) ** 2
    return np.argmin(dist2, axis=2).astype(np.int32)


def _build_sparse_feature(x_hwc: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_hwc, dtype=np.float32)
    mask = np.asarray(mask_hw, dtype=np.float32)[:, :, None]
    return np.transpose(x * mask, (2, 0, 1)).astype(np.float32, copy=False)


def _build_voronoi_feature(x_hwc: np.ndarray, mask_hw: np.ndarray, nearest_idx_hw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_hwc, dtype=np.float32)
    mask = np.asarray(mask_hw, dtype=bool)
    obs_values = x[mask]
    filled = obs_values[nearest_idx_hw]
    return np.transpose(filled, (2, 0, 1)).astype(np.float32, copy=False)


def _resolve_spatial_representation(model_dataset_spec: Dict[str, Any], model_cfg: Dict[str, Any]) -> str:
    return str(
        model_cfg.get(
            "input_representation",
            model_dataset_spec.get("input_representation", model_dataset_spec.get("feature_builder", "voronoi_per_channel_plus_mask")),
        )
    ).strip().lower()


def _build_spatial_feature_single(
    x_hwc: np.ndarray,
    *,
    mask_hw: np.ndarray,
    nearest_idx_hw: np.ndarray,
    noise_sigma: float,
    representation: str,
    include_mask_channel: bool,
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean_c: np.ndarray | None = None,
    norm_std_c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_hwc, dtype=np.float32)
    observed = x[np.asarray(mask_hw, dtype=bool)]
    noisy = add_gaussian_noise(observed, sigma=float(noise_sigma)).astype(np.float32, copy=False)
    x_noisy = x.copy()
    x_noisy[np.asarray(mask_hw, dtype=bool)] = noisy

    if norm_mean_c is not None and norm_std_c is not None:
        x_feature_src = _normalize_field_hwc(x_noisy, norm_mean_c, norm_std_c)
        x_target_src = _normalize_field_hwc(x, norm_mean_c, norm_std_c)
    else:
        x_feature_src = x_noisy
        x_target_src = x

    if representation in ("voronoi_per_channel_plus_mask", "per_channel_voronoi_plus_mask", "voronoi"):
        feature = _build_voronoi_feature(x_feature_src, mask_hw, nearest_idx_hw)
    elif representation in ("sparse_per_channel_plus_mask", "masked_sparse_channels_plus_mask", "sparse"):
        feature = _build_sparse_feature(x_feature_src, mask_hw)
    else:
        raise ValueError(f"Unsupported spatial input_representation='{representation}'")

    target = np.transpose(x_target_src, (2, 0, 1)).astype(np.float32, copy=False)
    obs_mask = _mask_to_channel(mask_hw)
    if include_mask_channel:
        feature = np.concatenate([feature, obs_mask], axis=0)

    feature = _pad_chw(feature, pad_hw)
    target = _pad_chw(target, pad_hw)
    obs_mask = _pad_chw(obs_mask, pad_hw)
    return feature, target, obs_mask


class SpatialObservationDataset(Dataset):
    def __init__(
        self,
        X_thwc: np.ndarray,
        *,
        mask_hw: np.ndarray,
        noise_sigma: float,
        representation: str,
        include_mask_channel: bool,
        patch_size: int,
        normalize: bool = True,
        norm_mean_c: np.ndarray | None = None,
        norm_std_c: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        X_thwc = np.asarray(X_thwc, dtype=np.float32)
        if X_thwc.ndim != 4:
            raise ValueError(f"X_thwc must be [T,H,W,C], got {X_thwc.shape}")

        self.X_thwc = X_thwc
        self.mask_hw = np.asarray(mask_hw, dtype=bool)
        if self.mask_hw.shape != tuple(X_thwc.shape[1:3]):
            raise ValueError(f"mask_hw shape {self.mask_hw.shape} != field spatial shape {X_thwc.shape[1:3]}")

        self.noise_sigma = float(noise_sigma)
        self.representation = str(representation)
        self.include_mask_channel = bool(include_mask_channel)
        self.normalize = bool(normalize)
        h, w = int(X_thwc.shape[1]), int(X_thwc.shape[2])
        self.pad_hw = (_compute_pad(h, int(patch_size)), _compute_pad(w, int(patch_size)))
        self.nearest_idx_hw = _build_nearest_seed_index(self.mask_hw)
        if self.normalize:
            if norm_mean_c is None or norm_std_c is None:
                norm_mean_c, norm_std_c = _compute_channelwise_mean_std(X_thwc)
            self.norm_mean_c = np.asarray(norm_mean_c, dtype=np.float32)
            self.norm_std_c = np.asarray(norm_std_c, dtype=np.float32)
        else:
            self.norm_mean_c = None
            self.norm_std_c = None

    def __len__(self) -> int:
        return int(self.X_thwc.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature, target, obs_mask = _build_spatial_feature_single(
            self.X_thwc[int(idx)],
            mask_hw=self.mask_hw,
            nearest_idx_hw=self.nearest_idx_hw,
            noise_sigma=self.noise_sigma,
            representation=self.representation,
            include_mask_channel=self.include_mask_channel,
            pad_hw=self.pad_hw,
            norm_mean_c=self.norm_mean_c,
            norm_std_c=self.norm_std_c,
        )
        return (
            torch.from_numpy(feature.astype(np.float32, copy=True)),
            torch.from_numpy(target.astype(np.float32, copy=True)),
            torch.from_numpy(obs_mask.astype(np.float32, copy=True)),
        )


@dataclass
class _TrainLoopConfig:
    num_epochs: int
    batch_size: int
    val_ratio: float
    device: str | None
    lr: float
    weight_decay: float
    verbose: bool
    live_line: bool
    live_every: int
    conv_window: int
    conv_slope_thresh: float
    plot_loss: bool
    plot_path: str | Path | None
    early_stop: bool
    early_patience: int
    early_min_delta: float
    early_warmup: int
    min_lr: float = 0.0
    warmup_epochs: int = 0
    use_cosine_schedule: bool = False
    seed: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None


def _print_epoch_line(prefix: str, epoch: int, num_epochs: int, avg_train_loss: float, avg_val_monitor: float, *, verbose: bool, live_line: bool) -> None:
    msg = (
        f"[{prefix}] Epoch {epoch:03d}/{num_epochs:03d} "
        f"train_loss={avg_train_loss:.4e}, val_monitor={avg_val_monitor:.4e}"
    )
    if not verbose:
        return
    if live_line:
        print("\r" + msg, end="", flush=True)
    else:
        print(msg)


def _finalize_training_info(
    *,
    prefix: str,
    train_losses: list[float],
    val_losses: list[float],
    num_epochs: int,
    stopped_early: bool,
    stop_epoch: int | None,
    batch_size: int,
    lr: float,
    device: str,
    conv_window: int,
    conv_slope_thresh: float,
    plot_loss: bool,
    plot_path: str | Path | None,
    verbose: bool,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    slope_log10_val = None
    plateau_like = None
    w = int(max(5, conv_window))
    if len(val_losses) >= w:
        y = np.asarray(val_losses[-w:], dtype=float)
        y = np.maximum(y, 1e-30)
        x = np.arange(len(y), dtype=float)
        a, _ = np.polyfit(x, np.log10(y), deg=1)
        slope_log10_val = float(a)
        plateau_like = bool(slope_log10_val > float(conv_slope_thresh))
        if verbose:
            verdict = "PLATEAU-ish" if plateau_like else "still improving"
            print(f"[{prefix}] Convergence check (last {w} epochs): slope(log10(val))={slope_log10_val:.3e} -> {verdict}")

    saved_plot = None
    if plot_loss:
        try:
            import matplotlib.pyplot as plt

            if plot_path is None:
                plot_path = Path(f"{prefix}_train_loss.png")
            plot_path = Path(plot_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)

            fig = plt.figure()
            ax = plt.gca()
            ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train")
            ax.plot(np.arange(1, len(val_losses) + 1), val_losses, label="val")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (log scale)")
            ax.set_title(f"{prefix} training curve")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            saved_plot = str(plot_path)
        except Exception as e:
            if verbose:
                print(f"[{prefix}] Warning: failed to plot loss curve: {type(e).__name__}: {e}")

    info: Dict[str, Any] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(min(val_losses)) if val_losses else float("inf"),
        "num_epochs": int(num_epochs),
        "epochs_ran": int(stop_epoch) if stopped_early and stop_epoch is not None else int(num_epochs),
        "stopped_early": bool(stopped_early),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(device),
        "conv_window": int(w) if slope_log10_val is not None else None,
        "slope_log10_val": slope_log10_val,
        "plateau_like": plateau_like,
        "loss_plot_path": saved_plot,
    }
    if extra_info:
        info.update(extra_info)
    return info


def _fit_model(
    *,
    prefix: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_step_fn: Callable[[nn.Module, Any, str], torch.Tensor],
    val_step_fn: Callable[[nn.Module, Any, str], tuple[torch.Tensor, torch.Tensor]],
    cfg: _TrainLoopConfig,
    extra_info: Optional[Dict[str, Any]] = None,
) -> tuple[nn.Module, Dict[str, Any]]:
    device = _resolve_torch_device(cfg.device)
    model = model.to(device)

    best_val_monitor = float("inf")
    best_state_dict = None
    train_losses: list[float] = []
    val_monitor_losses: list[float] = []
    patience_ctr = 0
    stopped_early = False
    stop_epoch = None
    last_lr = float(cfg.lr)

    for epoch in range(1, int(cfg.num_epochs) + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        num_train_batches = max(1, len(train_loader))
        for batch_idx, batch in enumerate(train_loader):
            if cfg.max_train_batches is not None and batch_idx >= int(cfg.max_train_batches):
                break
            if cfg.use_cosine_schedule:
                progress = float(epoch - 1) + float(batch_idx) / float(num_train_batches)
                last_lr = _adjust_learning_rate(
                    optimizer,
                    progress=progress,
                    base_lr=float(cfg.lr),
                    min_lr=float(cfg.min_lr),
                    total_epochs=int(cfg.num_epochs),
                    warmup_epochs=int(cfg.warmup_epochs),
                )
            optimizer.zero_grad()
            loss = train_step_fn(model, batch, device)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
            n_train_batches += 1

        if not cfg.use_cosine_schedule:
            last_lr = float(optimizer.param_groups[0]["lr"])

        avg_train_loss = total_train_loss / max(1, n_train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_monitor = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if cfg.max_val_batches is not None and batch_idx >= int(cfg.max_val_batches):
                    break
                _, val_monitor = val_step_fn(model, batch, device)
                total_val_monitor += float(val_monitor.item())
                n_val_batches += 1
        avg_val_monitor = total_val_monitor / max(1, n_val_batches)
        val_monitor_losses.append(avg_val_monitor)

        if cfg.verbose and (epoch == 1 or epoch == cfg.num_epochs or (epoch % max(1, int(cfg.live_every)) == 0)):
            _print_epoch_line(prefix, epoch, int(cfg.num_epochs), avg_train_loss, avg_val_monitor, verbose=cfg.verbose, live_line=cfg.live_line)

        improved = (best_val_monitor - avg_val_monitor) > float(cfg.early_min_delta)
        if improved:
            best_val_monitor = avg_val_monitor
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        elif cfg.early_stop and epoch >= int(cfg.early_warmup):
            patience_ctr += 1
            if patience_ctr >= int(cfg.early_patience):
                stopped_early = True
                stop_epoch = epoch
                if cfg.verbose and cfg.live_line:
                    print("")
                break

    if cfg.verbose and cfg.live_line:
        print("")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info = _finalize_training_info(
        prefix=prefix,
        train_losses=train_losses,
        val_losses=val_monitor_losses,
        num_epochs=int(cfg.num_epochs),
        stopped_early=stopped_early,
        stop_epoch=stop_epoch,
        batch_size=int(cfg.batch_size),
        lr=float(cfg.lr),
        device=device,
        conv_window=int(cfg.conv_window),
        conv_slope_thresh=float(cfg.conv_slope_thresh),
        plot_loss=bool(cfg.plot_loss),
        plot_path=cfg.plot_path,
        verbose=bool(cfg.verbose),
        extra_info=extra_info,
    )
    info["last_lr"] = float(last_lr)
    info["min_lr"] = float(cfg.min_lr)
    info["warmup_epochs"] = int(cfg.warmup_epochs)
    info["use_cosine_schedule"] = bool(cfg.use_cosine_schedule)
    info["seed"] = None if cfg.seed is None else int(cfg.seed)
    return model, info


def train_mlp_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    coeff_loss_weights: np.ndarray | None = None,
    loss_weighting: str = "none",
    loss_weight_power: float = 1.0,
    hidden_dims: Sequence[int] = (256, 256),
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    centered_pod: bool = True,
    verbose: bool = True,
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    early_stop: bool = True,
    early_patience: int = 20,
    early_min_delta: float = 0.0,
    early_warmup: int = 5,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> Tuple[PodMLP, Dict[str, Any]]:
    seed_value = None if seed is None else int(seed)
    _set_global_seed(seed_value)

    dataset = ObservationDataset(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=noise_sigma,
        centered_pod=centered_pod,
    )
    N = len(dataset)

    train_ds, val_ds, n_train, n_val = _split_train_val_dataset(
        dataset,
        val_ratio=val_ratio,
        seed=seed_value,
    )

    if verbose:
        print(f"[train_mlp] Dataset size: N={N}, train={n_train}, val={n_val}")
        print(f"[train_mlp] Obs dim M={dataset.M}, coeff dim r={dataset.r_eff}, device={_resolve_torch_device(device)}")
        print(f"[train_mlp] loss_weighting={loss_weighting}, loss_weight_power={loss_weight_power}")

    loader_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = PodMLP(in_dim=dataset.M, out_dim=dataset.r_eff, hidden_dims=tuple(int(v) for v in hidden_dims))
    criterion, effective_coeff_weights = build_mlp_loss(
        coeff_loss_weights=coeff_loss_weights,
        r_eff=dataset.r_eff,
        loss_weighting=loss_weighting,
        loss_weight_power=loss_weight_power,
        device=_resolve_torch_device(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _train_step(model: nn.Module, batch: Any, device_name: str) -> torch.Tensor:
        y_batch, a_batch = batch
        y_batch = y_batch.to(device_name)
        a_batch = a_batch.to(device_name)
        a_pred = model(y_batch)
        return criterion(a_pred, a_batch)

    def _val_step(model: nn.Module, batch: Any, device_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        y_batch, a_batch = batch
        y_batch = y_batch.to(device_name)
        a_batch = a_batch.to(device_name)
        a_pred = model(y_batch)
        loss = criterion(a_pred, a_batch)
        return loss, loss

    model, info = _fit_model(
        prefix="train_mlp",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        train_step_fn=_train_step,
        val_step_fn=_val_step,
        cfg=_TrainLoopConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            live_line=live_line,
            live_every=live_every,
            conv_window=conv_window,
            conv_slope_thresh=conv_slope_thresh,
            plot_loss=plot_loss,
            plot_path=plot_path,
            early_stop=early_stop,
            early_patience=early_patience,
            early_min_delta=early_min_delta,
            early_warmup=early_warmup,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        ),
        extra_info={
            "noise_sigma": float(noise_sigma),
            "loss_weighting": str(loss_weighting),
            "loss_weight_power": float(loss_weight_power),
            "hidden_dims": [int(v) for v in hidden_dims],
            "centered_pod": bool(centered_pod),
            "mask_obs_dim": int(dataset.M),
            "r_eff": int(dataset.r_eff),
            "coeff_weight_sum": None if effective_coeff_weights is None else float(np.sum(effective_coeff_weights)),
            "coeff_weight_min": None if effective_coeff_weights is None else float(np.min(effective_coeff_weights)),
            "coeff_weight_max": None if effective_coeff_weights is None else float(np.max(effective_coeff_weights)),
            "seed": None if seed_value is None else int(seed_value),
        },
    )
    return model, info


def train_pmrh_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    coarse_hidden_dims: Sequence[int] | None = None,
    refinement_hidden_dims: Sequence[int] | None = None,
    stage2_feature_dim: int | None = None,
    stage2_head_hidden_dim: int | None = None,
    stage3_feature_dim: int | None = None,
    stage3_head_hidden_dim: int | None = None,
    trunk_hidden_dims: Sequence[int] | None = None,
    stage1_hidden_dim: int | None = None,
    stage2_hidden_dim: int | None = None,
    stage3_hidden_dim: int | None = None,
    group_ratios: Sequence[int] = (1, 2, 5),
    stage1_low_rank: int | None = None,
    stage_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
    consistency_weight: float = 0.05,
    budget_weight: float = 1e-4,
    coarse_epochs: int | None = None,
    expansion_epochs: int | None = None,
    joint_epochs: int | None = None,
    phase1_epochs: int = 40,
    phase2_epochs: int = 60,
    phase3_epochs: int = 80,
    finetune_epochs: int = 20,
    stage2_freeze_epochs: int = 3,
    stage3_freeze_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    centered_pod: bool = True,
    verbose: bool = True,
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    joint_coarse_lr_scale: float = 0.15,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> Tuple[ProgressiveModalResidualHead, Dict[str, Any]]:
    seed_value = None if seed is None else int(seed)
    _set_global_seed(seed_value)

    if coarse_hidden_dims is None:
        if trunk_hidden_dims is not None:
            coarse_hidden_dims = tuple(int(v) for v in trunk_hidden_dims)
        else:
            coarse_dim = 64 if stage1_hidden_dim is None else int(stage1_hidden_dim)
            coarse_hidden_dims = (coarse_dim, coarse_dim)
    resolved_coarse_hidden_dims = tuple(int(v) for v in coarse_hidden_dims)
    if refinement_hidden_dims is None:
        resolved_refinement_hidden_dims = (
            int(stage2_feature_dim if stage2_feature_dim is not None else (192 if stage2_hidden_dim is None else stage2_hidden_dim)),
            int(stage3_feature_dim if stage3_feature_dim is not None else (192 if stage3_hidden_dim is None else stage3_hidden_dim)),
        )
    else:
        resolved_refinement_hidden_dims = tuple(int(v) for v in refinement_hidden_dims)
    if len(resolved_refinement_hidden_dims) != 2:
        raise ValueError(
            f"refinement_hidden_dims must have length 2, got {resolved_refinement_hidden_dims}"
        )
    resolved_stage2_feature_dim = int(resolved_refinement_hidden_dims[0])
    resolved_stage2_head_hidden_dim = int(
        resolved_stage2_feature_dim if stage2_head_hidden_dim is None else stage2_head_hidden_dim
    )
    resolved_stage3_feature_dim = int(resolved_refinement_hidden_dims[1])
    resolved_stage3_head_hidden_dim = int(
        resolved_stage3_feature_dim if stage3_head_hidden_dim is None else stage3_head_hidden_dim
    )
    resolved_stage1_low_rank = None if stage1_low_rank is None else int(stage1_low_rank)
    if resolved_stage1_low_rank is not None and resolved_stage1_low_rank <= 0:
        resolved_stage1_low_rank = None

    resolved_coarse_epochs = int(phase1_epochs) if coarse_epochs is None else int(coarse_epochs)
    resolved_expansion_epochs = int(phase2_epochs) + int(phase3_epochs) if expansion_epochs is None else int(expansion_epochs)
    resolved_joint_epochs = int(finetune_epochs) if joint_epochs is None else int(joint_epochs)
    resolved_coarse_epochs = max(0, resolved_coarse_epochs)
    resolved_expansion_epochs = max(0, resolved_expansion_epochs)
    resolved_joint_epochs = max(0, resolved_joint_epochs)

    dataset = ObservationDataset(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=noise_sigma,
        centered_pod=centered_pod,
    )
    N = len(dataset)
    train_ds, val_ds, n_train, n_val = _split_train_val_dataset(
        dataset,
        val_ratio=val_ratio,
        seed=seed_value,
    )

    if verbose:
        print(f"[train_pmrh] Dataset size: N={N}, train={n_train}, val={n_val}")
        print(f"[train_pmrh] Obs dim M={dataset.M}, coeff dim r={dataset.r_eff}, device={_resolve_torch_device(device)}")
        print(
            f"[train_pmrh] coarse_hidden_dims={resolved_coarse_hidden_dims}, "
            f"refinement_hidden_dims={resolved_refinement_hidden_dims}, "
            f"stage1_low_rank={resolved_stage1_low_rank}"
        )

    loader_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    train_indices = _subset_indices(train_ds, fallback_size=n_train)
    coeff_train = np.asarray(dataset.a_true_all[train_indices], dtype=np.float32)
    coeff_mean = np.mean(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.std(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.maximum(coeff_std, np.asarray(1e-6, dtype=np.float32))

    device_name = _resolve_torch_device(device)
    model = ProgressiveModalResidualHead(
        in_dim=dataset.M,
        out_dim=dataset.r_eff,
        coarse_hidden_dims=resolved_coarse_hidden_dims,
        refinement_hidden_dims=resolved_refinement_hidden_dims,
        stage2_feature_dim=resolved_stage2_feature_dim,
        stage2_head_hidden_dim=resolved_stage2_head_hidden_dim,
        stage3_feature_dim=resolved_stage3_feature_dim,
        stage3_head_hidden_dim=resolved_stage3_head_hidden_dim,
        group_ratios=tuple(int(v) for v in group_ratios),
        stage1_low_rank=resolved_stage1_low_rank,
    ).to(device_name)
    criterion = ProgressiveStageLoss(
        group_spec=model.group_spec,
        coeff_mean=torch.from_numpy(coeff_mean).to(device_name),
        coeff_std=torch.from_numpy(coeff_std).to(device_name),
        stage_weights=tuple(float(v) for v in stage_loss_weights),
        consistency_weight=float(consistency_weight),
        budget_weight=float(budget_weight),
    ).to(device_name)

    phases = [
        {
            "name": "coarse_pretrain",
            "active_stage": "stage1",
            "epochs": int(resolved_coarse_epochs),
            "enabled": {"coarse": True, "refine": False, "full_head": False},
            "lr_scales": {"coarse": 1.0},
        },
        {
            "name": "full_warmup",
            "active_stage": "full",
            "epochs": int(resolved_expansion_epochs),
            "enabled": {"coarse": False, "refine": True, "full_head": True},
            "lr_scales": {"refine": 1.0, "full_head": 1.0},
        },
        {
            "name": "joint_finetune",
            "active_stage": "full",
            "epochs": int(resolved_joint_epochs),
            "enabled": {"coarse": True, "refine": True, "full_head": True},
            "lr_scales": {"coarse": float(joint_coarse_lr_scale), "refine": 1.0, "full_head": 1.0},
        },
    ]

    stage_modules: dict[str, tuple[nn.Module, ...]] = {
        "coarse": (model.coarse_input_proj, model.coarse_hidden_proj, model.coarse_head),
        "refine": (model.refine_input_proj, model.refine_hidden_from_coarse, model.refine_hidden_from_refine),
        "full_head": (model.full_head,),
    }

    def _configure_phase(phase: Dict[str, Any]) -> torch.optim.Optimizer:
        enabled = dict(phase["enabled"])
        for key, modules in stage_modules.items():
            stage_on = bool(enabled.get(key, False))
            for module in modules:
                _set_module_trainable(module, stage_on)

        param_groups: list[dict[str, Any]] = []
        for key, modules in stage_modules.items():
            lr_scale = float(phase["lr_scales"].get(key, 0.0))
            params = [param for module in modules for param in module.parameters() if param.requires_grad]
            if not params or lr_scale <= 0.0:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": float(lr) * lr_scale,
                    "weight_decay": float(weight_decay),
                }
            )
        if not param_groups:
            raise ValueError(f"No trainable parameters configured for PMRH {phase['name']}")
        return torch.optim.Adam(param_groups)

    total_epochs = int(sum(int(phase["epochs"]) for phase in phases))
    train_losses: list[float] = []
    val_losses: list[float] = []
    phase_history: list[Dict[str, Any]] = []
    best_state_dict = None
    best_full_monitor = float("inf")
    best_val_by_stage = {
        "stage1": float("inf"),
        "stage2": float("inf"),
        "full": float("inf"),
    }
    global_epoch = 0

    for phase in phases:
        phase_epochs = int(phase["epochs"])
        if phase_epochs <= 0:
            continue

        optimizer = _configure_phase(phase)
        active_stage = str(phase["active_stage"])
        if verbose:
            print(f"[train_pmrh] start {phase['name']} active_stage={active_stage} epochs={phase_epochs}")

        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                y_batch, a_batch = batch
                y_batch = y_batch.to(device_name)
                a_batch = a_batch.to(device_name)

                optimizer.zero_grad()
                outputs = model.forward_stages(y_batch, stage=active_stage)
                loss, _ = criterion.compute_components(outputs, a_batch, active_stage=active_stage)
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.item())
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)
            train_losses.append(avg_train_loss)

            model.eval()
            val_totals = {
                "loss_total": 0.0,
                "loss_stage1": 0.0,
                "loss_stage2": 0.0,
                "loss_stage3": 0.0,
                "loss_consistency": 0.0,
                "loss_budget": 0.0,
            }
            n_val_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if max_val_batches is not None and batch_idx >= int(max_val_batches):
                        break
                    y_batch, a_batch = batch
                    y_batch = y_batch.to(device_name)
                    a_batch = a_batch.to(device_name)
                    outputs = model.forward_stages(y_batch, stage=active_stage)
                    _, metrics = criterion.compute_components(outputs, a_batch, active_stage=active_stage)
                    for key in val_totals:
                        val_totals[key] += float(metrics[key].item())
                    n_val_batches += 1

            avg_val_metrics = {key: value / max(1, n_val_batches) for key, value in val_totals.items()}
            avg_val_monitor = float(_pmrh_monitor_from_metrics(
                {key: torch.tensor(value) for key, value in avg_val_metrics.items()},
                active_stage,
            ).item())
            val_losses.append(avg_val_monitor)
            if active_stage in ("stage1", "coarse"):
                best_val_by_stage["stage1"] = min(best_val_by_stage["stage1"], avg_val_metrics["loss_stage1"])
            else:
                best_val_by_stage["stage2"] = min(best_val_by_stage["stage2"], avg_val_metrics["loss_stage2"])
                best_val_by_stage["full"] = min(best_val_by_stage["full"], avg_val_metrics["loss_stage3"])

            phase_history.append(
                {
                    "epoch": int(global_epoch),
                    "phase": str(phase["name"]),
                    "phase_epoch": int(phase_epoch),
                    "active_stage": str(active_stage),
                    "train_loss": float(avg_train_loss),
                    "val_monitor": float(avg_val_monitor),
                    **{key: float(value) for key, value in avg_val_metrics.items()},
                }
            )

            if verbose and (phase_epoch == 1 or phase_epoch == phase_epochs or (phase_epoch % max(1, int(live_every)) == 0)):
                _print_epoch_line(
                    f"train_pmrh:{phase['name']}",
                    phase_epoch,
                    phase_epochs,
                    avg_train_loss,
                    avg_val_monitor,
                    verbose=verbose,
                    live_line=live_line,
                )

            if active_stage == "full" and avg_val_monitor < best_full_monitor:
                best_full_monitor = float(avg_val_monitor)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if verbose and live_line:
        print("")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info = _finalize_training_info(
        prefix="train_pmrh",
        train_losses=train_losses,
        val_losses=val_losses,
        num_epochs=total_epochs,
        stopped_early=False,
        stop_epoch=None,
        batch_size=int(batch_size),
        lr=float(lr),
        device=device_name,
        conv_window=int(conv_window),
        conv_slope_thresh=float(conv_slope_thresh),
        plot_loss=bool(plot_loss),
        plot_path=plot_path,
        verbose=bool(verbose),
        extra_info={
            "noise_sigma": float(noise_sigma),
            "centered_pod": bool(centered_pod),
            "mask_obs_dim": int(dataset.M),
            "r_eff": int(dataset.r_eff),
            "model_variant": "v3_nested_width_budget_progressive",
            "group_spec": model.group_spec.as_dict(),
            "coarse_hidden_dims": [int(v) for v in resolved_coarse_hidden_dims],
            "refinement_hidden_dims": [int(v) for v in resolved_refinement_hidden_dims],
            "stage2_feature_dim": int(resolved_stage2_feature_dim),
            "stage2_head_hidden_dim": int(resolved_stage2_head_hidden_dim),
            "stage3_feature_dim": int(resolved_stage3_feature_dim),
            "stage3_head_hidden_dim": int(resolved_stage3_head_hidden_dim),
            "stage1_low_rank": None if resolved_stage1_low_rank is None else int(resolved_stage1_low_rank),
            "group_ratios": [int(v) for v in group_ratios],
            "stage_loss_weights": [float(v) for v in stage_loss_weights],
            "prefix_loss_weights": [float(v) for v in criterion.prefix_weights.detach().cpu().tolist()],
            "consistency_weight": float(consistency_weight),
            "budget_weight": float(budget_weight),
            "coarse_epochs": int(resolved_coarse_epochs),
            "expansion_epochs": int(resolved_expansion_epochs),
            "joint_epochs": int(resolved_joint_epochs),
            "phase_schedule": [
                {
                    "name": str(phase["name"]),
                    "active_stage": str(phase["active_stage"]),
                    "epochs": int(phase["epochs"]),
                        "enabled": {str(k): bool(v) for k, v in dict(phase["enabled"]).items()},
                    "lr_scales": {str(k): float(v) for k, v in phase["lr_scales"].items()},
                }
                for phase in phases
            ],
            "phase_history": phase_history,
            "compat_stage2_freeze_epochs": int(stage2_freeze_epochs),
            "compat_stage3_freeze_epochs": int(stage3_freeze_epochs),
            "best_full_val_monitor": None if not np.isfinite(best_full_monitor) else float(best_full_monitor),
            "best_stage1_val_loss": None if not np.isfinite(best_val_by_stage["stage1"]) else float(best_val_by_stage["stage1"]),
            "best_stage2_val_loss": None if not np.isfinite(best_val_by_stage["stage2"]) else float(best_val_by_stage["stage2"]),
            "best_stage3_val_loss": None if not np.isfinite(best_val_by_stage["full"]) else float(best_val_by_stage["full"]),
            "seed": None if seed_value is None else int(seed_value),
        },
    )
    return model, info


def train_v4a_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    stem_dim: int = 48,
    stage1_hidden_dims: Sequence[int] = (64, 64),
    stage2_hidden_dims: Sequence[int] = (128, 128),
    stage3_hidden_dims: Sequence[int] = (256, 256),
    group_ratios: Sequence[int] = (1, 2, 5),
    stage_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
    stage1_epochs: int | None = None,
    stage2_warmup_epochs: int | None = None,
    stage2_tune_epochs: int | None = None,
    stage3_warmup_epochs: int | None = None,
    stage3_tune_epochs: int | None = None,
    joint_epochs: int | None = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    centered_pod: bool = True,
    verbose: bool = True,
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    stem_lr_scale: float = 0.2,
    joint_sampling_weights: Sequence[float] | None = None,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> Tuple[SharedStemBudgetExpertMLP, Dict[str, Any]]:
    seed_value = None if seed is None else int(seed)
    _set_global_seed(seed_value)

    dataset = ObservationDataset(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=noise_sigma,
        centered_pod=centered_pod,
    )
    N = len(dataset)
    train_ds, val_ds, n_train, n_val = _split_train_val_dataset(
        dataset,
        val_ratio=val_ratio,
        seed=seed_value,
    )

    if verbose:
        print(f"[train_v4a] Dataset size: N={N}, train={n_train}, val={n_val}")
        print(f"[train_v4a] Obs dim M={dataset.M}, coeff dim r={dataset.r_eff}, device={_resolve_torch_device(device)}")
        print(
            f"[train_v4a] stem_dim={int(stem_dim)}, stage1_hidden_dims={tuple(int(v) for v in stage1_hidden_dims)}, "
            f"stage2_hidden_dims={tuple(int(v) for v in stage2_hidden_dims)}, stage3_hidden_dims={tuple(int(v) for v in stage3_hidden_dims)}"
        )

    loader_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    train_indices = _subset_indices(train_ds, fallback_size=n_train)
    coeff_train = np.asarray(dataset.a_true_all[train_indices], dtype=np.float32)
    coeff_mean = np.mean(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.std(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.maximum(coeff_std, np.asarray(1e-6, dtype=np.float32))

    device_name = _resolve_torch_device(device)
    model = SharedStemBudgetExpertMLP(
        in_dim=dataset.M,
        out_dim=dataset.r_eff,
        stem_dim=int(stem_dim),
        stage1_hidden_dims=tuple(int(v) for v in stage1_hidden_dims),
        stage2_hidden_dims=tuple(int(v) for v in stage2_hidden_dims),
        stage3_hidden_dims=tuple(int(v) for v in stage3_hidden_dims),
        group_ratios=tuple(int(v) for v in group_ratios),
    ).to(device_name)
    criterion = SharedStemBudgetLoss(
        group_spec=model.group_spec,
        coeff_mean=torch.from_numpy(coeff_mean).to(device_name),
        coeff_std=torch.from_numpy(coeff_std).to(device_name),
        stage_weights=tuple(float(v) for v in stage_loss_weights),
    ).to(device_name)

    stage_modules: dict[str, tuple[nn.Module, ...]] = {
        "stem": (model.stem_linear,),
        "stage1": (model.stage1_branch,),
        "stage2": (model.stage2_branch,),
        "stage3": (model.stage3_branch,),
    }

    def _configure_phase(phase: Dict[str, Any]) -> torch.optim.Optimizer:
        enabled = dict(phase["enabled"])
        for key, modules in stage_modules.items():
            is_enabled = bool(enabled.get(key, False))
            for module in modules:
                _set_module_trainable(module, is_enabled)

        param_groups: list[dict[str, Any]] = []
        for key, modules in stage_modules.items():
            lr_scale = float(phase["lr_scales"].get(key, 0.0))
            params = [param for module in modules for param in module.parameters() if param.requires_grad]
            if not params or lr_scale <= 0.0:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": float(lr) * lr_scale,
                    "weight_decay": float(weight_decay),
                }
            )
        if not param_groups:
            raise ValueError(f"No trainable parameters configured for v4a phase {phase['name']}")
        return torch.optim.Adam(param_groups)

    def _resolve_joint_sampling_probs() -> np.ndarray:
        raw = stage_loss_weights if joint_sampling_weights is None else joint_sampling_weights
        arr = np.asarray(list(raw), dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"joint_sampling_weights must have length 3, got {tuple(raw)}")
        arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return (arr / total).astype(np.float64, copy=False)

    rng = np.random.default_rng(seed_value)
    sampling_probs = _resolve_joint_sampling_probs()
    phase_specs = [
        {
            "name": "stage1_pretrain",
            "active_stage": "stage1",
            "epochs": int(0 if stage1_epochs is None else stage1_epochs),
            "enabled": {"stem": True, "stage1": True, "stage2": False, "stage3": False},
            "lr_scales": {"stem": 1.0, "stage1": 1.0},
        },
        {
            "name": "stage2_expert_warmup",
            "active_stage": "stage2",
            "epochs": int(0 if stage2_warmup_epochs is None else stage2_warmup_epochs),
            "enabled": {"stem": False, "stage1": False, "stage2": True, "stage3": False},
            "lr_scales": {"stage2": 1.0},
        },
        {
            "name": "stage2_tune",
            "active_stage": "stage2",
            "epochs": int(0 if stage2_tune_epochs is None else stage2_tune_epochs),
            "enabled": {"stem": True, "stage1": False, "stage2": True, "stage3": False},
            "lr_scales": {"stem": float(stem_lr_scale), "stage2": 1.0},
        },
        {
            "name": "stage3_expert_warmup",
            "active_stage": "full",
            "epochs": int(0 if stage3_warmup_epochs is None else stage3_warmup_epochs),
            "enabled": {"stem": False, "stage1": False, "stage2": False, "stage3": True},
            "lr_scales": {"stage3": 1.0},
        },
        {
            "name": "stage3_tune",
            "active_stage": "full",
            "epochs": int(0 if stage3_tune_epochs is None else stage3_tune_epochs),
            "enabled": {"stem": True, "stage1": False, "stage2": False, "stage3": True},
            "lr_scales": {"stem": float(stem_lr_scale), "stage3": 1.0},
        },
        {
            "name": "joint_finetune",
            "active_stage": "joint",
            "epochs": int(0 if joint_epochs is None else joint_epochs),
            "enabled": {"stem": True, "stage1": True, "stage2": True, "stage3": True},
            "lr_scales": {"stem": float(stem_lr_scale), "stage1": 1.0, "stage2": 1.0, "stage3": 1.0},
        },
    ]
    phases = [phase for phase in phase_specs if int(phase["epochs"]) > 0]
    if len(phases) == 0:
        raise ValueError("v4a training requires at least one positive epoch phase")

    def _eval_single_stage(stage_name: str, y_batch: torch.Tensor, a_batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = model.forward_stages(y_batch, stage=stage_name)
        return criterion.compute_components(outputs, a_batch, active_stage=stage_name)

    total_epochs = int(sum(int(phase["epochs"]) for phase in phases))
    train_losses: list[float] = []
    val_losses: list[float] = []
    phase_history: list[Dict[str, Any]] = []
    best_state_dict = None
    best_monitor = float("inf")
    best_val_by_stage = {
        "stage1": float("inf"),
        "stage2": float("inf"),
        "full": float("inf"),
    }
    global_epoch = 0

    for phase in phases:
        phase_epochs = int(phase["epochs"])
        optimizer = _configure_phase(phase)
        active_stage = str(phase["active_stage"])
        if verbose:
            print(f"[train_v4a] start {phase['name']} active_stage={active_stage} epochs={phase_epochs}")

        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0
            sampled_stage_counts = {"stage1": 0, "stage2": 0, "full": 0}

            for batch_idx, batch in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                y_batch, a_batch = batch
                y_batch = y_batch.to(device_name)
                a_batch = a_batch.to(device_name)

                optimizer.zero_grad()
                if active_stage == "joint":
                    sampled_idx = int(rng.choice(3, p=sampling_probs))
                    sampled_stage = ("stage1", "stage2", "full")[sampled_idx]
                    sampled_stage_counts[sampled_stage] += 1
                    loss, _ = _eval_single_stage(sampled_stage, y_batch, a_batch)
                else:
                    sampled_stage = active_stage
                    sampled_stage_counts[sampled_stage] += 1
                    loss, _ = _eval_single_stage(sampled_stage, y_batch, a_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.item())
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)
            train_losses.append(avg_train_loss)

            model.eval()
            val_totals = {
                "loss_total": 0.0,
                "loss_stage1": 0.0,
                "loss_stage2": 0.0,
                "loss_stage3": 0.0,
                "loss_consistency": 0.0,
                "loss_budget": 0.0,
            }
            n_val_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if max_val_batches is not None and batch_idx >= int(max_val_batches):
                        break
                    y_batch, a_batch = batch
                    y_batch = y_batch.to(device_name)
                    a_batch = a_batch.to(device_name)

                    if active_stage == "joint":
                        _, metrics1 = _eval_single_stage("stage1", y_batch, a_batch)
                        _, metrics2 = _eval_single_stage("stage2", y_batch, a_batch)
                        _, metrics3 = _eval_single_stage("full", y_batch, a_batch)
                        loss_stage1 = metrics1["loss_stage1"]
                        loss_stage2 = metrics2["loss_stage2"]
                        loss_stage3 = metrics3["loss_stage3"]
                        loss_total = criterion.combine_weighted(loss_stage1, loss_stage2, loss_stage3)
                        val_totals["loss_total"] += float(loss_total.item())
                        val_totals["loss_stage1"] += float(loss_stage1.item())
                        val_totals["loss_stage2"] += float(loss_stage2.item())
                        val_totals["loss_stage3"] += float(loss_stage3.item())
                    else:
                        _, metrics = _eval_single_stage(active_stage, y_batch, a_batch)
                        for key in val_totals:
                            val_totals[key] += float(metrics[key].item())
                    n_val_batches += 1

            avg_val_metrics = {key: value / max(1, n_val_batches) for key, value in val_totals.items()}
            avg_val_monitor = float(_pmrh_monitor_from_metrics(
                {key: torch.tensor(value) for key, value in avg_val_metrics.items()},
                ("full" if active_stage == "joint" else active_stage),
            ).item())
            if active_stage == "joint":
                avg_val_monitor = float(avg_val_metrics["loss_total"])
            val_losses.append(avg_val_monitor)
            if active_stage == "joint":
                best_val_by_stage["stage1"] = min(best_val_by_stage["stage1"], avg_val_metrics["loss_stage1"])
                best_val_by_stage["stage2"] = min(best_val_by_stage["stage2"], avg_val_metrics["loss_stage2"])
                best_val_by_stage["full"] = min(best_val_by_stage["full"], avg_val_metrics["loss_stage3"])
            elif active_stage == "stage1":
                best_val_by_stage["stage1"] = min(best_val_by_stage["stage1"], avg_val_metrics["loss_stage1"])
            elif active_stage == "stage2":
                best_val_by_stage["stage2"] = min(best_val_by_stage["stage2"], avg_val_metrics["loss_stage2"])
            else:
                best_val_by_stage["full"] = min(best_val_by_stage["full"], avg_val_metrics["loss_stage3"])

            phase_record = {
                "epoch": int(global_epoch),
                "phase": str(phase["name"]),
                "phase_epoch": int(phase_epoch),
                "active_stage": str(active_stage),
                "train_loss": float(avg_train_loss),
                "val_monitor": float(avg_val_monitor),
                **{key: float(value) for key, value in avg_val_metrics.items()},
            }
            if active_stage == "joint":
                phase_record["joint_stage_counts"] = {key: int(value) for key, value in sampled_stage_counts.items()}
            phase_history.append(phase_record)

            if verbose and (phase_epoch == 1 or phase_epoch == phase_epochs or (phase_epoch % max(1, int(live_every)) == 0)):
                _print_epoch_line(
                    f"train_v4a:{phase['name']}",
                    phase_epoch,
                    phase_epochs,
                    avg_train_loss,
                    avg_val_monitor,
                    verbose=verbose,
                    live_line=live_line,
                )

            should_track_best = active_stage == "joint" or (active_stage == "full" and int(joint_epochs or 0) <= 0)
            if should_track_best and avg_val_monitor < best_monitor:
                best_monitor = float(avg_val_monitor)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if verbose and live_line:
        print("")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info = _finalize_training_info(
        prefix="train_v4a",
        train_losses=train_losses,
        val_losses=val_losses,
        num_epochs=total_epochs,
        stopped_early=False,
        stop_epoch=None,
        batch_size=int(batch_size),
        lr=float(lr),
        device=device_name,
        conv_window=int(conv_window),
        conv_slope_thresh=float(conv_slope_thresh),
        plot_loss=bool(plot_loss),
        plot_path=plot_path,
        verbose=bool(verbose),
        extra_info={
            "noise_sigma": float(noise_sigma),
            "centered_pod": bool(centered_pod),
            "mask_obs_dim": int(dataset.M),
            "r_eff": int(dataset.r_eff),
            "model_variant": "v4a_shared_stem_parallel_budget_experts",
            "group_spec": model.group_spec.as_dict(),
            "stem_dim": int(stem_dim),
            "stage1_hidden_dims": [int(v) for v in stage1_hidden_dims],
            "stage2_hidden_dims": [int(v) for v in stage2_hidden_dims],
            "stage3_hidden_dims": [int(v) for v in stage3_hidden_dims],
            "group_ratios": [int(v) for v in group_ratios],
            "stage_loss_weights": [float(v) for v in stage_loss_weights],
            "joint_sampling_weights": [float(v) for v in sampling_probs.tolist()],
            "stem_lr_scale": float(stem_lr_scale),
            "stage1_epochs": int(0 if stage1_epochs is None else stage1_epochs),
            "stage2_warmup_epochs": int(0 if stage2_warmup_epochs is None else stage2_warmup_epochs),
            "stage2_tune_epochs": int(0 if stage2_tune_epochs is None else stage2_tune_epochs),
            "stage3_warmup_epochs": int(0 if stage3_warmup_epochs is None else stage3_warmup_epochs),
            "stage3_tune_epochs": int(0 if stage3_tune_epochs is None else stage3_tune_epochs),
            "joint_epochs": int(0 if joint_epochs is None else joint_epochs),
            "phase_schedule": [
                {
                    "name": str(phase["name"]),
                    "active_stage": str(phase["active_stage"]),
                    "epochs": int(phase["epochs"]),
                    "enabled": {str(k): bool(v) for k, v in dict(phase["enabled"]).items()},
                    "lr_scales": {str(k): float(v) for k, v in phase["lr_scales"].items()},
                }
                for phase in phases
            ],
            "phase_history": phase_history,
            "best_joint_val_monitor": None if not np.isfinite(best_monitor) else float(best_monitor),
            "best_stage1_val_loss": None if not np.isfinite(best_val_by_stage["stage1"]) else float(best_val_by_stage["stage1"]),
            "best_stage2_val_loss": None if not np.isfinite(best_val_by_stage["stage2"]) else float(best_val_by_stage["stage2"]),
            "best_stage3_val_loss": None if not np.isfinite(best_val_by_stage["full"]) else float(best_val_by_stage["full"]),
            "seed": None if seed_value is None else int(seed_value),
        },
    )
    return model, info


def train_v4b_on_observations(
    X_flat_all: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_flat: np.ndarray,
    noise_sigma: float = 0.0,
    *,
    stem_dim: int = 48,
    stage1_hidden_dims: Sequence[int] = (64, 64),
    stage2_hidden_dims: Sequence[int] = (128, 128),
    stage3_hidden_dims: Sequence[int] = (256, 256),
    adapter16_dim: int | None = None,
    adapter48_dim: int | None = None,
    group_ratios: Sequence[int] = (1, 2, 5),
    stage_loss_weights: Sequence[float] = (1.0, 1.0, 1.0),
    adapter_reg_weights: Sequence[float] = (0.0, 0.0),
    stage1_epochs: int | None = None,
    stage2_warmup_epochs: int | None = None,
    stage2_tune_epochs: int | None = None,
    stage3_warmup_epochs: int | None = None,
    stage3_tune_epochs: int | None = None,
    joint_epochs: int | None = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    centered_pod: bool = True,
    verbose: bool = True,
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    stem_lr_scale: float = 0.2,
    joint_sampling_weights: Sequence[float] | None = None,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> Tuple[LatentGuidedBudgetExpertMLP, Dict[str, Any]]:
    seed_value = None if seed is None else int(seed)
    _set_global_seed(seed_value)

    dataset = ObservationDataset(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=noise_sigma,
        centered_pod=centered_pod,
    )
    N = len(dataset)
    train_ds, val_ds, n_train, n_val = _split_train_val_dataset(
        dataset,
        val_ratio=val_ratio,
        seed=seed_value,
    )

    if verbose:
        print(f"[train_v4b] Dataset size: N={N}, train={n_train}, val={n_val}")
        print(f"[train_v4b] Obs dim M={dataset.M}, coeff dim r={dataset.r_eff}, device={_resolve_torch_device(device)}")
        print(
            f"[train_v4b] stem_dim={int(stem_dim)}, stage1_hidden_dims={tuple(int(v) for v in stage1_hidden_dims)}, "
            f"stage2_hidden_dims={tuple(int(v) for v in stage2_hidden_dims)}, stage3_hidden_dims={tuple(int(v) for v in stage3_hidden_dims)}"
        )

    loader_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    train_indices = _subset_indices(train_ds, fallback_size=n_train)
    coeff_train = np.asarray(dataset.a_true_all[train_indices], dtype=np.float32)
    coeff_mean = np.mean(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.std(coeff_train, axis=0, dtype=np.float64).astype(np.float32)
    coeff_std = np.maximum(coeff_std, np.asarray(1e-6, dtype=np.float32))

    device_name = _resolve_torch_device(device)
    model = LatentGuidedBudgetExpertMLP(
        in_dim=dataset.M,
        out_dim=dataset.r_eff,
        stem_dim=int(stem_dim),
        stage1_hidden_dims=tuple(int(v) for v in stage1_hidden_dims),
        stage2_hidden_dims=tuple(int(v) for v in stage2_hidden_dims),
        stage3_hidden_dims=tuple(int(v) for v in stage3_hidden_dims),
        adapter16_dim=(None if adapter16_dim is None else int(adapter16_dim)),
        adapter48_dim=(None if adapter48_dim is None else int(adapter48_dim)),
        group_ratios=tuple(int(v) for v in group_ratios),
    ).to(device_name)
    criterion = LatentGuidedBudgetLoss(
        group_spec=model.group_spec,
        coeff_mean=torch.from_numpy(coeff_mean).to(device_name),
        coeff_std=torch.from_numpy(coeff_std).to(device_name),
        stage_weights=tuple(float(v) for v in stage_loss_weights),
        adapter_reg_weights=tuple(float(v) for v in adapter_reg_weights),
    ).to(device_name)

    stage_modules: dict[str, tuple[nn.Module, ...]] = {
        "stem": (model.stem_linear,),
        "stage1": (model.stage1_branch,),
        "adapter16": (model.adapter_16_to_48,),
        "stage2": (model.stage2_branch,),
        "adapter48": (model.adapter_48_to_128,),
        "stage3": (model.stage3_branch,),
    }

    def _configure_phase(phase: Dict[str, Any]) -> torch.optim.Optimizer:
        enabled = dict(phase["enabled"])
        for key, modules in stage_modules.items():
            is_enabled = bool(enabled.get(key, False))
            for module in modules:
                _set_module_trainable(module, is_enabled)

        param_groups: list[dict[str, Any]] = []
        for key, modules in stage_modules.items():
            lr_scale = float(phase["lr_scales"].get(key, 0.0))
            params = [param for module in modules for param in module.parameters() if param.requires_grad]
            if not params or lr_scale <= 0.0:
                continue
            param_groups.append(
                {
                    "params": params,
                    "lr": float(lr) * lr_scale,
                    "weight_decay": float(weight_decay),
                }
            )
        if not param_groups:
            raise ValueError(f"No trainable parameters configured for v4b phase {phase['name']}")
        return torch.optim.Adam(param_groups)

    def _resolve_joint_sampling_probs() -> np.ndarray:
        raw = stage_loss_weights if joint_sampling_weights is None else joint_sampling_weights
        arr = np.asarray(list(raw), dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"joint_sampling_weights must have length 3, got {tuple(raw)}")
        arr = np.maximum(arr, 0.0)
        total = float(arr.sum())
        if total <= 0.0 or not np.isfinite(total):
            return np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return (arr / total).astype(np.float64, copy=False)

    rng = np.random.default_rng(seed_value)
    sampling_probs = _resolve_joint_sampling_probs()
    phase_specs = [
        {
            "name": "stage1_pretrain",
            "active_stage": "stage1",
            "epochs": int(0 if stage1_epochs is None else stage1_epochs),
            "enabled": {"stem": True, "stage1": True, "adapter16": False, "stage2": False, "adapter48": False, "stage3": False},
            "lr_scales": {"stem": 1.0, "stage1": 1.0},
        },
        {
            "name": "stage2_adapter_warmup",
            "active_stage": "stage2",
            "epochs": int(0 if stage2_warmup_epochs is None else stage2_warmup_epochs),
            "enabled": {"stem": False, "stage1": False, "adapter16": True, "stage2": True, "adapter48": False, "stage3": False},
            "lr_scales": {"adapter16": 1.0, "stage2": 1.0},
        },
        {
            "name": "stage2_tune",
            "active_stage": "stage2",
            "epochs": int(0 if stage2_tune_epochs is None else stage2_tune_epochs),
            "enabled": {"stem": True, "stage1": False, "adapter16": True, "stage2": True, "adapter48": False, "stage3": False},
            "lr_scales": {"stem": float(stem_lr_scale), "adapter16": 1.0, "stage2": 1.0},
        },
        {
            "name": "stage3_adapter_warmup",
            "active_stage": "full",
            "epochs": int(0 if stage3_warmup_epochs is None else stage3_warmup_epochs),
            "enabled": {"stem": False, "stage1": False, "adapter16": False, "stage2": False, "adapter48": True, "stage3": True},
            "lr_scales": {"adapter48": 1.0, "stage3": 1.0},
        },
        {
            "name": "stage3_tune",
            "active_stage": "full",
            "epochs": int(0 if stage3_tune_epochs is None else stage3_tune_epochs),
            "enabled": {"stem": True, "stage1": False, "adapter16": False, "stage2": False, "adapter48": True, "stage3": True},
            "lr_scales": {"stem": float(stem_lr_scale), "adapter48": 1.0, "stage3": 1.0},
        },
        {
            "name": "joint_finetune",
            "active_stage": "joint",
            "epochs": int(0 if joint_epochs is None else joint_epochs),
            "enabled": {"stem": True, "stage1": True, "adapter16": True, "stage2": True, "adapter48": True, "stage3": True},
            "lr_scales": {"stem": float(stem_lr_scale), "stage1": 1.0, "adapter16": 1.0, "stage2": 1.0, "adapter48": 1.0, "stage3": 1.0},
        },
    ]
    phases = [phase for phase in phase_specs if int(phase["epochs"]) > 0]
    if len(phases) == 0:
        raise ValueError("v4b training requires at least one positive epoch phase")

    def _eval_single_stage(stage_name: str, y_batch: torch.Tensor, a_batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = model.forward_stages(y_batch, stage=stage_name)
        return criterion.compute_components(outputs, a_batch, active_stage=stage_name)

    total_epochs = int(sum(int(phase["epochs"]) for phase in phases))
    train_losses: list[float] = []
    val_losses: list[float] = []
    phase_history: list[Dict[str, Any]] = []
    best_state_dict = None
    best_monitor = float("inf")
    best_val_by_stage = {
        "stage1": float("inf"),
        "stage2": float("inf"),
        "full": float("inf"),
    }
    global_epoch = 0

    for phase in phases:
        phase_epochs = int(phase["epochs"])
        optimizer = _configure_phase(phase)
        active_stage = str(phase["active_stage"])
        if verbose:
            print(f"[train_v4b] start {phase['name']} active_stage={active_stage} epochs={phase_epochs}")

        for phase_epoch in range(1, phase_epochs + 1):
            global_epoch += 1
            model.train()
            total_train_loss = 0.0
            n_train_batches = 0
            sampled_stage_counts = {"stage1": 0, "stage2": 0, "full": 0}

            for batch_idx, batch in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= int(max_train_batches):
                    break
                y_batch, a_batch = batch
                y_batch = y_batch.to(device_name)
                a_batch = a_batch.to(device_name)

                optimizer.zero_grad()
                if active_stage == "joint":
                    sampled_idx = int(rng.choice(3, p=sampling_probs))
                    sampled_stage = ("stage1", "stage2", "full")[sampled_idx]
                    sampled_stage_counts[sampled_stage] += 1
                    loss, _ = _eval_single_stage(sampled_stage, y_batch, a_batch)
                else:
                    sampled_stage = active_stage
                    sampled_stage_counts[sampled_stage] += 1
                    loss, _ = _eval_single_stage(sampled_stage, y_batch, a_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += float(loss.item())
                n_train_batches += 1

            avg_train_loss = total_train_loss / max(1, n_train_batches)
            train_losses.append(avg_train_loss)

            model.eval()
            val_totals = {
                "loss_total": 0.0,
                "loss_stage1": 0.0,
                "loss_stage2": 0.0,
                "loss_stage3": 0.0,
                "loss_adapter_48": 0.0,
                "loss_adapter_128": 0.0,
                "loss_consistency": 0.0,
                "loss_budget": 0.0,
            }
            n_val_batches = 0
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if max_val_batches is not None and batch_idx >= int(max_val_batches):
                        break
                    y_batch, a_batch = batch
                    y_batch = y_batch.to(device_name)
                    a_batch = a_batch.to(device_name)

                    if active_stage == "joint":
                        total1, metrics1 = _eval_single_stage("stage1", y_batch, a_batch)
                        total2, metrics2 = _eval_single_stage("stage2", y_batch, a_batch)
                        total3, metrics3 = _eval_single_stage("full", y_batch, a_batch)
                        val_totals["loss_total"] += float((total1 + total2 + total3).item())
                        val_totals["loss_stage1"] += float(metrics1["loss_stage1"].item())
                        val_totals["loss_stage2"] += float(metrics2["loss_stage2"].item())
                        val_totals["loss_stage3"] += float(metrics3["loss_stage3"].item())
                        val_totals["loss_adapter_48"] += float(metrics2["loss_adapter_48"].item())
                        val_totals["loss_adapter_128"] += float(metrics3["loss_adapter_128"].item())
                    else:
                        _, metrics = _eval_single_stage(active_stage, y_batch, a_batch)
                        for key in val_totals:
                            val_totals[key] += float(metrics[key].item())
                    n_val_batches += 1

            avg_val_metrics = {key: value / max(1, n_val_batches) for key, value in val_totals.items()}
            avg_val_monitor = float(_pmrh_monitor_from_metrics(
                {key: torch.tensor(value) for key, value in avg_val_metrics.items()},
                ("full" if active_stage == "joint" else active_stage),
            ).item())
            if active_stage == "joint":
                avg_val_monitor = float(avg_val_metrics["loss_total"])
            val_losses.append(avg_val_monitor)
            if active_stage == "joint":
                best_val_by_stage["stage1"] = min(best_val_by_stage["stage1"], avg_val_metrics["loss_stage1"])
                best_val_by_stage["stage2"] = min(best_val_by_stage["stage2"], avg_val_metrics["loss_stage2"])
                best_val_by_stage["full"] = min(best_val_by_stage["full"], avg_val_metrics["loss_stage3"])
            elif active_stage == "stage1":
                best_val_by_stage["stage1"] = min(best_val_by_stage["stage1"], avg_val_metrics["loss_stage1"])
            elif active_stage == "stage2":
                best_val_by_stage["stage2"] = min(best_val_by_stage["stage2"], avg_val_metrics["loss_stage2"])
            else:
                best_val_by_stage["full"] = min(best_val_by_stage["full"], avg_val_metrics["loss_stage3"])

            phase_record = {
                "epoch": int(global_epoch),
                "phase": str(phase["name"]),
                "phase_epoch": int(phase_epoch),
                "active_stage": str(active_stage),
                "train_loss": float(avg_train_loss),
                "val_monitor": float(avg_val_monitor),
                **{key: float(value) for key, value in avg_val_metrics.items()},
            }
            if active_stage == "joint":
                phase_record["joint_stage_counts"] = {key: int(value) for key, value in sampled_stage_counts.items()}
            phase_history.append(phase_record)

            if verbose and (phase_epoch == 1 or phase_epoch == phase_epochs or (phase_epoch % max(1, int(live_every)) == 0)):
                _print_epoch_line(
                    f"train_v4b:{phase['name']}",
                    phase_epoch,
                    phase_epochs,
                    avg_train_loss,
                    avg_val_monitor,
                    verbose=verbose,
                    live_line=live_line,
                )

            should_track_best = active_stage == "joint" or (active_stage == "full" and int(joint_epochs or 0) <= 0)
            if should_track_best and avg_val_monitor < best_monitor:
                best_monitor = float(avg_val_monitor)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if verbose and live_line:
        print("")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    info = _finalize_training_info(
        prefix="train_v4b",
        train_losses=train_losses,
        val_losses=val_losses,
        num_epochs=total_epochs,
        stopped_early=False,
        stop_epoch=None,
        batch_size=int(batch_size),
        lr=float(lr),
        device=device_name,
        conv_window=int(conv_window),
        conv_slope_thresh=float(conv_slope_thresh),
        plot_loss=bool(plot_loss),
        plot_path=plot_path,
        verbose=bool(verbose),
        extra_info={
            "noise_sigma": float(noise_sigma),
            "centered_pod": bool(centered_pod),
            "mask_obs_dim": int(dataset.M),
            "r_eff": int(dataset.r_eff),
            "model_variant": "v4b_shared_stem_latent_guided_budget_experts",
            "group_spec": model.group_spec.as_dict(),
            "stem_dim": int(stem_dim),
            "stage1_hidden_dims": [int(v) for v in stage1_hidden_dims],
            "stage2_hidden_dims": [int(v) for v in stage2_hidden_dims],
            "stage3_hidden_dims": [int(v) for v in stage3_hidden_dims],
            "adapter16_dim": int(model.adapter16_dim),
            "adapter48_dim": int(model.adapter48_dim),
            "group_ratios": [int(v) for v in group_ratios],
            "stage_loss_weights": [float(v) for v in stage_loss_weights],
            "adapter_reg_weights": [float(v) for v in adapter_reg_weights],
            "joint_sampling_weights": [float(v) for v in sampling_probs.tolist()],
            "stem_lr_scale": float(stem_lr_scale),
            "stage1_epochs": int(0 if stage1_epochs is None else stage1_epochs),
            "stage2_warmup_epochs": int(0 if stage2_warmup_epochs is None else stage2_warmup_epochs),
            "stage2_tune_epochs": int(0 if stage2_tune_epochs is None else stage2_tune_epochs),
            "stage3_warmup_epochs": int(0 if stage3_warmup_epochs is None else stage3_warmup_epochs),
            "stage3_tune_epochs": int(0 if stage3_tune_epochs is None else stage3_tune_epochs),
            "joint_epochs": int(0 if joint_epochs is None else joint_epochs),
            "phase_schedule": [
                {
                    "name": str(phase["name"]),
                    "active_stage": str(phase["active_stage"]),
                    "epochs": int(phase["epochs"]),
                    "enabled": {str(k): bool(v) for k, v in dict(phase["enabled"]).items()},
                    "lr_scales": {str(k): float(v) for k, v in phase["lr_scales"].items()},
                }
                for phase in phases
            ],
            "phase_history": phase_history,
            "best_joint_val_monitor": None if not np.isfinite(best_monitor) else float(best_monitor),
            "best_stage1_val_loss": None if not np.isfinite(best_val_by_stage["stage1"]) else float(best_val_by_stage["stage1"]),
            "best_stage2_val_loss": None if not np.isfinite(best_val_by_stage["stage2"]) else float(best_val_by_stage["stage2"]),
            "best_stage3_val_loss": None if not np.isfinite(best_val_by_stage["full"]) else float(best_val_by_stage["full"]),
            "seed": None if seed_value is None else int(seed_value),
        },
    )
    return model, info


def train_field_model_on_observations(
    *,
    model_type: str,
    X_thwc: np.ndarray,
    mask_hw: np.ndarray,
    noise_sigma: float = 0.0,
    model_dataset_spec: Dict[str, Any] | None = None,
    model_cfg: Dict[str, Any] | None = None,
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_ratio: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
    live_line: bool = True,
    live_every: int = 1,
    conv_window: int = 25,
    conv_slope_thresh: float = -1e-3,
    plot_loss: bool = False,
    plot_path: str | Path | None = None,
    early_stop: bool = True,
    early_patience: int = 20,
    early_min_delta: float = 0.0,
    early_warmup: int = 5,
    min_lr: float = 0.0,
    warmup_epochs: int = 10,
    use_cosine_schedule: bool = True,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    model_dataset_spec = dict(model_dataset_spec or {})
    model_cfg = dict(model_cfg or {})

    X_thwc = np.asarray(X_thwc, dtype=np.float32)
    _, H, W, C = X_thwc.shape
    representation = _resolve_spatial_representation(model_dataset_spec, model_cfg)
    include_mask_channel = bool(model_cfg.get("include_mask_channel", model_dataset_spec.get("include_mask_channel", True)))
    patch_size = int(model_cfg.get("patch_size", 1 if str(model_type) == "vcnn" else 4))
    normalize = bool(model_cfg.get("normalize_mean_std", True))
    seed_value = None if seed is None else int(seed)

    _set_global_seed(seed_value)

    dataset = SpatialObservationDataset(
        X_thwc,
        mask_hw=np.asarray(mask_hw, dtype=bool),
        noise_sigma=float(noise_sigma),
        representation=representation,
        include_mask_channel=include_mask_channel,
        patch_size=patch_size,
        normalize=normalize,
    )
    N = len(dataset)
    n_val = max(1, int(N * val_ratio))
    n_train = max(1, N - n_val)
    if n_train + n_val > N:
        n_val = max(1, N - n_train)
    split_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)

    loader_generator = None if seed_value is None else torch.Generator().manual_seed(seed_value)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    sample_feature, sample_target, sample_mask = dataset[0]
    in_channels = int(sample_feature.shape[0])
    out_channels = int(sample_target.shape[0])
    padded_hw = (int(sample_target.shape[1]), int(sample_target.shape[2]))

    loss_type = str(model_cfg.get("loss_type", "mae"))
    obs_weight = float(model_cfg.get("obs_weight", 1.0))
    eval_loss = RelativeL2Loss()

    if str(model_type) == "vcnn":
        model = VCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=int(model_cfg.get("hidden_channels", 48)),
            num_layers=int(model_cfg.get("num_layers", 8)),
            kernel_size=int(model_cfg.get("kernel_size", 7)),
        )
        train_loss = get_field_loss(loss_type=loss_type, obs_weight=obs_weight)
        aux_weight = 0.0
    elif str(model_type) == "vitae":
        variant = str(model_cfg.get("variant", model_cfg.get("architecture", "base")))
        model = build_vitae_model(
            input_size=padded_hw,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            variant=variant,
        )
        train_loss = build_vitae_loss(loss_type=loss_type, obs_weight=obs_weight)
        aux_weight = float(model_cfg.get("aux_loss_weight", 0.2))
    else:
        raise ValueError(f"Unsupported spatial model_type='{model_type}'")

    optimizer_name = str(model_cfg.get("optimizer", "adamw")).strip().lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            _build_optimizer_param_groups(model, weight_decay=float(weight_decay)),
            lr=lr,
            betas=(0.9, 0.95),
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer='{optimizer_name}' for spatial model branch")

    def _train_step(model: nn.Module, batch: Any, device_name: str) -> torch.Tensor:
        feature, target, obs_mask = batch
        feature = feature.to(device_name)
        target = target.to(device_name)
        obs_mask = obs_mask.to(device_name)

        output = model(feature)
        if isinstance(output, tuple):
            pred, pred_aux = output
            loss = (1.0 - aux_weight) * train_loss(target, pred, obs_mask) + aux_weight * train_loss(target, pred_aux, obs_mask)
            return loss
        return train_loss(target, output, obs_mask)

    def _val_step(model: nn.Module, batch: Any, device_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        feature, target, obs_mask = batch
        feature = feature.to(device_name)
        target = target.to(device_name)
        obs_mask = obs_mask.to(device_name)

        output = model(feature)
        if isinstance(output, tuple):
            pred, pred_aux = output
            loss = (1.0 - aux_weight) * train_loss(target, pred, obs_mask) + aux_weight * train_loss(target, pred_aux, obs_mask)
            pred_eval = pred
            target_eval = target
            if dataset.norm_mean_c is not None and dataset.norm_std_c is not None:
                mean = torch.as_tensor(dataset.norm_mean_c, dtype=pred.dtype, device=pred.device)[None, :, None, None]
                std = torch.as_tensor(dataset.norm_std_c, dtype=pred.dtype, device=pred.device)[None, :, None, None]
                pred_eval = pred * std + mean
                target_eval = target * std + mean
            monitor = eval_loss(target_eval, pred_eval, None)
            return loss, monitor
        loss = train_loss(target, output, obs_mask)
        pred_eval = output
        target_eval = target
        if dataset.norm_mean_c is not None and dataset.norm_std_c is not None:
            mean = torch.as_tensor(dataset.norm_mean_c, dtype=output.dtype, device=output.device)[None, :, None, None]
            std = torch.as_tensor(dataset.norm_std_c, dtype=output.dtype, device=output.device)[None, :, None, None]
            pred_eval = output * std + mean
            target_eval = target * std + mean
        monitor = eval_loss(target_eval, pred_eval, None)
        return loss, monitor

    model, info = _fit_model(
        prefix=f"train_{model_type}",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        train_step_fn=_train_step,
        val_step_fn=_val_step,
        cfg=_TrainLoopConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            val_ratio=val_ratio,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
            live_line=live_line,
            live_every=live_every,
            conv_window=conv_window,
            conv_slope_thresh=conv_slope_thresh,
            plot_loss=plot_loss,
            plot_path=plot_path,
            early_stop=early_stop,
            early_patience=early_patience,
            early_min_delta=early_min_delta,
            early_warmup=early_warmup,
            min_lr=min_lr,
            warmup_epochs=warmup_epochs,
            use_cosine_schedule=use_cosine_schedule,
            seed=seed_value,
            max_train_batches=max_train_batches,
            max_val_batches=max_val_batches,
        ),
        extra_info={
            "noise_sigma": float(noise_sigma),
            "loss_type": loss_type,
            "obs_weight": float(obs_weight),
            "input_representation": representation,
            "include_mask_channel": bool(include_mask_channel),
            "patch_size": int(patch_size),
            "in_channels": int(in_channels),
            "out_channels": int(out_channels),
            "padded_hw": [int(v) for v in padded_hw],
            "normalize_mean_std": bool(normalize),
            "norm_mean": None if dataset.norm_mean_c is None else [float(v) for v in dataset.norm_mean_c],
            "norm_std": None if dataset.norm_std_c is None else [float(v) for v in dataset.norm_std_c],
            "optimizer": optimizer_name,
        },
    )

    artifacts = {
        "representation": representation,
        "include_mask_channel": include_mask_channel,
        "patch_size": int(patch_size),
        "pad_hw": dataset.pad_hw,
        "nearest_idx_hw": dataset.nearest_idx_hw,
        "input_size": padded_hw,
        "in_channels": int(in_channels),
        "out_channels": int(out_channels),
        "norm_mean_c": dataset.norm_mean_c,
        "norm_std_c": dataset.norm_std_c,
    }
    return model, info, artifacts


def predict_field_model_batch(
    *,
    model_type: str,
    model: nn.Module,
    X_thwc: np.ndarray,
    mask_hw: np.ndarray,
    nearest_idx_hw: np.ndarray,
    representation: str,
    include_mask_channel: bool,
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    noise_sigma: float,
    device: str | None,
    batch_size: int = 16,
    norm_mean_c: np.ndarray | None = None,
    norm_std_c: np.ndarray | None = None,
) -> np.ndarray:
    device_name = _resolve_torch_device(device)
    model = model.to(device_name)
    model.eval()

    features: list[np.ndarray] = []
    for x_hwc in np.asarray(X_thwc, dtype=np.float32):
        feature, _, _ = _build_spatial_feature_single(
            x_hwc,
            mask_hw=np.asarray(mask_hw, dtype=bool),
            nearest_idx_hw=np.asarray(nearest_idx_hw, dtype=np.int32),
            noise_sigma=float(noise_sigma),
            representation=str(representation),
            include_mask_channel=bool(include_mask_channel),
            pad_hw=pad_hw,
            norm_mean_c=norm_mean_c,
            norm_std_c=norm_std_c,
        )
        features.append(feature)

    x_in = np.asarray(features, dtype=np.float32)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x_in.shape[0], int(batch_size)):
            chunk = torch.from_numpy(x_in[start : start + int(batch_size)]).to(device_name)
            output = model(chunk)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.detach().cpu().numpy())

    pred_nchw = np.concatenate(preds, axis=0)
    pred_cropped = np.stack([_crop_chw(pred_nchw[i], pad_hw) for i in range(pred_nchw.shape[0])], axis=0)
    if norm_mean_c is not None and norm_std_c is not None:
        pred_cropped = _denormalize_field_nchw(pred_cropped, norm_mean_c, norm_std_c)
    return _to_thwc(pred_cropped)