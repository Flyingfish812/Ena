from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..dataio.io_utils import ensure_dir, save_json
from ..metrics.cumulate_metrics import compute_nrmse_vs_r_coeff
from ..pod.project import project_to_pod
from ..sampling.noise import add_gaussian_noise
from ..viz.cumulate_plots import plot_nrmse_family_vs_r_curves
from .vcnn import RelativeL2Loss, VCNNMultiScale, get_field_loss


PREFIX_EVAL_STEPS: tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 128)
EXIT_LEVELS: tuple[int, ...] = (1, 2, 3)
DEFAULT_OUTPUT_MODE = "field"
DEFAULT_COEFF_LOSS_WEIGHTS: tuple[float, float, float] = (1.0, 1.0, 1.0)


CSV_FIELDNAMES: tuple[str, ...] = (
    "exp_id",
    "epoch",
    *tuple(f"E{prefix_dim}_exit{exit_idx}" for exit_idx in EXIT_LEVELS for prefix_dim in PREFIX_EVAL_STEPS),
    "latency_exit1",
    "latency_exit2",
    "latency_exit3",
    "param_count",
)


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


def _rescale_centered_field_nchw(x_nchw: np.ndarray, std_c: np.ndarray | None) -> np.ndarray:
    if std_c is None:
        return np.asarray(x_nchw, dtype=np.float32)
    std = np.asarray(std_c, dtype=np.float32)[None, :, None, None]
    return (np.asarray(x_nchw, dtype=np.float32) * std).astype(np.float32, copy=False)


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


def _crop_nchw_tensor(x: torch.Tensor, pad_hw: tuple[tuple[int, int], tuple[int, int]]) -> torch.Tensor:
    (top, bottom), (left, right) = pad_hw
    h_stop = None if bottom == 0 else -bottom
    w_stop = None if right == 0 else -right
    return x[:, :, top:h_stop, left:w_stop]


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
    noise_seed: int | None = None,
) -> np.ndarray:
    x = np.asarray(x_hwc, dtype=np.float32)
    mask_bool = np.asarray(mask_hw, dtype=bool)
    observed = x[mask_bool]
    noisy = add_gaussian_noise(observed, sigma=float(noise_sigma), seed=noise_seed).astype(np.float32, copy=False)
    x_noisy = x.copy()
    x_noisy[mask_bool] = noisy

    if norm_mean_c is not None and norm_std_c is not None:
        x_feature_src = _normalize_field_hwc(x_noisy, norm_mean_c, norm_std_c)
    else:
        x_feature_src = x_noisy

    rep = str(representation).strip().lower()
    if rep in ("voronoi_per_channel_plus_mask", "per_channel_voronoi_plus_mask", "voronoi"):
        feature = _build_voronoi_feature(x_feature_src, mask_bool, nearest_idx_hw)
    elif rep in ("sparse_per_channel_plus_mask", "masked_sparse_channels_plus_mask", "sparse"):
        feature = _build_sparse_feature(x_feature_src, mask_bool)
    else:
        raise ValueError(f"Unsupported spatial input_representation='{representation}'")

    if include_mask_channel:
        feature = np.concatenate([feature, _mask_to_channel(mask_bool)], axis=0)

    return _pad_chw(feature, pad_hw)


def _resolve_torch_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested = str(device).strip().lower()
    if requested in ("", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"Requested device '{device}' but CUDA is not available.")
    return str(device)


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


def _maybe_sync(device_name: str | torch.device) -> None:
    if str(device_name).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def _to_field_nchw(flat_batch: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
    h, w, c = [int(v) for v in spatial_shape]
    return flat_batch.view(-1, h, w, c).permute(0, 3, 1, 2).contiguous()


def _prefix_relative_error(a_pred: np.ndarray, a_true: np.ndarray, prefix_dim: int) -> float:
    k = max(1, int(prefix_dim))
    diff = np.asarray(a_pred[:, :k], dtype=np.float64) - np.asarray(a_true[:, :k], dtype=np.float64)
    denom = np.sum(np.asarray(a_true[:, :k], dtype=np.float64) ** 2, axis=1)
    denom = np.maximum(denom, 1e-12)
    return float(np.mean(np.sum(diff ** 2, axis=1) / denom))


def _resolve_prefix_steps(max_dim: int, prefix_steps: Sequence[int] | None = None) -> tuple[int, ...]:
    raw_steps = PREFIX_EVAL_STEPS if prefix_steps is None else tuple(int(v) for v in prefix_steps)
    out: list[int] = []
    cap = max(1, int(max_dim))
    for step in raw_steps:
        step_eff = min(max(1, int(step)), cap)
        if step_eff not in out:
            out.append(step_eff)
    if cap not in out:
        out.append(cap)
    return tuple(out)


def _subset_indices(subset: Dataset, fallback_size: int) -> np.ndarray:
    indices = getattr(subset, "indices", None)
    if indices is None:
        return np.arange(int(fallback_size), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _flatten_field_nchw(field_nchw: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(field_nchw, dtype=np.float32), (0, 2, 3, 1)).reshape(field_nchw.shape[0], -1)


def _to_numpy_image(field_hwc: np.ndarray) -> np.ndarray:
    field = np.asarray(field_hwc, dtype=np.float32)
    if field.ndim != 3:
        raise ValueError(f"Expected field [H,W,C], got {field.shape}")
    if field.shape[2] == 1:
        return field[:, :, 0]
    return np.sqrt(np.sum(field ** 2, axis=2, dtype=np.float64)).astype(np.float32)


def _field_stats(field_nchw: np.ndarray) -> dict[str, float]:
    flat = _flatten_field_nchw(np.asarray(field_nchw, dtype=np.float32))
    l2 = np.linalg.norm(flat, axis=1)
    abs_max = np.max(np.abs(flat), axis=1)
    return {
        "l2_mean": float(np.mean(l2)),
        "l2_std": float(np.std(l2)),
        "l2_min": float(np.min(l2)),
        "l2_max": float(np.max(l2)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "abs_max_mean": float(np.mean(abs_max)),
        "near_zero_l2_frac": float(np.mean(l2 <= 1e-8)),
        "near_zero_absmax_frac": float(np.mean(abs_max <= 1e-6)),
    }


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    return value


def _pad_nchw_tensor(x: torch.Tensor, pad_hw: tuple[tuple[int, int], tuple[int, int]]) -> torch.Tensor:
    (top, bottom), (left, right) = pad_hw
    if top == bottom == left == right == 0:
        return x
    return F.pad(x, (left, right, top, bottom))


def _set_module_trainable(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _build_trainable_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError("No trainable parameters are enabled for the current stage.")
    optimizer_key = str(optimizer_name).strip().lower()
    if optimizer_key == "adamw":
        return torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    if optimizer_key == "adam":
        return torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))
    raise ValueError(f"Unsupported optimizer='{optimizer_name}'")


def _relative_coeff_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    num = torch.sum(diff ** 2, dim=1)
    den = torch.sum(target ** 2, dim=1).clamp_min(1e-12)
    return torch.mean(torch.sqrt(num / den))


def _compute_coeff_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    name = str(loss_type or "mse").strip().lower()
    if name == "mae":
        return torch.mean(torch.abs(pred - target))
    if name == "l2norm":
        return _relative_coeff_error(pred, target)
    return torch.mean((pred - target) ** 2)


def _build_obs_mask_tensor(
    mask_hw: np.ndarray,
    *,
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    device: str | torch.device,
) -> torch.Tensor:
    mask = torch.as_tensor(np.asarray(mask_hw, dtype=np.float32)[None, None, :, :], device=device)
    return _pad_nchw_tensor(mask, pad_hw)


def _build_full_field_target(
    coeffs: torch.Tensor,
    *,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
) -> torch.Tensor:
    rank_eff = min(int(coeffs.shape[1]), int(basis.shape[1]))
    recon_flat = coeffs[:, :rank_eff] @ basis[:, :rank_eff].T + mean_flat[None, :]
    target = _to_field_nchw(recon_flat, spatial_shape)
    if norm_mean is not None and norm_std is not None:
        target = (target - norm_mean[None, :, None, None]) / norm_std[None, :, None, None]
    return _pad_nchw_tensor(target, pad_hw)


def _configure_two_stage_training(model: VCNNMultiScale, stage_name: str) -> tuple[int, ...]:
    stage = str(stage_name).strip().lower()
    if stage == "stage1":
        _set_module_trainable(model.hidden_blocks, True)
        _set_module_trainable(model.final_head, True)
        _set_module_trainable(model.exit1_head, False)
        _set_module_trainable(model.exit2_head, False)
        return (3,)
    if stage == "stage2":
        _set_module_trainable(model.hidden_blocks, False)
        _set_module_trainable(model.final_head, False)
        _set_module_trainable(model.exit1_head, True)
        _set_module_trainable(model.exit2_head, True)
        return (1, 2)
    raise ValueError(f"Unsupported stage_name='{stage_name}'")


def _compute_stage_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    coeffs: torch.Tensor,
    *,
    active_exits: Sequence[int],
    output_mode: str,
    loss_type: str,
    obs_mask: torch.Tensor | None,
    field_loss_fn: nn.Module | None,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    mode = _resolve_output_mode(output_mode)
    losses: dict[str, float] = {}
    total = coeffs.new_tensor(0.0)
    if mode == "coeff":
        target = coeffs
        for exit_level in active_exits:
            pred = outputs[int(exit_level) - 1]
            loss = _compute_coeff_loss(pred, target, loss_type)
            total = total + loss
            losses[f"exit{int(exit_level)}_loss"] = float(loss.detach().item())
        losses["loss_total"] = float(total.detach().item())
        return total, losses

    if field_loss_fn is None:
        raise ValueError("field_loss_fn is required for field output mode")
    target_field = _build_full_field_target(
        coeffs,
        basis=basis,
        mean_flat=mean_flat,
        spatial_shape=spatial_shape,
        norm_mean=norm_mean,
        norm_std=norm_std,
        pad_hw=pad_hw,
    )
    for exit_level in active_exits:
        pred = outputs[int(exit_level) - 1]
        loss = field_loss_fn(target_field, pred, obs_mask)
        total = total + loss
        losses[f"exit{int(exit_level)}_loss"] = float(loss.detach().item())
    losses["loss_total"] = float(total.detach().item())
    return total, losses


def _run_training_stage(
    *,
    stage_name: str,
    model: VCNNMultiScale,
    artifacts: MultiScaleArtifacts,
    device_name: str,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    num_epochs: int,
    min_lr: float,
    warmup_epochs: int,
    use_cosine_schedule: bool,
    early_stop: bool,
    early_patience: int,
    early_min_delta: float,
    early_warmup: int,
    max_train_batches: int | None,
    max_val_batches: int | None,
    verbose: bool,
    output_mode: str,
    loss_type: str,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
    obs_mask: torch.Tensor | None,
    field_loss_fn: nn.Module | None,
) -> dict[str, Any]:
    active_exits = _configure_two_stage_training(model, stage_name)
    optimizer = _build_trainable_optimizer(
        model,
        optimizer_name=optimizer_name,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    patience_ctr = 0
    epochs_ran = 0
    stopped_early = False
    last_lr = float(lr)

    for epoch in range(1, int(num_epochs) + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        num_train_batches = max(1, len(artifacts.train_loader))
        for batch_idx, (feature, coeff) in enumerate(artifacts.train_loader):
            if max_train_batches is not None and batch_idx >= int(max_train_batches):
                break
            if bool(use_cosine_schedule):
                progress = float(epoch - 1) + float(batch_idx) / float(num_train_batches)
                last_lr = _adjust_learning_rate(
                    optimizer,
                    progress=progress,
                    base_lr=float(lr),
                    min_lr=float(min_lr),
                    total_epochs=int(num_epochs),
                    warmup_epochs=int(warmup_epochs),
                )
            feature = feature.to(device_name)
            coeff = coeff.to(device_name)
            optimizer.zero_grad()
            outputs = model(feature)
            loss, _ = _compute_stage_loss(
                outputs,
                coeff,
                active_exits=active_exits,
                output_mode=output_mode,
                loss_type=loss_type,
                obs_mask=obs_mask,
                field_loss_fn=field_loss_fn,
                basis=basis,
                mean_flat=mean_flat,
                spatial_shape=artifacts.spatial_shape,
                pad_hw=artifacts.pad_hw,
                norm_mean=norm_mean,
                norm_std=norm_std,
            )
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
            train_batches += 1

        if not bool(use_cosine_schedule):
            last_lr = float(optimizer.param_groups[0]["lr"])

        avg_train_loss = total_train_loss / max(1, train_batches)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (feature, coeff) in enumerate(artifacts.val_loader):
                if max_val_batches is not None and batch_idx >= int(max_val_batches):
                    break
                feature = feature.to(device_name)
                coeff = coeff.to(device_name)
                outputs = model(feature)
                loss, _ = _compute_stage_loss(
                    outputs,
                    coeff,
                    active_exits=active_exits,
                    output_mode=output_mode,
                    loss_type=loss_type,
                    obs_mask=obs_mask,
                    field_loss_fn=field_loss_fn,
                    basis=basis,
                    mean_flat=mean_flat,
                    spatial_shape=artifacts.spatial_shape,
                    pad_hw=artifacts.pad_hw,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                )
                total_val_loss += float(loss.item())
                val_batches += 1

        avg_val_loss = total_val_loss / max(1, val_batches)
        val_losses.append(avg_val_loss)
        epochs_ran = int(epoch)

        if verbose:
            print(
                f"[train_vcnn_multiscale] {stage_name} epoch {epoch:03d}/{int(num_epochs):03d} "
                f"train_loss={avg_train_loss:.4e} val_loss={avg_val_loss:.4e} lr={last_lr:.3e}"
            )

        if avg_val_loss < (best_val_loss - float(early_min_delta)):
            best_val_loss = float(avg_val_loss)
            best_epoch = int(epoch)
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if bool(early_stop) and epoch >= int(early_warmup) and patience_ctr >= int(early_patience):
            stopped_early = True
            if verbose:
                print(f"[train_vcnn_multiscale] {stage_name} early stop at epoch {epoch}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "stage_name": str(stage_name),
        "active_exits": [int(v) for v in active_exits],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "stopped_early": bool(stopped_early),
        "last_lr": float(last_lr),
    }


def _collect_loader_predictions(
    model: VCNNMultiScale,
    dataloader: DataLoader,
    *,
    device: str | torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    device_name = str(device)
    pred1_list: list[np.ndarray] = []
    pred2_list: list[np.ndarray] = []
    pred3_list: list[np.ndarray] = []
    coeff_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (feature, coeff) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            outputs = model(feature.to(device_name))
            pred1_list.append(outputs[0].detach().cpu().numpy().astype(np.float32, copy=False))
            pred2_list.append(outputs[1].detach().cpu().numpy().astype(np.float32, copy=False))
            pred3_list.append(outputs[2].detach().cpu().numpy().astype(np.float32, copy=False))
            coeff_list.append(coeff.detach().cpu().numpy().astype(np.float32, copy=False))
    if not pred1_list:
        raise ValueError("No predictions collected from dataloader.")
    return {
        "pred1": np.concatenate(pred1_list, axis=0),
        "pred2": np.concatenate(pred2_list, axis=0),
        "pred3": np.concatenate(pred3_list, axis=0),
        "coeff": np.concatenate(coeff_list, axis=0),
    }


def _crop_nchw_array(x_nchw: np.ndarray, pad_hw: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    x = np.asarray(x_nchw, dtype=np.float32)
    top, bottom = pad_hw[0]
    left, right = pad_hw[1]
    h_stop = None if bottom == 0 else -bottom
    w_stop = None if right == 0 else -right
    return x[:, :, top:h_stop, left:w_stop].astype(np.float32, copy=False)


@dataclass
class MultiScaleArtifacts:
    train_loader: DataLoader
    val_loader: DataLoader
    train_indices: np.ndarray
    val_indices: np.ndarray
    mask_hw: np.ndarray
    spatial_shape: tuple[int, int, int]
    pad_hw: tuple[tuple[int, int], tuple[int, int]]
    norm_mean_c: np.ndarray | None
    norm_std_c: np.ndarray | None
    representation: str
    include_mask_channel: bool
    input_channels: int
    rank_steps: tuple[int, int, int]
    patch_size: int
    normalize: bool
    dataset_seed: int | None
    noise_sigma: float


class MultiScaleObservationDataset(Dataset):
    def __init__(
        self,
        X_thwc: np.ndarray,
        a_true: np.ndarray,
        *,
        mask_hw: np.ndarray,
        noise_sigma: float,
        representation: str,
        include_mask_channel: bool,
        patch_size: int,
        normalize: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.X_thwc = np.asarray(X_thwc, dtype=np.float32)
        self.a_true = np.asarray(a_true, dtype=np.float32)
        if self.X_thwc.ndim != 4:
            raise ValueError(f"X_thwc must be [T,H,W,C], got {self.X_thwc.shape}")
        if self.a_true.ndim != 2 or self.a_true.shape[0] != self.X_thwc.shape[0]:
            raise ValueError(f"a_true must be [T,r], got {self.a_true.shape}")

        self.mask_hw = np.asarray(mask_hw, dtype=bool)
        self.noise_sigma = float(noise_sigma)
        self.representation = str(representation)
        self.include_mask_channel = bool(include_mask_channel)
        self.seed = None if seed is None else int(seed)

        h, w = int(self.X_thwc.shape[1]), int(self.X_thwc.shape[2])
        self.pad_hw = (_compute_pad(h, int(patch_size)), _compute_pad(w, int(patch_size)))
        self.nearest_idx_hw = _build_nearest_seed_index(self.mask_hw)
        if normalize:
            self.norm_mean_c, self.norm_std_c = _compute_channelwise_mean_std(self.X_thwc)
        else:
            self.norm_mean_c = None
            self.norm_std_c = None

    def __len__(self) -> int:
        return int(self.X_thwc.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_idx = int(idx)
        noise_seed = None if self.seed is None else self.seed + item_idx
        feature = _build_spatial_feature_single(
            self.X_thwc[item_idx],
            mask_hw=self.mask_hw,
            nearest_idx_hw=self.nearest_idx_hw,
            noise_sigma=self.noise_sigma,
            representation=self.representation,
            include_mask_channel=self.include_mask_channel,
            pad_hw=self.pad_hw,
            norm_mean_c=self.norm_mean_c,
            norm_std_c=self.norm_std_c,
            noise_seed=noise_seed,
        )
        coeff = self.a_true[item_idx].astype(np.float32, copy=True)
        return torch.from_numpy(feature.astype(np.float32, copy=True)), torch.from_numpy(coeff)


def reconstruct_truncated_targets(
    coeffs: torch.Tensor,
    *,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    ranks: Sequence[int],
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> dict[int, torch.Tensor]:
    targets: dict[int, torch.Tensor] = {}
    for rank in dict.fromkeys(int(v) for v in ranks):
        rank_eff = max(1, min(int(rank), int(coeffs.shape[1]), int(basis.shape[1])))
        recon_flat = coeffs[:, :rank_eff] @ basis[:, :rank_eff].T + mean_flat[None, :]
        recon = _to_field_nchw(recon_flat, spatial_shape)
        if norm_mean is not None and norm_std is not None:
            recon = (recon - norm_mean[None, :, None, None]) / norm_std[None, :, None, None]
        targets[int(rank)] = recon
    return targets


def construct_band_targets(
    coeffs: torch.Tensor,
    *,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    rank_steps: Sequence[int] = (16, 48, 128),
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> tuple[dict[int, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    rank16, rank48, rank128 = [int(v) for v in rank_steps]
    cumulative = reconstruct_truncated_targets(
        coeffs,
        basis=basis,
        mean_flat=mean_flat,
        spatial_shape=spatial_shape,
        ranks=(rank16, rank48, rank128),
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    y_16 = cumulative[rank16]
    y_48 = cumulative[rank48]
    y_128 = cumulative[rank128]
    b1 = y_16
    b2 = y_48 - y_16
    b3 = y_128 - y_48
    return cumulative, (b1, b2, b3)


def project_field_to_pod_torch(
    field_nchw: torch.Tensor,
    *,
    basis: torch.Tensor,
    mean_flat: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> torch.Tensor:
    centered = field_nchw
    if norm_std is not None:
        centered = centered * norm_std[None, :, None, None]
    flat = centered.permute(0, 2, 3, 1).contiguous().view(centered.shape[0], -1)
    if mean_flat is not None:
        flat = flat - mean_flat[None, :]
    return flat @ basis


def _band_orthogonality_loss(
    a1_pred: torch.Tensor,
    a2_pred: torch.Tensor,
    a3_pred: torch.Tensor,
    rank_steps: Sequence[int],
) -> torch.Tensor:
    rank16, rank48, rank128 = [int(v) for v in rank_steps]
    loss1 = torch.sum(a1_pred[:, rank16:rank128] ** 2, dim=1)
    loss2 = torch.sum(a2_pred[:, :rank16] ** 2, dim=1) + torch.sum(a2_pred[:, rank48:rank128] ** 2, dim=1)
    loss3 = torch.sum(a3_pred[:, :rank48] ** 2, dim=1)
    return torch.mean(loss1 + loss2 + loss3)


def _mean_square_energy(target: torch.Tensor) -> torch.Tensor:
    return torch.mean(target.detach() ** 2).clamp_min(1e-12)


def _relative_field_error(pred: torch.Tensor, target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    diff = pred - target
    diff_energy = torch.sum(diff ** 2, dim=(1, 2, 3))
    ref_energy = torch.sum(reference.detach() ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return torch.mean(diff_energy / ref_energy)


def _coerce_stage_weights(weights: Sequence[float] | None, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if weights is None:
        return tuple(float(v) for v in default)
    values = tuple(float(v) for v in weights)
    if len(values) != 3:
        raise ValueError(f"Expected 3 stage weights, got {values}")
    return values


def _resolve_output_mode(output_mode: str | None) -> str:
    mode = str(output_mode or DEFAULT_OUTPUT_MODE).strip().lower()
    if mode not in ("field", "coeff"):
        raise ValueError(f"Unsupported output_mode='{output_mode}'. Expected 'field' or 'coeff'.")
    return mode


def _pad_coeff_predictions_torch(pred: torch.Tensor, total_dim: int) -> torch.Tensor:
    total = int(total_dim)
    if pred.shape[1] == total:
        return pred
    out = pred.new_zeros((pred.shape[0], total))
    width = min(int(pred.shape[1]), total)
    out[:, :width] = pred[:, :width]
    return out


def _pad_coeff_predictions_np(pred: np.ndarray, total_dim: int) -> np.ndarray:
    pred_np = np.asarray(pred, dtype=np.float32)
    total = int(total_dim)
    if pred_np.shape[1] == total:
        return pred_np.astype(np.float32, copy=False)
    out = np.zeros((pred_np.shape[0], total), dtype=np.float32)
    width = min(int(pred_np.shape[1]), total)
    out[:, :width] = pred_np[:, :width]
    return out


def _coeff_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def _coeff_batch_to_field_nchw(
    coeff_batch: np.ndarray,
    *,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    flat = np.asarray(coeff_batch, dtype=np.float32) @ np.asarray(basis, dtype=np.float32).T + np.asarray(mean_flat, dtype=np.float32)[None, :]
    h, w, c = [int(v) for v in spatial_shape]
    return flat.reshape(-1, h, w, c).transpose(0, 3, 1, 2).astype(np.float32, copy=False)


def _predict_cumulative_coeffs_from_raw_exits(
    pred1: np.ndarray,
    pred2: np.ndarray,
    pred3: np.ndarray,
    *,
    total_dim: int,
) -> dict[str, np.ndarray]:
    a_hat_16 = _pad_coeff_predictions_np(pred1, total_dim)
    a_hat_48 = _pad_coeff_predictions_np(pred2, total_dim)
    a_hat_128 = _pad_coeff_predictions_np(pred3, total_dim)
    return {
        "a_hat_16": a_hat_16,
        "a_hat_48": a_hat_48,
        "a_hat_128": a_hat_128,
    }


def _predict_cumulative_fields_from_coeff_exits(
    pred1: np.ndarray,
    pred2: np.ndarray,
    pred3: np.ndarray,
    *,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    coeffs = _predict_cumulative_coeffs_from_raw_exits(pred1, pred2, pred3, total_dim=int(basis.shape[1]))
    y_hat_16 = _coeff_batch_to_field_nchw(coeffs["a_hat_16"], basis=basis, mean_flat=mean_flat, spatial_shape=spatial_shape)
    y_hat_48 = _coeff_batch_to_field_nchw(coeffs["a_hat_48"], basis=basis, mean_flat=mean_flat, spatial_shape=spatial_shape)
    y_hat_128 = _coeff_batch_to_field_nchw(coeffs["a_hat_128"], basis=basis, mean_flat=mean_flat, spatial_shape=spatial_shape)
    return {
        **coeffs,
        "y_hat_16": y_hat_16,
        "y_hat_48": y_hat_48,
        "y_hat_128": y_hat_128,
        "band1": y_hat_16,
        "band2": (y_hat_48 - y_hat_16).astype(np.float32, copy=False),
        "band3": (y_hat_128 - y_hat_48).astype(np.float32, copy=False),
    }


def _compute_multiscale_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    coeffs: torch.Tensor,
    *,
    basis: torch.Tensor,
    mean_flat: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean: torch.Tensor | None,
    norm_std: torch.Tensor | None,
    rank_steps: tuple[int, int, int],
    output_mode: str,
    coeff_loss_weights: tuple[float, float, float],
    band_loss_weights: tuple[float, float, float],
    recon_loss_weights: tuple[float, float, float],
    ortho_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mode = _resolve_output_mode(output_mode)
    if mode == "coeff":
        rank16, rank48, rank128 = [int(v) for v in rank_steps]
        pred1, pred2, pred3 = outputs
        target1 = coeffs[:, :rank16]
        target2 = coeffs[:, :rank48]
        target3 = coeffs[:, :rank128]
        loss_coeff1 = _coeff_mse_loss(pred1, target1)
        loss_coeff2 = _coeff_mse_loss(pred2, target2)
        loss_coeff3 = _coeff_mse_loss(pred3, target3)
        total = (
            float(coeff_loss_weights[0]) * loss_coeff1
            + float(coeff_loss_weights[1]) * loss_coeff2
            + float(coeff_loss_weights[2]) * loss_coeff3
        )
        return total, {
            "output_mode": mode,
            "coeff_loss_weights": [float(v) for v in coeff_loss_weights],
            "loss_coeff1": float(loss_coeff1.detach().item()),
            "loss_coeff2": float(loss_coeff2.detach().item()),
            "loss_coeff3": float(loss_coeff3.detach().item()),
            "loss_total": float(total.detach().item()),
        }

    cumulative_targets, band_targets = construct_band_targets(
        coeffs,
        basis=basis,
        mean_flat=mean_flat,
        spatial_shape=spatial_shape,
        rank_steps=rank_steps,
        norm_mean=norm_mean,
        norm_std=norm_std,
    )
    y_exit1, y_exit2, y_exit3 = outputs
    pred1 = _crop_nchw_tensor(y_exit1, pad_hw)
    pred2 = _crop_nchw_tensor(y_exit2, pad_hw)
    pred3 = _crop_nchw_tensor(y_exit3, pad_hw)

    y_hat_16 = pred1
    y_hat_48 = pred1 + pred2
    y_hat_128 = pred1 + pred2 + pred3

    b1, b2, b3 = band_targets
    y_48 = cumulative_targets[int(rank_steps[1])]
    y_128 = cumulative_targets[int(rank_steps[2])]

    loss_band1 = _relative_field_error(pred1, b1, b1)
    loss_band2 = _relative_field_error(pred2, b2, b2)
    loss_band3 = _relative_field_error(pred3, b3, b3)
    loss_band = (
        float(band_loss_weights[0]) * loss_band1
        + float(band_loss_weights[1]) * loss_band2
        + float(band_loss_weights[2]) * loss_band3
    )

    loss_recon48 = _relative_field_error(y_hat_48, y_48, b2)
    loss_recon128 = _relative_field_error(y_hat_128, y_128, b3)
    loss_recon = (
        float(recon_loss_weights[1]) * loss_recon48
        + float(recon_loss_weights[2]) * loss_recon128
    )

    a1_pred = project_field_to_pod_torch(pred1, basis=basis, mean_flat=mean_flat, norm_std=norm_std)
    a2_pred = project_field_to_pod_torch(pred2, basis=basis, mean_flat=None, norm_std=norm_std)
    a3_pred = project_field_to_pod_torch(pred3, basis=basis, mean_flat=None, norm_std=norm_std)
    loss_ortho_raw = _band_orthogonality_loss(a1_pred, a2_pred, a3_pred, rank_steps)
    loss_ortho = float(ortho_loss_weight) * loss_ortho_raw

    total = loss_band + loss_recon + loss_ortho
    return total, {
        "output_mode": mode,
        "band_loss_weights": [float(v) for v in band_loss_weights],
        "recon_loss_weights": [float(v) for v in recon_loss_weights],
        "ortho_loss_weight": float(ortho_loss_weight),
        "energy_band1": float(_mean_square_energy(b1).detach().item()),
        "energy_band2": float(_mean_square_energy(b2).detach().item()),
        "energy_band3": float(_mean_square_energy(b3).detach().item()),
        "loss_band1": float(loss_band1.detach().item()),
        "loss_band2": float(loss_band2.detach().item()),
        "loss_band3": float(loss_band3.detach().item()),
        "loss_recon48": float(loss_recon48.detach().item()),
        "loss_recon128": float(loss_recon128.detach().item()),
        "loss_band": float(loss_band.detach().item()),
        "loss_recon": float(loss_recon.detach().item()),
        "loss_ortho_raw": float(loss_ortho_raw.detach().item()),
        "loss_ortho": float(loss_ortho.detach().item()),
        "loss_total": float(total.detach().item()),
    }


def build_multiscale_dataloaders(
    X_thwc: np.ndarray,
    a_true: np.ndarray,
    *,
    mask_hw: np.ndarray,
    noise_sigma: float,
    representation: str,
    include_mask_channel: bool,
    patch_size: int,
    batch_size: int,
    val_ratio: float,
    normalize: bool,
    seed: int | None,
    rank_steps: Sequence[int] = (16, 48, 128),
) -> MultiScaleArtifacts:
    dataset = MultiScaleObservationDataset(
        X_thwc,
        a_true,
        mask_hw=mask_hw,
        noise_sigma=noise_sigma,
        representation=representation,
        include_mask_channel=include_mask_channel,
        patch_size=patch_size,
        normalize=normalize,
        seed=seed,
    )
    total = int(len(dataset))
    n_val = min(max(1, int(round(float(total) * float(val_ratio)))), total - 1)
    n_train = total - n_val
    split_generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=split_generator)
    loader_generator = None if seed is None else torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=bool(n_train >= int(batch_size)),
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
    )

    sample_feature, _ = dataset[0]
    ranks = tuple(int(v) for v in rank_steps)
    train_indices = _subset_indices(train_ds, n_train)
    val_indices = _subset_indices(val_ds, n_val)
    return MultiScaleArtifacts(
        train_loader=train_loader,
        val_loader=val_loader,
        train_indices=train_indices,
        val_indices=val_indices,
        mask_hw=np.asarray(mask_hw, dtype=bool),
        spatial_shape=(int(X_thwc.shape[1]), int(X_thwc.shape[2]), int(X_thwc.shape[3])),
        pad_hw=dataset.pad_hw,
        norm_mean_c=dataset.norm_mean_c,
        norm_std_c=dataset.norm_std_c,
        representation=str(representation),
        include_mask_channel=bool(include_mask_channel),
        input_channels=int(sample_feature.shape[0]),
        rank_steps=(int(ranks[0]), int(ranks[1]), int(ranks[2])),
        patch_size=int(patch_size),
        normalize=bool(normalize),
        dataset_seed=None if seed is None else int(seed),
        noise_sigma=float(noise_sigma),
    )


def train_vcnn_multiscale_on_observations(
    *,
    X_thwc: np.ndarray,
    a_true: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    mask_hw: np.ndarray,
    representation: str,
    include_mask_channel: bool,
    noise_sigma: float,
    batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: str | None,
    hidden_channels: int = 48,
    num_layers: int = 8,
    kernel_size: int = 7,
    optimizer_name: str = "adamw",
    val_ratio: float = 0.1,
    patch_size: int = 1,
    normalize: bool = True,
    min_lr: float = 0.0,
    warmup_epochs: int = 0,
    use_cosine_schedule: bool = True,
    early_stop: bool = True,
    early_patience: int = 20,
    early_min_delta: float = 0.0,
    early_warmup: int = 5,
    seed: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    verbose: bool = True,
    rank_steps: Sequence[int] = (16, 48, 128),
    output_mode: str = DEFAULT_OUTPUT_MODE,
    loss_type: str = "mse",
    obs_weight: float = 1.0,
    stage2_num_epochs: int | None = None,
    stage2_lr: float | None = None,
    coeff_loss_weights: Sequence[float] | None = None,
    band_loss_weights: Sequence[float] | None = None,
    recon_loss_weights: Sequence[float] | None = None,
    ortho_loss_weight: float = 0.25,
) -> tuple[VCNNMultiScale, dict[str, Any], MultiScaleArtifacts]:
    rank_steps_eff = (
        min(int(rank_steps[0]), int(a_true.shape[1])),
        min(int(rank_steps[1]), int(a_true.shape[1])),
        min(int(rank_steps[2]), int(a_true.shape[1])),
    )
    artifacts = build_multiscale_dataloaders(
        X_thwc,
        a_true,
        mask_hw=mask_hw,
        noise_sigma=noise_sigma,
        representation=representation,
        include_mask_channel=include_mask_channel,
        patch_size=patch_size,
        batch_size=batch_size,
        val_ratio=val_ratio,
        normalize=normalize,
        seed=seed,
        rank_steps=rank_steps_eff,
    )

    device_name = _resolve_torch_device(device)
    output_mode_eff = _resolve_output_mode(output_mode)
    model = VCNNMultiScale(
        in_channels=int(artifacts.input_channels),
        out_channels=int(artifacts.spatial_shape[2]),
        hidden_channels=int(hidden_channels),
        num_layers=int(num_layers),
        kernel_size=int(kernel_size),
        output_mode=output_mode_eff,
        coeff_dims=(int(a_true.shape[1]), int(a_true.shape[1]), int(a_true.shape[1])),
    ).to(device_name)

    basis = torch.as_tensor(np.asarray(Ur_eff, dtype=np.float32), device=device_name)
    mean = torch.as_tensor(np.asarray(mean_flat, dtype=np.float32), device=device_name)
    norm_mean = None if artifacts.norm_mean_c is None else torch.as_tensor(np.asarray(artifacts.norm_mean_c, dtype=np.float32), device=device_name)
    norm_std = None if artifacts.norm_std_c is None else torch.as_tensor(np.asarray(artifacts.norm_std_c, dtype=np.float32), device=device_name)
    field_loss_fn = None if output_mode_eff == "coeff" else get_field_loss(loss_type=loss_type, obs_weight=float(obs_weight))
    obs_mask = None if output_mode_eff == "coeff" else _build_obs_mask_tensor(
        artifacts.mask_hw,
        pad_hw=artifacts.pad_hw,
        device=device_name,
    )

    stage1_info = _run_training_stage(
        stage_name="stage1",
        model=model,
        artifacts=artifacts,
        device_name=device_name,
        optimizer_name=optimizer_name,
        lr=float(lr),
        weight_decay=float(weight_decay),
        num_epochs=int(num_epochs),
        min_lr=float(min_lr),
        warmup_epochs=int(warmup_epochs),
        use_cosine_schedule=bool(use_cosine_schedule),
        early_stop=bool(early_stop),
        early_patience=int(early_patience),
        early_min_delta=float(early_min_delta),
        early_warmup=int(early_warmup),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        verbose=verbose,
        output_mode=output_mode_eff,
        loss_type=loss_type,
        basis=basis,
        mean_flat=mean,
        norm_mean=norm_mean,
        norm_std=norm_std,
        obs_mask=obs_mask,
        field_loss_fn=field_loss_fn,
    )
    stage2_epochs = int(num_epochs if stage2_num_epochs is None else stage2_num_epochs)
    stage2_learning_rate = float(lr if stage2_lr is None else stage2_lr)
    stage2_info = _run_training_stage(
        stage_name="stage2",
        model=model,
        artifacts=artifacts,
        device_name=device_name,
        optimizer_name=optimizer_name,
        lr=stage2_learning_rate,
        weight_decay=float(weight_decay),
        num_epochs=stage2_epochs,
        min_lr=float(min_lr),
        warmup_epochs=int(warmup_epochs),
        use_cosine_schedule=bool(use_cosine_schedule),
        early_stop=bool(early_stop),
        early_patience=int(early_patience),
        early_min_delta=float(early_min_delta),
        early_warmup=int(early_warmup),
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        verbose=verbose,
        output_mode=output_mode_eff,
        loss_type=loss_type,
        basis=basis,
        mean_flat=mean,
        norm_mean=norm_mean,
        norm_std=norm_std,
        obs_mask=obs_mask,
        field_loss_fn=field_loss_fn,
    )

    info = {
        "stage1": _sanitize_json_value(stage1_info),
        "stage2": _sanitize_json_value(stage2_info),
        "train_losses": list(stage1_info["train_losses"]) + list(stage2_info["train_losses"]),
        "val_losses": list(stage1_info["val_losses"]) + list(stage2_info["val_losses"]),
        "best_val_loss": float(stage2_info["best_val_loss"]),
        "best_epoch": int(stage2_info["best_epoch"]),
        "epochs_ran": int(stage1_info["epochs_ran"]) + int(stage2_info["epochs_ran"]),
        "stopped_early": bool(stage1_info["stopped_early"] or stage2_info["stopped_early"]),
        "device": str(device_name),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "stage2_lr": float(stage2_learning_rate),
        "last_lr": float(stage2_info["last_lr"]),
        "noise_sigma": float(noise_sigma),
        "representation": str(representation),
        "include_mask_channel": bool(include_mask_channel),
        "patch_size": int(patch_size),
        "normalize_mean_std": bool(normalize),
        "output_mode": output_mode_eff,
        "training_mode": "two_stage_full_target",
        "supervision_mode": "full_target_coeff" if output_mode_eff == "coeff" else "full_target_field",
        "loss_type": str(loss_type),
        "obs_weight": float(obs_weight),
        "rank_steps": [int(v) for v in artifacts.rank_steps],
        "coeff_dims": [int(a_true.shape[1]), int(a_true.shape[1]), int(a_true.shape[1])],
        "full_coeff_dim": int(a_true.shape[1]),
        "hidden_channels": int(hidden_channels),
        "num_layers": int(num_layers),
        "kernel_size": int(kernel_size),
        "param_count": int(_count_parameters(model)),
    }
    return model, info, artifacts


def evaluate_one_batch(
    model: VCNNMultiScale,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    device: str | torch.device,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean_c: np.ndarray | None,
    norm_std_c: np.ndarray | None,
    prefix_steps: Sequence[int] | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> dict[str, float]:
    feature, coeff = batch
    device_name = str(device)
    feature = feature.to(device_name)

    with torch.no_grad():
        outputs = model(feature)

    coeff_np = coeff.detach().cpu().numpy().astype(np.float32, copy=False)
    prefix_steps_eff = _resolve_prefix_steps(int(coeff.shape[1]), prefix_steps)
    output_mode_eff = _resolve_output_mode(output_mode)

    if output_mode_eff == "coeff":
        a_pred_by_exit = {
            1: outputs[0].detach().cpu().numpy().astype(np.float32, copy=False),
            2: outputs[1].detach().cpu().numpy().astype(np.float32, copy=False),
            3: outputs[2].detach().cpu().numpy().astype(np.float32, copy=False),
        }
        metrics: dict[str, float] = {}
        for exit_idx, a_pred in a_pred_by_exit.items():
            for prefix_dim in prefix_steps_eff:
                metrics[f"E{int(prefix_dim)}_exit{exit_idx}"] = _prefix_relative_error(a_pred, coeff_np, int(prefix_dim))
        return metrics

    y_hat_by_exit = {
        1: _crop_nchw_tensor(outputs[0].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
        2: _crop_nchw_tensor(outputs[1].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
        3: _crop_nchw_tensor(outputs[2].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
    }

    metrics: dict[str, float] = {}
    for exit_idx, pred_cum in y_hat_by_exit.items():
        if norm_mean_c is not None and norm_std_c is not None:
            pred_abs = _denormalize_field_nchw(pred_cum, norm_mean_c, norm_std_c)
        else:
            pred_abs = np.asarray(pred_cum, dtype=np.float32, copy=False)
        pred_flat = np.transpose(pred_abs, (0, 2, 3, 1)).reshape(pred_abs.shape[0], -1)
        a_pred = project_to_pod(
            pred_flat,
            np.asarray(Ur_eff, dtype=np.float32),
            np.asarray(mean_flat, dtype=np.float32),
        ).astype(np.float32, copy=False)

        for prefix_dim in prefix_steps_eff:
            metrics[f"E{int(prefix_dim)}_exit{exit_idx}"] = _prefix_relative_error(a_pred, coeff_np, int(prefix_dim))

    return metrics


def aggregate_metrics(
    model: VCNNMultiScale,
    dataloader: DataLoader,
    *,
    device: str | torch.device,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean_c: np.ndarray | None,
    norm_std_c: np.ndarray | None,
    prefix_steps: Sequence[int] | None = None,
    max_batches: int | None = None,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> dict[str, float]:
    aggregate: dict[str, float] = {}
    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch_metrics = evaluate_one_batch(
            model,
            batch,
            device=device,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
            spatial_shape=spatial_shape,
            pad_hw=pad_hw,
            norm_mean_c=norm_mean_c,
            norm_std_c=norm_std_c,
            prefix_steps=prefix_steps,
            output_mode=output_mode,
        )
        batch_size = int(batch[1].shape[0])
        total_samples += batch_size
        for key, value in batch_metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value) * float(batch_size)

    if total_samples == 0:
        raise ValueError("aggregate_metrics received an empty dataloader or max_batches=0")
    return {key: float(value / float(total_samples)) for key, value in aggregate.items()}


def measure_exit_latency_ms(
    model: VCNNMultiScale,
    feature_batch: torch.Tensor,
    *,
    exit_level: int,
    device: str | torch.device,
    repeats: int = 20,
    warmup: int = 3,
) -> float:
    device_name = str(device)
    x = feature_batch.to(device_name)
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            _ = model(x, exit_level=int(exit_level))
        _maybe_sync(device_name)
        start = time.perf_counter()
        for _ in range(max(1, int(repeats))):
            _ = model(x, exit_level=int(exit_level))
        _maybe_sync(device_name)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / max(1, int(repeats)))


def append_summary_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or [])
        if existing_fieldnames != list(CSV_FIELDNAMES):
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow({key: existing.get(key, "") for key in CSV_FIELDNAMES})

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDNAMES))
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def _predict_fields_from_raw_exits(
    pred1: np.ndarray,
    pred2: np.ndarray,
    pred3: np.ndarray,
    *,
    norm_mean_c: np.ndarray | None,
    norm_std_c: np.ndarray | None,
) -> dict[str, np.ndarray]:
    exit1_norm = np.asarray(pred1, dtype=np.float32)
    exit2_norm = np.asarray(pred2, dtype=np.float32)
    exit3_norm = np.asarray(pred3, dtype=np.float32)
    if norm_mean_c is not None and norm_std_c is not None:
        return {
            "exit1": _denormalize_field_nchw(exit1_norm, norm_mean_c, norm_std_c),
            "exit2": _denormalize_field_nchw(exit2_norm, norm_mean_c, norm_std_c),
            "exit3": _denormalize_field_nchw(exit3_norm, norm_mean_c, norm_std_c),
        }
    return {
        "exit1": exit1_norm.astype(np.float32, copy=False),
        "exit2": exit2_norm.astype(np.float32, copy=False),
        "exit3": exit3_norm.astype(np.float32, copy=False),
    }


def _project_field_predictions_to_coefficients(
    field_by_exit: dict[int, np.ndarray],
    *,
    basis: np.ndarray,
    mean_flat: np.ndarray,
) -> dict[int, np.ndarray]:
    coeff_by_exit: dict[int, np.ndarray] = {}
    for exit_idx, field in field_by_exit.items():
        field_flat = np.transpose(np.asarray(field, dtype=np.float32), (0, 2, 3, 1)).reshape(field.shape[0], -1)
        coeff_by_exit[int(exit_idx)] = project_to_pod(
            field_flat,
            np.asarray(basis, dtype=np.float32),
            np.asarray(mean_flat, dtype=np.float32),
        ).astype(np.float32, copy=False)
    return coeff_by_exit


def _build_exit_field_predictions(
    *,
    output_mode: str,
    collected: dict[str, np.ndarray],
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean_c: np.ndarray | None,
    norm_std_c: np.ndarray | None,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    mode = _resolve_output_mode(output_mode)
    if mode == "coeff":
        coeff_by_exit = {
            1: np.asarray(collected["pred1"], dtype=np.float32, copy=False),
            2: np.asarray(collected["pred2"], dtype=np.float32, copy=False),
            3: np.asarray(collected["pred3"], dtype=np.float32, copy=False),
        }
        field_by_exit = {
            exit_idx: _coeff_batch_to_field_nchw(coeff, basis=basis, mean_flat=mean_flat, spatial_shape=spatial_shape)
            for exit_idx, coeff in coeff_by_exit.items()
        }
        return field_by_exit, coeff_by_exit

    field_by_exit = {
        1: _crop_nchw_array(np.asarray(collected["pred1"], dtype=np.float32), pad_hw),
        2: _crop_nchw_array(np.asarray(collected["pred2"], dtype=np.float32), pad_hw),
        3: _crop_nchw_array(np.asarray(collected["pred3"], dtype=np.float32), pad_hw),
    }
    field_by_exit = {
        exit_idx: field
        for exit_idx, field in enumerate(
            (
                _predict_fields_from_raw_exits(
                    field_by_exit[1],
                    field_by_exit[2],
                    field_by_exit[3],
                    norm_mean_c=norm_mean_c,
                    norm_std_c=norm_std_c,
                )["exit1"],
                _predict_fields_from_raw_exits(
                    field_by_exit[1],
                    field_by_exit[2],
                    field_by_exit[3],
                    norm_mean_c=norm_mean_c,
                    norm_std_c=norm_std_c,
                )["exit2"],
                _predict_fields_from_raw_exits(
                    field_by_exit[1],
                    field_by_exit[2],
                    field_by_exit[3],
                    norm_mean_c=norm_mean_c,
                    norm_std_c=norm_std_c,
                )["exit3"],
            ),
            start=1,
        )
    }
    coeff_by_exit = _project_field_predictions_to_coefficients(
        field_by_exit,
        basis=basis,
        mean_flat=mean_flat,
    )
    return field_by_exit, coeff_by_exit


def _reconstruct_prefix_field_np(
    coeff: np.ndarray,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
    prefix_dim: int,
) -> np.ndarray:
    k = max(1, min(int(prefix_dim), int(coeff.shape[0]), int(basis.shape[1])))
    flat = np.asarray(coeff[:k], dtype=np.float32) @ np.asarray(basis[:, :k], dtype=np.float32).T + np.asarray(mean_flat, dtype=np.float32)
    return flat.reshape(spatial_shape).astype(np.float32, copy=False)


def _reconstruct_band_field_np(
    coeff: np.ndarray,
    basis: np.ndarray,
    spatial_shape: tuple[int, int, int],
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    start = max(0, int(start_idx))
    end = max(start, min(int(end_idx), int(coeff.shape[0]), int(basis.shape[1])))
    flat = np.asarray(coeff[start:end], dtype=np.float32) @ np.asarray(basis[:, start:end], dtype=np.float32).T
    return flat.reshape(spatial_shape).astype(np.float32, copy=False)


def save_vcnn_multiscale_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: VCNNMultiScale,
    train_info: dict[str, Any],
    artifacts: MultiScaleArtifacts,
    config_path: str | Path,
    exp_id: str,
    mask_rate: float,
    noise_sigma: float,
    model_cfg: dict[str, Any],
    pod_dir_used: str | Path | None = None,
    split_index: int | None = None,
    split_count: int | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    ensure_dir(checkpoint_path)
    payload = {
        "state_dict": model.state_dict(),
        "train_info": _sanitize_json_value(train_info),
        "exp_id": str(exp_id),
        "config_path": str(config_path),
        "mask_rate": float(mask_rate),
        "noise_sigma": float(noise_sigma),
        "mask_hw": np.asarray(artifacts.mask_hw, dtype=np.uint8),
        "representation": str(artifacts.representation),
        "include_mask_channel": bool(artifacts.include_mask_channel),
        "patch_size": int(artifacts.patch_size),
        "normalize_mean_std": bool(artifacts.normalize),
        "dataset_seed": None if artifacts.dataset_seed is None else int(artifacts.dataset_seed),
        "train_indices": np.asarray(artifacts.train_indices, dtype=np.int64),
        "val_indices": np.asarray(artifacts.val_indices, dtype=np.int64),
        "pad_hw": [[int(v) for v in pair] for pair in artifacts.pad_hw],
        "spatial_shape": [int(v) for v in artifacts.spatial_shape],
        "input_channels": int(artifacts.input_channels),
        "rank_steps": [int(v) for v in artifacts.rank_steps],
        "norm_mean_c": None if artifacts.norm_mean_c is None else np.asarray(artifacts.norm_mean_c, dtype=np.float32),
        "norm_std_c": None if artifacts.norm_std_c is None else np.asarray(artifacts.norm_std_c, dtype=np.float32),
        "hidden_channels": int(train_info.get("hidden_channels", model_cfg.get("hidden_channels", 48))),
        "num_layers": int(train_info.get("num_layers", model_cfg.get("num_layers", 8))),
        "kernel_size": int(train_info.get("kernel_size", model_cfg.get("kernel_size", 7))),
        "output_mode": str(train_info.get("output_mode", model_cfg.get("multiscale_output_mode", DEFAULT_OUTPUT_MODE))),
        "coeff_dims": [int(v) for v in train_info.get("coeff_dims", [artifacts.rank_steps[-1]] * 3)],
        "model_cfg": _sanitize_json_value(model_cfg),
        "pod_dir_used": None if pod_dir_used is None else str(pod_dir_used),
        "split_index": None if split_index is None else int(split_index),
        "split_count": None if split_count is None else int(split_count),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _resolve_split_bounds(width: int, split_index: int, split_count: int) -> tuple[int, int]:
    if split_count <= 0:
        raise ValueError(f"split_count must be positive, got {split_count}")
    if not (1 <= split_index <= split_count):
        raise ValueError(f"split_index must be in [1, {split_count}], got {split_index}")
    if width % split_count != 0:
        raise ValueError(f"Width {width} is not divisible by split_count={split_count}")
    split_w = width // split_count
    x0 = (split_index - 1) * split_w
    x1 = x0 + split_w
    return x0, x1


def _prepare_snapshots_with_optional_split(
    *,
    data_cfg,
    Ur: np.ndarray,
    mean_flat: np.ndarray,
    r_eff: int,
    split_index: int | None,
    split_count: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    from ..dataio.loader import describe_source, load_raw
    from ..eval.rebuild import _prepare_snapshots

    if split_index is None:
        return _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff=r_eff, verbose=verbose)

    if verbose:
        print(f"[rebuild] Loading full raw data from {describe_source(data_cfg)} for split verification ...")

    X_thwc = load_raw(data_cfg)  # [T,H,W,C]
    T, H, W, C = X_thwc.shape
    x0, x1 = _resolve_split_bounds(W, split_index=int(split_index), split_count=int(split_count))
    X_thwc = X_thwc[:, :, x0:x1, :]
    D = int(X_thwc.shape[1] * X_thwc.shape[2] * X_thwc.shape[3])

    if Ur.shape[0] != D:
        raise ValueError(
            "Split data and POD shape mismatch during verification: "
            f"Ur first dim={Ur.shape[0]} but split H*W*C={D}."
        )

    X_flat_all = X_thwc.reshape(T, D)
    Ur_eff = Ur[:, :r_eff]
    mean_flat_v = np.asarray(mean_flat, dtype=np.float32).reshape(1, D)
    A_true = ((X_flat_all - mean_flat_v) @ Ur_eff).astype(np.float32, copy=False)

    if verbose:
        print(
            f"  -> split_index={split_index}/{split_count}, x_range=[{x0},{x1}), "
            f"X_thwc={X_thwc.shape}, flat=[{T},{D}], r_eff={r_eff}"
        )

    return X_thwc, A_true


def load_vcnn_multiscale_checkpoint(checkpoint_path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {checkpoint_path}")
    return payload


def build_vcnn_multiscale_from_checkpoint(payload: dict[str, Any], *, device: str | torch.device) -> VCNNMultiScale:
    model = VCNNMultiScale(
        in_channels=int(payload["input_channels"]),
        out_channels=int(payload["spatial_shape"][2]),
        hidden_channels=int(payload["hidden_channels"]),
        num_layers=int(payload["num_layers"]),
        kernel_size=int(payload["kernel_size"]),
        output_mode=str(payload.get("output_mode", payload.get("train_info", {}).get("output_mode", DEFAULT_OUTPUT_MODE))),
        coeff_dims=tuple(int(v) for v in payload.get("coeff_dims", payload.get("rank_steps", (16, 48, 128)))),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(str(device))


def _metric_self_check(
    coeff_true: np.ndarray,
    *,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    prefix_steps: Sequence[int] | None = None,
) -> dict[str, float]:
    prefix_steps_eff = _resolve_prefix_steps(int(coeff_true.shape[1]), prefix_steps)
    checks: dict[str, float] = {}
    basis_np = np.asarray(basis, dtype=np.float32)
    for prefix_dim in prefix_steps_eff:
        k = int(prefix_dim)
        flat = np.asarray(coeff_true[:, :k], dtype=np.float32) @ basis_np[:, :k].T + np.asarray(mean_flat, dtype=np.float32)[None, :]
        a_proj = project_to_pod(flat, basis_np, np.asarray(mean_flat, dtype=np.float32)).astype(np.float32, copy=False)
        checks[f"E{k}"] = _prefix_relative_error(a_proj, coeff_true, k)
    return checks


def _sample_prefix_error(a_pred: np.ndarray, a_true: np.ndarray, prefix_dim: int) -> np.ndarray:
    k = max(1, int(prefix_dim))
    diff = np.asarray(a_pred[:, :k], dtype=np.float64) - np.asarray(a_true[:, :k], dtype=np.float64)
    denom = np.sum(np.asarray(a_true[:, :k], dtype=np.float64) ** 2, axis=1)
    denom = np.maximum(denom, 1e-12)
    return (np.sum(diff ** 2, axis=1) / denom).astype(np.float64, copy=False)


def _select_representative_positions(sample_errors: np.ndarray) -> list[int]:
    order = np.argsort(np.asarray(sample_errors, dtype=np.float64))
    if order.size == 0:
        return []
    positions = [0, order.size // 2, order.size - 1]
    out: list[int] = []
    for pos in positions:
        idx = int(order[int(pos)])
        if idx not in out:
            out.append(idx)
    return out


def _save_case_figure(
    out_path: Path,
    *,
    pred_coeffs_by_exit: dict[int, np.ndarray],
    true_coeff: np.ndarray,
    basis: np.ndarray,
    mean_flat: np.ndarray,
    spatial_shape: tuple[int, int, int],
    sample_title: str,
) -> None:
    prefix_steps = list(PREFIX_EVAL_STEPS)
    band_edges = [(0, 16), (16, 32), (32, 48), (48, 64), (64, 80), (80, 96), (96, 112), (112, 128)]

    # Layout tuning knobs for visual spacing and colorbar geometry.
    panel_width_in = 3.0
    panel_min_height_in = 0.58
    colorbar_height_in = 0.15
    grid_wspace = 0.05
    grid_hspace = 0.16
    colorbar_width_frac = 0.62
    colorbar_height_frac = 0.58
    colorbar_bottom_frac = 0.18

    rows: list[dict[str, Any]] = []
    for row_idx, prefix_dim in enumerate(prefix_steps):
        true_prefix = _to_numpy_image(_reconstruct_prefix_field_np(true_coeff, basis, mean_flat, spatial_shape, prefix_dim))
        pred_prefix_images = [
            _to_numpy_image(_reconstruct_prefix_field_np(pred_coeffs_by_exit[exit_idx], basis, mean_flat, spatial_shape, prefix_dim))
            for exit_idx in EXIT_LEVELS
        ]

        band_start, band_end = band_edges[row_idx]
        true_band = _to_numpy_image(_reconstruct_band_field_np(true_coeff, basis, spatial_shape, band_start, band_end))
        pred_band_images = [
            _to_numpy_image(_reconstruct_band_field_np(pred_coeffs_by_exit[exit_idx], basis, spatial_shape, band_start, band_end))
            for exit_idx in EXIT_LEVELS
        ]

        rows.append(
            {
                "prefix_dim": int(prefix_dim),
                "band_start": int(band_start),
                "band_end": int(band_end),
                "pred_prefix": pred_prefix_images,
                "true_prefix": true_prefix,
                "pred_band": pred_band_images,
                "true_band": true_band,
            }
        )

    sample_image = np.asarray(rows[0]["true_prefix"], dtype=np.float32)
    img_h, img_w = int(sample_image.shape[0]), int(sample_image.shape[1])
    image_aspect = float(max(1, img_h)) / float(max(1, img_w))
    panel_height_in = max(panel_min_height_in, panel_width_in * image_aspect)
    fig_height = float(8.0 * panel_height_in + 8.0 * colorbar_height_in + 1.0)

    fig = plt.figure(figsize=(28.0, fig_height))
    height_ratios: list[float] = []
    for _ in range(8):
        height_ratios.extend([1.0, max(0.12, colorbar_height_in / max(panel_height_in, 1e-6))])
    gs = fig.add_gridspec(
        nrows=16,
        ncols=8,
        left=0.03,
        right=0.99,
        top=0.955,
        bottom=0.03,
        wspace=grid_wspace,
        hspace=grid_hspace,
        height_ratios=height_ratios,
    )

    for row_idx, row_data in enumerate(rows):
        prefix_dim = int(row_data["prefix_dim"])
        band_start = int(row_data["band_start"])
        band_end = int(row_data["band_end"])
        pred_prefix_images = [np.asarray(v, dtype=np.float32) for v in list(row_data["pred_prefix"])]
        true_prefix = np.asarray(row_data["true_prefix"], dtype=np.float32)
        pred_band_images = [np.asarray(v, dtype=np.float32) for v in list(row_data["pred_band"])]
        true_band = np.asarray(row_data["true_band"], dtype=np.float32)

        prefix_group = [pred_prefix_images[0], pred_prefix_images[1], pred_prefix_images[2], true_prefix]
        band_group = [pred_band_images[0], pred_band_images[1], pred_band_images[2], true_band]

        prefix_arr = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in prefix_group], axis=0)
        band_arr = np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in band_group], axis=0)
        prefix_center = float(np.mean(prefix_arr))
        band_center = float(np.mean(band_arr))
        prefix_span = float(max(np.max(np.abs(prefix_arr - prefix_center)), 1e-8))
        band_span = float(max(np.max(np.abs(band_arr - band_center)), 1e-8))
        prefix_norm = mcolors.TwoSlopeNorm(
            vmin=prefix_center - prefix_span,
            vcenter=prefix_center,
            vmax=prefix_center + prefix_span,
        )
        band_norm = mcolors.TwoSlopeNorm(
            vmin=band_center - band_span,
            vcenter=band_center,
            vmax=band_center + band_span,
        )

        images = (
            pred_prefix_images[0],
            pred_prefix_images[1],
            pred_prefix_images[2],
            true_prefix,
            pred_band_images[0],
            pred_band_images[1],
            pred_band_images[2],
            true_band,
        )
        titles = (
            f"Exit1 prefix 1-{prefix_dim}",
            f"Exit2 prefix 1-{prefix_dim}",
            f"Exit3 prefix 1-{prefix_dim}",
            f"True prefix 1-{prefix_dim}",
            f"Exit1 band {band_start + 1}-{band_end}",
            f"Exit2 band {band_start + 1}-{band_end}",
            f"Exit3 band {band_start + 1}-{band_end}",
            f"True band {band_start + 1}-{band_end}",
        )

        for col_idx, (image, title) in enumerate(zip(images, titles)):
            ax = fig.add_subplot(gs[2 * row_idx, col_idx])
            norm = prefix_norm if col_idx < 4 else band_norm
            ax.imshow(image, origin="lower", cmap="RdBu_r", norm=norm)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"row {row_idx + 1}", fontsize=8)

        prefix_cax = fig.add_subplot(gs[2 * row_idx + 1, 0:4])
        band_cax = fig.add_subplot(gs[2 * row_idx + 1, 4:8])
        for cax in (prefix_cax, band_cax):
            pos = cax.get_position()
            width = pos.width * float(colorbar_width_frac)
            height = pos.height * float(colorbar_height_frac)
            left = pos.x0 + 0.5 * (pos.width - width)
            bottom = pos.y0 + float(colorbar_bottom_frac) * (pos.height - height)
            cax.set_position([left, bottom, width, height])
        prefix_sm = plt.cm.ScalarMappable(norm=prefix_norm, cmap="RdBu_r")
        band_sm = plt.cm.ScalarMappable(norm=band_norm, cmap="RdBu_r")
        cbar_prefix = fig.colorbar(prefix_sm, cax=prefix_cax, orientation="horizontal")
        cbar_band = fig.colorbar(band_sm, cax=band_cax, orientation="horizontal")
        cbar_prefix.ax.tick_params(labelsize=6, length=2, pad=1)
        cbar_band.ax.tick_params(labelsize=6, length=2, pad=1)
        cbar_prefix.set_label(f"prefix mean={prefix_center:.2e}", fontsize=6, labelpad=1)
        cbar_band.set_label(f"band mean={band_center:.2e}", fontsize=6, labelpad=1)

    fig.suptitle(sample_title, fontsize=13)

    ensure_dir(out_path)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_nrmse_vs_r_figure(
    out_dir: Path,
    *,
    exp_id: str,
    coeff_true: np.ndarray,
    coeff_by_exit: dict[int, np.ndarray],
) -> dict[str, Any]:
    curves = []
    curve_payload: dict[str, Any] = {}
    for exit_idx in EXIT_LEVELS:
        r_grid, nrmse = compute_nrmse_vs_r_coeff(
            np.asarray(coeff_by_exit[int(exit_idx)], dtype=np.float32),
            np.asarray(coeff_true, dtype=np.float32),
        )
        key = f"exit{int(exit_idx)}"
        curve_payload[key] = {
            "r_grid": np.asarray(r_grid, dtype=np.int32).tolist(),
            "nrmse_full": np.asarray(nrmse["nrmse_full"], dtype=np.float64).tolist(),
            "nrmse_prefix": np.asarray(nrmse["nrmse_prefix"], dtype=np.float64).tolist(),
        }
        curves.append(
            {
                "label": key,
                "x": np.asarray(r_grid, dtype=np.float64),
                "nrmse": {
                    "nrmse_full": np.asarray(nrmse["nrmse_full"], dtype=np.float64),
                    "nrmse_prefix": np.asarray(nrmse["nrmse_prefix"], dtype=np.float64),
                },
            }
        )

    fig = plot_nrmse_family_vs_r_curves(
        curves,
        title=f"{exp_id} | NRMSE vs r by exit",
        nrmse_kinds=("nrmse_full", "nrmse_prefix"),
        legend_outside=True,
    )
    fig_path = out_dir / "nrmse_vs_r.png"
    data_path = out_dir / "nrmse_vs_r.json"
    if fig is not None:
        ensure_dir(fig_path)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    save_json(data_path, curve_payload)
    return {
        "figure_path": str(fig_path),
        "data_path": str(data_path),
        "curves": curve_payload,
    }


def _sample_nrmse_from_coeff(a_pred: np.ndarray, a_true: np.ndarray) -> np.ndarray:
    pred = np.asarray(a_pred, dtype=np.float64)
    true = np.asarray(a_true, dtype=np.float64)
    diff = pred - true
    num = np.sum(diff ** 2, axis=1)
    den = np.sum(true ** 2, axis=1)
    den = np.maximum(den, 1e-12)
    return np.sqrt(num / den).astype(np.float64, copy=False)


def _compute_nrmse_vs_t_curves(
    *,
    model: VCNNMultiScale,
    dataloader: DataLoader,
    device: str | torch.device,
    output_mode: str,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
    pad_hw: tuple[tuple[int, int], tuple[int, int]],
    norm_mean_c: np.ndarray | None,
    norm_std_c: np.ndarray | None,
    max_batches: int | None = None,
) -> dict[int, np.ndarray]:
    mode = _resolve_output_mode(output_mode)
    device_name = str(device)
    basis_np = np.asarray(Ur_eff, dtype=np.float32)
    mean_np = np.asarray(mean_flat, dtype=np.float32)
    out: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}

    model.eval()
    with torch.no_grad():
        for batch_idx, (feature, coeff) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= int(max_batches):
                break

            outputs = model(feature.to(device_name))
            coeff_true = coeff.detach().cpu().numpy().astype(np.float32, copy=False)

            if mode == "coeff":
                coeff_pred_by_exit = {
                    1: outputs[0].detach().cpu().numpy().astype(np.float32, copy=False),
                    2: outputs[1].detach().cpu().numpy().astype(np.float32, copy=False),
                    3: outputs[2].detach().cpu().numpy().astype(np.float32, copy=False),
                }
            else:
                pred_by_exit = {
                    1: _crop_nchw_tensor(outputs[0].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
                    2: _crop_nchw_tensor(outputs[1].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
                    3: _crop_nchw_tensor(outputs[2].detach().cpu(), pad_hw).numpy().astype(np.float32, copy=False),
                }
                coeff_pred_by_exit = {}
                for exit_idx, pred_nchw in pred_by_exit.items():
                    if norm_mean_c is not None and norm_std_c is not None:
                        pred_abs = _denormalize_field_nchw(pred_nchw, norm_mean_c, norm_std_c)
                    else:
                        pred_abs = np.asarray(pred_nchw, dtype=np.float32, copy=False)
                    pred_flat = np.transpose(pred_abs, (0, 2, 3, 1)).reshape(pred_abs.shape[0], -1)
                    coeff_pred_by_exit[exit_idx] = project_to_pod(pred_flat, basis_np, mean_np).astype(np.float32, copy=False)

            for exit_idx in EXIT_LEVELS:
                out[int(exit_idx)].append(_sample_nrmse_from_coeff(coeff_pred_by_exit[int(exit_idx)], coeff_true))

    return {
        int(exit_idx): np.concatenate(values, axis=0) if values else np.asarray([], dtype=np.float64)
        for exit_idx, values in out.items()
    }


def _save_nrmse_vs_t_figure(
    out_dir: Path,
    *,
    exp_id: str,
    nrmse_by_exit: dict[int, np.ndarray],
) -> dict[str, Any]:
    t_axis = np.arange(int(max((len(v) for v in nrmse_by_exit.values()), default=0)), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(12.0, 4.2))
    color_map = {1: "tab:blue", 2: "tab:orange", 3: "tab:green"}
    payload: dict[str, Any] = {"t": t_axis.tolist(), "nrmse": {}}

    for exit_idx in EXIT_LEVELS:
        curve = np.asarray(nrmse_by_exit.get(int(exit_idx), np.asarray([], dtype=np.float64)), dtype=np.float64)
        if curve.size == 0:
            payload["nrmse"][f"exit{int(exit_idx)}"] = []
            continue
        x = np.arange(curve.shape[0], dtype=np.int32)
        ax.plot(x, curve, linewidth=1.2, color=color_map[int(exit_idx)], label=f"exit{int(exit_idx)}")
        payload["nrmse"][f"exit{int(exit_idx)}"] = curve.tolist()

    ax.set_title(f"{exp_id} | NRMSE vs t (all samples)")
    ax.set_xlabel("t (sample index)")
    ax.set_ylabel("NRMSE")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    fig_path = out_dir / "nrmse_vs_t.png"
    data_path = out_dir / "nrmse_vs_t.json"
    ensure_dir(fig_path)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    save_json(data_path, payload)

    return {
        "figure_path": str(fig_path),
        "data_path": str(data_path),
        "length": int(t_axis.shape[0]),
    }


def run_vcnn_multiscale_verification(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    batch_size: int | None = None,
    max_batches: int | None = None,
    pod_dir: str | Path | None = None,
    split_index: int | None = None,
    split_count: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    from ..config.yaml_io import load_experiment_yaml
    from ..eval.rebuild import _load_or_build_pod

    checkpoint = load_vcnn_multiscale_checkpoint(checkpoint_path, map_location="cpu")
    device_name = _resolve_torch_device(device or checkpoint.get("train_info", {}).get("device", "auto"))
    output_root = Path(output_dir)
    ensure_dir(output_root)

    data_cfg, pod_cfg, _, _ = load_experiment_yaml(checkpoint["config_path"])
    pod_dir_eff = pod_dir if pod_dir is not None else checkpoint.get("pod_dir_used")
    if pod_dir_eff:
        pod_cfg.save_dir = Path(str(pod_dir_eff))

    split_index_eff = split_index if split_index is not None else checkpoint.get("split_index")
    split_count_eff = int(split_count if split_count is not None else checkpoint.get("split_count", 4))

    Ur, mean_flat, pod_meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    r_eff = int(min(int(Ur.shape[1]), int(pod_meta.get("r_used", Ur.shape[1])), int(pod_cfg.r)))
    X_thwc, A_true = _prepare_snapshots_with_optional_split(
        data_cfg=data_cfg,
        Ur=Ur,
        mean_flat=mean_flat,
        r_eff=r_eff,
        split_index=None if split_index_eff is None else int(split_index_eff),
        split_count=split_count_eff,
        verbose=verbose,
    )
    Ur_eff = np.asarray(Ur[:, :r_eff], dtype=np.float32)

    dataset = MultiScaleObservationDataset(
        X_thwc,
        A_true,
        mask_hw=np.asarray(checkpoint["mask_hw"], dtype=bool),
        noise_sigma=float(checkpoint["noise_sigma"]),
        representation=str(checkpoint["representation"]),
        include_mask_channel=bool(checkpoint["include_mask_channel"]),
        patch_size=int(checkpoint["patch_size"]),
        normalize=bool(checkpoint["normalize_mean_std"]),
        seed=None if checkpoint.get("dataset_seed", None) is None else int(checkpoint["dataset_seed"]),
    )
    val_indices = np.asarray(checkpoint["val_indices"], dtype=np.int64)
    val_subset = Subset(dataset, val_indices.tolist())
    val_batch_size = int(batch_size or checkpoint.get("train_info", {}).get("batch_size", 8))
    val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False, drop_last=False)

    model = build_vcnn_multiscale_from_checkpoint(checkpoint, device=device_name)
    collected = _collect_loader_predictions(model, val_loader, device=device_name, max_batches=max_batches)
    coeff_val = collected["coeff"]
    output_mode = _resolve_output_mode(checkpoint.get("output_mode", checkpoint.get("train_info", {}).get("output_mode", DEFAULT_OUTPUT_MODE)))

    field_by_exit, coeff_by_exit = _build_exit_field_predictions(
        output_mode=output_mode,
        collected=collected,
        pad_hw=dataset.pad_hw,
        norm_mean_c=dataset.norm_mean_c,
        norm_std_c=dataset.norm_std_c,
        basis=Ur_eff,
        mean_flat=np.asarray(mean_flat, dtype=np.float32),
        spatial_shape=(int(X_thwc.shape[1]), int(X_thwc.shape[2]), int(X_thwc.shape[3])),
    )
    a_pred_final = np.asarray(coeff_by_exit[3], dtype=np.float32, copy=False)
    sample_errors = _sample_prefix_error(a_pred_final, coeff_val, min(128, coeff_val.shape[1]))
    representative_positions = _select_representative_positions(sample_errors)

    self_check = _metric_self_check(
        coeff_val,
        basis=Ur_eff,
        mean_flat=np.asarray(mean_flat, dtype=np.float32),
        prefix_steps=PREFIX_EVAL_STEPS,
    )
    if verbose:
        print("[verify_vcnn_mod] metric self-check")
        for key, value in self_check.items():
            print(f"  {key}(true,true)={value:.6e}")

    energy_stats = {
        "exit1": _field_stats(field_by_exit[1]),
        "exit2": _field_stats(field_by_exit[2]),
        "exit3": _field_stats(field_by_exit[3]),
    }
    if verbose:
        print("[verify_vcnn_mod] exit energy/amplitude summary")
        for key, stats in energy_stats.items():
            print(
                f"  {key}: l2_mean={stats['l2_mean']:.4e} mean={stats['mean']:.4e} "
                f"min={stats['min']:.4e} max={stats['max']:.4e} near_zero_l2_frac={stats['near_zero_l2_frac']:.3f}"
            )

    representative_cases: list[dict[str, Any]] = []
    for label, pos in zip(("low_error", "median_error", "high_error"), representative_positions):
        global_idx = int(val_indices[int(pos)])
        true_coeff = np.asarray(coeff_val[int(pos)], dtype=np.float32)
        fig_path = output_root / f"{label}_compare.png"
        _save_case_figure(
            fig_path,
            pred_coeffs_by_exit={
                1: np.asarray(coeff_by_exit[1][int(pos)], dtype=np.float32),
                2: np.asarray(coeff_by_exit[2][int(pos)], dtype=np.float32),
                3: np.asarray(coeff_by_exit[3][int(pos)], dtype=np.float32),
            },
            true_coeff=true_coeff,
            basis=Ur_eff,
            mean_flat=np.asarray(mean_flat, dtype=np.float32),
            spatial_shape=(int(X_thwc.shape[1]), int(X_thwc.shape[2]), int(X_thwc.shape[3])),
            sample_title=f"{checkpoint['exp_id']} | {label} | sample={global_idx} | E128={sample_errors[int(pos)]:.4e}",
        )
        representative_cases.append(
            {
                "label": label,
                "position_in_val": int(pos),
                "global_index": int(global_idx),
                "E128": float(sample_errors[int(pos)]),
                "figure_path": str(fig_path),
            }
        )

    nrmse_summary = _save_nrmse_vs_r_figure(
        output_root,
        exp_id=str(checkpoint["exp_id"]),
        coeff_true=coeff_val,
        coeff_by_exit=coeff_by_exit,
    )

    full_loader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)
    nrmse_t_curves = _compute_nrmse_vs_t_curves(
        model=model,
        dataloader=full_loader,
        device=device_name,
        output_mode=output_mode,
        Ur_eff=Ur_eff,
        mean_flat=np.asarray(mean_flat, dtype=np.float32),
        pad_hw=dataset.pad_hw,
        norm_mean_c=dataset.norm_mean_c,
        norm_std_c=dataset.norm_std_c,
        max_batches=max_batches,
    )
    nrmse_t_summary = _save_nrmse_vs_t_figure(
        output_root,
        exp_id=str(checkpoint["exp_id"]),
        nrmse_by_exit=nrmse_t_curves,
    )

    review_summary = {
        "exp_id": str(checkpoint["exp_id"]),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(checkpoint["config_path"]),
        "mask_rate": float(checkpoint["mask_rate"]),
        "noise_sigma": float(checkpoint["noise_sigma"]),
        "output_mode": output_mode,
        "self_check": {key: float(value) for key, value in self_check.items()},
        "energy_stats": energy_stats,
        "representative_cases": representative_cases,
        "nrmse_vs_r": nrmse_summary,
        "nrmse_vs_t": nrmse_t_summary,
        "val_size": int(val_indices.shape[0]),
        "full_size": int(len(dataset)),
    }
    save_json(output_root / "review_summary.json", review_summary)
    with (output_root / "review_summary.txt").open("w", encoding="utf-8") as f:
        f.write(json.dumps(review_summary, ensure_ascii=False, indent=2))
    return review_summary
