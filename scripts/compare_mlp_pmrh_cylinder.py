from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.schemas import resolve_model_dataset_specs
from backend.config.yaml_io import load_experiment_yaml
from backend.eval.rebuild import (
    _encode_rate,
    _load_or_build_pod,
    _make_noisy_observations_batch,
    _mask_strategy_from_data_cfg,
    _predict_coeff_model_batch,
    _prepare_snapshots,
    _resolve_branch_train_config,
)
from backend.models.train import train_mlp_on_observations, train_v4a_on_observations, train_v4b_on_observations
from backend.sampling.masks import flatten_mask, generate_observation_mask_hw


def _parse_noise_sigmas(raw: str) -> list[float]:
    out: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        out.append(float(text))
    return out


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is not None:
            merged[str(key)] = value
    return merged


def _count_model_bytes(model: torch.nn.Module) -> tuple[int, int]:
    param_count = int(sum(param.numel() for param in model.parameters()))
    param_bytes = int(sum(param.numel() * param.element_size() for param in model.parameters()))
    return param_count, param_bytes


def _count_active_model_bytes(model: torch.nn.Module, stage: str | None = None) -> tuple[int, int]:
    if stage is None or not hasattr(model, "get_stage_modules"):
        return _count_model_bytes(model)

    modules = tuple(model.get_stage_modules(stage=stage))
    seen: set[int] = set()
    param_count = 0
    param_bytes = 0
    for module in modules:
        for param in module.parameters():
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            param_count += int(param.numel())
            param_bytes += int(param.numel() * param.element_size())
    return int(param_count), int(param_bytes)


def _prefix_metrics(pred: np.ndarray, target: np.ndarray, prefix_dim: int) -> dict[str, float]:
    pred_prefix = np.asarray(pred[:, :prefix_dim], dtype=np.float64)
    target_prefix = np.asarray(target[:, :prefix_dim], dtype=np.float64)
    diff = pred_prefix - target_prefix
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = np.linalg.norm(target_prefix, axis=1)
    denom = np.maximum(denom, 1e-12)
    rel_l2 = float(np.mean(np.linalg.norm(diff, axis=1) / denom))
    return {
        "coeff_rmse": rmse,
        "coeff_rel_l2": rel_l2,
    }


def _full_metrics_from_prefix_pred(pred_prefix: np.ndarray, target_full: np.ndarray, prefix_dim: int) -> dict[str, float]:
    pred_prefix = np.asarray(pred_prefix, dtype=np.float64)
    target_full = np.asarray(target_full, dtype=np.float64)
    pred_full = np.zeros_like(target_full, dtype=np.float64)
    pred_full[:, : int(prefix_dim)] = pred_prefix[:, : int(prefix_dim)]
    diff = pred_full - target_full
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = np.linalg.norm(target_full, axis=1)
    denom = np.maximum(denom, 1e-12)
    rel_l2 = float(np.mean(np.linalg.norm(diff, axis=1) / denom))
    return {
        "coeff_rmse": rmse,
        "coeff_rel_l2": rel_l2,
    }


def _maybe_sync(device_name: str | torch.device) -> None:
    if str(device_name).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_prediction_ms(
    predict_fn,
    *,
    repeats: int,
    device_name: str | torch.device,
) -> float:
    _maybe_sync(device_name)
    start = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        predict_fn()
    _maybe_sync(device_name)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / max(1, int(repeats)))


def _efficiency_metrics(prefix_rmse: float, infer_ms: float, param_count: int) -> dict[str, float]:
    safe_ms = max(float(infer_ms), 1e-12)
    safe_param = max(int(param_count), 2)
    return {
        "rmse_per_ms": float(prefix_rmse / safe_ms),
        "rmse_per_log_param": float(prefix_rmse / np.log(float(safe_param))),
    }


def _resolve_v4_budget_epochs(cfg: dict[str, Any], default_total_epochs: int) -> dict[str, int]:
    total_epochs = int(cfg.get("max_epochs", default_total_epochs))
    stage1_epochs = cfg.get("stage1_epochs", None)
    stage2_warmup_epochs = cfg.get("stage2_warmup_epochs", None)
    stage2_tune_epochs = cfg.get("stage2_tune_epochs", None)
    stage3_warmup_epochs = cfg.get("stage3_warmup_epochs", None)
    stage3_tune_epochs = cfg.get("stage3_tune_epochs", None)
    joint_epochs = cfg.get("joint_epochs", None)
    if all(value is None for value in (stage1_epochs, stage2_warmup_epochs, stage2_tune_epochs, stage3_warmup_epochs, stage3_tune_epochs, joint_epochs)):
        if total_epochs <= 1:
            alloc = [1, 0, 0, 0, 0, 0]
        elif total_epochs == 2:
            alloc = [1, 0, 1, 0, 0, 0]
        elif total_epochs == 3:
            alloc = [1, 0, 1, 0, 1, 0]
        else:
            base = np.asarray([0.20, 0.10, 0.20, 0.10, 0.25, 0.15], dtype=np.float64)
            alloc = np.floor(base * float(total_epochs)).astype(np.int64)
            minima = np.asarray([1, 0, 1, 0, 1, 1], dtype=np.int64)
            alloc = np.maximum(alloc, minima)
            while int(alloc.sum()) > total_epochs:
                for idx in (5, 1, 3, 0, 2, 4):
                    if int(alloc.sum()) <= total_epochs:
                        break
                    if alloc[idx] > minima[idx]:
                        alloc[idx] -= 1
            remainder = int(total_epochs - int(alloc.sum()))
            if remainder > 0:
                frac = base * float(total_epochs) - np.floor(base * float(total_epochs))
                for idx in np.argsort(-frac)[:remainder]:
                    alloc[int(idx)] += 1
            alloc = alloc.astype(np.int64).tolist()
        stage1_epochs, stage2_warmup_epochs, stage2_tune_epochs, stage3_warmup_epochs, stage3_tune_epochs, joint_epochs = [int(v) for v in alloc]
    else:
        stage1_epochs = int(0 if stage1_epochs is None else stage1_epochs)
        stage2_warmup_epochs = int(0 if stage2_warmup_epochs is None else stage2_warmup_epochs)
        stage2_tune_epochs = int(0 if stage2_tune_epochs is None else stage2_tune_epochs)
        stage3_warmup_epochs = int(0 if stage3_warmup_epochs is None else stage3_warmup_epochs)
        stage3_tune_epochs = int(0 if stage3_tune_epochs is None else stage3_tune_epochs)
        joint_epochs = int(0 if joint_epochs is None else joint_epochs)
    return {
        "stage1_epochs": int(stage1_epochs),
        "stage2_warmup_epochs": int(stage2_warmup_epochs),
        "stage2_tune_epochs": int(stage2_tune_epochs),
        "stage3_warmup_epochs": int(stage3_warmup_epochs),
        "stage3_tune_epochs": int(stage3_tune_epochs),
        "joint_epochs": int(joint_epochs),
    }


def _append_rows(csv_path: Path, rows: Iterable[dict[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_fieldnames: list[str] = []
    for row in row_list:
        for key in row.keys():
            if key not in new_fieldnames:
                new_fieldnames.append(str(key))

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fieldnames = list(reader.fieldnames or [])
        fieldnames = list(existing_fieldnames)
        for key in new_fieldnames:
            if key not in fieldnames:
                fieldnames.append(key)
        all_rows = []
        for row in existing_rows:
            all_rows.append({key: row.get(key, "") for key in fieldnames})
        for row in row_list:
            all_rows.append({key: row.get(key, "") for key in fieldnames})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        return

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(row_list)


def _load_existing_baseline_ref(
    csv_path: Path,
    *,
    config: str,
    mask_rate: float,
    noise_sigma: float,
    model_label: str,
) -> dict[str, float] | None:
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None

    selected: dict[str, float] | None = None
    for row in rows:
        if str(row.get("model_type", "")).strip().lower() != "mlp":
            continue
        if str(row.get("model_label", "")).strip().lower() != str(model_label).strip().lower():
            continue
        try:
            row_config = str(row.get("config", "")).strip()
            row_mask_rate = float(row.get("mask_rate", "nan"))
            row_noise_sigma = float(row.get("noise_sigma", "nan"))
            row_rmse = float(row.get("prefix_coeff_rmse", "nan"))
            row_infer_ms = float(row.get("infer_ms", "nan"))
            row_param_count = float(row.get("param_count", "nan"))
        except Exception:
            continue

        if row_config != str(config):
            continue
        if abs(row_mask_rate - float(mask_rate)) > 1e-12:
            continue
        if abs(row_noise_sigma - float(noise_sigma)) > 1e-12:
            continue
        if not np.isfinite(row_rmse) or not np.isfinite(row_infer_ms) or not np.isfinite(row_param_count):
            continue

        selected = {
            "prefix_coeff_rmse": float(row_rmse),
            "infer_ms": float(row_infer_ms),
            "param_count": float(row_param_count),
        }
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline MLPs and the selected v4 improved budget-expert model on cylinder2d POD coefficients.")
    parser.add_argument("--config", default="configs/cylinder_exp_full.yaml")
    parser.add_argument("--mask-rate", type=float, default=0.0005)
    parser.add_argument("--noise-sigmas", default="0,0.001,0.01")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-csv", default="artifacts/experiments/pmrh_benchmarks/cylinder2d_mlp_pmrh_compare.csv")
    parser.add_argument("--timing-repeats", type=int, default=10)
    parser.add_argument("--timing-max-frames", type=int, default=256)
    parser.add_argument("--override-max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--run-mode",
        choices=("baseline-only", "both", "improved-only"),
        default="baseline-only",
        help="baseline-only: 仅跑 MLP baseline；both: baseline+所选 v4；improved-only: 仅跑所选 v4",
    )
    parser.add_argument("--model-version", choices=("v4a", "v4b"), default="v4a")
    parser.add_argument("--tag", default="cylinder2d_mlp_vs_v4")
    args = parser.parse_args()
    model_version = str(args.model_version).strip().lower()

    noise_sigmas = _parse_noise_sigmas(args.noise_sigmas)
    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(args.config)
    if train_cfg is None:
        raise ValueError("train config is required")

    Ur, mean_flat, pod_meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=True)
    r_eff = int(min(int(Ur.shape[1]), int(pod_meta.get("r_used", Ur.shape[1])), int(pod_cfg.r)))
    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff=r_eff, verbose=True)
    X_flat_all = X_thwc.reshape(X_thwc.shape[0], -1).astype(np.float32, copy=False)
    Ur_eff = np.asarray(Ur[:, :r_eff], dtype=np.float32)

    T, H, W, C = X_thwc.shape
    model_dataset_specs = resolve_model_dataset_specs(data_cfg, num_channels=C)
    mask_strategy, mask_strategy_kwargs_base = _mask_strategy_from_data_cfg(
        data_cfg,
        X_thwc=X_thwc,
        Ur_eff=Ur_eff,
        H=H,
        W=W,
        C=C,
        pod_cfg=pod_cfg,
    )
    mask_rate = float(args.mask_rate)
    mask_seed = int(getattr(data_cfg, "observation_mask_seed", 0))
    mask_strategy_kwargs = dict(mask_strategy_kwargs_base)
    if mask_strategy.strip().lower() in ("cylinder_structure_aware", "structure_aware", "region_importance"):
        template_count = max(1, int(mask_strategy_kwargs.get("num_templates", 1)))
        mask_strategy_kwargs["template_index"] = int(_encode_rate(mask_rate) % template_count)
    mask_hw = generate_observation_mask_hw(
        H,
        W,
        mask_rate=mask_rate,
        seed=mask_seed,
        strategy=mask_strategy,
        strategy_kwargs=mask_strategy_kwargs,
    )
    mask_flat = flatten_mask(mask_hw, C=C)
    Y_true = X_flat_all[:, mask_flat]
    mean_masked = mean_flat[mask_flat]

    base_common_cfg = {
        "noise_sigma": float(getattr(train_cfg, "noise_sigma", 0.0)),
        "batch_size": int(getattr(train_cfg, "batch_size", 64)),
        "lr": float(getattr(train_cfg, "lr", 1e-3)),
        "weight_decay": float(getattr(train_cfg, "weight_decay", 0.0)),
        "val_ratio": float(getattr(train_cfg, "val_ratio", 0.1)),
        "device": args.device or str(getattr(train_cfg, "device", "auto")),
        "seed": int(getattr(train_cfg, "seed", 0)),
        "max_train_batches": args.max_train_batches,
        "max_val_batches": args.max_val_batches,
    }
    mlp_cfg = _merge_overrides(
        _resolve_branch_train_config(train_cfg, model_type="mlp", mask_rate=mask_rate),
        {
            "hidden_dims": (256, 256),
            "device": base_common_cfg["device"],
            "seed": base_common_cfg["seed"],
            "max_train_batches": base_common_cfg["max_train_batches"],
            "max_val_batches": base_common_cfg["max_val_batches"],
        },
    )
    if args.override_max_epochs is not None:
        mlp_cfg["max_epochs"] = int(args.override_max_epochs)

    improved_cfg = _merge_overrides(
        _resolve_branch_train_config(train_cfg, model_type="pmrh", mask_rate=mask_rate),
        {
            "stem_dim": 48,
            "stage1_hidden_dims": (64, 64),
            "stage2_hidden_dims": (128, 128),
            "stage3_hidden_dims": (256, 256),
            "adapter16_dim": 64,
            "adapter48_dim": 128,
            "group_ratios": (1, 2, 5),
            "stage_loss_weights": (1.0, 1.0, 1.0),
            "adapter_reg_weights": (0.0, 0.0),
            "stem_lr_scale": 0.2,
            "device": base_common_cfg["device"],
            "seed": base_common_cfg["seed"],
            "max_train_batches": base_common_cfg["max_train_batches"],
            "max_val_batches": base_common_cfg["max_val_batches"],
        },
    )
    if args.override_max_epochs is not None:
        improved_cfg["max_epochs"] = int(args.override_max_epochs)
    improved_budget_epochs = _resolve_v4_budget_epochs(improved_cfg, default_total_epochs=int(getattr(train_cfg, "max_epochs", 200)))

    mlp_specs = [
        ("mlp16", min(16, int(r_eff)), (256, 256)),
        ("mlp48", min(48, int(r_eff)), (256, 256)),
        ("mlp128", int(r_eff), (256, 256)),
        # Extra compact baselines for low-budget references.
        ("mlp16_h64", min(16, int(r_eff)), (64, 64)),
        ("mlp48_h128", min(48, int(r_eff)), (128, 128)),
    ]

    run_baseline = str(args.run_mode) in ("baseline-only", "both")
    run_improved = str(args.run_mode) in ("both", "improved-only")

    trained_mlps: list[tuple[str, int, tuple[int, ...], torch.nn.Module, dict[str, Any], str | torch.device, int, int, int]] = []
    if run_baseline:
        for model_label, prefix_dim, hidden_dims in mlp_specs:
            print(f"\n=== train baseline {model_label} (out_dim={prefix_dim}, hidden_dims={hidden_dims}) ===")
            Ur_prefix = np.asarray(Ur_eff[:, : int(prefix_dim)], dtype=np.float32)
            model, train_info = train_mlp_on_observations(
                X_flat_all=X_flat_all,
                Ur_eff=Ur_prefix,
                mean_flat=mean_flat,
                mask_flat=mask_flat,
                noise_sigma=float(mlp_cfg.get("noise_sigma", base_common_cfg["noise_sigma"])),
                coeff_loss_weights=None,
                loss_weighting="none",
                loss_weight_power=1.0,
                hidden_dims=tuple(int(v) for v in hidden_dims),
                batch_size=int(mlp_cfg.get("batch_size", base_common_cfg["batch_size"])),
                num_epochs=int(mlp_cfg.get("max_epochs", getattr(train_cfg, "max_epochs", 400))),
                lr=float(mlp_cfg.get("lr", base_common_cfg["lr"])),
                weight_decay=float(mlp_cfg.get("weight_decay", base_common_cfg["weight_decay"])),
                val_ratio=float(mlp_cfg.get("val_ratio", base_common_cfg["val_ratio"])),
                device=str(mlp_cfg.get("device", base_common_cfg["device"])),
                centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
                verbose=True,
                live_line=True,
                live_every=int(mlp_cfg.get("live_every", 1)),
                conv_window=int(mlp_cfg.get("conv_window", 25)),
                conv_slope_thresh=float(mlp_cfg.get("conv_slope_thresh", -1e-3)),
                plot_loss=bool(mlp_cfg.get("plot_loss", False)),
                plot_path=mlp_cfg.get("plot_path", None),
                early_stop=bool(mlp_cfg.get("early_stop", True)),
                early_patience=int(mlp_cfg.get("early_patience", 20)),
                early_min_delta=float(mlp_cfg.get("early_min_delta", 0.0)),
                early_warmup=int(mlp_cfg.get("early_warmup", 5)),
                seed=int(mlp_cfg.get("seed", base_common_cfg["seed"])),
                max_train_batches=(None if mlp_cfg.get("max_train_batches", None) is None else int(mlp_cfg.get("max_train_batches"))),
                max_val_batches=(None if mlp_cfg.get("max_val_batches", None) is None else int(mlp_cfg.get("max_val_batches"))),
            )
            model_device = next(model.parameters()).device
            model_chunk = int(mlp_cfg.get("eval_chunk_size", getattr(train_cfg, "eval_chunk_size", 2048)))
            param_count, param_bytes = _count_model_bytes(model)
            trained_mlps.append(
                (
                    model_label,
                    int(prefix_dim),
                    tuple(int(v) for v in hidden_dims),
                    model,
                    train_info,
                    model_device,
                    model_chunk,
                    param_count,
                    param_bytes,
                )
            )

    improved_model: torch.nn.Module | None = None
    improved_train_info: dict[str, Any] = {}
    improved_device: str | torch.device | None = None
    improved_chunk = int(improved_cfg.get("eval_chunk_size", getattr(train_cfg, "eval_chunk_size", 2048)))
    improved_total_param_count = 0
    improved_total_param_bytes = 0
    improved_group_spec = {
        "coarse_dim": int(min(16, int(r_eff))),
        "mid_dim": int(min(max(48, 16), int(r_eff)) - min(16, int(r_eff))),
        "fine_dim": int(max(0, int(r_eff) - min(max(48, 16), int(r_eff)))),
        "stage2_dim": int(min(48, int(r_eff))),
        "total_dim": int(r_eff),
    }
    improved_best_val_by_stage = {"stage1": float("nan"), "stage2": float("nan"), "full": float("nan")}
    improved_variant = "v4a_shared_stem_parallel_budget_experts" if model_version == "v4a" else "v4b_shared_stem_latent_guided_budget_experts"
    improved_model_type = f"mlp_{model_version}"
    improved_model_label = "mlp_v4a_branch_a" if model_version == "v4a" else "mlp_v4b_branch_b"
    if run_improved:
        print(f"\n=== train {model_version} improved budget experts ===")
        if model_version == "v4a":
            improved_model, improved_train_info = train_v4a_on_observations(
                X_flat_all=X_flat_all,
                Ur_eff=Ur_eff,
                mean_flat=mean_flat,
                mask_flat=mask_flat,
                noise_sigma=float(improved_cfg.get("noise_sigma", base_common_cfg["noise_sigma"])),
                stem_dim=int(improved_cfg.get("stem_dim", 48)),
                stage1_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage1_hidden_dims", (64, 64))),
                stage2_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage2_hidden_dims", (128, 128))),
                stage3_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage3_hidden_dims", (256, 256))),
                group_ratios=tuple(int(v) for v in improved_cfg.get("group_ratios", (1, 2, 5))),
                stage_loss_weights=tuple(float(v) for v in improved_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0))),
                stage1_epochs=int(improved_budget_epochs["stage1_epochs"]),
                stage2_warmup_epochs=int(improved_budget_epochs["stage2_warmup_epochs"]),
                stage2_tune_epochs=int(improved_budget_epochs["stage2_tune_epochs"]),
                stage3_warmup_epochs=int(improved_budget_epochs["stage3_warmup_epochs"]),
                stage3_tune_epochs=int(improved_budget_epochs["stage3_tune_epochs"]),
                joint_epochs=int(improved_budget_epochs["joint_epochs"]),
                batch_size=int(improved_cfg.get("batch_size", base_common_cfg["batch_size"])),
                lr=float(improved_cfg.get("lr", base_common_cfg["lr"])),
                weight_decay=float(improved_cfg.get("weight_decay", base_common_cfg["weight_decay"])),
                val_ratio=float(improved_cfg.get("val_ratio", base_common_cfg["val_ratio"])),
                device=str(improved_cfg.get("device", base_common_cfg["device"])),
                centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
                verbose=True,
                live_line=True,
                live_every=int(improved_cfg.get("live_every", 1)),
                conv_window=int(improved_cfg.get("conv_window", 25)),
                conv_slope_thresh=float(improved_cfg.get("conv_slope_thresh", -1e-3)),
                plot_loss=bool(improved_cfg.get("plot_loss", False)),
                plot_path=improved_cfg.get("plot_path", None),
                stem_lr_scale=float(improved_cfg.get("stem_lr_scale", 0.2)),
                joint_sampling_weights=tuple(float(v) for v in improved_cfg.get("joint_sampling_weights", improved_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0)))),
                seed=int(improved_cfg.get("seed", base_common_cfg["seed"])),
                max_train_batches=(None if improved_cfg.get("max_train_batches", None) is None else int(improved_cfg.get("max_train_batches"))),
                max_val_batches=(None if improved_cfg.get("max_val_batches", None) is None else int(improved_cfg.get("max_val_batches"))),
            )
        else:
            improved_model, improved_train_info = train_v4b_on_observations(
                X_flat_all=X_flat_all,
                Ur_eff=Ur_eff,
                mean_flat=mean_flat,
                mask_flat=mask_flat,
                noise_sigma=float(improved_cfg.get("noise_sigma", base_common_cfg["noise_sigma"])),
                stem_dim=int(improved_cfg.get("stem_dim", 48)),
                stage1_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage1_hidden_dims", (64, 64))),
                stage2_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage2_hidden_dims", (128, 128))),
                stage3_hidden_dims=tuple(int(v) for v in improved_cfg.get("stage3_hidden_dims", (256, 256))),
                adapter16_dim=(None if improved_cfg.get("adapter16_dim", None) is None else int(improved_cfg.get("adapter16_dim"))),
                adapter48_dim=(None if improved_cfg.get("adapter48_dim", None) is None else int(improved_cfg.get("adapter48_dim"))),
                group_ratios=tuple(int(v) for v in improved_cfg.get("group_ratios", (1, 2, 5))),
                stage_loss_weights=tuple(float(v) for v in improved_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0))),
                adapter_reg_weights=tuple(float(v) for v in improved_cfg.get("adapter_reg_weights", (0.0, 0.0))),
                stage1_epochs=int(improved_budget_epochs["stage1_epochs"]),
                stage2_warmup_epochs=int(improved_budget_epochs["stage2_warmup_epochs"]),
                stage2_tune_epochs=int(improved_budget_epochs["stage2_tune_epochs"]),
                stage3_warmup_epochs=int(improved_budget_epochs["stage3_warmup_epochs"]),
                stage3_tune_epochs=int(improved_budget_epochs["stage3_tune_epochs"]),
                joint_epochs=int(improved_budget_epochs["joint_epochs"]),
                batch_size=int(improved_cfg.get("batch_size", base_common_cfg["batch_size"])),
                lr=float(improved_cfg.get("lr", base_common_cfg["lr"])),
                weight_decay=float(improved_cfg.get("weight_decay", base_common_cfg["weight_decay"])),
                val_ratio=float(improved_cfg.get("val_ratio", base_common_cfg["val_ratio"])),
                device=str(improved_cfg.get("device", base_common_cfg["device"])),
                centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
                verbose=True,
                live_line=True,
                live_every=int(improved_cfg.get("live_every", 1)),
                conv_window=int(improved_cfg.get("conv_window", 25)),
                conv_slope_thresh=float(improved_cfg.get("conv_slope_thresh", -1e-3)),
                plot_loss=bool(improved_cfg.get("plot_loss", False)),
                plot_path=improved_cfg.get("plot_path", None),
                stem_lr_scale=float(improved_cfg.get("stem_lr_scale", 0.2)),
                joint_sampling_weights=tuple(float(v) for v in improved_cfg.get("joint_sampling_weights", improved_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0)))),
                seed=int(improved_cfg.get("seed", base_common_cfg["seed"])),
                max_train_batches=(None if improved_cfg.get("max_train_batches", None) is None else int(improved_cfg.get("max_train_batches"))),
                max_val_batches=(None if improved_cfg.get("max_val_batches", None) is None else int(improved_cfg.get("max_val_batches"))),
            )
        improved_device = next(improved_model.parameters()).device
        improved_total_param_count, improved_total_param_bytes = _count_model_bytes(improved_model)
        improved_group_spec = improved_model.group_spec.as_dict()
        improved_variant = str(improved_train_info.get("model_variant", improved_variant))
        improved_best_val_by_stage = {
            "stage1": float(improved_train_info.get("best_stage1_val_loss", float("nan"))),
            "stage2": float(improved_train_info.get("best_stage2_val_loss", float("nan"))),
            "full": float(improved_train_info.get("best_stage3_val_loss", float("nan"))),
        }

    timing_frames = int(min(T, max(1, int(args.timing_max_frames))))
    Y_timing = Y_true[:timing_frames]
    mean_timing = mean_masked

    mlp_group_spec = {
        "coarse_dim": int(min(16, int(r_eff))),
        "stage2_dim": int(min(48, int(r_eff))),
        "total_dim": int(r_eff),
    }

    improved_eval_specs = [
        ("stage1", int(improved_group_spec["coarse_dim"]), "prefix16"),
        ("stage2", int(improved_group_spec["stage2_dim"]), "prefix48"),
        ("full", int(improved_group_spec["total_dim"]), "prefix128"),
    ]

    rows: list[dict[str, Any]] = []
    for noise_sigma in noise_sigmas:
        print(f"\n=== eval noise_sigma={noise_sigma:.4e} ===")
        Y_noisy = _make_noisy_observations_batch(
            Y_true,
            noise_sigma=float(noise_sigma),
            centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
            mean_masked=mean_masked,
        )
        Y_noisy_timing = _make_noisy_observations_batch(
            Y_timing,
            noise_sigma=float(noise_sigma),
            centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
            mean_masked=mean_timing,
        )

        baseline_refs: dict[str, dict[str, float] | None] = {
            "prefix16": None,
            "prefix48": None,
            "prefix128": None,
        }

        for model_label, prefix_dim, hidden_dims, model, train_info, model_device, model_chunk, param_count, param_bytes in trained_mlps:
            pred = _predict_coeff_model_batch(model, Y_noisy, device=model_device, chunk_size=model_chunk)
            infer_ms = _measure_prediction_ms(
                lambda: _predict_coeff_model_batch(model, Y_noisy_timing, device=model_device, chunk_size=model_chunk),
                repeats=int(args.timing_repeats),
                device_name=model_device,
            )

            prefix_target = np.asarray(A_true[:, : int(prefix_dim)], dtype=np.float64)
            prefix_metrics = _prefix_metrics(pred, prefix_target, int(prefix_dim))
            full_metrics = _full_metrics_from_prefix_pred(pred, A_true, int(prefix_dim))
            efficiency = _efficiency_metrics(prefix_metrics["coeff_rmse"], infer_ms, param_count)

            row = {
                "tag": str(args.tag),
                "config": str(args.config),
                "mask_rate": float(mask_rate),
                "noise_sigma": float(noise_sigma),
                "model_type": "mlp",
                "model_variant": "baseline_direct_heads",
                "model_label": str(model_label),
                "predict_stage": "full",
                "stage_label": f"prefix{int(prefix_dim)}",
                "prefix_dim": int(prefix_dim),
                "prefix_coeff_rmse": float(prefix_metrics["coeff_rmse"]),
                "prefix_coeff_rel_l2": float(prefix_metrics["coeff_rel_l2"]),
                "full_coeff_rmse": float(full_metrics["coeff_rmse"]),
                "full_coeff_rel_l2": float(full_metrics["coeff_rel_l2"]),
                "infer_ms": float(infer_ms),
                "timing_frames": int(timing_frames),
                "param_count": int(param_count),
                "param_bytes": int(param_bytes),
                "model_param_count_total": int(param_count),
                "model_param_bytes_total": int(param_bytes),
                "rmse_per_ms": float(efficiency["rmse_per_ms"]),
                "rmse_per_log_param": float(efficiency["rmse_per_log_param"]),
                "train_noise_sigma": float(getattr(train_cfg, "noise_sigma", 0.0)),
                "centered_pod": bool(getattr(eval_cfg, "centered_pod", True)),
                "group_spec_json": json.dumps(mlp_group_spec, ensure_ascii=False),
                "best_val_loss": float(train_info.get("best_val_loss", float("nan"))),
                "hidden_dims": json.dumps([int(v) for v in hidden_dims], ensure_ascii=False),
                "model_version": "baseline",
            }
            rows.append(row)
            if int(prefix_dim) == int(mlp_group_spec["coarse_dim"]):
                baseline_refs["prefix16"] = {
                    "prefix_coeff_rmse": float(prefix_metrics["coeff_rmse"]),
                    "infer_ms": float(infer_ms),
                    "param_count": float(param_count),
                }
            elif int(prefix_dim) == int(mlp_group_spec["stage2_dim"]):
                baseline_refs["prefix48"] = {
                    "prefix_coeff_rmse": float(prefix_metrics["coeff_rmse"]),
                    "infer_ms": float(infer_ms),
                    "param_count": float(param_count),
                }
            elif int(prefix_dim) == int(mlp_group_spec["total_dim"]):
                baseline_refs["prefix128"] = {
                    "prefix_coeff_rmse": float(prefix_metrics["coeff_rmse"]),
                    "infer_ms": float(infer_ms),
                    "param_count": float(param_count),
                }
            print(
                f"[{model_label}] prefix_dim={prefix_dim} "
                f"prefix_rel_l2={prefix_metrics['coeff_rel_l2']:.4e} full_rel_l2={full_metrics['coeff_rel_l2']:.4e} infer_ms={infer_ms:.3f}"
            )

        baseline_labels = {
            "prefix16": "mlp16_h64",
            "prefix48": "mlp48_h128",
            "prefix128": "mlp128",
        }
        for stage_label, model_label in baseline_labels.items():
            if baseline_refs[stage_label] is None:
                baseline_refs[stage_label] = _load_existing_baseline_ref(
                    Path(args.output_csv),
                    config=str(args.config),
                    mask_rate=float(mask_rate),
                    noise_sigma=float(noise_sigma),
                    model_label=str(model_label),
                )

        if run_improved and improved_model is not None and improved_device is not None:
            for predict_stage, prefix_dim, stage_label in improved_eval_specs:
                pred = _predict_coeff_model_batch(
                    improved_model,
                    Y_noisy,
                    device=improved_device,
                    chunk_size=improved_chunk,
                    predict_stage=predict_stage,
                )
                infer_ms = _measure_prediction_ms(
                    lambda: _predict_coeff_model_batch(
                        improved_model,
                        Y_noisy_timing,
                        device=improved_device,
                        chunk_size=improved_chunk,
                        predict_stage=predict_stage,
                    ),
                    repeats=int(args.timing_repeats),
                    device_name=improved_device,
                )

                prefix_target = np.asarray(A_true[:, : int(prefix_dim)], dtype=np.float64)
                prefix_metrics = _prefix_metrics(pred, prefix_target, int(prefix_dim))
                full_metrics = _full_metrics_from_prefix_pred(pred, A_true, int(prefix_dim))
                active_param_count, active_param_bytes = _count_active_model_bytes(improved_model, stage=predict_stage)
                efficiency = _efficiency_metrics(prefix_metrics["coeff_rmse"], infer_ms, active_param_count)
                baseline_ref = baseline_refs.get(stage_label)
                target_baseline_label = baseline_labels[str(stage_label)]
                rmse_ratio_vs_target = None
                infer_ratio_vs_target = None
                param_ratio_vs_target = None
                rmse_ratio_vs_mlp16 = None
                infer_ratio_vs_mlp16 = None
                param_ratio_vs_mlp16 = None
                if baseline_ref is not None:
                    rmse_ratio_vs_target = float(prefix_metrics["coeff_rmse"] / max(float(baseline_ref["prefix_coeff_rmse"]), 1e-12))
                    infer_ratio_vs_target = float(infer_ms / max(float(baseline_ref["infer_ms"]), 1e-12))
                    param_ratio_vs_target = float(active_param_count / max(float(baseline_ref["param_count"]), 1.0))
                    if str(stage_label) == "prefix16":
                        rmse_ratio_vs_mlp16 = rmse_ratio_vs_target
                        infer_ratio_vs_mlp16 = infer_ratio_vs_target
                        param_ratio_vs_mlp16 = param_ratio_vs_target

                row = {
                    "tag": str(args.tag),
                    "config": str(args.config),
                    "mask_rate": float(mask_rate),
                    "noise_sigma": float(noise_sigma),
                    "model_type": str(improved_model_type),
                    "model_version": str(model_version),
                    "model_variant": str(improved_variant),
                    "model_label": str(improved_model_label),
                    "predict_stage": str(predict_stage),
                    "budget_level": ("coarse" if str(predict_stage) == "stage1" else ("mid" if str(predict_stage) == "stage2" else "full")),
                    "stage_label": str(stage_label),
                    "prefix_dim": int(prefix_dim),
                    "prefix_coeff_rmse": float(prefix_metrics["coeff_rmse"]),
                    "prefix_coeff_rel_l2": float(prefix_metrics["coeff_rel_l2"]),
                    "full_coeff_rmse": float(full_metrics["coeff_rmse"]),
                    "full_coeff_rel_l2": float(full_metrics["coeff_rel_l2"]),
                    "infer_ms": float(infer_ms),
                    "timing_frames": int(timing_frames),
                    "param_count": int(active_param_count),
                    "param_bytes": int(active_param_bytes),
                    "model_param_count_total": int(improved_total_param_count),
                    "model_param_bytes_total": int(improved_total_param_bytes),
                    "rmse_per_ms": float(efficiency["rmse_per_ms"]),
                    "rmse_per_log_param": float(efficiency["rmse_per_log_param"]),
                    "train_noise_sigma": float(getattr(train_cfg, "noise_sigma", 0.0)),
                    "centered_pod": bool(getattr(eval_cfg, "centered_pod", True)),
                    "group_spec_json": json.dumps(improved_group_spec, ensure_ascii=False),
                    "best_val_loss": float(improved_best_val_by_stage[predict_stage]),
                    "target_baseline_label": str(target_baseline_label),
                    "rmse_ratio_vs_target_baseline": rmse_ratio_vs_target,
                    "infer_ratio_vs_target_baseline": infer_ratio_vs_target,
                    "param_ratio_vs_target_baseline": param_ratio_vs_target,
                    "rmse_ratio_vs_mlp16": rmse_ratio_vs_mlp16,
                    "infer_ratio_vs_mlp16": infer_ratio_vs_mlp16,
                    "param_ratio_vs_mlp16": param_ratio_vs_mlp16,
                }
                rows.append(row)
                print(
                    f"[{improved_model_label}:{predict_stage}] prefix_dim={prefix_dim} "
                    f"prefix_rel_l2={prefix_metrics['coeff_rel_l2']:.4e} full_rel_l2={full_metrics['coeff_rel_l2']:.4e} infer_ms={infer_ms:.3f}"
                )

    output_csv = Path(args.output_csv)
    _append_rows(output_csv, rows)
    print(f"\nSaved {len(rows)} rows to {output_csv}")
    print(f"dataset_spec[mlp]={model_dataset_specs.get('mlp', {})}")
    print(f"dataset_spec[{model_version}]={model_dataset_specs.get('pmrh', model_dataset_specs.get('mlp', {}))}")


if __name__ == "__main__":
    main()