# backend/eval/rebuild.py
# v2.0: Level-2 producer (train + raw predictions) extracted from reconstruction.py
#
# Purpose:
#   - Split out "reconstruction" (raw model outputs) from "analysis/metrics".
#   - Produce and persist Level-2 artifacts:
#       * predicted POD coefficients A_hat_all for each (mask_rate, noise_sigma)
#       * minimal metadata and optional training artifacts (MLP)
#
# Storage layout (draft, consistent with pipeline_train.py):
#   exp_dir/
#     L2_rebuild/
#       meta.json
#       linear/
#         p=..._s=.../
#           pred_coeffs.npz
#           obs.npz
#           entry.json
#       mlp/
#         p=..._s=.../
#           pred_coeffs.npz
#           obs.npz
#           entry.json
#           train.json            (optional, only if train_info exists)
#
# Notes:
#   - Level-2 "raw" output is A_hat_all (predicted POD coefficients).
#   - Spatial models may also persist X_pred_flat_all so L4 examples can use the native
#     full-field prediction instead of the POD-projected surrogate.
#   - All multi-line commentary stays outside function bodies.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import json
import numpy as np

from ..config.schemas import (
    SPATIAL_FIELD_MODEL_TYPES,
    DataConfig,
    PodConfig,
    EvalConfig,
    TrainConfig,
    resolve_model_dataset_specs,
    resolve_model_types_from_train_cfg,
)
from ..pod.compute import build_pod
from ..pod.project import project_to_pod
from ..dataio.loader import load_raw, describe_source
from ..dataio.io_utils import load_numpy, load_json, ensure_dir
from ..sampling.masks import build_structure_importance_map, generate_observation_mask_hw, flatten_mask
from ..sampling.noise import add_gaussian_noise
from ..models.linear_baseline import solve_pod_coeffs_least_squares
from ..models.train import (
    predict_field_model_batch,
    train_field_model_on_observations,
    train_mlp_on_observations,
    train_pmrh_on_observations,
)


# ------------------------------
# helpers: json / npz writers
# ------------------------------

def _save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# ------------------------------
# Level-2 entry npz schema (must stay stable across versions)
# ------------------------------

L2_NPZ_KEYS: Tuple[str, ...] = (
    "A_hat_all",         # float32, [T, r_eff] or [T, ...]
    "A_true_all",        # float32, optional, same shape family as A_hat_all
    "X_pred_flat_all",   # float32, optional, [T, D] native full-field prediction for spatial models
    "mask_flat",         # bool, [D] or [H*W*C]
    "mask_rate",         # float32 scalar (0-d array)
    "noise_sigma",       # float32 scalar (0-d array)
    "centered_pod",      # bool scalar (0-d array)
    "model_type",        # str scalar (0-d array or 1-element)
    "train_info_json",   # str scalar, optional (MLP only)
    "train_cfg_json",    # str scalar, optional (MLP only)
    "projection_residual_nmse",  # float32 scalar, optional (for full-field models projected back to POD)
    "prediction_target", # str scalar, optional
)


# ------------------------------
# key / indexing
# ------------------------------

@dataclass(frozen=True)
class RebuildEntryKey:
    model_type: str   # "linear" | "mlp" | "pmrh" | spatial models
    mask_rate: float
    noise_sigma: float


def _entry_dir(l2_root: Path, key: RebuildEntryKey) -> Path:
    return Path(l2_root)


def _encode_rate(x: float, scale: int = 10000) -> int:
    return int(round(float(x) * int(scale)))


def _entry_filename(key: RebuildEntryKey) -> str:
    p_code = _encode_rate(key.mask_rate)
    s_code = _encode_rate(key.noise_sigma)
    return f"{key.model_type}_p{p_code:04d}_s{s_code:04d}.npz"


def _entry_path(l2_root: Path, key: RebuildEntryKey) -> Path:
    return _entry_dir(l2_root, key) / _entry_filename(key)

# ------------------------------
# POD loading/building (copied in spirit from reconstruction.py)
# ------------------------------

def _load_or_build_pod(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    save_dir = pod_cfg.save_dir
    Ur_path = save_dir / "Ur.npy"
    mean_path = save_dir / "mean_flat.npy"
    meta_path = save_dir / "pod_meta.json"

    if not (Ur_path.exists() and mean_path.exists() and meta_path.exists()):
        if verbose:
            print(f"[rebuild] POD artifacts not found in {save_dir}, building POD...")
        ensure_dir(save_dir)
        build_pod(data_cfg, pod_cfg, verbose=verbose, plot=False)
    else:
        if verbose:
            print(f"[rebuild] Found existing POD in {save_dir}, skip rebuilding.")

    Ur = load_numpy(Ur_path)
    mean_flat = load_numpy(mean_path)
    meta = load_json(meta_path)
    return Ur, mean_flat, meta


def _load_pod_mode_energy_weights(save_dir: Path, *, r_eff: int) -> np.ndarray:
    mode_energy_path = Path(save_dir) / "mode_energy_ratio.npy"
    singular_values_path = Path(save_dir) / "singular_values.npy"

    if mode_energy_path.exists():
        weights = np.asarray(load_numpy(mode_energy_path), dtype=np.float32).reshape(-1)
    elif singular_values_path.exists():
        singular_values = np.asarray(load_numpy(singular_values_path), dtype=np.float32).reshape(-1)
        energy = singular_values ** 2
        total_energy = float(energy.sum()) if energy.size > 0 else 0.0
        if total_energy <= 0.0:
            raise ValueError(f"Invalid singular_values in {singular_values_path}: total energy must be positive.")
        weights = (energy / total_energy).astype(np.float32, copy=False)
    else:
        raise FileNotFoundError(
            f"Cannot find POD energy weights under {save_dir}: expected mode_energy_ratio.npy or singular_values.npy"
        )

    if weights.shape[0] < int(r_eff):
        raise ValueError(f"POD energy weights length {weights.shape[0]} < r_eff={r_eff}")
    return weights[:r_eff].astype(np.float32, copy=False)


def _prepare_snapshots(
    data_cfg: DataConfig,
    Ur: np.ndarray,
    mean_flat: np.ndarray,
    r_eff: int,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(f"[rebuild] Loading full raw data from {describe_source(data_cfg)} ...")

    X_thwc = load_raw(data_cfg)  # [T,H,W,C]
    T, H, W, C = X_thwc.shape
    D = H * W * C

    if Ur.shape[0] != D:
        raise ValueError(f"Ur first dim {Ur.shape[0]} != H*W*C={D}")

    X_flat_all = X_thwc.reshape(T, D)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  -> X_thwc={X_thwc.shape}, flat=[{T},{D}], r_eff={r_eff}")

    A_true = project_to_pod(X_flat_all, Ur_eff, mean_flat)  # [T, r_eff]

    if verbose:
        print("  -> Projected snapshots: A_true shape =", A_true.shape)

    return X_thwc, A_true


def _gather_observations_batch(X_flat_all: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    return X_flat_all[:, mask_flat]


def _make_noisy_observations_batch(
    Y_true: np.ndarray,
    *,
    noise_sigma: float,
    centered_pod: bool,
    mean_masked: np.ndarray,
) -> np.ndarray:
    Y_noisy = add_gaussian_noise(Y_true, sigma=float(noise_sigma))
    if centered_pod:
        Y_noisy = Y_noisy - mean_masked[None, :]
    return Y_noisy


def _predict_coeff_model_batch(
    model_coeff,
    Y_noisy: np.ndarray,
    *,
    device,
    chunk_size: int = 2048,
    predict_stage: str = "full",
) -> np.ndarray:
    import torch

    T = int(Y_noisy.shape[0])
    out: List[np.ndarray] = []
    model_coeff.eval()
    stage_name = str(predict_stage).strip().lower()

    with torch.no_grad():
        for i in range(0, T, int(chunk_size)):
            y_chunk = Y_noisy[i : i + int(chunk_size)]
            y_tensor = torch.from_numpy(y_chunk.astype(np.float32)).to(device)
            if stage_name in ("", "full", "stage3"):
                try:
                    a_chunk = model_coeff(y_tensor).detach().cpu().numpy()
                except TypeError:
                    a_chunk = model_coeff(y_tensor, stage="full").detach().cpu().numpy()
            else:
                a_chunk = model_coeff(y_tensor, stage=stage_name).detach().cpu().numpy()
            out.append(a_chunk)

    return np.concatenate(out, axis=0)


def _project_field_prediction_to_pod(
    X_pred_flat: np.ndarray,
    *,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
) -> tuple[np.ndarray, float]:
    X_pred_flat = np.asarray(X_pred_flat, dtype=np.float32)
    if X_pred_flat.ndim != 2:
        raise ValueError(f"Projected spatial prediction must be [T,D], got {X_pred_flat.shape}")

    A_hat_all = project_to_pod(X_pred_flat, Ur_eff, mean_flat).astype(np.float32, copy=False)
    X_proj_flat = (A_hat_all @ Ur_eff.T) + mean_flat[None, :]

    diff = np.asarray(X_pred_flat, dtype=np.float64) - np.asarray(X_proj_flat, dtype=np.float64)
    denom = np.sum(np.asarray(X_pred_flat, dtype=np.float64) ** 2, axis=1)
    denom = np.maximum(denom, 1e-12)
    residual_nmse = np.mean(np.sum(diff ** 2, axis=1) / denom)
    return A_hat_all, float(residual_nmse)


def _resolve_branch_train_config(
    train_cfg: Optional[TrainConfig],
    *,
    model_type: str,
    mask_rate: float,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "mask_rate": float(mask_rate),
    }
    if train_cfg is None:
        return merged

    merged.update(
        {
            "noise_sigma": float(getattr(train_cfg, "noise_sigma", 0.0)),
            "hidden_dims": tuple(int(v) for v in getattr(train_cfg, "hidden_dims", (256, 256))),
            "lr": float(getattr(train_cfg, "lr", 1e-3)),
            "weight_decay": float(getattr(train_cfg, "weight_decay", 0.0)),
            "use_weighted_loss": bool(getattr(train_cfg, "use_weighted_loss", False)),
            "loss_weighting": str(getattr(train_cfg, "loss_weighting", "none")),
            "loss_weight_power": float(getattr(train_cfg, "loss_weight_power", 1.0)),
            "seed": int(getattr(train_cfg, "seed", 0)),
            "val_ratio": float(getattr(train_cfg, "val_ratio", 0.1)),
            "batch_size": int(getattr(train_cfg, "batch_size", 64)),
            "max_epochs": int(getattr(train_cfg, "max_epochs", 50)),
            "device": str(getattr(train_cfg, "device", "auto")),
            "min_lr": float(getattr(train_cfg, "min_lr", 0.0)),
            "warmup_epochs": int(getattr(train_cfg, "warmup_epochs", 0)),
            "eval_chunk_size": int(getattr(train_cfg, "eval_chunk_size", 2048)),
            "live_line": bool(getattr(train_cfg, "live_line", True)),
            "live_every": int(getattr(train_cfg, "live_every", 1)),
            "conv_window": int(getattr(train_cfg, "conv_window", 25)),
            "conv_slope_thresh": float(getattr(train_cfg, "conv_slope_thresh", -1e-3)),
            "plot_loss": bool(getattr(train_cfg, "plot_loss", False)),
            "plot_path": getattr(train_cfg, "plot_path", None),
            "early_stop": bool(getattr(train_cfg, "early_stop", True)),
            "early_patience": int(getattr(train_cfg, "early_patience", 20)),
            "early_min_delta": float(getattr(train_cfg, "early_min_delta", 0.0)),
            "early_warmup": int(getattr(train_cfg, "early_warmup", 5)),
        }
    )

    model_overrides = dict(getattr(train_cfg, "model_configs", {}) or {}).get(str(model_type), {}) or {}
    merged.update(dict(model_overrides))
    if "hidden_dims" in merged and merged["hidden_dims"] is not None:
        merged["hidden_dims"] = tuple(int(v) for v in merged["hidden_dims"])
    return merged


def _mask_strategy_meta_from_data_cfg(data_cfg: DataConfig) -> Dict[str, Any]:
    strategy = str(getattr(data_cfg, "observation_mask_strategy", "random") or "random")
    kwargs = dict(getattr(data_cfg, "observation_mask_kwargs", {}) or {})
    meta: Dict[str, Any] = {
        "strategy": strategy,
        "seed": int(getattr(data_cfg, "observation_mask_seed", 0)),
    }

    if strategy.strip().lower() in ("radial_spiral", "spiral"):
        meta["spiral_max_radius_frac"] = float(getattr(data_cfg, "observation_spiral_max_radius_frac", 0.875))
    if kwargs:
        meta["kwargs"] = kwargs

    return meta


def _mask_strategy_from_data_cfg(
    data_cfg: DataConfig,
    *,
    X_thwc: np.ndarray | None = None,
    Ur_eff: np.ndarray | None = None,
    H: int | None = None,
    W: int | None = None,
    C: int | None = None,
    pod_cfg: PodConfig | None = None,
) -> tuple[str, Dict[str, Any]]:
    strategy = str(getattr(data_cfg, "observation_mask_strategy", "random") or "random")
    kwargs: Dict[str, Any] = dict(getattr(data_cfg, "observation_mask_kwargs", {}) or {})

    strategy_name = strategy.strip().lower()

    if strategy_name in ("radial_spiral", "spiral"):
        kwargs["max_radius_frac"] = float(getattr(data_cfg, "observation_spiral_max_radius_frac", 0.875))
        return strategy, kwargs

    if strategy_name in ("cylinder_structure_aware", "structure_aware", "region_importance"):
        if X_thwc is None or Ur_eff is None or H is None or W is None or C is None:
            raise ValueError("Structure-aware observation mask requires X_thwc, Ur_eff, H, W, C")

        importance_source = str(kwargs.get("importance_source", "temporal_variance"))
        channel_reduce = str(kwargs.get("importance_channel_reduce", "l2"))
        pod_top_k = int(kwargs.get("importance_top_k", min(16, Ur_eff.shape[1])))

        hotspot_cfg = dict(kwargs.get("hotspot", {}) or {})
        hotspot_center_raw = hotspot_cfg.get("center", None)
        hotspot_center = None if hotspot_center_raw is None else (
            float(hotspot_center_raw[0]),
            float(hotspot_center_raw[1]),
        )
        hotspot_sigma_raw = hotspot_cfg.get("sigma", None)
        hotspot_sigma = None if hotspot_sigma_raw is None else (
            float(hotspot_sigma_raw[0]),
            float(hotspot_sigma_raw[1]),
        )
        hotspot_weight = float(hotspot_cfg.get("weight", 0.0))

        mode_energy_weights = None
        if importance_source.strip().lower() in ("pod_energy", "pod", "pod_modes"):
            if pod_cfg is None:
                raise ValueError("pod_cfg is required when importance_source=pod_energy")
            mode_energy_weights = _load_pod_mode_energy_weights(pod_cfg.save_dir, r_eff=Ur_eff.shape[1])

        kwargs["importance_map"] = build_structure_importance_map(
            H=H,
            W=W,
            C=C,
            data_thwc=X_thwc,
            Ur=Ur_eff,
            mode_energy_weights=mode_energy_weights,
            source=importance_source,
            channel_reduce=channel_reduce,
            pod_top_k=pod_top_k,
            gradient_mix=float(kwargs.get("importance_gradient_mix", 0.0)),
            importance_power=float(kwargs.get("importance_power", 1.0)),
            hotspot_center=hotspot_center,
            hotspot_sigma=hotspot_sigma,
            hotspot_weight=hotspot_weight,
        )

    return strategy, kwargs


"""
Public API: run_rebuild_sweep

Level-2 producer:
  - load/build POD
  - load snapshots, compute A_true
  - for each mask_rate:
      * generate mask
      * (linear) precompute Ur_masked
      * (mlp) train one model for this mask_rate (same as original design)
    then for each noise_sigma:
      * add noise to observations
      * predict A_hat_all
      * save Level-2 entry
"""

def run_rebuild_sweep(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: Optional[TrainConfig],
    *,
    exp_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    l2_root = Path(exp_dir) / "L2"
    ensure_dir(l2_root)

    model_types: Tuple[str, ...] = resolve_model_types_from_train_cfg(train_cfg)

    # POD + snapshots
    Ur, mean_flat, pod_meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = int(pod_meta["H"]), int(pod_meta["W"]), int(pod_meta["C"])
    T = int(pod_meta["T"])
    r_used = int(pod_meta["r_used"])
    r_eff = int(min(int(pod_cfg.r), r_used))
    Ur_eff = Ur[:, :r_eff]
    model_dataset_specs = resolve_model_dataset_specs(data_cfg, num_channels=C)

    if verbose:
        print(f"[rebuild] meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}, r_eff={r_eff}")
        print(f"[rebuild] model_types={model_types}")

    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)
    D = H * W * C
    X_flat_all = X_thwc.reshape(T, D)

    # root meta
    meta = {
        "schema_version": "v2.0",
        "level": 2,
        "root_dirname": "L2",
        "model_types": list(model_types),
        "entry_naming": {
            "layout": "flat",
            "filename": "{model}_p{p_code:04d}_s{s_code:04d}.npz",
            "scale": 10000,
            "p_code": "round(mask_rate * scale)",
            "s_code": "round(noise_sigma * scale)",
        },
        "entry_npz_schema": {
            "format": "npz",
            "keys": list(L2_NPZ_KEYS),
            "required_keys": [
                "A_hat_all",
                "mask_flat",
                "mask_rate",
                "noise_sigma",
                "centered_pod",
                "model_type",
            ],
            "optional_keys": [
                "A_true_all",
                "X_pred_flat_all",
                "train_info_json",
                "train_cfg_json",
            ],
            "notes": {
                "mask_flat": "boolean vector aligned with flattened field dimension (H*W*C).",
                "A_hat_all": "predicted POD coefficients (raw model output, or POD projection of raw spatial prediction).",
                "A_true_all": "optional ground-truth POD coefficients for convenience.",
                "X_pred_flat_all": "optional raw full-field prediction [T,D] for spatial models; examples modules should prefer it when present.",
                "train_info_json": "optional; JSON-encoded training info for MLP.",
            },
        },
        "pod": {
            "save_dir": str(pod_cfg.save_dir),
            "r_eff": int(r_eff),
            "r_used": int(r_used),
            "center": bool(pod_meta.get("center", True)),
            "H": H,
            "W": W,
            "C": C,
            "T": T,
        },
        "eval": {
            "mask_rates": [float(x) for x in getattr(eval_cfg, "mask_rates", [])],
            "noise_sigmas": [float(x) for x in getattr(eval_cfg, "noise_sigmas", [])],
            "centered_pod": bool(getattr(eval_cfg, "centered_pod", False)),
        },
        "model_dataset_specs": model_dataset_specs,
        "observation_mask": _mask_strategy_meta_from_data_cfg(data_cfg),
        "note": "Level-2 raw artifacts: POD coeffs for coeff-regression models, or POD projection of full-field models plus optional raw X_pred_flat_all for examples and projection_residual_nmse for diagnostics.",
    }
    _save_json(l2_root / "meta.json", meta)

    entries_index: List[Dict[str, Any]] = []
    mask_strategy, mask_strategy_kwargs_base = _mask_strategy_from_data_cfg(
        data_cfg,
        X_thwc=X_thwc,
        Ur_eff=Ur_eff,
        H=H,
        W=W,
        C=C,
        pod_cfg=pod_cfg,
    )
    mask_seed = int(getattr(data_cfg, "observation_mask_seed", 0))

    for model_type in model_types:
        model_dataset_spec = dict(model_dataset_specs.get(str(model_type), {}))
        if verbose:
            print(f"\n=== [rebuild-{model_type}] start ===")
            print(f"[rebuild-{model_type}] dataset_spec={model_dataset_spec}")

        for mask_rate in getattr(eval_cfg, "mask_rates", []):
            mask_rate_f = float(mask_rate)
            mask_strategy_kwargs = dict(mask_strategy_kwargs_base)
            if mask_strategy.strip().lower() in ("cylinder_structure_aware", "structure_aware", "region_importance"):
                template_count = max(1, int(mask_strategy_kwargs.get("num_templates", 1)))
                mask_strategy_kwargs["template_index"] = int(mask_strategy_kwargs.get("template_index", _encode_rate(mask_rate_f) % template_count))
            if verbose:
                print(f"[rebuild-{model_type}] mask_rate={mask_rate_f:.6g}")

            mask_hw = generate_observation_mask_hw(
                H,
                W,
                mask_rate=mask_rate_f,
                seed=mask_seed,
                strategy=mask_strategy,
                strategy_kwargs=mask_strategy_kwargs,
            )
            mask_flat = flatten_mask(mask_hw, C=C)
            n_obs = int(mask_flat.sum())

            if verbose:
                extra = ""
                if "template_index" in mask_strategy_kwargs:
                    extra = f", template_index={int(mask_strategy_kwargs['template_index'])}"
                print(f"  -> mask_strategy={mask_strategy}, n_obs={n_obs} (with C={C}){extra}")

            Y_true = _gather_observations_batch(X_flat_all, mask_flat)
            mean_masked = mean_flat[mask_flat]

            # per-mask precompute / train
            train_art = train_model_for_entry(
                model_type=model_type,
                data_cfg=data_cfg,
                pod_cfg=pod_cfg,
                train_cfg=train_cfg,
                centered_pod=bool(getattr(eval_cfg, "centered_pod", True)),
                mask_rate=mask_rate_f,
                noise_sigma=float(min(getattr(eval_cfg, "noise_sigmas", [0.0]))),
                verbose=verbose,
                model_dataset_spec=model_dataset_spec,
                mask_hw=mask_hw,
                spatial_shape=(H, W, C),
                mask_flat=mask_flat,
                Ur_eff=Ur_eff,
                X_flat_all=X_flat_all,
                mean_flat=mean_flat,
            )

            for noise_sigma in getattr(eval_cfg, "noise_sigmas", []):
                noise_sigma_f = float(noise_sigma)
                if verbose:
                    print(f"  [rebuild-{model_type}] noise_sigma={noise_sigma_f:.4e}")

                pred = predict_coeffs_for_entry(
                    model_type=model_type,
                    data_cfg=data_cfg,
                    pod_cfg=pod_cfg,
                    eval_cfg=eval_cfg,
                    train_artifacts=train_art,
                    mask_rate=mask_rate_f,
                    noise_sigma=noise_sigma_f,
                    verbose=verbose,
                    Y_true=Y_true,
                    mean_masked=mean_masked,
                    A_true=A_true,
                    mask_hw=mask_hw,
                    mask_flat=mask_flat,
                    Ur_eff=Ur_eff,
                    mean_flat=mean_flat,
                )

                key = RebuildEntryKey(model_type=str(model_type), mask_rate=mask_rate_f, noise_sigma=noise_sigma_f)
                out_path = _entry_path(l2_root, key)

                save_level2_entry(
                    out_path=out_path,
                    pred=pred,
                    train_artifacts=train_art,
                    verbose=verbose,
                )

                entries_index.append(
                    {
                        "key": {"model_type": key.model_type, "mask_rate": key.mask_rate, "noise_sigma": key.noise_sigma},
                        "path": str(out_path),
                    }
                )

        if verbose:
            print(f"=== [rebuild-{model_type}] done ===")

    return {
        "root": str(l2_root),
        "meta": meta,
        "entries": entries_index,
        "model_types": model_types,
    }


"""
train_model_for_entry

In original reconstruction.py:
  - linear: per-mask precompute Ur_masked
  - mlp: per-mask train one model using train_cfg.noise_sigma (train noise), then reuse for all test noise

We keep that: training happens per mask_rate, not per (mask_rate, noise_sigma).
"""

def train_model_for_entry(
    *,
    model_type: str,
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: Optional[TrainConfig],
    centered_pod: bool,
    mask_rate: float,
    noise_sigma: float,
    verbose: bool = True,
    model_dataset_spec: Dict[str, Any] | None,
    mask_hw: np.ndarray | None,
    spatial_shape: tuple[int, int, int] | None,
    mask_flat: np.ndarray,
    Ur_eff: np.ndarray,
    X_flat_all: np.ndarray,
    mean_flat: np.ndarray,
) -> Dict[str, Any]:
    branch_cfg = _resolve_branch_train_config(train_cfg, model_type=model_type, mask_rate=mask_rate)

    # linear: cache Ur_masked for least squares
    if str(model_type) == "linear":
        Ur_masked = Ur_eff[mask_flat, :]  # [n_obs, r_eff]
        return {
            "model_type": "linear",
            "model_dataset_spec": dict(model_dataset_spec or {}),
            "model_task": "pod_coeff_regression",
            "Ur_masked": Ur_masked,
            "train_info": None,
            "train_cfg": branch_cfg,
        }

    # mlp: train one model per mask_rate (same logic as reconstruction backend)
    if str(model_type) == "mlp":
        if train_cfg is None:
            raise ValueError("train_cfg is required for model_type='mlp'")

        use_weighted_loss = bool(branch_cfg.get("use_weighted_loss", False))
        effective_loss_weighting = (
            str(branch_cfg.get("loss_weighting", "none")) if use_weighted_loss else "none"
        )
        coeff_loss_weights = None
        if effective_loss_weighting.strip().lower() == "pod_energy":
            coeff_loss_weights = _load_pod_mode_energy_weights(pod_cfg.save_dir, r_eff=int(Ur_eff.shape[1]))

        if verbose:
            print(
                f"  -> training MLP: train_noise_sigma={float(branch_cfg.get('noise_sigma', 0.0)):.4e}, "
                f"batch_size={int(branch_cfg.get('batch_size', 64))}, max_epochs={int(branch_cfg.get('max_epochs', 50))}, lr={float(branch_cfg.get('lr', 1e-3)):.3g}"
            )
            print(f"     use_weighted_loss: {use_weighted_loss}")
            if effective_loss_weighting.strip().lower() == "pod_energy" and coeff_loss_weights is not None:
                print(
                    f"     loss weighting: pod_energy, first weights="
                    f"{[float(x) for x in coeff_loss_weights[: min(5, coeff_loss_weights.shape[0])]]}"
                )

        model_mlp, train_info = train_mlp_on_observations(
            X_flat_all=X_flat_all,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
            mask_flat=mask_flat,
            noise_sigma=float(branch_cfg.get("noise_sigma", 0.0)),
            coeff_loss_weights=coeff_loss_weights,
            loss_weighting=effective_loss_weighting,
            loss_weight_power=float(branch_cfg.get("loss_weight_power", 1.0)),
            hidden_dims=tuple(int(v) for v in branch_cfg.get("hidden_dims", (256, 256))),
            batch_size=int(branch_cfg.get("batch_size", 64)),
            num_epochs=int(branch_cfg.get("max_epochs", 50)),
            lr=float(branch_cfg.get("lr", 1e-3)),
            weight_decay=float(branch_cfg.get("weight_decay", 0.0)),
            val_ratio=float(branch_cfg.get("val_ratio", 0.1)),
            device=str(branch_cfg.get("device", "auto")),
            centered_pod=bool(centered_pod),
            verbose=verbose,
            live_line=bool(branch_cfg.get("live_line", True)),
            live_every=int(branch_cfg.get("live_every", 1)),
            conv_window=int(branch_cfg.get("conv_window", 25)),
            conv_slope_thresh=float(branch_cfg.get("conv_slope_thresh", -1e-3)),
            plot_loss=bool(branch_cfg.get("plot_loss", False)),
            plot_path=branch_cfg.get("plot_path", None),
            early_stop=bool(branch_cfg.get("early_stop", True)),
            early_patience=int(branch_cfg.get("early_patience", 20)),
            early_min_delta=float(branch_cfg.get("early_min_delta", 0.0)),
            early_warmup=int(branch_cfg.get("early_warmup", 5)),
            seed=int(branch_cfg.get("seed", 0)),
            max_train_batches=(None if branch_cfg.get("max_train_batches", None) is None else int(branch_cfg.get("max_train_batches"))),
            max_val_batches=(None if branch_cfg.get("max_val_batches", None) is None else int(branch_cfg.get("max_val_batches"))),
        )

        model_mlp.eval()
        device = next(model_mlp.parameters()).device

        eval_chunk_size = int(branch_cfg.get("eval_chunk_size", 2048))

        return {
            "model_type": "mlp",
            "model_dataset_spec": dict(model_dataset_spec or {}),
            "model_task": "pod_coeff_regression",
            "model_mlp": model_mlp,
            "device": device,
            "eval_chunk_size": eval_chunk_size,
            "train_info": train_info,
            "train_cfg": {
                **branch_cfg,
                "mask_rate": float(branch_cfg.get("mask_rate", mask_rate)),
                "hidden_dims": [int(v) for v in branch_cfg.get("hidden_dims", ())],
                "use_weighted_loss": bool(use_weighted_loss),
                "loss_weighting": str(effective_loss_weighting),
                "eval_chunk_size": int(eval_chunk_size),
                "centered_pod": bool(centered_pod),
            },
        }

    if str(model_type) == "pmrh":
        if train_cfg is None:
            raise ValueError("train_cfg is required for model_type='pmrh'")

        total_epochs = int(branch_cfg.get("max_epochs", 200))
        phase1_epochs = branch_cfg.get("phase1_epochs", None)
        phase2_epochs = branch_cfg.get("phase2_epochs", None)
        phase3_epochs = branch_cfg.get("phase3_epochs", None)
        finetune_epochs = branch_cfg.get("finetune_epochs", None)
        if phase1_epochs is None and phase2_epochs is None and phase3_epochs is None and finetune_epochs is None:
            phase1_epochs = max(1, int(round(total_epochs * 0.2)))
            phase2_epochs = max(1, int(round(total_epochs * 0.2)))
            phase3_epochs = max(1, int(round(total_epochs * 0.4)))
            finetune_epochs = max(1, total_epochs - int(phase1_epochs) - int(phase2_epochs) - int(phase3_epochs))
        else:
            phase1_epochs = int(0 if phase1_epochs is None else phase1_epochs)
            phase2_epochs = int(0 if phase2_epochs is None else phase2_epochs)
            phase3_epochs = int(0 if phase3_epochs is None else phase3_epochs)
            finetune_epochs = int(0 if finetune_epochs is None else finetune_epochs)

        if verbose:
            print(
                f"  -> training PMRH: train_noise_sigma={float(branch_cfg.get('noise_sigma', 0.0)):.4e}, "
                f"batch_size={int(branch_cfg.get('batch_size', 64))}, lr={float(branch_cfg.get('lr', 1e-3)):.3g}, "
                f"phase_epochs=({int(phase1_epochs)}, {int(phase2_epochs)}, {int(phase3_epochs)}, {int(finetune_epochs)})"
            )

        model_pmrh, train_info = train_pmrh_on_observations(
            X_flat_all=X_flat_all,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
            mask_flat=mask_flat,
            noise_sigma=float(branch_cfg.get("noise_sigma", 0.0)),
            coarse_hidden_dims=tuple(int(v) for v in branch_cfg.get("coarse_hidden_dims", branch_cfg.get("trunk_hidden_dims", (64, 64)))),
            stage2_feature_dim=int(branch_cfg.get("stage2_feature_dim", branch_cfg.get("stage2_hidden_dim", 96))),
            stage2_head_hidden_dim=int(branch_cfg.get("stage2_head_hidden_dim", branch_cfg.get("stage2_hidden_dim", 96))),
            stage3_feature_dim=int(branch_cfg.get("stage3_feature_dim", branch_cfg.get("stage3_hidden_dim", 128))),
            stage3_head_hidden_dim=int(branch_cfg.get("stage3_head_hidden_dim", branch_cfg.get("stage3_hidden_dim", 128))),
            group_ratios=tuple(int(v) for v in branch_cfg.get("group_ratios", (1, 2, 5))),
            stage1_low_rank=(None if branch_cfg.get("stage1_low_rank", None) is None else int(branch_cfg.get("stage1_low_rank"))),
            stage_loss_weights=tuple(float(v) for v in branch_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0))),
            consistency_weight=float(branch_cfg.get("consistency_weight", 0.05)),
            budget_weight=float(branch_cfg.get("budget_weight", 1e-4)),
            phase1_epochs=int(phase1_epochs),
            phase2_epochs=int(phase2_epochs),
            phase3_epochs=int(phase3_epochs),
            finetune_epochs=int(finetune_epochs),
            stage2_freeze_epochs=int(branch_cfg.get("stage2_freeze_epochs", 3)),
            stage3_freeze_epochs=int(branch_cfg.get("stage3_freeze_epochs", 3)),
            batch_size=int(branch_cfg.get("batch_size", 64)),
            lr=float(branch_cfg.get("lr", 1e-3)),
            weight_decay=float(branch_cfg.get("weight_decay", 0.0)),
            val_ratio=float(branch_cfg.get("val_ratio", 0.1)),
            device=str(branch_cfg.get("device", "auto")),
            centered_pod=bool(centered_pod),
            verbose=verbose,
            live_line=bool(branch_cfg.get("live_line", True)),
            live_every=int(branch_cfg.get("live_every", 1)),
            conv_window=int(branch_cfg.get("conv_window", 25)),
            conv_slope_thresh=float(branch_cfg.get("conv_slope_thresh", -1e-3)),
            plot_loss=bool(branch_cfg.get("plot_loss", False)),
            plot_path=branch_cfg.get("plot_path", None),
            seed=int(branch_cfg.get("seed", 0)),
            max_train_batches=(None if branch_cfg.get("max_train_batches", None) is None else int(branch_cfg.get("max_train_batches"))),
            max_val_batches=(None if branch_cfg.get("max_val_batches", None) is None else int(branch_cfg.get("max_val_batches"))),
        )

        model_pmrh.eval()
        device = next(model_pmrh.parameters()).device
        eval_chunk_size = int(branch_cfg.get("eval_chunk_size", 2048))

        return {
            "model_type": "pmrh",
            "model_dataset_spec": dict(model_dataset_spec or {}),
            "model_task": "pod_coeff_regression",
            "model_pmrh": model_pmrh,
            "device": device,
            "eval_chunk_size": eval_chunk_size,
            "predict_stage": str(branch_cfg.get("predict_stage", "full")),
            "train_info": train_info,
            "train_cfg": {
                **branch_cfg,
                "mask_rate": float(branch_cfg.get("mask_rate", mask_rate)),
                "coarse_hidden_dims": [int(v) for v in branch_cfg.get("coarse_hidden_dims", branch_cfg.get("trunk_hidden_dims", (64, 64)))],
                "stage2_feature_dim": int(branch_cfg.get("stage2_feature_dim", branch_cfg.get("stage2_hidden_dim", 96))),
                "stage2_head_hidden_dim": int(branch_cfg.get("stage2_head_hidden_dim", branch_cfg.get("stage2_hidden_dim", 96))),
                "stage3_feature_dim": int(branch_cfg.get("stage3_feature_dim", branch_cfg.get("stage3_hidden_dim", 128))),
                "stage3_head_hidden_dim": int(branch_cfg.get("stage3_head_hidden_dim", branch_cfg.get("stage3_hidden_dim", 128))),
                "stage1_low_rank": (None if branch_cfg.get("stage1_low_rank", None) is None else int(branch_cfg.get("stage1_low_rank"))),
                "group_ratios": [int(v) for v in branch_cfg.get("group_ratios", (1, 2, 5))],
                "stage_loss_weights": [float(v) for v in branch_cfg.get("stage_loss_weights", (1.0, 1.0, 1.0))],
                "consistency_weight": float(branch_cfg.get("consistency_weight", 0.05)),
                "budget_weight": float(branch_cfg.get("budget_weight", 1e-4)),
                "phase1_epochs": int(phase1_epochs),
                "phase2_epochs": int(phase2_epochs),
                "phase3_epochs": int(phase3_epochs),
                "finetune_epochs": int(finetune_epochs),
                "stage2_freeze_epochs": int(branch_cfg.get("stage2_freeze_epochs", 3)),
                "stage3_freeze_epochs": int(branch_cfg.get("stage3_freeze_epochs", 3)),
                "eval_chunk_size": int(eval_chunk_size),
                "centered_pod": bool(centered_pod),
            },
        }

    if str(model_type) in SPATIAL_FIELD_MODEL_TYPES:
        if train_cfg is None:
            raise ValueError(f"train_cfg is required for model_type='{model_type}'")
        if mask_hw is None:
            raise ValueError(f"Spatial model '{model_type}' requires mask_hw.")
        if spatial_shape is None:
            raise ValueError(f"Spatial model '{model_type}' requires spatial_shape=(H,W,C).")

        if verbose:
            print(
                f"  -> training {model_type}: train_noise_sigma={float(branch_cfg.get('noise_sigma', 0.0)):.4e}, "
                f"batch_size={int(branch_cfg.get('batch_size', 16))}, max_epochs={int(branch_cfg.get('max_epochs', 20))}, lr={float(branch_cfg.get('lr', 1e-3)):.3g}"
            )

        n_frames = int(X_flat_all.shape[0])
        D_full = int(X_flat_all.shape[1])
        H, W, C = [int(v) for v in spatial_shape]
        if H * W * C != D_full:
            raise ValueError(f"Invalid spatial_shape={spatial_shape}: H*W*C={H*W*C} != D={D_full}")
        X_thwc_local = X_flat_all.reshape(n_frames, H, W, C)

        model_spatial, train_info, spatial_artifacts = train_field_model_on_observations(
            model_type=str(model_type),
            X_thwc=X_thwc_local,
            mask_hw=np.asarray(mask_hw, dtype=bool),
            noise_sigma=float(branch_cfg.get("noise_sigma", 0.0)),
            model_dataset_spec=dict(model_dataset_spec or {}),
            model_cfg=branch_cfg,
            batch_size=int(branch_cfg.get("batch_size", 16)),
            num_epochs=int(branch_cfg.get("max_epochs", 20)),
            lr=float(branch_cfg.get("lr", 1e-3)),
            weight_decay=float(branch_cfg.get("weight_decay", 0.0)),
            val_ratio=float(branch_cfg.get("val_ratio", 0.1)),
            device=str(branch_cfg.get("device", "auto")),
            verbose=verbose,
            live_line=bool(branch_cfg.get("live_line", True)),
            live_every=int(branch_cfg.get("live_every", 1)),
            conv_window=int(branch_cfg.get("conv_window", 25)),
            conv_slope_thresh=float(branch_cfg.get("conv_slope_thresh", -1e-3)),
            plot_loss=bool(branch_cfg.get("plot_loss", False)),
            plot_path=branch_cfg.get("plot_path", None),
            early_stop=bool(branch_cfg.get("early_stop", True)),
            early_patience=int(branch_cfg.get("early_patience", 20)),
            early_min_delta=float(branch_cfg.get("early_min_delta", 0.0)),
            early_warmup=int(branch_cfg.get("early_warmup", 5)),
            min_lr=float(branch_cfg.get("min_lr", 0.0)),
            warmup_epochs=int(branch_cfg.get("warmup_epochs", 10)),
            use_cosine_schedule=bool(branch_cfg.get("use_cosine_schedule", True)),
            seed=int(branch_cfg.get("seed", 0)),
            max_train_batches=(None if branch_cfg.get("max_train_batches", None) is None else int(branch_cfg.get("max_train_batches"))),
            max_val_batches=(None if branch_cfg.get("max_val_batches", None) is None else int(branch_cfg.get("max_val_batches"))),
        )

        device_name = str(branch_cfg.get("device", "auto"))
        eval_batch_size = int(branch_cfg.get("eval_batch_size", branch_cfg.get("batch_size", 16)))

        def _predict_field_batch_fn(*, noise_sigma: float, mask_hw: np.ndarray, mask_flat: np.ndarray, centered_pod: bool) -> np.ndarray:
            _ = mask_flat, centered_pod
            return predict_field_model_batch(
                model_type=str(model_type),
                model=model_spatial,
                X_thwc=X_thwc_local,
                mask_hw=np.asarray(mask_hw, dtype=bool),
                nearest_idx_hw=spatial_artifacts["nearest_idx_hw"],
                representation=str(spatial_artifacts["representation"]),
                include_mask_channel=bool(spatial_artifacts["include_mask_channel"]),
                pad_hw=spatial_artifacts["pad_hw"],
                noise_sigma=float(noise_sigma),
                device=device_name,
                batch_size=eval_batch_size,
                norm_mean_c=spatial_artifacts.get("norm_mean_c", None),
                norm_std_c=spatial_artifacts.get("norm_std_c", None),
            )

        return {
            "model_type": str(model_type),
            "model_task": "spatial_field_reconstruction",
            "model_dataset_spec": dict(model_dataset_spec or {}),
            "model_spatial": model_spatial,
            "predict_field_batch_fn": _predict_field_batch_fn,
            "device": device_name,
            "train_info": train_info,
            "train_cfg": {
                **branch_cfg,
                "mask_rate": float(branch_cfg.get("mask_rate", mask_rate)),
                "spatial_shape": [H, W, C],
                "eval_batch_size": int(eval_batch_size),
                "input_representation": str(spatial_artifacts["representation"]),
                "include_mask_channel": bool(spatial_artifacts["include_mask_channel"]),
                "patch_size": int(spatial_artifacts["patch_size"]),
                "in_channels": int(spatial_artifacts["in_channels"]),
                "out_channels": int(spatial_artifacts["out_channels"]),
                "normalize_mean_std": bool(branch_cfg.get("normalize_mean_std", True)),
                "optimizer": str(branch_cfg.get("optimizer", "adamw")),
                "use_cosine_schedule": bool(branch_cfg.get("use_cosine_schedule", True)),
            },
        }

    raise ValueError(f"Unsupported model_type='{model_type}'")


"""
predict_coeffs_for_entry

Computes A_hat_all for a specific (mask_rate, noise_sigma) given per-mask train_artifacts.
This is the "raw reconstruction output" saved as Level-2 artifact.
"""

def predict_coeffs_for_entry(
    *,
    model_type: str,
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_artifacts: Optional[Dict[str, Any]],
    mask_rate: float,
    noise_sigma: float,
    verbose: bool = True,
    Y_true: np.ndarray,
    mean_masked: np.ndarray,
    A_true: np.ndarray,
    mask_hw: np.ndarray,
    mask_flat: np.ndarray,
    Ur_eff: np.ndarray,
    mean_flat: np.ndarray,
) -> Dict[str, Any]:
    centered_pod = bool(getattr(eval_cfg, "centered_pod", False))

    # noisy observations
    Y_noisy = _make_noisy_observations_batch(
        Y_true,
        noise_sigma=float(noise_sigma),
        centered_pod=centered_pod,
        mean_masked=mean_masked,
    )

    # predict coeffs
    if str(model_type) == "linear":
        if train_artifacts is None or "Ur_masked" not in train_artifacts:
            raise ValueError("linear predict requires train_artifacts['Ur_masked']")
        Ur_masked = train_artifacts["Ur_masked"]
        A_hat_all = solve_pod_coeffs_least_squares(Y_noisy, Ur_masked)  # [T, r_eff]
        projection_residual_nmse = 0.0
    elif str(model_type) == "mlp":
        if train_artifacts is None or "model_mlp" not in train_artifacts:
            raise ValueError("mlp predict requires train_artifacts['model_mlp']")
        model_mlp = train_artifacts["model_mlp"]
        device = train_artifacts["device"]
        chunk_size = int(train_artifacts.get("eval_chunk_size", 2048))
        A_hat_all = _predict_coeff_model_batch(model_mlp, Y_noisy, device=device, chunk_size=chunk_size)
        projection_residual_nmse = 0.0
    elif str(model_type) == "pmrh":
        if train_artifacts is None or "model_pmrh" not in train_artifacts:
            raise ValueError("pmrh predict requires train_artifacts['model_pmrh']")
        model_pmrh = train_artifacts["model_pmrh"]
        device = train_artifacts["device"]
        chunk_size = int(train_artifacts.get("eval_chunk_size", 2048))
        predict_stage = str(train_artifacts.get("predict_stage", "full"))
        A_hat_all = _predict_coeff_model_batch(
            model_pmrh,
            Y_noisy,
            device=device,
            chunk_size=chunk_size,
            predict_stage=predict_stage,
        )
        projection_residual_nmse = 0.0
    elif str(model_type) in SPATIAL_FIELD_MODEL_TYPES:
        if train_artifacts is None or train_artifacts.get("predict_field_batch_fn", None) is None:
            raise NotImplementedError(
                f"Spatial model '{model_type}' requires train_artifacts['predict_field_batch_fn'] to return predicted fields."
            )
        field_predict_fn = train_artifacts["predict_field_batch_fn"]
        X_pred = field_predict_fn(
            noise_sigma=float(noise_sigma),
            mask_hw=np.asarray(mask_hw, dtype=bool),
            mask_flat=np.asarray(mask_flat, dtype=bool),
            centered_pod=bool(centered_pod),
        )
        X_pred = np.asarray(X_pred, dtype=np.float32)
        if X_pred.ndim == 4:
            X_pred = X_pred.reshape(X_pred.shape[0], -1)
        A_hat_all, projection_residual_nmse = _project_field_prediction_to_pod(
            X_pred,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
        )
        X_pred_flat_all = np.asarray(X_pred, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported model_type='{model_type}'")

    # pack outputs (keep compact)
    payload = {
        "model_type": str(model_type),
        "mask_rate": float(mask_rate),
        "noise_sigma": float(noise_sigma),
        "centered_pod": bool(centered_pod),
        "A_hat_all": np.asarray(A_hat_all, dtype=np.float32),
        "A_true_all": np.asarray(A_true, dtype=np.float32),
        "mask_hw": np.asarray(mask_hw, dtype=bool),
        "mask_flat": np.asarray(mask_flat, dtype=bool),
        "prediction_target": str(
            "full_field_projected_to_pod" if str(model_type) in SPATIAL_FIELD_MODEL_TYPES else "pod_coefficients"
        ),
        "projection_residual_nmse": float(projection_residual_nmse),
    }
    if str(model_type) in SPATIAL_FIELD_MODEL_TYPES:
        payload["X_pred_flat_all"] = X_pred_flat_all
    return payload


"""
save_level2_entry

Writes Level-2 artifact files for one entry directory.
"""

def save_level2_entry(
    *,
    out_path: Path,
    pred: Dict[str, Any],
    train_artifacts: Optional[Dict[str, Any]],
    verbose: bool = True,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # core arrays
    A_hat_all = np.asarray(pred["A_hat_all"], dtype=np.float32)
    A_true_all = pred.get("A_true_all", None)
    if A_true_all is not None:
        A_true_all = np.asarray(A_true_all, dtype=np.float32)

    mask_flat = np.asarray(pred["mask_flat"], dtype=bool)
    centered_pod = bool(pred.get("centered_pod", False))

    # store small scalars as 0-d arrays (portable in npz)
    mask_rate = np.asarray(float(pred.get("mask_rate", 0.0)), dtype=np.float32)
    noise_sigma = np.asarray(float(pred.get("noise_sigma", 0.0)), dtype=np.float32)
    centered_pod_arr = np.asarray(centered_pod, dtype=bool)

    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return value

    # optional: train_info as json string
    train_info_json = None
    train_cfg_json = None
    if train_artifacts is not None and train_artifacts.get("train_info", None) is not None:
        train_info_json = json.dumps(_json_safe(train_artifacts["train_info"]), ensure_ascii=False)
        train_cfg_json = json.dumps(_json_safe(train_artifacts.get("train_cfg", None)), ensure_ascii=False)

    projection_residual_nmse = pred.get("projection_residual_nmse", None)
    prediction_target = pred.get("prediction_target", None)

    # write one npz
    payload = {
        "A_hat_all": A_hat_all,
        "mask_flat": mask_flat,
        "mask_rate": mask_rate,
        "noise_sigma": noise_sigma,
        "centered_pod": centered_pod_arr,
        "model_type": np.asarray(str(pred.get("model_type", ""))),
    }
    if A_true_all is not None:
        payload["A_true_all"] = A_true_all
    if pred.get("X_pred_flat_all", None) is not None:
        payload["X_pred_flat_all"] = np.asarray(pred["X_pred_flat_all"], dtype=np.float32)
    if train_info_json is not None:
        payload["train_info_json"] = np.asarray(train_info_json)
    if train_cfg_json is not None:
        payload["train_cfg_json"] = np.asarray(train_cfg_json)
    if projection_residual_nmse is not None:
        payload["projection_residual_nmse"] = np.asarray(float(projection_residual_nmse), dtype=np.float32)
    if prediction_target is not None:
        payload["prediction_target"] = np.asarray(str(prediction_target))

    np.savez_compressed(out_path, **payload)

    if verbose:
        print(f"    -> saved L2 entry: {out_path}")

"""
Level-2 entry loader / summarizer / plotter
"""
def load_l2_npz(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    data = np.load(path, allow_pickle=True)

    out: Dict[str, Any] = {}
    for k in data.files:
        v = data[k]
        # np.load may give 0-d arrays for scalars/strings; unwrap them
        if isinstance(v, np.ndarray) and v.shape == ():
            out[k] = v.item()
        else:
            out[k] = v
    return out


def summarize_l2_npz(path: str | Path, *, max_array_elems: int = 8) -> Dict[str, Any]:
    path = Path(path)
    d = load_l2_npz(path)

    summary: Dict[str, Any] = {"path": str(path), "keys": list(d.keys()), "items": {}}

    for k, v in d.items():
        if isinstance(v, np.ndarray):
            info = {"type": "ndarray", "dtype": str(v.dtype), "shape": tuple(v.shape)}
            if v.size <= max_array_elems:
                info["value"] = v
            summary["items"][k] = info
        else:
            summary["items"][k] = {"type": type(v).__name__, "value": v}

    return summary


def plot_l2_coeffs_preview(
    path: str | Path,
    *,
    coeff_idx: int = 0,
    show_true: bool = True,
    max_T: Optional[int] = None,
):
    import matplotlib.pyplot as plt

    d = load_l2_npz(path)
    A_hat = d["A_hat_all"]
    A_true = d.get("A_true_all", None)

    if A_hat.ndim != 2:
        raise ValueError(f"A_hat_all expected 2D [T, r], got shape {A_hat.shape}")

    T, r = A_hat.shape
    i = int(np.clip(coeff_idx, 0, r - 1))

    t = np.arange(T)
    if max_T is not None:
        t = t[: int(max_T)]
    y_hat = A_hat[: len(t), i]

    plt.figure()
    plt.plot(t, y_hat, label="A_hat")

    if show_true and A_true is not None:
        if A_true.shape != A_hat.shape:
            raise ValueError(f"A_true_all shape {A_true.shape} != A_hat_all shape {A_hat.shape}")
        y_true = A_true[: len(t), i]
        plt.plot(t, y_true, label="A_true")

    plt.xlabel("t (frame index)")
    plt.ylabel(f"coeff[{i}]")
    title = f"{Path(path).name}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt.gcf()
