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
#   - Level-2 "raw" output is A_hat_all (predicted POD coefficients), not full field X_hat_thwc.
#     X_hat is linearly reproducible from A_hat_all + (Ur, mean_flat), so we keep L2 compact.
#   - All multi-line commentary stays outside function bodies.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import json
import numpy as np

from ..config.schemas import DataConfig, PodConfig, EvalConfig, TrainConfig
from ..pod.compute import build_pod
from ..pod.project import project_to_pod
from ..dataio.nc_loader import load_raw_nc
from ..dataio.io_utils import load_numpy, load_json, ensure_dir
from ..sampling.masks import generate_random_mask_hw, flatten_mask
from ..sampling.noise import add_gaussian_noise
from ..models.linear_baseline import solve_pod_coeffs_least_squares
from ..models.train_mlp import train_mlp_on_observations


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
    "mask_flat",         # bool, [D] or [H*W*C]
    "mask_rate",         # float32 scalar (0-d array)
    "noise_sigma",       # float32 scalar (0-d array)
    "centered_pod",      # bool scalar (0-d array)
    "model_type",        # str scalar (0-d array or 1-element)
    "train_info_json",   # str scalar, optional (MLP only)
    "train_cfg_json",    # str scalar, optional (MLP only)
)


# ------------------------------
# key / indexing
# ------------------------------

@dataclass(frozen=True)
class RebuildEntryKey:
    model_type: str   # "linear" | "mlp"
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


def _prepare_snapshots(
    data_cfg: DataConfig,
    Ur: np.ndarray,
    mean_flat: np.ndarray,
    r_eff: int,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if verbose:
        print(f"[rebuild] Loading full raw data from {data_cfg.nc_path} ...")

    X_thwc = load_raw_nc(data_cfg)  # [T,H,W,C]
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


def _predict_mlp_coeffs_batch(model_mlp, Y_noisy: np.ndarray, *, device, chunk_size: int = 2048) -> np.ndarray:
    import torch

    T = int(Y_noisy.shape[0])
    out: List[np.ndarray] = []
    model_mlp.eval()

    with torch.no_grad():
        for i in range(0, T, int(chunk_size)):
            y_chunk = Y_noisy[i : i + int(chunk_size)]
            y_tensor = torch.from_numpy(y_chunk.astype(np.float32)).to(device)
            a_chunk = model_mlp(y_tensor).detach().cpu().numpy()
            out.append(a_chunk)

    return np.concatenate(out, axis=0)


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

    model_types: Tuple[str, ...] = ("linear",) if train_cfg is None else ("linear", "mlp")

    # POD + snapshots
    Ur, mean_flat, pod_meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = int(pod_meta["H"]), int(pod_meta["W"]), int(pod_meta["C"])
    T = int(pod_meta["T"])
    r_used = int(pod_meta["r_used"])
    r_eff = int(min(int(pod_cfg.r), r_used))
    Ur_eff = Ur[:, :r_eff]

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
                "train_info_json",
                "train_cfg_json",
            ],
            "notes": {
                "mask_flat": "boolean vector aligned with flattened field dimension (H*W*C).",
                "A_hat_all": "predicted POD coefficients (raw model output).",
                "A_true_all": "optional ground-truth POD coefficients for convenience.",
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
        "note": "Level-2 raw artifacts: predicted POD coeffs A_hat_all (+ minimal reconstruction dependencies).",
    }
    _save_json(l2_root / "meta.json", meta)

    entries_index: List[Dict[str, Any]] = []

    for model_type in model_types:
        if verbose:
            print(f"\n=== [rebuild-{model_type}] start ===")

        for mask_rate in getattr(eval_cfg, "mask_rates", []):
            mask_rate_f = float(mask_rate)
            if verbose:
                print(f"[rebuild-{model_type}] mask_rate={mask_rate_f:.6g}")

            mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate_f, seed=0)
            mask_flat = flatten_mask(mask_hw, C=C)
            n_obs = int(mask_flat.sum())

            if verbose:
                print(f"  -> n_obs={n_obs} (with C={C})")

            Y_true = _gather_observations_batch(X_flat_all, mask_flat)
            mean_masked = mean_flat[mask_flat]

            # per-mask precompute / train
            train_art = train_model_for_entry(
                model_type=model_type,
                data_cfg=data_cfg,
                pod_cfg=pod_cfg,
                train_cfg=train_cfg,
                mask_rate=mask_rate_f,
                noise_sigma=float(min(getattr(eval_cfg, "noise_sigmas", [0.0]))),
                verbose=verbose,
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
    mask_rate: float,
    noise_sigma: float,
    verbose: bool = True,
    mask_flat: np.ndarray,
    Ur_eff: np.ndarray,
    X_flat_all: np.ndarray,
    mean_flat: np.ndarray,
) -> Dict[str, Any]:
    # linear: cache Ur_masked for least squares
    if str(model_type) == "linear":
        Ur_masked = Ur_eff[mask_flat, :]  # [n_obs, r_eff]
        return {
            "model_type": "linear",
            "Ur_masked": Ur_masked,
            "train_info": None,
        }

    # mlp: train one model per mask_rate (same logic as reconstruction backend)
    if str(model_type) == "mlp":
        if train_cfg is None:
            raise ValueError("train_cfg is required for model_type='mlp'")

        if verbose:
            print(
                f"  -> training MLP: train_noise_sigma={float(train_cfg.noise_sigma):.4e}, "
                f"batch_size={int(train_cfg.batch_size)}, max_epochs={int(train_cfg.max_epochs)}, lr={float(train_cfg.lr):.3g}"
            )

        model_mlp, train_info = train_mlp_on_observations(
            X_flat_all=X_flat_all,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
            mask_flat=mask_flat,
            noise_sigma=float(train_cfg.noise_sigma),
            batch_size=int(train_cfg.batch_size),
            num_epochs=int(train_cfg.max_epochs),
            lr=float(train_cfg.lr),
            verbose=verbose,
        )

        model_mlp.eval()
        device = next(model_mlp.parameters()).device

        eval_chunk_size = int(getattr(train_cfg, "eval_chunk_size", 2048))

        return {
            "model_type": "mlp",
            "model_mlp": model_mlp,
            "device": device,
            "eval_chunk_size": eval_chunk_size,
            "train_info": train_info,
            "train_cfg": {
                "mask_rate": float(getattr(train_cfg, "mask_rate", mask_rate)),
                "noise_sigma": float(getattr(train_cfg, "noise_sigma", 0.0)),
                "hidden_dims": list(getattr(train_cfg, "hidden_dims", [])),
                "lr": float(getattr(train_cfg, "lr", 0.0)),
                "batch_size": int(getattr(train_cfg, "batch_size", 0)),
                "max_epochs": int(getattr(train_cfg, "max_epochs", 0)),
                "device": str(getattr(train_cfg, "device", "")),
                "eval_chunk_size": int(eval_chunk_size),
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
    elif str(model_type) == "mlp":
        if train_artifacts is None or "model_mlp" not in train_artifacts:
            raise ValueError("mlp predict requires train_artifacts['model_mlp']")
        model_mlp = train_artifacts["model_mlp"]
        device = train_artifacts["device"]
        chunk_size = int(train_artifacts.get("eval_chunk_size", 2048))
        A_hat_all = _predict_mlp_coeffs_batch(model_mlp, Y_noisy, device=device, chunk_size=chunk_size)
    else:
        raise ValueError(f"Unsupported model_type='{model_type}'")

    # pack outputs (keep compact)
    return {
        "model_type": str(model_type),
        "mask_rate": float(mask_rate),
        "noise_sigma": float(noise_sigma),
        "centered_pod": bool(centered_pod),
        "A_hat_all": np.asarray(A_hat_all, dtype=np.float32),
        "A_true_all": np.asarray(A_true, dtype=np.float32),
        "mask_hw": np.asarray(mask_hw, dtype=bool),
        "mask_flat": np.asarray(mask_flat, dtype=bool),
    }


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

    # optional: train_info as json string
    train_info_json = None
    train_cfg_json = None
    if train_artifacts is not None and train_artifacts.get("train_info", None) is not None:
        train_info_json = json.dumps(train_artifacts["train_info"], ensure_ascii=False)
        train_cfg_json = json.dumps(train_artifacts.get("train_cfg", None), ensure_ascii=False)

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
    if train_info_json is not None:
        payload["train_info_json"] = np.asarray(train_info_json)
    if train_cfg_json is not None:
        payload["train_cfg_json"] = np.asarray(train_cfg_json)

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
