# backend/config/yaml_io.py

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .schemas import DataConfig, PodConfig, EvalConfig, FourierConfig, TrainConfig


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_experiment_yaml(
    path: str | Path,
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = _to_serializable(asdict(data_cfg))
    pod_dict = _to_serializable(asdict(pod_cfg))
    eval_dict = _to_serializable(asdict(eval_cfg))
    train_dict = _to_serializable(asdict(train_cfg)) if train_cfg is not None else None

    config: Dict[str, Any] = {
        "data": data_dict,
        "pod": pod_dict,
        "eval": eval_dict,
    }
    if train_dict is not None:
        config["train"] = train_dict

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def load_experiment_yaml(
    path: str | Path,
) -> Tuple[DataConfig, PodConfig, EvalConfig, TrainConfig | None]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # ---------- data ----------
    data_raw: Dict[str, Any] = config.get("data", {}) or {}
    nc_path = Path(data_raw.get("nc_path", "data/cylinder2d.nc"))
    var_keys = tuple(data_raw.get("var_keys", ("u", "v")))
    cache_dir = data_raw.get("cache_dir", None)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    data_cfg = DataConfig(
        nc_path=nc_path,
        var_keys=var_keys,
        cache_dir=cache_dir,
    )

    # ---------- pod ----------
    pod_raw: Dict[str, Any] = config.get("pod", {}) or {}
    pod_save_dir = Path(pod_raw.get("save_dir", "artifacts/pod"))

    dx = float(pod_raw.get("dx", 1.0))
    dy = float(pod_raw.get("dy", 1.0))

    scale_channel_reduce = str(pod_raw.get("scale_channel_reduce", "l2")),
    enable_scale_analysis = bool(pod_raw.get("enable_scale_analysis", False))
    scale_analysis = dict(pod_raw.get("scale_analysis", {}) or {})

    enable_basis_spectrum = bool(pod_raw.get("enable_basis_spectrum", False))
    fft_basis = dict(pod_raw.get("fft_basis", {}) or {})

    pod_cfg = PodConfig(
        r=int(pod_raw.get("r", 128)),
        center=bool(pod_raw.get("center", True)),
        save_dir=pod_save_dir,

        dx=dx,
        dy=dy,

        scale_channel_reduce=scale_channel_reduce,
        enable_scale_analysis=enable_scale_analysis,
        scale_analysis=scale_analysis,

        enable_basis_spectrum=enable_basis_spectrum,
        fft_basis=fft_basis,
    )

    # ---------- eval ----------
    eval_raw: Dict[str, Any] = config.get("eval", {}) or {}
    mask_rates = list(eval_raw.get("mask_rates", [0.0001, 0.0004, 0.0016]))
    noise_sigmas = list(eval_raw.get("noise_sigmas", [0.0, 0.01, 0.1]))

    pod_bands_raw = eval_raw.get("pod_bands", {}) or {}
    pod_bands = {name: (int(v[0]), int(v[1])) for name, v in pod_bands_raw.items()}
    centered_pod = bool(eval_raw.get("centered_pod", True))
    eval_save_dir = Path(eval_raw.get("save_dir", "artifacts/eval"))

    # ----- Fourier -----
    fourier_raw: Dict[str, Any] = eval_raw.get("fourier", {}) or {}

    enabled = bool(fourier_raw.get("enabled", True))
    band_scheme = str(fourier_raw.get("band_scheme", "physical"))
    grid_meta = dict(fourier_raw.get("grid_meta", {}) or {})

    binning = str(fourier_raw.get("binning", "log"))
    num_bins = int(fourier_raw.get("num_bins", 64))
    k_min_eval = float(fourier_raw.get("k_min_eval", 0.25))
    sample_frames = int(fourier_raw.get("sample_frames", 8))
    kstar_threshold = float(fourier_raw.get("kstar_threshold", 1.0))
    mean_mode_true = str(fourier_raw.get("mean_mode_true", "global"))
    save_curve = bool(fourier_raw.get("save_curve", False))

    band_names = tuple(fourier_raw.get("band_names", ("L", "M", "H")))
    lambda_edges_raw = fourier_raw.get("lambda_edges", (1.0, 0.25))
    lambda_edges = [float(v) for v in lambda_edges_raw]

    save_fft2_2d_stats = bool(fourier_raw.get("save_fft2_2d_stats", False))
    fft2_2d_stats_what_raw = fourier_raw.get(
        "fft2_2d_stats_what",
        ("P_true", "P_pred", "P_err", "C_tp", "coh", "H"),
    )
    fft2_2d_stats_what = tuple(str(x) for x in (fft2_2d_stats_what_raw or ()))

    fft2_2d_stats_avg_over_frames = bool(
        fourier_raw.get("fft2_2d_stats_avg_over_frames", True)
    )
    fft2_2d_stats_dtype = str(fourier_raw.get("fft2_2d_stats_dtype", "complex64"))
    fft2_2d_stats_store_shifted = bool(
        fourier_raw.get("fft2_2d_stats_store_shifted", False)
    )

    fft2_2d_stats_sample_frames_raw = fourier_raw.get("fft2_2d_stats_sample_frames", None)
    if fft2_2d_stats_sample_frames_raw is None:
        fft2_2d_stats_sample_frames = None
    else:
        fft2_2d_stats_sample_frames = int(fft2_2d_stats_sample_frames_raw)

    fourier_cfg = FourierConfig(
        enabled=enabled,
        band_scheme=band_scheme,
        grid_meta=grid_meta,
        binning=binning,
        num_bins=num_bins,
        k_min_eval=k_min_eval,
        sample_frames=sample_frames,
        kstar_threshold=kstar_threshold,
        mean_mode_true=mean_mode_true,
        save_curve=save_curve,
        band_names=band_names,
        lambda_edges=lambda_edges,
        save_fft2_2d_stats=save_fft2_2d_stats,
        fft2_2d_stats_what=fft2_2d_stats_what,
        fft2_2d_stats_avg_over_frames=fft2_2d_stats_avg_over_frames,
        fft2_2d_stats_dtype=fft2_2d_stats_dtype,
        fft2_2d_stats_store_shifted=fft2_2d_stats_store_shifted,
        fft2_2d_stats_sample_frames=fft2_2d_stats_sample_frames,
    )

    eval_cfg = EvalConfig(
        mask_rates=mask_rates,
        noise_sigmas=noise_sigmas,
        pod_bands=pod_bands,
        centered_pod=centered_pod,
        save_dir=eval_save_dir,
        fourier=fourier_cfg,
    )

    # ---------- train (optional) ----------
    train_raw: Dict[str, Any] | None = config.get("train", None)
    if train_raw is None:
        train_cfg = None
    else:
        train_save_dir = Path(train_raw.get("save_dir", "artifacts/nn"))
        hidden_dims = tuple(train_raw.get("hidden_dims", (256, 256)))
        train_cfg = TrainConfig(
            mask_rate=float(train_raw.get("mask_rate", 0.02)),
            noise_sigma=float(train_raw.get("noise_sigma", 0.01)),
            hidden_dims=hidden_dims,
            lr=float(train_raw.get("lr", 1e-3)),
            batch_size=int(train_raw.get("batch_size", 64)),
            max_epochs=int(train_raw.get("max_epochs", 50)),
            device=str(train_raw.get("device", "cuda")),
            save_dir=train_save_dir,
        )

    return data_cfg, pod_cfg, eval_cfg, train_cfg
