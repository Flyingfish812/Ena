# backend/config/presets.py

"""
为实验提供若干默认配置组合。

这些函数会在 notebook / GUI 中被调用。
"""

from pathlib import Path
from .schemas import DataConfig, PodConfig, TrainConfig, FourierConfig, EvalConfig


def default_data_config() -> DataConfig:
    return DataConfig(
        nc_path=Path("data/cylinder2d.nc"),
        var_keys=("u", "v"),
        cache_dir=None,
    )


def default_pod_config() -> PodConfig:
    """
    常用 POD 配置（r=128，去均值，保存到 artifacts/pod）。
    v2.x: 默认给 dx/dy=1.0；需要物理标定时在 notebook/GUI 改。
    """
    return PodConfig(
        r=128,
        center=True,
        save_dir=Path("artifacts/pod"),

        dx=1.0,
        dy=1.0,

        scale_channel_reduce="l2",
        enable_scale_analysis=False,
        scale_analysis={
            "method": "B_robust_energy_centroid",
            "k_min": None,
            "k_max": None,
            "demean_line": True,
        },

        enable_basis_spectrum=False,
        fft_basis={
            "demean": True,
            "window": None,
            "norm": None,
        },
    )


def default_train_config(mask_rate: float, noise_sigma: float) -> TrainConfig:
    subdir = f"p{mask_rate:.4f}_sigma{noise_sigma:.3g}".replace(".", "p")
    save_dir = Path("artifacts/nn") / subdir
    return TrainConfig(
        mask_rate=mask_rate,
        noise_sigma=noise_sigma,
        hidden_dims=(256, 256),
        lr=1e-3,
        batch_size=64,
        max_epochs=100,
        device="cuda",
        save_dir=save_dir,
    )


def default_eval_config() -> EvalConfig:
    mask_rates = [0.0001, 0.0004, 0.0016]
    noise_sigmas = [0.0, 0.01, 0.1]

    pod_bands = {
        "L1": (0, 16),
        "L2": (16, 32),
        "M1": (32, 48),
        "M2": (48, 64),
        "M3": (64, 80),
        "H1": (80, 96),
        "H2": (96, 112),
        "H3": (112, 128),
    }

    fourier = FourierConfig(
        enabled=True,
        band_scheme="physical",
        grid_meta={
            "Lx": 8.0,
            "Ly": 1.0,
            "obstacle_diameter": 0.125,
            "dx": 0.0125,
            "dy": 0.0125,
            "angular": False,
        },
        binning="log",
        num_bins=64,
        k_min_eval=0.25,
        sample_frames=8,
        kstar_threshold=1.0,
        mean_mode_true="global",
        save_curve=True,
        band_names=("L", "M", "H"),
        lambda_edges=(1.0, 0.25),

        save_fft2_2d_stats=False,
        fft2_2d_stats_what=("P_true", "P_pred", "P_err", "C_tp", "coh", "H"),
        fft2_2d_stats_avg_over_frames=True,
        fft2_2d_stats_dtype="complex64",
        fft2_2d_stats_store_shifted=False,
        fft2_2d_stats_sample_frames=None,
    )

    return EvalConfig(
        mask_rates=mask_rates,
        noise_sigmas=noise_sigmas,
        pod_bands=pod_bands,
        centered_pod=True,
        save_dir=Path("artifacts/eval"),
        fourier=fourier,
    )
