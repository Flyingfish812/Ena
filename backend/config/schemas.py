# backend/config/schemas.py

"""
所有实验相关配置的结构定义。

使用 dataclass 方便在 notebook / GUI / CLI 中构建和编辑。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Sequence, Dict, Any


@dataclass
class DataConfig:
    """
    原始数据集配置。
    """

    # -----------------
    # Source routing
    # -----------------
    # Supported:
    #   - "netcdf"  : use nc_path/var_keys (legacy default)
    #   - "h5_rdb"  : data/2D_rdb_NA_NA.h5
    #   - "mat_sst" : data/sst_weekly.mat (MAT v7.3 / HDF5)
    source: str = "netcdf"

    # New generic path (preferred for new datasets). If omitted, fall back to nc_path.
    path: Path | None = None

    # Legacy field kept for backward compatibility.
    nc_path: Path | None = None

    # NetCDF only
    var_keys: Tuple[str, ...] = ("u", "v")

    # Optional cache directory for loaders to store derived arrays (e.g. sampled memmaps)
    cache_dir: Path | None = None

    # -----------------
    # h5_rdb options
    # -----------------
    h5_rdb_dataset_key: str = "data"
    h5_rdb_group_count: int = 50
    h5_rdb_group_start: int = 0
    h5_rdb_group_step: int = 1
    h5_rdb_frames_per_group: int = 10
    h5_rdb_frame_sampling: str = "linspace"  # "linspace" only for now

    # -----------------
    # mat_sst options
    # -----------------
    mat_key: str = "sst"
    sst_fill_nan: str = "per_frame_mean"  # "per_frame_mean" | "global_mean" | "zero"
    sst_reshape: str = "360x180_rot90"    # match scripts/analyze_new_datasets.py
    sst_max_frames: int | None = None


@dataclass
class PodConfig:
    """
    POD 基底构建配置（Level-1）。

    v2.x additions:
    - dx, dy: 空间网格步长（用于尺度换算/频谱网格）
    - enable_scale_analysis + scale_analysis: 生成 scale_table.csv / scale_meta.json
    - enable_basis_spectrum + fft_basis: 生成 basis_spectrum.npz
    """
    r: int = 128
    center: bool = True
    save_dir: Path = Path("artifacts/pod")

    # --- v2.x: grid meta for scale/fft ---
    dx: float = 1.0
    dy: float = 1.0

    # --- v2.x: scale table ---
    scale_channel_reduce: str = "l2"
    enable_scale_analysis: bool = False
    scale_analysis: Dict[str, Any] = field(default_factory=lambda: {
        # 主线默认方法名（scaler.py 内部按这个分派）
        "method": "B_robust_energy_centroid",
        # 频率截断（角波数，单位 rad/len；None 表示不截断）
        "k_min": None,
        "k_max": None,
        # 逐线 FFT 前是否去均值
        "demean_line": True,
    })

    # --- v2.x: basis spectrum dict ---
    enable_basis_spectrum: bool = False
    fft_basis: Dict[str, Any] = field(default_factory=lambda: {
        "demean": True,
        # 预留：window 等（现阶段 scaler.py 可先不实现，但 schema 允许）
        "window": None,
        "norm": None,
    })


@dataclass
class TrainConfig:
    """
    MLP 训练配置。
    """
    mask_rate: float
    noise_sigma: float
    hidden_dims: Tuple[int, ...]
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    device: str = "cuda"
    save_dir: Path = Path("artifacts/nn")


@dataclass
class FourierConfig:
    """
    eval.fourier.* 频域多尺度配置（仅保留新版本 YAML 写法）。
    """
    enabled: bool = True
    band_scheme: str = "physical"
    grid_meta: Dict[str, Any] = field(default_factory=dict)

    binning: str = "log"
    num_bins: int = 64
    k_min_eval: float = 0.25
    sample_frames: int = 8

    kstar_threshold: float = 1.0
    mean_mode_true: str = "global"

    save_curve: bool = False
    band_names: Sequence[str] = ("L", "M", "H")
    lambda_edges: Sequence[float] = (1.0, 0.25)

    # v2.1 additions
    save_fft2_2d_stats: bool = False
    fft2_2d_stats_what: Sequence[str] = ("P_true", "P_pred", "P_err", "C_tp", "coh", "H")
    fft2_2d_stats_avg_over_frames: bool = True
    fft2_2d_stats_dtype: str = "complex64"
    fft2_2d_stats_store_shifted: bool = False
    fft2_2d_stats_sample_frames: int | None = None


@dataclass
class EvalConfig:
    """
    评估阶段配置（新版本 YAML schema）。
    """
    mask_rates: Sequence[float]
    noise_sigmas: Sequence[float]
    pod_bands: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    centered_pod: bool = True
    save_dir: Path = Path("artifacts/eval")
    fourier: FourierConfig = field(default_factory=FourierConfig)