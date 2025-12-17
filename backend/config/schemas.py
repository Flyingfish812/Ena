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

    Attributes
    ----------
    nc_path:
        NetCDF 文件路径。
    var_keys:
        要读取的变量名，例如 ("u", "v")。
    cache_dir:
        可选，缓存中间 numpy 数组（例如预处理后的全量快照）。
    """
    nc_path: Path
    var_keys: Tuple[str, ...] = ("u", "v")
    cache_dir: Path | None = None


@dataclass
class PodConfig:
    """
    POD 基底构建配置。

    Attributes
    ----------
    r:
        截断的模态数量。
    center:
        是否先对每个空间点做去均值。
    save_dir:
        保存 POD 基底与相关元数据的目录。
    """
    r: int = 128
    center: bool = True
    save_dir: Path = Path("artifacts/pod")


@dataclass
class TrainConfig:
    """
    MLP 训练配置。

    Attributes
    ----------
    mask_rate:
        观测点比例（0-1），例如 0.02 表示 2% 网格点观测。
    noise_sigma:
        观测噪声标准差。
    hidden_dims:
        隐藏层尺寸列表，例如 (4*r, 2*r)。
    lr:
        学习率。
    batch_size:
        batch 大小。
    max_epochs:
        训练轮数上限。
    device:
        "cpu" 或 "cuda"。
    save_dir:
        保存模型与训练日志的目录。
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
class EvalConfig:
    """
    评估阶段配置。

    POD 多尺度（legacy）：
      - pod_bands / centered_pod

    Fourier 频域多尺度（v1.12+）：
      - fourier_enabled: 是否启用频域尺度分析
      - fourier_grid: 指定 dx/dy 或 Lx/Ly（用于构造 k 网格）
      - fourier_num_bins: 径向谱分箱数量
      - fourier_k_edges: 频带边界（None 则按能量分位数自动选）
      - fourier_kstar_threshold: k* 的阈值（NRMSE(k) <= threshold）
      - fourier_sample_frames: 频域统计抽样帧数（控制开销）
      - fourier_save_curve: 是否把谱曲线存进 entry（很大，默认 False）
    """
    mask_rates: Sequence[float]
    noise_sigmas: Sequence[float]
    pod_bands: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    centered_pod: bool = True
    save_dir: Path = Path("artifacts/eval")

    # ===== Fourier frequency-space multiscale (Batch 3/6) =====
    fourier_enabled: bool = True

    # grid meta: user may provide:
    #   {"dx":..., "dy":..., "angular": True/False}
    # or {"Lx":..., "Ly":..., "angular": True/False}  (dx=Lx/W, dy=Ly/H inferred later)
    fourier_grid: Dict[str, Any] = field(default_factory=dict)

    # radial spectrum bins
    fourier_num_bins: int = 64
    fourier_k_max: float | None = None  # None -> max(k)

    # band partition
    fourier_k_edges: Sequence[float] | None = None
    fourier_band_names: Tuple[str, ...] = ("L", "M", "H")
    fourier_auto_edges_quantiles: Tuple[float, ...] = (0.80, 0.95)

    # optional weak split (soft weights); 0 => hard split
    fourier_soft_transition: float = 0.0

    # k* definition
    fourier_kstar_threshold: float = 1.0
    fourier_monotone_envelope: bool = True

    # compute cost control
    fourier_sample_frames: int = 8
    fourier_save_curve: bool = False

    # mean removal mode in FFT of x_true
    fourier_mean_mode_true: str = "global"
