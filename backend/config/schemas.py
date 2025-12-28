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
class FourierConfig:
    """
    eval.fourier.* 频域多尺度配置（仅保留新版本 YAML 写法）。

    YAML 期望结构示例（v2.1 增补了 fft2_2d_stats_*）：
      fourier:
        enabled: true
        band_scheme: physical
        grid_meta:
          Lx: 8.0
          Ly: 1.0
          obstacle_diameter: 0.125
          dx: 0.0125
          dy: 0.0125
          angular: false
        num_bins: 64
        sample_frames: 8
        kstar_threshold: 1.0
        mean_mode_true: global
        save_curve: true
        band_names: [L, M, H]
        lambda_edges: [1.0, 0.25]

        # --- v2.1: L3 额外存盘：2D FFT 统计量（用于 coherence / transfer function 等新指标） ---
        save_fft2_2d_stats: false
        fft2_2d_stats_what: [P_true, P_pred, P_err, C_tp, coh, H]
        fft2_2d_stats_avg_over_frames: true
        fft2_2d_stats_dtype: complex64
        fft2_2d_stats_store_shifted: false
        fft2_2d_stats_sample_frames: null
    """
    enabled: bool = True
    band_scheme: str = "physical"   # e.g. "physical"
    grid_meta: Dict[str, Any] = field(default_factory=dict)

    binning: str = "log"  # "linear" or "log"
    num_bins: int = 64
    k_min_eval: float = 0.25
    sample_frames: int = 8

    kstar_threshold: float = 1.0
    mean_mode_true: str = "global"

    save_curve: bool = False
    band_names: Sequence[str] = ("L", "M", "H")
    lambda_edges: Sequence[float] = (1.0, 0.25)

    # v2.1 additions: L3 额外存盘 2D FFT 统计量（默认关闭）
    # 是否在 L3 产物里保存 2D FFT 的“帧平均统计量”
    save_fft2_2d_stats: bool = False

    # 保存哪些 2D 统计量（用字符串约定）
    # - P_true: mean(|F_true|^2)
    # - P_pred: mean(|F_pred|^2)
    # - P_err : mean(|F_err|^2)
    # - C_tp  : mean(F_true * conj(F_pred))  (cross spectrum)
    # - coh   : |C_tp|^2 / (P_true * P_pred)
    # - H     : C_tp / (P_true)
    fft2_2d_stats_what: Sequence[str] = ("P_true", "P_pred", "P_err", "C_tp", "coh", "H")

    # 是否对抽样帧做平均后再保存（推荐 True：体积可控且足够支撑指标）
    fft2_2d_stats_avg_over_frames: bool = True

    # 2D 统计量写盘 dtype（约定字符串，实际落盘由实现决定）
    # - "float32"/"float64": 仅适用于纯实数阵列
    # - "complex64"/"complex128": 适用于互谱/传递函数
    fft2_2d_stats_dtype: str = "complex64"

    # 是否保存为 fftshift 后的布局（默认 False；建议只在纯可视化需求时 True）
    fft2_2d_stats_store_shifted: bool = False

    # 2D 统计量计算时的抽帧数（None 表示沿用 sample_frames；-1 表示全量帧）
    fft2_2d_stats_sample_frames: int | None = None


@dataclass
class EvalConfig:
    """
    评估阶段配置（新版本 YAML schema）。

    - pod_bands / centered_pod: legacy POD 多尺度
    - fourier: 频域多尺度配置（FourierConfig）
    """
    mask_rates: Sequence[float]
    noise_sigmas: Sequence[float]
    pod_bands: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    centered_pod: bool = True
    save_dir: Path = Path("artifacts/eval")
    fourier: FourierConfig = field(default_factory=FourierConfig)