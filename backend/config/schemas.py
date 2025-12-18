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

    YAML 期望结构示例：
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
    """
    enabled: bool = True
    band_scheme: str = "physical"   # e.g. "physical"
    grid_meta: Dict[str, Any] = field(default_factory=dict)

    num_bins: int = 64
    sample_frames: int = 8

    kstar_threshold: float = 1.0
    mean_mode_true: str = "global"

    save_curve: bool = False
    band_names: Sequence[str] = ("L", "M", "H")
    lambda_edges: Sequence[float] = (1.0, 0.25)


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