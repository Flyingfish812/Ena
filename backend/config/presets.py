# backend/config/presets.py

"""
为实验提供若干默认配置组合。

这些函数会在 notebook / GUI 中被调用。
"""

from pathlib import Path
from .schemas import DataConfig, PodConfig, TrainConfig, EvalConfig


def default_data_config() -> DataConfig:
    """
    返回一份默认的数据配置。

    注意：nc_path 需要在 notebook 中根据实际环境手动修改。
    """
    raise NotImplementedError


def default_pod_config() -> PodConfig:
    """
    返回论文实验中常用的一份 POD 配置。
    """
    raise NotImplementedError


def default_train_config(mask_rate: float, noise_sigma: float) -> TrainConfig:
    """
    构造一份针对指定 mask_rate 和 noise_sigma 的训练配置。
    """
    raise NotImplementedError


def default_eval_config() -> EvalConfig:
    """
    返回一份默认评估配置（包含多组 mask_rate / noise_sigma 与 POD 分段）。
    """
    raise NotImplementedError
