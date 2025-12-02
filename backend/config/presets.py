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

    注意：
    - nc_path 只是一个占位路径，实际使用时请在 notebook / GUI 中改成你真实的数据位置。
    - var_keys 默认假定数据集中有 "u", "v" 两个通道。
    """
    return DataConfig(
        nc_path=Path("data/cylinder2d.nc"),  # TODO: 在 notebook 里改成真实路径
        var_keys=("u", "v"),
        cache_dir=None,
    )


def default_pod_config() -> PodConfig:
    """
    返回论文实验中常用的一份 POD 配置。

    约定：
    - 截断模态数 r = 128
    - 进行去均值
    - 结果存到 artifacts/pod 下面
    """
    return PodConfig(
        r=128,
        center=True,
        save_dir=Path("artifacts/pod"),
    )


def default_train_config(mask_rate: float, noise_sigma: float) -> TrainConfig:
    """
    构造一份针对指定 mask_rate 和 noise_sigma 的训练配置。

    设计约定：
    - hidden_dims 固定用 (256, 256)（与当前 PodMLP 实现保持一致）
    - 训练轮数适中：max_epochs=100
    - lr=1e-3, batch_size=64
    - 默认跑在 "cuda"，如果机器上没有 GPU，train_mlp_on_observations 内部会 fallback 到 CPU
    - save_dir 统一放在 artifacts/nn 下面，按 mask_rate + noise_sigma 再细分一层目录
    """
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
    """
    返回一份默认评估配置（包含多组 mask_rate / noise_sigma 与 POD 分段）。

    设计约定（你后面可以在 notebook 里改）：
    - mask_rates: 更关心稀疏观测区域：1%, 2%, 5%, 10%
    - noise_sigmas: 无噪声 / 中等噪声 / 稍大的噪声：0.0, 0.01, 0.02
    - pod_bands:
        L: 低频 (0~16)
        M: 中频 (16~64)
        H: 高频 (64~128)
      要求和 PodConfig.r 保持一致（这里 r=128）
    - save_dir: artifacts/eval
    """
    mask_rates = [0.01, 0.02, 0.05, 0.10]
    noise_sigmas = [0.0, 0.01, 0.02]

    pod_bands = {
        "L": (0, 16),
        "M": (16, 64),
        "H": (64, 128),
    }

    return EvalConfig(
        mask_rates=mask_rates,
        noise_sigmas=noise_sigmas,
        pod_bands=pod_bands,
        save_dir=Path("artifacts/eval"),
    )
