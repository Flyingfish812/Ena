# backend/eval/reconstruction.py

"""
在线性基线与 MLP 之间进行对比评估。

负责：
- 加载 test 集 snapshot
- 对每个 mask_rate / noise_sigma 组合执行重建
- 计算全场与多尺度误差
"""

from typing import Dict, Any

import numpy as np

from ..config.schemas import DataConfig, PodConfig, EvalConfig, TrainConfig
from ..pod.project import project_to_pod, reconstruct_from_pod
from ..sampling.masks import flatten_mask, apply_mask_flat
from ..sampling.noise import add_gaussian_noise
from ..models.linear_baseline import solve_pod_coeffs_least_squares
from ..metrics.errors import nmse, nmae, psnr
from ..metrics.multiscale import compute_pod_band_errors


def run_linear_baseline_experiment(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
) -> Dict[str, Any]:
    """
    在 test 集上，对一组 (mask_rate, noise_sigma) 组合运行线性基线重建。

    返回
    ----
    result:
        结构化结果，其中包含每个组合的：
        - 全场误差指标
        - POD band 误差
        - 可选择存储部分重建样例
    """
    raise NotImplementedError


def run_mlp_experiment(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    在 test 集上，对一组 (mask_rate, noise_sigma) 组合运行 MLP 重建。

    需要加载已经训练好的模型权重。

    返回结构同 run_linear_baseline_experiment。
    """
    raise NotImplementedError
