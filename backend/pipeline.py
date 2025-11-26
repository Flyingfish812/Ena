# backend/pipeline.py

"""
为 notebook / GUI 提供的高层“一键调用”接口。

目标：在 ipynb 里一行代码完成某一阶段的实验。
"""

from pathlib import Path
from typing import Dict, Any

from .config.schemas import DataConfig, PodConfig, TrainConfig, EvalConfig


def run_build_pod_pipeline(data_cfg: DataConfig, pod_cfg: PodConfig) -> Dict[str, Any]:
    """
    顶层 POD 构建入口。

    - 读取原始数据
    - 执行 SVD / POD
    - 截断并保存基底与均值
    - 返回能量谱等元信息，供 notebook/GUI 作图
    """
    raise NotImplementedError


def run_train_mlp_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    顶层 MLP 训练入口。

    - 使用给定 POD 基底与数据配置构建训练/验证数据集
    - 训练 MLP 模型预测 POD 系数
    - 保存最优模型与训练日志
    - 返回训练曲线等信息
    """
    raise NotImplementedError


def run_full_eval_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
) -> Dict[str, Any]:
    """
    顶层评估入口。

    - 在 test 集上对比线性基线与 MLP
    - 扫描 mask_rate / noise_sigma 等参数
    - 计算全场与多尺度误差
    - 返回可直接喂给 viz 的结构化结果
    """
    raise NotImplementedError
