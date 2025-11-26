# backend/__init__.py

"""
Ena backend 核心包。

包含：
- config: 配置结构与默认预设
- dataio: 原始数据读取与切分
- pod: POD 基底构建与投影
- sampling: 观测 mask 与噪声
- models: 线性基线与 MLP
- metrics: 误差与多尺度指标
- eval: 实验编排与结果汇总
- viz: 各类可视化
"""

__all__ = [
    "config",
    "dataio",
    "pod",
    "sampling",
    "models",
    "metrics",
    "eval",
    "viz",
]
