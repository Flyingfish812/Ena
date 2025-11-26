# backend/eval/__init__.py

"""
实验评估与结果汇总。
"""

from .reconstruction import run_linear_baseline_experiment, run_mlp_experiment

__all__ = [
    "run_linear_baseline_experiment",
    "run_mlp_experiment",
]
