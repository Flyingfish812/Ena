# backend/eval/reports.py

"""
将评估结果整理成表格 / CSV，方便导出到论文或报告。
"""

from typing import Dict, Any

import pandas as pd
from pathlib import Path


def results_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    将 run_linear_baseline_experiment / run_mlp_experiment 返回的结果
    转换为 pandas DataFrame 形式，便于绘图或导出。
    """
    raise NotImplementedError


def save_results_csv(df: pd.DataFrame, path: Path) -> None:
    """
    将结果 DataFrame 保存为 CSV 文件。
    """
    raise NotImplementedError
