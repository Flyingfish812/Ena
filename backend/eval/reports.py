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

    DataFrame 列示例：
    - model_type
    - mask_rate
    - noise_sigma
    - nmse_mean, nmse_std
    - nmae_mean, nmae_std
    - psnr_mean, psnr_std
    - band_<name>  （例如 band_L, band_M, band_S）
    """
    model_type = result.get("model_type", "model")
    entries = result.get("entries", [])

    rows: list[dict[str, Any]] = []

    # 收集所有 band 名字，确保列名完整
    band_names: set[str] = set()
    for e in entries:
        band_errors = e.get("band_errors", {})
        band_names.update(band_errors.keys())

    band_names_sorted = sorted(band_names)

    for e in entries:
        row: dict[str, Any] = {
            "model_type": model_type,
            "mask_rate": e["mask_rate"],
            "noise_sigma": e["noise_sigma"],
            "nmse_mean": e["nmse_mean"],
            "nmse_std": e["nmse_std"],
            "nmae_mean": e["nmae_mean"],
            "nmae_std": e["nmae_std"],
            "psnr_mean": e["psnr_mean"],
            "psnr_std": e["psnr_std"],
            "n_frames": e.get("n_frames", None),
            "n_obs": e.get("n_obs", None),
        }

        band_errors = e.get("band_errors", {})
        for name in band_names_sorted:
            key = f"band_{name}"
            row[key] = float(band_errors.get(name, float("nan")))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_results_csv(df: pd.DataFrame, path: Path) -> None:
    """
    将结果 DataFrame 保存为 CSV 文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
