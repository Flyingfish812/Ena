# backend/eval/reports.py

"""
将评估结果整理成表格 / CSV / Markdown 报告，方便导出到论文或报告。
"""

from __future__ import annotations

from typing import Dict, Any

from pathlib import Path
import json

import pandas as pd


# ----------------------------------------------------------------------
# 1. DataFrame & CSV
# ----------------------------------------------------------------------


def results_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    将 run_linear_baseline_experiment / run_mlp_experiment 返回的结果
    转换为 pandas DataFrame 形式，便于绘图或导出。

    v1.12+（Fourier 多尺度）新增列：
    - k_star
    - fourier_band_nrmse_<band>
    """
    model_type = result.get("model_type", "model")
    entries = result.get("entries", []) or []

    rows: list[dict[str, Any]] = []

    # --- 先扫一遍，收集所有可能出现的 key，保证列名完备 ---
    band_rmse_names: set[str] = set()
    band_nrmse_names: set[str] = set()
    group_names: set[str] = set()
    partial_names: set[str] = set()

    fourier_band_names: set[str] = set()
    has_kstar = False

    for e in entries:
        band_errors = e.get("band_errors", {}) or {}
        band_rmse_names.update(band_errors.keys())

        band_nrmse = e.get("band_nrmse", {}) or {}
        band_nrmse_names.update(band_nrmse.keys())

        group_err = e.get("field_nmse_per_group", {}) or {}
        group_names.update(group_err.keys())

        partial_err = e.get("field_nmse_partial", {}) or {}
        partial_names.update(partial_err.keys())

        # Fourier
        if "k_star" in e:
            has_kstar = True
        fb = e.get("fourier_band_nrmse", {}) or {}
        if isinstance(fb, dict):
            fourier_band_names.update(fb.keys())

    band_rmse_sorted = sorted(band_rmse_names)
    band_nrmse_sorted = sorted(band_nrmse_names)
    group_sorted = sorted(group_names)
    partial_sorted = sorted(partial_names)

    fourier_band_sorted = sorted(fourier_band_names)

    # --- 逐 entry 构造行 ---
    for e in entries:
        row: dict[str, Any] = {
            "model_type": model_type,
            "mask_rate": e.get("mask_rate", None),
            "noise_sigma": e.get("noise_sigma", None),
            "nmse_mean": e.get("nmse_mean", None),
            "nmse_std": e.get("nmse_std", None),
            "nmae_mean": e.get("nmae_mean", None),
            "nmae_std": e.get("nmae_std", None),
            "psnr_mean": e.get("psnr_mean", None),
            "psnr_std": e.get("psnr_std", None),
            "n_frames": e.get("n_frames", None),
            "n_obs": e.get("n_obs", None),
            # 有效模态等级（legacy POD band）
            "effective_band": e.get("effective_band", None),
            "effective_r_cut": e.get("effective_r_cut", None),
        }

        band_errors = e.get("band_errors", {}) or {}
        for name in band_rmse_sorted:
            key = f"band_{name}"
            row[key] = float(band_errors.get(name, float("nan")))

        band_nrmse = e.get("band_nrmse", {}) or {}
        for name in band_nrmse_sorted:
            key = f"band_nrmse_{name}"
            row[key] = float(band_nrmse.get(name, float("nan")))

        group_err = e.get("field_nmse_per_group", {}) or {}
        for name in group_sorted:
            key = f"group_nmse_{name}"
            row[key] = float(group_err.get(name, float("nan")))

        partial_err = e.get("field_nmse_partial", {}) or {}
        for name in partial_sorted:
            key = f"partial_nmse_{name}"
            row[key] = float(partial_err.get(name, float("nan")))

        # Fourier flat columns
        if has_kstar:
            row["k_star"] = e.get("k_star", float("nan"))

        fb = e.get("fourier_band_nrmse", {}) or {}
        for b in fourier_band_sorted:
            row[f"fourier_band_nrmse_{b}"] = float(fb.get(b, float("nan")))

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def save_results_csv(df: pd.DataFrame, path: Path | str) -> None:
    """
    将结果 DataFrame 保存为 CSV 文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _strip_heavy_fields(result: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """
    从单个 model 结果 dict 中移除体积巨大的字段（如 examples / example_recon / fourier_curve），
    只保留 entries + meta 等数值型摘要，避免 JSON 爆炸到上百 MB。
    """
    if result is None:
        return None

    out = dict(result)

    out.pop("examples", None)
    out.pop("example_recon", None)

    # 如果 entry 内含 fourier_curve（save_curve=true），JSON 会迅速变大
    entries = out.get("entries", None)
    if isinstance(entries, list):
        new_entries = []
        for e in entries:
            if not isinstance(e, dict):
                new_entries.append(e)
                continue
            ee = dict(e)
            ee.pop("fourier_curve", None)
            new_entries.append(ee)
        out["entries"] = new_entries

    return out


def save_full_experiment_results(
    all_result: Dict[str, Any],
    base_dir: Path | str,
    experiment_name: str,
) -> Dict[str, Path]:
    """
    保存一次完整实验（linear + mlp）的结果到 JSON/CSV。
    """
    base_dir = Path(base_dir) / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # 1) JSON（瘦身）
    linear_res_full = all_result.get("linear", None)
    linear_res = _strip_heavy_fields(linear_res_full)
    if linear_res is not None:
        p_json = base_dir / "linear_results.json"
        with p_json.open("w", encoding="utf-8") as f:
            json.dump(linear_res, f, indent=2, ensure_ascii=False)
        paths["linear_json"] = p_json

    mlp_res_full = all_result.get("mlp", None)
    mlp_res = _strip_heavy_fields(mlp_res_full)
    if mlp_res is not None:
        p_json = base_dir / "mlp_results.json"
        with p_json.open("w", encoding="utf-8") as f:
            json.dump(mlp_res, f, indent=2, ensure_ascii=False)
        paths["mlp_json"] = p_json

    # 2) CSV
    df_lin = all_result.get("df_linear", None)
    if df_lin is not None:
        p_csv = base_dir / "linear_results.csv"
        save_results_csv(df_lin, p_csv)
        paths["linear_csv"] = p_csv

    df_mlp = all_result.get("df_mlp", None)
    if df_mlp is not None:
        p_csv = base_dir / "mlp_results.csv"
        save_results_csv(df_mlp, p_csv)
        paths["mlp_csv"] = p_csv

    return paths


def load_full_experiment_results(
    base_dir: Path | str,
    experiment_name: str | None = None,
) -> Dict[str, Any]:
    """
    从磁盘加载一次完整实验（linear + mlp）的数值结果，供后续绘图 / 报告使用。
    """
    base_dir = Path(base_dir)
    if experiment_name is not None:
        exp_dir = base_dir / experiment_name
    else:
        exp_dir = base_dir

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    linear_res: Dict[str, Any] | None = None
    mlp_res: Dict[str, Any] | None = None
    df_lin = None
    df_mlp = None

    p_lin_json = exp_dir / "linear_results.json"
    if p_lin_json.exists():
        with p_lin_json.open("r", encoding="utf-8") as f:
            linear_res = json.load(f)

    p_mlp_json = exp_dir / "mlp_results.json"
    if p_mlp_json.exists():
        with p_mlp_json.open("r", encoding="utf-8") as f:
            mlp_res = json.load(f)

    p_lin_csv = exp_dir / "linear_results.csv"
    if p_lin_csv.exists():
        df_lin = pd.read_csv(p_lin_csv)

    p_mlp_csv = exp_dir / "mlp_results.csv"
    if p_mlp_csv.exists():
        df_mlp = pd.read_csv(p_mlp_csv)

    return {
        "exp_dir": exp_dir,
        "linear": linear_res,
        "mlp": mlp_res,
        "df_linear": df_lin,
        "df_mlp": df_mlp,
    }


# ----------------------------------------------------------------------
# 2. Markdown 实验报告生成
# ----------------------------------------------------------------------

def generate_experiment_report_md(
    all_result: Dict[str, Any],
    out_path: Path | str,
    experiment_name: str = "Ena experiment",
    config_yaml: Path | str | None = None,
) -> Path:
    """
    根据一次完整实验的结果，生成一个模板化的 report.md。

    v1.12+：
    - 保留原 POD-band 多尺度（论文旧版可对照）
    - 新增 Fourier 频域尺度：k* 与 Fourier band NRMSE
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    linear_res = all_result.get("linear", None)
    mlp_res = all_result.get("mlp", None)

    if linear_res is None:
        raise ValueError("all_result['linear'] is required to generate report.")

    df_lin = all_result.get("df_linear", results_to_dataframe(linear_res))
    df_mlp = None
    if mlp_res is not None:
        df_mlp = all_result.get("df_mlp", results_to_dataframe(mlp_res))

    meta = linear_res.get("meta", {})
    H = meta.get("H", "?")
    W = meta.get("W", "?")
    C = meta.get("C", "?")
    T = meta.get("T", "?")
    r_eff = meta.get("r_eff", "?")
    pod_bands = meta.get("pod_bands", {})

    mask_rates = linear_res.get("mask_rates", [])
    noise_sigmas = linear_res.get("noise_sigmas", [])

    saved_paths: Dict[str, Any] = all_result.get("saved_paths", {}) or {}

    fig_nmse_vs_mask = saved_paths.get("fig_nmse_vs_mask", None)
    fig_nmse_vs_noise = saved_paths.get("fig_nmse_vs_noise", None)
    fig_example_linear = saved_paths.get("fig_example_linear", None)
    fig_example_mlp = saved_paths.get("fig_example_mlp", None)
    fig_multiscale_example = saved_paths.get("fig_multiscale_example", None)

    # Fourier figs (Batch 4)
    fig_kstar_linear = saved_paths.get("fig_kstar_linear", None)
    fig_kstar_mlp = saved_paths.get("fig_kstar_mlp", None)
    fig_fourier_band_vs_mask_linear = saved_paths.get("fig_fourier_band_vs_mask_linear", None)
    fig_fourier_band_vs_mask_mlp = saved_paths.get("fig_fourier_band_vs_mask_mlp", None)
    fig_fourier_band_vs_noise_linear = saved_paths.get("fig_fourier_band_vs_noise_linear", None)
    fig_fourier_band_vs_noise_mlp = saved_paths.get("fig_fourier_band_vs_noise_mlp", None)

    # ------------------------------------------------------------------
    # 1) summary 统计
    # ------------------------------------------------------------------
    best_lin = df_lin.loc[df_lin["nmse_mean"].idxmin()]
    worst_lin = df_lin.loc[df_lin["nmse_mean"].idxmax()]

    summary_lines: list[str] = []
    summary_lines.append(
        f"- Linear baseline 最佳场级 NMSE = {best_lin['nmse_mean']:.4e} "
        f"(p={best_lin['mask_rate']:.3g}, σ={best_lin['noise_sigma']:.3g})"
    )
    summary_lines.append(
        f"- Linear baseline 最差场级 NMSE = {worst_lin['nmse_mean']:.4e} "
        f"(p={worst_lin['mask_rate']:.3g}, σ={worst_lin['noise_sigma']:.3g})"
    )

    if df_mlp is not None and len(df_mlp) > 0:
        best_mlp = df_mlp.loc[df_mlp["nmse_mean"].idxmin()]
        worst_mlp = df_mlp.loc[df_mlp["nmse_mean"].idxmax()]
        summary_lines.append(
            f"- MLP baseline 最佳场级 NMSE = {best_mlp['nmse_mean']:.4e} "
            f"(p={best_mlp['mask_rate']:.3g}, σ={best_mlp['noise_sigma']:.3g})"
        )
        summary_lines.append(
            f"- MLP baseline 最差场级 NMSE = {worst_mlp['nmse_mean']:.4e} "
            f"(p={worst_mlp['mask_rate']:.3g}, σ={worst_mlp['noise_sigma']:.3g})"
        )

    # Fourier quick summary: best/worst k* if exists
    fourier_summary: list[str] = []
    if "k_star" in df_lin.columns:
        try:
            best_k = df_lin.loc[df_lin["k_star"].idxmax()]
            worst_k = df_lin.loc[df_lin["k_star"].idxmin()]
            fourier_summary.append(
                f"- Linear 的最大 k* = {best_k['k_star']:.4g} (p={best_k['mask_rate']:.3g}, σ={best_k['noise_sigma']:.3g})"
            )
            fourier_summary.append(
                f"- Linear 的最小 k* = {worst_k['k_star']:.4g} (p={worst_k['mask_rate']:.3g}, σ={worst_k['noise_sigma']:.3g})"
            )
        except Exception:
            pass

    if df_mlp is not None and "k_star" in df_mlp.columns:
        try:
            best_k = df_mlp.loc[df_mlp["k_star"].idxmax()]
            worst_k = df_mlp.loc[df_mlp["k_star"].idxmin()]
            fourier_summary.append(
                f"- MLP 的最大 k* = {best_k['k_star']:.4g} (p={best_k['mask_rate']:.3g}, σ={best_k['noise_sigma']:.3g})"
            )
            fourier_summary.append(
                f"- MLP 的最小 k* = {worst_k['k_star']:.4g} (p={worst_k['mask_rate']:.3g}, σ={worst_k['noise_sigma']:.3g})"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2) 旧 POD-band 多尺度摘要（保留）
    # ------------------------------------------------------------------
    multiscale_lines: list[str] = []
    if df_mlp is not None and len(df_mlp) > 0:
        train_meta = mlp_res.get("meta", {}).get("train_cfg", {}) if mlp_res else {}
        p_train = train_meta.get("mask_rate", None)
        s_train = train_meta.get("noise_sigma", None)

        def _pick_row(df: pd.DataFrame, p: float, s: float):
            sel = df[(df["mask_rate"] == p) & (df["noise_sigma"] == s)]
            if len(sel) == 0:
                return None
            return sel.iloc[0]

        row_lin = None
        row_mlp = None
        if p_train is not None and s_train is not None:
            row_lin = _pick_row(df_lin, p_train, s_train)
            row_mlp = _pick_row(df_mlp, p_train, s_train)

        if row_lin is None or row_mlp is None:
            row_mlp = df_mlp.iloc[0]
            p_train = row_mlp["mask_rate"]
            s_train = row_mlp["noise_sigma"]
            row_lin = _pick_row(df_lin, p_train, s_train)

        if row_lin is not None and row_mlp is not None:
            multiscale_lines.append(
                f"选取典型点 p={p_train:.3g}, σ={s_train:.3g}，比较各 POD band 的系数 RMSE / NRMSE："
            )
            band_names = sorted(pod_bands.keys())
            for band_name in band_names:
                key_rmse = f"band_{band_name}"
                key_nrmse = f"band_nrmse_{band_name}"
                val_lin = row_lin.get(key_rmse, float("nan"))
                val_mlp = row_mlp.get(key_rmse, float("nan"))
                val_mlp_nrmse = row_mlp.get(key_nrmse, float("nan"))
                if not (pd.isna(val_lin) or pd.isna(val_mlp)):
                    line = (
                        f"- Band {band_name}: "
                        f"Linear RMSE={val_lin:.4e}, "
                        f"MLP RMSE={val_mlp:.4e}"
                    )
                    if not pd.isna(val_mlp_nrmse):
                        line += f",  MLP NRMSE≈{val_mlp_nrmse:.3f}"
                    multiscale_lines.append(line)

    # ------------------------------------------------------------------
    # 3) Fourier 多尺度摘要（新）
    # ------------------------------------------------------------------
    fourier_lines: list[str] = []
    fourier_cols = [c for c in df_lin.columns if c.startswith("fourier_band_nrmse_")]
    if fourier_cols or ("k_star" in df_lin.columns):
        fourier_lines.append("本次实验同时在空间频域做了尺度评估：")
        fourier_lines.append("- **k\\***：定义为 NRMSE(k) 首次不超过阈值（默认 1.0）的最大可恢复波数。")
        fourier_lines.append("- **Fourier band NRMSE**：把频谱按径向波数分段后，对每段计算归一化误差。")
        if fourier_summary:
            fourier_lines.append("")
            fourier_lines.append("k* 的简单统计：")
            fourier_lines.extend(fourier_summary)

    # ------------------------------------------------------------------
    # 4) 写 Markdown
    # ------------------------------------------------------------------
    lines: list[str] = []
    lines.append(f"# 实验报告：{experiment_name}")
    lines.append("")

    lines.append("## 1. 实验配置概述")
    lines.append("")
    lines.append(f"- 数据集路径: `{meta.get('nc_path', 'N/A')}`")
    lines.append(f"- 空间尺寸: H={H}, W={W}, C={C}")
    lines.append(f"- 时间帧数: T={T}")
    lines.append(f"- POD 截断模态数 r_eff = {r_eff}")
    lines.append(f"- 观测率集合: {mask_rates}")
    lines.append(f"- 噪声强度集合: {noise_sigmas}")
    lines.append(f"- POD bands: {pod_bands}")
    if config_yaml is not None:
        lines.append(f"- 配置文件: `{Path(config_yaml)}`")
    lines.append("")

    lines.append("## 2. 全局重建性能摘要（场空间）")
    lines.append("")
    lines.extend(summary_lines)
    lines.append("")

    if (fig_nmse_vs_mask is not None) or (fig_nmse_vs_noise is not None):
        lines.append("## 3. 全局误差曲线（文件路径）")
        lines.append("")
        if fig_nmse_vs_mask is not None:
            lines.append(f"- NMSE vs 观测率: `{Path(fig_nmse_vs_mask)}`")
        if fig_nmse_vs_noise is not None:
            lines.append(f"- NMSE vs 噪声强度: `{Path(fig_nmse_vs_noise)}`")
        lines.append("")

    if (fig_example_linear is not None) or (fig_example_mlp is not None):
        lines.append("## 4. 典型重建可视化（文件路径）")
        lines.append("")
        if fig_example_linear is not None:
            lines.append(f"- Linear 示例四联图: `{Path(fig_example_linear)}`")
        if fig_example_mlp is not None:
            lines.append(f"- MLP 示例四联图: `{Path(fig_example_mlp)}`")
        lines.append("")

    lines.append("## 5. 量化结果一览（全部组合）")
    lines.append("")
    lines.append("```text")
    lines.append("Linear baseline:")
    lines.append(df_lin.to_string(index=False))
    if df_mlp is not None:
        lines.append("")
        lines.append("MLP baseline:")
        lines.append(df_mlp.to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.append("## 6. 多尺度分析：POD band（旧版保留）")
    lines.append("")
    if multiscale_lines:
        lines.extend(multiscale_lines)
    else:
        lines.append("当前结果中未找到可用的 POD band 多尺度统计信息。")
    if fig_multiscale_example is not None:
        lines.append("")
        lines.append(f"- 典型 POD 多尺度图: `{Path(fig_multiscale_example)}`")
    lines.append("")

    lines.append("## 7. 多尺度分析：Fourier 频域（新版）")
    lines.append("")
    if fourier_lines:
        lines.extend(fourier_lines)
    else:
        lines.append("当前结果未提供 Fourier 频域多尺度信息（可能 fourier_enabled=false 或未写入 entry）。")
    lines.append("")
    lines.append("### 7.1 k* 热力图（文件路径）")
    lines.append("")
    if fig_kstar_linear is not None:
        lines.append(f"- k* heatmap (linear): `{Path(fig_kstar_linear)}`")
    if fig_kstar_mlp is not None:
        lines.append(f"- k* heatmap (mlp): `{Path(fig_kstar_mlp)}`")
    if (fig_kstar_linear is None) and (fig_kstar_mlp is None):
        lines.append("- （无）")
    lines.append("")

    lines.append("### 7.2 Fourier band NRMSE 曲线（文件路径）")
    lines.append("")
    if fig_fourier_band_vs_mask_linear is not None:
        lines.append(f"- Fourier band vs mask (linear): `{Path(fig_fourier_band_vs_mask_linear)}`")
    if fig_fourier_band_vs_mask_mlp is not None:
        lines.append(f"- Fourier band vs mask (mlp): `{Path(fig_fourier_band_vs_mask_mlp)}`")
    if fig_fourier_band_vs_noise_linear is not None:
        lines.append(f"- Fourier band vs noise (linear): `{Path(fig_fourier_band_vs_noise_linear)}`")
    if fig_fourier_band_vs_noise_mlp is not None:
        lines.append(f"- Fourier band vs noise (mlp): `{Path(fig_fourier_band_vs_noise_mlp)}`")
    if (
        fig_fourier_band_vs_mask_linear is None
        and fig_fourier_band_vs_mask_mlp is None
        and fig_fourier_band_vs_noise_linear is None
        and fig_fourier_band_vs_noise_mlp is None
    ):
        lines.append("- （无）")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
