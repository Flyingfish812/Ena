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

    DataFrame 典型列包括：
    - model_type
    - mask_rate
    - noise_sigma
    - nmse_mean, nmse_std
    - nmae_mean, nmae_std
    - psnr_mean, psnr_std
    - n_frames, n_obs
    - band_<name>             ：各 POD band 的系数 RMSE
    - band_nrmse_<name>       ：各 POD band 的系数 NRMSE
    - group_nmse_<group>      ：场级（单 band）NMSE（φ 分组）
    - partial_nmse_<group>    ：场级（累积到该 band 位置）的 NMSE
    - effective_band          ：启发式判定的“有效 band 名字”
    - effective_r_cut         ：对应的有效模态截止 r 值
    """
    model_type = result.get("model_type", "model")
    entries = result.get("entries", []) or []

    rows: list[dict[str, Any]] = []

    # --- 先扫一遍，收集所有可能出现的 key，保证列名完备 ---
    band_rmse_names: set[str] = set()
    band_nrmse_names: set[str] = set()
    group_names: set[str] = set()
    partial_names: set[str] = set()

    for e in entries:
        band_errors = e.get("band_errors", {}) or {}
        band_rmse_names.update(band_errors.keys())

        band_nrmse = e.get("band_nrmse", {}) or {}
        band_nrmse_names.update(band_nrmse.keys())

        group_err = e.get("field_nmse_per_group", {}) or {}
        group_names.update(group_err.keys())

        partial_err = e.get("field_nmse_partial", {}) or {}
        partial_names.update(partial_err.keys())

    band_rmse_sorted = sorted(band_rmse_names)
    band_nrmse_sorted = sorted(band_nrmse_names)
    group_sorted = sorted(group_names)
    partial_sorted = sorted(partial_names)

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
            # 有效模态等级（如有）
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


def save_full_experiment_results(
    all_result: Dict[str, Any],
    base_dir: Path | str,
    experiment_name: str,
) -> Dict[str, Path]:
    """
    将一次完整实验（linear + mlp）的结果全部保存下来，便于后续复查。

    约定 all_result 的结构为 run_full_eval_pipeline / quick_full_experiment 返回的形式：
        {
            "linear": {...},
            "mlp": {... or None},
            "df_linear": DataFrame,
            "df_mlp": DataFrame or None,
            ...
        }

    保存内容：
    - {base_dir}/{experiment_name}/linear_results.json
    - {base_dir}/{experiment_name}/mlp_results.json (若存在)
    - {base_dir}/{experiment_name}/linear_results.csv
    - {base_dir}/{experiment_name}/mlp_results.csv (若存在)
    """
    base_dir = Path(base_dir) / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {}

    # 1) JSON 原始结果
    linear_res = all_result.get("linear", None)
    if linear_res is not None:
        p_json = base_dir / "linear_results.json"
        with p_json.open("w", encoding="utf-8") as f:
            json.dump(linear_res, f, indent=2, ensure_ascii=False)
        paths["linear_json"] = p_json

    mlp_res = all_result.get("mlp", None)
    if mlp_res is not None:
        p_json = base_dir / "mlp_results.json"
        with p_json.open("w", encoding="utf-8") as f:
            json.dump(mlp_res, f, indent=2, ensure_ascii=False)
        paths["mlp_json"] = p_json

    # 2) CSV 形式
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

    说明（v1.06 更新）：
    - 全局重建性能仍然以场空间 NMSE / NMAE / PSNR 为主。
    - 多尺度部分区分两类量：
        1) band_*/band_nrmse_*：在 POD 系数空间的 per-band 误差
        2) partial_nmse_*     ：在场空间的部分重建 NMSE（累积到某个 band 为止）
    - “有效模态等级”不再依赖 legacy 的 effective_r_cut，而是
      从 partial NMSE 曲线中自动识别“误差几乎不再下降”的 band，
      用于描述模型的有效恢复尺度。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    linear_res = all_result.get("linear", None)
    mlp_res = all_result.get("mlp", None)

    if linear_res is None:
        raise ValueError("all_result['linear'] is required to generate report.")

    # DataFrame: 优先使用已缓存版本, 否则从原始结果构造
    df_lin = all_result.get("df_linear", results_to_dataframe(linear_res))
    df_mlp = None
    if mlp_res is not None:
        df_mlp = all_result.get("df_mlp", results_to_dataframe(mlp_res))

    # 元信息
    meta = linear_res.get("meta", {})
    H = meta.get("H", "?")
    W = meta.get("W", "?")
    C = meta.get("C", "?")
    T = meta.get("T", "?")
    r_eff = meta.get("r_eff", "?")
    pod_bands = meta.get("pod_bands", {})

    mask_rates = linear_res.get("mask_rates", [])
    noise_sigmas = linear_res.get("noise_sigmas", [])

    # 从 all_result 中读取已保存的文件路径(若存在)
    saved_paths: Dict[str, Any] = all_result.get("saved_paths", {}) or {}
    fig_nmse_vs_mask = saved_paths.get("fig_nmse_vs_mask", None)
    fig_nmse_vs_noise = saved_paths.get("fig_nmse_vs_noise", None)
    fig_example_linear = saved_paths.get("fig_example_linear", None)
    fig_example_mlp = saved_paths.get("fig_example_mlp", None)
    fig_multiscale_example = saved_paths.get("fig_multiscale_example", None)

    # ------------------------------------------------------------------
    # 1) 一些 summary 统计：最好 / 最差 NMSE（场空间）
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

        # 尝试比较同一 (p,σ) 下的提升（仍然是场空间 NMSE）
        try:
            p_ref = sorted(set(df_lin["mask_rate"]))[0]
            s_ref = sorted(set(df_lin["noise_sigma"]))[0]

            lin_ref = df_lin[
                (df_lin["mask_rate"] == p_ref) & (df_lin["noise_sigma"] == s_ref)
            ]["nmse_mean"].iloc[0]
            mlp_ref = df_mlp[
                (df_mlp["mask_rate"] == p_ref) & (df_mlp["noise_sigma"] == s_ref)
            ]["nmse_mean"].iloc[0]

            if lin_ref > 0:
                rel_improve = (lin_ref - mlp_ref) / lin_ref * 100.0
                summary_lines.append(
                    f"- 在 p={p_ref:.3g}, σ={s_ref:.3g} 处，MLP 相比 Linear 的场级 NMSE "
                    f"相对改善约 {rel_improve:.2f}%"
                )
        except Exception:
            # 配置不完整或筛选失败时，默默跳过即可
            pass

    # ------------------------------------------------------------------
    # 2) 多尺度 band-wise 对比（系数空间）——只做数值罗列，不乱下“128 模态全恢复”的结论
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
            # fallback: 用 df_mlp 的第一行
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
    # 3) 有效模态等级：从 partial NMSE（场空间）中自动识别“误差几乎不再下降”的 band
    # ------------------------------------------------------------------
    effective_lines: list[str] = []
    if df_mlp is not None and len(df_mlp) > 0:
        # 仍然选用训练配置或其 fallback 那一组 (p,σ)
        train_meta = mlp_res.get("meta", {}).get("train_cfg", {}) if mlp_res else {}
        p_train = train_meta.get("mask_rate", None)
        s_train = train_meta.get("noise_sigma", None)

        def _pick_row(df: pd.DataFrame, p: float, s: float):
            sel = df[(df["mask_rate"] == p) & (df["noise_sigma"] == s)]
            if len(sel) == 0:
                return None
            return sel.iloc[0]

        row_mlp = None
        if p_train is not None and s_train is not None:
            row_mlp = _pick_row(df_mlp, p_train, s_train)

        if row_mlp is None:
            row_mlp = df_mlp.iloc[0]
            p_train = row_mlp["mask_rate"]
            s_train = row_mlp["noise_sigma"]

        # 抽取当前 (p,σ) 下的所有 partial_nmse_* 列
        partial_cols = [c for c in df_mlp.columns if c.startswith("partial_nmse_")]
        partial_cols = sorted(partial_cols)

        if row_mlp is not None and partial_cols:
            effective_lines.append(
                f"选取 MLP 在 p={p_train:.3g}, σ={s_train:.3g} 条件下，"
                "分析 partial NMSE 曲线以估计有效恢复模态等级："
            )
            effective_lines.append("")

            vals: list[tuple[str, float]] = []
            min_val = float("inf")
            for col in partial_cols:
                name = col.replace("partial_nmse_", "")
                val = float(row_mlp[col])
                vals.append((name, val))
                min_val = min(min_val, val)
                effective_lines.append(
                    f"- 累积到 {name} 时，场级 NMSE ≈ {val:.4e}"
                )

            # 定义一个容忍度：到达整体最小值 5% 以内就认为“已基本收敛”
            tol = 0.05
            eff_name = None
            for name, val in vals:
                if val <= (1.0 + tol) * min_val:
                    eff_name = name
                    break

            if eff_name is not None:
                effective_lines.append("")
                effective_lines.append(
                    f"可以看到，累积到 {eff_name} 时，partial NMSE 已接近全局最小值 "
                    f"(在 {tol*100:.1f}% 容忍度以内)，"
                    "继续加入更高频 band 对误差下降的贡献已经非常有限。"
                )
                effective_lines.append(
                    "因此，可以将该累积 band 视为当前配置下的“有效恢复尺度上限”，"
                    "高于该尺度的高频模态主要受能量极低和可辨识性限制影响，"
                    "难以从稀疏有噪观测中可靠重建。"
                )
        else:
            effective_lines.append(
                "当前结果中未显式提供 partial_nmse_* 信息，"
                "无法从场空间曲线自动推断有效模态等级。"
            )

    # ------------------------------------------------------------------
    # 4) 拼 Markdown 文本
    # ------------------------------------------------------------------
    lines: list[str] = []

    # 标题
    lines.append(f"# 实验报告：{experiment_name}")
    lines.append("")

    # 1. 配置概述
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

    # 2. 指标说明 + 全局摘要
    lines.append("## 2. 指标说明与全局重建性能摘要")
    lines.append("")
    lines.append("### 2.1 指标说明")
    lines.append("")
    lines.append(
        "- **NMSE** (Normalized MSE)：在物理场空间计算的归一化均方误差，"
        "形式上为 E[‖û−u‖²]/E[‖u‖²]，直接刻画整体能量意义下的重建精度。"
    )
    lines.append(
        "- **NMAE** (Normalized MAE)：归一化后的平均绝对误差，相比 NMSE 对少量极端异常值不那么敏感。"
    )
    lines.append(
        "- **PSNR** (Peak Signal-to-Noise Ratio)：以 dB 表示的峰值信噪比，越高表示重建场与真值之间的对比度越好。"
    )
    lines.append(
        "- **band_***：在 POD 系数空间中，各 POD band 的系数 RMSE，用于观察不同尺度下系数误差的绝对大小。"
    )
    lines.append(
        "- **band_nrmse_***：在 POD 系数空间中，以模态能量 λₖ 归一化后的 per-band NRMSE，"
        "近似刻画“该 band 的误差能量 / 该 band 真实能量”的比例。"
    )
    lines.append(
        "  场级 NMSE 与模态 NRMSE 之间满足能量加权关系："
        "NMSE_field = (∑ₖ λₖ·NRMSEₖ²)/(∑ₖ λₖ)，"
        "因此即便部分高频 band 的 NRMSE 接近 1，只要其特征值 λₖ 很小，"
        "对整体场级 NMSE 的贡献仍然可以忽略。"
    )
    lines.append(
        "- **partial_nmse_***：在场空间中，累积到某个 band 为止的部分重建 NMSE "
        "（例如 S1, S1+S2, ...），用于分析“加入更多模态后整体误差是否仍显著下降”，"
        "从而推断模型的有效恢复尺度。"
    )
    lines.append("")

    lines.append("### 2.2 全局重建性能摘要")
    lines.append("")
    lines.append("主要结论要点：")
    lines.extend(summary_lines)
    lines.append("")

    # 若存在已保存的典型误差曲线图，则在此列出文件路径
    if (fig_nmse_vs_mask is not None) or (fig_nmse_vs_noise is not None):
        lines.append("### 2.3 典型误差曲线（文件路径）")
        lines.append("")
        lines.append("本次实验中，典型的全局误差曲线已保存为：")
        if fig_nmse_vs_mask is not None:
            lines.append(f"- NMSE vs 观测率: `{Path(fig_nmse_vs_mask)}`")
        if fig_nmse_vs_noise is not None:
            lines.append(f"- NMSE vs 噪声强度: `{Path(fig_nmse_vs_noise)}`")
        lines.append(
            "在撰写论文或正式报告时，可以将上述图像直接嵌入对应章节，"
            "作为整体误差随观测率/噪声变化的可视化证据。"
        )
        lines.append("")

    # 若存在典型重建场图，也在此列出
    if (fig_example_linear is not None) or (fig_example_mlp is not None):
        lines.append("### 2.4 典型重建场图（文件路径）")
        lines.append("")
        lines.append("用于直观展示在典型 (p,σ) 条件下的重建效果：")
        if fig_example_linear is not None:
            lines.append(f"- Linear baseline 示例场图: `{Path(fig_example_linear)}`")
        if fig_example_mlp is not None:
            lines.append(f"- MLP baseline 示例场图: `{Path(fig_example_mlp)}`")
        lines.append("")

    # 3. 量化结果(全部行)
    lines.append("## 3. 量化结果一览（全部组合）")
    lines.append("")
    lines.append("下面给出所有 (mask_rate, noise_sigma) 组合下的统计量，便于逐行查阅和后处理。")
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

    # 4. 多尺度分析 + 有效模态等级（新版逻辑）
    lines.append("## 4. 多尺度（POD band）恢复性能与有效模态等级分析")
    lines.append("")

    lines.append("### 4.1 band-wise 系数误差对比")
    lines.append("")
    if multiscale_lines:
        lines.extend(multiscale_lines)
    else:
        lines.append("当前结果中未找到可用的 band-wise 统计信息。")
    lines.append("")
    lines.append(
        "在正式论文中可以在本节插入对应的 band-wise 柱状图或误差对比图，"
        "用于展示低频/中频/高频结构的恢复能力差异。"
    )
    if fig_multiscale_example is not None:
        lines.append("")
        lines.append(
            f"本次实验中，已将一幅典型的 POD 多尺度误差图保存为："
            f"`{Path(fig_multiscale_example)}`，可直接用于插图。"
        )
    lines.append("")

    lines.append("### 4.2 基于 partial NMSE 的有效模态等级与自适应截断")
    lines.append("")
    if effective_lines:
        lines.extend(effective_lines)
    else:
        lines.append(
            "当前结果中未显式提供 partial_nmse_* 信息，无法自动从场空间曲线推断有效模态等级。"
        )
    lines.append("")

    # 5. 结果讨论与展望（模板）
    lines.append("## 5. 结果讨论与展望（模板）")
    lines.append("")
    lines.append(
        "- 从 4.1 与 4.2 的结果可以看出，模型在 POD 能量主导的低频 / 中低频 band 上"
        "具有较好的重建精度，而在高频 band 上误差显著增大，这与稀疏观测条件下"
        "高频模态本身能量低、可辨识性差的特性相一致；"
    )
    lines.append(
        "- partial NMSE 曲线表明，累积到某一 band 后整体误差已基本收敛，"
        "进一步加入高频模态对场级 NMSE 的改善有限，可以据此选择一个有效截断等级，"
        "在保证主要物理结构重建的前提下控制模型复杂度；"
    )
    lines.append(
        "- 后续可以尝试在 MLP 结构中引入卷积/注意力或物理正则项，"
        "以改善中频尺度的恢复能力，并系统比较不同网络结构在“可恢复能量带宽”上的差异。"
    )
    lines.append("")
    lines.append(
        "> 注：以上条目为模板化描述，可根据实际数值结果进行适当修改和细化。"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
