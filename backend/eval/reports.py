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

    DataFrame 列示例：
    - model_type
    - mask_rate
    - noise_sigma
    - nmse_mean, nmse_std
    - nmae_mean, nmae_std
    - psnr_mean, psnr_std
    - band_<name>  （例如 band_L, band_M, band_H）
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

    参数
    ----
    all_result:
        run_full_eval_pipeline / quick_full_experiment 的返回值。
        建议包含:
          - "linear", "mlp" (可选)
          - "df_linear", "df_mlp" (可选)
          - "saved_paths" (可选, 内含 CSV / JSON / 图像文件路径)
    out_path:
        输出的 markdown 路径。
    experiment_name:
        报告标题中的实验名称。
    config_yaml:
        若提供，将在报告中附上该 YAML 配置的路径。

    返回
    ----
    out_path (Path)
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

    # 1) 一些 summary 统计：最好 / 最差 NMSE
    best_lin = df_lin.loc[df_lin["nmse_mean"].idxmin()]
    worst_lin = df_lin.loc[df_lin["nmse_mean"].idxmax()]

    summary_lines: list[str] = []

    summary_lines.append(
        f"- Linear baseline 最佳 NMSE = {best_lin['nmse_mean']:.4e} "
        f"(p={best_lin['mask_rate']:.3g}, σ={best_lin['noise_sigma']:.3g})"
    )
    summary_lines.append(
        f"- Linear baseline 最差 NMSE = {worst_lin['nmse_mean']:.4e} "
        f"(p={worst_lin['mask_rate']:.3g}, σ={worst_lin['noise_sigma']:.3g})"
    )

    if df_mlp is not None and len(df_mlp) > 0:
        best_mlp = df_mlp.loc[df_mlp["nmse_mean"].idxmin()]
        worst_mlp = df_mlp.loc[df_mlp["nmse_mean"].idxmax()]

        summary_lines.append(
            f"- MLP baseline 最佳 NMSE = {best_mlp['nmse_mean']:.4e} "
            f"(p={best_mlp['mask_rate']:.3g}, σ={best_mlp['noise_sigma']:.3g})"
        )
        summary_lines.append(
            f"- MLP baseline 最差 NMSE = {worst_mlp['nmse_mean']:.4e} "
            f"(p={worst_mlp['mask_rate']:.3g}, σ={worst_mlp['noise_sigma']:.3g})"
        )

        # 尝试比较同一 (p,σ) 下的提升
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
                    f"- 在 p={p_ref:.3g}, σ={s_ref:.3g} 处，MLP 相比 Linear 的 NMSE "
                    f"相对改善约 {rel_improve:.2f}%"
                )
        except Exception:
            # 配置不完整或筛选失败时，默默跳过即可
            pass

    # 2) 多尺度 band-wise 对比（选取一组典型 (p,σ)，优先训练配置）
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
                f"选取典型点 p={p_train:.3g}, σ={s_train:.3g}，比较各 POD band 的系数 RMSE："
            )

            for band_name in sorted(pod_bands.keys()):
                key = f"band_{band_name}"
                val_lin = row_lin.get(key, float("nan"))
                val_mlp = row_mlp.get(key, float("nan"))
                if not (pd.isna(val_lin) or pd.isna(val_mlp)):
                    multiscale_lines.append(
                        f"- Band {band_name}: Linear={val_lin:.4e}, MLP={val_mlp:.4e}"
                    )

    # 2') 有效模态等级统计（基于 effective_r_cut）
    effective_lines: list[str] = []
    if df_mlp is not None and len(df_mlp) > 0 and "effective_r_cut" in df_mlp.columns:
        df_eff = df_mlp.dropna(subset=["effective_r_cut"]).copy()
        if len(df_eff) > 0:
            # 先看噪声较小的一条曲线下，随着 mask_rate 变化，有效模态数如何变化
            s_min = df_eff["noise_sigma"].min()
            sub_smin = df_eff[df_eff["noise_sigma"] == s_min].sort_values("mask_rate")
            if len(sub_smin) > 0:
                effective_lines.append(
                    f"在噪声较小 (σ≈{s_min:.3g}) 的情况下，不同观测率下 MLP 的有效模态等级为："
                )
                for _, row in sub_smin.iterrows():
                    effective_lines.append(
                        f"- p={row['mask_rate']:.3g} 时，有效模态截止约为 r≈{int(row['effective_r_cut'])} "
                        f"(effective_band='{row.get('effective_band', 'N/A')}')"
                    )

            # 再比较噪声最小 / 最大时的平均有效模态数
            s_max = df_eff["noise_sigma"].max()
            r_min = float(df_eff[df_eff["noise_sigma"] == s_min]["effective_r_cut"].mean())
            r_max = float(df_eff[df_eff["noise_sigma"] == s_max]["effective_r_cut"].mean())

            effective_lines.append("")
            effective_lines.append(
                f"在噪声最小 σ≈{s_min:.3g} 与最大 σ≈{s_max:.3g} 的对比下："
            )
            effective_lines.append(
                f"- 低噪声下平均有效模态数约为 r≈{r_min:.1f}；高噪声下约为 r≈{r_max:.1f}。"
            )

            if r_max < 0.8 * r_min:
                effective_lines.append(
                    "- 随着噪声水平升高，有效可恢复模态数显著下降，高频模态更容易被噪声主导，"
                    "在实际应用中可考虑在 r≈高噪声下的有效截止附近进行自适应截断。"
                )
            else:
                effective_lines.append(
                    "- 在所考察的噪声范围内，有效恢复模态数变化相对缓和，模型对噪声具备一定的尺度鲁棒性。"
                )

    # 3) 拼 Markdown 文本
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
    lines.append("- **NMSE** (Normalized MSE)：对整体能量归一化后的均方误差，越小表示整体重建越准确。")
    lines.append("- **NMAE** (Normalized MAE)：归一化后的平均绝对误差，相比 NMSE 对少量极端异常值不那么敏感。")
    lines.append("- **PSNR** (Peak Signal-to-Noise Ratio)：以 dB 表示的峰值信噪比，越高表示重建场与真值之间的对比度越好。")
    lines.append(
        "- **band_***：在 POD 系数空间中，对低频/中频/高频等 band 的系数 RMSE，"
        "用于刻画不同尺度结构（大尺度流型 vs 细节涡量）的恢复能力。"
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

    # 4. 多尺度分析 + 有效模态等级
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
    lines.append("")

    lines.append("### 4.2 有效模态等级与自适应截断")
    lines.append("")
    if effective_lines:
        lines.extend(effective_lines)
    else:
        lines.append("当前结果中未显式提供 effective_r_cut 信息，无法自动推断有效模态等级。")
    lines.append("")

    # 5. 结果讨论与展望（依然是模板，可基于上方结论做微调）
    lines.append("## 5. 结果讨论与展望（模板）")
    lines.append("")
    lines.append("- 当观测率较低时，线性基线在高频 band 上的误差通常更大，表明高频结构对稀疏观测更敏感；")
    lines.append("- MLP baseline 在中高频 POD band 上往往优于线性基线，特别是在中等噪声水平下更能保持结构细节；")
    lines.append("- 随着观测率提升，两者的性能差距可能逐渐缩小，说明在高采样场景中线性方法已经接近饱和；")
    lines.append(
        "- 结合 4.2 节的有效模态等级分析，可以在 r≈有效截止附近进行自适应截断，"
        "在保证主要流动结构重建质量的前提下，显著压缩模型复杂度与推理成本；"
    )
    lines.append("- 后续可以尝试引入卷积/注意力结构，或在损失中加入物理约束，以进一步改善高频恢复表现与泛化能力。")
    lines.append("")
    lines.append(
        "> 注：以上条目为模板化描述，可根据实际数值结果进行适当修改和细化。"
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
