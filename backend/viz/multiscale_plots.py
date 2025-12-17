# backend/viz/multiscale_plots.py

"""
多尺度 POD 误差与能量谱的可视化工具。

包含：
- plot_multiscale_bar: 单模型的 POD band 系数 RMSE 柱状图
- plot_multiscale_summary: 线性 vs MLP 的多尺度四合一对比图
"""

from typing import Dict, Mapping, Sequence, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_multiscale_bar(
    band_errors: Dict[str, float],
    ax: plt.Axes | None = None,
    title: str = "",
) -> plt.Axes:
    """
    绘制一个组合的 POD band 误差柱状图。

    band_errors 形如 {"L": 0.01, "M": 0.02, "H": 0.05}。
    y 轴用线性坐标，数值是系数 RMSE（越低越好）。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    names = list(band_errors.keys())
    values = [float(band_errors[k]) for k in names]

    ax.bar(names, values)
    ax.set_xlabel("POD band")
    ax.set_ylabel("Coefficient RMSE")
    ax.set_title(title or "POD band-wise coefficient error")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    return ax


def plot_multiscale_summary(
    entry: Mapping[str, object],
    *,
    energy_cum: Optional[Sequence[float]] = None,
    title_prefix: str = "",
    model_label: str = "model",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """
    为“单个模型在一组 (p,σ) 上”的结果画出四种多尺度曲线：

    1. Per-band NRMSE vs band index  （条形图）
    2. Per-mode NRMSE vs mode index  （折线图）
    3. 部分重建误差 E_(n,*)          （折线图）
    4. 能量比例 E^(n)                 （累计能量曲线，只跟 POD 有关）

    参数
    ----
    entry:
        run_linear_baseline_experiment / run_mlp_experiment 中某个 (p,σ) 的 entry：
            - "band_nrmse": Dict[str,float]
            - "coeff_nrmse_per_mode": List[float]
            - "field_nmse_partial": Dict[str,float]

    energy_cum:
        POD 模态的累计能量比例数组，长度为 r_eff。

    model_label:
        图例/标题中用于标记该模型的名字，例如 "linear" 或 "mlp"。

    返回
    ----
    fig, (ax_band, ax_mode, ax_partial, ax_energy)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax_band, ax_mode, ax_partial, ax_energy = axes.ravel()

    # ---------------- 1) Per-band NRMSE 条形图 ----------------
    band_nrmse = dict(entry.get("band_nrmse", {})) if entry else {}
    band_names = band_nrmse.keys()

    if band_names:
        x = np.arange(len(band_names))
        y = np.asarray([float(band_nrmse[k]) for k in band_names], dtype=float)

        # 颜色分段：0–0.5 深绿，0.5–0.9 深蓝，0.9+ 深红
        colors: list[str] = []
        for val in y:
            if np.isnan(val):
                colors.append("gray")
            elif val <= 0.5:
                colors.append("darkgreen")
            elif val <= 0.9:
                colors.append("navy")
            else:
                colors.append("darkred")

        bars = ax_band.bar(x, y, width=0.6, color=colors)

        ax_band.set_xticks(x)
        ax_band.set_xticklabels(band_names)
        ax_band.set_xlabel("POD band")
        ax_band.set_ylabel("Per-band NRMSE")
        ax_band.set_title(f"Per-band NRMSE ({model_label})")
        ax_band.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        # 在柱顶标数值
        if len(y) > 0:
            offset = 0.01 * float(np.nanmax(np.abs(y))) if np.isfinite(np.nanmax(np.abs(y))) else 0.0
        else:
            offset = 0.0
        for xi, yi in zip(x, y):
            if np.isnan(yi):
                label = "nan"
                yy = 0.0
            else:
                label = f"{yi:.3f}"
                yy = yi
            ax_band.text(
                xi,
                yy + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    else:
        ax_band.text(
            0.5,
            0.5,
            "No band_nrmse available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_band.transAxes,
        )
        ax_band.set_axis_off()

    # ---------------- 2) Per-mode NRMSE 谱线 ----------------
    coeff_nrmse = entry.get("coeff_nrmse_per_mode", None)
    if coeff_nrmse is not None:
        coeff = np.asarray(coeff_nrmse, dtype=float)
        modes = np.arange(1, coeff.shape[0] + 1)
        ax_mode.plot(modes, coeff, linewidth=1.0)
        ax_mode.set_xlabel("Mode index k")
        ax_mode.set_ylabel("Per-mode NRMSE")
        ax_mode.set_title(f"Per-mode NRMSE vs mode index ({model_label})")
        ax_mode.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax_mode.text(
            0.5,
            0.5,
            "No per-mode NRMSE available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_mode.transAxes,
        )
        ax_mode.set_axis_off()

    # ---------------- 3) 部分重建误差 E_(n,*) ----------------
    field_partial = dict(entry.get("field_nmse_partial", {})) if entry else {}
    part_names = sorted(field_partial.keys())

    if part_names:
        x_idx = np.arange(1, len(part_names) + 1)
        y_vals = [float(field_partial[k]) for k in part_names]
        ax_partial.plot(
            x_idx,
            y_vals,
            marker="o",
            linestyle="-",
            linewidth=1.0,
        )
        ax_partial.set_xticks(x_idx)
        ax_partial.set_xticklabels(part_names, rotation=45, ha="right", fontsize=8)
        ax_partial.set_xlabel("Cumulative band index")
        ax_partial.set_ylabel(r"$E_{(n),\star}$ (NMSE)")
        ax_partial.set_title(f"Partial reconstruction NMSE ({model_label})")
        ax_partial.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    else:
        ax_partial.text(
            0.5,
            0.5,
            "No partial NMSE available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_partial.transAxes,
        )
        ax_partial.set_axis_off()

    # ---------------- 4) 能量比例累积曲线 + 阈值虚线 ----------------
    if energy_cum is not None:
        e = np.asarray(energy_cum, dtype=float)
        n_modes = e.shape[0]
        x_e = np.arange(1, n_modes + 1)

        ax_energy.plot(x_e, e, linewidth=1.0)
        ax_energy.set_xlabel("Mode index n")
        ax_energy.set_ylabel(r"Cumulative energy $\mathcal{E}^{(n)}$")
        ax_energy.set_title("Cumulative POD energy ratio")
        ax_energy.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax_energy.set_ylim(0.0, 1.05)

        # 在 0.9 / 0.99 / 0.999 / 0.9999 四个阈值处标出使用的模态数
        thresholds = [0.9, 0.99, 0.999, 0.9999]
        # 对应的 m1, m2, m3, m4 编号，仅用于标注文本时区分
        labels = ["m1", "m2", "m3", "m4"]

        for thr, lbl in zip(thresholds, labels):
            # 找到最小的 n 使得 e[n-1] >= thr
            idx = int(np.searchsorted(e, thr, side="left")) + 1  # 转成 1-based index
            if idx <= n_modes:
                # 标出 (m, thr) 点
                ax_energy.plot(idx, thr, marker="o", markersize=4, color="k")
                # 竖直虚线：从 (m, 0) 到 (m, thr)
                ax_energy.vlines(
                    idx,
                    0.0,
                    thr,
                    linestyles="--",
                    linewidth=0.8,
                    color="gray",
                )
                # 水平虚线：从 (1, thr) 到 (m, thr)
                ax_energy.hlines(
                    thr,
                    1,
                    idx,
                    linestyles="--",
                    linewidth=0.8,
                    color="gray",
                )
                # 在点上方标注 (m_k, 阈值)
                ax_energy.text(
                    idx,
                    thr + 0.01,
                    f"{lbl}={idx}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    else:
        ax_energy.text(
            0.5,
            0.5,
            "No energy_cum available",
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_energy.transAxes,
        )
        ax_energy.set_axis_off()

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, (ax_band, ax_mode, ax_partial, ax_energy)

def _infer_band_order_from_meta(
    meta: Mapping[str, object] | None,
    example_entry: Mapping[str, object] | None,
) -> list[str]:
    """
    从 result['meta']['pod_bands'] 或某个 entry['band_nrmse'] 的键
    推断 band 的顺序。
    """
    if meta is not None:
        pod_bands = meta.get("pod_bands", None)
        if isinstance(pod_bands, dict):
            try:
                items = list(pod_bands.items())
                # pod_bands[band] = [start, end]，按起始模态排序
                items.sort(
                    key=lambda kv: tuple(kv[1])
                    if hasattr(kv[1], "__iter__") else (0, 0)
                )
                return [name for name, _ in items]
            except Exception:
                pass

    if example_entry is not None:
        band_nrmse = example_entry.get("band_nrmse", None)
        if isinstance(band_nrmse, dict):
            return list(band_nrmse.keys())

    return []

def plot_per_band_nrmse_vs_p(
    linear_results: Mapping[str, object],
    mlp_results: Mapping[str, object] | None = None,
    *,
    sigma_values: Sequence[float] | None = None,
    band_order: Sequence[str] | None = None,
    log_eps: float = 1e-8,
) -> Dict[float, plt.Figure]:
    """
    对每个 σ 画一张图，图内左右双子图：

        左：linear，各 band 的 NRMSE vs p（log y）
        右：mlp，各 band 的 NRMSE vs p（log y）

    注意：左右两个子图的 y 轴 **不共享**，分别按各自数据的范围
    自动设置 log 纵轴的上下界，从而避免 mlp 曲线挤在底部。
    返回 {sigma: Figure}
    """

    def _group_by_sigma(res: Mapping[str, object] | None):
        out: Dict[float, list[Mapping[str, object]]] = {}
        if res is None:
            return out
        entries = res.get("entries", []) or []
        for e in entries:
            try:
                s = float(e["noise_sigma"])
            except Exception:
                continue
            out.setdefault(s, []).append(e)
        for s in list(out.keys()):
            out[s].sort(key=lambda ex: float(ex.get("mask_rate", 0.0)))
        return out

    lin_by_sigma = _group_by_sigma(linear_results)
    mlp_by_sigma = _group_by_sigma(mlp_results)

    all_sigmas = sorted(set(lin_by_sigma.keys()) | set(mlp_by_sigma.keys()))
    if sigma_values is None:
        sigma_values = all_sigmas

    # ---- 推断 band 顺序 ----
    if band_order is None:
        example_entry = None
        if all_sigmas:
            s0 = all_sigmas[0]
            if lin_by_sigma.get(s0):
                example_entry = lin_by_sigma[s0][0]
            elif mlp_by_sigma.get(s0):
                example_entry = mlp_by_sigma[s0][0]
        meta = linear_results.get("meta", None) if hasattr(linear_results, "get") else None
        band_order = _infer_band_order_from_meta(meta, example_entry)

    band_order = list(band_order)
    figs: Dict[float, plt.Figure] = {}

    def _autoscale_log_y(ax: plt.Axes):
        """根据该轴上所有折线数据，自动设置 log y 轴的上下界。"""
        ys_all = []
        for line in ax.get_lines():
            y = np.asarray(line.get_ydata(), dtype=float)
            ys_all.append(y)
        if not ys_all:
            return
        ys = np.concatenate(ys_all)
        mask = np.isfinite(ys) & (ys > log_eps)
        ys = ys[mask]
        if ys.size == 0:
            return
        ymin, ymax = ys.min(), ys.max()
        # 给一点余量，避免贴边
        ax.set_ylim(ymin * 0.8, ymax * 1.2)

    for sigma in sigma_values:
        lin_entries = lin_by_sigma.get(float(sigma), [])
        mlp_entries = mlp_by_sigma.get(float(sigma), [])

        if not lin_entries and not mlp_entries:
            continue

        # 统一 p 轴
        ps = sorted({
            *[float(e.get("mask_rate", 0.0)) for e in lin_entries],
            *[float(e.get("mask_rate", 0.0)) for e in mlp_entries],
        })

        # 有 mlp 就画两个子图，否则只画 linear
        if mlp_results is not None and mlp_entries:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
            ax_lin, ax_mlp = axes
        else:
            fig, ax_lin = plt.subplots(1, 1, figsize=(5, 4))
            ax_mlp = None

        def _plot_model(ax, entries, model_name: str):
            if not entries:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {model_name}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
                return

            idx = {float(e.get("mask_rate", 0.0)): e for e in entries}

            for band in band_order:
                ys = []
                for p in ps:
                    e = idx.get(p, None)
                    if e is None:
                        ys.append(np.nan)
                    else:
                        bn = e.get("band_nrmse", {}) or {}
                        val = bn.get(band, np.nan)
                        v = float(val) if np.isfinite(val) else np.nan
                        if np.isnan(v) or v <= 0.0:
                            v = log_eps
                        ys.append(v)

                ys_arr = np.asarray(ys, dtype=float)
                if np.all(np.isnan(ys_arr)):
                    continue

                ax.plot(
                    ps,
                    ys_arr,
                    marker="o",
                    linewidth=1.0,
                    label=f"{band}",
                )

            ax.set_xlabel("Mask rate p")
            ax.set_ylabel("Per-band NRMSE")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            ax.legend(title=model_name, fontsize=7, ncol=2)

            _autoscale_log_y(ax)

        _plot_model(ax_lin, lin_entries, "linear")
        if ax_mlp is not None:
            _plot_model(ax_mlp, mlp_entries, "mlp")

        fig.suptitle(f"Per-band NRMSE vs p  (σ={sigma})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        figs[float(sigma)] = fig

    return figs

def plot_effective_cutoff_mode_distribution(
    linear_results: Mapping[str, object],
    mlp_results: Mapping[str, object] | None = None,
    *,
    band_order: Sequence[str] | None = None,
    modes_per_band: int = 16,
) -> Dict[str, plt.Figure]:
    """
    绘制“有效截止模态分布图”。

    对于每个模型（linear / mlp）：
        - 横轴: mask_rate p
        - 纵轴: noise_sigma σ
        - 每个 (p,σ) 网格点的数值为 n*：

          a) 基于 per-mode NRMSE 的精细估计:
             - 对 coeff_nrmse_per_mode 做长度为 modes_per_band 的滑动平均，
               得到 e_smooth[k]（对应原始索引区间 [k-w+1, k]）。
             - 找最后一个 e_smooth[k] < 1 的 k*，令 n_window = k*；
               若没有任何 e_smooth < 1，则 n_window = 0。

          b) 基于 per-band NRMSE 的粗略下界:
             - 找出最后一个 band_nrmse < 1 的 band 序号 b*，
               令 n_band = b* * modes_per_band（band 从 1 开始计数）；
               若没有任何 band 满足，则 n_band = 0。

          最终有效截止模态:
              n* = max(n_window, n_band)

          这样可以避免前几个模态的异常 spike 直接把 n* 判为 0，
          同时又能利用 per-mode 信息在 band 的基础上进一步“向后扩展”。
    """

    def _compute_cutoff_from_per_mode(coeff_nrmse, window: int) -> int:
        coeff = np.asarray(coeff_nrmse, dtype=float)
        if coeff.ndim != 1 or coeff.size == 0:
            return 0

        w = min(window, coeff.size)
        if w <= 0:
            return 0

        # 长度为 w 的滑动平均，mode='valid'：
        # smooth[i] 对应原始索引区间 [i, i+w-1]，i 从 0 开始
        kernel = np.ones(w, dtype=float) / float(w)
        smooth = np.convolve(coeff, kernel, mode="valid")  # 长度 r-w+1

        # 对应的“窗口末端索引”k = w..r
        k_indices = np.arange(w, coeff.size + 1, dtype=int)

        mask_ok = smooth < 1.0
        if not np.any(mask_ok):
            return 0

        # 最后一个满足阈值的 k
        k_star = int(k_indices[mask_ok][-1])
        return max(k_star, 0)

    def _compute_cutoff_from_band(
        band_nrmse: Mapping[str, float],
        bands: Sequence[str],
    ) -> int:
        if not bands:
            # 如果没有 band 顺序信息，就退回 dict 的键顺序
            bands = list(band_nrmse.keys())

        last_idx = -1
        for i, name in enumerate(bands):
            v = band_nrmse.get(name, np.nan)
            v = float(v) if v is not None else np.nan
            if np.isfinite(v) and v < 1.0:
                last_idx = i

        if last_idx < 0:
            return 0
        # band 从 1 开始计数，乘以每 band 的模态数
        return (last_idx + 1) * modes_per_band

    def _build_figure(res: Mapping[str, object] | None, model_label: str):
        if res is None:
            return None

        entries = res.get("entries", []) or []
        if not entries:
            return None

        ps = sorted({float(e.get("mask_rate", 0.0)) for e in entries})
        sigmas = sorted({float(e.get("noise_sigma", 0.0)) for e in entries})

        p_to_idx = {p: j for j, p in enumerate(ps)}
        s_to_idx = {s: i for i, s in enumerate(sigmas)}

        grid = np.full((len(sigmas), len(ps)), np.nan, dtype=float)

        for e in entries:
            try:
                p = float(e.get("mask_rate", 0.0))
                s = float(e.get("noise_sigma", 0.0))
            except Exception:
                continue

            band_nrmse = e.get("band_nrmse", {}) or {}
            # 1) band 级别的下界
            n_band = _compute_cutoff_from_band(
                band_nrmse,
                band_order or list(band_nrmse.keys()),
            )

            # 2) per-mode 级别的上修（若有）
            coeff_nrmse = e.get("coeff_nrmse_per_mode", None)
            if coeff_nrmse is not None:
                n_window = _compute_cutoff_from_per_mode(
                    coeff_nrmse,
                    modes_per_band,
                )
            else:
                n_window = 0

            n_eff = max(n_band, n_window)

            i = s_to_idx[s]
            j = p_to_idx[p]
            grid[i, j] = n_eff

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        im = ax.imshow(grid, origin="lower", aspect="auto")

        ax.set_xticks(range(len(ps)))
        ax.set_xticklabels([f"{p:.4g}" for p in ps], rotation=45, ha="right")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f"{s:g}" for s in sigmas])
        ax.set_xlabel("Mask rate p")
        ax.set_ylabel("Noise sigma σ")
        ax.set_title(f"Effective cutoff n*  ({model_label})")

        finite_vals = grid[np.isfinite(grid)]
        mean_val = float(finite_vals.mean()) if finite_vals.size > 0 else 0.0

        for i, s in enumerate(sigmas):
            for j, p in enumerate(ps):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                color = "white" if val > mean_val else "black"
                ax.text(
                    j,
                    i,
                    f"{int(val):d}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )

        fig.colorbar(
            im,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="n* (effective cutoff modes)",
        )
        fig.tight_layout()
        return fig

    # band_order 只在 band 兜底/下界里用；没传的话就推断一次
    if band_order is None:
        entries = linear_results.get("entries", []) or []
        example_entry = entries[0] if entries else None
        meta = linear_results.get("meta", None)
        band_order = _infer_band_order_from_meta(meta, example_entry)

    figs: Dict[str, plt.Figure] = {}
    fig_lin = _build_figure(linear_results, "linear")
    if fig_lin is not None:
        figs["linear"] = fig_lin
    if mlp_results is not None:
        fig_mlp = _build_figure(mlp_results, "mlp")
        if fig_mlp is not None:
            figs["mlp"] = fig_mlp
    return figs

def save_multiscale_summary_figures_from_dir(
    exp_dir: Path | str,
    *,
    modes_per_band: int = 16,
) -> Dict[str, Any]:
    """
    从已有的实验结果目录中加载 linear/mlp 的 JSON/CSV，
    调用多尺度绘图函数，并在该目录下保存：

    - per-band NRMSE vs p, 按 σ 分面：一张 σ 一张图
        文件名: per_band_nrmse_vs_p_sigma{σ}.png

    - 有效截止模态分布图：linear / mlp 各一张
        文件名: effective_cutoff_modes_linear.png
                effective_cutoff_modes_mlp.png

    返回:
    {
        "exp_dir": Path,
        "per_band_figs": { sigma: Path },
        "cutoff_figs": { "linear": Path, "mlp": Path? },
    }
    """
    from backend.eval.reports import load_full_experiment_results
    from backend.viz.fourier_plots import (
        plot_kstar_heatmap,
        plot_fourier_band_nrmse_curves,
    )

    exp_dir = Path(exp_dir)
    # 这里 experiment_name=None，表示 exp_dir 本身就是单次实验目录
    loaded = load_full_experiment_results(exp_dir, experiment_name=None)
    exp_dir = loaded["exp_dir"]
    linear_res = loaded.get("linear", None)
    mlp_res = loaded.get("mlp", None)

    if linear_res is None:
        raise ValueError(f"No linear_results.json found in {exp_dir}")

    per_band_figs: Dict[float, Path] = {}
    cutoff_figs: Dict[str, Path] = {}
    fourier_figs: Dict[str, Path] = {}

    # 1) per-band NRMSE vs p（按 σ 分面）
    figs_sigma = plot_per_band_nrmse_vs_p(
        linear_results=linear_res,
        mlp_results=mlp_res,
    )
    for sigma, fig in figs_sigma.items():
        # 用 σ 的简洁字符串作为文件名的一部分
        sigma_str = f"{sigma:g}".replace(".", "_")
        out_path = exp_dir / f"per_band_nrmse_vs_p_sigma{sigma_str}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        per_band_figs[float(sigma)] = out_path

    # 2) 有效截止模态分布图
    figs_cutoff = plot_effective_cutoff_mode_distribution(
        linear_results=linear_res,
        mlp_results=mlp_res,
        modes_per_band=modes_per_band,
    )
    for model_name, fig in figs_cutoff.items():
        out_path = exp_dir / f"effective_cutoff_modes_{model_name}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        cutoff_figs[model_name] = out_path

    # 3) ===== Batch 6: Fourier figs from CSV (no recompute) =====
    df_lin = loaded.get("df_linear", None)
    df_mlp = loaded.get("df_mlp", None)

    # 3.1 k* heatmap
    try:
        fig = plot_kstar_heatmap(df_lin, df_mlp, model="linear", title="k* heatmap")
        if fig is not None:
            out_path = exp_dir / "kstar_heatmap_linear.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            fourier_figs["kstar_linear"] = out_path
    except Exception:
        pass

    try:
        fig = plot_kstar_heatmap(df_lin, df_mlp, model="mlp", title="k* heatmap")
        if fig is not None:
            out_path = exp_dir / "kstar_heatmap_mlp.png"
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            fourier_figs["kstar_mlp"] = out_path
    except Exception:
        pass

    # 3.2 Fourier band curves
    try:
        # 自动推断 band 名（按列名 fourier_band_nrmse_*）
        band_cols = []
        if df_lin is not None:
            band_cols = [c for c in df_lin.columns if c.startswith("fourier_band_nrmse_")]
        band_names = tuple(c.replace("fourier_band_nrmse_", "") for c in band_cols) if band_cols else ("L", "M", "H")

        figs_fourier = plot_fourier_band_nrmse_curves(df_lin, df_mlp, band_names=band_names)

        # 逐个保存（None 就跳过）
        name_to_file = {
            "fig_fourier_band_vs_mask_linear": "fourier_band_vs_mask_linear.png",
            "fig_fourier_band_vs_mask_mlp": "fourier_band_vs_mask_mlp.png",
            "fig_fourier_band_vs_noise_linear": "fourier_band_vs_noise_linear.png",
            "fig_fourier_band_vs_noise_mlp": "fourier_band_vs_noise_mlp.png",
        }
        for k, fname in name_to_file.items():
            fig = figs_fourier.get(k, None)
            if fig is None:
                continue
            out_path = exp_dir / fname
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            fourier_figs[k] = out_path
    except Exception:
        pass

    return {
        "exp_dir": exp_dir,
        "per_band_figs": per_band_figs,
        "cutoff_figs": cutoff_figs,
        "fourier_figs": fourier_figs,
    }
