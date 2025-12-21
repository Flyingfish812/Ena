# backend/pipeline.py

"""
为 notebook / GUI 提供的高层“一键调用”接口。
"""

from pathlib import Path
from typing import Dict, Any, Sequence

from .config.schemas import DataConfig, PodConfig, TrainConfig, EvalConfig
from .pod.compute import build_pod

from .dataio.io_utils import load_numpy, load_json, ensure_dir
from .dataio.nc_loader import load_raw_nc
from .pod.project import project_to_pod, reconstruct_from_pod
from .sampling.masks import generate_random_mask_hw, flatten_mask, apply_mask_flat
from .sampling.noise import add_gaussian_noise
from .models.linear_baseline import solve_pod_coeffs_least_squares
from .metrics.errors import (
    nmse,
    nmae,
    mse,
    mae,
    rmse,
    linf,
    psnr,
    compute_basic_errors,
)
from .viz.field_plots import plot_field_comparison, plot_error_map, plot_recon_quadruple, plot_example_from_npz
from .viz.pod_plots import plot_pod_mode_groups
from .viz.curves import plot_eval_nmse_curves
from .viz.fourier_plots import (
    plot_kstar_heatmap,
    plot_fourier_band_nrmse_curves,
    plot_kstar_curve_from_entry,
    plot_spatial_fourier_band_decomposition,
    plot_spatial_fourier_band_decomposition_examples,
    plot_energy_spectrum_with_band_edges,
)

from .models.train_mlp import train_mlp_on_observations
import numpy as np
import json
import matplotlib.pyplot as plt

from .eval.reconstruction import (
    run_linear_baseline_experiment,
    run_mlp_experiment,
)
from .eval.reports import results_to_dataframe
from .viz.multiscale_plots import plot_multiscale_bar, plot_multiscale_summary
from .config.yaml_io import load_experiment_yaml, save_experiment_yaml  # 可选用

def compute_full_eval_results(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    只做“数值实验”：线性基线 sweep + (可选) MLP sweep，
    返回原始结果和对应的 DataFrame，不进行任何绘图或保存。

    返回结构:
    {
        "linear": linear_results_dict,
        "mlp": mlp_results_dict | None,
        "df_linear": pd.DataFrame,
        "df_mlp": pd.DataFrame | None,
    }
    """
    if verbose:
        print("=== [full-eval] compute_full_eval_results ===")
        print("[full-eval] Running linear baseline sweep ...")

    # 1) 线性基线 sweep
    linear_results = run_linear_baseline_experiment(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        verbose=verbose,
    )
    df_linear = results_to_dataframe(linear_results)

    # 2) 可选的 MLP sweep
    mlp_results = None
    df_mlp = None
    if train_cfg is not None:
        if verbose:
            print("[full-eval] Running MLP sweep ...")
        mlp_results = run_mlp_experiment(
            data_cfg=data_cfg,
            pod_cfg=pod_cfg,
            eval_cfg=eval_cfg,
            train_cfg=train_cfg,
            verbose=verbose,
        )
        df_mlp = results_to_dataframe(mlp_results)

    return {
        "linear": linear_results,
        "mlp": mlp_results,
        "df_linear": df_linear,
        "df_mlp": df_mlp,
    }

def build_eval_figures(
    linear_results: Dict[str, Any],
    mlp_results: Dict[str, Any] | None = None,
    *,
    df_linear: Any | None = None,
    df_mlp: Any | None = None,
    include_example_figures: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    只做“画图”：基于 eval 结果构建所有用于论文/报告的 Figure。

    设计原则（最终版）：
    - 默认 include_example_figures=False：不生成任何 example 级图（四联图、Fourier spatial decomp）。
      example 级图统一由 pipeline.redraw_all_example_figures_from_npz 从 npz 重绘。
    - include_example_figures=True：仅用于调试/对照时，才在这里额外生成 example 级图。
    - Fourier band_names / k_edges / grid_meta 统一从 results['meta'] 或 entry['fourier_curve'] 读取，
      不再硬编码 ("L","M","H")。
    """

    # -----------------------------
    # 0) helpers（只在本函数内部用）
    # -----------------------------
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _safe_build(name: str, fn):
        try:
            out = fn()
            if out is not None:
                _log(f"[build-figs] {name}")
            return out
        except Exception as e:
            _log(f"[build-figs] WARNING: {name} failed: {e}")
            return None

    def _cfg_name(p: float, s: float) -> str:
        return f"p{p:.4f}_sigma{s:.3g}".replace(".", "-")

    def _parse_cfg_name(cfg_name: str) -> tuple[float | None, float | None]:
        try:
            p_part, s_part = cfg_name.split("_sigma")
            p_str = p_part[1:].replace("-", ".")
            s_str = s_part.replace("-", ".")
            return float(p_str), float(s_str)
        except Exception:
            return None, None

    def _ensure_df(results: Dict[str, Any] | None, df: Any | None) -> Any | None:
        if results is None:
            return None
        return df if df is not None else results_to_dataframe(results)

    def _get_examples(results: Dict[str, Any] | None) -> list[Dict[str, Any]]:
        if results is None:
            return []
        return results.get("examples", []) or []

    def _group_examples_by_cfg(examples: list[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
        g: Dict[str, list[Dict[str, Any]]] = {}
        for ex in examples or []:
            try:
                p = float(ex["mask_rate"])
                s = float(ex["noise_sigma"])
            except Exception:
                continue
            cfg = _cfg_name(p, s)
            g.setdefault(cfg, []).append(ex)

        def _frame_key(ex: Dict[str, Any]) -> int:
            try:
                return int(ex.get("frame_idx", 0))
            except Exception:
                return 0

        for cfg, lst in g.items():
            lst.sort(key=_frame_key)
        return g

    def _index_entries(results: Dict[str, Any] | None) -> Dict[tuple[float, float], Dict[str, Any]]:
        idx: Dict[tuple[float, float], Dict[str, Any]] = {}
        if results is None:
            return idx
        for e in (results.get("entries", []) or []):
            try:
                p = float(e.get("mask_rate"))
                s = float(e.get("noise_sigma"))
            except Exception:
                continue
            idx[(p, s)] = e
        return idx

    def _plot_quadruple_from_ex(ex: Dict[str, Any], *, model_key: str, title_prefix: str) -> Any | None:
        """
        model_key: "x_lin" or "x_mlp"
        """
        try:
            p = float(ex["mask_rate"])
            s = float(ex["noise_sigma"])
            x_true = np.asarray(ex["x_true"])
            x_out = np.asarray(ex[model_key])
            x_in = np.asarray(ex["x_interp"])
            mask_hw = np.asarray(ex["mask_hw"]) if "mask_hw" in ex else None
            title = f"{title_prefix} (frame={ex.get('frame_idx', '?')}, p={p:.3g}, σ={s:.3g})"
            return plot_recon_quadruple(
                x_input_hw=x_in,
                x_output_hw=x_out,
                x_target_hw=x_true,
                mask_hw=mask_hw,
                title=title,
            )
        except Exception as e:
            _log(f"[build-figs] WARNING: quadruple plot failed ({title_prefix}): {e}")
            return None

    def _pick_meta() -> Dict[str, Any]:
        meta = (mlp_results.get("meta", {}) if mlp_results is not None else {}) or {}
        if not meta:
            meta = (linear_results.get("meta", {}) or {})
        return meta
    
    def _pick_fourier_meta(*, entry: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        统一 Fourier meta 的来源优先级（从强到弱）：
        1) entry["fourier_meta"] / entry["fourier"]（每个 cfg 独有，最精确）
        2) global meta: _pick_meta() 里的 fourier_*（全局兜底，用于定义图/缺字段时）
        """
        fm: Dict[str, Any] = {}
        if isinstance(entry, dict):
            for k in ("fourier_meta", "fourier"):
                v = entry.get(k)
                if isinstance(v, dict) and v:
                    fm.update(v)

        gm = _pick_meta()
        # 兼容两种风格：要么直接 fourier_grid_meta，要么包在 fourier_meta 里
        if isinstance(gm.get("fourier_meta"), dict):
            fm.update({k: v for k, v in gm["fourier_meta"].items() if v is not None})

        # 把 global 的四个关键键补齐进来（不覆盖 entry 已有）
        for k in ("fourier_k_centers", "fourier_energy_k", "fourier_k_edges", "fourier_grid_meta"):
            v = gm.get(k, None)
            if v is not None and k not in fm:
                fm[k] = v

        # 统一字段名：grid_meta
        if "grid_meta" not in fm:
            if isinstance(fm.get("fourier_grid_meta"), dict):
                fm["grid_meta"] = fm["fourier_grid_meta"]
            elif isinstance(fm.get("grid"), dict):
                fm["grid_meta"] = fm["grid"]

        return fm

    def _band_names_from_anything() -> tuple[str, ...]:
        # 1) meta 优先（新 schema）
        meta = _pick_meta()
        bn = meta.get("fourier_band_names", None)
        if isinstance(bn, (list, tuple)) and len(bn) > 0:
            return tuple(str(x) for x in bn)

        # 2) df 列名兜底（兼容旧保存结果）
        try:
            if df_linear is not None:
                cols = [c for c in df_linear.columns if str(c).startswith("fourier_band_nrmse_")]
                names = tuple(str(c).replace("fourier_band_nrmse_", "") for c in cols)
                if len(names) > 0:
                    return names
        except Exception:
            pass

        return ("L", "M", "H")

    def _k_edges_from_entry_or_meta(entry: Dict[str, Any] | None) -> list[float] | None:
        # per-entry curve 更“就地”
        if entry is not None:
            fc = entry.get("fourier_curve") or {}
            ke = fc.get("k_edges", None)
            if isinstance(ke, (list, tuple)) and len(ke) > 0:
                return [float(x) for x in ke]

        # meta 兜底
        meta = _pick_meta()
        ke = meta.get("fourier_k_edges", None)
        if isinstance(ke, (list, tuple)) and len(ke) > 0:
            return [float(x) for x in ke]

        return None

    def _band_names_from_entry_or_meta(entry: Dict[str, Any] | None) -> tuple[str, ...]:
        if entry is not None:
            fc = entry.get("fourier_curve") or {}
            bn = fc.get("band_names", None)
            if isinstance(bn, (list, tuple)) and len(bn) > 0:
                return tuple(str(x) for x in bn)
        return _band_names_from_anything()

    # -----------------------------
    # 1) DataFrame 准备
    # -----------------------------
    df_linear = _ensure_df(linear_results, df_linear)
    df_mlp = _ensure_df(mlp_results, df_mlp) if mlp_results is not None else None

    # examples（无论 include_example_figures 与否都先取，避免作用域/NameError）
    lin_examples = _get_examples(linear_results)
    mlp_examples = _get_examples(mlp_results)

    # -----------------------------
    # 2) 全局 NMSE 曲线
    # -----------------------------
    curve_figs = _safe_build(
        "Global NMSE curves built via viz.curves.plot_eval_nmse_curves",
        lambda: plot_eval_nmse_curves(df_linear, df_mlp),
    ) or {}

    fig_mask_linear = curve_figs.get("fig_nmse_vs_mask_linear")
    fig_mask_mlp = curve_figs.get("fig_nmse_vs_mask_mlp")
    fig_noise_linear = curve_figs.get("fig_nmse_vs_noise_linear")
    fig_noise_mlp = curve_figs.get("fig_nmse_vs_noise_mlp")

    # -----------------------------
    # 3) per-(p,σ) 四联图（example 级）：默认不生成
    # -----------------------------
    fig_examples_linear: Dict[str, list[Any]] = {}
    fig_examples_mlp: Dict[str, list[Any]] = {}

    if include_example_figures:
        if lin_examples:
            _log("[build-figs] Generating per-(p,σ) linear quadruple figures ...")
            for ex in lin_examples:
                try:
                    p = float(ex["mask_rate"])
                    s = float(ex["noise_sigma"])
                    cfg = _cfg_name(p, s)
                except Exception:
                    continue
                fig = _plot_quadruple_from_ex(ex, model_key="x_lin", title_prefix="Linear")
                if fig is not None:
                    fig_examples_linear.setdefault(cfg, []).append(fig)

        if mlp_examples:
            _log("[build-figs] Generating per-(p,σ) MLP quadruple figures ...")
            for ex in mlp_examples:
                try:
                    p = float(ex["mask_rate"])
                    s = float(ex["noise_sigma"])
                    cfg = _cfg_name(p, s)
                except Exception:
                    continue
                fig = _plot_quadruple_from_ex(ex, model_key="x_mlp", title_prefix="MLP")
                if fig is not None:
                    fig_examples_mlp.setdefault(cfg, []).append(fig)

    # -----------------------------
    # 4) Fourier 多尺度可视化（汇总 + per-cfg 解释图）
    # -----------------------------

    # 4.0 entries index：后面多个 Fourier 图都会用到；提前构建，避免重复
    lin_entry_idx = _index_entries(linear_results)
    mlp_entry_idx = _index_entries(mlp_results)

    def _pick_any_entry_for_fourier_meta() -> Dict[str, Any] | None:
        """从 entries 中挑一个最可能带 fourier 信息的 entry，用于提取全局 grid_meta。"""
        # 优先 linear（因为你现在要先把 linear heatmap 画对）
        for e in (lin_entry_idx.values() or []):
            if isinstance(e, dict) and (e.get("fourier_curve") is not None or e.get("fourier_meta") or e.get("fourier")):
                return e
        # 再尝试 mlp
        for e in (mlp_entry_idx.values() or []):
            if isinstance(e, dict) and (e.get("fourier_curve") is not None or e.get("fourier_meta") or e.get("fourier")):
                return e
        return None

    # 4.0.1 统一提取 heatmap 要用的 grid_meta（用于 Nyquist/ℓ_N 标注与 k*>kN 判无效）
    _entry_for_heatmap = _pick_any_entry_for_fourier_meta()
    fm_heat = _pick_fourier_meta(entry=_entry_for_heatmap)
    grid_meta_heat = fm_heat.get("grid_meta", None)

    # 4.1 k* heatmap（新版：内部会以 ℓ*=1/k* 为主，并根据 grid_meta 计算 Nyquist）
    fig_kstar_linear = _safe_build(
        "k* heatmap (linear)",
        lambda gm=grid_meta_heat: plot_kstar_heatmap(
            df_linear,
            df_mlp,
            model="linear",
            title="Resolvable scale ℓ* heatmap",
            grid_meta=gm,
        ),
    )

    fig_kstar_mlp = None
    if mlp_results is not None:
        fig_kstar_mlp = _safe_build(
            "k* heatmap (mlp)",
            lambda gm=grid_meta_heat: plot_kstar_heatmap(
                df_linear,
                df_mlp,
                model="mlp",
                title="Resolvable scale ℓ* heatmap",
                grid_meta=gm,
            ),
        )

    # 4.2 Fourier band NRMSE curves（随 p/σ）
    band_names_global = _band_names_from_anything()
    fourier_curve_figs = _safe_build(
        "Fourier band curves",
        lambda: plot_fourier_band_nrmse_curves(df_linear, df_mlp, band_names=band_names_global),
    ) or {}

    # 4.3 per-(p,σ) Fourier 解释图：k* 曲线（entry 级，默认保留）
    # all cfg names：至少来自 entries；若 include_example_figures=True 再 union examples
    all_cfg_names: set[str] = set()
    for (p, s) in set(lin_entry_idx.keys()) | set(mlp_entry_idx.keys()):
        all_cfg_names.add(_cfg_name(p, s))

    lin_ex_by_cfg: Dict[str, list[Dict[str, Any]]] = {}
    mlp_ex_by_cfg: Dict[str, list[Dict[str, Any]]] = {}
    if include_example_figures:
        lin_ex_by_cfg = _group_examples_by_cfg(lin_examples)
        mlp_ex_by_cfg = _group_examples_by_cfg(mlp_examples)
        all_cfg_names |= set(lin_ex_by_cfg.keys()) | set(mlp_ex_by_cfg.keys())

    fig_fourier_kstar_curves_linear: Dict[str, Any] = {}
    fig_fourier_kstar_curves_mlp: Dict[str, Any] = {}

    # Fourier spatial decomposition（example 级，默认不生成）
    fig_fourier_decomp_linear: Dict[str, list[Any]] = {}
    fig_fourier_decomp_mlp: Dict[str, list[Any]] = {}

    for cfg in sorted(all_cfg_names):
        p_val, s_val = _parse_cfg_name(cfg)
        if p_val is None or s_val is None:
            continue

        e_lin = lin_entry_idx.get((p_val, s_val))
        e_mlp = mlp_entry_idx.get((p_val, s_val))

        # ----- k* curve (linear) -----
        if e_lin and e_lin.get("fourier_curve") is not None:
            fm = _pick_fourier_meta(entry=e_lin)
            fig = _safe_build(
                f"k* curve (linear) {cfg}",
                lambda e=e_lin, fm_=fm: plot_kstar_curve_from_entry(
                    e,
                    title=f"NRMSE(k) vs spatial scale ℓ [linear] | {cfg}",
                    band_names=fm_.get("band_names", band_names_global),
                    k_edges=fm_.get("k_edges", None),          # 可选：显式传，避免 entry 缺字段时画不出 band
                    grid_meta=fm_.get("grid_meta", None),       # 用于 λ/D 标注等
                ),
            )
            if fig is not None:
                fig_fourier_kstar_curves_linear[cfg] = fig

        # ----- k* curve (mlp) -----
        if e_mlp and e_mlp.get("fourier_curve") is not None:
            fm = _pick_fourier_meta(entry=e_mlp)
            fig = _safe_build(
                f"k* curve (mlp) {cfg}",
                lambda e=e_mlp, fm_=fm: plot_kstar_curve_from_entry(
                    e,
                    title=f"NRMSE(k) vs spatial scale ℓ [mlp] | {cfg}",
                    band_names=fm_.get("band_names", band_names_global),
                    k_edges=fm_.get("k_edges", None),
                    grid_meta=fm_.get("grid_meta", None),
                ),
            )
            if fig is not None:
                fig_fourier_kstar_curves_mlp[cfg] = fig

        # spatial decomposition：仅在 include_example_figures=True 才画
        if include_example_figures:
            # 线性：优先用 e_lin 的 band 定义；没有的话用 e_mlp/meta 兜底
            ex_lin_list = lin_ex_by_cfg.get(cfg, [])
            if ex_lin_list:
                src_entry = e_lin if e_lin is not None else e_mlp
                fm = _pick_fourier_meta(entry=src_entry)
                k_edges = fm.get("k_edges", None) or _k_edges_from_entry_or_meta(src_entry)
                bn = fm.get("band_names", None) or _band_names_from_entry_or_meta(src_entry)
                if k_edges is not None:
                    figs_list: list[Any] = []
                    for ex in ex_lin_list:
                        fi = ex.get("frame_idx", "?")
                        fig = _safe_build(
                            f"Fourier decomp (linear) {cfg} frame={fi}",
                            lambda ex_=ex, ke=k_edges, bn_=bn, cfg_=cfg, fi_=fi: plot_spatial_fourier_band_decomposition(
                                x_true_hw=np.asarray(ex_["x_true"]),
                                x_pred_hw=np.asarray(ex_["x_lin"]),
                                k_edges=ke,
                                band_names=bn_,
                                channel=0,
                                title=f"Fourier bands spatial view (linear) | {cfg_} | frame={fi_}",
                                center_mode="target_mean",
                                robust_q=99.5,
                                max_cols=5,
                            ),
                        )
                        if fig is not None:
                            figs_list.append(fig)
                    if figs_list:
                        fig_fourier_decomp_linear[cfg] = figs_list

            # MLP：优先用 e_mlp 的 band 定义；没有的话用 e_lin/meta 兜底
            ex_mlp_list = mlp_ex_by_cfg.get(cfg, [])
            if ex_mlp_list:
                src_entry = e_mlp if e_mlp is not None else e_lin
                fm = _pick_fourier_meta(entry=src_entry)
                k_edges = fm.get("k_edges", None) or _k_edges_from_entry_or_meta(src_entry)
                bn = fm.get("band_names", None) or _band_names_from_entry_or_meta(src_entry)
                if k_edges is not None:
                    figs_list: list[Any] = []
                    for ex in ex_mlp_list:
                        fi = ex.get("frame_idx", "?")
                        fig = _safe_build(
                            f"Fourier decomp (mlp) {cfg} frame={fi}",
                            lambda ex_=ex, ke=k_edges, bn_=bn, cfg_=cfg, fi_=fi: plot_spatial_fourier_band_decomposition(
                                x_true_hw=np.asarray(ex_["x_true"]),
                                x_pred_hw=np.asarray(ex_["x_mlp"]),
                                k_edges=ke,
                                band_names=bn_,
                                channel=0,
                                title=f"Fourier bands spatial view (mlp) | {cfg_} | frame={fi_}",
                                center_mode="target_mean",
                                robust_q=99.5,
                                max_cols=5,
                            ),
                        )
                        if fig is not None:
                            figs_list.append(fig)
                    if figs_list:
                        fig_fourier_decomp_mlp[cfg] = figs_list

    # 4.4 全局 band 定义图（E(k)+edges）
    fm_global = _pick_fourier_meta(entry=None)  # 只从全局 meta 来
    fig_energy_spectrum = _safe_build(
        "Energy spectrum definition figure",
        lambda: (
            plot_energy_spectrum_with_band_edges(
                k_centers=np.asarray(fm_global.get("fourier_k_centers")),
                energy_k=np.asarray(fm_global.get("fourier_energy_k")),
                k_edges=fm_global.get("fourier_k_edges"),
                band_names=fm_global.get("band_names", band_names_global),
                grid_meta=fm_global.get("grid_meta", None),
                title="Energy spectrum E(k) with band edges",
            )
            if (
                fm_global.get("fourier_k_centers") is not None
                and fm_global.get("fourier_energy_k") is not None
                and fm_global.get("fourier_k_edges") is not None
            )
            else None
        ),
    )

    # -----------------------------
    # 5) assemble return
    # -----------------------------
    return {
        "fig_nmse_vs_mask_linear": fig_mask_linear,
        "fig_nmse_vs_mask_mlp": fig_mask_mlp,
        "fig_nmse_vs_noise_linear": fig_noise_linear,
        "fig_nmse_vs_noise_mlp": fig_noise_mlp,
        # example级（默认 None）
        "fig_examples_linear": fig_examples_linear or None,
        "fig_examples_mlp": fig_examples_mlp or None,
        # Fourier汇总
        "fig_kstar_linear": fig_kstar_linear,
        "fig_kstar_mlp": fig_kstar_mlp,
        "fig_fourier_kstar_curves_linear": fig_fourier_kstar_curves_linear or None,
        "fig_fourier_kstar_curves_mlp": fig_fourier_kstar_curves_mlp or None,
        # Fourier example级（默认 None）
        "fig_fourier_decomp_linear": fig_fourier_decomp_linear or None,
        "fig_fourier_decomp_mlp": fig_fourier_decomp_mlp or None,
        "fig_fourier_energy_spectrum": fig_energy_spectrum,
        # 其他 Fourier 曲线图（plot_fourier_band_nrmse_curves 产生的多个 fig）
        **fourier_curve_figs,
    }

def run_build_pod_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    *,
    verbose: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    顶层 POD 构建入口。

    - 读取原始数据
    - 执行 SVD / POD
    - 截断并保存基底与均值
    - 返回能量谱等元信息，供 notebook/GUI 作图

    参数
    ----
    verbose:
        是否打印中间过程信息。
    plot:
        是否直接画出奇异值谱与累计能量图。
    """
    result = build_pod(data_cfg, pod_cfg, verbose=verbose, plot=plot)
    return result

def run_train_mlp_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: TrainConfig,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    顶层 MLP 训练入口。

    流程：
    1. 若 pod_cfg.save_dir 下没有 POD 结果，则先调用 build_pod 构建；
    2. 加载 Ur / mean / meta；
    3. 加载全量场数据 X_thwc，并展平成 [T,D]；
    4. 生成与 train_cfg.mask_rate 对应的固定空间 mask；
    5. 在该 mask + 噪声强度 train_cfg.noise_sigma 下调用 train_mlp_on_observations 训练；
    6. 返回 model 与训练信息、mask 以及部分元信息，供 notebook / GUI 使用。

    注意：
    - 这里只负责“训练一个 MLP”，不做大规模 sweep；
    - eval sweep 在 run_full_eval_pipeline 里处理。
    """
    # 1) 确保 POD 已经存在
    pod_dir = pod_cfg.save_dir
    Ur_path = pod_dir / "Ur.npy"
    mean_path = pod_dir / "mean_flat.npy"
    meta_path = pod_dir / "pod_meta.json"

    if not (Ur_path.exists() and mean_path.exists() and meta_path.exists()):
        if verbose:
            print(f"[train-pipeline] POD not found in {pod_dir}, building POD first...")
        ensure_dir(pod_dir)
        build_pod(data_cfg, pod_cfg, verbose=verbose, plot=False)
    else:
        if verbose:
            print(f"[train-pipeline] Found existing POD in {pod_dir}, skip rebuilding.")

    # 2) 加载 POD 结果
    Ur = load_numpy(Ur_path)           # [D,r0]
    mean_flat = load_numpy(mean_path)  # [D]
    meta = load_json(meta_path)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]

    # 真正使用的模态数 r_eff
    r_eff = min(pod_cfg.r, r_used)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(
            f"[train-pipeline] meta: T={T}, H={H}, W={W}, C={C}, "
            f"r_used={r_used}, r_eff={r_eff}"
        )

    # 3) 加载全量数据，展平
    if verbose:
        print(f"[train-pipeline] Loading full raw data from {data_cfg.nc_path} ...")
    X_thwc = load_raw_nc(data_cfg)  # [T,H,W,C]
    T2, H2, W2, C2 = X_thwc.shape
    if (T2, H2, W2, C2) != (T, H, W, C):
        raise ValueError(
            f"Data shape {X_thwc.shape} mismatch meta (T={T},H={H},W={W},C={C})"
        )

    D = H * W * C
    X_flat_all = X_thwc.reshape(T, D)

    if verbose:
        print(f"[train-pipeline] X_thwc shape = {X_thwc.shape}, flattened = [{T}, {D}]")

    # 4) 生成固定 mask
    if verbose:
        print(
            f"[train-pipeline] Generating spatial mask with mask_rate={train_cfg.mask_rate:.4f} ..."
        )

    mask_hw = generate_random_mask_hw(H, W, mask_rate=train_cfg.mask_rate, seed=0)
    mask_flat = flatten_mask(mask_hw, C=C)
    n_obs = int(mask_flat.sum())

    if verbose:
        print(
            f"[train-pipeline] total observed entries (with {C} channels) = {n_obs}, "
            f"effective mask_rate ≈ {n_obs / (H * W * C):.4f}"
        )

    # 5) 训练 MLP
    if verbose:
        print(
            f"[train-pipeline] Training MLP with noise_sigma={train_cfg.noise_sigma:.4e}, "
            f"batch_size={train_cfg.batch_size}, max_epochs={train_cfg.max_epochs}, "
            f"lr={train_cfg.lr}"
        )

    model, train_info = train_mlp_on_observations(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=train_cfg.noise_sigma,
        batch_size=train_cfg.batch_size,
        num_epochs=train_cfg.max_epochs,
        lr=train_cfg.lr,
        device=train_cfg.device,
        verbose=verbose,
    )

    result: Dict[str, Any] = {
        "model": model,
        "train_info": train_info,
        "mask_rate": float(train_cfg.mask_rate),
        "noise_sigma": float(train_cfg.noise_sigma),
        "mask_flat": mask_flat,
        "pod": {
            "Ur_eff": Ur_eff,
            "mean_flat": mean_flat,
            "r_eff": r_eff,
            "meta": meta,
        },
        "data_cfg": {
            "nc_path": str(data_cfg.nc_path),
            "var_keys": data_cfg.var_keys,
        },
        "train_cfg": {
            "mask_rate": train_cfg.mask_rate,
            "noise_sigma": train_cfg.noise_sigma,
            "hidden_dims": train_cfg.hidden_dims,
            "lr": train_cfg.lr,
            "batch_size": train_cfg.batch_size,
            "max_epochs": train_cfg.max_epochs,
            "device": train_cfg.device,
            "save_dir": str(train_cfg.save_dir),
        },
    }

    if verbose:
        print("[train-pipeline] Training finished.")

    return result

def run_full_eval_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    顶层评估入口：一行代码跑完整的小论文实验 sweep。

    v1.08 起：
    - 先调用 compute_full_eval_results 做纯数值实验（linear + 可选 mlp）
    - 再调用 build_eval_figures 统一生成所有 Figure
    - 返回 dict 里同时包含结果和图像句柄
    """
    if verbose:
        print("=== [full-eval] Start full evaluation pipeline ===")

    eval_results = compute_full_eval_results(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        verbose=verbose,
    )

    figs = build_eval_figures(
        linear_results=eval_results["linear"],
        mlp_results=eval_results.get("mlp"),
        df_linear=eval_results.get("df_linear"),
        df_mlp=eval_results.get("df_mlp"),
        verbose=verbose,
    )

    if verbose:
        print("[full-eval] Done.")

    # 汇总：保持原有字段命名，兼容 quick_full_experiment / run_experiment_from_yaml
    result: Dict[str, Any] = {
        "linear": eval_results["linear"],
        "mlp": eval_results.get("mlp"),
        "df_linear": eval_results.get("df_linear"),
        "df_mlp": eval_results.get("df_mlp"),
    }
    result.update(figs)
    return result

def quick_figs_from_saved_experiment(
    base_dir: str | Path,
    experiment_name: str,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    从磁盘已保存的实验结果目录中恢复所有“结果级别”的 Figure，
    不再重新跑任何训练 / 评估。

    约定:
        - base_dir/experiment_name 目录下存在由 save_full_experiment_results
          写出的 linear_results.json / mlp_results.json / *.csv 等文件。
    """
    from .eval.reports import load_full_experiment_results

    if verbose:
        print(
            f"[quick-figs] Loading numeric results from "
            f"{Path(base_dir) / experiment_name} ..."
        )

    loaded = load_full_experiment_results(
        base_dir=base_dir,
        experiment_name=experiment_name,
    )

    figs = build_eval_figures(
        linear_results=loaded["linear"],
        mlp_results=loaded.get("mlp"),
        df_linear=loaded.get("df_linear"),
        df_mlp=loaded.get("df_mlp"),
        verbose=verbose,
    )

    if verbose:
        print("[quick-figs] Figures rebuilt from saved numeric results.")

    return figs

def quick_build_pod(
    nc_path: str | Path,
    save_dir: str | Path = "artifacts/pod",
    r: int = 128,
    center: bool = True,
    var_keys: tuple[str, ...] = ("u", "v"),
    *,
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    提供给 notebook 的“一行跑 POD”接口。

    示例用法（ipynb）：
    >>> from backend.pipeline import quick_build_pod
    >>> res = quick_build_pod("data/my.nc", "artifacts/pod_nc_r128", r=128)

    参数
    ----
    nc_path:
        NetCDF 文件路径。
    save_dir:
        POD 基底输出目录。
    r:
        截断模态数。
    center:
        是否去均值。
    var_keys:
        需要读取的变量名元组，例如 ("u","v")。
    verbose:
        是否打印中间过程信息。
    plot:
        是否绘制奇异值谱与累计能量曲线。

    返回
    ----
    result:
        与 build_pod 返回结构相同，多一个可选的 "fig_pod"。
    """
    data_cfg = DataConfig(
        nc_path=Path(nc_path),
        var_keys=var_keys,
        cache_dir=None,
    )
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=Path(save_dir),
    )
    return run_build_pod_pipeline(
        data_cfg,
        pod_cfg,
        verbose=verbose,
        plot=plot,
    )

def quick_test_linear_baseline(
    nc_path: str | Path,
    pod_dir: str | Path = "artifacts/pod",
    *,
    r: int = 128,
    center: bool = True,
    var_keys: tuple[str, ...] = ("u", "v"),
    frame_idx: int = 0,
    mask_rate: float = 0.02,
    noise_sigma: float = 0.01,
    max_modes: int = 64,
    modes_per_fig: int = 16,  # 现在理解为"group_size"
    channel: int = 0,
    save_dir: str | Path | None = None,
    fig_prefix: str = "quick_linear",
    save_dpi: int = 300,
    centered_pod: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一行代码测试完整流程（单帧）：

    - 若 pod_dir 下不存在 POD 结果，则自动构建 POD。
    - 加载 Ur / mean / meta。
    - 从 nc_path 中取一帧 snapshot。
    - 做 POD 截断自重建，计算一组误差指标 (MSE/MAE/RMSE/L∞/NMSE/NMAE/PSNR)。
    - 生成随机 mask + 加噪观测。
    - 线性最小二乘解 POD 系数，重建场，计算同一组误差指标。
    - 可视化：
        - Figure 1: [0]=True, [1]=POD truncation, [2]=Linear baseline
        - Figure 2: Linear 误差图
        - Figure 3: POD 模态分组叠加（每 group_size=modes_per_fig 个模态一组）

    若 save_dir 不为 None，则自动将上述 Figure 保存到该目录。

    返回 result 中会包含以上误差与 Figure。
    """
    nc_path = Path(nc_path)
    pod_dir = Path(pod_dir)

    data_cfg = DataConfig(
        nc_path=nc_path,
        var_keys=var_keys,
        cache_dir=None,
    )
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=pod_dir,
    )

    # 1) 检查 / 构建 POD
    Ur_path = pod_dir / "Ur.npy"
    mean_path = pod_dir / "mean_flat.npy"
    meta_path = pod_dir / "pod_meta.json"

    if not (Ur_path.exists() and mean_path.exists() and meta_path.exists()):
        if verbose:
            print(f"[quick_test] POD artifacts not found in {pod_dir}, building POD...")
        ensure_dir(pod_dir)
        _ = run_build_pod_pipeline(
            data_cfg,
            pod_cfg,
            verbose=verbose,
            plot=True,
        )
    else:
        if verbose:
            print(f"[quick_test] Found existing POD in {pod_dir}, skip rebuilding.")

    # 2) 加载 POD
    if verbose:
        print("[quick_test] Loading POD artifacts...")

    Ur = load_numpy(Ur_path)
    mean_flat = load_numpy(mean_path)
    meta = load_json(meta_path)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}")
        if r > r_used:
            print(f"  [warn] requested r={r} > r_used={r_used}, 实际最多只用 r_eff={r_used} 阶。")

    # 3) 取一帧 snapshot
    if frame_idx < 0 or frame_idx >= T:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {T-1}]")

    if verbose:
        print(f"[quick_test] Loading raw data from {nc_path} ...")

    X_thwc = load_raw_nc(data_cfg)
    x = X_thwc[frame_idx]      # [H,W,C]
    x_flat = x.reshape(-1)     # [D]

    if verbose:
        print(f"  - Using frame_idx={frame_idx}, x shape = {x.shape}, flat D={x_flat.size}")

    # 截断 Ur 到 r_eff
    r_eff = min(r, Ur.shape[1])
    Ur_eff = Ur[:, :r_eff]

    # 4) POD 截断自重建 + 误差
    if verbose:
        print("[quick_test] POD truncation self-reconstruction...")
        print(f"  -> 使用 r_eff = {r_eff} 个模态，对该帧 snapshot 投影后叠加重构")

    a_true = project_to_pod(x_flat, Ur_eff, mean_flat)
    x_pod_flat = reconstruct_from_pod(a_true, Ur_eff, mean_flat)
    x_pod = x_pod_flat.reshape(H, W, C)

    metrics_pod = compute_basic_errors(x_pod, x, data_range=None)
    nmse_pod = metrics_pod["nmse"]
    if verbose:
        print(
            "  -> POD truncation errors: "
            f"NMSE={metrics_pod['nmse']:.4e}, "
            f"NMAE={metrics_pod['nmae']:.4e}, "
            f"MAE={metrics_pod['mae']:.4e}, "
            f"L∞={metrics_pod['linf']:.4e}"
        )

    # 5) mask + 噪声
    if verbose:
        print("[quick_test] Generating mask and noisy observations...")

    mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
    mask_flat = flatten_mask(mask_hw, C=C)
    n_obs = int(mask_flat.sum())

    if verbose:
        print(f"  - mask_rate target = {mask_rate:.4f}, actual ≈ {n_obs / (H*W*C):.4f} over H×W×C")
        print(f"  - total observed entries (with {C} channels) = {n_obs}")

    y = apply_mask_flat(x_flat, mask_flat)
    y_noisy = add_gaussian_noise(y, sigma=noise_sigma, seed=0)
    if centered_pod:
        mean_obs = apply_mask_flat(mean_flat, mask_flat)
        y_noisy = y_noisy - mean_obs

    # 6) 线性最小二乘重建 + 误差
    if verbose:
        print("[quick_test] Linear least-squares reconstruction in POD space...")

    Ur_masked = Ur_eff[mask_flat, :]   # [M, r_eff]
    a_lin = solve_pod_coeffs_least_squares(y_noisy, Ur_masked)
    x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
    x_lin = x_lin_flat.reshape(H, W, C)

    metrics_linear = compute_basic_errors(x_lin, x, data_range=None)
    nmse_linear = metrics_linear["nmse"]
    if verbose:
        print(
            "  -> Linear baseline errors: "
            f"NMSE={metrics_linear['nmse']:.4e}, "
            f"NMAE={metrics_linear['nmae']:.4e}, "
            f"MAE={metrics_linear['mae']:.4e}, "
            f"L∞={metrics_linear['linf']:.4e}"
        )

    # 7) 场对比图
    if verbose:
        print("[quick_test] Plotting field comparison and error map...")
        print("  -> Figure layout: [0]=True, [1]=POD truncation, [2]=Linear baseline")

    fig_fields = plot_field_comparison(
        x_true_hw=x,
        x_lin_hw=x_pod,
        x_nn_hw=x_lin,
        names=(
            "True field",
            f"POD truncation (r={r_eff})",
            "Linear baseline",
        ),
    )
    fig_fields.suptitle("True vs POD truncation vs Linear baseline", fontsize=12)

    fig_error = plot_error_map(
        x_true_hw=x,
        x_hat_hw=x_lin,
        title="|Linear baseline - True|",
    )

    # 8) POD 模态分组图
    if verbose:
        print("[quick_test] Plotting POD mode groups (partial sums)...")
        print(f"  -> 可视化前 {min(max_modes, r_eff)} 个模态，每组 {modes_per_fig} 个相加为一幅图")

    fig_modes = plot_pod_mode_groups(
        Ur_eff,
        H=H,
        W=W,
        C=C,
        max_modes=max_modes,
        group_size=modes_per_fig,
        channel=channel,
    )

    # 9) 如需，将 Figure 保存到磁盘
    if save_dir is not None:
        save_path = Path(save_dir)
        ensure_dir(save_path)
        base = (
            f"{fig_prefix}_frame{frame_idx}_p{mask_rate:.4f}_sigma{noise_sigma:.3g}"
            .replace(".", "-")
        )
        fig_fields.savefig(
            save_path / f"{base}_fields.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        fig_error.savefig(
            save_path / f"{base}_error.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        fig_modes.savefig(
            save_path / f"{base}_modes.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        if verbose:
            print(f"[quick_test] Figures saved to {save_path}")

    if verbose:
        print("[quick_test] Done.")

    result: Dict[str, Any] = {
        "nmse_pod": nmse_pod,
        "nmse_linear": nmse_linear,
        "metrics_pod": metrics_pod,
        "metrics_linear": metrics_linear,
        "frame_idx": frame_idx,
        "mask_rate": mask_rate,
        "noise_sigma": noise_sigma,
        "n_obs": n_obs,
        "fig_fields": fig_fields,
        "fig_error": fig_error,
        "fig_modes": fig_modes,
        "meta": meta,
    }
    return result

def quick_test_mlp_baseline(
    nc_path: str | Path,
    pod_dir: str | Path = "artifacts/pod",
    *,
    r: int = 128,
    center: bool = True,
    var_keys: tuple[str, ...] = ("u", "v"),
    frame_idx: int = 0,
    mask_rate: float = 0.02,
    noise_sigma: float = 0.01,
    mlp_noise_sigma: float | None = None,
    batch_size: int = 64,
    num_epochs: int = 50,
    lr: float = 1e-3,
    max_modes: int = 64,
    modes_per_fig: int = 16,
    channel: int = 0,
    save_dir: str | Path | None = None,
    fig_prefix: str = "quick_mlp",
    save_dpi: int = 300,
    centered_pod: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一行代码测试完整流程（单帧，包含 MLP）：

    - 若无 POD 则自动构建 POD；
    - 基于整段时间序列构建 ObservationDataset，训练 MLP：
          y (masked noisy obs) -> a_true (POD coeffs)
    - 在指定 frame_idx 上计算：
          - POD 截断自身误差（全家桶: MSE/MAE/RMSE/L∞/NMSE/NMAE/PSNR）
          - 线性 least-squares baseline 误差
          - MLP baseline 误差
    - 可视化：
          Figure 1: True / POD / Linear / MLP 四列对比
          Figure 2: MLP 误差图 |MLP - True|
          Figure 3: POD 模态分组叠加（每组 modes_per_fig 个模态求和）

    若 save_dir 不为 None，则自动将上述 Figure 保存到该目录。

    返回 result 字典。
    """
    nc_path = Path(nc_path)
    pod_dir = Path(pod_dir)

    data_cfg = DataConfig(
        nc_path=nc_path,
        var_keys=var_keys,
        cache_dir=None,
    )
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=pod_dir,
    )

    # 1) 检查 POD
    Ur_path = pod_dir / "Ur.npy"
    mean_path = pod_dir / "mean_flat.npy"
    meta_path = pod_dir / "pod_meta.json"

    if not (Ur_path.exists() and mean_path.exists() and meta_path.exists()):
        if verbose:
            print(f"[quick_test_mlp] POD artifacts not found in {pod_dir}, building POD...")
        ensure_dir(pod_dir)
        _ = run_build_pod_pipeline(
            data_cfg,
            pod_cfg,
            verbose=verbose,
            plot=True,
        )
    else:
        if verbose:
            print(f"[quick_test_mlp] Found existing POD in {pod_dir}, skip rebuilding.")

    # 2) 加载 POD
    if verbose:
        print("[quick_test_mlp] Loading POD artifacts...")

    Ur = load_numpy(Ur_path)
    mean_flat = load_numpy(mean_path)
    meta = load_json(meta_path)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}")
        if r > r_used:
            print(f"  [warn] requested r={r} > r_used={r_used}, 实际最多只用 r_eff={r_used} 阶。")

    # 截断 Ur 到 r_eff
    r_eff = min(r, Ur.shape[1])
    Ur_eff = Ur[:, :r_eff]

    # 3) 读取全部数据并展平
    if verbose:
        print(f"[quick_test_mlp] Loading full raw data from {nc_path} ...")

    X_thwc = load_raw_nc(data_cfg)          # [T,H,W,C]
    if X_thwc.shape[0] != T:
        # 理论上不会出错，只是稳一下
        T = X_thwc.shape[0]

    X_flat_all = X_thwc.reshape(T, -1)      # [T,D]
    D = X_flat_all.shape[1]
    if verbose:
        print(f"  - X_thwc shape = {X_thwc.shape}, flattened = [{T}, {D}]")

    # 4) 生成固定 mask
    if verbose:
        print("[quick_test_mlp] Generating fixed spatial mask...")

    mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
    mask_flat = flatten_mask(mask_hw, C=C)
    n_obs = int(mask_flat.sum())

    if verbose:
        print(f"  - mask_rate target = {mask_rate:.4f}, actual ≈ {n_obs / (H*W*C):.4f} over H×W×C")
        print(f"  - total observed entries (with {C} channels) = {n_obs}")

    # 5) POD 截断自重建（在目标 frame_idx 上）+ 误差
    if frame_idx < 0 or frame_idx >= T:
        raise IndexError(f"frame_idx {frame_idx} out of range [0, {T-1}]")

    if verbose:
        print("[quick_test_mlp] POD truncation self-reconstruction on target frame...")
        print(f"  -> 使用 r_eff = {r_eff} 个模态，对该帧 snapshot 投影后叠加重构")

    x = X_thwc[frame_idx]             # [H,W,C]
    x_flat = x.reshape(-1)            # [D]

    a_true_frame = project_to_pod(x_flat, Ur_eff, mean_flat)
    x_pod_flat = reconstruct_from_pod(a_true_frame, Ur_eff, mean_flat)
    x_pod = x_pod_flat.reshape(H, W, C)

    metrics_pod = compute_basic_errors(x_pod, x, data_range=None)
    nmse_pod = metrics_pod["nmse"]
    if verbose:
        print(
            "  -> POD truncation errors: "
            f"NMSE={metrics_pod['nmse']:.4e}, "
            f"NMAE={metrics_pod['nmae']:.4e}, "
            f"MAE={metrics_pod['mae']:.4e}, "
            f"L∞={metrics_pod['linf']:.4e}"
        )

    # 6) 线性 least-squares baseline（同一 mask）+ 误差
    if verbose:
        print("[quick_test_mlp] Linear least-squares baseline on target frame...")

    y_true = apply_mask_flat(x_flat, mask_flat)        # [M]
    y_true_noisy = add_gaussian_noise(y_true, sigma=noise_sigma, seed=0)
    if centered_pod:
        mean_obs = apply_mask_flat(mean_flat, mask_flat)
        y_true_noisy = y_true_noisy - mean_obs

    Ur_masked = Ur_eff[mask_flat, :]                   # [M,r_eff]
    a_lin = solve_pod_coeffs_least_squares(y_true_noisy, Ur_masked)
    x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
    x_lin = x_lin_flat.reshape(H, W, C)

    metrics_linear = compute_basic_errors(x_lin, x, data_range=None)
    nmse_linear = metrics_linear["nmse"]
    if verbose:
        print(
            "  -> Linear baseline errors: "
            f"NMSE={metrics_linear['nmse']:.4e}, "
            f"NMAE={metrics_linear['nmae']:.4e}, "
            f"MAE={metrics_linear['mae']:.4e}, "
            f"L∞={metrics_linear['linf']:.4e}"
        )

    # 7) 训练 MLP
    if mlp_noise_sigma is None:
        mlp_noise_sigma = noise_sigma

    if verbose:
        print("[quick_test_mlp] Training MLP on full time series observations...")
        print(f"  -> MLP noise_sigma={mlp_noise_sigma}, batch_size={batch_size}, "
              f"num_epochs={num_epochs}, lr={lr}")

    model_mlp, train_info = train_mlp_on_observations(
        X_flat_all=X_flat_all,
        Ur_eff=Ur_eff,
        mean_flat=mean_flat,
        mask_flat=mask_flat,
        noise_sigma=mlp_noise_sigma,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr=lr,
        verbose=verbose,
    )

    # 8) 用 MLP 在目标帧上推断 + 误差
    if verbose:
        print("[quick_test_mlp] Inference with trained MLP on target frame...")
        print("  -> 注意: 使用与训练相同的固定掩膜 mask_flat")

    model_mlp.eval()
    import torch
    device = next(model_mlp.parameters()).device

    y_frame = apply_mask_flat(x_flat, mask_flat)               # [M]
    y_frame_noisy = add_gaussian_noise(y_frame, sigma=noise_sigma, seed=123)
    if centered_pod:
        mean_obs = apply_mask_flat(mean_flat, mask_flat)
        y_frame_noisy = y_frame_noisy - mean_obs

    y_tensor = torch.from_numpy(y_frame_noisy.astype(np.float32)).to(device)
    with torch.no_grad():
        a_mlp_tensor = model_mlp(y_tensor[None, :])            # [1,r_eff]
    a_mlp = a_mlp_tensor.cpu().numpy().reshape(-1)             # [r_eff]

    x_mlp_flat = reconstruct_from_pod(a_mlp, Ur_eff, mean_flat)
    x_mlp = x_mlp_flat.reshape(H, W, C)

    metrics_mlp = compute_basic_errors(x_mlp, x, data_range=None)
    nmse_mlp = metrics_mlp["nmse"]
    if verbose:
        print(
            "  -> MLP baseline errors: "
            f"NMSE={metrics_mlp['nmse']:.4e}, "
            f"NMAE={metrics_mlp['nmae']:.4e}, "
            f"MAE={metrics_mlp['mae']:.4e}, "
            f"L∞={metrics_mlp['linf']:.4e}"
        )

    # 9) 场对比 + 误差可视化
    if verbose:
        print("[quick_test_mlp] Plotting field comparison and error map...")
        print("  -> Figure layout: [0]=True, [1]=POD truncation, "
              "[2]=Linear baseline, [3]=MLP baseline")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))

    fields = [
        ("True field", x[..., channel]),
        (f"POD truncation (r={r_eff})", x_pod[..., channel]),
        ("Linear baseline", x_lin[..., channel]),
        ("MLP baseline", x_mlp[..., channel]),
    ]

    vmin = min(f.min() for _, f in fields)
    vmax = max(f.max() for _, f in fields)

    for ax, (name, f) in zip(axes, fields, strict=False):
        im = ax.imshow(f, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("True vs POD vs Linear vs MLP", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_fields_all = fig

    fig_error_mlp = plot_error_map(
        x_true_hw=x,
        x_hat_hw=x_mlp,
        title="|MLP baseline - True|",
    )

    # 10) POD 模态分组叠加
    if verbose:
        print("[quick_test_mlp] Plotting POD mode groups (partial sums)...")
        print(f"  -> 可视化前 {min(max_modes, r_eff)} 个模态，每组 {modes_per_fig} 个相加为一幅图")

    fig_modes = plot_pod_mode_groups(
        Ur_eff,
        H=H,
        W=W,
        C=C,
        max_modes=max_modes,
        group_size=modes_per_fig,
        channel=channel,
    )

    # 11) 如需，将 Figure 保存到磁盘
    if save_dir is not None:
        save_path = Path(save_dir)
        ensure_dir(save_path)
        base = (
            f"{fig_prefix}_frame{frame_idx}_p{mask_rate:.4f}_sigma{noise_sigma:.3g}"
            .replace(".", "-")
        )
        fig_fields_all.savefig(
            save_path / f"{base}_fields.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        fig_error_mlp.savefig(
            save_path / f"{base}_error_mlp.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        fig_modes.savefig(
            save_path / f"{base}_modes.png",
            dpi=save_dpi,
            bbox_inches="tight",
        )
        if verbose:
            print(f"[quick_test_mlp] Figures saved to {save_path}")

    if verbose:
        print("[quick_test_mlp] Done.")

    result: Dict[str, Any] = {
        "nmse_pod": nmse_pod,
        "nmse_linear": nmse_linear,
        "nmse_mlp": nmse_mlp,
        "metrics_pod": metrics_pod,
        "metrics_linear": metrics_linear,
        "metrics_mlp": metrics_mlp,
        "frame_idx": frame_idx,
        "mask_rate": mask_rate,
        "noise_sigma": noise_sigma,
        "n_obs": n_obs,
        "train_info": train_info,
        "fig_fields": fig_fields_all,
        "fig_error_mlp": fig_error_mlp,
        "fig_modes": fig_modes,
        "meta": meta,
    }
    return result

def quick_full_experiment(
    nc_path: str | Path = "data/cylinder2d.nc",
    *,

    var_keys: tuple[str, ...] = ("u", "v"),
    r: int = 128,
    center: bool = True,
    mask_rates: Sequence[float] | None = None,
    noise_sigmas: Sequence[float] | None = None,
    pod_bands: Dict[str, tuple[int, int]] | None = None,
    train_mask_rate: float = 0.02,
    train_noise_sigma: float = 0.01,
    hidden_dims: tuple[int, ...] = (256, 256),
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 50,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一行跑完“小论文主实验”的高层接口。
    """
    nc_path = Path(nc_path)

    if mask_rates is None:
        mask_rates = [0.01, 0.02, 0.05, 0.10]
    if noise_sigmas is None:
        noise_sigmas = [0.0, 0.01, 0.02]

    if pod_bands is None:
        r_L = min(16, r)
        r_M = min(64, r)
        pod_bands = {
            "L": (0, r_L),
            "M": (r_L, r_M),
            "H": (r_M, r),
        }

    data_cfg = DataConfig(
        nc_path=nc_path,
        var_keys=var_keys,
        cache_dir=None,
    )

    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=Path(f"artifacts/pod_r{r}"),
    )

    eval_cfg = EvalConfig(
        mask_rates=list(mask_rates),
        noise_sigmas=list(noise_sigmas),
        pod_bands=pod_bands,
        save_dir=Path("artifacts/eval"),
    )

    train_cfg = TrainConfig(
        mask_rate=float(train_mask_rate),
        noise_sigma=float(train_noise_sigma),
        hidden_dims=hidden_dims,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        device=device,
        save_dir=Path("artifacts/nn")
        / f"p{train_mask_rate:.4f}_sigma{train_noise_sigma:.3g}".replace(".", "-"),
    )

    result = run_full_eval_pipeline(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        verbose=verbose,
    )

    # ===== 代表性多尺度图：linear / mlp 各一张 =====
    fig_ms_lin = None
    fig_ms_mlp = None

    lin_results = result.get("linear", None)
    mlp_results = result.get("mlp", None)

    if lin_results is not None and mlp_results is not None:
        lin_entries = lin_results.get("entries", []) or []
        mlp_entries = mlp_results.get("entries", []) or []

        if lin_entries and mlp_entries:
            def _build_index(entries):
                m = {}
                for e in entries:
                    p = float(e.get("mask_rate", -1.0))
                    s = float(e.get("noise_sigma", -1.0))
                    m[(p, s)] = e
                return m

            lin_idx = _build_index(lin_entries)
            mlp_idx = _build_index(mlp_entries)

            train_meta = mlp_results.get("meta", {}).get("train_cfg", {}) or {}
            p_train = train_meta.get("mask_rate", None)
            s_train = train_meta.get("noise_sigma", None)

            entry_lin = None
            entry_mlp = None

            if p_train is not None and s_train is not None:
                key = (float(p_train), float(s_train))
                entry_lin = lin_idx.get(key, None)
                entry_mlp = mlp_idx.get(key, None)

            if entry_mlp is None and mlp_entries:
                entry_mlp = mlp_entries[0]
                key = (float(entry_mlp.get("mask_rate", -1.0)),
                       float(entry_mlp.get("noise_sigma", -1.0)))
                entry_lin = lin_idx.get(key, None)

            energy_cum = (
                mlp_results.get("meta", {}).get("energy_cum")
                or lin_results.get("meta", {}).get("energy_cum")
            )

            if entry_lin is not None:
                title = (
                    f"Multiscale summary (linear, p={entry_lin['mask_rate']}, "
                    f"σ={entry_lin['noise_sigma']})"
                )
                fig_ms_lin, _ = plot_multiscale_summary(
                    entry_lin,
                    energy_cum=energy_cum,
                    title_prefix=title,
                    model_label="linear",
                )

            if entry_mlp is not None:
                title = (
                    f"Multiscale summary (mlp, p={entry_mlp['mask_rate']}, "
                    f"σ={entry_mlp['noise_sigma']})"
                )
                fig_ms_mlp, _ = plot_multiscale_summary(
                    entry_mlp,
                    energy_cum=energy_cum,
                    title_prefix=title,
                    model_label="mlp",
                )

    # 兼容旧字段：默认把 mlp 的那张作为 fig_multiscale_example
    result["fig_multiscale_example_linear"] = fig_ms_lin
    result["fig_multiscale_example_mlp"] = fig_ms_mlp
    result["fig_multiscale_example"] = fig_ms_mlp

    result["configs"] = {
        "data_cfg": data_cfg,
        "pod_cfg": pod_cfg,
        "eval_cfg": eval_cfg,
        "train_cfg": train_cfg,
    }

    return result

def extract_and_save_figures(
    eval_results: Dict[str, Any],
    figs: Dict[str, Any],
    exp_dir: Path,
    *,
    verbose: bool = True,
    redraw_examples_from_npz: bool = True,
    fourier_suffix: str = "_fourier",
) -> Dict[str, Any]:
    """
    从 eval 结果与 Figure bundle 中“抽出”需要保存的图像/原始数据并写入 exp_dir。

    最终版设计（收敛口径）：
    - 汇总/曲线/heatmap/定义图：仍从 figs 保存 PNG
    - example 级图（四联图 + Fourier 联图）：不从 figs 保存；统一保存 npz 后调用
      redraw_all_example_figures_from_npz(exp_dir) 覆盖式重绘生成 PNG

    返回:
      fig_paths: {key: Path | List[Path] | Dict[...] | ...}
    """
    ensure_dir(exp_dir)

    fig_paths: Dict[str, Any] = {}

    lin_res = eval_results.get("linear", None)
    mlp_res = eval_results.get("mlp", None)

    lin_mask_map = (lin_res or {}).get("mask_hw_map", {}) or {}
    mlp_mask_map = (mlp_res or {}).get("mask_hw_map", {}) or {}

    # -----------------------------
    # helpers
    # -----------------------------
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    def _cfg_name(p_val: float, s_val: float) -> str:
        return f"p{p_val:.4f}_sigma{s_val:.3g}".replace(".", "-")

    def _group_examples(ex_list: list[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
        out: Dict[str, list[Dict[str, Any]]] = {}
        for ex in ex_list:
            try:
                p_val = float(ex["mask_rate"])
                s_val = float(ex["noise_sigma"])
            except Exception:
                continue
            cfg = _cfg_name(p_val, s_val)
            out.setdefault(cfg, []).append(ex)

        # 同一 cfg 内按 frame_idx 排序，确保 png/npz 对齐
        def _frame_key(ex: Dict[str, Any]) -> int:
            try:
                return int(ex.get("frame_idx", 0))
            except Exception:
                return 0

        for cfg, lst in out.items():
            lst.sort(key=_frame_key)
        return out

    def _infer_mask_hw_from_map(ex: Dict[str, Any], mask_map: Dict[str, Any]) -> Any | None:
        try:
            p_val = float(ex.get("mask_rate", 0.0))
            mask_key = f"{p_val:.6g}"
            if mask_key in mask_map:
                return np.asarray(mask_map[mask_key], dtype=bool)
        except Exception:
            pass
        return None

    def _get_fourier_meta_from_example_or_results(ex: Dict[str, Any], *, meta_fallback: Dict[str, Any]) -> tuple[list[float] | None, list[str] | None, float, float]:
        """
        返回 (k_edges, band_names, dx, dy)
        优先级：
          1) ex["fourier_curve"]（最“就地”）
          2) meta_fallback["fourier_*"]
          3) dx/dy 默认 1.0
        """
        k_edges: list[float] | None = None
        band_names: list[str] | None = None

        fc = ex.get("fourier_curve", None)
        if isinstance(fc, dict):
            ke = fc.get("k_edges", None)
            bn = fc.get("band_names", None)
            if isinstance(ke, (list, tuple)) and len(ke) > 0:
                k_edges = [float(x) for x in ke]
            if isinstance(bn, (list, tuple)) and len(bn) > 0:
                band_names = [str(x) for x in bn]

        if k_edges is None:
            ke = meta_fallback.get("fourier_k_edges", None)
            if isinstance(ke, (list, tuple)) and len(ke) > 0:
                k_edges = [float(x) for x in ke]

        if band_names is None:
            bn = meta_fallback.get("fourier_band_names", None)
            if isinstance(bn, (list, tuple)) and len(bn) > 0:
                band_names = [str(x) for x in bn]

        grid_meta = meta_fallback.get("fourier_grid_meta", None) or {}
        if isinstance(grid_meta, dict):
            dx = float(grid_meta.get("dx", 1.0))
            dy = float(grid_meta.get("dy", 1.0))
        else:
            dx, dy = 1.0, 1.0

        return k_edges, band_names, dx, dy

    # -----------------------------
    # 1) 全局 NMSE 曲线（PNG）
    # -----------------------------
    for key, fname, legacy_key in [
        ("fig_nmse_vs_mask_linear", "nmse_vs_mask_linear.png", "fig_nmse_vs_mask"),
        ("fig_nmse_vs_mask_mlp", "nmse_vs_mask_mlp.png", "fig_nmse_vs_mask"),
        ("fig_nmse_vs_noise_linear", "nmse_vs_noise_linear.png", "fig_nmse_vs_noise"),
        ("fig_nmse_vs_noise_mlp", "nmse_vs_noise_mlp.png", "fig_nmse_vs_noise"),
    ]:
        fig_ = figs.get(key, None)
        if fig_ is not None:
            p = exp_dir / fname
            fig_.savefig(p, dpi=300, bbox_inches="tight")
            fig_paths[key] = p
            fig_paths.setdefault(legacy_key, [])
            fig_paths[legacy_key].append(p)
            _log(f"[yaml-extract] Saved figure: {p}")

    # -----------------------------
    # 2) per-(p,σ) example：只保存 NPZ（不保存 PNG）
    #    PNG 统一由 redraw_all_example_figures_from_npz 从 NPZ 重绘
    # -----------------------------
    lin_examples = (lin_res or {}).get("examples", []) or []
    mlp_examples = (mlp_res or {}).get("examples", []) or []

    lin_groups = _group_examples(lin_examples)
    mlp_groups = _group_examples(mlp_examples)

    all_cfg_names = sorted(set(lin_groups.keys()) | set(mlp_groups.keys()))

    # 用于 fallback 的 meta（优先 mlp，再 linear）
    meta_fallback = ((mlp_res or {}).get("meta", {}) or (lin_res or {}).get("meta", {}) or {})

    for cfg_name in all_cfg_names:
        cfg_dir = exp_dir / cfg_name
        ensure_dir(cfg_dir)

        # --- linear NPZ ---
        for idx, ex in enumerate(lin_groups.get(cfg_name, []) or []):
            mask_hw = _infer_mask_hw_from_map(ex, lin_mask_map)

            # Fourier meta 写入每个 NPZ（关键：锁定重绘口径）
            k_edges, band_names, dx, dy = _get_fourier_meta_from_example_or_results(ex, meta_fallback=meta_fallback)

            p_npz = cfg_dir / f"linear_example_{idx:02d}.npz"
            kwargs = dict(
                x_true=np.asarray(ex["x_true"]),
                x_hat=np.asarray(ex["x_lin"]),
                x_interp=np.asarray(ex["x_interp"]),
                mask_rate=float(ex.get("mask_rate", 0.0)),
                noise_sigma=float(ex.get("noise_sigma", 0.0)),
                frame_idx=int(ex.get("frame_idx", -1)),
                model_type="linear",
                fourier_dx=float(dx),
                fourier_dy=float(dy),
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            if k_edges is not None:
                kwargs["fourier_k_edges"] = np.asarray(k_edges, dtype=float)
            if band_names is not None:
                kwargs["fourier_band_names"] = np.asarray([str(x) for x in band_names], dtype="U32")

            np.savez_compressed(p_npz, **kwargs)
            key = f"examples_linear_npz/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_npz)
            _log(f"[yaml-extract] Saved linear NPZ: {p_npz}")

        # --- mlp NPZ ---
        for idx, ex in enumerate(mlp_groups.get(cfg_name, []) or []):
            mask_hw = _infer_mask_hw_from_map(ex, mlp_mask_map)

            # Fourier meta 写入每个 NPZ（关键：锁定重绘口径）
            k_edges, band_names, dx, dy = _get_fourier_meta_from_example_or_results(ex, meta_fallback=meta_fallback)

            p_npz = cfg_dir / f"mlp_example_{idx:02d}.npz"
            kwargs = dict(
                x_true=np.asarray(ex["x_true"]),
                x_hat=np.asarray(ex["x_mlp"]),
                x_interp=np.asarray(ex["x_interp"]),
                mask_rate=float(ex.get("mask_rate", 0.0)),
                noise_sigma=float(ex.get("noise_sigma", 0.0)),
                frame_idx=int(ex.get("frame_idx", -1)),
                model_type="mlp",
                fourier_dx=float(dx),
                fourier_dy=float(dy),
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            if k_edges is not None:
                kwargs["fourier_k_edges"] = np.asarray(k_edges, dtype=float)
            if band_names is not None:
                kwargs["fourier_band_names"] = np.asarray([str(x) for x in band_names], dtype="U32")

            np.savez_compressed(p_npz, **kwargs)
            key = f"examples_mlp_npz/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_npz)
            _log(f"[yaml-extract] Saved mlp NPZ: {p_npz}")

    # -----------------------------
    # 3) per-(p,σ) 多尺度 summary 图 (PNG) —— 保持原逻辑
    # -----------------------------
    fig_multi_lin = figs.get("fig_multiscale_summary_linear") or {}
    fig_multi_mlp = figs.get("fig_multiscale_summary_mlp") or {}

    if fig_multi_lin or fig_multi_mlp:
        # 以 figs 内已有 cfg_name 为准（更稳）
        cfg_names = sorted(set(fig_multi_lin.keys()) | set(fig_multi_mlp.keys()))
        for cfg_name in cfg_names:
            cfg_dir = exp_dir / cfg_name
            ensure_dir(cfg_dir)

            fig_lin = fig_multi_lin.get(cfg_name, None)
            if fig_lin is not None:
                out = cfg_dir / "multiscale_summary_linear.png"
                fig_lin.savefig(out, dpi=300, bbox_inches="tight")
                fig_paths[f"multiscale_summary_linear/{cfg_name}"] = out
                _log(f"[yaml-extract] Saved figure: {out}")

            fig_mlp = fig_multi_mlp.get(cfg_name, None)
            if fig_mlp is not None:
                out = cfg_dir / "multiscale_summary_mlp.png"
                fig_mlp.savefig(out, dpi=300, bbox_inches="tight")
                fig_paths[f"multiscale_summary_mlp/{cfg_name}"] = out
                _log(f"[yaml-extract] Saved figure: {out}")

    # -----------------------------
    # 4) Fourier：全局曲线、heatmap、定义图、per-cfg k* 曲线（PNG）
    # -----------------------------
    # 4.1 fourier band vs mask/noise 曲线（key 与 build_eval_figures 返回保持一致）
    for k, fname in [
        ("fig_fourier_band_vs_mask_linear", "fourier_band_vs_mask_linear.png"),
        ("fig_fourier_band_vs_mask_mlp", "fourier_band_vs_mask_mlp.png"),
        ("fig_fourier_band_vs_noise_linear", "fourier_band_vs_noise_linear.png"),
        ("fig_fourier_band_vs_noise_mlp", "fourier_band_vs_noise_mlp.png"),
    ]:
        fig_ = figs.get(k, None)
        if fig_ is not None:
            p = exp_dir / fname
            fig_.savefig(p, dpi=300, bbox_inches="tight")
            fig_paths[k] = p
            _log(f"[yaml-extract] Saved figure: {p}")

    # 4.2 k* heatmap
    for k, fname in [
        ("fig_kstar_linear", "kstar_heatmap_linear.png"),
        ("fig_kstar_mlp", "kstar_heatmap_mlp.png"),
    ]:
        fig_ = figs.get(k, None)
        if fig_ is not None:
            p = exp_dir / fname
            fig_.savefig(p, dpi=300, bbox_inches="tight")
            fig_paths[k] = p
            _log(f"[yaml-extract] Saved figure: {p}")

    # 4.3 Fourier 分带定义图：E(k) + edges
    fig_energy = figs.get("fig_fourier_energy_spectrum", None)
    if fig_energy is not None:
        out = exp_dir / "fourier_energy_spectrum_definition.png"
        fig_energy.savefig(out, dpi=300, bbox_inches="tight")
        fig_paths["fig_fourier_energy_spectrum"] = out
        _log(f"[yaml-extract] Saved figure: {out}")

    # 4.4 per-cfg：k* 曲线解释图（PNG + 原始曲线NPZ）
    curves_lin = figs.get("fig_fourier_kstar_curves_linear") or {}
    curves_mlp = figs.get("fig_fourier_kstar_curves_mlp") or {}

    # --- build entry index (p,s) -> entry, to fetch raw curve data ---
    def _index_entries(results: Dict[str, Any] | None) -> Dict[tuple[float, float], Dict[str, Any]]:
        idx: Dict[tuple[float, float], Dict[str, Any]] = {}
        if results is None:
            return idx
        for e in (results.get("entries", []) or []):
            try:
                p = float(e.get("mask_rate"))
                s = float(e.get("noise_sigma"))
            except Exception:
                continue
            idx[(p, s)] = e
        return idx

    def _parse_cfg_name(cfg: str) -> tuple[float | None, float | None]:
        # cfg like: "p0-0004_sigma0" or "p0-0004_sigma0-01"
        try:
            p_part, s_part = cfg.split("_sigma")
            p = float(p_part[1:].replace("-", "."))
            s = float(s_part.replace("-", "."))
            return p, s
        except Exception:
            return None, None

    lin_entry_idx = _index_entries(lin_res)
    mlp_entry_idx = _index_entries(mlp_res)

    # meta fallback (for grid_meta if not embedded in curve)
    meta = ((mlp_res or {}).get("meta", {}) or (lin_res or {}).get("meta", {}) or {})
    grid_meta_fallback = meta.get("fourier_grid_meta", None) or {}

    def _save_kstar_curve_npz(
        *,
        out_path: Path,
        entry: Dict[str, Any],
        model: str,
        cfg_name: str,
    ) -> bool:
        """
        Save raw curve data needed to redraw k* curve / ℓ curve.
        Returns True if saved, False if no curve data.
        """
        fc = entry.get("fourier_curve", None)
        if not isinstance(fc, dict):
            return False

        # Required for curve plot
        k_centers = fc.get("k_centers", None)
        nrmse_k = fc.get("nrmse_k", None)
        if not isinstance(k_centers, (list, tuple)) or not isinstance(nrmse_k, (list, tuple)):
            return False

        # Optional fields
        e_true = fc.get("E_true_k", None)
        e_err = fc.get("E_err_k", None)
        k_edges = fc.get("k_edges", None)
        band_names = fc.get("band_names", None)

        k_star = fc.get("k_star", entry.get("k_star", None))
        k_thr = fc.get("k_star_threshold", None)

        # grid_meta: prefer embedded (if you add it later), else fallback
        gm = fc.get("grid_meta", None)
        if not isinstance(gm, dict):
            gm = grid_meta_fallback if isinstance(grid_meta_fallback, dict) else {}

        np.savez_compressed(
            out_path,
            model_type=str(model),
            cfg_name=str(cfg_name),
            mask_rate=float(entry.get("mask_rate", float("nan"))),
            noise_sigma=float(entry.get("noise_sigma", float("nan"))),
            k_centers=np.asarray(k_centers, dtype=float),
            nrmse_k=np.asarray(nrmse_k, dtype=float),
            E_true_k=np.asarray(e_true, dtype=float) if isinstance(e_true, (list, tuple)) else np.asarray([], dtype=float),
            E_err_k=np.asarray(e_err, dtype=float) if isinstance(e_err, (list, tuple)) else np.asarray([], dtype=float),
            k_edges=np.asarray(k_edges, dtype=float) if isinstance(k_edges, (list, tuple)) else np.asarray([], dtype=float),
            band_names=np.asarray([str(x) for x in band_names], dtype="U32") if isinstance(band_names, (list, tuple)) else np.asarray([], dtype="U32"),
            k_star=np.nan if k_star is None else float(k_star),
            k_star_threshold=np.nan if k_thr is None else float(k_thr),
            grid_meta_json=json.dumps(gm, ensure_ascii=False),
        )
        return True

    cfg_names = sorted(set(curves_lin.keys()) | set(curves_mlp.keys()))
    for cfg_name in cfg_names:
        cfg_dir = exp_dir / cfg_name
        ensure_dir(cfg_dir)

        p_val, s_val = _parse_cfg_name(cfg_name)
        # 如果 parse 失败，就只保存 PNG，不保存 NPZ
        key_lin = (p_val, s_val) if (p_val is not None and s_val is not None) else None

        # ---------- linear ----------
        fig = curves_lin.get(cfg_name, None)
        if fig is not None:
            out_png = cfg_dir / "fourier_kstar_curve_linear.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig_paths[f"fourier_kstar_curve_linear/{cfg_name}"] = out_png
            _log(f"[yaml-extract] Saved figure: {out_png}")

        if key_lin is not None:
            e_lin = lin_entry_idx.get(key_lin, None)
            if e_lin is not None:
                out_npz = cfg_dir / "fourier_kstar_curve_linear.npz"
                if _save_kstar_curve_npz(out_path=out_npz, entry=e_lin, model="linear", cfg_name=cfg_name):
                    fig_paths[f"fourier_kstar_curve_linear_npz/{cfg_name}"] = out_npz
                    _log(f"[yaml-extract] Saved k* curve NPZ: {out_npz}")

        # ---------- mlp ----------
        fig = curves_mlp.get(cfg_name, None)
        if fig is not None:
            out_png = cfg_dir / "fourier_kstar_curve_mlp.png"
            fig.savefig(out_png, dpi=300, bbox_inches="tight")
            fig_paths[f"fourier_kstar_curve_mlp/{cfg_name}"] = out_png
            _log(f"[yaml-extract] Saved figure: {out_png}")

        if key_lin is not None:
            e_mlp = mlp_entry_idx.get(key_lin, None)
            if e_mlp is not None:
                out_npz = cfg_dir / "fourier_kstar_curve_mlp.npz"
                if _save_kstar_curve_npz(out_path=out_npz, entry=e_mlp, model="mlp", cfg_name=cfg_name):
                    fig_paths[f"fourier_kstar_curve_mlp_npz/{cfg_name}"] = out_npz
                    _log(f"[yaml-extract] Saved k* curve NPZ: {out_npz}")

    # -----------------------------
    # 5) 保存 Fourier 元信息（fourier_meta.json）—— 保留
    # -----------------------------
    meta = ((mlp_res or {}).get("meta", {}) or (lin_res or {}).get("meta", {}) or {})
    fourier_meta = {
        "fourier_enabled": meta.get("fourier_enabled", None),
        "fourier_band_scheme": meta.get("fourier_band_scheme", None),
        "fourier_band_names": meta.get("fourier_band_names", None),
        "fourier_lambda_edges": meta.get("fourier_lambda_edges", None),
        "fourier_k_edges": meta.get("fourier_k_edges", None),  # interior edges
        "fourier_k_centers": meta.get("fourier_k_centers", None),
        "fourier_grid_meta": meta.get("fourier_grid_meta", None),
    }
    if any(v is not None for v in fourier_meta.values()):
        out_json = exp_dir / "fourier_meta.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(fourier_meta, f, indent=2, ensure_ascii=False)
        fig_paths["fourier_meta_json"] = out_json
        _log(f"[yaml-extract] Saved Fourier meta: {out_json}")

    # -----------------------------
    # 6) 统一从 NPZ 重绘 example PNG（四联图 + Fourier 联图）
    # -----------------------------
    if redraw_examples_from_npz:
        try:
            redraw_info = redraw_all_example_figures_from_npz(
                exp_dir,
                recursive=True,
                redraw_fourier=True,
                fourier_meta_name="fourier_meta.json",
                fourier_suffix=fourier_suffix,
                verbose=verbose,
            )
            fig_paths["redraw_from_npz"] = redraw_info
        except Exception as e:
            _log(f"[yaml-extract] WARNING: redraw_all_example_figures_from_npz failed: {e}")

    return fig_paths

def run_experiment_from_yaml(
    yaml_path: str | Path,
    *,
    experiment_name: str | None = None,
    save_root: str | Path = "artifacts/experiments",
    generate_report: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    从 YAML 配置运行一轮完整实验，并将结果与图像全部落盘。

    v1.08 结构：
    1) compute_full_eval_results -> eval_results (只含 linear/mlp/df_*)
    2) build_eval_figures        -> figs (只含 Figure 对象)
    3) extract_and_save_figures  -> fig_paths (图像路径)
    4) save_full_experiment_results(eval_results, ...) -> numeric_paths (JSON/CSV)
    5) 将 numeric_paths + fig_paths 合并写入 all_result['saved_paths']
       供 generate_experiment_report_md 使用。
    """
    yaml_path = Path(yaml_path)
    if experiment_name is None:
        experiment_name = yaml_path.stem

    if verbose:
        print(f"[yaml-experiment] Loading configs from {yaml_path} ...")

    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    if verbose:
        print("[yaml-experiment] Running full evaluation (compute only) ...")

    # 1) 只做数值实验
    eval_results = compute_full_eval_results(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        verbose=verbose,
    )

    save_root = Path(save_root)
    exp_dir = save_root / experiment_name
    ensure_dir(exp_dir)

    # 2) 画出所有需要的 Figure（仅在内存中）
    if verbose:
        print("[yaml-experiment] Building figures from eval results ...")
    figs = build_eval_figures(
        linear_results=eval_results["linear"],
        mlp_results=eval_results.get("mlp"),
        df_linear=eval_results.get("df_linear"),
        df_mlp=eval_results.get("df_mlp"),
        verbose=verbose,
    )

    # 3) 保存所有图像到磁盘
    if verbose:
        print(f"[yaml-experiment] Saving figures under {exp_dir} ...")
    fig_paths = extract_and_save_figures(
        eval_results=eval_results,
        figs=figs,
        exp_dir=exp_dir,
        verbose=verbose,
    )

    # 4) 保存数值结果（JSON / CSV 等）——不再把 Figure 塞进保存用 dict
    if verbose:
        print("[yaml-experiment] Saving numeric results (JSON/CSV) ...")
    from .eval.reports import save_full_experiment_results  # 避免循环导入顶部

    numeric_paths = save_full_experiment_results(
        eval_results,
        base_dir=save_root,
        experiment_name=experiment_name,
    )

    saved_paths = {}
    saved_paths.update(numeric_paths)
    saved_paths.update(fig_paths)

    # 5) 组装 all_result：给后续 report.md 使用
    all_result: Dict[str, Any] = {
        "linear": eval_results["linear"],
        "mlp": eval_results.get("mlp"),
        "df_linear": eval_results.get("df_linear"),
        "df_mlp": eval_results.get("df_mlp"),
        "saved_paths": saved_paths,
        "exp_dir": exp_dir,
        "yaml_path": yaml_path,
    }

    # 6) （可选）生成 report.md
    report_path = None
    if generate_report:
        if verbose:
            print("[yaml-experiment] Generating report.md ...")
        from .eval.reports import generate_experiment_report_md

        report_path = generate_experiment_report_md(
            all_result,
            out_path=exp_dir / "report.md",
            experiment_name=experiment_name,
            config_yaml=yaml_path,
        )
    all_result["report_path"] = report_path

    if verbose:
        print(f"[yaml-experiment] Done. Results saved under {exp_dir}")

    return all_result

def redraw_all_example_figures_from_npz(
    exp_root: str | Path,
    *,
    recursive: bool = True,
    redraw_fourier: bool = True,
    fourier_meta_name: str = "fourier_meta.json",
    fourier_suffix: str = "_fourier",
    verbose: bool = True,
) -> Dict[str, list[Path]]:
    """
    遍历实验目录下的 example npz（至少包含 x_true/x_hat/x_interp），
    覆盖式重绘：
      - 四联图：<npz_stem>.png
      - Fourier 联图（可选）：<npz_stem>{fourier_suffix}.png

    新设计：每个 npz 只 load 一次，然后分别调用：
      - viz.field_plots.plot_example_from_npz_data(data)
      - viz.fourier_plots.plot_fourier_example_from_npz_data(data)
    Fourier 元信息优先从 npz 内的 fourier_* 字段读取；
    若缺失，则回退读取 exp_root/fourier_meta.json（用于兼容老结果）。
    """
    from pathlib import Path
    import json
    import numpy as np

    from backend.viz.field_plots import plot_example_from_npz_data
    from backend.viz.fourier_plots import plot_fourier_example_from_npz_data
    from backend.dataio.io_utils import ensure_dir

    exp_root = Path(exp_root)
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")

    # --------- load Fourier meta once (fallback for legacy npz) ----------
    fallback_band_names = None
    fallback_k_edges = None
    fallback_dx = None
    fallback_dy = None

    if redraw_fourier:
        meta_path = exp_root / fourier_meta_name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                fallback_band_names = meta.get("fourier_band_names", None)
                fallback_k_edges = meta.get("fourier_k_edges", None)
                grid_meta = meta.get("fourier_grid_meta", None) or {}
                if isinstance(grid_meta, dict):
                    fallback_dx = grid_meta.get("dx", None)
                    fallback_dy = grid_meta.get("dy", None)

                # 没有 k_edges 就别 redraw_fourier
                if not isinstance(fallback_k_edges, (list, tuple)) or len(fallback_k_edges) == 0:
                    if verbose:
                        print(f"[redraw] WARNING: {meta_path} has no valid fourier_k_edges; skip Fourier redraw.")
                    redraw_fourier = False
            except Exception as e:
                if verbose:
                    print(f"[redraw] WARNING: failed to read {meta_path}: {e}")
                redraw_fourier = False
        else:
            # 不强制要求 meta.json：如果 npz 自带 fourier_* 也能画
            pass

    # --------- collect npz ----------
    npz_paths = sorted(exp_root.rglob("*.npz")) if recursive else sorted(exp_root.glob("*.npz"))

    if verbose:
        print(f"[redraw] Scanning {exp_root} ... found {len(npz_paths)} npz files")

    updated: list[Path] = []
    skipped: list[Path] = []
    failed: list[Path] = []
    updated_fourier: list[Path] = []
    failed_fourier: list[Path] = []

    for npz_path in npz_paths:
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                files = set(data.files)

                # 只处理 example npz
                if not {"x_true", "x_hat", "x_interp"}.issubset(files):
                    skipped.append(npz_path)
                    continue

                # --------- 1) redraw quadruple ----------
                try:
                    fig = plot_example_from_npz_data(data)
                    png_path = npz_path.with_suffix(".png")
                    ensure_dir(png_path.parent)
                    fig.savefig(png_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    updated.append(png_path)
                    if verbose:
                        print(f"[redraw] Updated: {png_path}")
                except Exception as e:
                    failed.append(npz_path)
                    if verbose:
                        print(f"[redraw] FAILED quadruple: {npz_path} ({e})")

                # --------- 2) redraw Fourier (optional) ----------
                if redraw_fourier:
                    try:
                        # 优先用 npz 内的 fourier_*；缺失则用 fallback（兼容老 npz）
                        k_edges = None
                        band_names = None
                        dx = None
                        dy = None

                        if "fourier_k_edges" in files:
                            k_edges = np.asarray(data["fourier_k_edges"], dtype=float).tolist()
                        else:
                            k_edges = [float(x) for x in fallback_k_edges] if fallback_k_edges is not None else None

                        if "fourier_band_names" in files:
                            band_names = [str(x) for x in np.asarray(data["fourier_band_names"]).tolist()]
                        else:
                            band_names = fallback_band_names

                        if "fourier_dx" in files:
                            dx = float(data["fourier_dx"])
                        else:
                            dx = float(fallback_dx) if fallback_dx is not None else None

                        if "fourier_dy" in files:
                            dy = float(data["fourier_dy"])
                        else:
                            dy = float(fallback_dy) if fallback_dy is not None else None

                        fig_f = plot_fourier_example_from_npz_data(
                            data,
                            k_edges=k_edges,
                            band_names=band_names,
                            dx=dx,
                            dy=dy,
                        )
                        if fig_f is not None:
                            out_f = npz_path.with_suffix("").with_name(npz_path.stem + f"{fourier_suffix}.png")
                            ensure_dir(out_f.parent)
                            fig_f.savefig(out_f, dpi=300, bbox_inches="tight")
                            plt.close(fig_f)
                            updated_fourier.append(out_f)
                            if verbose:
                                print(f"[redraw] Updated Fourier: {out_f}")
                    except Exception as e:
                        failed_fourier.append(npz_path)
                        if verbose:
                            print(f"[redraw] FAILED Fourier: {npz_path} ({e})")

        except Exception as e:
            failed.append(npz_path)
            if verbose:
                print(f"[redraw] FAILED to load npz: {npz_path}  ({e})")
            continue

    return {
        "updated_png": updated,
        "updated_fourier_png": updated_fourier,
        "skipped_npz": skipped,
        "failed_npz": failed,
        "failed_fourier_npz": failed_fourier,
    }
