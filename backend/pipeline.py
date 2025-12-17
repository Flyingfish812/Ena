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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    只做“画图”：基于 eval 结果构建所有用于论文/报告的 Figure。
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

    def _ensure_df(results: Dict[str, Any], df: Any | None) -> Any:
        return df if df is not None else results_to_dataframe(results)

    def _get_examples(results: Dict[str, Any] | None) -> list[Dict[str, Any]]:
        if results is None:
            return []
        return results.get("examples", []) or []

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

    def _first_example_by_cfg(examples: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        m: Dict[str, Dict[str, Any]] = {}
        for ex in examples or []:
            try:
                p = float(ex["mask_rate"])
                s = float(ex["noise_sigma"])
            except Exception:
                continue
            cfg = _cfg_name(p, s)
            if cfg not in m:
                m[cfg] = ex
        return m

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
            title = f"{title_prefix} (frame={ex['frame_idx']}, p={p:.3g}, σ={s:.3g})"
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

    def _band_names_from_df(df: Any) -> tuple[str, ...]:
        try:
            cols = [c for c in df.columns if str(c).startswith("fourier_band_nrmse_")]
            names = tuple(str(c).replace("fourier_band_nrmse_", "") for c in cols)
            return names if len(names) > 0 else ("L", "M", "H")
        except Exception:
            return ("L", "M", "H")

    # -----------------------------
    # 1) DataFrame 准备
    # -----------------------------
    df_linear = _ensure_df(linear_results, df_linear)
    df_mlp = _ensure_df(mlp_results, df_mlp) if mlp_results is not None else None

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
    # 3) 单个代表性 example 四联图
    # -----------------------------
    fig_example_linear = None
    ex_lin = linear_results.get("example_recon", None)
    if ex_lin is not None:
        fig_example_linear = _plot_quadruple_from_ex(ex_lin, model_key="x_lin", title_prefix="Linear example")

    fig_example_mlp = None
    if mlp_results is not None:
        ex_mlp = mlp_results.get("example_recon", None)
        if ex_mlp is not None:
            fig_example_mlp = _plot_quadruple_from_ex(ex_mlp, model_key="x_mlp", title_prefix="MLP example")

    # -----------------------------
    # 4) per-(p,σ) 四联图（按 cfg 分组）
    # -----------------------------
    lin_examples = _get_examples(linear_results)
    mlp_examples = _get_examples(mlp_results)

    fig_examples_linear: Dict[str, list[Any]] = {}
    fig_examples_mlp: Dict[str, list[Any]] = {}

    if lin_examples:
        _log("[build-figs] Generating per-(p,σ) linear quadruple figures via viz.field_plots ...")
        for ex in lin_examples:
            try:
                p = float(ex["mask_rate"]); s = float(ex["noise_sigma"])
                cfg = _cfg_name(p, s)
            except Exception:
                continue
            fig = _plot_quadruple_from_ex(ex, model_key="x_lin", title_prefix="Linear")
            if fig is not None:
                fig_examples_linear.setdefault(cfg, []).append(fig)

    if mlp_examples:
        _log("[build-figs] Generating per-(p,σ) MLP quadruple figures via viz.field_plots ...")
        for ex in mlp_examples:
            try:
                p = float(ex["mask_rate"]); s = float(ex["noise_sigma"])
                cfg = _cfg_name(p, s)
            except Exception:
                continue
            fig = _plot_quadruple_from_ex(ex, model_key="x_mlp", title_prefix="MLP")
            if fig is not None:
                fig_examples_mlp.setdefault(cfg, []).append(fig)

    # -----------------------------
    # 5) Fourier 多尺度可视化（汇总 + per-cfg 解释图）
    # -----------------------------
    # 5.1 k* heatmap
    fig_kstar_linear = _safe_build(
        "k* heatmap (linear)",
        lambda: plot_kstar_heatmap(df_linear, df_mlp, model="linear", title="k* heatmap"),
    )
    fig_kstar_mlp = None
    if mlp_results is not None:
        fig_kstar_mlp = _safe_build(
            "k* heatmap (mlp)",
            lambda: plot_kstar_heatmap(df_linear, df_mlp, model="mlp", title="k* heatmap"),
        )

    # 5.2 Fourier band NRMSE curves（L/M/H 随 p/σ 的曲线）
    band_names = _band_names_from_df(df_linear)
    fourier_curve_figs = _safe_build(
        "Fourier band curves",
        lambda: plot_fourier_band_nrmse_curves(df_linear, df_mlp, band_names=band_names),
    ) or {}

    # 5.3 per-(p,σ) Fourier 解释图（k* 曲线 + 空间分解）
    lin_entry_idx = _index_entries(linear_results)
    mlp_entry_idx = _index_entries(mlp_results)
    lin_ex_first = _first_example_by_cfg(lin_examples)
    mlp_ex_first = _first_example_by_cfg(mlp_examples)

    all_cfg_names = set(lin_ex_first.keys()) | set(mlp_ex_first.keys())
    for (p, s) in set(lin_entry_idx.keys()) | set(mlp_entry_idx.keys()):
        all_cfg_names.add(_cfg_name(p, s))

    fig_fourier_kstar_curves_linear: Dict[str, Any] = {}
    fig_fourier_kstar_curves_mlp: Dict[str, Any] = {}
    fig_fourier_decomp_linear: Dict[str, Any] = {}
    fig_fourier_decomp_mlp: Dict[str, Any] = {}

    for cfg in sorted(all_cfg_names):
        p_val, s_val = _parse_cfg_name(cfg)
        if p_val is None or s_val is None:
            continue

        # k* curve
        e_lin = lin_entry_idx.get((p_val, s_val))
        if e_lin and e_lin.get("fourier_curve") is not None:
            fig = _safe_build(
                f"k* curve (linear) {cfg}",
                lambda e=e_lin: plot_kstar_curve_from_entry(e, title_prefix="NRMSE(k) [linear]"),
            )
            if fig is not None:
                fig_fourier_kstar_curves_linear[cfg] = fig

        e_mlp = mlp_entry_idx.get((p_val, s_val))
        if e_mlp and e_mlp.get("fourier_curve") is not None:
            fig = _safe_build(
                f"k* curve (mlp) {cfg}",
                lambda e=e_mlp: plot_kstar_curve_from_entry(e, title_prefix="NRMSE(k) [mlp]"),
            )
            if fig is not None:
                fig_fourier_kstar_curves_mlp[cfg] = fig

        # spatial decomposition（需要 example + k_edges）
        ex_lin = lin_ex_first.get(cfg)
        if ex_lin is not None and e_lin is not None:
            k_edges = (e_lin.get("fourier_curve") or {}).get("k_edges")
            if k_edges is not None:
                fig = _safe_build(
                    f"Fourier decomp (linear) {cfg}",
                    lambda ex=ex_lin, ke=k_edges: plot_spatial_fourier_band_decomposition(
                        x_true_hw=np.asarray(ex["x_true"]),
                        x_pred_hw=np.asarray(ex["x_lin"]),
                        k_edges=ke,
                        band_names=("L", "M", "H"),
                        channel=0,
                        title=f"Fourier bands spatial view (linear) | {cfg}",
                    ),
                )
                if fig is not None:
                    fig_fourier_decomp_linear[cfg] = fig

        ex_mlp = mlp_ex_first.get(cfg)
        if ex_mlp is not None and e_mlp is not None:
            k_edges = (e_mlp.get("fourier_curve") or {}).get("k_edges")
            if k_edges is not None:
                fig = _safe_build(
                    f"Fourier decomp (mlp) {cfg}",
                    lambda ex=ex_mlp, ke=k_edges: plot_spatial_fourier_band_decomposition(
                        x_true_hw=np.asarray(ex["x_true"]),
                        x_pred_hw=np.asarray(ex["x_mlp"]),
                        k_edges=ke,
                        band_names=("L", "M", "H"),
                        channel=0,
                        title=f"Fourier bands spatial view (mlp) | {cfg}",
                    ),
                )
                if fig is not None:
                    fig_fourier_decomp_mlp[cfg] = fig

    # 5.4 全局 band 定义图（E(k)+edges）：依赖 meta（有则画，没有就 None）
    def _pick_meta() -> Dict[str, Any]:
        meta = (mlp_results.get("meta", {}) if mlp_results is not None else {}) or {}
        if not meta:
            meta = (linear_results.get("meta", {}) or {})
        return meta

    fig_energy_spectrum = _safe_build(
        "Energy spectrum definition figure",
        lambda: (
            plot_energy_spectrum_with_band_edges(
                k_centers=np.asarray(_pick_meta().get("fourier_k_centers")),
                energy_k=np.asarray(_pick_meta().get("fourier_energy_k")),
                k_edges=_pick_meta().get("fourier_k_edges"),
                band_names=("L", "M", "H"),
                title="Energy spectrum E(k) with band edges (definition of L/M/H)",
            )
            if (
                _pick_meta().get("fourier_k_centers") is not None
                and _pick_meta().get("fourier_energy_k") is not None
                and _pick_meta().get("fourier_k_edges") is not None
            )
            else None
        ),
    )

    # -----------------------------
    # 6) assemble return
    # -----------------------------
    return {
        "fig_nmse_vs_mask_linear": fig_mask_linear,
        "fig_nmse_vs_mask_mlp": fig_mask_mlp,
        "fig_nmse_vs_noise_linear": fig_noise_linear,
        "fig_nmse_vs_noise_mlp": fig_noise_mlp,
        "fig_example_linear": fig_example_linear,
        "fig_example_mlp": fig_example_mlp,
        "fig_examples_linear": fig_examples_linear or None,
        "fig_examples_mlp": fig_examples_mlp or None,
        "fig_kstar_linear": fig_kstar_linear,
        "fig_kstar_mlp": fig_kstar_mlp,
        "fig_fourier_kstar_curves_linear": fig_fourier_kstar_curves_linear or None,
        "fig_fourier_kstar_curves_mlp": fig_fourier_kstar_curves_mlp or None,
        "fig_fourier_decomp_linear": fig_fourier_decomp_linear or None,
        "fig_fourier_decomp_mlp": fig_fourier_decomp_mlp or None,
        "fig_fourier_energy_spectrum": fig_energy_spectrum,
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
) -> Dict[str, Any]:
    """
    从 eval 结果与 Figure bundle 中“抽出”需要保存的图像/原始数据并写入 exp_dir。

    - 负责保存 4 张全局 NMSE 曲线图 (PNG)
    - 负责保存代表性 example 四联图 (PNG + npz)
    - 负责保存 per-(p,σ) 四联图 (PNG + npz)
    - 负责保存 per-(p,σ) 多尺度 summary 图 (PNG)

    返回:
        fig_paths: { key(str) -> Path 或 List[Path] }
        其中 key 的命名保持与旧版 run_experiment_from_yaml 大体兼容,
        新增:
        - "example_linear_npz" / "example_mlp_npz"
        - "examples_linear_npz/{cfg_name}"
        - "examples_mlp_npz/{cfg_name}"
    """
    ensure_dir(exp_dir)
    fig_paths: Dict[str, Any] = {}

    # 方便后面取 examples / mask
    lin_res = eval_results.get("linear", None)
    mlp_res = eval_results.get("mlp", None)

    lin_examples_all = (lin_res or {}).get("examples", []) or []
    mlp_examples_all = (mlp_res or {}).get("examples", []) or []

    lin_mask_map = (lin_res or {}).get("mask_hw_map", {}) or {}
    mlp_mask_map = (mlp_res or {}).get("mask_hw_map", {}) or {}

    def _build_ex_index(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """按 (p,σ) -> cfg_name 分组 examples."""
        idx: dict[str, list[dict[str, Any]]] = {}
        for ex in examples:
            try:
                p = float(ex["mask_rate"])
                s = float(ex["noise_sigma"])
            except Exception:
                continue
            cfg_name = f"p{p:.4f}_sigma{s:.3g}".replace(".", "-")
            idx.setdefault(cfg_name, []).append(ex)
        return idx

    lin_ex_index = _build_ex_index(lin_examples_all)
    mlp_ex_index = _build_ex_index(mlp_examples_all)

    # === 1) 4 张全局 NMSE 曲线图 ===
    fig_mask_linear = figs.get("fig_nmse_vs_mask_linear", None)
    if fig_mask_linear is not None:
        p = exp_dir / "nmse_vs_mask_linear.png"
        fig_mask_linear.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_mask_linear"] = p
        # 给 report.md 也塞一个通用键（用 linear 版）
        fig_paths.setdefault("fig_nmse_vs_mask", p)
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    fig_mask_mlp = figs.get("fig_nmse_vs_mask_mlp", None)
    if fig_mask_mlp is not None:
        p = exp_dir / "nmse_vs_mask_mlp.png"
        fig_mask_mlp.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_mask_mlp"] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    fig_noise_linear = figs.get("fig_nmse_vs_noise_linear", None)
    if fig_noise_linear is not None:
        p = exp_dir / "nmse_vs_noise_linear.png"
        fig_noise_linear.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_noise_linear"] = p
        fig_paths.setdefault("fig_nmse_vs_noise", p)
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    fig_noise_mlp = figs.get("fig_nmse_vs_noise_mlp", None)
    if fig_noise_mlp is not None:
        p = exp_dir / "nmse_vs_noise_mlp.png"
        fig_noise_mlp.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_noise_mlp"] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    # === 2) 代表性 example 四联图（各 1 张），PNG + npz，供 report.md / debug 使用 ===
    fig_example_linear = figs.get("fig_example_linear", None)
    if fig_example_linear is not None:
        p_png = exp_dir / "example_linear.png"
        fig_example_linear.savefig(p_png, dpi=300, bbox_inches="tight")
        fig_paths["fig_example_linear"] = p_png
        if verbose:
            print(f"[yaml-extract] Saved example_linear PNG: {p_png}")

        ex_lin = (lin_res or {}).get("example_recon", None)
        if ex_lin is not None:
            # 从 meta 里恢复 mask
            mask_hw = None
            try:
                p_val = float(ex_lin["mask_rate"])
                mask_key = f"{p_val:.6g}"
                if mask_key in lin_mask_map:
                    mask_hw = np.asarray(lin_mask_map[mask_key], dtype=bool)
            except Exception:
                pass

            p_npz = exp_dir / "example_linear.npz"
            kwargs = dict(
                x_true=np.asarray(ex_lin["x_true"]),
                x_hat=np.asarray(ex_lin["x_lin"]),
                x_interp=np.asarray(ex_lin["x_interp"]),
                mask_rate=float(ex_lin.get("mask_rate", 0.0)),
                noise_sigma=float(ex_lin.get("noise_sigma", 0.0)),
                frame_idx=int(ex_lin.get("frame_idx", -1)),
                model_type="linear",
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            np.savez_compressed(p_npz, **kwargs)
            fig_paths["example_linear_npz"] = p_npz
            if verbose:
                print(f"[yaml-extract] Saved example_linear NPZ: {p_npz}")

    fig_example_mlp = figs.get("fig_example_mlp", None)
    if fig_example_mlp is not None:
        p_png = exp_dir / "example_mlp.png"
        fig_example_mlp.savefig(p_png, dpi=300, bbox_inches="tight")
        fig_paths["fig_example_mlp"] = p_png
        if verbose:
            print(f"[yaml-extract] Saved example_mlp PNG: {p_png}")

        ex_mlp = (mlp_res or {}).get("example_recon", None)
        if ex_mlp is not None:
            mask_hw = None
            try:
                p_val = float(ex_mlp["mask_rate"])
                mask_key = f"{p_val:.6g}"
                if mask_key in mlp_mask_map:
                    mask_hw = np.asarray(mlp_mask_map[mask_key], dtype=bool)
            except Exception:
                pass

            p_npz = exp_dir / "example_mlp.npz"
            kwargs = dict(
                x_true=np.asarray(ex_mlp["x_true"]),
                x_hat=np.asarray(ex_mlp["x_mlp"]),
                x_interp=np.asarray(ex_mlp["x_interp"]),
                mask_rate=float(ex_mlp.get("mask_rate", 0.0)),
                noise_sigma=float(ex_mlp.get("noise_sigma", 0.0)),
                frame_idx=int(ex_mlp.get("frame_idx", -1)),
                model_type="mlp",
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            np.savez_compressed(p_npz, **kwargs)
            fig_paths["example_mlp_npz"] = p_npz
            if verbose:
                print(f"[yaml-extract] Saved example_mlp NPZ: {p_npz}")

    # === 3) per-(p,σ) 四联图：PNG + npz ===
    fig_examples_linear = figs.get("fig_examples_linear") or {}
    fig_examples_mlp = figs.get("fig_examples_mlp") or {}

    if fig_examples_linear or fig_examples_mlp:
        if verbose:
            print("[yaml-extract] Saving per-(p,σ) quadruple figures and npz ...")

    all_cfg_names = set(fig_examples_linear.keys()) | set(fig_examples_mlp.keys()) \
        | set(lin_ex_index.keys()) | set(mlp_ex_index.keys())

    for cfg_name in sorted(all_cfg_names):
        cfg_dir = exp_dir / cfg_name
        ensure_dir(cfg_dir)

        # --- 3.1 linear: PNG ---
        lin_figs = fig_examples_linear.get(cfg_name, []) or []
        for idx, fig in enumerate(lin_figs):
            p_png = cfg_dir / f"linear_example_{idx:02d}.png"
            fig.savefig(p_png, dpi=300, bbox_inches="tight")
            key = f"fig_examples_linear/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_png)
            if verbose:
                print(f"[yaml-extract] Saved figure: {p_png}")

        # --- 3.2 linear: NPZ (原始数据) ---
        lin_ex_list = lin_ex_index.get(cfg_name, []) or []
        for idx, ex in enumerate(lin_ex_list):
            # 对应 p,σ 的 mask
            mask_hw = None
            try:
                p_val = float(ex["mask_rate"])
                mask_key = f"{p_val:.6g}"
                if mask_key in lin_mask_map:
                    mask_hw = np.asarray(lin_mask_map[mask_key], dtype=bool)
            except Exception:
                pass

            p_npz = cfg_dir / f"linear_example_{idx:02d}.npz"
            kwargs = dict(
                x_true=np.asarray(ex["x_true"]),
                x_hat=np.asarray(ex["x_lin"]),
                x_interp=np.asarray(ex["x_interp"]),
                mask_rate=float(ex.get("mask_rate", 0.0)),
                noise_sigma=float(ex.get("noise_sigma", 0.0)),
                frame_idx=int(ex.get("frame_idx", -1)),
                model_type="linear",
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            np.savez_compressed(p_npz, **kwargs)

            key = f"examples_linear_npz/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_npz)
            if verbose:
                print(f"[yaml-extract] Saved linear NPZ: {p_npz}")

        # --- 3.3 mlp: PNG ---
        mlp_figs = fig_examples_mlp.get(cfg_name, []) or []
        for idx, fig in enumerate(mlp_figs):
            p_png = cfg_dir / f"mlp_example_{idx:02d}.png"
            fig.savefig(p_png, dpi=300, bbox_inches="tight")
            key = f"fig_examples_mlp/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_png)
            if verbose:
                print(f"[yaml-extract] Saved figure: {p_png}")

        # --- 3.4 mlp: NPZ ---
        mlp_ex_list = mlp_ex_index.get(cfg_name, []) or []
        for idx, ex in enumerate(mlp_ex_list):
            mask_hw = None
            try:
                p_val = float(ex["mask_rate"])
                mask_key = f"{p_val:.6g}"
                if mask_key in mlp_mask_map:
                    mask_hw = np.asarray(mlp_mask_map[mask_key], dtype=bool)
            except Exception:
                pass

            p_npz = cfg_dir / f"mlp_example_{idx:02d}.npz"
            kwargs = dict(
                x_true=np.asarray(ex["x_true"]),
                x_hat=np.asarray(ex["x_mlp"]),
                x_interp=np.asarray(ex["x_interp"]),
                mask_rate=float(ex.get("mask_rate", 0.0)),
                noise_sigma=float(ex.get("noise_sigma", 0.0)),
                frame_idx=int(ex.get("frame_idx", -1)),
                model_type="mlp",
            )
            if mask_hw is not None:
                kwargs["mask_hw"] = mask_hw
            np.savez_compressed(p_npz, **kwargs)

            key = f"examples_mlp_npz/{cfg_name}"
            fig_paths.setdefault(key, [])
            fig_paths[key].append(p_npz)
            if verbose:
                print(f"[yaml-extract] Saved mlp NPZ: {p_npz}")

    # === 4) per-(p,σ) 多尺度 summary 图 ===
    lin_entries = lin_res.get("entries", []) if lin_res is not None else []
    mlp_entries = mlp_res.get("entries", []) if mlp_res is not None else []

    if lin_entries or mlp_entries:
        if verbose:
            print("[yaml-extract] Generating multiscale summary figures for each (p,σ) ...")

        def _build_index(entries):
            m = {}
            for e in entries:
                p = float(e.get("mask_rate", -1.0))
                s = float(e.get("noise_sigma", -1.0))
                m[(p, s)] = e
            return m

        lin_idx = _build_index(lin_entries)
        mlp_idx = _build_index(mlp_entries)

        energy_cum = (
            (mlp_res or {}).get("meta", {}).get("energy_cum")
            or (lin_res or {}).get("meta", {}).get("energy_cum")
        )

        all_ps = sorted(set(lin_idx.keys()) | set(mlp_idx.keys()))

        for (p_val, s_val) in all_ps:
            entry_lin = lin_idx.get((p_val, s_val), None)
            entry_mlp = mlp_idx.get((p_val, s_val), None)

            cfg_name = f"p{p_val:.4f}_sigma{s_val:.3g}".replace(".", "-")
            cfg_dir = exp_dir / cfg_name
            ensure_dir(cfg_dir)

            # 线性模型的多尺度图
            if entry_lin is not None:
                title_lin = f"Multiscale summary (linear, p={p_val}, σ={s_val})"
                fig_lin, _ = plot_multiscale_summary(
                    entry_lin,
                    energy_cum=energy_cum,
                    title_prefix=title_lin,
                    model_label="linear",
                )
                out_lin = cfg_dir / "multiscale_linear.png"
                fig_lin.savefig(out_lin, dpi=300, bbox_inches="tight")
                key = f"multiscale_linear/{cfg_name}"
                fig_paths.setdefault(key, [])
                fig_paths[key].append(out_lin)
                # 拿第一张 linear 多尺度图给 report.md 用
                fig_paths.setdefault("fig_multiscale_example", out_lin)
                if verbose:
                    print(f"[yaml-extract] Saved multiscale linear: {out_lin}")

            # MLP 模型的多尺度图
            if entry_mlp is not None:
                title_mlp = f"Multiscale summary (mlp, p={p_val}, σ={s_val})"
                fig_mlp, _ = plot_multiscale_summary(
                    entry_mlp,
                    energy_cum=energy_cum,
                    title_prefix=title_mlp,
                    model_label="mlp",
                )
                out_mlp = cfg_dir / "multiscale_mlp.png"
                fig_mlp.savefig(out_mlp, dpi=300, bbox_inches="tight")
                key = f"multiscale_mlp/{cfg_name}"
                fig_paths.setdefault(key, [])
                fig_paths[key].append(out_mlp)
                if verbose:
                    print(f"[yaml-extract] Saved multiscale mlp: {out_mlp}")

    # === 5) Fourier figs（Batch 4）保存 ===
    # 5.1 k* heatmap
    fig_kstar_linear = figs.get("fig_kstar_linear", None)
    if fig_kstar_linear is not None:
        p = exp_dir / "kstar_heatmap_linear.png"
        fig_kstar_linear.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_kstar_linear"] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    fig_kstar_mlp = figs.get("fig_kstar_mlp", None)
    if fig_kstar_mlp is not None:
        p = exp_dir / "kstar_heatmap_mlp.png"
        fig_kstar_mlp.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_kstar_mlp"] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    # 5.2 Fourier band curves
    k = "fig_fourier_band_vs_mask_linear"
    fig_ = figs.get(k, None)
    if fig_ is not None:
        p = exp_dir / "fourier_band_vs_mask_linear.png"
        fig_.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths[k] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    k = "fig_fourier_band_vs_mask_mlp"
    fig_ = figs.get(k, None)
    if fig_ is not None:
        p = exp_dir / "fourier_band_vs_mask_mlp.png"
        fig_.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths[k] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    k = "fig_fourier_band_vs_noise_linear"
    fig_ = figs.get(k, None)
    if fig_ is not None:
        p = exp_dir / "fourier_band_vs_noise_linear.png"
        fig_.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths[k] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")

    k = "fig_fourier_band_vs_noise_mlp"
    fig_ = figs.get(k, None)
    if fig_ is not None:
        p = exp_dir / "fourier_band_vs_noise_mlp.png"
        fig_.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths[k] = p
        if verbose:
            print(f"[yaml-extract] Saved figure: {p}")
    
    # 5.3 per-(p,σ) Fourier 解释图保存
    # 5.3.1 全局定义图：E(k) + edges
    fig_energy = figs.get("fig_fourier_energy_spectrum", None)
    if fig_energy is not None:
        out = exp_dir / "fourier_energy_spectrum_definition.png"
        fig_energy.savefig(out, dpi=300, bbox_inches="tight")
        fig_paths["fig_fourier_energy_spectrum"] = out
        if verbose:
            print(f"[yaml-extract] Saved figure: {out}")

    # 5.3.2 per-cfg：k* 曲线解释图
    curves_lin = figs.get("fig_fourier_kstar_curves_linear") or {}
    curves_mlp = figs.get("fig_fourier_kstar_curves_mlp") or {}

    # 5.3.3 per-cfg：空间域 L/M/H 分解解释图
    decomp_lin = figs.get("fig_fourier_decomp_linear") or {}
    decomp_mlp = figs.get("fig_fourier_decomp_mlp") or {}

    all_cfg = set(curves_lin.keys()) | set(curves_mlp.keys()) | set(decomp_lin.keys()) | set(decomp_mlp.keys())

    if all_cfg and verbose:
        print("[yaml-extract] Saving per-(p,σ) Fourier explanatory figures ...")

    for cfg_name in sorted(all_cfg):
        cfg_dir = exp_dir / cfg_name
        ensure_dir(cfg_dir)

        # k* curve
        fig = curves_lin.get(cfg_name, None)
        if fig is not None:
            out = cfg_dir / "fourier_kstar_curve_linear.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fourier_kstar_curve_linear/{cfg_name}", out)
            if verbose:
                print(f"[yaml-extract] Saved figure: {out}")

        fig = curves_mlp.get(cfg_name, None)
        if fig is not None:
            out = cfg_dir / "fourier_kstar_curve_mlp.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fourier_kstar_curve_mlp/{cfg_name}", out)
            if verbose:
                print(f"[yaml-extract] Saved figure: {out}")

        # spatial decomposition
        fig = decomp_lin.get(cfg_name, None)
        if fig is not None:
            out = cfg_dir / "fourier_band_decomp_linear.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fourier_band_decomp_linear/{cfg_name}", out)
            if verbose:
                print(f"[yaml-extract] Saved figure: {out}")

        fig = decomp_mlp.get(cfg_name, None)
        if fig is not None:
            out = cfg_dir / "fourier_band_decomp_mlp.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fourier_band_decomp_mlp/{cfg_name}", out)
            if verbose:
                print(f"[yaml-extract] Saved figure: {out}")

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
    verbose: bool = True,
) -> Dict[str, list[Path]]:
    """
    遍历某个实验根目录下的所有 npz 文件，凡是符合 example 格式的
    (至少包含 x_true/x_hat/x_interp)，就用 plot_example_from_npz 重新绘制
    四联图，并覆盖保存对应的 PNG。

    约定：
    - PNG 文件名 = NPZ 文件名去掉后缀再加 .png
      例如：
        example_linear.npz      -> example_linear.png
        linear_example_00.npz   -> linear_example_00.png
        mlp_example_03.npz      -> mlp_example_03.png
    - 目录结构不作假设，直接用 glob / rglob 查找所有 .npz，
      只要键满足就重画，其他一律跳过。

    返回:
        {
            "updated": [Path(...), ...],   # 成功重绘并保存的 PNG 列表
            "skipped": [Path(...), ...],   # 不是 example 格式而被跳过的 NPZ
            "failed":  [Path(...), ...],   # 解析或绘制出错的 NPZ
        }
    """
    from pathlib import Path

    exp_root = Path(exp_root)
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {exp_root}")

    if recursive:
        npz_paths = sorted(exp_root.rglob("*.npz"))
    else:
        npz_paths = sorted(exp_root.glob("*.npz"))

    if verbose:
        print(f"[redraw] Scanning {exp_root} ... found {len(npz_paths)} npz files")

    updated: list[Path] = []
    skipped: list[Path] = []
    failed: list[Path] = []

    for npz_path in npz_paths:
        try:
            data = np.load(npz_path, allow_pickle=False)
        except Exception as e:
            failed.append(npz_path)
            if verbose:
                print(f"[redraw] FAILED to load npz: {npz_path}  ({e})")
            continue

        files = set(data.files)

        # 只处理我们约定的 example 格式
        if not {"x_true", "x_hat", "x_interp"}.issubset(files):
            skipped.append(npz_path)
            continue

        try:
            # 利用已有的单文件绘图函数（会自动处理 mask、标题等细节）
            fig = plot_example_from_npz(npz_path)

            png_path = npz_path.with_suffix(".png")
            # 目录本身应当已存在，这里保险起见再 ensure 一次
            ensure_dir(png_path.parent)
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            updated.append(png_path)
            if verbose:
                print(f"[redraw] Updated figure: {png_path}")

        except Exception as e:
            failed.append(npz_path)
            if verbose:
                print(f"[redraw] FAILED to redraw: {npz_path}  ({e})")

    if verbose:
        print(
            f"[redraw] Done. updated={len(updated)}, "
            f"skipped={len(skipped)}, failed={len(failed)}"
        )

    return {
        "updated": updated,
        "skipped": skipped,
        "failed": failed,
    }
