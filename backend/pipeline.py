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
from .viz.field_plots import plot_field_comparison, plot_error_map
from .viz.pod_plots import plot_pod_mode_groups
from .models.train_mlp import train_mlp_on_observations
import numpy as np
import json
import matplotlib.pyplot as plt

from .eval.reconstruction import (
    run_linear_baseline_experiment,
    run_mlp_experiment,
)
from .eval.reports import results_to_dataframe
from .viz.curves import plot_nmse_vs_mask_rate, plot_nmse_vs_noise
from .viz.multiscale_plots import plot_multiscale_bar, plot_multiscale_summary
from .config.yaml_io import load_experiment_yaml, save_experiment_yaml  # 可选用
from .eval.reports import (
    results_to_dataframe,
    save_full_experiment_results,
    generate_experiment_report_md,
)

def _plot_recon_quadruple(
    x_interp: np.ndarray,
    x_hat: np.ndarray,
    x_true: np.ndarray,
    *,
    mask_hw: np.ndarray | None = None,
    model_name: str = "linear",
    channel: int = 0,
    title_prefix: str = "",
) -> plt.Figure:
    """
    画一张四联图：
        [0] Input: interpolation (+ 可选采样点)
        [1] Output: 模型预测
        [2] Target: 真值
        [3] Error: true - pred

    统一色标，并把 colorbar 放到图像下方，避免覆盖到图像上。
    若提供 mask_hw（[H,W] bool），则在 input 图上叠加采样点。
    """
    # 保证是三维
    if x_true.ndim != 3 or x_hat.ndim != 3 or x_interp.ndim != 3:
        raise ValueError("x_true / x_hat / x_interp must all be [H,W,C].")

    x_true_ch = x_true[..., channel]
    x_hat_ch = x_hat[..., channel]
    x_interp_ch = x_interp[..., channel]

    err = x_true_ch - x_hat_ch

    vmin = float(min(x_true_ch.min(), x_hat_ch.min(), x_interp_ch.min()))
    vmax = float(max(x_true_ch.max(), x_hat_ch.max(), x_interp_ch.max()))

    err_max = float(np.max(np.abs(err)))
    err_vmin, err_vmax = -err_max, err_max

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    titles = [
        f"Input (interp, ch={channel})",
        f"Output ({model_name}, ch={channel})",
        f"Target (true, ch={channel})",
        f"Error (true - {model_name}, ch={channel})",
    ]

    ims = []
    ims.append(
        axes[0].imshow(
            x_interp_ch,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
    )
    ims.append(
        axes[1].imshow(
            x_hat_ch,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
    )
    ims.append(
        axes[2].imshow(
            x_true_ch,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
    )
    ims.append(
        axes[3].imshow(
            err,
            origin="lower",
            cmap="RdBu_r",
            vmin=err_vmin,
            vmax=err_vmax,
        )
    )

    # 在 input 子图上标出采样点
    if mask_hw is not None:
        mask_hw = np.asarray(mask_hw, dtype=bool)
        yy, xx = np.where(mask_hw)
        # 画成空心小圆点，避免挡住色块
        axes[0].scatter(
            xx,
            yy,
            s=6,
            facecolors="none",
            edgecolors="k",
            linewidths=0.4,
            zorder=2,
        )

    for ax, t in zip(axes, titles, strict=False):
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # 先收紧布局，再在底部留出空间给 colorbar
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.18, wspace=0.25)

    # 共享 colorbar（物理场共一个水平色条）
    cbar1 = fig.colorbar(
        ims[0],
        ax=axes[:3],
        orientation="horizontal",
        fraction=0.04,
        pad=0.12,
    )
    cbar1.ax.set_xlabel("Field value", fontsize=8)

    # 误差场单独一个色条
    cbar2 = fig.colorbar(
        ims[3],
        ax=axes[3],
        orientation="horizontal",
        fraction=0.04,
        pad=0.25,
    )
    cbar2.ax.set_xlabel("Error", fontsize=8)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=11)

    return fig

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

    流程：
    1. 线性基线 sweep：run_linear_baseline_experiment
    2. 若提供了 train_cfg：
        - 在 eval_cfg.mask_rates / noise_sigmas 上跑 MLP sweep：run_mlp_experiment
        - 自动生成两类典型曲线：
            * NMSE vs mask_rate（固定最小 noise_sigma）
            * NMSE vs noise_sigma（固定最小 mask_rate）
    3. 从 linear / mlp 结果中：
        - 保留一个代表样本四联图（fig_example_linear / fig_example_mlp）；
        - 若结果中包含 "examples" 与 "mask_hw_map"，
          则为每个 (p,σ) 生成若干张四联图：
              fig_examples_linear: {config_name: [Figure, ...]}
              fig_examples_mlp:   {config_name: [Figure, ...]}

    返回 result 字典包含：
    - "linear": 线性基线的原始结果
    - "mlp":    MLP 的原始结果（若 train_cfg 为 None 则为 None）
    - "df_linear": 线性结果转成的 DataFrame
    - "df_mlp":    MLP 结果转成的 DataFrame（或 None）
    - "fig_nmse_vs_mask": 叠加 linear / mlp 的 mask_rate 曲线图（或 None）
    - "fig_nmse_vs_noise": 叠加 linear / mlp 的 noise_sigma 曲线图（或 None）
    - "fig_example_linear": 单个代表性四联图（或 None）
    - "fig_example_mlp":    同上（或 None）
    - "fig_examples_linear": 每个 (p,σ) 的多张四联图映射（或 None）
    - "fig_examples_mlp":   同上（或 None）
    """
    if verbose:
        print("=== [full-eval] Start full evaluation pipeline ===")

    # 1) 线性基线 sweep
    if verbose:
        print("[full-eval] Running linear baseline sweep ...")
    linear_results = run_linear_baseline_experiment(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        verbose=verbose,
    )
    df_linear = results_to_dataframe(linear_results)

    mlp_results = None
    df_mlp = None
    fig_mask = None
    fig_noise = None

    # 2) 若有 train_cfg，则跑 MLP sweep
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

        # 3) 生成两张最典型的曲线图（linear + mlp）
        if verbose:
            print("[full-eval] Plotting NMSE curves ...")

        fig_mask, ax_mask = plt.subplots(1, 1, figsize=(4, 3))
        plot_nmse_vs_mask_rate(linear_results, ax=ax_mask, label="linear")
        if mlp_results is not None:
            plot_nmse_vs_mask_rate(mlp_results, ax=ax_mask, label="mlp")
        ax_mask.legend()
        ax_mask.set_title("NMSE vs mask_rate (linear vs mlp)")

        fig_noise, ax_noise = plt.subplots(1, 1, figsize=(4, 3))
        plot_nmse_vs_noise(linear_results, ax=ax_noise, label="linear")
        if mlp_results is not None:
            plot_nmse_vs_noise(mlp_results, ax=ax_noise, label="mlp")
        ax_noise.legend()
        ax_noise.set_title("NMSE vs noise_sigma (linear vs mlp)")

    # 3) 代表性四联图（旧接口，单个 example）
    fig_example_linear = None
    fig_example_mlp = None

    # 3.1 Linear example：input / output / target / error
    ex_lin = linear_results.get("example_recon", None)
    if ex_lin is not None:
        x_true = np.asarray(ex_lin["x_true"])
        x_lin = np.asarray(ex_lin["x_lin"])
        x_interp = np.asarray(ex_lin["x_interp"])

        # optional：从 linear 的 mask_hw_map 中拿对应 mask（用于代表图）
        mask_hw = None
        mask_map = linear_results.get("mask_hw_map", {}) or {}
        try:
            p = float(ex_lin["mask_rate"])
            mask_key = f"{p:.6g}"
            if mask_key in mask_map:
                mask_hw = np.asarray(mask_map[mask_key], dtype=bool)
        except Exception:
            mask_hw = None

        title = (
            f"Linear example (frame={ex_lin['frame_idx']}, "
            f"p={ex_lin['mask_rate']:.3g}, σ={ex_lin['noise_sigma']:.3g})"
        )
        fig_example_linear = _plot_recon_quadruple(
            x_interp=x_interp,
            x_hat=x_lin,
            x_true=x_true,
            mask_hw=mask_hw,
            model_name="linear",
            channel=0,
            title_prefix=title,
        )

    # 3.2 MLP example：input / output / target / error
    if mlp_results is not None:
        ex_mlp = mlp_results.get("example_recon", None)
        if ex_mlp is not None:
            x_true = np.asarray(ex_mlp["x_true"])
            x_mlp = np.asarray(ex_mlp["x_mlp"])
            x_interp = np.asarray(ex_mlp["x_interp"])

            mask_hw = None
            mask_map_mlp = mlp_results.get("mask_hw_map", {}) or {}
            try:
                p = float(ex_mlp["mask_rate"])
                mask_key = f"{p:.6g}"
                if mask_key in mask_map_mlp:
                    mask_hw = np.asarray(mask_map_mlp[mask_key], dtype=bool)
            except Exception:
                mask_hw = None

            title = (
                f"MLP example (frame={ex_mlp['frame_idx']}, "
                f"p={ex_mlp['mask_rate']:.3g}, σ={ex_mlp['noise_sigma']:.3g})"
            )
            fig_example_mlp = _plot_recon_quadruple(
                x_interp=x_interp,
                x_hat=x_mlp,
                x_true=x_true,
                mask_hw=mask_hw,
                model_name="mlp",
                channel=0,
                title_prefix=title,
            )

    # 4) 新：为每个 (p,σ) 生成多张四联图，按配置名组织
    fig_examples_linear: Dict[str, list[plt.Figure]] = {}
    fig_examples_mlp: Dict[str, list[plt.Figure]] = {}

    # 4.1 Linear 多样本
    lin_examples = linear_results.get("examples", []) or []
    lin_mask_map = linear_results.get("mask_hw_map", {}) or {}
    if lin_examples:
        if verbose:
            print("[full-eval] Generating per-(p,σ) linear quadruple figures ...")
        for ex in lin_examples:
            p = float(ex["mask_rate"])
            s = float(ex["noise_sigma"])
            cfg_name = f"p{p:.4f}_sigma{s:.3g}".replace(".", "p")

            x_true = np.asarray(ex["x_true"])
            x_lin = np.asarray(ex["x_lin"])
            x_interp = np.asarray(ex["x_interp"])

            mask_key = f"{p:.6g}"
            mask_hw = None
            if mask_key in lin_mask_map:
                mask_hw = np.asarray(lin_mask_map[mask_key], dtype=bool)

            title = (
                f"Linear (frame={ex['frame_idx']}, "
                f"p={p:.3g}, σ={s:.3g})"
            )
            fig = _plot_recon_quadruple(
                x_interp=x_interp,
                x_hat=x_lin,
                x_true=x_true,
                mask_hw=mask_hw,
                model_name="linear",
                channel=0,
                title_prefix=title,
            )

            fig_examples_linear.setdefault(cfg_name, []).append(fig)

    # 4.2 MLP 多样本
    if mlp_results is not None:
        mlp_examples = mlp_results.get("examples", []) or []
        mlp_mask_map = mlp_results.get("mask_hw_map", {}) or []
        if mlp_examples:
            if verbose:
                print("[full-eval] Generating per-(p,σ) MLP quadruple figures ...")
            for ex in mlp_examples:
                p = float(ex["mask_rate"])
                s = float(ex["noise_sigma"])
                cfg_name = f"p{p:.4f}_sigma{s:.3g}".replace(".", "p")

                x_true = np.asarray(ex["x_true"])
                x_mlp = np.asarray(ex["x_mlp"])
                x_interp = np.asarray(ex["x_interp"])

                mask_key = f"{p:.6g}"
                mask_hw = None
                if mask_key in mlp_mask_map:
                    mask_hw = np.asarray(mlp_mask_map[mask_key], dtype=bool)

                title = (
                    f"MLP (frame={ex['frame_idx']}, "
                    f"p={p:.3g}, σ={s:.3g})"
                )
                fig = _plot_recon_quadruple(
                    x_interp=x_interp,
                    x_hat=x_mlp,
                    x_true=x_true,
                    mask_hw=mask_hw,
                    model_name="mlp",
                    channel=0,
                    title_prefix=title,
                )

                fig_examples_mlp.setdefault(cfg_name, []).append(fig)

    if verbose:
        print("[full-eval] Done.")

    return {
        "linear": linear_results,
        "mlp": mlp_results,
        "df_linear": df_linear,
        "df_mlp": df_mlp,
        "fig_nmse_vs_mask": fig_mask,
        "fig_nmse_vs_noise": fig_noise,
        "fig_example_linear": fig_example_linear,
        "fig_example_mlp": fig_example_mlp,
        # 新增：每个 (p,σ) 多帧四联图
        "fig_examples_linear": fig_examples_linear or None,
        "fig_examples_mlp": fig_examples_mlp or None,
    }

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
            .replace(".", "p")
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
            .replace(".", "p")
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
        / f"p{train_mask_rate:.4f}_sigma{train_noise_sigma:.3g}".replace(".", "p"),
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

def run_experiment_from_yaml(
    yaml_path: str | Path,
    *,
    experiment_name: str | None = None,
    save_root: str | Path = "artifacts/experiments",
    generate_report: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)
    if experiment_name is None:
        experiment_name = yaml_path.stem

    if verbose:
        print(f"[yaml-experiment] Loading configs from {yaml_path} ...")

    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    if verbose:
        print("[yaml-experiment] Running full evaluation pipeline ...")

    all_result = run_full_eval_pipeline(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        verbose=verbose,
    )

    save_root = Path(save_root)
    exp_dir = save_root / experiment_name
    figs_dir = exp_dir / "figs"
    ensure_dir(figs_dir)

    fig_paths: Dict[str, Any] = {}

    fig_mask = all_result.get("fig_nmse_vs_mask", None)
    if fig_mask is not None:
        p = figs_dir / "nmse_vs_mask.png"
        fig_mask.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_mask"] = p
        if verbose:
            print(f"[yaml-experiment] Saved figure: {p}")

    fig_noise = all_result.get("fig_nmse_vs_noise", None)
    if fig_noise is not None:
        p = figs_dir / "nmse_vs_noise.png"
        fig_noise.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_nmse_vs_noise"] = p
        if verbose:
            print(f"[yaml-experiment] Saved figure: {p}")

    fig_example_linear = all_result.get("fig_example_linear", None)
    if fig_example_linear is not None:
        p = figs_dir / "example_linear.png"
        fig_example_linear.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_example_linear"] = p
        if verbose:
            print(f"[yaml-experiment] Saved figure: {p}")

    fig_example_mlp = all_result.get("fig_example_mlp", None)
    if fig_example_mlp is not None:
        p = figs_dir / "example_mlp.png"
        fig_example_mlp.savefig(p, dpi=300, bbox_inches="tight")
        fig_paths["fig_example_mlp"] = p
        if verbose:
            print(f"[yaml-experiment] Saved figure: {p}")

    # 四联图 per-(p,σ)
    fig_examples_linear = all_result.get("fig_examples_linear") or {}
    fig_examples_mlp = all_result.get("fig_examples_mlp") or {}

    if fig_examples_linear or fig_examples_mlp:
        if verbose:
            print("[yaml-experiment] Saving per-(p,σ) quadruple figures ...")

    all_cfg_names = set(fig_examples_linear.keys()) | set(fig_examples_mlp.keys())

    for cfg_name in sorted(all_cfg_names):
        cfg_dir = exp_dir / cfg_name
        ensure_dir(cfg_dir)

        lin_figs = fig_examples_linear.get(cfg_name, []) or []
        for idx, fig in enumerate(lin_figs):
            p = cfg_dir / f"linear_example_{idx:02d}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fig_examples_linear/{cfg_name}", [])
            fig_paths[f"fig_examples_linear/{cfg_name}"].append(p)
            if verbose:
                print(f"[yaml-experiment] Saved figure: {p}")

        mlp_figs = fig_examples_mlp.get(cfg_name, []) or []
        for idx, fig in enumerate(mlp_figs):
            p = cfg_dir / f"mlp_example_{idx:02d}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            fig_paths.setdefault(f"fig_examples_mlp/{cfg_name}", [])
            fig_paths[f"fig_examples_mlp/{cfg_name}"].append(p)
            if verbose:
                print(f"[yaml-experiment] Saved figure: {p}")

    # ===== 多尺度 “四合一” 图：linear 和 mlp 分开画 =====
    lin_res = all_result.get("linear", None)
    mlp_res = all_result.get("mlp", None)

    if lin_res is not None and mlp_res is not None:
        lin_entries = lin_res.get("entries", []) or []
        mlp_entries = mlp_res.get("entries", []) or []

        if verbose:
            print("[yaml-experiment] Generating multiscale summary figures for each (p,σ) ...")

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
            mlp_res.get("meta", {}).get("energy_cum")
            or lin_res.get("meta", {}).get("energy_cum")
        )

        all_ps = sorted(set(lin_idx.keys()) | set(mlp_idx.keys()))

        for (p_val, s_val) in all_ps:
            entry_lin = lin_idx.get((p_val, s_val), None)
            entry_mlp = mlp_idx.get((p_val, s_val), None)

            cfg_name = f"p{p_val:.4f}_sigma{s_val:.3g}".replace(".", "p")
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
                fig_paths.setdefault(f"multiscale_linear/{cfg_name}", [])
                fig_paths[f"multiscale_linear/{cfg_name}"].append(out_lin)
                if verbose:
                    print(f"[yaml-experiment] Saved multiscale linear: {out_lin}")

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
                fig_paths.setdefault(f"multiscale_mlp/{cfg_name}", [])
                fig_paths[f"multiscale_mlp/{cfg_name}"].append(out_mlp)
                if verbose:
                    print(f"[yaml-experiment] Saved multiscale mlp: {out_mlp}")

        # 选一个代表 (p,σ)，在 figs/ 下再各存一张 linear / mlp 示例
        train_meta = mlp_res.get("meta", {}).get("train_cfg", {}) or {}
        p_train = train_meta.get("mask_rate", None)
        s_train = train_meta.get("noise_sigma", None)

        rep_lin = None
        rep_mlp = None

        if p_train is not None and s_train is not None:
            key = (float(p_train), float(s_train))
            rep_lin = lin_idx.get(key, None)
            rep_mlp = mlp_idx.get(key, None)

        if rep_mlp is None and mlp_entries:
            rep_mlp = mlp_entries[0]
            key = (float(rep_mlp.get("mask_rate", -1.0)),
                   float(rep_mlp.get("noise_sigma", -1.0)))
            rep_lin = lin_idx.get(key, None)

        if rep_lin is not None:
            title = (
                f"Multiscale summary (linear, p={rep_lin['mask_rate']}, "
                f"σ={rep_lin['noise_sigma']})"
            )
            fig_lin_rep, _ = plot_multiscale_summary(
                rep_lin,
                energy_cum=energy_cum,
                title_prefix=title,
                model_label="linear",
            )
            p_lin = figs_dir / "multiscale_linear_example.png"
            fig_lin_rep.savefig(p_lin, dpi=300, bbox_inches="tight")
            fig_paths["fig_multiscale_linear_example"] = p_lin
            if verbose:
                print(f"[yaml-experiment] Saved representative multiscale linear: {p_lin}")

        if rep_mlp is not None:
            title = (
                f"Multiscale summary (mlp, p={rep_mlp['mask_rate']}, "
                f"σ={rep_mlp['noise_sigma']})"
            )
            fig_mlp_rep, _ = plot_multiscale_summary(
                rep_mlp,
                energy_cum=energy_cum,
                title_prefix=title,
                model_label="mlp",
            )
            p_mlp = figs_dir / "multiscale_mlp_example.png"
            fig_mlp_rep.savefig(p_mlp, dpi=300, bbox_inches="tight")
            fig_paths["fig_multiscale_mlp_example"] = p_mlp
            # 兼容旧字段名：fig_multiscale_example 指向 mlp 的例子
            fig_paths["fig_multiscale_example"] = p_mlp
            if verbose:
                print(f"[yaml-experiment] Saved representative multiscale mlp: {p_mlp}")

    saved_paths = save_full_experiment_results(
        all_result,
        base_dir=save_root,
        experiment_name=experiment_name,
    )
    saved_paths.update(fig_paths)
    all_result["saved_paths"] = saved_paths

    report_path = None
    if generate_report:
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
