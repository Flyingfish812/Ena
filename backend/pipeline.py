# backend/pipeline.py

"""
为 notebook / GUI 提供的高层“一键调用”接口。
"""

from pathlib import Path
from typing import Dict, Any

from .config.schemas import DataConfig, PodConfig, TrainConfig, EvalConfig
from .pod.compute import build_pod

from .dataio.io_utils import load_numpy, load_json, ensure_dir
from .dataio.nc_loader import load_raw_nc
from .pod.project import project_to_pod, reconstruct_from_pod
from .sampling.masks import generate_random_mask_hw, flatten_mask, apply_mask_flat
from .sampling.noise import add_gaussian_noise
from .models.linear_baseline import solve_pod_coeffs_least_squares
from .metrics.errors import nmse
from .viz.field_plots import plot_field_comparison, plot_error_map
from .viz.pod_plots import plot_pod_mode_groups
from .models.train_mlp import train_mlp_on_observations
from .viz.pod_plots import plot_pod_mode_groups
import numpy as np
import json
import matplotlib.pyplot as plt

from .eval.reconstruction import (
    run_linear_baseline_experiment,
    run_mlp_experiment,
)
from .eval.reports import results_to_dataframe
from .viz.curves import plot_nmse_vs_mask_rate, plot_nmse_vs_noise


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
            - NMSE vs mask_rate（固定最小 noise_sigma）
            - NMSE vs noise_sigma（固定最小 mask_rate）

    返回 result 字典包含：
    - "linear": 线性基线的原始结果
    - "mlp":    MLP 的原始结果（若 train_cfg 为 None 则为 None）
    - "df_linear": 线性结果转成的 DataFrame
    - "df_mlp":    MLP 结果转成的 DataFrame（或 None）
    - "fig_nmse_vs_mask": 叠加 linear / mlp 的 mask_rate 曲线图（或 None）
    - "fig_nmse_vs_noise": 叠加 linear / mlp 的 noise_sigma 曲线图（或 None）
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

        # 3) 生成两张最典型的曲线图
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

    if verbose:
        print("[full-eval] Done.")

    return {
        "linear": linear_results,
        "mlp": mlp_results,
        "df_linear": df_linear,
        "df_mlp": df_mlp,
        "fig_nmse_vs_mask": fig_mask,
        "fig_nmse_vs_noise": fig_noise,
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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一行代码测试完整流程（单帧）：

    - 若 pod_dir 下不存在 POD 结果，则自动构建 POD。
    - 加载 Ur / mean / meta。
    - 从 nc_path 中取一帧 snapshot。
    - 做 POD 截断自重建，计算 NMSE_pod，并打印“使用 r_eff 个模态”。
    - 生成随机 mask + 加噪观测。
    - 线性最小二乘解 POD 系数，重建场，计算 NMSE_linear。
    - 可视化：
        - Figure 1: [0]=True, [1]=POD truncation, [2]=Linear baseline
        - Figure 2: Linear 误差图
        - Figure 3: POD 模态分组叠加（每 group_size=modes_per_fig 个模态一组）

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

    # 4) POD 截断自重建
    if verbose:
        print("[quick_test] POD truncation self-reconstruction...")
        print(f"  -> 使用 r_eff = {r_eff} 个模态，对该帧 snapshot 投影后叠加重构")

    a_true = project_to_pod(x_flat, Ur_eff, mean_flat)
    x_pod_flat = reconstruct_from_pod(a_true, Ur_eff, mean_flat)
    x_pod = x_pod_flat.reshape(H, W, C)

    nmse_pod = nmse(x_pod, x)
    if verbose:
        print(f"  -> NMSE(POD truncation, r_eff={r_eff}) = {nmse_pod:.4e}")

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

    # 6) 线性最小二乘重建
    if verbose:
        print("[quick_test] Linear least-squares reconstruction in POD space...")

    Ur_masked = Ur_eff[mask_flat, :]   # [M, r_eff]
    a_lin = solve_pod_coeffs_least_squares(y_noisy, Ur_masked)
    x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
    x_lin = x_lin_flat.reshape(H, W, C)

    nmse_linear = nmse(x_lin, x)
    if verbose:
        print(f"  -> NMSE(Linear baseline, r_eff={r_eff}) = {nmse_linear:.4e}")

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

    # 8) POD 模态“低/中/高频分组唬人图”
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

    if verbose:
        print("[quick_test] Done.")

    result: Dict[str, Any] = {
        "nmse_pod": nmse_pod,
        "nmse_linear": nmse_linear,
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
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    一行代码测试完整流程（单帧，包含 MLP）：

    - 若无 POD 则自动构建 POD；
    - 基于整段时间序列构建 ObservationDataset，训练 MLP：
          y (masked noisy obs) -> a_true (POD coeffs)
    - 在指定 frame_idx 上计算：
          - POD 截断自身误差 NMSE_pod
          - 线性 least-squares baseline NMSE_linear
          - MLP baseline NMSE_mlp
    - 可视化：
          Figure 1: [0]=True, [1]=POD truncation, [2]=Linear baseline, [3]=MLP
          Figure 2: MLP 误差图 |MLP - True|
          Figure 3: POD 模态分组叠加（每组 modes_per_fig 个模态求和）

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

    # 5) POD 截断自重建（在目标 frame_idx 上）
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

    nmse_pod = nmse(x_pod, x)
    if verbose:
        print(f"  -> NMSE(POD truncation, r_eff={r_eff}) = {nmse_pod:.4e}")

    # 6) 线性 least-squares baseline（同一 mask）
    if verbose:
        print("[quick_test_mlp] Linear least-squares baseline on target frame...")

    y_true = apply_mask_flat(x_flat, mask_flat)        # [M]
    y_true_noisy = add_gaussian_noise(y_true, sigma=noise_sigma, seed=0)

    Ur_masked = Ur_eff[mask_flat, :]                   # [M,r_eff]
    a_lin = solve_pod_coeffs_least_squares(y_true_noisy, Ur_masked)
    x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
    x_lin = x_lin_flat.reshape(H, W, C)

    nmse_linear = nmse(x_lin, x)
    if verbose:
        print(f"  -> NMSE(Linear baseline, r_eff={r_eff}) = {nmse_linear:.4e}")

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

    # 8) 用 MLP 在目标帧上推断
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

    nmse_mlp = nmse(x_mlp, x)
    if verbose:
        print(f"  -> NMSE(MLP baseline, r_eff={r_eff}) = {nmse_mlp:.4e}")

    # 9) 场对比 + 误差可视化
    if verbose:
        print("[quick_test_mlp] Plotting field comparison and error map...")
        print("  -> Figure layout: [0]=True, [1]=POD truncation, "
              "[2]=Linear baseline, [3]=MLP baseline")

    fig_fields = plot_field_comparison(
        x_true_hw=x,
        x_lin_hw=x_pod,
        x_nn_hw=None,   # 先用两个额外通道来装 linear 和 mlp，我们会手动调用 imshow
    )

    # 为了控制顺序，这里重画一张 4 列版：
    import matplotlib.pyplot as plt
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

    if verbose:
        print("[quick_test_mlp] Done.")

    result: Dict[str, Any] = {
        "nmse_pod": nmse_pod,
        "nmse_linear": nmse_linear,
        "nmse_mlp": nmse_mlp,
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
