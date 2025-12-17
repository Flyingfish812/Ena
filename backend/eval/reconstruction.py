# backend/eval/reconstruction.py

"""
在线性基线与 MLP 之间进行对比评估。

负责：
- 加载 snapshot
- 对每个 mask_rate / noise_sigma 组合执行重建
- 计算全场与多尺度误差
"""

from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from ..config.schemas import DataConfig, PodConfig, EvalConfig, TrainConfig
from ..pod.compute import build_pod
from ..pod.project import project_to_pod, reconstruct_from_pod
from ..dataio.nc_loader import load_raw_nc
from ..dataio.io_utils import load_numpy, load_json, ensure_dir
from ..sampling.masks import generate_random_mask_hw, flatten_mask, apply_mask_flat
from ..sampling.noise import add_gaussian_noise
from ..models.linear_baseline import solve_pod_coeffs_least_squares
from ..models.train_mlp import train_mlp_on_observations
from ..metrics.errors import nmse, nmae, psnr
from ..metrics.multiscale import compute_pod_band_errors
from ..metrics.metrics import (
    rmse_per_mode,
    nrmse_per_mode,
    nrmse_per_band,
    partial_recon_nmse,
)
from ..metrics.fourier_metrics import (
    fourier_radial_nrmse_curve,
    kstar_from_radial_curve,
    fourier_band_nrmse,
    energy_spectrum,
)
from ..fourier.filters import auto_pick_k_edges_from_energy

def _load_pod_aux_info(
    pod_cfg: PodConfig,
    r_eff: int,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray | None, list[dict]]:
    """
    从 POD 目录中加载：
    - eigenvalues.npy（若存在）
    - phi_groups.json（若存在）

    若某个文件不存在，则做合理降级：
    - eigenvalues 为空则在 nrmse_per_mode 中退化为用 std 做归一化；
    - phi_groups 为空则根据 r_eff 和 group_size=16 现场构造 S1, S2, ...。
    """
    save_dir = pod_cfg.save_dir
    eigen_path = save_dir / "eigenvalues.npy"
    phi_path = save_dir / "phi_groups.json"

    eigenvalues: np.ndarray | None
    if eigen_path.exists():
        eigenvalues = load_numpy(eigen_path).astype(np.float64)
        if eigenvalues.shape[0] < r_eff:
            # 容错：老版本可能只存了更少的特征值
            eigenvalues = eigenvalues
        else:
            eigenvalues = eigenvalues[:r_eff]
        if verbose:
            print(f"[eval] Loaded eigenvalues from {eigen_path}, shape={eigenvalues.shape}")
    else:
        eigenvalues = None
        if verbose:
            print(f"[eval] eigenvalues.npy not found in {save_dir}, NRMSE_per_mode will use std-based normalization.")

    phi_groups: list[dict] = []
    if phi_path.exists():
        phi_json = load_json(phi_path)
        phi_groups = list(phi_json.get("groups", []))
        if verbose:
            print(f"[eval] Loaded phi_groups from {phi_path}, count={len(phi_groups)}")
    else:
        # 回退：按 16 模态一组构造
        from ..pod.compute import _build_phi_groups  # 局部导入避免循环
        phi_groups = _build_phi_groups(r_used=r_eff, group_size=16)
        if verbose:
            print(f"[eval] phi_groups.json not found, fallback to 16-modes groups, count={len(phi_groups)}")

    return eigenvalues, phi_groups

def _compute_interp_baseline(
    x: np.ndarray,      # [H,W,C]
    mask_hw: np.ndarray # [H,W]，True 表示该空间格点被观测到（对所有通道）
) -> np.ndarray:
    """
    非物理但简单可重复的插值 baseline：

    - 在未观测格点使用“该帧中观测点的通道均值”填充；
    - 在观测点直接使用真值 x（不考虑噪声）。
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [H,W,C], got {x.shape}")
    H, W, C = x.shape
    if mask_hw.shape != (H, W):
        raise ValueError(f"mask_hw shape {mask_hw.shape} != (H,W)=({H},{W})")

    x_interp = np.empty_like(x)
    for c in range(C):
        obs_vals = x[..., c][mask_hw]
        if obs_vals.size == 0:
            mean_val = 0.0
        else:
            mean_val = float(obs_vals.mean())
        # 未观测位置用均值填，观测位置用真值
        x_interp[..., c].fill(mean_val)
        x_interp[..., c][mask_hw] = x[..., c][mask_hw]

    return x_interp

def _load_or_build_pod(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    工具函数：若 save_dir 下无 POD，则先构建；然后加载 Ur / mean_flat / meta。
    """
    save_dir = pod_cfg.save_dir
    Ur_path = save_dir / "Ur.npy"
    mean_path = save_dir / "mean_flat.npy"
    meta_path = save_dir / "pod_meta.json"

    if not (Ur_path.exists() and mean_path.exists() and meta_path.exists()):
        if verbose:
            print(f"[eval] POD artifacts not found in {save_dir}, building POD...")
        ensure_dir(save_dir)
        build_pod(data_cfg, pod_cfg, verbose=verbose, plot=False)
    else:
        if verbose:
            print(f"[eval] Found existing POD in {save_dir}, skip rebuilding.")

    Ur = load_numpy(Ur_path)           # [D,r0]
    mean_flat = load_numpy(mean_path)  # [D]
    meta = load_json(meta_path)
    return Ur, mean_flat, meta

def _prepare_snapshots(
    data_cfg: DataConfig,
    Ur: np.ndarray,
    mean_flat: np.ndarray,
    r_eff: int,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载全量数据，返回：
    - X_thwc: [T,H,W,C]
    - A_true: [T,r_eff] 对每一帧的真实 POD 系数
    """
    if verbose:
        print(f"[eval] Loading full raw data from {data_cfg.nc_path} ...")

    X_thwc = load_raw_nc(data_cfg)         # [T,H,W,C]
    T, H, W, C = X_thwc.shape
    D = H * W * C

    if Ur.shape[0] != D:
        raise ValueError(
            f"Ur first dim {Ur.shape[0]} not equal to H*W*C={D} inferred from data."
        )

    # 展平为 [T,D]
    X_flat_all = X_thwc.reshape(T, D)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  -> X_thwc shape = {X_thwc.shape}, flatten = [{T}, {D}], r_eff={r_eff}")

    # 所有帧的 POD 真系数
    A_true = project_to_pod(X_flat_all, Ur_eff, mean_flat)  # [T,r_eff]

    if verbose:
        print("  -> Projected all snapshots to POD space: A_true shape =", A_true.shape)

    return X_thwc, A_true

def _select_sample_frames(T: int, n: int) -> list[int]:
    """Pick n frame indices uniformly from [0, T-1]."""
    if T <= 0:
        return []
    n = int(n)
    if n <= 0:
        return []
    n = min(n, T)
    return [int(x) for x in np.linspace(0, T - 1, num=n, dtype=int)]

def _infer_fourier_grid_meta(eval_cfg: EvalConfig, *, H: int, W: int) -> dict:
    """Map eval_cfg.fourier_grid to grid_meta consumed by fourier_metrics."""
    g = dict(getattr(eval_cfg, "fourier_grid", {}) or {})
    # If user gives Lx/Ly but not dx/dy, metrics will compute dx=Lx/W, dy=Ly/H
    # Also allow angular flag
    return g

def _get_fourier_settings(eval_cfg: EvalConfig) -> dict:
    """Collect Fourier-related settings from eval_cfg with defaults.
    Support both legacy flat fields (fourier_enabled, fourier_grid, ...)
    and nested YAML block (eval_cfg.fourier.enabled, eval_cfg.fourier.grid_meta, ...).
    """
    f = getattr(eval_cfg, "fourier", None)

    def _f_get(key: str, default=None):
        if f is None:
            return default
        if isinstance(f, dict):
            return f.get(key, default)
        return getattr(f, key, default)

    # enabled: prefer legacy flat field if present, else nested
    enabled_flat = getattr(eval_cfg, "fourier_enabled", None)
    enabled = bool(enabled_flat) if enabled_flat is not None else bool(_f_get("enabled", False))

    # grid_meta: legacy flat field name is fourier_grid; nested uses fourier.grid_meta
    grid_flat = getattr(eval_cfg, "fourier_grid", None)
    grid_meta = grid_flat if grid_flat is not None else _f_get("grid_meta", {})
    grid_meta = dict(grid_meta or {})

    # band scheme: legacy flat field fourier_band_scheme; nested fourier.band_scheme
    band_scheme_flat = getattr(eval_cfg, "fourier_band_scheme", None)
    band_scheme = band_scheme_flat if band_scheme_flat is not None else _f_get("band_scheme", "energy_quantile")

    # lambda_edges: legacy flat field fourier_lambda_edges; nested fourier.lambda_edges
    lambda_edges_flat = getattr(eval_cfg, "fourier_lambda_edges", None)
    lambda_edges = lambda_edges_flat if lambda_edges_flat is not None else _f_get("lambda_edges", None)

    return {
        "enabled": enabled,
        "grid_meta": grid_meta,

        "mean_mode_true": getattr(eval_cfg, "fourier_mean_mode_true", _f_get("mean_mode_true", "global")),
        "num_bins": int(getattr(eval_cfg, "fourier_num_bins", _f_get("num_bins", 64))),
        "k_max": getattr(eval_cfg, "fourier_k_max", _f_get("k_max", None)),

        "kstar_thr": float(getattr(eval_cfg, "fourier_kstar_threshold", _f_get("kstar_threshold", 1.0))),
        "mono_env": bool(getattr(eval_cfg, "fourier_monotone_envelope", _f_get("monotone_envelope", True))),
        "sample_frames": int(getattr(eval_cfg, "fourier_sample_frames", _f_get("sample_frames", 8))),
        "save_curve": bool(getattr(eval_cfg, "fourier_save_curve", _f_get("save_curve", False))),

        # band edges
        "k_edges_cfg": getattr(eval_cfg, "fourier_k_edges", _f_get("k_edges", None)),
        "auto_edges_quantiles": getattr(eval_cfg, "fourier_auto_edges_quantiles", _f_get("auto_edges_quantiles", (0.80, 0.95))),
        "band_names_cfg": tuple(getattr(eval_cfg, "fourier_band_names", _f_get("band_names", ("L", "M", "H")))),

        "band_scheme": band_scheme,
        "lambda_edges": lambda_edges,
    }

def _compute_fourier_for_setting(
    *,
    X_thwc: np.ndarray,             # [T,H,W,C]
    A_hat_all: np.ndarray,          # [T,r_eff]  (linear or mlp coeffs)
    Ur_eff: np.ndarray,             # [D,r_eff]
    mean_flat: np.ndarray,          # [D]
    eval_cfg: EvalConfig,
    H: int,
    W: int,
) -> tuple[dict | None, float | None, dict | None, dict | None]:
    """
    Returns:
      fourier_band_nrmse_out: Dict[str,float] | None
      k_star_out: float | None
      fourier_curve_out: Dict[str,Any] | None
      meta_hint: Dict[str,Any] | None   (用于 result["meta"] 的三件套填充)
    """
    fs = _get_fourier_settings(eval_cfg)
    if not fs["enabled"]:
        return None, None, None, None

    T = int(X_thwc.shape[0])
    t_samples = _select_sample_frames(T, fs["sample_frames"])
    if len(t_samples) == 0:
        return None, None, None, None

    Et_sum = None
    Ee_sum = None
    k_centers_ref = None

    # 1) 累积 radial 能量（true & error）
    for t_s in t_samples:
        x_true = X_thwc[t_s]  # [H,W,C]
        a_hat = A_hat_all[t_s]
        x_hat = reconstruct_from_pod(a_hat, Ur_eff, mean_flat).reshape(H, W, -1)

        k_centers, Et, Ee, _nrmse_k = fourier_radial_nrmse_curve(
            x_hat=x_hat,
            x_true=x_true,
            num_bins=fs["num_bins"],
            k_max=fs["k_max"],
            grid_meta=fs["grid_meta"],
            mean_mode=fs["mean_mode_true"],
        )

        if k_centers_ref is None:
            k_centers_ref = k_centers
            Et_sum = np.zeros_like(Et, dtype=float)
            Ee_sum = np.zeros_like(Ee, dtype=float)

        Et_sum += Et
        Ee_sum += Ee

    if k_centers_ref is None:
        return None, None, None, None

    # 2) NRMSE(k) + k*
    nrmse_k_mean = np.sqrt(Ee_sum / (Et_sum + 1e-12))
    k_star_out = kstar_from_radial_curve(
        k_centers_ref,
        nrmse_k_mean,
        threshold=fs["kstar_thr"],
        monotone_enforce=fs["mono_env"],
    )

    # 3) k_edges
    #    内部用 full_edges=[k0,k1,...,kN]（含端点）做统计；
    #    对外输出 interior_edges=[k1,...,k_{N-1}] 供画图/空间分解，避免多一段 band3。
    full_edges: list[float] | None = None

    scheme = str(fs.get("band_scheme", "energy_quantile")).lower()

    if scheme == "physical":
        # 物理尺度：用波长 λ 的分界来定义 bands（k = 1/λ）
        lam_edges = fs.get("lambda_edges", None)
        if lam_edges is None:
            # 给一个稳健默认：如果提供了 obstacle_diameter，用 [8D, 2D]；
            # 否则退化到 [Ly, Ly/4]
            g = fs.get("grid_meta", {}) or {}
            D = g.get("obstacle_diameter", None)
            Ly = g.get("Ly", None)
            if D is not None:
                lam_edges = [8.0 * float(D), 2.0 * float(D)]
            elif Ly is not None:
                lam_edges = [float(Ly), float(Ly) / 4.0]
            else:
                lam_edges = [1.0, 0.25]

        lam_edges = [float(x) for x in lam_edges]
        # L/M/H: λ 大 -> k 小；所以要把边界转成递增的 k
        k_interior = sorted([1.0 / max(le, 1e-12) for le in lam_edges])
        k0 = 0.0
        kN = float(np.max(k_centers_ref)) if fs["k_max"] is None else float(fs["k_max"])
        full_edges = [k0] + k_interior + [kN]

    else:
        # energy_quantile（旧逻辑）或显式 k_edges_cfg
        if fs["k_edges_cfg"] is None:
            full_edges = auto_pick_k_edges_from_energy(
                k_centers_ref,
                Et_sum,
                quantiles=fs["auto_edges_quantiles"],
            )
        else:
            # 允许用户传：
            # - interior edges: [k1,k2]
            # - full edges: [k0,k1,k2,kN]
            cfg_edges = [float(v) for v in fs["k_edges_cfg"]]
            if len(cfg_edges) >= 2 and cfg_edges[0] <= 1e-12:
                # 看起来像 full edges
                full_edges = cfg_edges
            else:
                # 当作 interior edges
                k0 = 0.0
                kN = float(np.max(k_centers_ref)) if fs["k_max"] is None else float(fs["k_max"])
                full_edges = [k0] + cfg_edges + [kN]

    if full_edges is None or len(full_edges) < 3:
        return None, None, None, None

    # 对外输出的“内部边界”
    interior_edges = [float(v) for v in full_edges[1:-1]]

    # 4) band names 对齐 band 数
    nb = len(full_edges) - 1
    band_names = list(fs["band_names_cfg"])
    if len(band_names) != nb:
        band_names = [f"band_{i}" for i in range(nb)]

    # 5) band NRMSE：对 sampled frames 逐帧算后取均值（保持你原本语义）
    band_vals = []
    for t_s in t_samples:
        x_true = X_thwc[t_s]
        a_hat = A_hat_all[t_s]
        x_hat = reconstruct_from_pod(a_hat, Ur_eff, mean_flat).reshape(H, W, -1)

        b = fourier_band_nrmse(
            x_hat=x_hat,
            x_true=x_true,
            k_edges=full_edges,
            grid_meta=fs["grid_meta"],
            mean_mode=fs["mean_mode_true"],
            band_names=band_names,
        )
        band_vals.append(b)

    fourier_band_nrmse_out = {
        name: float(np.mean([bv[name] for bv in band_vals])) for name in band_names
    }

    # 6) curve（可选保存，用于 per-(p,σ) 的解释图）
    fourier_curve_out = None
    if fs["save_curve"]:
        fourier_curve_out = {
            "k_centers": k_centers_ref.tolist(),
            "E_true_k": Et_sum.tolist(),
            "E_err_k": Ee_sum.tolist(),
            "nrmse_k": nrmse_k_mean.tolist(),
            "k_edges": interior_edges,
            "band_names": band_names,
            "k_star": None if k_star_out is None else float(k_star_out),
            "k_star_threshold": float(fs["kstar_thr"]),
        }

    # 7) meta_hint：用于 result["meta"] 全局定义图（你 Batch 2 的 fig_energy_spectrum 依赖这个）
    meta_hint = {
        "fourier_k_centers": k_centers_ref.tolist(),
        "fourier_energy_k": Et_sum.tolist(),      # 用 “true energy spectrum” 做定义图
        "fourier_k_edges": interior_edges,
        "fourier_grid_meta": dict(fs.get("grid_meta", {}) or {}),
    }

    return fourier_band_nrmse_out, (None if k_star_out is None else float(k_star_out)), fourier_curve_out, meta_hint

def run_linear_baseline_experiment(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    在全数据集上，对一组 (mask_rate, noise_sigma) 组合运行线性基线重建。

    返回
    ----
    result:
        {
            "model_type": "linear",
            "mask_rates": [...],
            "noise_sigmas": [...],
            "meta": {... POD 与数据集信息 ...},
            "entries": [... 原有统计 ...],

            # 旧字段：保留一个代表样本（最小 p / σ，t=0），兼容旧绘图逻辑
            "example_recon": { ... } | None,

            # 新字段：用于多组四联图绘制
            "examples": [...],

            # 新字段：每个 mask_rate 对应的空间掩膜（为画采样点服务）
            "mask_hw_map": {...},
        }
    """
    if verbose:
        print("=== [eval-linear] Start linear baseline experiment ===")

    # 1) POD 基底
    Ur, mean_flat, meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]
    r_eff = min(pod_cfg.r, r_used)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}, r_eff={r_eff}")

    # 1.1 载入 POD 辅助信息（特征值 + φ 分组）
    eigenvalues, phi_groups = _load_pod_aux_info(
        pod_cfg=pod_cfg,
        r_eff=r_eff,
        verbose=verbose,
    )

    # 1.2 从 eigenvalues 推出能量比例与累计能量（供多尺度能量曲线使用）
    energy_frac = None
    energy_cum = None
    if eigenvalues is not None:
        e = np.asarray(eigenvalues, dtype=float)
        total = float(e.sum())
        if total > 0.0:
            frac = e / total
            energy_frac = frac.tolist()
            energy_cum = np.cumsum(frac).tolist()

    # 2) 全数据 + 真系数
    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)

    # 典型样本选择：取最小 mask_rate / noise_sigma，时间帧 t=0（旧逻辑保留）
    p_ref = float(min(eval_cfg.mask_rates))
    s_ref = float(min(eval_cfg.noise_sigmas))
    t_ref = 0
    example_recon: Dict[str, Any] | None = None

    # 新：为每个 (p,σ) 采样若干帧作为可视化样本，这里最多取 3 帧，均匀分布在时间序列上
    n_example_frames = min(3, T)
    example_t_indices = np.linspace(0, T - 1, num=n_example_frames, dtype=int)
    example_t_set = set(int(t) for t in example_t_indices)

    examples: List[Dict[str, Any]] = []
    mask_hw_map: Dict[str, Any] = {}

    # 3) 遍历 (mask_rate, noise_sigma)
    entries: List[Dict[str, Any]] = []
    fourier_meta_hint: Dict[str, Any] | None = None

    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-linear] mask_rate = {mask_rate:.4f}")

        # 同一 mask_rate 复用一个随机 mask（固定 seed 以保证可复现）
        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        # 记录下该 mask_rate 对应的 mask_hw，后续绘图时用来标采样点
        mask_key = f"{float(mask_rate):.6g}"
        mask_hw_map[mask_key] = mask_hw.astype(bool).tolist()

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")

        Ur_masked = Ur_eff[mask_flat, :]  # [M,r_eff]

        for noise_sigma in eval_cfg.noise_sigmas:
            if verbose:
                print(f"  [eval-linear] noise_sigma = {noise_sigma:.4e}")

            nmse_list: List[float] = []
            nmae_list: List[float] = []
            psnr_list: List[float] = []

            # 收集该组合下所有帧的线性系数，便于后面做 band-wise / per-mode / partial 误差
            A_lin_all = np.empty_like(A_true)  # [T,r_eff]

            for t in range(T):
                x = X_thwc[t]                 # [H,W,C]
                x_flat = x.reshape(-1)        # [D]
                a_true_t = A_true[t]          # [r_eff]  # noqa: F841

                # mask + 噪声
                y = apply_mask_flat(x_flat, mask_flat)  # [M]
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)
                if eval_cfg.centered_pod:
                    y_noisy -= apply_mask_flat(mean_flat, mask_flat)

                # 线性最小二乘在 POD 空间求系数
                a_lin = solve_pod_coeffs_least_squares(y_noisy, Ur_masked)  # [r_eff]
                A_lin_all[t] = a_lin

                # 重建到物理空间
                x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
                x_lin = x_lin_flat.reshape(H, W, C)

                nmse_list.append(nmse(x_lin, x))
                nmae_list.append(nmae(x_lin, x))
                psnr_list.append(psnr(x_lin, x))

                # 旧逻辑：记录一个典型样本，用于后续全实验可视化（单个例子）
                if (
                    example_recon is None
                    and float(mask_rate) == p_ref
                    and float(noise_sigma) == s_ref
                    and t == t_ref
                ):
                    x_interp_ref = _compute_interp_baseline(x, mask_hw)
                    example_recon = {
                        "frame_idx": int(t),
                        "mask_rate": float(mask_rate),
                        "noise_sigma": float(noise_sigma),
                        "x_true": x.tolist(),
                        "x_lin": x_lin.tolist(),
                        "x_interp": x_interp_ref.tolist(),
                    }

                # 新逻辑：为每个 (p,σ) 挑选若干帧作为四联图样本
                if int(t) in example_t_set:
                    x_interp = _compute_interp_baseline(x, mask_hw)
                    examples.append(
                        {
                            "frame_idx": int(t),
                            "mask_rate": float(mask_rate),
                            "noise_sigma": float(noise_sigma),
                            "x_true": x.tolist(),
                            "x_lin": x_lin.tolist(),
                            "x_interp": x_interp.tolist(),
                        }
                    )

            nmse_arr = np.array(nmse_list)
            nmae_arr = np.array(nmae_list)
            psnr_arr = np.array(psnr_list)

            # === 3.1 POD 系数多尺度误差 ===

            # (a) band-wise 系数 RMSE（沿用旧接口）
            band_errors = compute_pod_band_errors(
                a_hat=A_lin_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )
            band_errors = {k: float(v) for k, v in band_errors.items()}

            # (b) band-wise 系数 NRMSE（相对误差）
            band_nrmse = nrmse_per_band(
                a_hat=A_lin_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )
            band_nrmse = {k: float(v) for k, v in band_nrmse.items()}

            # (c) per-mode RMSE / NRMSE 谱线
            coeff_rmse = rmse_per_mode(A_lin_all, A_true)                      # [r_eff]
            coeff_nrmse = nrmse_per_mode(A_lin_all, A_true, eigenvalues=eigenvalues)  # [r_eff]

            # === 3.2 基于 φ 分组的场级 partial 重建 NMSE ===

            partial_info = partial_recon_nmse(
                a_hat=A_lin_all,
                a_true=A_true,
                Ur=Ur_eff,
                groups=phi_groups,
                mean_flat=mean_flat,
                sample_indices=None,     # 使用全部时间帧
                reduction="mean",        # 每个组 / 累积一个标量
            )

            field_nmse_per_group = {
                name: float(val)
                for name, val in partial_info["group_nmse"].items()
            }
            field_nmse_partial = {
                name: float(val)
                for name, val in partial_info["cumulative_nmse"].items()
            }

            # === 3.3 从 band_errors 推断“有效模态等级”（保持原有行为） ===

            effective_band = None
            effective_r_cut = None
            if band_errors and eval_cfg.pod_bands:
                # 按 band 起始索引排序，确保从低频到高频
                band_items = sorted(
                    eval_cfg.pod_bands.items(),
                    key=lambda kv: kv[1][0],
                )
                names = [name for name, _ in band_items]
                errs = []
                for name in names:
                    v = band_errors.get(name, float("nan"))
                    errs.append(v)

                errs = np.asarray(errs, dtype=float)
                # 若全是 NaN，就放弃推断
                if np.isfinite(errs).any():
                    # 用一个简单的“跳变阈值”来判断从哪一层开始误差显著增大
                    jump_ratio = 3.0
                    eff_idx = len(names) - 1
                    for i in range(len(names) - 1):
                        if not np.isfinite(errs[i]) or not np.isfinite(errs[i + 1]):
                            continue
                        if errs[i + 1] > jump_ratio * errs[i]:
                            eff_idx = i
                            break

                    effective_band = names[eff_idx]
                    effective_r_cut = int(eval_cfg.pod_bands[effective_band][1])

            # ===== Fourier multiscale (refactored) =====
            fourier_band_nrmse_out = None
            k_star_out = None
            fourier_curve_out = None

            fb, ks, fc, meta_hint = _compute_fourier_for_setting(
                X_thwc=X_thwc,
                A_hat_all=A_lin_all,
                Ur_eff=Ur_eff,
                mean_flat=mean_flat,
                eval_cfg=eval_cfg,
                H=H,
                W=W,
            )
            fourier_band_nrmse_out = fb
            k_star_out = ks
            fourier_curve_out = fc

            # 把 “L/M/H 是什么” 的全局定义信息写一次进 meta（用于后续统一画 E(k)+edges）
            if fourier_meta_hint is None and meta_hint is not None:
                fourier_meta_hint = meta_hint

            entry = {
                "mask_rate": float(mask_rate),
                "noise_sigma": float(noise_sigma),

                # 全场误差统计
                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),

                # POD 系数级误差
                "band_errors": band_errors,                         # RMSE
                "band_nrmse": band_nrmse,                           # NRMSE
                "coeff_rmse_per_mode": coeff_rmse.tolist(),         # [r_eff]
                "coeff_nrmse_per_mode": coeff_nrmse.tolist(),       # [r_eff]

                # 场级多尺度误差（φ 分组）
                "field_nmse_per_group": field_nmse_per_group,       # S1, S2, ...
                "field_nmse_partial": field_nmse_partial,           # S1, S1+S2, ...

                # 有效模态等级
                "effective_band": effective_band,
                "effective_r_cut": effective_r_cut,

                "n_frames": int(T),
                "n_obs": int(n_obs),

                # 傅里叶多尺度等级
                "fourier_band_nrmse": fourier_band_nrmse_out,
                "k_star": None if k_star_out is None else float(k_star_out),
                "fourier_curve": fourier_curve_out,
            }
            entries.append(entry)

            if verbose:
                print(
                    f"    -> NMSE(mean±std) = {entry['nmse_mean']:.4e} ± {entry['nmse_std']:.4e}, "
                    f"NMAE = {entry['nmae_mean']:.4e}, PSNR = {entry['psnr_mean']:.2f} dB, "
                    f"effective_band={effective_band}, r_cut={effective_r_cut}"
                )

    result: Dict[str, Any] = {
        "model_type": "linear",
        "mask_rates": list(eval_cfg.mask_rates),
        "noise_sigmas": list(eval_cfg.noise_sigmas),
        "meta": {
            "H": H,
            "W": W,
            "C": C,
            "T": T,
            "r_eff": r_eff,
            "pod_bands": eval_cfg.pod_bands,
            "center": meta.get("center", True),
            "energy_frac": energy_frac,
            "energy_cum": energy_cum,
            "fourier_k_centers": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_centers"),
            "fourier_energy_k": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_energy_k"),
            "fourier_k_edges": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_edges"),
            "fourier_grid_meta": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_grid_meta"),
        },
        "entries": entries,
        "example_recon": example_recon,
        "examples": examples,
        "mask_hw_map": mask_hw_map,
    }

    if verbose:
        print("\n=== [eval-linear] Done ===")

    return result

def run_mlp_experiment(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    在全数据集上，对一组 (mask_rate, noise_sigma) 组合运行 MLP 重建。

    新增：
    - "examples": 每个 (mask_rate, noise_sigma) 下若干帧的 x_true / x_mlp / x_interp，
      用于后续四联图绘制；
    - "mask_hw_map": 与线性基线同结构，用于标出采样点。
    """
    if verbose:
        print("=== [eval-mlp] Start MLP experiment ===")

    # 1) POD 基底
    Ur, mean_flat, meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]
    r_eff = min(pod_cfg.r, r_used)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}, r_eff={r_eff}")

    # 1.1 载入 POD 辅助信息（特征值 + φ 分组）
    eigenvalues, phi_groups = _load_pod_aux_info(
        pod_cfg=pod_cfg,
        r_eff=r_eff,
        verbose=verbose,
    )

    # 1.2 从 eigenvalues 推出能量比例与累计能量（供多尺度能量曲线使用）
    energy_frac = None
    energy_cum = None
    if eigenvalues is not None:
        e = np.asarray(eigenvalues, dtype=float)
        total = float(e.sum())
        if total > 0.0:
            frac = e / total
            energy_frac = frac.tolist()
            energy_cum = np.cumsum(frac).tolist()

    # 2) 全数据 + 真系数
    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)
    D = H * W * C
    X_flat_all = X_thwc.reshape(T, D)

    # 典型样本选择：取最小 mask_rate / noise_sigma，时间帧 t=0（兼容旧逻辑）
    p_ref = float(min(eval_cfg.mask_rates))
    s_ref = float(min(eval_cfg.noise_sigmas))
    t_ref = 0
    example_recon: Dict[str, Any] | None = None

    # 新：统一为每个 (p,σ) 选若干帧作为可视化样本
    n_example_frames = min(3, T)
    example_t_indices = np.linspace(0, T - 1, num=n_example_frames, dtype=int)
    example_t_set = set(int(t) for t in example_t_indices)

    examples: List[Dict[str, Any]] = []
    mask_hw_map: Dict[str, Any] = {}

    entries: List[Dict[str, Any]] = []
    fourier_meta_hint: Dict[str, Any] | None = None

    # 3) 对每个 mask_rate：训练一个 MLP
    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-mlp] mask_rate(train/test) = {mask_rate:.4f}")

        # 3.1 构造观测掩码（训练和测试共用）
        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        # 记录 mask_hw
        mask_key = f"{float(mask_rate):.6g}"
        mask_hw_map[mask_key] = mask_hw.astype(bool).tolist()

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")

            print(
                f"  -> Training MLP with train_noise_sigma={train_cfg.noise_sigma:.4e}, "
                f"batch_size={train_cfg.batch_size}, max_epochs={train_cfg.max_epochs}, "
                f"hidden_dims={train_cfg.hidden_dims}"
            )

        # 3.2 训练 MLP（使用 train_cfg.noise_sigma 作为训练噪声）
        model_mlp, train_info = train_mlp_on_observations(
            X_flat_all=X_flat_all,
            Ur_eff=Ur_eff,
            mean_flat=mean_flat,
            mask_flat=mask_flat,
            noise_sigma=train_cfg.noise_sigma,
            batch_size=train_cfg.batch_size,
            num_epochs=train_cfg.max_epochs,
            lr=train_cfg.lr,
            verbose=verbose,
        )

        # 4) 在不同 noise_sigma 上评估
        import torch

        model_mlp.eval()
        device = next(model_mlp.parameters()).device

        for noise_sigma in eval_cfg.noise_sigmas:
            if verbose:
                print(f"  [eval-mlp] noise_sigma(test) = {noise_sigma:.4e}")

            nmse_list: List[float] = []
            nmae_list: List[float] = []
            psnr_list: List[float] = []

            A_mlp_all = np.empty_like(A_true)  # [T,r_eff]

            for t in range(T):
                x = X_thwc[t]            # [H,W,C]
                x_flat = x.reshape(-1)   # [D]
                a_true_t = A_true[t]     # [r_eff]  # noqa: F841

                # 观测 + 测试噪声
                y = apply_mask_flat(x_flat, mask_flat)  # [M]
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)
                if eval_cfg.centered_pod:
                    y_noisy -= apply_mask_flat(mean_flat, mask_flat)

                # MLP 预测系数
                y_tensor = torch.from_numpy(y_noisy.astype(np.float32)).to(device)
                with torch.no_grad():
                    a_pred_t = model_mlp(y_tensor[None, :])[0].cpu().numpy()  # [r_eff]

                A_mlp_all[t] = a_pred_t

                # 重建到物理空间
                x_mlp_flat = reconstruct_from_pod(a_pred_t, Ur_eff, mean_flat)
                x_mlp = x_mlp_flat.reshape(H, W, C)

                nmse_list.append(nmse(x_mlp, x))
                nmae_list.append(nmae(x_mlp, x))
                psnr_list.append(psnr(x_mlp, x))

                # 旧单例 example_recon（最小 p/σ，t=0）
                if (
                    example_recon is None
                    and float(mask_rate) == p_ref
                    and float(noise_sigma) == s_ref
                    and t == t_ref
                ):
                    x_interp_ref = _compute_interp_baseline(x, mask_hw)
                    example_recon = {
                        "frame_idx": int(t),
                        "mask_rate": float(mask_rate),
                        "noise_sigma": float(noise_sigma),
                        "x_true": x.tolist(),
                        "x_mlp": x_mlp.tolist(),
                        "x_interp": x_interp_ref.tolist(),
                    }

                # 新：多帧样本
                if int(t) in example_t_set:
                    x_interp = _compute_interp_baseline(x, mask_hw)
                    examples.append(
                        {
                            "frame_idx": int(t),
                            "mask_rate": float(mask_rate),
                            "noise_sigma": float(noise_sigma),
                            "x_true": x.tolist(),
                            "x_mlp": x_mlp.tolist(),
                            "x_interp": x_interp.tolist(),
                        }
                    )

            nmse_arr = np.array(nmse_list)
            nmae_arr = np.array(nmae_list)
            psnr_arr = np.array(psnr_list)

            # === 4.1 POD 系数多尺度误差 ===

            # (a) band-wise 系数 RMSE（与 linear 保持一致）
            band_errors = compute_pod_band_errors(
                a_hat=A_mlp_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )
            band_errors = {k: float(v) for k, v in band_errors.items()}

            # (b) band-wise 系数 NRMSE
            band_nrmse = nrmse_per_band(
                a_hat=A_mlp_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )
            band_nrmse = {k: float(v) for k, v in band_nrmse.items()}

            # (c) per-mode RMSE / NRMSE 谱线
            coeff_rmse = rmse_per_mode(A_mlp_all, A_true)                          # [r_eff]
            coeff_nrmse = nrmse_per_mode(A_mlp_all, A_true, eigenvalues=eigenvalues)  # [r_eff]

            # === 4.2 基于 φ 分组的场级 partial 重建 NMSE ===

            partial_info = partial_recon_nmse(
                a_hat=A_mlp_all,
                a_true=A_true,
                Ur=Ur_eff,
                groups=phi_groups,
                mean_flat=mean_flat,
                sample_indices=None,     # 使用全部时间帧
                reduction="mean",        # 每个组 / 累积一个标量
            )

            field_nmse_per_group = {
                name: float(val)
                for name, val in partial_info["group_nmse"].items()
            }
            field_nmse_partial = {
                name: float(val)
                for name, val in partial_info["cumulative_nmse"].items()
            }

            # === 4.3 从 band_errors 推断“有效模态等级”（与 linear 同逻辑） ===

            effective_band = None
            effective_r_cut = None
            if band_errors and eval_cfg.pod_bands:
                # 按 band 起始索引排序，确保从低频到高频
                band_items = sorted(
                    eval_cfg.pod_bands.items(),
                    key=lambda kv: kv[1][0],
                )
                names = [name for name, _ in band_items]
                errs = []
                for name in names:
                    v = band_errors.get(name, float("nan"))
                    errs.append(v)

                errs = np.asarray(errs, dtype=float)
                if np.isfinite(errs).any():
                    jump_ratio = 3.0
                    eff_idx = len(names) - 1
                    for i in range(len(names) - 1):
                        if not np.isfinite(errs[i]) or not np.isfinite(errs[i + 1]):
                            continue
                        if errs[i + 1] > jump_ratio * errs[i]:
                            eff_idx = i
                            break

                    effective_band = names[eff_idx]
                    effective_r_cut = int(eval_cfg.pod_bands[effective_band][1])

            # ===== Fourier multiscale (refactored) =====
            fourier_band_nrmse_out = None
            k_star_out = None
            fourier_curve_out = None

            fb, ks, fc, meta_hint = _compute_fourier_for_setting(
                X_thwc=X_thwc,
                A_hat_all=A_mlp_all,
                Ur_eff=Ur_eff,
                mean_flat=mean_flat,
                eval_cfg=eval_cfg,
                H=H,
                W=W,
            )
            fourier_band_nrmse_out = fb
            k_star_out = ks
            fourier_curve_out = fc

            if fourier_meta_hint is None and meta_hint is not None:
                fourier_meta_hint = meta_hint

            entry = {
                "mask_rate": float(mask_rate),
                "noise_sigma": float(noise_sigma),

                # 全场误差统计
                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),

                # POD 系数级误差
                "band_errors": band_errors,                         # RMSE
                "band_nrmse": band_nrmse,                           # NRMSE
                "coeff_rmse_per_mode": coeff_rmse.tolist(),         # [r_eff]
                "coeff_nrmse_per_mode": coeff_nrmse.tolist(),       # [r_eff]

                # 场级多尺度误差（φ 分组）
                "field_nmse_per_group": field_nmse_per_group,       # S1, S2, ...
                "field_nmse_partial": field_nmse_partial,           # S1, S1+S2, ...

                # 有效模态等级
                "effective_band": effective_band,
                "effective_r_cut": effective_r_cut,

                # 数据规模 / 观测数
                "n_frames": int(T),
                "n_obs": int(n_obs),

                # 傅里叶多尺度等级
                "fourier_band_nrmse": fourier_band_nrmse_out,
                "k_star": None if k_star_out is None else float(k_star_out),
                "fourier_curve": fourier_curve_out,

                # 训练信息（保持原来的结构）
                "train_info": train_info,
            }
            entries.append(entry)

            if verbose:
                print(
                    f"    -> NMSE(mean±std) = {entry['nmse_mean']:.4e} ± {entry['nmse_std']:.4e}, "
                    f"NMAE = {entry['nmae_mean']:.4e}, PSNR = {entry['psnr_mean']:.2f} dB, "
                    f"effective_band={effective_band}, r_cut={effective_r_cut}"
                )

    # 5) 汇总结果
    result: Dict[str, Any] = {
        "model_type": "mlp",
        "mask_rates": list(eval_cfg.mask_rates),
        "noise_sigmas": list(eval_cfg.noise_sigmas),
        "meta": {
            "H": H,
            "W": W,
            "C": C,
            "T": T,
            "r_eff": r_eff,
            "pod_bands": eval_cfg.pod_bands,
            "center": meta.get("center", True),
            "energy_frac": energy_frac,
            "energy_cum": energy_cum,
            "train_cfg": {
                "mask_rate": train_cfg.mask_rate,
                "noise_sigma": train_cfg.noise_sigma,
                "hidden_dims": train_cfg.hidden_dims,
                "lr": train_cfg.lr,
                "batch_size": train_cfg.batch_size,
                "max_epochs": train_cfg.max_epochs,
                "device": train_cfg.device,
            },
            "fourier_k_centers": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_centers"),
            "fourier_energy_k": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_energy_k"),
            "fourier_k_edges": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_edges"),
            "fourier_grid_meta": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_grid_meta"),
        },
        "entries": entries,
        "example_recon": example_recon,
        "examples": examples,
        "mask_hw_map": mask_hw_map,
    }

    if verbose:
        print("\n=== [eval-mlp] Done ===")

    return result
