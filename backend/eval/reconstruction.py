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
    """
    NEW SCHEMA ONLY:
    Read Fourier grid meta from eval_cfg.fourier.grid_meta.

    - If dx/dy missing but Lx/Ly present: infer dx=Lx/W, dy=Ly/H
    - If Lx/Ly missing but dx/dy present: infer Lx=dx*W, Ly=dy*H
    - Ensure 'angular' exists (default False)
    """
    f = getattr(eval_cfg, "fourier", None)
    grid = dict(getattr(f, "grid_meta", {}) or {}) if f is not None else {}

    # defaults
    grid.setdefault("angular", False)

    # infer dx/dy from Lx/Ly
    if ("dx" not in grid or "dy" not in grid) and ("Lx" in grid and "Ly" in grid):
        Lx = float(grid["Lx"])
        Ly = float(grid["Ly"])
        if W > 0 and H > 0:
            grid.setdefault("dx", Lx / float(W))
            grid.setdefault("dy", Ly / float(H))

    # infer Lx/Ly from dx/dy
    if ("Lx" not in grid or "Ly" not in grid) and ("dx" in grid and "dy" in grid):
        dx = float(grid["dx"])
        dy = float(grid["dy"])
        grid.setdefault("Lx", dx * float(W))
        grid.setdefault("Ly", dy * float(H))

    return grid

def _get_fourier_settings(eval_cfg: EvalConfig, *, H: int, W: int) -> dict:
    """
    NEW SCHEMA ONLY.
    Collect Fourier-related settings from eval_cfg.fourier with strict defaults.
    No legacy flat fields are supported.

    Returns a dict used by _compute_fourier_for_setting.
    """
    f = getattr(eval_cfg, "fourier", None)
    if f is None:
        return {
            "enabled": False,
        }

    grid_meta = _infer_fourier_grid_meta(eval_cfg, H=H, W=W)

    enabled = bool(getattr(f, "enabled", False))
    band_scheme = str(getattr(f, "band_scheme", "physical")).strip().lower()
    band_names = tuple(getattr(f, "band_names", ("L", "M", "H")))

    lambda_edges = getattr(f, "lambda_edges", None)
    if lambda_edges is not None:
        lambda_edges = [float(v) for v in lambda_edges]

    return {
        "enabled": enabled,
        "grid_meta": grid_meta,

        "num_bins": int(getattr(f, "num_bins", 64)),
        "sample_frames": int(getattr(f, "sample_frames", 8)),

        "kstar_thr": float(getattr(f, "kstar_threshold", 1.0)),
        "mean_mode_true": str(getattr(f, "mean_mode_true", "global")),
        "save_curve": bool(getattr(f, "save_curve", False)),

        "band_scheme": band_scheme,
        "band_names": band_names,
        "lambda_edges": lambda_edges,
    }

def _compute_fourier_for_setting(
    *,
    X_thwc: np.ndarray,             # [T,H,W,C]
    A_hat_all: np.ndarray,          # [T,r_eff]
    Ur_eff: np.ndarray,             # [D,r_eff]
    mean_flat: np.ndarray,          # [D]
    eval_cfg: EvalConfig,
    H: int,
    W: int,
) -> tuple[dict | None, float | None, dict | None, dict | None]:
    """
    NEW SCHEMA ONLY.

    Returns:
      fourier_band_nrmse_out: Dict[str,float] | None
      k_star_out: float | None
      fourier_curve_out: Dict[str,Any] | None
      meta_hint: Dict[str,Any] | None   (用于 result["meta"] 的三件套填充)
    """
    fs = _get_fourier_settings(eval_cfg, H=H, W=W)
    if not fs.get("enabled", False):
        return None, None, None, None

    T = int(X_thwc.shape[0])
    t_samples = _select_sample_frames(T, fs["sample_frames"])
    if len(t_samples) == 0:
        return None, None, None, None

    Et_sum = None
    Ee_sum = None
    k_centers_ref = None

    # 1) accumulate radial energy (true & error) across sampled frames
    for t_s in t_samples:
        x_true = X_thwc[t_s]  # [H,W,C]
        a_hat = A_hat_all[t_s]
        x_hat = reconstruct_from_pod(a_hat, Ur_eff, mean_flat).reshape(H, W, -1)

        k_centers, Et, Ee, _nrmse_k = fourier_radial_nrmse_curve(
            x_hat=x_hat,
            x_true=x_true,
            num_bins=fs["num_bins"],
            k_max=None,                    # NEW schema does not expose k_max
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

    # 2) NRMSE(k) and k*
    nrmse_k_mean = np.sqrt(Ee_sum / (Et_sum + 1e-12))
    k_star_out = kstar_from_radial_curve(
        k_centers_ref,
        nrmse_k_mean,
        threshold=float(fs["kstar_thr"]),
    )

    # 3) decide band edges
    band_scheme = fs["band_scheme"]
    band_names = list(fs["band_names"])

    if band_scheme == "physical":
        lam = fs.get("lambda_edges", None)
        if not lam:
            raise ValueError("fourier.band_scheme='physical' requires fourier.lambda_edges in YAML.")
        # interior edges in k-space
        interior_edges = sorted([1.0 / float(v) for v in lam if float(v) > 0.0])
        if len(interior_edges) == 0:
            raise ValueError("fourier.lambda_edges must contain positive values.")
        kN = float(np.max(k_centers_ref))
        full_edges = [0.0] + interior_edges + [kN]

    elif band_scheme == "energy_quantile":
        # default quantiles (can be exposed later in FourierConfig if needed)
        full_edges = auto_pick_k_edges_from_energy(
            k_centers_ref,
            Et_sum,
            quantiles=(0.80, 0.95),
        )
        interior_edges = [float(v) for v in full_edges[1:-1]]

    else:
        raise ValueError(f"Unsupported fourier.band_scheme='{band_scheme}'. Use 'physical' or 'energy_quantile'.")

    # 4) ensure band_names match number of bands
    n_bands = max(0, len(full_edges) - 1)
    if len(band_names) != n_bands:
        # strict but safe: auto-generate to avoid index errors
        band_names = [f"B{i+1}" for i in range(n_bands)]

    # 5) band nrmse
    band_vals = []
    for t_s in t_samples:
        x_true = X_thwc[t_s]
        a_hat = A_hat_all[t_s]
        x_hat = reconstruct_from_pod(a_hat, Ur_eff, mean_flat).reshape(H, W, -1)

        # compute band nrmse for this frame
        k_centers, Et, Ee, _nrmse_k = fourier_radial_nrmse_curve(
            x_hat=x_hat,
            x_true=x_true,
            num_bins=fs["num_bins"],
            k_max=None,
            grid_meta=fs["grid_meta"],
            mean_mode=fs["mean_mode_true"],
        )

        band_nrmse_t = fourier_band_nrmse(
            k_centers=k_centers,
            E_true_k=Et,
            E_err_k=Ee,
            full_edges=full_edges,
            band_names=band_names,
            monotone_enforce=True,  # NEW schema: fixed default
        )
        band_vals.append(band_nrmse_t)

    fourier_band_nrmse_out = {name: float(np.mean([bv[name] for bv in band_vals])) for name in band_names}

    # 6) optional curve (for per-(p,σ) explanation plots)
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

    # 7) meta_hint for build_eval_figures gate and definition plot
    meta_hint = {
        "fourier_k_centers": k_centers_ref.tolist(),
        "fourier_energy_k": Et_sum.tolist(),
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
    新版：仅支持 eval_cfg.fourier.*（嵌套结构）：
      - eval_cfg.fourier.enabled
      - eval_cfg.fourier.band_scheme
      - eval_cfg.fourier.grid_meta
      - eval_cfg.fourier.lambda_edges / band_names / ...
    """
    if verbose:
        print("=== [eval-linear] Start linear baseline experiment ===")

    # ---- Fourier config (NEW ONLY) ----
    fourier_cfg = getattr(eval_cfg, "fourier", None)
    fourier_enabled = bool(getattr(fourier_cfg, "enabled", False)) if fourier_cfg is not None else False
    fourier_grid_meta = dict(getattr(fourier_cfg, "grid_meta", {}) or {}) if fourier_cfg is not None else {}
    fourier_band_scheme = str(getattr(fourier_cfg, "band_scheme", "physical")) if fourier_cfg is not None else "physical"
    fourier_lambda_edges = getattr(fourier_cfg, "lambda_edges", None) if fourier_cfg is not None else None
    fourier_band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H"))) if fourier_cfg is not None else ("L", "M", "H")
    fourier_num_bins = int(getattr(fourier_cfg, "num_bins", 64)) if fourier_cfg is not None else 64
    fourier_sample_frames = int(getattr(fourier_cfg, "sample_frames", 8)) if fourier_cfg is not None else 8
    fourier_kstar_threshold = float(getattr(fourier_cfg, "kstar_threshold", 1.0)) if fourier_cfg is not None else 1.0
    fourier_mean_mode_true = str(getattr(fourier_cfg, "mean_mode_true", "global")) if fourier_cfg is not None else "global"
    fourier_save_curve = bool(getattr(fourier_cfg, "save_curve", False)) if fourier_cfg is not None else False

    if fourier_enabled and verbose:
        print(
            "[eval-linear][fourier] enabled=True, "
            f"scheme={fourier_band_scheme}, bins={fourier_num_bins}, sample_frames={fourier_sample_frames}, "
            f"kstar_thr={fourier_kstar_threshold}, mean_mode={fourier_mean_mode_true}, "
            f"band_names={fourier_band_names}, lambda_edges={fourier_lambda_edges}"
        )
        if not fourier_grid_meta:
            print("[eval-linear][fourier][WARN] grid_meta is empty. Your physical scaling may be wrong.")
        else:
            # 关键字段不强制报错（避免打断），但会提醒
            for k in ("dx", "dy", "angular"):
                if k not in fourier_grid_meta:
                    print(f"[eval-linear][fourier][WARN] grid_meta missing '{k}'.")

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

    # 1.2 从 eigenvalues 推出能量比例与累计能量
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

    # 典型样本（旧逻辑保留）
    p_ref = float(min(eval_cfg.mask_rates))
    s_ref = float(min(eval_cfg.noise_sigmas))
    t_ref = 0
    example_recon: Dict[str, Any] | None = None

    # 新：每个 (p,σ) 选若干帧作为可视化样本
    n_example_frames = min(3, T)
    example_t_indices = np.linspace(0, T - 1, num=n_example_frames, dtype=int)
    example_t_set = set(int(t) for t in example_t_indices)

    examples: List[Dict[str, Any]] = []
    mask_hw_map: Dict[str, Any] = {}

    entries: List[Dict[str, Any]] = []
    fourier_meta_hint: Dict[str, Any] | None = None

    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-linear] mask_rate = {mask_rate:.4f}")

        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        mask_key = f"{float(mask_rate):.6g}"
        mask_hw_map[mask_key] = mask_hw.astype(bool).tolist()

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")

        Ur_masked = Ur_eff[mask_flat, :]

        for noise_sigma in eval_cfg.noise_sigmas:
            if verbose:
                print(f"  [eval-linear] noise_sigma = {noise_sigma:.4e}")

            nmse_list: List[float] = []
            nmae_list: List[float] = []
            psnr_list: List[float] = []

            A_lin_all = np.empty_like(A_true)

            for t in range(T):
                x = X_thwc[t]
                x_flat = x.reshape(-1)

                y = apply_mask_flat(x_flat, mask_flat)
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)
                if eval_cfg.centered_pod:
                    y_noisy -= apply_mask_flat(mean_flat, mask_flat)

                a_lin = solve_pod_coeffs_least_squares(y_noisy, Ur_masked)
                A_lin_all[t] = a_lin

                x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
                x_lin = x_lin_flat.reshape(H, W, C)

                nmse_list.append(nmse(x_lin, x))
                nmae_list.append(nmae(x_lin, x))
                psnr_list.append(psnr(x_lin, x))

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

            band_errors = compute_pod_band_errors(a_hat=A_lin_all, a_true=A_true, bands=eval_cfg.pod_bands)
            band_errors = {k: float(v) for k, v in band_errors.items()}

            band_nrmse = nrmse_per_band(a_hat=A_lin_all, a_true=A_true, bands=eval_cfg.pod_bands)
            band_nrmse = {k: float(v) for k, v in band_nrmse.items()}

            coeff_rmse = rmse_per_mode(A_lin_all, A_true)
            coeff_nrmse = nrmse_per_mode(A_lin_all, A_true, eigenvalues=eigenvalues)

            partial_info = partial_recon_nmse(
                a_hat=A_lin_all,
                a_true=A_true,
                Ur=Ur_eff,
                groups=phi_groups,
                mean_flat=mean_flat,
                sample_indices=None,
                reduction="mean",
            )
            field_nmse_per_group = {name: float(val) for name, val in partial_info["group_nmse"].items()}
            field_nmse_partial = {name: float(val) for name, val in partial_info["cumulative_nmse"].items()}

            effective_band = None
            effective_r_cut = None
            if band_errors and eval_cfg.pod_bands:
                band_items = sorted(eval_cfg.pod_bands.items(), key=lambda kv: kv[1][0])
                names = [name for name, _ in band_items]
                errs = np.asarray([band_errors.get(name, float("nan")) for name in names], dtype=float)
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

            # ===== Fourier multiscale =====
            fourier_band_nrmse_out = None
            k_star_out = None
            fourier_curve_out = None

            if fourier_enabled:
                fb, ks, fc, meta_hint = _compute_fourier_for_setting(
                    X_thwc=X_thwc,
                    A_hat_all=A_lin_all,
                    Ur_eff=Ur_eff,
                    mean_flat=mean_flat,
                    eval_cfg=eval_cfg,  # 下游你后面会继续清理为只读 eval_cfg.fourier
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

                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),

                "band_errors": band_errors,
                "band_nrmse": band_nrmse,
                "coeff_rmse_per_mode": coeff_rmse.tolist(),
                "coeff_nrmse_per_mode": coeff_nrmse.tolist(),

                "field_nmse_per_group": field_nmse_per_group,
                "field_nmse_partial": field_nmse_partial,

                "effective_band": effective_band,
                "effective_r_cut": effective_r_cut,

                "n_frames": int(T),
                "n_obs": int(n_obs),

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

            # 让绘图端能稳定拿到“Fourier 分带定义”
            "fourier_enabled": bool(fourier_enabled),
            "fourier_band_scheme": fourier_band_scheme,
            "fourier_band_names": list(fourier_band_names),
            "fourier_lambda_edges": None if fourier_lambda_edges is None else [float(x) for x in fourier_lambda_edges],
            "fourier_grid_meta": dict(fourier_grid_meta),

            # 供 build_eval_figures gate 的三件套（来自一次 meta_hint）
            "fourier_k_centers": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_centers"),
            "fourier_energy_k": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_energy_k"),
            "fourier_k_edges": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_edges"),
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
    新版：仅支持 eval_cfg.fourier.*（嵌套结构）。
    """
    if verbose:
        print("=== [eval-mlp] Start MLP experiment ===")

    # ---- Fourier config (NEW ONLY) ----
    fourier_cfg = getattr(eval_cfg, "fourier", None)
    fourier_enabled = bool(getattr(fourier_cfg, "enabled", False)) if fourier_cfg is not None else False
    fourier_grid_meta = dict(getattr(fourier_cfg, "grid_meta", {}) or {}) if fourier_cfg is not None else {}
    fourier_band_scheme = str(getattr(fourier_cfg, "band_scheme", "physical")) if fourier_cfg is not None else "physical"
    fourier_lambda_edges = getattr(fourier_cfg, "lambda_edges", None) if fourier_cfg is not None else None
    fourier_band_names = tuple(getattr(fourier_cfg, "band_names", ("L", "M", "H"))) if fourier_cfg is not None else ("L", "M", "H")
    fourier_num_bins = int(getattr(fourier_cfg, "num_bins", 64)) if fourier_cfg is not None else 64
    fourier_sample_frames = int(getattr(fourier_cfg, "sample_frames", 8)) if fourier_cfg is not None else 8
    fourier_kstar_threshold = float(getattr(fourier_cfg, "kstar_threshold", 1.0)) if fourier_cfg is not None else 1.0
    fourier_mean_mode_true = str(getattr(fourier_cfg, "mean_mode_true", "global")) if fourier_cfg is not None else "global"
    fourier_save_curve = bool(getattr(fourier_cfg, "save_curve", False)) if fourier_cfg is not None else False

    if fourier_enabled and verbose:
        print(
            "[eval-mlp][fourier] enabled=True, "
            f"scheme={fourier_band_scheme}, bins={fourier_num_bins}, sample_frames={fourier_sample_frames}, "
            f"kstar_thr={fourier_kstar_threshold}, mean_mode={fourier_mean_mode_true}, "
            f"band_names={fourier_band_names}, lambda_edges={fourier_lambda_edges}"
        )

    # 1) POD 基底
    Ur, mean_flat, meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]
    r_eff = min(pod_cfg.r, r_used)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}, r_eff={r_eff}")

    eigenvalues, phi_groups = _load_pod_aux_info(
        pod_cfg=pod_cfg,
        r_eff=r_eff,
        verbose=verbose,
    )

    energy_frac = None
    energy_cum = None
    if eigenvalues is not None:
        e = np.asarray(eigenvalues, dtype=float)
        total = float(e.sum())
        if total > 0.0:
            frac = e / total
            energy_frac = frac.tolist()
            energy_cum = np.cumsum(frac).tolist()

    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)
    D = H * W * C
    X_flat_all = X_thwc.reshape(T, D)

    p_ref = float(min(eval_cfg.mask_rates))
    s_ref = float(min(eval_cfg.noise_sigmas))
    t_ref = 0
    example_recon: Dict[str, Any] | None = None

    n_example_frames = min(3, T)
    example_t_indices = np.linspace(0, T - 1, num=n_example_frames, dtype=int)
    example_t_set = set(int(t) for t in example_t_indices)

    examples: List[Dict[str, Any]] = []
    mask_hw_map: Dict[str, Any] = {}

    entries: List[Dict[str, Any]] = []
    fourier_meta_hint: Dict[str, Any] | None = None

    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-mlp] mask_rate(train/test) = {mask_rate:.4f}")

        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        mask_key = f"{float(mask_rate):.6g}"
        mask_hw_map[mask_key] = mask_hw.astype(bool).tolist()

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")
            print(
                f"  -> Training MLP with train_noise_sigma={train_cfg.noise_sigma:.4e}, "
                f"batch_size={train_cfg.batch_size}, max_epochs={train_cfg.max_epochs}, "
                f"hidden_dims={train_cfg.hidden_dims}"
            )

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

        import torch

        model_mlp.eval()
        device = next(model_mlp.parameters()).device

        for noise_sigma in eval_cfg.noise_sigmas:
            if verbose:
                print(f"  [eval-mlp] noise_sigma(test) = {noise_sigma:.4e}")

            nmse_list: List[float] = []
            nmae_list: List[float] = []
            psnr_list: List[float] = []

            A_mlp_all = np.empty_like(A_true)

            for t in range(T):
                x = X_thwc[t]
                x_flat = x.reshape(-1)

                y = apply_mask_flat(x_flat, mask_flat)
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)
                if eval_cfg.centered_pod:
                    y_noisy -= apply_mask_flat(mean_flat, mask_flat)

                y_tensor = torch.from_numpy(y_noisy.astype(np.float32)).to(device)
                with torch.no_grad():
                    a_pred_t = model_mlp(y_tensor[None, :])[0].cpu().numpy()

                A_mlp_all[t] = a_pred_t

                x_mlp_flat = reconstruct_from_pod(a_pred_t, Ur_eff, mean_flat)
                x_mlp = x_mlp_flat.reshape(H, W, C)

                nmse_list.append(nmse(x_mlp, x))
                nmae_list.append(nmae(x_mlp, x))
                psnr_list.append(psnr(x_mlp, x))

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

            band_errors = compute_pod_band_errors(a_hat=A_mlp_all, a_true=A_true, bands=eval_cfg.pod_bands)
            band_errors = {k: float(v) for k, v in band_errors.items()}

            band_nrmse = nrmse_per_band(a_hat=A_mlp_all, a_true=A_true, bands=eval_cfg.pod_bands)
            band_nrmse = {k: float(v) for k, v in band_nrmse.items()}

            coeff_rmse = rmse_per_mode(A_mlp_all, A_true)
            coeff_nrmse = nrmse_per_mode(A_mlp_all, A_true, eigenvalues=eigenvalues)

            partial_info = partial_recon_nmse(
                a_hat=A_mlp_all,
                a_true=A_true,
                Ur=Ur_eff,
                groups=phi_groups,
                mean_flat=mean_flat,
                sample_indices=None,
                reduction="mean",
            )
            field_nmse_per_group = {name: float(val) for name, val in partial_info["group_nmse"].items()}
            field_nmse_partial = {name: float(val) for name, val in partial_info["cumulative_nmse"].items()}

            effective_band = None
            effective_r_cut = None
            if band_errors and eval_cfg.pod_bands:
                band_items = sorted(eval_cfg.pod_bands.items(), key=lambda kv: kv[1][0])
                names = [name for name, _ in band_items]
                errs = np.asarray([band_errors.get(name, float("nan")) for name in names], dtype=float)
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

            # ===== Fourier multiscale =====
            fourier_band_nrmse_out = None
            k_star_out = None
            fourier_curve_out = None

            if fourier_enabled:
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

                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),

                "band_errors": band_errors,
                "band_nrmse": band_nrmse,
                "coeff_rmse_per_mode": coeff_rmse.tolist(),
                "coeff_nrmse_per_mode": coeff_nrmse.tolist(),

                "field_nmse_per_group": field_nmse_per_group,
                "field_nmse_partial": field_nmse_partial,

                "effective_band": effective_band,
                "effective_r_cut": effective_r_cut,

                "n_frames": int(T),
                "n_obs": int(n_obs),

                "fourier_band_nrmse": fourier_band_nrmse_out,
                "k_star": None if k_star_out is None else float(k_star_out),
                "fourier_curve": fourier_curve_out,

                "train_info": train_info,
            }
            entries.append(entry)

            if verbose:
                print(
                    f"    -> NMSE(mean±std) = {entry['nmse_mean']:.4e} ± {entry['nmse_std']:.4e}, "
                    f"NMAE = {entry['nmae_mean']:.4e}, PSNR = {entry['psnr_mean']:.2f} dB, "
                    f"effective_band={effective_band}, r_cut={effective_r_cut}"
                )

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

            "fourier_enabled": bool(fourier_enabled),
            "fourier_band_scheme": fourier_band_scheme,
            "fourier_band_names": list(fourier_band_names),
            "fourier_lambda_edges": None if fourier_lambda_edges is None else [float(x) for x in fourier_lambda_edges],
            "fourier_grid_meta": dict(fourier_grid_meta),

            "fourier_k_centers": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_centers"),
            "fourier_energy_k": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_energy_k"),
            "fourier_k_edges": None if fourier_meta_hint is None else fourier_meta_hint.get("fourier_k_edges"),
        },
        "entries": entries,
        "example_recon": example_recon,
        "examples": examples,
        "mask_hw_map": mask_hw_map,
    }

    if verbose:
        print("\n=== [eval-mlp] Done ===")

    return result
