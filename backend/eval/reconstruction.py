# backend/eval/reconstruction.py

"""
在线性基线与 MLP 之间进行对比评估。

负责：
- 加载 snapshot
- 对每个 mask_rate / noise_sigma 组合执行重建
- 计算全场与多尺度误差
"""

from typing import Dict, Any, List, Tuple

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
            "entries": [
                {
                    "mask_rate": float,
                    "noise_sigma": float,
                    "nmse_mean": float,
                    "nmse_std": float,
                    "nmae_mean": float,
                    "nmae_std": float,
                    "psnr_mean": float,
                    "psnr_std": float,
                    "band_errors": { band_name: float, ... }  # 系数 RMSE
                },
                ...
            ]
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

    # 2) 全数据 + 真系数
    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)

    # 3) 遍历 (mask_rate, noise_sigma)
    entries: List[Dict[str, Any]] = []

    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-linear] mask_rate = {mask_rate:.4f}")

        # 同一 mask_rate 复用一个随机 mask（固定 seed 以保证可复现）
        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")

        Ur_masked = Ur_eff[mask_flat, :]  # [M,r_eff]

        for noise_sigma in eval_cfg.noise_sigmas:
            if verbose:
                print(f"  [eval-linear] noise_sigma = {noise_sigma:.4e}")

            nmse_list: List[float] = []
            nmae_list: List[float] = []
            psnr_list: List[float] = []

            # 收集该组合下所有帧的线性系数，便于后面做 band-wise 误差
            A_lin_all = np.empty_like(A_true)  # [T,r_eff]

            for t in range(T):
                x = X_thwc[t]                 # [H,W,C]
                x_flat = x.reshape(-1)        # [D]
                a_true_t = A_true[t]          # [r_eff]

                # mask + 噪声
                y = apply_mask_flat(x_flat, mask_flat)  # [M]
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)

                # 线性最小二乘在 POD 空间求系数
                a_lin = solve_pod_coeffs_least_squares(y_noisy, Ur_masked)  # [r_eff]
                A_lin_all[t] = a_lin

                # 重建到物理空间
                x_lin_flat = reconstruct_from_pod(a_lin, Ur_eff, mean_flat)
                x_lin = x_lin_flat.reshape(H, W, C)

                nmse_list.append(nmse(x_lin, x))
                nmae_list.append(nmae(x_lin, x))
                psnr_list.append(psnr(x_lin, x))

            nmse_arr = np.array(nmse_list)
            nmae_arr = np.array(nmae_list)
            psnr_arr = np.array(psnr_list)

            # POD 系数 band-wise 误差（系数 RMSE）
            band_errors = compute_pod_band_errors(
                a_hat=A_lin_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )

            entry = {
                "mask_rate": float(mask_rate),
                "noise_sigma": float(noise_sigma),
                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),
                "band_errors": {k: float(v) for k, v in band_errors.items()},
                "n_frames": int(T),
                "n_obs": int(n_obs),
            }
            entries.append(entry)

            if verbose:
                print(
                    f"    -> NMSE(mean±std) = {entry['nmse_mean']:.4e} ± {entry['nmse_std']:.4e}, "
                    f"NMAE = {entry['nmae_mean']:.4e}, PSNR = {entry['psnr_mean']:.2f} dB"
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
        },
        "entries": entries,
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

    设计选择：
    - 对于每个 mask_rate，训练一个对应的 MLP（训练噪声强度由 train_cfg.noise_sigma 决定）；
    - 对该 mask_rate 下的多个 noise_sigma（测试噪声）进行评估；
    - 每个组合输出全场误差 + POD band 系数 RMSE。

    返回结构与 run_linear_baseline_experiment 相同，只是 model_type="mlp"，
    meta 中会额外记录部分训练超参数。
    """
    if verbose:
        print("=== [eval-mlp] Start MLP baseline experiment ===")

    # 1) POD 基底
    Ur, mean_flat, meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=verbose)
    H, W, C = meta["H"], meta["W"], meta["C"]
    T = meta["T"]
    r_used = meta["r_used"]
    r_eff = min(pod_cfg.r, r_used)
    Ur_eff = Ur[:, :r_eff]

    if verbose:
        print(f"  - meta: T={T}, H={H}, W={W}, C={C}, r_used={r_used}, r_eff={r_eff}")

    # 2) 全数据 + 真系数
    X_thwc, A_true = _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff, verbose=verbose)
    D = H * W * C
    X_flat_all = X_thwc.reshape(T, D)

    entries: List[Dict[str, Any]] = []

    for mask_rate in eval_cfg.mask_rates:
        if verbose:
            print(f"\n[eval-mlp] mask_rate = {mask_rate:.4f}")

        # 固定 mask（与 linear experiment 保持同样风格：每个 mask_rate 一张掩膜）
        mask_hw = generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=0)
        mask_flat = flatten_mask(mask_hw, C=C)
        n_obs = int(mask_flat.sum())

        if verbose:
            print(f"  -> total observed entries (with {C} channels) = {n_obs}")
            print(
                f"  -> Training MLP with train_noise_sigma={train_cfg.noise_sigma:.4e}, "
                f"batch_size={train_cfg.batch_size}, max_epochs={train_cfg.max_epochs}, "
                f"hidden_dims={train_cfg.hidden_dims}"
            )

        # 3) 训练 MLP（使用 train_cfg.noise_sigma 作为训练噪声）
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
                x = X_thwc[t]
                x_flat = x.reshape(-1)
                a_true_t = A_true[t]

                # 观测 + 测试噪声
                y = apply_mask_flat(x_flat, mask_flat)  # [M]
                y_noisy = add_gaussian_noise(y, sigma=noise_sigma)

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

            nmse_arr = np.array(nmse_list)
            nmae_arr = np.array(nmae_list)
            psnr_arr = np.array(psnr_list)

            band_errors = compute_pod_band_errors(
                a_hat=A_mlp_all,
                a_true=A_true,
                bands=eval_cfg.pod_bands,
            )

            entry = {
                "mask_rate": float(mask_rate),
                "noise_sigma": float(noise_sigma),
                "nmse_mean": float(nmse_arr.mean()),
                "nmse_std": float(nmse_arr.std()),
                "nmae_mean": float(nmae_arr.mean()),
                "nmae_std": float(nmae_arr.std()),
                "psnr_mean": float(psnr_arr.mean()),
                "psnr_std": float(psnr_arr.std()),
                "band_errors": {k: float(v) for k, v in band_errors.items()},
                "n_frames": int(T),
                "n_obs": int(n_obs),
                "train_info": train_info,  # 可以方便你调参，后续也可以去掉
            }
            entries.append(entry)

            if verbose:
                print(
                    f"    -> NMSE(mean±std) = {entry['nmse_mean']:.4e} ± {entry['nmse_std']:.4e}, "
                    f"NMAE = {entry['nmae_mean']:.4e}, PSNR = {entry['psnr_mean']:.2f} dB"
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
            "train_cfg": {
                "mask_rate": train_cfg.mask_rate,
                "noise_sigma": train_cfg.noise_sigma,
                "hidden_dims": train_cfg.hidden_dims,
                "lr": train_cfg.lr,
                "batch_size": train_cfg.batch_size,
                "max_epochs": train_cfg.max_epochs,
                "device": train_cfg.device,
            },
        },
        "entries": entries,
    }

    if verbose:
        print("\n=== [eval-mlp] Done ===")

    return result
