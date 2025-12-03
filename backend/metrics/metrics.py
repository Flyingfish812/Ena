"""
高阶误差指标与多尺度辅助工具。

在 errors.py 提供的标量误差基础上，这里实现：

- rmse_per_mode(a_hat, a_true)
    逐模态 RMSE 谱线。

- nrmse_per_mode(a_hat, a_true, eigenvalues=None)
    逐模态 NRMSE 谱线，可选用 POD 特征值 lambda_k 做归一化。

- rmse_per_band(a_hat, a_true, bands)
    按给定 POD band 区间计算系数 RMSE（与 multiscale.compute_pod_band_errors 一致）。

- nrmse_per_band(a_hat, a_true, bands)
    按 band 计算系数 NRMSE，定义为：
        NRMSE_band = sqrt( sum(e^2) / sum(a_true^2) )

- field_nmse(x_hat, x_true, reduction="mean")
    场空间上的 NMSE，支持 [N,H,W,C] / [N,D] 等，
    reduction="none" 时返回每个样本的 NMSE 向量。

- partial_recon_nmse(...)
    基于 POD 基底 Ur + 模态分组信息，对“单组 / 累积”重建场
    计算 NMSE，用于多尺度有效模态等级分析。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------
# 内部工具：统一成 float64 + 形状检查
# ---------------------------------------------------------------------


def _to_2d_coeff_arrays(
    a_hat: ArrayLike,
    a_true: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将系数数组统一成 float64、形状 [N, r]。

    - 若输入是一维 [r]，则视作单样本 [1, r]。
    - 二维 [N, r] 直接返回。
    """
    a_hat = np.asarray(a_hat, dtype=np.float64)
    a_true = np.asarray(a_true, dtype=np.float64)

    if a_hat.shape != a_true.shape:
        raise ValueError(f"a_hat shape {a_hat.shape} != a_true shape {a_true.shape}")

    if a_hat.ndim == 1:
        a_hat = a_hat[None, :]
        a_true = a_true[None, :]
    elif a_hat.ndim != 2:
        raise ValueError(f"a_hat must be 1D or 2D, got {a_hat.shape}")

    return a_hat, a_true


# ---------------------------------------------------------------------
# 逐模态 RMSE / NRMSE
# ---------------------------------------------------------------------


def rmse_per_mode(
    a_hat: ArrayLike,
    a_true: ArrayLike,
) -> np.ndarray:
    """
    逐模态 RMSE 谱线。

    定义（对每个模态 k）：
        RMSE_k = sqrt( mean_t ( (a_hat[t,k] - a_true[t,k])^2 ) )

    返回：
        rmse: float32 数组，形状 [r]。
    """
    a_hat, a_true = _to_2d_coeff_arrays(a_hat, a_true)  # [N,r]
    diff = a_hat - a_true
    mse_mode = np.mean(diff ** 2, axis=0)  # [r]
    rmse = np.sqrt(mse_mode)               # [r]
    return rmse.astype(np.float32, copy=False)


def nrmse_per_mode(
    a_hat: ArrayLike,
    a_true: ArrayLike,
    eigenvalues: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    逐模态 NRMSE 谱线。

    若提供 eigenvalues（通常是 POD 特征值 lambda_k），则定义：
        NRMSE_k = RMSE_k / sqrt(lambda_k)

    否则，使用该模态在真实系数上的标准差做归一化：
        denom_k = std_t( a_true[t,k] )
        NRMSE_k = RMSE_k / max(denom_k, eps)

    返回：
        nrmse: float32 数组，形状 [r]。
    """
    a_hat, a_true = _to_2d_coeff_arrays(a_hat, a_true)  # [N,r]
    rmse = rmse_per_mode(a_hat, a_true)                # [r]

    if eigenvalues is not None:
        ev = np.asarray(eigenvalues, dtype=np.float64)
        if ev.ndim != 1:
            raise ValueError(f"eigenvalues must be 1D, got {ev.shape}")
        if ev.shape[0] < rmse.shape[0]:
            raise ValueError(
                f"eigenvalues length {ev.shape[0]} < r={rmse.shape[0]} (need at least r)"
            )
        ev = ev[: rmse.shape[0]]
        denom = np.sqrt(ev)  # [r]
    else:
        # 用真实系数的 std 做归一化
        denom = np.std(a_true, axis=0)  # [r]

    denom = np.maximum(denom, eps)
    nrmse = rmse.astype(np.float64) / denom
    return nrmse.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# 按 band 统计系数 RMSE / NRMSE
# ---------------------------------------------------------------------


def rmse_per_band(
    a_hat: ArrayLike,
    a_true: ArrayLike,
    bands: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    """
    按给定的 POD 模态区间（band）计算每个 band 的系数 RMSE。

    参数
    ----
    a_hat, a_true:
        系数数组，形状 [N,r] 或 [r]，含义与 rmse_per_mode 一致。
    bands:
        例如 {"L": (0,10), "M": (10,40), "S": (40,128)}，
        下标区间为半开区间 [start, end)，采用 0-based 索引。

    返回
    ----
    band_rmse:
        例如 {"L": 0.01, "M": 0.02, "S": 0.05}，
        这里的数值是该 band 内所有样本、所有模态的**系数 RMSE**：
            RMSE_band = sqrt( mean( (a_hat - a_true)^2 ) )
    """
    a_hat, a_true = _to_2d_coeff_arrays(a_hat, a_true)  # [N,r]
    N, r = a_hat.shape

    band_errors: Dict[str, float] = {}
    for name, (start, end) in bands.items():
        if not (0 <= start < end <= r):
            raise ValueError(
                f"Band '{name}' with range [{start},{end}) is invalid for r={r}"
            )
        diff = a_hat[:, start:end] - a_true[:, start:end]  # [N, r_band]
        mse = float(np.mean(diff ** 2))
        rmse = float(np.sqrt(mse))
        band_errors[name] = rmse

    return band_errors


def nrmse_per_band(
    a_hat: ArrayLike,
    a_true: ArrayLike,
    bands: Dict[str, Tuple[int, int]],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    按给定 POD band 区间计算系数 NRMSE，定义为：

        NRMSE_band = sqrt( sum(e^2) / sum(a_true^2) )

    其中 e = a_hat - a_true，在该 band 内累加所有样本、所有模态。

    若某个 band 内 sum(a_true^2) = 0，则返回 NaN。
    """
    a_hat, a_true = _to_2d_coeff_arrays(a_hat, a_true)  # [N,r]
    N, r = a_hat.shape

    band_errors: Dict[str, float] = {}
    for name, (start, end) in bands.items():
        if not (0 <= start < end <= r):
            raise ValueError(
                f"Band '{name}' with range [{start},{end}) is invalid for r={r}"
            )

        diff = a_hat[:, start:end] - a_true[:, start:end]  # [N, r_band]
        num = float(np.sum(diff ** 2))
        denom = float(np.sum(a_true[:, start:end] ** 2))
        if denom <= eps:
            band_errors[name] = float("nan")
        else:
            band_errors[name] = float(np.sqrt(num / denom))

    return band_errors


# ---------------------------------------------------------------------
# 场空间 NMSE：支持 [N,H,W,C] / [N,D] / [H,W] 等
# ---------------------------------------------------------------------


def field_nmse(
    x_hat: ArrayLike,
    x_true: ArrayLike,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """
    场空间 NMSE：

        NMSE = ||x_hat - x_true||^2 / ||x_true||^2

    支持输入形状：
    - [N,H,W,C] / [N,H,W] / [N,D]：视作 batch 维度 N。
    - [H,W,C] / [H,W] / [D]：视作单样本，返回一个标量。

    参数
    ----
    reduction:
        "mean" : 对 batch 求平均，返回标量（常用）。
        "none" : 返回每个样本的 NMSE 向量，形状 [N]。

    返回
    ----
    float 或 np.ndarray:
        根据 reduction 决定。
    """
    x_hat = np.asarray(x_hat, dtype=np.float64)
    x_true = np.asarray(x_true, dtype=np.float64)

    if x_hat.shape != x_true.shape:
        raise ValueError(f"x_hat shape {x_hat.shape} != x_true shape {x_true.shape}")

    if x_hat.ndim == 1:
        # [D] -> 单样本 [1,D]
        x_hat = x_hat[None, :]
        x_true = x_true[None, :]
    elif x_hat.ndim >= 2:
        # 视第 0 维为 batch，其余展平
        N = x_hat.shape[0]
        x_hat = x_hat.reshape(N, -1)
        x_true = x_true.reshape(N, -1)
    else:
        raise ValueError(f"Unsupported ndim for x_hat: {x_hat.ndim}")

    diff = x_hat - x_true            # [N, D]
    num = np.sum(diff ** 2, axis=1)  # [N]
    denom = np.sum(x_true ** 2, axis=1)  # [N]

    # 避免除零：denom==0 -> NaN
    mask_zero = denom <= eps
    nmse_vec = np.empty_like(num, dtype=np.float64)
    nmse_vec[mask_zero] = np.nan
    nmse_vec[~mask_zero] = num[~mask_zero] / denom[~mask_zero]

    if reduction == "none":
        return nmse_vec.astype(np.float32, copy=False)
    elif reduction == "mean":
        # 忽略 NaN
        return float(np.nanmean(nmse_vec))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# ---------------------------------------------------------------------
# 基于 POD 的单组 / 累积重建 NMSE
# ---------------------------------------------------------------------


def _reconstruct_from_coeffs(
    Ur: np.ndarray,          # [D,r]
    coeffs: np.ndarray,      # [N,r]
    mean_flat: Optional[np.ndarray] = None,  # [D] or None
) -> np.ndarray:
    """
    X = mean_flat + coeffs @ Ur^T

    返回形状 [N, D] 的展平场。
    """
    Ur = np.asarray(Ur, dtype=np.float64)       # [D,r]
    coeffs = np.asarray(coeffs, dtype=np.float64)  # [N,r]

    X = coeffs @ Ur.T  # [N,D]
    if mean_flat is not None:
        mf = np.asarray(mean_flat, dtype=np.float64).reshape(1, -1)  # [1,D]
        X = X + mf
    return X  # [N,D]


def partial_recon_nmse(
    a_hat: ArrayLike,
    a_true: ArrayLike,
    Ur: np.ndarray,
    groups: Sequence[dict],
    mean_flat: Optional[np.ndarray] = None,
    sample_indices: Optional[Sequence[int]] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    基于 POD 模态分组的“单组 / 累积”重建 NMSE。

    输入
    ----
    a_hat, a_true:
        预测 / 真实 POD 系数，形状 [N,r] 或 [r]。
    Ur:
        POD 空间基底，形状 [D,r]，通常来自 Ur.npy。
    groups:
        φ 分组信息列表，每个元素形如：
            {
                "group_index": 1,
                "name": "S1",
                "k_start": 1,   # 1-based inclusive
                "k_end": 16,   # 1-based inclusive
            }
        注意：k_start / k_end 是 1-based，内部会转换为 0-based [start, end)。
    mean_flat:
        可选，展平的均值场 [D]，若 None 则不加均值（在 NMSE 定义中不会影响差值的分子，
        但会影响分母；这里采用“对重建场本身做 NMSE”的定义，因此推荐传入）。
    sample_indices:
        可选，若仅在部分时间样本上做分析，可以传 [n0, n1, ...]。
        若为 None，则使用全部样本。
    reduction:
        传给 field_nmse(...) 的 reduction 参数：
            - "mean" : 每个 band / 累积返回一个标量（常用）
            - "none" : 返回每个样本一条 NMSE 向量
    返回
    ----
    result: dict
        {
            "group_nmse": {group_name: scalar_or_vec, ...},
            "cumulative_nmse": {group_name_cum: scalar_or_vec, ...},
        }

        其中：
        - group_nmse[name]       : 只使用该组模态 S_g 的重建 NMSE
        - cumulative_nmse[name]  : 使用从 S1 累积到当前组的所有模态的重建 NMSE
    """
    a_hat, a_true = _to_2d_coeff_arrays(a_hat, a_true)  # [N,r]
    N, r = a_hat.shape

    Ur = np.asarray(Ur, dtype=np.float64)
    if Ur.shape[1] < r:
        raise ValueError(f"Ur shape {Ur.shape} has fewer modes than r={r}")

    # 选取部分样本（若指定）
    if sample_indices is not None:
        idx = np.asarray(sample_indices, dtype=np.int64)
        a_hat = a_hat[idx]
        a_true = a_true[idx]

    N_eff = a_hat.shape[0]

    # 预先构造一个全零模板
    zeros_template = np.zeros((N_eff, r), dtype=np.float64)

    group_nmse: Dict[str, Union[float, np.ndarray]] = {}
    cumulative_nmse: Dict[str, Union[float, np.ndarray]] = {}

    # 用于累积的掩码
    cumulative_mask = np.zeros(r, dtype=bool)

    for g in groups:
        name = g.get("name", f"S{g.get('group_index', 0)}")
        k_start_1 = int(g["k_start"])
        k_end_1 = int(g["k_end"])
        if not (1 <= k_start_1 <= k_end_1 <= r):
            raise ValueError(
                f"Invalid group [{k_start_1},{k_end_1}] for r={r}"
            )

        # 转成 0-based 半开区间 [start0, end0)
        start0 = k_start_1 - 1
        end0 = k_end_1

        # -------------------------
        # 1) 单组重建：只保留该组模态
        # -------------------------
        coeff_true_group = zeros_template.copy()
        coeff_hat_group = zeros_template.copy()
        coeff_true_group[:, start0:end0] = a_true[:, start0:end0]
        coeff_hat_group[:, start0:end0] = a_hat[:, start0:end0]

        X_true_g = _reconstruct_from_coeffs(Ur, coeff_true_group, mean_flat=mean_flat)
        X_hat_g = _reconstruct_from_coeffs(Ur, coeff_hat_group, mean_flat=mean_flat)

        nmse_g = field_nmse(X_hat_g, X_true_g, reduction=reduction, eps=eps)
        group_nmse[name] = nmse_g

        # -------------------------
        # 2) 累积重建：从 S1 到当前组
        # -------------------------
        cumulative_mask[start0:end0] = True
        coeff_true_cum = zeros_template.copy()
        coeff_hat_cum = zeros_template.copy()
        coeff_true_cum[:, cumulative_mask] = a_true[:, cumulative_mask]
        coeff_hat_cum[:, cumulative_mask] = a_hat[:, cumulative_mask]

        X_true_cum = _reconstruct_from_coeffs(Ur, coeff_true_cum, mean_flat=mean_flat)
        X_hat_cum = _reconstruct_from_coeffs(Ur, coeff_hat_cum, mean_flat=mean_flat)

        nmse_cum = field_nmse(X_hat_cum, X_true_cum, reduction=reduction, eps=eps)
        cumulative_nmse[name] = nmse_cum

    return {
        "group_nmse": group_nmse,
        "cumulative_nmse": cumulative_nmse,
    }
