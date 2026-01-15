# backend/metrics/cumulate_metrics.py
from __future__ import annotations

from typing import Literal, Tuple, Any, Dict, Optional
import numpy as np


def compute_nrmse_prefix_coeff(
    A_hat: np.ndarray,
    A_true: np.ndarray,
    *,
    eps: float = 1e-12,
    max_r: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    系数域“快路径”：
      NRMSE(r) = ||A_true[:,:r] - A_hat[:,:r]||_F / (||A_true[:,:r]||_F + eps)

    返回：
      r_grid: shape [R] -> 1..R
      nrmse_r: shape [R]
    """
    if A_hat.ndim != 2 or A_true.ndim != 2:
        raise ValueError(f"A_hat/A_true must be 2D [T,R]. Got {A_hat.shape}, {A_true.shape}")
    if A_hat.shape != A_true.shape:
        raise ValueError(f"A_hat and A_true shape mismatch: {A_hat.shape} vs {A_true.shape}")

    T, R = A_true.shape
    R_use = int(R if max_r is None else min(int(max_r), int(R)))
    if R_use <= 0:
        raise ValueError(f"Invalid R_use={R_use} (R={R}, max_r={max_r})")

    # 前缀累计：先算每个 mode 的能量，再 cumsum
    diff = (A_true[:, :R_use] - A_hat[:, :R_use]).astype(np.float64, copy=False)
    tru = A_true[:, :R_use].astype(np.float64, copy=False)

    # 每列（mode）平方和 -> [R_use]
    diff_e = np.sum(diff * diff, axis=0)
    tru_e = np.sum(tru * tru, axis=0)

    diff_prefix = np.cumsum(diff_e, axis=0)
    tru_prefix = np.cumsum(tru_e, axis=0)

    nrmse_r = np.sqrt(diff_prefix / (tru_prefix + float(eps)))
    r_grid = np.arange(1, R_use + 1, dtype=np.int32)
    return r_grid, nrmse_r


def compute_nrmse_prefix(
    A_hat: np.ndarray,
    A_true: np.ndarray,
    *,
    mode: Literal["coeff"] = "coeff",
    eps: float = 1e-12,
    max_r: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    统一入口（Batch-1 默认 coeff；field 暂不实现，后续 Batch-2/3 再接 POD 重建）。
    """
    if mode != "coeff":
        raise NotImplementedError(
            f"mode='{mode}' not implemented in Batch-1. Use mode='coeff' now."
        )
    r_grid, nrmse_r = compute_nrmse_prefix_coeff(A_hat, A_true, eps=eps, max_r=max_r)
    return r_grid, nrmse_r, "coeff"


def merge_leff_and_nrmse(
    *,
    leff_pack: Dict[str, Any],
    r_grid_nrmse: np.ndarray,
    nrmse_r: np.ndarray,
    max_r_cfg: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对齐并截断到共同 r_grid，输出联合 payload（供 L4 json 落盘/绘图）。
    规则：R_used = min(R_scale, R_nrmse, max_r_cfg(if provided))
    """
    r_grid_scale = leff_pack.get("r_grid", None)
    if r_grid_scale is None:
        raise ValueError("leff_pack missing r_grid")
    r_grid_scale = np.asarray(r_grid_scale)

    r_grid_nrmse = np.asarray(r_grid_nrmse)
    nrmse_r = np.asarray(nrmse_r, dtype=float)

    R_scale = int(len(r_grid_scale))
    R_nrmse = int(len(r_grid_nrmse))
    R_used = min(R_scale, R_nrmse)

    if max_r_cfg is not None:
        R_used = min(R_used, int(max_r_cfg))

    if R_used <= 0:
        raise ValueError(f"Invalid R_used={R_used} (R_scale={R_scale}, R_nrmse={R_nrmse}, max_r_cfg={max_r_cfg})")

    # 统一输出 r_grid 为 1..R_used
    r_grid = np.arange(1, R_used + 1, dtype=np.int32)

    def _cut(v):
        if v is None:
            return None
        vv = np.asarray(v, dtype=float)
        return vv[:R_used].tolist()

    leff_x = _cut(leff_pack.get("leff_x", None))
    leff_y = _cut(leff_pack.get("leff_y", None))
    leff_agg = _cut(leff_pack.get("leff_agg", None))

    payload = {
        "r_grid": r_grid.tolist(),
        "R_used": int(R_used),
        "R_scale": int(R_scale),
        "R_nrmse": int(R_nrmse),
        "leff": {
            "x": leff_x,
            "y": leff_y,
            "agg": leff_agg,
            "colmap": leff_pack.get("colmap", None),
            "scale_table_path": leff_pack.get("scale_table_path", None),
        },
        "nrmse_r": nrmse_r[:R_used].astype(float).tolist(),
    }
    return payload