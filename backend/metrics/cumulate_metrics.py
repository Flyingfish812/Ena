# backend/metrics/cumulate_metrics.py
from __future__ import annotations

from typing import Literal, Tuple, Any, Dict, Optional, List
import numpy as np


def merge_scales_and_nrmsepack(
    *,
    scales_pack: Dict[str, Any],
    r_grid_nrmse: np.ndarray,
    nrmse_pack: Dict[str, Any],
    max_r_cfg: Optional[int] = None,
) -> Dict[str, Any]:
    """
    对齐并截断到共同 r_grid，输出联合 payload（供 L4 json 落盘/绘图）。
    规则：R_used = min(R_scale, R_nrmse, max_r_cfg(if provided))

    scales_pack 期望来自 cumulate_io.load_pod_mode_scales_standardized(ctx)，包含：
      - r_grid, R
      - ell_x_med/prefix/tail
      - ell_y_med/prefix/tail
      - colmap, scale_table_path

    nrmse_pack 期望包含三条曲线（list/ndarray 均可）：
      - nrmse_full
      - nrmse_prefix
      - nrmse_tail

    返回：
      {
        "r_grid": [1..R_used],
        "R_used": int,
        "R_scale": int,
        "R_nrmse": int,
        "scales": {...},
        "nrmse": {
          "nrmse_full": [...],
          "nrmse_prefix": [...],
          "nrmse_tail": [...],
        },
      }
    """
    r_grid_scale = scales_pack.get("r_grid", None)
    if r_grid_scale is None:
        raise ValueError("scales_pack missing r_grid")
    r_grid_scale = np.asarray(r_grid_scale)

    r_grid_nrmse = np.asarray(r_grid_nrmse)

    R_scale = int(len(r_grid_scale))
    R_nrmse = int(len(r_grid_nrmse))
    R_used = min(R_scale, R_nrmse)

    if max_r_cfg is not None:
        R_used = min(R_used, int(max_r_cfg))

    if R_used <= 0:
        raise ValueError(
            f"Invalid R_used={R_used} (R_scale={R_scale}, R_nrmse={R_nrmse}, max_r_cfg={max_r_cfg})"
        )

    # 统一输出 r_grid 为 1..R_used
    r_grid = np.arange(1, R_used + 1, dtype=np.int32)

    def _cut_arr(v: Any) -> List[float]:
        vv = np.asarray(v, dtype=float).reshape(-1)
        return vv[:R_used].astype(float).tolist()

    required_scale_keys = [
        "ell_x_med",
        "ell_x_prefix",
        "ell_x_tail",
        "ell_y_med",
        "ell_y_prefix",
        "ell_y_tail",
    ]
    missing = [k for k in required_scale_keys if k not in scales_pack]
    if missing:
        raise KeyError(f"scales_pack missing keys: {missing}")

    scales = {k: _cut_arr(scales_pack[k]) for k in required_scale_keys}
    scales["colmap"] = scales_pack.get("colmap", None)
    scales["scale_table_path"] = scales_pack.get("scale_table_path", None)

    # nrmse pack (strict, no compat)
    for k in ("nrmse_full", "nrmse_prefix", "nrmse_tail"):
        if k not in nrmse_pack:
            raise KeyError(f"nrmse_pack missing key: {k}")

    payload = {
        "r_grid": r_grid.tolist(),
        "R_used": int(R_used),
        "R_scale": int(R_scale),
        "R_nrmse": int(R_nrmse),
        "scales": scales,
        "nrmse": {
            "nrmse_full": _cut_arr(nrmse_pack["nrmse_full"]),
            "nrmse_prefix": _cut_arr(nrmse_pack["nrmse_prefix"]),
            "nrmse_tail": _cut_arr(nrmse_pack["nrmse_tail"]),
        },
    }
    return payload


def compute_nrmse_vs_r_coeff(
    A_hat: np.ndarray,
    A_true: np.ndarray,
    *,
    eps: float = 1e-12,
    max_r: int | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    系数域计算三种 NRMSE 曲线，随 r(=前缀模态数) 变化：

    - nrmse_prefix(r): ||A_true[:,:r] - A_hat[:,:r]||_F / (||A_true[:,:r]||_F + eps)

    - nrmse_full(r):   ||A_true - A_hat_prefix(r)||_F / (||A_true||_F + eps)
      A_hat_prefix(r) 仅保留前 r 列，后面列置零。
      等价：num2 = diff_prefix^2 + tru_tail^2

    - nrmse_tail(r):   ||A_true[:,r:] - A_hat[:,r:]||_F / (||A_true[:,r:]||_F + eps)
      tail 采用“后 R-r 个模态”（0-based 起点 start_idx=r）

    返回:
      r_grid: [1..R_use]
      nrmse: {"nrmse_full","nrmse_prefix","nrmse_tail"}  (末尾 nrmse_tail 可能为 nan)
    """
    if A_hat.ndim != 2 or A_true.ndim != 2:
        raise ValueError(f"A_hat/A_true must be 2D [T,R]. Got {A_hat.shape}, {A_true.shape}")
    if A_hat.shape != A_true.shape:
        raise ValueError(f"A_hat and A_true shape mismatch: {A_hat.shape} vs {A_true.shape}")

    T, R = A_true.shape
    R_use = int(R if max_r is None else min(int(max_r), int(R)))
    if R_use <= 0:
        raise ValueError(f"Invalid R_use={R_use} (R={R}, max_r={max_r})")

    tru = A_true[:, :R_use].astype(np.float64, copy=False)
    hat = A_hat[:, :R_use].astype(np.float64, copy=False)
    diff = tru - hat

    tru_e = np.sum(tru * tru, axis=0)      # [R_use]
    diff_e = np.sum(diff * diff, axis=0)   # [R_use]

    tru_prefix = np.cumsum(tru_e, axis=0)    # [R_use]
    diff_prefix = np.cumsum(diff_e, axis=0)  # [R_use]
    tru_total = float(np.sum(tru_e))

    nrmse_prefix = np.sqrt(diff_prefix / (tru_prefix + float(eps)))

    num2_full = diff_prefix + (tru_total - tru_prefix)
    nrmse_full = np.sqrt(num2_full / (tru_total + float(eps)))

    diff_suffix = np.zeros((R_use + 1,), dtype=np.float64)
    tru_suffix = np.zeros((R_use + 1,), dtype=np.float64)
    diff_suffix[:R_use] = np.cumsum(diff_e[::-1], axis=0)[::-1]
    tru_suffix[:R_use] = np.cumsum(tru_e[::-1], axis=0)[::-1]

    nrmse_tail = np.full((R_use,), np.nan, dtype=np.float64)
    for r in range(1, R_use + 1):
        start_idx = r
        if start_idx >= R_use:
            continue
        nrmse_tail[r - 1] = np.sqrt(diff_suffix[start_idx] / (tru_suffix[start_idx] + float(eps)))

    r_grid = np.arange(1, R_use + 1, dtype=np.int32)
    return r_grid, {
        "nrmse_full": nrmse_full,
        "nrmse_prefix": nrmse_prefix,
        "nrmse_tail": nrmse_tail,
    }


def compute_nrmse_vs_r(
    A_hat: np.ndarray,
    A_true: np.ndarray,
    *,
    mode: Literal["coeff"] = "coeff",
    eps: float = 1e-12,
    max_r: int | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """
    统一入口：目前只实现 coeff。
    """
    if mode != "coeff":
        raise NotImplementedError(f"mode='{mode}' not implemented. Use mode='coeff' now.")
    r_grid, nrmse = compute_nrmse_vs_r_coeff(A_hat, A_true, eps=eps, max_r=max_r)
    return r_grid, nrmse, "coeff"
