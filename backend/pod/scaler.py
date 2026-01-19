# backend/pod/scaler.py
"""
POD Level-1 扩展：
- 模态特征尺度计算（多方法可选）
- 模态复数频谱字典缓存
"""

from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json

# ------------------------
# 基础工具
# ------------------------

def _fft2(q: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    q: [H, W] real
    return: complex FFT2, same shape
    """
    if demean:
        q = q - q.mean()
    return np.fft.fft2(q)


def _freq_grids(H: int, W: int, dx: float = 1.0, dy: float = 1.0):
    kx = np.fft.fftfreq(W, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=dy) * 2 * np.pi
    kxg, kyg = np.meshgrid(kx, ky)
    kr = np.sqrt(kxg**2 + kyg**2)
    return kxg, kyg, kr


# ------------------------
# 尺度计算方法
# ------------------------

def scale_from_energy_centroid_1d(
    signal: np.ndarray,
    dx: float,
    k_min: float | None = None,
    k_max: float | None = None,
    demean: bool = True,
) -> float:
    """
    一维稳健能量质心尺度：
    ell = 2*pi / <k>_E
    """
    sig = signal.astype(np.float64, copy=False)
    if demean:
        sig = sig - np.mean(sig)
    f = np.fft.fft(sig)
    k = np.fft.fftfreq(signal.size, d=dx) * 2 * np.pi
    E = np.abs(f)**2

    mask = (k > 0)
    if k_min is not None:
        mask &= (np.abs(k) >= k_min)
    if k_max is not None:
        mask &= (np.abs(k) <= k_max)

    k_use = np.abs(k[mask])
    E_use = E[mask]
    if E_use.sum() <= 0:
        return np.inf

    k_bar = (k_use * E_use).sum() / E_use.sum()
    return 2 * np.pi / max(k_bar, 1e-12)


def scale_from_peak_1d(
    signal: np.ndarray,
    dx: float,
    k_min: float | None = None,
    k_max: float | None = None,
    demean: bool = True,
) -> float:
    """
    一维主导峰值尺度（对应“卷积+相位最大化”）：

    你导师口述的：
        F(n, theta) = <s, sin(ω_n x + theta)>
        A(n) = max_theta F(n, theta)
    在离散实现里等价于取该频率的复系数幅值 |FFT(s)(k)|。

    本函数做法：
    - 在指定频率范围内取 |F(k)| 最大处的 k*（只取正频率，排除 k=0）
    - 返回 ell = 2*pi / k*
    """
    sig = signal.astype(np.float64, copy=False)
    if demean:
        sig = sig - np.mean(sig)

    f = np.fft.fft(sig)
    k = np.fft.fftfreq(sig.size, d=dx) * 2 * np.pi
    A = np.abs(f)

    mask = (k > 0)
    if k_min is not None:
        mask &= (k >= k_min)
    if k_max is not None:
        mask &= (k <= k_max)

    if not np.any(mask):
        return np.inf

    k_use = k[mask]
    A_use = A[mask]
    idx = int(np.argmax(A_use))
    k_star = float(k_use[idx])
    if not np.isfinite(k_star) or k_star <= 0:
        return np.inf
    return 2 * np.pi / k_star


def _robust_stat(v: np.ndarray) -> Dict[str, float]:
    """
    对一组尺度样本 v 做稳健统计（过滤 nan/inf），保持旧字段兼容并补充稳定性信息。
    """
    vv = np.asarray(v, dtype=np.float64)
    n_total = int(vv.size)
    vv = vv[np.isfinite(vv)]
    n_finite = int(vv.size)

    if n_finite <= 0:
        return {
            "n_total": float(n_total),
            "n_finite": float(0),
            "frac_bad": float(1.0 if n_total > 0 else 0.0),
            "med": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "mad": float("nan"),
        }

    med = float(np.median(vv))
    p25 = float(np.percentile(vv, 25))
    p75 = float(np.percentile(vv, 75))
    mean = float(np.mean(vv))
    std = float(np.std(vv))
    mad = float(np.median(np.abs(vv - med)))
    frac_bad = float((n_total - n_finite) / max(n_total, 1))

    return {
        "n_total": float(n_total),
        "n_finite": float(n_finite),
        "frac_bad": float(frac_bad),
        "med": med,
        "p25": p25,
        "p75": p75,
        "mean": mean,
        "std": std,
        "mad": mad,
    }

def estimate_mode_scales(
    q: np.ndarray,
    dx: float,
    dy: float,
    cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    q: [H, W] 单模态
    """
    H, W = q.shape
    method = cfg.get("method", "B_robust_energy_centroid")
    k_min = cfg.get("k_min", None)
    k_max = cfg.get("k_max", None)
    demean = bool(cfg.get("demean", True))

    # 仅依据 method 字符串决定尺度提取定义
    m = (method or "").lower()
    if ("peak" in m) or ("argmax" in m) or m.startswith("a_"):
        scale_fn = scale_from_peak_1d
    else:
        # 默认：稳健能量质心
        scale_fn = scale_from_energy_centroid_1d

    # x / y 方向：逐行 / 逐列统计
    ell_x_list = []
    for j in range(H):
        ell_x_list.append(scale_fn(q[j, :], dx, k_min, k_max, demean=demean))

    ell_y_list = []
    for i in range(W):
        ell_y_list.append(scale_fn(q[:, i], dy, k_min, k_max, demean=demean))

    sx = _robust_stat(np.asarray(ell_x_list, dtype=np.float64))
    sy = _robust_stat(np.asarray(ell_y_list, dtype=np.float64))

    # 综合尺度：默认取更细方向（用于“可达最细尺度”叙事）
    ell_min = float(np.nanmin([sx["med"], sy["med"]]))
    ell_geo = (
        float(np.sqrt(sx["med"] * sy["med"]))
        if np.isfinite(sx["med"]) and np.isfinite(sy["med"])
        else float("nan")
    )

    return {
        "ell_x_med": sx["med"],
        "ell_x_p25": sx["p25"],
        "ell_x_p75": sx["p75"],
        "ell_x_mean": sx["mean"],
        "ell_x_std": sx["std"],
        "ell_x_mad": sx["mad"],

        "ell_y_med": sy["med"],
        "ell_y_p25": sy["p25"],
        "ell_y_p75": sy["p75"],
        "ell_y_mean": sy["mean"],
        "ell_y_std": sy["std"],
        "ell_y_mad": sy["mad"],

        "ell_min": ell_min,
        "ell_geo": ell_geo,
        "method": method,
    }

# ------------------------
# ScaleTable 构建
# ------------------------

def build_scale_table(
    q_modes: np.ndarray,
    grid_meta: Dict[str, Any],
    cfg_scale: Dict[str, Any],
    out_csv: Path,
    out_meta: Path,
    preview: bool = True,
):
    """
    q_modes: [R, H, W]
    """
    dx = float(grid_meta.get("dx", 1.0))
    dy = float(grid_meta.get("dy", 1.0))

    rows = []
    for i, q in enumerate(q_modes):
        row = {"mode": i}
        row.update(estimate_mode_scales(q, dx, dy, cfg_scale))
        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------
    # 新增：三类尺度信息
    # 1) ell_x_med / ell_y_med：单模态尺度（已由 estimate_mode_scales 给出）
    # 2) ell_x_prefix / ell_y_prefix：前 r 个模态的累积有效尺度（最细尺度）
    #    定义：ell_prefix(r) = min_{i<=r} ell_i
    # 3) ell_x_tail / ell_y_tail：后 R-r 个模态的“最大可预测尺度”（最粗尺度）
    #    定义：ell_tail(r) = max_{i>r} ell_i
    #
    # 备注：
    # - 这里仍保留旧字段 ell_x_eff/ell_y_eff 作为兼容别名（= ell_x_prefix/ell_y_prefix）。
    # - ell_min_eff / ell_geo_eff 先保留并放在最后两列（用于后续研究尺度塌缩时再启用）。
    # ------------------------

    def _suffix_nanmax_exclusive(v: np.ndarray) -> np.ndarray:
        """tail[i] = nanmax(v[i+1:])；若后缀为空或全为 NaN/Inf，则返回 NaN。"""
        vv = np.asarray(v, dtype=np.float64)
        vv = np.where(np.isfinite(vv), vv, np.nan)
        n = int(vv.size)
        out = np.full((n,), np.nan, dtype=np.float64)
        if n <= 1:
            return out
        cur = np.nan
        # 从后往前滚动维护 nanmax（exclusive 版本：先写 out，再更新 cur）
        for i in range(n - 1, -1, -1):
            out[i] = cur
            x = vv[i]
            if np.isnan(cur):
                cur = x
            elif not np.isnan(x) and x > cur:
                cur = x
        return out

    sx = pd.Series(df["ell_x_med"], dtype="float64")
    sy = pd.Series(df["ell_y_med"], dtype="float64")

    # prefix：最细尺度（前缀最小值）
    df["ell_x_prefix"] = sx.cummin()
    df["ell_y_prefix"] = sy.cummin()

    # tail：最大可预测尺度（尾部最大值，exclusive）
    df["ell_x_tail"] = _suffix_nanmax_exclusive(sx.to_numpy())
    df["ell_y_tail"] = _suffix_nanmax_exclusive(sy.to_numpy())

    # 兼容字段（旧名）
    df["ell_x_eff"] = df["ell_x_prefix"]
    df["ell_y_eff"] = df["ell_y_prefix"]

    # 先保留：尺度塌缩相关（仍用 prefix-cummin 定义）
    df["ell_min_eff"] = pd.Series(df["ell_min"], dtype="float64").cummin()
    df["ell_geo_eff"] = pd.Series(df["ell_geo"], dtype="float64").cummin()

    # 确保 ell_min_eff / ell_geo_eff 位于最后两列
    last2 = ["ell_min_eff", "ell_geo_eff"]
    cols = [c for c in df.columns if c not in last2] + last2
    df = df[cols]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "grid_meta": grid_meta,
        "scale_cfg": cfg_scale,
        "n_modes": int(q_modes.shape[0]),
        "effective_scale_def": {
            "ell_x_med": "per-mode robust scale in x (median over lines)",
            "ell_y_med": "per-mode robust scale in y (median over lines)",
            "ell_x_prefix": "cummin(ell_x_med)",
            "ell_y_prefix": "cummin(ell_y_med)",
            "ell_x_tail": "suffix max (exclusive): max_{i>r} ell_x_med(i)",
            "ell_y_tail": "suffix max (exclusive): max_{i>r} ell_y_med(i)",
            "ell_x_eff": "alias of ell_x_prefix (backward compatible)",
            "ell_y_eff": "alias of ell_y_prefix (backward compatible)",
            "ell_min_eff": "cummin(ell_min)",
            "ell_geo_eff": "cummin(ell_geo)",
        },
    }
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    if preview:
        print("=== ScaleTable Preview ===")
        print(df.head())
        print(df[["mode", "ell_x_prefix", "ell_y_prefix", "ell_x_tail", "ell_y_tail", "ell_min_eff", "ell_geo_eff"]].head(10))
        print(df[["ell_min", "ell_min_eff", "ell_geo", "ell_geo_eff"]].describe())

    return df


# ------------------------
# 频谱字典缓存
# ------------------------

def build_basis_spectrum(
    q_modes: np.ndarray,
    *,
    grid_meta: Dict[str, Any],
    out_npz: Path,
    fft_cfg: Dict[str, Any],
):
    """
    支持：
    - q_modes: [r,H,W] 或 [r,H,W,C]
    保存：
    - Q: [r,H,W] (complex64) 或 [r,H,W,C] (complex64)
    - kx, ky, kr
    - grid_meta, fft_cfg
    """
    dx = float(grid_meta.get("dx", 1.0))
    dy = float(grid_meta.get("dy", 1.0))
    demean = bool(fft_cfg.get("demean", True))

    q_modes = np.asarray(q_modes)
    if q_modes.ndim == 3:
        r, H, W = q_modes.shape
        Q = np.empty((r, H, W), dtype=np.complex64)
        for i in range(r):
            Q[i] = _fft2(q_modes[i], demean=demean).astype(np.complex64, copy=False)
    elif q_modes.ndim == 4:
        r, H, W, C = q_modes.shape
        Q = np.empty((r, H, W, C), dtype=np.complex64)
        for i in range(r):
            for c in range(C):
                Q[i, :, :, c] = _fft2(q_modes[i, :, :, c], demean=demean).astype(np.complex64, copy=False)
    else:
        raise ValueError(f"q_modes must be 3D or 4D, got {q_modes.shape}")

    kx, ky, kr = _freq_grids(H, W, dx, dy)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        Q=Q,
        kx=kx,
        ky=ky,
        kr=kr,
        grid_meta=grid_meta,
        fft_cfg=fft_cfg,
    )

    print(f"[basis_spectrum] saved: {out_npz}  Q.shape={Q.shape}")
    return out_npz
