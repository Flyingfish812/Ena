# backend/pipelines/eval_mods/fourier_io.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import json
import numpy as np

# pandas 是为了复用旧版 fourier_plots 的 df 接口（plot_kstar_heatmap / plot_fourier_band_nrmse_curves）
# 如果你未来决定彻底去 df 化，可以在这里替换掉。
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class L3FourierEntry:
    model_type: str
    mask_rate: float
    noise_sigma: float
    l3_fft_path: str
    k_star: float


def read_l3_index(exp_dir: str | Path) -> Dict[str, Any]:
    exp_dir = Path(exp_dir)
    l3_root = exp_dir / "L3_fft"
    idx_path = l3_root / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing L3 index.json: {idx_path}")
    with idx_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_l3_meta(exp_dir: str | Path) -> Dict[str, Any]:
    exp_dir = Path(exp_dir)
    meta_path = exp_dir / "L3_fft" / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_l3_entries(l3_index: Dict[str, Any]) -> List[L3FourierEntry]:
    out: List[L3FourierEntry] = []
    for e in (l3_index.get("entries", []) or []):
        try:
            out.append(
                L3FourierEntry(
                    model_type=str(e["model_type"]),
                    mask_rate=float(e["mask_rate"]),
                    noise_sigma=float(e["noise_sigma"]),
                    l3_fft_path=str(e["l3_fft_path"]),
                    k_star=float(e.get("k_star", np.nan)),
                )
            )
        except Exception:
            continue
    return out


def _lambda_edges_to_k_interior(lambda_edges: Sequence[float]) -> List[float]:
    # lambda_edges: [λ1, λ2, ...] 代表从“大尺度到小尺度”的分界
    # interior k edges: [1/λ1, 1/λ2, ...]
    ks: List[float] = []
    for lam in lambda_edges:
        lam = float(lam)
        if np.isfinite(lam) and lam > 0:
            ks.append(1.0 / lam)
    ks = sorted(set(ks))
    return ks


def compute_band_nrmse_from_l3_npz(
    npz_path: str | Path,
    *,
    k_edges_interior: Sequence[float],
    band_names: Sequence[str],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    从 L3 npz（E_true_k / E_err_k / k_centers）计算每个 band 的 NRMSE：
      NRMSE_band = sqrt( sum(E_err) / (sum(E_true)+eps) )
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        k = np.asarray(z["k_centers"], dtype=np.float64).reshape(-1)
        Et = np.asarray(z["E_true_k"], dtype=np.float64).reshape(-1)
        Ee = np.asarray(z["E_err_k"], dtype=np.float64).reshape(-1)

    # full edges: [0, k1, k2, ..., +inf]（最后用 k.max 作为上界）
    interior = [float(x) for x in k_edges_interior if np.isfinite(float(x)) and float(x) > 0]
    interior = sorted(interior)
    kmax = float(np.nanmax(k)) if k.size > 0 else 0.0
    edges = [0.0] + interior + [kmax + 1e-9]

    out: Dict[str, float] = {}
    nb = len(edges) - 1
    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        name = band_names[i] if i < len(band_names) else f"B{i}"
        m = (k >= lo) & (k < hi)
        if not np.any(m):
            out[f"fourier_band_nrmse_{name}"] = float("nan")
            continue
        num = float(np.nansum(Ee[m]))
        den = float(np.nansum(Et[m])) + float(eps)
        out[f"fourier_band_nrmse_{name}"] = float(np.sqrt(max(num, 0.0) / den))
    return out


def build_fourier_df_from_l3(
    *,
    entries: Sequence[L3FourierEntry],
    eval_cfg_fourier: Any,
    model_type: str,
) -> Any:
    """
    生成给旧版 fourier_plots 复用的 DataFrame，至少包含：
      mask_rate, noise_sigma, k_star
      fourier_band_nrmse_<band>
    """
    if pd is None:
        raise RuntimeError("pandas is required for fourier plotting functions (df-based).")

    band_names = tuple(getattr(eval_cfg_fourier, "band_names", ("L", "M", "H")))
    lambda_edges = list(getattr(eval_cfg_fourier, "lambda_edges", (1.0, 0.25)))
    k_edges_interior = _lambda_edges_to_k_interior(lambda_edges)

    rows: List[Dict[str, Any]] = []
    for e in entries:
        if str(e.model_type) != str(model_type):
            continue
        row: Dict[str, Any] = {
            "mask_rate": float(e.mask_rate),
            "noise_sigma": float(e.noise_sigma),
            "k_star": float(e.k_star),
        }
        row.update(
            compute_band_nrmse_from_l3_npz(
                e.l3_fft_path,
                k_edges_interior=k_edges_interior,
                band_names=band_names,
            )
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # 排序便于绘图一致
    df = df.sort_values(["mask_rate", "noise_sigma"]).reset_index(drop=True)
    return df


def pick_representative_l3_npz(
    *,
    entries: Sequence[L3FourierEntry],
    prefer_model: str = "linear",
) -> Optional[str]:
    # 用于画 “E(k)+band edges” 的解释图：随便挑一个最具代表性的 entry
    # 策略：优先 linear；其次任意；在该模型内选最小噪声、最大 p（观测最充分，谱更稳）
    cand = [e for e in entries if e.model_type == prefer_model]
    if not cand:
        cand = list(entries)
    if not cand:
        return None
    cand = sorted(cand, key=lambda x: (x.noise_sigma, -x.mask_rate))
    return str(cand[0].l3_fft_path)


def lambda_edges_to_k_edges_interior(eval_cfg_fourier: Any) -> List[float]:
    lambda_edges = list(getattr(eval_cfg_fourier, "lambda_edges", (1.0, 0.25)))
    return _lambda_edges_to_k_interior(lambda_edges)
