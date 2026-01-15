# backend/pipelines/eval_mods/cumulate_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import csv
import numpy as np

from backend.pipelines.eval.utils import load_npz, read_json


# -----------------------------
# POD (L1) artifacts
# -----------------------------

@dataclass(frozen=True)
class PodArtifactsPaths:
    pod_dir: Path
    Ur_npy: Path
    mean_flat_npy: Path
    pod_meta_json: Path
    scale_table_csv: Path
    scale_meta_json: Path


def resolve_pod_dir(ctx) -> Path:
    """
    单一可信来源：ctx.pod_cfg.save_dir
    不做猜路径。
    """
    pod_cfg = getattr(ctx, "pod_cfg", None)
    if pod_cfg is None:
        raise ValueError("ctx.pod_cfg is None; cannot resolve POD save_dir.")
    save_dir = getattr(pod_cfg, "save_dir", None)
    if save_dir is None:
        raise ValueError("ctx.pod_cfg.save_dir is None; cannot resolve POD save_dir.")
    return Path(save_dir)


def pod_artifacts_paths(ctx) -> PodArtifactsPaths:
    pod_dir = resolve_pod_dir(ctx)
    return PodArtifactsPaths(
        pod_dir=pod_dir,
        Ur_npy=pod_dir / "Ur.npy",
        mean_flat_npy=pod_dir / "mean_flat.npy",
        pod_meta_json=pod_dir / "pod_meta.json",
        scale_table_csv=pod_dir / "scale_table.csv",
        scale_meta_json=pod_dir / "scale_meta.json",
    )


def try_load_npy(path: Path) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rep
    try:
        arr = np.load(path, allow_pickle=False)
        rep.update({"ok": True, "dtype": str(arr.dtype), "shape": tuple(arr.shape)})
    except Exception as e:
        rep.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return rep


def try_read_json(path: Path) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rep
    try:
        obj = read_json(path)
        rep["ok"] = True
        # 只 peek 一点，避免把大 json 全塞进报告
        rep["keys"] = sorted(list(obj.keys())) if isinstance(obj, dict) else None
        rep["peek"] = obj if not isinstance(obj, dict) else {k: obj.get(k) for k in list(obj.keys())[:8]}
    except Exception as e:
        rep["ok"] = False
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


def try_peek_scale_table_csv(path: Path, *, max_rows: int = 3) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rep
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rep["ok"] = True
        rep["n_lines"] = int(len(lines))
        rep["head"] = lines[: 1 + int(max_rows)]
        if lines:
            rep["columns"] = [x.strip() for x in lines[0].split(",")]
    except Exception as e:
        rep["ok"] = False
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep


def load_scale_table_csv(path: Path) -> Dict[str, Any]:
    """
    轻量 CSV loader（不依赖 pandas）：
    返回 dict: { "columns": [...], "rows": [...], "data": {col: np.ndarray} }
    注意：此处只负责 IO，不做统计。
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows: List[Dict[str, Any]] = []
        for r in reader:
            rows.append(r)

    # 尝试把能转成 float 的列都转成 float ndarray
    data: Dict[str, Any] = {}
    for c in cols:
        col_vals = [rows[i].get(c, "") for i in range(len(rows))]
        # try numeric
        ok_num = True
        out_num: List[float] = []
        for v in col_vals:
            try:
                out_num.append(float(v))
            except Exception:
                ok_num = False
                break
        if ok_num:
            data[c] = np.asarray(out_num, dtype=float)
        else:
            data[c] = np.asarray(col_vals, dtype=object)

    return {"columns": cols, "rows": rows, "data": data}


# -----------------------------
# L2 helpers
# -----------------------------

def find_any_l2_npz(l2_root: Path) -> Optional[Path]:
    l2_root = Path(l2_root)
    cands = sorted(list(l2_root.glob("*.npz")))
    if len(cands) == 0:
        cands = sorted(list(l2_root.rglob("*.npz")))
    return cands[0] if len(cands) > 0 else None


def peek_l2_npz(path: Path) -> Dict[str, Any]:
    rep: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return rep

    try:
        z = load_npz(path, allow_pickle=False)
        keys = list(z.get("_keys", []))
        rep["ok"] = True
        rep["keys"] = keys

        def _shape(k: str) -> Optional[Tuple[int, ...]]:
            if k not in z:
                return None
            v = z[k]
            if isinstance(v, np.ndarray):
                return tuple(v.shape)
            return None

        rep["peek"] = {
            "A_hat_all.shape": _shape("A_hat_all"),
            "A_true_all.shape": _shape("A_true_all"),
            "mask_flat.shape": _shape("mask_flat"),
            "mask_rate": z.get("mask_rate", None),
            "noise_sigma": z.get("noise_sigma", None),
            "model_type": z.get("model_type", None),
            "centered_pod": z.get("centered_pod", None),
        }
    except Exception as e:
        rep["ok"] = False
        rep["error"] = f"{type(e).__name__}: {e}"
    return rep

def _pick_first_key(z: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        if k in z:
            return k
    return None


def load_coeff_pair_from_l2(z: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    从 L2 npz(dict) 中抽取 A_hat / A_true（并做形状校验与兼容 key）。
    返回 (A_hat, A_true, meta_peek)
    """
    k_hat = _pick_first_key(z, ("A_hat_all", "A_hat"))
    k_tru = _pick_first_key(z, ("A_true_all", "A_true"))
    if k_hat is None or k_tru is None:
        raise KeyError(f"L2 missing coeff keys: need A_hat_all/A_hat and A_true_all/A_true. keys={list(z.keys())}")

    A_hat = z[k_hat]
    A_true = z[k_tru]

    if not isinstance(A_hat, np.ndarray) or not isinstance(A_true, np.ndarray):
        raise TypeError(f"A_hat/A_true must be numpy arrays. Got {type(A_hat)}, {type(A_true)}")

    # 兼容 [T,R,1] -> [T,R]
    if A_hat.ndim == 3 and A_hat.shape[-1] == 1:
        A_hat = A_hat[..., 0]
    if A_true.ndim == 3 and A_true.shape[-1] == 1:
        A_true = A_true[..., 0]

    if A_hat.ndim != 2 or A_true.ndim != 2:
        raise ValueError(f"A_hat/A_true must be 2D [T,R]. Got {A_hat.shape}, {A_true.shape}")

    if A_hat.shape != A_true.shape:
        raise ValueError(f"A_hat/A_true mismatch: {A_hat.shape} vs {A_true.shape}")

    meta = {
        "k_hat": k_hat,
        "k_true": k_tru,
        "T": int(A_true.shape[0]),
        "R": int(A_true.shape[1]),
        "mask_rate": z.get("mask_rate", None),
        "noise_sigma": z.get("noise_sigma", None),
        "model_type": z.get("model_type", None),
        "centered_pod": z.get("centered_pod", None),
    }
    return A_hat, A_true, meta


def load_coeff_pair(ctx, model_type: str, mask_rate: float, noise_sigma: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Path]:
    """
    从 ctx.load_l2 读取后抽系数对。
    返回 (A_hat, A_true, meta_peek, npz_path)
    """
    npz_path = ctx.get_l2_path(model_type, mask_rate, noise_sigma)
    z = ctx.load_l2(model_type, mask_rate, noise_sigma)
    A_hat, A_true, meta = load_coeff_pair_from_l2(z)
    return A_hat, A_true, meta, Path(npz_path)

# === Batch-2: POD cumulative effective scales loader ===

def _pick_col(data_cols: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    """优先精确匹配，其次大小写不敏感匹配。"""
    s = set(data_cols)
    for c in candidates:
        if c in s:
            return c
    lower_map = {c.lower(): c for c in data_cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def load_pod_mode_scales_standardized(ctx, *, agg_kind: str = "min") -> Dict[str, Any]:
    """
    读取 POD save_dir/scale_table.csv 并标准化输出（按你的 scale_table 列名规范）。

    CSV header (given):
      mode, ..., ell_x_eff, ell_y_eff, ell_min_eff, ell_geo_eff

    输出：
      r_grid: 1..R
      leff_x/leff_y/leff_agg: np.ndarray[float]
      colmap: 使用到的列名映射
    """
    pp = pod_artifacts_paths(ctx)
    tab = load_scale_table_csv(pp.scale_table_csv)
    cols: List[str] = tab.get("columns", [])
    data: Dict[str, Any] = tab.get("data", {})

    required = ["mode", "ell_x_eff", "ell_y_eff"]
    for k in required:
        if k not in data:
            raise KeyError(f"[cumulate_io] scale_table.csv missing column '{k}'. cols={cols}")

    if agg_kind not in ("min", "geo"):
        raise ValueError(f"[cumulate_io] agg_kind must be 'min' or 'geo', got '{agg_kind}'")
    agg_col = "ell_min_eff" if agg_kind == "min" else "ell_geo_eff"
    if agg_col not in data:
        raise KeyError(f"[cumulate_io] scale_table.csv missing column '{agg_col}'. cols={cols}")

    mode_raw = np.asarray(data["mode"], dtype=np.int64)
    if mode_raw.size == 0:
        raise ValueError("[cumulate_io] empty 'mode' column in scale_table.csv")

    # 0-based -> 1-based
    if int(mode_raw.min()) == 0:
        mode_raw = mode_raw + 1

    order = np.argsort(mode_raw)

    leff_x = np.asarray(data["ell_x_eff"], dtype=float)[order]
    leff_y = np.asarray(data["ell_y_eff"], dtype=float)[order]
    leff_agg = np.asarray(data[agg_col], dtype=float)[order]

    R = int(len(mode_raw))
    # 统一输出 r_grid 为 1..R 的前缀，确保后续合并用“长度对齐”不会错位
    r_grid = np.arange(1, R + 1, dtype=np.int32)

    colmap = {
        "r": "mode",
        "x": "ell_x_eff",
        "y": "ell_y_eff",
        "agg": agg_col,
        "agg_kind": agg_kind,
    }

    return {
        "r_grid": r_grid,
        "leff_x": leff_x,
        "leff_y": leff_y,
        "leff_agg": leff_agg,
        "colmap": colmap,
        "R": R,
        "scale_table_path": str(pp.scale_table_csv),
    }