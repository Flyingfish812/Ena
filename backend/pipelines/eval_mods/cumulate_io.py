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
    """单一可信来源：ctx.pod_cfg.save_dir。不做猜路径。"""
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
    """轻量 CSV loader（不依赖 pandas）。

    返回：
      {
        "columns": [...],
        "rows": [ {col: raw_str, ...}, ... ],
        "data": {col: np.ndarray}
      }

    说明：这里只负责 IO 与类型推断（能转 float 的列转 float），不做任何统计。
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        rows: List[Dict[str, Any]] = [r for r in reader]

    data: Dict[str, Any] = {}
    for c in cols:
        col_vals = [rows[i].get(c, "") for i in range(len(rows))]
        ok_num = True
        out_num: List[float] = []
        for v in col_vals:
            try:
                out_num.append(float(v))
            except Exception:
                ok_num = False
                break
        data[c] = np.asarray(out_num, dtype=float) if ok_num else np.asarray(col_vals, dtype=object)

    return {"columns": cols, "rows": rows, "data": data}


# -----------------------------
# POD scales (L1) loader (Batch-0)
# -----------------------------


def load_pod_mode_scales_standardized(ctx) -> Dict[str, Any]:
    """读取 POD save_dir/scale_table.csv，并一次性输出后续画图需要的全部尺度列（长度均为 R）。

    期望 scale_table.csv 至少包含这些列（大小写必须精确匹配）：
      - mode
      - ell_x_med, ell_x_prefix, ell_x_tail
      - ell_y_med, ell_y_prefix, ell_y_tail

    返回：
      {
        "r_grid": np.ndarray[int]    # 1..R
        "mode": np.ndarray[int]      # 与 r_grid 同序（1-based）
        "R": int,
        "ell_x_med": np.ndarray[float],
        "ell_x_prefix": np.ndarray[float],
        "ell_x_tail": np.ndarray[float],
        "ell_y_med": np.ndarray[float],
        "ell_y_prefix": np.ndarray[float],
        "ell_y_tail": np.ndarray[float],
        "colmap": Dict[str,str],
        "scale_table_path": str,
      }

    说明：
      - 仅做“按 mode 排序 + 0/1-based 对齐 + 数值列提取”。
      - 不做任何拟合/统计，避免在 IO 层混入分析逻辑。
    """
    pp = pod_artifacts_paths(ctx)
    tab = load_scale_table_csv(pp.scale_table_csv)
    cols: List[str] = tab.get("columns", [])
    data: Dict[str, Any] = tab.get("data", {})

    required_cols = [
        "mode",
        "ell_x_med",
        "ell_x_prefix",
        "ell_x_tail",
        "ell_y_med",
        "ell_y_prefix",
        "ell_y_tail",
    ]
    missing = [c for c in required_cols if c not in data]
    if missing:
        raise KeyError(f"[cumulate_io] scale_table.csv missing columns: {missing}. cols={cols}")

    mode_raw = np.asarray(data["mode"], dtype=np.int64)
    if mode_raw.size == 0:
        raise ValueError("[cumulate_io] empty 'mode' column in scale_table.csv")

    # 0-based -> 1-based
    if int(mode_raw.min()) == 0:
        mode_raw = mode_raw + 1

    order = np.argsort(mode_raw)
    mode_sorted = mode_raw[order]

    def _as_float(col: str) -> np.ndarray:
        return np.asarray(data[col], dtype=float)[order]

    ell_x_med = _as_float("ell_x_med")
    ell_x_prefix = _as_float("ell_x_prefix")
    ell_x_tail = _as_float("ell_x_tail")

    ell_y_med = _as_float("ell_y_med")
    ell_y_prefix = _as_float("ell_y_prefix")
    ell_y_tail = _as_float("ell_y_tail")

    R = int(len(mode_sorted))
    r_grid = np.arange(1, R + 1, dtype=np.int32)

    colmap = {
        "r": "mode",
        "ell_x_med": "ell_x_med",
        "ell_x_prefix": "ell_x_prefix",
        "ell_x_tail": "ell_x_tail",
        "ell_y_med": "ell_y_med",
        "ell_y_prefix": "ell_y_prefix",
        "ell_y_tail": "ell_y_tail",
    }

    return {
        "r_grid": r_grid,
        "mode": mode_sorted.astype(np.int32, copy=False),
        "R": R,
        "ell_x_med": ell_x_med,
        "ell_x_prefix": ell_x_prefix,
        "ell_x_tail": ell_x_tail,
        "ell_y_med": ell_y_med,
        "ell_y_prefix": ell_y_prefix,
        "ell_y_tail": ell_y_tail,
        "colmap": colmap,
        "scale_table_path": str(pp.scale_table_csv),
    }


# -----------------------------
# L2 helpers (kept minimal)
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
            return tuple(v.shape) if isinstance(v, np.ndarray) else None

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


def _pick_first_key(z: dict, keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        if k in z:
            return k
    return None


def load_coeff_pair_from_l2(z: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """从 L2 npz(dict) 中抽取 A_hat / A_true（并做形状校验与兼容 key）。"""
    k_hat = _pick_first_key(z, ("A_hat_all", "A_hat"))
    k_tru = _pick_first_key(z, ("A_true_all", "A_true"))
    if k_hat is None or k_tru is None:
        raise KeyError(
            f"L2 missing coeff keys: need A_hat_all/A_hat and A_true_all/A_true. keys={list(z.keys())}"
        )

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


def load_coeff_pair(
    ctx, model_type: str, mask_rate: float, noise_sigma: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Path]:
    """从 ctx.load_l2 读取后抽系数对。

    返回 (A_hat, A_true, meta_peek, npz_path)
    """
    npz_path = ctx.get_l2_path(model_type, mask_rate, noise_sigma)
    z = ctx.load_l2(model_type, mask_rate, noise_sigma)
    A_hat, A_true, meta = load_coeff_pair_from_l2(z)
    return A_hat, A_true, meta, Path(npz_path)
