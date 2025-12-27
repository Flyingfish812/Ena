# backend/pipelines/eval/utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np

from backend.dataio.io_utils import ensure_dir, load_json


def encode_rate(x: float, *, scale: int = 10000) -> int:
    return int(np.round(float(x) * float(scale)))


def entry_name(model_type: str, mask_rate: float, noise_sigma: float, *, scale: int = 10000) -> str:
    p_code = encode_rate(mask_rate, scale=scale)
    s_code = encode_rate(noise_sigma, scale=scale)
    return f"{str(model_type)}_p{p_code:04d}_s{s_code:04d}.npz"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return load_json(path)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def unwrap_np_scalar(v: Any) -> Any:
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def load_npz(path: Path, *, allow_pickle: bool = False) -> Dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=allow_pickle) as z:
        out = {k: unwrap_np_scalar(z[k]) for k in z.files}
    out["_path"] = str(path)
    out["_keys"] = [k for k in out.keys() if not k.startswith("_")]
    return out


def pick_config_yaml(exp_dir: Path) -> Path:
    cand = [
        exp_dir / "config_used.yaml",
        exp_dir / "config.yaml",
        exp_dir / "experiment.yaml",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find config_used.yaml (or config.yaml/experiment.yaml) under: {exp_dir}")


def pick_l2_root(exp_dir: Path) -> Path:
    cand = [exp_dir / "L2", exp_dir / "L2_rebuild"]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find Level-2 root under {exp_dir} (expected L2/ or L2_rebuild/).")


def pick_l3_root(exp_dir: Path) -> Path:
    p = exp_dir / "L3_fft"
    if not p.exists():
        raise FileNotFoundError(f"Cannot find Level-3 root under {exp_dir} (expected L3_fft/).")
    return p


def ensure_dir_path(p: Path) -> Path:
    ensure_dir(p)
    return p


def parse_flat_name(stem: str, *, scale: float = 10000.0) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    # <model>_pXXXX_sXXXX
    if "_p" not in stem or "_s" not in stem:
        return None, None, None
    try:
        mt = stem.split("_p", 1)[0]
        rest = stem.split("_p", 1)[1]
        p_code_str, s_part = rest.split("_s", 1)

        s_code_str = s_part
        for sep in ["_", "-"]:
            if sep in s_code_str:
                s_code_str = s_code_str.split(sep, 1)[0]

        p_code = int(p_code_str)
        s_code = int(s_code_str)
        return str(mt), float(p_code) / scale, float(s_code) / scale
    except Exception:
        return None, None, None
