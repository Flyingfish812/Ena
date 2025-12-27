"""backend/pipelines/eval_mods/examples_io.py

Level-4 examples IO helpers.

This file is intentionally self-contained and only assumes the v2.0
Level-4 context defined in `backend/pipelines/eval_mods/core.py`:

  - ctx.exp_dir: Path
  - ctx.out_dir: Path
  - ctx.cfg: dict with keys {data_cfg, pod_cfg, eval_cfg, train_cfg}
  - ctx.caches: dict

No dependency on the legacy `backend.pipelines.eval.*` toolchain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import numpy as np

from backend.dataio.io_utils import ensure_dir


@dataclass(frozen=True)
class PodBasis:
    Ur: np.ndarray          # [D, r]
    mean_flat: np.ndarray   # [D]
    H: int
    W: int
    C: int


def _infer_pod_meta(pod_dir: Path) -> Tuple[int, int, int]:
    """
    推断 (H, W, C)。优先读 pod_dir/meta.json；读不到就报错。
    """
    meta_path = pod_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"POD meta.json not found under: {pod_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # 常见字段命名：H/W/C 或 height/width/channels
    H = int(meta.get("H", meta.get("height", 0)))
    W = int(meta.get("W", meta.get("width", 0)))
    C = int(meta.get("C", meta.get("channels", 0)))
    if H <= 0 or W <= 0 or C <= 0:
        raise ValueError(f"Invalid POD meta (H,W,C) from {meta_path}: {meta}")
    return H, W, C


def _load_pod_basis(pod_dir: Path) -> PodBasis:
    """
    加载 POD 基：Ur + mean_flat
    约定：pod_dir/Ur.npy 和 pod_dir/mean.npy（或 mean_flat.npy）
    """
    cand_ur = [pod_dir / "Ur.npy", pod_dir / "Ur_eff.npy"]
    cand_mean = [pod_dir / "mean.npy", pod_dir / "mean_flat.npy"]

    ur_path = next((p for p in cand_ur if p.exists()), None)
    if ur_path is None:
        raise FileNotFoundError(f"Cannot find Ur basis under {pod_dir}: tried {cand_ur}")

    mean_path = next((p for p in cand_mean if p.exists()), None)
    if mean_path is None:
        raise FileNotFoundError(f"Cannot find mean under {pod_dir}: tried {cand_mean}")

    Ur = np.load(ur_path)
    mean_flat = np.load(mean_path)

    H, W, C = _infer_pod_meta(pod_dir)

    if mean_flat.ndim != 1:
        mean_flat = mean_flat.reshape(-1)
    if Ur.ndim != 2:
        raise ValueError(f"Ur must be 2D, got shape={Ur.shape} from {ur_path}")

    D = C * H * W
    if Ur.shape[0] != D or mean_flat.shape[0] != D:
        raise ValueError(
            f"POD basis shape mismatch: D={D}, Ur={Ur.shape}, mean={mean_flat.shape}"
        )

    return PodBasis(Ur=Ur, mean_flat=mean_flat, H=H, W=W, C=C)


def _pick_l2_root(exp_dir: Path) -> Path:
    """Pick Level-2 root directory.

    v2.0 convention (as documented in `pipelines_code.md`):
      - exp_dir/L2
      - exp_dir/L2_rebuild
    """
    cand = [exp_dir / "L2", exp_dir / "L2_rebuild"]
    for p in cand:
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        f"Cannot find Level-2 root under {exp_dir} (expected L2/ or L2_rebuild/). Tried: {cand}"
    )


def _l2_entry_path(
    l2_root: Path,
    *,
    model_type: str,
    mask_rate: float,
    noise_sigma: float,
    scale: int = 10000,
) -> Path:
    """Flat filename convention: <model>_pXXXX_sXXXX.npz"""
    p_code = int(np.round(float(mask_rate) * float(scale)))
    s_code = int(np.round(float(noise_sigma) * float(scale)))
    return l2_root / f"{str(model_type)}_p{p_code:04d}_s{s_code:04d}.npz"


def load_l2_npz(
    ctx: Any,
    *,
    model_type: str,
    mask_rate: float,
    noise_sigma: float,
) -> "np.lib.npyio.NpzFile":
    """Load a Level-2 entry NPZ (read-only handle)."""
    exp_dir = Path(getattr(ctx, "exp_dir"))
    l2_root = _pick_l2_root(exp_dir)
    p = _l2_entry_path(l2_root, model_type=model_type, mask_rate=mask_rate, noise_sigma=noise_sigma)
    if not p.exists():
        raise FileNotFoundError(f"L2 entry not found: {p}")
    return np.load(p)


def _pick_frames(T: int, *, sample_frames: int, seed: int = 0) -> List[int]:
    """
    sample_frames:
      -1 => all
      >0 => uniform-ish sample
    """
    if sample_frames == -1 or sample_frames >= T:
        return list(range(T))
    if sample_frames <= 0:
        return [0]

    rng = np.random.default_rng(seed)
    # 更稳一点：先均匀取，再随机补齐（避免 T 很大时全挤在前面）
    base = np.linspace(0, T - 1, num=sample_frames, dtype=int).tolist()
    base = sorted(set(base))
    while len(base) < sample_frames:
        base.append(int(rng.integers(0, T)))
        base = sorted(set(base))
    return base[:sample_frames]


def reconstruct_fields_from_l2(
    ctx: Any,
    *,
    model_type: str,
    mask_rate: float,
    noise_sigma: float,
    channel: int = 0,
    sample_frames: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    从 L2 的系数重建空间场（按需取帧）。

    返回：
      {
        "H","W","C","T","frames",
        "x_pred": [N, H, W],
        "x_true": [N, H, W],
        "x_err":  [N, H, W],
        "mask_hw": [H, W] 或 None,
      }
    """
    # ---- cache: POD basis ----
    pod_cfg = (getattr(ctx, "cfg", None) or {}).get("pod_cfg", None)
    pod_dir = Path(getattr(pod_cfg, "save_dir", "artifacts/pod"))
    cache_key = ("pod_basis", str(pod_dir))
    if cache_key in ctx.caches:
        basis: PodBasis = ctx.caches[cache_key]
    else:
        basis = _load_pod_basis(pod_dir)
        ctx.caches[cache_key] = basis

    # ---- load L2 npz ----
    # 约定：Level-2 NPZ 至少包含 A_hat_all / A_true_all（或 A_hat / A_true）
    with load_l2_npz(ctx, model_type=model_type, mask_rate=float(mask_rate), noise_sigma=float(noise_sigma)) as z:
        A_hat_raw = z.get("A_hat_all", z.get("A_hat", None))
        A_true_raw = z.get("A_true_all", z.get("A_true", None))
        mf_raw = z.get("mask_flat", None)
        if A_hat_raw is None or A_true_raw is None:
            raise KeyError(
                f"L2 npz missing A_hat/A_true for {(model_type, mask_rate, noise_sigma)}"
            )
        A_hat = np.asarray(A_hat_raw)
        A_true = np.asarray(A_true_raw)
        mask_flat = np.asarray(mf_raw).reshape(-1) if mf_raw is not None else None

    if A_hat.ndim != 2 or A_true.ndim != 2:
        raise ValueError(f"A_hat/A_true must be [T,r]. got {A_hat.shape}, {A_true.shape}")

    T = int(A_hat.shape[0])
    frames = _pick_frames(T, sample_frames=sample_frames, seed=seed)

    # ---- reconstruct flat -> THWC ----
    Ur = basis.Ur  # [D,r]
    mean = basis.mean_flat  # [D]
    H, W, C = basis.H, basis.W, basis.C

    # 只重建选中的帧：减少开销
    A_hat_sel = A_hat[frames]  # [N,r]
    A_true_sel = A_true[frames]  # [N,r]

    # X = Ur @ a + mean
    # [D,r] @ [N,r].T -> [D,N] -> [N,D]
    X_hat_flat = (Ur @ A_hat_sel.T).T + mean[None, :]
    X_true_flat = (Ur @ A_true_sel.T).T + mean[None, :]

    X_hat_thwc = X_hat_flat.reshape(len(frames), C, H, W).transpose(0, 2, 3, 1)  # [N,H,W,C]
    X_true_thwc = X_true_flat.reshape(len(frames), C, H, W).transpose(0, 2, 3, 1)

    if not (0 <= channel < C):
        raise ValueError(f"channel out of range: {channel} (C={C})")

    x_pred = X_hat_thwc[..., channel]  # [N,H,W]
    x_true = X_true_thwc[..., channel]
    x_err = x_pred - x_true

    mask_hw = None
    if mask_flat is not None:
        if mask_flat.size == H * W:
            mask_hw = mask_flat.reshape(H, W)
        # 如果你存的是 C*H*W 的 mask，也可以扩展这里做降维

    return {
        "H": H,
        "W": W,
        "C": C,
        "T": T,
        "frames": frames,
        "x_pred": x_pred,
        "x_true": x_true,
        "x_err": x_err,
        "mask_hw": mask_hw,
    }


def ensure_subdir(root: Path, name: str) -> Path:
    p = root / name
    ensure_dir(p)
    return p
