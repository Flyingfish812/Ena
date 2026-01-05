# backend/pipelines/eval_mods/examples_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from backend.pipelines.eval.utils import read_json  # 你已有的工具
from backend.dataio.io_utils import ensure_dir  # 你项目里已有


@dataclass(frozen=True)
class PodBasis:
    Ur: np.ndarray          # [D, r]
    mean_flat: np.ndarray   # [D]
    H: int
    W: int
    C: int
    D: int


def _infer_pod_meta(pod_dir: Path) -> Tuple[int, int, int, int]:
    """
    从 pod_dir/pod_meta.json 推断 (H, W, C, D)。
    这是 L4 系统的唯一 POD 元信息来源（不兼容旧字段）。
    """
    meta_path = pod_dir / "pod_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"POD pod_meta.json not found under: {pod_dir}")

    meta = read_json(meta_path)

    H = int(meta["H"])
    W = int(meta["W"])
    C = int(meta["C"])
    D = int(meta["D"])

    if H <= 0 or W <= 0 or C <= 0 or D <= 0:
        raise ValueError(f"Invalid POD meta from {meta_path}: {meta}")

    # 严格一致性检查：D 必须等于 H*W*C
    D_expected = H * W * C
    if D != D_expected:
        raise ValueError(
            f"POD meta inconsistent: D={D} but H*W*C={D_expected} "
            f"(H={H}, W={W}, C={C}) from {meta_path}"
        )

    return H, W, C, D


def _load_pod_basis(pod_dir: Path) -> PodBasis:
    """
    L4 统一 POD basis 加载：
      - pod_dir/pod_meta.json
      - pod_dir/Ur.npy
      - pod_dir/mean_flat.npy

    不兼容旧文件名。
    """
    ur_path = pod_dir / "Ur.npy"
    mean_path = pod_dir / "mean_flat.npy"

    if not ur_path.exists():
        raise FileNotFoundError(f"Cannot find POD basis: {ur_path}")
    if not mean_path.exists():
        raise FileNotFoundError(f"Cannot find POD mean: {mean_path}")

    Ur = np.load(ur_path)
    mean_flat = np.load(mean_path)

    if Ur.ndim != 2:
        raise ValueError(f"Ur must be 2D [D,r], got shape={Ur.shape} from {ur_path}")
    mean_flat = np.asarray(mean_flat).reshape(-1)

    H, W, C, D = _infer_pod_meta(pod_dir)

    if Ur.shape[0] != D:
        raise ValueError(f"Ur shape mismatch: Ur.shape[0]={Ur.shape[0]} != D={D} from pod_meta.json")
    if mean_flat.shape[0] != D:
        raise ValueError(f"mean_flat mismatch: mean_flat.shape[0]={mean_flat.shape[0]} != D={D} from pod_meta.json")

    return PodBasis(Ur=Ur, mean_flat=mean_flat, H=H, W=W, C=C, D=D)


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

    依赖统一引擎 EvalContext 的接口：
      - ctx.paths.l2_root / ctx.paths.l4_root
      - ctx.load_l2(model_type, mask_rate, noise_sigma)
      - ctx.pod_cfg.save_dir
      - ctx.eval_cfg / ctx.iter_cfgs / ctx.model_types 等（由上层 mod 用）

    返回：
      {
        "H","W","C","T","frames",
        "x_pred": [N, H, W],
        "x_true": [N, H, W],
        "x_err":  [N, H, W],
        "mask_hw": [H, W] 或 None,
      }
    """
    assert ctx.paths is not None

    # ---- make sure caches exists (engine统一字段；若你还没加，就在 ctx 上挂一个) ----
    if not hasattr(ctx, "caches") or getattr(ctx, "caches") is None:
        setattr(ctx, "caches", {})
    caches: Dict[Any, Any] = getattr(ctx, "caches")

    # ---- cache: POD basis ----
    pod_dir = Path(getattr(ctx.pod_cfg, "save_dir", "artifacts/pod"))
    cache_key = ("pod_basis", str(pod_dir))
    if cache_key in caches:
        basis: PodBasis = caches[cache_key]
    else:
        basis = _load_pod_basis(pod_dir)
        caches[cache_key] = basis

    # ---- load L2 npz ----
    z = ctx.load_l2(str(model_type), float(mask_rate), float(noise_sigma))
    A_hat = z.get("A_hat_all", z.get("A_hat", None))
    A_true = z.get("A_true_all", z.get("A_true", None))

    if A_hat is None or A_true is None:
        raise KeyError(f"L2 npz missing A_hat/A_true for {(model_type, mask_rate, noise_sigma)}")

    A_hat = np.asarray(A_hat)
    A_true = np.asarray(A_true)

    if A_hat.ndim != 2 or A_true.ndim != 2:
        raise ValueError(f"A_hat/A_true must be [T,r]. got {A_hat.shape}, {A_true.shape}")

    T = int(A_hat.shape[0])
    frames = _pick_frames(T, sample_frames=sample_frames, seed=seed)

    # ---- reconstruct flat -> [N,H,W,C] ----
    Ur = basis.Ur
    mean = basis.mean_flat
    H, W, C, D = basis.H, basis.W, basis.C, basis.D

    A_hat_sel = A_hat[frames]
    A_true_sel = A_true[frames]

    X_hat_flat = (Ur @ A_hat_sel.T).T + mean[None, :]
    X_true_flat = (Ur @ A_true_sel.T).T + mean[None, :]

    X_hat_thwc = X_hat_flat.reshape(len(frames), H, W, C)
    X_true_thwc = X_true_flat.reshape(len(frames), H, W, C)

    if not (0 <= channel < C):
        raise ValueError(f"channel out of range: {channel} (C={C})")

    x_pred = X_hat_thwc[..., channel]
    x_true = X_true_thwc[..., channel]
    x_err = x_pred - x_true

    mask_hw = None
    mf = z.get("mask_flat", None)
    if mf is not None:
        mf = np.asarray(mf).reshape(-1)
        expected_hw = H * W
        expected_hwc = H * W * C
        print(f"[reconstruct] mask_flat size: {mf.size}, expected hw: {expected_hw}, expected hwc: {expected_hwc}")

        if mf.size == expected_hw:
            mask_hw = mf.reshape(H, W).astype(bool)

        elif mf.size == expected_hwc:
            # mask stored per-channel: [H, W, C] flattened
            mask_hwc = mf.reshape(H, W, C).astype(bool)
            # reduce to position mask: sampled if any channel sampled
            mask_hw = np.any(mask_hwc, axis=2)

        else:
            print(f"[warn] Unrecognized mask_flat size {mf.size}; cannot convert to HxW mask.")
            mask_hw = None

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
