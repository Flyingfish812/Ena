# backend/pipelines/eval/mods_builtin.py
from __future__ import annotations

from typing import Any, Dict, List

from backend.pipelines.eval.context import EvalContext
from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval.utils import write_json


def _mod_manifest(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    assert ctx.paths is not None

    l2_meta = ctx.l2_meta()
    l3_meta = ctx.l3_meta() if ctx.paths.l3_root.exists() else {}
    l3_index = ctx.l3_index() if (ctx.paths.l3_root / "index.json").exists() else {}

    l2_count = len(list(ctx.paths.l2_root.glob("*.npz")))
    l3_count = len(list(ctx.paths.l3_root.glob("*.npz"))) if ctx.paths.l3_root.exists() else 0

    cfg_grid = [(p, s) for (p, s) in ctx.iter_cfgs()]

    payload = {
        "schema_version": "v2.0-L4",
        "exp_dir": str(ctx.paths.exp_dir),
        "config_used": str(ctx.paths.cfg_path),
        "roots": {
            "L2": str(ctx.paths.l2_root),
            "L3_fft": str(ctx.paths.l3_root),
            "L4_eval": str(ctx.paths.l4_root),
        },
        "model_types": list(ctx.model_types or ()),
        "grid": {
            "mask_rates": [float(x) for x in getattr(ctx.eval_cfg, "mask_rates", [])],
            "noise_sigmas": [float(x) for x in getattr(ctx.eval_cfg, "noise_sigmas", [])],
            "count": int(len(cfg_grid)),
        },
        "l2": {
            "meta_path": str(ctx.paths.l2_root / "meta.json"),
            "has_meta": (ctx.paths.l2_root / "meta.json").exists(),
            "npz_count": int(l2_count),
            "meta": l2_meta,
        },
        "l3": {
            "enabled": bool(l3_meta.get("enabled", False)) if l3_meta else False,
            "meta_path": str(ctx.paths.l3_root / "meta.json"),
            "index_path": str(ctx.paths.l3_root / "index.json"),
            "has_meta": (ctx.paths.l3_root / "meta.json").exists(),
            "has_index": (ctx.paths.l3_root / "index.json").exists(),
            "npz_count": int(l3_count),
            "meta": l3_meta,
            "index": l3_index,
        },
    }

    out_path = ctx.paths.l4_root / "manifest.json"
    write_json(out_path, payload)
    return {"written": {"manifest_json": str(out_path)}, "manifest": payload}


def _mod_quick_check(ctx: EvalContext, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    assert ctx.paths is not None

    missing: List[str] = []

    # L2
    if not ctx.paths.l2_root.exists():
        missing.append(str(ctx.paths.l2_root))
    else:
        if len(list(ctx.paths.l2_root.glob("*.npz"))) == 0:
            missing.append(str(ctx.paths.l2_root / "*.npz"))

    # L3 (only if enabled)
    fourier_cfg = getattr(ctx.eval_cfg, "fourier", None)
    fourier_enabled = bool(getattr(fourier_cfg, "enabled", False)) if fourier_cfg is not None else False
    if fourier_enabled:
        if not ctx.paths.l3_root.exists():
            missing.append(str(ctx.paths.l3_root))
        else:
            if not (ctx.paths.l3_root / "meta.json").exists():
                missing.append(str(ctx.paths.l3_root / "meta.json"))
            if not (ctx.paths.l3_root / "index.json").exists():
                missing.append(str(ctx.paths.l3_root / "index.json"))

    rep = {"ok": (len(missing) == 0), "fourier_enabled": bool(fourier_enabled), "missing": missing}
    out_path = ctx.paths.l4_root / "quick_check.json"
    write_json(out_path, rep)

    return {"written": {"quick_check_json": str(out_path)}, "report": rep}


def register_builtin_mods() -> None:
    register_mod(
        EvalMod(
            name="manifest",
            requires=(),
            description="Write Level-4 index/manifest JSON (paths, meta, inventory).",
            run=_mod_manifest,
        )
    )
    register_mod(
        EvalMod(
            name="quick_check",
            requires=(),
            description="Lightweight existence checks for L1/L2/L3 (no computation).",
            run=_mod_quick_check,
        )
    )
