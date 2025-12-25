# backend/pipeline_train.py
# v2.0 draft: training + reconstruction (Level-2 artifacts) pipeline entry for ipynb

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.config.yaml_io import load_experiment_yaml, save_experiment_yaml
from backend.dataio.io_utils import ensure_dir

from backend.eval.rebuild import run_rebuild_sweep
from backend.eval.pre_analysis import run_pre_analysis_from_storage

import json
import numpy as np

# ------------------------------
# Artifacts storage convention (draft)
#
# exp_dir/
#   config_used.yaml
#   L2_rebuild/
#     meta.json
#     linear/
#       p=..._s=.../
#         pred_coeffs.npz
#         obs.npz
#         entry.json
#     mlp/...
#   L3_fft/
#     meta.json
#     linear/...
# ------------------------------


# ============================================================
# compute_level2_rebuild
#
# Level-2 producer (train + predict + save raw artifacts).
#
# Responsibilities:
#   - 读取 YAML -> (data_cfg, pod_cfg, eval_cfg, train_cfg)
#   - 调用 rebuild 层：训练/推理并保存 Level-2 产物（原始预测结果 + 训练元信息）
#   - 训练完成后再把本次实际使用的配置写入 exp_dir/config_used.yaml 方便溯源
#
# Notes:
#   - 不计算 FFT，不做任何指标统计
# ============================================================
def compute_level2_rebuild(
    yaml_path: str | Path,
    *,
    experiment_name: str | None = None,
    save_root: str | Path = "artifacts/experiments",
    verbose: bool = True,
    save_used_yaml: bool = True,
) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)
    if experiment_name is None:
        experiment_name = yaml_path.stem

    save_root = Path(save_root)
    exp_dir = save_root / str(experiment_name)
    ensure_dir(exp_dir)

    # Load configs from yaml
    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    # Level-2: train + predict + save raw artifacts
    l2_index = run_rebuild_sweep(
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
        exp_dir=exp_dir,
        verbose=verbose,
    )

    # Save the effective config AFTER training finishes
    if save_used_yaml:
        try:
            save_experiment_yaml(
                path=exp_dir / "config_used.yaml",
                data_cfg=data_cfg,
                pod_cfg=pod_cfg,
                eval_cfg=eval_cfg,
                train_cfg=train_cfg,
            )
        except Exception as e:
            if verbose:
                print(f"[L2] WARNING: failed to save config_used.yaml: {e}")

    return {
        "yaml_path": str(yaml_path),
        "experiment_name": str(experiment_name),
        "exp_dir": str(exp_dir),
        "L2": l2_index,
        "eval_cfg": eval_cfg,
    }


# ============================================================
# compute_level3_fft_from_level2
#
# Level-3 entrypoint (ipynb):
#   - Locate config_used.yaml under exp_dir
#   - Load eval_cfg from yaml
#   - Infer model_types from L2 meta/index if not provided
#   - Run pre-analysis from storage and save L3 artifacts
# ============================================================
def compute_level3_fft_from_level2(
    *,
    exp_dir: str | Path,
    model_types: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    exp_dir = Path(exp_dir)
    ensure_dir(exp_dir)

    # 1) Load configs from saved experiment package (prefer config_used.yaml)
    cfg_candidates = [
        exp_dir / "config_used.yaml",
        exp_dir / "config.yaml",
        exp_dir / "experiment.yaml",
    ]
    cfg_path = None
    for p in cfg_candidates:
        if p.exists():
            cfg_path = p
            break
    if cfg_path is None:
        raise FileNotFoundError(
            f"Cannot find config_used.yaml (or config.yaml/experiment.yaml) under: {exp_dir}"
        )

    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(cfg_path)

    # 2) Infer model_types from L2 package if not specified
    if model_types is None:
        # Prefer L2/meta.json for model_types
        l2_meta_candidates = [
            exp_dir / "L2" / "meta.json",
            exp_dir / "L2_rebuild" / "meta.json",
        ]
        mt = None
        for mp in l2_meta_candidates:
            if mp.exists():
                try:
                    with open(mp, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    mt = meta.get("model_types", None)
                except Exception:
                    mt = None
                break

        if mt is None:
            # Fall back: if train_cfg exists, assume linear+mlp, else linear only
            model_types = ("linear", "mlp") if train_cfg is not None else ("linear",)
        else:
            model_types = tuple(str(x) for x in mt)

    # 3) Run Level-3 pre-analysis (FFT package) from stored L2 artifacts
    l3_index = run_pre_analysis_from_storage(
        exp_dir=exp_dir,
        eval_cfg=eval_cfg,
        model_types=tuple(model_types),
        verbose=verbose,
    )

    return {
        "exp_dir": str(exp_dir),
        "config_used": str(cfg_path),
        "model_types": list(model_types),
        "L3": l3_index,
    }


# ============================================================
# compute_full_eval_results (compat wrapper)
#
# Backward-style orchestration:
#   - 先跑 L2，再（可选）跑 L3
#
# Notes:
#   - 仍然保持“一个函数完成全流程”的用法，但底层已经是 L2/L3 分层
# ============================================================
def compute_full_eval_results(
    yaml_path: str | Path,
    *,
    experiment_name: str | None = None,
    save_root: str | Path = "artifacts/experiments",
    run_pre_analysis: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    # Run Level-2 first
    l2_pack = compute_level2_rebuild(
        yaml_path=yaml_path,
        experiment_name=experiment_name,
        save_root=save_root,
        verbose=verbose,
    )

    # Optionally run Level-3 from stored Level-2
    l3_pack = None
    if run_pre_analysis:
        l3_pack = compute_level3_fft_from_level2(
            eval_cfg=l2_pack["eval_cfg"],
            exp_dir=l2_pack["exp_dir"],
            l2_index=l2_pack["L2"],
            verbose=verbose,
        )

    return {
        "yaml_path": l2_pack["yaml_path"],
        "experiment_name": l2_pack["experiment_name"],
        "exp_dir": l2_pack["exp_dir"],
        "L2": l2_pack["L2"],
        "L3": None if l3_pack is None else l3_pack["L3"],
    }

# ============================================================
# Final Checks
# ============================================================

def check_level123_artifacts_ready(
    yaml_path: str | Path,
    *,
    experiment_name: str | None = None,
    save_root: str | Path = "artifacts/experiments",
    require_level: int = 3,
    model_types: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    # One-click checker for Level-1/2/3 artifacts readiness.
    # require_level:
    #   1 -> only check POD artifacts
    #   2 -> check POD + L2
    #   3 -> check POD + L2 + L3

    if require_level not in (1, 2, 3):
        raise ValueError("require_level must be 1, 2, or 3")

    yaml_path = Path(yaml_path)
    if experiment_name is None:
        experiment_name = yaml_path.stem

    save_root = Path(save_root)
    exp_dir = save_root / str(experiment_name)

    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    # Decide model_types (prefer caller override, else infer from train_cfg)
    if model_types is None:
        model_types = ("linear", "mlp") if train_cfg is not None else ("linear",)
    model_types = tuple(str(x) for x in model_types)

    report: Dict[str, Any] = {
        "yaml_path": str(yaml_path),
        "experiment_name": str(experiment_name),
        "exp_dir": str(exp_dir),
        "require_level": int(require_level),
        "model_types": list(model_types),
        "levels": {},
        "ok": True,
    }

    # -------- L1 (POD) --------
    l1 = _check_level1_pod(pod_dir=Path(pod_cfg.save_dir), verbose=verbose)
    report["levels"]["L1"] = l1
    if not l1["ok"]:
        report["ok"] = False
        if require_level == 1:
            _print_check_report(report, verbose=verbose)
            return report

    # -------- L2 (raw predictions) --------
    if require_level >= 2:
        l2 = _check_level2_pack(
            exp_dir=exp_dir,
            eval_cfg=eval_cfg,
            model_types=model_types,
            verbose=verbose,
        )
        report["levels"]["L2"] = l2
        if not l2["ok"]:
            report["ok"] = False
            if require_level == 2:
                _print_check_report(report, verbose=verbose)
                return report

    # -------- L3 (FFT packages) --------
    if require_level >= 3:
        l3 = _check_level3_pack(
            exp_dir=exp_dir,
            eval_cfg=eval_cfg,
            model_types=model_types,
            verbose=verbose,
        )
        report["levels"]["L3"] = l3
        if not l3["ok"]:
            report["ok"] = False

    _print_check_report(report, verbose=verbose)
    return report


def _check_level1_pod(*, pod_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    # Check POD artifacts existence.
    pod_dir = Path(pod_dir)

    required = [
        pod_dir / "Ur.npy",
        pod_dir / "mean_flat.npy",
        pod_dir / "pod_meta.json",
    ]
    missing = [str(p) for p in required if not p.exists()]

    ok = (len(missing) == 0)
    out = {
        "ok": ok,
        "pod_dir": str(pod_dir),
        "missing": missing,
    }

    if ok:
        # Optional: light sanity (no heavy loads)
        out["present"] = [str(p) for p in required]
    return out


def _check_level2_pack(
    *,
    exp_dir: Path,
    eval_cfg,
    model_types: Sequence[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    # Check Level-2 artifacts under exp_dir/L2 (flat layout) or exp_dir/L2_rebuild.
    exp_dir = Path(exp_dir)

    l2_root = None
    if (exp_dir / "L2").exists():
        l2_root = exp_dir / "L2"
    elif (exp_dir / "L2_rebuild").exists():
        # allow legacy folder name (but we still expect flat npz inside after your upgrade)
        l2_root = exp_dir / "L2_rebuild"

    if l2_root is None:
        return {
            "ok": False,
            "l2_root": None,
            "missing": [str(exp_dir / "L2")],
            "found_npz": [],
            "expected_npz": [],
        }

    expected_npz = _expected_entry_filenames(
        model_types=model_types,
        mask_rates=getattr(eval_cfg, "mask_rates", []),
        noise_sigmas=getattr(eval_cfg, "noise_sigmas", []),
        prefix_style="l2",  # filename pattern is the same for L2/L3
    )
    found_npz = sorted([p.name for p in l2_root.glob("*.npz") if p.is_file()])

    missing_npz = [name for name in expected_npz if not (l2_root / name).exists()]

    # meta/index are helpful but not strictly required for “ready to proceed”
    meta_path = l2_root / "meta.json"
    has_meta = meta_path.exists()

    ok = (len(missing_npz) == 0)

    return {
        "ok": ok,
        "l2_root": str(l2_root),
        "has_meta": bool(has_meta),
        "meta_path": str(meta_path),
        "expected_npz_count": int(len(expected_npz)),
        "found_npz_count": int(len(found_npz)),
        "missing_npz_count": int(len(missing_npz)),
        "missing_npz": missing_npz[:50],  # cap to avoid huge prints
        "expected_npz": expected_npz[:50],
        "found_npz": found_npz[:50],
    }


def _check_level3_pack(
    *,
    exp_dir: Path,
    eval_cfg,
    model_types: Sequence[str],
    verbose: bool = True,
) -> Dict[str, Any]:
    # Check Level-3 FFT packages under exp_dir/L3_fft (flat layout).
    exp_dir = Path(exp_dir)
    l3_root = exp_dir / "L3_fft"
    if not l3_root.exists():
        return {
            "ok": False,
            "l3_root": str(l3_root),
            "missing": [str(l3_root)],
            "found_npz": [],
            "expected_npz": [],
        }

    # If Fourier disabled, consider L3 “not required”
    fourier_cfg = getattr(eval_cfg, "fourier", None)
    if fourier_cfg is None or not bool(getattr(fourier_cfg, "enabled", False)):
        return {
            "ok": True,
            "l3_root": str(l3_root),
            "note": "eval_cfg.fourier.enabled is False -> L3 not required",
            "expected_npz_count": 0,
            "found_npz_count": len(list(l3_root.glob("*.npz"))),
        }

    expected_npz = _expected_entry_filenames(
        model_types=model_types,
        mask_rates=getattr(eval_cfg, "mask_rates", []),
        noise_sigmas=getattr(eval_cfg, "noise_sigmas", []),
        prefix_style="l3",  # same filename pattern
    )
    found_npz = sorted([p.name for p in l3_root.glob("*.npz") if p.is_file()])
    missing_npz = [name for name in expected_npz if not (l3_root / name).exists()]

    meta_path = l3_root / "meta.json"
    index_path = l3_root / "index.json"
    has_meta = meta_path.exists()
    has_index = index_path.exists()

    ok = (len(missing_npz) == 0) and has_meta and has_index

    return {
        "ok": ok,
        "l3_root": str(l3_root),
        "has_meta": bool(has_meta),
        "has_index": bool(has_index),
        "meta_path": str(meta_path),
        "index_path": str(index_path),
        "expected_npz_count": int(len(expected_npz)),
        "found_npz_count": int(len(found_npz)),
        "missing_npz_count": int(len(missing_npz)),
        "missing_npz": missing_npz[:50],
        "expected_npz": expected_npz[:50],
        "found_npz": found_npz[:50],
    }


def _expected_entry_filenames(
    *,
    model_types: Sequence[str],
    mask_rates: Sequence[float],
    noise_sigmas: Sequence[float],
    prefix_style: str,
    scale: int = 10000,
) -> List[str]:
    # Generate flat filenames: <model>_pXXXX_sXXXX.npz
    # This matches your L2/L3 flat storage convention.
    names: List[str] = []
    for mt in model_types:
        for p in mask_rates:
            for s in noise_sigmas:
                p_code = int(np.round(float(p) * float(scale)))
                s_code = int(np.round(float(s) * float(scale)))
                names.append(f"{str(mt)}_p{p_code:04d}_s{s_code:04d}.npz")
    return names


def _print_check_report(report: Dict[str, Any], *, verbose: bool = True) -> None:
    # Pretty printer for check results.
    if not verbose:
        return

    print("\n=== [check] Level-1/2/3 artifacts readiness ===")
    print(f"- yaml: {report.get('yaml_path')}")
    print(f"- exp : {report.get('exp_dir')}")
    print(f"- require_level: {report.get('require_level')} | model_types: {report.get('model_types')}")
    print(f"- overall ok: {report.get('ok')}")

    levels = report.get("levels", {})
    for lvl in ["L1", "L2", "L3"]:
        if lvl not in levels:
            continue
        info = levels[lvl]
        ok = bool(info.get("ok", False))
        print(f"\n[{lvl}] ok={ok}")
        for k in ["pod_dir", "l2_root", "l3_root", "has_meta", "has_index", "note"]:
            if k in info:
                print(f"  - {k}: {info[k]}")
        for k in ["expected_npz_count", "found_npz_count", "missing_npz_count"]:
            if k in info:
                print(f"  - {k}: {info[k]}")
        if info.get("missing", None):
            print(f"  - missing: {info['missing']}")
        if info.get("missing_npz_count", 0) > 0:
            print("  - missing_npz (first 20):")
            for name in info.get("missing_npz", [])[:20]:
                print(f"      * {name}")