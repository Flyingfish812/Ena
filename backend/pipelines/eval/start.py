# backend/pipelines/eval/start.py
# v2.0: Level-4 producer (metrics + plots) modular buffet-style

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.config.yaml_io import load_experiment_yaml
from backend.dataio.io_utils import ensure_dir

from backend.pipelines.eval_mods import (
    EvalContext,
    ModRegistry,
    run_mods,
    register_fourier_mods,
)


def _pick_config_used_yaml(exp_dir: Path) -> Path:
    cand = [
        exp_dir / "config_used.yaml",
        exp_dir / "config.yaml",
        exp_dir / "experiment.yaml",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Cannot find config_used.yaml (or config.yaml/experiment.yaml) under: {exp_dir}"
    )


def compute_level4_eval_mods(
    *,
    exp_dir: str | Path,
    assemble: Iterable[str],
    mod_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    out_dirname: str = "L4_eval",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Level-4 主入口（ipynb 调用）：
      exp_dir: 实验目录（已存在 L2/L3）
      assemble: 需要运行的模组声明列表
      mod_kwargs: 对某些模组传参（例如 max_plots）
    """
    exp_dir = Path(exp_dir)
    ensure_dir(exp_dir)

    cfg_path = _pick_config_used_yaml(exp_dir)
    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(cfg_path)

    out_dir = exp_dir / out_dirname
    ensure_dir(out_dir)

    ctx = EvalContext(
        exp_dir=exp_dir,
        out_dir=out_dir,
        cfg={
            "config_used": str(cfg_path),
            "data_cfg": data_cfg,
            "pod_cfg": pod_cfg,
            "eval_cfg": eval_cfg,
            "train_cfg": train_cfg,
        },
        verbose=verbose,
    )

    registry = ModRegistry()

    # 批次2：注册 Fourier 模组
    register_fourier_mods(registry)

    # 后续批次你会继续 register_*_mods(...)
    pack = run_mods(
        ctx,
        registry,
        assemble=list(assemble),
        mod_kwargs=mod_kwargs,
        save_index=True,
    )
    return pack
