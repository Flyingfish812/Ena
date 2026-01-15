# backend/pipelines/eval/start.py
# v2.0: Level-4 producer (metrics + plots) modular buffet-style

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.pipelines.eval.context import EvalContext
from backend.pipelines.eval.mods_builtin import register_builtin_mods
from backend.pipelines.eval.runner import run_eval_mods

from backend.pipelines.eval_mods.fourier_mods import register_fourier_mods
from backend.pipelines.eval_mods.examples_mods import register_example_mods
from backend.pipelines.eval_mods.scale_mods import register_scale_mods
from backend.pipelines.eval_mods.cumulate_mods import register_cumulate_mods


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

    # 统一上下文（唯一 EvalContext 实现）
    ctx = EvalContext(exp_dir=exp_dir).resolve(verbose=verbose)

    # 允许自定义 L4 输出目录名（默认 L4_eval）
    assert ctx.paths is not None
    if out_dirname != "L4_eval":
        ctx.paths.l4_root = (ctx.paths.exp_dir / out_dirname)  # type: ignore[attr-defined]
        ctx.paths.l4_root.mkdir(parents=True, exist_ok=True)

    # 统一注册表：eval/registry.py
    register_builtin_mods()
    register_fourier_mods()
    register_example_mods()
    register_scale_mods()
    register_cumulate_mods()

    # 统一执行器：eval/runner.py
    pack = run_eval_mods(
        ctx,
        assemble=list(assemble),
        mod_kwargs=mod_kwargs,
        save_index=True,
    )
    return pack