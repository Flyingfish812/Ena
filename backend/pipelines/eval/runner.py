# backend/pipelines/eval/runner.py
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from backend.pipelines.eval.context import EvalContext
from backend.pipelines.eval.registry import get_mod, resolve_mods


def run_eval_mods(
    ctx: EvalContext,
    *,
    assemble: Iterable[str],
    mod_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    save_index: bool = True,
) -> Dict[str, Any]:
    """
    统一的 L4 执行器（唯一入口执行链路）：

    - assemble: ["manifest", "quick_check", "fourier.kstar_curves", ...]
    - mod_kwargs: {"fourier.kstar_curves": {"max_plots": 30}, ...}

    输出：
      {
        "out_dir": ".../L4_eval",
        "results": {mod: {...}},
        "fig_paths": {mod: [png,...]}
      }
    """
    assert ctx.paths is not None, "EvalContext must be resolved before running mods."

    mod_kwargs = mod_kwargs or {}
    names = resolve_mods(list(assemble))

    results: Dict[str, Any] = {}
    fig_paths: Dict[str, List[str]] = {}

    for name in names:
        mod = get_mod(str(name))
        kwargs = dict(mod_kwargs.get(mod.name, {}) or {})

        print(f"[L4] run mod: {mod.name}")
        out = mod.run(ctx, kwargs) or {}
        results[mod.name] = out

        fps = out.get("fig_paths", None) if isinstance(out, dict) else None
        if fps is not None:
            if isinstance(fps, (list, tuple)):
                fig_paths[mod.name] = [str(x) for x in fps]
            elif isinstance(fps, str):
                fig_paths[mod.name] = [str(fps)]

    pack = {
        "out_dir": str(ctx.paths.l4_root),
        "results": results,
        "fig_paths": fig_paths,
    }

    if save_index:
        index_path = ctx.paths.l4_root / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)

    return pack
