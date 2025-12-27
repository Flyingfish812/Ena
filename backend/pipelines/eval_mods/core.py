# backend/pipelines/eval_mods/core.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import json


@dataclass
class EvalContext:
    """
    Level-4 评估上下文（storage-driven）：
      - exp_dir：实验根目录
      - out_dir：L4 输出目录
      - cfg：从 config_used.yaml 读出来的 (data_cfg, pod_cfg, eval_cfg, train_cfg)
      - caches：模组之间共享缓存（读盘一次，多处复用）
    """
    exp_dir: Path
    out_dir: Path
    cfg: Dict[str, Any] = field(default_factory=dict)
    caches: Dict[str, Any] = field(default_factory=dict)
    verbose: bool = True

    def log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


@dataclass(frozen=True)
class EvalMod:
    name: str
    fn: Callable[[EvalContext, Dict[str, Any]], Dict[str, Any]]
    default_enabled: bool = True
    description: str = ""


class ModRegistry:
    def __init__(self) -> None:
        self._mods: Dict[str, EvalMod] = {}

    def register(self, mod: EvalMod) -> None:
        if mod.name in self._mods:
            raise ValueError(f"Duplicate mod name: {mod.name}")
        self._mods[mod.name] = mod

    def get(self, name: str) -> EvalMod:
        if name not in self._mods:
            raise KeyError(f"Unknown mod: {name}. Available: {sorted(self._mods.keys())}")
        return self._mods[name]

    def list(self) -> List[str]:
        return sorted(self._mods.keys())


def run_mods(
    ctx: EvalContext,
    registry: ModRegistry,
    *,
    assemble: Iterable[str],
    mod_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    save_index: bool = True,
) -> Dict[str, Any]:
    """
    运行装配的模组集合：
      assemble: ["fourier.kstar_heatmap", "fourier.kstar_curves", ...]
      mod_kwargs: {"fourier.kstar_curves": {"max_plots": 30}, ...}

    返回：
      {
        "out_dir": ...,
        "results": {mod_name: {...}},
        "fig_paths": {mod_name: [png paths...]},
        "meta_paths": ...
      }
    """
    mod_kwargs = mod_kwargs or {}

    ctx.out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    fig_paths: Dict[str, List[str]] = {}

    for name in assemble:
        mod = registry.get(str(name))
        kwargs = dict(mod_kwargs.get(mod.name, {}) or {})

        ctx.log(f"[L4] run mod: {mod.name}")
        out = mod.fn(ctx, kwargs)
        results[mod.name] = out or {}

        fps = out.get("fig_paths", None) if isinstance(out, dict) else None
        if fps is not None:
            if isinstance(fps, (list, tuple)):
                fig_paths[mod.name] = [str(x) for x in fps]
            elif isinstance(fps, str):
                fig_paths[mod.name] = [str(fps)]

    pack = {
        "out_dir": str(ctx.out_dir),
        "results": results,
        "fig_paths": fig_paths,
    }

    if save_index:
        index_path = ctx.out_dir / "index.json"
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2, ensure_ascii=False)

    return pack
