# backend/pipelines/eval/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from backend.pipelines.eval.context import EvalContext


@dataclass(frozen=True)
class EvalMod:
    name: str
    requires: Tuple[str, ...] = ()
    description: str = ""
    run: Callable[[EvalContext], Dict[str, Any]] = lambda ctx: {}


_REGISTRY: Dict[str, EvalMod] = {}


def register_mod(mod: EvalMod) -> None:
    if mod.name in _REGISTRY:
        raise KeyError(f"Eval mod '{mod.name}' already registered.")
    _REGISTRY[mod.name] = mod


def get_mod(name: str) -> EvalMod:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown eval mod: {name}. Available: {list_mods()}")
    return _REGISTRY[name]


def list_mods() -> List[str]:
    return sorted(list(_REGISTRY.keys()))


def describe_mod(name: str) -> Dict[str, Any]:
    m = get_mod(name)
    return {"name": m.name, "requires": list(m.requires), "description": m.description}


def resolve_mods(mods: Optional[Sequence[str]]) -> List[str]:
    # Minimal v1:
    #   - None -> default set
    #   - explicit list -> run those (support "default" alias)
    if mods is None:
        out = ["manifest"]
    else:
        out: List[str] = []
        for x in mods:
            if str(x) == "default":
                out.extend(["manifest"])
            else:
                out.append(str(x))

    # de-dup preserving order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq
