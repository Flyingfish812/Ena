from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_MAP_DISPLAY_SOURCES = {"mat_sst", "sst_mat", "mat"}


def _extract_source(candidate: Any) -> str | None:
    if candidate is None:
        return None
    if isinstance(candidate, Mapping):
        value = candidate.get("source")
    else:
        value = getattr(candidate, "source", None)
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def infer_display_origin(
    *,
    source: str | None = None,
    data_cfg: Any = None,
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Infer the correct imshow origin for a dataset-backed spatial field."""
    if source is not None:
        source_text = str(source).strip().lower()
        if source_text in {"lower", "upper"}:
            return source_text

    explicit = None
    if meta is not None:
        explicit = meta.get("display_origin")
    if explicit is not None:
        origin = str(explicit).strip().lower()
        if origin in {"lower", "upper"}:
            return origin

    source_candidates = [
        source,
        _extract_source(data_cfg),
        _extract_source(meta),
        _extract_source((meta or {}).get("data_cfg") if meta is not None else None),
    ]
    for candidate in source_candidates:
        if candidate in _MAP_DISPLAY_SOURCES:
            return "upper"

    return "lower"