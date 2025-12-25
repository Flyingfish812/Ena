# backend/advance_analysis.py
# v2.0 draft: compute Level-4 artifacts (final analysis products) using L1/L2/L3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from backend.dataio.io_utils import ensure_dir, save_json, load_json


@dataclass(frozen=True)
class AdvanceAnalysisConfig:
    """Draft knobs for advanced analysis toolchain."""
    enabled: bool = True
    # add later: kstar_threshold, band_scheme, aggregation rules, etc.


def run_advance_analysis(
    *,
    exp_dir: str | Path,
    adv_cfg: Optional[AdvanceAnalysisConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Level-4 producer (draft).

    Inputs:
      - Level-1 POD artifacts (basis, mean, coeff tables) [referenced]
      - Level-2 raw predictions (coeffs, model states)
      - Level-3 FFT artifacts (spectra, band inputs)

    Outputs:
      - Level-4 consolidated results: tables, summaries, k* curves, etc.
      - (Plotting remains out-of-scope for now; will be split later.)

    Returns:
      index dict with paths to Level-4 outputs.
    """
    exp_dir = Path(exp_dir)
    adv_cfg = adv_cfg or AdvanceAnalysisConfig()

    l4_root = exp_dir / "L4_final"
    ensure_dir(l4_root)

    # Placeholder: read L2/L3 meta to understand what exists
    l2_meta_path = exp_dir / "L2_rebuild" / "meta.json"
    l3_meta_path = exp_dir / "L3_fft" / "meta.json"

    meta = {
        "schema_version": "v2.0-draft",
        "has_L2": l2_meta_path.exists(),
        "has_L3": l3_meta_path.exists(),
        "note": "Level-4 final analysis products. Implement later.",
    }
    save_json(meta, l4_root / "meta.json")

    # Placeholder outputs
    save_json({"placeholder": True}, l4_root / "results.json")

    return {"root": str(l4_root), "meta": meta}


# Future: toolchain components (to be implemented)
def compute_kstar_curves(
    *,
    exp_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Placeholder: compute k* curves using stored FFT artifacts and POD meta."""
    raise NotImplementedError


def aggregate_metrics_tables(
    *,
    exp_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Placeholder: build final tables for papers (Level-4)."""
    raise NotImplementedError
