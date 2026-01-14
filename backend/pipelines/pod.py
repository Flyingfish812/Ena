from pathlib import Path
from typing import Any, Dict, Optional

from backend.pod.compute import build_pod
from backend.config.schemas import DataConfig, PodConfig


def quick_build_pod(
    nc_path: str | Path,
    save_dir: str | Path = "artifacts/pod",
    r: int = 128,
    center: bool = True,
    var_keys: tuple[str, ...] = ("u", "v"),
    *,
    # ===== v2.x additions for Level-1 =====
    dx: float = 1.0,
    dy: float = 1.0,
    enable_scale_analysis: bool = False,
    scale_analysis: Optional[Dict[str, Any]] = None,
    scale_channel_reduce: str = "l2",   # "l2" | "sum" | "u" | "v"
    enable_basis_spectrum: bool = False,
    fft_basis: Optional[Dict[str, Any]] = None,
    # ===== misc =====
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    # DataConfig
    data_cfg = DataConfig(
        nc_path=Path(nc_path),
        var_keys=var_keys,
        cache_dir=None,
    )

    # PodConfig
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=Path(save_dir),

        dx=dx,
        dy=dy,

        enable_scale_analysis=enable_scale_analysis,
        scale_analysis=scale_analysis if scale_analysis is not None else {
            "method": "B_robust_energy_centroid",
            "k_min": None,
            "k_max": None,
            "demean_line": True,
        },
        scale_channel_reduce=scale_channel_reduce,

        enable_basis_spectrum=enable_basis_spectrum,
        fft_basis=fft_basis if fft_basis is not None else {
            "demean": True,
            "window": None,
            "norm": None,
        },
    )

    # Run
    result = build_pod(
        data_cfg,
        pod_cfg,
        verbose=verbose,
        plot=plot,
    )
    return result
