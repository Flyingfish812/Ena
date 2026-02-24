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


def quick_build_pod_rdb_h5(
    h5_path: str | Path = "data/2D_rdb_NA_NA.h5",
    save_dir: str | Path = "artifacts/pod/rdb_h5",
    r: int = 128,
    center: bool = True,
    *,
    group_count: int = 50,
    frames_per_group: int = 10,
    group_start: int = 0,
    group_step: int = 1,
    dataset_key: str = "data",
    cache_dir: str | Path | None = "artifacts/preanalysis_cache",
    # grid meta for scale/fft tools
    dx: float = 1.0,
    dy: float = 1.0,
    enable_scale_analysis: bool = False,
    scale_analysis: Optional[Dict[str, Any]] = None,
    scale_channel_reduce: str = "l2",
    enable_basis_spectrum: bool = False,
    fft_basis: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """One-click Level-1 POD for data/2D_rdb_NA_NA.h5.

    Sampling defaults are chosen for feasibility:
      T = group_count * frames_per_group
    """
    data_cfg = DataConfig(
        source="h5_rdb",
        path=Path(h5_path),
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        h5_rdb_dataset_key=str(dataset_key),
        h5_rdb_group_count=int(group_count),
        h5_rdb_group_start=int(group_start),
        h5_rdb_group_step=int(group_step),
        h5_rdb_frames_per_group=int(frames_per_group),
        h5_rdb_frame_sampling="linspace",
    )

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

    return build_pod(data_cfg, pod_cfg, verbose=verbose, plot=plot)


def quick_build_pod_sst_mat(
    mat_path: str | Path = "data/sst_weekly.mat",
    save_dir: str | Path = "artifacts/pod/sst_weekly",
    r: int = 128,
    center: bool = True,
    *,
    mat_key: str = "sst",
    fill_nan: str = "per_frame_mean",
    reshape_mode: str = "360x180_rot90",
    max_frames: int | None = None,
    cache_dir: str | Path | None = "artifacts/preanalysis_cache",
    # grid meta for scale/fft tools
    dx: float = 1.0,
    dy: float = 1.0,
    enable_scale_analysis: bool = False,
    scale_analysis: Optional[Dict[str, Any]] = None,
    scale_channel_reduce: str = "l2",
    enable_basis_spectrum: bool = False,
    fft_basis: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """One-click Level-1 POD for data/sst_weekly.mat (MAT v7.3)."""
    data_cfg = DataConfig(
        source="mat_sst",
        path=Path(mat_path),
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
        mat_key=str(mat_key),
        sst_fill_nan=str(fill_nan),
        sst_reshape=str(reshape_mode),
        sst_max_frames=max_frames,
    )

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

    return build_pod(data_cfg, pod_cfg, verbose=verbose, plot=plot)
