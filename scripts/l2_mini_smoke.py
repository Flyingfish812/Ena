#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

from backend.config.yaml_io import load_experiment_yaml
from backend.dataio.loader import load_raw
from backend.pipelines.train import compute_level2_rebuild


def _pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def _check_dataset_volume(yaml_path: Path, full_frames: int, *, lower: float, upper: float) -> None:
    data_cfg, _, _, _ = load_experiment_yaml(yaml_path)
    x = load_raw(data_cfg)

    t, h, w, c = x.shape
    bytes_n = int(x.nbytes)
    ratio = float(t) / float(full_frames)

    print(f"[mini-check] {yaml_path.name}: shape={x.shape}, bytes={bytes_n}, ratio={_pct(ratio)}")

    if bytes_n > 5 * 1024 * 1024:
        raise RuntimeError(
            f"mini payload too large for {yaml_path.name}: {bytes_n} bytes > 5MB"
        )
    if not (lower <= ratio <= upper):
        raise RuntimeError(
            f"mini ratio out of range for {yaml_path.name}: {ratio:.6f} not in [{lower}, {upper}]"
        )


def _run_l2(yaml_path: Path, exp_name: str, save_root: Path, verbose: bool) -> None:
    print(f"[mini-run] start: {yaml_path} -> {exp_name}")
    pack = compute_level2_rebuild(
        yaml_path,
        experiment_name=exp_name,
        save_root=save_root,
        verbose=verbose,
    )
    print(f"[mini-run] done : exp_dir={pack['exp_dir']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="L2 mini smoke for rdb_h5 + sst_weekly")
    parser.add_argument("--save-root", default="artifacts/experiments")
    parser.add_argument("--skip-l2", action="store_true", help="only check mini volume, skip L2 run")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    rdb_yaml = repo_root / "configs" / "rdb_h5_exp_full_mini.yaml"
    sst_yaml = repo_root / "configs" / "sst_weekly_exp_full_mini.yaml"

    # full frames reference:
    # - rdb_h5: 1000 groups * 101 frames
    # - sst_weekly: 1914 frames
    _check_dataset_volume(rdb_yaml, full_frames=101000, lower=0.0005, upper=0.01)
    _check_dataset_volume(sst_yaml, full_frames=1914, lower=0.001, upper=0.01)

    if not args.skip_l2:
        save_root = Path(args.save_root)
        _run_l2(
            rdb_yaml,
            exp_name="rdb_h5_exp_full_mini",
            save_root=save_root,
            verbose=args.verbose,
        )
        _run_l2(
            sst_yaml,
            exp_name="sst_weekly_exp_full_mini",
            save_root=save_root,
            verbose=args.verbose,
        )

    print("[mini-smoke] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
