from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from backend.config.schemas import normalize_model_types, resolve_model_types_from_train_cfg
from backend.config.yaml_io import load_experiment_yaml, save_experiment_yaml
from backend.pipelines.train import compute_level2_rebuild
from backend.pod.compute import build_pod


SAVE_ROOT = Path("artifacts/experiments")
RUN_L1 = False
RUN_L2 = True
RUN_COMPARE = True

# Keep this as mlp + pmrh for baseline/improved comparison.
RUN_MODEL_TYPES = ("mlp", "pmrh")

# One experiment folder per run.
EXPERIMENT_NAME = "cylinder_mlp_pmrh_20260331"

JOBS = [
    {
        "name": "cylinder2d",
        "yaml_path": "configs/cylinder_exp_full.yaml",
        "experiment_name": EXPERIMENT_NAME,
    },
]

# Compare script settings.
COMPARE_MASK_RATE = 0.0005
COMPARE_NOISE_SIGMAS = "0,0.001,0.01"
COMPARE_DEVICE = "cpu"  # set to "cuda" when GPU is available
COMPARE_TIMING_REPEATS = 10
COMPARE_TIMING_MAX_FRAMES = 256
COMPARE_OVERRIDE_MAX_EPOCHS: int | None = None
COMPARE_MAX_TRAIN_BATCHES: int | None = None
COMPARE_MAX_VAL_BATCHES: int | None = None


def run_level1_from_yaml(yaml_path: str | Path, *, verbose: bool = True) -> dict:
    data_cfg, pod_cfg, _, _ = load_experiment_yaml(yaml_path)

    if verbose:
        print(f"\n=== Level-1 POD build: {yaml_path} ===")
        print("pod_save_dir:", pod_cfg.save_dir)

    return build_pod(
        data_cfg,
        pod_cfg,
        verbose=verbose,
        plot=False,
    )


def validate_job_config(yaml_path: str | Path) -> None:
    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    if not bool(pod_cfg.center):
        raise ValueError(f"POD centering must be enabled for this experiment: {yaml_path}")
    if not bool(eval_cfg.centered_pod):
        raise ValueError(f"eval.centered_pod must be true for this experiment: {yaml_path}")
    if train_cfg is None:
        raise ValueError(f"train config is required for L2 experiment: {yaml_path}")

    model_types = resolve_model_types_from_train_cfg(train_cfg)
    if len(model_types) == 0:
        raise ValueError(f"No train.model_types resolved from yaml: {yaml_path}")

    pod_ready = (pod_cfg.save_dir / "Ur.npy").exists() and (pod_cfg.save_dir / "pod_meta.json").exists()
    if not RUN_L1 and not pod_ready:
        raise ValueError(
            f"RUN_L1 is False but existing L1 artifacts are missing for {yaml_path}: "
            f"expected Ur.npy and pod_meta.json under {pod_cfg.save_dir}"
        )

    print("config check:")
    print("  pod.save_dir      :", pod_cfg.save_dir)
    print("  pod.ready         :", pod_ready)
    print("  pod.center        :", pod_cfg.center)
    print("  eval.centered_pod :", eval_cfg.centered_pod)
    print("  train.model_types :", model_types)
    print("  train.device      :", train_cfg.device)


def prepare_effective_yaml(
    yaml_path: str | Path,
    *,
    experiment_name: str,
    save_root: str | Path,
) -> Path:
    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)
    if train_cfg is None:
        raise ValueError(f"train config is required for override run: {yaml_path}")

    requested = normalize_model_types(RUN_MODEL_TYPES)
    if len(requested) == 0:
        raise ValueError(f"RUN_MODEL_TYPES resolves to empty selection: {RUN_MODEL_TYPES}")

    train_cfg.model_types = requested

    # Keep selected model configs; ensure pmrh entry exists.
    original_model_configs = dict(getattr(train_cfg, "model_configs", {}) or {})
    filtered = {
        str(model_type): dict(cfg or {})
        for model_type, cfg in original_model_configs.items()
        if str(model_type).strip().lower() in requested
    }
    if "pmrh" in requested and "pmrh" not in filtered:
        filtered["pmrh"] = {"task_type": "pod_coeff_regression"}
    train_cfg.model_configs = filtered

    run_cfg_dir = Path(save_root) / "_run_configs"
    run_cfg_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = run_cfg_dir / f"{experiment_name}_{'_'.join(requested)}.yaml"
    save_experiment_yaml(
        out_yaml,
        data_cfg=data_cfg,
        pod_cfg=pod_cfg,
        eval_cfg=eval_cfg,
        train_cfg=train_cfg,
    )
    return out_yaml


def run_compare(
    *,
    effective_yaml_path: Path,
    exp_dir: Path,
    tag: str,
) -> None:
    output_csv = exp_dir / "mlp_vs_pmrh_compare.csv"

    cmd: list[str] = [
        sys.executable,
        "scripts/compare_mlp_pmrh_cylinder.py",
        "--config",
        str(effective_yaml_path),
        "--mask-rate",
        str(float(COMPARE_MASK_RATE)),
        "--noise-sigmas",
        str(COMPARE_NOISE_SIGMAS),
        "--device",
        str(COMPARE_DEVICE),
        "--timing-repeats",
        str(int(COMPARE_TIMING_REPEATS)),
        "--timing-max-frames",
        str(int(COMPARE_TIMING_MAX_FRAMES)),
        "--output-csv",
        str(output_csv),
        "--tag",
        str(tag),
    ]

    if COMPARE_OVERRIDE_MAX_EPOCHS is not None:
        cmd.extend(["--override-max-epochs", str(int(COMPARE_OVERRIDE_MAX_EPOCHS))])
    if COMPARE_MAX_TRAIN_BATCHES is not None:
        cmd.extend(["--max-train-batches", str(int(COMPARE_MAX_TRAIN_BATCHES))])
    if COMPARE_MAX_VAL_BATCHES is not None:
        cmd.extend(["--max-val-batches", str(int(COMPARE_MAX_VAL_BATCHES))])

    print("\n=== Compare MLP vs PMRH ===")
    print("command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("compare_csv:", output_csv)


def main() -> None:
    pod_runs: dict[str, dict] = {}
    l2_runs: dict[str, dict] = {}

    for job in JOBS:
        name = str(job["name"])
        yaml_path = str(job["yaml_path"])
        exp_name = str(job["experiment_name"])

        validate_job_config(yaml_path)
        effective_yaml_path = prepare_effective_yaml(
            yaml_path,
            experiment_name=exp_name,
            save_root=SAVE_ROOT,
        )
        print("effective_yaml:", effective_yaml_path)
        validate_job_config(effective_yaml_path)

        if RUN_L1:
            pod_res = run_level1_from_yaml(effective_yaml_path, verbose=True)
            pod_runs[name] = pod_res
            print("r_used:", pod_res.get("r_used"))
            print("save_dir:", pod_res.get("save_dir"))

        exp_dir = SAVE_ROOT / exp_name
        if RUN_L2:
            print(f"\n=== Level-2 rebuild: {name} ===")
            l2_res = compute_level2_rebuild(
                effective_yaml_path,
                experiment_name=exp_name,
                save_root=SAVE_ROOT,
                verbose=True,
            )
            l2_runs[name] = l2_res
            exp_dir = Path(l2_res["exp_dir"])
            print("exp_dir:", exp_dir)
            print("trained model_types:", l2_res["L2"].get("model_types", ()))
            print("L2 entry count:", len(l2_res["L2"].get("entries", [])))

        if RUN_COMPARE:
            run_compare(
                effective_yaml_path=Path(effective_yaml_path),
                exp_dir=exp_dir,
                tag=exp_name,
            )

    if RUN_L1:
        print("\n=== L1 POD build done ===")
        print({key: value.get("save_dir") for key, value in pod_runs.items()})

    if RUN_L2:
        print("\n=== L2 rebuild all done ===")
        print({key: value["exp_dir"] for key, value in l2_runs.items()})


if __name__ == "__main__":
    main()
