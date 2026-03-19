from pathlib import Path

from backend.config.yaml_io import load_experiment_yaml
from backend.pipelines.train import compute_level2_rebuild
from backend.pod.compute import build_pod


SAVE_ROOT = "artifacts/experiments"
RUN_L1 = False
RUN_L2 = True

# JOBS = [
#     {
#         "name": "cylinder2d",
#         "yaml_path": "configs/cylinder_exp_full.yaml",
#         "experiment_name": "cylinder_exp_full_3",
#     },
# ]
# JOBS = [
#     {
#         "name": "rdb_h5",
#         "yaml_path": "configs/rdb_h5_exp_full.yaml",
#         "experiment_name": "rdb_h5_exp_full_3",
#     },
# ]
JOBS = [
    {
        "name": "sst_weekly",
        "yaml_path": "configs/sst_weekly_exp_full.yaml",
        "experiment_name": "sst_weekly_exp_full_3",
    },
]


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
    _, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(yaml_path)

    if not bool(pod_cfg.center):
        raise ValueError(f"POD centering must be enabled for this experiment: {yaml_path}")
    if not bool(eval_cfg.centered_pod):
        raise ValueError(f"eval.centered_pod must be true for this experiment: {yaml_path}")
    if train_cfg is None:
        raise ValueError(f"train config is required for L2 MLP experiment: {yaml_path}")

    print("config check:")
    print("  pod.center      :", pod_cfg.center)
    print("  eval.centered_pod:", eval_cfg.centered_pod)
    print("  train.hidden_dims:", train_cfg.hidden_dims)
    print("  train.device    :", train_cfg.device)
    print("  train.noise_sigma:", train_cfg.noise_sigma)
    print("  train.use_weighted_loss:", getattr(train_cfg, "use_weighted_loss", False))
    print("  train.loss_weighting:", getattr(train_cfg, "loss_weighting", "none"))
    print("  train.loss_weight_power:", getattr(train_cfg, "loss_weight_power", 1.0))


def main() -> None:
    pod_runs = {}
    l2_runs = {}

    for job in JOBS:
        name = str(job["name"])
        yaml_path = str(job["yaml_path"])
        exp_name = str(job["experiment_name"])

        validate_job_config(yaml_path)

        if RUN_L1:
            pod_res = run_level1_from_yaml(yaml_path, verbose=True)
            pod_runs[name] = pod_res
            print("r_used:", pod_res.get("r_used"))
            print("save_dir:", pod_res.get("save_dir"))

        if RUN_L2:
            print(f"\n=== Level-2 rebuild: {name} ===")
            l2_res = compute_level2_rebuild(
                yaml_path,
                experiment_name=exp_name,
                save_root=SAVE_ROOT,
                verbose=True,
            )
            l2_runs[name] = l2_res
            print("exp_dir:", l2_res["exp_dir"])

    if RUN_L1:
        print("\n=== L1 POD build done ===")
        print({k: v.get("save_dir") for k, v in pod_runs.items()})

    if RUN_L2:
        print("\n=== L2 rebuild all done ===")
        print({k: v["exp_dir"] for k, v in l2_runs.items()})


if __name__ == "__main__":
    main()
