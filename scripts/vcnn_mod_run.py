from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.config.schemas import resolve_model_dataset_specs
from backend.config.yaml_io import load_experiment_yaml
from backend.dataio.loader import describe_source, load_raw
from backend.eval.rebuild import (
    _encode_rate,
    _load_or_build_pod,
    _mask_strategy_from_data_cfg,
    _prepare_snapshots,
    _resolve_branch_train_config,
)
from backend.models.vcnn_multiscale import (
    PREFIX_EVAL_STEPS,
    append_summary_csv,
    aggregate_metrics,
    measure_exit_latency_ms,
    run_vcnn_multiscale_verification,
    save_vcnn_multiscale_checkpoint,
    train_vcnn_multiscale_on_observations,
)
from backend.sampling.masks import generate_observation_mask_hw


def _default_exp_id(config_path: str, mask_rate: float, noise_sigma: float) -> str:
    cfg_stem = Path(config_path).stem
    return f"vcnn_ms_{cfg_stem}_p{_encode_rate(mask_rate):04d}_s{_encode_rate(noise_sigma):04d}"


def _first_feature_batch(dataloader):
    for feature, _ in dataloader:
        return feature
    raise ValueError("Validation dataloader is empty.")


def _resolve_split_bounds(width: int, split_index: int, split_count: int) -> tuple[int, int]:
    if split_count <= 0:
        raise ValueError(f"split_count must be positive, got {split_count}")
    if not (1 <= split_index <= split_count):
        raise ValueError(f"split_index must be in [1, {split_count}], got {split_index}")
    if width % split_count != 0:
        raise ValueError(f"Width {width} is not divisible by split_count={split_count}")
    split_w = width // split_count
    x0 = (split_index - 1) * split_w
    x1 = x0 + split_w
    return x0, x1


def _prepare_snapshots_for_vcnn(
    data_cfg,
    Ur: np.ndarray,
    mean_flat: np.ndarray,
    r_eff: int,
    *,
    split_index: int | None,
    split_count: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if split_index is None:
        return _prepare_snapshots(data_cfg, Ur, mean_flat, r_eff=r_eff, verbose=verbose)

    if verbose:
        print(f"[rebuild] Loading full raw data from {describe_source(data_cfg)} for split training ...")
    X_thwc = load_raw(data_cfg)  # [T,H,W,C]
    T, H, W, C = X_thwc.shape
    x0, x1 = _resolve_split_bounds(W, split_index=split_index, split_count=split_count)
    X_thwc = X_thwc[:, :, x0:x1, :]
    D = int(X_thwc.shape[1] * X_thwc.shape[2] * X_thwc.shape[3])

    if Ur.shape[0] != D:
        raise ValueError(
            "Split data and POD shape mismatch: "
            f"Ur first dim={Ur.shape[0]} but split H*W*C={D}. "
            "Please ensure --pod-dir points to the same split index."
        )

    X_flat_all = X_thwc.reshape(T, D)
    Ur_eff = Ur[:, :r_eff]
    mean_flat_v = np.asarray(mean_flat, dtype=np.float32).reshape(1, D)
    A_true = ((X_flat_all - mean_flat_v) @ Ur_eff).astype(np.float32, copy=False)

    if verbose:
        print(
            f"  -> split_index={split_index}/{split_count}, x_range=[{x0},{x1}), "
            f"X_thwc={X_thwc.shape}, flat=[{T},{D}], r_eff={r_eff}"
        )

    return X_thwc, A_true


def _load_mask_from_csv(mask_csv: str | Path, *, H: int, W: int) -> np.ndarray:
    csv_path = Path(mask_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"mask csv not found: {csv_path}")

    rows: list[int] = []
    cols: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"mask csv is empty: {csv_path}")
        names = {str(v).strip().lower() for v in reader.fieldnames}
        if not {"row", "col"}.issubset(names):
            raise ValueError(f"mask csv must contain header columns 'row,col': {csv_path}")

        for line_idx, rec in enumerate(reader, start=2):
            if rec is None:
                continue
            try:
                r = int(rec.get("row", ""))
                c = int(rec.get("col", ""))
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"invalid row/col at line {line_idx} in {csv_path}") from exc
            if not (0 <= r < int(H) and 0 <= c < int(W)):
                raise ValueError(
                    f"mask point out of bounds at line {line_idx}: (row={r}, col={c}) for shape {(H, W)}"
                )
            rows.append(r)
            cols.append(c)

    mask_hw = np.zeros((int(H), int(W)), dtype=bool)
    if rows:
        mask_hw[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = True
    if int(mask_hw.sum()) <= 0:
        raise ValueError(f"mask csv has no valid points: {csv_path}")
    return mask_hw


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VCNN multi-scale early-exit experiment on cylinder2d and append one summary row.")
    parser.add_argument("--config", default="configs/cylinder_exp_full.yaml")
    parser.add_argument("--mask-rate", type=float, default=0.0005)
    parser.add_argument(
        "--mask-csv",
        type=str,
        default=None,
        help="Path to custom mask CSV with header 'row,col'. If set, this mask overrides strategy-based generation.",
    )
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--exp-id", default=None)
    parser.add_argument("--output-csv", default="artifacts/experiments/vcnn_benchmarks/vcnn_multiscale_exp.csv")
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--override-max-epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--pod-dir",
        default=None,
        help="Override pod.save_dir, e.g. artifacts/pod/cylinder2d-1",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="1-based split index for width partition (use with --split-count).",
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=4,
        help="Total split count along width when --split-index is enabled.",
    )
    args = parser.parse_args()

    data_cfg, pod_cfg, eval_cfg, train_cfg = load_experiment_yaml(args.config)
    if train_cfg is None:
        raise ValueError("train config is required")

    if args.split_index is not None and not args.pod_dir:
        pod_cfg.save_dir = Path(f"{pod_cfg.save_dir}-{int(args.split_index)}")
    if args.pod_dir:
        pod_cfg.save_dir = Path(args.pod_dir)

    Ur, mean_flat, pod_meta = _load_or_build_pod(data_cfg, pod_cfg, verbose=True)
    r_eff = int(min(int(Ur.shape[1]), int(pod_meta.get("r_used", Ur.shape[1])), int(pod_cfg.r)))
    X_thwc, A_true = _prepare_snapshots_for_vcnn(
        data_cfg,
        Ur,
        mean_flat,
        r_eff=r_eff,
        split_index=args.split_index,
        split_count=int(args.split_count),
        verbose=True,
    )
    Ur_eff = np.asarray(Ur[:, :r_eff], dtype=np.float32)

    _, H, W, C = X_thwc.shape
    model_dataset_specs = resolve_model_dataset_specs(data_cfg, num_channels=C)
    model_dataset_spec = dict(model_dataset_specs.get("vcnn", {}))

    mask_rate = float(args.mask_rate)
    custom_mask_csv = None if args.mask_csv is None else str(Path(args.mask_csv))
    if custom_mask_csv:
        mask_hw = _load_mask_from_csv(custom_mask_csv, H=H, W=W)
        mask_rate = float(mask_hw.mean())
        mask_source = "custom_csv"
        mask_strategy = "custom_csv"
        mask_strategy_kwargs = {"mask_csv": custom_mask_csv}
        if True:
            print(
                f"[vcnn_multiscale] use custom mask csv: {custom_mask_csv} "
                f"obs={int(mask_hw.sum())} density={mask_rate:.6f}"
            )
    else:
        mask_strategy, mask_strategy_kwargs_base = _mask_strategy_from_data_cfg(
            data_cfg,
            X_thwc=X_thwc,
            Ur_eff=Ur_eff,
            H=H,
            W=W,
            C=C,
            pod_cfg=pod_cfg,
        )
        mask_seed = int(getattr(data_cfg, "observation_mask_seed", 0))
        mask_strategy_kwargs = dict(mask_strategy_kwargs_base)
        if str(mask_strategy).strip().lower() in ("cylinder_structure_aware", "structure_aware", "region_importance"):
            template_count = max(1, int(mask_strategy_kwargs.get("num_templates", 1)))
            mask_strategy_kwargs["template_index"] = int(_encode_rate(mask_rate) % template_count)
        mask_hw = generate_observation_mask_hw(
            H,
            W,
            mask_rate=mask_rate,
            seed=mask_seed,
            strategy=mask_strategy,
            strategy_kwargs=mask_strategy_kwargs,
        )
        mask_source = "generated"

    model_cfg = _resolve_branch_train_config(train_cfg, model_type="vcnn", mask_rate=mask_rate)
    model_cfg = dict(model_cfg)
    model_cfg["mask_source"] = str(mask_source)
    model_cfg["mask_strategy"] = str(mask_strategy)
    model_cfg["mask_strategy_kwargs"] = dict(mask_strategy_kwargs)
    model_cfg["custom_mask_csv_path"] = custom_mask_csv
    if args.override_max_epochs is not None:
        model_cfg["max_epochs"] = int(args.override_max_epochs)

    noise_sigma = float(
        args.noise_sigma
        if args.noise_sigma is not None
        else model_cfg.get("noise_sigma", getattr(train_cfg, "noise_sigma", 0.0))
    )
    representation = str(model_cfg.get("input_representation", model_dataset_spec.get("input_representation", "voronoi_per_channel_plus_mask")))
    include_mask_channel = bool(model_cfg.get("include_mask_channel", model_dataset_spec.get("include_mask_channel", True)))
    patch_size = int(model_cfg.get("patch_size", 1))
    normalize = bool(model_cfg.get("normalize_mean_std", True))
    device_name = args.device or str(model_cfg.get("device", getattr(train_cfg, "device", "auto")))
    output_mode = str(model_cfg.get("multiscale_output_mode", "field"))

    exp_id = str(args.exp_id or _default_exp_id(args.config, mask_rate, noise_sigma))
    print(
        f"[vcnn_multiscale] exp_id={exp_id} mask_rate={mask_rate:.4e} noise_sigma={noise_sigma:.4e} "
        f"representation={representation} output_mode={output_mode}"
    )
    print(f"[vcnn_multiscale] mask_source={mask_source} strategy={mask_strategy}")

    model, train_info, artifacts = train_vcnn_multiscale_on_observations(
        X_thwc=X_thwc,
        a_true=A_true,
        Ur_eff=Ur_eff,
        mean_flat=np.asarray(mean_flat, dtype=np.float32),
        mask_hw=mask_hw,
        representation=representation,
        include_mask_channel=include_mask_channel,
        noise_sigma=noise_sigma,
        batch_size=int(model_cfg.get("batch_size", getattr(train_cfg, "batch_size", 12))),
        num_epochs=int(model_cfg.get("max_epochs", getattr(train_cfg, "max_epochs", 25))),
        lr=float(model_cfg.get("lr", getattr(train_cfg, "lr", 1e-3))),
        weight_decay=float(model_cfg.get("weight_decay", getattr(train_cfg, "weight_decay", 0.0))),
        device=device_name,
        hidden_channels=int(model_cfg.get("hidden_channels", 48)),
        num_layers=int(model_cfg.get("num_layers", 8)),
        kernel_size=int(model_cfg.get("kernel_size", 7)),
        optimizer_name=str(model_cfg.get("optimizer", "adamw")),
        val_ratio=float(model_cfg.get("val_ratio", getattr(train_cfg, "val_ratio", 0.1))),
        patch_size=patch_size,
        normalize=normalize,
        min_lr=float(model_cfg.get("min_lr", getattr(train_cfg, "min_lr", 0.0))),
        warmup_epochs=int(model_cfg.get("warmup_epochs", getattr(train_cfg, "warmup_epochs", 0))),
        use_cosine_schedule=bool(model_cfg.get("use_cosine_schedule", True)),
        early_stop=bool(model_cfg.get("early_stop", getattr(train_cfg, "early_stop", True))),
        early_patience=int(model_cfg.get("early_patience", getattr(train_cfg, "early_patience", 20))),
        early_min_delta=float(model_cfg.get("early_min_delta", getattr(train_cfg, "early_min_delta", 0.0))),
        early_warmup=int(model_cfg.get("early_warmup", getattr(train_cfg, "early_warmup", 5))),
        seed=int(model_cfg.get("seed", getattr(train_cfg, "seed", 0))),
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        verbose=True,
        rank_steps=(16, 48, 128),
        output_mode=output_mode,
        loss_type=str(model_cfg.get("loss_type", "mse")),
        obs_weight=float(model_cfg.get("obs_weight", 1.0)),
        stage2_num_epochs=model_cfg.get("stage2_max_epochs"),
        stage2_lr=model_cfg.get("stage2_lr"),
    )

    metrics = aggregate_metrics(
        model,
        artifacts.val_loader,
        device=str(train_info["device"]),
        Ur_eff=Ur_eff,
        mean_flat=np.asarray(mean_flat, dtype=np.float32),
        spatial_shape=artifacts.spatial_shape,
        pad_hw=artifacts.pad_hw,
        norm_mean_c=artifacts.norm_mean_c,
        norm_std_c=artifacts.norm_std_c,
        prefix_steps=PREFIX_EVAL_STEPS,
        max_batches=args.max_val_batches,
        output_mode=output_mode,
    )

    timing_batch = _first_feature_batch(artifacts.val_loader)
    latency_exit1 = measure_exit_latency_ms(
        model,
        timing_batch,
        exit_level=1,
        device=str(train_info["device"]),
        repeats=int(args.timing_repeats),
    )
    latency_exit2 = measure_exit_latency_ms(
        model,
        timing_batch,
        exit_level=2,
        device=str(train_info["device"]),
        repeats=int(args.timing_repeats),
    )
    latency_exit3 = measure_exit_latency_ms(
        model,
        timing_batch,
        exit_level=3,
        device=str(train_info["device"]),
        repeats=int(args.timing_repeats),
    )

    summary_row = {
        "exp_id": exp_id,
        "epoch": int(train_info.get("epochs_ran", train_info.get("best_epoch", 0))),
        "latency_exit1": float(latency_exit1),
        "latency_exit2": float(latency_exit2),
        "latency_exit3": float(latency_exit3),
        "param_count": int(train_info["param_count"]),
    }
    for exit_idx in (1, 2, 3):
        for prefix_dim in PREFIX_EVAL_STEPS:
            key = f"E{int(prefix_dim)}_exit{exit_idx}"
            if key in metrics:
                summary_row[key] = float(metrics[key])

    benchmark_root = Path(args.output_csv).resolve().parent
    checkpoint_path = benchmark_root / "vcnn_multiscale_latest.pt"
    review_dir = benchmark_root / "vcnn_multiscale_latest_review"
    save_vcnn_multiscale_checkpoint(
        checkpoint_path,
        model=model,
        train_info=train_info,
        artifacts=artifacts,
        config_path=args.config,
        exp_id=exp_id,
        mask_rate=mask_rate,
        noise_sigma=noise_sigma,
        model_cfg=model_cfg,
        pod_dir_used=pod_cfg.save_dir,
        split_index=args.split_index,
        split_count=(int(args.split_count) if args.split_index is not None else None),
    )
    review_summary = run_vcnn_multiscale_verification(
        checkpoint_path=checkpoint_path,
        output_dir=review_dir,
        device=device_name,
        batch_size=int(model_cfg.get("batch_size", getattr(train_cfg, "batch_size", 12))),
        max_batches=args.max_val_batches,
        pod_dir=pod_cfg.save_dir,
        split_index=args.split_index,
        split_count=(int(args.split_count) if args.split_index is not None else None),
        verbose=True,
    )
    append_summary_csv(args.output_csv, summary_row)

    print("[vcnn_multiscale] validation summary")
    for exit_idx in (1, 2, 3):
        for prefix_dim in PREFIX_EVAL_STEPS:
            key = f"E{int(prefix_dim)}_exit{exit_idx}"
            if key in summary_row:
                print(f"  {key}={summary_row[key]:.6e}")
    print(
        "[vcnn_multiscale] latency_ms "
        f"exit1={latency_exit1:.3f} exit2={latency_exit2:.3f} exit3={latency_exit3:.3f} "
        f"param_count={summary_row['param_count']} csv={args.output_csv}"
    )
    print(
        f"[vcnn_multiscale] checkpoint={checkpoint_path} review_dir={review_dir} "
        f"representative_cases={len(review_summary.get('representative_cases', []))}"
    )


if __name__ == "__main__":
    main()