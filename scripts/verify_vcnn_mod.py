from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.models.vcnn_multiscale import run_vcnn_multiscale_verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Reload a saved two-stage VCNN early-exit checkpoint and regenerate review artifacts.")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/experiments/vcnn_benchmarks/vcnn_multiscale_latest.pt",
        help="Path to the saved VCNN multiscale checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/vcnn_benchmarks/vcnn_multiscale_latest_review",
        help="Directory where verification artifacts are overwritten.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--pod-dir",
        default=None,
        help="Override pod.save_dir during verification (for split POD).",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="1-based width split index used during training.",
    )
    parser.add_argument(
        "--split-count",
        type=int,
        default=None,
        help="Total width split count used during training.",
    )
    args = parser.parse_args()

    summary = run_vcnn_multiscale_verification(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        pod_dir=args.pod_dir,
        split_index=args.split_index,
        split_count=args.split_count,
        verbose=True,
    )
    print(
        f"[verify_vcnn_mod] done exp_id={summary['exp_id']} "
        f"review_summary={Path(args.output_dir) / 'review_summary.json'} "
        f"nrmse_plot={Path(args.output_dir) / 'nrmse_vs_r.png'}"
    )


if __name__ == "__main__":
    main()