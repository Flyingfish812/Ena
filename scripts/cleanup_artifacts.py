#!/usr/bin/env python3
"""Artifact cleanup helper (safe by default).

This repo accumulates large experiment artifacts (L1/L2/L3/L4). For paper writing we
usually only need a small, reproducible subset. This script implements a *whitelist*
policy and removes everything else.

Default behavior is DRY-RUN (no changes). To actually change the filesystem, pass
`--apply`.

Modes when `--apply` is set:
- move: move removal candidates into artifacts/_trash/<timestamp>/... (reversible)
- delete: permanently delete removal candidates

Current keep-set decision:
- Keep L1 POD dirs: artifacts/pod/{cylinder2d,rdb_h5,sst_weekly}/
- Keep experiments: artifacts/experiments/*_exp_full_2 for the 3 full datasets
  - keep exp_dir/config_used.yaml
  - keep exp_dir/L2/ (all)
  - keep exp_dir/L4_eval/cumulate/ and minimal L4 json indices
- Drop everything else (including L3_fft, preanalysis_cache, preanalysis_figs)

"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"


@dataclass(frozen=True)
class KeepSpec:
	pod_subdirs: Tuple[str, ...]
	exp_dirs: Tuple[str, ...]


DEFAULT_KEEP = KeepSpec(
	pod_subdirs=("cylinder2d", "rdb_h5", "sst_weekly"),
	exp_dirs=(
		"cylinder_exp_full_2",
		"rdb_h5_exp_full_2",
		"sst_weekly_exp_full_2",
	),
)


def _is_relative_to(path: Path, base: Path) -> bool:
	try:
		path.relative_to(base)
		return True
	except Exception:
		return False


def _pretty_rel(path: Path, base: Path) -> str:
	try:
		return str(path.relative_to(base))
	except Exception:
		return str(path)


def _du(path: Path) -> int:
	"""Compute total size in bytes for a file/dir (best-effort)."""
	try:
		if path.is_file() or path.is_symlink():
			return path.lstat().st_size
		total = 0
		for p in path.rglob("*"):
			try:
				if p.is_file() or p.is_symlink():
					total += p.lstat().st_size
			except Exception:
				pass
		return total
	except Exception:
		return 0


def _rm_tree(path: Path) -> None:
	if path.is_symlink() or path.is_file():
		path.unlink(missing_ok=True)
		return
	shutil.rmtree(path, ignore_errors=False)


def _move_to_trash(*, src: Path, trash_root: Path, artifacts_root: Path) -> Path:
	if not src.exists():
		return src

	rel = src.relative_to(artifacts_root)
	dst = trash_root / rel
	dst.parent.mkdir(parents=True, exist_ok=True)

	if dst.exists():
		stem = dst.name
		for i in range(1, 1000):
			cand = dst.with_name(f"{stem}.__{i}")
			if not cand.exists():
				dst = cand
				break

	return Path(shutil.move(str(src), str(dst)))


def build_removal_candidates(
	*,
	artifacts_root: Path,
	keep: KeepSpec,
	delete_preanalysis_cache: bool,
	delete_preanalysis_figs: bool,
) -> List[Path]:
	"""Return a list of removal candidates under artifacts/.

	The list is intentionally coarse-grained (dirs/files) for fast move/delete.
	"""
	candidates: List[Path] = []

	# --- preanalysis ---
	if delete_preanalysis_cache:
		p = artifacts_root / "preanalysis_cache"
		if p.exists():
			candidates.append(p)
	if delete_preanalysis_figs:
		p = artifacts_root / "preanalysis_figs"
		if p.exists():
			candidates.append(p)

	# --- POD dirs ---
	pod_root = artifacts_root / "pod"
	if pod_root.exists():
		keep_pod = {pod_root / name for name in keep.pod_subdirs}
		for child in pod_root.iterdir():
			if child in keep_pod:
				continue
			candidates.append(child)

	# --- experiments ---
	exp_root = artifacts_root / "experiments"
	if exp_root.exists():
		keep_exp = {exp_root / name for name in keep.exp_dirs}
		for child in exp_root.iterdir():
			if child in keep_exp:
				continue
			candidates.append(child)

		# Inside kept experiments
		for exp_dir in sorted(keep_exp):
			if not exp_dir.exists():
				continue

			l3 = exp_dir / "L3_fft"
			if l3.exists():
				candidates.append(l3)

			l4 = exp_dir / "L4_eval"
			if l4.exists():
				keep_names = {
					"cumulate",
					"index.json",
					"manifest.json",
					"quick_check.json",
				}
				for c in l4.iterdir():
					if c.name in keep_names:
						continue
					candidates.append(c)

	# Dedup; keep only paths under artifacts_root
	uniq: List[Path] = []
	seen = set()
	base = artifacts_root.resolve()
	for p in candidates:
		try:
			rp = p.resolve()
		except Exception:
			rp = p
		if not _is_relative_to(rp, base):
			continue
		k = str(rp)
		if k in seen:
			continue
		seen.add(k)
		uniq.append(rp)

	uniq.sort(key=lambda x: (len(_pretty_rel(x, base)), _pretty_rel(x, base)))
	return uniq


def validate_keep_set(*, artifacts_root: Path, keep: KeepSpec) -> List[str]:
	warnings: List[str] = []

	for name in keep.pod_subdirs:
		d = artifacts_root / "pod" / name
		if not d.exists():
			warnings.append(f"missing keep pod dir: {_pretty_rel(d, artifacts_root)}")

	exp_root = artifacts_root / "experiments"
	for name in keep.exp_dirs:
		d = exp_root / name
		if not d.exists():
			warnings.append(f"missing keep exp dir: {_pretty_rel(d, artifacts_root)}")
		else:
			cfg = d / "config_used.yaml"
			if not cfg.exists():
				warnings.append(
					f"keep exp dir missing config_used.yaml: {_pretty_rel(cfg, artifacts_root)}"
				)
			l2 = d / "L2"
			if not l2.exists():
				warnings.append(f"keep exp dir missing L2/: {_pretty_rel(l2, artifacts_root)}")

	return warnings


def main(argv: Optional[Sequence[str]] = None) -> int:
	ap = argparse.ArgumentParser(
		description="Cleanup artifacts with whitelist policy (safe by default)"
	)
	ap.add_argument(
		"--apply",
		action="store_true",
		help="Actually perform changes (default: dry-run)",
	)
	ap.add_argument(
		"--mode",
		choices=("move", "delete"),
		default="move",
		help="When --apply: move to trash or permanently delete (default: move)",
	)
	ap.add_argument(
		"--trash-root",
		default=str(ARTIFACTS_ROOT / "_trash"),
		help="Trash root for mode=move (default: artifacts/_trash)",
	)
	ap.add_argument(
		"--timestamp",
		default=None,
		help="Timestamp folder name under trash root (default: now, like 20260223_142233)",
	)

	ap.add_argument(
		"--delete-preanalysis-cache",
		action="store_true",
		default=True,
		help="Remove artifacts/preanalysis_cache (default: true)",
	)
	ap.add_argument(
		"--keep-preanalysis-cache",
		action="store_false",
		dest="delete_preanalysis_cache",
		help="Keep artifacts/preanalysis_cache",
	)
	ap.add_argument(
		"--delete-preanalysis-figs",
		action="store_true",
		default=True,
		help="Remove artifacts/preanalysis_figs (default: true)",
	)
	ap.add_argument(
		"--keep-preanalysis-figs",
		action="store_false",
		dest="delete_preanalysis_figs",
		help="Keep artifacts/preanalysis_figs",
	)

	args = ap.parse_args(list(argv) if argv is not None else None)

	artifacts_root = ARTIFACTS_ROOT
	if not artifacts_root.exists():
		print(f"[cleanup] artifacts root not found: {artifacts_root}", file=sys.stderr)
		return 2

	warnings = validate_keep_set(artifacts_root=artifacts_root, keep=DEFAULT_KEEP)
	if warnings:
		print("[cleanup] WARNINGS (keep-set validation):")
		for w in warnings:
			print(f"  - {w}")

	candidates = build_removal_candidates(
		artifacts_root=artifacts_root,
		keep=DEFAULT_KEEP,
		delete_preanalysis_cache=bool(args.delete_preanalysis_cache),
		delete_preanalysis_figs=bool(args.delete_preanalysis_figs),
	)
	candidates = [p for p in candidates if p.exists()]

	total_bytes = sum(_du(p) for p in candidates)
	print("[cleanup] Removal candidates under artifacts/:")
	for p in candidates:
		print(f"  - {_pretty_rel(p, artifacts_root)}")
	print(f"[cleanup] Count: {len(candidates)}")
	print(f"[cleanup] Approx size: {total_bytes/1024/1024:.1f} MiB")

	if not args.apply:
		print("[cleanup] DRY-RUN only. Re-run with --apply to perform changes.")
		return 0

	if args.mode == "move":
		ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
		trash_root = Path(args.trash_root) / ts
		trash_root.mkdir(parents=True, exist_ok=True)

		moved = 0
		for p in candidates:
			rel = _pretty_rel(p, artifacts_root)
			try:
				dst = _move_to_trash(
					src=p, trash_root=trash_root, artifacts_root=artifacts_root
				)
				print(
					f"[cleanup] MOVE  {rel} -> {_pretty_rel(dst, artifacts_root)}"
				)
				moved += 1
			except Exception as e:
				print(
					f"[cleanup] FAIL MOVE {rel}: {type(e).__name__}: {e}",
					file=sys.stderr,
				)

		print(
			f"[cleanup] Done. Moved {moved}/{len(candidates)} to trash: {_pretty_rel(trash_root, artifacts_root)}"
		)
		return 0

	deleted = 0
	for p in candidates:
		rel = _pretty_rel(p, artifacts_root)
		try:
			_rm_tree(p)
			print(f"[cleanup] DELETE {rel}")
			deleted += 1
		except Exception as e:
			print(
				f"[cleanup] FAIL DELETE {rel}: {type(e).__name__}: {e}",
				file=sys.stderr,
			)

	print(f"[cleanup] Done. Deleted {deleted}/{len(candidates)}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
