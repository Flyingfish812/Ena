# backend/pipelines/eval_mods/cumulate_mods.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval.utils import read_json, write_json, entry_name

from backend.pipelines.eval_mods.cumulate_io import (
    pod_artifacts_paths,
    try_load_npy,
    try_read_json,
    try_peek_scale_table_csv,
    find_any_l2_npz,
    peek_l2_npz,
    load_coeff_pair,
    load_pod_mode_scales_standardized,
)
from backend.metrics.cumulate_metrics import compute_nrmse_prefix, merge_leff_and_nrmse
from backend.viz.cumulate_plots import plot_nrmse_leff_curves

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, *, dpi: int = 160) -> None:
    if fig is None:
        return
    _ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)

def mod_cumulate_quick_check(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-0（IO链路检查）：
      - 检查 POD(L1) 关键文件存在/可读
      - 抽样 peek 一个 L2 npz 的 keys 与 shape
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- POD (L1) ----
    pp = pod_artifacts_paths(ctx)
    pod_report = {
        "pod_dir": str(pp.pod_dir),
        "Ur.npy": try_load_npy(pp.Ur_npy),
        "mean_flat.npy": try_load_npy(pp.mean_flat_npy),
        "pod_meta.json": try_read_json(pp.pod_meta_json),
        "scale_table.csv": try_peek_scale_table_csv(pp.scale_table_csv, max_rows=int(kwargs.get("max_scale_csv_rows", 3))),
        "scale_meta.json": try_read_json(pp.scale_meta_json),
    }

    # ---- L2 ----
    l2_root = Path(ctx.paths.l2_root)
    eg = find_any_l2_npz(l2_root)
    l2_report = {
        "l2_root": str(l2_root),
        "example_npz": (str(eg) if eg is not None else None),
        "example_npz_peek": (peek_l2_npz(eg) if eg is not None else {"ok": False, "error": "no npz found under l2_root"}),
    }

    # ---- print ----
    print("=== [L4:cumulate.quick_check] ===")
    print(f"- POD dir: {pod_report['pod_dir']}")
    for k in ["Ur.npy", "mean_flat.npy", "pod_meta.json", "scale_table.csv", "scale_meta.json"]:
        v = pod_report.get(k, {})
        if isinstance(v, dict):
            print(f"  * {k}: exists={v.get('exists', None)} ok={v.get('ok', None)}")
    print(f"- L2 root: {l2_report['l2_root']}")
    print(f"  * example_npz: {l2_report['example_npz']}")
    egp = l2_report.get("example_npz_peek", {})
    if isinstance(egp, dict):
        print(f"    keys={egp.get('keys', None)}")
        print(f"    peek={egp.get('peek', None)}")

    report = {"pod": pod_report, "l2": l2_report}

    out_json = out_dir / "quick_check.json"
    write_json(out_json, report)

    return {
        "out_dir": str(out_dir),
        "written": {"quick_check_json": str(out_json)},
        "report": report,
    }

def mod_cumulate_nrmse_prefix(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-1：对每个 (model_type, p, σ) 从 L2 计算 NRMSE(r)（纯数值，不画图）。
    输出：L4_eval/cumulate/nrmse_prefix/json/<entry>.json
    """
    assert ctx.paths is not None
    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_prefix"
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    mode = str(kwargs.get("mode", "coeff"))
    eps = float(kwargs.get("eps", 1e-12))
    max_r = kwargs.get("max_r", None)
    max_r = None if max_r is None else int(max_r)

    written = []
    ok_cnt = 0
    fail_cnt = 0

    for model_type in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            stem = entry_name(str(model_type), float(p), float(s))
            out_json = json_dir / f"{stem}.json"

            try:
                A_hat, A_true, meta, npz_path = load_coeff_pair(ctx, str(model_type), float(p), float(s))
                r_grid, nrmse_r, method = compute_nrmse_prefix(A_hat, A_true, mode=mode, eps=eps, max_r=max_r)

                payload = {
                    "stem": stem,
                    "meta": {
                        "model_type": str(model_type),
                        "mask_rate": float(p),
                        "noise_sigma": float(s),
                        "source_npz": str(npz_path),
                        "eps": float(eps),
                        "method": method,
                        "coeff_keys": {
                            "A_hat": meta.get("k_hat", None),
                            "A_true": meta.get("k_true", None),
                        },
                        "T": int(meta.get("T", A_true.shape[0])),
                        "R_total": int(meta.get("R", A_true.shape[1])),
                        "R_used": int(len(r_grid)),
                        "centered_pod": meta.get("centered_pod", None),
                    },
                    "r_grid": r_grid.astype(int).tolist(),
                    "nrmse_r": np.asarray(nrmse_r, dtype=float).tolist(),
                }

                write_json(out_json, payload)
                written.append(str(out_json))
                ok_cnt += 1
                print(f"[L4:cumulate.nrmse_prefix] OK  {stem}  R={len(r_grid)}  -> {out_json.name}")

            except Exception as e:
                fail_cnt += 1
                print(f"[L4:cumulate.nrmse_prefix] FAIL {stem}: {type(e).__name__}: {e}")

    index = {
        "mod": "cumulate.nrmse_prefix",
        "out_dir": str(out_dir),
        "mode": mode,
        "eps": eps,
        "max_r": max_r,
        "written_json": written,
        "ok_count": ok_cnt,
        "fail_count": fail_cnt,
    }
    write_json(out_dir / "index.json", index)

    return {
        "out_dir": str(out_dir),
        "json_dir": str(json_dir),
        "index_json": str(out_dir / "index.json"),
        "written_json": written,
        "ok_count": ok_cnt,
        "fail_count": fail_cnt,
    }

def mod_cumulate_nrmse_leff_pack(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_leff"
    json_dir = out_dir / "json"
    _ensure_dir(json_dir)

    mode = str(kwargs.get("mode", "coeff"))
    eps = float(kwargs.get("eps", 1e-12))
    max_r = kwargs.get("max_r", None)
    max_r = None if max_r is None else int(max_r)

    agg_kind = str(kwargs.get("agg_kind", "min"))  # "min"|"geo"
    leff_pack = load_pod_mode_scales_standardized(ctx, agg_kind=agg_kind)

    written = []
    ok_cnt = 0
    fail_cnt = 0

    for model_type in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            stem = entry_name(str(model_type), float(p), float(s))
            out_json = json_dir / f"{stem}.json"
            try:
                A_hat, A_true, meta, npz_path = load_coeff_pair(ctx, str(model_type), float(p), float(s))
                r_grid, nrmse_r, method = compute_nrmse_prefix(A_hat, A_true, mode=mode, eps=eps, max_r=max_r)

                merged = merge_leff_and_nrmse(
                    leff_pack=leff_pack,
                    r_grid_nrmse=r_grid,
                    nrmse_r=nrmse_r,
                    max_r_cfg=max_r,
                )

                payload = {
                    "stem": stem,
                    "meta": {
                        "model_type": str(model_type),
                        "mask_rate": float(p),
                        "noise_sigma": float(s),
                        "source_npz": str(npz_path),
                        "eps": float(eps),
                        "method": method,
                        "max_r_cfg": max_r,
                        "coeff_keys": {
                            "A_hat": meta.get("k_hat", None),
                            "A_true": meta.get("k_true", None),
                        },
                        "T": int(meta.get("T", A_true.shape[0])),
                        "R_total": int(meta.get("R", A_true.shape[1])),
                        "centered_pod": meta.get("centered_pod", None),
                    },
                    "merged": merged,
                }

                write_json(out_json, payload)
                written.append(str(out_json))
                ok_cnt += 1
                print(f"[L4:cumulate.nrmse_leff_pack] OK  {stem}  R_used={merged['R_used']} -> {out_json.name}")
            except Exception as e:
                fail_cnt += 1
                print(f"[L4:cumulate.nrmse_leff_pack] FAIL {stem}: {type(e).__name__}: {e}")

    index = {
        "mod": "cumulate.nrmse_leff_pack",
        "out_dir": str(out_dir),
        "json_dir": str(json_dir),
        "written_json": written,
        "ok_count": ok_cnt,
        "fail_count": fail_cnt,
    }
    write_json(out_dir / "index.json", index)

    return index


def mod_cumulate_nrmse_leff_plot(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-3：读取 nrmse_leff/json/<stem>.json，并按分组出图 + 输出拟合参数表。

    group_by:
      - "model_type"（默认）：每个 model_type 一张图，含所有(p,σ)
      - "all"：所有 model_type 混在一张图（总览用）
      - "p"：按固定 p 分组出图（同 p 比 σ）
      - "sigma"：按固定 σ 分组出图（同 σ 比 p）
    """
    assert ctx.paths is not None

    import csv

    base_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_leff"
    json_dir = base_dir / "json"
    if not json_dir.exists():
        raise FileNotFoundError(
            f"[cumulate.nrmse_leff_plot] missing json_dir: {json_dir}. "
            f"Run cumulate.nrmse_leff_pack first."
        )

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_leff_plot"
    fig_dir = out_dir / "figs"
    fit_dir = out_dir / "fit_params"
    _ensure_dir(fig_dir)
    _ensure_dir(fit_dir)

    # ---- unified kwargs (plot config) ----
    which_leff = str(kwargs.get("which_leff", "agg"))          # "x"|"y"|"agg"
    sort_by_leff = bool(kwargs.get("sort_by_leff", False))
    group_by = str(kwargs.get("group_by", "model_type"))       # "model_type"|"all"|"p"|"sigma"
    dpi = int(kwargs.get("dpi", 180))

    plot_mode = str(kwargs.get("plot_mode", "scatter_fit"))    # "scatter_fit"|"scatter_only"|"line"
    fit_kind = str(kwargs.get("fit_kind", "poly"))             # "poly"|"log"|"exp"|"power"
    fit_degree = int(kwargs.get("fit_degree", 2))              # only for poly
    fit_points = int(kwargs.get("fit_points", 200))
    fit_eps = float(kwargs.get("fit_eps", 1e-12))

    annotate_every = int(kwargs.get("annotate_every", 20))
    annotate_mode = str(kwargs.get("annotate_mode", "none"))   # 默认关：20组会很乱
    annotate_fontsize = int(kwargs.get("annotate_fontsize", 9))

    legend_mode = str(kwargs.get("legend_mode", "fit"))        # "fit"|"short"|"none"
    label_digits = int(kwargs.get("label_digits", 3))
    legend_outside = bool(kwargs.get("legend_outside", True))

    if group_by not in ("model_type", "all", "p", "sigma"):
        raise ValueError(f"[cumulate.nrmse_leff_plot] invalid group_by='{group_by}'")

    # ---- read curves & group them ----
    curves_by_group: Dict[str, List[Dict[str, Any]]] = {}

    for model_type in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            stem = entry_name(str(model_type), float(p), float(s))
            path = json_dir / f"{stem}.json"
            if not path.exists():
                continue

            obj = read_json(path)
            merged = obj.get("merged", {})
            leff = merged.get("leff", {}) if isinstance(merged.get("leff", {}), dict) else {}
            nrmse = merged.get("nrmse_r", None)
            if nrmse is None:
                continue

            x = leff.get(which_leff, None)
            if x is None:
                continue

            curve = {
                "x": np.asarray(x, dtype=float),
                "y": np.asarray(nrmse, dtype=float),
                # base label: 简短但可辨识（详细参数交给 fit label）
                "label": f"p={float(p):.3g}, σ={float(s):.3g}",
                # meta for fit export
                "meta": {
                    "model_type": str(model_type),
                    "mask_rate": float(p),
                    "noise_sigma": float(s),
                    "stem": stem,
                },
            }

            if group_by == "all":
                g = "ALL"
            elif group_by == "model_type":
                g = str(model_type)
            elif group_by == "p":
                g = f"{model_type}_p={float(p):.3g}"
            else:  # "sigma"
                g = f"{model_type}_sigma={float(s):.3g}"

            curves_by_group.setdefault(g, []).append(curve)

    fig_paths: Dict[str, str] = {}
    fit_json_paths: Dict[str, str] = {}
    fit_csv_paths: Dict[str, str] = {}

    for g, curves in curves_by_group.items():
        if not curves:
            continue

        title = f"NRMSE vs ℓ_eff(r) [{g}]"
        xlabel = f"ℓ_eff(r) ({which_leff})"
        ylabel = "NRMSE(r)"

        make_zoom = bool(kwargs.get("make_zoom", False))
        yscale = str(kwargs.get("yscale", "linear"))
        ymin = kwargs.get("ymin", None)  # None or float
        zoom_ymax = float(kwargs.get("zoom_ymax", 0.05))

        if make_zoom:
            fig, fit_summaries, extra = plot_nrmse_leff_curves(
                curves,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                which_leff=which_leff,
                sort_by_leff=sort_by_leff,
                plot_mode=plot_mode,
                fit_kind=fit_kind,
                fit_degree=fit_degree,
                fit_points=fit_points,
                fit_eps=fit_eps,
                annotate_every=annotate_every,
                annotate_mode=annotate_mode,
                annotate_fontsize=annotate_fontsize,
                legend_mode=legend_mode,
                label_digits=label_digits,
                legend_outside=legend_outside,
                # --- new knobs ---
                yscale=yscale,
                ymin=ymin,
                make_zoom=True,
                zoom_ymax=zoom_ymax,
                # --- return control ---
                return_fit_summaries=True,
                return_extra_figs=True,
            )
        else:
            fig, fit_summaries = plot_nrmse_leff_curves(
                curves,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                which_leff=which_leff,
                sort_by_leff=sort_by_leff,
                plot_mode=plot_mode,
                fit_kind=fit_kind,
                fit_degree=fit_degree,
                fit_points=fit_points,
                fit_eps=fit_eps,
                annotate_every=annotate_every,
                annotate_mode=annotate_mode,
                annotate_fontsize=annotate_fontsize,
                legend_mode=legend_mode,
                label_digits=label_digits,
                legend_outside=legend_outside,
                # --- new knobs ---
                yscale=yscale,
                ymin=ymin,
                # --- return control ---
                return_fit_summaries=True,
            )
            extra = {}

        safe_g = g.replace("/", "_").replace(" ", "_")
        png = fig_dir / f"nrmse_leff_{safe_g}_{which_leff}_{plot_mode}_{fit_kind}.png"
        _save_fig(fig, png, dpi=dpi)
        fig_paths[g] = str(png)

        # ---- write fit summaries (json) ----
        fit_json = fit_dir / f"fit_{safe_g}_{which_leff}_{fit_kind}.json"
        write_json(fit_json, {"group": g, "which_leff": which_leff, "fit_kind": fit_kind, "items": fit_summaries})
        fit_json_paths[g] = str(fit_json)

        # ---- write fit summaries (csv) ----
        fit_csv = fit_dir / f"fit_{safe_g}_{which_leff}_{fit_kind}.csv"

        # 统一输出列（poly 也能落 coef；exp/log/power 落 a,b）
        fieldnames = [
            "group",
            "model_type",
            "mask_rate",
            "noise_sigma",
            "stem",
            "fit_kind",
            "a",
            "b",
            "degree",
            "coef",   # json string
        ]

        with fit_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()

            for it in fit_summaries:
                meta = it.get("fit_meta", {}) if isinstance(it.get("fit_meta", {}), dict) else {}
                k = str(it.get("fit_kind", meta.get("kind", "")))

                row = {
                    "group": g,
                    "model_type": it.get("model_type", ""),
                    "mask_rate": it.get("mask_rate", ""),
                    "noise_sigma": it.get("noise_sigma", ""),
                    "stem": it.get("stem", ""),
                    "fit_kind": k,
                    "a": "",
                    "b": "",
                    "degree": "",
                    "coef": "",
                }

                if k in ("exp", "log", "power"):
                    row["a"] = meta.get("a", "")
                    row["b"] = meta.get("b", "")
                elif k == "poly":
                    row["degree"] = meta.get("degree", "")
                    row["coef"] = str(meta.get("coef", ""))

                w.writerow(row)

        fit_csv_paths[g] = str(fit_csv)

    index = {
        "mod": "cumulate.nrmse_leff_plot",
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "fit_dir": str(fit_dir),
        "fig_paths": fig_paths,
        "fit_json_paths": fit_json_paths,
        "fit_csv_paths": fit_csv_paths,
        "which_leff": which_leff,
        "sort_by_leff": sort_by_leff,
        "group_by": group_by,
        "dpi": dpi,
        "plot_mode": plot_mode,
        "fit_kind": fit_kind,
        "fit_degree": fit_degree,
        "fit_points": fit_points,
        "fit_eps": fit_eps,
        "annotate_every": annotate_every,
        "annotate_mode": annotate_mode,
        "legend_mode": legend_mode,
        "label_digits": label_digits,
        "legend_outside": legend_outside,
    }
    write_json(out_dir / "index.json", index)
    return index


def register_cumulate_mods() -> None:
    register_mod(
        EvalMod(
            name="cumulate.quick_check",
            run=mod_cumulate_quick_check,
            description="cumulate: IO quick check for POD(L1)+L2 inputs (Batch-0).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.nrmse_prefix",
            run=mod_cumulate_nrmse_prefix,
            description="cumulate: compute NRMSE(r) prefix curves from L2 coeffs (Batch-1, numeric only).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.nrmse_leff_pack",
            run=mod_cumulate_nrmse_leff_pack,
            description="cumulate: merge leff(r) from L1 with NRMSE(r) from L2 (Batch-2, numeric only).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.nrmse_leff_plot",
            run=mod_cumulate_nrmse_leff_plot,
            description="cumulate: plot NRMSE vs leff(r) curves (Batch-3).",
        )
    )
