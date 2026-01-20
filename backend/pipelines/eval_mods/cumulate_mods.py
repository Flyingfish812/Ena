# backend/pipelines/eval_mods/cumulate_mods.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval.utils import read_json, write_json, entry_name

from backend.pipelines.eval_mods.cumulate_io import (
    load_coeff_pair,
    load_pod_mode_scales_standardized,
)

from backend.metrics.cumulate_metrics import compute_nrmse_prefix, merge_scales_and_nrmse

from backend.viz.cumulate_plots import (
    render_save_nrmse_vs_r,
    render_save_dual_vs_r,
)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_stem(s: str) -> str:
    """把 group 名转成安全文件名。"""
    t = str(s).strip()
    for ch in [" ", "/", "\\", ":", ";", ",", "|", "=", "(", ")", "[", "]", "{", "}", "<", ">", "@", "#"]:
        t = t.replace(ch, "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_") if t else "group"


def _load_nrmse_scales_pack_entries(ctx) -> List[Dict[str, Any]]:
    """
    读取 Batch-1 产物目录：
      L4_eval/cumulate/nrmse_scales/json/*.json
    返回每个 entry 的 dict（原样）。
    """
    root = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_scales" / "json"
    if not root.exists():
        raise FileNotFoundError(f"nrmse_scales json dir not found: {root}")

    files = sorted(root.glob("*.json"))
    if len(files) == 0:
        raise FileNotFoundError(f"no json found under: {root}")

    return [read_json(fp) for fp in files]


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


def mod_cumulate_nrmse_scales_pack(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-1（联合落盘，为 Batch-2/4 提供统一输入）：
      - 从 L1 读取一次 scale_table.csv -> 6条尺度曲线（x/y × med/prefix/tail）
      - 对每个 (model_type, p, σ) 从 L2 计算 NRMSE(r)
      - merge 对齐截断 -> 写入 L4 json

    输出：
      L4_eval/cumulate/nrmse_scales/json/<stem>.json
      其中 merged 内含：
        - r_grid
        - nrmse_r
        - scales: 6条尺度数组 + colmap/path
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_scales"
    json_dir = out_dir / "json"
    _ensure_dir(json_dir)

    mode = str(kwargs.get("mode", "coeff"))
    eps = float(kwargs.get("eps", 1e-12))
    max_r = kwargs.get("max_r", None)
    max_r = None if max_r is None else int(max_r)

    # ---- load scales once (avoid repeated IO) ----
    scales_pack = load_pod_mode_scales_standardized(ctx)

    written: List[str] = []
    ok_cnt = 0
    fail_cnt = 0

    for model_type in (ctx.model_types or ()):
        for (p, s) in ctx.iter_cfgs():
            stem = entry_name(str(model_type), float(p), float(s))
            out_json = json_dir / f"{stem}.json"

            try:
                A_hat, A_true, meta, npz_path = load_coeff_pair(ctx, str(model_type), float(p), float(s))
                r_grid, nrmse_r, method = compute_nrmse_prefix(A_hat, A_true, mode=mode, eps=eps, max_r=max_r)

                merged = merge_scales_and_nrmse(
                    scales_pack=scales_pack,
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
                        "R_used": int(merged.get("R_used", len(merged.get("r_grid", [])))),
                        "centered_pod": meta.get("centered_pod", None),
                    },
                    "merged": merged,
                }

                write_json(out_json, payload)
                written.append(str(out_json))
                ok_cnt += 1
                print(f"[L4:cumulate.nrmse_scales_pack] OK  {stem}  R_used={payload['meta']['R_used']} -> {out_json.name}")

            except Exception as e:
                fail_cnt += 1
                print(f"[L4:cumulate.nrmse_scales_pack] FAIL {stem}: {type(e).__name__}: {e}")

    index = {
        "mod": "cumulate.nrmse_scales_pack",
        "out_dir": str(out_dir),
        "json_dir": str(json_dir),
        "mode": mode,
        "eps": eps,
        "max_r": max_r,
        "written_json": written,
        "ok_count": ok_cnt,
        "fail_count": fail_cnt,
    }
    write_json(out_dir / "index.json", index)
    return index


def mod_cumulate_nrmse_vs_r_plot(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-2：画 NRMSE(r) vs r
    - 只读 Batch-1 产物 json
    - 不做 plt / savefig / close（全部委托给 viz.render_save_*）
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_vs_r"
    fig_dir = out_dir / "fig"
    _ensure_dir(fig_dir)

    entries = _load_nrmse_scales_pack_entries(ctx)

    # plot knobs
    yscale = str(kwargs.get("yscale", "log"))
    annotate_every = int(kwargs.get("annotate_every", 20))
    legend_outside = bool(kwargs.get("legend_outside", True))
    make_zoom = bool(kwargs.get("make_zoom", False))
    zoom_ymax = float(kwargs.get("zoom_ymax", 0.05))
    dpi = int(kwargs.get("dpi", 200))
    title = str(kwargs.get("title", "NRMSE(r) vs r"))

    # grouping
    group_by = str(kwargs.get("group_by", "all"))  # "all"|"model_type"|"mask_rate"|"noise_sigma"
    label_mode = str(kwargs.get("label_mode", "stem"))  # "stem"|"short"

    def _group_key(e: Dict[str, Any]) -> str:
        meta = e.get("meta", {}) or {}
        if group_by == "model_type":
            return str(meta.get("model_type", "unknown"))
        if group_by == "mask_rate":
            return f"p={meta.get('mask_rate', 'unknown')}"
        if group_by == "noise_sigma":
            return f"sigma={meta.get('noise_sigma', 'unknown')}"
        return "all"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(_group_key(e), []).append(e)

    written: List[str] = []

    for gname, els in groups.items():
        curves = []
        for e in els:
            meta = e.get("meta", {}) or {}
            merged = e.get("merged", {}) or {}
            r_grid = np.asarray(merged.get("r_grid", []), dtype=float)
            nrmse_r = np.asarray(merged.get("nrmse_r", []), dtype=float)

            if label_mode == "short":
                label = f"{meta.get('model_type','?')}, p={meta.get('mask_rate','?')}, s={meta.get('noise_sigma','?')}"
            else:
                label = str(e.get("stem", ""))

            curves.append({"label": label, "x": r_grid, "y": nrmse_r})

        stem = "all" if gname == "all" else safe_stem(gname)
        out_png = fig_dir / f"nrmse_vs_r__{stem}.png"
        out_pngz = fig_dir / f"nrmse_vs_r__{stem}__zoom.png" if make_zoom else None

        w = render_save_nrmse_vs_r(
            curves,
            out_png=out_png,
            out_png_zoom=out_pngz,
            dpi=dpi,
            title=(title if gname == "all" else f"{title} | {gname}"),
            yscale=yscale,
            annotate_every=annotate_every,
            legend_outside=legend_outside,
            make_zoom=make_zoom,
            zoom_ymax=zoom_ymax,
        )

        if w.get("main"):
            written.append(w["main"])
        if w.get("zoom"):
            written.append(w["zoom"])

    index = {
        "mod": "cumulate.nrmse_vs_r_plot",
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "group_by": group_by,
        "label_mode": label_mode,
        "yscale": yscale,
        "annotate_every": annotate_every,
        "legend_outside": legend_outside,
        "make_zoom": make_zoom,
        "zoom_ymax": zoom_ymax,
        "dpi": dpi,
        "written": written,
    }
    write_json(out_dir / "index.json", index)
    return index


def mod_cumulate_dual_vs_r_plot(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-4：双纵轴图：scale(left) + nrmse(right) vs r
    - 只读 Batch-1 产物 json
    - scale_key 由 (axis,family) 选择：ell_{x|y}_{med|prefix|tail}
    - 不做 plt / savefig / close（全部委托给 viz.render_save_*）
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "dual_vs_r"
    fig_dir = out_dir / "fig"
    _ensure_dir(fig_dir)

    entries = _load_nrmse_scales_pack_entries(ctx)

    scale_axis = str(kwargs.get("scale_axis", "x")).lower().strip()
    if scale_axis not in ("x", "y"):
        scale_axis = "x"
    scale_family = str(kwargs.get("scale_family", "prefix")).lower().strip()
    if scale_family not in ("med", "prefix", "tail"):
        scale_family = "prefix"
    scale_key = f"ell_{scale_axis}_{scale_family}"

    invert_left = bool(kwargs.get("invert_left", True))
    left_yscale = str(kwargs.get("left_yscale", "log"))
    right_yscale = str(kwargs.get("right_yscale", "log"))
    annotate_every = int(kwargs.get("annotate_every", 20))
    annotate_on = str(kwargs.get("annotate_on", "right"))  # "left"|"right"|"none"
    legend_outside = bool(kwargs.get("legend_outside", True))
    dpi = int(kwargs.get("dpi", 200))
    title = str(kwargs.get("title", f"{scale_key}(left) + NRMSE(right) vs r"))

    group_by = str(kwargs.get("group_by", "all"))
    label_mode = str(kwargs.get("label_mode", "stem"))

    def _group_key(e: Dict[str, Any]) -> str:
        meta = e.get("meta", {}) or {}
        if group_by == "model_type":
            return str(meta.get("model_type", "unknown"))
        if group_by == "mask_rate":
            return f"p={meta.get('mask_rate', 'unknown')}"
        if group_by == "noise_sigma":
            return f"sigma={meta.get('noise_sigma', 'unknown')}"
        return "all"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(_group_key(e), []).append(e)

    written: List[str] = []

    for gname, els in groups.items():
        curves = []
        for e in els:
            meta = e.get("meta", {}) or {}
            merged = e.get("merged", {}) or {}
            r_grid = np.asarray(merged.get("r_grid", []), dtype=float)
            nrmse_r = np.asarray(merged.get("nrmse_r", []), dtype=float)
            scales = (merged.get("scales", {}) or {})
            y_left = np.asarray(scales.get(scale_key, []), dtype=float)

            if label_mode == "short":
                label = f"{meta.get('model_type','?')}, p={meta.get('mask_rate','?')}, s={meta.get('noise_sigma','?')}"
            else:
                label = str(e.get("stem", ""))

            curves.append({"label": label, "x": r_grid, "y_left": y_left, "y_right": nrmse_r})

        stem = "all" if gname == "all" else safe_stem(gname)
        out_png = fig_dir / f"dual_vs_r__{scale_key}__{stem}.png"

        p = render_save_dual_vs_r(
            curves,
            out_png=out_png,
            dpi=dpi,
            title=(title if gname == "all" else f"{title} | {gname}"),
            ylabel_left=f"{scale_key}(r)  (smaller=finer)",
            ylabel_right="NRMSE(r)",
            left_yscale=left_yscale,
            right_yscale=right_yscale,
            invert_left=invert_left,
            annotate_every=annotate_every,
            annotate_on=annotate_on,
            legend_outside=legend_outside,
        )
        if p:
            written.append(p)

    index = {
        "mod": "cumulate.dual_vs_r_plot",
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "group_by": group_by,
        "label_mode": label_mode,
        "scale_axis": scale_axis,
        "scale_family": scale_family,
        "scale_key": scale_key,
        "invert_left": invert_left,
        "left_yscale": left_yscale,
        "right_yscale": right_yscale,
        "annotate_every": annotate_every,
        "annotate_on": annotate_on,
        "legend_outside": legend_outside,
        "dpi": dpi,
        "written": written,
    }
    write_json(out_dir / "index.json", index)
    return index


def register_cumulate_mods() -> None:
    register_mod(
        EvalMod(
            name="cumulate.nrmse_scales_pack",
            run=mod_cumulate_nrmse_scales_pack,
            description="cumulate: merge 6-scale curves (med/prefix/tail, x/y) with NRMSE(r) into packed json (Batch-1).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.nrmse_vs_r_plot",
            run=mod_cumulate_nrmse_vs_r_plot,
            description="cumulate: plot NRMSE(r) vs r from packed json (Batch-2).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.dual_vs_r_plot",
            run=mod_cumulate_dual_vs_r_plot,
            description="cumulate: plot dual-axis scale(left)+NRMSE(right) vs r from packed json (Batch-4).",
        )
    )
