# backend/pipelines/eval_mods/cumulate_mods.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from backend.pipelines.eval.registry import EvalMod, register_mod
from backend.pipelines.eval.utils import read_json, write_json, entry_name

from backend.pipelines.eval_mods.cumulate_io import (
    load_coeff_pair,
    load_pod_mode_scales_standardized,
)

from backend.metrics.cumulate_metrics import compute_nrmse_vs_r, merge_scales_and_nrmsepack

from backend.viz.cumulate_plots import (
    render_save_nrmse_family_vs_r,
    render_save_dual_xy_vs_r,
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


def mod_cumulate_nrmse_scales_pack(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Batch-1（联合落盘，为后续画图提供统一输入）：
      - 从 L1 读取 scale_table.csv -> 6条尺度曲线（x/y × med/prefix/tail）
      - 对每个 (model_type, p, σ) 从 L2 计算三种 NRMSE 曲线：nrmse_full/nrmse_prefix/nrmse_tail
      - merge 对齐截断 -> 写入 L4 json

    输出：
      L4_eval/cumulate/nrmse_scales/json/<stem>.json
      merged 内含：
        - r_grid
        - scales: 6条尺度数组 + colmap/path
        - nrmse: {"nrmse_full","nrmse_prefix","nrmse_tail"}
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_scales"
    json_dir = out_dir / "json"
    _ensure_dir(json_dir)

    mode = str(kwargs.get("mode", "coeff"))
    eps = float(kwargs.get("eps", 1e-12))
    max_r = kwargs.get("max_r", None)
    max_r = None if max_r is None else int(max_r)

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

                r_grid, nrmse, method = compute_nrmse_vs_r(A_hat, A_true, mode=mode, eps=eps, max_r=max_r)

                merged = merge_scales_and_nrmsepack(
                    scales_pack=scales_pack,
                    r_grid_nrmse=r_grid,
                    nrmse_pack=nrmse,
                    max_r_cfg=max_r,
                )

                R_used = int(merged.get("R_used", len(merged.get("r_grid", []))))

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
                        "R_used": int(R_used),
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
    Batch-2：画 “NRMSE family vs r”（可选 nrmse_full/nrmse_prefix/nrmse_tail）
    - 只读 Batch-1 packed json（nrmse_scales/json/*.json）
    - 线性坐标轴（常数轴）
    - 不拟合
    - 同一(model_type,p,s)的三条线同色，不同线型
    - legend 放到图外，不遮挡数轴
    """
    assert ctx.paths is not None

    out_dir = Path(ctx.paths.l4_root) / "cumulate" / "nrmse_vs_r"
    fig_dir = out_dir / "fig"
    _ensure_dir(fig_dir)

    entries = _load_nrmse_scales_pack_entries(ctx)

    group_by = str(kwargs.get("group_by", "p")).lower().strip()  # "p"|"sigma"
    if group_by not in ("p", "sigma"):
        group_by = "p"

    nrmse_kinds = kwargs.get("nrmse_kinds", ["nrmse_full", "nrmse_prefix", "nrmse_tail"])
    if isinstance(nrmse_kinds, str):
        nrmse_kinds = [nrmse_kinds]
    nrmse_kinds = [str(x).strip() for x in list(nrmse_kinds)]
    allowed = {"nrmse_full", "nrmse_prefix", "nrmse_tail"}
    nrmse_kinds = [k for k in nrmse_kinds if k in allowed]
    if len(nrmse_kinds) == 0:
        nrmse_kinds = ["nrmse_full", "nrmse_prefix", "nrmse_tail"]

    label_mode = str(kwargs.get("label_mode", "short")).lower().strip()  # "stem"|"short"
    dpi = int(kwargs.get("dpi", 200))
    title = str(kwargs.get("title", "NRMSE family vs r (linear axis)"))
    legend_outside = bool(kwargs.get("legend_outside", True))

    def _group_key(e: Dict[str, Any]) -> str:
        meta = e.get("meta", {}) or {}
        if group_by == "p":
            return f"p={meta.get('mask_rate', 'unknown')}"
        return f"sigma={meta.get('noise_sigma', 'unknown')}"

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
            nrmse = (merged.get("nrmse", {}) or {})

            if label_mode == "stem":
                label = str(e.get("stem", ""))
            else:
                label = f"{meta.get('model_type','?')}, p={meta.get('mask_rate','?')}, s={meta.get('noise_sigma','?')}"

            curves.append({
                "label": label,
                "x": r_grid,
                "nrmse": {
                    "nrmse_full": np.asarray(nrmse.get("nrmse_full", []), dtype=float),
                    "nrmse_prefix": np.asarray(nrmse.get("nrmse_prefix", []), dtype=float),
                    "nrmse_tail": np.asarray(nrmse.get("nrmse_tail", []), dtype=float),
                },
            })

        stem = safe_stem(gname)
        out_png = fig_dir / f"nrmse_vs_r_by_{group_by}_{stem}.png"

        pth = render_save_nrmse_family_vs_r(
            curves,
            out_png=out_png,
            dpi=dpi,
            title=f"{title} | {gname}",
            nrmse_kinds=nrmse_kinds,
            legend_outside=legend_outside,
        )
        if pth:
            written.append(pth)

    index = {
        "mod": "cumulate.nrmse_vs_r_plot",
        "out_dir": str(out_dir),
        "fig_dir": str(fig_dir),
        "group_by": group_by,
        "nrmse_kinds": nrmse_kinds,
        "label_mode": label_mode,
        "legend_outside": legend_outside,
        "dpi": dpi,
        "written": written,
    }
    write_json(out_dir / "index.json", index)
    return index


def mod_cumulate_dual_vs_r_plot(ctx, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    新版 Batch-4：dual plot（x/y 分图）

    默认生成三个类别（每类一个文件夹）：
      1) (ell_prefix, nrmse_full)
      2) (ell_prefix, nrmse_prefix)
      3) (ell_tail,   nrmse_tail)

    特性：
      - x/y 分别画（左右两个子图）
      - ℓ_*：黑色 scatter，仅一组（与配置无关）
      - NRMSE：彩色实线，多配置区分
      - 常数坐标轴
      - legend 统一放底部
      - 支持按 p 或 σ 分组
    """
    assert ctx.paths is not None

    group_by = str(kwargs.get("group_by", "p")).lower().strip()
    if group_by not in ("p", "sigma"):
        group_by = "p"

    dpi = int(kwargs.get("dpi", 200))
    legend_outside = bool(kwargs.get("legend_outside", True))

    # 三个默认类别
    categories = kwargs.get(
        "categories",
        [
            ("prefix", "nrmse_full"),
            ("prefix", "nrmse_prefix"),
            ("tail", "nrmse_tail"),
        ],
    )

    entries = _load_nrmse_scales_pack_entries(ctx)

    def _group_key(e: Dict[str, Any]) -> str:
        meta = e.get("meta", {}) or {}
        if group_by == "p":
            return f"p={meta.get('mask_rate', 'unknown')}"
        return f"sigma={meta.get('noise_sigma', 'unknown')}"

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(_group_key(e), []).append(e)

    written: List[str] = []

    base_out = Path(ctx.paths.l4_root) / "cumulate" / "dual_xy_vs_r"
    _ensure_dir(base_out)

    for scale_family, nrmse_kind in categories:
        cat_dir = base_out / f"{scale_family}_{nrmse_kind}"
        fig_dir = cat_dir / "fig"
        _ensure_dir(fig_dir)

        for gname, els in groups.items():
            # ---- scales: take from first entry only ----
            merged0 = els[0]["merged"]
            r_grid = np.asarray(merged0["r_grid"], dtype=float)

            ell_x = np.asarray(merged0["scales"][f"ell_x_{scale_family}"], dtype=float)
            ell_y = np.asarray(merged0["scales"][f"ell_y_{scale_family}"], dtype=float)

            curves_right = []
            for e in els:
                meta = e.get("meta", {}) or {}
                merged = e.get("merged", {}) or {}
                y = np.asarray(merged["nrmse"][nrmse_kind], dtype=float)

                label = f"{meta.get('model_type','?')}, p={meta.get('mask_rate','?')}, s={meta.get('noise_sigma','?')}"
                curves_right.append(
                    {
                        "label": label,
                        "y": y,
                        "color": None,  # matplotlib auto cycle
                    }
                )

            # assign colors consistently
            for i, c in enumerate(curves_right):
                c["color"] = plt.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]

            stem = safe_stem(gname)
            out_png = fig_dir / f"dual_xy_vs_r_{stem}.png"

            p = render_save_dual_xy_vs_r(
                r_grid=r_grid,
                ell_x=ell_x,
                ell_y=ell_y,
                curves_right=curves_right,
                out_png=out_png,
                dpi=dpi,
                title=f"scale_{scale_family} + {nrmse_kind} | {gname}",
                legend_outside=legend_outside,
            )
            if p:
                written.append(p)

        write_json(
            cat_dir / "index.json",
            {
                "scale_family": scale_family,
                "nrmse_kind": nrmse_kind,
                "group_by": group_by,
                "fig_dir": str(fig_dir),
                "written": written,
            },
        )

    return {
        "mod": "cumulate.dual_vs_r_plot",
        "out_dir": str(base_out),
        "group_by": group_by,
        "categories": categories,
        "written": written,
    }


def register_cumulate_mods() -> None:
    register_mod(
        EvalMod(
            name="cumulate.nrmse_scales_pack",
            run=mod_cumulate_nrmse_scales_pack,
            description="cumulate: merge 6-scale curves (med/prefix/tail, x/y) with NRMSE family into packed json (Batch-1).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.nrmse_vs_r_plot",
            run=mod_cumulate_nrmse_vs_r_plot,
            description="cumulate: plot selectable NRMSE kinds (nrmse_full/prefix/tail) vs r on linear axis, grouped by p or sigma (Batch-2).",
        )
    )
    register_mod(
        EvalMod(
            name="cumulate.dual_vs_r_plot",
            run=mod_cumulate_dual_vs_r_plot,
            description="cumulate: plot dual-subplot ell_x/ell_y (left) + selected NRMSE kind (right) vs r from packed json (new Batch-4).",
        )
    )
