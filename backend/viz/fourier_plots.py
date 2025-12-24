# backend/viz/fourier_plots.py

from __future__ import annotations

from typing import Dict, Any, Iterable, Sequence, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def _ensure_hw(x: np.ndarray, *, channel: int = 0) -> np.ndarray:
    """
    将输入转成 [H,W] 单通道数组：
    - x: [H,W] or [H,W,C]
    """
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        if channel < 0 or channel >= x.shape[2]:
            raise ValueError(f"channel {channel} out of range for x shape {x.shape}")
        return x[..., channel]
    raise ValueError(f"Expected x.ndim in {{2,3}}, got {x.ndim} with shape {x.shape}")


def _radial_k_grid(
    H: int,
    W: int,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    构造以 FFT 频率为基础的 radial wavenumber 网格 k(y,x)。
    这里的单位是 cycles / length（即频率），不是角频率。
    """
    ky = np.fft.fftfreq(H, d=dy)  # cycles/len
    kx = np.fft.fftfreq(W, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX**2 + KY**2)


def _fft_bandpass_2d(
    x_hw: np.ndarray,
    *,
    k_low: float | None,
    k_high: float | None,
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    在频域做一个“radial band-pass”并 iFFT 回空间域。
    - k_low=None 表示 0
    - k_high=None 表示 +inf
    """
    x_hw = np.asarray(x_hw, dtype=float)
    H, W = x_hw.shape
    K = _radial_k_grid(H, W, dx=dx, dy=dy)

    X = np.fft.fft2(x_hw)
    mask = np.ones_like(K, dtype=bool)
    if k_low is not None:
        mask &= (K >= float(k_low))
    if k_high is not None:
        mask &= (K < float(k_high))

    X_f = X * mask
    x_f = np.fft.ifft2(X_f).real
    return x_f


def plot_energy_spectrum_with_band_edges(
    k_centers: np.ndarray,
    energy_k: np.ndarray,
    *,
    k_edges: Sequence[float] | None = None,          # interior edges: [k1,k2,...]
    band_names: Sequence[str] = ("L", "M", "H"),
    grid_meta: dict | None = None,                   # NEW: for physical annotation
    title: str = "Energy spectrum E(k) with band edges",
    xlabel: str = "wavenumber k (cycles / length unit)",
    ylabel: str = "E(k)",
    loglog: bool = True,
) -> Optional[plt.Figure]:
    """
    解释图：E(k) + band edges。
    - k 单位：cycles / length unit（与 np.fft.fftfreq 的 d=dx/dy 一致）
    - 顶轴显示波长 λ=1/k，帮助把“频域划分”翻译成“空间尺度划分”
    """
    k = np.asarray(k_centers, dtype=float).reshape(-1)
    e = np.asarray(energy_k, dtype=float).reshape(-1)
    if k.size == 0 or e.size == 0 or k.size != e.size:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.6, 4.1))
    ax.plot(k, e, linewidth=2.0)

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    # ---- top axis: wavelength λ = 1/k ----
    eps = 1e-12

    def k_to_lam(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / np.maximum(x, eps)

    def lam_to_k(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / np.maximum(x, eps)

    secax = ax.secondary_xaxis("top", functions=(k_to_lam, lam_to_k))
    secax.set_xlabel("wavelength λ (length unit)")

    # optional: if obstacle diameter is known, also show λ/D as a small note
    g = dict(grid_meta or {})
    D = g.get("obstacle_diameter", None)
    if D is not None and float(D) > 0:
        secax.set_xlabel("wavelength λ (length unit)   [λ/D shown in band labels]")

    # ---- band edges + labels ----
    if k_edges is not None:
        edges = sorted([float(v) for v in k_edges if v is not None and np.isfinite(v) and v > 0])
        # vertical lines
        for ke in edges:
            ax.axvline(ke, linestyle="--", linewidth=1.0)

        # segments: [0,k1],[k1,k2],...,[klast, inf] (use view range to place text)
        x_min, x_max = ax.get_xlim()
        segs = [max(x_min, eps)] + [ke for ke in edges if x_min < ke < x_max] + [x_max]

        # build labels per visible segment
        nb = max(1, len(segs) - 1)
        for i in range(nb):
            a, b = segs[i], segs[i + 1]
            if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 and b > a):
                continue

            # geometric mid (log-axis friendly)
            xm = np.sqrt(a * b)
            name = band_names[i] if i < len(band_names) else f"B{i}"

            lam_hi = 1.0 / max(a, eps)  # larger wavelength at smaller k
            lam_lo = 1.0 / max(b, eps)

            # text: include λ range
            if i == 0:
                lam_text = f"λ ≥ {lam_lo:.3g}"
            elif i == nb - 1:
                lam_text = f"λ < {lam_hi:.3g}"
            else:
                lam_text = f"{lam_lo:.3g} ≤ λ < {lam_hi:.3g}"

            if D is not None and float(D) > 0:
                # show λ/D (dimensionless) as extra cue
                if i == 0:
                    lamD_text = f"λ/D ≥ {lam_lo/float(D):.3g}"
                elif i == nb - 1:
                    lamD_text = f"λ/D < {lam_hi/float(D):.3g}"
                else:
                    lamD_text = f"{lam_lo/float(D):.3g} ≤ λ/D < {lam_hi/float(D):.3g}"
                label = f"{name}\n({lam_text})\n({lamD_text})"
            else:
                label = f"{name}\n({lam_text})"

            # place near the top inside axes
            y_top = ax.get_ylim()[1]
            ax.text(xm, y_top, label, ha="center", va="top")

    fig.tight_layout()
    return fig


def plot_kstar_curve_from_entry(
    entry: Dict[str, Any],
    *,
    # --- style / labels (aligned with plot_energy_spectrum_with_band_edges) ---
    title: str | None = None,
    xlabel: str = "wavenumber k (cycles / length unit)",
    ylabel: str = r"Cumulative NRMSE$_{≤ K}$",
    loglog: bool = True,
    show_edges: bool = True,
    show_kstar: bool = True,
    # --- NEW: keep local curve as diagnostic overlay ---
    show_local_curve: bool = True,
    local_curve_label: str = "Local NRMSE(k) (diagnostic)",
    # --- meta (can be injected, otherwise auto-resolved from entry/curve) ---
    k_edges: Sequence[float] | None = None,                 # interior edges in k-domain: [k1,k2,...]
    band_names: Sequence[str] = ("L", "M", "H"),
    grid_meta: dict | None = None,                          # dx/dy/obstacle_diameter/Lx/Ly ...
) -> Optional[plt.Figure]:
    """
    【解释图 v1.17】单个配置的 Fourier 尺度恢复能力曲线图。

    主曲线（对外口径 / k* 定义依据）：
      - cumulative curve: NRMSE_{<=K} vs K  (curve["nrmse_cum"])

    诊断曲线（辅助解释，不作为 k* 的定义依据）：
      - local curve: NRMSE(k) vs k (curve["nrmse_k"])

    变更（v1.17 绘图可读性升级）：
      - 图内 band 只保留大标签（Background/Big/Medium/Small/Tiny）
      - 详细括号信息（λ 与 λ/D 范围）移动到图外 info panel（右侧）
      - 不改变 band 边界（由 k_edges 决定）
    """
    curve = entry.get("fourier_curve", None)
    if not isinstance(curve, dict):
        return None

    # ---- resolve meta (prefer explicit args, otherwise from entry/curve) ----
    meta = (
        entry.get("fourier_meta", None)
        or entry.get("fourier", None)
        or entry.get("meta", None)
        or {}
    )
    meta = meta if isinstance(meta, dict) else {}

    # grid_meta: explicit > meta["grid_meta"] > meta["grid"] > curve["grid_meta"] ...
    if grid_meta is None:
        grid_meta = (
            meta.get("grid_meta", None)
            or meta.get("grid", None)
            or curve.get("grid_meta", None)
            or curve.get("grid", None)
        )
    g = dict(grid_meta or {})
    dx = float(g.get("dx", 1.0))
    dy = float(g.get("dy", 1.0))
    D = g.get("obstacle_diameter", None)

    # k_edges: explicit > curve > meta
    if k_edges is None:
        k_edges = curve.get("k_edges", None) or meta.get("k_edges", None)

    # band_names: explicit > curve > meta
    bn_from_curve = curve.get("band_names", None) or meta.get("band_names", None)
    if bn_from_curve is not None:
        band_names = bn_from_curve

    # ---- load k centers ----
    k = np.asarray(curve.get("k_centers", []), dtype=float).reshape(-1)
    if k.size == 0:
        return None

    # ---- load cumulative curve (MAIN) ----
    y_cum = np.asarray(curve.get("nrmse_cum", []), dtype=float).reshape(-1)
    if y_cum.size == 0 or y_cum.size != k.size:
        return None

    # ---- load local curve (optional diagnostic) ----
    y_local = None
    if show_local_curve:
        y_local_arr = curve.get("nrmse_k", None)
        if y_local_arr is not None:
            y_local = np.asarray(y_local_arr, dtype=float).reshape(-1)
            if y_local.size != k.size:
                y_local = None

    # ---- clean + sort by k ----
    eps = 1e-12
    mask_main = np.isfinite(k) & np.isfinite(y_cum) & (k > 0)
    if not np.any(mask_main):
        return None

    k_main = k[mask_main]
    y_main = y_cum[mask_main]
    order = np.argsort(k_main)
    k_sorted = k_main[order]
    y_sorted = y_main[order]

    # diagnostic curve aligned to k_sorted where possible
    y_local_sorted = None
    if y_local is not None:
        mask_local = np.isfinite(k) & np.isfinite(y_local) & (k > 0)
        if np.any(mask_local):
            k_loc = k[mask_local]
            y_loc = y_local[mask_local]
            ord2 = np.argsort(k_loc)
            k_loc_s = k_loc[ord2]
            y_loc_s = y_loc[ord2]
            if k_loc_s.size == k_sorted.size and np.allclose(k_loc_s, k_sorted, rtol=0, atol=0):
                y_local_sorted = y_loc_s

    # ---- title suffix: show p, sigma, model ----
    p = entry.get("mask_rate", None)
    sigma = entry.get("noise_sigma", None)
    model = entry.get("model_name", entry.get("model", None))

    if title is None:
        parts = []
        if p is not None:
            try:
                parts.append(f"p={float(p):.4g}")
            except Exception:
                parts.append(f"p={p}")
        if sigma is not None:
            try:
                parts.append(f"σ={float(sigma):.4g}")
            except Exception:
                parts.append(f"σ={sigma}")
        if model is not None:
            parts.append(str(model))
        suffix = ("  [" + ", ".join(parts) + "]") if parts else ""
        title = r"Cumulative NRMSE$_{≤ K}$ with band edges" + suffix

    # ---- plot (wider, reserve right margin for info panel) ----
    fig, ax = plt.subplots(figsize=(8.8, 4.2))

    # main curve (cumulative)
    ax.plot(k_sorted, y_sorted, linewidth=2.2, label=r"Cumulative NRMSE$_{≤ K}$")

    # diagnostic overlay (local)
    if y_local_sorted is not None:
        ax.plot(
            k_sorted,
            y_local_sorted,
            linewidth=1.3,
            linestyle="--",
            alpha=0.65,
            label=local_curve_label,
        )

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    # ---- top axis: ℓ = 1/k ----
    def k_to_ell(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / np.maximum(x, eps)

    def ell_to_k(x):
        x = np.asarray(x, dtype=float)
        return 1.0 / np.maximum(x, eps)

    secax = ax.secondary_xaxis("top", functions=(k_to_ell, ell_to_k))
    secax.set_xlabel(r"spatial scale $\ell$ (length unit)   [$\ell$ = 1/k]")

    # -----------------------------
    # band edges + big labels on plot
    # -----------------------------
    band_info_lines: list[str] = []
    if show_edges and k_edges is not None:
        edges_k = sorted([
            float(v) for v in k_edges
            if v is not None and np.isfinite(v) and float(v) > 0
        ])

        if len(edges_k) > 0:
            for ke in edges_k:
                ax.axvline(ke, linestyle="--", linewidth=1.0, alpha=0.9)

            x_min, x_max = ax.get_xlim()
            segs = [max(x_min, eps)] + [ke for ke in edges_k if x_min < ke < x_max] + [x_max]
            nb = max(1, len(segs) - 1)

            # Decide "big labels" shown inside plot.
            # If band_names looks like generic ("L","M","H"), we override with descriptive labels.
            default_big = ["Background", "Big", "Medium", "Small", "Tiny"]
            use_big_names = list(default_big[:nb])
            # If user provided meaningful names (not just L/M/H), respect them as the in-plot label.
            try:
                bn_norm = [str(x).strip() for x in band_names]
                generic = set(["L", "M", "H", "LMH", "low", "mid", "high"])
                if len(bn_norm) >= nb and not all((b in generic or len(b) <= 2) for b in bn_norm[:nb]):
                    use_big_names = bn_norm[:nb]
            except Exception:
                pass

            # helper: make detailed text for info panel
            def _detail_for_segment(i: int, a: float, b: float) -> str:
                name = use_big_names[i] if 0 <= i < len(use_big_names) else f"B{i}"
                if i == 0:
                    lam_text = f"λ ≥ {1.0/max(b,eps):.3g}"
                elif i == nb - 1:
                    lam_text = f"λ < {1.0/max(a,eps):.3g}"
                else:
                    lam_text = f"{1.0/max(b,eps):.3g} < λ ≤ {1.0/max(a,eps):.3g}"

                if D is not None and float(D) > 0:
                    Df = float(D)
                    if i == 0:
                        lamD_text = f"λ/D ≥ {(1.0/max(b,eps))/Df:.3g}"
                    elif i == nb - 1:
                        lamD_text = f"λ/D < {(1.0/max(a,eps))/Df:.3g}"
                    else:
                        lamD_text = f"{(1.0/max(b,eps))/Df:.3g} < λ/D ≤ {(1.0/max(a,eps))/Df:.3g}"
                    return f"{name}: ({lam_text}), ({lamD_text})"
                return f"{name}: ({lam_text})"

            # draw alternating spans and put ONLY big labels in-plot
            # use axes y-coord so it won't depend on log-y limits
            y_text = 0.87  # in axes fraction
            for i in range(nb):
                a, b = segs[i], segs[i + 1]
                if not (np.isfinite(a) and np.isfinite(b) and b > a > 0):
                    continue
                if i % 2 == 0:
                    ax.axvspan(a, b, alpha=0.06)

                xm = np.sqrt(a * b)

                # big label only
                ax.text(
                    xm,
                    y_text,
                    use_big_names[i] if i < len(use_big_names) else f"B{i}",
                    transform=ax.get_xaxis_transform(),  # x in data, y in axes
                    ha="center",
                    va="center",
                    fontsize=12,
                    alpha=0.95,
                )

                # collect detailed line for external panel
                band_info_lines.append(_detail_for_segment(i, a, b))

    # ---- legacy threshold (optional, diagnostic only) ----
    thr_legacy = (
        curve.get("k_star_threshold_legacy", None)
        or curve.get("k_star_threshold", None)
        or curve.get("nrmse_threshold", None)
        or meta.get("k_star_threshold", None)
        or meta.get("nrmse_threshold", None)
    )
    if thr_legacy is not None and np.isfinite(float(thr_legacy)):
        thr_legacy = float(thr_legacy)
        ax.axhline(thr_legacy, linestyle=":", linewidth=1.1, alpha=0.7)
        ax.text(
            ax.get_xlim()[0],
            thr_legacy,
            f"  legacy thr={thr_legacy:g}",
            va="bottom",
            ha="left",
            fontsize=9,
            alpha=0.8,
        )

    # ---- show k* marker (MAIN, on k-axis) ----
    if show_kstar:
        k_star = (
            curve.get("k_star_cum", None)
            or entry.get("k_star", None)
            or curve.get("k_star", None)
            or meta.get("k_star", None)
        )
        if k_star is not None:
            try:
                k_star = float(k_star)
            except Exception:
                k_star = None

        if k_star is not None and np.isfinite(k_star) and k_star > 0:
            ell_star = 1.0 / max(k_star, eps)
            kstar_label = rf"$k^*$={k_star:.3g} ($\ell^*$={ell_star:.3g})"
            ax.axvline(
                k_star,
                linestyle="-.",
                linewidth=1.7,
                color="tab:red",
                label=kstar_label,
                zorder=3,
            )

    # ---- Nyquist marker ----
    try:
        dmin = float(min(dx, dy))
        if np.isfinite(dmin) and dmin > 0:
            k_nyq = 1.0 / (2.0 * dmin)
            nyq_label = rf"Nyquist $k_N$={k_nyq:.3g}"
            ax.axvline(
                k_nyq,
                linestyle=":",
                linewidth=1.4,
                color="0.35",
                label=nyq_label,
                zorder=2,
            )
    except Exception:
        pass

    # ---- legend ----
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="lower right", frameon=True, fontsize=9)

    # --------------------------------
    # External info panel for band details (right side, outside axes)
    # --------------------------------
    if band_info_lines:
        # Make room on right
        fig.subplots_adjust(right=0.78)

        # Add a small text box outside the axes
        info_text = "Band definitions:\n" + "\n".join(f"- {ln}" for ln in band_info_lines)
        fig.text(
            0.80, 0.50,
            info_text,
            ha="left", va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor="0.6"),
        )
    else:
        fig.tight_layout()

    return fig


def plot_spatial_fourier_band_decomposition(
    x_true_hw: np.ndarray,
    x_pred_hw: np.ndarray,
    *,
    k_edges: Sequence[float],
    band_names: Sequence[str] = ("L", "M", "H"),
    dx: float = 1.0,
    dy: float = 1.0,
    channel: int = 0,
    title: str = "Spatial decomposition by Fourier bands",
    center_mode: str = "target_mean",   # "none" | "target_mean"
    robust_q: float = 99.5,             # for vlim
    max_cols: int = 5,                  # wrap columns when bands are many
) -> Optional[plt.Figure]:
    """
    空间域“多尺度解释图”：
    行：True / Pred / Error
    列：Full + 各 band（按 k_edges 划分）
    - 支持零中心化：减去 target 的空间均值（仅用于可视化）
    - 支持 band 多时自动换行（wrap），避免图过长/过宽
    - 上方为 value colorbar（标题下方），下方为 error colorbar，互不重叠
    - 通过 set_box_aspect(H/W) 让每个子图轴本身呈现条带比例，从根源压缩留白
    """
    xT = _ensure_hw(x_true_hw, channel=channel)
    xP = _ensure_hw(x_pred_hw, channel=channel)
    if xT.shape != xP.shape:
        raise ValueError(f"x_true shape {xT.shape} != x_pred shape {xP.shape}")

    H, W = xT.shape

    edges = [float(v) for v in k_edges if v is not None]
    if len(edges) == 0:
        return None
    nb = len(edges) + 1  # band count

    # ---- optional centering (visualization only) ----
    if center_mode == "target_mean":
        mu = float(np.mean(xT))
        xT0 = xT - mu
        xP0 = xP - mu
    elif center_mode == "none":
        xT0 = xT
        xP0 = xP
    else:
        raise ValueError(f"Unknown center_mode={center_mode}")

    # ---- bands: [0,e1], [e1,e2], ... , [elast, inf] ----
    lows = [None] + edges
    highs = edges + [None]

    # names padding
    need = nb
    names = list(band_names[:need])
    if len(names) < need:
        names += [f"B{i}" for i in range(len(names), need)]

    # ---- compute comps (Full + bands) ----
    true_comps = [xT0]
    pred_comps = [xP0]
    for lo, hi in zip(lows[:nb], highs[:nb], strict=False):
        true_comps.append(_fft_bandpass_2d(xT0, k_low=lo, k_high=hi, dx=dx, dy=dy))
        pred_comps.append(_fft_bandpass_2d(xP0, k_low=lo, k_high=hi, dx=dx, dy=dy))
    err_comps = [p - t for p, t in zip(pred_comps, true_comps, strict=False)]

    # ---- unified, robust color limits ----
    # main：只用 Full 的 true 来定标（与你现在一致）
    vals = true_comps[0].ravel()
    max_abs_main = float(np.percentile(np.abs(vals), robust_q))
    if max_abs_main == 0.0:
        max_abs_main = 1.0
    vmin_main, vmax_main = -max_abs_main, max_abs_main

    # err：只用 Full 的 error 来定标
    err0 = err_comps[0].ravel()
    max_abs_err = float(np.percentile(np.abs(err0), robust_q))
    if max_abs_err == 0.0:
        max_abs_err = max_abs_main
    vmin_err, vmax_err = -max_abs_err, max_abs_err

    # ---- layout with wrapping ----
    cols_all = ["Full"] + list(names)        # Full + nb bands
    n_panels = 1 + nb                        # Full + nb (注意：这里和 true_comps/pred_comps 的索引一致)

    max_cols = max(1, int(max_cols))
    ncols = min(max_cols, n_panels)          # band 少时不留空列
    n_blocks = int(np.ceil(n_panels / ncols))

    # 分区：Grid 只占中间区域，上下分别给标题/上 colorbar 和下 colorbar
    GRID_TOP = 0.78
    GRID_BOTTOM = 0.18

    # figsize：宽随列数，高随 block 数；这里高度不要太大，否则看起来松
    fig_w = 2.7 * ncols
    fig_h = 0.80 * 3 * n_blocks + 1.10
    fig = plt.figure(figsize=(fig_w, fig_h))

    # 每一行很扁：height_ratios + hspace 控行距（留白主要靠 set_box_aspect 再压一刀）
    row_heights = []
    for _ in range(n_blocks):
        row_heights.extend([0.16, 0.16, 0.16])

    gs = fig.add_gridspec(
        nrows=3 * n_blocks,
        ncols=ncols,
        height_ratios=row_heights,
        hspace=0.01,
        wspace=0.10,
        left=0.06,
        right=0.99,
        top=GRID_TOP,
        bottom=GRID_BOTTOM,
    )

    first_main_im = None
    first_err_im = None

    # 让轴框也变成条带比例（关键！）：box_aspect = H/W（很小）
    box_aspect = float(H) / float(W) if W > 0 else 1.0

    def _style_ax(ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        # 关键：把“轴框高度”压成与数据一致的条带比例，空白会显著减少
        try:
            ax.set_box_aspect(box_aspect)
        except Exception:
            # 老版本 matplotlib 可能没有 set_box_aspect
            pass

    for b in range(n_blocks):
        start = b * ncols
        end = min((b + 1) * ncols, n_panels)

        for j, col_idx in enumerate(range(start, end)):
            col_title = cols_all[col_idx]

            for r in range(3):
                ax = fig.add_subplot(gs[3 * b + r, j])

                if r == 0:
                    im = ax.imshow(
                        true_comps[col_idx],
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=vmin_main,
                        vmax=vmax_main,
                        aspect="equal",  # 不拉伸像素
                    )
                    if first_main_im is None:
                        first_main_im = im
                    ax.set_title(col_title, fontsize=10)
                    if j == 0:
                        ax.set_ylabel("True")

                elif r == 1:
                    ax.imshow(
                        pred_comps[col_idx],
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=vmin_main,
                        vmax=vmax_main,
                        aspect="equal",
                    )
                    if j == 0:
                        ax.set_ylabel("Pred")

                else:
                    im = ax.imshow(
                        err_comps[col_idx],
                        origin="lower",
                        cmap="RdBu_r",
                        vmin=vmin_err,
                        vmax=vmax_err,
                        aspect="equal",
                    )
                    if first_err_im is None:
                        first_err_im = im
                    if j == 0:
                        ax.set_ylabel("Error")

                _style_ax(ax)

        # 块内不足 ncols 的空位关掉
        for j in range(end - start, ncols):
            for r in range(3):
                ax = fig.add_subplot(gs[3 * b + r, j])
                ax.axis("off")

    # ---- title ----
    fig.suptitle(title, fontsize=12, y=0.99)

    # ---- manual colorbars: 与 grid 分区，互不干扰 ----
    cbar_h = 0.02
    cbar_w = 0.58
    cbar_left = 0.5 - cbar_w / 2.0

    # 上方 value colorbar：放在 GRID_TOP 之上，label 在上，不压到网格
    if first_main_im is not None:
        cax1 = fig.add_axes([cbar_left, GRID_TOP + 0.075, cbar_w, cbar_h])
        cb1 = fig.colorbar(first_main_im, cax=cax1, orientation="horizontal")
        cb1.set_label(
            "value (centered by target mean)" if center_mode == "target_mean" else "value",
            labelpad=4,
        )
        cb1.ax.xaxis.set_label_position("top")
        cb1.ax.xaxis.set_ticks_position("bottom")

    # 下方 error colorbar：放在 GRID_BOTTOM 之下
    if first_err_im is not None:
        cax2 = fig.add_axes([cbar_left, GRID_BOTTOM - 0.08, cbar_w, cbar_h])
        cb2 = fig.colorbar(first_err_im, cax=cax2, orientation="horizontal")
        cb2.set_label("error (Pred - True)", labelpad=4)
        cb2.ax.xaxis.set_label_position("top")
        cb2.ax.xaxis.set_ticks_position("bottom")

    return fig


def plot_spatial_fourier_band_decomposition_examples(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    *,
    k_edges: Sequence[float],
    band_names: Sequence[str] = ("L", "M", "H"),
    dx: float = 1.0,
    dy: float = 1.0,
    channel: int = 0,
    frame_indices: Sequence[int] | None = None,
    title_prefix: str = "Fourier bands spatial view",
    max_cols: int = 5,
) -> Dict[int, Optional[plt.Figure]]:
    """
    多帧版本：对若干 frame 生成傅里叶 band 空间分解图。
    支持输入:
      - [H,W] / [H,W,C]
      - [T,H,W] / [T,H,W,C]
    返回: {frame_idx: fig}
    """
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    # 统一成 [T,H,W,C?] 的视角
    def _as_THW(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:          # [H,W]
            return x[None, ...]
        if x.ndim == 3:
            # 可能是 [H,W,C] 或 [T,H,W]
            if x.shape[-1] <= 4: # 经验：小通道数更像 [H,W,C]
                return x[..., channel][None, ...]
            return x             # [T,H,W]
        if x.ndim == 4:          # [T,H,W,C]
            return x[..., channel]
        raise ValueError(f"Unsupported shape {x.shape}")

    XT = _as_THW(x_true)   # [T,H,W]
    XP = _as_THW(x_pred)   # [T,H,W]
    if XT.shape != XP.shape:
        raise ValueError(f"x_true shape {XT.shape} != x_pred shape {XP.shape}")

    T = XT.shape[0]
    if frame_indices is None:
        # 默认 3 帧：0 / 中间 / 最后
        frame_indices = [0, T // 2, T - 1] if T >= 3 else list(range(T))

    figs: Dict[int, Optional[plt.Figure]] = {}
    for fi in frame_indices:
        fi = int(fi)
        if fi < 0 or fi >= T:
            continue
        title = f"{title_prefix} | frame={fi}"
        figs[fi] = plot_spatial_fourier_band_decomposition(
            XT[fi],
            XP[fi],
            k_edges=k_edges,
            band_names=band_names,
            dx=dx,
            dy=dy,
            channel=0,  # 已经抽出单通道了
            title=title,
            max_cols=max_cols,
            center_mode="target_mean",
            robust_q=99.5,
        )
    return figs


def plot_kstar_heatmap(
    df_lin,
    df_mlp=None,
    *,
    title: str = "k* heatmap",
    model: str = "linear",
    grid_meta: dict | None = None,
    show_numbers: bool = True,
    number_fmt_k: str = "{:.3g}",
    number_fmt_ell: str = "{:.3g}",
    # --- new knobs ---
    use_log10_ell: bool = False,
    hatch_no_scale: str = "///",
    hatch_facecolor: tuple[float, float, float, float] = (0.90, 0.90, 0.90, 1.0),
) -> Optional[plt.Figure]:
    """
    画 k* 的 (p, σ) 热力图（v1.17：主口径为累计低通误差曲线的自适应截止 k*）。

    v1.17 定义（与数学手册对齐）：
      - 先计算累计低通误差曲线 NRMSE_{<=K}
      - k* 为该累计曲线进入“平台期”的起点（plateau onset）
      - ℓ* = 1/k* 表示模型可恢复的最小可信尺度

    仍显式考虑 Nyquist：
      - 若 k* > k_N，则认为该格子“越 Nyquist（无物理意义）”，用 × 标记
      - NaN/<=0 则认为 “No resolvable scale”
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D

    df = df_lin if model == "linear" else df_mlp
    if df is None or len(df) == 0:
        return None
    if "k_star" not in df.columns:
        return None

    # ---------------------------
    # resolve grid_meta / dx,dy
    # ---------------------------
    def _try_parse_meta_obj(x):
        if x is None:
            return None
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            try:
                import json
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None

    def _resolve_grid_meta_from_df(df_):
        if isinstance(grid_meta, dict) and len(grid_meta) > 0:
            return dict(grid_meta)

        for col in ("fourier_meta", "fourier", "meta"):
            if col in df_.columns:
                for v in df_[col].tolist():
                    meta_obj = _try_parse_meta_obj(v)
                    if isinstance(meta_obj, dict) and meta_obj:
                        gm = (
                            meta_obj.get("grid_meta", None)
                            or meta_obj.get("grid", None)
                            or meta_obj.get("gridMeta", None)
                            or meta_obj.get("gridmeta", None)
                        )
                        gm_obj = _try_parse_meta_obj(gm) or (gm if isinstance(gm, dict) else None)
                        if isinstance(gm_obj, dict) and gm_obj:
                            return dict(gm_obj)
                        if any(k in meta_obj for k in ("dx", "dy")):
                            return dict(meta_obj)

        g = {}
        if "dx" in df_.columns:
            try:
                g["dx"] = float(df_["dx"].dropna().iloc[0])
            except Exception:
                pass
        if "dy" in df_.columns:
            try:
                g["dy"] = float(df_["dy"].dropna().iloc[0])
            except Exception:
                pass
        return g if g else {}

    g = _resolve_grid_meta_from_df(df)
    dx = float(g.get("dx", 1.0))
    dy = float(g.get("dy", 1.0))

    eps = 1e-12
    k_nyq = None
    ell_nyq = None
    try:
        dmin = float(min(dx, dy))
        if np.isfinite(dmin) and dmin > 0:
            k_nyq = 1.0 / (2.0 * dmin)
            ell_nyq = 1.0 / max(k_nyq, eps)
    except Exception:
        k_nyq = None
        ell_nyq = None

    # ---------------------------
    # pivot table
    # ---------------------------
    try:
        pv_k = df.pivot(index="noise_sigma", columns="mask_rate", values="k_star")
    except Exception:
        return None
    pv_k = pv_k.sort_index(axis=0).sort_index(axis=1)
    K = pv_k.to_numpy(dtype=float)

    # ℓ* (physical meaning)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = 1.0 / np.maximum(K, eps)
        L[~np.isfinite(K) | (K <= 0)] = np.nan

    # classify cells
    no_scale = ~np.isfinite(L)  # includes NaN / invalid k_star
    invalid_nyq = np.zeros_like(K, dtype=bool)
    if k_nyq is not None and np.isfinite(k_nyq):
        invalid_nyq = np.isfinite(K) & (K > float(k_nyq))

    # values used for colormap (clamp invalid to boundary so cmap doesn't blow up)
    L_show = L.copy()
    if ell_nyq is not None and np.isfinite(ell_nyq):
        L_show[invalid_nyq] = float(ell_nyq)

    # for colormap: optionally log10(ℓ*)
    Z = L_show.copy()
    if use_log10_ell:
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.log10(np.maximum(L_show, eps))
            Z[~np.isfinite(L_show)] = np.nan  # keep NaNs

    finite_Z = Z[np.isfinite(Z)]
    if finite_Z.size == 0:
        return None
    vmin = float(np.nanmin(finite_Z))
    vmax = float(np.nanmax(finite_Z))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (vmin - 1.0), (vmin + 1.0)

    # ---------------------------
    # layout: main heatmap + right info panel (no overlay)
    # ---------------------------
    fig = plt.figure(figsize=(8.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.56])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=hatch_facecolor)  # NaN cells base color

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_title(f"{title} ({model})")
    ax.set_xticks(np.arange(pv_k.shape[1]))
    ax.set_xticklabels([f"{c:.3g}" for c in pv_k.columns], rotation=45, ha="right")
    ax.set_xlabel("mask_rate p")
    ax.set_yticks(np.arange(pv_k.shape[0]))
    ax.set_yticklabels([f"{r:.3g}" for r in pv_k.index])
    ax.set_ylabel("noise_sigma σ")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if use_log10_ell:
        cb.set_label(r"$\log_{10}(\ell^*)$  where $\ell^*=1/k^*$  (length unit)")
    else:
        cb.set_label(r"resolvable scale $\ell^* = 1/k^*$  (length unit)")

    # ---------------------------
    # overlay hatch for "no resolvable scale" cells
    # ---------------------------
    H_, W_ = Z.shape
    for i in range(H_):
        for j in range(W_):
            if no_scale[i, j]:
                rect = Rectangle(
                    (j - 0.5, i - 0.5),
                    1.0,
                    1.0,
                    facecolor=hatch_facecolor,
                    edgecolor="0.65",
                    hatch=hatch_no_scale,
                    linewidth=0.0,
                    zorder=2,
                )
                ax.add_patch(rect)

    # ---------------------------
    # numbers in each cell
    # ---------------------------
    if show_numbers:
        norm = im.norm
        for i in range(H_):
            for j in range(W_):
                k_val = K[i, j]
                ell = L[i, j]  # true ell* (not clamped)

                if no_scale[i, j]:
                    txt = "No scale"
                else:
                    if (k_nyq is not None) and np.isfinite(k_nyq) and (k_val > k_nyq):
                        if ell_nyq is not None and np.isfinite(ell_nyq):
                            txt = (
                                rf"$\ell^*<{number_fmt_ell.format(ell_nyq)}$" "\n"
                                rf"$k^*>{number_fmt_k.format(k_nyq)}$"
                            )
                        else:
                            txt = rf"$k^*>{number_fmt_k.format(k_nyq)}$"
                    else:
                        txt = (
                            rf"$\ell^*={number_fmt_ell.format(ell)}$" "\n"
                            rf"$k^*={number_fmt_k.format(k_val)}$"
                        )

                val_for_color = Z[i, j]
                if np.isfinite(val_for_color):
                    rgba = cmap(norm(val_for_color))
                    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    tcolor = "black" if lum > 0.55 else "white"
                else:
                    tcolor = "black"

                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=tcolor,
                    zorder=4,
                )

    if invalid_nyq.any():
        ys, xs = np.where(invalid_nyq)
        ax.scatter(xs, ys, marker="x", s=60, linewidths=2.0, color="white", zorder=5)

    # ---------------------------
    # info panel: definition + status legend
    # ---------------------------
    def_lines = [
        r"$k^*$: plateau onset of cumulative low-pass error",
        r"$\mathrm{NRMSE}_{≤ K}=\sqrt{\sum_{k≤ K}E_e(k) / \sum_{k≤ K}E_t(k)}$",
        r"$\ell^*=1/k^*$: smallest trustworthy spatial scale",
    ]
    if k_nyq is not None and np.isfinite(k_nyq):
        def_lines.append(rf"Nyquist: $k_N=1/(2\,\min(dx,dy))={k_nyq:.3g}$")
    if ell_nyq is not None and np.isfinite(ell_nyq):
        def_lines.append(rf"$\ell_N=1/k_N={ell_nyq:.3g}$")

    ax_info.text(
        0.02,
        0.98,
        "\n".join(def_lines),
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95, edgecolor="0.6"),
        transform=ax_info.transAxes,
    )

    handles = []
    labels = []
    handles.append(Patch(facecolor=hatch_facecolor, edgecolor="0.65", hatch=hatch_no_scale))
    labels.append("No resolvable scale (k* undefined)")
    handles.append(Line2D([0], [0], marker="x", color="none", markeredgecolor="black",
                          markersize=8, markeredgewidth=2))
    labels.append(r"Invalid: $k^*>k_N$")

    ax_info.legend(
        handles,
        labels,
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        fontsize=9,
        borderpad=0.6,
        handlelength=1.6,
        handletextpad=0.8,
    )

    return fig


def plot_fourier_band_nrmse_curves(
    df_lin,
    df_mlp=None,
    *,
    band_names: Sequence[str] = ("L", "M", "H"),
) -> Dict[str, Optional[plt.Figure]]:
    """
    画频域 band NRMSE 的两类曲线：
    - 固定 σ：band_nrmse vs p
    - 固定 p：band_nrmse vs σ
    需要列名: fourier_band_nrmse_<band>
    """
    figs: Dict[str, Optional[plt.Figure]] = {
        "fig_fourier_band_vs_mask_linear": None,
        "fig_fourier_band_vs_mask_mlp": None,
        "fig_fourier_band_vs_noise_linear": None,
        "fig_fourier_band_vs_noise_mlp": None,
    }

    def _has_cols(df) -> bool:
        if df is None or len(df) == 0:
            return False
        for b in band_names:
            if f"fourier_band_nrmse_{b}" not in df.columns:
                return False
        return True

    def _plot_vs_mask(df, model: str) -> Optional[plt.Figure]:
        if not _has_cols(df):
            return None

        # 每个 sigma 一条曲线；每条曲线内部对 band 画多条
        sigmas = sorted(set(df["noise_sigma"].tolist()))
        ps = sorted(set(df["mask_rate"].tolist()))
        if not sigmas or not ps:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        for s in sigmas:
            sub = df[df["noise_sigma"] == s].sort_values("mask_rate")
            if len(sub) == 0:
                continue
            for b in band_names:
                y = sub[f"fourier_band_nrmse_{b}"].to_numpy()
                x = sub["mask_rate"].to_numpy()
                ax.plot(x, y, marker="o", linewidth=1.5, label=f"σ={s:.3g}  band={b}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mask_rate p (log)")
        ax.set_ylabel("Fourier band NRMSE (log)")
        ax.set_title(f"Fourier band NRMSE vs mask_rate ({model})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

        fig.subplots_adjust(right=0.78)
        return fig

    def _plot_vs_noise(df, model: str) -> Optional[plt.Figure]:
        if not _has_cols(df):
            return None

        ps = sorted(set(df["mask_rate"].tolist()))
        sigmas = sorted(set(df["noise_sigma"].tolist()))
        if not ps or not sigmas:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.2))
        for p in ps:
            sub = df[df["mask_rate"] == p].sort_values("noise_sigma")
            if len(sub) == 0:
                continue
            for b in band_names:
                y = sub[f"fourier_band_nrmse_{b}"].to_numpy()
                x = sub["noise_sigma"].to_numpy()
                ax.plot(x, y, marker="o", linewidth=1.5, label=f"p={p:.3g}  band={b}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("noise_sigma σ (log)")
        ax.set_ylabel("Fourier band NRMSE (log)")
        ax.set_title(f"Fourier band NRMSE vs noise_sigma ({model})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )

        fig.subplots_adjust(right=0.78)
        return fig

    figs["fig_fourier_band_vs_mask_linear"] = _plot_vs_mask(df_lin, "linear")
    if df_mlp is not None:
        figs["fig_fourier_band_vs_mask_mlp"] = _plot_vs_mask(df_mlp, "mlp")

    figs["fig_fourier_band_vs_noise_linear"] = _plot_vs_noise(df_lin, "linear")
    if df_mlp is not None:
        figs["fig_fourier_band_vs_noise_mlp"] = _plot_vs_noise(df_mlp, "mlp")

    return figs


def plot_fourier_curve_from_entry(
    entry: Dict[str, Any],
    *,
    title: str = "Fourier scale curve",
) -> Optional[plt.Figure]:
    """
    兼容旧接口：仍然叫 plot_fourier_curve_from_entry。

    v1.17 起，它返回的就是“k* 解释型曲线图”：
      - 主曲线：累计低通 NRMSE_{<=K} vs K
      - 可选叠加：局部 NRMSE(k) vs k（诊断）
    """
    return plot_kstar_curve_from_entry(
        entry,
        title=title,
        show_edges=True,
        show_kstar=True,
        show_local_curve=True,
    )


def plot_fourier_example_from_npz_data(
    data: "np.lib.npyio.NpzFile | dict[str, object]",
    *,
    title_prefix: str | None = None,
    # allow override / fallback
    k_edges: Sequence[float] | None = None,
    band_names: Sequence[str] | None = None,
    dx: float | None = None,
    dy: float | None = None,
    # passthrough controls
    center_mode: str = "target_mean",
    robust_q: float = 99.5,
    channel: int = 0,
) -> Optional[plt.Figure]:
    """
    从已加载的 example npz 数据对象中恢复一张“傅里叶多尺度空间联图”（不做任何 IO）。

    预期包含（至少）：
        - x_true, x_hat
    建议包含（用于保证口径一致）：
        - fourier_k_edges
        - fourier_band_names
        - fourier_dx, fourier_dy

    若上述 fourier_* 字段不存在，会按优先级回退到函数参数（k_edges/band_names/dx/dy）。
    """
    files = getattr(data, "files", None)
    has = (lambda k: (k in files) if files is not None else (k in data))
    get = (lambda k: data[k])

    x_true = np.asarray(get("x_true"))
    x_hat = np.asarray(get("x_hat"))

    # --- resolve meta with fallback ---
    if k_edges is None and has("fourier_k_edges"):
        k_edges = np.asarray(get("fourier_k_edges")).astype(float).tolist()
    if band_names is None and has("fourier_band_names"):
        raw = get("fourier_band_names")
        band_names = [str(x) for x in np.asarray(raw).tolist()]
    if dx is None and has("fourier_dx"):
        dx = float(get("fourier_dx"))
    if dy is None and has("fourier_dy"):
        dy = float(get("fourier_dy"))

    if k_edges is None:
        raise ValueError(
            "plot_fourier_example_from_npz_data: missing k_edges "
            "(fourier_k_edges not in npz and k_edges not provided)"
        )
    if band_names is None:
        band_names = ("L", "M", "H")
    if dx is None:
        dx = 1.0
    if dy is None:
        dy = 1.0

    # title
    if title_prefix is None:
        model_name = str(get("model_type")) if has("model_type") else "model"

        parts = [f"Fourier bands spatial view ({model_name})"]
        if has("mask_rate"):
            parts.append(f"p={float(get('mask_rate')):.3g}")
        if has("noise_sigma"):
            parts.append(f"σ={float(get('noise_sigma')):.3g}")
        if has("frame_idx"):
            parts.append(f"frame={int(get('frame_idx'))}")
        title_prefix = " | ".join(parts)

    return plot_spatial_fourier_band_decomposition(
        x_true_hw=x_true,
        x_pred_hw=x_hat,
        k_edges=k_edges,
        band_names=band_names,
        dx=dx,
        dy=dy,
        title=title_prefix,
        channel=channel,
        center_mode=center_mode,
        robust_q=robust_q,
    )


def plot_fourier_example_from_npz(
    npz_path: str | Path,
    *,
    title_prefix: str | None = None,
    k_edges: Sequence[float] | None = None,
    band_names: Sequence[str] | None = None,
    dx: float | None = None,
    dy: float | None = None,
    center_mode: str = "target_mean",
    robust_q: float = 99.5,
    channel: int = 0,
) -> Optional[plt.Figure]:
    """
    从保存的 example npz 文件中恢复一张“傅里叶多尺度空间联图”。

    这是一个薄封装：负责 IO，实际绘制逻辑在 plot_fourier_example_from_npz_data() 内。
    """
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        return plot_fourier_example_from_npz_data(
            data,
            title_prefix=title_prefix,
            k_edges=k_edges,
            band_names=band_names,
            dx=dx,
            dy=dy,
            center_mode=center_mode,
            robust_q=robust_q,
            channel=channel,
        )