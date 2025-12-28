# backend/viz/scale_plots.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# small helpers
# -------------------------

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


def _resolve_dxdyangular(grid_meta: dict | None) -> tuple[float, float, bool]:
    g = dict(grid_meta or {})
    dx = float(g.get("dx", 1.0))
    dy = float(g.get("dy", 1.0))
    angular = bool(g.get("angular", False))
    return dx, dy, angular


def _nyquist_k(dx: float, dy: float, angular: bool) -> tuple[Optional[float], Optional[float]]:
    """
    Return (k_N, ell_N) where ell_N = 1/k_N.
    - If angular=False: k_N = 1/(2*dmin)  (cycles/unit)
    - If angular=True:  k_N = pi/dmin     (rad/unit)
    """
    eps = 1e-12
    try:
        dmin = float(min(dx, dy))
        if not np.isfinite(dmin) or dmin <= 0:
            return None, None
        if angular:
            kN = float(np.pi / dmin)
        else:
            kN = float(1.0 / (2.0 * dmin))
        ellN = float(1.0 / max(kN, eps))
        return kN, ellN
    except Exception:
        return None, None


# -------------------------
# 1) unified profile curves
# -------------------------

def plot_profile_curves(
    profiles: Sequence[Dict[str, Any]],
    *,
    title: str = "Scale profile curves",
    xlabel: str = "wavenumber k",
    ylabel: str = "profile",
    xlog: bool = True,
    ylog: bool = False,
    ylim: tuple[float, float] | None = None,
    grid_meta: dict | None = None,
    show_nyquist: bool = True,
    show_secondary_lambda: bool = True,
    secondary_label: str = "ℓ = 1/k (length unit)",
    legend_loc: str = "best",
) -> Optional[plt.Figure]:
    """
    Unified curve plot for:
      - coherence(k)  (0..1)
      - SNR(k) or log10(SNR(k))
      - other scale-response y(k)

    `profiles` element schema:
      {
        "k": ndarray-like,
        "y": ndarray-like,
        "label": str,
        "k_eff": float|None,
        "k_eff_label": str|None,
        "style": dict|None   # passed to ax.plot
      }
    """
    if profiles is None or len(profiles) == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.2))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)

    # plot curves
    any_plotted = False
    for p in profiles:
        k = np.asarray(p.get("k", []), dtype=float).reshape(-1)
        y = np.asarray(p.get("y", []), dtype=float).reshape(-1)
        m = np.isfinite(k) & np.isfinite(y)
        if xlog:
            m &= (k > 0)
        if ylog:
            m &= (y > 0)

        k = k[m]
        y = y[m]
        if k.size == 0:
            continue

        label = str(p.get("label", "curve"))
        style = p.get("style", None)
        if not isinstance(style, dict):
            style = {}

        ax.plot(k, y, label=label, **style)
        any_plotted = True

        k_eff = p.get("k_eff", None)
        if k_eff is not None and np.isfinite(float(k_eff)) and float(k_eff) > 0:
            k_eff = float(k_eff)
            lab = p.get("k_eff_label", None)
            if lab is None:
                lab = rf"$k_{{eff}}$={k_eff:.3g}"
            ax.axvline(k_eff, linestyle="-.", linewidth=1.4, alpha=0.9, label=str(lab))

    if not any_plotted:
        plt.close(fig)
        return None

    # Nyquist marker
    if show_nyquist and grid_meta is not None:
        dx, dy, angular = _resolve_dxdyangular(grid_meta)
        kN, ellN = _nyquist_k(dx, dy, angular)
        if kN is not None and np.isfinite(kN):
            ax.axvline(kN, linestyle=":", linewidth=1.4, alpha=0.9, label=rf"Nyquist $k_N$={kN:.3g}")

    # secondary axis (ell=1/k)
    if show_secondary_lambda:
        eps = 1e-12

        def k_to_ell(x):
            x = np.asarray(x, dtype=float)
            return 1.0 / np.maximum(x, eps)

        def ell_to_k(x):
            x = np.asarray(x, dtype=float)
            return 1.0 / np.maximum(x, eps)

        secax = ax.secondary_xaxis("top", functions=(k_to_ell, ell_to_k))
        secax.set_xlabel(secondary_label)

    ax.legend(loc=legend_loc, fontsize=9)
    fig.tight_layout()
    return fig


# -------------------------
# 2) cutoff heatmap (k_eff / ell_eff)
# -------------------------

def plot_cutoff_heatmap(
    *,
    mask_rates: Sequence[float],
    noise_sigmas: Sequence[float],
    k_eff_matrix: Sequence[Sequence[float]],
    title: str = "k_eff heatmap",
    grid_meta: dict | None = None,
    show_numbers: bool = True,
    number_fmt_k: str = "{:.3g}",
    number_fmt_ell: str = "{:.3g}",
    use_log10_ell: bool = False,
    hatch_no_scale: str = "///",
    hatch_invalid_nyq: str = "\\\\\\",
    hatch_facecolor: tuple[float, float, float, float] = (0.90, 0.90, 0.90, 1.0),
    info_lines: Sequence[str] = (),
) -> Optional[plt.Figure]:
    """
    Heatmap layout aligned with plot_kstar_heatmap style:
      - main heatmap uses ell_eff = 1/k_eff (or log10 ell_eff)
      - hatch NaN / invalid
      - mark cells where k_eff > k_N as "beyond Nyquist" (hatch)
      - right side info panel: Nyquist, dx/dy/angular, any extra info_lines

    Axes:
      x: mask_rate (p)
      y: noise_sigma (σ)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D

    mrs = [float(x) for x in mask_rates]
    nss = [float(y) for y in noise_sigmas]
    K = np.asarray(k_eff_matrix, dtype=float)
    if K.ndim != 2 or K.shape != (len(nss), len(mrs)):
        raise ValueError(f"k_eff_matrix shape must be (len(noise_sigmas), len(mask_rates)) = {(len(nss), len(mrs))}, got {K.shape}")

    eps = 1e-12

    # resolve Nyquist
    dx, dy, angular = _resolve_dxdyangular(grid_meta)
    k_nyq, ell_nyq = _nyquist_k(dx, dy, angular)

    # ell_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        L = 1.0 / np.maximum(K, eps)
        L[~np.isfinite(K) | (K <= 0)] = np.nan

    no_scale = ~np.isfinite(L)
    invalid_nyq = np.zeros_like(K, dtype=bool)
    if k_nyq is not None and np.isfinite(k_nyq):
        invalid_nyq = np.isfinite(K) & (K > float(k_nyq))

    # values for colormap
    L_show = L.copy()
    if ell_nyq is not None and np.isfinite(ell_nyq):
        # clamp beyond-nyquist to ell_nyq to keep cmap stable
        L_show[invalid_nyq] = float(ell_nyq)

    Z = L_show.copy()
    if use_log10_ell:
        with np.errstate(divide="ignore", invalid="ignore"):
            Z = np.log10(np.maximum(L_show, eps))
            Z[~np.isfinite(L_show)] = np.nan

    finite_Z = Z[np.isfinite(Z)]
    if finite_Z.size == 0:
        return None
    vmin = float(np.nanmin(finite_Z))
    vmax = float(np.nanmax(finite_Z))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = (vmin - 1.0), (vmin + 1.0)

    # layout: heatmap + info panel
    fig = plt.figure(figsize=(9.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 0.58])

    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color=hatch_facecolor)

    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("mask_rate p")
    ax.set_ylabel("noise_sigma σ")

    # ticks as actual values (show all by default; caller can thin beforehand if too dense)
    ax.set_xticks(np.arange(len(mrs)))
    ax.set_xticklabels([f"{v:.3g}" for v in mrs], rotation=0)
    ax.set_yticks(np.arange(len(nss)))
    ax.set_yticklabels([f"{v:.3g}" for v in nss], rotation=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
    cbar.set_label("log10(ℓ_eff)" if use_log10_ell else "ℓ_eff = 1/k_eff")

    # numbers + hatch overlays
    for yi in range(len(nss)):
        for xi in range(len(mrs)):
            if no_scale[yi, xi]:
                # hatch no-scale
                rect = Rectangle(
                    (xi - 0.5, yi - 0.5),
                    1.0, 1.0,
                    fill=False,
                    hatch=hatch_no_scale,
                    edgecolor="k",
                    linewidth=0.0,
                )
                ax.add_patch(rect)
                continue

            if invalid_nyq[yi, xi]:
                rect = Rectangle(
                    (xi - 0.5, yi - 0.5),
                    1.0, 1.0,
                    fill=False,
                    hatch=hatch_invalid_nyq,
                    edgecolor="k",
                    linewidth=0.0,
                )
                ax.add_patch(rect)

            if show_numbers:
                # ---- k value ----
                k_val = float(K[yi, xi]) if np.isfinite(K[yi, xi]) else None

                # ---- ell value (1/k) from L matrix ----
                ell_cell = L[yi, xi]
                ell_val = float(ell_cell) if np.isfinite(ell_cell) else None

                # ---- decide text ----
                if k_val is None:
                    txt = "—\n(—)"
                else:
                    # beyond Nyquist -> show capped info
                    if k_nyq is not None and np.isfinite(k_nyq) and k_val > float(k_nyq):
                        # ell_nyq might be None or non-finite
                        if ell_nyq is not None and np.isfinite(float(ell_nyq)):
                            ell_txt = number_fmt_ell.format(float(ell_nyq))
                        else:
                            ell_txt = "—"
                        txt = f">{number_fmt_k.format(float(k_nyq))}\n({ell_txt})"
                    else:
                        # normal cell
                        if ell_val is None:
                            txt = f"{number_fmt_k.format(k_val)}\n(—)"
                        else:
                            txt = f"{number_fmt_k.format(k_val)}\n({number_fmt_ell.format(ell_val)})"

                ax.text(xi, yi, txt, ha="center", va="center", fontsize=8)

    # legend for hatches
    handles = [
        Patch(facecolor=hatch_facecolor, edgecolor="k", hatch=hatch_no_scale, label="No resolvable scale"),
    ]
    if k_nyq is not None and np.isfinite(k_nyq):
        handles.append(Patch(facecolor=hatch_facecolor, edgecolor="k", hatch=hatch_invalid_nyq, label="Beyond Nyquist"))

    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    # info panel
    info: List[str] = []
    info.append("Scale cutoff heatmap")
    info.append("")
    info.append(f"dx={dx:.3g}, dy={dy:.3g}, angular={angular}")
    if k_nyq is not None and np.isfinite(k_nyq):
        info.append(f"Nyquist k_N={k_nyq:.3g}")
        if ell_nyq is not None and np.isfinite(ell_nyq):
            info.append(f"Nyquist ℓ_N={ell_nyq:.3g}")
    else:
        info.append("Nyquist: unavailable (dx/dy invalid)")
    if info_lines:
        info.append("")
        info.extend([str(x) for x in info_lines])

    ax_info.text(0.0, 1.0, "\n".join(info), ha="left", va="top", fontsize=10)

    return fig
