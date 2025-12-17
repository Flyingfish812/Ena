# backend/viz/fourier_plots.py

from __future__ import annotations

from typing import Dict, Any, Iterable, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_kstar_heatmap(
    df_lin,
    df_mlp=None,
    *,
    title: str = "k* heatmap",
    model: str = "linear",
) -> Optional[plt.Figure]:
    """
    画 k* 的 (p, σ) 热力图。
    - df 必须含列: mask_rate, noise_sigma, k_star
    """
    df = df_lin if model == "linear" else df_mlp
    if df is None or len(df) == 0:
        return None
    if "k_star" not in df.columns:
        return None

    # pivot -> rows: noise_sigma, cols: mask_rate
    try:
        pv = df.pivot(index="noise_sigma", columns="mask_rate", values="k_star")
    except Exception:
        return None

    # 保证排序
    pv = pv.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.0))
    im = ax.imshow(pv.values, origin="lower", aspect="auto")
    ax.set_title(f"{title} ({model})")

    ax.set_xticks(np.arange(pv.shape[1]))
    ax.set_xticklabels([f"{c:.3g}" for c in pv.columns], rotation=45, ha="right")
    ax.set_xlabel("mask_rate p")

    ax.set_yticks(np.arange(pv.shape[0]))
    ax.set_yticklabels([f"{r:.3g}" for r in pv.index])
    ax.set_ylabel("noise_sigma σ")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("k*")

    fig.tight_layout()
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
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
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
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
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
    title: str = "NRMSE(k)",
) -> Optional[plt.Figure]:
    """
    可选：如果 entry["fourier_curve"] 存在，就画 NRMSE(k)。
    需要 fourier_curve 结构：
      {k_centers, nrmse_k, k_edges(optional), band_names(optional)}
    """
    curve = entry.get("fourier_curve", None)
    if curve is None:
        return None

    k = np.asarray(curve.get("k_centers", []), dtype=float)
    y = np.asarray(curve.get("nrmse_k", []), dtype=float)
    if k.size == 0 or y.size == 0:
        return None

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.6))
    ax.plot(k, y, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("k (rad/len or cycles/len)")
    ax.set_ylabel("NRMSE(k) (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    # optionally draw band edges
    k_edges = curve.get("k_edges", None)
    if k_edges is not None:
        for ke in k_edges:
            try:
                ax.axvline(float(ke), linestyle="--", linewidth=1.0)
            except Exception:
                pass

    fig.tight_layout()
    return fig
