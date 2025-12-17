# backend/viz/fourier_plots.py

from __future__ import annotations

from typing import Dict, Any, Iterable, Sequence, Optional
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
    title_prefix: str = "NRMSE(k)",
    show_edges: bool = True,
    show_kstar: bool = True,
) -> Optional[plt.Figure]:
    """
    【解释图-A2/B1】单个配置的 NRMSE(k) + k* 标注图。
    需要 entry["fourier_curve"] 至少包含:
      - k_centers, nrmse_k
    可选:
      - k_edges: 用虚线画 band 边界
      - k_star: 用竖线画 k*
      - k_star_threshold 或 nrmse_threshold: 用水平线画阈值
    """
    curve = entry.get("fourier_curve", None)
    if curve is None:
        return None

    k = np.asarray(curve.get("k_centers", []), dtype=float)
    y = np.asarray(curve.get("nrmse_k", []), dtype=float)
    if k.size == 0 or y.size == 0 or k.size != y.size:
        return None

    p = entry.get("mask_rate", None)
    s = entry.get("noise_sigma", None)
    suffix = []
    if p is not None:
        suffix.append(f"p={float(p):.3g}")
    if s is not None:
        suffix.append(f"σ={float(s):.3g}")
    suffix = (", " + ", ".join(suffix)) if suffix else ""

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.8))
    ax.plot(k, y, linewidth=2.0)

    ax.set_yscale("log")
    ax.set_xlabel("k (cycles / length)")
    ax.set_ylabel("NRMSE(k) (log)")
    ax.set_title(f"{title_prefix}{suffix}")
    ax.grid(True, which="both", alpha=0.3)

    if show_edges:
        k_edges = curve.get("k_edges", None)
        if k_edges is not None:
            for ke in k_edges:
                try:
                    ax.axvline(float(ke), linestyle="--", linewidth=1.0)
                except Exception:
                    pass

    # k* + threshold
    if show_kstar:
        k_star = curve.get("k_star", entry.get("k_star", None))
        if k_star is not None:
            try:
                ax.axvline(float(k_star), linestyle="-.", linewidth=1.5)
                ax.text(float(k_star), ax.get_ylim()[0], " k*", va="bottom", ha="left")
            except Exception:
                pass

        thr = curve.get("k_star_threshold", curve.get("nrmse_threshold", None))
        if thr is not None:
            try:
                ax.axhline(float(thr), linestyle=":", linewidth=1.2)
                ax.text(ax.get_xlim()[0], float(thr), " threshold ", va="bottom", ha="left")
            except Exception:
                pass

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
) -> Optional[plt.Figure]:
    """
    【解释图-A3/B2】空间域“多尺度解释图”：
    行：True / Pred / Error
    列：Full / L / M / H（按 k_edges 划分）

    注意：
    - 这里的 band-pass 是对单帧单通道做径向频带滤波（radial band-pass）。
    - k_edges 的单位必须与 fftfreq 构造的 k 一致（cycles/length）。
    """
    xT = _ensure_hw(x_true_hw, channel=channel)
    xP = _ensure_hw(x_pred_hw, channel=channel)
    if xT.shape != xP.shape:
        raise ValueError(f"x_true shape {xT.shape} != x_pred shape {xP.shape}")

    edges = [float(v) for v in k_edges]
    if len(edges) == 0:
        return None

    # 构造 bands: [0,e1], [e1,e2], ..., [elast, inf]
    segs = [None] + edges + [None]
    # 但我们更明确：low/high 数值
    lows = [None] + edges
    highs = edges + [None]

    # Full + bands
    cols = ["Full"] + list(band_names[: (len(lows))])  # 最多匹配到 edges+1 个 band
    # 实际 band 数 = len(edges)+1
    nb = len(edges) + 1
    cols = ["Full"] + list(band_names[:nb])
    if len(cols) < nb + 1:
        # 如果 band_names 不够，补默认名
        for i in range(len(cols) - 1, nb):
            cols.append(f"B{i}")

    # 预计算分量
    true_comps = [xT]
    pred_comps = [xP]
    for lo, hi in zip(lows[:nb], highs[:nb], strict=False):
        true_comps.append(_fft_bandpass_2d(xT, k_low=lo, k_high=hi, dx=dx, dy=dy))
        pred_comps.append(_fft_bandpass_2d(xP, k_low=lo, k_high=hi, dx=dx, dy=dy))
    err_comps = [p - t for p, t in zip(pred_comps, true_comps, strict=False)]

    # 颜色范围：True/Pred 用同一范围；Error 用以 0 为中心的对称范围
    vmin = float(np.min(xT))
    vmax = float(np.max(xT))
    emax = float(np.max(np.abs(err_comps[0])))

    nrows = 3
    ncols = 1 + nb  # Full + nb bands
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))

    row_titles = ["True", "Pred", "Error"]
    for j in range(ncols):
        axes[0, j].set_title(cols[j])

    # True row
    ims = []
    for j in range(ncols):
        ax = axes[0, j]
        im = ax.imshow(true_comps[j], origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ims.append(im)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("True")

    # Pred row
    for j in range(ncols):
        ax = axes[1, j]
        im = ax.imshow(pred_comps[j], origin="lower", vmin=vmin, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Pred")

    # Error row (0-center)
    for j in range(ncols):
        ax = axes[2, j]
        im = ax.imshow(err_comps[j], origin="lower", vmin=-emax, vmax=emax, cmap="RdBu_r")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Error")

    # 每行一个 colorbar（横向，覆盖该行全部子图）
    cb0 = fig.colorbar(axes[0, 0].images[0], ax=axes[0, :], orientation="horizontal", fraction=0.046, pad=0.06)
    cb0.set_label("value (True)")

    cb1 = fig.colorbar(axes[1, 0].images[0], ax=axes[1, :], orientation="horizontal", fraction=0.046, pad=0.06)
    cb1.set_label("value (Pred)")

    cb2 = fig.colorbar(axes[2, 0].images[0], ax=axes[2, :], orientation="horizontal", fraction=0.046, pad=0.06)
    cb2.set_label("error (Pred - True)")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


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
    兼容旧接口：仍然叫 plot_fourier_curve_from_entry，
    但升级为“解释型”曲线图：NRMSE(k) + (可选) k* + (可选) threshold + (可选) band edges。
    """
    return plot_kstar_curve_from_entry(
        entry,
        title_prefix=title,
        show_edges=True,
        show_kstar=True,
    )