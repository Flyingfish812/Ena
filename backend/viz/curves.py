# backend/viz/curves.py

"""
NMSE/NMAE/PSNR 随 mask_rate / noise_sigma 变化的曲线。
"""

from typing import Dict, Any

import matplotlib.pyplot as plt


def _filter_entries(
    results: Dict[str, Any],
    *,
    mask_rate: float | None = None,
    noise_sigma: float | None = None,
):
    """
    从 results["entries"] 里按条件筛选记录。
    """
    entries = results.get("entries", [])
    out = []
    for e in entries:
        if mask_rate is not None and abs(e["mask_rate"] - mask_rate) > 1e-12:
            continue
        if noise_sigma is not None and abs(e["noise_sigma"] - noise_sigma) > 1e-12:
            continue
        out.append(e)
    return out


def plot_nmse_vs_mask_rate(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 mask_rate 下的 NMSE 曲线。

    约定：
    - 若 results 中包含多种 noise_sigma，则默认选取其中最小的 noise_sigma；
    - x 轴: mask_rate，y 轴: nmse_mean。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    mask_rates = sorted(set(float(m) for m in results.get("mask_rates", [])))
    noise_sigmas = sorted(set(float(s) for s in results.get("noise_sigmas", [])))

    if not mask_rates or not noise_sigmas:
        raise ValueError("results 中缺少 mask_rates 或 noise_sigmas 信息")

    sigma_ref = noise_sigmas[0]  # 默认用噪声最小的那一组
    entries = _filter_entries(results, noise_sigma=sigma_ref)

    x = []
    y = []
    for mr in mask_rates:
        # 找到该 mask_rate 下的条目
        e_list = [e for e in entries if abs(e["mask_rate"] - mr) < 1e-12]
        if not e_list:
            continue
        e = e_list[0]
        x.append(mr)
        y.append(e["nmse_mean"])

    model_label = label or results.get("model_type", "model")
    ax.plot(x, y, marker="o", label=f"{model_label} (σ={sigma_ref:.3g})")
    ax.set_xlabel("mask_rate")
    ax.set_ylabel("NMSE (mean)")
    ax.set_title("NMSE vs mask_rate")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return ax


def plot_nmse_vs_noise(
    results: Dict[str, Any],
    ax: plt.Axes | None = None,
    label: str | None = None,
) -> plt.Axes:
    """
    绘制在不同 noise_sigma 下的 NMSE 曲线。

    约定：
    - 若 results 中包含多种 mask_rate，则默认选取其中最小的 mask_rate；
    - x 轴: noise_sigma，y 轴: nmse_mean。
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    mask_rates = sorted(set(float(m) for m in results.get("mask_rates", [])))
    noise_sigmas = sorted(set(float(s) for s in results.get("noise_sigmas", [])))

    if not mask_rates or not noise_sigmas:
        raise ValueError("results 中缺少 mask_rates 或 noise_sigmas 信息")

    mr_ref = mask_rates[0]  # 默认用采样率最小的那一组
    entries = _filter_entries(results, mask_rate=mr_ref)

    x = []
    y = []
    for sigma in noise_sigmas:
        e_list = [e for e in entries if abs(e["noise_sigma"] - sigma) < 1e-12]
        if not e_list:
            continue
        e = e_list[0]
        x.append(sigma)
        y.append(e["nmse_mean"])

    model_label = label or results.get("model_type", "model")
    ax.plot(x, y, marker="o", label=f"{model_label} (p={mr_ref:.3g})")
    ax.set_xlabel("noise_sigma")
    ax.set_ylabel("NMSE (mean)")
    ax.set_title("NMSE vs noise_sigma")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return ax
