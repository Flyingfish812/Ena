# backend/pod/compute.py

"""
构建 POD 基底：从原始数据计算 SVD / POD，保存截断基底与元数据。
"""

from typing import Dict, Any

import numpy as np

from ..config.schemas import DataConfig, PodConfig
from ..dataio.nc_loader import load_raw_nc
from ..dataio.io_utils import ensure_dir, save_numpy, save_json


def build_pod(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    *,
    verbose: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    完整执行一次 POD 构建流程，并可选打印中间信息和作图。

    步骤：
    1. 调用 dataio.load_raw_nc 读取 [T,H,W,C] 数组。
    2. 将数据 reshape 为 [N, D]，其中 N = T，D = H*W*C。
    3. 对每个空间点做去均值（若 pod_cfg.center 为 True）。
    4. 对 X 做 SVD，得到空间 POD 基底。
    5. 截断到前 r 个模态，保存 U_r、均值场、奇异值与基础元数据。
    6. 返回包含能量谱等信息的字典。

    可视化：
    - verbose=True 时打印关键维度、均值、能量占比等
    - plot=True 时绘制奇异值谱和累计能量曲线
    """
    if verbose:
        print("=== [POD] Step 1: 读取原始数据 ===")
        print(f"  - nc_path : {data_cfg.nc_path}")
        print(f"  - var_keys: {data_cfg.var_keys}")

    # 1) 读取原始数据
    X_thwc = load_raw_nc(data_cfg)   # [T,H,W,C]
    T, H, W, C = X_thwc.shape
    N = T
    D = H * W * C

    if verbose:
        print(f"  -> 数据形状 [T,H,W,C] = [{T}, {H}, {W}, {C}]")
        print(f"  -> 展平后：N = {N} snapshots, D = {D} 空间维度")
        print(f"  -> X_thwc dtype: {X_thwc.dtype}, min={X_thwc.min():.4e}, max={X_thwc.max():.4e}")

    # 2) reshape 为 [N,D]
    X_flat = X_thwc.reshape(N, D).astype(np.float64)  # 用 float64 算 SVD 稳一些

    # 3) 去均值
    if verbose:
        print("=== [POD] Step 2: 去均值 ===")
        print(f"  - center: {pod_cfg.center}")

    if pod_cfg.center:
        mean_flat = X_flat.mean(axis=0, keepdims=True)
        Xc = X_flat - mean_flat
        if verbose:
            print(f"  -> mean_flat shape: {mean_flat.shape}, mean|mean_flat|={np.mean(np.abs(mean_flat)):.4e}")
    else:
        mean_flat = np.zeros((1, D), dtype=np.float64)
        Xc = X_flat
        if verbose:
            print("  -> 未去均值，mean_flat 全零")

    if verbose:
        err_mean = np.mean(Xc)
        std_Xc = np.std(Xc)
        print(f"  -> 去均值后整体均值 ~ {err_mean:.4e}, std = {std_Xc:.4e}")

    # 4) SVD：Xc = U S V^T
    if verbose:
        print("=== [POD] Step 3: SVD 分解 ===")
        print(f"  -> 调用 np.linalg.svd(Xc)，形状 [N,D] = [{N}, {D}]")
        print("  -> full_matrices=False 以节省内存")

    U_svd, S, Vh = np.linalg.svd(Xc, full_matrices=False)  # U[N,r0], S[r0], Vh[r0,D]
    r0 = S.shape[0]

    if verbose:
        print(f"  -> SVD 完成，r0 = {r0} (可用的最大模态数)")
        print(f"  -> 奇异值 S: max={S.max():.4e}, min={S.min():.4e}")
        print("  -> 前几阶奇异值:", ", ".join(f"{v:.3e}" for v in S[:5]))

    # 5) 截断
    r = min(pod_cfg.r, r0)
    if verbose:
        print("=== [POD] Step 4: 截断并计算能量谱 ===")
        print(f"  - 请求 r = {pod_cfg.r}, 实际可用 r0 = {r0}, 使用 r = {r}")

    Vr = Vh[:r, :].T.astype(np.float32, copy=False)          # [D,r]
    mean_flat32 = mean_flat.astype(np.float32, copy=False).reshape(-1)  # [D]
    S = S.astype(np.float32, copy=False)

    energy = (S ** 2)
    total_energy = float(energy.sum()) if energy.size > 0 else 1.0
    energy_frac = energy / total_energy
    cum_energy = np.cumsum(energy_frac)

    if verbose:
        def find_r_for_threshold(th: float) -> int:
            idx = np.searchsorted(cum_energy, th)
            return int(idx + 1)  # 1-based 模态数
        r90 = find_r_for_threshold(0.90)
        r95 = find_r_for_threshold(0.95)
        r99 = find_r_for_threshold(0.99)

        print(f"  -> 总能量: {total_energy:.4e}")
        print(f"  -> 用前 r={r} 阶累计能量 = {cum_energy[r-1]:.4f}")
        print(f"  -> 达到 90% 能量需要的模态数约: {r90}")
        print(f"  -> 达到 95% 能量需要的模态数约: {r95}")
        print(f"  -> 达到 99% 能量需要的模态数约: {r99}")

    # 6) 保存
    save_dir = pod_cfg.save_dir
    ensure_dir(save_dir)

    if verbose:
        print("=== [POD] Step 5: 保存结果 ===")
        print(f"  -> save_dir = {save_dir}")
        print("  -> 保存 Ur.npy, mean_flat.npy, singular_values.npy, pod_meta.json")

    from ..dataio.io_utils import save_numpy, save_json  # 已在顶部导入，这里只是提醒

    save_numpy(save_dir / "Ur.npy", Vr)                   # [D,r]
    save_numpy(save_dir / "mean_flat.npy", mean_flat32)   # [D]
    save_numpy(save_dir / "singular_values.npy", S)       # [r0]

    meta = {
        "T": T,
        "H": H,
        "W": W,
        "C": C,
        "N": N,
        "D": D,
        "r_requested": pod_cfg.r,
        "r_used": r,
        "center": pod_cfg.center,
        "total_energy": float(total_energy),
    }
    save_json(save_dir / "pod_meta.json", meta)

    result: Dict[str, Any] = {
        "singular_values": S,
        "energy": energy_frac,
        "cum_energy": cum_energy,
        "r_used": r,
        "mean_field": mean_flat32,
        "meta": meta,
        "save_dir": str(save_dir),
    }

    # 7) 可选作图
    if plot:
        try:
            import matplotlib.pyplot as plt

            # 整体风格稍微紧凑一点
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

            k = np.arange(1, len(S) + 1)

            # --- 左：奇异值谱（半对数，线+小点） ---
            ax1.semilogy(
                k,
                S,
                marker="o",
                markersize=2,      # 点缩小
                linewidth=0.5,     # 细线
            )
            ax1.set_xlabel("Mode index k")
            ax1.set_ylabel("Singular value σ_k")
            ax1.set_title("POD Singular Value Spectrum")
            ax1.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
            ax1.set_ylim(1e-4, S.max()*1.1)

            # --- 右：剩余能量 (1 - cumulative energy) 的对数图 ---
            cum = cum_energy
            residual = 1 - cum
            K_zoom = min(500, len(cum))

            ax2.semilogy(
                k[:K_zoom],
                residual[:K_zoom],
                marker="o",
                markersize=2,
                linewidth=0.5,
            )

            ax2.set_xlabel("Mode index k (zoomed)")
            ax2.set_ylabel("Remaining energy 1 - cum(k)")
            ax2.set_title(f"Log-scale Remaining Energy (first {K_zoom} modes)")

            # 参考线：不同数量级的“剩余能量阈值”
            for thr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
                ax2.axhline(thr, ls="--", lw=0.6, label=f"{thr:.0e}")

            ax2.set_ylim(
                residual[:K_zoom].min() * 0.8,
                1.0
            )
            ax2.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
            ax2.legend(fontsize=7, loc="upper right")

            fig.tight_layout()
            result["fig_pod"] = fig

        except Exception as e:
            if verbose:
                print(f"[POD] 绘图失败（可忽略）：{e}")

    if verbose:
        print("=== [POD] 完成 ===")

    return result