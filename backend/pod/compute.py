# backend/pod/compute.py

"""
构建 POD 基底：从原始数据计算 SVD / POD，保存截断基底与元数据。
"""

from typing import Dict, Any, Optional

import numpy as np

from backend.config.schemas import DataConfig, PodConfig
from backend.dataio.nc_loader import load_raw_nc
from backend.dataio.io_utils import ensure_dir, save_numpy, save_json
from backend.pod.scaler import build_scale_table, build_basis_spectrum
from pathlib import Path


def _build_phi_groups(r_used: int, group_size: int = 16):
    """
    将前 r_used 个模态按 group_size 分组，返回一个列表：
    [
        {"group_index": 1, "name": "S1", "k_start": 1,  "k_end": min(16, r_used)},
        {"group_index": 2, "name": "S2", "k_start": 17, "k_end": min(32, r_used)},
        ...
    ]
    """
    groups = []
    k_start = 1  # 1-based index
    g_idx = 1
    while k_start <= r_used:
        k_end = min(k_start + group_size - 1, r_used)
        groups.append(
            {
                "group_index": g_idx,
                "name": f"S{g_idx}",
                "k_start": k_start,
                "k_end": k_end,
            }
        )
        g_idx += 1
        k_start = k_end + 1
    return groups

def plot_pod_outputs(
    *,
    save_dir: Path,
    pod_cfg: Optional[PodConfig] = None,
    verbose: bool = True,
    # 允许直接传入内存变量以避免重复 IO；为空则自动从磁盘读取
    singular_values: Optional[np.ndarray] = None,
    cum_energy: Optional[np.ndarray] = None,
    df_scale: Any = None,
    dpi: int = 180,
) -> Dict[str, Any]:
    """
    绘制 POD 相关图像（可被 build_pod 调用，也可外部单独调用，基于 save_dir 的存盘文件作图）

    读取优先级：
    - singular_values: 参数传入 > save_dir/singular_values.npy
    - df_scale: 参数传入 > save_dir/scale_table.csv（若存在）
    - cum_energy: 参数传入；若为空则由 singular_values 计算得到

    输出：
    - 返回 {"fig_pod": fig, "fig_scale_eff": fig2, "fig_paths": [...]}（有就返回）
    - 同时将 png 保存到 save_dir 下
    """
    out: Dict[str, Any] = {}
    fig_paths = []

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        if verbose:
            print(f"[plot_pod_outputs] matplotlib 不可用：{e}")
        return out

    # ---------
    # 1) 读取/准备 singular_values & cum_energy
    # ---------
    S = singular_values
    if S is None:
        npy = save_dir / "singular_values.npy"
        if npy.exists():
            S = np.load(npy)
        else:
            if verbose:
                print(f"[plot_pod_outputs] 缺少 singular_values：{npy}")
            return out

    S = np.asarray(S)

    if cum_energy is None:
        energy = (S.astype(np.float64) ** 2)
        total = float(energy.sum()) if energy.size > 0 else 1.0
        energy_frac = energy / total
        cum_energy = np.cumsum(energy_frac)

    cum = np.asarray(cum_energy)

    # ------------------------
    # 图 1：奇异值谱 + 剩余能量
    # ------------------------
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    k = np.arange(1, len(S) + 1)

    ax1.semilogy(
        k,
        S,
        marker="o",
        markersize=2,
        linewidth=0.5,
    )
    ax1.set_xlabel("Mode index k")
    ax1.set_ylabel("Singular value σ_k")
    ax1.set_title("POD Singular Value Spectrum")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)

    # y 轴下界别硬编码 1e-4：对不同数据更鲁棒
    s_pos = S[np.isfinite(S) & (S > 0)]
    ymin = float(np.min(s_pos)) * 0.8 if s_pos.size > 0 else 1e-8
    ymax = float(np.max(s_pos)) * 1.1 if s_pos.size > 0 else 1.0
    ax1.set_ylim(max(ymin, 1e-12), max(ymax, 1e-11))

    residual = 1.0 - cum
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

    for thr in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
        ax2.axhline(thr, ls="--", lw=0.6, label=f"{thr:.0e}")

    # residual 的最小值可能为 0（数值累积），避免 log 下界炸掉
    res_pos = residual[:K_zoom][np.isfinite(residual[:K_zoom]) & (residual[:K_zoom] > 0)]
    res_ymin = float(np.min(res_pos)) * 0.8 if res_pos.size > 0 else 1e-12
    ax2.set_ylim(max(res_ymin, 1e-12), 1.0)

    ax2.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
    ax2.legend(fontsize=7, loc="upper right")

    fig1.tight_layout()
    png1 = save_dir / "fig_pod_spectrum.png"
    fig1.savefig(png1, dpi=dpi, bbox_inches="tight")
    fig_paths.append(str(png1))
    out["fig_pod"] = fig1

    # ---------
    # 2) 读取/准备 df_scale（可选）
    # ---------
    if df_scale is None:
        csv = save_dir / "scale_table.csv"
        if csv.exists():
            try:
                import pandas as pd
                df_scale = pd.read_csv(csv)
            except Exception as e:
                if verbose:
                    print(f"[plot_pod_outputs] 读取 scale_table.csv 失败：{e}")
                df_scale = None

    # ------------------------
    # 图 2：模态堆叠数 - 有效尺度（散点 + 拐点拟合线，四条曲线同图）
    # ------------------------
    enable_scale = getattr(pod_cfg, "enable_scale_analysis", None)
    # 外部调用时 pod_cfg 可能为 None：只要 scale_table.csv 存在就画
    if (enable_scale is None or bool(enable_scale)) and (df_scale is not None):
        fig2, ax = plt.subplots(1, 1, figsize=(7.8, 5.2))

        r = df_scale["mode"].values.astype(np.int64) + 1  # 1-based

        def _extract_step_points(x: np.ndarray, y: np.ndarray, *, eps: float = 1e-12):
            """
            从“有效尺度”（通常是 cummin / 单调包络）中提取阶梯拐点：
            只保留 y 发生“严格下降”的那些点（以及第一个点）。
            返回：x_step, y_step
            """
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            if x.size == 0:
                return x, y

            idx = [0]
            cur = y[0]
            for i in range(1, len(y)):
                if y[i] < cur - eps:
                    idx.append(i)
                    cur = y[i]
            return x[idx], y[idx]

        def _fit_exp_floor(
            x: np.ndarray,
            y: np.ndarray,
            *,
            floor_quantile: float = 0.05,
            floor_eps: float = 1e-6,
        ):
            """
            带下限指数拟合：y = c + a * exp(b*x)

            轻量实现（无 scipy）：
            - c 用 y 的低分位数估计（地板）
            - 对 (y-c) 做 ln 线性拟合：ln(y-c) = ln(a) + b*x

            返回：a, b, c, eq_str
            """
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)

            m0 = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if np.count_nonzero(m0) < 4:
                return float("nan"), float("nan"), float("nan"), "y = c + a·exp(bx) (fit failed)"

            xx = x[m0]
            yy = y[m0]

            # floor 估计：低分位数比 min 更抗离群
            c0 = float(np.quantile(yy, np.clip(floor_quantile, 0.0, 0.49)))
            c = max(0.0, c0 - floor_eps)

            y_shift = yy - c
            m1 = y_shift > 0
            if np.count_nonzero(m1) < 3:
                # 退回到 min 附近
                c = max(0.0, float(np.min(yy)) - floor_eps)
                y_shift = yy - c
                m1 = y_shift > 0
                if np.count_nonzero(m1) < 3:
                    return float("nan"), float("nan"), c, "y = c + a·exp(bx) (fit failed)"

            xx2 = xx[m1]
            ly = np.log(y_shift[m1])

            b, ln_a = np.polyfit(xx2, ly, deg=1)
            a = float(np.exp(ln_a))
            eq = f"y = {c:.3g} + {a:.3g}·exp({b:.3g}x)"
            return a, float(b), float(c), eq

        series = [
            ("ell_x_eff",   df_scale["ell_x_eff"].values,   "o"),
            ("ell_y_eff",   df_scale["ell_y_eff"].values,   "s"),
            ("ell_min_eff", df_scale["ell_min_eff"].values, "D"),
            ("ell_geo_eff", df_scale["ell_geo_eff"].values, "^"),
        ]

        for name, y, marker in series:
            y = np.asarray(y, dtype=np.float64)

            color = ax._get_lines.get_next_color()

            # 1) 全量散点：真实点（淡）
            ax.scatter(
                r, y,
                s=14,
                marker=marker,
                alpha=0.35,
                linewidths=0.0,
                color=color,
            )

            # 2) 提取阶梯拐点（用于拟合）
            r_step, y_step = _extract_step_points(r, y)

            # 拐点更醒目一些（同色、更不透明）
            ax.scatter(
                r_step, y_step,
                s=28,
                marker=marker,
                alpha=0.9,
                linewidths=0.0,
                color=color,
            )

            # 3) 在拐点上做带下限指数拟合：y = c + a*exp(bx)
            a, b, c, eq = _fit_exp_floor(
                r_step, y_step,
                floor_quantile=0.05,
                floor_eps=1e-6,
            )
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
                y_fit = c + a * np.exp(b * r.astype(np.float64))
                ax.plot(
                    r, y_fit,
                    linestyle="--",
                    linewidth=2.0,
                    color=color,
                    label=f"{name}: {eq}",
                )
            else:
                ax.plot([], [], label=f"{name}: {eq}")

        ax.set_title("Effective Scale vs Mode Count (scatter + step-fit exp-with-floor)")
        ax.set_xlabel("Mode count r (1-based)")
        ax.set_ylabel("effective scale")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.3, alpha=0.5)
        ax.legend(fontsize=8, loc="upper right", frameon=True)

        fig2.tight_layout()
        png2 = save_dir / "fig_scale_eff_scatter_fit.png"
        fig2.savefig(png2, dpi=dpi, bbox_inches="tight")
        fig_paths.append(str(png2))
        out["fig_scale_eff"] = fig2

    out["fig_paths"] = fig_paths
    return out

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
    5. 截断到前 r 个模态，保存 U_r、均值场、奇异值、真实 POD 系数 A_true、
       POD 特征值 eigenvalues 以及基础元数据与模态分组信息。
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
    del X_thwc  # 及时释放内存

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

    # 5) 截断，并计算能量谱 / POD 特征值 / 真实系数
    r = min(pod_cfg.r, r0)
    if verbose:
        print("=== [POD] Step 4: 截断并计算能量谱 / POD 特征值 / 系数矩阵 ===")
        print(f"  - 请求 r = {pod_cfg.r}, 实际可用 r0 = {r0}, 使用 r = {r}")

    # 注意：此处 S 仍是长度 r0 的一维数组，我们只对前 r 阶做系数和特征值
    S = S.astype(np.float32, copy=False)
    S_r = S[:r]  # [r]

    # POD 基底（空间模态）：Vr[D,r]
    Vr = Vh[:r, :].T.astype(np.float32, copy=False)   # [D,r]

    # 去均值场（展平）
    mean_flat32 = mean_flat.astype(np.float32, copy=False).reshape(-1)  # [D]

    # 能量谱（基于全部 r0 阶奇异值）
    energy = (S ** 2)        # [r0]
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

    # === 新增：计算真实 POD 系数 A_true 与 POD 特征值 eigenvalues ===
    # 对于中心化后的数据 Xc = U S V^T，前 r 阶的时间系数 a_true(t,k) = U[:,k] * S[k]
    # 这里只保留前 r 个模态的系数，形状 [N, r]
    U_r = U_svd[:, :r].astype(np.float32, copy=False)    # [N,r]
    A_true = U_r * S_r.reshape(1, r)                     # [N,r]，广播乘

    # POD 特征值 λ_k = S_k^2 / (N - 1)，仅前 r 阶
    # 若 N=1（极端情况），避免除零，直接使用 S_k^2
    if N > 1:
        eigenvalues = (S_r.astype(np.float64) ** 2) / float(N - 1)
    else:
        eigenvalues = (S_r.astype(np.float64) ** 2)
    eigenvalues = eigenvalues.astype(np.float32, copy=False)  # [r]

    if verbose:
        print("  -> 已计算 A_true (真实 POD 系数) 和 eigenvalues (POD 特征值)")
        print(f"     A_true shape      : {A_true.shape}")
        print(f"     eigenvalues shape : {eigenvalues.shape}")

    # === 新增：构建按 16 模态一组的 φ 分组信息 ===
    phi_groups = _build_phi_groups(r_used=r, group_size=16)
    if verbose:
        print(f"  -> φ 被分成 {len(phi_groups)} 组，每组最多 16 个模态")

    # === Level-1 扩展产物：模态尺度表 & 频谱字典 ===
    q_modes_full = Vr.T.reshape(r, H, W, C)  # [r,H,W,C]

    if verbose:
        print("=== [POD:L1] 构建模态尺度表与频谱字典 ===")
        print(f"  -> q_modes_full shape = {q_modes_full.shape}  (r,H,W,C)")
        print(f"  -> scale_channel_reduce = {getattr(pod_cfg, 'scale_channel_reduce', 'l2')}")

    # 将多通道聚合成单标量场用于“尺度”主线
    reduce_mode = getattr(pod_cfg, "scale_channel_reduce", "l2")
    if reduce_mode == "l2":
        q_modes = np.sqrt(
            np.sum(q_modes_full.astype(np.float64) ** 2, axis=-1)
        ).astype(np.float32)
    elif reduce_mode == "sum":
        q_modes = np.sum(q_modes_full, axis=-1).astype(np.float32)
    elif reduce_mode == "u":
        q_modes = q_modes_full[..., 0].astype(np.float32)
    elif reduce_mode == "v":
        q_modes = q_modes_full[..., 1].astype(np.float32)
    else:
        raise ValueError(f"Unknown scale_channel_reduce: {reduce_mode}")

    if verbose:
        print(f"  -> q_modes (for scale) shape = {q_modes.shape}  (r,H,W)")
        vmin = float(np.nanmin(q_modes))
        vmax = float(np.nanmax(q_modes))
        vmean = float(np.nanmean(q_modes))
        print(f"     value range: min={vmin:.3e}, max={vmax:.3e}, mean={vmean:.3e}")

    grid_meta = {
        "H": H,
        "W": W,
        "C": C,
        "dx": getattr(pod_cfg, "dx", 1.0),
        "dy": getattr(pod_cfg, "dy", 1.0),
        "scale_channel_reduce": reduce_mode,
    }

    if verbose:
        print(f"  -> grid_meta: H={H}, W={W}, C={C}, dx={grid_meta['dx']}, dy={grid_meta['dy']}")

    # --- ScaleTable ---
    if getattr(pod_cfg, "enable_scale_analysis", False):
        if verbose:
            print("  -> 生成 ScaleTable (scale_table.csv)")
            print(f"     scale_analysis cfg = {pod_cfg.scale_analysis}")

        df_scale = build_scale_table(
            q_modes=q_modes,
            grid_meta=grid_meta,
            cfg_scale=pod_cfg.scale_analysis,
            out_csv=pod_cfg.save_dir / "scale_table.csv",
            out_meta=pod_cfg.save_dir / "scale_meta.json",
            preview=verbose,
        )

        if verbose:
            print("     ScaleTable 写入完成")

    else:
        df_scale = None
        if verbose:
            print("  -> 跳过 ScaleTable（enable_scale_analysis = False）")

    # --- Basis spectrum (Q_i) ---
    if getattr(pod_cfg, "enable_basis_spectrum", False):
        if verbose:
            print("  -> 生成 basis_spectrum.npz（模态复数频谱字典）")
            print(f"     fft_basis cfg = {pod_cfg.fft_basis}")

        build_basis_spectrum(
            q_modes=q_modes_full,   # [r,H,W,C]
            grid_meta=grid_meta,
            out_npz=pod_cfg.save_dir / "basis_spectrum.npz",
            fft_cfg=pod_cfg.fft_basis,
        )

        if verbose:
            print("     basis_spectrum.npz 写入完成")

    else:
        if verbose:
            print("  -> 跳过 basis_spectrum（enable_basis_spectrum = False）")

    # 释放大数组（除了 Vr / mean_flat32 / A_true 这些后续要用的）
    del U_svd, Vh, X_flat, Xc

    # 6) 保存
    save_dir = pod_cfg.save_dir
    ensure_dir(save_dir)

    if verbose:
        print("=== [POD] Step 5: 保存结果 ===")
        print(f"  -> save_dir = {save_dir}")
        print("  -> 保存 Ur.npy, mean_flat.npy, singular_values.npy,")
        print("          A_true.npy, eigenvalues.npy, pod_meta.json, phi_groups.json")

    # 空间基底与均值场
    save_numpy(save_dir / "Ur.npy", Vr)                   # [D,r]
    save_numpy(save_dir / "mean_flat.npy", mean_flat32)   # [D]
    # 全部奇异值（r0 阶）
    save_numpy(save_dir / "singular_values.npy", S)       # [r0]

    # 新增：真实 POD 系数矩阵（仅前 r_used 阶）
    save_numpy(save_dir / "A_true.npy", A_true)           # [N,r]
    # 新增：POD 特征值（仅前 r_used 阶）
    save_numpy(save_dir / "eigenvalues.npy", eigenvalues) # [r]

    # 元数据与分组信息
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
        "phi_group_size": 16,
        "phi_group_count": len(phi_groups),
    }
    save_json(save_dir / "pod_meta.json", meta)

    # 新增：φ 分组信息单独保存，便于多尺度分析直接读取
    save_json(save_dir / "phi_groups.json", {"groups": phi_groups})

    result: Dict[str, Any] = {
        "singular_values": S,          # [r0]
        "energy": energy_frac,         # [r0]
        "cum_energy": cum_energy,      # [r0]
        "r_used": r,
        "mean_field": mean_flat32,     # [D]
        "Ur": Vr,                      # [D,r]
        "A_true": A_true,              # [N,r]
        "eigenvalues": eigenvalues,    # [r]
        "phi_groups": phi_groups,      # list of dicts
        "meta": meta,
        "save_dir": str(save_dir),
    }

    # 7) 可选作图（拆出到独立函数，便于外部基于存盘复绘）
    if plot:
        try:
            plot_out = plot_pod_outputs(
                save_dir=pod_cfg.save_dir,
                pod_cfg=pod_cfg,
                verbose=verbose,
                singular_values=S,
                cum_energy=cum_energy,
                df_scale=df_scale,
                dpi=int(getattr(pod_cfg, "plot_dpi", 180)),
            )
            # 合并回 result，保持原先 build_pod 的返回习惯
            result.update({k: v for k, v in (plot_out or {}).items() if v is not None})
        except Exception as e:
            if verbose:
                print(f"[POD] 绘图失败（可忽略）：{e}")


    if verbose:
        print("=== [POD] 完成 ===")

    return result