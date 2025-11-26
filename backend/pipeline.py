# backend/pipeline.py

"""
为 notebook / GUI 提供的高层“一键调用”接口。
"""

from pathlib import Path
from typing import Dict, Any

from .config.schemas import DataConfig, PodConfig, TrainConfig, EvalConfig
from .pod.compute import build_pod


def run_build_pod_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    *,
    verbose: bool = True,
    plot: bool = False,
) -> Dict[str, Any]:
    """
    顶层 POD 构建入口。

    - 读取原始数据
    - 执行 SVD / POD
    - 截断并保存基底与均值
    - 返回能量谱等元信息，供 notebook/GUI 作图

    参数
    ----
    verbose:
        是否打印中间过程信息。
    plot:
        是否直接画出奇异值谱与累计能量图。
    """
    result = build_pod(data_cfg, pod_cfg, verbose=verbose, plot=plot)
    return result


def run_train_mlp_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    train_cfg: TrainConfig,
) -> Dict[str, Any]:
    """
    顶层 MLP 训练入口。

    这一批暂不实现，占位。
    """
    raise NotImplementedError


def run_full_eval_pipeline(
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
) -> Dict[str, Any]:
    """
    顶层评估入口。

    这一批暂不实现，占位。
    """
    raise NotImplementedError


def quick_build_pod(
    nc_path: str | Path,
    save_dir: str | Path = "artifacts/pod",
    r: int = 128,
    center: bool = True,
    var_keys: tuple[str, ...] = ("u", "v"),
    *,
    verbose: bool = True,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    提供给 notebook 的“一行跑 POD”接口。

    示例用法（ipynb）：
    >>> from backend.pipeline import quick_build_pod
    >>> res = quick_build_pod("data/my.nc", "artifacts/pod_nc_r128", r=128)

    参数
    ----
    nc_path:
        NetCDF 文件路径。
    save_dir:
        POD 基底输出目录。
    r:
        截断模态数。
    center:
        是否去均值。
    var_keys:
        需要读取的变量名元组，例如 ("u","v")。
    verbose:
        是否打印中间过程信息。
    plot:
        是否绘制奇异值谱与累计能量曲线。

    返回
    ----
    result:
        与 build_pod 返回结构相同，多一个可选的 "fig_pod"。
    """
    data_cfg = DataConfig(
        nc_path=Path(nc_path),
        var_keys=var_keys,
        cache_dir=None,
    )
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=Path(save_dir),
    )
    return run_build_pod_pipeline(
        data_cfg,
        pod_cfg,
        verbose=verbose,
        plot=plot,
    )
