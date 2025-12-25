from backend.pod.compute import build_pod
from backend.config.schemas import DataConfig, PodConfig
from typing import Any, Dict
from pathlib import Path


"""
POD 构建的一站式入口 (Notebook / GUI 推荐接口)

功能：
- 从 NetCDF 文件读取原始数据
- 执行 SVD / POD 分解
- 按 r 截断并保存基底与均值
- 返回能量谱、累计能量、实际使用模态数等元信息

示例 (ipynb): 
from backend.pipeline_pod import quick_build_pod
res = quick_build_pod(
    nc_path="data/cylinder2d.nc",
    r=128,
    center=True,
    var_keys=("u", "v"),
    verbose=True,
    plot=True,
)
print(res["r_used"], res["cum_energy"][:10])

参数：
- nc_path: NetCDF 文件路径
- save_dir: POD 基底输出目录
- r: 截断模态数
- center: 是否去均值
- var_keys: 读取的变量名
- verbose: 是否打印中间信息
- plot: 是否绘制奇异值谱与累计能量图

返回：
- Dict[str, Any]，结构与 build_pod 返回一致
"""

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
    # 构造 DataConfig
    data_cfg = DataConfig(
        nc_path=Path(nc_path),
        var_keys=var_keys,
        cache_dir=None,
    )

    # 构造 PodConfig
    pod_cfg = PodConfig(
        r=r,
        center=center,
        save_dir=Path(save_dir),
    )

    # 执行 POD 构建
    result = build_pod(
        data_cfg,
        pod_cfg,
        verbose=verbose,
        plot=plot,
    )

    return result
