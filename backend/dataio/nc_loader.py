# backend/dataio/nc_loader.py

"""
读取原始 NetCDF 数据，统一为 numpy 数组格式。

目标：
- 仅支持当前 Ena 使用的这份数据集
- 将若干变量拼成 [T, H, W, C] 的 float32 数组
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import netCDF4 as nc

from ..config.schemas import DataConfig


def _reorder_to_THW(arr: np.ndarray) -> np.ndarray:
    """
    将一个 3D 数组重排为 [T, H, W]。

    约定（针对 Ena 当前数据集）：
    - 数组有 3 个轴，对应 {T, H, W} 的某种排列；
    - 最大尺寸轴视为时间 T；
    - 最小尺寸轴视为高度 H；
    - 剩余的那个轴视为宽度 W。
    """
    import numpy as np

    if arr.ndim != 3:
        raise ValueError(f"Expect 3D array for variable, got shape {arr.shape}")

    shape = np.array(arr.shape)
    order = np.argsort(shape)  # 小→大

    iy = int(order[0])         # H = 最小维
    it = int(order[-1])        # T = 最大维
    # 剩下的那个就是 W
    ix_candidates = [i for i in range(3) if i not in (iy, it)]
    if not ix_candidates:
        raise RuntimeError(f"Cannot infer W axis from shape {arr.shape}")
    ix = int(ix_candidates[0])

    # Debug 输出（你要的话可以留着）
    # print(f"[debug] arr.shape = {arr.shape}, (it,iy,ix) = ({it},{iy},{ix})")

    # 目标顺序是 [T,H,W]
    arr_thw = np.moveaxis(arr, (it, iy, ix), (0, 1, 2))
    return arr_thw.astype(np.float32, copy=False)

def load_raw_nc(cfg: DataConfig) -> np.ndarray:
    """
    读取 NetCDF 文件中的指定变量，返回 [T, H, W, C] 的 float32 数组。

    参数
    ----
    cfg:
        DataConfig，其中包含 nc_path 与 var_keys。

    返回
    ----
    X:
        形状为 [T, H, W, C] 的数组，其中 C = len(var_keys)。
    """
    nc_path: Path = cfg.nc_path
    var_keys = cfg.var_keys

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    ds = nc.Dataset(str(nc_path), "r")
    try:
        thw_list = []
        for k in var_keys:
            if k not in ds.variables:
                raise KeyError(f"Variable '{k}' not found in NetCDF file: {nc_path}")
            raw = np.asarray(ds.variables[k][:])
            thw = _reorder_to_THW(raw)  # [T,H,W]
            thw_list.append(thw)

        # [T,H,W,C]
        X = np.stack(thw_list, axis=-1)
        return X.astype(np.float32, copy=False)
    finally:
        ds.close()


def infer_thwc_shape(ds: nc.Dataset, var_key: str) -> Tuple[int, int, int, int]:
    """
    从给定变量中推断 (T, H, W, C) 中的 T/H/W 形状。

    在 Ena 中，本函数目前主要用于调试或 sanity check，
    实际数据读取使用 load_raw_nc 内部的 _reorder_to_THW。

    返回
    ----
    T, H, W, C:
        其中 C 暂时固定为 1（单变量）。
    """
    if var_key not in ds.variables:
        raise KeyError(f"Variable '{var_key}' not found in dataset.")
    arr = np.asarray(ds.variables[var_key][:])
    if arr.ndim != 3:
        raise ValueError(f"Expect 3D array for variable, got shape {arr.shape}")

    shape = np.array(arr.shape)
    order = np.argsort(shape)
    H = int(shape[order[0]])
    W = int(shape[order[-1]])
    it_candidates = [i for i in range(3) if i not in (order[0], order[-1])]
    T = int(shape[it_candidates[0]])
    C = 1
    return T, H, W, C
