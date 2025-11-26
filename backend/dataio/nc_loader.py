# backend/dataio/nc_loader.py

"""
读取原始 NetCDF 数据，统一为 numpy 数组格式。

目标：对 Ena 而言，这里只需要支持当前这一个数据集，
不追求对任意 NC 的完全泛化。
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import netCDF4 as nc

from ..config.schemas import DataConfig


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
    raise NotImplementedError


def infer_thwc_shape(ds: nc.Dataset, var_key: str) -> Tuple[int, int, int, int]:
    """
    从给定变量中推断 (T, H, W, C) 中的 T/H/W 形状。

    Ena 的数据集是已知格式，这里可以写成对当前数据集的
    “半硬编码”推断逻辑，而不用做极端通用。
    """
    raise NotImplementedError
