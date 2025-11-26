# backend/dataio/__init__.py

"""
数据输入输出与切分相关模块。
"""

from .nc_loader import load_raw_nc
from .splits import train_val_test_split

__all__ = [
    "load_raw_nc",
    "train_val_test_split",
]
