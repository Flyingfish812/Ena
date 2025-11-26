# backend/dataio/io_utils.py

"""
数组与元数据的简单读写工具，用于缓存 POD/模型/结果等。
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import json


def ensure_dir(path: Path) -> None:
    """
    确保目录存在，若不存在则创建。
    """
    raise NotImplementedError


def save_numpy(path: Path, arr: np.ndarray) -> None:
    """
    将 numpy 数组保存到 .npy 文件。
    """
    raise NotImplementedError


def load_numpy(path: Path) -> np.ndarray:
    """
    从 .npy 文件读取 numpy 数组。
    """
    raise NotImplementedError


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    将字典保存为 .json 文件，保证缩进和 UTF-8 编码。
    """
    raise NotImplementedError


def load_json(path: Path) -> Dict[str, Any]:
    """
    从 .json 文件加载字典。
    """
    raise NotImplementedError
