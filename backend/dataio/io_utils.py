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

    注意：
    - 如果传入的是文件路径，则创建其父目录。
    - 如果传入的是目录路径，则直接创建该目录。
    """
    path = Path(path)
    # 简单判断：有后缀就认为是文件路径
    if path.suffix:
        dir_path = path.parent
    else:
        dir_path = path
    dir_path.mkdir(parents=True, exist_ok=True)


def save_numpy(path: Path, arr: np.ndarray) -> None:
    """
    将 numpy 数组保存到 .npy 文件。
    """
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, arr)


def load_numpy(path: Path) -> np.ndarray:
    """
    从 .npy 文件读取 numpy 数组。
    """
    path = Path(path)
    return np.load(path)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    将字典保存为 .json 文件，保证缩进和 UTF-8 编码。
    """
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    """
    从 .json 文件加载字典。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
