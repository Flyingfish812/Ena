# backend/pod/compute.py

"""
构建 POD 基底：从原始数据计算 SVD / POD，保存截断基底与元数据。
"""

from typing import Dict, Any

import numpy as np

from ..config.schemas import DataConfig, PodConfig


def build_pod(data_cfg: DataConfig, pod_cfg: PodConfig) -> Dict[str, Any]:
    """
    完整执行一次 POD 构建流程。

    步骤（实现时）：
    1. 调用 dataio.load_raw_nc 读取 [T,H,W,C] 数组。
    2. 将数据 reshape 为 [N, D]，其中 N = T，D = H*W*C。
    3. 对每个空间点做去均值（若 pod_cfg.center 为 True）。
    4. 对 X 做 SVD 或特征分解，得到 POD 基底。
    5. 截断到前 r 个模态，保存 U_r、均值场、奇异值等到 pod_cfg.save_dir。
    6. 返回包含能量谱等信息的字典。

    返回
    ----
    result:
        {
            "singular_values": np.ndarray,
            "energy": np.ndarray,
            "cum_energy": np.ndarray,
            "r_used": int,
            "mean_field": np.ndarray,
        }
    """
    raise NotImplementedError
