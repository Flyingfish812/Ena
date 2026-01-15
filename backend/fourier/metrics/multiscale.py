"""
基于 POD 模态的多尺度误差评估。

这里提供的是一个针对「系数按 band 统计」的简洁封装：
    - compute_pod_band_errors(a_hat, a_true, bands)

其内部委托给 metrics.rmse_per_band，实现与其它指标接口的统一。
"""

from typing import Dict, Tuple

import numpy as np

from .metrics import rmse_per_band  # 与本目录下 metrics.py 协同工作


def compute_pod_band_errors(
    a_hat: np.ndarray,
    a_true: np.ndarray,
    bands: Dict[str, Tuple[int, int]],
) -> Dict[str, float]:
    """
    按给定的 POD 模态区间（band）计算每个 band 的系数 RMSE。

    参数
    ----
    a_hat:
        预测的 POD 系数，形状 [N, r] 或 [r]。
    a_true:
        真实的 POD 系数，形状 [N, r] 或 [r]。
    bands:
        例如 {"L": (0,10), "M": (10,40), "S": (40,128)}，
        下标区间为半开区间 [start, end)，0-based 索引。

    返回
    ----
    band_errors:
        例如 {"L": 0.01, "M": 0.02, "S": 0.05}，这里的数值是该 band 内所有样本、
        所有模态的 **系数 RMSE**：

            RMSE_band = sqrt( mean( (a_hat - a_true)^2 ) )

    说明
    ----
    该函数等价于 backend.metrics.metrics.rmse_per_band(...)，
    保留此别名是为了兼容旧代码和清晰的语义。
    """
    return rmse_per_band(a_hat=a_hat, a_true=a_true, bands=bands)
