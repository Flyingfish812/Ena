# backend/sampling/__init__.py

"""
观测 mask 与噪声注入相关工具。
"""

from .masks import generate_random_mask_hw, apply_mask_flat
from .noise import add_gaussian_noise

__all__ = [
    "generate_random_mask_hw",
    "apply_mask_flat",
    "add_gaussian_noise",
]
