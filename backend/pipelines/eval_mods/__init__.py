# backend/pipelines/eval_mods/__init__.py

"""
eval_mods:
仅存放 Level-4 的“具体评估 / 画图模组实现”。
注意：
- eval_mods 不再包含任何“引擎 / 上下文 / 注册表”实现
- 所有核心逻辑统一位于 backend.pipelines.eval.*
- 本包只对外暴露各类 register_*_mods() 函数
"""
from .fourier_mods import register_fourier_mods
from .examples_mods import register_example_mods

__all__ = [
    "register_fourier_mods",
    "register_example_mods",
]