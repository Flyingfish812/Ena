# backend/models/mlp.py

"""
POD 系数回归用的简单 MLP 模型与其专用损失。
"""

import copy
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


class PodMLP(nn.Module):
    """
    一个简单的全连接 MLP，用于从观测向量 y ∈ R^M 预测 POD 系数 a ∈ R^r。

    结构示例:
        M -> 256 -> 256 -> r
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Iterable[int] = (256, 256),
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        def _act() -> nn.Module:
            return copy.deepcopy(activation)

        dims: List[int] = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 2:
                layers.append(_act())

        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: [batch, in_dim]
        返回: [batch, out_dim]
        """
        return self.net(y)


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        rank_value = int(rank)
        if rank_value <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_proj = nn.Linear(int(in_features), rank_value, bias=False)
        self.out_proj = nn.Linear(rank_value, int(out_features), bias=bool(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.in_proj(x))


class BudgetExpertMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        def _act() -> nn.Module:
            return copy.deepcopy(activation)

        dims = [int(in_dim)] + [int(v) for v in hidden_dims]
        if len(dims) < 2:
            raise ValueError(f"hidden_dims must contain at least one value, got {tuple(hidden_dims)}")

        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(_act())
        self.feature_net = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], int(out_dim))

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        return features, self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, out = self.forward_features(x)
        return out


@dataclass(frozen=True)
class ProgressiveGroupSpec:
    coarse_dim: int
    mid_dim: int
    fine_dim: int

    @property
    def stage2_dim(self) -> int:
        return int(self.coarse_dim + self.mid_dim)

    @property
    def total_dim(self) -> int:
        return int(self.coarse_dim + self.mid_dim + self.fine_dim)

    def as_dict(self) -> dict[str, int]:
        return {
            "coarse_dim": int(self.coarse_dim),
            "mid_dim": int(self.mid_dim),
            "fine_dim": int(self.fine_dim),
            "stage2_dim": int(self.stage2_dim),
            "total_dim": int(self.total_dim),
        }


def compute_progressive_group_spec(
    out_dim: int,
    ratios: Sequence[int] = (1, 2, 5),
) -> ProgressiveGroupSpec:
    total_dim = int(out_dim)
    ratio_arr = np.asarray(list(ratios), dtype=np.float64).reshape(-1)
    if total_dim <= 0:
        raise ValueError(f"out_dim must be positive, got {out_dim}")
    if ratio_arr.shape[0] != 3:
        raise ValueError(f"ratios must have length 3, got {tuple(ratio_arr.shape)}")
    if np.any(ratio_arr <= 0.0):
        raise ValueError(f"ratios must be positive, got {tuple(ratios)}")

    raw_sizes = float(total_dim) * ratio_arr / float(np.sum(ratio_arr))
    sizes = np.floor(raw_sizes).astype(np.int64)
    remainder = int(total_dim - int(sizes.sum()))
    if remainder > 0:
        frac = raw_sizes - sizes.astype(np.float64)
        for idx in np.argsort(-frac)[:remainder]:
            sizes[int(idx)] += 1

    if np.any(sizes <= 0):
        raise ValueError(
            f"Computed progressive group sizes must be positive, got {tuple(int(v) for v in sizes)} for out_dim={out_dim}"
        )

    return ProgressiveGroupSpec(
        coarse_dim=int(sizes[0]),
        mid_dim=int(sizes[1]),
        fine_dim=int(sizes[2]),
    )


class ProgressiveModalResidualHead(nn.Module):
    """
    第三代 PMRH：共享输入、分级宽度、弱耦合监督的可裁剪 MLP。

    - stage1 只计算一条真正的窄子网络: M -> 64 -> 64 -> 16
    - full 预算在保留 coarse 子空间的同时增开 refinement width
    - stage2 仅作为 full 输出前缀的兼容别名，不再拥有独立结构
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        coarse_hidden_dims: Sequence[int] = (64, 64),
        refinement_hidden_dims: Sequence[int] | None = None,
        stage2_feature_dim: int | None = None,
        stage2_head_hidden_dim: int | None = None,
        stage3_feature_dim: int | None = None,
        stage3_head_hidden_dim: int | None = None,
        group_ratios: Sequence[int] = (1, 2, 5),
        stage1_low_rank: int | None = None,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        def _act() -> nn.Module:
            return copy.deepcopy(activation)

        coarse_dims = [int(v) for v in coarse_hidden_dims]
        if len(coarse_dims) != 2:
            raise ValueError(f"coarse_hidden_dims must have length 2, got {coarse_hidden_dims}")
        if coarse_dims[0] <= 0 or coarse_dims[1] <= 0:
            raise ValueError(f"coarse_hidden_dims must be positive, got {coarse_hidden_dims}")

        if refinement_hidden_dims is None:
            refinement_hidden_dims = (
                192 if stage2_feature_dim is None else int(stage2_feature_dim),
                192 if stage3_feature_dim is None else int(stage3_feature_dim),
            )
        refine_dims = [int(v) for v in refinement_hidden_dims]
        if len(refine_dims) != 2:
            raise ValueError(f"refinement_hidden_dims must have length 2, got {refinement_hidden_dims}")
        if refine_dims[0] <= 0 or refine_dims[1] <= 0:
            raise ValueError(f"refinement_hidden_dims must be positive, got {refinement_hidden_dims}")
        if stage2_head_hidden_dim is not None and int(stage2_head_hidden_dim) <= 0:
            raise ValueError(f"stage2_head_hidden_dim must be positive, got {stage2_head_hidden_dim}")
        if stage3_head_hidden_dim is not None and int(stage3_head_hidden_dim) <= 0:
            raise ValueError(f"stage3_head_hidden_dim must be positive, got {stage3_head_hidden_dim}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_spec = compute_progressive_group_spec(out_dim=self.out_dim, ratios=group_ratios)
        self.coarse_hidden_dims = (int(coarse_dims[0]), int(coarse_dims[1]))
        self.refinement_hidden_dims = (int(refine_dims[0]), int(refine_dims[1]))
        self.stage1_feature_dim = int(self.coarse_hidden_dims[-1])
        self.stage2_feature_dim = int(self.refinement_hidden_dims[0])
        self.stage3_feature_dim = int(self.refinement_hidden_dims[1])

        coarse_input_proj: nn.Module
        if stage1_low_rank is None:
            coarse_input_proj = nn.Linear(self.in_dim, self.coarse_hidden_dims[0])
        else:
            coarse_input_proj = LowRankLinear(self.in_dim, self.coarse_hidden_dims[0], int(stage1_low_rank))

        self.coarse_input_proj = coarse_input_proj
        self.coarse_input_activation = _act()
        self.coarse_hidden_proj = nn.Linear(self.coarse_hidden_dims[0], self.coarse_hidden_dims[1])
        self.coarse_hidden_activation = _act()

        self.refine_input_proj = nn.Linear(self.in_dim, self.refinement_hidden_dims[0])
        self.refine_input_activation = _act()
        self.refine_hidden_from_coarse = nn.Linear(self.coarse_hidden_dims[0], self.refinement_hidden_dims[1], bias=False)
        self.refine_hidden_from_refine = nn.Linear(self.refinement_hidden_dims[0], self.refinement_hidden_dims[1])
        self.refine_hidden_activation = _act()

        self.coarse_head = nn.Linear(self.coarse_hidden_dims[1], self.group_spec.coarse_dim)
        self.full_head = nn.Linear(self.coarse_hidden_dims[1] + self.refinement_hidden_dims[1], self.group_spec.total_dim)

    def get_stage_modules(self, stage: str = "full") -> tuple[nn.Module, ...]:
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return (self.coarse_input_proj, self.coarse_hidden_proj, self.coarse_head)
        if stage_name in ("stage2", "full", "stage3"):
            return (
                self.coarse_input_proj,
                self.coarse_hidden_proj,
                self.refine_input_proj,
                self.refine_hidden_from_coarse,
                self.refine_hidden_from_refine,
                self.full_head,
            )
        raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

    def _encode_coarse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1_coarse = self.coarse_input_activation(self.coarse_input_proj(y))
        h2_coarse = self.coarse_hidden_activation(self.coarse_hidden_proj(h1_coarse))
        return h1_coarse, h2_coarse

    def forward_stages(self, y: torch.Tensor, stage: str = "full") -> dict[str, torch.Tensor]:
        stage_name = str(stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

        h1_coarse, h2_coarse = self._encode_coarse(y)
        y_coarse = self.coarse_head(h2_coarse)
        outputs: dict[str, torch.Tensor] = {
            "h1_coarse": h1_coarse,
            "h2_coarse": h2_coarse,
            "stage1": y_coarse,
            "coarse_solution": y_coarse,
        }
        if stage_name in ("stage1", "coarse"):
            return outputs

        h1_refine = self.refine_input_activation(self.refine_input_proj(y))
        h2_refine = self.refine_hidden_activation(
            self.refine_hidden_from_coarse(h1_coarse) + self.refine_hidden_from_refine(h1_refine)
        )
        full_hidden = torch.cat([h2_coarse, h2_refine], dim=-1)
        y_full = self.full_head(full_hidden)
        outputs.update(
            {
                "h1_refine": h1_refine,
                "h2_refine": h2_refine,
                "h_full": full_hidden,
                "stage2": y_full[:, : self.group_spec.stage2_dim],
                "full": y_full,
                "stage3": y_full,
                "full_solution": y_full,
            }
        )
        return outputs

    def forward(self, y: torch.Tensor, stage: str = "full") -> torch.Tensor:
        outputs = self.forward_stages(y, stage=stage)
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return outputs["stage1"]
        if stage_name == "stage2":
            return outputs["stage2"]
        return outputs["full"]


class SharedStemBudgetExpertMLP(nn.Module):
    """
    第四代 v4a：轻量共享 stem + 三个并行预算专家。

    - stage1/coarse: stem -> expert16 -> 16 维前缀
    - stage2:        stem -> expert48 -> 48 维前缀
    - full/stage3:   stem -> expert128 -> 全维输出
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        stem_dim: int = 48,
        stage1_hidden_dims: Sequence[int] = (64, 64),
        stage2_hidden_dims: Sequence[int] = (128, 128),
        stage3_hidden_dims: Sequence[int] = (256, 256),
        group_ratios: Sequence[int] = (1, 2, 5),
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        def _act() -> nn.Module:
            return copy.deepcopy(activation)

        stem_width = int(stem_dim)
        if stem_width <= 0:
            raise ValueError(f"stem_dim must be positive, got {stem_dim}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_spec = compute_progressive_group_spec(out_dim=self.out_dim, ratios=group_ratios)
        self.stem_dim = stem_width
        self.stage1_hidden_dims = tuple(int(v) for v in stage1_hidden_dims)
        self.stage2_hidden_dims = tuple(int(v) for v in stage2_hidden_dims)
        self.stage3_hidden_dims = tuple(int(v) for v in stage3_hidden_dims)
        if len(self.stage1_hidden_dims) == 0 or len(self.stage2_hidden_dims) == 0 or len(self.stage3_hidden_dims) == 0:
            raise ValueError("Each stage hidden_dims must contain at least one value")
        if any(v <= 0 for v in self.stage1_hidden_dims + self.stage2_hidden_dims + self.stage3_hidden_dims):
            raise ValueError("All expert hidden dims must be positive")

        self.stem_linear = nn.Linear(self.in_dim, self.stem_dim)
        self.stem_activation = _act()
        self.stage1_branch = BudgetExpertMLP(
            self.stem_dim,
            self.group_spec.coarse_dim,
            self.stage1_hidden_dims,
            activation=activation,
        )
        self.stage2_branch = BudgetExpertMLP(
            self.stem_dim,
            self.group_spec.stage2_dim,
            self.stage2_hidden_dims,
            activation=activation,
        )
        self.stage3_branch = BudgetExpertMLP(
            self.stem_dim,
            self.group_spec.total_dim,
            self.stage3_hidden_dims,
            activation=activation,
        )

    def encode_stem(self, y: torch.Tensor) -> torch.Tensor:
        return self.stem_activation(self.stem_linear(y))

    def get_stage_modules(self, stage: str = "full") -> tuple[nn.Module, ...]:
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return (self.stem_linear, self.stage1_branch)
        if stage_name == "stage2":
            return (self.stem_linear, self.stage2_branch)
        if stage_name in ("full", "stage3"):
            return (self.stem_linear, self.stage3_branch)
        raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

    def forward_stages(self, y: torch.Tensor, stage: str = "full") -> dict[str, torch.Tensor]:
        stage_name = str(stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

        h0 = self.encode_stem(y)
        outputs: dict[str, torch.Tensor] = {"stem": h0}
        if stage_name in ("stage1", "coarse"):
            z16, y16 = self.stage1_branch.forward_features(h0)
            outputs.update({"z16": z16, "stage1": y16, "coarse_solution": y16})
            return outputs
        if stage_name == "stage2":
            z48, y48 = self.stage2_branch.forward_features(h0)
            outputs.update({"z48": z48, "stage2": y48, "mid_solution": y48})
            return outputs

        z128, y128 = self.stage3_branch.forward_features(h0)
        outputs.update({"z128": z128, "full": y128, "stage3": y128, "full_solution": y128})
        return outputs

    def forward(self, y: torch.Tensor, stage: str = "full") -> torch.Tensor:
        outputs = self.forward_stages(y, stage=stage)
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return outputs["stage1"]
        if stage_name == "stage2":
            return outputs["stage2"]
        return outputs["full"]


class LatentGuidedBudgetExpertMLP(nn.Module):
    """
    第四代 v4b：轻量共享 stem + 受控 latent adapter 的预算专家。

    - stage1/coarse: stem -> expert16 -> 16 维前缀
    - stage2:        stem + sg(T16(z16)) -> expert48 -> 48 维前缀
    - full/stage3:   stem + sg(T48(z48)) -> expert128 -> 全维输出
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        stem_dim: int = 48,
        stage1_hidden_dims: Sequence[int] = (64, 64),
        stage2_hidden_dims: Sequence[int] = (128, 128),
        stage3_hidden_dims: Sequence[int] = (256, 256),
        adapter16_dim: int | None = None,
        adapter48_dim: int | None = None,
        group_ratios: Sequence[int] = (1, 2, 5),
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        def _act() -> nn.Module:
            return copy.deepcopy(activation)

        stem_width = int(stem_dim)
        if stem_width <= 0:
            raise ValueError(f"stem_dim must be positive, got {stem_dim}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.group_spec = compute_progressive_group_spec(out_dim=self.out_dim, ratios=group_ratios)
        self.stem_dim = stem_width
        self.stage1_hidden_dims = tuple(int(v) for v in stage1_hidden_dims)
        self.stage2_hidden_dims = tuple(int(v) for v in stage2_hidden_dims)
        self.stage3_hidden_dims = tuple(int(v) for v in stage3_hidden_dims)
        if len(self.stage1_hidden_dims) == 0 or len(self.stage2_hidden_dims) == 0 or len(self.stage3_hidden_dims) == 0:
            raise ValueError("Each stage hidden_dims must contain at least one value")
        if any(v <= 0 for v in self.stage1_hidden_dims + self.stage2_hidden_dims + self.stage3_hidden_dims):
            raise ValueError("All expert hidden dims must be positive")

        self.stage1_feature_dim = int(self.stage1_hidden_dims[-1])
        self.stage2_feature_dim = int(self.stage2_hidden_dims[-1])
        self.stage3_feature_dim = int(self.stage3_hidden_dims[-1])
        self.adapter16_dim = int(self.stage1_feature_dim if adapter16_dim is None else adapter16_dim)
        self.adapter48_dim = int(self.stage2_feature_dim if adapter48_dim is None else adapter48_dim)
        if self.adapter16_dim <= 0 or self.adapter48_dim <= 0:
            raise ValueError("adapter dims must be positive")

        self.stem_linear = nn.Linear(self.in_dim, self.stem_dim)
        self.stem_activation = _act()
        self.stage1_branch = BudgetExpertMLP(
            self.stem_dim,
            self.group_spec.coarse_dim,
            self.stage1_hidden_dims,
            activation=activation,
        )
        self.adapter_16_to_48 = nn.Linear(self.stage1_feature_dim, self.adapter16_dim)
        self.stage2_branch = BudgetExpertMLP(
            self.stem_dim + self.adapter16_dim,
            self.group_spec.stage2_dim,
            self.stage2_hidden_dims,
            activation=activation,
        )
        self.adapter_48_to_128 = nn.Linear(self.stage2_feature_dim, self.adapter48_dim)
        self.stage3_branch = BudgetExpertMLP(
            self.stem_dim + self.adapter48_dim,
            self.group_spec.total_dim,
            self.stage3_hidden_dims,
            activation=activation,
        )

    def encode_stem(self, y: torch.Tensor) -> torch.Tensor:
        return self.stem_activation(self.stem_linear(y))

    def get_stage_modules(self, stage: str = "full") -> tuple[nn.Module, ...]:
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return (self.stem_linear, self.stage1_branch)
        if stage_name == "stage2":
            return (self.stem_linear, self.stage1_branch, self.adapter_16_to_48, self.stage2_branch)
        if stage_name in ("full", "stage3"):
            return (
                self.stem_linear,
                self.stage1_branch,
                self.adapter_16_to_48,
                self.stage2_branch,
                self.adapter_48_to_128,
                self.stage3_branch,
            )
        raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

    def forward_stages(self, y: torch.Tensor, stage: str = "full") -> dict[str, torch.Tensor]:
        stage_name = str(stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported stage='{stage}'. Use 'stage1', 'stage2', or 'full'.")

        h0 = self.encode_stem(y)
        z16, y16 = self.stage1_branch.forward_features(h0)
        outputs: dict[str, torch.Tensor] = {
            "stem": h0,
            "z16": z16,
            "stage1": y16,
            "coarse_solution": y16,
        }
        if stage_name in ("stage1", "coarse"):
            return outputs

        u16_to_48 = self.adapter_16_to_48(z16.detach())
        stage2_input = torch.cat([h0, u16_to_48], dim=-1)
        z48, y48 = self.stage2_branch.forward_features(stage2_input)
        outputs.update(
            {
                "u16_to_48": u16_to_48,
                "z48": z48,
                "stage2": y48,
                "mid_solution": y48,
            }
        )
        if stage_name == "stage2":
            return outputs

        u48_to_128 = self.adapter_48_to_128(z48.detach())
        stage3_input = torch.cat([h0, u48_to_128], dim=-1)
        z128, y128 = self.stage3_branch.forward_features(stage3_input)
        outputs.update(
            {
                "u48_to_128": u48_to_128,
                "z128": z128,
                "full": y128,
                "stage3": y128,
                "full_solution": y128,
            }
        )
        return outputs

    def forward(self, y: torch.Tensor, stage: str = "full") -> torch.Tensor:
        outputs = self.forward_stages(y, stage=stage)
        stage_name = str(stage).strip().lower()
        if stage_name in ("stage1", "coarse"):
            return outputs["stage1"]
        if stage_name == "stage2":
            return outputs["stage2"]
        return outputs["full"]


class ProgressiveStageLoss(nn.Module):
    """第三代 PMRH 的弱耦合预算损失。"""

    def __init__(
        self,
        *,
        group_spec: ProgressiveGroupSpec,
        coeff_mean: torch.Tensor,
        coeff_std: torch.Tensor,
        stage_weights: Sequence[float] = (1.0, 1.0, 1.0),
        consistency_weight: float = 0.0,
        budget_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if coeff_mean.ndim != 1 or coeff_std.ndim != 1:
            raise ValueError("coeff_mean/coeff_std must be 1D tensors")
        if coeff_mean.shape != coeff_std.shape:
            raise ValueError(f"coeff_mean/coeff_std shape mismatch: {tuple(coeff_mean.shape)} vs {tuple(coeff_std.shape)}")
        if int(coeff_mean.shape[0]) != int(group_spec.total_dim):
            raise ValueError(
                f"coeff stats dim {tuple(coeff_mean.shape)} must match total_dim={group_spec.total_dim}"
            )
        stage_weights_arr = np.asarray(list(stage_weights), dtype=np.float32).reshape(-1)
        if stage_weights_arr.shape[0] != 3:
            raise ValueError(f"stage_weights must have length 3, got {tuple(stage_weights)}")

        self.group_spec = group_spec
        self.register_buffer("coeff_mean", coeff_mean.detach().clone().float())
        self.register_buffer("coeff_std", torch.clamp(coeff_std.detach().clone().float(), min=1e-6))
        self.register_buffer("stage_weights", torch.as_tensor(stage_weights_arr, dtype=torch.float32))
        self.register_buffer(
            "prefix_weights",
            torch.as_tensor((0.5 * float(stage_weights_arr[1]), float(stage_weights_arr[1])), dtype=torch.float32),
        )
        self.consistency_weight = float(consistency_weight)
        self.budget_weight = float(budget_weight)

    def _normalize_prefix(self, x: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        mean = self.coeff_mean[:prefix_dim].view(1, -1)
        std = self.coeff_std[:prefix_dim].view(1, -1)
        return (x - mean) / std

    def _prefix_mse(self, pred: torch.Tensor, target: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        pred_n = self._normalize_prefix(pred[:, :prefix_dim], prefix_dim)
        target_n = self._normalize_prefix(target[:, :prefix_dim], prefix_dim)
        return torch.mean((pred_n - target_n) ** 2)

    def _prefix_consistency(self, pred: torch.Tensor, ref: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        pred_n = self._normalize_prefix(pred[:, :prefix_dim], prefix_dim)
        ref_n = self._normalize_prefix(ref[:, :prefix_dim], prefix_dim)
        return torch.mean((pred_n - ref_n) ** 2)

    def compute_components(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stage_name = str(active_stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported active_stage='{active_stage}'")

        g1 = int(self.group_spec.coarse_dim)
        g12 = int(self.group_spec.stage2_dim)
        g123 = int(self.group_spec.total_dim)

        zero = target.new_zeros(())
        loss_stage1 = self._prefix_mse(outputs["stage1"], target, g1)
        loss_stage2 = zero
        loss_stage3 = zero
        loss_prefix16 = zero
        loss_prefix48 = zero

        total = self.stage_weights[0] * loss_stage1
        if stage_name in ("stage2", "full", "stage3"):
            loss_stage2 = self._prefix_mse(outputs["full"], target, g12)
            loss_stage3 = self._prefix_mse(outputs["full"], target, g123)
            loss_prefix16 = self._prefix_mse(outputs["full"], target, g1)
            loss_prefix48 = loss_stage2
            total = total + self.stage_weights[2] * loss_stage3
            total = total + self.prefix_weights[0] * loss_prefix16 + self.prefix_weights[1] * loss_prefix48

        consistency_penalty = zero
        budget_penalty = zero

        metrics = {
            "loss_total": total.detach(),
            "loss_stage1": loss_stage1.detach(),
            "loss_stage2": loss_stage2.detach(),
            "loss_stage3": loss_stage3.detach(),
            "loss_prefix16": loss_prefix16.detach(),
            "loss_prefix48": loss_prefix48.detach(),
            "loss_consistency": consistency_penalty.detach(),
            "loss_budget": budget_penalty.detach(),
        }
        return total, metrics

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> torch.Tensor:
        total, _ = self.compute_components(outputs, target, active_stage=active_stage)
        return total


class SharedStemBudgetLoss(nn.Module):
    """v4a 的按预算独立归一化前缀损失。"""

    def __init__(
        self,
        *,
        group_spec: ProgressiveGroupSpec,
        coeff_mean: torch.Tensor,
        coeff_std: torch.Tensor,
        stage_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        if coeff_mean.ndim != 1 or coeff_std.ndim != 1:
            raise ValueError("coeff_mean/coeff_std must be 1D tensors")
        if coeff_mean.shape != coeff_std.shape:
            raise ValueError(f"coeff_mean/coeff_std shape mismatch: {tuple(coeff_mean.shape)} vs {tuple(coeff_std.shape)}")
        if int(coeff_mean.shape[0]) != int(group_spec.total_dim):
            raise ValueError(
                f"coeff stats dim {tuple(coeff_mean.shape)} must match total_dim={group_spec.total_dim}"
            )

        stage_weights_arr = np.asarray(list(stage_weights), dtype=np.float32).reshape(-1)
        if stage_weights_arr.shape[0] != 3:
            raise ValueError(f"stage_weights must have length 3, got {tuple(stage_weights)}")

        self.group_spec = group_spec
        self.register_buffer("coeff_mean", coeff_mean.detach().clone().float())
        self.register_buffer("coeff_std", torch.clamp(coeff_std.detach().clone().float(), min=1e-6))
        self.register_buffer("stage_weights", torch.as_tensor(stage_weights_arr, dtype=torch.float32))

    def _normalize_prefix(self, x: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        mean = self.coeff_mean[:prefix_dim].view(1, -1)
        std = self.coeff_std[:prefix_dim].view(1, -1)
        return (x[:, :prefix_dim] - mean) / std

    def _prefix_mse(self, pred: torch.Tensor, target: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        pred_n = self._normalize_prefix(pred, prefix_dim)
        target_n = self._normalize_prefix(target, prefix_dim)
        return torch.mean((pred_n - target_n) ** 2)

    def compute_components(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stage_name = str(active_stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported active_stage='{active_stage}'")

        g1 = int(self.group_spec.coarse_dim)
        g12 = int(self.group_spec.stage2_dim)
        g123 = int(self.group_spec.total_dim)
        zero = target.new_zeros(())
        loss_stage1 = zero
        loss_stage2 = zero
        loss_stage3 = zero

        if stage_name in ("stage1", "coarse"):
            loss_stage1 = self._prefix_mse(outputs["stage1"], target, g1)
            total = self.stage_weights[0] * loss_stage1
        elif stage_name == "stage2":
            loss_stage2 = self._prefix_mse(outputs["stage2"], target, g12)
            total = self.stage_weights[1] * loss_stage2
        else:
            loss_stage3 = self._prefix_mse(outputs["full"], target, g123)
            total = self.stage_weights[2] * loss_stage3

        metrics = {
            "loss_total": total.detach(),
            "loss_stage1": loss_stage1.detach(),
            "loss_stage2": loss_stage2.detach(),
            "loss_stage3": loss_stage3.detach(),
            "loss_consistency": zero.detach(),
            "loss_budget": zero.detach(),
        }
        return total, metrics

    def combine_weighted(self, loss_stage1: torch.Tensor, loss_stage2: torch.Tensor, loss_stage3: torch.Tensor) -> torch.Tensor:
        return (
            self.stage_weights[0] * loss_stage1
            + self.stage_weights[1] * loss_stage2
            + self.stage_weights[2] * loss_stage3
        )

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> torch.Tensor:
        total, _ = self.compute_components(outputs, target, active_stage=active_stage)
        return total


class LatentGuidedBudgetLoss(nn.Module):
    """v4b 的按预算独立前缀损失，附带弱 adapter 幅度正则。"""

    def __init__(
        self,
        *,
        group_spec: ProgressiveGroupSpec,
        coeff_mean: torch.Tensor,
        coeff_std: torch.Tensor,
        stage_weights: Sequence[float] = (1.0, 1.0, 1.0),
        adapter_reg_weights: Sequence[float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        if coeff_mean.ndim != 1 or coeff_std.ndim != 1:
            raise ValueError("coeff_mean/coeff_std must be 1D tensors")
        if coeff_mean.shape != coeff_std.shape:
            raise ValueError(f"coeff_mean/coeff_std shape mismatch: {tuple(coeff_mean.shape)} vs {tuple(coeff_std.shape)}")
        if int(coeff_mean.shape[0]) != int(group_spec.total_dim):
            raise ValueError(
                f"coeff stats dim {tuple(coeff_mean.shape)} must match total_dim={group_spec.total_dim}"
            )

        stage_weights_arr = np.asarray(list(stage_weights), dtype=np.float32).reshape(-1)
        if stage_weights_arr.shape[0] != 3:
            raise ValueError(f"stage_weights must have length 3, got {tuple(stage_weights)}")
        adapter_reg_arr = np.asarray(list(adapter_reg_weights), dtype=np.float32).reshape(-1)
        if adapter_reg_arr.shape[0] != 2:
            raise ValueError(f"adapter_reg_weights must have length 2, got {tuple(adapter_reg_weights)}")

        self.group_spec = group_spec
        self.register_buffer("coeff_mean", coeff_mean.detach().clone().float())
        self.register_buffer("coeff_std", torch.clamp(coeff_std.detach().clone().float(), min=1e-6))
        self.register_buffer("stage_weights", torch.as_tensor(stage_weights_arr, dtype=torch.float32))
        self.register_buffer("adapter_reg_weights", torch.as_tensor(adapter_reg_arr, dtype=torch.float32))

    def _normalize_prefix(self, x: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        mean = self.coeff_mean[:prefix_dim].view(1, -1)
        std = self.coeff_std[:prefix_dim].view(1, -1)
        return (x[:, :prefix_dim] - mean) / std

    def _prefix_mse(self, pred: torch.Tensor, target: torch.Tensor, prefix_dim: int) -> torch.Tensor:
        pred_n = self._normalize_prefix(pred, prefix_dim)
        target_n = self._normalize_prefix(target, prefix_dim)
        return torch.mean((pred_n - target_n) ** 2)

    def _adapter_energy(self, x: torch.Tensor | None) -> torch.Tensor:
        if x is None:
            return self.coeff_mean.new_zeros(())
        return torch.mean(x ** 2)

    def compute_components(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stage_name = str(active_stage).strip().lower()
        if stage_name not in ("stage1", "coarse", "stage2", "full", "stage3"):
            raise ValueError(f"Unsupported active_stage='{active_stage}'")

        g1 = int(self.group_spec.coarse_dim)
        g12 = int(self.group_spec.stage2_dim)
        g123 = int(self.group_spec.total_dim)
        zero = target.new_zeros(())
        loss_stage1 = zero
        loss_stage2 = zero
        loss_stage3 = zero
        loss_adapter_48 = zero
        loss_adapter_128 = zero

        if stage_name in ("stage1", "coarse"):
            loss_stage1 = self._prefix_mse(outputs["stage1"], target, g1)
            total = self.stage_weights[0] * loss_stage1
        elif stage_name == "stage2":
            loss_stage2 = self._prefix_mse(outputs["stage2"], target, g12)
            loss_adapter_48 = self._adapter_energy(outputs.get("u16_to_48"))
            total = self.stage_weights[1] * loss_stage2 + self.adapter_reg_weights[0] * loss_adapter_48
        else:
            loss_stage3 = self._prefix_mse(outputs["full"], target, g123)
            loss_adapter_128 = self._adapter_energy(outputs.get("u48_to_128"))
            total = self.stage_weights[2] * loss_stage3 + self.adapter_reg_weights[1] * loss_adapter_128

        metrics = {
            "loss_total": total.detach(),
            "loss_stage1": loss_stage1.detach(),
            "loss_stage2": loss_stage2.detach(),
            "loss_stage3": loss_stage3.detach(),
            "loss_adapter_48": loss_adapter_48.detach(),
            "loss_adapter_128": loss_adapter_128.detach(),
            "loss_consistency": zero.detach(),
            "loss_budget": zero.detach(),
        }
        return total, metrics

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
        *,
        active_stage: str = "full",
    ) -> torch.Tensor:
        total, _ = self.compute_components(outputs, target, active_stage=active_stage)
        return total


class WeightedCoefficientMSELoss(nn.Module):
    def __init__(self, coeff_weights: torch.Tensor) -> None:
        super().__init__()
        if coeff_weights.ndim != 1:
            raise ValueError(f"coeff_weights must be 1D, got shape {tuple(coeff_weights.shape)}")
        self.register_buffer("coeff_weights", coeff_weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.shape[-1] != int(self.coeff_weights.shape[0]):
            raise ValueError(
                f"Last dim of pred/target must match coeff_weights: "
                f"{tuple(pred.shape)} vs {tuple(self.coeff_weights.shape)}"
            )
        sq_err = (pred - target) ** 2
        weighted_sq_err = sq_err * self.coeff_weights.view(1, -1)
        return weighted_sq_err.sum(dim=-1).mean()


def build_coeff_loss_weights(
    coeff_loss_weights: np.ndarray | None,
    *,
    r_eff: int,
    loss_weighting: str,
    loss_weight_power: float,
) -> np.ndarray | None:
    mode = str(loss_weighting or "none").strip().lower()
    if mode in ("none", "uniform", "off"):
        return None
    if mode != "pod_energy":
        raise ValueError(f"Unsupported loss_weighting='{loss_weighting}'.")
    if coeff_loss_weights is None:
        raise ValueError("coeff_loss_weights is required when loss_weighting='pod_energy'.")

    weights = np.asarray(coeff_loss_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] != int(r_eff):
        raise ValueError(f"coeff_loss_weights length {weights.shape[0]} != r_eff={r_eff}")
    if np.any(weights < 0.0):
        raise ValueError("coeff_loss_weights must be nonnegative.")

    power = float(loss_weight_power)
    if power <= 0.0:
        raise ValueError(f"loss_weight_power must be positive, got {loss_weight_power}")
    if power != 1.0:
        weights = np.power(weights, power).astype(np.float32, copy=False)

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("coeff_loss_weights sum must be positive.")
    return weights.astype(np.float32, copy=False)


def build_mlp_loss(
    *,
    coeff_loss_weights: np.ndarray | None,
    r_eff: int,
    loss_weighting: str,
    loss_weight_power: float,
    device: str | torch.device,
) -> tuple[nn.Module, np.ndarray | None]:
    effective_coeff_weights = build_coeff_loss_weights(
        coeff_loss_weights,
        r_eff=r_eff,
        loss_weighting=loss_weighting,
        loss_weight_power=loss_weight_power,
    )
    if effective_coeff_weights is None:
        return nn.MSELoss(), None

    weight_tensor = torch.from_numpy(effective_coeff_weights.astype(np.float32, copy=False)).to(device)
    return WeightedCoefficientMSELoss(weight_tensor), effective_coeff_weights
