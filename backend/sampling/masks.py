# backend/sampling/masks.py

"""
生成稀疏空间观测的 mask，并在展平向量上应用。
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _resolve_num_obs(num_points: int, mask_rate: float | None, mask_num: int | None) -> int:
    if mask_num is not None:
        num_obs = int(mask_num)
        if num_obs <= 0:
            raise ValueError(f"mask_num must be positive, got {mask_num}")
        return min(num_obs, num_points)

    if mask_rate is None:
        raise ValueError("Either mask_rate or mask_num must be provided.")
    if not (0 < mask_rate <= 1.0):
        raise ValueError(f"mask_rate must be in (0,1], got {mask_rate}")
    return max(1, int(round(num_points * mask_rate)))


def generate_random_mask_hw(
    H: int,
    W: int,
    mask_rate: float | None = None,
    seed: int | None = None,
    mask_num: int | None = None,
) -> np.ndarray:
    """
    在 H×W 网格上生成随机均匀采样的观测 mask。

    参数
    ----
    mask_rate:
        观测比例 (0,1]。若同时给定 mask_rate 与 mask_num，则优先使用 mask_num。
    mask_num:
        观测点个数（以空间网格点计，不含通道）。若给定则直接使用该个数。

    返回
    ----
    mask:
        形状为 [H, W] 的 bool 数组，其中 True 表示被观测。
    """
    num_points = H * W
    num_obs = _resolve_num_obs(num_points, mask_rate, mask_num)

    rng = np.random.RandomState(seed)
    flat_mask = np.zeros(num_points, dtype=bool)
    idx = rng.choice(num_points, size=num_obs, replace=False)
    flat_mask[idx] = True

    return flat_mask.reshape(H, W)


def generate_radial_spiral_mask_hw(
    H: int,
    W: int,
    mask_rate: float | None = None,
    seed: int | None = None,
    mask_num: int | None = None,
    *,
    max_radius_frac: float = 0.875,
) -> np.ndarray:
    """
    在 H×W 网格上生成“径向螺旋式”观测 mask。

    规则：
    - 采样半径从中心开始均匀增加，最大到 max_radius_frac * min(H, W) / 2
    - 每个半径对应一个独立的随机角度，使采样点不会全部落在同一射线上
    - 若离散化后发生重复点，则在相近半径处回退到最近的未使用网格点
    """
    num_points = H * W
    num_obs = _resolve_num_obs(num_points, mask_rate, mask_num)

    if not (0.0 <= float(max_radius_frac) <= 1.0):
        raise ValueError(f"max_radius_frac must be in [0,1], got {max_radius_frac}")

    rng = np.random.RandomState(seed)
    cy = 0.5 * (H - 1)
    cx = 0.5 * (W - 1)
    radius_limit = float(min(H, W)) * 0.5 * float(max_radius_frac)
    radii = np.linspace(0.0, radius_limit, num_obs, endpoint=True, dtype=np.float64)

    used: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []

    yy, xx = np.indices((H, W), dtype=np.float64)
    radial_dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    for radius in radii:
        point: tuple[int, int] | None = None
        for _ in range(64):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            row = int(round(cy + radius * np.sin(theta)))
            col = int(round(cx + radius * np.cos(theta)))
            row = int(np.clip(row, 0, H - 1))
            col = int(np.clip(col, 0, W - 1))
            candidate = (row, col)
            if candidate not in used:
                point = candidate
                break

        if point is None:
            free_mask = np.ones((H, W), dtype=bool)
            for row, col in used:
                free_mask[row, col] = False

            free_rows, free_cols = np.where(free_mask)
            if free_rows.size == 0:
                break

            free_dist = radial_dist[free_rows, free_cols]
            order = np.argsort(np.abs(free_dist - radius), kind="stable")
            best_idx = int(order[0])
            point = (int(free_rows[best_idx]), int(free_cols[best_idx]))

        used.add(point)
        coords.append(point)

    if len(coords) != num_obs:
        raise RuntimeError(f"Failed to place {num_obs} unique spiral samples on grid {H}x{W}; got {len(coords)}")

    mask = np.zeros((H, W), dtype=bool)
    for row, col in coords:
        mask[row, col] = True
    return mask


def _normalize_probability_map(prob: np.ndarray, *, fallback_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float64)
    if arr.shape != fallback_shape:
        raise ValueError(f"Probability map shape mismatch: expected {fallback_shape}, got {arr.shape}")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(fallback_shape, 1.0 / float(np.prod(fallback_shape)), dtype=np.float64)
    return arr / total


def _normalize_vector(prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float64).reshape(-1)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float64)
    return arr / total


def _fraction_to_index(length: int, frac: float, *, lower_bound: int, upper_bound: int) -> int:
    value = int(round(float(length) * float(frac)))
    return int(np.clip(value, lower_bound, upper_bound))


def _make_center_row_probabilities(
    H: int,
    *,
    center_bias_strength: float,
    center_bias_sigma_frac: float,
    flank_weight: float = 0.0,
    flank_offset_frac: float = 0.12,
    flank_sigma_frac: float | None = None,
) -> np.ndarray:
    rows = np.arange(H, dtype=np.float64)
    center = 0.5 * float(H - 1)
    sigma = max(1e-6, float(center_bias_sigma_frac) * float(H))
    center_gaussian = np.exp(-0.5 * ((rows - center) / sigma) ** 2)
    center_gaussian = center_gaussian ** max(1e-6, float(center_bias_strength))

    flank_mix = float(np.clip(flank_weight, 0.0, 1.0))
    if flank_mix <= 0.0:
        return _normalize_vector(center_gaussian)

    flank_sigma = sigma if flank_sigma_frac is None else max(1e-6, float(flank_sigma_frac) * float(H))
    offset = float(flank_offset_frac) * float(H)
    upper_center = center - offset
    lower_center = center + offset
    upper = np.exp(-0.5 * ((rows - upper_center) / flank_sigma) ** 2)
    lower = np.exp(-0.5 * ((rows - lower_center) / flank_sigma) ** 2)
    flank_profile = 0.5 * (upper + lower)
    mixed = (1.0 - flank_mix) * center_gaussian + flank_mix * flank_profile
    return _normalize_vector(mixed)


def _gradient_magnitude_hw(field_hw: np.ndarray) -> np.ndarray:
    field = np.asarray(field_hw, dtype=np.float64)
    grad_y, grad_x = np.gradient(field)
    return np.sqrt(grad_y ** 2 + grad_x ** 2, dtype=np.float64)


def _reduce_channel_map(map_hwc: np.ndarray, channel_reduce: str) -> np.ndarray:
    reduce_name = str(channel_reduce or "l2").strip().lower()
    if map_hwc.ndim != 3:
        raise ValueError(f"Expected [H,W,C] channel map, got {map_hwc.shape}")

    if reduce_name == "l2":
        return np.sqrt(np.sum(np.square(map_hwc), axis=2, dtype=np.float64))
    if reduce_name in ("mean", "avg"):
        return np.mean(map_hwc, axis=2, dtype=np.float64)
    if reduce_name == "max":
        return np.max(map_hwc, axis=2)
    if reduce_name.startswith("channel:"):
        channel_idx = int(reduce_name.split(":", 1)[1])
        if not (0 <= channel_idx < map_hwc.shape[2]):
            raise ValueError(f"Channel index out of range for {map_hwc.shape}: {channel_idx}")
        return map_hwc[:, :, channel_idx]

    raise ValueError(f"Unsupported channel_reduce: {channel_reduce!r}")


def build_structure_importance_map(
    *,
    H: int,
    W: int,
    C: int,
    data_thwc: np.ndarray | None = None,
    Ur: np.ndarray | None = None,
    mode_energy_weights: np.ndarray | None = None,
    source: str = "temporal_variance",
    channel_reduce: str = "l2",
    pod_top_k: int = 16,
    gradient_mix: float = 0.0,
    importance_power: float = 1.0,
    hotspot_center: tuple[float, float] | None = None,
    hotspot_sigma: tuple[float, float] | None = None,
    hotspot_weight: float = 0.0,
) -> np.ndarray:
    source_name = str(source or "temporal_variance").strip().lower()

    if source_name in ("temporal_variance", "variance", "var"):
        if data_thwc is None:
            raise ValueError("data_thwc is required for temporal_variance importance")
        data = np.asarray(data_thwc, dtype=np.float32)
        if data.ndim != 4 or data.shape[1:] != (H, W, C):
            raise ValueError(f"Expected data_thwc shape [T,{H},{W},{C}], got {data.shape}")
        var_hwc = np.var(data, axis=0, dtype=np.float64)
        importance_hw = _reduce_channel_map(var_hwc, channel_reduce)
    elif source_name in ("pod_energy", "pod", "pod_modes"):
        if Ur is None:
            raise ValueError("Ur is required for pod_energy importance")
        basis = np.asarray(Ur, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != H * W * C:
            raise ValueError(f"Expected Ur shape [{H * W * C}, r], got {basis.shape}")

        top_k = max(1, min(int(pod_top_k), basis.shape[1]))
        basis_hwck = basis[:, :top_k].reshape(H, W, C, top_k)
        if mode_energy_weights is None:
            weights = np.ones(top_k, dtype=np.float64)
        else:
            full_weights = np.asarray(mode_energy_weights, dtype=np.float64).reshape(-1)
            if full_weights.shape[0] < top_k:
                raise ValueError(f"mode_energy_weights length {full_weights.shape[0]} < pod_top_k={top_k}")
            weights = full_weights[:top_k]
        weighted = np.square(np.abs(basis_hwck)) * weights[None, None, None, :]
        importance_hw = weighted.sum(axis=(2, 3), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported structure importance source: {source!r}")

    importance_hw = np.clip(np.asarray(importance_hw, dtype=np.float64), 0.0, None)

    grad_mix = float(np.clip(gradient_mix, 0.0, 1.0))
    if grad_mix > 0.0:
        grad_hw = _gradient_magnitude_hw(importance_hw)
        grad_hw = _normalize_probability_map(grad_hw, fallback_shape=(H, W))
        base_hw = _normalize_probability_map(importance_hw, fallback_shape=(H, W))
        importance_hw = (1.0 - grad_mix) * base_hw + grad_mix * grad_hw

    power = max(1e-6, float(importance_power))
    if abs(power - 1.0) > 1e-12:
        importance_hw = np.power(np.clip(importance_hw, 0.0, None), power, dtype=np.float64)

    if hotspot_center is not None and float(hotspot_weight) > 0.0:
        cy, cx = float(hotspot_center[0]), float(hotspot_center[1])
        if hotspot_sigma is None:
            hotspot_sigma = (max(1.0, 0.08 * H), max(1.0, 0.08 * W))
        sy = max(1e-6, float(hotspot_sigma[0]))
        sx = max(1e-6, float(hotspot_sigma[1]))
        yy, xx = np.indices((H, W), dtype=np.float64)
        hotspot = np.exp(-0.5 * (((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
        importance_hw = importance_hw + float(hotspot_weight) * hotspot

    return _normalize_probability_map(importance_hw, fallback_shape=(H, W))


def _counts_from_ratios(total_count: int, ratios: list[float]) -> list[int]:
    if total_count <= 0:
        return [0 for _ in ratios]

    weight = np.asarray(ratios, dtype=np.float64)
    weight = np.clip(weight, 0.0, None)
    if float(weight.sum()) <= 0.0:
        weight = np.full_like(weight, 1.0 / float(weight.size))
    else:
        weight = weight / float(weight.sum())

    exact = weight * float(total_count)
    counts = np.floor(exact).astype(int)
    remainder = int(total_count - int(counts.sum()))
    if remainder > 0:
        order = np.argsort(-(exact - counts), kind="stable")
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return counts.tolist()


def _sample_without_replacement(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    *,
    count: int,
    rng: np.random.RandomState,
    used: set[tuple[int, int]],
    coords: list[tuple[int, int]],
    min_distance: float,
) -> list[tuple[int, int]]:
    if count <= 0 or candidates.size == 0:
        return []

    probs = _normalize_vector(probabilities)
    available = np.ones(candidates.shape[0], dtype=bool)
    picked: list[tuple[int, int]] = []
    min_dist_sq = float(min_distance) ** 2

    def _distance_ok(point: tuple[int, int]) -> bool:
        if min_dist_sq <= 0.0:
            return True
        py, px = point
        for qy, qx in coords:
            dy = float(py - qy)
            dx = float(px - qx)
            if dy * dy + dx * dx < min_dist_sq:
                return False
        return True

    while len(picked) < count and bool(np.any(available)):
        active = np.flatnonzero(available)
        choice_p = _normalize_vector(probs[active])
        pick_pos = int(rng.choice(active, p=choice_p))
        available[pick_pos] = False
        point = (int(candidates[pick_pos, 0]), int(candidates[pick_pos, 1]))
        if point in used or not _distance_ok(point):
            continue
        used.add(point)
        coords.append(point)
        picked.append(point)

    while len(picked) < count and bool(np.any(available)):
        active = np.flatnonzero(available)
        choice_p = _normalize_vector(probs[active])
        pick_pos = int(rng.choice(active, p=choice_p))
        available[pick_pos] = False
        point = (int(candidates[pick_pos, 0]), int(candidates[pick_pos, 1]))
        if point in used:
            continue
        used.add(point)
        coords.append(point)
        picked.append(point)

    return picked


def _skeleton_points(
    H: int,
    W: int,
    *,
    skeleton_count: int,
    line_offsets_frac: list[float],
) -> list[tuple[int, int]]:
    if skeleton_count <= 0:
        return []

    center_row = 0.5 * float(H - 1)
    row_candidates: list[int] = []
    for frac in line_offsets_frac:
        row = int(round(center_row + float(frac) * float(H)))
        row = int(np.clip(row, 0, H - 1))
        if row not in row_candidates:
            row_candidates.append(row)
    if not row_candidates:
        row_candidates = [int(round(center_row))]

    x_count = max(1, int(np.ceil(float(skeleton_count) / float(len(row_candidates)))))
    x_positions = np.unique(np.round(np.linspace(0, W - 1, num=x_count)).astype(int))

    all_points: list[tuple[int, int]] = []
    for row in row_candidates:
        for col in x_positions.tolist():
            all_points.append((int(row), int(col)))

    if len(all_points) <= skeleton_count:
        return all_points

    pick_idx = np.round(np.linspace(0, len(all_points) - 1, num=skeleton_count)).astype(int)
    return [all_points[int(idx)] for idx in pick_idx.tolist()]


def generate_structure_aware_mask_hw(
    H: int,
    W: int,
    mask_rate: float | None = None,
    seed: int | None = None,
    mask_num: int | None = None,
    *,
    num_templates: int = 1,
    template_index: int = 0,
    region_bounds_frac: tuple[float, float] = (0.15, 0.5),
    region_ratios: dict[str, float] | None = None,
    center_bias_strength: float = 1.75,
    center_bias_sigma_frac: float = 0.12,
    flank_weight: float = 0.0,
    flank_offset_frac: float = 0.12,
    flank_sigma_frac: float | None = None,
    importance_map: np.ndarray | None = None,
    importance_mix: float = 0.65,
    skeleton_points: int = 12,
    skeleton_line_offsets_frac: tuple[float, ...] = (0.0, -0.1, 0.1),
    min_distance: float = 0.0,
) -> np.ndarray:
    num_points = H * W
    num_obs = _resolve_num_obs(num_points, mask_rate, mask_num)

    template_count = max(1, int(num_templates))
    template_id = int(template_index) % template_count
    base_seed = 0 if seed is None else int(seed)
    rng = np.random.RandomState(base_seed + 1009 * template_id)

    left_frac, middle_frac = float(region_bounds_frac[0]), float(region_bounds_frac[1])
    if not (0.0 < left_frac < middle_frac < 1.0):
        raise ValueError(f"region_bounds_frac must satisfy 0 < left < middle < 1, got {region_bounds_frac}")

    left_end = _fraction_to_index(W, left_frac, lower_bound=1, upper_bound=max(1, W - 2))
    middle_end = _fraction_to_index(W, middle_frac, lower_bound=left_end + 1, upper_bound=max(left_end + 1, W - 1))

    ratios = dict(region_ratios or {})
    left_ratio = float(ratios.get("left", 0.35))
    middle_ratio = float(ratios.get("middle", 0.35))
    right_ratio = float(ratios.get("right", 0.20))
    global_ratio = float(ratios.get("global_random", 0.10))

    row_prob = _make_center_row_probabilities(
        H,
        center_bias_strength=center_bias_strength,
        center_bias_sigma_frac=center_bias_sigma_frac,
        flank_weight=flank_weight,
        flank_offset_frac=flank_offset_frac,
        flank_sigma_frac=flank_sigma_frac,
    )
    if importance_map is None:
        importance_hw = np.full((H, W), 1.0 / float(H * W), dtype=np.float64)
    else:
        importance_hw = _normalize_probability_map(np.asarray(importance_map), fallback_shape=(H, W))
    mix = float(np.clip(importance_mix, 0.0, 1.0))

    max_skeleton = max(0, min(int(skeleton_points), max(1, num_obs // 3) if num_obs >= 3 else num_obs))
    skeleton = _skeleton_points(
        H,
        W,
        skeleton_count=max_skeleton,
        line_offsets_frac=[float(x) for x in skeleton_line_offsets_frac],
    )

    used: set[tuple[int, int]] = set()
    coords: list[tuple[int, int]] = []
    for point in skeleton:
        if point not in used:
            used.add(point)
            coords.append(point)

    remaining = num_obs - len(coords)
    region_counts = _counts_from_ratios(remaining, [left_ratio, middle_ratio, right_ratio, global_ratio])
    region_slices = [
        slice(0, left_end),
        slice(left_end, middle_end),
        slice(middle_end, W),
    ]

    for region_slice, count in zip(region_slices, region_counts[:3]):
        if count <= 0:
            continue
        cols = np.arange(region_slice.start, region_slice.stop, dtype=int)
        rows = np.arange(H, dtype=int)
        yy, xx = np.meshgrid(rows, cols, indexing="ij")
        candidates = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)

        base_prob = np.repeat(row_prob[:, None], cols.size, axis=1)
        base_prob = _normalize_probability_map(base_prob, fallback_shape=base_prob.shape)
        importance_region = importance_hw[:, region_slice]
        mixed_prob = _normalize_probability_map((1.0 - mix) * base_prob + mix * importance_region, fallback_shape=base_prob.shape)

        _sample_without_replacement(
            candidates,
            mixed_prob.reshape(-1),
            count=count,
            rng=rng,
            used=used,
            coords=coords,
            min_distance=min_distance,
        )

    global_count = int(region_counts[3])
    if global_count > 0:
        yy, xx = np.indices((H, W), dtype=int)
        candidates = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
        uniform = np.full((H, W), 1.0 / float(H * W), dtype=np.float64)
        global_prob = _normalize_probability_map((1.0 - mix) * uniform + mix * importance_hw, fallback_shape=(H, W))
        _sample_without_replacement(
            candidates,
            global_prob.reshape(-1),
            count=global_count,
            rng=rng,
            used=used,
            coords=coords,
            min_distance=min_distance,
        )

    if len(coords) < num_obs:
        yy, xx = np.indices((H, W), dtype=int)
        candidates = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
        fallback_prob = _normalize_probability_map(importance_hw, fallback_shape=(H, W)).reshape(-1)
        _sample_without_replacement(
            candidates,
            fallback_prob,
            count=num_obs - len(coords),
            rng=rng,
            used=used,
            coords=coords,
            min_distance=0.0,
        )

    if len(coords) != num_obs:
        raise RuntimeError(f"Failed to place {num_obs} unique structure-aware samples on grid {H}x{W}; got {len(coords)}")

    mask = np.zeros((H, W), dtype=bool)
    for row, col in coords:
        mask[row, col] = True
    return mask


def generate_observation_mask_hw(
    H: int,
    W: int,
    mask_rate: float | None = None,
    seed: int | None = None,
    mask_num: int | None = None,
    *,
    strategy: str = "random",
    strategy_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    strategy_name = str(strategy or "random").strip().lower()
    kwargs = dict(strategy_kwargs or {})

    if strategy_name == "random":
        return generate_random_mask_hw(H, W, mask_rate=mask_rate, seed=seed, mask_num=mask_num)
    if strategy_name in ("radial_spiral", "spiral"):
        return generate_radial_spiral_mask_hw(
            H,
            W,
            mask_rate=mask_rate,
            seed=seed,
            mask_num=mask_num,
            max_radius_frac=float(kwargs.get("max_radius_frac", 0.875)),
        )
    if strategy_name in ("cylinder_structure_aware", "structure_aware", "region_importance"):
        return generate_structure_aware_mask_hw(
            H,
            W,
            mask_rate=mask_rate,
            seed=seed,
            mask_num=mask_num,
            num_templates=int(kwargs.get("num_templates", 1)),
            template_index=int(kwargs.get("template_index", 0)),
            region_bounds_frac=tuple(kwargs.get("region_bounds_frac", (0.15, 0.5))),
            region_ratios=dict(kwargs.get("region_ratios", {}) or {}),
            center_bias_strength=float(kwargs.get("center_bias_strength", 1.75)),
            center_bias_sigma_frac=float(kwargs.get("center_bias_sigma_frac", 0.12)),
            flank_weight=float(kwargs.get("flank_weight", 0.0)),
            flank_offset_frac=float(kwargs.get("flank_offset_frac", 0.12)),
            flank_sigma_frac=(None if kwargs.get("flank_sigma_frac", None) is None else float(kwargs.get("flank_sigma_frac"))),
            importance_map=kwargs.get("importance_map", None),
            importance_mix=float(kwargs.get("importance_mix", 0.65)),
            skeleton_points=int(kwargs.get("skeleton_points", 12)),
            skeleton_line_offsets_frac=tuple(kwargs.get("skeleton_line_offsets_frac", (0.0, -0.1, 0.1))),
            min_distance=float(kwargs.get("min_distance", 0.0)),
        )

    raise ValueError(f"Unsupported observation mask strategy: {strategy!r}")


def flatten_mask(mask_hw: np.ndarray, C: int) -> np.ndarray:
    """
    将 H×W 的空间 mask 扩展到包含通道维度后展平为长度 D 的向量。

    例如：
    - 输入 mask_hw 形状 [H,W]
    - 输出 mask_flat 形状 [H*W*C]
    """
    mask_hw = np.asarray(mask_hw, dtype=bool)
    if mask_hw.ndim != 2:
        raise ValueError(f"mask_hw must be 2D [H,W], got {mask_hw.shape}")

    H, W = mask_hw.shape
    # [H,W] -> [H,W,1] -> [H,W,C]
    mask_hwc = np.repeat(mask_hw[:, :, None], C, axis=2)
    mask_flat = mask_hwc.reshape(-1)  # [H*W*C]

    return mask_flat


def apply_mask_flat(
    x_flat: np.ndarray,
    mask_flat: np.ndarray,
) -> np.ndarray:
    """
    在展平向量上应用 mask，只保留被观测的元素。

    参数
    ----
    x_flat:
        形状为 [D] 或 [N,D] 的数组。
    mask_flat:
        形状为 [D] 的 bool 数组。

    返回
    ----
    y:
        观测值向量：
        - 若输入为 [D]，则输出为 [M]
        - 若输入为 [N,D]，则输出为 [N,M]
        其中 M 为 mask 中 True 的个数。
    """
    x = np.asarray(x_flat)
    mask = np.asarray(mask_flat, dtype=bool)

    if x.ndim == 1:
        if x.shape[0] != mask.shape[0]:
            raise ValueError(f"Dimension mismatch: x[{x.shape}] vs mask[{mask.shape}]")
        return x[mask]
    elif x.ndim == 2:
        if x.shape[1] != mask.shape[0]:
            raise ValueError(f"Dimension mismatch: x[{x.shape}] vs mask[{mask.shape}]")
        return x[:, mask]
    else:
        raise ValueError(f"x_flat must be 1D or 2D, got {x.shape}")
