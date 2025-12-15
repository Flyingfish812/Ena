# backend/config/yaml_io.py

"""
实验配置的 YAML 读写工具。

约定 YAML 结构大致如下：

data:
  nc_path: data/cylinder2d.nc
  var_keys: [u, v]

pod:
  r: 128
  center: true
  save_dir: artifacts/pod_r128

eval:
  mask_rates: [0.01, 0.02, 0.05, 0.1]
  noise_sigmas: [0.0, 0.01, 0.02]
  pod_bands:
    L: [0, 16]
    M: [16, 64]
    H: [64, 128]
  save_dir: artifacts/eval

train:
  mask_rate: 0.02
  noise_sigma: 0.01
  hidden_dims: [256, 256]
  lr: 0.001
  batch_size: 64
  max_epochs: 50
  device: cuda
  save_dir: artifacts/nn
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml  # 需要 pip install pyyaml

from .schemas import DataConfig, PodConfig, EvalConfig, TrainConfig


def _to_serializable(obj: Any) -> Any:
    """
    将 dataclass 字典中的 Path 等对象转为 YAML 友好的类型。
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_experiment_yaml(
    path: str | Path,
    data_cfg: DataConfig,
    pod_cfg: PodConfig,
    eval_cfg: EvalConfig,
    train_cfg: TrainConfig | None = None,
) -> None:
    """
    将当前的一整套实验配置写入 YAML 文件。

    - path: YAML 文件路径
    - train_cfg 可以为 None（例如只做线性基线）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = _to_serializable(asdict(data_cfg))
    pod_dict = _to_serializable(asdict(pod_cfg))
    eval_dict = _to_serializable(asdict(eval_cfg))
    train_dict = _to_serializable(asdict(train_cfg)) if train_cfg is not None else None

    config: Dict[str, Any] = {
        "data": data_dict,
        "pod": pod_dict,
        "eval": eval_dict,
    }
    if train_dict is not None:
        config["train"] = train_dict

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def load_experiment_yaml(
    path: str | Path,
) -> Tuple[DataConfig, PodConfig, EvalConfig, TrainConfig | None]:
    """
    从 YAML 文件中恢复 DataConfig / PodConfig / EvalConfig / TrainConfig。

    - 若 YAML 中缺少 train 字段，则返回的 TrainConfig 为 None。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # data
    data_raw: Dict[str, Any] = config.get("data", {})
    nc_path = Path(data_raw.get("nc_path", "data/cylinder2d.nc"))
    var_keys = tuple(data_raw.get("var_keys", ("u", "v")))
    cache_dir = data_raw.get("cache_dir", None)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)

    data_cfg = DataConfig(
        nc_path=nc_path,
        var_keys=var_keys,
        cache_dir=cache_dir,
    )

    # pod
    pod_raw: Dict[str, Any] = config.get("pod", {})
    pod_save_dir = Path(pod_raw.get("save_dir", "artifacts/pod"))
    pod_cfg = PodConfig(
        r=int(pod_raw.get("r", 128)),
        center=bool(pod_raw.get("center", True)),
        save_dir=pod_save_dir,
    )

    # eval
    eval_raw: Dict[str, Any] = config.get("eval", {})
    mask_rates = list(eval_raw.get("mask_rates", [0.01, 0.02, 0.05, 0.10]))
    noise_sigmas = list(eval_raw.get("noise_sigmas", [0.0, 0.01, 0.02]))

    pod_bands_raw = eval_raw.get("pod_bands", None)
    if pod_bands_raw is None:
        pod_bands: Dict[str, tuple[int, int]] = {
            "L": (0, 16),
            "M": (16, 64),
            "H": (64, 128),
        }
    else:
        pod_bands = {
            name: (int(v[0]), int(v[1]))
            for name, v in pod_bands_raw.items()
        }
    centered_pod = bool(eval_raw.get("centered_pod", True))

    eval_save_dir = Path(eval_raw.get("save_dir", "artifacts/eval"))
    eval_cfg = EvalConfig(
        mask_rates=mask_rates,
        noise_sigmas=noise_sigmas,
        pod_bands=pod_bands,
        centered_pod=centered_pod,
        save_dir=eval_save_dir,
    )

    # train（可选）
    train_raw: Dict[str, Any] | None = config.get("train", None)
    if train_raw is None:
        train_cfg = None
    else:
        train_save_dir = Path(train_raw.get("save_dir", "artifacts/nn"))
        hidden_dims = tuple(train_raw.get("hidden_dims", (256, 256)))
        train_cfg = TrainConfig(
            mask_rate=float(train_raw.get("mask_rate", 0.02)),
            noise_sigma=float(train_raw.get("noise_sigma", 0.01)),
            hidden_dims=hidden_dims,
            lr=float(train_raw.get("lr", 1e-3)),
            batch_size=int(train_raw.get("batch_size", 64)),
            max_epochs=int(train_raw.get("max_epochs", 50)),
            device=str(train_raw.get("device", "cuda")),
            save_dir=train_save_dir,
        )

    return data_cfg, pod_cfg, eval_cfg, train_cfg
