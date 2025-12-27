# backend/pipelines/eval/context.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from backend.config.yaml_io import load_experiment_yaml

from backend.pipelines.eval.utils import (
    entry_name,
    parse_flat_name,
    pick_config_yaml,
    pick_l2_root,
    load_npz,
    read_json,
    ensure_dir_path,
)


@dataclass
class EvalPaths:
    exp_dir: Path
    cfg_path: Path
    l2_root: Path
    l3_root: Path
    l4_root: Path


@dataclass
class EvalContext:
    yaml_path: Optional[Path] = None
    experiment_name: Optional[str] = None
    save_root: Path = Path("artifacts/experiments")
    exp_dir: Optional[Path] = None

    model_types: Optional[Tuple[str, ...]] = None

    paths: Optional[EvalPaths] = None
    data_cfg: Any = None
    pod_cfg: Any = None
    eval_cfg: Any = None
    train_cfg: Any = None

    _l2_meta: Optional[Dict[str, Any]] = None
    _l3_meta: Optional[Dict[str, Any]] = None
    _l3_index: Optional[Dict[str, Any]] = None

    _l2_filemap: Dict[Tuple[str, float, float], Path] = field(default_factory=dict)
    _l3_filemap: Dict[Tuple[str, float, float], Path] = field(default_factory=dict)

    def resolve(self, *, verbose: bool = True) -> "EvalContext":
        # Decide exp_dir
        if self.exp_dir is None:
            if self.yaml_path is None:
                raise ValueError("Either exp_dir or yaml_path must be provided.")
            if self.experiment_name is None:
                self.experiment_name = Path(self.yaml_path).stem
            self.exp_dir = Path(self.save_root) / str(self.experiment_name)

        self.exp_dir = Path(self.exp_dir)
        ensure_dir_path(self.exp_dir)

        cfg_path = pick_config_yaml(self.exp_dir)
        self.data_cfg, self.pod_cfg, self.eval_cfg, self.train_cfg = load_experiment_yaml(cfg_path)

        l2_root = pick_l2_root(self.exp_dir)
        l3_root = self.exp_dir / "L3_fft"  # may not exist if Fourier disabled
        l4_root = ensure_dir_path(self.exp_dir / "L4_eval")

        self.paths = EvalPaths(
            exp_dir=self.exp_dir,
            cfg_path=cfg_path,
            l2_root=l2_root,
            l3_root=l3_root,
            l4_root=l4_root,
        )

        # Infer model_types if not provided
        if self.model_types is None:
            mt = None
            meta_path = l2_root / "meta.json"
            if meta_path.exists():
                try:
                    m = read_json(meta_path)
                    mt = m.get("model_types", None)
                except Exception:
                    mt = None
            if mt is None:
                self.model_types = ("linear", "mlp") if (self.train_cfg is not None) else ("linear",)
            else:
                self.model_types = tuple(str(x) for x in mt)

        if verbose:
            print("=== [L4] EvalContext resolved ===")
            print(f"- exp_dir: {self.paths.exp_dir}")
            print(f"- cfg    : {self.paths.cfg_path}")
            print(f"- L2     : {self.paths.l2_root}")
            print(f"- L3     : {self.paths.l3_root} (exists={self.paths.l3_root.exists()})")
            print(f"- L4     : {self.paths.l4_root}")
            print(f"- model_types: {list(self.model_types)}")

        self._build_l2_filemap()
        if self.paths.l3_root.exists():
            self._build_l3_filemap()

        return self

    # ---------- inventory / indexing ----------
    def iter_cfgs(self) -> Iterable[Tuple[float, float]]:
        mrs = [float(x) for x in getattr(self.eval_cfg, "mask_rates", [])]
        nss = [float(x) for x in getattr(self.eval_cfg, "noise_sigmas", [])]
        for p in mrs:
            for s in nss:
                yield (p, s)

    def _build_l2_filemap(self) -> None:
        assert self.paths is not None
        l2_root = self.paths.l2_root
        self._l2_filemap.clear()

        npz_paths = sorted(list(l2_root.glob("*.npz")))
        if len(npz_paths) == 0:
            npz_paths = sorted(list(l2_root.rglob("*.npz")))

        for p in npz_paths:
            mt, mr, ns = parse_flat_name(p.stem)
            if mt is None:
                continue
            key = (mt, float(mr), float(ns))
            if key not in self._l2_filemap or len(str(p)) < len(str(self._l2_filemap[key])):
                self._l2_filemap[key] = p

    def _build_l3_filemap(self) -> None:
        assert self.paths is not None
        l3_root = self.paths.l3_root
        self._l3_filemap.clear()

        npz_paths = sorted(list(l3_root.glob("*.npz")))
        for p in npz_paths:
            mt, mr, ns = parse_flat_name(p.stem)
            if mt is None:
                continue
            self._l3_filemap[(mt, float(mr), float(ns))] = p

    # ---------- meta/index ----------
    def l2_meta(self) -> Dict[str, Any]:
        if self._l2_meta is not None:
            return self._l2_meta
        assert self.paths is not None
        meta_path = self.paths.l2_root / "meta.json"
        self._l2_meta = read_json(meta_path) if meta_path.exists() else {}
        return self._l2_meta

    def l3_meta(self) -> Dict[str, Any]:
        if self._l3_meta is not None:
            return self._l3_meta
        assert self.paths is not None
        meta_path = self.paths.l3_root / "meta.json"
        self._l3_meta = read_json(meta_path) if meta_path.exists() else {}
        return self._l3_meta

    def l3_index(self) -> Dict[str, Any]:
        if self._l3_index is not None:
            return self._l3_index
        assert self.paths is not None
        idx_path = self.paths.l3_root / "index.json"
        self._l3_index = read_json(idx_path) if idx_path.exists() else {}
        return self._l3_index

    # ---------- L2 / L3 access ----------
    def get_l2_path(self, model_type: str, mask_rate: float, noise_sigma: float) -> Path:
        key = (str(model_type), float(mask_rate), float(noise_sigma))
        if key in self._l2_filemap:
            return self._l2_filemap[key]
        assert self.paths is not None
        p = self.paths.l2_root / entry_name(model_type, mask_rate, noise_sigma)
        if p.exists():
            return p
        raise FileNotFoundError(f"L2 entry not found for {key}. Tried: {p}")

    def get_l3_path(self, model_type: str, mask_rate: float, noise_sigma: float) -> Path:
        key = (str(model_type), float(mask_rate), float(noise_sigma))
        if key in self._l3_filemap:
            return self._l3_filemap[key]
        assert self.paths is not None
        p = self.paths.l3_root / entry_name(model_type, mask_rate, noise_sigma)
        if p.exists():
            return p
        raise FileNotFoundError(f"L3 entry not found for {key}. Tried: {p}")

    def load_l2(self, model_type: str, mask_rate: float, noise_sigma: float) -> Dict[str, Any]:
        return load_npz(self.get_l2_path(model_type, mask_rate, noise_sigma), allow_pickle=False)

    def load_l3(self, model_type: str, mask_rate: float, noise_sigma: float) -> Dict[str, Any]:
        return load_npz(self.get_l3_path(model_type, mask_rate, noise_sigma), allow_pickle=False)
