## Ena Project — Copilot 指南

本文件为 AI 编码代理（Copilot / assistant）快速上手本仓库的精简说明，包含结构概览、重要入口、运行/数据约定与常见模式示例。

**仓库总体**: 本项目以 POD（Proper Orthogonal Decomposition）为核心，分层产生实验产物：
- **L1 (POD)**: 在 `artifacts/pod` 下保存 `Ur.npy`, `mean_flat.npy`, `pod_meta.json` 等（见 `backend/pod/compute.py`）。
- **L2 (raw predictions)**: 由 `backend/eval/rebuild.py` 产生，保存为 flat `.npz`（文件名模式见下）。
- **L3 (FFT / pre-analysis)**: 在 `artifacts/experiments/<exp>/L3_fft` 保存 Fourier 相关分析产物。

**主要目录 & 核心文件**:
- `backend/pipeline.py`: 高层一键入口（`run_experiment_from_yaml`、`compute_level2_rebuild`、`compute_level3_fft_from_level2`、检查函数 `check_level123_artifacts_ready`）。
- `backend/config/schemas.py`、`backend/config/yaml_io.py`: YAML → dataclass 的配置 schema 与读写（实验配置请参考 `configs/` 下示例）。
- `backend/pod/compute.py`: POD 构建（`build_pod`），生成 L1 产物与 scale/basis 表。
- `backend/eval/rebuild.py`: 产生 Level-2 原始预测（`run_rebuild_sweep`）并定义 L2 NPZ schema。
- `backend/pod/project.py`: POD 投影/重建工具（`project_to_pod` / `reconstruct_from_pod`）。
- `backend/models/train_mlp.py`、`backend/models/mlp.py`: MLP 训练与模型（MLP 使用 PyTorch）。
- `backend/dataio/nc_loader.py`: netCDF 数据读取；默认数据文件 `data/cylinder2d.nc`。

**重要约定（必须遵守）**:
- **YAML → dataclasses**: Use `backend/config/yaml_io.py::load_experiment_yaml` to get `DataConfig, PodConfig, EvalConfig, TrainConfig` instances — functions expect these dataclasses, not raw dicts.
- **Artifact levels**: code heavily relies on L1/L2/L3 separation. Don’t bypass L2 when L3 expects its layout.
- **L2 `.npz` schema**: `backend/eval/rebuild.py` 定义 `L2_NPZ_KEYS`。关键 keys: `A_hat_all`, `mask_flat`, `mask_rate`, `noise_sigma`, `centered_pod`, `model_type` 等。保持这些字段名和形状兼容性。
- **L2 filename encoding**: filenames follow `<model>_p{p_code:04d}_s{s_code:04d}.npz`，`p_code`/`s_code` = round(value * 10000)。参见 `rebuild.py::_entry_filename` 与 `entry_naming` metadata。
- **I/O formats**: Numpy `.npy`/`.npz` + UTF-8 JSON for metadata. Use `backend/dataio/io_utils.py` helpers.

**运行 & 调试快速示例**:
- 以 YAML 运行完整实验（示例）:
  - `python run.py`  # 注意：`run.py` 中示例会调用 `run_experiment_from_yaml`; 修改配置路径到 `configs/` 下的实际文件。
- 手动构建 POD（交互 / 小改动时）:
  - ```bash
    python -c "from backend.config.yaml_io import load_experiment_yaml; from backend.pod.compute import build_pod; dc, pc, ec, tc = load_experiment_yaml('configs/cylinder_exp_full.yaml'); build_pod(dc, pc, verbose=True, plot=True)"
    ```
- 只生成 Level-2（训练+预测并保存）:
  - ```bash
    python - <<'PY'
    from backend.pipelines.train import compute_level2_rebuild
    compute_level2_rebuild('configs/cylinder_exp_full.yaml', experiment_name='myexp', save_root='artifacts/experiments')
    PY
    ```

**依赖与运行时注意**:
- MLP 训练依赖 PyTorch（见 `backend/models/train_mlp.py`），绘图依赖 `matplotlib`，部分表格/绘图会用 `pandas`（可选）。
- 大量操作会分配内存（SVD），`build_pod` 使用 `np.linalg.svd`；在数据大时请在有足够内存的环境中运行或降低 `PodConfig.r`/采样。

**常见编码模式 & 风格提示**:
- 大量函数接受 dataclass 配置对象（`DataConfig/PodConfig/EvalConfig/TrainConfig`），优先使用 `load_experiment_yaml` 来构造这些对象。
- 代码倾向于把“计算”（数值产物）与“绘图/报告”分离：L2/L3 负责数据产物，`pipeline.py` 负责把产物组合成报告/图像。
- 善用 `backend/dataio/io_utils.py` 中 `ensure_dir`, `save_numpy`, `save_json` 保持 I/O 一致性。

**当你要改动/新增功能时，优先检查**:
- 是否改变了 L2 `.npz` 的 key / shape（会影响下游分析）；若必须更改，务必同时更新 `rebuild.py` 中 `L2_NPZ_KEYS` 与 `entry_npz_schema` metadata。
- 配置 schema（`backend/config/schemas.py`）是否需要扩展，若改变请同步更新 `yaml_io.py` 的读写和默认值。

请阅读以上引用的文件以获取细节：`backend/pipeline.py`, `backend/eval/rebuild.py`, `backend/pod/compute.py`, `backend/config/yaml_io.py`, `backend/config/schemas.py`。

如果有某部分不够详尽（例如某个 YAML 字段含义或训练细节），告诉我具体文件或场景，我将把该部分补入或举例。 

**Artifacts 目录（实际产物示例）**
- `artifacts/pod/`: L1 POD 产物（示例文件：`Ur.npy`, `mean_flat.npy`, `singular_values.npy`, `pod_meta.json`, `scale_table.csv`, `basis_spectrum.npz`）。这些文件是 L4 分析的基础依赖。
- `artifacts/experiments/<exp>/L2/`: Level-2 原始预测 `.npz` 文件，示例：`linear_p0001_s0000.npz`, `mlp_p0001_s0010.npz`，并带 `meta.json`。L2 文件遵循 `backend/eval/rebuild.py` 中的 `L2_NPZ_KEYS` schema。
- `artifacts/experiments/<exp>/L3_fft/`: Level-3 FFT / pre-analysis 产物，示例包含 `meta.json`, `index.json`, 以及按 cfg 命名的 `.npz`（例如 `linear_p0004_s0000.npz`）。
- `artifacts/experiments/<exp>/L4_eval/`: Level-4 输出目录（由 L4 引擎写入），包含 `index.json`, 各种圖像子目录與 JSON 索引。

**來自 `console.ipynb` 的常用命令（可作为复制粘贴示例）**
- 构建 POD (L1):
  - `from backend.pod.compute import build_pod`
  - `build_pod(dc, pc, verbose=True, plot=True)` — `dc, pc` 来自 `load_experiment_yaml`。
- 生成 Level-2（训练 + 重建）:
  - `from backend.pipelines.train import compute_level2_rebuild`
  - `compute_level2_rebuild('configs/cylinder_exp_full.yaml', experiment_name='cylinder_exp_full_2', save_root='artifacts/experiments')`
- 生成 Level-3（FFT）:
  - `from backend.pipelines.train import compute_level3_fft_from_level2`
  - `compute_level3_fft_from_level2(exp_dir='artifacts/experiments/cylinder_exp_full_2')`
- 生成 Level-4（评估 + 图像）：示例 assemble 列表见 notebook
  - ```python
    from backend.pipelines.eval.start import compute_level4_eval_mods
    pack = compute_level4_eval_mods(
        exp_dir='artifacts/experiments/cylinder_exp_compare',
        assemble=[
            'fourier.energy_spectrum_legend',
            'fourier.kstar_heatmap',
            'fourier.band_nrmse_curves',
            'fourier.kstar_curves',
        ],
        mod_kwargs={
            'fourier.kstar_curves': {'max_plots': 40, 'show_local_curve': True},
            'fourier.kstar_heatmap': {'use_log10_ell': False},
        },
        out_dirname='L4_eval',
        verbose=True,
    )
    ```

**如何把 L1/L2/L3 产物用于 L4（详细）**
- L1 (POD) 提供：`Ur.npy`（D x r）、`mean_flat.npy`（D,）、`pod_meta.json`（包含 H,W,C,T,r_used 等）。L4 某些模块（例如 spatial recon / band 分解）可能直接读取这些文件；POD 的尺度信息保存在 `scale_table.csv` / `scale_meta.json`。
- L2 (.npz) 是 L4 的主数据源：`EvalContext` 會在 `resolve()` 時構建 `_l2_filemap`，通過 `ctx.get_l2_path(model_type, mask_rate, noise_sigma)` 可以直接拿到對應 `.npz` 路徑，並通過 `ctx.load_l2(...)` 讀取一個 dict。L2 中 `A_hat_all`（預測系數）是還原出整個場的最小信息單元。
- L3 (FFT) 提供預計算的頻域統計，`ctx.get_l3_path(...)` / `ctx.load_l3(...)` 可直接訪問；L3 的 presence 會影響某些 fourier/scale 模組跳過重複計算並直接使用已有分析結果。

**Level-4 (評估) mod 引擎結構要點**
- 入口: `backend/pipelines/eval/start.py::compute_level4_eval_mods` — 構建 `EvalContext`，註冊內置 mod（`register_builtin_mods()` + 各類 `register_*_mods()`），然後調用 `run_eval_mods`。
- 上下文: `backend/pipelines/eval/context.py::EvalContext` — 負責解析 `exp_dir`、加載 YAML（`load_experiment_yaml`）、定位 `L2`/`L3` 根目錄並構建文件映射（`_build_l2_filemap/_build_l3_filemap`）。模組通過該上下文訪問所有數據：`ctx.load_l2(...)`, `ctx.load_l3(...)`, `ctx.l2_meta()` 等。
- 註冊/解析: `backend/pipelines/eval/registry.py` 提供 `EvalMod` dataclass 與註冊 API（`register_mod`、`get_mod`、`resolve_mods`）。模組以字符串 name 註冊，支持按 notebook 彙編 `assemble` 列表運行。
- 運行器: `backend/pipelines/eval/runner.py::run_eval_mods` — 按 `assemble` 的解析順序運行每個 mod：
  - 調用 `mod.run(ctx, kwargs)`，期望返回 `dict`。
  - 如果返回 dict 包含 `fig_paths`（列表或單字符串），會被收集到 `pack['fig_paths']`。
  - 最終返回 pack 格式: `{'out_dir': str(l4_root), 'results': {mod_name: result_dict}, 'fig_paths': {mod_name: [paths...]}}` 並寫入 `L4_eval/index.json`（若 `save_index=True`）。

**如何編寫/註冊新 mod（快速指南）**
1. 在 `backend/pipelines/eval_mods/` 下添加一個模組文件（例如 `my_mods.py`），實現一個註冊函數 `register_my_mods()` 或直接在模組頂層創建並調用 `registry.register_mod(EvalMod(...))`。
2. `EvalMod` 的 `run(ctx, kwargs)` 簽名應接受 `EvalContext` 與 `dict`，並返回包含需要的數值/路徑的 `dict`。若需要輸出圖片，返回值中應包含 `fig_paths: ["...png"]`。
3. 在 `backend/pipelines/eval/start.py` 中導入並註冊你的 `register_my_mods()`（或在 notebook 中手動調用註冊函數），然後把 mod 名稱加到 `assemble` 列表中運行。

**pack / index.json 示例（運行後產物）**
- `L4_eval/index.json`（示例結構）：
  - `out_dir`: 輸出目錄路徑
  - `results`: 按 mod 名稱的結果字典（每個 mod 內部定義格式）
  - `fig_paths`: 每個 mod 輸出的圖片路徑列表

**調試要點**
- 若某個 mod 找不到 L2/L3 文件：先用 `ctx.paths.l2_root` 檢查目錄佈局；可用 notebook 中 `from backend.pipelines.eval.context import EvalContext; EvalContext(exp_dir='...').resolve()` 複現查找邏輯。
- 若要修改 L2 schema（新增 key），同時更新 `backend/eval/rebuild.py::L2_NPZ_KEYS` 與 `entry_npz_schema`，並確保 `EvalContext._build_l2_filemap` / `load_npz` 仍能向後兼容舊文件。
