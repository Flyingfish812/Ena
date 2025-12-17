from backend.pipeline import run_experiment_from_yaml

res = run_experiment_from_yaml(
    "configs/cylinder_exp2.yaml",
    experiment_name="cylinder_exp4_1",
    save_root="artifacts/experiments",
    generate_report=True,
    verbose=True,
)

print("报告路径:", res["report_path"])
res["df_linear"].head()
