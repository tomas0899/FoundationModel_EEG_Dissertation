from pathlib import Path
import sys

import pandas as pd


# ============================================================
# 1. Find project root
# ============================================================

current_file = Path(__file__).resolve()

project_root = None

for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break

if project_root is None:
    raise RuntimeError(
        "Project root not found. "
        "Could not find a parent folder containing 'src'."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================
# 2. Import project module
# ============================================================

from src.modules import tools_EEG_models as TEEG_mod


# ============================================================
# 3. Define input and output folders
# ============================================================

classification_tables_dir = (
    project_root
    / "results"
    / "together_results"
    / "classification_tables"
)

plots_output_dir = (
    project_root
    / "results"
    / "together_results"
    / "summary_plots"
)

tables_output_dir = (
    project_root
    / "results"
    / "together_results"
    / "summary_tables"
)

plots_output_dir.mkdir(parents=True, exist_ok=True)
tables_output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. Load classification table paths
# ============================================================

csv_files = sorted(
    classification_tables_dir.glob(
        "*classification_table.csv"
    )
)

print(f"Number of CSV files found: {len(csv_files)}")

if not csv_files:
    raise FileNotFoundError(
        "No classification table CSV files were found in: "
        f"{classification_tables_dir}"
    )

for file in csv_files:
    print(file.name)


# ============================================================
# 5. Parse classification filenames
# ============================================================

metadata_df = pd.DataFrame([
    TEEG_mod.parse_classification_filename(file)
    for file in csv_files
])

print("\nFiles per dataset and model:")
print(
    metadata_df
    .groupby(["dataset", "model_group"])
    .size()
)


# ============================================================
# 6. Build summary dataframes
# ============================================================

summary_metrics_df = (
    TEEG_mod.build_full_metrics_summary_df(
        metadata_df
    )
)

summary_df_long = (
    TEEG_mod.build_long_metrics_summary_df(
        summary_metrics_df
    )
)


# ============================================================
# 7. Generate summary plots
# ============================================================

metrics_to_plot = [
    "Accuracy",
    "Weighted F1",
    "Preictal F1",
    "Ictal F1",
]

for dataset_name in ["Test", "Validation"]:

    for metric_name in metrics_to_plot:

        safe_metric_name = TEEG_mod.clean_filename(
            metric_name
        )

        safe_dataset_name = TEEG_mod.clean_filename(
            dataset_name
        )

        TEEG_mod.plot_metric_from_summary_long(
            summary_df_long=summary_df_long,
            metric_name=metric_name,
            dataset_to_plot=dataset_name,
            baseline=0.5,
            output_path=(
                plots_output_dir
                / (
                    f"{safe_dataset_name}_"
                    f"{safe_metric_name}_summary_plot.svg"
                )
            ),
            show_plot=False,
            add_stats_text=True,
            add_median_line=True
        )


# ============================================================
# 8. Build compact best-model table
# ============================================================

compact_best_model_table = (
    TEEG_mod.build_best_model_compact_table(
        summary_metrics_df=summary_metrics_df,
        dataset_to_use="Test",
        output_path=(
            tables_output_dir
            / "best_model_per_patient_compact_Test.csv"
        )
    )
)


# ============================================================
# 9. Save summary dataframes
# ============================================================

metadata_df.to_csv(
    tables_output_dir
    / "classification_files_metadata.csv",
    index=False
)

summary_df_long.to_csv(
    tables_output_dir
    / "classification_metrics_long.csv",
    index=False
)

summary_metrics_df.to_csv(
    tables_output_dir
    / "full_metrics_summary.csv",
    index=False
)


# ============================================================
# 10. Display final compact table
# ============================================================

print("\nCompact best-model table:")
print(compact_best_model_table)