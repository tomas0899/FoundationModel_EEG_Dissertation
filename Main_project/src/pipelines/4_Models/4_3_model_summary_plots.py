from pathlib import Path
from datetime import date
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
from pathlib import Path

current_file = Path(__file__).resolve()

project_root = None

for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break

if project_root is None:
    raise RuntimeError(
        "Project root not found. Could not find a parent folder containing 'src'."
    )

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules import tools_EEG_models as TEEG_mod
# ============================================================
# Model order
# ============================================================

model_order = [
    "FEATURES + SVM",
    "PCA + SVM",
    "PCA + Decision Tree",
    "PCA + Random Forest",
]


# ============================================================
# Patient ID mapping
# ============================================================

patient_number_map = {
    "10OXG": "P1",
    "1JSZ6": "P2",
    "3ZL8B": "P3",
    "F88R2": "P4",
    "FP628": "P5",
    "JYXFE": "P6",
    "PN12G": "P7",
    "RQXZ1": "P8",
    "XB47Y": "P9"
}

patient_order = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "P9"
]


# ============================================================
# Patient color palette
# ============================================================

CB_color_cycle = [
    "#2E0F4F",  # P1 - very dark violet
    "#FF5F00",  # P2 - very bright orange
    "#1B7837",  # P3 - dark green
    "#f781bf",  # P4 - pink
    "#a65628",  # P5 - brown
    "#984ea3",  # P6 - purple
    "#999999",  # P7 - grey
    "#e41a1c",  # P8 - red
    "#dede00"   # P9 - yellow
]

patient_colors = {
    patient: CB_color_cycle[i]
    for i, patient in enumerate(patient_order)
}


# ============================================================
# Helper function for clean filenames
# ============================================================

def clean_filename(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text


# ============================================================
# 1. Function to extract one metric from one classification CSV
# ============================================================

def extract_classification_metric(
    csv_path,
    metric_row="weighted avg",
    metric_col="f1-score"
):
    """
    Extracts one value from a classification table CSV.

    Example:
    metric_row = "weighted avg"
    metric_col = "f1-score"
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Remove accidental unnamed index columns if present
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Safety check
    if "class_or_metric" not in df.columns:
        raise ValueError(
            f"'class_or_metric' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    if metric_col not in df.columns:
        raise ValueError(
            f"'{metric_col}' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    # Normalize labels to avoid problems with spaces/capitalization
    df["class_or_metric_clean"] = (
        df["class_or_metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if isinstance(metric_row, str):
        metric_row = [metric_row]
    
    metric_rows_clean = [
        str(value).strip().lower()
        for value in metric_row
    ]
    
    matched_row = df[
        df["class_or_metric_clean"].isin(metric_rows_clean)
    ]

    if matched_row.empty:
        return np.nan
    
    return float(matched_row[metric_col].iloc[0])

    value = metric_match[metric_col].iloc[0]

    return float(value)
def build_multi_metric_summary_df(metadata_df):
    """
    Builds a long-format summary dataframe with:
    1. weighted avg f1-score
    2. preictal f1-score
    3. seizure f1-score
    """

    metrics_to_extract = [
        {
            "metric_name": "Weighted F1",
            "metric_row": "weighted avg",
            "metric_col": "f1-score"
        },
        {
            "metric_name": "Preictal F1",
            "metric_row": "preictal",
            "metric_col": "f1-score"
        },
        {
            "metric_name": "Ictal F1",
            "metric_row": ["ictal", "seizure"],
            "metric_col": "f1-score"
        },
    ]

    summary_rows = []

    for _, row in metadata_df.iterrows():

        if row.get("parse_error", False):
            continue

        csv_path = row["filepath"]

        for metric in metrics_to_extract:

            metric_value = extract_classification_metric(
                csv_path=csv_path,
                metric_row=metric["metric_row"],
                metric_col=metric["metric_col"]
            )

            summary_rows.append({
                "patient_id": row["patient_id"],
                "dataset": row["dataset"],
                "input_type": row["input_type"],
                "model": row["model"],
                "model_group": row["model_group"],
                "metric_name": metric["metric_name"],
                "metric_row": metric["metric_row"],
                "metric_col": metric["metric_col"],
                "metric_value": metric_value,
                "filename": row["filename"],
                "filepath": row["filepath"]
            })

    summary_df_long = pd.DataFrame(summary_rows)

    return summary_df_long
def parse_classification_filename(filepath):
    """
    Extract metadata from classification table CSV filenames.
 
    Handles:
    1. SVM filenames:
       JYXFE_IN-PCA_SVM-SCORING-F1-MACRO_20260608_v01_Test_classification_table.csv

    2. Decision Tree / Random Forest filenames:
       PAT-JYXFE_Test_-_Decision_Tree_classification_table.csv

    Rule:
    Decision Tree and Random Forest are always PCA-based.
    """

    name = Path(filepath).name

    # -------------------------------------------------
    # Pattern 1: SVM
    # -------------------------------------------------
    svm_pattern = re.match(
        r"(?P<patient_id>.+?)_IN-(?P<input_type>FEATURES|PCA)_"
        r"(?P<model>SVM)-SCORING-(?P<scoring>.+?)_"
        r"(?P<date>\d{8})_(?P<version>v\d+)_(?P<dataset>Test|Validation)_classification_table\.csv",
        name
    )

    if svm_pattern:
        info = svm_pattern.groupdict()

        info["model"] = "SVM"
        info["model_group"] = f"{info['input_type']} + SVM"
        info["filename"] = name
        info["filepath"] = str(filepath)
        info["parse_error"] = False

        return info

    # -------------------------------------------------
    # Pattern 2: Decision Tree / Random Forest
    # -------------------------------------------------
    tree_rf_pattern = re.match(
        r"PAT-(?P<patient_id>.+?)_"
        r"(?P<dataset>Test|Validation)_-_"
        r"(?P<model>Decision_Tree|Random_Forest)_classification_table\.csv",
        name
    )

    if tree_rf_pattern:
        info = tree_rf_pattern.groupdict()

        # Fixed rule: DT and RF are always PCA-based
        info["input_type"] = "PCA"
        info["model"] = info["model"].replace("_", " ")
        info["scoring"] = None
        info["date"] = None
        info["version"] = None
        info["model_group"] = f"{info['input_type']} + {info['model']}"
        info["filename"] = name
        info["filepath"] = str(filepath)
        info["parse_error"] = False

        return info

    # -------------------------------------------------
    # If no pattern matched
    # -------------------------------------------------
    return {
        "patient_id": None,
        "input_type": None,
        "model": None,
        "scoring": None,
        "date": None,
        "version": None,
        "dataset": None,
        "model_group": None,
        "filename": name,
        "filepath": str(filepath),
        "parse_error": True
    }

# ============================================================
# 1. Define folder containing all classification table CSVs
# ============================================================

classification_tables_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/classification_tables"
)

# Load all CSV paths in that folder
csv_files = sorted(classification_tables_dir.glob("*classification_table.csv"))

print(f"Number of CSV files found: {len(csv_files)}")

for file in csv_files:
    print(file.name)

metadata_df = pd.DataFrame([
    parse_classification_filename(file)
    for file in csv_files
])

metadata_df
metadata_df.groupby(["dataset", "model_group"]).size()
summary_df_long = build_multi_metric_summary_df(metadata_df)
def plot_metric_from_summary_long(
    summary_df_long,
    metric_name,
    dataset_to_plot="Test",
    baseline=0.5,
    output_path=None,
    show_plot=False,
    add_stats_text=True,
    add_median_line=True
):
    """
    Plot one metric from summary_df_long.

    Each point represents one patient.

    Uses:
    - Numeric patient labels: P1-P9
    - Same patient color palette used in previous plots
    - Same circular marker for all patients
    - Fixed horizontal offset per patient to avoid overlap
    - Optional baseline at 0.5
    - Optional median line per ML model
    - Mean/median/n text above the 0-1 metric range
    - Saves plot as SVG
    """

    df_plot = summary_df_long.copy()

    df_plot = df_plot[
        (df_plot["dataset"] == dataset_to_plot) &
        (df_plot["metric_name"] == metric_name)
    ].copy()

    # -------------------------------------------------
    # Convert original patient IDs to numeric patient labels
    # -------------------------------------------------

    df_plot["Patient"] = df_plot["patient_id"].map(patient_number_map)

    already_numeric_mask = df_plot["patient_id"].isin(patient_order)
    df_plot.loc[already_numeric_mask, "Patient"] = df_plot.loc[
        already_numeric_mask,
        "patient_id"
    ]

    unmapped_patients = df_plot.loc[
        df_plot["Patient"].isna(),
        "patient_id"
    ].unique()

    if len(unmapped_patients) > 0:
        print("Warning: these patient IDs were not mapped:", unmapped_patients)

    df_plot = df_plot.dropna(subset=["Patient"])

    df_plot["Patient"] = pd.Categorical(
        df_plot["Patient"],
        categories=patient_order,
        ordered=True
    )

    # -------------------------------------------------
    # Keep model order consistent
    # -------------------------------------------------

    current_order = [
        model for model in model_order
        if model in df_plot["model_group"].unique()
    ]

    df_plot["model_group"] = pd.Categorical(
        df_plot["model_group"],
        categories=current_order,
        ordered=True
    )

    df_plot = df_plot.sort_values(["model_group", "Patient"])

    # -------------------------------------------------
    # Calculate mean and median per model category
    # -------------------------------------------------

    stats_df = (
        df_plot
        .groupby("model_group", observed=True)["metric_value"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )

    print(f"\n{dataset_to_plot} - {metric_name}")
    print(stats_df)

    # -------------------------------------------------
    # Patient offset mapping
    # -------------------------------------------------

    present_patients = [
        patient for patient in patient_order
        if patient in df_plot["Patient"].astype(str).unique()
    ]

    if len(present_patients) == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.16, 0.16, len(present_patients))

    patient_to_offset = {
        patient: offsets[i]
        for i, patient in enumerate(present_patients)
    }

    model_to_x = {
        model: i
        for i, model in enumerate(current_order)
    }

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_upper = 1.12
    y_stats = 1.09

    for _, row in df_plot.iterrows():

        model_group = row["model_group"]
        patient = str(row["Patient"])
        y_value = row["metric_value"]

        x_value = model_to_x[model_group] + patient_to_offset[patient]

        ax.scatter(
            x_value,
            y_value,
            marker="o",                      # same shape for all patients
            s=95,
            facecolors=patient_colors[patient],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.90,
            zorder=3
        )

    # -------------------------------------------------
    # Median line per model
    # -------------------------------------------------

    if add_median_line:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            median_value = row["median"].iloc[0]

            ax.hlines(
                y=median_value,
                xmin=i - 0.24,
                xmax=i + 0.24,
                color="black",
                linewidth=2.0,
                alpha=0.45,
                zorder=4
            )

    # -------------------------------------------------
    # Optional horizontal baseline
    # -------------------------------------------------

    if baseline is not None:
        ax.axhline(
            y=baseline,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.7,
            zorder=1
        )

    # -------------------------------------------------
    # Add mean and median text above plot points
    # -------------------------------------------------

    if add_stats_text:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            mean_value = row["mean"].iloc[0]
            median_value = row["median"].iloc[0]
            count_value = int(row["count"].iloc[0])

            stats_text = (
                f"mean={mean_value:.2f}\n"
                f"med={median_value:.2f}\n"
                f"n={count_value}"
            )

            ax.text(
                x=i,
                y=y_stats,
                s=stats_text,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                ),
                zorder=5
            )

    # -------------------------------------------------
    # Axis formatting
    # -------------------------------------------------

    ax.set_title(f"{dataset_to_plot} - {metric_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)

    ax.set_ylim(0, y_upper)

    ax.set_xticks(range(len(current_order)))
    ax.set_xticklabels(current_order, rotation=25, ha="right")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # -------------------------------------------------
    # Custom legend for patients, median and baseline
    # -------------------------------------------------

    patient_handles = [
        Line2D(
            [0],
            [0],
            marker="o",                      # no different shapes
            color="w",
            markerfacecolor=patient_colors[patient],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            markersize=8,
            label=patient
        )
        for patient in present_patients
    ]

    legend_handles = patient_handles.copy()

    if add_median_line:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.0,
                alpha=0.45,
                label="Median"
            )
        )

    if baseline is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline = {baseline}"
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Patient",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()

    # -------------------------------------------------
    # Save as SVG
    # -------------------------------------------------

    if output_path is None:
        today = date.today().strftime("%Y-%m-%d")
        filename = (
            f"{clean_filename(dataset_to_plot)}_"
            f"{clean_filename(metric_name)}_"
            f"model_comparison_patient_palette_median_line_{today}.svg"
        )
        output_path = Path.cwd() / filename
    else:
        output_path = Path(output_path)

        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight"
    )

    print(f"SVG saved in: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()
# TEST!!!
output_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/summary_plots"
)

metrics_to_plot = [
    "Weighted F1",
    "Preictal F1",
    "Ictal F1",
]

for metric_name in metrics_to_plot:
    safe_metric_name = metric_name.lower().replace(" ", "_")

    plot_metric_from_summary_long(
        summary_df_long=summary_df_long,
        metric_name=metric_name,
        dataset_to_plot="Test",
        baseline=0.5,
        output_path=output_dir / f"Test_{safe_metric_name}_summary_plot.svg",
        show_plot=False
    )
# VALIDATION!!
for metric_name in metrics_to_plot:
    safe_metric_name = metric_name.lower().replace(" ", "_")

    plot_metric_from_summary_long(
        summary_df_long=summary_df_long,
        metric_name=metric_name,
        dataset_to_plot="Validation",
        baseline=0.5,
        output_path=output_dir / f"Validation_{safe_metric_name}_summary_plot.svg",
        show_plot=False,
        add_stats_text=True
    )


def extract_accuracy_metric(csv_path):
    """
    Extract accuracy from one classification table CSV.
    In your CSV, the accuracy value is stored in the 'accuracy' row,
    usually under the 'f1-score' column.
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Remove accidental unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "class_or_metric" not in df.columns:
        raise ValueError(
            f"'class_or_metric' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    df["class_or_metric_clean"] = (
        df["class_or_metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    acc_row = df[df["class_or_metric_clean"] == "accuracy"]

    if acc_row.empty:
        raise ValueError(
            f"'accuracy' row not found in {csv_path.name}. "
            f"Rows found: {df['class_or_metric'].tolist()}"
        )

    # In your classification table, this should work
    accuracy_value = acc_row["f1-score"].iloc[0]

    return float(accuracy_value)
def build_accuracy_summary_df(metadata_df):
    """
    Builds one summary dataframe for accuracy.

    Each CSV becomes one row:
    patient_id | dataset | model_group | accuracy
    """

    summary_rows = []

    for _, row in metadata_df.iterrows():

        if row.get("parse_error", False):
            continue

        csv_path = row["filepath"]

        accuracy_value = extract_accuracy_metric(csv_path)

        summary_rows.append({
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "input_type": row["input_type"],
            "model": row["model"],
            "model_group": row["model_group"],
            "accuracy": accuracy_value,
            "filename": row["filename"],
            "filepath": row["filepath"]
        })

    summary_accuracy_df = pd.DataFrame(summary_rows)

    return summary_accuracy_df
def plot_accuracy_by_dataset(
    summary_accuracy_df,
    dataset_to_plot="Test",
    baseline=0.5,
    output_path=None,
    show_plot=False,
    add_stats_text=True,
    add_median_line=True
):
    """
    Plot accuracy for one dataset: Test or Validation.

    Each point represents one patient.

    Uses:
    - Numeric patient labels: P1-P9
    - Same patient color palette used in previous plots
    - Same circular marker for all patients
    - Fixed horizontal offset per patient to avoid overlap
    - Optional baseline at 0.5
    - Optional median line per ML model
    - Saves plot as SVG
    """

    df_plot = summary_accuracy_df.copy()

    df_plot = df_plot[
        df_plot["dataset"] == dataset_to_plot
    ].copy()

    # -------------------------------------------------
    # Convert original patient IDs to numeric patient labels
    # -------------------------------------------------

    df_plot["Patient"] = df_plot["patient_id"].map(patient_number_map)

    already_numeric_mask = df_plot["patient_id"].isin(patient_order)
    df_plot.loc[already_numeric_mask, "Patient"] = df_plot.loc[
        already_numeric_mask,
        "patient_id"
    ]

    unmapped_patients = df_plot.loc[
        df_plot["Patient"].isna(),
        "patient_id"
    ].unique()

    if len(unmapped_patients) > 0:
        print("Warning: these patient IDs were not mapped:", unmapped_patients)

    df_plot = df_plot.dropna(subset=["Patient"])

    df_plot["Patient"] = pd.Categorical(
        df_plot["Patient"],
        categories=patient_order,
        ordered=True
    )

    # -------------------------------------------------
    # Keep model order consistent
    # -------------------------------------------------

    current_order = [
        model for model in model_order
        if model in df_plot["model_group"].unique()
    ]

    df_plot["model_group"] = pd.Categorical(
        df_plot["model_group"],
        categories=current_order,
        ordered=True
    )

    df_plot = df_plot.sort_values(["model_group", "Patient"])

    # -------------------------------------------------
    # Calculate mean and median per model category
    # -------------------------------------------------

    stats_df = (
        df_plot
        .groupby("model_group", observed=True)["accuracy"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )

    print(f"\nAccuracy - {dataset_to_plot}")
    print(stats_df)

    # -------------------------------------------------
    # Patient offset mapping
    # -------------------------------------------------

    present_patients = [
        patient for patient in patient_order
        if patient in df_plot["Patient"].astype(str).unique()
    ]

    if len(present_patients) == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.16, 0.16, len(present_patients))

    patient_to_offset = {
        patient: offsets[i]
        for i, patient in enumerate(present_patients)
    }

    model_to_x = {
        model: i
        for i, model in enumerate(current_order)
    }

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_upper = 1.12
    y_stats = 1.09

    for _, row in df_plot.iterrows():

        model_group = row["model_group"]
        patient = str(row["Patient"])
        y_value = row["accuracy"]

        x_value = model_to_x[model_group] + patient_to_offset[patient]

        ax.scatter(
            x_value,
            y_value,
            marker="o",                      # same shape for all patients
            s=95,
            facecolors=patient_colors[patient],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.90,
            zorder=3
        )

    # -------------------------------------------------
    # Median line per model
    # -------------------------------------------------

    if add_median_line:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            median_value = row["median"].iloc[0]

            ax.hlines(
                y=median_value,
                xmin=i - 0.24,
                xmax=i + 0.24,
                color="black",
                linewidth=2.0,
                alpha=0.45,
                zorder=4
            )

    # -------------------------------------------------
    # Baseline
    # -------------------------------------------------

    if baseline is not None:
        ax.axhline(
            y=baseline,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.7,
            zorder=1
        )

    # -------------------------------------------------
    # Add mean and median text above plot points
    # -------------------------------------------------

    if add_stats_text:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            mean_value = row["mean"].iloc[0]
            median_value = row["median"].iloc[0]
            count_value = int(row["count"].iloc[0])

            stats_text = (
                f"mean={mean_value:.2f}\n"
                f"med={median_value:.2f}\n"
                f"n={count_value}"
            )

            ax.text(
                x=i,
                y=y_stats,
                s=stats_text,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                ),
                zorder=5
            )

    # -------------------------------------------------
    # Formatting
    # -------------------------------------------------

    ax.set_title(f"Accuracy - {dataset_to_plot}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")

    ax.set_ylim(0, y_upper)

    ax.set_xticks(range(len(current_order)))
    ax.set_xticklabels(current_order, rotation=25, ha="right")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------

    patient_handles = [
        Line2D(
            [0],
            [0],
            marker="o",                      # no different shapes
            color="w",
            markerfacecolor=patient_colors[patient],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            markersize=8,
            label=patient
        )
        for patient in present_patients
    ]

    legend_handles = patient_handles.copy()

    if add_median_line:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.0,
                alpha=0.45,
                label="Median"
            )
        )

    if baseline is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline = {baseline}"
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Patient",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()

    # -------------------------------------------------
    # Save as SVG
    # -------------------------------------------------

    if output_path is None:
        today = date.today().strftime("%Y-%m-%d")
        filename = (
            f"accuracy_"
            f"{clean_filename(dataset_to_plot)}_"
            f"model_comparison_patient_palette_median_line_{today}.svg"
        )
        output_path = Path.cwd() / filename
    else:
        output_path = Path(output_path)

        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight"
    )

    print(f"SVG saved in: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

summary_accuracy_df = build_accuracy_summary_df(metadata_df)

summary_accuracy_df[[
    "patient_id",
    "dataset",
    "model_group",
    "accuracy",
    "filename"
]]
output_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/summary_plots"
)

for dataset_name in ["Test", "Validation"]:

    plot_accuracy_by_dataset(
        summary_accuracy_df=summary_accuracy_df,
        dataset_to_plot=dataset_name,
        baseline=0.5,
        output_path=output_dir / f"Accuracy_{dataset_name}_summary_plot.svg",
        show_plot=False,
        add_stats_text=True
    )



# ============================================================
# 1. Extract all relevant metrics from one classification table
# ============================================================

def extract_classification_metrics(csv_path):
    """
    Extract accuracy, weighted F1, preictal F1 and ictal/seizure F1
    from one classification report CSV.

    Expected rows in class_or_metric:
    - preictal
    - seizure or ictal
    - accuracy
    - weighted avg
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Remove accidental unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "class_or_metric" not in df.columns:
        raise ValueError(
            f"'class_or_metric' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    if "f1-score" not in df.columns:
        raise ValueError(
            f"'f1-score' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    df["class_or_metric_clean"] = (
        df["class_or_metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def get_f1_from_row(row_name_options):
        """
        Extract f1-score from the first matching row.
        row_name_options can be a string or a list of strings.
        """

        if isinstance(row_name_options, str):
            row_name_options = [row_name_options]

        row_name_options = [
            str(x).strip().lower()
            for x in row_name_options
        ]

        matched_row = df[
            df["class_or_metric_clean"].isin(row_name_options)
        ]

        if matched_row.empty:
            return np.nan

        return float(matched_row["f1-score"].iloc[0])

    metrics = {
        "accuracy": get_f1_from_row("accuracy"),
        "weighted_f1": get_f1_from_row("weighted avg"),
        "preictal_f1": get_f1_from_row("preictal"),
        "ictal_f1": get_f1_from_row(["ictal", "seizure"])
    }

    return metrics


# ============================================================
# 2. Build full metrics summary dataframe
# ============================================================

def build_full_metrics_summary_df(metadata_df):
    """
    Each classification CSV becomes one row:
    patient_id | dataset | input_type | model | model_group |
    accuracy | weighted_f1 | preictal_f1 | ictal_f1
    """

    summary_rows = []

    for _, row in metadata_df.iterrows():

        if row.get("parse_error", False):
            continue

        csv_path = row["filepath"]

        metrics = extract_classification_metrics(csv_path)

        summary_rows.append({
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "input_type": row["input_type"],
            "model": row["model"],
            "model_group": row["model_group"],
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "preictal_f1": metrics["preictal_f1"],
            "ictal_f1": metrics["ictal_f1"],
            "filename": row["filename"],
            "filepath": row["filepath"]
        })

    summary_metrics_df = pd.DataFrame(summary_rows)

    return summary_metrics_df


# ============================================================
# 3. Create compact best-model table
# ============================================================

def build_best_model_compact_table(
    summary_metrics_df,
    dataset_to_use="Test",
    output_path=None
):
    """
    Build compact table:

    patient_id
    best_by_accuracy
    accuracy
    best_by_weighted_f1
    weighted_f1
    preictal_f1
    ictal_f1
    same_best_model

    The F1 values reported are from the model selected by weighted F1.
    """

    df = summary_metrics_df.copy()

    df = df[df["dataset"] == dataset_to_use].copy()

    if df.empty:
        raise ValueError(f"No rows found for dataset: {dataset_to_use}")

    # --------------------------------------------------------
    # Best model per patient by accuracy
    # --------------------------------------------------------

    idx_best_acc = (
        df
        .groupby("patient_id")["accuracy"]
        .idxmax()
    )

    best_acc_df = (
        df.loc[idx_best_acc, [
            "patient_id",
            "model_group",
            "accuracy"
        ]]
        .rename(columns={
            "model_group": "best_by_accuracy",
            "accuracy": "best_accuracy"
        })
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Best model per patient by weighted F1
    # --------------------------------------------------------

    idx_best_f1 = (
        df
        .groupby("patient_id")["weighted_f1"]
        .idxmax()
    )

    best_f1_df = (
        df.loc[idx_best_f1, [
            "patient_id",
            "model_group",
            "weighted_f1",
            "preictal_f1",
            "ictal_f1",
            "accuracy"
        ]]
        .rename(columns={
            "model_group": "best_by_weighted_f1",
            "accuracy": "accuracy_of_best_weighted_f1_model"
        })
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Merge both selections
    # --------------------------------------------------------

    compact_table = best_acc_df.merge(
        best_f1_df,
        on="patient_id",
        how="outer"
    )

    compact_table["same_best_model"] = (
        compact_table["best_by_accuracy"]
        == compact_table["best_by_weighted_f1"]
    )

    # --------------------------------------------------------
    # Optional: make values dissertation-friendly
    # --------------------------------------------------------

    metric_cols = [
        "best_accuracy",
        "weighted_f1",
        "preictal_f1",
        "ictal_f1",
        "accuracy_of_best_weighted_f1_model"
    ]

    for col in metric_cols:
        compact_table[col] = compact_table[col].round(3)

    compact_table = compact_table.sort_values("patient_id").reset_index(drop=True)

    # --------------------------------------------------------
    # Save if requested
    # --------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            compact_table.to_csv(output_path, index=False)
        elif output_path.suffix in [".xlsx", ".xls"]:
            compact_table.to_excel(output_path, index=False)
        else:
            raise ValueError("output_path must end in .csv or .xlsx")

        print(f"Compact table saved to: {output_path}")

    return compact_table


# ============================================================
# 4. Run table generation
# ============================================================


summary_metrics_df = build_full_metrics_summary_df(metadata_df)

tables_output_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/summary_tables"
)

tables_output_dir.mkdir(parents=True, exist_ok=True)

compact_best_model_table = build_best_model_compact_table(
    summary_metrics_df=summary_metrics_df,
    dataset_to_use="Test",
    output_path=tables_output_dir / "best_model_per_patient_compact_Test.csv"
)

metadata_df.to_csv(
    tables_output_dir / "classification_files_metadata.csv",
    index=False
)

summary_df_long.to_csv(
    tables_output_dir / "classification_metrics_long.csv",
    index=False
)

summary_accuracy_df.to_csv(
    tables_output_dir / "accuracy_summary.csv",
    index=False
)

summary_metrics_df.to_csv(
    tables_output_dir / "full_metrics_summary.csv",
    index=False
)

print(compact_best_model_table)