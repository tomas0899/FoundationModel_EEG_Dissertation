from pathlib import Path
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr


# ============================================================
# 1. Define paths
# ============================================================

project_root = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026"
)

summary_tables_dir = (
    project_root
    / "Main_project"
    / "results"
    / "together_results"
    / "summary_tables"
)

output_dir = (
    project_root
    / "Main_project"
    / "results"
    / "together_results"
    / "MRA"
)

output_dir.mkdir(parents=True, exist_ok=True)

input_path = summary_tables_dir / "totalValmerged.pkl"

if not input_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_path}"
    )


# ============================================================
# 2. Load merged dataframe
# ============================================================

final_merged_df = pd.read_pickle(input_path)

print(f"Loaded dataframe from: {input_path}")
print(f"Shape: {final_merged_df.shape}")


# ============================================================
# 3. Patient definitions
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
    "P1", "P2", "P3", "P4", "P5",
    "P6", "P7", "P8", "P9"
]

CB_color_cycle = [
    "#2E0F4F",  # P1
    "#FF5F00",  # P2
    "#1B7837",  # P3
    "#f781bf",  # P4
    "#a65628",  # P5
    "#984ea3",  # P6
    "#999999",  # P7
    "#e41a1c",  # P8
    "#dede00"   # P9
]

patient_colors = {
    patient: CB_color_cycle[i]
    for i, patient in enumerate(patient_order)
}


# ============================================================
# 4. Helper function to prepare plotting dataframe
# ============================================================

def prepare_plot_df(df, required_columns):
    df_plot = df.copy()

    df_plot[required_columns] = df_plot[required_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if "Study label" in df_plot.columns:
        df_plot["Patient"] = df_plot["Study label"].astype(str).str.strip()
    else:
        df_plot["Patient"] = df_plot["Patient ID"].map(patient_number_map)

    df_plot["Patient ID"] = (
        df_plot["Patient ID"]
        .astype(str)
        .str.strip()
    )

    df_plot = df_plot.dropna(
        subset=["Patient ID", "Patient"] + required_columns
    ).copy()

    unmapped_patients = df_plot.loc[
        ~df_plot["Patient"].isin(patient_order),
        "Patient ID"
    ].unique()

    if len(unmapped_patients) > 0:
        print(
            "Warning: these Patient IDs were not mapped:",
            unmapped_patients
        )

    df_plot = df_plot[
        df_plot["Patient"].isin(patient_order)
    ].copy()

    df_plot["Patient"] = pd.Categorical(
        df_plot["Patient"],
        categories=patient_order,
        ordered=True
    )

    df_plot = (
        df_plot
        .sort_values("Patient")
        .reset_index(drop=True)
    )

    return df_plot


# ============================================================
# 5. Helper function to generate bubble plots
# ============================================================

def make_bubble_plot(
    df,
    x_col,
    y_col,
    size_col,
    x_label,
    y_label,
    title,
    output_filename,
    add_accuracy_reference_line=False
):
    df_plot = prepare_plot_df(
        df,
        required_columns=[x_col, y_col, size_col]
    )

    x = df_plot[x_col]
    y = df_plot[y_col]
    density = df_plot[size_col]

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(x, y)

    print(f"\nPearson correlation for {x_col} vs {y_col}:")
    print(f"r = {pearson_r:.3f}, p = {pearson_p:.3f}")

    # Bubble sizes
    if density.max() == density.min():
        bubble_sizes = pd.Series(600, index=df_plot.index)
    else:
        bubble_sizes = 200 + (
            (density - density.min())
            / (density.max() - density.min())
        ) * 1200

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, row in df_plot.iterrows():
        ax.scatter(
            row[x_col],
            row[y_col],
            s=bubble_sizes.loc[i],
            color=patient_colors[row["Patient"]],
            alpha=0.78,
            edgecolor="black",
            linewidth=0.8
        )

        x_offset = 0.8 if x.max() > 5 else 0.02
        y_offset = 0.005 if y.max() <= 1.5 else 0.3

        ax.text(
            row[x_col] + x_offset,
            row[y_col] + y_offset,
            row["Patient"],
            fontsize=9,
            fontweight="bold"
        )

    # Linear trend line
    m, b = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 200)

    ax.plot(
        x_line,
        m * x_line + b,
        linestyle="--",
        color="black",
        alpha=0.7
    )

    # Correlation annotation
    ax.text(
        0.05,
        0.95,
        f"Pearson r = {pearson_r:.2f}\np = {pearson_p:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8
        }
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.grid(
        True,
        linestyle="--",
        alpha=0.4
    )

    if add_accuracy_reference_line:
        ax.axhline(
            0.5,
            linestyle="--",
            color="grey",
            alpha=0.6
        )

    patients_present = df_plot["Patient"].astype(str).unique()

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=patient,
            markerfacecolor=patient_colors[patient],
            markeredgecolor="black",
            markersize=8
        )
        for patient in patient_order
        if patient in patients_present
    ]

    ax.legend(
        handles=legend_handles,
        title="Patient",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()

    output_path = output_dir / output_filename

    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight"
    )

    plt.show()

    print(f"SVG saved to: {output_path.resolve()}")

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "slope": m,
        "intercept": b
    }


# ============================================================
# 6. Generate plots
# ============================================================

today = date.today().strftime("%Y-%m-%d")

# 6.1 Data capture rate (%) vs Test accuracy
result_1 = make_bubble_plot(
    df=final_merged_df,
    x_col="Data capture rate, %",
    y_col="Test accuracy",
    size_col="Seizure density",
    x_label="Data capture rate (%)",
    y_label="Test accuracy",
    title=(
        "Relationship between data capture rate, "
        "test accuracy, and seizure density"
    ),
    output_filename=(
        f"bubble_data_capture_test_accuracy_pearson_{today}.svg"
    ),
    add_accuracy_reference_line=True
)

# 6.2 Significant features (%) vs Test accuracy
result_2 = make_bubble_plot(
    df=final_merged_df,
    x_col="Significant features (%)",
    y_col="Test accuracy",
    size_col="Seizure density",
    x_label="Significant features (%)",
    y_label="Test accuracy",
    title=(
        "Relationship between significant features, "
        "test accuracy, and seizure density"
    ),
    output_filename=(
        f"bubble_significant_features_test_accuracy_pearson_{today}.svg"
    ),
    add_accuracy_reference_line=True
)

# 6.3 Data capture rate (%) vs Significant features (%)
result_3 = make_bubble_plot(
    df=final_merged_df,
    x_col="Data capture rate, %",
    y_col="Significant features (%)",
    size_col="Seizure density",
    x_label="Data capture rate (%)",
    y_label="Significant features (%)",
    title=(
        "Relationship between data capture rate, "
        "significant features, and seizure density"
    ),
    output_filename=(
        f"bubble_data_capture_significant_features_pearson_{today}.svg"
    ),
    add_accuracy_reference_line=False
)


# ============================================================
# 7. Multiple linear regression
# Outcome = Test accuracy
# Predictors = Data capture rate (%) + Significant features (%)
# ============================================================

df_regression = final_merged_df.copy()

regression_columns = [
    "Data capture rate, %",
    "Significant features (%)",
    "Test accuracy",
]

df_regression[regression_columns] = (
    df_regression[regression_columns]
    .apply(pd.to_numeric, errors="coerce")
)

df_regression = df_regression.dropna(
    subset=regression_columns
).copy()

# Safety check in case significant features are still 0–1
if df_regression["Significant features (%)"].max() <= 1:
    df_regression["Significant features (%)"] *= 100

X = df_regression[
    [
        "Data capture rate, %",
        "Significant features (%)",
    ]
]

X = sm.add_constant(X)

y = df_regression["Test accuracy"]

model = sm.OLS(y, X).fit()

print("\nMultiple linear regression summary:")
print(model.summary())


# ============================================================
# 8. Save regression summary
# ============================================================

regression_output_path = (
    output_dir
    / f"multiple_linear_regression_test_accuracy_{today}.txt"
)

with open(regression_output_path, "w") as f:
    f.write(model.summary().as_text())

print(f"\nRegression summary saved to: {regression_output_path.resolve()}")