from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from datetime import date

# ============================================================
# 1. Import CSVs and merge dataframes
# ============================================================

input_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/MRA"
)

output_dir = Path(
    "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/MRA"
)

output_dir.mkdir(parents=True, exist_ok=True)

df_characteristics = pd.read_csv(input_dir / "dataset_characteristics.csv")
df_performance = pd.read_csv(input_dir / "model_performance.csv")

df_merged = pd.merge(
    df_characteristics,
    df_performance,
    on="Patient ID",
    how="inner"
)

df_merged



# ============================================================
# Bubble scatter plot
# X = Data capture rate (%)
# Y = Weighted F1
# Bubble size = Seizure density
# Colour = Patient number
# Label = Patient number
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

patient_order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

df_plot = df_merged.copy()

# Replace original patient IDs with assigned patient numbers
df_plot["Patient"] = df_plot["Patient ID"].map(patient_number_map)

# Optional check: identify unmapped patients
unmapped_patients = df_plot.loc[df_plot["Patient"].isna(), "Patient ID"].unique()
if len(unmapped_patients) > 0:
    print("Warning: these Patient IDs were not mapped:", unmapped_patients)

# Sort by assigned patient number
df_plot["Patient"] = pd.Categorical(
    df_plot["Patient"],
    categories=patient_order,
    ordered=True
)

df_plot = df_plot.sort_values("Patient")

CB_color_cycle = [
    '#2E0F4F',  # P1 - very dark violet
    '#FF5F00',  # P2 - very bright orange
    '#1B7837',  # P3 - dark green
    '#f781bf',  # P4 - pink
    '#a65628',  # P5 - brown
    '#984ea3',  # P6 - purple
    '#999999',  # P7 - grey
    '#e41a1c',  # P8 - red
    '#dede00'   # P9 - yellow
]
patient_colors = {
    patient: CB_color_cycle[i]
    for i, patient in enumerate(patient_order)
}

# Bubble size scaling
density = df_plot["Seizure density"]

if density.max() == density.min():
    bubble_sizes = pd.Series(700, index=df_plot.index)
else:
    bubble_sizes = 200 + (
        (density - density.min()) / (density.max() - density.min())
    ) * 1200

plt.figure(figsize=(10, 7))

for i, row in df_plot.iterrows():
    plt.scatter(
        row["Data capture rate, %"],
        row["Weighted F1"],
        s=bubble_sizes.loc[i],
        color=patient_colors[row["Patient"]],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8
    )

    plt.text(
        row["Data capture rate, %"] + 0.8,
        row["Weighted F1"] + 0.005,
        row["Patient"],
        fontsize=9,
        fontweight="bold"
    )

plt.xlabel("Data capture rate (%)")
plt.ylabel("Weighted F1")
plt.title("Relationship between data capture rate, weighted F1, and seizure density")

plt.grid(True, linestyle="--", alpha=0.4)

# Optional reference line
plt.axhline(0.5, linestyle="--", color="grey", alpha=0.6)

# Colour legend
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
]

plt.legend(
    handles=legend_handles,
    title="Patient",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True
)

plt.tight_layout()

# Save as SVG in current notebook folder
today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"bubble_data_capture_weighted_f1_seizure_density_CB_palette_{today}.svg"

plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import date

# ============================================================
# Bubble scatter plot
# X = Data capture rate (%)
# Y = Weighted F1
# Bubble size = Seizure density
# Colour = Patient number
# Label = Patient number
# + Pearson correlation
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

patient_order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

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

df_plot = df_merged.copy()

# Keep only rows needed for this plot/correlation
df_plot = df_plot.dropna(
    subset=["Patient ID", "Data capture rate, %", "Weighted F1", "Seizure density"]
)

# Replace original patient IDs with assigned patient numbers
df_plot["Patient"] = df_plot["Patient ID"].map(patient_number_map)

# Optional check: identify unmapped patients
unmapped_patients = df_plot.loc[df_plot["Patient"].isna(), "Patient ID"].unique()
if len(unmapped_patients) > 0:
    print("Warning: these Patient IDs were not mapped:", unmapped_patients)

# Remove unmapped patients if any
df_plot = df_plot.dropna(subset=["Patient"])

# Sort by assigned patient number
df_plot["Patient"] = pd.Categorical(
    df_plot["Patient"],
    categories=patient_order,
    ordered=True
)

df_plot = df_plot.sort_values("Patient")

# ============================================================
# Correlation: Data capture rate vs Weighted F1
# ============================================================

x = df_plot["Data capture rate, %"]
y = df_plot["Weighted F1"]

pearson_r, pearson_p = pearsonr(x, y)

print("Pearson correlation between data capture rate and Weighted F1:")
print(f"r = {pearson_r:.3f}, p = {pearson_p:.3f}")

# ============================================================
# Bubble size scaling
# matplotlib 's' is area, not radius
# ============================================================

density = df_plot["Seizure density"]

if density.max() == density.min():
    bubble_sizes = pd.Series(600, index=df_plot.index)
else:
    bubble_sizes = 200 + (
        (density - density.min()) / (density.max() - density.min())
    ) * 1200

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 7))

for i, row in df_plot.iterrows():
    plt.scatter(
        row["Data capture rate, %"],
        row["Weighted F1"],
        s=bubble_sizes.loc[i],
        color=patient_colors[row["Patient"]],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8
    )

    plt.text(
        row["Data capture rate, %"] + 0.8,
        row["Weighted F1"] + 0.005,
        row["Patient"],
        fontsize=9,
        fontweight="bold"
    )

# ============================================================
# Linear trend line
# ============================================================

m, b = np.polyfit(x, y, 1)

plt.plot(
    x,
    m * x + b,
    linestyle="--",
    color="black",
    alpha=0.7,
    label="Linear trend"
)

# ============================================================
# Correlation annotation inside plot
# ============================================================

plt.text(
    0.05,
    0.95,
    f"Pearson r = {pearson_r:.2f}\np = {pearson_p:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.xlabel("Data capture rate (%)")
plt.ylabel("Weighted F1")
plt.title("Relationship between data capture rate, weighted F1, and seizure density")

plt.grid(True, linestyle="--", alpha=0.4)

# Optional reference line
plt.axhline(0.5, linestyle="--", color="grey", alpha=0.6)

# ============================================================
# Colour legend
# ============================================================

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
]

plt.legend(
    handles=legend_handles,
    title="Patient",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True
)

plt.tight_layout()

# ============================================================
# Save as SVG in current notebook folder
# ============================================================

today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"bubble_data_capture_weighted_f1_pearson_CB_palette_{today}.svg"

plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import date

# ============================================================
# Bubble scatter plot
# X = Data capture rate (%)
# Y = Best accuracy
# Bubble size = Seizure density
# Colour = Patient number
# Label = Patient number
# + Pearson correlation
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

patient_order = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

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

df_plot = df_merged.copy()

# Keep only rows needed for this plot/correlation
df_plot = df_plot.dropna(
    subset=["Patient ID", "Data capture rate, %", "Best accuracy", "Seizure density"]
)

# Replace original patient IDs with assigned patient numbers
df_plot["Patient"] = df_plot["Patient ID"].map(patient_number_map)

# Optional check: identify unmapped patients
unmapped_patients = df_plot.loc[df_plot["Patient"].isna(), "Patient ID"].unique()
if len(unmapped_patients) > 0:
    print("Warning: these Patient IDs were not mapped:", unmapped_patients)

# Remove unmapped patients if any
df_plot = df_plot.dropna(subset=["Patient"])

# Sort by assigned patient number
df_plot["Patient"] = pd.Categorical(
    df_plot["Patient"],
    categories=patient_order,
    ordered=True
)

df_plot = df_plot.sort_values("Patient")

# ============================================================
# Correlation: Data capture rate vs Best accuracy
# ============================================================

x = df_plot["Data capture rate, %"]
y = df_plot["Best accuracy"]

pearson_r, pearson_p = pearsonr(x, y)

print("Pearson correlation between data capture rate and Best accuracy:")
print(f"r = {pearson_r:.3f}, p = {pearson_p:.3f}")

# ============================================================
# Bubble size scaling
# matplotlib 's' is area, not radius
# ============================================================

density = df_plot["Seizure density"]

if density.max() == density.min():
    bubble_sizes = pd.Series(600, index=df_plot.index)
else:
    bubble_sizes = 200 + (
        (density - density.min()) / (density.max() - density.min())
    ) * 1200

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(10, 7))

for i, row in df_plot.iterrows():
    plt.scatter(
        row["Data capture rate, %"],
        row["Best accuracy"],
        s=bubble_sizes.loc[i],
        color=patient_colors[row["Patient"]],
        alpha=0.78,
        edgecolor="black",
        linewidth=0.8
    )

    plt.text(
        row["Data capture rate, %"] + 0.8,
        row["Best accuracy"] + 0.005,
        row["Patient"],
        fontsize=9,
        fontweight="bold"
    )

# ============================================================
# Linear trend line
# ============================================================

m, b = np.polyfit(x, y, 1)

plt.plot(
    x,
    m * x + b,
    linestyle="--",
    color="black",
    alpha=0.7
)

# ============================================================
# Correlation annotation inside plot
# ============================================================

plt.text(
    0.05,
    0.95,
    f"Pearson r = {pearson_r:.2f}\np = {pearson_p:.3f}",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.xlabel("Data capture rate (%)")
plt.ylabel("Best accuracy")
plt.title("Relationship between data capture rate, best accuracy, and seizure density")

plt.grid(True, linestyle="--", alpha=0.4)

# Optional reference line
plt.axhline(0.5, linestyle="--", color="grey", alpha=0.6)

# ============================================================
# Colour legend
# ============================================================

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
]

plt.legend(
    handles=legend_handles,
    title="Patient",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True
)

plt.tight_layout()

# ============================================================
# Save as SVG in current notebook folder
# ============================================================

today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"bubble_data_capture_best_accuracy_pearson_CB_palette_{today}.svg"

plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")

import statsmodels.api as sm

X = df_plot[["Data capture rate, %", "prop_significant_features"]]
X = sm.add_constant(X)

y = df_plot["Best accuracy"]

model = sm.OLS(y, X).fit()
print(model.summary())