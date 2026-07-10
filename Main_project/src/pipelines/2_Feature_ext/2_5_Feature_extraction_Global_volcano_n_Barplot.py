from pathlib import Path
from datetime import date
from matplotlib import font_manager
import sys
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import re





project_root = Path("/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project")
# ============================================================
# Output directory
# ============================================================

output_dir = (
    project_root /
    "results" /
    "together_results" /
    "summary_volcano_barplot"
)

output_dir.mkdir(parents=True, exist_ok=True)
src_dir = project_root / "src" / "modules"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import tools_EEG_FE as TEEG_FE

importlib.reload(TEEG_FE)



available_fonts = {f.name for f in font_manager.fontManager.ttflist}

if "Arial" in available_fonts:
    chosen_font = "Arial"
elif "Liberation Sans" in available_fonts:
    chosen_font = "Liberation Sans"   
else:
    chosen_font = "DejaVu Sans"      

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [chosen_font],
    "axes.unicode_minus": False
})

print(f"Using font: {chosen_font}")




def load_all_patient_mannwhitney_results(
    root_dir,
    filename_pattern="*.pkl",
    add_source_file=True,
    verbose=True
):
    root_dir = Path(root_dir)

    required_columns = [
        "patient_id",
        "feature",
        "n_preictal",
        "n_ictal",
        "mannwhitney_U_preictal",
        "p_value",
        "neg_log10_p",
        "effect_size_log2_median_fold_change",
        "abs_effect_size_log2_median_fold_change",
        "fold_change_median_ictal_over_preictal",
        "valid_log2_fold_change",
        "median_preictal",
        "median_ictal",
        "mean_preictal",
        "mean_ictal",
        "median_difference_ictal_minus_preictal",
        "mean_difference_ictal_minus_preictal",
        "feature_subcategory",
        "feature_color",
        "result",
    ]

    pkl_files = sorted(root_dir.rglob(filename_pattern))

    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No pkl files found in: {root_dir}")

    dfs = []
    skipped_files = []

    for pkl_path in pkl_files:
        try:
            df = pd.read_pickle(pkl_path)

            if not isinstance(df, pd.DataFrame):
                skipped_files.append((pkl_path, "Not a dataframe"))
                continue

            missing_cols = [col for col in required_columns if col not in df.columns]

            if missing_cols:
                skipped_files.append((pkl_path, f"Missing columns: {missing_cols}"))
                continue

            df = df.copy()

            if add_source_file:
                df["source_file"] = str(pkl_path)

            dfs.append(df)

        except Exception as e:
            skipped_files.append((pkl_path, str(e)))

    if verbose:
        print(f"Found {len(pkl_files)} pkl files")
        print(f"Loaded {len(dfs)} valid dataframe files")
        print(f"Skipped {len(skipped_files)} files")

        if skipped_files:
            print("\nSkipped files:")
            for path, reason in skipped_files:
                print(f"- {path.name}: {reason}")

    if len(dfs) == 0:
        raise ValueError("No valid dataframes were loaded.")

    df_all = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\nCombined dataframe shape: {df_all.shape}")
        print(f"Patients loaded: {df_all['patient_id'].nunique()}")

    return df_all
root_dir = "/home/tperezsanchez/Tomas_PS_DissertationKCL2026/Main_project/results/together_results/pvalue_effectsize"

df_all_mannwhitney = load_all_patient_mannwhitney_results(
    root_dir=root_dir,
    filename_pattern="*.pkl"
)

df_all_mannwhitney.head()
# ============================================================
# Volcano plot 1
# Global volcano plot using all patient-feature observations
# ============================================================
alpha = 0.05

subcategory_palette = TEEG_FE.get_feature_subcategory_palette()

df_global_volcano = df_all_mannwhitney.query(
    "valid_log2_fold_change == True"
).copy()

# Output path: current notebook working directory + today's date
today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"global_volcano_plot_{today}.svg"
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df_global_volcano,
    x="effect_size_log2_median_fold_change",
    y="neg_log10_p",
    hue="feature_subcategory",
    palette=subcategory_palette,
    s=70,
    edgecolor="black",
    linewidth=0.4,
    alpha=0.8
)

plt.axvline(0, linestyle="--", color="black", linewidth=1)

# Thresholds at log2FC = -1 and +1
plt.axvline(-1, linestyle="--", color="red", linewidth=1)
plt.axvline(1, linestyle="--", color="red", linewidth=1)

plt.axhline(-np.log10(alpha), linestyle="--", color="grey", linewidth=1)

plt.xlabel("log2 median fold-change")
plt.ylabel("-log10(p-value)")
plt.title("Global volcano plot: feature discriminability across patients")

plt.legend(
    title="Feature subcategory",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")
# ============================================================
# Volcano plot 2
# Global volcano plot using median feature discriminability
# across patients
# ============================================================

alpha = 0.05

subcategory_palette = TEEG_FE.get_feature_subcategory_palette()

df_feature_global = (
    df_all_mannwhitney
    .query("valid_log2_fold_change == True")
    .groupby(["feature", "feature_subcategory", "feature_color"], as_index=False)
    .agg(
        median_log2_effect_size=("effect_size_log2_median_fold_change", "median"),
        median_neg_log10_p=("neg_log10_p", "median"),
        mean_neg_log10_p=("neg_log10_p", "mean"),
        n_patients=("patient_id", "nunique")
    )
)

# Output path: current notebook working directory + today's date
today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"global_volcano_median_feature_discriminability_{today}.svg"
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df_feature_global,
    x="median_log2_effect_size",
    y="median_neg_log10_p",
    hue="feature_subcategory",
    palette=subcategory_palette,
    s=90,
    edgecolor="black",
    linewidth=0.4,
    alpha=0.85
)

# Reference line at 0
plt.axvline(0, linestyle="--", color="black", linewidth=1)

# Thresholds at log2FC = -1 and +1
plt.axvline(-1, linestyle="--", color="red", linewidth=1)
plt.axvline(1, linestyle="--", color="red", linewidth=1)

# Significance threshold
plt.axhline(-np.log10(alpha), linestyle="--", color="grey", linewidth=1)

plt.xlabel("Median log2 median fold-change across patients")
plt.ylabel("Median -log10(p-value) across patients")
plt.title("Global volcano plot: median feature discriminability across patients")

plt.legend(
    title="Feature subcategory",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

# Save as SVG
plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")



# ============================================================
# Bar plot
# Top discriminative features ranked by median significance
# ============================================================
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 16
TICK_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 18
LEGEND_TITLE_FONTSIZE = 16

def format_feature_label_with_channel(feature):
    feature = str(feature)

    match = re.match(r"^(?P<base>.+?)_(?P<channel>EEG_.+)$", feature)

    if match:
        base = match.group("base")
        channel = match.group("channel")

        channel_map = {
            "EEG_SQ_P_SQ_C": "Ch-1",
            "EEG_SQ_D_SQ_C": "Ch-2"
        }

        channel_display = channel_map.get(channel, channel)
        base_display = base.replace("_", " ").capitalize()

        return f"{base_display} [{channel_display}]"

    return feature.replace("_", " ").capitalize()


top_n = 20

df_valid = (
    df_all_mannwhitney
    .query("valid_log2_fold_change == True")
    .copy()
)

df_feature_summary = (
    df_valid
    .groupby(["feature", "feature_subcategory", "feature_color"], as_index=False)
    .agg(
        median_log2_effect_size=("effect_size_log2_median_fold_change", "median"),
        median_neg_log10_p=("neg_log10_p", "median"),
        n_patients=("patient_id", "nunique")
    )
)

df_bar = (
    df_feature_summary
    .sort_values("median_neg_log10_p", ascending=False)
    .head(top_n)
    .copy()
)

df_bar["feature_display"] = (
    df_bar["feature"]
    .apply(format_feature_label_with_channel)
)

df_bar = (
    df_bar
    .sort_values("median_neg_log10_p", ascending=True)
    .reset_index(drop=True)
)

# Output path
today = date.today().strftime("%Y-%m-%d")
output_path = output_dir / f"top_{top_n}_global_features_barplot_{today}.svg"
fig, ax = plt.subplots(figsize=(19, 0.65 * len(df_bar) + 3))

y_pos = np.arange(len(df_bar))

ax.barh(
    y=y_pos,
    width=df_bar["median_log2_effect_size"],
    color=df_bar["feature_color"],
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85
)

ax.axvline(0, linestyle="--", color="black", linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(
    df_bar["feature_display"],
    fontsize=TICK_LABEL_FONTSIZE
)

ax.tick_params(
    axis="x",
    labelsize=TICK_LABEL_FONTSIZE
)

ax.set_xlabel(
    "Median log2 fold-change across patients",
    fontsize=AXIS_LABEL_FONTSIZE
)

ax.set_ylabel(
    "Feature",
    fontsize=AXIS_LABEL_FONTSIZE
)

ax.set_title(
    f"Top {top_n} global features ordered by median -log10(p-value)",
    fontsize=TITLE_FONTSIZE,
    pad=14
)

subcategory_palette = (
    df_bar
    .drop_duplicates("feature_subcategory")
    .set_index("feature_subcategory")["feature_color"]
    .to_dict()
)

legend_patches = [
    mpatches.Patch(color=color, label=subcategory)
    for subcategory, color in subcategory_palette.items()
]

legend = ax.legend(
    handles=legend_patches,
    title="Feature subcategory",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=LEGEND_FONTSIZE,
    title_fontsize=LEGEND_TITLE_FONTSIZE
)

plt.tight_layout()

plt.savefig(output_path, format="svg", bbox_inches="tight")



print(f"SVG saved in: {output_path}")