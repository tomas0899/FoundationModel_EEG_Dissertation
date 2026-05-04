import json
from pathlib import Path
from datetime import datetime

# ==========================================================
# JSON CONFIG GENERATOR FOR EEG PIPELINE
# ==========================================================
# This script creates a JSON configuration file for the EEG
# processing pipeline.
#
# The user only needs to define:
# - patient information
# - input paths
# - output root directory
# - filtering parameters
# - plotting parameters
#
# Output file names and folders are generated automatically
# using:
# - patient_id
# - input folder name
# - amplitude cutoff
# - bandpass filter
# - notch filter
# - z-score setting
# - current date
# ==========================================================


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def clean_number_for_name(x):
    """
    Convert numbers into filename-safe strings.
    Example:
    0.5  -> 0p5
    48.0 -> 48
    34.5 -> 34p5
    """
    if x is None:
        return "None"

    x = float(x)

    if x.is_integer():
        return str(int(x))

    return str(x).replace(".", "p")


def build_experiment_id(
    patient_id,
    input_dir,
    amp_threshold,
    lowcut,
    highcut,
    notch_freq,
    do_zscore,
    date_code
):
    """
    Build a reproducible experiment ID based on input data and parameters.
    """

    input_name = Path(input_dir).name

    amp_code = f"AMP{clean_number_for_name(amp_threshold)}"
    bp_code = f"BP{clean_number_for_name(lowcut)}-{clean_number_for_name(highcut)}Hz"

    if notch_freq is None:
        notch_code = "NO-NOTCH"
    else:
        notch_code = f"NOTCH{clean_number_for_name(notch_freq)}Hz"

    zscore_code = "ZSCORE" if do_zscore else "NOZSCORE"

    experiment_id = (
        f"{patient_id}_"
        f"IN-{input_name}_"
        f"{amp_code}_"
        f"{bp_code}_"
        f"{notch_code}_"
        f"{zscore_code}_"
        f"{date_code}"
    )

    return experiment_id


# ==========================================================
# USER-DEFINED SETTINGS
# ==========================================================

# ----------------------------------------------------------
# PATIENT INFORMATION
# ----------------------------------------------------------
patient_id = "XB47Y"

# ----------------------------------------------------------
# INPUT PATHS
# ----------------------------------------------------------
input_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/"

seizure_file = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/XB47Y_seizures.xlsx"

# ----------------------------------------------------------
# USER-DEFINED OUTPUT ROOT DIRECTORY
# ----------------------------------------------------------
# The user chooses ONLY this folder.
# The script will automatically create a subfolder inside it
# using the generated experiment_id.
output_root_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Pre_processing"

# ----------------------------------------------------------
# CONFIG OUTPUT DIRECTORY
# ----------------------------------------------------------
# Folder where the generated JSON config will be saved.
config_output_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/src/pipelines/1_Pre_process/configs"


# ==========================================================
# FILTERING / PREPROCESSING PARAMETERS
# ==========================================================

filtering = {
    "amp_threshold": 200.0,
    "lowcut": 0.5,
    "highcut": 48.0,
    "order": 4,
    "do_zscore": False,
    "notch_freq": 34.5
}


# ==========================================================
# PLOTTING PARAMETERS
# ==========================================================

plotting = {
    "show_plot": True,
    "channel_idx_1": 0,
    "channel_idx_2": 1,
    "window_sec": 10,
    "n_windows": 12,
    "pre_onset_sec": 60,
    "vertical_offset_uv": 100
}


# ==========================================================
# OPTIONAL INSPECTION PARAMETERS
# ==========================================================

inspection = {
    "target_date": "2019-12-11"
}


# ==========================================================
# AUTOMATIC NAMING
# ==========================================================

date_code = datetime.now().strftime("%Y%m%d")
created_at = datetime.now().isoformat(timespec="seconds")

experiment_id = build_experiment_id(
    patient_id=patient_id,
    input_dir=input_dir,
    amp_threshold=filtering["amp_threshold"],
    lowcut=filtering["lowcut"],
    highcut=filtering["highcut"],
    notch_freq=filtering["notch_freq"],
    do_zscore=filtering["do_zscore"],
    date_code=date_code
)

# Main experiment output directory
experiment_output_dir = Path(output_root_dir) / experiment_id

# Specific output directories
npz_output_dir = experiment_output_dir / "npz"
viz_output_dir = experiment_output_dir / "visualizations"
maps_output_dir = experiment_output_dir / "maps"

# Create directories
experiment_output_dir.mkdir(parents=True, exist_ok=True)
npz_output_dir.mkdir(parents=True, exist_ok=True)
viz_output_dir.mkdir(parents=True, exist_ok=True)
maps_output_dir.mkdir(parents=True, exist_ok=True)

# Automatically generated output files
map_output_path = maps_output_dir / f"{experiment_id}_seizure_availability_map.png"


# ==========================================================
# DEFINE CONFIG DICTIONARY
# ==========================================================

config = {
    # ------------------------------------------------------
    # EXPERIMENT INFORMATION
    # ------------------------------------------------------
    "experiment_id": experiment_id,
    "created_at": created_at,
    "date_code": date_code,

    # ------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------
    "patient_id": patient_id,

    # ------------------------------------------------------
    # FILE AND FOLDER PATHS
    # ------------------------------------------------------
    "paths": {
        "input_dir": str(input_dir),
        "seizure_file": str(seizure_file),

        "output_root_dir": str(output_root_dir),
        "experiment_output_dir": str(experiment_output_dir),

        "map_output_path": str(map_output_path),
        "npz_output_dir": str(npz_output_dir),
        "viz_output_dir": str(viz_output_dir)
    },

    # ------------------------------------------------------
    # AUTOMATIC NAMING INFORMATION
    # ------------------------------------------------------
    "naming": {
        "input_name": Path(input_dir).name,
        "amp_code": f"AMP{clean_number_for_name(filtering['amp_threshold'])}",
        "bandpass_code": f"BP{clean_number_for_name(filtering['lowcut'])}-{clean_number_for_name(filtering['highcut'])}Hz",
        "notch_code": (
            "NO-NOTCH"
            if filtering["notch_freq"] is None
            else f"NOTCH{clean_number_for_name(filtering['notch_freq'])}Hz"
        ),
        "zscore_code": "ZSCORE" if filtering["do_zscore"] else "NOZSCORE"
    },

    # ------------------------------------------------------
    # FILTERING / PREPROCESSING PARAMETERS
    # ------------------------------------------------------
    "filtering": filtering,

    # ------------------------------------------------------
    # PLOTTING PARAMETERS
    # ------------------------------------------------------
    "plotting": plotting,

    # ------------------------------------------------------
    # OPTIONAL INSPECTION PARAMETERS
    # ------------------------------------------------------
    "inspection": inspection
}


# ==========================================================
# SAVE JSON CONFIG FILE
# ==========================================================

config_output_dir = Path(config_output_dir)
config_output_dir.mkdir(parents=True, exist_ok=True)

config_path = config_output_dir / f"config_{experiment_id}.json"

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print("Config saved at:")
print(config_path)

print("\nExperiment ID:")
print(experiment_id)

print("\nExperiment output directory:")
print(experiment_output_dir)

print("\nGenerated output paths:")
print(f"Map output: {map_output_path}")
print(f"NPZ output dir: {npz_output_dir}")
print(f"Visualization output dir: {viz_output_dir}")