import json
from pathlib import Path

# ==========================================================
# INSTRUCTIONS FOR THE USER
# ==========================================================
# This script creates a JSON configuration file ("config.json")
# that will be used as input for the EEG processing pipeline.
#
# Before running this script, the user should:
# 1. Replace the example patient information with their own patient data.
# 2. Check that all input and output paths are correct.
# 3. Adjust filtering and plotting parameters if needed.
#
# What this script does:
# - Builds a Python dictionary called "config"
# - Saves it as a JSON file named "config.json"
#
# Important:
# - Edit only the values, not the dictionary structure or key names.
# - Keep quotation marks around text values.
# - In Python, boolean values must be written as True / False
#   (not true / false).
# ==========================================================

# Define config as a Python dictionary
config = {
    # ------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------
    # Replace "XB47Y" with the patient ID you want to process.
    # This ID should match the corresponding patient folder/files.
    "patient_id": "XB47Y",

    # ------------------------------------------------------
    # FILE AND FOLDER PATHS
    # ------------------------------------------------------
    # Update these paths so they point to the correct input data
    # and desired output locations for this patient.
    "paths": {
        # Folder containing the patient's .mat EEG files
        "input_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/",

        # Excel file containing seizure annotations / seizure diary
        "seizure_file": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/XB47Y_seizures.xlsx",

        # Output path for the seizure availability map plot
        "map_output_path": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_CONFIG_24042026MAPWITHSEIZ.png",

        # Output directory where generated .npz files will be saved
        "npz_output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47YConfig_test_pipeline_24042026/",

        # Output directory where seizure-window visualizations will be saved
        "viz_output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/seizureCONFIG_pipeline_test_NOT_normalized24042026"
    },

    # ------------------------------------------------------
    # FILTERING / PREPROCESSING PARAMETERS
    # ------------------------------------------------------
    # These values control how the EEG signal is filtered
    # and preprocessed before saving the .npz outputs.
    "filtering": {
        # Amplitude threshold used to reject extreme signal values/artifacts
        "amp_threshold": 200.0,

        # Lower cutoff frequency for bandpass filtering (Hz)
        "lowcut": 0.5,

        # Upper cutoff frequency for bandpass filtering (Hz)
        "highcut": 48.0,

        # Filter order
        "order": 4,

        # Set to True if you want to z-score normalize the full recording
        # Set to False if you want to keep the signal in its filtered/raw scale
        "do_zscore": False,

        # Frequency for notch filtering (Hz), used to reduce a specific noise peak
        "notch_freq": 34.5
    },

    # ------------------------------------------------------
    # PLOTTING PARAMETERS
    # ------------------------------------------------------
    # These settings control how seizure windows are visualized.
    "plotting": {
        # If True, show plots during execution
        # If False, plots may only be saved to disk depending on the function
        "show_plot": True,

        # Index of the first EEG channel to plot
        "channel_idx_1": 0,

        # Index of the second EEG channel to plot
        "channel_idx_2": 1,

        # Window size in seconds
        "window_sec": 10,

        # Number of windows to display around the event
        "n_windows": 12,

        # Number of seconds before seizure onset to include
        "pre_onset_sec": 60,

        # Vertical offset between channels in microvolts for easier visualization
        "vertical_offset_uv": 100
    },

    # ------------------------------------------------------
    # OPTIONAL INSPECTION PARAMETERS
    # ------------------------------------------------------
    # This section is optional and is mainly useful for manual checking
    # or filtering by a specific date during exploratory analysis.
    "inspection": {
        # Example date to inspect recordings/events
        # Format must be YYYY-MM-DD
        "target_date": "2019-12-11"
    }
}

# ----------------------------------------------------------
# SAVE JSON FILE
# ----------------------------------------------------------
# Change the file name here if you want a more specific name, and select the path to save the config. 
# for example: Path("config_XB47Y.json")
config_path = Path("configs/config_XB47Y.json")

# Write the Python dictionary into a JSON file with indentation
with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

# Confirmation message
print(f"Config saved at: {config_path}")