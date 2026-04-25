import json
from pathlib import Path

# ==========================================================
# INSTRUCTIONS FOR THE USER
# ==========================================================
# This script creates a JSON configuration file for the EEG
# global channel normalization pipeline.
#
# Before running this script, the user should:
# 1. Replace the patient ID with the correct patient.
# 2. Check that the input folder contains the .npz files.
# 3. Check that the output folder is where normalized .npz files
#    should be saved.
# 4. Adjust validation parameters if needed.
# ==========================================================

config = {
    # ------------------------------------------------------
    # PATIENT INFORMATION
    # ------------------------------------------------------
    "patient_id": "XB47Y",

    # ------------------------------------------------------
    # FILE AND FOLDER PATHS
    # ------------------------------------------------------
    "paths": {
        # Folder containing the original/non-normalized .npz files
        "input_npz_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_testALL_25032026/",

        # Example .npz file used only for inspection
        "example_npz_file": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_testALL_25032026/XB47Y_182_preproc_full.npz",

        # Output folder where globally normalized .npz files will be saved
        "normalized_output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/"
    },

    # ------------------------------------------------------
    # FILE SELECTION PARAMETERS
    # ------------------------------------------------------
    "file_selection": {
        # Pattern used to find input files inside input_npz_dir
        "file_pattern": "*.npz"
    },

    # ------------------------------------------------------
    # NORMALIZATION PARAMETERS
    # ------------------------------------------------------
    "normalization": {
        # Expected number of EEG channels
        "expected_n_channels": 2,

        # Small value used to avoid division by zero
        "eps": 1e-8,

        # Whether np.load should allow pickle objects
        "allow_pickle": True,

        # Metadata keys that should be preserved in the normalized .npz files
        "metadata_keys": [
            "mu",
            "sigma",
            "fs",
            "channel_names",
            "source_file",
            "seizure_onsets",
            "T0",
            "TF"
        ]
    },

    # ------------------------------------------------------
    # PIPELINE STEPS
    # ------------------------------------------------------
    "pipeline_steps": {
        # Inspect one example .npz file before processing all files
        "inspect_example_file": True,

        # Compute global mean/std per channel across all recordings
        "compute_global_channel_stats": True,

        # Apply global channel normalization to each recording
        "apply_global_normalization": True,

        # Run sanity check comparing original vs normalized folders
        "run_sanity_check": True
    }
}

# ----------------------------------------------------------
# SAVE JSON FILE
# ----------------------------------------------------------
config_path = Path("configs/config_XB47Y_global_normalization.json")
config_path.parent.mkdir(parents=True, exist_ok=True)

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Config saved at: {config_path}")