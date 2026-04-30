import json
from pathlib import Path

# ==========================================================
# INSTRUCTIONS FOR THE USER
# ==========================================================
# This script creates a JSON configuration file for the EEG
# windowing + labeling pipeline.
#
# Before running this script, the user should:
# 1. Replace the patient ID with the correct patient.
# 2. Check that the input folder contains the normalized .npz files.
# 3. Check that the output folder is correct.
# 4. Adjust window size and labeling ranges if needed.
# ==========================================================


def create_labeling_config():

    config = {
        # ------------------------------------------------------
        # PATIENT INFORMATION
        # ------------------------------------------------------
        "patient_id": "XB47Y",

        # ------------------------------------------------------
        # FILE AND FOLDER PATHS
        # ------------------------------------------------------
        "paths": {
            # Folder containing normalized .npz files
            "input_npz_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/",

            # Output folder where pickles will be saved
            "output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part1_labeling"
        },

        # ------------------------------------------------------
        # FILE SELECTION PARAMETERS
        # ------------------------------------------------------
        "file_selection": {
            # Pattern used to find input files
            "file_pattern": "*full.npz"
        },

        # ------------------------------------------------------
        # WINDOWING PARAMETERS
        # ------------------------------------------------------
        "windowing": {
            # Window size in seconds
            "window_sec": 10
        },

        # ------------------------------------------------------
        # LABELING PARAMETERS
        # ------------------------------------------------------
        "labeling": {
            # Preictal window range (in minutes before seizure)
            "preictal_range_min": [-6, -5],

            # Ictal window range (in minutes after seizure onset)
            "ictal_range_min": [0, 1],

            # Whether to label gaps as interictal (0)
            "include_gap_as_interictal": True
        },

        # ------------------------------------------------------
        # FILTERING PARAMETERS
        # ------------------------------------------------------
        "filtering": {
            # Keep only preictal (1) and seizure (2)
            "keep_only_preictal_seizure": True
        },

        # ------------------------------------------------------
        # OUTPUT FILES
        # ------------------------------------------------------
        "output_files": {
            "df_labeled_filename": "df_labeled_all_1min.pkl",
            "df_final_filename": "df_ictal_Vs_Preictal_1min.pkl"
        },

        # ------------------------------------------------------
        # PIPELINE STEPS
        # ------------------------------------------------------
        "pipeline_steps": {
            # Run metadata extraction
            "load_metadata": True,

            # Run temporal sanity checks
            "temporal_sanity_check": True,

            # Clean seizure onsets
            "clean_onsets": True,

            # Create windows
            "run_windowing": True,

            # Apply labeling
            "run_labeling": True,

            # Apply filtering
            "run_filtering": True,

            # Save outputs
            "save_outputs": True
        }
    }

    # ----------------------------------------------------------
    # SAVE JSON FILE
    # ----------------------------------------------------------
    config_path = Path("configs/config_XB47Y_FE_Part1_labeling_1min30042026.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved at: {config_path.resolve()}")


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    create_labeling_config()

#how to run script:
# python 2_1_Feature_ext....py configs/config_XB47Y_labeling.json