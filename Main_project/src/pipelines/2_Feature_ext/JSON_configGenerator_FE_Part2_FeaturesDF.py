import json
from pathlib import Path

# ==========================================================
# INSTRUCTIONS FOR THE USER
# ==========================================================
# This script creates a JSON configuration file for the EEG
# feature extraction pipeline.
#
# Before running this script, the user should:
# 1. Replace the patient ID with the correct patient.
# 2. Check that the input pickle files exist.
# 3. Check that the NPZ folder contains the normalized .npz files.
# 4. Check that the output folder is correct.
# ==========================================================


def create_feature_extraction_config():

    config = {
        # ------------------------------------------------------
        # PATIENT INFORMATION
        # ------------------------------------------------------
        "patient_id": "XB47Y",

        # ------------------------------------------------------
        # FILE AND FOLDER PATHS
        # ------------------------------------------------------
        "paths": {
            # Folder containing the labeled pickle files
            "input_pickle_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part1_labeling",

            # Folder containing the normalized/preprocessed NPZ files
            "input_npz_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/",

            # Output folder where feature pickle files will be saved
            "output_dir": "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part2_features"
        },

        # ------------------------------------------------------
        # INPUT FILES
        # ------------------------------------------------------
        "input_files": {
            "df_labeled_all": "df_labeled_all.pkl",
            "df_ictalVspreictal": "df_ictal_Vs_Preictal.pkl"
        },

        # ------------------------------------------------------
        # FEATURE EXTRACTION PARAMETERS
        # ------------------------------------------------------
        "feature_extraction": {
            # Whether to extract features from all labeled windows
            "extract_features_all_windows": True,

            # Whether to extract features only from preictal vs seizure windows
            "extract_features_ictalVspreictal": True,

            # Whether to use file cache when loading NPZ files
            "use_file_cache": True,

            # Whether np.load should allow pickle objects
            "allow_pickle": True
        },

        # ------------------------------------------------------
        # OUTPUT FILES
        # ------------------------------------------------------
        "output_files": {
            "df_features_all": "df_features_all.pkl",
            "df_features_ictalVspreictal": "df_features_ictal_Vs_Preictal.pkl"
        },

        # ------------------------------------------------------
        # PIPELINE STEPS
        # ------------------------------------------------------
        "pipeline_steps": {
            # Load input pickle files
            "load_input_pickles": True,

            # Extract features for ictal vs preictal dataframe
            "run_features_ictalVspreictal": True,

            # Extract features for all labeled windows
            "run_features_all": True,

            # Save output feature dataframes
            "save_outputs": True
        }
    }

    # ----------------------------------------------------------
    # SAVE JSON FILE
    # ----------------------------------------------------------
    config_path = Path("configs/config_XB47Y_feature_extraction.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved at: {config_path.resolve()}")


# ==========================================================
# RUN SCRIPT
# ==========================================================
if __name__ == "__main__":
    create_feature_extraction_config()