from pathlib import Path
import sys

# Get current file location
current_file = Path(__file__).resolve()

# Go up until you find the project root (where "src" exists)
for parent in current_file.parents:
    if (parent / "src").exists():
        project_root = parent
        break

# Add to PYTHONPATH if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import works
from src.modules import tools_EEG as TEEG

#==========================
#==========================
#==========================
# 1. Open/load .mat files
# open all the .mat files from the folder of a single patient
# create a df with all the information
path = "//home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/"
df_patient, error_list = TEEG.process_eeg_mat_files_1_1(path)
#print(df_patient.head())
#==========================
#==========================
#==========================
# 1.2 Visualize distribution of daily accumulated recording time
TEEG.plot_daily_recording_histogram_1_2(df_patient, patient_id="XB47Y")

#==========================
#==========================
#==========================
# 1.3 Gather seizure data from CSV
file_path = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/XB47Y_seizures.xlsx"
df_sq, df_di = TEEG.preprocess_seizure_data_1_3(file_path)
#==========================
#==========================
#==========================
# 1.4 Mapping of seizures in all mat files
df_matches = TEEG.plot_eeg_availability_with_onsetsV2_1_4(
    df_files=df_patient, 
    df_onsets=df_sq, 
    output_path="/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_MAPWITHSEIZ.png",
    show_plot=True
)
#search for the mat file with the onset on the 2019-12-11
import pandas as pd

# 1. Ensure T0 is in datetime format (just in case)
df_matches['T0'] = pd.to_datetime(df_matches['T0'])

# 2. Filter by comparing only the date part (.dt.date)
# Note: both pd.Timestamp or datetime.date objects work for this comparison
target_date = pd.to_datetime("2019-12-11").date()
df_filtered = df_matches[df_matches['T0'].dt.date == target_date]

# Show results
#print(f"Found {len(df_filtered)} records for the date: {target_date}")
#display(df_matches.head())
df_matches
#==========================
#==========================
#==========================
# 1.5 Merge both Df 
# Step 1: Create a reduced version of df_matches with only relevant columns
# This does NOT modify df_matches; it creates a new DataFrame
df_match_small = df_matches[["file", "onset", "captured"]]

# Step 2: Merge df_patient with df_match_small using a LEFT JOIN on "file"
# - Keeps ALL rows from df_patient
# - Adds "onset" and "captured" where a match is found
# - If no match exists, NaN values are assigned
# - Result is stored in a new DataFrame (df_merged), original DataFrames remain unchanged
df_merged = df_patient.merge(df_match_small, on="file", how="left")

# Final result: df_merged contains EEG data + matched clinical/event metadata
df_merged

# checking that the files with onset are repeated. as they should be, because the 
#print(df_matches.columns.tolist())
#print(df_matches[df_matches["file"] == "patients.mat"])

# checking that the files with onset are repeated. as they should be, because the df should keep the onset records
#print(df_merged.columns.tolist())
#print(df_merged[df_merged["file"] == "XB47Y_182.mat"])
#==========================
#==========================
#==========================

# 1.6 GET LIST FOR UNIQUE MATCH
# PRINT ALL THE MAT FILES THAT HAVE A PRESENCE OF A SEIZURE
#df_matches
df_Unique_match = df_matches['file'].unique()
df_Unique_match
#print(type(df_Unique_match))
#for file in df_Unique_match:
#    print(file)
list_Unique_match = df_Unique_match.tolist()
#print(type(list_Unique_match))
# unique list from all the mat files
# this is the input for my function to get the npz
files_to_process = sorted(df_patient["file"].dropna().astype(str).unique().tolist())
#print(files_to_process)
#==========================
#==========================
#==========================

# 1.7 GENERATE ALL .NPZ FILES
import os

input_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/data/Working/XB47Y/"
output_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_test_pipeline_24042026/"
os.makedirs(output_dir, exist_ok=True)
#Output .npz contains:
 # X:              (C, N) full z-scored signal
 # mu:             (C,)   mean per channel
 # sigma:          (C,)   standard deviation per channel
 # fs:             float  sampling rate
 # channel_names:  (C,)
 # source_file:    (1,)
 # seizure_onsets: (K,)   ISO-format datetimes associated with this .mat file (K may be 0)
#T0:              (K,)   recording start timestamps (ISO format) from
                           #df_matches (may repeat if multiple matches exist)
 # TF:              (K,)   recording end timestamps (ISO format) from
                           #df_matches (may repeat if multiple matches exist)

TEEG.full_recording_from_matfiles_1_9_V2(
    input_dir=input_dir,
    output_dir=output_dir,
    files_to_process=files_to_process,
    df_matches=df_merged,
    amp_threshold=200.0,
    lowcut=0.5,
    highcut=48.0,
    order=4,
    do_zscore=False,
    notch_freq=34.5,
)

#==========================
#==========================
#==========================

# 1.8 VISUALIZE WINDOWS FROM NPZ FILES
# VERSION With channel overlap and pre-ictal
# Final
import os

directory = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_test_pipeline_24042026/"

for file_name in sorted(os.listdir(directory)):

    if file_name.endswith("_preproc_full.npz"):

        full_path = os.path.join(directory, file_name)

        print(f"\nProcessing: {file_name}")


        TEEG.visualize_seizure_windows_from_npz_1_10V3(
    npz_path=full_path,
    channel_idx_1=0,
    channel_idx_2=1,
    window_sec=10,
    n_windows=12,
    pre_onset_sec=60,
    vertical_offset_uv=100,
    output_dir="/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results//seizure_pipelinetest_NOT_normalized24042026"
)
