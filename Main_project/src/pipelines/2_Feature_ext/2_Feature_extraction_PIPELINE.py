import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
#==========================
#==========================
#==========================
# 0.1 Load modules and json config
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
# 0.2 Load FILES
files = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/")
base_files = sorted(files.glob("*full.npz"))

print(f"Found {len(base_files)} .npz files")

rows = []
from datetime import datetime, timezone
import numpy as np


def parse_timestamp(val):
    """Accept Unix float, datetime string, or repeated timestamp arrays."""

    if isinstance(val, np.ndarray):
        flat = val.ravel()
        cleaned = [str(x).strip() for x in flat if str(x).strip() != ""]

        if len(cleaned) == 0:
            return None

        unique_vals = list(dict.fromkeys(cleaned))

        if len(unique_vals) == 1:
            val = unique_vals[0]
        else:
            raise ValueError(f"Timestamp array has multiple different values: {unique_vals}")

    try:
        return float(val)
    except (ValueError, TypeError):
        pass

    val = str(val).strip()
    dt = datetime.strptime(val, "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=timezone.utc).timestamp() #==========================
#==========================
#==========================
for base_NPZ_path in base_files:
    meta = {"file_name": base_NPZ_path.name, "file_path": str(base_NPZ_path.resolve()), "load_error": None}

    try:
        with np.load(base_NPZ_path, allow_pickle=True) as data:
            keys = set(data.files)
            # store data from key into meta df, convert them into float or str first
            
            meta["fs"]         = float(data["fs"]) if "fs" in keys else None
            meta["source_file"]= str(data["source_file"]) if "source_file" in keys else None
            meta["T0"] = parse_timestamp(data["T0"]) if "T0" in keys else None
            meta["TF"] = parse_timestamp(data["TF"]) if "TF" in keys else None
            meta["is_normalized"] = "mu" in keys and "sigma" in keys

            meta["channel_names"]  = list(data["channel_names"]) if "channel_names" in keys else []
            meta["seizure_onsets"] = list(data["seizure_onsets"]) if "seizure_onsets" in keys else []

            # Read X shape only — avoids loading the full array into memory
            if "X" in keys:
                shape = data["X"].shape
                meta["n_channels"] = int(shape[0]) if len(shape) == 2 else None
                meta["n_samples"]  = int(shape[1]) if len(shape) == 2 else None
            else:
                meta["n_channels"] = None
                meta["n_samples"]  = None

    except Exception as e:
        meta["load_error"] = str(e)
        print(f"Failed to load {base_NPZ_path.name}: {e}")

    rows.append(meta)

print(f"\nLoaded metadata from {len(rows)} file(s)")
#==========================
#==========================
#==========================
# 2. Temporal sanity check

from datetime import datetime

for meta in rows:
    T0 = meta.get("T0")
    TF = meta.get("TF") 

    meta["start_time"] = datetime.fromtimestamp(T0) if T0 is not None else None
    meta["end_time"]   = datetime.fromtimestamp(TF) if TF is not None else None
    meta["duration_s"] = round(TF - T0, 3) if (T0 is not None and TF is not None) else None

    # Cross-check: does n_samples / fs match TF - T0?
    fs, n = meta.get("fs"), meta.get("n_samples")
    if fs and n and meta["duration_s"] is not None:
        expected = round(n / fs, 3)
        meta["duration_check_ok"] = abs(expected - meta["duration_s"]) < 1.0
    else:
        meta["duration_check_ok"] = None

bad = [m for m in rows if m["duration_check_ok"] is False]
print(len(bad), "files with duration mismatch")

#==========================
#==========================
#==========================
# 3. General Dataframe for every file
COLUMNS = [
    "file_name", "file_path",
    "start_time", "end_time", "duration_s",
    "fs", "n_channels", "n_samples",
    "channel_names", "seizure_onsets",
    "is_normalized", "source_file",
    "duration_check_ok", "load_error",
]

df = pd.DataFrame(rows, columns=COLUMNS)
df = df.sort_values("start_time", na_position="last").reset_index(drop=True)

print(df.shape)
df.head()

print("── Sanity Report ──────────────────────────────────────")
print(f"  Total recordings   : {len(df)}")
print(f"  Load errors        : {df['load_error'].notna().sum()}")
print(f"  Missing T0/TF      : {df['start_time'].isna().sum()}")
print(f"  Duration check ✗   : {(df['duration_check_ok'] == False).sum()}")
print(f"  Missing fs         : {df['fs'].isna().sum()}")
print(f"  Normalized files   : {df['is_normalized'].sum()}")

# Check for overlapping recordings
valid = df.dropna(subset=["start_time", "end_time"]).sort_values("start_time")
overlaps = 0
for i in range(len(valid) - 1):
    if valid.iloc[i]["end_time"] > valid.iloc[i + 1]["start_time"]:
        overlaps += 1
print(f"  Overlapping pairs  : {overlaps}")
print("───────────────────────────────────────────────────────")
#==========================
#==========================
#==========================
# 4. Clean onset: solution to problems with "nan"
def clean_onsets(x):
    if isinstance(x, (list, np.ndarray)):
        return [i for i in x if not pd.isna(i)]
    elif pd.isna(x):
        return []
    else:
        return [x]

df["seizure_onsets_clean"] = df["seizure_onsets"].apply(clean_onsets)
#==========================
#==========================
#==========================
# 5. Windowing
# Create an empty list that will store one dictionary per EEG window
df_windows = TEEG.create_eeg_windows_2_3(df, window_sec=10)
# Convert the list of dictionaries into a new dataframe
# Each row now represents one 10-second window


print(df_windows.head())
print(df_windows.shape) # rows vs. columns
df_windows[df_windows["seizure_onsets"].apply(lambda x: not pd.isna(x).all() if isinstance(x, (list, np.ndarray)) else not pd.isna(x))].head(10)
#==========================
#==========================
#==========================
# 6. Labeling
from datetime import timedelta
import pandas as pd
import numpy as np

# -------------------------------
# Helper: clean seizure_onsets
# -------------------------------
def clean_onsets(x):
    if isinstance(x, (list, np.ndarray, pd.Series)):
        return [i for i in x if not pd.isna(i)]
    elif pd.isna(x) or x is None:
        return []
    else:
        return [x]

# -------------------------------
# Copy dataframe to avoid overwriting original
# -------------------------------
df_windows_labeled = df_windows.copy()

# Create new columns
df_windows_labeled["window_start_time"] = pd.NaT
df_windows_labeled["window_end_time"] = pd.NaT
df_windows_labeled["class_label"] = np.nan
df_windows_labeled["label_name"] = pd.NA

# -------------------------------
# Main labeling loop
# -------------------------------
for idx, row in df_windows_labeled.iterrows():
    
    # Match recording-level metadata
    rec = df[df["file_name"] == row["file_name"]].iloc[0]

    # Recording start time
    recording_start = pd.to_datetime(rec["start_time"])
    fs = row["fs"]
    
    # Convert sample indices to seconds
    start_sec = row["start_sample"] / fs
    end_sec = row["end_sample"] / fs

    # Compute real datetime of each window
    window_start_time = recording_start + pd.Timedelta(seconds=start_sec)
    window_end_time = recording_start + pd.Timedelta(seconds=end_sec)

    # Save as datetime first
    df_windows_labeled.at[idx, "window_start_time"] = window_start_time
    df_windows_labeled.at[idx, "window_end_time"] = window_end_time
    
    # Default = excluded
    assigned_label = np.nan
    assigned_name = pd.NA

    # Clean seizure_onsets
    seizure_onsets = clean_onsets(row["seizure_onsets"])
    
    # Loop through every onset
    for onset in seizure_onsets:
        onset = pd.to_datetime(onset)
        
        # Define intervals
        preictal_start = onset - pd.Timedelta(minutes=10)
        preictal_end   = onset - pd.Timedelta(minutes=5)
        seizure_start  = onset
        seizure_end    = onset + pd.Timedelta(minutes=5)
        
        # Overlap function
        def overlaps(a_start, a_end, b_start, b_end):
            return (a_start < b_end) and (a_end > b_start)

        # 1) Seizure has priority
        if overlaps(window_start_time, window_end_time, seizure_start, seizure_end):
            assigned_label = 1
            assigned_name = "seizure"
            break
        
        # 2) Preictal only if not seizure
        elif overlaps(window_start_time, window_end_time, preictal_start, preictal_end):
            if pd.isna(assigned_label) or assigned_label != 1:
                assigned_label = 0
                assigned_name = "preictal"
    
    df_windows_labeled.at[idx, "class_label"] = assigned_label
    df_windows_labeled.at[idx, "label_name"] = assigned_name

# -------------------------------
# Keep only preictal and seizure windows
# -------------------------------
df_final = df_windows_labeled[df_windows_labeled["class_label"].isin([0, 1])].copy()

# Make class_label integer
df_final["class_label"] = df_final["class_label"].astype(int)

# -------------------------------
# Convert datetime columns to visible string format
# -------------------------------
df_final["window_start_time"] = pd.to_datetime(df_final["window_start_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
df_final["window_end_time"] = pd.to_datetime(df_final["window_end_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------
# Quick sanity check
# -------------------------------
print(df_final[[
    "file_name",
    "window_id",
    "window_start_time",
    "window_end_time",
    "seizure_onsets",
    "class_label",
    "label_name"
]].head(30))

print("\nClass counts:")
print(df_final["label_name"].value_counts())

#directory with all the npz
npz_base_path = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_28032026Normalized/")
#single file
npz_file_example = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_28032026Normalized/XB47Y_41_preproc_full.npz"

npz = np.load(npz_file_example, allow_pickle=True)
print(npz.files)
print(npz["X"].shape)
#==========================
#==========================
#==========================
# 6. Main loop
#directory with all the npz
npz_base_path = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/")
#single file
npz_file_example = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/XB47Y_41_preproc_full.npz"

npz = np.load(npz_file_example, allow_pickle=True)
print(npz.files)
print(npz["X"].shape)
import os
import numpy as np

# 1) select one row
row = df_final.iloc[0]

# 2) build file path
npz_path = os.path.join(npz_base_path, row["file_name"])

# 3) load npz
npz_data = np.load(npz_path, allow_pickle=True)

# 4) extract signal and metadate from npz
X = npz_data["X"]                     # shape (C, N)
fs = float(npz_data["fs"])
channel_names = npz_data["channel_names"]

print("file:", row["file_name"])
print("X shape:", X.shape)
print("fs:", fs)
print("channel_names:", channel_names)

# 5) extract windows time boundaries
start_sample = int(row["start_sample"])
end_sample = int(row["end_sample"])

print("start_sample:", start_sample)
print("end_sample:", end_sample)
print("window length in samples:", end_sample - start_sample)

# 6) cut window
window = X[:, start_sample:end_sample]

print("window shape:", window.shape)

# checking the features extraction from one window
time_features = extract_time_features(window, channel_names=channel_names)
freq_features = extract_frequency_features(window, fs=fs, channel_names=channel_names)

print("n_time_features:", len(time_features))
print("n_freq_features:", len(freq_features))

all_features = {**time_features, **freq_features}

print("n_total_features:", len(all_features))
#check a few of them
for i, (k, v) in enumerate(all_features.items()):
    print(k, v)
    if i >= 9:
        break
# Check whether any extracted feature is NaN
nan_features = {k: v for k, v in all_features.items() if np.isnan(v)}

print("Number of NaN features:", len(nan_features))

# Print NaN features if any appear
for k, v in nan_features.items():
    print(k, ":", v)

import pandas as pd

# Convert the original metadata row into a dictionary
row_dict = row.to_dict()

# Merge metadata and extracted features into a single flat dictionary
full_row = {**row_dict, **all_features}

# Convert the merged dictionary into a one-row DataFrame
df_one_window_features = pd.DataFrame([full_row])

# Print output shape
print("Output shape:", df_one_window_features.shape)

# Print all column names
print(df_one_window_features.columns.tolist())

# Show the first row transposed for easier inspection
print(df_one_window_features.T.head(25))

# Check for duplicated column names
print("Duplicated columns:", df_one_window_features.columns.duplicated().sum())

# Check data types
print(df_one_window_features.dtypes)

import os
import numpy as np

def extract_features_from_row_cached(row, npz_base_path, file_cache):
    """
    Extract metadata + per-channel features for a single EEG window row,
    using a cache so repeated NPZ files are not reloaded.

    Parameters
    ----------
    row : pd.Series
        One row from df_final. Must contain:
        - file_name
        - start_sample
        - end_sample

    npz_base_path : str
        Directory containing the preprocessed NPZ files.

    file_cache : dict
        Dictionary used to store already loaded NPZ content by file name.

    Returns
    -------
    full_row : dict
        Dictionary containing original metadata plus extracted features.
    """

    # Get file name from the dataframe row
    file_name = row["file_name"]

    # Load the NPZ only once per file and store it in the cache
    if file_name not in file_cache:
        npz_path = os.path.join(npz_base_path, file_name)
        npz_data = np.load(npz_path, allow_pickle=True)

        file_cache[file_name] = {
            "X": npz_data["X"],
            "fs": float(npz_data["fs"]),
            "channel_names": npz_data["channel_names"]
        }

    # Retrieve cached data for the current file
    X = file_cache[file_name]["X"]
    fs = file_cache[file_name]["fs"]
    channel_names = file_cache[file_name]["channel_names"]

    # Read window boundaries from the dataframe row
    start_sample = int(row["start_sample"])
    end_sample = int(row["end_sample"])

    # Slice the current window from the recording
    window = X[:, start_sample:end_sample]

    # Extract time-domain features
    time_features = extract_time_features(
        window,
        channel_names=channel_names
    )

    # Extract frequency-domain features
    freq_features = extract_frequency_features(
        window,
        fs=fs,
        channel_names=channel_names
    )

    # Merge all extracted features
    all_features = {**time_features, **freq_features}

    # Merge original metadata and extracted features
    full_row = {**row.to_dict(), **all_features}

    return full_row
import pandas as pd

# List to collect one output dictionary per window
rows_with_features = []

# Dictionary to cache already loaded NPZ files
file_cache = {}

# Iterate through all rows in df_final
for idx, row in df_final.iterrows():
    full_row = extract_features_from_row_cached(
        row,
        npz_base_path=npz_base_path,
        file_cache=file_cache
    )
    rows_with_features.append(full_row)

# Convert the list of dictionaries into the final feature dataframe
df_features = pd.DataFrame(rows_with_features)

# Print final shape
print("Final dataframe shape:", df_features.shape)

# Print number of unique cached files used
print("Number of cached files:", len(file_cache))

# Show first rows
print(df_features.head())

#==========================
#==========================
#==========================
# 7. Saving pickles files as a back up and for statistcal 
# Save the final feature dataframe as a pickle file
df_features.to_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y/Feature_ext/df_features_ictalVSPreictal18_4.pkl")
# Load the feature dataframe from the pickle file
import pandas as pd

df_features = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y/Feature_ext/df_features_ictalVSPreictal18_4.pkl")