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
# Get current file location
import pandas as pd

df_labeled_all = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part1_labeling/df_labeled_all.pkl")

df_final_ictalVspreictal = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y/Feature_ext/Part1_labeling/df_ictal_Vs_Preictal.pkl")

print(df_labeled_all.shape)
print(df_final_ictalVspreictal.shape)
#==========================
#==========================
#==========================
# 1. df only including ictal vs. preictal
import pandas as pd
# path with all the preprocessed npz
npz_base_path = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/")

# List to collect one output dictionary per window
rows_with_features = []

# Dictionary to cache already loaded NPZ files
file_cache = {}

# Iterate through all rows in df_final
for idx, row in df_final_ictalVspreictal.iterrows():
    full_row = TEEG.extract_features_from_row_cached_2_7(
        row,
        npz_base_path=npz_base_path,
        file_cache=file_cache
    )
    rows_with_features.append(full_row)

# Convert the list of dictionaries into the final feature dataframe
df_features_ictalVspreictal = pd.DataFrame(rows_with_features)

# Print final shape
print("Final dataframe shape:", df_features_ictalVspreictal.shape)

# Print number of unique cached files used
print("Number of cached files:", len(file_cache))

# Show first rows
print(df_features_ictalVspreictal.head())
#==========================
#==========================
#==========================
# 2. df for all the windows
import pandas as pd
# path with all the preprocessed npz
npz_base_path = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_28032026Normalized/")

# List to collect one output dictionary per window
rows_with_features = []

# Dictionary to cache already loaded NPZ files
file_cache = {}

# Iterate through all rows in df_final
for idx, row in df_labeled_all.iterrows():
    full_row = TEEG.extract_features_from_row_cached_2_7(
        row,
        npz_base_path=npz_base_path,
        file_cache=file_cache
    )
    rows_with_features.append(full_row)

# Convert the list of dictionaries into the final feature dataframe
df_features_all = pd.DataFrame(rows_with_features)

# Print final shape
print("Final dataframe shape:", df_features_all.shape)

# Print number of unique cached files used
print("Number of cached files:", len(file_cache))

# Show first rows
print(df_features_all.head())
#==========================
#==========================
#==========================
# 3. save both df as pickles files
output_dir = Path(config["paths"]["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

df_features_all = config["output_files"]["df_features_all"]
df_features_ictalVspreictal = config["output_files"]["df_features_ictalVspreictal"]

path_all = output_dir / df_features_all
path_ictalVspreictal = output_dir / df_features_ictalVspreictal

print(f"Output directory: {output_dir.resolve()}")
print(f"df_labeled filename: {df_features_all}")
print(f"df_final filename: {df_features_ictalVspreictal}")

df_features_all.to_pickle(path_all)
df_features_ictalVspreictal.to_pickle(path_ictalVspreictal)

print(f"df_features_all saved in: {path_all.resolve()}")
print(f"df_features_ictalVspreictal saved in: {path_ictalVspreictal.resolve()}")