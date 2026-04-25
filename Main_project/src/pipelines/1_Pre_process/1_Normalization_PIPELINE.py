from pathlib import Path
import sys
import json
import pandas as pd
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

from pathlib import Path
import numpy as np
#==========================
#==========================
#==========================
# 1. load files
# Cambia esto por la ruta a uno de tus archivos reales
ruta_archivo = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_testALL_25032026/XB47Y_182_preproc_full.npz")

# Cargar el archivo
data = np.load(ruta_archivo, allow_pickle=True)

# Ver qué claves tiene
print("Claves encontradas:", list(data.keys()))
# expected: Claves encontradas: ['X', 'mu', 'sigma', 'fs', 'channel_names', 'source_file', 'seizure_onsets', 'T0', 'TF']
#==========================
#==========================
#==========================
# 2. file inspection
# Leer el array X
X = data["X"]

print("Type of data:", X.dtype)
print("(shape):", X.shape)
print(f"  → Channels (C): {X.shape[0]}")
print(f"  →  Samples (N): {X.shape[1]}")
print()
print("Val mínimo:", np.nanmin(X))
print("Val máximo:", np.nanmax(X))
print("¿NaN?:  ", np.any(np.isnan(X)))
print("¿Inf?:  ", np.any(np.isinf(X)))

from pathlib import Path

# Change this to your actual input folder
input_dir = Path("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/XB47Y_testALL_25032026/")

# Find all .npz files
all_files = sorted(input_dir.glob("*.npz"))

print(f"Files found: {len(all_files)}")

#==========================
#==========================
#==========================
# 3. Calculate mean and std per channel
ch_mean, ch_std, ch_count, skipped = TEEG.compute_global_channel_stats_1_15(all_files)

#==========================
#==========================
#==========================
# 4. Data normalization per recording + save it in a new npz

ch_mean, ch_std, ch_count, skipped = TEEG.compute_global_channel_stats_1_16(all_files)

output_dir = "/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_28032026Normalized/"

stats_npz_path, summary = apply_global_channel_normalization(
    all_files=all_files,
    ch_mean=ch_mean,
    ch_std=ch_std,
    ch_count=ch_count,
    output_dir=output_dir
)
#==========================
#==========================
#==========================
# 5. Sanity Check
df_check = TEEG.sanity_check_global_zscore_npz_1_14(
    original_dir="/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_testALL_25032026/",
    normalized_dir="/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y_28032026Normalized/"
)