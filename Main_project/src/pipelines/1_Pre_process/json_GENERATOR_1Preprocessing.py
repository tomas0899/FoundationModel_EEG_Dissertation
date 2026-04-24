import json
from pathlib import Path

# Define config as a Python dictionary
config = {
    "patient_id": "XB47Y",
    "input_path": "data/Originals/Subermat_anon/XB47Y/",
    "output_path": "results/XB47Y/",
    "window_size_sec": 10,
    "bandpass": [0.5, 48],
    "amplitude_threshold": 200,
    "target_date": "2019-12-11"
}

# Save JSON file
config_path = Path("config.json")

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)

print(f"Config saved at: {config_path}")