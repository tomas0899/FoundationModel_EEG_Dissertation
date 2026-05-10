# Importing the df from pickle
import pandas as pd
from pathlib import Path
# Load the feature dataframe from the pickle file
import pandas as pd
df_features = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/Main_project/results/RQXZ1/Feature_ext/Part2_features/RQXZ1_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260510_v01_FEAT-TIME-FREQ_20260510_v01/RQXZ1_IN-normalized_npz_FP-fullnpz_W10s_PRE6to5min_ICT0to1min_GAPasINT_FINAL-PREvsSEIZ_20260510_v01_FEAT-TIME-FREQ_20260510_v01_df_features_ictalVspreictal.pkl")

df_features.head()
df_SVM = df_features.copy()
df_SVM["window_start_time"] = pd.to_datetime(df_SVM["window_start_time"])
df_SVM = df_SVM.sort_values("window_start_time").reset_index(drop=True)
df_SVM = df_SVM.dropna(axis=1, how="all").copy()

metadata_cols = [
    "window_id",
    "start_sample",
    "end_sample",
    "fs",
    "n_channels",
    "window_sec",
    "seizure_onsets",
    "file_name",
    "window_start_time",
    "window_end_time",
    "class_label",
    "label_name"
]

feature_cols = [col for col in df_SVM.columns if col not in metadata_cols]

X = df_SVM[feature_cols].copy()
y = df_SVM["class_label"].copy()

print("df_SVM shape:", df_SVM.shape)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("NaNs in X:", X.isna().sum().sum())
print("Class counts:")
print(y.value_counts())
import numpy as np

# -------------------------------
# 1. Define target variable
# -------------------------------
y = df_SVM["class_label"].copy()

# -------------------------------
# 2. Drop metadata, identifiers, temporal info, target columns,
#    and columns that are not EEG-derived features
# -------------------------------
cols_to_drop = [
    "file_name",
    "window_id",
    "start_sample",
    "end_sample",
    "fs",
    "n_channels",
    "window_sec",
    "seizure_onsets",
    "window_start_time",
    "window_end_time",
    "class_label",
    "label_name",
    "excluded_reason"   # important: this column is completely empty
]

# Drop only columns that actually exist
cols_to_drop = [col for col in cols_to_drop if col in df_SVM.columns]

X = df_SVM.drop(columns=cols_to_drop).copy()

# -------------------------------
# 3. Keep only numeric EEG-derived features
# -------------------------------
X = X.select_dtypes(include=[np.number])

# -------------------------------
# 4. Replace infinite values with NaN
# -------------------------------
X = X.replace([np.inf, -np.inf], np.nan)

# -------------------------------
# 5. Keep only rows without missing values in EEG features
# -------------------------------
mask = X.notna().all(axis=1)

X = X.loc[mask].copy()
y = y.loc[mask].copy()

# -------------------------------
# 6. Remove constant columns
# -------------------------------
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)

# -------------------------------
# 7. Sanity checks
# -------------------------------
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Removed constant columns:", constant_cols)
print("Total NaNs in X:", X.isna().sum().sum())
print("Class counts:")
print(y.value_counts())# Convert labels to binary format:
# 0 = preictal
# 1 = seizure
y_binary = y.map({
    1: 0,
    2: 1
})

print(y_binary.value_counts())
y = y_binary.copy()
# Use the cleaned dataset size, not the original dataframe size
n = len(X)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)

ideal_train = 0.70
ideal_val = 0.15
ideal_test = 0.15

# generate candidates for the splitting

# possible positions where training ends
train_candidates = range(int(0.70 * n), int(0.90 * n), max(1, n // 1000))
# possible positions where validation ends 
val_candidates = range(int(0.05 * n), int(0.20 * n), max(1, n // 1000))

# search for the best temporal split
best = None
best_score = np.inf
# This procedure creates a temporal split by preserving the chronological order
# of the samples, while selecting the train/validation/test boundaries that
# best match the desired split proportions and maintain class ratios close to
# the global dataset distribution.
for train_end in train_candidates:
    for val_size in val_candidates:
        val_end = train_end + val_size

        if val_end >= n:
            continue

        # temporal subsets because df is already sorted by window_start_time
        # SPLIT STARTS HERE
        y_train = y.iloc[:train_end]
        y_val = y.iloc[train_end:val_end]
        y_test = y.iloc[val_end:]

        # require both classes in every split
        # MAKE SURE OF THE PRESENCE OF EACH CLASS IN EVERY SET
        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            continue

        # size closeness to 75/15/15
        # measure how close is to the ideal proportion
        train_frac = len(y_train) / n
        val_frac = len(y_val) / n
        test_frac = len(y_test) / n

        size_score = (
            abs(train_frac - 0.70) +
            abs(val_frac - 0.15) +
            abs(test_frac - 0.15)
        )

        # class ratio closeness to global ratio
        # measure how close is the preictal proportions 
        ratio_score = (
            abs(y_train.mean() - global_ratio) +
            abs(y_val.mean() - global_ratio) +
            abs(y_test.mean() - global_ratio)
        )

        # combined score for choosing the best score
        score = size_score + ratio_score * 3

        if score < best_score:
            best_score = score
            best = (train_end, val_end)
# Unpack the best split boundaries found previously
# train_end = index where the training set stops
# val_end = index where the validation set stops
train_end, val_end = best

# Create the training set:
# from the beginning of the dataset up to train_end (excluded)
X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

# Create the validation set:
# from train_end up to val_end (excluded)
X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

# Create the test set:
# from val_end to the end of the dataset
X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    make_scorer,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# -----------------------------
# Build the machine learning pipeline
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

# -----------------------------
# Temporal cross-validation
# -----------------------------
# If each row = 1 minute, gap=5 leaves a 5-minute gap between train and validation
tscv = TimeSeriesSplit(n_splits=4, gap=5)

# -----------------------------
# Define multiple evaluation metrics
# -----------------------------
scoring = {
    "f1_macro": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "recall_seizure": make_scorer(recall_score, pos_label=1),
    "precision_seizure": make_scorer(precision_score, pos_label=1, zero_division=0)
}

# -----------------------------
# Define hyperparameter search space
# -----------------------------
param_grid = {
    "svm__C": np.logspace(-2, 3, 6),
    "svm__gamma": ["scale"] + list(np.logspace(-4, 0, 5))
}

# -----------------------------
# Grid search
# -----------------------------
grid_f1 = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="f1_macro",
    cv=tscv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# -----------------------------
# Fit only on training set
# -----------------------------
grid_f1.fit(X_train, y_train)

# -----------------------------
# Best model
# -----------------------------
best_model_f1 = grid_f1.best_estimator_

print("Best parameters based on f1_macro:")
print(grid_f1.best_params_)

print("\nBest mean CV f1_macro:")
print(grid_f1.best_score_)