# Importing the df from pickle
import pandas as pd
from pathlib import Path
# Load the feature dataframe from the pickle file
import pandas as pd

df_features = pd.read_pickle("/home/tperezsanchez/FoundationModel_EEG_Dissertation/EEG_data_vis/results/XB47Y/Feature_ext/df_features_ictalVSPreictal18_4.pkl")
df_features.head()
df_SVM = df_features.copy()
df_SVM["window_start_time"] = pd.to_datetime(df_SVM["window_start_time"])
df_SVM = df_SVM.sort_values("window_start_time").reset_index(drop=True)
import numpy as np
# Prepare feature matrix for SVM by excluding metadata, identifiers,
# temporal information, and target columns, keeping only EEG-derived features.
# These columns were excluded because they represent metadata, indexing,
# or target information, and do not provide useful physiological signal
# information for SVM training.

# Define target variable
y = df_SVM["class_label"].copy()

# Drop metadata, identifiers, time columns, and target columns
cols_to_drop = [
    "file_name",          # file identifier
    "window_id",          # window index / identifier
    "start_sample",       # sample-based position in recording
    "end_sample",         # sample-based position in recording
    "fs",                 # sampling frequency (often constant)
    "n_channels",         # number of channels (often constant)
    "seizure_onsets",     # metadata, not a model feature
    "window_start_time",  # timestamp metadata
    "window_end_time",    # timestamp metadata
    "class_label",        # numeric target
    "label_name"          # text version of target
]
# Keep only EEG-derived numerical features
X = df_SVM.drop(columns=cols_to_drop).copy()

# Replace inf values with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Keep only rows without missing values
mask = X.notna().all(axis=1)
X = X.loc[mask].copy()
y = y.loc[mask].copy()

# Remove constant columns (no variance, no predictive value)
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
X = X.drop(columns=constant_cols)

# y.mean() corresponds to the proportion of class 1 (seizure)
global_ratio = y.mean()
print("Global seizure ratio:", global_ratio)

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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# -----------------------------
# Build the machine learning pipeline
# -----------------------------
# Step 1: Standardize the features so they have comparable scale
# Step 2: Train an SVM with RBF kernel
# class_weight="balanced" gives more importance to the minority class
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", class_weight="balanced"))
])

# -----------------------------
# Define temporal cross-validation
# -----------------------------
# TimeSeriesSplit preserves chronological order:
# earlier samples are used for training, later samples for validation
# This is important to avoid temporal leakage
tscv = TimeSeriesSplit(n_splits=4)

# -----------------------------
# Define the hyperparameter search space
# -----------------------------
# C controls how strongly the model penalizes misclassification
# gamma controls how flexible the RBF decision boundary is
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
}

# -----------------------------
# Set up grid search
# -----------------------------
# GridSearchCV will try all combinations of C and gamma
# scoring="accuracy" means models are ranked by mean validation accuracy
# cv=tscv applies temporal cross-validation
# n_jobs=-1 uses all available CPU cores
# refit=True retrains the best model on the full training set at the end
grid_acc = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=tscv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# -----------------------------
# Fit grid search on the training set only
# -----------------------------
# This tests all parameter combinations using temporal CV
grid_acc.fit(X_train, y_train)

# -----------------------------
# Retrieve the best model
# -----------------------------
# best_estimator_ contains the pipeline with the best hyperparameters found
best_model_acc = grid_acc.best_estimator_

# Print the best hyperparameters
print("Best parameters (accuracy):")
print(grid_acc.best_params_)

# Print the best mean cross-validation accuracy
print("\nBest mean CV accuracy:")
print(grid_acc.best_score_)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Predict on validation set
y_val_pred = best_model_acc.predict(X_val)

# Correct class names
class_names = ["preictal", "seizure"]

# -----------------------------
# Confusion matrix
# -----------------------------
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.title("Confusion Matrix - Validation Set")
plt.show()

# -----------------------------
# Classification report as table
# -----------------------------
report_dict = classification_report(
    y_val,
    y_val_pred,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose().round(3)
print(report_df)
# Predict on test set
y_test_pred = best_model_acc.predict(X_test)

# Confusion matrix
cm_test = confusion_matrix(y_test, y_test_pred)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
disp_test.plot()
plt.title("Confusion Matrix - Test Set")
plt.show()

# Classification report
report_test = classification_report(
    y_test,
    y_test_pred,
    target_names=class_names,
    output_dict=True
)

report_test_df = pd.DataFrame(report_test).transpose().round(3)
print(report_test_df)