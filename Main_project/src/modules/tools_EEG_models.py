import numpy as np
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from datetime import date
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



#TOOLS EEG MODELS:
#=================================================================================
#=================================================================================
#=================================================================================
# Function #1

# -------------------------------------------------
# Helper: safe filename
# -------------------------------------------------
def sanitize_filename(text):
    """
    Convert text into a safe filename string.
    """
    text = str(text)
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


# -------------------------------------------------
# Function: plot confusion matrix as percentages
# -------------------------------------------------
def plot_confusion_matrix_percent(
    y_true,
    y_pred,
    class_names,
    title="Confusion Matrix",
    patient_id=None,
    labels=None,
    save_pdf_path=None,
    show_plot=False
):
    """
    Plot a row-normalized confusion matrix in percentage format.
    Each row sums to 100%.

    Optionally:
    - includes patient ID in the title
    - saves the figure as PDF
    """

    # Force label order to match class_names
    if labels is None:
        labels = list(range(len(class_names)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Avoid division by zero if one class is absent in y_true
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0
    ) * 100

    # Add patient ID to plot title if provided
    if patient_id is not None:
        title = f"Patient {patient_id} - {title}"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_percent, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )

    # Add percentage text inside each cell
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            ax.text(
                j, i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="black"
            )

    plt.tight_layout()

    # Save figure as PDF
    if save_pdf_path is not None:
        save_pdf_path = Path(save_pdf_path)
        save_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_pdf_path, format="pdf", bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return cm, cm_percent


# -------------------------------------------------
# Function: evaluate model on a dataset
# -------------------------------------------------
def evaluate_and_plot_3_1(
    model,
    X_data,
    y_true,
    class_names,
    dataset_name="Validation",
    patient_id=None,
    output_dir=None,
    labels=None,
    show_plot=False,
    output_prefix=None
):
    """
    1. Predicts labels
    2. Prints classification table
    3. Prints global metrics
    4. Plots confusion matrix in percentages
    5. Saves confusion matrix PDF
    6. Saves confusion matrix as CSV
    7. Saves classification table as CSV
    8. Optionally adds patient ID to outputs
    """

    y_pred = model.predict(X_data)

    # Force label order to match class_names
    if labels is None:
        labels = list(range(len(class_names)))

    # -------------------------------------------------
    # Classification report as dataframe
    # -------------------------------------------------
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    # Add patient and dataset information to the classification table
    if patient_id is not None:
        report_df.insert(0, "patient_id", patient_id)

    report_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    # -------------------------------------------------
    # Print results
    # -------------------------------------------------
    print(f"\n{'='*40}")

    if patient_id is not None:
        print(f"PATIENT: {patient_id}")

    print(f"{dataset_name.upper()} SET")
    print(f"{'='*40}")

    print("\nClassification table:")
    print(report_df)

    print("\nGlobal metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")

    # -------------------------------------------------
    # Build output filenames
    # -------------------------------------------------
    save_pdf_path = None
    cm_counts_csv_path = None
    cm_percent_csv_path = None
    classification_csv_path = None
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        dataset_tag = sanitize_filename(dataset_name)
    
        if output_prefix is not None:
            base_stem = f"{sanitize_filename(output_prefix)}_{dataset_tag}"
        else:
            patient_tag = (
                f"PAT-{sanitize_filename(patient_id)}"
                if patient_id is not None
                else "PAT-unknown"
            )
            base_stem = f"{patient_tag}_{dataset_tag}"
    
        file_stem = f"{base_stem}_confusion_matrix"
    
        save_pdf_path = output_dir / f"{file_stem}.pdf"
        cm_counts_csv_path = output_dir / f"{file_stem}_counts.csv"
        cm_percent_csv_path = output_dir / f"{file_stem}_percent.csv"
    
        classification_csv_path = output_dir / f"{base_stem}_classification_table.csv"
    # -------------------------------------------------
    # Plot and optionally save confusion matrix PDF
    # -------------------------------------------------
    cm_counts, cm_percent = plot_confusion_matrix_percent(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        title=f"{dataset_name} Confusion Matrix (%)",
        patient_id=patient_id,
        labels=labels,
        save_pdf_path=save_pdf_path,
        show_plot=show_plot
    )

    # -------------------------------------------------
    # Return confusion matrices as dataframes
    # -------------------------------------------------
    cm_counts_df = pd.DataFrame(
        cm_counts,
        index=[f"True {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names]
    )

    cm_percent_df = pd.DataFrame(
        cm_percent,
        index=[f"True {c}" for c in class_names],
        columns=[f"Pred {c}" for c in class_names]
    )

    # Add patient and dataset labels to confusion matrix tables
    if patient_id is not None:
        cm_counts_df.insert(0, "patient_id", patient_id)
        cm_percent_df.insert(0, "patient_id", patient_id)

    cm_counts_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    cm_percent_df.insert(
        1 if patient_id is not None else 0,
        "dataset",
        dataset_name
    )

    # -------------------------------------------------
    # Save outputs
    # -------------------------------------------------
    if output_dir is not None:

        cm_counts_df.to_csv(cm_counts_csv_path, index=True)
        cm_percent_df.to_csv(cm_percent_csv_path, index=True)

        # This saves the printed classification table as CSV
        report_df.to_csv(
            classification_csv_path,
            index=True,
            index_label="class_or_metric"
        )

        print("\nSaved outputs:")
        print(f"PDF: {save_pdf_path}")
        print(f"CSV counts: {cm_counts_csv_path}")
        print(f"CSV percent: {cm_percent_csv_path}")
        print(f"Classification table CSV: {classification_csv_path}")

    return {
        "patient_id": patient_id,
        "dataset": dataset_name,
        "y_pred": y_pred,
        "classification_table": report_df,
        "confusion_counts": cm_counts_df,
        "confusion_percent": cm_percent_df,
        "pdf_path": save_pdf_path,
        "confusion_counts_csv_path": cm_counts_csv_path,
        "confusion_percent_csv_path": cm_percent_csv_path,
        "classification_csv_path": classification_csv_path
    }
#=================================================================================
#=================================================================================
#=================================================================================
# Function #2
def find_best_temporal_split_3_2(
    y,
    ideal_train=0.70,
    ideal_val=0.15,
    ideal_test=0.15,
    train_search_range=(0.70, 0.90),
    val_search_range=(0.05, 0.20),
    ratio_weight=3
):
    """
    Find the best temporal train/validation/test split.

    The function preserves chronological order and searches for split boundaries
    that are close to the desired train/validation/test proportions while also
    keeping the class ratio similar across the three sets.

    Parameters
    ----------
    y : pd.Series
        Target variable ordered chronologically.

    ideal_train : float
        Desired proportion of the dataset for the training set.

    ideal_val : float
        Desired proportion of the dataset for the validation set.

    ideal_test : float
        Desired proportion of the dataset for the test set.

    train_search_range : tuple
        Range of possible train end positions as proportions of the dataset.
        Example: (0.70, 0.90) searches train_end between 70% and 90%.

    val_search_range : tuple
        Range of possible validation sizes as proportions of the dataset.
        Example: (0.05, 0.20) searches validation sizes between 5% and 20%.

    ratio_weight : float
        Weight applied to the class-ratio score.
        Higher values force the split to preserve class balance more strongly.

    Returns
    -------
    train_end : int
        Index where the training set ends.

    val_end : int
        Index where the validation set ends.

    best_score : float
        Score of the selected split. Lower is better.
    """

    # Number of samples in the cleaned dataset
    n = len(y)

    # Global proportion of class 1
    # In this binary classification setup, class 1 represents seizures
    global_ratio = y.mean()

    # Candidate positions where the training set may end
    train_candidates = range(
        int(train_search_range[0] * n),
        int(train_search_range[1] * n),
        max(1, n // 1000)
    )

    # Candidate validation set sizes
    val_candidates = range(
        int(val_search_range[0] * n),
        int(val_search_range[1] * n),
        max(1, n // 1000)
    )

    best = None
    best_score = np.inf

    # Search for the best temporal split
    for train_end in train_candidates:
        for val_size in val_candidates:

            val_end = train_end + val_size

            # Skip invalid split where validation exceeds dataset length
            if val_end >= n:
                continue

            # Temporal split
            y_train_candidate = y.iloc[:train_end]
            y_val_candidate = y.iloc[train_end:val_end]
            y_test_candidate = y.iloc[val_end:]

            # Require both classes in train, validation, and test
            if (
                y_train_candidate.nunique() < 2
                or y_val_candidate.nunique() < 2
                or y_test_candidate.nunique() < 2
            ):
                continue

            # Measure how close the split sizes are to the ideal proportions
            train_frac = len(y_train_candidate) / n
            val_frac = len(y_val_candidate) / n
            test_frac = len(y_test_candidate) / n

            size_score = (
                abs(train_frac - ideal_train)
                + abs(val_frac - ideal_val)
                + abs(test_frac - ideal_test)
            )

            # Measure how close the class ratios are to the global class ratio
            ratio_score = (
                abs(y_train_candidate.mean() - global_ratio)
                + abs(y_val_candidate.mean() - global_ratio)
                + abs(y_test_candidate.mean() - global_ratio)
            )

            # Combined score
            score = size_score + ratio_score * ratio_weight

            # Keep the best split found so far
            if score < best_score:
                best_score = score
                best = (train_end, val_end)

    # Safety check in case no valid split was found
    if best is None:
        raise ValueError(
            "No valid temporal split found. "
            "Try changing candidate ranges or check class distribution over time."
        )

    train_end, val_end = best

    return train_end, val_end, best_score
#=================================================================================
#=================================================================================
#=================================================================================
# Function #3

def train_svm_gridsearch_3_3(
    X_train,
    y_train,
    n_splits=4,
    scoring="f1_macro",
    param_grid=None,
    n_jobs=-1,
    verbose=1
):
    """
    Train an SVM classifier using a sklearn Pipeline and temporal cross-validation.

    The pipeline includes:
    1. StandardScaler:
       Standardizes features using only the training fold during cross-validation.

    2. SVC with RBF kernel:
       Non-linear Support Vector Machine classifier.

    TimeSeriesSplit is used to preserve chronological order during hyperparameter tuning.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used to rank models during GridSearchCV.

    param_grid : dict or None
        Hyperparameter grid for the SVM.
        If None, a default grid is used.

    n_jobs : int
        Number of CPU cores used by GridSearchCV.
        -1 means use all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    best_model : sklearn Pipeline
        Best trained pipeline selected by GridSearchCV.

    grid_search : GridSearchCV
        Full fitted GridSearchCV object containing results, scores, and best parameters.
    """

    # --------------------------------------------------------
    # 1. Build machine learning pipeline
    # --------------------------------------------------------
    # StandardScaler is inside the pipeline to avoid data leakage.
    # During cross-validation, it is fitted only on each training fold.

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced"))
    ])

    # --------------------------------------------------------
    # 2. Define temporal cross-validation
    # --------------------------------------------------------
    # TimeSeriesSplit preserves chronological order.
    # This is important for EEG windows because future data should not be used
    # to validate past data.

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # --------------------------------------------------------
    # 3. Define default hyperparameter grid
    # --------------------------------------------------------
    # C controls the penalty for misclassification.
    # gamma controls the influence radius of each sample in the RBF kernel.

    if param_grid is None:
        param_grid = {
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.001, 0.01, 0.1, 1]
        }

    # --------------------------------------------------------
    # 4. Set up GridSearchCV
    # --------------------------------------------------------
    # GridSearchCV tests all combinations of C and gamma.
    # refit=True means that the best model is refitted on the full training set.

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # --------------------------------------------------------
    # 5. Fit grid search on training set only
    # --------------------------------------------------------
    # Validation and test sets are not used here.

    grid_search.fit(X_train, y_train)

    # --------------------------------------------------------
    # 6. Retrieve best model
    # --------------------------------------------------------

    best_model = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(f"\nBest mean CV {scoring}:")
    print(grid_search.best_score_)

    return best_model, grid_search
#=================================================================================
#=================================================================================
#=================================================================================
# Function #4



def train_decision_tree_gridsearch_3_4(
    X_train,
    y_train,
    n_splits: int = 4,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """
    Trains a Decision Tree classifier using TimeSeriesSplit and GridSearchCV.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features. 

    y_train : array-like or Series
        Training labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used by GridSearchCV. Default is "f1_macro".

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores used by GridSearchCV. -1 uses all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    grid_dt : GridSearchCV object
        Full fitted GridSearchCV object.

    best_model_dt : Pipeline
        Best fitted Decision Tree pipeline.

    best_params_dt : dict
        Best hyperparameters found.

    best_score_dt : float
        Best mean cross-validation score.
    """

    # ----------------------------------------------------------
    # 1. Build Decision Tree pipeline
    # ----------------------------------------------------------
    pipeline_dt = Pipeline([
        ("tree", DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced"
        ))
    ])

    # ----------------------------------------------------------
    # 2. Define temporal cross-validation
    # ----------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ----------------------------------------------------------
    # 3. Define hyperparameter grid
    # ----------------------------------------------------------
    param_grid_dt = {
        "tree__criterion": ["gini", "entropy"],
        "tree__max_depth": [2, 3, 4, 5, 6, None],
        "tree__min_samples_split": [2, 5, 10, 20],
        "tree__min_samples_leaf": [1, 2, 5, 10]
    }

    # ----------------------------------------------------------
    # 4. Set up GridSearchCV
    # ----------------------------------------------------------
    grid_dt = GridSearchCV(
        estimator=pipeline_dt,
        param_grid=param_grid_dt,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # ----------------------------------------------------------
    # 5. Train model
    # ----------------------------------------------------------
    grid_dt.fit(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Extract best model and results
    # ----------------------------------------------------------
    best_model_dt = grid_dt.best_estimator_
    best_params_dt = grid_dt.best_params_
    best_score_dt = grid_dt.best_score_

    print("Best Decision Tree parameters:")
    print(best_params_dt)

    print("\nBest mean CV macro F1:")
    print(best_score_dt)

    return grid_dt, best_model_dt, best_params_dt, best_score_dt


#=================================================================================
#=================================================================================
#=================================================================================
# Function #5

def train_random_forest_gridsearch_3_5(
    X_train,
    y_train,
    n_splits: int = 4,
    scoring: str = "f1_macro",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
):
    """
    Trains a Random Forest classifier using TimeSeriesSplit and GridSearchCV.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Training features. In your case, this can be the PCA-transformed features.

    y_train : array-like or Series
        Training labels.

    n_splits : int
        Number of temporal cross-validation splits.

    scoring : str
        Metric used by GridSearchCV. Default is "f1_macro".

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of CPU cores used by GridSearchCV. -1 uses all available cores.

    verbose : int
        Verbosity level for GridSearchCV.

    Returns
    -------
    grid_rf : GridSearchCV object
        Full fitted GridSearchCV object.

    best_model_rf : Pipeline
        Best fitted Random Forest pipeline.

    best_params_rf : dict
        Best hyperparameters found.

    best_score_rf : float
        Best mean cross-validation score.
    """

    # ----------------------------------------------------------
    # 1. Build Random Forest pipeline
    # ----------------------------------------------------------
    pipeline_rf = Pipeline([
        ("forest", RandomForestClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1
        ))
    ])

    # ----------------------------------------------------------
    # 2. Define temporal cross-validation
    # ----------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ----------------------------------------------------------
    # 3. Define hyperparameter grid
    # ----------------------------------------------------------
    param_grid_rf = {
        "forest__n_estimators": [100, 200, 500],
        "forest__max_depth": [3, 5, 10, None],
        "forest__min_samples_split": [2, 5, 10],
        "forest__min_samples_leaf": [1, 2, 5, 10],
        "forest__max_features": ["sqrt", "log2", None]
    }

    # ----------------------------------------------------------
    # 4. Set up GridSearchCV
    # ----------------------------------------------------------
    grid_rf = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid_rf,
        scoring=scoring,
        cv=tscv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )

    # ----------------------------------------------------------
    # 5. Train model
    # ----------------------------------------------------------
    grid_rf.fit(X_train, y_train)

    # ----------------------------------------------------------
    # 6. Extract best model and results
    # ----------------------------------------------------------
    best_model_rf = grid_rf.best_estimator_
    best_params_rf = grid_rf.best_params_
    best_score_rf = grid_rf.best_score_

    print("Best Random Forest parameters:")
    print(best_params_rf)

    print("\nBest mean CV macro F1:")
    print(best_score_rf)

    return grid_rf, best_model_rf, best_params_rf, best_score_rf

model_order = [
    "FEATURES + SVM",
    "PCA + SVM",
    "PCA + Decision Tree",
    "PCA + Random Forest",
]

patient_number_map = {
    "10OXG": "P1",
    "1JSZ6": "P2",
    "3ZL8B": "P3",
    "F88R2": "P4",
    "FP628": "P5",
    "JYXFE": "P6",
    "PN12G": "P7",
    "RQXZ1": "P8",
    "XB47Y": "P9"
}

patient_order = [
    "P1", "P2", "P3", "P4", "P5",
    "P6", "P7", "P8", "P9"
]

CB_color_cycle = [
    "#2E0F4F",
    "#FF5F00",
    "#1B7837",
    "#f781bf",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00"
]

patient_colors = {
    patient: CB_color_cycle[i]
    for i, patient in enumerate(patient_order)
}

#=================================================================================
#=================================================================================
#=================================================================================
# Function #5


def clean_filename(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text

def parse_classification_filename(filepath):
    """
    Extract metadata from classification table CSV filenames.
 
    Handles:
    1. SVM filenames:
       JYXFE_IN-PCA_SVM-SCORING-F1-MACRO_20260608_v01_Test_classification_table.csv

    2. Decision Tree / Random Forest filenames:
       PAT-JYXFE_Test_-_Decision_Tree_classification_table.csv

    Rule:
    Decision Tree and Random Forest are always PCA-based.
    """

    name = Path(filepath).name

    # -------------------------------------------------
    # Pattern 1: SVM
    # -------------------------------------------------
    svm_pattern = re.match(
        r"(?P<patient_id>.+?)_IN-(?P<input_type>FEATURES|PCA)_"
        r"(?P<model>SVM)-SCORING-(?P<scoring>.+?)_"
        r"(?P<date>\d{8})_(?P<version>v\d+)_(?P<dataset>Test|Validation)_classification_table\.csv",
        name
    )

    if svm_pattern:
        info = svm_pattern.groupdict()

        info["model"] = "SVM"
        info["model_group"] = f"{info['input_type']} + SVM"
        info["filename"] = name
        info["filepath"] = str(filepath)
        info["parse_error"] = False

        return info

    # -------------------------------------------------
    # Pattern 2: Decision Tree / Random Forest
    # -------------------------------------------------
    tree_rf_pattern = re.match(
        r"PAT-(?P<patient_id>.+?)_"
        r"(?P<dataset>Test|Validation)_-_"
        r"(?P<model>Decision_Tree|Random_Forest)_classification_table\.csv",
        name
    )

    if tree_rf_pattern:
        info = tree_rf_pattern.groupdict()

   
        info["input_type"] = "PCA"
        info["model"] = info["model"].replace("_", " ")
        info["scoring"] = None
        info["date"] = None
        info["version"] = None
        info["model_group"] = f"{info['input_type']} + {info['model']}"
        info["filename"] = name
        info["filepath"] = str(filepath)
        info["parse_error"] = False

        return info

    # -------------------------------------------------
    # If no pattern matched
    # -------------------------------------------------
    return {
        "patient_id": None,
        "input_type": None,
        "model": None,
        "scoring": None,
        "date": None,
        "version": None,
        "dataset": None,
        "model_group": None,
        "filename": name,
        "filepath": str(filepath),
        "parse_error": True
    }
def plot_metric_from_summary_long(
    summary_df_long,
    metric_name,
    dataset_to_plot="Test",
    baseline=0.5,
    output_path=None,
    show_plot=False,
    add_stats_text=True,
    add_median_line=True
):
    """
    Plot one metric from summary_df_long.

    Each point represents one patient.

    Uses:
    - Numeric patient labels: P1-P9
    - Same patient color palette used in previous plots
    - Same circular marker for all patients
    - Fixed horizontal offset per patient to avoid overlap
    - Optional baseline at 0.5
    - Optional median line per ML model
    - Mean/median/n text above the 0-1 metric range
    - Saves plot as SVG
    """

    df_plot = summary_df_long.copy()

    df_plot = df_plot[
        (df_plot["dataset"] == dataset_to_plot) &
        (df_plot["metric_name"] == metric_name)
    ].copy()

    # -------------------------------------------------
    # Convert original patient IDs to numeric patient labels
    # -------------------------------------------------

    df_plot["Patient"] = df_plot["patient_id"].map(patient_number_map)

    already_numeric_mask = df_plot["patient_id"].isin(patient_order)
    df_plot.loc[already_numeric_mask, "Patient"] = df_plot.loc[
        already_numeric_mask,
        "patient_id"
    ]

    unmapped_patients = df_plot.loc[
        df_plot["Patient"].isna(),
        "patient_id"
    ].unique()

    if len(unmapped_patients) > 0:
        print("Warning: these patient IDs were not mapped:", unmapped_patients)

    df_plot = df_plot.dropna(subset=["Patient"])

    df_plot["Patient"] = pd.Categorical(
        df_plot["Patient"],
        categories=patient_order,
        ordered=True
    )

    # -------------------------------------------------
    # Keep model order consistent
    # -------------------------------------------------

    current_order = [
        model for model in model_order
        if model in df_plot["model_group"].unique()
    ]

    df_plot["model_group"] = pd.Categorical(
        df_plot["model_group"],
        categories=current_order,
        ordered=True
    )

    df_plot = df_plot.sort_values(["model_group", "Patient"])

    # -------------------------------------------------
    # Calculate mean and median per model category
    # -------------------------------------------------

    stats_df = (
        df_plot
        .groupby("model_group", observed=True)["metric_value"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )

    print(f"\n{dataset_to_plot} - {metric_name}")
    print(stats_df)

    # -------------------------------------------------
    # Patient offset mapping
    # -------------------------------------------------

    present_patients = [
        patient for patient in patient_order
        if patient in df_plot["Patient"].astype(str).unique()
    ]

    if len(present_patients) == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.16, 0.16, len(present_patients))

    patient_to_offset = {
        patient: offsets[i]
        for i, patient in enumerate(present_patients)
    }

    model_to_x = {
        model: i
        for i, model in enumerate(current_order)
    }

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(11, 5.5))

    y_upper = 1.12
    y_stats = 1.09

    for _, row in df_plot.iterrows():

        model_group = row["model_group"]
        patient = str(row["Patient"])
        y_value = row["metric_value"]

        x_value = model_to_x[model_group] + patient_to_offset[patient]

        ax.scatter(
            x_value,
            y_value,
            marker="o",                      
            s=95,
            facecolors=patient_colors[patient],
            edgecolors="black",
            linewidths=1.0,
            alpha=0.90,
            zorder=3
        )

    # -------------------------------------------------
    # Median line per model
    # -------------------------------------------------

    if add_median_line:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            median_value = row["median"].iloc[0]

            ax.hlines(
                y=median_value,
                xmin=i - 0.24,
                xmax=i + 0.24,
                color="black",
                linewidth=2.0,
                alpha=0.45,
                zorder=4
            )

    # -------------------------------------------------
    # Optional horizontal baseline
    # -------------------------------------------------

    if baseline is not None:
        ax.axhline(
            y=baseline,
            linestyle="--",
            linewidth=1.5,
            color="black",
            alpha=0.7,
            zorder=1
        )

    # -------------------------------------------------
    # Add mean and median text above plot points
    # -------------------------------------------------

    if add_stats_text:
        for i, model_group in enumerate(current_order):

            row = stats_df[stats_df["model_group"] == model_group]

            if row.empty:
                continue

            mean_value = row["mean"].iloc[0]
            median_value = row["median"].iloc[0]
            count_value = int(row["count"].iloc[0])

            stats_text = (
                f"mean={mean_value:.2f}\n"
                f"med={median_value:.2f}\n"
                f"n={count_value}"
            )

            ax.text(
                x=i,
                y=y_stats,
                s=stats_text,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.8
                ),
                zorder=5
            )

    # -------------------------------------------------
    # Axis formatting
    # -------------------------------------------------

    ax.set_title(f"{dataset_to_plot} - {metric_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_name)

    ax.set_ylim(0, y_upper)

    ax.set_xticks(range(len(current_order)))
    ax.set_xticklabels(current_order, rotation=25, ha="right")

    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # -------------------------------------------------
    # Custom legend for patients, median and baseline
    # -------------------------------------------------

    patient_handles = [
        Line2D(
            [0],
            [0],
            marker="o",                      
            color="w",
            markerfacecolor=patient_colors[patient],
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            markersize=8,
            label=patient
        )
        for patient in present_patients
    ]

    legend_handles = patient_handles.copy()

    if add_median_line:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.0,
                alpha=0.45,
                label="Median"
            )
        )

    if baseline is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Baseline = {baseline}"
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Patient",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

    plt.tight_layout()

    # -------------------------------------------------
    # Save as SVG
    # -------------------------------------------------

    if output_path is None:
        today = date.today().strftime("%Y-%m-%d")
        filename = (
            f"{clean_filename(dataset_to_plot)}_"
            f"{clean_filename(metric_name)}_"
            f"model_comparison_patient_palette_median_line_{today}.svg"
        )
        output_path = Path.cwd() / filename
    else:
        output_path = Path(output_path)

        if output_path.suffix.lower() != ".svg":
            output_path = output_path.with_suffix(".svg")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path,
        format="svg",
        bbox_inches="tight"
    )

    print(f"SVG saved in: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

def extract_classification_metrics(csv_path):
    """
    Extract accuracy, weighted F1, preictal F1 and ictal/seizure F1
    from one classification report CSV.

    Expected rows in class_or_metric:
    - preictal
    - seizure or ictal
    - accuracy
    - weighted avg
    """

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)


    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    if "class_or_metric" not in df.columns:
        raise ValueError(
            f"'class_or_metric' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    if "f1-score" not in df.columns:
        raise ValueError(
            f"'f1-score' column not found in {csv_path.name}. "
            f"Columns found: {list(df.columns)}"
        )

    df["class_or_metric_clean"] = (
        df["class_or_metric"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    def get_f1_from_row(row_name_options):
        """
        Extract f1-score from the first matching row.
        row_name_options can be a string or a list of strings.
        """

        if isinstance(row_name_options, str):
            row_name_options = [row_name_options]

        row_name_options = [
            str(x).strip().lower()
            for x in row_name_options
        ]

        matched_row = df[
            df["class_or_metric_clean"].isin(row_name_options)
        ]

        if matched_row.empty:
            return np.nan

        return float(matched_row["f1-score"].iloc[0])

    metrics = {
        "accuracy": get_f1_from_row("accuracy"),
        "weighted_f1": get_f1_from_row("weighted avg"),
        "preictal_f1": get_f1_from_row("preictal"),
        "ictal_f1": get_f1_from_row(["ictal", "seizure"])
    }

    return metrics


# ============================================================
# 2. Build full metrics summary dataframe
# ============================================================

def build_full_metrics_summary_df(metadata_df):
    """
    Each classification CSV becomes one row:
    patient_id | dataset | input_type | model | model_group |
    accuracy | weighted_f1 | preictal_f1 | ictal_f1
    """

    summary_rows = []

    for _, row in metadata_df.iterrows():

        if row.get("parse_error", False):
            continue

        csv_path = row["filepath"]

        metrics = extract_classification_metrics(csv_path)

        summary_rows.append({
            "patient_id": row["patient_id"],
            "dataset": row["dataset"],
            "input_type": row["input_type"],
            "model": row["model"],
            "model_group": row["model_group"],
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "preictal_f1": metrics["preictal_f1"],
            "ictal_f1": metrics["ictal_f1"],
            "filename": row["filename"],
            "filepath": row["filepath"]
        })

    summary_metrics_df = pd.DataFrame(summary_rows)

    return summary_metrics_df

def build_long_metrics_summary_df(summary_metrics_df):
    """
    Convert the full metrics dataframe from wide to long format.
    """

    metric_name_map = {
        "accuracy": "Accuracy",
        "weighted_f1": "Weighted F1",
        "preictal_f1": "Preictal F1",
        "ictal_f1": "Ictal F1",
    }

    id_columns = [
        "patient_id",
        "dataset",
        "input_type",
        "model",
        "model_group",
        "filename",
        "filepath",
    ]

    summary_df_long = summary_metrics_df.melt(
        id_vars=id_columns,
        value_vars=list(metric_name_map.keys()),
        var_name="metric_key",
        value_name="metric_value",
    )

    summary_df_long["metric_name"] = (
        summary_df_long["metric_key"].map(metric_name_map)
    )

    return summary_df_long
# ============================================================
# 3. Create compact best-model table
# ============================================================

def build_best_model_compact_table(
    summary_metrics_df,
    dataset_to_use="Test",
    output_path=None
):
    """
    Build compact table:

    patient_id
    best_by_accuracy
    accuracy
    best_by_weighted_f1
    weighted_f1
    preictal_f1
    ictal_f1
    same_best_model

    The F1 values reported are from the model selected by weighted F1.
    """

    df = summary_metrics_df.copy()

    df = df[df["dataset"] == dataset_to_use].copy()

    if df.empty:
        raise ValueError(f"No rows found for dataset: {dataset_to_use}")

    # --------------------------------------------------------
    # Best model per patient by accuracy
    # --------------------------------------------------------

    idx_best_acc = (
        df
        .groupby("patient_id")["accuracy"]
        .idxmax()
    )

    best_acc_df = (
        df.loc[idx_best_acc, [
            "patient_id",
            "model_group",
            "accuracy"
        ]]
        .rename(columns={
            "model_group": "best_by_accuracy",
            "accuracy": "best_accuracy"
        })
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Best model per patient by weighted F1
    # --------------------------------------------------------

    idx_best_f1 = (
        df
        .groupby("patient_id")["weighted_f1"]
        .idxmax()
    )

    best_f1_df = (
        df.loc[idx_best_f1, [
            "patient_id",
            "model_group",
            "weighted_f1",
            "preictal_f1",
            "ictal_f1",
            "accuracy"
        ]]
        .rename(columns={
            "model_group": "best_by_weighted_f1",
            "accuracy": "accuracy_of_best_weighted_f1_model"
        })
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Merge both selections
    # --------------------------------------------------------

    compact_table = best_acc_df.merge(
        best_f1_df,
        on="patient_id",
        how="outer"
    )

    compact_table["same_best_model"] = (
        compact_table["best_by_accuracy"]
        == compact_table["best_by_weighted_f1"]
    )



    metric_cols = [
        "best_accuracy",
        "weighted_f1",
        "preictal_f1",
        "ictal_f1",
        "accuracy_of_best_weighted_f1_model"
    ]

    for col in metric_cols:
        compact_table[col] = compact_table[col].round(3)

    compact_table = compact_table.sort_values("patient_id").reset_index(drop=True)

    # --------------------------------------------------------
    # Save if requested
    # --------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".csv":
            compact_table.to_csv(output_path, index=False)
        elif output_path.suffix in [".xlsx", ".xls"]:
            compact_table.to_excel(output_path, index=False)
        else:
            raise ValueError("output_path must end in .csv or .xlsx")

        print(f"Compact table saved to: {output_path}")

    return compact_table
from pathlib import Path

import pandas as pd


def build_validation_selected_model_table(
    summary_metrics_df,
    output_path=None,
):
    """
    Select the best model configuration for each patient using validation
    weighted F1 and retrieve the corresponding test-set metrics.

    Parameters
    ----------
    summary_metrics_df : pandas.DataFrame
        Summary dataframe containing one row per patient, dataset and
        model configuration.

        Required columns:
        - patient_id
        - dataset
        - model_group
        - accuracy
        - weighted_f1
        - preictal_f1
        - ictal_f1

    output_path : str or pathlib.Path, optional
        CSV path where the final table will be saved.

    Returns
    -------
    pandas.DataFrame
        Compact patient-level table containing the validation-selected
        configuration and its corresponding test performance.
    """

    required_columns = {
        "patient_id",
        "dataset",
        "model_group",
        "accuracy",
        "weighted_f1",
        "preictal_f1",
        "ictal_f1",
    }

    missing_columns = (
        required_columns
        - set(summary_metrics_df.columns)
    )

    if missing_columns:
        raise KeyError(
            "The following required columns are missing from "
            f"summary_metrics_df: {sorted(missing_columns)}"
        )

    working_df = summary_metrics_df.copy()

    # ----------------------------------------------------------
    # 1. Normalise dataset names
    # ----------------------------------------------------------

    working_df["_dataset_normalized"] = (
        working_df["dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    validation_df = working_df[
        working_df["_dataset_normalized"].isin(
            ["validation", "val"]
        )
    ].copy()

    test_df = working_df[
        working_df["_dataset_normalized"].eq("test")
    ].copy()

    if validation_df.empty:
        raise ValueError(
            "No validation results were found in summary_metrics_df."
        )

    if test_df.empty:
        raise ValueError(
            "No test results were found in summary_metrics_df."
        )

    # Ensure weighted F1 is numeric before using idxmax
    validation_df["weighted_f1"] = pd.to_numeric(
        validation_df["weighted_f1"],
        errors="coerce",
    )

    if validation_df["weighted_f1"].isna().any():
        invalid_rows = validation_df.loc[
            validation_df["weighted_f1"].isna(),
            ["patient_id", "model_group"],
        ]

        raise ValueError(
            "Some validation weighted F1 values could not be "
            f"interpreted as numbers:\n{invalid_rows}"
        )

    # ----------------------------------------------------------
    # 2. Select configuration using validation weighted F1
    # ----------------------------------------------------------

    best_validation_idx = (
        validation_df
        .groupby(
            "patient_id",
            observed=True,
        )["weighted_f1"]
        .idxmax()
    )

    selected_validation_df = (
        validation_df.loc[
            best_validation_idx,
            [
                "patient_id",
                "model_group",
                "weighted_f1",
            ],
        ]
        .rename(
            columns={
                "model_group":
                    "Configuration selected by validation",
                "weighted_f1":
                    "Validation weighted F1",
            }
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # 3. Prepare test metrics
    # ----------------------------------------------------------

    selected_test_metrics_df = (
        test_df[
            [
                "patient_id",
                "model_group",
                "accuracy",
                "weighted_f1",
                "preictal_f1",
                "ictal_f1",
            ]
        ]
        .rename(
            columns={
                "model_group":
                    "Configuration selected by validation",
                "accuracy":
                    "Test accuracy",
                "weighted_f1":
                    "Test weighted F1",
                "preictal_f1":
                    "Test preictal F1",
                "ictal_f1":
                    "Test ictal F1",
            }
        )
    )

    # There should be only one test row per patient/configuration
    duplicated_test_rows = selected_test_metrics_df.duplicated(
        subset=[
            "patient_id",
            "Configuration selected by validation",
        ],
        keep=False,
    )

    if duplicated_test_rows.any():
        duplicates = selected_test_metrics_df.loc[
            duplicated_test_rows,
            [
                "patient_id",
                "Configuration selected by validation",
            ],
        ]

        raise ValueError(
            "Multiple test rows were found for the same patient "
            f"and configuration:\n{duplicates}"
        )

    # ----------------------------------------------------------
    # 4. Match validation-selected configurations with test
    # ----------------------------------------------------------

    final_model_summary_df = selected_validation_df.merge(
        selected_test_metrics_df,
        on=[
            "patient_id",
            "Configuration selected by validation",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_test_results = final_model_summary_df[
        "Test weighted F1"
    ].isna()

    if missing_test_results.any():
        missing_rows = final_model_summary_df.loc[
            missing_test_results,
            [
                "patient_id",
                "Configuration selected by validation",
            ],
        ]

        raise FileNotFoundError(
            "No matching test result was found for the following "
            f"validation-selected configurations:\n{missing_rows}"
        )

    # ----------------------------------------------------------
    # 5. Format final table
    # ----------------------------------------------------------

    final_model_summary_df = (
        final_model_summary_df
        .rename(
            columns={
                "patient_id": "Patient ID",
            }
        )
        [
            [
                "Patient ID",
                "Configuration selected by validation",
                "Validation weighted F1",
                "Test accuracy",
                "Test weighted F1",
                "Test preictal F1",
                "Test ictal F1",
            ]
        ]
        .sort_values("Patient ID")
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------
    # 6. Save table
    # ----------------------------------------------------------

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_model_summary_df.to_csv(
            output_path,
            index=False,
        )

    return final_model_summary_df