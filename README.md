# Developing a Baseline Machine Learning Model for Seizure Classification Using Novel Ultra-Long-term Subcutaneous EEG

**MSc Dissertation – Applied Bioinformatics**  
**King's College London, 2026**  
**Dissertation repository**

**Author:** Tomás Pérez Sánchez  
**Supervisors:** Dr. Dominic Burrows and Dr. Richard Rosch

---

## Repository logic: start here

The source code is divided into three directories with different purposes:

### `test_and_develop/`

This directory contains exploratory notebooks, debugging work, prototype functions, intermediate analyses, and code used during development.

- It documents how the analysis was developed.
- Its contents are not required to reproduce the final workflow.
- Code in this directory may include older or experimental versions of functions.
- Finalised functionality was transferred into `modules/` and called from `pipelines/`.

### `modules/`

This directory contains reusable Python functions used by the final analysis.

- `tools_EEG_Preprocess.py`: loading, signal mapping, preprocessing, filtering, normalisation, and storage utilities.
- `tools_EEG_FE.py`: windowing, labelling, feature extraction, statistical analysis, PCA, and feature visualisation utilities.
- `tools_EEG_models.py`: machine-learning, evaluation, summary, and model-comparison utilities.
- The module files are imported by pipeline scripts and are not normally executed directly.

### `pipelines/`

This directory contains the final executable scripts used to reproduce the dissertation workflow.

- Scripts are organised in the order in which the analysis was performed.
- Pipeline scripts call functions from `modules/`.
- Patient-specific paths and analysis parameters are separated from the core code through JSON configuration files.
- For each configurable stage, first edit and run the corresponding JSON generator, then pass the generated JSON file to the pipeline script.

The intended workflow is therefore:

```text
test_and_develop/  -> exploratory development only
modules/           -> reusable analysis functions
pipelines/         -> final executable workflow
```

and, during execution:

```text
Edit JSON generator
        |
        v
Generate configuration file
        |
        v
Run pipeline script with the JSON configuration
        |
        v
Save processed data, figures, tables, and model outputs
```

---

## 1. Project overview

This repository contains the code developed for an MSc Applied Bioinformatics dissertation investigating whether interpretable handcrafted features extracted from ultra-long-term subcutaneous electroencephalography (sqEEG) can distinguish between preictal and ictal periods.

The project develops patient-specific baseline machine-learning models using real-world sqEEG recordings. Its main purpose is to establish a transparent and reproducible benchmark before progressing towards more complex seizure-prediction or seizure-forecasting approaches.

The implemented task is **retrospective binary classification of preictal and ictal windows**. It is not a prospective seizure-warning system and does not perform continuous seizure-risk forecasting. Instead, it evaluates whether the selected sqEEG representation contains discriminative information that could support future forecasting research.

---

## 2. Scientific motivation

Seizure unpredictability has a substantial effect on the autonomy and quality of life of people with epilepsy. Patient-reported seizure diaries can also be incomplete or inaccurate, which motivates the use of objective long-term cerebral monitoring.

Subcutaneous EEG provides a minimally invasive compromise between conventional scalp EEG and intracranial EEG. It enables near-continuous, real-world monitoring over periods that are not feasible with routine scalp EEG, while avoiding the invasiveness associated with intracranial recordings.

This project uses ultra-long-term sqEEG as a basis for developing an interpretable patient-specific classification framework and for assessing how seizure-related signal patterns vary across patients.

---

## 3. Study scope

The analysis used a subset of **nine patients with treatment-resistant focal epilepsy** from the prospective sqEEG cohort reported by Viana et al. (2025).

The dataset included:

- Two sqEEG channels per patient.
- A sampling frequency of approximately 207 Hz.
- Sequential raw EEG recordings stored as `.mat` files.
- Recording metadata, including sampling frequency, channel labels, and timestamps.
- Clinically reviewed seizure-onset annotations.
- Long-duration real-world recordings with substantial variation in recording coverage, seizure frequency, and patient-specific signal characteristics.

The analysis was conducted independently for each patient to preserve patient-specific temporal and electrophysiological patterns.

---

## 4. Research aim

The overall aim was to develop and evaluate interpretable baseline machine-learning models using handcrafted sqEEG features to discriminate between ictal and preictal windows.

The methodological framework addresses the following questions:

1. Which handcrafted sqEEG features show the strongest discriminability between preictal and ictal windows?
2. Does PCA-based dimensionality reduction improve SVM classification compared with the original feature representation?
3. How does performance vary across SVM, Decision Tree, and Random Forest classifiers?
4. Are differences in patient-level performance associated with feature discriminability or recording characteristics?
5. Can handcrafted-feature models provide an interpretable reference for future seizure-forecasting research?

---

## 5. Analysis workflow

### 5.1 Raw-data exploration and temporal mapping

The raw `.mat` structure was inspected to identify:

- EEG signal arrays.
- Sampling frequency.
- Channel labels.
- Recording start times.
- Recording duration.
- Seizure-onset timestamps.

Sample indices were converted to elapsed recording time using the sampling frequency. Seizure annotations were then mapped onto the reconstructed recording timeline.

Daily recording-availability plots and seizure-onset maps were used as quality-control checks before downstream processing.

### 5.2 Preprocessing

The final preprocessing workflow included:

1. **Amplitude cut-off:** extreme values outside ±200 µV were treated as non-physiological artefacts.
2. **Band-pass filtering:** a channel-wise Butterworth filter between 0.5 and 48 Hz.
3. **Optional notch filtering:** applied when persistent narrow-band technical peaks were identified.
4. **Global within-patient, channel-wise normalisation:** the mean and standard deviation were calculated from all valid finite samples for each patient and channel, and the same reference statistics were applied across that patient's recordings.
5. **Compressed storage:** processed recordings and relevant metadata were stored as `.npz` files.

Global within-patient normalisation was selected instead of independently normalising each recording so that inter-recording amplitude differences were not removed before extracting amplitude-sensitive features.

### 5.3 Windowing and labels

Recordings were divided into fixed-length **10-second windows**.

Windows were initially assigned to one of three temporal categories:

- **Preictal:** the one-minute interval from 6 to 5 minutes before seizure onset.
- **Ictal:** the one-minute interval immediately after seizure onset.
- **Interictal:** all remaining windows.

The final statistical and machine-learning analyses were restricted to preictal and ictal windows. Equal-duration class intervals were used to provide a standardised exploratory baseline. These temporal definitions are configurable and should not be interpreted as universal physiological definitions of the preictal state.

### 5.4 Handcrafted feature extraction

Fourteen features were calculated separately for each sqEEG channel, producing **28 features per window**.

| Feature subgroup | Extracted features |
|---|---|
| Amplitude level and magnitude | Mean amplitude, root mean square, peak-to-peak amplitude |
| Amplitude variability | Standard deviation, variance |
| Temporal waveform change / signal complexity | Line length |
| Amplitude distribution shape | Skewness, kurtosis |
| Band-specific spectral power | Delta, theta, alpha, beta, gamma power |
| Dominant spectral frequency | Peak frequency |

Frequency-domain features were derived from Welch power spectral density estimates.

### 5.5 Feature discriminability

Each channel-wise feature was compared between preictal and ictal windows using a Mann–Whitney U test.

The exploratory criteria were:

- **Statistically significant feature:** `p < 0.05`.
- **Large effect feature:** `|log2FC| > 1`.

No multiple-comparison correction was applied in this exploratory baseline analysis.

P-values were represented as negative log10 p-values so that stronger statistical evidence appears as larger positive values. Effect magnitude and direction were represented using the log2 median fold-change between ictal and preictal feature values:

- Positive log2FC: higher feature values during ictal windows.
- Negative log2FC: higher feature values during preictal windows.
- Zero: no median fold-change between classes.

Feature-level outputs included patient-specific statistical tables, violin plots, volcano plots, and cross-patient summary plots.

### 5.6 Principal Component Analysis

PCA was applied as an alternative representation of the 28 handcrafted features.

- Features were standardised before PCA.
- PCA was fitted to the extracted feature dataset after feature standardisation.
- The retained components explained approximately 90% of cumulative variance.

This representation was compared with the original handcrafted-feature representation in the SVM analysis.

### 5.7 Temporal dataset splitting

Windows were ordered chronologically and divided without random shuffling:

- Training set: approximately 70%.
- Validation set: approximately 15%.
- Test set: approximately 15%.

Split boundaries were selected to approximate the requested proportions while preserving a similar preictal-to-ictal class ratio across subsets.

Chronological splitting and temporally ordered cross-validation were used to reduce leakage between adjacent EEG windows.

### 5.8 Machine-learning models

The following patient-specific configurations were evaluated:

| Input representation | Classifier |
|---|---|
| Original handcrafted features | Support Vector Machine with RBF kernel |
| PCA-transformed features | Support Vector Machine with RBF kernel |
| PCA-transformed features | Decision Tree |
| PCA-transformed features | Random Forest |

Hyperparameter optimisation was conducted using the training set. Model selection during cross-validation was based on macro F1-score, after which the selected model was refitted and evaluated on the validation and test sets.

Reported outputs included:

- Accuracy.
- Weighted F1-score.
- Preictal F1-score.
- Ictal F1-score.
- Classification reports.
- Confusion matrices.
- Patient-level and cohort-level model-comparison plots.

---

## 6. Repository structure

```text
.
├── README.md
├── environment.yml
└── Main_project
    └── src
        ├── modules
        │   ├── Guide.ipynb
        │   ├── guide_Tools_EEG.ipynb
        │   ├── tools_EEG_Preprocess.py
        │   ├── tools_EEG_FE.py
        │   └── tools_EEG_models.py
        │
        ├── pipelines
        │   ├── 1_Pre_process
        │   │   ├── configs
        │   │   ├── json_GENERATOR_1Preprocessing.py
        │   │   ├── json_GENERATOR_1Normalization.py
        │   │   ├── 1_preprocess_PIPELINE.py
        │   │   └── 1_Normalization_PIPELINE.py
        │   │
        │   ├── 2_Feature_ext
        │   │   ├── configs
        │   │   ├── JSON_configGenerator_FE_Part1_Labeling.py
        │   │   ├── JSON_configGenerator_FE_Part2_FeaturesDF.py
        │   │   ├── JSON_configGenerator_FE_Part3_PCA.py
        │   │   ├── JSON_configGenerator_FE_Part4_stats.py
        │   │   ├── 2_1_Feature_extraction_Labeling_PIPELINE.py
        │   │   ├── 2_2_Feature_extraction_FE_PIPELINE.py
        │   │   ├── 2_3_Feature_extraction_PCA_PIPELINE.py
        │   │   ├── 2_4_Feature_extraction_Stats_PIPELINE.py
        │   │   └── 2_5_Feature_extraction_Global_volcano_n_Barplot.py
        │   │
        │   ├── 3_SVM_pilot
        │   │   ├── configs
        │   │   ├── JSONconfig_3_SVM.py
        │   │   └── 3_SVM_pipeline_PCAorFeatures.py
        │   │
        │   └── 4_Models
        │       ├── configs
        │       ├── JSON_generator_DTandRF.py
        │       ├── 4_DTandRF.py
        │       ├── 4_2_MRA_analysis.py
        │       └── 4_3_model_summary_plots.py
        │
        └── test_and_develop
            ├── 1_Pre_processing
            ├── 2_Feature_extraction
            ├── 3_SVM_pilot
            └── 4_Models
```

Generated data, figures, model objects, and summary tables are written to output directories specified in the JSON configuration files.

---

## 7. Installation

### 7.1 Clone the repository

```bash
git clone https://github.com/tomas0899/Tomas_PS_DissertationKCL2026.git
cd Tomas_PS_DissertationKCL2026
```

### 7.2 Create the software environment

The project was developed using Python 3.10 in a micromamba environment named `domain_expansion`.

```bash
micromamba env create -f environment.yml
micromamba activate domain_expansion
```

The environment can also be created with Conda:

```bash
conda env create -f environment.yml
conda activate domain_expansion
```

The exported environment contains Linux-specific packages because the analysis was developed on a Linux x86-64 remote workstation. Users on macOS or Windows may need to create a platform-adjusted environment using the principal dependencies listed below.

Main analysis dependencies include:

- Python 3.10
- NumPy
- pandas
- SciPy
- scikit-learn
- statsmodels
- Matplotlib
- seaborn
- h5py
- openpyxl
- JupyterLab
- tqdm
- PyYAML

---

## 8. Running the pipeline

Run the stages in numerical order:

```text
1_Pre_process
2_Feature_ext
3_SVM_pilot
4_Models
```

For each configurable stage:

1. Open the corresponding JSON generator script.
2. Set the patient identifier, input paths, output paths, and analysis parameters.
3. Replace any existing absolute paths with paths valid on the local system.
4. Run the generator to create a configuration file inside the relevant `configs/` directory.
5. Run the corresponding pipeline script and provide the generated JSON file as its argument.

General pattern:

```bash
python <JSON_generator_script>.py
python <pipeline_script>.py <path/to/generated_config.json>
```

Example structure:

```bash
cd Main_project/src/pipelines/1_Pre_process

python json_GENERATOR_1Preprocessing.py
python 1_preprocess_PIPELINE.py configs/<generated_preprocessing_config>.json
```

Repeat the same generator-to-pipeline logic for normalisation, labelling, feature extraction, PCA, statistics, SVM modelling, and Decision Tree/Random Forest modelling.

### Important path configuration

Some generator scripts contain absolute paths from the original development workstation. These must be changed before running the code on another system.

Input and output locations should be kept outside the reusable function modules and defined through the JSON configuration files wherever possible.

---

## 9. Expected outputs

Depending on the selected pipeline stage, outputs may include:

- Processed and normalised `.npz` recordings.
- Recording-availability and seizure-mapping plots.
- Window-level label tables.
- Channel-wise handcrafted-feature datasets.
- PCA-transformed datasets.
- Mann–Whitney U statistical summaries.
- Negative log10 p-values and log2FC tables.
- Violin plots, volcano plots, and ranked feature plots.
- Serialised machine-learning models.
- Validation and test predictions.
- Confusion matrices and classification reports.
- Patient-level model summaries.
- Cross-patient performance plots.
- Exploratory multiple-regression summaries.

---

## 10. Data availability and privacy

The raw and processed sqEEG recordings are not distributed with this public repository.

All patient identifiers used in the analysis code, figures, tables, and generated outputs are fully anonymised. The repository therefore contains the computational framework and configuration structure, but not the underlying patient recordings.

---

## 11. Interpretation and limitations

This project provides an exploratory patient-specific baseline and should be interpreted within the following scope:

- The task compares restricted preictal and ictal intervals rather than performing continuous forecasting.
- Interictal windows were excluded from the final binary classification analysis.
- The preictal definition is an adjustable methodological choice, not a universally established biomarker.
- sqEEG has restricted spatial coverage because it uses a small number of subcutaneous electrodes.
- Models were evaluated within patients; cross-patient generalisation was not tested.
- The software is intended for research and academic use and is not a clinical decision-support system.

---

## 12. Future directions

Potential extensions include:

- Adding interictal windows through balanced negative sampling.
- Evaluating alternative preictal horizons and window durations.
- Performing patient-specific feature selection.
- Adding relationships between features, such as spectral-band ratios.
- Evaluating temporal changes across consecutive windows.
- Investigating circadian and sleep-related effects.
- Performing leave-one-patient-out and other cross-patient validation strategies.
- Comparing the baseline with deep-learning and self-supervised representations.
- Using the interpretable baseline to support future seizure-risk forecasting models.

---

## 13. Reference dataset

The sqEEG dataset analysed in this dissertation was derived from:

> Viana, P. F., Duun-Henriksen, J., Biondi, A., Winston, J. S., Freestone, D. R., Schulze-Bonhage, A., Brinkmann, B. H., & Richardson, M. P. (2025). Real-world epilepsy monitoring with ultra-long-term subcutaneous electroencephalography: A 15-month prospective study. *Epilepsia, 66*(11), 4476–4489. https://doi.org/10.1111/epi.18566

---

## 15. Author

**Tomás Pérez Sánchez**  
MSc Applied Bioinformatics  
King's College London

**Supervisors:** Dr. Dominic Burrows and Dr. Richard Rosch