# EEG Foundation Model for Seizure Forecasting  
MSc Dissertation – Applied Bioinformatics  
King’s College London  

---

## 1. Motivation

Epilepsy affects approximately 1% of the global population, and up to 30–40% of patients continue to experience uncontrolled seizures despite medication. One of the main challenges in epilepsy management is the unpredictability of seizures, which can occur with little or no warning and significantly affect patients’ ability to work, travel, and plan daily activities safely.

Reliable seizure forecasting has the potential to improve quality of life and clinical decision-making, representing a meaningful step toward precision neurology.

---

## 2. Project Vision

This project contributes to the development of a pilot foundation model for seizure forecasting using EEG.

The core hypothesis is that training flexible deep learning architectures (e.g., Transformer-based models) on large, heterogeneous EEG datasets using self-supervised objectives can produce transferable internal representations of seizure dynamics.

Rather than overfitting to a single curated dataset, the goal is to learn a shared embedding space that generalises across:

- Different hospitals and acquisition protocols  
- Scalp EEG and intracranial EEG  
- Heterogeneous patient populations  
- Potentially different species (human and rodent recordings)

---

## 3. Rationale for a Foundation-Style Approach

Most existing seizure prediction models:

- Rely on hand-crafted features  
- Use relatively small, curated datasets  
- Show limited cross-patient generalisation  
- Require patient-specific tuning  

A foundation-style model aims to:

- Learn representations directly from raw EEG  
- Capture general pre-ictal and ictal dynamics  
- Enable transfer learning across patients and centres  
- Provide a reusable embedding space for downstream tasks such as detection, forecasting, and network analysis  

---

## 4. Scope of This MSc Project

The broader research programme aims to train a cross-dataset EEG foundation model across heterogeneous EEG/iEEG datasets.

This MSc project focuses on a core component of that effort:

### Data Engineering and Preprocessing Pipeline

Specifically:

- Loading and harmonising large-scale `.mat` EEG recordings  
- Extracting and structuring recording metadata (T0, TF, sampling rate, channels)  
- Aligning seizure annotations with EEG signals  
- Implementing reproducible preprocessing steps  
- Generating windowed datasets suitable for self-supervised learning  

This repository contains the tools and exploratory analyses developed for this stage.

---

## 5. Repository Structure

```text
.
└── EEG_data_vis
    └── code
        ├── basic_reader_of_mats.ipynb
        ├── csv_reader.ipynb
        ├── debuggin_mapper.ipynb
        ├── mapper_of_signals.ipynb
        ├── mix_readers.ipynb
        ├── preProcessPipeline_test.ipynb
        ├── test_sampleXB47Y.ipynb
        ├── tools_EEG.py
        └── util_EEG.ipynb
```

---

## 6. Module Overview

### Data Loading

- `basic_reader_of_mats.ipynb`  
- `mix_readers.ipynb`  
- `csv_reader.ipynb`  

Load raw `.mat` EEG recordings and associated metadata.  
Parse timestamps, channel information, and sampling rate.

---

### Signal Mapping and Annotation Alignment

- `mapper_of_signals.ipynb`  
- `debuggin_mapper.ipynb`  

Align EEG recordings with seizure onset annotations.  
Construct recording availability maps and validate temporal consistency.

---

### Preprocessing Utilities

- `preProcessPipeline_test.ipynb`  
- `tools_EEG.py`  
- `util_EEG.ipynb`  

Implements:

- Bandpass filtering (e.g., 0.5–40 Hz)  
- Z-score normalisation  
- Amplitude thresholding  
- Fixed-length windowing (e.g., 10-second segments)  
- Data reshaping for model input  

---

### Dataset Validation

- `test_sampleXB47Y.ipynb`  

Performs sanity checks on:

- Signal distributions  
- Recording duration consistency  
- Alignment between seizure diary and EEG-confirmed seizures  

---

## 7. Technical Focus Areas

This project lies at the intersection of:

- Artificial Intelligence for Health  
- Neurophysiology  
- Time-Series Analysis  
- Large-Scale Data Engineering  

Methodological components include:

- Self-supervised learning (e.g., masked reconstruction, contrastive learning)  
- Representation learning for EEG  
- Statistical analysis of seizure clustering  
- Circadian pattern analysis  
- Cross-patient generalisation challenges  

---

## 8. Long-Term Research Direction

The embedding space learned by a future foundation model could support:

- Seizure forecasting  
- Seizure detection  
- Patient similarity modelling  
- Identification of seizure-generating networks  
- Chronotherapy-informed treatment optimisation  

---

## 9. Author

Tomás Pérez  
MSc Applied Bioinformatics  
King’s College London  

Research interests:

- Foundation models for biomedical signals  
- AI in precision medicine  
- Translational neuroinformatics  
- Seizure forecasting  
