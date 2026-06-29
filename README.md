# GeoPredict
### Advanced Multi-Model ML Pipeline for Automated Subsurface Facies Classification

An end-to-end data science pipeline that automates expert-level lithofacies interpretation from raw wireline log curves, wrapped in a live interactive web application for use by asset teams and petrophysicists.

🔗 **[Live Interactive Dashboard](https://facies-prediction-dashboard-yrvvff4umzeyf36ptmqbxu.streamlit.app/)**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Engineering Pipeline](#engineering-pipeline)
- [Dashboard Features](#dashboard-features)
- [Performance Metrics](#performance-metrics)
- [Local Setup](#local-setup)
- [Tech Stack](#tech-stack)

---

## Project Overview

Identifying rock facies (lithology types) from wireline well logs is foundational to reservoir characterization, stratigraphic correlation, and economic asset evaluation. Traditionally, this workflow depends on physical core inspections manually correlated with log signatures by senior domain experts — a process that is time-consuming, expensive, and scales poorly across large basins.

GeoPredict treats subsurface characterization as a **multi-class sequence classification task**. Using a rich, highly imbalanced dataset of 9 wells from the **Council Grove Field (Hugoton & Panoma gas reservoirs, Kansas)** as ground truth, the pipeline ingests standard wireline log curves and outputs instant lithology classifications.

### Core Objectives

| Objective | Description |
|---|---|
| **Multi-Model Benchmarking** | Systematically evaluate Random Forest and XGBoost against a Sequential 1D-CNN |
| **Spatial Window Hypothesis** | Demonstrate that depth-windowed sequences outperform per-sample classification |
| **Facies 5 Stability** | Resolve classification instability over Facies 5, a volatile petrophysical transition zone |

---

## Repository Structure

```
├── app.py                                        # Production Streamlit web application
├── requirements.txt                              # Python 3.12 package dependencies
├── scaler.joblib                                 # Fitted StandardScaler (training baseline)
├── best_baseline_rf.joblib                       # Pre-trained Random Forest model
├── best_baseline_xgb.joblib                      # Pre-trained XGBoost model
├── best_1d_cnn.h5                                # Pre-trained Keras 1D-CNN model
│
├── notebooks/
│   └── facies_classification_training.ipynb      # EDA, hyperparameter tuning, SHAP analysis
│
└── sample data/
    └── sample_well.las                           # Example wireline log for evaluation testing
```

---

## Engineering Pipeline

### Phase A — Exploratory Data Analysis & QC

- **Class Imbalance Mapping** — Quantified frequency variance across 9 target facies layers to prevent naive models from ignoring thin, economically vital reservoir beds.
- **Smart Imputation** — Missing Photoelectric Effect (PE) curves are filled using a per-well median, falling back to a global average only when a well omits the tool string entirely.
- **Data Leakage Prevention** — The StandardScaler is fit strictly on the training split before transforming validation and test arrays, ensuring mathematically honest metrics.

### Phase B — Feature Engineering

- **GR_PHIND_ratio** — A custom feature interaction mapping Gamma Ray to Neutron Porosity, isolating clean porous reservoir rock from high-API shaly zones.
- **Windowed Sequences** — 3-sample rolling means and standard deviations computed per well, preventing depth-boundary contamination across geographically separate wells.

### Phase C — Model Architecture

**Random Forest**
Configured with `class_weight='balanced'` and tuned via 3-fold cross-validated `GridSearchCV` over tree depths and estimator counts.

**XGBoost**
Uses a `multi:softprob` objective. Targets are shifted (`y - 1`) during training for 0-indexed compatibility and mapped back to the true 1–9 scale during inference.

**Sequential 1D-CNN**
Built with TensorFlow/Keras to model continuous vertical depositional patterns. A sequence-slicing window (size = 5) converts 2D scaled data into a 3D array `(samples, 5, features)`, enabling the model to leverage contextual depth intervals above and below each sample — outperforming tree models on complex transitional boundaries.

---

## Dashboard Features

The Streamlit application replicates a high-end commercial petrophysical software experience:

- **Automated Alias Resolution** — Maps vendor-dependent column headers (e.g., `GGCE → GR`, `AHT90 → ILD`) via an internal regex dictionary, with manual override dropdowns for full user control.
- **Multi-Model Inference Workspace** — Sidebar toggle to switch between Random Forest, XGBoost, and 1D-CNN predictions. Missing model assets trigger a graceful fallback with an inline warning.
- **Advanced Visual Log Strips** — Vertical Matplotlib strip charts mimicking professional log layout records, with optional engineering tracks (Gamma Ray, Deep Resistivity, NM_M, RELPOS) alongside the facies prediction strip.
- **Interactive Plotly Crossplots** — White-themed scatter plots of Gamma Ray vs. Resistivity with hover inspection of depth, logging metrics, and classification labels.
- **Automated Performance Scorecard** — If the uploaded `.las` file contains pre-labeled `FACIES` columns, the dashboard auto-generates global accuracy and a full `classification_report`.
- **Data Export** — One-click download of processed log data and continuous facies predictions as a clean `.csv` for downstream workflows.

---

## Performance Metrics

Due to severe class imbalance, the pipeline is evaluated on **Macro-Averaged F1-Score** alongside raw accuracy, following [SEG 2016 ML Contest](https://github.com/seg/2016-ml-contest) guidelines.

| Metric | Target |
|---|---|
| Micro Accuracy | ~70% |
| Macro F1-Score | ~0.69 |

**SHAP Explainability** — SHAP (SHapley Additive exPlanations) analysis is integrated into the training notebook to verify that model predictions are driven by physically meaningful log variations (e.g., high GR → shale, low porosity → dense limestone) rather than arbitrary noise.

---

## Local Setup

**Prerequisites:** Python 3.11 or 3.12 (avoid 3.14 for TensorFlow wheel compatibility)

```bash
# 1. Clone the repository
git clone https://github.com/prgoff/facies-prediction-dashboard.git
cd facies-prediction-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `XGBoost` · `Scikit-learn` · `Streamlit` · `Matplotlib` · `Plotly` · `SHAP` · `LAS File Processing`

**Domains:** Geophysics · Petrophysics · Well-Log Analysis · Machine Learning · Deep Learning (1D-CNN)· Lithofacies Classification · Explainable AI
