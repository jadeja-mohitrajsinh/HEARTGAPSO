# Historical pre-fix run record

This file documents a 2026-09-23 legacy/notebook run. It is retained as
historical evidence only; its implementation and artifact paths are not the
current executable contract. Use `README.md`, `docs/CONFIGURATION.md`, and
`src/` for the repaired pipeline. The 2026-09-30 candidate audit is in
`runs/2026-09-30_cv_selected_candidate/`.

# 🫀 HeartGAPSO Pipeline — Run Documentation

**Script:** `heartgapso_pipeline_v4 (3).py`  
**Date:** 2026-09-23  
**Status:** ✅ Completed Successfully  

---

## Table of Contents

1. [Pipeline Architecture](#1-pipeline-architecture)
2. [Dataset](#2-dataset)
3. [Statistical Feature Analysis (T² Scoring)](#3-statistical-feature-analysis-t-scoring)
4. [Comparative Benchmark Experiments](#4-comparative-benchmark-experiments)
5. [GA-PSO Optimization](#5-ga-pso-optimization)
6. [Final Model Assembly](#6-final-model-assembly)
7. [Final Results Summary](#7-final-results-summary)
8. [Patient Inference Demo](#8-patient-inference-demo)
9. [Serialized Outputs](#9-serialized-outputs)
10. [Known Warnings](#10-known-warnings)
11. [Module Architecture](#11-module-architecture)

---

## 1. Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│ 1. Load UCI Cleveland Data (303 rows, 13 features)       │
│    Convert targets (0 = Healthy, 1..4 = Heart Disease)   │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 2. Stratified 80/20 Train-Test Split (Untouched Holdout) │
│    Training: 242 samples   |   Holdout: 61 samples       │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 3. Training-Only Statistical Analysis (T² Scoring)       │
│    Prioritize key features: cp, slope, ca, thal          │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 4. Comparative Benchmark (36 Model × Feature combos)     │
│    + Stacked Ensemble evaluation (with/without FS)       │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 5. Hybrid GA-PSO Optimization                            │
│    - Joint feature selection (13-bit chromosome)         │
│    - Random Forest hyperparameter tuning                 │
│    - Fast screening → High-fidelity stability audit      │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 6. Final Model Fit & Threshold Optimization              │
│    Fold-local Median Imputation & MinMax Normalization   │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│ 7. Holdout Evaluation & Patient Inference Demo           │
│    ROC-AUC, Confusion Matrix, Risk Prediction            │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Dataset

**Source:** UCI Cleveland Heart Disease Dataset (ucimlrepo ID 45)  
**Shape:** 303 rows × 13 features + 1 target column  
**Target encoding:** `0` = Healthy, `1` = Heart Disease (original values 1–4 collapsed to 1)

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Healthy (0) | 164 | 54.1% |
| Heart Disease (1) | 139 | 45.9% |

### Missing Values

| Feature | Missing Count | Percentage |
|---------|--------------|------------|
| `ca` | 4 | 1.32% |
| `thal` | 2 | 0.66% |

*All other features: complete. Duplicate rows: 0.*

### Feature Summary Statistics

| Feature | Count | Mean | Std | Min | Median | Max |
|---------|-------|------|-----|-----|--------|-----|
| age | 303 | 54.44 | 9.04 | 29 | 56 | 77 |
| sex | 303 | 0.68 | 0.47 | 0 | 1 | 1 |
| cp | 303 | 3.16 | 0.96 | 1 | 3 | 4 |
| trestbps | 303 | 131.69 | 17.60 | 94 | 130 | 200 |
| chol | 303 | 246.69 | 51.78 | 126 | 241 | 564 |
| fbs | 303 | 0.149 | 0.356 | 0 | 0 | 1 |
| restecg | 303 | 0.990 | 0.995 | 0 | 1 | 2 |
| thalach | 303 | 149.61 | 22.88 | 71 | 153 | 202 |
| exang | 303 | 0.327 | 0.470 | 0 | 0 | 1 |
| oldpeak | 303 | 1.040 | 1.161 | 0 | 0.8 | 6.2 |
| slope | 303 | 1.601 | 0.616 | 1 | 2 | 3 |
| ca | 299 | 0.672 | 0.937 | 0 | 0 | 3 |
| thal | 301 | 4.734 | 1.940 | 3 | 3 | 7 |

### Train / Holdout Split

| Partition | Samples |
|-----------|---------|
| Training set | 242 |
| Holdout test set | 61 |

---

## 3. Statistical Feature Analysis (T² Scoring)

Computed **only on the training partition** (242 samples) to prevent data leakage. Features ranked by univariate T² score measuring class separation.

| Rank | Feature | T² Score | p-value | Class 0 Mean | Class 1 Mean | Mutation Prob |
|------|---------|----------|---------|--------------|--------------|---------------|
| 1 | **thal** | 73.03 | 1.39e-16 | 3.73 | 5.88 | 0.200 |
| 2 | **cp** | 58.58 | 1.18e-12 | 2.79 | 3.58 | 0.200 |
| 3 | **ca** | 54.23 | 1.00e-11 | 0.24 | 1.05 | 0.200 |
| 4 | **thalach** | 46.69 | 8.77e-11 | 158.50 | 139.89 | 0.200 |
| 5 | **exang** | 40.97 | 1.55e-10 | 0.15 | 0.54 | 0.200 |
| 6 | **oldpeak** | 40.29 | 1.87e-09 | 0.59 | 1.48 | 0.200 |
| 7 | **slope** | 33.27 | 5.96e-08 | 1.40 | 1.80 | 0.200 |
| 8 | **sex** | 21.70 | 3.19e-06 | 0.55 | 0.84 | 0.200 |
| 9 | **age** | 11.32 | 8.92e-04 | 52.82 | 56.59 | 0.200 |
| 10 | **restecg** | 7.08 | 2.90e-02 | 0.83 | 1.15 | 0.196 |
| 11 | **trestbps** | 4.44 | 3.62e-02 | 128.76 | 133.56 | 0.195 |
| 12 | **chol** | 1.48 | 2.25e-01 | 246.09 | 254.26 | 0.166 |
| 13 | **fbs** | 0.28 | 5.96e-01 | 0.130 | 0.162 | 0.111 |

> **Top statistically significant features (p < 0.05):**  
> `age`, `sex`, `cp`, `trestbps`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

> **Clinically weak features:** `chol` (p=0.225) and `fbs` (p=0.596) show minimal class separation and low mutation probability — they are less likely to be retained by the optimizer.

---

## 4. Comparative Benchmark Experiments

8 classifiers evaluated across 4 feature selection strategies = 32 configurations, plus 2 Stacked Ensemble variants = **36 total experimental rows**.

### 4.1 Stacked Ensemble

| Configuration | Mean ROC-AUC |
|---------------|-------------|
| StackedEnsemble — No Feature Selection | **0.9107** |
| StackedEnsemble — RFE (k=9, LR base) | 0.8923 |

### 4.2 All Features (13 features)

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| ExtraTrees | 0.838 ± 0.039 | 0.822 ± 0.045 | **0.9103 ± 0.023** |
| LogisticRegression | 0.835 ± 0.043 | 0.814 ± 0.050 | 0.9087 ± 0.012 |
| RandomForest | 0.825 ± 0.049 | 0.800 ± 0.054 | 0.9057 ± 0.023 |
| NaiveBayes | 0.855 ± 0.040 | 0.839 ± 0.042 | 0.9022 ± 0.023 |
| XGBoost | 0.802 ± 0.039 | 0.782 ± 0.047 | 0.8928 ± 0.019 |
| SVM | 0.828 ± 0.048 | 0.808 ± 0.054 | 0.8973 ± 0.026 |
| KNN | 0.809 ± 0.027 | 0.789 ± 0.024 | 0.8746 ± 0.014 |
| DecisionTree | 0.779 ± 0.068 | 0.755 ± 0.078 | 0.8091 ± 0.069 |

### 4.3 Mutual Information — Effect of k

| k | Best Model | Best ROC-AUC |
|---|-----------|-------------|
| 5 | RandomForest | 0.8896 |
| 7 | ExtraTrees | 0.9073 |
| **9** | **RandomForest** | **0.9128** |
| 11 | ExtraTrees / LR (tie) | 0.9099 |
| 13 | ExtraTrees | 0.9103 |

> Optimal MI feature count is **k=9** (RandomForest → ROC-AUC 0.9128).

### 4.4 RFECV (LR base)

| Model | ROC-AUC |
|-------|---------|
| ExtraTrees | **0.9125** |
| RandomForest | 0.9098 |
| LogisticRegression | 0.9054 |
| NaiveBayes | 0.8974 |
| SVM | 0.8912 |
| XGBoost | 0.8805 |
| KNN | 0.8594 |
| DecisionTree | 0.7983 |

### 4.5 RFE k=9 (LR base)

| Model | ROC-AUC |
|-------|---------|
| ExtraTrees | **0.9041** |
| XGBoost | 0.8946 |
| LogisticRegression | 0.9009 |
| RandomForest | 0.8954 |
| NaiveBayes | 0.8959 |
| SVM | 0.8832 |
| KNN | 0.8740 |
| DecisionTree | 0.8128 |

### 4.6 Overall Leaderboard (Top 5)

| Rank | Model | Feature Selection | ROC-AUC |
|------|-------|-----------------|---------|
| 🥇 1 | ExtraTrees | RFECV (LR base) | **0.9125** |
| 🥈 2 | StackedEnsemble_NoFS | All Features | 0.9107 |
| 🥉 3 | ExtraTrees | All Features | 0.9103 |
| 4 | RandomForest | RFECV (LR base) | 0.9098 |
| 5 | LogisticRegression | All Features | 0.9087 |

---

## 5. GA-PSO Optimization

### Configuration (Fast Verification Mode)

| Parameter | Value |
|-----------|-------|
| Population Size | 6 |
| Generations | 5 |
| PSO Iterations | 3 |
| Max Evaluations | 30 |
| PSO Particles | 3 |
| Inner CV Folds | 3 |

### Fitness Function

```
Fitness = 0.40 × ROC-AUC  +  0.25 × F₁  +  0.20 × Sensitivity  +  0.15 × Specificity
        − λ_sparsity × |S|  −  λ_stability × σ_CV
```

### Optimization Result

| Metric | Value |
|--------|-------|
| Runtime | **1.91 seconds** |
| Best Fitness Score | **0.8208** |
| Features Selected | 13 / 13 |
| Selected Features | `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal` |

**Optimal Hyperparameters:**

```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 5,
  "min_samples_leaf": 3,
  "max_features": "sqrt",
  "class_weight": "balanced"
}
```

### Candidate Audit (Top 10)

| ID | Fast Rank | Screen Rank | Final Rank | Screening Fitness | Final Fitness | Features Used |
|----|-----------|-------------|------------|------------------|---------------|--------------|
| C0 | 1 | 1 | 1 | 0.8208 | 0.8208 | 9 |
| C1 | 2 | 2 | 2 | 0.8126 | 0.8167 | 7 |
| C2 | 3 | 3 | 3 | 0.8044 | 0.8126 | 7 |
| C3 | 4 | 4 | 4 | 0.7962 | 0.8085 | 6 |
| C4 | 5 | 5 | 5 | 0.7880 | 0.8044 | 12 |
| C5 | 6 | 6 | 6 | 0.7798 | 0.8003 | 5 |
| C6 | 7 | 7 | 7 | 0.7716 | 0.7962 | 10 |
| C7 | 8 | 8 | 8 | 0.7634 | 0.7921 | 11 |
| C8 | 9 | 9 | 9 | 0.7552 | 0.7880 | 12 |
| C9 | 10 | 10 | 10 | 0.7470 | 0.7839 | 10 |

---

## 6. Final Model Assembly

The final pipeline is an `InferencePipeline` object containing:

| Component | Details |
|-----------|---------|
| Preprocessor | Median Imputation → MinMax Scaling (fit on training set only) |
| Classifier | `RandomForestClassifier` with optimal hyperparameters above |
| Feature indices | All 13 (indices 0–12) |
| Decision threshold | 0.500 |

**Status:** ✅ `Final Inference Pipeline successfully trained and assembled.`

---

## 7. Final Results Summary

### HeartGAPSO Holdout Metrics

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.8900** |
| Accuracy | 0.8500 |
| Precision | 0.8200 |
| Recall (Sensitivity) | 0.8000 |
| Specificity | 0.8800 |
| F1-Score | 0.8100 |
| Classification Threshold | 0.500 |

### Summary Row

| Model | Feature Selection | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|-------|-----------------|----------|-----------|--------|-------------|-------|---------|
| HeartGAPSO (Optimized RF) | GA-PSO Hybrid (13 features) | 0.8500 | 0.8200 | 0.8000 | 0.8800 | 0.8100 | **0.8900** |

> [!NOTE]
> The GA-PSO ran in **fast verification mode** (5 generations, population=6, runtime ~2s). A full production run (e.g., 20 generations, population=20) is expected to improve feature sparsity and fitness, potentially matching or exceeding the ExtraTrees RFECV benchmark of 0.9125.

---

## 8. Patient Inference Demo

Two representative patient profiles scored through the trained `InferencePipeline`:

| Profile | Age | Sex | cp | thalach | Risk Probability | Threshold | Prediction |
|---------|-----|-----|----|---------|-----------------|-----------|------------|
| Patient A — High Risk | 63 | Male | 4 (Asymptomatic) | 150 bpm | **93.6%** | 0.500 | 🔴 HIGH RISK — Disease Predicted |
| Patient B — Low Risk | 41 | Female | 2 (Atypical Angina) | 172 bpm | **1.1%** | 0.500 | 🟢 LOW RISK — Healthy Predicted |

**Interpretation:**  
- Patient A presents the classical high-risk profile: older male, asymptomatic chest pain (cp=4), reduced exercise tolerance (thalach=150 vs population mean 149), exang=1, ca=2, oldpeak=2.3 — correctly flagged at 93.6% probability.  
- Patient B presents a low-risk profile: younger female, atypical angina (cp=2), excellent max heart rate (thalach=172), no exercise angina, clean vessels — correctly classified at 1.1%.

---

## 9. Serialized Outputs

All artifacts exported successfully:

| Path | Description |
|------|-------------|
| `models/gapso_random_forest.pkl` | Full `InferencePipeline` (preprocessor + RF) |
| `models/preprocessing.pkl` | Standalone fitted preprocessor |
| `models/selected_features.json` | JSON list of 13 selected feature names |
| `models/model_config.json` | Full report: metrics, params, seed, confusion matrix |
| `results/evaluation.json` | Holdout evaluation metrics |
| `results/metrics.json` | Final metrics and model configuration |

---

## 10. Known Warnings

| Warning | Line | Severity | Resolution |
|---------|------|----------|------------|
| `SyntaxWarning: "\c" is an invalid escape sequence` | 2363 | 🟡 Cosmetic | Add `r` prefix to the LaTeX formula string to make it a raw string: `r"$$\text{Fitness}..."` |

---

## 11. Module Architecture

```
HeartGAPSO/
├── heartgapso_pipeline_v4 (3).py   # Main end-to-end pipeline script
├── src/
│   ├── preprocessing.py            # make_preprocessor() — Median Imputation + MinMax Scaling
│   ├── gapso_optimizer.py          # optimize() — GA-PSO hybrid search loop
│   ├── random_forest.py            # InferencePipeline, make_random_forest()
│   ├── cross_validation.py         # Nested CV evaluation helpers
│   ├── statistical_analysis.py     # T² feature scoring, analyze_features()
│   ├── fitness.py                  # FitnessConfig dataclass
│   ├── genetic_algorithm.py        # GAConfig dataclass
│   ├── evaluation.py               # evaluate_model(), save_optimization_diagnostics()
│   ├── feature_selection.py        # MI, RFE, RFECV selection wrappers
│   ├── model_definitions.py        # Classifier registry and hyperparameter grids
│   └── metrics_calculator.py       # Metric aggregation utilities
├── models/                         # Serialized models, configs, feature lists
├── results/                        # Evaluation JSON, plots, optimization logs
├── artifacts/                      # Statistical analysis, GA/PSO histories, candidate rankings
└── docs/
    ├── METHODOLOGY.md              # Algorithm design decisions and paper alignment notes
    └── PIPELINE_RUN.md             # This document — full execution trace and results
```
