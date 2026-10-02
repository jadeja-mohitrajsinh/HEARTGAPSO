# -*- coding: utf-8 -*-
"""HeartGAPSO Pipeline v4: Complete Machine Learning & Metaheuristic System
Hybrid Genetic Algorithm (GA) & Particle Swarm Optimization (PSO) with Multi-Model Benchmarking, Nested CV & Clinical Inference.
"""

import copy
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

# Force UTF-8 output on Windows terminals (CP1252 can't handle emoji)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress verbose deprecation/future warnings from sklearn & xgboost
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

# Safe display helper for terminal and Jupyter
try:
    display  # type: ignore # noqa: F821
except NameError:
    def display(x):
        if isinstance(x, pd.DataFrame):
            print(x.to_string())
        else:
            print(x)

# Safe ucimlrepo loader with offline fallback
try:
    from ucimlrepo import fetch_ucirepo
    HAS_UCIMLREPO = True
except ImportError:
    HAS_UCIMLREPO = False

# Ensure current working directory is first in sys.path
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

# Import modular components from src
from src.data_loader import FEATURE_NAMES, load_cleveland
from src.preprocessing import make_preprocessor
from src.statistical_analysis import analyze_features, save_analysis
from src.fitness import FitnessConfig
from src.genetic_algorithm import GAConfig
from src.gapso_optimizer import optimize
from src.random_forest import InferencePipeline, make_random_forest
from src.evaluation import evaluate_model, save_optimization_diagnostics
from src.model_definitions import get_models, get_stacked_ensemble
from src.cross_validation import run_nested_cv
from src.metrics_calculator import calculate_metrics, format_metrics


def main():
    print("=" * 75)
    print("🫀 HeartGAPSO Pipeline v4: End-to-End Execution")
    print("=" * 75)

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)

    # -------------------------------------------------------------
    # 1. Data Ingestion
    # -------------------------------------------------------------
    print("\n[Step 1/8] Ingesting UCI Cleveland Heart Disease Dataset...")
    X, y = None, None
    if HAS_UCIMLREPO:
        try:
            print("Fetching from UCI ML Repository (id=45)...")
            heart_disease = fetch_ucirepo(id=45)
            X_raw = heart_disease.data.features.copy()
            y_raw = heart_disease.data.targets.copy()
            X_raw.columns = [str(c).strip() for c in X_raw.columns]
            target_col = y_raw.columns[0]
            y_series = pd.to_numeric(y_raw[target_col], errors="coerce")
            y = (y_series > 0).astype(int)
            X = X_raw[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
            print(f"✓ Fetched {len(X)} instances from ucimlrepo.")
        except Exception as e:
            print(f"Notice: ucimlrepo fetch failed ({e}). Using local dataset.")

    if X is None or y is None:
        local_path = Path("heart+disease/processed.cleveland.data")
        print(f"Loading local dataset from {local_path}...")
        X, y = load_cleveland(local_path)
        print(f"✓ Loaded {len(X)} instances from local storage.")

    df_preview = X.copy()
    df_preview["target"] = y

    print(f"Dataset Shape: {X.shape[0]} patients, {X.shape[1]} features")
    print(f"Class Balance: Healthy (0) = {(y == 0).sum()} ({(y == 0).mean():.1%}), Disease (1) = {(y == 1).sum()} ({(y == 1).mean():.1%})")

    # -------------------------------------------------------------
    # 2. Stratified 80/20 Holdout Split
    # -------------------------------------------------------------
    print("\n[Step 2/8] Creating Stratified 80/20 Train-Test Split (Untouched Holdout)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Training partition (for CV & Optimization): {X_train.shape[0]} samples")
    print(f"Untouched holdout test partition: {X_test.shape[0]} samples")

    # -------------------------------------------------------------
    # 3. Statistical Feature Significance Analysis (T² Scoring)
    # -------------------------------------------------------------
    print("\n[Step 3/8] Computing Training-Only T² Statistical Significance...")
    statistical_analysis = analyze_features(X_train, y_train)
    save_analysis(statistical_analysis, "artifacts/statistical_feature_analysis.json")
    print(f"Top Statistically Significant Features for GA Seeding: {statistical_analysis['important_features']}")

    stats_df = pd.DataFrame(statistical_analysis["features"])
    print("\nTop 5 Statistical Features by T² Score:")
    display(stats_df[["importance_rank", "feature", "t2_statistical_score", "class_0_mean", "class_1_mean"]].head(5))

    # -------------------------------------------------------------
    # 4. Multi-Model Baseline & Feature Selection Benchmark
    # -------------------------------------------------------------
    print("\n[Step 4/8] Running 5-Fold Nested Cross-Validation Benchmark Suite...")
    models = get_models(RANDOM_SEED)
    feature_selection_configs = [
        {"name": "All Features (13)", "method": "all", "fs_params": {}},
        {"name": "Mutual Information (k=8)", "method": "mutual_info", "fs_params": {"k": 8}},
        {"name": "RFE (Logistic Regression, 8)", "method": "rfe", "fs_params": {"n_features_to_select": 8}},
    ]

    param_grids = {
        "DecisionTree": {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]},
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [4, 6, 8, None]},
        # CalibratedClassifierCV wraps SVC, so params use estimator__ prefix
        "SVM": {"estimator__C": [0.1, 1.0, 10.0], "estimator__kernel": ["linear", "rbf"]},
    }

    benchmark_results = []
    for fs_cfg in feature_selection_configs:
        fs_name = fs_cfg["name"]
        fs_method = fs_cfg["method"]
        fs_params = fs_cfg["fs_params"]

        for model_name, model_est in models.items():
            res = run_nested_cv(
                X=X_train,
                y=y_train,
                model_name=model_name,
                model_estimator=model_est,
                feature_selection_method=fs_method,
                fs_params=fs_params,
                param_grid=param_grids.get(model_name, {}),
                outer_cv_folds=5,
                inner_cv_folds=3,
                random_state=RANDOM_SEED,
            )
            res["feature_selection"] = fs_name
            benchmark_results.append(res)

    stacked_clf = get_stacked_ensemble(models, random_state=RANDOM_SEED)
    stacked_res = run_nested_cv(
        X=X_train,
        y=y_train,
        model_name="Stacked Ensemble (SVM+XGB+ET)",
        model_estimator=stacked_clf,
        feature_selection_method="all",
        outer_cv_folds=5,
        inner_cv_folds=3,
        random_state=RANDOM_SEED,
    )
    stacked_res["feature_selection"] = "All Features (13)"
    benchmark_results.append(stacked_res)

    benchmark_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Feature Selection": r["feature_selection"],
            "CV Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.3f}",
            "CV ROC-AUC": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.3f}",
            "CV F1-Score": f"{r['f1_mean']:.4f} ± {r['f1_std']:.3f}",
            "CV Sensitivity": f"{r['recall_mean']:.4f} ± {r['recall_std']:.3f}",
            "CV Specificity": f"{r['specificity_mean']:.4f} ± {r['specificity_std']:.3f}",
            "_raw_auc": r["roc_auc_mean"],
        }
        for r in benchmark_results
    ]).sort_values(by="_raw_auc", ascending=False).drop(columns=["_raw_auc"])

    print("\n--- Baseline Cross-Validation Benchmark (Top 5) ---")
    display(benchmark_df.head(5))

    # -------------------------------------------------------------
    # 5. Hybrid GAPSO Optimization Engine
    # -------------------------------------------------------------
    print("\n[Step 5/8] Running Hybrid GAPSO Optimization Engine...")
    FITNESS_CONFIG = FitnessConfig(
        folds=5,
        repeats=5,
        roc_auc_weight=0.40,
        f1_weight=0.30,
        sensitivity_weight=0.15,
        specificity_weight=0.15,
        feature_penalty=0.002,
        stability_penalty=0.06,
    )

    GA_CONFIG = GAConfig(
        population_size=60,
        generations=50,
        pso_iterations=25,
        max_evaluations=1200,
        pso_particles=25,
    )

    start_time = time.perf_counter()
    opt_result = optimize(
        X=X_train,
        y=y_train,
        seed=RANDOM_SEED,
        ga_config=GA_CONFIG,
        fitness_config=FITNESS_CONFIG,
        high_fidelity_candidate_limit=40,
        stability_candidates=8,
        fast_promotions=40,
        stability_seeds=7,
        statistical_analysis=statistical_analysis,
    )
    opt_runtime = time.perf_counter() - start_time

    selected_feature_indices = [idx for idx, bit in enumerate(opt_result["features"]) if bit]
    selected_feature_names = [FEATURE_NAMES[idx] for idx in selected_feature_indices]

    print(f"\n✓ GAPSO Optimization finished in {opt_runtime:.2f}s!")
    print(f"Best Fitness Score: {opt_result['fitness']:.4f}")
    print(f"Selected Features ({len(selected_feature_names)}/13): {selected_feature_names}")
    print(f"Optimal Hyperparameters: {json.dumps(opt_result['params'], indent=2)}")
    print(f"Calibrated Classification Threshold: {opt_result['threshold']:.3f}")

    # -------------------------------------------------------------
    # 6. Final Model Training & Pipeline Assembly
    # -------------------------------------------------------------
    print("\n[Step 6/8] Fitting Final Random Forest & Assembling Inference Pipeline...")
    preprocessor = make_preprocessor()
    X_train_selected = X_train.iloc[:, selected_feature_indices]
    X_train_processed = preprocessor.fit_transform(X_train_selected)

    final_rf = make_random_forest(opt_result["params"], RANDOM_SEED)
    final_rf.fit(X_train_processed, y_train)

    pipeline = InferencePipeline(
        preprocessing=preprocessor,
        classifier=final_rf,
        indices=selected_feature_indices,
        threshold=opt_result["threshold"],
    )

    # -------------------------------------------------------------
    # 7. Holdout Evaluation & Diagnostics
    # -------------------------------------------------------------
    print("\n[Step 7/8] Evaluating on Untouched Holdout Test Partition (N=61)...")
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = pipeline.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_test_prob)

    metrics_summary = {
        "Accuracy": float(acc),
        "ROC-AUC": float(roc_auc),
        "Sensitivity / Recall": float(rec),
        "Specificity": float(spec),
        "Precision": float(prec),
        "F1-Score": float(f1),
        "Classification Threshold": float(pipeline.threshold),
    }

    print("\n=== Final Holdout Test Results ===")
    for k, v in metrics_summary.items():
        print(f" - {k:25s}: {v:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Save Diagnostics & Curves
    save_optimization_diagnostics(opt_result["history"], opt_result, "results")

    # ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1b9e77", lw=2.5, label=f"HeartGAPSO (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance")
    plt.title("Holdout Receiver Operating Characteristic (ROC)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=160)
    plt.close()

    # Confusion Matrix Plot
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=np.array([[tn, fp], [fn, tp]]),
        display_labels=["Healthy (0)", "Disease (1)"],
    )
    cm_display.plot(cmap="Blues", colorbar=False)
    plt.title(f"Holdout Confusion Matrix (Acc={acc:.1%})")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=160)
    plt.close()

    # -------------------------------------------------------------
    # 8. Serialization & Artifact Export
    # -------------------------------------------------------------
    print("\n[Step 8/8] Serializing Models and Reports...")
    report = {
        **metrics_summary,
        "selected_features": selected_feature_names,
        "number_selected_features": len(selected_feature_names),
        "best_rf_parameters": opt_result["params"],
        "optimization_fitness": opt_result["fitness"],
        "optimization_runtime_seconds": opt_runtime,
        "classification_threshold": pipeline.threshold,
        "seed": RANDOM_SEED,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    with open("models/gapso_random_forest.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    with open("models/preprocessing.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    Path("models/selected_features.json").write_text(json.dumps(selected_feature_names, indent=2))
    Path("models/model_config.json").write_text(json.dumps(report, indent=2, default=str))
    Path("results/evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    Path("results/metrics.json").write_text(json.dumps(report, indent=2, default=str))

    # -------------------------------------------------------------
    # 9. Patient Inference Demo
    # -------------------------------------------------------------
    print("\n--- Clinical Patient Risk Inference Demo ---")
    sample_patients = pd.DataFrame([
        {
            "Patient": "Patient A (Severe Presentation)",
            "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 1,
            "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.6, "slope": 2,
            "ca": 3, "thal": 7,
        },
        {
            "Patient": "Patient B (Low Risk Healthy)",
            "age": 41, "sex": 0, "cp": 2, "trestbps": 120, "chol": 175, "fbs": 0,
            "restecg": 0, "thalach": 178, "exang": 0, "oldpeak": 0.0, "slope": 1,
            "ca": 0, "thal": 3,
        },
    ])

    patient_features = sample_patients[FEATURE_NAMES]
    probs = pipeline.predict_proba(patient_features)[:, 1]
    preds = pipeline.predict(patient_features)

    for i, row in sample_patients.iterrows():
        pred_label = "HIGH RISK (Disease)" if preds[i] == 1 else "LOW RISK (Healthy)"
        print(f"[{row['Patient']}] Predicted Probability: {probs[i]:.1%}, Threshold: {pipeline.threshold:.3f} => {pred_label}")

    print("\n✓ Full pipeline execution completed successfully!")


if __name__ == "__main__":
    main()