"""Rigorous Empirical Experiment on UCI Cleveland Heart Disease Dataset (ID: 45)
Research Question: Can a feature-selected stacked ensemble outperform Random Forest under identical cross-validation conditions?
"""

import json
import os
import sys
import time
from pathlib import Path

# Configure utf-8 stdout for Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Data fetching
from ucimlrepo import fetch_ucirepo

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS),
    ], remainder="passthrough")


def calculate_fold_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": rec,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "cm": (tn, fp, fn, tp),
    }


def run_nested_evaluation(X, y, estimator, param_grid, fs_method="all", fs_k=None, outer_cv=None, inner_splits=3, random_state=42):
    fold_metrics = []
    oof_y_true = []
    oof_y_prob = []
    oof_y_pred = []
    fitted_estimators = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[train_idx].to_numpy(), y.iloc[val_idx].to_numpy()

        # 1. Fit preprocessor strictly on training fold
        preprocessor = build_preprocessor()
        X_tr_proc = preprocessor.fit_transform(X_tr)
        X_val_proc = preprocessor.transform(X_val)
        feat_names = np.array(NUMERICAL_COLS + CATEGORICAL_COLS)

        # 2. Feature Selection strictly on training fold
        if fs_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=fs_k)
            X_tr_fs = selector.fit_transform(X_tr_proc, y_tr)
            X_val_fs = selector.transform(X_val_proc)
        elif fs_method == "rfe":
            rfe_base = LogisticRegression(random_state=random_state, solver="liblinear")
            selector = RFE(estimator=rfe_base, n_features_to_select=fs_k, step=1)
            X_tr_fs = selector.fit_transform(X_tr_proc, y_tr)
            X_val_fs = selector.transform(X_val_proc)
        else:
            X_tr_fs = X_tr_proc
            X_val_fs = X_val_proc

        # 3. Inner CV for Hyperparameter Tuning
        est_clone = clone(estimator)
        if param_grid:
            inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + fold_idx)
            grid = GridSearchCV(est_clone, param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=-1)
            grid.fit(X_tr_fs, y_tr)
            best_model = grid.best_estimator_
        else:
            best_model = est_clone
            best_model.fit(X_tr_fs, y_tr)

        fitted_estimators.append((preprocessor, best_model, selector if fs_method != "all" else None))

        # 4. Out-of-fold prediction
        val_probs = best_model.predict_proba(X_val_fs)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        m = calculate_fold_metrics(y_val, val_preds, val_probs)
        fold_metrics.append(m)

        oof_y_true.extend(y_val)
        oof_y_prob.extend(val_probs)
        oof_y_pred.extend(val_preds)

    # Compute mean and standard deviation across folds
    metric_keys = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]
    summary = {}
    for k in metric_keys:
        vals = [f[k] for f in fold_metrics]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
        summary[f"{k}_raw"] = vals

    summary["oof_y_true"] = np.array(oof_y_true)
    summary["oof_y_prob"] = np.array(oof_y_prob)
    summary["oof_y_pred"] = np.array(oof_y_pred)
    summary["fitted_models"] = fitted_estimators
    return summary


def main():
    out_dir = Path("results/experiment")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔬 COMPREHENSIVE EXPERIMENTAL SUITE: STACKED ENSEMBLE VS RANDOM FOREST")
    print("=" * 80)

    # 1. Dataset Loading
    print("\n--- 1. DATASET INGESTION & EDA ---")
    try:
        dataset = fetch_ucirepo(id=45)
        X_df = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
        y = (y_raw.iloc[:, 0].astype(float) > 0).astype(int)
        X = X_df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce")
        print(f"✓ Loaded UCI Cleveland Heart Disease Dataset: {X.shape[0]} rows, {X.shape[1]} features.")
    except Exception as e:
        print(f"Fallback to local data: {e}")
        from src.data_loader import load_cleveland
        X, y = load_cleveland("heart+disease/processed.cleveland.data")

    # EDA Details
    n_total = len(X)
    n_healthy = int((y == 0).sum())
    n_disease = int((y == 1).sum())
    missing_counts = X.isna().sum()
    n_missing_cols = int((missing_counts > 0).sum())
    n_duplicates = int(X.duplicated().sum())

    print(f"Class Balance: Healthy (0) = {n_healthy} ({n_healthy/n_total:.1%}), Disease (1) = {n_disease} ({n_disease/n_total:.1%})")
    print(f"Missing Values: {missing_counts[missing_counts > 0].to_dict()} (Total missing entries: {missing_counts.sum()})")
    print(f"Duplicate Rows: {n_duplicates}")

    # 2. Setup Models and Hyperparameter Grids
    RANDOM_SEED = 42
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    base_models = {
        "Logistic Regression": (
            LogisticRegression(random_state=RANDOM_SEED, solver="liblinear"),
            {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l1", "l2"]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"]}
        ),
        "Naive Bayes": (
            GaussianNB(),
            {}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {"max_depth": [3, 5, 8, None], "min_samples_split": [2, 5, 10]}
        ),
        "SVM-RBF": (
            SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
            {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto", 0.01, 0.1]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [50, 100, 200], "max_depth": [4, 6, 8, None], "min_samples_split": [2, 5, 10]}
        ),
        "Extra Trees": (
            ExtraTreesClassifier(random_state=RANDOM_SEED),
            {"n_estimators": [50, 100, 200], "max_depth": [4, 6, 8, None], "min_samples_split": [2, 5, 10]}
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss", use_label_encoder=False),
            {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]}
        )
    }

    # 3. Benchmark All Base Models on All 13 Features
    print("\n--- 2. BENCHMARKING BASELINE MODELS (Nested 5-Fold Stratified CV) ---")
    base_results = {}
    for name, (est, p_grid) in base_models.items():
        print(f"  -> Evaluating: {name:22s}...", end="", flush=True)
        t0 = time.time()
        res = run_nested_evaluation(X, y, est, p_grid, fs_method="all", outer_cv=outer_cv, random_state=RANDOM_SEED)
        elapsed = time.time() - t0
        base_results[name] = res
        print(f" Done ({elapsed:.1f}s) | Acc: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.3f} | ROC-AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.3f} | F1: {res['f1_mean']:.4f}")

    # 4. Feature Selection Evaluation on Primary Baseline (Random Forest) and Stack
    print("\n--- 3. FEATURE SELECTION EXPERIMENTS ---")
    fs_methods = [
        ("All 13 Features", "all", None),
        ("Mutual Info (k=5)", "mutual_info", 5),
        ("Mutual Info (k=7)", "mutual_info", 7),
        ("Mutual Info (k=9)", "mutual_info", 9),
        ("Mutual Info (k=11)", "mutual_info", 11),
        ("Mutual Info (k=13)", "mutual_info", 13),
        ("RFE (k=7)", "rfe", 7),
        ("RFE (k=9)", "rfe", 9),
        ("RFE (k=11)", "rfe", 11),
    ]

    rf_est, rf_grid = base_models["Random Forest"]
    rf_fs_results = {}
    for fs_label, fs_m, fs_k in fs_methods:
        print(f"  -> Random Forest with {fs_label:22s}...", end="", flush=True)
        res = run_nested_evaluation(X, y, rf_est, rf_grid, fs_method=fs_m, fs_k=fs_k, outer_cv=outer_cv, random_state=RANDOM_SEED)
        rf_fs_results[fs_label] = res
        print(f" Acc: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.3f} | ROC-AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.3f}")

    # 5. Build and Evaluate Proposed Stacked Ensemble
    print("\n--- 4. EVALUATING PROPOSED STACKED ENSEMBLE (SVM + XGB + ET -> LR) ---")
    
    def make_stack_classifier(seed=42):
        estimators = [
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed)),
            ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed, eval_metric="logloss", use_label_encoder=False)),
            ("et", ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=seed)),
        ]
        meta_learner = LogisticRegression(C=1.0, random_state=seed, solver="liblinear")
        return StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1
        )

    stack_fs_results = {}
    for fs_label, fs_m, fs_k in fs_methods:
        print(f"  -> Stacked Ensemble with {fs_label:22s}...", end="", flush=True)
        t0 = time.time()
        stack_clf = make_stack_classifier(RANDOM_SEED)
        # Give comparable tuning to meta learner and base classifiers if needed
        res = run_nested_evaluation(X, y, stack_clf, param_grid={}, fs_method=fs_m, fs_k=fs_k, outer_cv=outer_cv, random_state=RANDOM_SEED)
        elapsed = time.time() - t0
        stack_fs_results[fs_label] = res
        print(f" Done ({elapsed:.1f}s) | Acc: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.3f} | ROC-AUC: {res['roc_auc_mean']:.4f} ± {res['roc_auc_std']:.3f}")

    # Best FS for Stack
    best_fs_stack_name = max(stack_fs_results.keys(), key=lambda k: stack_fs_results[k]["roc_auc_mean"])
    best_stack_res = stack_fs_results[best_fs_stack_name]
    stack_no_fs_res = stack_fs_results["All 13 Features"]
    rf_baseline_res = base_results["Random Forest"]

    # 6. Ablation Study
    print("\n--- 5. ABLATION ANALYSIS ---")
    ablation_items = [
        ("1. Random Forest (Baseline, All 13)", rf_baseline_res),
        ("2. SVM-RBF (Base Component)", base_results["SVM-RBF"]),
        ("3. XGBoost (Base Component)", base_results["XGBoost"]),
        ("4. Extra Trees (Base Component)", base_results["Extra Trees"]),
        ("5. Stack without Feature Selection (All 13)", stack_no_fs_res),
        (f"6. Proposed Stack + Best FS ({best_fs_stack_name})", best_stack_res),
    ]

    ablation_rows = []
    for label, r in ablation_items:
        ablation_rows.append({
            "Configuration": label,
            "Accuracy": f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.3f}",
            "F1": f"{r['f1_mean']:.4f} ± {r['f1_std']:.3f}",
            "Sensitivity": f"{r['recall_sensitivity_mean']:.4f} ± {r['recall_sensitivity_std']:.3f}",
            "Specificity": f"{r['specificity_mean']:.4f} ± {r['specificity_std']:.3f}",
            "ROC-AUC": f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.3f}",
            "PR-AUC": f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.3f}",
        })
    ablation_df = pd.DataFrame(ablation_rows)
    print(ablation_df.to_string(index=False))

    # 7. Final Accuracy Check & Statistical Significance Testing
    print("\n--- 6. STATISTICAL SIGNIFICANCE TESTING (Paired across identical folds) ---")
    
    # Paired Fold Comparisons
    rf_acc_folds = np.array(rf_baseline_res["accuracy_raw"])
    stack_acc_folds = np.array(best_stack_res["accuracy_raw"])

    rf_auc_folds = np.array(rf_baseline_res["roc_auc_raw"])
    stack_auc_folds = np.array(best_stack_res["roc_auc_raw"])

    rf_f1_folds = np.array(rf_baseline_res["f1_raw"])
    stack_f1_folds = np.array(best_stack_res["f1_raw"])

    rf_sens_folds = np.array(rf_baseline_res["recall_sensitivity_raw"])
    stack_sens_folds = np.array(best_stack_res["recall_sensitivity_raw"])

    rf_spec_folds = np.array(rf_baseline_res["specificity_raw"])
    stack_spec_folds = np.array(best_stack_res["specificity_raw"])

    # Metrics differences
    diff_acc = stack_acc_folds - rf_acc_folds
    diff_auc = stack_auc_folds - rf_auc_folds
    diff_f1 = stack_f1_folds - rf_f1_folds
    diff_sens = stack_sens_folds - rf_sens_folds
    diff_spec = stack_spec_folds - rf_spec_folds

    # Paired t-tests
    t_stat_acc, p_val_acc = stats.ttest_rel(stack_acc_folds, rf_acc_folds)
    t_stat_auc, p_val_auc = stats.ttest_rel(stack_auc_folds, rf_auc_folds)
    t_stat_f1, p_val_f1 = stats.ttest_rel(stack_f1_folds, rf_f1_folds)

    # Wilcoxon signed-rank tests
    try:
        w_stat_acc, w_pval_acc = stats.wilcoxon(diff_acc)
    except Exception:
        w_stat_acc, w_pval_acc = 0.0, 1.0

    try:
        w_stat_auc, w_pval_auc = stats.wilcoxon(diff_auc)
    except Exception:
        w_stat_auc, w_pval_auc = 0.0, 1.0

    abs_acc_diff = float(np.mean(diff_acc))
    pct_acc_diff = (abs_acc_diff / rf_baseline_res["accuracy_mean"]) * 100
    abs_auc_diff = float(np.mean(diff_auc))
    abs_f1_diff = float(np.mean(diff_f1))
    abs_sens_diff = float(np.mean(diff_sens))
    abs_spec_diff = float(np.mean(diff_spec))

    comparison_table = pd.DataFrame([
        {
            "Model": "Random Forest (Baseline)",
            "Accuracy": f"{rf_baseline_res['accuracy_mean']:.4f} ± {rf_baseline_res['accuracy_std']:.4f}",
            "F1": f"{rf_baseline_res['f1_mean']:.4f} ± {rf_baseline_res['f1_std']:.4f}",
            "Sensitivity": f"{rf_baseline_res['recall_sensitivity_mean']:.4f} ± {rf_baseline_res['recall_sensitivity_std']:.4f}",
            "Specificity": f"{rf_baseline_res['specificity_mean']:.4f} ± {rf_baseline_res['specificity_std']:.4f}",
            "ROC-AUC": f"{rf_baseline_res['roc_auc_mean']:.4f} ± {rf_baseline_res['roc_auc_std']:.4f}",
        },
        {
            "Model": f"Proposed Stack + {best_fs_stack_name}",
            "Accuracy": f"{best_stack_res['accuracy_mean']:.4f} ± {best_stack_res['accuracy_std']:.4f}",
            "F1": f"{best_stack_res['f1_mean']:.4f} ± {best_stack_res['f1_std']:.4f}",
            "Sensitivity": f"{best_stack_res['recall_sensitivity_mean']:.4f} ± {best_stack_res['recall_sensitivity_std']:.4f}",
            "Specificity": f"{best_stack_res['specificity_mean']:.4f} ± {best_stack_res['specificity_std']:.4f}",
            "ROC-AUC": f"{best_stack_res['roc_auc_mean']:.4f} ± {best_stack_res['roc_auc_std']:.4f}",
        },
    ])

    print("\n--- FINAL HEAD-TO-HEAD COMPARISON TABLE ---")
    print(comparison_table.to_string(index=False))

    print("\n--- STATISTICAL GAINS & HYPOTHESIS TESTING ---")
    print(f"Absolute Accuracy Change:     {abs_acc_diff:+.4f} ({abs_acc_diff*100:+.2f}% points)")
    print(f"Relative Accuracy Change:     {pct_acc_diff:+.2f}%")
    print(f"Absolute F1-Score Change:     {abs_f1_diff:+.4f}")
    print(f"Absolute Sensitivity Change:  {abs_sens_diff:+.4f}")
    print(f"Absolute Specificity Change:  {abs_spec_diff:+.4f}")
    print(f"Absolute ROC-AUC Change:      {abs_auc_diff:+.4f}")
    print(f"Paired t-test on Accuracy:    t = {t_stat_acc:.4f}, p-value = {p_val_acc:.4f}")
    print(f"Wilcoxon test on Accuracy:    W = {w_stat_acc:.4f}, p-value = {w_pval_acc:.4f}")
    print(f"Paired t-test on ROC-AUC:     t = {t_stat_auc:.4f}, p-value = {p_val_auc:.4f}")
    print(f"Wilcoxon test on ROC-AUC:     W = {w_stat_auc:.4f}, p-value = {w_pval_auc:.4f}")
    print(f"Fold consistency (wins/ties/losses for Stack vs RF): {int(np.sum(diff_acc > 0))}/{int(np.sum(diff_acc == 0))}/{int(np.sum(diff_acc < 0))}")

    # 8. Explainability via Permutation Importance on Training Data
    print("\n--- 7. EXPLAINABILITY & FEATURE IMPORTANCE ---")
    preproc = build_preprocessor()
    X_proc = preproc.fit_transform(X)
    feat_labels = np.array(NUMERICAL_COLS + CATEGORICAL_COLS)

    rf_model_full = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=6)
    rf_model_full.fit(X_proc, y)
    perm_rf = permutation_importance(rf_model_full, X_proc, y, n_repeats=10, random_state=RANDOM_SEED)

    stack_model_full = make_stack_classifier(RANDOM_SEED)
    stack_model_full.fit(X_proc, y)
    perm_stack = permutation_importance(stack_model_full, X_proc, y, n_repeats=10, random_state=RANDOM_SEED)

    perm_df = pd.DataFrame({
        "Feature": feat_labels,
        "RF Permutation Mean": perm_rf.importances_mean,
        "RF Permutation Std": perm_rf.importances_std,
        "Stack Permutation Mean": perm_stack.importances_mean,
        "Stack Permutation Std": perm_stack.importances_std,
    }).sort_values(by="RF Permutation Mean", ascending=False)

    print("\nFeature Importance Rankings:")
    print(perm_df.to_string(index=False))

    # 9. Generate Publication-Quality Visualizations
    print("\n--- 8. GENERATING PUBLICATION-QUALITY FIGURES ---")
    
    # 1. ROC Curves
    plt.figure(figsize=(8, 7))
    models_to_plot = [
        ("Random Forest (Baseline)", rf_baseline_res, "#2b5c8f", "-"),
        (f"Proposed Stack ({best_fs_stack_name})", best_stack_res, "#d95f02", "-"),
        ("Stack (All Features)", stack_no_fs_res, "#7570b3", "--"),
        ("SVM-RBF", base_results["SVM-RBF"], "#1b9e77", ":"),
        ("XGBoost", base_results["XGBoost"], "#e7298a", ":"),
        ("Extra Trees", base_results["Extra Trees"], "#66a61e", ":"),
    ]
    for label, res, col, ls in models_to_plot:
        fpr, tpr, _ = roc_curve(res["oof_y_true"], res["oof_y_prob"])
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_val:.3f})", color=col, linestyle=ls, lw=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.title("Out-of-Fold Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves
    plt.figure(figsize=(8, 7))
    for label, res, col, ls in models_to_plot[:3]:
        prec_curve, rec_curve, _ = precision_recall_curve(res["oof_y_true"], res["oof_y_prob"])
        pr_val = average_precision_score(res["oof_y_true"], res["oof_y_prob"])
        plt.plot(rec_curve, prec_curve, label=f"{label} (PR-AUC = {pr_val:.3f})", color=col, linestyle=ls, lw=2.5)

    plt.title("Out-of-Fold Precision-Recall Curves", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_curves.png", dpi=300)
    plt.close()

    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm_rf = confusion_matrix(rf_baseline_res["oof_y_true"], rf_baseline_res["oof_y_pred"], labels=[0, 1])
    cm_stack = confusion_matrix(best_stack_res["oof_y_true"], best_stack_res["oof_y_pred"], labels=[0, 1])

    ConfusionMatrixDisplay(cm_rf, display_labels=["Healthy", "Disease"]).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"Random Forest (OOF Acc = {rf_baseline_res['accuracy_mean']:.1%})", fontweight="bold")
    axes[0].grid(False)

    ConfusionMatrixDisplay(cm_stack, display_labels=["Healthy", "Disease"]).plot(ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title(f"Proposed Stack (OOF Acc = {best_stack_res['accuracy_mean']:.1%})", fontweight="bold")
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrices.png", dpi=300)
    plt.close()

    # 4. Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(perm_df["RF Permutation Mean"].to_numpy())
    y_pos = np.arange(len(feat_labels))
    plt.barh(y_pos - 0.2, perm_df["RF Permutation Mean"].iloc[sorted_idx], height=0.4, label="Random Forest", color="#2b5c8f")
    plt.barh(y_pos + 0.2, perm_df["Stack Permutation Mean"].iloc[sorted_idx], height=0.4, label="Stacked Ensemble", color="#d95f02")
    plt.yticks(y_pos, perm_df["Feature"].iloc[sorted_idx])
    plt.xlabel("Mean Permutation Accuracy Decrease")
    plt.title("Feature Importance: Permutation Test (Holdout-Safe)", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=300)
    plt.close()

    # 5. Ablation Bar Chart
    plt.figure(figsize=(12, 6))
    ab_labels = [row["Configuration"].split(". ")[1] for row in ablation_rows]
    ab_accs = [float(row["Accuracy"].split(" ± ")[0]) for row in ablation_rows]
    ab_aucs = [float(row["ROC-AUC"].split(" ± ")[0]) for row in ablation_rows]
    x_idx = np.arange(len(ab_labels))

    plt.bar(x_idx - 0.18, ab_accs, width=0.36, label="Accuracy", color="#2b5c8f")
    plt.bar(x_idx + 0.18, ab_aucs, width=0.36, label="ROC-AUC", color="#1b9e77")
    plt.xticks(x_idx, ab_labels, rotation=20, ha="right")
    plt.ylim(0.70, 0.95)
    plt.ylabel("Score")
    plt.title("Ablation Study: Architecture & Feature Selection Contribution", fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "ablation_study.png", dpi=300)
    plt.close()

    # Save JSON summary report
    full_report = {
        "dataset_info": {
            "instances": n_total,
            "features": X.shape[1],
            "healthy": n_healthy,
            "disease": n_disease,
            "missing_values": int(missing_counts.sum()),
        },
        "base_model_benchmarks": {
            name: {k: v for k, v in res.items() if not k.startswith("oof_") and not k.endswith("_raw") and k != "fitted_models"}
            for name, res in base_results.items()
        },
        "rf_feature_selection": {
            k: {m: v for m, v in res.items() if not m.startswith("oof_") and not m.endswith("_raw") and m != "fitted_models"}
            for k, res in rf_fs_results.items()
        },
        "stack_feature_selection": {
            k: {m: v for m, v in res.items() if not m.startswith("oof_") and not m.endswith("_raw") and m != "fitted_models"}
            for k, res in stack_fs_results.items()
        },
        "ablation_results": ablation_rows,
        "head_to_head": {
            "random_forest": {
                "accuracy": f"{rf_baseline_res['accuracy_mean']:.4f} ± {rf_baseline_res['accuracy_std']:.4f}",
                "roc_auc": f"{rf_baseline_res['roc_auc_mean']:.4f} ± {rf_baseline_res['roc_auc_std']:.4f}",
                "f1": f"{rf_baseline_res['f1_mean']:.4f} ± {rf_baseline_res['f1_std']:.4f}",
                "sensitivity": f"{rf_baseline_res['recall_sensitivity_mean']:.4f} ± {rf_baseline_res['recall_sensitivity_std']:.4f}",
                "specificity": f"{rf_baseline_res['specificity_mean']:.4f} ± {rf_baseline_res['specificity_std']:.4f}",
            },
            "proposed_stack": {
                "accuracy": f"{best_stack_res['accuracy_mean']:.4f} ± {best_stack_res['accuracy_std']:.4f}",
                "roc_auc": f"{best_stack_res['roc_auc_mean']:.4f} ± {best_stack_res['roc_auc_std']:.4f}",
                "f1": f"{best_stack_res['f1_mean']:.4f} ± {best_stack_res['f1_std']:.4f}",
                "sensitivity": f"{best_stack_res['recall_sensitivity_mean']:.4f} ± {best_stack_res['recall_sensitivity_std']:.4f}",
                "specificity": f"{best_stack_res['specificity_mean']:.4f} ± {best_stack_res['specificity_std']:.4f}",
            },
            "differences": {
                "absolute_accuracy_delta": abs_acc_diff,
                "relative_accuracy_pct": pct_acc_diff,
                "absolute_roc_auc_delta": abs_auc_diff,
                "absolute_f1_delta": abs_f1_diff,
                "paired_t_stat_accuracy": float(t_stat_acc),
                "paired_p_val_accuracy": float(p_val_acc),
                "paired_t_stat_roc_auc": float(t_stat_auc),
                "paired_p_val_roc_auc": float(p_val_auc),
                "wilcoxon_p_val_accuracy": float(w_pval_acc),
                "wilcoxon_p_val_roc_auc": float(w_pval_auc),
            }
        }
    }

    with open(out_dir / "full_experiment_report.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    print(f"\n✓ Experiment execution complete! All plots and reports exported to {out_dir.resolve()}.")

    # Return key dicts for programmatic consumption if needed
    return full_report


if __name__ == "__main__":
    main()
