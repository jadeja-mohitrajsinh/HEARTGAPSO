import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.stats import ttest_ind, chi2_contingency

def analyze_features(X, y):
    """
    Performs statistical significance analysis on features using Welch's t-test
    for continuous features and Chi-Square for categorical features.
    """
    analysis_results = {"features": [], "important_features": []}
    categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    for feature_name in X.columns:
        class_0 = X[y == 0][feature_name].dropna()
        class_1 = X[y == 1][feature_name].dropna()

        class_0_mean = float(class_0.mean()) if not class_0.empty else 0.0
        class_1_mean = float(class_1.mean()) if not class_1.empty else 0.0

        if feature_name in categorical_cols:
            contingency_table = pd.crosstab(X[feature_name], y)
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
                stat_score = float(chi2)
            else:
                p_val = 1.0
                stat_score = 0.0
        else:
            if len(class_0) > 1 and len(class_1) > 1:
                t_stat, p_val = ttest_ind(class_0, class_1, equal_var=False)
                stat_score = float(t_stat ** 2) if not np.isnan(t_stat) else 0.0
            else:
                p_val = 1.0
                stat_score = 0.0

        if np.isnan(p_val):
            p_val = 1.0

        mutation_probability = float(np.clip(0.05 + 0.15 * (1.0 - p_val), 0.05, 0.25))

        feature_info = {
            "feature": feature_name,
            "t2_statistical_score": stat_score,
            "p_value": float(p_val),
            "class_0_mean": class_0_mean,
            "class_1_mean": class_1_mean,
            "mutation_probability": mutation_probability
        }
        analysis_results["features"].append(feature_info)

        if p_val < 0.05:
            analysis_results["important_features"].append(feature_name)

    analysis_results["features"].sort(key=lambda x: x["t2_statistical_score"], reverse=True)
    for i, feature_info in enumerate(analysis_results["features"]):
        feature_info["importance_rank"] = i + 1

    return analysis_results

def save_analysis(analysis_results, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(analysis_results, f, indent=4)
