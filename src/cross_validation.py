import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from src.preprocessing_pipeline import create_preprocessing_pipeline
from src.feature_selection import get_feature_selector
from src.metrics_calculator import calculate_metrics


def run_nested_cv(X, y, model_name, model_estimator, feature_selection_method="all", fs_params=None, param_grid=None, outer_cv_folds=5, inner_cv_folds=3, random_state=42):
    if fs_params is None:
        fs_params = {}
    if param_grid is None:
        param_grid = {}

    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=random_state)
    metrics_accumulator = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'specificity': [],
        'f1': [],
        'roc_auc': [],
        'pr_auc': []
    }

    best_estimators = []
    selected_features_sets = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Step 1: Preprocessing
        preprocessor = create_preprocessing_pipeline(X_train)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        feature_names = preprocessor.get_feature_names_out()
        X_train_proc_df = pd.DataFrame(X_train_proc, columns=feature_names)
        X_test_proc_df = pd.DataFrame(X_test_proc, columns=feature_names)

        # Step 2: Feature Selection
        selected_features = list(feature_names)
        if feature_selection_method != 'all':
            clean_fs_params = fs_params.copy()
            fs_estimator = clean_fs_params.pop('estimator', None)
            if fs_estimator is None:
                if hasattr(model_estimator, 'coef_') or hasattr(model_estimator, 'feature_importances_'):
                    fs_estimator = clone(model_estimator)
                else:
                    fs_estimator = LogisticRegression(random_state=random_state, solver='liblinear')

            selector = get_feature_selector(
                feature_selection_method,
                estimator=fs_estimator,
                cv=inner_cv_folds,
                random_state=random_state,
                **clean_fs_params
            )

            if selector:
                selector.fit(X_train_proc_df, y_train)
                support = selector.get_support()
                selected_features = [f for f, s in zip(feature_names, support) if s]

                if not selected_features:
                    selected_features = list(feature_names)

        selected_features_sets.append(selected_features)
        X_train_fs = X_train_proc_df[selected_features]
        X_test_fs = X_test_proc_df[selected_features]

        # Step 3: Hyperparameter Tuning / Model Fitting
        estimator_clone = clone(model_estimator)
        if param_grid:
            inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(
                estimator=estimator_clone,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train_fs, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = estimator_clone
            best_model.fit(X_train_fs, y_train)

        best_estimators.append(best_model)

        # Step 4: Out-of-fold Prediction & Metrics
        y_pred = best_model.predict(X_test_fs)
        y_prob = best_model.predict_proba(X_test_fs)[:, 1] if hasattr(best_model, 'predict_proba') else None

        fold_metrics = calculate_metrics(y_test, y_pred, y_prob)
        for key in metrics_accumulator:
            metrics_accumulator[key].append(fold_metrics[key])

    summary = {
        'model': model_name,
        'feature_selection': feature_selection_method,
        'accuracy_mean': float(np.mean(metrics_accumulator['accuracy'])),
        'accuracy_std': float(np.std(metrics_accumulator['accuracy'])),
        'roc_auc_mean': float(np.mean(metrics_accumulator['roc_auc'])),
        'roc_auc_std': float(np.std(metrics_accumulator['roc_auc'])),
        'f1_mean': float(np.mean(metrics_accumulator['f1'])),
        'f1_std': float(np.std(metrics_accumulator['f1'])),
        'recall_mean': float(np.mean(metrics_accumulator['recall'])),
        'recall_std': float(np.std(metrics_accumulator['recall'])),
        'specificity_mean': float(np.mean(metrics_accumulator['specificity'])),
        'specificity_std': float(np.std(metrics_accumulator['specificity'])),
        'precision_mean': float(np.mean(metrics_accumulator['precision'])),
        'precision_std': float(np.std(metrics_accumulator['precision'])),
        'selected_features_folds': selected_features_sets,
        'raw_metrics': metrics_accumulator
    }
    return summary
