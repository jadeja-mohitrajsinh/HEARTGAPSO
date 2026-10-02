# HeartGAPSO — API & Module Reference

Complete reference for every public function and class in the `src/` package.

---

## Table of Contents

1. [src/fitness.py](#srcfitnesspy)
2. [src/genetic_algorithm.py](#srcgenetic_algorithmpy)
3. [src/gapso_optimizer.py](#srcgapso_optimizerpy)
4. [src/particle_swarm.py](#srcparticle_swarmpy)
5. [src/statistical_analysis.py](#srcstatistical_analysispy)
6. [src/preprocessing.py](#srcpreprocessingpy)
7. [src/preprocessing_pipeline.py](#srcpreprocessing_pipelinepy)
8. [src/random_forest.py](#srcrandom_forestpy)
9. [src/cross_validation.py](#srccross_validationpy)
10. [src/feature_selection.py](#srcfeature_selectionpy)
11. [src/model_definitions.py](#srcmodel_definitionspy)
12. [src/metrics_calculator.py](#srcmetrics_calculatorpy)
13. [src/evaluation.py](#srcevaluationpy)
14. [src/data_loader.py](#srcdata_loaderpy)

---

## src/fitness.py

### `FitnessConfig`

```python
@dataclass
class FitnessConfig:
    folds: int
    repeats: int
    roc_auc_weight: float
    f1_weight: float
    sensitivity_weight: float
    specificity_weight: float
```

**Configuration dataclass** for the fitness evaluation protocol.

| Field | Type | Description |
|-------|------|-------------|
| `folds` | int | Number of stratified CV folds |
| `repeats` | int | Number of times to repeat the CV |
| `roc_auc_weight` | float | Multiplier for ROC-AUC in fitness sum |
| `f1_weight` | float | Multiplier for F1-Score |
| `sensitivity_weight` | float | Multiplier for Sensitivity (Recall) |
| `specificity_weight` | float | Multiplier for Specificity |

---

## src/genetic_algorithm.py

### `GAConfig`

```python
@dataclass
class GAConfig:
    population_size: int
    generations: int
    pso_iterations: int
    max_evaluations: int
    pso_particles: int
```

**Configuration dataclass** for the GA optimizer.

| Field | Type | Description |
|-------|------|-------------|
| `population_size` | int | Number of chromosomes per generation |
| `generations` | int | Maximum number of GA generations |
| `pso_iterations` | int | PSO repair iterations per generation |
| `max_evaluations` | int | Hard budget on total fitness evaluations |
| `pso_particles` | int | Number of PSO particles per generation |

---

## src/gapso_optimizer.py

### `_evaluate_individual(features, params, X, y, fitness_config, seed)`

**Internal.** Evaluates a single (feature_mask, hyperparams) candidate using stratified k-fold CV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `features` | list[bool] | 13-element binary feature mask |
| `params` | dict | RF hyperparameters (n_estimators, max_depth, etc.) |
| `X` | DataFrame | Full training feature matrix |
| `y` | Series | Training labels |
| `fitness_config` | FitnessConfig | CV folds and metric weights |
| `seed` | int | Random seed for reproducibility |

**Returns:** `float` — mean fitness score across folds.

**Notes:**
- Returns `0.0` for empty feature masks.
- Fits a fresh preprocessor inside each fold on training rows only.
- Computes: ROC-AUC, F1, Sensitivity, Specificity per fold.

---

### `optimize(X, y, seed, ga_config, fitness_config, statistical_analysis, **kwargs)`

**Main entry point.** Runs the GA-PSO hybrid optimization.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | DataFrame | Training feature matrix |
| `y` | Series | Training binary labels |
| `seed` | int | Master random seed |
| `ga_config` | GAConfig | GA configuration |
| `fitness_config` | FitnessConfig | Fitness evaluation configuration |
| `statistical_analysis` | dict | Output of `analyze_features()` |
| `**kwargs` | - | `high_fidelity_candidate_limit`, `stability_candidates`, `fast_promotions`, `stability_seeds` |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `features` | list[bool] | Best feature mask (13 bits) |
| `params` | dict | Best RF hyperparameters |
| `fitness` | float | Best fitness score |
| `threshold` | float | Best classification threshold |
| `history` | list[dict] | Per-generation stats |
| `candidate_audit` | list[dict] | All screened candidates with ranks |

---

## src/particle_swarm.py

### `repair_rejected(rejected, evaluate, dimensions, seed, iterations, inertia, cognitive, social)`

Runs PSO on a set of rejected GA individuals and returns their personal bests.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rejected` | list[tuple] | — | List of (position, score) from rejected GA individuals |
| `evaluate` | callable | — | `(features, params) -> (score, metrics)` evaluation function |
| `dimensions` | int | — | Number of feature dimensions (13) |
| `seed` | int | — | Random seed |
| `iterations` | int | 20 | Maximum PSO iterations |
| `inertia` | float | 0.65 | Velocity inertia weight w |
| `cognitive` | float | 1.4 | Personal best attraction coefficient c₁ |
| `social` | float | 1.4 | Global best attraction coefficient c₂ |

**Returns:** `(results, history)` where:
- `results`: list of `(personal_best_position, score, metrics)` — one entry per particle
- `history`: list of dicts with per-particle update details

**Behaviour:**
- Initializes velocities with Gaussian noise (σ=0.08).
- Clips positions to [0, 1] after each update.
- Binarizes the feature sub-vector via sigmoid transform.
- Applies early stopping if global best stagnates for 3 iterations.
- Skips duplicate positions (evaluated_positions set).

---

## src/statistical_analysis.py

### `analyze_features(X, y)`

Computes per-feature class-separability statistics using the training partition only.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | DataFrame | Feature matrix (training set only) |
| `y` | Series | Binary target labels |

**Returns:** `dict` with keys:
- `"features"`: list of per-feature dicts (sorted by T² score descending)
- `"important_features"`: list of feature names with p < 0.05

**Per-feature dict fields:**

| Field | Type | Description |
|-------|------|-------------|
| `feature` | str | Feature name |
| `t2_statistical_score` | float | T² (continuous) or χ² (categorical) statistic |
| `p_value` | float | Statistical significance |
| `class_0_mean` | float | Mean value for healthy class |
| `class_1_mean` | float | Mean value for disease class |
| `mutation_probability` | float | GA mutation rate for this feature |
| `importance_rank` | int | 1 = most significant |

**Method selection:**
- Categorical (`sex, cp, fbs, restecg, exang, slope, ca, thal`): Chi-square test
- Continuous (`age, trestbps, chol, thalach, oldpeak`): Welch's t-test → T² = t²

---

### `save_analysis(analysis_results, file_path)`

Saves `analyze_features()` output to a JSON file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `analysis_results` | dict | Output of `analyze_features()` |
| `file_path` | str | Output path (parent dirs created automatically) |

---

## src/preprocessing.py

### `make_preprocessor(feature_names=None, **kwargs)`

Creates a sklearn `Pipeline` with median imputation followed by MinMax scaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `feature_names` | list[str] or None | Accepted but unused; for API compatibility |
| `**kwargs` | — | Ignored; for forward-compatibility |

**Returns:** `sklearn.pipeline.Pipeline` with steps:
1. `('imputer', SimpleImputer(strategy='median'))`
2. `('scaler', MinMaxScaler())`

**Usage:**
```python
preprocessor = make_preprocessor(feature_names=selected_feature_names)
X_train_proc = preprocessor.fit_transform(X_train_selected)
X_val_proc   = preprocessor.transform(X_val_selected)
```

---

## src/preprocessing_pipeline.py

### `DataFramePreprocessor`

A wrapper around the sklearn Pipeline that preserves DataFrame column names.

```python
class DataFramePreprocessor:
    def fit(self, X, y=None) -> self
    def transform(self, X) -> np.ndarray
    def fit_transform(self, X, y=None) -> np.ndarray
    def get_feature_names_out(self, input_features=None) -> list[str]
```

- `fit()`: Stores feature names from the input DataFrame, then fits the pipeline.
- `transform()`: Applies median imputation + MinMax scaling.
- `get_feature_names_out()`: Returns the feature names stored during `fit()`.

### `create_preprocessing_pipeline(X=None)`

Factory function. Returns a new `DataFramePreprocessor` instance.

---

## src/random_forest.py

### `make_random_forest(params, seed)`

Creates a `RandomForestClassifier` with the given hyperparameters and seed.

| Parameter | Type | Description |
|-----------|------|-------------|
| `params` | dict | RF hyperparameters (n_estimators, max_depth, etc.) |
| `seed` | int | `random_state` for reproducibility |

**Returns:** `RandomForestClassifier`

---

### `InferencePipeline`

A self-contained inference wrapper combining column selection, preprocessing, and classification.

```python
class InferencePipeline:
    def __init__(self, preprocessing, classifier, indices, threshold)
    def predict_proba(self, X_raw) -> np.ndarray   # shape (N, 2)
    def predict(self, X_raw) -> np.ndarray          # shape (N,), binary {0,1}
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `preprocessing` | Pipeline | Fitted preprocessor |
| `classifier` | RandomForestClassifier | Fitted RF model |
| `indices` | list[int] | Column indices of selected features |
| `threshold` | float | Classification threshold (default 0.5) |

**`predict_proba(X_raw)`:**
1. Selects `X_raw.iloc[:, self.indices]`
2. Applies `self.preprocessing.transform()`
3. Returns `self.classifier.predict_proba()` (N × 2 array)

**`predict(X_raw)`:**
1. Calls `predict_proba()` and extracts column 1 (disease probability)
2. Returns `(proba >= self.threshold).astype(int)`

**Input requirement:** `X_raw` must be a DataFrame with all 13 feature columns in the standard order.

---

## src/cross_validation.py

### `run_nested_cv(X, y, model_name, model_estimator, feature_selection_method, fs_params, param_grid, outer_cv_folds, inner_cv_folds, random_state)`

Runs full nested stratified k-fold cross-validation with optional feature selection and hyperparameter tuning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | DataFrame | — | Full feature matrix |
| `y` | Series | — | Target labels |
| `model_name` | str | — | Display name for this configuration |
| `model_estimator` | estimator | — | sklearn-compatible classifier |
| `feature_selection_method` | str | `"all"` | One of the supported method strings |
| `fs_params` | dict | `{}` | Extra params for the feature selector |
| `param_grid` | dict | `{}` | Hyperparameter grid for GridSearchCV |
| `outer_cv_folds` | int | `5` | Number of outer CV folds |
| `inner_cv_folds` | int | `3` | Number of inner CV folds (for GridSearchCV / RFECV) |
| `random_state` | int | `42` | Random seed |

**Returns:** `dict` with summary statistics:

| Key | Type | Description |
|-----|------|-------------|
| `model` | str | Model name |
| `feature_selection` | str | Feature selection method |
| `accuracy_mean` | float | Mean accuracy across outer folds |
| `accuracy_std` | float | Std dev of accuracy |
| `roc_auc_mean` | float | Mean ROC-AUC |
| `roc_auc_std` | float | Std dev of ROC-AUC |
| `f1_mean` | float | Mean F1 |
| `recall_mean` | float | Mean Sensitivity |
| `specificity_mean` | float | Mean Specificity |
| `precision_mean` | float | Mean Precision |
| `selected_features_folds` | list[list[str]] | Features selected per fold |
| `raw_metrics` | dict | All per-fold metric lists |

---

## src/feature_selection.py

### `get_feature_selector(method, k, estimator, step, cv, random_state, n_features_to_select, **kwargs)`

Factory that returns a sklearn feature selector for the given method.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `"all"` | Selection method identifier |
| `k` | int | `8` | Number of features to select (for fixed-k methods) |
| `estimator` | estimator | LR | Base estimator for RFE/RFECV |
| `step` | int | `1` | Features removed per RFE step |
| `cv` | int | `3` | CV folds for RFECV |
| `random_state` | int | `42` | Seed for reproducibility |

**Returns:** sklearn selector or `None` (if method is `"all"`)

---

## src/model_definitions.py

### `get_models(random_state=42)`

Returns a dict of all benchmark classifiers.

| Key | Classifier |
|-----|-----------|
| `'LogisticRegression'` | `LogisticRegression(solver='liblinear')` |
| `'KNN'` | `KNeighborsClassifier(n_neighbors=5)` |
| `'NaiveBayes'` | `GaussianNB()` |
| `'DecisionTree'` | `DecisionTreeClassifier(max_depth=5)` |
| `'SVM'` | `CalibratedClassifierCV(SVC())` |
| `'RandomForest'` | `RandomForestClassifier(n_estimators=100)` |
| `'ExtraTrees'` | `ExtraTreesClassifier(n_estimators=100)` |
| `'XGBoost'` | `XGBClassifier()` or `GradientBoostingClassifier()` (fallback) |

---

### `get_stacked_ensemble(base_models, meta_learner, random_state)`

Creates a `StackingClassifier` with SVM + XGBoost + ExtraTrees base models and Logistic Regression meta-learner.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_models` | dict or None | `get_models()` | Override base estimators |
| `meta_learner` | estimator or None | `LogisticRegression` | Final level meta-classifier |

**Returns:** `StackingClassifier` with `stack_method='predict_proba'`, `cv=3`, `n_jobs=-1`

---

## src/metrics_calculator.py

### `calculate_metrics(y_true, y_pred, y_prob=None)`

Computes all evaluation metrics from predictions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `y_true` | array | Ground truth binary labels |
| `y_pred` | array | Predicted binary labels |
| `y_prob` | array or None | Predicted probabilities for positive class |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `accuracy` | float | (TP+TN)/(TP+TN+FP+FN) |
| `precision` | float | TP/(TP+FP) |
| `recall` | float | TP/(TP+FN) — same as Sensitivity |
| `specificity` | float | TN/(TN+FP) |
| `f1` | float | 2×Precision×Recall/(Precision+Recall) |
| `roc_auc` | float | Area under ROC curve (0.5 if y_prob is None) |
| `pr_auc` | float | Area under Precision-Recall curve (0.5 if y_prob is None) |

---

### `format_metrics(metrics)`

Formats metric dict values as strings for display.

- `float` values → `"0.8500"`
- `list[float]` values → `"0.8500 ± 0.0230"` (mean ± std)
- Other → `str(value)`

---

## src/evaluation.py

### `evaluate_model(y_true, y_pred, y_prob, threshold)`

Stub evaluator. Returns a fixed summary dict (to be expanded in production).

**Returns:** `{"roc_auc": 0.85, "f1": 0.75, "threshold": threshold}`

---

### `save_optimization_diagnostics(history, opt_result, output_dir)`

Saves optimization diagnostics to disk.

| Parameter | Type | Description |
|-----------|------|-------------|
| `history` | list[dict] | Per-generation history from optimizer |
| `opt_result` | dict | Full optimizer result dict |
| `output_dir` | str | Output directory path |

**Outputs:**
- `{output_dir}/fitness_convergence.png` — best fitness per generation plot
- `{output_dir}/optimization_summary.json` — `{"best_fitness": ..., "selected_features_count": ...}`

---

## src/data_loader.py

### `FEATURE_NAMES`

```python
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
```

The canonical ordered list of 13 feature column names. Used throughout the codebase to map bit indices to feature names.
