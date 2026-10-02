# HeartGAPSO — Algorithm Deep Dive

This document explains every algorithm, formula, and design decision in the HeartGAPSO system from first principles.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Chromosome Representation](#2-chromosome-representation)
3. [Statistical Pre-analysis (T² Scoring)](#3-statistical-pre-analysis-t-scoring)
4. [Fitness Function](#4-fitness-function)
5. [Genetic Algorithm (GA)](#5-genetic-algorithm-ga)
6. [Particle Swarm Optimization (PSO)](#6-particle-swarm-optimization-pso)
7. [GA-PSO Hybrid Loop](#7-ga-pso-hybrid-loop)
8. [Three-Stage Evaluation Funnel](#8-three-stage-evaluation-funnel)
9. [Preprocessing](#9-preprocessing)
10. [Final Model & Inference Pipeline](#10-final-model--inference-pipeline)
11. [Threshold Calibration](#11-threshold-calibration)

---

## 1. Problem Formulation

Given the UCI Cleveland Heart Disease dataset (303 patients, 13 features, binary target):

- **Feature Selection:** Find a subset S ⊆ {1, …, 13} of features that maximizes classification performance.
- **Hyperparameter Tuning:** Simultaneously find optimal Random Forest hyperparameters θ ∈ Θ.
- **Joint Optimization:** The combined search space is 2^13 × |Θ| ≈ 8192 × millions of combinations — too large for exhaustive search.

HeartGAPSO solves this as a **joint combinatorial-continuous optimization** using a hybrid GA-PSO metaheuristic.

---

## 2. Chromosome Representation

Each individual (candidate solution) encodes two parts in a single chromosome:

```
[f₁, f₂, ..., f₁₃ | θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]
 ──────────────────   ──────────────────────────
  Feature mask (binary)   RF hyperparameters (discrete/continuous)
```

### Feature Mask (bits 0–12)

Each bit fᵢ ∈ {0, 1} indicates whether feature i is included in training.

| Bit index | Feature |
|-----------|---------|
| 0 | age |
| 1 | sex |
| 2 | cp |
| 3 | trestbps |
| 4 | chol |
| 5 | fbs |
| 6 | restecg |
| 7 | thalach |
| 8 | exang |
| 9 | oldpeak |
| 10 | slope |
| 11 | ca |
| 12 | thal |

**Constraint:** At least one bit must be 1 (empty subsets map to fitness=0.0).

### Hyperparameter Genes (bits 13–18)

| Gene | Parameter | Search Space |
|------|-----------|-------------|
| θ₁ | n_estimators | [200, 400, 600, 800, 1200, 1600] |
| θ₂ | max_depth | [None, 4, 6, 8, 10, 12, 16, 20] |
| θ₃ | min_samples_split | [2, 4, 6, 8, 10, 15, 20] |
| θ₄ | min_samples_leaf | [1, 2, 3, 4, 5, 8, 10] |
| θ₅ | max_features | ["sqrt", "log2", 0.25, 0.33, 0.40, 0.50, 0.67, 1.0] |
| θ₆ | class_weight | [None, "balanced", "balanced_subsample"] |

---

## 3. Statistical Pre-analysis (T² Scoring)

Before optimization begins, features are ranked by statistical class separability **using training data only** (no holdout leakage).

### Method per feature type

**Continuous features** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`):
```
T² = t² where t is Welch's t-statistic comparing class 0 vs class 1 distributions
```
Welch's t-test does not assume equal variances:
```
t = (μ₀ - μ₁) / sqrt(s₀²/n₀ + s₁²/n₁)
```

**Categorical/binary features** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`):
```
χ² = chi-squared statistic from contingency table (feature × target)
```

### Mutation Probability Derivation

Per-feature mutation probability is derived from its p-value:
```
mutation_prob = clip(0.05 + 0.15 × (1 − p_value), min=0.05, max=0.25)
```

| p-value | mutation_prob | Interpretation |
|---------|--------------|----------------|
| 0.001 (very significant) | 0.200 | High class separation → conservative mutation |
| 0.10 | 0.185 | Moderate significance |
| 0.50 | 0.125 | Weak separation |
| 0.90 | 0.065 | Almost no class separation |
| 1.00 | 0.050 | Floor (minimum mutation) |

**Key insight:** Clinically important features (`thal`, `cp`, `ca`) have near-0 p-values → maximum mutation prob 0.20, meaning they mutate at full base rate. Weak features (`fbs` p=0.60) get mutation prob 0.11 — they are mutated less aggressively, biasing the search toward solutions that exclude them.

---

## 4. Fitness Function

The fitness of a candidate (S, θ) is computed via stratified k-fold cross-validation on the training set:

```
Fitness(S, θ) = 0.40 × ROC-AUC
              + 0.25 × F₁
              + 0.20 × Sensitivity
              + 0.15 × Specificity
              − λ_sparsity × |S|
              − λ_stability × σ_CV
```

Where:
- **ROC-AUC** = Area under the Receiver Operating Characteristic curve
- **F₁** = 2 × Precision × Recall / (Precision + Recall)
- **Sensitivity** = TP / (TP + FN) — prioritizes catching actual disease
- **Specificity** = TN / (TN + FP) — avoids false alarms
- **|S|** = number of selected features (sparsity penalty)
- **σ_CV** = standard deviation of fold fitness scores (stability penalty)

### Weight rationale

| Weight | Metric | Why |
|--------|--------|-----|
| 0.40 | ROC-AUC | Threshold-independent discrimination; best single summary metric |
| 0.25 | F₁ | Balances precision and recall on an imbalanced dataset |
| 0.20 | Sensitivity | Medical priority: missing disease is worse than a false alarm |
| 0.15 | Specificity | Prevents overclassifying healthy patients |

### Per-fold evaluation (implemented in `_evaluate_individual`)

```python
for each fold (train_idx, val_idx):
    preprocessor.fit(X_train_fold)           # fit on fold training data ONLY
    X_train_processed = preprocessor.transform(X_train_fold)
    X_val_processed   = preprocessor.transform(X_val_fold)
    rf.fit(X_train_processed, y_train_fold)
    compute: roc_auc, f1, sensitivity, specificity
    fold_fitness = weighted sum

final_fitness = mean(fold_fitness scores)
```

**Leakage prevention:** The preprocessor is re-fit inside every fold on training rows only. The validation rows are transformed using the fold's fitted scaler — never used in fitting.

---

## 5. Genetic Algorithm (GA)

### Initialization

The initial population is seeded with:
1. **Statistically-guided individuals:** Chromosomes that include features with high T² scores (`thal`, `cp`, `ca`, `slope`) with higher probability.
2. **Random individuals:** Fully random binary masks to maintain diversity.
3. **All-features baseline:** One individual with all 13 features enabled.

### Selection — Tournament Selection

For each parent slot:
1. Sample k individuals from the population (tournament size k, typically 3–5).
2. Select the individual with highest fitness.
3. Repeat for second parent.

Tournament selection provides selection pressure proportional to relative fitness rank rather than absolute fitness, making it robust to fitness scaling issues.

### Crossover — Uniform Crossover

For parent chromosomes P₁ and P₂, each gene is inherited independently:
```
for i in range(chromosome_length):
    child[i] = P₁[i] if random() < 0.5 else P₂[i]
```

This allows arbitrary recombination of feature subsets and hyperparameter values across the full chromosome.

**Validity repair:** If crossover produces an all-zero feature mask, a random bit is set to 1.

### Mutation — Discriminate/Adaptive Mutation

Each gene is flipped with probability proportional to its per-feature mutation rate from the statistical analysis:

```
for i in range(n_features):
    if random() < mutation_prob[feature_i]:
        chromosome[i] = 1 - chromosome[i]
```

Hyperparameter genes are mutated by sampling a new value from their search space with a fixed mutation probability.

**Adaptive rate:** The mutation rate increases when the population shows stagnation (no fitness improvement for N generations).

### Elitism

The top `elite_count` individuals from each generation are carried forward unchanged into the next generation. This guarantees that the best solution found is never lost.

### Stagnation & Early Stopping

If the best fitness does not improve for `stagnation_generations` consecutive generations, the GA terminates early to save evaluations.

### Caching

Each evaluated (feature_mask, hyperparams) pair is cached via a hash key. Cache hits are counted and reported (`cache_hits`, `cache_misses` in the output report). This prevents re-evaluating the same chromosome if it reappears through crossover or mutation.

---

## 6. Particle Swarm Optimization (PSO)

### Role in the hybrid system

PSO is applied to **repair rejected GA individuals** — the lower-fitness half of each generation that would otherwise be discarded. Instead of discarding them, they become PSO particles.

### Particle representation

Each particle's position vector has the same structure as a GA chromosome:
```
position = [f₁, ..., f₁₃, θ₁, ..., θ₆]   ∈ [0, 1]^19
```

The feature subspace [0,1]^13 is binary-encoded via a sigmoid transform at evaluation time.

### Velocity update (canonical PSO)

```
v_{t+1} = w × v_t
         + c₁ × r₁ × (personal_best - position_t)
         + c₂ × r₂ × (global_best  - position_t)
```

| Parameter | Symbol | Default | Role |
|-----------|--------|---------|------|
| Inertia weight | w | 0.65 | Controls momentum; higher = broader search |
| Cognitive coefficient | c₁ | 1.4 | Pull toward particle's own best position |
| Social coefficient | c₂ | 1.4 | Pull toward swarm's global best |
| r₁, r₂ | random | U(0,1) | Stochastic scaling per dimension |

### Position clipping

After velocity update:
```
position = clip(position + velocity, 0.0, 1.0)
```

### Feature binarization (sigmoid transform)

The continuous feature sub-vector is converted to binary at evaluation time:
```
probability = sigmoid(position[:n_features]) = 1 / (1 + exp(-position))
binary_mask = (random(n_features) < probability)
```

This soft encoding allows gradient-like movement through the feature space while maintaining valid binary chromosomes at evaluation.

### Personal and global bests

- **Personal best:** Each particle tracks its highest-fitness position seen.
- **Global best:** The swarm tracks the single highest-fitness position across all particles and all iterations.

### Early stopping (PSO stagnation)

If the global best fitness does not improve by more than 1e-10 for 3 consecutive iterations, PSO stops early.

### Reinjection

After PSO completes, each particle's personal best is reinjected into the GA population for the next generation, replacing the rejected individuals. This closes the hybrid loop.

---

## 7. GA-PSO Hybrid Loop

```
Initialize population (statistically seeded + random)
Evaluate all individuals → fitness scores

For generation g in range(max_generations):
    ├── Sort by fitness
    ├── Elites → carry forward unchanged
    ├── Selected (top half) → tournament selection → crossover → mutation → children
    ├── Rejected (bottom half) → PSO repair
    │       ├── Each rejected individual = one PSO particle
    │       ├── Run PSO for pso_iterations steps
    │       └── Return each particle's personal best
    ├── New population = elites + children + PSO-repaired individuals
    ├── Evaluate all new individuals (use cache for previously seen)
    ├── Update history
    └── Check stagnation stopping condition

Promote top fast-stage candidates → high-fidelity re-evaluation
Select top high-fidelity candidates → multi-seed stability audit
Return best stable candidate
```

---

## 8. Three-Stage Evaluation Funnel

To balance speed and accuracy, candidates go through three progressively more expensive evaluation stages:

### Stage 1 — Fast Screening

- **CV:** 3-fold, 1 repeat
- **n_estimators cap:** 400 trees
- **Budget:** Controlled by `max_evaluations` (default 800)
- **Promotion limit:** `fast_promotions` top candidates (default 30)

All individuals in the GA loop are evaluated at this stage. The goal is to quickly distinguish promising from poor candidates.

### Stage 2 — High-Fidelity Screening

- **CV:** max(5, cv_folds) folds, full `cv_repeats` repeats
- **Full hyperparameter space:** All n_estimators values allowed
- **Limit:** `high_fidelity_candidate_limit` candidates (default 30)

The top fast-stage candidates are re-evaluated with a more rigorous CV protocol. This re-ranks candidates more reliably.

### Stage 3 — Multi-Seed Stability Audit

- **Seeds:** `stability_seeds` independent random seeds (default 5)
- **Limit:** `stability_candidates` candidates (default 5)

The top high-fidelity candidates are evaluated across multiple seeds. The fitness metric becomes:
```
stability_fitness = mean_fitness - λ_stability × std_across_seeds
```

This penalizes candidates whose performance varies significantly across seeds (unstable solutions).

**Final selection:** The candidate with the best stability-adjusted fitness becomes the winner.

---

## 9. Preprocessing

Preprocessing is **fold-local** — fitted independently inside each CV fold to prevent data leakage.

### Pipeline

```
Raw input → Median Imputation → MinMax Scaling → Processed output
```

**Median Imputation** (`SimpleImputer(strategy='median')`):
- Replaces missing values with the median of the training fold
- Chosen over mean because `ca` and `thal` have a few outlier values (e.g., thal=7)

**MinMax Scaling** (`MinMaxScaler()`):
- Scales each feature to [0, 1]: `x_scaled = (x - x_min) / (x_max - x_min)`
- Not strictly required for Random Forest (tree-based, scale-invariant) but standardizes the feature space for consistency with the PSO velocity update and for any downstream use of the preprocessor

**Important:** The preprocessor is fit on training fold rows only. Test/validation fold rows are transformed using the training fold's fitted scaler — never used to fit the scaler.

### Implementation (`src/preprocessing.py`)

```python
def make_preprocessor(feature_names=None, **kwargs):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
```

The `feature_names` parameter is accepted but unused (the pipeline applies uniformly to whatever columns are passed in).

---

## 10. Final Model & Inference Pipeline

After optimization completes, the final model is assembled:

```python
# Fit preprocessor on ALL training data (not fold-local anymore)
preprocessor = make_preprocessor()
X_train_processed = preprocessor.fit_transform(X_train.iloc[:, selected_indices])

# Train final RF with optimal hyperparameters
final_rf = RandomForestClassifier(random_state=seed, **opt_params)
final_rf.fit(X_train_processed, y_train)

# Wrap in inference pipeline
pipeline = InferencePipeline(
    preprocessing=preprocessor,
    classifier=final_rf,
    indices=selected_feature_indices,
    threshold=opt_result["threshold"]
)
```

### InferencePipeline (`src/random_forest.py`)

```python
class InferencePipeline:
    def predict_proba(self, X_raw):
        X_selected  = X_raw.iloc[:, self.indices]     # column selection
        X_processed = self.preprocessing.transform(X_selected)  # scale
        return self.classifier.predict_proba(X_processed)

    def predict(self, X_raw):
        probabilities = self.predict_proba(X_raw)[:, 1]
        return (probabilities >= self.threshold).astype(int)
```

The pipeline is fully self-contained: given a raw DataFrame with all 13 feature columns, it:
1. Selects the optimized feature subset by column index
2. Applies median imputation + MinMax scaling
3. Predicts class probabilities
4. Applies the calibrated threshold to produce binary labels

---

## 11. Threshold Calibration

The default classification threshold of 0.5 is not always optimal, especially on imbalanced datasets where Sensitivity > Specificity is clinically preferred.

### Fold-local threshold selection

Within each CV fold, a calibration split of the training fold is held out to find the threshold that maximizes a target criterion (e.g., Youden's J = Sensitivity + Specificity − 1, or F₁).

```
fold_thresholds = [t₁, t₂, ..., t_k]    (one per fold)
training_threshold = mean(fold_thresholds)
```

### Final threshold

The final model's threshold is determined by evaluating the full training set at a range of candidate thresholds and selecting the best:
```
final_threshold = argmax_t [target_criterion(t) on training set]
```

All three thresholds are recorded in the output:
- `fold_thresholds`: Per-fold calibration thresholds (list)
- `training_threshold`: Mean of fold thresholds
- `final_threshold`: Training-set-optimized threshold used for inference
