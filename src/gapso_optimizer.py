"""Leakage-safe GA--PSO search for feature masks and RF hyperparameters."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from .particle_swarm import repair_rejected
from .preprocessing import make_preprocessor
from .random_forest import make_random_forest


PARAMETER_SPACE = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [None, 4, 6, 8, 10, 14],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 3, 4, 8, 10],
    "max_features": ["sqrt", "log2", 0.33, 0.5, 1.0],
    "class_weight": [None, "balanced", "balanced_subsample"],
}
PARAMETER_NAMES = tuple(PARAMETER_SPACE)
THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.01), 2)


def _decode_genes(genes: np.ndarray, fast: bool) -> dict:
    params = {}
    for gene, name in zip(np.asarray(genes, dtype=float), PARAMETER_NAMES):
        choices = PARAMETER_SPACE[name]
        index = int(np.clip(round(float(gene) * (len(choices) - 1)), 0, len(choices) - 1))
        params[name] = choices[index]
    if fast:
        params["n_estimators"] = min(int(params["n_estimators"]), 200)
    return params


def _encode_params(params: dict) -> np.ndarray:
    encoded = []
    for name in PARAMETER_NAMES:
        choices = PARAMETER_SPACE[name]
        value = params[name]
        encoded.append(choices.index(value) / max(1, len(choices) - 1))
    return np.asarray(encoded, dtype=float)


def _metrics(y_true, probabilities: np.ndarray, threshold: float) -> dict:
    prediction = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
    }


def _objective(metrics: dict, feature_count: int, fitness_config) -> float:
    return (
        fitness_config.roc_auc_weight * metrics["roc_auc"]
        + fitness_config.f1_weight * metrics["f1"]
        + fitness_config.sensitivity_weight * metrics["sensitivity"]
        + fitness_config.specificity_weight * metrics["specificity"]
        - fitness_config.feature_penalty * feature_count
    )


def _evaluate_candidate(mask: np.ndarray, genes: np.ndarray, X, y, fitness_config, seed: int, fast: bool) -> dict:
    """Evaluate one candidate with fold-local preprocessing and pooled CV OOF thresholding."""
    selected = np.flatnonzero(mask).tolist()
    if not selected:
        raise ValueError("Candidate mask cannot be empty")
    params = _decode_genes(genes, fast=fast)
    folds = min(3, fitness_config.folds) if fast else fitness_config.folds
    repeats = 1 if fast else fitness_config.repeats
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    X_selected = X.iloc[:, selected]
    all_y, all_probabilities, fold_payloads = [], [], []
    for train_index, validation_index in splitter.split(X_selected, y):
        X_train, X_validation = X_selected.iloc[train_index], X_selected.iloc[validation_index]
        y_train, y_validation = y.iloc[train_index], y.iloc[validation_index]
        preprocessor = make_preprocessor()
        model = make_random_forest(params, seed)
        model.fit(preprocessor.fit_transform(X_train), y_train)
        probabilities = model.predict_proba(preprocessor.transform(X_validation))[:, 1]
        all_y.extend(y_validation.tolist())
        all_probabilities.extend(probabilities.tolist())
        fold_payloads.append((y_validation.to_numpy(), probabilities))

    pooled_y = np.asarray(all_y, dtype=int)
    pooled_probabilities = np.asarray(all_probabilities, dtype=float)
    threshold_analysis = []
    for threshold in THRESHOLDS:
        threshold_metrics = _metrics(pooled_y, pooled_probabilities, float(threshold))
        threshold_analysis.append({"threshold": float(threshold), **threshold_metrics,
                                   "fitness": _objective(threshold_metrics, len(selected), fitness_config)})
    best_threshold_row = max(threshold_analysis, key=lambda item: (item["fitness"], item["sensitivity"], item["specificity"]))
    threshold = best_threshold_row["threshold"]
    fold_scores = [_objective(_metrics(fold_y, fold_probability, threshold), len(selected), fitness_config)
                   for fold_y, fold_probability in fold_payloads]
    metrics = _metrics(pooled_y, pooled_probabilities, threshold)
    cv_std = float(np.std(fold_scores, ddof=0))
    fitness = _objective(metrics, len(selected), fitness_config) - fitness_config.stability_penalty * cv_std
    return {
        "mask": np.asarray(mask, dtype=bool),
        "genes": np.asarray(genes, dtype=float),
        "params": params,
        "feature_subset": selected,
        "feature_count": len(selected),
        "fitness": float(fitness),
        "threshold": float(threshold),
        "metrics": metrics,
        "cv_std": cv_std,
        "fold_thresholds": [float(threshold)] * len(fold_payloads),
        "threshold_analysis": threshold_analysis,
        "folds": folds,
        "repeats": repeats,
        "seed": int(seed),
        "stage": "fast" if fast else "high_fidelity",
    }


def optimize(X, y, seed, ga_config, fitness_config, statistical_analysis, **kwargs) -> dict:
    """Run GA screening, PSO repair, high-fidelity CV, and stability selection.

    The holdout is not supplied here; every selection decision is therefore
    confined to the caller's training partition.
    """
    started = perf_counter()
    rng = np.random.default_rng(seed)
    n_features = X.shape[1]
    max_evaluations = ga_config.max_evaluations
    fast_promotions = int(kwargs.get("fast_promotions", 20))
    high_limit = int(kwargs.get("high_fidelity_candidate_limit", 12))
    stability_limit = int(kwargs.get("stability_candidates", 3))
    stability_seeds = int(kwargs.get("stability_seeds", 3))
    cache: dict[tuple, dict] = {}
    fast_records: dict[tuple, dict] = {}
    cache_hits = 0
    cache_misses = 0
    pso_evaluations = 0

    def canonical(mask, genes) -> tuple:
        return tuple(np.asarray(mask, dtype=bool).astype(int).tolist()) + tuple(np.round(np.asarray(genes), 5).tolist())

    def evaluate(mask, genes, *, fast: bool, evaluation_seed: int, pso: bool = False):
        nonlocal cache_hits, cache_misses, pso_evaluations
        normalized_mask = np.asarray(mask, dtype=float) >= 0.5
        if not normalized_mask.any():
            normalized_mask[int(np.argmax(np.asarray(mask, dtype=float)))] = True
        normalized_genes = np.clip(np.asarray(genes, dtype=float), 0.0, 1.0)
        key = ("fast" if fast else "high", evaluation_seed, canonical(normalized_mask, normalized_genes))
        if key in cache:
            cache_hits += 1
            return cache[key]
        if fast and cache_misses >= max_evaluations:
            return None
        record = _evaluate_candidate(normalized_mask, normalized_genes, X, y, fitness_config, evaluation_seed, fast)
        cache[key] = record
        cache_misses += 1
        if pso:
            pso_evaluations += 1
        if fast:
            fast_records[canonical(normalized_mask, normalized_genes)] = record
        return record

    important = set(statistical_analysis.get("important_features", []))
    seeded_mask = np.asarray([name in important for name in X.columns], dtype=bool)
    if not seeded_mask.any():
        seeded_mask[:] = True
    baseline_params = {
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
        "min_samples_leaf": 3, "max_features": "sqrt", "class_weight": "balanced",
    }
    population: list[tuple[np.ndarray, np.ndarray]] = [
        (np.ones(n_features, dtype=bool), _encode_params(baseline_params)),
        (seeded_mask.copy(), _encode_params(baseline_params)),
    ]
    while len(population) < ga_config.population_size:
        mask = rng.random(n_features) < 0.55
        if not mask.any():
            mask[rng.integers(n_features)] = True
        population.append((mask, rng.random(len(PARAMETER_NAMES))))

    history, pso_history = [], []
    best_record = None
    stagnant = 0
    for generation in range(ga_config.generations):
        scored = []
        for mask, genes in population:
            result = evaluate(mask, genes, fast=True, evaluation_seed=seed)
            if result is not None:
                scored.append(result)
        if not scored:
            break
        scored.sort(key=lambda item: item["fitness"], reverse=True)
        generation_best = scored[0]
        if best_record is None or generation_best["fitness"] > best_record["fitness"] + 1e-12:
            best_record, stagnant = generation_best, 0
        else:
            stagnant += 1
        history.append({
            "generation": generation,
            "best_fitness": float(generation_best["fitness"]),
            "mean_fitness": float(np.mean([item["fitness"] for item in scored])),
            "selected_feature_counts": [int(item["feature_count"]) for item in scored],
        })
        if cache_misses >= max_evaluations or stagnant >= ga_config.stagnation_generations:
            break

        def tournament():
            candidates = rng.choice(scored, size=min(ga_config.tournament_size, len(scored)), replace=False)
            return max(candidates, key=lambda item: item["fitness"])

        next_population = [(item["mask"].copy(), item["genes"].copy())
                           for item in scored[:min(ga_config.elite_count, len(scored))]]
        while len(next_population) < ga_config.population_size:
            first, second = tournament(), tournament()
            mask = np.where(rng.random(n_features) < 0.5, first["mask"], second["mask"]).astype(bool)
            genes = np.where(rng.random(len(PARAMETER_NAMES)) < 0.5, first["genes"], second["genes"])
            if rng.random() > ga_config.crossover_rate:
                mask, genes = first["mask"].copy(), first["genes"].copy()
            mutation_rate = 0.12
            mask ^= rng.random(n_features) < mutation_rate
            genes = np.clip(genes + rng.normal(0, 0.12, len(PARAMETER_NAMES)) * (rng.random(len(PARAMETER_NAMES)) < mutation_rate), 0, 1)
            if not mask.any():
                mask[rng.integers(n_features)] = True
            next_population.append((mask, genes))

        rejected = scored[-min(ga_config.pso_particles, max(1, len(scored) // 2)):]
        packed = [(np.concatenate([item["mask"].astype(float), item["genes"]]), item["fitness"]) for item in rejected]

        def pso_evaluate(feature_values, gene_values):
            record = evaluate(feature_values, gene_values, fast=True, evaluation_seed=seed, pso=True)
            if record is None:
                return -1.0, {"fitness": -1.0}
            return record["fitness"], record

        repaired, repair_history = repair_rejected(
            packed, pso_evaluate, n_features, seed + generation + 1,
            iterations=min(ga_config.pso_iterations, 3),
        )
        pso_history.extend([{**entry, "generation": generation} for entry in repair_history])
        for index, (position, _, _) in enumerate(repaired[:len(next_population)]):
            replacement = (np.asarray(position[:n_features]) >= 0.5, np.clip(position[n_features:], 0, 1))
            next_population[-(index + 1)] = replacement
        population = next_population

    if best_record is None:
        raise RuntimeError("The optimizer exhausted its budget before producing a candidate")

    fast_ranked = sorted(fast_records.values(), key=lambda item: item["fitness"], reverse=True)
    promotions = fast_ranked[:min(fast_promotions, len(fast_ranked))]
    high_records = []
    for record in promotions[:min(high_limit, len(promotions))]:
        high_records.append(evaluate(record["mask"], record["genes"], fast=False, evaluation_seed=seed))
    high_records.sort(key=lambda item: item["fitness"], reverse=True)
    if not high_records:
        raise RuntimeError("No candidates reached high-fidelity evaluation")

    finalists = high_records[:min(stability_limit, len(high_records))]
    stability = []
    for finalist_index, record in enumerate(finalists):
        seed_records = [record]
        for repeat_index in range(1, stability_seeds):
            seed_records.append(evaluate(record["mask"], record["genes"], fast=False,
                                         evaluation_seed=seed + 1000 * (finalist_index + 1) + repeat_index))
        scores = [item["fitness"] for item in seed_records]
        stability.append({"record": record, "mean": float(np.mean(scores)), "std": float(np.std(scores)),
                          "scores": [float(score) for score in scores]})
    stability.sort(key=lambda item: item["mean"] - fitness_config.stability_penalty * item["std"], reverse=True)
    winner = stability[0]["record"]

    audit = []
    high_index = {canonical(item["mask"], item["genes"]): index + 1 for index, item in enumerate(high_records)}
    final_index = {canonical(item["record"]["mask"], item["record"]["genes"]): index + 1 for index, item in enumerate(stability)}
    stability_by_key = {canonical(item["record"]["mask"], item["record"]["genes"]): item for item in stability}
    for fast_rank, record in enumerate(promotions, start=1):
        key = canonical(record["mask"], record["genes"])
        high_rank = high_index.get(key)
        high_record = next((candidate for candidate in high_records if canonical(candidate["mask"], candidate["genes"]) == key), None)
        stable = stability_by_key.get(key)
        audit.append({
            "candidate_id": int(fast_rank), "fast_rank": int(fast_rank), "screening_rank": int(fast_rank),
            "final_rank": final_index.get(key), "feature_subset": record["feature_subset"],
            "feature_count": record["feature_count"], "rf_parameters": record["params"],
            "screening_fitness": record["fitness"], "high_fidelity_fitness": high_record["fitness"] if high_record else None,
            "final_fitness": (stable["mean"] - fitness_config.stability_penalty * stable["std"]) if stable else None,
            "threshold": high_record["threshold"] if high_record else record["threshold"],
            "cv_roc_auc": (high_record or record)["metrics"]["roc_auc"],
            "cv_accuracy": (high_record or record)["metrics"]["accuracy"],
            "cv_f1": (high_record or record)["metrics"]["f1"],
            "cv_sensitivity": (high_record or record)["metrics"]["sensitivity"],
            "cv_specificity": (high_record or record)["metrics"]["specificity"],
            "cv_std": (high_record or record)["cv_std"],
            "seed_mean_fitness": stable["mean"] if stable else None,
            "seed_std": stable["std"] if stable else None,
            "seed_scores": stable["scores"] if stable else [],
        })

    return {
        "features": winner["mask"].astype(bool).tolist(), "params": winner["params"],
        "fitness": float(stability[0]["mean"] - fitness_config.stability_penalty * stability[0]["std"]),
        "threshold": winner["threshold"], "training_threshold": winner["threshold"],
        "final_threshold": winner["threshold"], "fitness_metrics": {
            **winner["metrics"], "cv_std": winner["cv_std"], "fold_thresholds": winner["fold_thresholds"],
            "threshold": winner["threshold"], "threshold_analysis": winner["threshold_analysis"],
            "metric_standard_deviations": {},
        },
        "history": history, "pso_history": pso_history, "candidate_audit": audit,
        "evaluations": cache_misses, "cache_hits": cache_hits, "cache_misses": cache_misses,
        "pso_evaluations": pso_evaluations, "high_fidelity_evaluations": len(high_records),
        "fast_promotions": len(promotions), "runtime_seconds": perf_counter() - started,
    }
