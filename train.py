import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
import numpy
import pandas
import sklearn

from src.data_loader import FEATURE_NAMES, load_cleveland
from src.evaluation import evaluate_model
from src.evaluation import save_optimization_diagnostics
from src.fitness import FitnessConfig
from src.gapso_optimizer import PARAMETER_SPACE, optimize
from src.genetic_algorithm import GAConfig
from src.preprocessing import make_preprocessor
from src.random_forest import InferencePipeline, make_random_forest
from src.statistical_analysis import analyze_features, save_analysis


def main():
    parser = argparse.ArgumentParser(description="Train the HeartGAPSO model")
    parser.add_argument("--data", default="heart+disease/processed.cleveland.data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--pso-iterations", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--max-evaluations", type=int, default=90)
    parser.add_argument("--high-fidelity-candidates", type=int, default=10)
    parser.add_argument("--stability-candidates", type=int, default=3)
    parser.add_argument("--fast-promotions", type=int, default=15)
    parser.add_argument("--stability-seeds", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, help="Directory for a non-promoted candidate run")
    parser.add_argument("--promote", action="store_true", help="Replace models/, artifacts/, and results/ after review")
    args = parser.parse_args()
    run_dir = args.run_dir or Path("runs") / datetime.now().strftime("%Y-%m-%d_%H%M%S_candidate")
    model_dir = Path("models") if args.promote else run_dir / "models"
    artifact_dir = Path("artifacts") if args.promote else run_dir / "artifacts"
    results_dir = Path("results") if args.promote else run_dir / "results"
    for directory in (model_dir, artifact_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Loading dataset...", flush=True)
    X, y = load_cleveland(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed,
    )
    print("Preprocessing... (median imputation and Min-Max normalization are fold-local)", flush=True)
    print("Statistical analysis...", flush=True)
    statistical_analysis = analyze_features(X_train, y_train)
    print(f"Top statistically significant features: {statistical_analysis['important_features']}", flush=True)
    save_analysis(statistical_analysis, artifact_dir / "statistical_feature_analysis.json")
    print("Starting GA... RF fitness evaluations will report after each candidate.", flush=True)
    fitness_config = FitnessConfig(folds=args.cv_folds, repeats=args.cv_repeats)
    ga_config = GAConfig(population_size=args.population, generations=args.generations,
                         pso_iterations=args.pso_iterations,
                         max_evaluations=args.max_evaluations,
                         pso_particles=20)
    result = optimize(
        X_train, y_train, args.seed,
        ga_config,
        fitness_config,
        high_fidelity_candidate_limit=args.high_fidelity_candidates,
        stability_candidates=args.stability_candidates,
        fast_promotions=args.fast_promotions,
        stability_seeds=args.stability_seeds,
        statistical_analysis=statistical_analysis,
    )
    selected = [index for index, bit in enumerate(result["features"]) if bit]
    if not selected:
        raise RuntimeError("Optimizer returned an empty feature subset")
    print(f"Best feature subset found: {[FEATURE_NAMES[i] for i in selected]}", flush=True)
    print(f"Best RF parameters found: {result['params']}", flush=True)
    preprocessor = make_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train.iloc[:, selected])
    print("Training final model...", flush=True)
    model = make_random_forest(result["params"], args.seed)
    model.fit(X_train_processed, y_train)
    pipeline = InferencePipeline(
        preprocessor, model, selected, result["threshold"],
        feature_names=[FEATURE_NAMES[index] for index in selected],
    )
    print("Evaluating untouched holdout exactly once...", flush=True)
    metrics = evaluate_model(pipeline, X_test, y_test, selected, results_dir, result["threshold"])
    runtime = time.perf_counter() - started
    report = {
        **metrics, "selected_features": [FEATURE_NAMES[i] for i in selected],
        "number_selected_features": len(selected), "best_rf_parameters": result["params"],
        "optimization_fitness": result["fitness"], "optimization_runtime_seconds": runtime,
        "classification_threshold": result["threshold"],
        "training_threshold": result["training_threshold"],
        "final_threshold": result["final_threshold"],
        "cross_validation_mean": {
            key: value for key, value in result["fitness_metrics"].items()
            if key not in {"cv_std", "threshold", "fold_thresholds", "baseline_threshold",
                           "threshold_analysis", "metric_standard_deviations"}
        },
        "cross_validation_standard_deviation": result["fitness_metrics"]["cv_std"],
        "cross_validation_metric_standard_deviations": result["fitness_metrics"].get(
            "metric_standard_deviations", {}
        ),
        "threshold_analysis": result["fitness_metrics"].get("threshold_analysis", []),
        "baseline_threshold_metrics": next(
            (item for item in result["fitness_metrics"].get("threshold_analysis", [])
             if item["threshold"] == 0.5), None,
        ),
        "optimizer_evaluations": result["evaluations"],
        "optimizer_cache_hits": result["cache_hits"],
        "optimizer_cache_misses": result["cache_misses"],
        "pso_evaluations": result["pso_evaluations"],
        "high_fidelity_evaluations": result["high_fidelity_evaluations"],
        "fast_promotions": result["fast_promotions"],
        "candidate_audit": [
            {
                **candidate,
                "feature_subset": [FEATURE_NAMES[index] for index in candidate["feature_subset"]],
            }
            for candidate in result["candidate_audit"]
        ],
        "optimizer_runtime_seconds": result["runtime_seconds"],
        "cv_configuration": {"folds": args.cv_folds, "repeats": args.cv_repeats},
        "fitness_weights": {
            "roc_auc": fitness_config.roc_auc_weight,
            "f1": fitness_config.f1_weight,
            "sensitivity": fitness_config.sensitivity_weight,
            "specificity": fitness_config.specificity_weight,
        },
        "optimization_configuration": {
            "population_size": ga_config.population_size,
            "generations": ga_config.generations,
            "pso_iterations": ga_config.pso_iterations,
            "elite_count": ga_config.elite_count,
            "stagnation_generations": ga_config.stagnation_generations,
            "rf_search_space": PARAMETER_SPACE,
            "fast_stage": {
                "cv_folds": 3,
                "cv_repeats": 1,
                "n_estimators": [100, 200],
                "rf_tree_cap": 200,
            },
            "pso_particles": ga_config.pso_particles,
            "high_fidelity_candidate_limit": args.high_fidelity_candidates,
            "stability_candidate_limit": args.stability_candidates,
            "high_fidelity_cv_folds": max(5, args.cv_folds),
            "max_evaluations": ga_config.max_evaluations,
            "fast_promotions": args.fast_promotions,
            "stability_seeds": args.stability_seeds,
        },
        "seed": args.seed,
        "statistical_analysis": statistical_analysis,
        "preprocessing": ["median_imputation", "min_max_normalization"],
    }
    with (model_dir / "gapso_random_forest.pkl").open("wb") as file:
        pickle.dump(pipeline, file)
    with (model_dir / "preprocessing.pkl").open("wb") as file:
        pickle.dump(preprocessor, file)
    (model_dir / "selected_features.json").write_text(json.dumps(report["selected_features"], indent=2))
    (model_dir / "model_config.json").write_text(json.dumps(report, indent=2, default=str))
    (results_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
    (results_dir / "candidate_audit.json").write_text(
        json.dumps(report["candidate_audit"], indent=2, default=str), encoding="utf-8",
    )
    (artifact_dir / "ga_history.json").write_text(
        json.dumps(result["history"], indent=2, default=str), encoding="utf-8",
    )
    (artifact_dir / "pso_history.json").write_text(
        json.dumps(result["pso_history"], indent=2, default=str), encoding="utf-8",
    )
    (artifact_dir / "candidate_audit.json").write_text(
        json.dumps(report["candidate_audit"], indent=2, default=str), encoding="utf-8",
    )
    (artifact_dir / "selected_features.json").write_text(
        json.dumps(report["selected_features"], indent=2), encoding="utf-8",
    )
    (artifact_dir / "final_model_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8",
    )
    with (artifact_dir / "final_model.pkl").open("wb") as file:
        pickle.dump(pipeline, file)
    with (artifact_dir / "preprocessing.pkl").open("wb") as file:
        pickle.dump(preprocessor, file)
    (artifact_dir / "feature_names.json").write_text(
        json.dumps(report["selected_features"], indent=2), encoding="utf-8",
    )
    (artifact_dir / "final_threshold.json").write_text(
        json.dumps({
            "fold_thresholds": result["fitness_metrics"].get("fold_thresholds", []),
            "training_threshold": result["training_threshold"],
            "final_threshold": result["final_threshold"],
        }, indent=2), encoding="utf-8",
    )
    (artifact_dir / "threshold_analysis.json").write_text(
        json.dumps(result["fitness_metrics"].get("threshold_analysis", []), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "screening_ranking.json").write_text(
        json.dumps(report["candidate_audit"], indent=2, default=str), encoding="utf-8",
    )
    (artifact_dir / "stability_ranking.json").write_text(
        json.dumps([
            candidate for candidate in report["candidate_audit"] if candidate["final_rank"] is not None
        ], indent=2, default=str), encoding="utf-8",
    )
    def sha256(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "data_path": str(Path(args.data).as_posix()),
        "data_sha256": sha256(args.data),
        "model_sha256": sha256(model_dir / "gapso_random_forest.pkl"),
        "preprocessor_sha256": sha256(model_dir / "preprocessing.pkl"),
        "source_sha256": {
            str(path): sha256(path)
            for path in [Path("train.py"), *sorted(Path("src").glob("*.py"))]
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "holdout_policy": "Seed-42 stratified 20% Cleveland holdout; excluded from optimizer selection.",
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.promote:
        (model_dir / "MODEL_STATUS.md").write_text(
            "# Model status\n\n"
            "This repaired GAPSO-RF candidate was explicitly promoted after review. "
            f"Its one reserved holdout result is accuracy {report['accuracy']:.6f} "
            f"and ROC-AUC {report['roc_auc']:.6f} on {report['holdout_samples']} samples.\n",
            encoding="utf-8",
        )
        (artifact_dir / "RUN_STATUS.md").write_text(
            "# Current artifact status\n\n"
            "These artifacts correspond to the currently promoted model in `models/`.\n",
            encoding="utf-8",
        )
    save_optimization_diagnostics(result["history"], result, results_dir)
    print(f"Saved candidate run to {run_dir if not args.promote else Path.cwd()}", flush=True)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
