# GAPSO-RF Methodology

## Reference paper method

The reference workflow performs statistical analysis before feature selection, Min-Max normalization, binary feature chromosomes, modified tournament selection, uniform crossover, discriminate mutation, and PSO rehabilitation of rejected GA individuals. Random Forest is used as the fitness evaluator and final classifier.

## Current implementation

HeartGAPSO follows that architecture on the Cleveland UCI dataset. Missing values are imputed and Min-Max normalization is fitted independently inside every training fold. Statistical analysis is computed only on the outer training portion and produces class means, standard deviations, a univariate T2-style score, correlations, ranks, and per-feature mutation probabilities. `cp`, `slope`, `ca`, and `thal` are favored during initialization and receive a lower, configurable mutation probability; they are not forced into the final subset.

The GA chromosome contains 13 binary feature coordinates and six bounded Random Forest hyperparameter coordinates. Tournament selection uses the RF-based `screening_fitness`. Uniform crossover and adaptive/discriminate mutation create valid non-empty masks. Rejected candidates are sent to a binary/discrete PSO repair stage, where feature coordinates are sampled through a sigmoid velocity transform, personal bests and a global best are retained, and improved particles are injected into the next GA population.

The fitness objective is:

`0.40 ROC-AUC + 0.25 F1 + 0.20 sensitivity + 0.15 specificity`

The feature-count penalty and fold-stability penalty are applied once, after the weighted objective. Thresholds are selected from an internal calibration split within each training fold. Fold thresholds, the training threshold, and final threshold are stored separately. The untouched holdout is used only once for final reporting.

The implementation intentionally adds a fast-search to high-fidelity screening funnel, deterministic multi-seed stability evaluation, explicit ranking assertions, and detailed GA/PSO telemetry. These are engineering enhancements for runtime, reproducibility, and auditability; they are not claimed to be identical to the paper's experimental implementation.