# HeartGAPSO configuration

The executable contract is `train.py` and the modules in `src/`. This document
matches the repaired default training command:

```text
python train.py --seed 42 --generations 10 --population 12 \
  --pso-iterations 2 --cv-folds 5 --cv-repeats 2 \
  --max-evaluations 90 --fast-promotions 15 \
  --high-fidelity-candidates 10 --stability-candidates 3 \
  --stability-seeds 3
```

| Setting | Purpose |
| --- | --- |
| `--seed` | Master split, GA/PSO, and RF seed. |
| `--generations`, `--population` | GA search size. |
| `--pso-iterations` | PSO repair steps per GA generation, capped at three for a bounded run. |
| `--max-evaluations` | Fast 3-fold screening cache-miss budget. |
| `--cv-folds`, `--cv-repeats` | High-fidelity repeated stratified CV. |
| `--fast-promotions`, `--high-fidelity-candidates` | Candidates moved from screening to high-fidelity evaluation. |
| `--stability-candidates`, `--stability-seeds` | Final multi-seed stability audit. |

The chromosome jointly selects a non-empty subset of the 13 Cleveland input
features and these RF parameters:

| Parameter | Values |
| --- | --- |
| `n_estimators` | 100, 200, 400, 600 (fast stage caps at 200) |
| `max_depth` | None, 4, 6, 8, 10, 14 |
| `min_samples_split` | 2, 5, 10, 15 |
| `min_samples_leaf` | 1, 2, 3, 4, 8, 10 |
| `max_features` | sqrt, log2, 0.33, 0.5, 1.0 |
| `class_weight` | None, balanced, balanced_subsample |

Fitness is `0.40 ROC-AUC + 0.25 F1 + 0.20 sensitivity + 0.15 specificity`,
minus feature-count and fold-stability penalties. Thresholds are selected from
pooled CV out-of-fold predictions in the 0.30--0.70 range; the reserved 20%
holdout is not used during selection.
