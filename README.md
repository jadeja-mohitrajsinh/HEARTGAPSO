# HeartGAPSO

HeartGAPSO trains a Random Forest for the binary UCI Cleveland heart-disease
target (severity 1--4 becomes disease). It uses GA feature/hyperparameter
screening, PSO repair, fold-local preprocessing, pooled out-of-fold threshold
selection, high-fidelity repeated CV, and a multi-seed stability check.

## Current model status

`models/gapso_random_forest.pkl` is a retained legacy release. Its post-fix
compatibility check reproduces **0.885246 accuracy** and **0.961039 ROC-AUC**
on its fixed, 61-row Cleveland holdout. It is not a clinical device and must
not be used for diagnosis or treatment decisions.

The repaired pipeline's first CV-selected candidate is preserved—not promoted—
under `runs/2026-09-30_cv_selected_candidate/`. It was selected without
holdout input but did not clear the release accuracy gate. See
[`models/MODEL_STATUS.md`](models/MODEL_STATUS.md) for the explicit lineage
decision.

## Run

```text
pip install -r requirements.txt
python test_model.py
python predict.py
python train.py
```

`test_model.py` recreates the model's saved seed-42 stratified 20% Cleveland
holdout. Supply `--data external.csv` to test a truly independent, labelled
dataset. `predict.py --input patients.csv` requires the 13 named Cleveland
feature columns.

The default training run uses 12 GA candidates, 10 generations, two PSO repair
iterations, a 90-evaluation fast-screen budget, 5-fold x 2-repeat candidate
CV, 10 promoted finalists, and three-seed stability selection. The holdout is
split before optimization and is used once, after model selection.

Training writes a new candidate directory under `runs/` and never replaces the
released `models/` files unless `--promote` is supplied after review.

## Reproducibility and artifacts

Every completed repaired training run writes:

- `models/` — promoted inference model and configuration.
- `artifacts/` — candidate audit, CV threshold analysis, GA/PSO history, and
  a SHA-256 run manifest for reproducible candidates.
- `results/` — one final holdout report and diagnostic figures.

Run snapshots are never deleted automatically. The pre-fix model lineage is in
`runs/2026-09-30_pre_fix/`; the repaired CV candidate is in
`runs/2026-09-30_cv_selected_candidate/`.
