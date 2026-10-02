# Model status

`models/gapso_random_forest.pkl` is the retained legacy release. Its first
post-fix compatibility verification reproduced its recorded fixed-holdout
result: accuracy 0.885246 and ROC-AUC 0.961039 on the 61-row, seed-42
Cleveland holdout. It remains the prediction target because the newly repaired
CV-selected candidate did not clear this release-quality gate on that holdout.

The legacy release predates the repaired training source, so it is usable but
not source-reproducible. Its original files are preserved in
`runs/2026-09-30_pre_fix/models/`.

The repaired, reproducible candidate and its complete model, CV audit,
manifest, and holdout result are preserved separately in
`runs/2026-09-30_cv_selected_candidate/`. It selected six features by 5-fold,
two-repeat CV plus three-seed stability analysis, then obtained 0.836066
holdout accuracy and 0.945887 ROC-AUC. It was not promoted to `models/`.
