# Current artifact status

The JSON and pickle files in this folder record the repaired 2026-09-30
CV-selected candidate, not the retained legacy model in `models/`. This is
intentional: the candidate is a traceable failed release candidate rather than
an inferred provenance match for the legacy model. See
`models/MODEL_STATUS.md` for the release decision and the two immutable run
snapshots under `runs/` for each lineage.
