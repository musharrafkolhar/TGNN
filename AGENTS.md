# TGNN v1 contributor instructions

Read README.md, ARCHITECTURE.md and VERIFICATION.md before editing. This is the local v1 prototype for synthetic laundering transaction classification. Implement only the active report scope; research extensions are deferred.

## Correctness rules

1. Use chronological splits, preserving timestamp ties. Fit encoders, scalers and class weights on train only.
2. Exclude labels, attempts and typologies from model inputs/messages/memory.
3. Score every event at a timestamp before inserting any same-time event into history. Never shuffle training. Reset history each epoch; detach recurrent state appropriately.
4. Select a checkpoint using validation AP. Reload it and reconstruct history with fixed weights before final held-out predictions. Continue history through split boundaries without resets.
5. Choose thresholds on validation only. Report AP as average_precision_score, separately from trapezoidal PR-AUC. Report measured precision/recall/F1; do not substitute accuracy.
6. Save predictions, config, seed, environment and provenance with every run. No fake results. Explicitly label fixtures and unavailable attention.
7. Require identical dataset identity and held-out edge IDs/splits/labels for comparisons. Verify canonical joins before alert export.
8. Historical context is strictly earlier than the target. Temporal motifs need supporting edges and correct time order. Motifs/attention are descriptive, not fraud proof or causal attribution.
9. Account identity is the bank/account pair. Preserve strings and original source row IDs. Keep data, checkpoints, runs and credentials out of Git.

## Module and workflow rules

- Stages communicate through files. Evaluation imports no model code. App reads saved JSON/SQLite and imports no training/model code. Utilities import no project modules.
- Use YAML configuration, pathlib, public type hints, logging and seed utilities. Record changed contracts in ARCHITECTURE.md alongside code.
- Existing stack: Python 3.11/3.12, PyTorch/PyG, XGBoost, scikit-learn, pandas/NumPy/Parquet, NetworkX, Plotly, Streamlit and SQLite. Ask before adding infrastructure or dependencies outside this scope unless the user's request authorizes them.
- For code changes, add meaningful regression tests, then run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`. A TGN skip does not verify TGN. Fix failures rather than weakening assertions.
- Make targets mirror documented module commands. Do not push or merge GitHub changes without authorization.
- Summarize verified behavior and remaining limitations honestly. Team ownership must be agreed by humans; do not invent assignments.
