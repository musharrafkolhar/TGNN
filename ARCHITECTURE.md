# TGNN v1 architecture and contracts

This document describes the implemented prototype. Research extensions are deferred in README.md. Team ownership and repository integration require human agreement.

## File-based pipeline

`IBM CSV → preprocess → canonical data/features → XGBoost and TGN runs → evaluation → historical context/SQLite → Streamlit`

Stages communicate through saved artifacts. Evaluation imports no model code. The app imports no training/model code and reads saved JSON/SQLite only. Utilities import no project modules. The scorer owns TGN history/checkpoint state.

## Canonical data

`data/ibm_pilot_processed/edges.parquet` contains:

- `edge_id`: contiguous int64 row address in canonical chronological order, starting at zero; unique within the source/subset identity.
- `original_row`: original zero-based CSV data-row ID, retained through chunking/subsetting/sorting.
- `timestamp`: UTC calendar timestamp. `calendar_seconds`: Unix seconds. `ts`: integer seconds relative to the full source dataset start, retained separately in the manifest.
- `src`, `dst`: integer node IDs. Node identity is the exact `(bank, account)` string pair, including leading zeros. `node_index.parquet` saves that mapping. Global node IDs are addressing only; they are not learned features.
- Paid/received amounts, from/to bank, paid/received currencies and payment format.
- `split`: train/val/test. `label`: binary synthetic laundering annotation. `attempt_id` and `typology` are nullable evaluation-only placeholders.

The complete input is inspected in configurable bounded CSV chunks. Input need not be sorted. Candidate retention keeps the earliest `max_events+1` rows, then canonicalization excludes the whole timestamp group at the cutoff. All events within the resulting interval are retained. Stable sorting uses timestamp then original row ID. Start is inclusive, end exclusive. Source checksum, schema, parser dtypes, missingness, timestamp range, counts/prevalence and settings are saved in `preprocess_report.json`; selected interval/split boundaries are also in `splits.json`.

Approximately 60/20/20 chronological splits preserve ties. Preprocessing requires nonempty splits with both classes, failing with a clear instruction to revise the documented subset. Metric evaluation separately reports a no-positive test set explicitly; validation needs both classes to choose an F1 cutoff.

## Features

`features.npy` is aligned with canonical rows, float32 and label-free. `transformers.pkl` is a trusted-local scikit-learn transformer fitted only on train rows.

Allowlist: log paid/received amounts, same-bank indicator, same-currency indicator, calendar time-of-day sine/cosine, and categorical from/to bank, currencies and payment format. OneHotEncoder caps each categorical field at 32 encoded columns, groups train-infrequent categories and maps unseen categories to zeros. No account IDs, labels, attempts, typologies or future aggregates enter features. Capped encodings keep the dense pilot feature matrix bounded on a laptop.

## Models and temporal state

XGBoost uses current transaction features with train-derived class weight. Parameters and seed are in YAML.

TGN uses PyG TGNMemory, IdentityMessage, LastAggregator, LastNeighborLoader, TransformerConv with time-encoded historical edges, and an MLP transaction classification head receiving source embedding, destination embedding and current features. The objective is weighted binary transaction classification, not link prediction.

At each timestamp: score every chunk, compute training gradients, then make one optimizer update and observe the complete timestamp group. No same-time event enters another's score. Reset memory/neighbours each training epoch; detach recurrent memory after observation. Use validation AP to save the best checkpoint. Reload it, reset history and replay train → validation → test with fixed weights; no resets at split boundaries and no parameter updates in held-out scoring. Replay requires canonical IDs and finite aligned features.

`checkpoint_best.pt` is a trusted-local PyTorch checkpoint. `environment.json`, `seed.txt`, resolved `config.yaml` and `training_history.json` record the run. XGBoost saves `model.json`. Every run has `data_manifest.json` including canonical edges SHA-256.

## Predictions, metrics and comparisons

Every model writes `predictions.parquet` with `edge_id`, `split` (val/test), `score` (finite [0,1]), `label`, `model` (xgboost/tgn), `run_id`. The same edge in different runs/models is a distinct result. Risk scores are not assumed calibrated.

Prediction validation rejects duplicate/invalid IDs, blank identities, mixed runs/models, bad splits/labels and invalid scores. Comparison requires identical canonical dataset hashes and identical held-out IDs/splits/labels. Each model selects its own maximum-validation-F1 threshold with highest-cutoff tie breaking. Freeze it for test. Metrics include AP, precision, recall, illicit-class F1, counts/prevalence and TN/FP/FN/TP. Zero division is zero; no-positive test AP is null with a reason. AP is scikit-learn average precision, not trapezoidal PR-AUC. Save `metrics.json` and self-contained `pr_curve.html`; comparison JSON contains actual results and verified set identity.

## Context and attention

Each alert has JSON in `runs/RUN/contexts/EDGE.json` and in SQLite. Fields include run/model/target identity, score, threshold, window, history policy, context limits/truncation, target separately, nodes, historical edges, motif evidence, attention status, factual summary and interpretation.

History satisfies `target.ts - window <= ts < target.ts`. Bounded hop expansion starts at both endpoints and keeps the most recent `max_edges`. A cycle requires a directed historical path from target destination to target source with increasing times; the target closes it. Fan-out counts distinct non-self destinations. Supporting edge IDs and time span are retained. The motif detector rejects nonhistorical input.

TGN replay captures attention for the same checkpoint/history used for scoring. Records identify target and actual historical edge IDs; endpoint embedding attention is averaged across heads/directions where applicable. It is not complete causal attribution of the classifier. Missing attention is explicitly unavailable; empty motif lists are valid.

## SQLite and dashboard

`alerts` stores run/model/edge, dataset-relative time, nodes, amount/currency, saved score/threshold, evaluation-only label and investigator status. Uniqueness is `(run_id, model, edge_id)`. `explanations` joins context JSON to alert ID. Upserts preserve review status. Allowed states are new/reviewed/dismissed; none automatically confirms fraud.

Export verifies canonical dataset hash and prediction ID/split/label joins, then saves the top configured threshold-passing test alerts. A demo bundle includes SQLite, metrics, predictions, PR curve, config, environment, seed and provenance. The dashboard works without checkpoints/raw CSV. Queue/detail share `app/Home.py`, with score/time/node/status filters, evaluation-mode label display, calendar timestamps, directed Plotly graph and evidence tables. Missing/incomplete outputs produce setup guidance.

## Verification and deferred work

Tests exercise ordering, chunk/source preservation, train-only transforms, metric known answers, frozen thresholds, schema/comparison/joins, context causality, motifs, attention timing, actual TGN memory learning, checkpoint replay, deterministic training, SQLite idempotency and Streamlit views. Run pytest and Ruff checks from README.md. TGN skips when its dependency is absent; skips cannot count as TGN verification. Actual execution status/results live in VERIFICATION.md and saved run artifacts.

No Docker/API/production stream/authentication, static GNN, graph-feature baseline, multi-seed runs, pattern matching, GNNExplainer or bank integrations are required for v1. No additional infrastructure dependencies were introduced. Python 3.11/3.12 and supplied CPU pins are the local target. CUDA/Colab require separate checks.
