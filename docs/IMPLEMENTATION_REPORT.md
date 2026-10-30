# TGNN v1 Prototype Implementation Report

**Project:** Temporal Graph Neural Networks for Coordinated Financial Fraud Detection  
**Team:** Musharraf Kolhar, Prajwal Vitkar, Priyanshu Pande and Aaryan Karandikar  
**Guide:** Prof. Ashwini Patil  
**Institution:** JSPM's Jayawantrao Sawant College of Engineering, Hadapsar, Pune  
**Prepared:** 13 October 2026  
**Purpose:** Implementation and coordination plan for the four-member prototype team

## 1 Prototype objective

We will build a working prototype that processes timestamped account-to-account transactions, trains a Temporal Graph Network (TGN), produces transaction risk scores and presents saved alerts in a Streamlit dashboard. An evaluator can select an alert and inspect its transaction details and historical graph context.

The prototype uses the IBM synthetic anti-money-laundering dataset. Accounts are graph nodes and transactions are directed timestamped edges. The available label indicates laundering, so results must be described as performance on synthetic laundering transactions, rather than proof of broad real-world fraud detection.

The prototype will contain one XGBoost baseline and one TGN. Its purpose is to demonstrate a correct, connected implementation. TGN does not have to outperform XGBoost for the prototype to be complete. Any comparison must report actual measured results.

**Planning status:** This report defines work to implement. It does not claim completed modules, trained models or measured results. Updated prototype documentation must be committed by the team before it becomes the branch's working agreement.

## 2 Scope for v1

| Required component | Prototype behaviour |
|---|---|
| Data pipeline | Validate input files and create a reproducible chronological subset |
| Graph representation | Accounts as nodes and transactions as timestamped edges |
| Baseline | One XGBoost model using current transaction features |
| Main model | One TGN with a transaction edge-classification head |
| Evaluation | Average precision, precision, recall and illicit-class F1 |
| Threshold | Selected on validation predictions and frozen for test |
| Context | Bounded historical transaction subgraph and basic cycle/fan-out detection |
| Attention | Included only when the TGN owner can supply matching prediction-time values |
| Dashboard | Alert queue and alert detail page using saved model outputs |
| Execution | Local Python environment; training may use Colab/Kaggle |
| Verification | Data, ordering, leakage, metrics, schema and integration checks |

Docker is **not required for v1**. A live API, production stream, authentication, bank integration and online model learning are also excluded.

Static GNN comparisons, graph-feature XGBoost, ablations, multiple seeds, full-scale experiments, GNNExplainer, replay controls and broader explanation evaluation are deferred to later development.

## 3 Four member responsibilities

Only Prajwal's workstream is assigned in this report. The other three owners must be agreed by the team; the order of team names does not determine ownership.

| Member | Owner | Workstream | Main handoff |
|---|---|---|---|
| 1 | To be assigned | Data and XGBoost baseline | Canonical data, split manifest and baseline predictions |
| 2 | To be assigned | TGN model and scoring | Checkpoint and held-out predictions |
| 3 | Prajwal Vitkar | Evaluation and context | Metrics, threshold, context JSON and pattern evidence |
| 4 | To be assigned | Dashboard and integration | Alert database and working local demonstration |

Every member owns documentation and relevant tests for their module. Everyone must understand the overall model, dataset and limitations for the viva.

### Member 1 Data and baseline

**First task:** Inspect the actual downloaded input and record filenames, columns, data types, timestamp range, missing values and class balance.

Responsibilities:
- Rename ambiguous account columns explicitly and preserve original row identifiers.
- Treat an account as a unique bank-and-account pair.
- Build a stable node index and transaction edge IDs.
- Select a manageable contiguous time interval, preserve all events within it and record its boundaries.
- Create approximately 60/20/20 chronological splits, keeping equal timestamps together.
- Report positive counts and prevalence for every split. An interval with no positives in a required split cannot support minority-class evaluation.
- Fit encoders/scalers on training data only; handle unseen categories explicitly.
- Keep label, attempt ID and typology out of model features.
- Train XGBoost on transaction features and export common-format validation/test predictions.
- Coordinate patterns-file matching with Prajwal if ground-truth pattern annotations are available.

Planned files:
- `src/tgnn_fraud/data/preprocess.py`
- `src/tgnn_fraud/data/splits.py`
- `src/tgnn_fraud/data/features.py`
- `src/tgnn_fraud/data/event_stream.py`
- `src/tgnn_fraud/models/baselines.py`
- `src/tgnn_fraud/training/train_baselines.py`

Deliverables:
- `edges.parquet`, `node_index.parquet`, `splits.json`
- `preprocess_report.json`, fitted transformers and subset/source manifest
- Saved XGBoost model, run config and `predictions.parquet`

Acceptance: another member can reproduce the subset and splits; chronology and preprocessing checks pass; baseline scores are generated by a trained model.

### Member 2 TGN and scoring

**First task:** Establish a compatible PyTorch/PyG environment and validate TGN components using tiny timestamped test data.

Responsibilities:
- Build node memory, recent-neighbour sampling and a temporal graph embedding module.
- Use an edge classifier to score actual transactions, rather than a link-prediction objective.
- Include current transaction features alongside source/destination embeddings.
- Train chronologically without shuffling; score before updating event memory.
- Reset training memory/neighbours at each epoch and detach state as required by the implemented training strategy.
- Select a checkpoint using validation performance.
- For held-out evaluation, reload the selected checkpoint and rebuild historical memory with that checkpoint before scoring.
- Own `scorer.py` and all model-state logic.
- Provide predictions to Prajwal and attention records where feasible.
- Record seed, configuration and tested library versions.

Planned files:
- `src/tgnn_fraud/models/tgn.py`
- `src/tgnn_fraud/models/edge_classifier.py`
- `src/tgnn_fraud/training/train_tgn.py`
- `src/tgnn_fraud/training/losses.py`
- `src/tgnn_fraud/serving/scorer.py`

Deliverables: checkpoint, resolved config, training record and common-format validation/test predictions. Attention data must identify the target transaction and historical edge IDs.

Acceptance: genuine held-out scores exist; memory-order and label-exclusion checks pass; loading the checkpoint and rebuilding state reproduces the documented scoring procedure.

### Member 3 Prajwal Evaluation and context

**First task:** Build evaluation on small labelled prediction fixtures with known expected answers. Fixtures are test data, not project results.

Implementation order:
1. Validate prediction schema, score range, duplicate IDs and finite values.
2. Compute average precision from continuous scores.
3. Select a threshold by maximum validation F1, using a documented deterministic tie rule.
4. Apply the same threshold to test scores and compute precision, recall and F1.
5. Save metrics, threshold and provenance; produce a precision-recall curve.
6. Extract bounded historical context around the flagged transaction.
7. Detect basic cycles and fan-outs and record actual supporting edges.
8. Produce context JSON for the dashboard.
9. Integrate attention if Member 2 supplies compatible records.

Responsibilities:
- Evaluate both models on identical held-out edge sets.
- Define zero-division handling and explicitly report splits with no positives.
- Keep threshold tuning separate from test evaluation.
- Extract history strictly before the target timestamp; display the current transaction separately.
- Check time order when naming a transaction chain or temporal cycle. A cycle in an aggregated graph is not automatically a time-valid money-flow cycle.
- Distinguish repeated transfers to one account from fan-out to distinct accounts.
- Limit time window, hops and edge count to keep graphs usable.
- Use descriptive templates grounded in graph facts; no invented explanatory claims.
- Explain that attention and motifs do not prove fraud or causal model attribution.

Planned files:
- `src/tgnn_fraud/evaluation/metrics.py`
- `src/tgnn_fraud/evaluation/run_eval.py`
- `src/tgnn_fraud/explain/context.py`
- `src/tgnn_fraud/explain/motifs.py`
- `src/tgnn_fraud/explain/attention.py` when supported
- `src/tgnn_fraud/explain/report.py`
- `tests/test_metrics.py`, context and motif tests

Deliverables: `metrics.json`, saved threshold, PR curve and context JSON indexed by run and target edge.

Acceptance: known-answer tests pass; future transactions do not alter earlier context; missing motifs/attention produce explicit empty or unavailable values; JSON joins to the correct transaction.

### Member 4 Dashboard and integration

**First task:** Create a basic Streamlit interface and SQLite storage using clearly labelled fixtures while models are being developed.

Responsibilities:
- Define alert export/storage with Member 3 and preserve run identifiers.
- Display saved model scores, selected threshold and evaluation metrics.
- Build queue filters by score, time and account.
- Implement detail view with amount, timestamp, source/destination, graph and motif descriptions.
- Display ground-truth labels only in clearly marked evaluation mode.
- Read files and SQLite; do not train models inside the app.
- Handle missing output files with a useful setup message.
- Avoid duplicate alerts on repeated imports.
- Document local setup and package a small demo bundle with access instructions.
- Integrate genuine model outputs before declaring the demo complete.

Planned files:
- `src/tgnn_fraud/serving/alert_store.py`
- `src/tgnn_fraud/serving/build_demo_bundle.py`
- `app/Home.py`
- `app/pages/1_Alert_Queue.py`
- `app/pages/2_Alert_Detail.py`

Deliverables: `alerts.db`, Streamlit app, local setup guide and verified demo bundle.

Acceptance: another member can launch the app, select an alert and inspect matching context without retraining. Re-importing the same run must not duplicate alerts.

## 4 Shared data contracts

Agree schemas before working independently. Record changes in ARCHITECTURE.md and notify the consuming member.

### Transaction table

Minimum prototype fields: `edge_id`, `ts`, `src`, `dst`, `amount_paid`, `amount_received`, bank/currency/payment-format fields and `label`.

Timestamps are seconds relative to dataset start. Preserve the start timestamp separately for calendar display and time-of-day features. Stable account mappings are stored in `node_index.parquet`.

Ground-truth label and optional attempt/typology columns are evaluation metadata. They must not be included in model features.

### Prediction table

| Field | Meaning |
|---|---|
| edge_id | Stable identifier joining to the transaction table |
| split | val or test |
| score | Finite model risk score in the agreed 0 to 1 range |
| label | Ground truth for evaluation |
| model | xgboost or tgn |
| run_id | Experiment identifier |

Both models must cover the same evaluation edge IDs. A sigmoid score is not assumed to be a calibrated probability.

### Evaluation output

Store run ID, model, dataset/subset identity, per-split counts and prevalence, AP, validation-selected threshold, threshold-selection method, test precision/recall/F1 and confusion-matrix counts.

Average precision uses `average_precision_score`; it must be named AP rather than silently treated as trapezoidal PR-AUC. Preserve seed and configuration with the model run.

### Context JSON

Agree fields for:
- Run/model/target edge identifiers, score and threshold.
- Window duration, timestamp policy and context size limits.
- Context nodes and historical edges; target edge separately identified.
- Motif type, actual edge IDs, involved nodes and time span.
- Attention values when available, with a clear availability status.
- Factual summary and context truncation indicator.

Alert storage must distinguish the same edge scored by different runs/models. A uniqueness rule should cover `run_id`, `model` and `edge_id`. Reviewed status is investigator workflow metadata, not an automatic fraud confirmation.

## 5 Build sequence and parallel work

| Stage | Member 1 | Member 2 | Prajwal | Member 4 |
|---|---|---|---|---|
| P0 Setup | Inspect raw data | Test environment and tiny TGN | Agree prediction schema; metric fixtures | App/storage scaffold |
| P1 Data and baseline | Produce subset and XGBoost scores | Implement training on canonical data | Evaluate baseline; validate thresholds | Queue/detail using labelled fixtures |
| P2 Model | Support feature corrections | Produce TGN checkpoint and scores | Compare actual outputs on shared splits | Read real predictions |
| P3 Context | Support account/pattern mapping | Supply attention where feasible | Historical context, motifs and JSON | Integrate context and alerts |
| P4 Demo | Verify data reproduction | Verify checkpoint scoring | Verify metrics and context | End-to-end local demo |

Stages are dependency milestones, not promised dates. Assign dates after a small training run establishes the team's compute and time requirements. UI fixtures and metric tests can proceed early; final research experiments do not block the prototype.

## 6 Essential correctness checks

| Check | Primary owner | Required behaviour |
|---|---|---|
| Split ordering | Member 1 | Train precedes validation, which precedes test; ties stay together |
| Preprocessing leakage | Member 1 | Encoders/scalers fit on train only |
| Label exclusion | Members 1 and 2 | Labels/typologies never reach model features/messages |
| Memory ordering | Member 2 | Current/future events cannot influence earlier scores |
| Equal timestamps | Members 1 and 2 | Strictly-before policy does not expose same-time events |
| Metric correctness | Prajwal | Known-answer cases and threshold isolation pass |
| Context causality | Prajwal | Future-event changes do not alter earlier context |
| Motif validity | Prajwal | Planted cycle/fan-out tests and negative examples pass |
| Schema and joins | All members | Every artifact maps to the correct run and transaction |
| Demo integration | Member 4 with team | Saved outputs appear correctly on a fresh local setup |

Report failed checks and unavailable outputs honestly. Never replace a failed model run with invented scores or benchmark values.

## 7 Git and coordination

If the team adopts `v1` as the prototype integration branch, each member creates a feature branch from it. Suggested branches:
- `member1/data-baseline`
- `member2/tgn-model`
- `prajwal/evaluation-context`
- `member4/dashboard`

Use small pull requests into `v1`. Review shared schemas before merging. Do not modify another member's module without coordination. A PR should state the changed behaviour, checks performed and outputs needed by the next member.

Keep raw datasets, checkpoints, run folders, credentials and large generated files out of Git. Distribute a demo bundle through an agreed location and document how to obtain it. Do not merge into `main` until the team is ready to integrate the prototype.

## 8 Local setup without Docker

Use Python 3.10/3.11 with a tested dependency file and an isolated virtual environment. Record compatible PyTorch/PyG versions. Training may use Colab/Kaggle; the dashboard reads saved outputs on a laptop.

Once package metadata and dependencies are implemented, Windows setup can use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

These are target instructions, not a claim that the current repository already runs. Verify them on another member's machine. Python module commands should be available when Make is not installed.

## 9 Demonstration and completion checklist

The demo starts with the dataset/subset and real evaluation metrics. The evaluator browses saved alerts, chooses a transaction and sees the score, frozen threshold, accounts, amount, timestamp and bounded historical graph. Patterns are presented as descriptive context. No supported motif is a valid output.

v1 is complete when:
- [ ] Team owners and shared schemas are agreed.
- [ ] Dataset/subset and chronological split are reproducible.
- [ ] XGBoost and TGN are genuinely trained and have saved held-out predictions.
- [ ] Evaluation uses validation-selected thresholds and measured test metrics.
- [ ] Alerts and context join to the correct run and transaction.
- [ ] Basic motif and causality tests pass.
- [ ] Dashboard uses real saved outputs and clearly identifies evaluation labels.
- [ ] Another member can set up and run the local demo.
- [ ] Documentation reflects what works and what remains deferred.

## 10 Limitations and references

Synthetic data, one pilot subset, one seed and descriptive graph context limit the conclusions. Legitimate transactions may form cycles or bursts. Scores and patterns assist investigation; they do not establish wrongdoing. The first demo uses precomputed outputs and is not production real-time detection.

Sources and working references:
- Team repository: https://github.com/musharrafkolhar/TGNN
- Prototype branch: https://github.com/musharrafkolhar/TGNN/tree/v1
- Dataset listing: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Prototype README and architecture updates prepared in this session; commit them before treating them as branch-authoritative.

**Prajwal's immediate next step:** implement and test evaluation on the agreed prediction schema, then connect real baseline/TGN outputs before building context and motif integration.

