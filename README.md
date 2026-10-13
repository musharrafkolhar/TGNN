# TGNN-Fraud: Coordinated Financial Fraud Detection

> **BE final-year project prototype (v1)** · JSPM's Jayawantrao Sawant College of Engineering, Hadapsar, Pune · SPPU
>
> **Guide:** Prof. Ashwini Patil  
> **Team:** Musharraf Kolhar · Prajwal Vitkar · Priyanshu Pande · Aaryan Karandikar
>
> **Current status: planning / pre-implementation.** The repository contains design documents and a placeholder Python file. The modules, commands and folder layout below are planned, not working features. No trained model or measured results are available yet.

## 1. Prototype objective

Build a working path from timestamped account-to-account transactions to TGN risk scores, saved alerts and an investigator dashboard. An evaluator should be able to select an alert and inspect the transaction, its historical graph context and detected patterns.

The first prototype uses IBM's **synthetic anti-money-laundering dataset**. It classifies transactions labelled as laundering; it does not establish criminal intent or cover every kind of financial fraud. Accounts are nodes and transactions are timestamped directed edges.

TGN learns from account history and recent neighbours. Pattern detection adds descriptive context such as cycles or fan-outs. A pattern is not proof of fraud, and attention weights are not a verified causal explanation of a model decision.

## 2. v1 scope and completion criteria

| Component | Required for v1 | Complete when |
|---|---|---|
| Environment | Python 3.10/3.11, PyTorch/PyG, core data and dashboard dependencies | A documented setup can import dependencies and run smoke checks |
| Data | Verified HI-Small input; reproducible contiguous chronological subset | Clean edges, node index, split manifest and data validation report exist |
| Baseline | One XGBoost model using transaction features | Saved validation/test predictions and measured metrics exist |
| Main model | One TGN with an edge-classification head | Training completes; checkpoint and held-out predictions are saved |
| Evaluation | Average precision (reported as AP), precision, recall and illicit-class F1 | Both models use identical splits; threshold is selected on validation and frozen for test |
| Alert context | Historical subgraph, basic cycle and fan-out detection; attention where supported | A selected alert displays actual graph facts; missing motifs/attention are shown honestly |
| Dashboard | Alert queue and alert detail in Streamlit | Selecting an alert opens its matching transaction, score and graph |
| Demo | Saved model outputs on a documented test slice | Demo launches locally with clear labels and no fabricated scores |
| Verification | Split, leakage, model-state ordering, schema and basic end-to-end checks | Checks pass and setup can be repeated by another member |

**v1 does not need to outperform XGBoost to be a working prototype.** Report the observed result, including poor performance. A rule-only dashboard or mock scores do not satisfy the TGN requirement.

### Deferred final-version work

- XGBoost with causal historical graph features.
- Static GNN, no-memory/no-time ablations and multiple seeds.
- Full HI-Small experiments, LI-Small generalisation and detailed typology analysis.
- GNNExplainer, SHAP and quantitative explanation evaluation.
- Live inference replay controls, additional dashboard pages and Docker packaging.
- TGAT, another dataset or FastAPI only if separately agreed.

These remain research goals, not prerequisites for the first demo. No devices, merchants, geographic features, real bank integration, online learning or production serving are included in v1.

## 3. Implementation status

| Feature | Status | Evidence |
|---|---|---|
| Scope and architecture documentation | Documented | README.md, ARCHITECTURE.md |
| Python entry point | Placeholder | tgnn/main.py contains a comment only |
| Environment and package skeleton | Planned | Not implemented |
| Data validation and preprocessing | Planned | Not implemented |
| XGBoost baseline | Planned | No saved run |
| TGN training and scoring | Planned | No checkpoint |
| Metrics and alert context | Planned | No predictions or explanations |
| Dashboard and demo | Planned | No runnable app |

Update a row to **implemented** when code exists and **verified** only after its acceptance check passes. Record the verifying command and commit/run ID.

## 4. Dataset and subset policy

Primary source: [IBM Transactions for Anti-Money Laundering (AML), Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

Download HI-Small transactions and its accompanying patterns file into `data/raw/`. **Exact filenames, schema, row counts, time span and class balance must be recorded after inspecting the downloaded files.** The schema in ARCHITECTURE.md is an assumption until verified. Do not store Kaggle credentials in Git.

Prototype subset procedure:

1. Sort transactions by timestamp and stable original row order.
2. Choose one contiguous time interval for a manageable pilot run. Record its boundaries and selection rule in config; do not choose it based on test model performance.
3. Keep all normal and illicit events in that interval. Do not randomly sample or oversample the transaction stream.
4. Split chronologically approximately 60/20/20, keeping equal timestamps together.
5. Report total events, unique accounts, positive counts and prevalence separately for train, validation and test. If a split has no positives, report the pilot as unsuitable for minority-class evaluation and revise the documented interval before modelling.
6. Save the source file checksum, subset boundaries, split boundaries and seed/config with the run.

The subset is for feasibility testing; results describe that subset, not the full dataset or real banks. If the patterns file is unavailable or cannot be matched, basic graph-pattern displays can still work, but ground-truth typology metrics remain unavailable.

## 5. Stack

| Layer | v1 choice |
|---|---|
| Model | Python, PyTorch, PyTorch Geometric TGN components |
| Baseline and metrics | XGBoost, scikit-learn |
| Data | Pandas, NumPy, Parquet |
| Graph context | NetworkX |
| Dashboard | Streamlit, PyVis, Plotly |
| Alerts | SQLite |
| Configuration and checks | YAML configuration, pytest, ruff |

Pin a tested environment once verified. GPU training can run on Colab/Kaggle; the first dashboard reads saved outputs on CPU. Neo4j, PostgreSQL, Kafka, LLMs and an API are unnecessary for v1.

## 6. Planned interfaces and layout

The actual repository currently has three design documents and `tgnn/main.py`. The following are **target paths to create**, not existing modules:

| Planned path | Responsibility |
|---|---|
| `configs/prototype.yaml` | Subset, paths, split, model and demo settings |
| `src/tgnn_fraud/data/` | Validation, preprocessing, splits, encoders, event stream |
| `src/tgnn_fraud/models/` | XGBoost, TGN and edge classifier |
| `src/tgnn_fraud/training/` | Training and saved run outputs |
| `src/tgnn_fraud/evaluation/` | File-based evaluation and threshold selection |
| `src/tgnn_fraud/explain/` | Historical context, attention and basic motifs |
| `src/tgnn_fraud/serving/` | Checkpoint scorer, alert export and SQLite |
| `src/tgnn_fraud/utils/` | Config, seeding and logging |
| `app/` | Streamlit alert queue and detail |
| `tests/` | Tiny-data checks |
| `data/`, `runs/`, `reports/` | Generated data, checkpoints and metrics; not committed |

Shared outputs:

- Data: `edges.parquet`, `node_index.parquet`, `splits.json`, `preprocess_report.json`.
- Models: `predictions.parquet`, checkpoint/model file, config and seed.
- Evaluation: `metrics.json`, selected threshold and PR curve.
- Alert export: `alerts.db` with explanation/context JSON.

Detailed schemas are in ARCHITECTURE.md. All models use the same prediction schema.

## 7. Setup and commands

### Available now: clone the prototype branch

```powershell
git clone --branch v1 https://github.com/musharrafkolhar/TGNN.git
cd TGNN
```

### Planned Windows environment setup

These steps require `requirements.txt` and package metadata to be implemented first:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Install a compatible PyTorch/PyG build using their official installation guides and record the verified versions. Do not assume CUDA is available.

### Planned pipeline (not yet runnable)

| Step | Planned Make target | Planned Python equivalent for Windows without Make |
|---|---|---|
| Prepare data | `make data` | `python -m tgnn_fraud.data.preprocess --config configs/prototype.yaml` |
| Baseline | `make train-baselines` | `python -m tgnn_fraud.training.train_baselines --config configs/prototype.yaml` |
| TGN | `make train-tgn` | `python -m tgnn_fraud.training.train_tgn --config configs/prototype.yaml` |
| Evaluate | `make evaluate RUN=<run_id>` | `python -m tgnn_fraud.evaluation.run_eval --run runs/<run_id>` |
| Export context and alerts | `make explain RUN=<run_id>` | `python -m tgnn_fraud.explain.run_explain --run runs/<run_id>` |
| Dashboard | `make app` | `python -m streamlit run app/Home.py` |
| Tests | `make test` | `python -m pytest` |
| Lint | `make lint` | `python -m ruff check .` and `python -m ruff format --check .` |

Implement and verify CLI arguments and Make targets together. Use the venv Python executable if the environment is not activated. A future demo bundle must document where to obtain its data and saved outputs.

## 8. ML correctness and evaluation

- No random splits. Equal-timestamp events stay in one split.
- Fit scalers, categorical encoders and loss class weights on train only.
- Labels, attempt IDs and typology annotations are never model features or memory messages.
- Predict before updating memory/neighbour state. No shuffling.
- For final held-out evaluation, load the selected checkpoint, reset state and rebuild historical state chronologically using that checkpoint. Continue from train into validation into test without further resets or parameter updates.
- Define equal-timestamp handling so events at time `t` cannot see other events at `t` if the policy is strictly-before history.
- Use one recorded seed for v1. Multiple seeds belong to the final comparison.
- Report **average precision (AP)** using scikit-learn average_precision_score and name it explicitly; it is not identical to trapezoidal PR-AUC.
- Select a decision threshold on validation, freeze it for test, and save it with metrics. ROC-AUC is optional; accuracy is not a useful headline.
- A sigmoid model score is not automatically a calibrated fraud probability.

## 9. Prototype demonstration

1. Launch the dashboard and show the dataset subset, run ID and evaluation metrics.
2. Browse the saved alert queue ranked by the selected model's score.
3. Select an alert to view its timestamp, accounts, amount, score and threshold.
4. Inspect a bounded historical subgraph and any detected cycle/fan-out.
5. View ground truth only in clearly labelled evaluation mode.

The first demo displays **precomputed model outputs**. It does not perform live inference or retrain when opened. Attention is included only if extracted from the matching prediction-time state. Display “no supported motif found” when appropriate; never invent an explanation. Graph context uses no future events.

## 10. Prototype build order

| Milestone | Deliverable | Acceptance |
|---|---|---|
| P0 | Package skeleton, config, environment, tiny test data | Imports and smoke checks pass |
| P1 | Subset pipeline, splits, XGBoost and metrics | Reproducible data plus a measured baseline run |
| P2 | TGN training and checkpoint evaluation | Genuine held-out scores; ordering/leakage checks pass |
| P3 | Alert export, historical context and basic motifs | Correct transaction-to-alert joins and context |
| P4 | Queue/detail dashboard and local demo | Another teammate can launch and inspect saved alerts |

UI scaffolding with labelled fixtures can proceed in parallel. Connect it to real outputs after P2/P3; final research comparisons do not block the prototype.

## 11. Four-member ownership (assign names before implementation)

| Workstream | Owner | Boundary |
|---|---|---|
| Data and baseline | Unassigned | Data artifacts, subset manifest and XGBoost |
| TGN and scorer | Unassigned | Model training, checkpoint, memory lifecycle and scorer |
| Evaluation and context | Unassigned | Shared metrics, thresholding, motifs, attention and context JSON |
| Dashboard and integration | Unassigned | SQLite storage, alert export integration and Streamlit |

Do not infer ownership from the order of team names. Agree schemas first, use separate branches and small pull requests into `v1` for prototype integration. The model owner controls scoring state; the dashboard owner consumes saved outputs.

## 12. Results and limitations

**No results yet.** Add only measured values linked to a saved run.

| Model | Subset/run ID | AP | Precision | Recall | F1 (illicit) |
|---|---|---|---|---|---|
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| TGN | TBD | TBD | TBD | TBD | TBD |

Synthetic data, a small pilot subset, one seed and descriptive graph context limit the conclusions. Suspicious patterns may occur in legitimate transfers. The prototype assists inspection; it does not verify wrongdoing.

## 13. Final research direction and references

The final version will investigate improvements over graph-feature and static-GNN baselines, temporal ablations, per-pattern detection and explanation quality. These questions are deferred until the prototype works.

Core references to complete with verified bibliographic links:
- Rossi et al., Temporal Graph Networks for Deep Learning on Dynamic Graphs (2020).
- Altman et al., Realistic Synthetic Financial Transactions for Anti-Money Laundering Models (NeurIPS 2023).
- Temporal explanation references from the literature review require verification before use.

License: **TBD**; the team must choose a code license and respect the dataset's separate terms.

