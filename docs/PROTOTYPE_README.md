# TGNN v1 — temporal transaction investigation

A local Python prototype for **synthetic laundering transaction classification** using XGBoost and a PyTorch Geometric Temporal Graph Network. The Streamlit dashboard reads saved scores, metrics and SQLite alerts; it never trains models.

## Start the saved demonstration

The delivery includes genuine trained-model outputs for an IBM HI-Small chronological pilot. See `VERIFICATION.md` for measured results, exact run IDs and remaining limitations.

Use Python **3.11 or 3.12**. The complete CPU pipeline and dependency pins were exercised on Windows with Python 3.12.4.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Open the local address printed by Streamlit. After installation, `Launch-Dashboard.cmd` also launches the app. For a dashboard-only laptop, install `requirements-core.txt`; training TGN requires the full requirements.

Select a saved model run in the sidebar. Browse the queue, filter score/account/time/status, switch to Alert detail, and inspect the target transaction, directed historical graph, supporting motif edges and review status. Ground-truth labels appear only when **Show evaluation ground truth** is enabled. Displayed alerts are the top 40 test events passing that model's frozen validation threshold, not every test transaction.

## Reproduce training from your downloaded CSV

Download `HI-Small_Trans.csv` from the [IBM AML dataset listing](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data?select=HI-Small_Trans.csv). Keep the raw CSV outside Git. Supply its actual path:

```powershell
.\.venv\Scripts\python.exe scripts/run_prototype.py --config configs/ibm_prototype.yaml --input "E:\Downloads\HI-Small_Trans.csv" --model both
```

The pipeline inspects the **entire CSV in bounded chunks**, preserves original row IDs and bank/account strings, and selects a contiguous chronological prefix after the configured interval start. It sorts globally, since the source CSV is not chronological. It excludes the entire timestamp group at the event-limit boundary. It trains both models, evaluates, exports alerts/context and verifies identical held-out events before saving `runs/comparison.json`.

The delivered pilot starts at **2022-09-02 00:00 UTC**. The dataset opening period has a dense concentration of transactions; an opening-prefix inspection produced only one positive in each held-out split. The documented next-day interval was chosen before evaluating its models. The final pilot has 199,583 events, with 47/15/14 positive labels in train/validation/test. This is a small pilot, not a full-dataset experiment.

Input fields are Timestamp, From Bank, Account, To Bank, Account (the second duplicate becomes Account.1), Amount Received, Receiving Currency, Amount Paid, Payment Currency, Payment Format and Is Laundering. Missing data, invalid amounts and invalid labels fail explicitly. The parser records source SHA-256, raw schema/dtypes/missingness, full timestamp range, class counts, selected interval and split counts.

## Test with a fixture

```powershell
.\.venv\Scripts\python.exe scripts/run_prototype.py --fixture --model both
```

This creates planted synthetic test data and trains real models on it. Fixture scores are explicitly labelled and **must not be presented as IBM results**. Tests regenerate their own tiny fixtures.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Tests cover known-answer AP/F1, threshold isolation, finite prediction identities, chronological splits, train-only transformations, chunked unsorted input, retained source rows, dataset-relative seconds, context causality, temporal motifs, cross-model held-out matching, joins, idempotent SQLite imports, ground-truth visibility, incomplete outputs, actual TGN training, memory parameter learning, deterministic training, matching prediction-time attention and checkpoint replay.

The TGN module skips only when PyG is absent. A suite with that skip does not verify TGN. The delivered verification ran with PyG installed and no skips.

## Individual stages

After editable installation:

```powershell
python -m tgnn_fraud.data.preprocess --config configs/ibm_prototype.yaml
python -m tgnn_fraud.training.train_baselines --config configs/ibm_prototype.yaml
python -m tgnn_fraud.training.train_tgn --config configs/ibm_prototype.yaml
python -m tgnn_fraud.evaluation.run_eval --run runs/YOUR_RUN
python -m tgnn_fraud.explain.run_explain --run runs/YOUR_RUN --config configs/ibm_prototype.yaml
python -m tgnn_fraud.serving.build_demo_bundle --run runs/YOUR_RUN --config configs/ibm_prototype.yaml
python -m tgnn_fraud.evaluation.compare --runs runs/XGBOOST_RUN runs/TGN_RUN
```

For the individual preprocess command, first set the real input path in the YAML. Make equivalents are supplied with `CONFIG=configs/ibm_prototype.yaml` and `RUN=runs/YOUR_RUN`.

## Correctness boundaries

- All events at a timestamp are scored before any event from that timestamp is observed. Training does not shuffle; memory/neighbours reset each epoch.
- Checkpoint selection uses validation AP. Final scoring reloads that checkpoint and rebuilds chronological history using fixed weights, then continues through validation and test.
- Only an explicit allowlist of current transaction features enters the models. Labels, attempt IDs and typologies never enter features/messages/memory. Encoders and scalers fit on train only. Categorical encodings are capped at 32 columns per field, using train-defined infrequent buckets and all-zero encoding for unseen categories.
- AP means `average_precision_score`, not trapezoidal PR-AUC. Maximum validation F1 chooses the threshold; ties choose the highest threshold. The test threshold is frozen. Zero-division metrics use zero; a test split without positives reports AP as null with an explicit reason.
- Context uses strictly earlier timestamps; the current transaction is separate. Time window, hops, edge count and motif count are bounded. Cycles require increasing transaction times; fan-out counts distinct destinations. Attention is captured during the matching fixed-checkpoint replay.
- Scores are not calibrated probabilities. Attention and motifs are descriptive context, not causal model attribution or proof of wrongdoing.
- Alert uniqueness is `(run_id, model, edge_id)`. Re-importing preserves investigator review status. Processed-data hashes and canonical label/split/ID joins prevent mismatched exports.

## Scope and delivery

Implemented: data preprocessing, one XGBoost, one TGN, evaluation, historical context/cycles/fan-out, optional matching attention, saved alert database, dashboard, local demo and tests. Attention is explicitly unavailable for XGBoost. No motif is a valid result.

Deferred: pattern-file ground-truth matching, static GNN, ablations, multiple seeds, full-data experiments, GNNExplainer, live APIs/streams, authentication and Docker. GPU/Colab execution is not verified here; the notebook is a starter only.

The report assigns evaluation/context to Prajwal Vitkar. The remaining owners require team agreement; software cannot establish that agreement. An independent teammate setup check and GitHub integration are pending. No GitHub changes were pushed. Raw data/checkpoints/run folders are gitignored; the delivery ZIP distributes saved demo artifacts separately from Git.
