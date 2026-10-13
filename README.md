# TGNN-Fraud: Temporal Graph Neural Networks for Coordinated Financial Fraud Detection

> **BE Final Year Project** · JSPM's Jayawantrao Sawant College of Engineering, Hadapsar, Pune
> Affiliated to Savitribai Phule Pune University
> **Guide:** Prof. Ashwini Patil
> **Team:** Musharraf Kolhar · Prajwal Vitkar · Priyanshu Pande · Aaryan Karandikar

> **Status: planning / pre-implementation.** Sections marked **TBD** are filled in as the project is built.
> Do not treat any number in this file as a result until the "Results" section says so.

---

## 1. Overview

Financial fraud and money laundering are rarely one-off events. Rings of accounts move money together through
chains, cycles, fan-outs and bursts. A model that scores each transaction in isolation cannot see this, because it
ignores **who transacts with whom, and when**.

This project models transactions as a **continuous-time dynamic graph** (accounts are nodes, time-stamped transactions
are edges) and uses a **Temporal Graph Network (TGN)** to classify each transaction as suspicious or normal.
Every alert comes with an **explanation**: the temporal subgraph, the attention weights, and the detected laundering
pattern (for example "4-account circular flow within 6 hours") that drove the score. An **investigator dashboard**
(Streamlit) lets an analyst browse alerts and inspect them.

## 2. Research questions

| ID | Question | How we answer it |
|----|----------|------------------|
| RQ1 | Does a TGN detect illicit transactions better than per-transaction and graph-feature baselines? | PR-AUC, minority-class F1, recall@k vs. XGBoost baselines |
| RQ2 | Does temporal modelling help beyond a static GNN? | Ablations: static GNN, TGN without memory, TGN without time encoding |
| RQ3 | Which laundering typologies benefit most from temporal graph modelling? | Per-typology recall (cycle, fan-in, fan-out, scatter-gather, ...) |
| RQ4 | Do explanations point at the true laundering accounts and transactions? | Overlap between explained subgraph and ground-truth pattern edges |

## 3. Scope

### In scope
- Offline training and evaluation on a **public dataset** (IBM AML synthetic transactions, see section 5).
- Accounts as nodes; bank, currency and payment format as features.
- Transaction-level (edge) classification with a TGN.
- Baselines (XGBoost, XGBoost + graph features, static GNN) and TGN ablations.
- Explainability: attention weights, GNNExplainer (if feasible), motif detection.
- Simulated streaming by **chronological replay** of the test period.
- Streamlit investigator dashboard.

### Out of scope
- Real bank data and production-grade real-time serving.
- Devices, merchants, customers and payment instruments as node types. No public dataset provides them.
- Geographic or "location" features. They do not exist in the chosen dataset.
- Online learning / continual updating of the model.

### Priority tiers (cut line if time runs short)

| Tier | Items |
|------|-------|
| **Must** | Preprocessing, XGBoost baselines (raw + graph features), TGN with edge-classification head, evaluation suite, attention + motif explanations, dashboard (alert queue + alert detail) |
| **Should** | Static GNN baseline, TGN ablations, GNNExplainer, replay simulator page, LI-Small generalisation run, Docker image |
| **Stretch** | TGAT comparison, Elliptic Bitcoin dataset check, FastAPI inference endpoint |

## 4. Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.10 / 3.11 |
| Deep learning | PyTorch |
| Graph / temporal GNN | PyTorch Geometric (TGN: `TGNMemory`, `LastNeighborLoader`, `TransformerConv`) |
| Baselines | scikit-learn, XGBoost |
| Data | Pandas, NumPy, Parquet |
| Graph analysis | NetworkX |
| Explainability | Attention weights, `torch_geometric.explain.GNNExplainer`, NetworkX motif detection, SHAP (XGBoost only) |
| Dashboard | Streamlit, PyVis (graphs), Plotly (charts) |
| Alert storage | SQLite |
| Experiment tracking | TensorBoard (or Weights & Biases) |
| Packaging | Docker (dashboard / inference demo), Git + GitHub |
| Training compute | Google Colab / Kaggle GPU |

**Deliberately not used:** Neo4j, PostgreSQL, FastAPI (stretch only). See `ARCHITECTURE.md`, decision log.

## 5. Dataset

**Primary:** IBM *Transactions for Anti-Money Laundering (AML)*, the synthetic AMLworld data from
Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023
(Datasets & Benchmarks). Use the **HI-Small** version first (about 5M transactions, roughly 0.1% illicit; verify after
download), then **LI-Small** for generalisation. Do not use the Medium/Large versions; they are too heavy for free GPUs.

Why this dataset: it has timestamps, account-to-account edges, illicit labels, and a patterns file that labels each
laundering attempt by typology (cycle, fan-in, fan-out, scatter-gather, etc.), which lets us answer RQ3 and RQ4.

**Download (Kaggle):** search Kaggle for "IBM Transactions for Anti-Money Laundering (AML)". Place the files in
`data/raw/`. Verify the exact file names and columns after download; `ARCHITECTURE.md` section 4 documents the
assumed schema.

> The dataset is **synthetic**. Say so in the report. It limits how far conclusions transfer to real banks.

## 6. Repository structure

```
tgnn-fraud-detection/
├── README.md
├── ARCHITECTURE.md          # system design, module contracts, decision log
├── AGENTS.md                # instructions for AI coding assistants
├── CLAUDE.md                # imports AGENTS.md (Claude Code)
├── pyproject.toml
├── requirements.txt
├── Makefile
├── Dockerfile
├── configs/                 # YAML configs (data, baselines, tgn, explain)
├── data/                    # gitignored: raw/, interim/, processed/, demo/
├── src/tgnn_fraud/
│   ├── data/                # download check, preprocess, splits, features
│   ├── models/              # baselines, static GNN, TGN, edge classifier
│   ├── training/            # training scripts, losses
│   ├── evaluation/          # metrics, typology recall, explanation eval
│   ├── explain/             # attention, GNNExplainer, motifs, report
│   ├── serving/             # stream replay, scorer, alert store
│   └── utils/               # config, seeding, logging
├── app/                     # Streamlit dashboard (multipage)
├── notebooks/               # EDA, Colab training notebook
├── tests/
├── runs/                    # gitignored: one folder per experiment
├── reports/                 # figures and tables for the thesis
└── docs/images/             # screenshots
```

## 7. Installation

> Commands below describe the **target** workflow. They start working as each module is implemented.

### 7.1 Local (CPU is fine for development on a small subset)

```bash
git clone <REPO_URL> tgnn-fraud-detection
cd tgnn-fraud-detection
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 1) Install PyTorch for your platform first: https://pytorch.org/get-started/locally/
# 2) Then:
pip install -r requirements.txt
pip install -e .
```

If a PyG import fails with a message mentioning `torch-scatter` or `pyg-lib`, install the wheels matching your
exact PyTorch/CUDA version from the PyG installation guide. Pin the working versions in `requirements.txt`
once the team's environment is stable.

### 7.2 Google Colab / Kaggle (training)

Open `notebooks/colab_train.ipynb`. It clones the repo, installs dependencies, mounts storage for data and
checkpoints, and runs `make train-tgn`.

### 7.3 Docker (dashboard / demo only)

```bash
docker build -t tgnn-fraud .
docker run -p 8501:8501 tgnn-fraud      # then open http://localhost:8501
```

The image runs the dashboard on CPU using a small **demo bundle** (`data/demo/`: a slice of the test period,
a trained checkpoint, precomputed alerts and explanations). It does not train models.

## 8. Usage

All steps are driven by YAML configs in `configs/` and wrapped by `make` targets.

```bash
make data              # validate raw files, build data/processed/edges.parquet + splits.json
make train-baselines   # XGBoost (raw) and XGBoost (+ graph features), static GNN
make train-tgn         # TGN + edge classifier -> runs/<run_id>/
make evaluate RUN=<run_id>   # metrics, PR curves, per-typology recall -> reports/
make explain  RUN=<run_id>   # explanations for top-K alerts -> alerts.db
make demo-bundle RUN=<run_id>   # package small demo slice for Docker
make app               # streamlit run app/Home.py
make test              # pytest
```

Every model writes the same `predictions.parquet` schema (see `ARCHITECTURE.md` section 7), so evaluation,
explanation and the dashboard work with any model.

## 9. Evaluation protocol

- **Chronological split** (60% train / 20% val / 20% test by timestamp). Never a random split.
- Class imbalance handled with weighted loss (or focal loss); scalers fit on train only.
- Primary metric: **PR-AUC**. Also: F1 on the illicit class, precision@k / recall@k (alert budget),
  per-typology recall, ROC-AUC (secondary; misleading under heavy imbalance), inference latency per batch.
- Decision threshold chosen on the validation set, then frozen for test.
- Every reported number comes from a saved run folder with its config and seed. At least 3 seeds for the final
  comparison.

## 10. Results

**TBD.** Filled in after experiments. Leave this table empty until real runs exist.

### 10.1 Model comparison (HI-Small, test split)

| Model | PR-AUC | F1 (illicit) | Recall@0.5% | Precision@0.5% |
|-------|--------|--------------|-------------|----------------|
| XGBoost (raw tx features) | TBD | TBD | TBD | TBD |
| XGBoost (+ graph features) | TBD | TBD | TBD | TBD |
| Static GNN | TBD | TBD | TBD | TBD |
| TGN, no memory | TBD | TBD | TBD | TBD |
| TGN, no time encoding | TBD | TBD | TBD | TBD |
| **TGN (full)** | TBD | TBD | TBD | TBD |

### 10.2 Recall by laundering typology

| Typology | XGBoost (+ graph) | TGN |
|----------|-------------------|-----|
| Cycle | TBD | TBD |
| Fan-out | TBD | TBD |
| Fan-in | TBD | TBD |
| Scatter-gather | TBD | TBD |
| Other | TBD | TBD |

### 10.3 Explanation quality (RQ4)

| Method | Edge precision@k vs. ground truth | Edge recall@k |
|--------|-----------------------------------|---------------|
| Attention | TBD | TBD |
| GNNExplainer | TBD | TBD |
| Attention + motif detection | TBD | TBD |

## 11. Screenshots

**TBD.** Add after the dashboard exists. Planned images in `docs/images/`:

| File | Content |
|------|---------|
| `overview.png` | Overview page: model comparison and metrics |
| `alert_queue.png` | Ranked alert queue with filters |
| `alert_detail.png` | Subgraph view, timeline, explanation text |
| `replay.png` | Stream replay simulator |

## 12. Roadmap

- [ ] **M0**: Repo, environment, CI-free sanity checks, tiny synthetic dataset for tests
- [ ] **M1**: Preprocessing, chronological split, XGBoost baselines, evaluation suite
- [ ] **M2**: TGN with edge classifier trained on HI-Small; first PR-AUC
- [ ] **M3**: Static GNN + TGN ablations; multi-seed runs
- [ ] **M4**: Explainability (attention, motifs, GNNExplainer feasibility); explanation evaluation
- [ ] **M5**: Replay + alert store + Streamlit dashboard
- [ ] **M6**: Final experiments, Docker demo bundle, report and presentation

Adjust timing to your college schedule. Acceptance criteria per milestone are in `ARCHITECTURE.md` section 16.

## 13. Team and ownership

| Workstream | Owner | Main modules |
|------------|-------|--------------|
| Data and baselines | TBD | `data/`, `models/baselines.py`, `evaluation/` |
| TGN model | TBD | `models/tgn.py`, `training/` |
| Explainability | TBD | `explain/`, `evaluation/explanation_eval.py` |
| Dashboard and packaging | TBD | `app/`, `serving/`, `Dockerfile` |

## 14. Known limitations

- Synthetic data; real-world transfer is untested.
- Streaming is simulated by replay, not a production pipeline.
- Explainability for TGN is approximate. GNNExplainer is not designed for memory-based temporal models, so
  attention + motif detection is the primary explanation method.
- Accounts are the only node type.

## 15. References

1. TMR-GGNN: Credit Card Fraud Detection Based on Time-Aware Multi-Relational Guided Graph Neural Network (2026)
2. Self-Adaptive Graph Analytics for Real-Time Fraud Detection in Transaction Networks (2026)
3. Multi-Temporal Partitioned Graph Attention Networks for Financial Fraud Detection (2025)
4. Catching Scam Tokens With Temporal Graph Learning in Decentralized Finance (2026)
5. Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs*, 2020 (TGN)
6. Xu et al., *Inductive Representation Learning on Temporal Graphs*, ICLR 2020 (TGAT)
7. Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023
8. Ying et al., *GNNExplainer: Generating Explanations for Graph Neural Networks*, NeurIPS 2019
9. Xia et al., *Explaining Temporal Graph Models through an Explorer-Navigator Framework*, ICLR 2023 (T-GNNExplainer)
10. Chen and Ying, *TempME: Towards the Explainability of Temporal Graph Neural Networks via Motif Discovery*, NeurIPS 2023

> Add authors and links for references 1 to 4 from the review presentation.

## 16. License

**TBD.** Choose before making the repository public.
