# AGENTS.md

Instructions for AI coding assistants (Claude Code, Codex, Cursor, Copilot, etc.) working on **TGNN-Fraud**.
Humans: read `README.md` first. This file is the working agreement for AI-assisted changes.

## Project in one paragraph

BE final-year project: detect coordinated financial fraud by classifying transactions with a **Temporal Graph
Network (TGN)** on the IBM AML synthetic dataset, explain alerts (attention + motif detection), and present them in a
Streamlit dashboard. Models are compared against XGBoost baselines and a static GNN. The active delivery is the v1 prototype; final research questions are deferred in `README.md`.

## Active scope: v1 prototype

Use README.md prototype milestones P0-P4 and ARCHITECTURE.md section 0. Preserve the ML safeguards, but do not require final-research models, ablations, multi-seed runs, GNNExplainer, replay controls or Docker before completing the prototype. One real TGN and one XGBoost baseline are required. No fake scores or results.

The default demo reads saved outputs. Attention is optional when unavailable and must never be fabricated. Motifs are descriptive context, not proof of wrongdoing or causal explanations.

## Read before coding

1. `ARCHITECTURE.md`, especially sections 4 (data contracts), 6 (models), 7 (prediction schema) and 13 (dependency rules).
2. The module you are changing and its tests.

If a task conflicts with `ARCHITECTURE.md`, stop and say so. Do not silently diverge. If you change a contract,
update `ARCHITECTURE.md` in the same change.

## Stack (fixed)

Python 3.10/3.11 · PyTorch · PyTorch Geometric · scikit-learn · XGBoost · Pandas/NumPy/Parquet · NetworkX ·
Streamlit + PyVis + Plotly · SQLite · TensorBoard · Docker.

**Do not add** Neo4j, PostgreSQL, Spark, Kafka, FastAPI, LangChain or any LLM dependency unless the user explicitly
asks. FastAPI is a stretch item only. Ask before adding any new dependency, and state why.

## Commands

```bash
make setup              # create venv, install requirements, pip install -e .
make test               # pytest (must pass before you finish)
make lint               # ruff check + ruff format --check
make data               # preprocess raw data -> data/processed/
make train-baselines
make train-tgn
make evaluate RUN=<run_id>
make explain  RUN=<run_id>
make app                # streamlit run app/Home.py
```

If a target does not exist yet, create it in the `Makefile` as part of your change instead of inventing one-off commands.

## Non-negotiable ML rules

These prevent the most likely silent bugs. Violating them invalidates results.

1. **Chronological splits only.** Never use `train_test_split` with shuffling on transactions. Use `splits.json`.
2. **No future leakage.** Features, scalers, encoders and class weights are computed from **train** data only.
   Any windowed feature for an edge at time `t` uses events strictly before `t`.
3. **TGN order of operations:** predict, compute loss, then update memory and neighbour loader. No shuffling.
   Reset memory at the start of each training epoch. After loading the selected checkpoint for held-out evaluation, reset and rebuild historical state using that checkpoint, then continue across train, val and test boundaries without further resets. Handle equal timestamps without same-time leakage.
4. **Labels never enter model inputs**, messages, memory or features.
5. **Imbalance:** report average precision (AP) as the prototype's primary metric; distinguish it from trapezoidal PR-AUC. Do not report accuracy. Do not claim a result from ROC-AUC alone.
6. **Every result comes from a saved run folder** (`runs/<id>/` with config, seed, predictions, metrics). Never hardcode
   or invent numbers in code, docs, notebooks or README tables. Leave `TBD` until a real run exists.
7. **Set seeds** via `utils/seed.py`. Record library versions in the run config.
8. **Do not assume dataset columns.** Open the file, check the schema, and update `ARCHITECTURE.md` section 4.1 if it differs.

## Architecture rules

- Stages communicate through files (`edges.parquet`, `predictions.parquet`, `alerts.db`), not by importing each
  other's internals.
- Every model writes the same `predictions.parquet` schema (`ARCHITECTURE.md` 7.3).
- `evaluation/` depends only on parquet files, never on model code.
- `app/` must not import from `models/` or `training/`. It reads files and SQLite only.
- `utils/` imports nothing else from the project. No circular imports.
- Ablations are config flags, not copied code.
- Explanations must come from the model and graph (attention, motifs, GNNExplainer). Never generate explanations with
  an LLM or hand-written guesses.

## Code conventions

- Type hints on all public functions; docstrings state shapes for tensors (`[N, F]`) and units for time (seconds).
- Format and lint with `ruff`. Keep functions small and pure where possible.
- Config via YAML in `configs/`; no magic numbers, paths or hyperparameters in code.
- Use `pathlib.Path`, not string paths. Use `logging`, not `print`, outside scripts and the app.
- Data files, checkpoints and `runs/` are gitignored. Never commit them. Never commit credentials or Kaggle tokens.
- Keep notebooks for exploration only. Reusable logic belongs in `src/`.

## Testing expectations

- Add or update tests with every change. Use the tiny synthetic dataset (about 2k edges with a planted cycle and
  fan-out) so tests run in seconds on CPU.
- Required test types: split ordering, leakage, memory-order, determinism, prediction schema, motif detection,
  explanation JSON schema (`ARCHITECTURE.md` section 14).
- For implementation changes, run `make test` and `make lint` (or the documented Python equivalents) before declaring work done. Documentation-only scope changes require consistency and diff checks, not nonexistent test targets. If required checks could not run, say so.

## Workflow for a task

1. Restate the task and the files you will touch. For anything spanning more than one module, outline a short plan first.
2. Make the smallest change that satisfies the task. Do not refactor unrelated code or rename public interfaces.
3. Write or update tests, then implement.
4. Run tests and lint. Fix failures instead of weakening tests.
5. Update docs that your change affects (`README.md`, `ARCHITECTURE.md`, config comments).
6. Summarise: what changed, what you verified, what remains uncertain.

## Scope discipline

Follow the required-v1 and deferred-final scope in `README.md`. Finish prototype milestones before adding deferred items. Do not add features, pages, models or datasets that are not in scope. Suggest them instead.

Out of scope: real bank data, devices/merchants/customers as node types, geographic features, online learning,
production real-time serving.

## Honesty requirements

- The dataset is **synthetic**; never describe results as real-world fraud performance.
- Streaming is a **replay simulation**; never describe it as production real-time detection.
- If something fails, is slow, or you are unsure (for example GNNExplainer on TGN), say so plainly and propose a
  fallback. Do not hide errors with broad `try/except` or by silently skipping steps.
- Do not fabricate citations, benchmark numbers, library APIs or dataset facts. If unsure of a PyG API, check the
  installed version's source or docs before using it.

## Definition of done

- [ ] Code follows the dependency and ML rules above
- [ ] Tests added or updated; `make test` and `make lint` pass
- [ ] No new unapproved dependencies; no committed data, checkpoints or secrets
- [ ] Docs updated where behaviour or contracts changed
- [ ] Any reported metric is traceable to a run folder

