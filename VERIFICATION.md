# Verification — 24 October 2026 (India time)

The software pipeline is implemented and exercised locally. This is a working research prototype, **not a successful fraud-detection benchmark**: both models have zero test recall at their frozen validation-selected cutoffs on the recorded pilot.

## Actual IBM input and subset

Source: `E:\Downloads\HI-Small_Trans.csv` from the IBM AML Kaggle listing. The complete input was scanned in bounded chunks, including validation of amounts/labels/missing values.

- Raw transactions: **5,078,345**; laundering labels: **5,177**; no missing input fields.
- Source timestamp range: 2022-09-01 00:00 through 2022-09-18 16:18. Naive source dates are normalized as UTC for a consistent internal/display convention; the CSV does not specify a source timezone.
- Source SHA-256: `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040`.
- Pilot: **199,583** contiguous events from 2022-09-02 00:00 through 05:58, **157,344** bank/account nodes, **106** float32 features. The whole boundary timestamp group was excluded at the configured 200,000-event cutoff.
- Canonical edges SHA-256: `1130adb83ad18f6f11dd350e7415366b702940284a5efb7363ef97657b9a8df5`.

| Split | Transactions | Laundering labels | Time range on 2022-09-02 |
|---|---:|---:|---|
| Train | 119,707 | 47 | 00:00–03:23 |
| Validation | 39,940 | 15 | 03:24–04:41 |
| Test | 39,936 | 14 | 04:42–05:58 |

An initial opening-prefix inspection had only one positive in validation and one in test, with unusually dense opening activity. The next-day pilot interval was documented before evaluating its models. Preliminary/fixture runs are retained only in task scratch space; the delivery contains the final pilot outputs. There is one pilot seed (42), not a full-data or multi-seed result.

## Measured results

Each model selected its own threshold by maximum validation F1, choosing the highest cutoff on ties. Test labels did not choose either cutoff. AP is scikit-learn average precision from continuous scores.

| Model | Test AP | Frozen threshold | Test precision | Test recall | Test F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.0034910074 | 0.7606453300 | 0 | 0 | 0 | 0 | 166 | 14 |
| TGN | 0.0013643895 | 0.8674738407 | 0 | 0 | 0 | 0 | 17 | 14 |

The TGN was trained for three chronological epochs and selected epoch 3 by validation AP (0.0023807413). The baseline validation AP was 0.0040425929. XGBoost ranked better than TGN on this pilot. **Neither model provides useful detection at its selected cutoff in this test period.** No scores, labels or thresholds were altered to hide this result. The dashboard displays the limitation explicitly. Patterns in exported alerts describe graph structure and do not establish laundering.

Run directories (IDs use UTC timestamps):

- `runs/20261023_185426_xgboost_3e54ac`
- `runs/20261023_185430_tgn_b558e6`

Both contain **79,876 identical held-out event IDs/splits/labels**, saved predictions, metrics, resolved config, seed, environment and provenance. XGBoost has its saved model; TGN has its selected checkpoint, training history and prediction-time attention. Comparison is in `runs/comparison.json` and `runs/ibm_comparison.json`.

## Checks that passed

- **28 tests passed, zero skips**, including actual PyG training, learned memory weights, equal timestamps across chunks, deterministic fixture training, checkpoint replay, label/future invariance, train-only transformations, chunked unsorted source selection, retained original rows, known-answer AP/F1 and deterministic threshold ties.
- Schema, finite identities/scores, threshold isolation, equal held-out sets, canonical joins, future-context invariance, time-valid cycle/fan-out positives and negatives, prediction-time attention history, idempotent SQLite storage, ground-truth visibility, review status, incomplete setup handling and dashboard views passed.
- `ruff check .` and `ruff format --check .` passed.
- Required dependency versions resolved successfully with pip's dry-run installation check.
- Reloaded XGBoost and the selected TGN checkpoint reproduce saved IBM scores within absolute tolerance **1e-7**. TGN history was reset and rebuilt from the selected checkpoint, without parameter updates. See `runs/artifact_verification.json` and `runs/supported_environment_verification.json`.
- The saved transformer reproduces **every** stored feature exactly under the supported pandas pin.
- All **57 exported alerts** (40 XGBoost, 17 TGN) match saved model scores, thresholds, run/model identities, canonical transactions, source accounts, historical time bounds and standalone context JSON. Their **16 motif records** have actual supporting edges; cycle evidence is time ordered.
- Browser and Streamlit AppTest checks exercised the live queue/detail views and historical graph. The app uses saved outputs without retraining.

Test execution emitted two existing protobuf/Python deprecation warnings; they did not fail checks. These are recorded rather than suppressed.

## Environment and reproducibility

Python 3.12.4, CPU; PyTorch 2.2.2, PyG 2.6.1, NumPy 1.26.4, scikit-learn 1.6.1, XGBoost 3.0.0, PyArrow 18.1.0, NetworkX 3.5, Streamlit 1.32.0, Plotly 5.18.0, PyYAML 6.0.2, pytest 8.4.1 and Ruff 0.12.2.

The initial IBM training used the existing system pandas 3.0.6, recorded honestly in each run's environment. Streamlit 1.32.0 declares pandas <3, so the **installation files pin pandas 2.2.0**. The full 28-test suite, saved transformer reproduction and IBM model reload/replay verification were also run with pandas 2.2.0. PyG and the supported pandas were staged in a task-local dependency directory rather than changing the user's global environment. A fresh team setup should use the documented virtual environment and requirements.

To independently recheck artifacts after installation:

```powershell
python scripts/verify_artifacts.py --data data/ibm_pilot_processed --runs runs/20261023_185426_xgboost_3e54ac runs/20261023_185430_tgn_b558e6 --source "E:\Downloads\HI-Small_Trans.csv"
```

The ZIP includes source/config/tests/docs, canonical pilot data and fitted transformer, both trained-model artifacts, real predictions/metrics/attention/context, portable demo bundles and checksums. It excludes the original 475 MB raw CSV, temporary dependencies, virtual environments and caches. Raw data/checkpoints/runs remain ignored by Git even though saved artifacts are distributed in the local delivery ZIP.

## Remaining limits and human milestones

- Only 14 positive test examples, one seed and one short synthetic interval: no claim of real-world fraud performance, temporal superiority or stable research conclusions.
- Model quality needs further research. Do not tune repeatedly against this held-out test set; define a new validation protocol for future experiments.
- Pattern-file matching, typology metrics, static GNN, ablations, multiple seeds, full-scale runs, GNNExplainer, GPU/Colab execution and production features remain deferred.
- Team owners beyond Prajwal's documented workstream have not been agreed by the team. Independent setup on another member's machine remains pending.
- This is a local delivery. No GitHub commit, PR, merge or deployment was made.
