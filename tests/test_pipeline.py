"""Known-answer correctness, causality and real baseline integration checks."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tgnn_fraud.data.features import feature_frame, fit_features
from tgnn_fraud.data.preprocess import canonicalize, preprocess
from tgnn_fraud.data.synthetic import make_fixture
from tgnn_fraud.evaluation.metrics import evaluate_predictions, select_threshold
from tgnn_fraud.evaluation.run_eval import run_evaluation
from tgnn_fraud.explain.context import historical_context
from tgnn_fraud.explain.motifs import detect_motifs
from tgnn_fraud.explain.report import build_context
from tgnn_fraud.explain.run_explain import export_alerts
from tgnn_fraud.serving.alert_store import (
    initialize,
    read_alerts,
    read_context,
    save_alert,
    update_status,
)
from tgnn_fraud.serving.build_demo_bundle import build_bundle
from tgnn_fraud.training.train_baselines import train


@pytest.fixture
def config(tmp_path):
    with open("configs/prototype.yaml", encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    result["data"]["input"] = str(tmp_path / "fixture.csv")
    result["data"]["output"] = str(tmp_path / "processed")
    result["runs_dir"] = str(tmp_path / "runs")
    result["baseline"]["n_estimators"] = 10
    return result


def predictions():
    return pd.DataFrame(
        {
            "edge_id": np.arange(8),
            "split": ["val"] * 4 + ["test"] * 4,
            "score": [0.1, 0.4, 0.35, 0.8, 0.1, 0.4, 0.35, 0.8],
            "label": [0, 0, 1, 1] * 2,
            "model": "test",
            "run_id": "fixture",
        }
    )


def test_metric_known_answers():
    result = evaluate_predictions(predictions())
    assert result["threshold"] == pytest.approx(0.35)
    assert result["test"]["average_precision"] == pytest.approx(5 / 6)
    assert result["test"]["precision"] == pytest.approx(2 / 3)
    assert result["test"]["recall"] == 1
    assert result["test"]["f1"] == pytest.approx(0.8)
    assert result["test"]["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}


def test_test_labels_do_not_select_threshold():
    frame = predictions()
    before = evaluate_predictions(frame)["threshold"]
    frame.loc[frame.split == "test", "label"] = [1, 1, 0, 0]
    assert evaluate_predictions(frame)["threshold"] == before


@pytest.mark.parametrize("column,value", [("score", np.nan), ("score", 1.2), ("label", 2)])
def test_reject_bad_predictions(column, value):
    frame = predictions()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate_predictions(frame)


def test_reject_ambiguous_and_empty_splits():
    frame = predictions()
    frame.loc[0, "edge_id"] = 1
    with pytest.raises(ValueError):
        evaluate_predictions(frame)
    frame = predictions()
    frame.loc[frame.split == "val", "label"] = 0
    with pytest.raises(ValueError, match="both"):
        evaluate_predictions(frame)
    with pytest.raises(ValueError):
        select_threshold(np.zeros(3), np.ones(3))


def test_timestamp_ties_and_train_only_features(config):
    raw = make_fixture()
    raw.loc[1199:1202, "timestamp"] = raw.loc[1199, "timestamp"]
    edges, _, splits = canonicalize(raw, config["data"])
    assert splits["train"]["max_ts"] < splits["val"]["min_ts"]
    assert splits["val"]["max_ts"] < splits["test"]["min_ts"]
    _, before = fit_features(edges)
    changed = edges.copy()
    changed.loc[changed.split == "test", "amount_paid"] = 1e12
    changed.loc[changed.split == "test", "payment_format"] = "future_only"
    _, after = fit_features(changed)
    np.testing.assert_allclose(before[edges.split == "train"], after[edges.split == "train"])
    changed.label = 1 - changed.label
    pd.testing.assert_frame_equal(
        feature_frame(edges),
        feature_frame(
            changed.assign(amount_paid=edges.amount_paid, payment_format=edges.payment_format)
        ),
    )


def test_reject_invalid_schema_amounts(config):
    raw = make_fixture()
    with pytest.raises(ValueError, match="Missing columns"):
        canonicalize(raw.drop(columns="timestamp"), config["data"])
    raw.loc[0, "amount_paid"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        canonicalize(raw, config["data"])


def test_context_no_future_or_same_time(config):
    edges, _, _ = canonicalize(make_fixture(), config["data"])
    target = 1000
    first, _ = historical_context(edges, target, config["explain"])
    changed = edges.copy()
    changed.loc[changed.ts >= edges.iloc[target].ts, "amount_paid"] = 1e10
    second, _ = historical_context(changed, target, config["explain"])
    pd.testing.assert_frame_equal(first, second)
    assert (first.ts < edges.iloc[target].ts).all()


def test_temporal_cycle_and_fanout(config):
    edges, _, _ = canonicalize(make_fixture(), config["data"])
    settings = config["explain"]
    history, _ = historical_context(edges, 23, settings)
    found = detect_motifs(history, edges.iloc[23], settings)
    assert any(m["type"] == "cycle" and m["edge_ids"] == [20, 21, 22, 23] for m in found)
    history, _ = historical_context(edges, 27, settings)
    found = detect_motifs(history, edges.iloc[27], settings)
    assert any(m["type"] == "fan_out" for m in found)
    # Reverse event order makes the aggregate cycle non-temporal.
    bad_history = edges.iloc[20:23].copy()
    bad_history["ts"] = [3, 2, 1]
    target = edges.iloc[23].copy()
    target["ts"] = 4
    assert not any(m["type"] == "cycle" for m in detect_motifs(bad_history, target, settings))


def test_repeated_same_counterparty_is_not_fanout(config):
    history = pd.DataFrame(
        {"edge_id": [0, 1, 2], "src": [0, 0, 0], "dst": [1, 1, 1], "ts": [0, 1, 2]}
    )
    target = pd.Series({"edge_id": 3, "src": 0, "dst": 1, "ts": 3})
    assert detect_motifs(history, target, config["explain"]) == []


def test_context_json_and_idempotent_storage(config, tmp_path):
    edges, nodes, _ = canonicalize(make_fixture(), config["data"])
    prediction = pd.Series({"edge_id": 23, "run_id": "test", "model": "fixture", "score": 0.8})
    context = build_context(edges, nodes, prediction, 0.5, config["explain"])
    json.dumps(context, allow_nan=False)
    assert context["target"]["edge_id"] == 23
    assert context["attention_status"].startswith("unavailable")
    database = tmp_path / "alerts.db"
    initialize(database)
    save_alert(database, context, 1)
    update_status(database, 1, "reviewed")
    save_alert(database, context, 1)
    assert len(read_alerts(database)) == 1
    assert read_alerts(database).iloc[0].status == "reviewed"
    assert read_context(database, 1)["edge_id"] == 23
    second = copy.deepcopy(context)
    second["run_id"] = "different_run"
    save_alert(database, second, 1)
    assert len(read_alerts(database)) == 2


def test_baseline_end_to_end_and_data_guard(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    make_fixture().to_csv(config["data"]["input"], index=False)
    preprocess(config)
    run = train(config)
    result = run_evaluation(run)
    assert result["data"]["source_kind"] == "test_fixture"
    assert 0 <= result["test"]["average_precision"] <= 1
    database = export_alerts(run, config)
    assert database.exists()
    first_count = len(read_alerts(database))
    export_alerts(run, config)
    assert len(read_alerts(database)) == first_count
    bundle = build_bundle(run, config)
    assert (bundle / "FIXTURE_ONLY.txt").exists()
    from pathlib import Path

    path = Path(config["data"]["output"]) / "edges.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "amount_paid"] += 1
    changed.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="changed"):
        export_alerts(run, config)


def test_alert_export_rejects_mismatched_ground_truth(config):
    make_fixture(400).to_csv(config["data"]["input"], index=False)
    preprocess(config)
    run = train(config)
    run_evaluation(run)
    path = run / "predictions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], "label"] = 1 - frame.iloc[0].label
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="canonical"):
        export_alerts(run, config)
