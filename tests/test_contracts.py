"""Regression checks for provenance, comparison, joins and boundary cases."""

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from tgnn_fraud.data.preprocess import canonicalize
from tgnn_fraud.data.synthetic import make_fixture
from tgnn_fraud.evaluation.metrics import (
    evaluate_predictions,
    select_threshold,
    validate_comparison,
)
from tgnn_fraud.explain.motifs import detect_motifs


def prediction_fixture(model="xgboost"):
    return pd.DataFrame(
        {
            "edge_id": np.arange(8),
            "split": ["val"] * 4 + ["test"] * 4,
            "score": [0.1, 0.4, 0.35, 0.8] * 2,
            "label": [0, 0, 1, 1] * 2,
            "model": model,
            "run_id": model + "_fixture",
        }
    )


def test_comparison_requires_identical_ids_splits_and_labels():
    left, right = prediction_fixture(), prediction_fixture("tgn")
    validate_comparison(left, right.iloc[::-1])
    for column, value in [("edge_id", 99), ("split", "train"), ("label", 1)]:
        changed = right.copy()
        changed.loc[0, column] = value
        with pytest.raises(ValueError, match="identical"):
            validate_comparison(left, changed)


def test_no_test_positives_reported_explicitly():
    frame = prediction_fixture()
    frame.loc[frame.split == "test", "label"] = 0
    result = evaluate_predictions(frame)
    assert result["test"]["average_precision"] is None
    assert result["test"]["positive_class_status"] == "no positives; AP undefined"
    assert result["test"]["recall"] == 0


@pytest.mark.parametrize("column,value", [("edge_id", np.inf), ("run_id", " ")])
def test_prediction_identity_must_be_finite_and_nonempty(column, value):
    frame = prediction_fixture()
    if column == "edge_id":
        frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        evaluate_predictions(frame)


def test_threshold_f1_tie_chooses_highest_cutoff():
    assert select_threshold(np.array([0, 1, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.4, 0.5])) == 0.5


def test_bank_encoding_is_bounded_and_train_fitted():
    from tgnn_fraud.data.features import fit_features

    settings = yaml.safe_load(open("configs/prototype.yaml", encoding="utf-8"))["data"]
    raw = make_fixture(400)
    raw["from_bank"] = [f"bank_{index}" for index in range(len(raw))]
    edges, _, _ = canonicalize(raw, settings)
    transformer, features = fit_features(edges)
    assert features.dtype == np.float32
    assert features.shape[1] <= 6 + 5 * 32
    encoder = transformer.named_transformers_["category"]
    assert "bank_399" not in encoder.categories_[0]


def test_relative_seconds_preserve_dataset_origin_for_later_subset():
    settings = yaml.safe_load(open("configs/prototype.yaml", encoding="utf-8"))["data"]
    raw = make_fixture(400)
    settings = copy.deepcopy(settings)
    settings["subset_start"] = raw.timestamp.iloc[40].isoformat()
    edges, _, manifest = canonicalize(raw, settings)
    assert edges.ts.iloc[0] == 1200
    assert pd.Timestamp(manifest["dataset_start_utc"]) == raw.timestamp.iloc[0].tz_localize("UTC")
    assert pd.Timestamp(manifest["subset_start_utc"]) == raw.timestamp.iloc[40].tz_localize("UTC")


def test_chunked_unsorted_source_matches_full_read(tmp_path):
    from tgnn_fraud.data.preprocess import read_source

    settings = yaml.safe_load(open("configs/prototype.yaml", encoding="utf-8"))["data"]
    settings.update(max_events=300, csv_chunk_rows=37)
    raw = make_fixture(400).sample(frac=1, random_state=14).reset_index(drop=True)
    # Equal timestamps must survive source chunks and the chronological cutoff.
    raw.loc[raw.timestamp == "2024-01-01 02:30:00", "timestamp"] = pd.Timestamp(
        "2024-01-01 02:29:30"
    )
    path = tmp_path / "input.csv"
    raw.to_csv(path, index=False)
    selected, inspection = read_source(path, settings)
    streamed, _, _ = canonicalize(selected, settings)
    whole, _, _ = canonicalize(raw, settings)
    assert streamed.original_row.tolist() == whole.original_row.tolist()
    assert streamed.ts.tolist() == whole.ts.tolist()
    assert inspection["raw_rows"] == 400
    assert inspection["input_class_counts"]["1"] == int(raw.label.sum())


def test_motif_detector_rejects_nonhistorical_input():
    settings = yaml.safe_load(open("configs/prototype.yaml", encoding="utf-8"))["explain"]
    history = pd.DataFrame({"edge_id": [0, 1], "src": [0, 0], "dst": [1, 2], "ts": [1, 4]})
    target = pd.Series({"edge_id": 2, "src": 0, "dst": 3, "ts": 4})
    with pytest.raises(ValueError, match="strictly earlier"):
        detect_motifs(history, target, settings)
