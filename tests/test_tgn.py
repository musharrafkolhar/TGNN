"""PyG integration tests; explicitly skipped when the optional runtime is unavailable."""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric", reason="Install torch-geometric to verify actual TGN")
import torch  # noqa: E402
import yaml  # noqa: E402

from tgnn_fraud.data.features import fit_features  # noqa: E402
from tgnn_fraud.data.preprocess import canonicalize  # noqa: E402
from tgnn_fraud.data.synthetic import make_fixture  # noqa: E402
from tgnn_fraud.models.tgn import TemporalFraudModel  # noqa: E402
from tgnn_fraud.serving.scorer import load_checkpoint, replay  # noqa: E402


def test_tgn_future_label_invariance_and_attention(tmp_path):
    with open("configs/prototype.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    edges, _, _ = canonicalize(make_fixture(400), config["data"])
    # Equal-time group at a split-safe location tests chunk boundaries.
    edges.loc[261:263, "ts"] = edges.loc[261, "ts"]
    _, features = fit_features(edges)
    model = TemporalFraudModel(
        int(max(edges.src.max(), edges.dst.max())) + 1, features.shape[1], config["tgn"]
    )
    original = replay(model, edges, features, 1, tmp_path / "attention.jsonl")
    labels_changed = edges.copy()
    labels_changed.label = 1 - labels_changed.label
    again = replay(model, labels_changed, features, 10)
    np.testing.assert_allclose(original.score, again.score, atol=1e-6)
    changed_features = features.copy()
    changed_features[350:] = 500
    future = replay(model, edges, changed_features, 10)
    np.testing.assert_allclose(
        original.loc[original.edge_id < 350, "score"],
        future.loc[future.edge_id < 350, "score"],
        atol=1e-6,
    )
    assert (tmp_path / "attention.jsonl").stat().st_size > 0
    for line in (tmp_path / "attention.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        target_ts = edges.iloc[record["edge_id"]].ts
        assert all(edges.iloc[int(edge_id)].ts < target_ts for edge_id in record["weights"])


def test_tgn_training_and_checkpoint(tmp_path):
    from tgnn_fraud.data.preprocess import preprocess
    from tgnn_fraud.training.train_tgn import train

    with open("configs/prototype.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["data"]["input"] = str(tmp_path / "fixture.csv")
    config["data"]["output"] = str(tmp_path / "processed")
    config["runs_dir"] = str(tmp_path / "runs")
    config["tgn"]["epochs"] = 1
    make_fixture(400).to_csv(config["data"]["input"], index=False)
    raw = pd.read_csv(config["data"]["input"])
    raw.loc[100:103, "timestamp"] = raw.loc[100, "timestamp"]
    raw.to_csv(config["data"]["input"], index=False)
    config["tgn"]["max_events_per_chunk"] = 1
    preprocess(config)
    run = train(config)
    assert (run / "checkpoint_best.pt").exists()
    assert (run / "predictions.parquet").exists()
    edges = pd.read_parquet(tmp_path / "processed/edges.parquet")
    features = np.load(tmp_path / "processed/features.npy")
    loaded = load_checkpoint(run / "checkpoint_best.pt")
    actual = replay(loaded, edges, features, 31)
    saved = pd.read_parquet(run / "predictions.parquet")
    np.testing.assert_allclose(actual.score, saved.score, atol=1e-6)
    torch.manual_seed(config["seed"])
    initial = TemporalFraudModel(loaded.num_nodes, features.shape[1], config["tgn"])
    assert not torch.equal(initial.memory.gru.weight_ih, loaded.memory.gru.weight_ih)
    repeated_run = train(config)
    repeated = pd.read_parquet(repeated_run / "predictions.parquet")
    np.testing.assert_array_equal(saved.score, repeated.score)
