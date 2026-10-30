"""Independently reload actual model artifacts and verify their saved outputs."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tgnn_fraud.evaluation.metrics import (  # noqa: E402
    evaluate_predictions,
    validate_comparison,
)
from tgnn_fraud.utils.config import file_hash, write_json  # noqa: E402


def verify(data: Path, runs: list[Path], source: Path | None = None) -> dict:
    """Fail on a mismatch; replay the trusted local TGN checkpoint from zero history."""
    edges = pd.read_parquet(data / "edges.parquet").set_index("edge_id", drop=False)
    nodes = pd.read_parquet(data / "node_index.parquet").set_index("node_id")
    features = np.load(data / "features.npy", allow_pickle=False)
    held = edges.split.isin(["val", "test"])
    frames, results = [], []
    for run in runs:
        manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
        assert file_hash(data / "edges.parquet") == manifest["edges_sha256"]
        if source:
            assert file_hash(source) == manifest["source_sha256"]
        frame = pd.read_parquet(run / "predictions.parquet")
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        actual_metrics = evaluate_predictions(frame)
        for key, value in actual_metrics.items():
            assert metrics[key] == value, f"Metric mismatch: {key}"
        assert frame.edge_id.tolist() == edges.loc[held, "edge_id"].tolist()
        assert frame.label.tolist() == edges.loc[held, "label"].tolist()
        if metrics["model"] == "xgboost":
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(run / "model.json")
            restored_scores = model.predict_proba(features[held.to_numpy()])[:, 1]
        elif metrics["model"] == "tgn":
            import torch

            from tgnn_fraud.serving.scorer import load_checkpoint, replay

            torch.set_num_threads(1)
            model = load_checkpoint(run / "checkpoint_best.pt")
            restored_scores = replay(model, edges.reset_index(drop=True), features, 256).score
        else:
            raise ValueError("Unsupported model")
        np.testing.assert_allclose(restored_scores, frame.score, rtol=0, atol=1e-7)
        lookup = frame.set_index("edge_id")
        with sqlite3.connect(run / "alerts.db") as connection:
            rows = connection.execute(
                "SELECT a.edge_id,a.run_id,a.model,a.score,a.threshold,a.label,e.json "
                "FROM alerts a JOIN explanations e USING(alert_id)"
            ).fetchall()
        motif_count = 0
        for edge_id, run_id, model_name, score, threshold, label, payload in rows:
            context = json.loads(payload)
            target = edges.loc[edge_id]
            assert context["edge_id"] == edge_id == context["target"]["edge_id"]
            assert context["run_id"] == run_id == metrics["run_id"]
            assert context["model"] == model_name == metrics["model"]
            assert context["score"] == score == float(lookup.loc[edge_id].score)
            assert context["threshold"] == threshold == metrics["threshold"]
            assert label == int(target.label)
            assert int(context["target"]["ts"]) == int(target.ts)
            assert score >= threshold
            assert len(context["subgraph"]["edges"]) <= context["context_limits"]["max_edges"]
            evidence = {edge_id}
            for record in context["subgraph"]["edges"]:
                event = edges.loc[record["edge_id"]]
                assert target.ts - context["window_seconds"] <= event.ts < target.ts
                for key in ["src", "dst", "ts"]:
                    assert record[key] == int(event[key])
                assert record["amount"] == float(event.amount_paid)
                assert record["currency"] == str(event.currency_paid)
                evidence.add(record["edge_id"])
            for node in context["subgraph"]["nodes"]:
                account = nodes.loc[node["node_id"]]
                assert node["label"] == f"{account.bank}:{account.account}"
            for motif in context["motifs"]:
                assert set(motif["edge_ids"]) <= evidence
                support = edges.loc[motif["edge_ids"]]
                if motif["type"] == "cycle":
                    assert (np.diff(support.ts.to_numpy()) > 0).all()
                    assert support.dst.iloc[:-1].tolist() == support.src.iloc[1:].tolist()
                    assert support.dst.iloc[-1] == support.src.iloc[0]
                motif_count += 1
            disk_context = json.loads(
                (run / "contexts" / f"{edge_id}.json").read_text(encoding="utf-8")
            )
            assert disk_context == context
        frames.append(frame)
        results.append(
            {
                "run_id": run.name,
                "model": metrics["model"],
                "prediction_rows": len(frame),
                "model_reload_matches": True,
                "verified_alerts": len(rows),
                "verified_motifs": motif_count,
            }
        )
    for frame in frames[1:]:
        validate_comparison(frames[0], frame)
    return {"passed": True, "identical_held_out_sets": True, "runs": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/artifact_verification.json"))
    args = parser.parse_args()
    write_json(args.output, verify(args.data, args.runs, args.source))


if __name__ == "__main__":
    main()
