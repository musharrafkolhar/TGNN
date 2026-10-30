"""Export top threshold-passing test alerts and grounded historical context."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from tgnn_fraud.evaluation.metrics import validate_predictions
from tgnn_fraud.explain.report import build_context
from tgnn_fraud.serving.alert_store import initialize, save_alert
from tgnn_fraud.utils.config import configure_logging, file_hash, read_config, write_json


def export_alerts(run: Path, config: dict) -> Path:
    """Export actual predictions; fail if processed data no longer matches the run."""
    data = Path(config["data"]["output"])
    manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
    if manifest["edges_sha256"] != file_hash(data / "edges.parquet"):
        raise ValueError("Processed data changed since training; use the matching data bundle")
    edges = pd.read_parquet(data / "edges.parquet")
    nodes = pd.read_parquet(data / "node_index.parquet")
    predictions = pd.read_parquet(run / "predictions.parquet")
    validate_predictions(predictions)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if (
        metrics["run_id"] != predictions.run_id.iloc[0]
        or metrics["model"] != predictions.model.iloc[0]
    ):
        raise ValueError("Metrics and prediction run/model identifiers differ")
    joined = predictions.merge(
        edges[["edge_id", "split", "label"]],
        on="edge_id",
        how="left",
        suffixes=("", "_canonical"),
        validate="one_to_one",
    )
    if (
        joined.split.ne(joined.split_canonical).any()
        or joined.label.ne(joined.label_canonical).any()
    ):
        raise ValueError("Prediction labels/splits/IDs do not match canonical transactions")
    selected = predictions.loc[
        (predictions.split == "test") & (predictions.score >= metrics["threshold"])
    ].sort_values(["score", "edge_id"], ascending=[False, True])
    total = len(selected)
    selected = selected.head(config["explain"]["top_k"])
    selected_ids = set(selected.edge_id)
    attention = {}
    source = run / "attention.jsonl"
    if source.exists():
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record["edge_id"] in selected_ids:
                    attention[record["edge_id"]] = record
    database = run / "alerts.db"
    initialize(database)
    # Replace the generated selection, preserving statuses for alerts still selected.
    import sqlite3

    with sqlite3.connect(database) as connection:
        existing = connection.execute("SELECT alert_id,edge_id FROM alerts").fetchall()
        for alert_id, edge_id in existing:
            if edge_id not in selected_ids:
                connection.execute("DELETE FROM explanations WHERE alert_id=?", (alert_id,))
                connection.execute("DELETE FROM alerts WHERE alert_id=?", (alert_id,))
    for _, prediction in selected.iterrows():
        context = build_context(
            edges,
            nodes,
            prediction,
            metrics["threshold"],
            config["explain"],
            attention.get(int(prediction.edge_id)),
        )
        save_alert(database, context, int(prediction.label))
        write_json(run / "contexts" / f"{int(prediction.edge_id)}.json", context)
    write_json(
        run / "alert_export.json",
        {
            "threshold_passing_test_events": total,
            "exported_alerts": len(selected),
            "top_k_limit": config["explain"]["top_k"],
            "source_kind": manifest["source_kind"],
            "history": "strictly earlier timestamp; target separate",
        },
    )
    logging.info("Exported %s/%s passing alerts to %s", len(selected), total, database)
    return database


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    export_alerts(args.run, read_config(args.config))


if __name__ == "__main__":
    main()
