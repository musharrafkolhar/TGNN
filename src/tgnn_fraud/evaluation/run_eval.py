"""Evaluate a saved prediction file without importing any model code."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve

from tgnn_fraud.evaluation.metrics import evaluate_predictions
from tgnn_fraud.utils.config import configure_logging, write_json


def run_evaluation(run: Path) -> dict:
    """Write metrics and self-contained PR curve HTML into the run folder."""
    predictions = pd.read_parquet(run / "predictions.parquet")
    result = evaluate_predictions(predictions)
    metadata = run / "data_manifest.json"
    if metadata.exists():
        result["data"] = json.loads(metadata.read_text(encoding="utf-8"))
    write_json(run / "metrics.json", result)
    figure = go.Figure()
    for split in ["val", "test"]:
        part = predictions.loc[predictions.split == split]
        precision, recall, _ = precision_recall_curve(part.label, part.score)
        figure.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=split))
    figure.update_layout(
        title="Precision recall curves", xaxis_title="Recall", yaxis_title="Precision"
    )
    figure.write_html(run / "pr_curve.html", include_plotlyjs=True)
    logging.info("Evaluated %s: test AP=%s", run.name, result["test"]["average_precision"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    run_evaluation(args.run)


if __name__ == "__main__":
    main()
