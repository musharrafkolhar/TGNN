"""Train XGBoost and write the universal prediction interface."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tgnn_fraud.models.baselines import train_baseline
from tgnn_fraud.utils.config import configure_logging, file_hash, new_run, read_config, write_json
from tgnn_fraud.utils.seed import seed_all


def train(config: dict) -> Path:
    """Create a saved XGBoost run without evaluating or importing other stages."""
    seed_all(config["seed"])
    data = Path(config["data"]["output"])
    edges = pd.read_parquet(data / "edges.parquet")
    features = np.load(data / "features.npy", allow_pickle=False)
    train_mask = (edges.split == "train").to_numpy()
    model = train_baseline(
        features[train_mask], edges.label.to_numpy()[train_mask], config["baseline"], config["seed"]
    )
    run = new_run(config, "xgboost")
    model.save_model(run / "model.json")
    held = edges.loc[edges.split.isin(["val", "test"]), ["edge_id", "split", "label"]].copy()
    held["score"] = model.predict_proba(features[~train_mask])[:, 1]
    held["model"] = "xgboost"
    held["run_id"] = run.name
    held.to_parquet(run / "predictions.parquet", index=False)
    report = json.loads((data / "preprocess_report.json").read_text(encoding="utf-8"))
    report["edges_sha256"] = file_hash(data / "edges.parquet")
    write_json(run / "data_manifest.json", report)
    logging.info("Saved XGBoost run: %s", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    train(read_config(args.config))


if __name__ == "__main__":
    main()
