"""Train TGN and export checkpoint-consistent held-out predictions."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from tgnn_fraud.data.event_stream import time_groups
from tgnn_fraud.models.tgn import TemporalFraudModel
from tgnn_fraud.serving.scorer import load_checkpoint, replay
from tgnn_fraud.utils.config import configure_logging, file_hash, new_run, read_config, write_json
from tgnn_fraud.utils.seed import seed_all


def train(config: dict) -> Path:
    """Fit on training labels only and select a checkpoint by validation AP."""
    seed_all(config["seed"])
    torch.set_num_threads(1)
    data = Path(config["data"]["output"])
    edges = pd.read_parquet(data / "edges.parquet")
    features = np.load(data / "features.npy", allow_pickle=False)
    settings = config["tgn"]
    if settings["epochs"] < 1 or settings["max_events_per_chunk"] < 1:
        raise ValueError("TGN epochs and chunk size must be positive")
    device = settings["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; set device: cpu")
    model = TemporalFraudModel(
        int(max(edges.src.max(), edges.dst.max())) + 1, features.shape[1], settings
    ).to(device)
    src = torch.as_tensor(edges.src.to_numpy(copy=True), dtype=torch.long, device=device)
    dst = torch.as_tensor(edges.dst.to_numpy(copy=True), dtype=torch.long, device=device)
    times = torch.as_tensor(edges.ts.to_numpy(copy=True), dtype=torch.long, device=device)
    messages = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(edges.label.to_numpy(copy=True), dtype=torch.float32, device=device)
    train_count = int((edges.split == "train").sum())
    positives = float(labels[:train_count].sum())
    if not 0 < positives < train_count:
        raise ValueError("TGN training requires both classes in training data")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((train_count - positives) / positives, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    run = new_run(config, "tgn")
    best_ap = -1.0
    history = []
    for epoch in range(settings["epochs"]):
        model.train()
        model.reset_history()
        losses = []
        for group in time_groups(edges.ts.to_numpy()[:train_count]):
            optimizer.zero_grad()
            # Accumulate chunk gradients, but update weights/history only once per timestamp.
            for start in range(0, len(group), settings["max_events_per_chunk"]):
                rows = group[start : start + settings["max_events_per_chunk"]]
                logits, _ = model.score(src[rows], dst[rows], messages[rows], messages, times)
                loss = loss_fn(logits, labels[rows]) * len(rows) / len(group)
                loss.backward()
                losses.append(float(loss.detach()))
            optimizer.step()
            model.observe(src[group], dst[group], times[group], messages[group])
        # Rebuild history with final epoch weights. Never score test during checkpoint selection.
        validation_stop = int((edges.split != "test").sum())
        val_predictions = replay(
            model,
            edges.iloc[:validation_stop],
            features[:validation_stop],
            settings["max_events_per_chunk"],
        )
        val_ap = float(average_precision_score(val_predictions.label, val_predictions.score))
        history.append({"epoch": epoch + 1, "loss_sum": sum(losses), "val_ap": val_ap})
        logging.info("Epoch %s val AP %.4f", epoch + 1, val_ap)
        if val_ap > best_ap:
            best_ap = val_ap
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_nodes": model.num_nodes,
                    "feature_dim": features.shape[1],
                    "model_config": settings,
                },
                run / "checkpoint_best.pt",
            )
    selected = load_checkpoint(run / "checkpoint_best.pt", device)
    predictions = replay(
        selected,
        edges,
        features,
        settings["max_events_per_chunk"],
        attention_path=run / "attention.jsonl",
    )
    predictions["model"] = "tgn"
    predictions["run_id"] = run.name
    predictions.to_parquet(run / "predictions.parquet", index=False)
    write_json(run / "training_history.json", history)
    report = json.loads((data / "preprocess_report.json").read_text(encoding="utf-8"))
    report["edges_sha256"] = file_hash(data / "edges.parquet")
    write_json(run / "data_manifest.json", report)
    logging.info("Saved TGN run: %s", run)
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    configure_logging()
    train(read_config(args.config))


if __name__ == "__main__":
    main()
