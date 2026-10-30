"""Fixed-checkpoint chronological replay; no online parameter updates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tgnn_fraud.data.event_stream import time_groups
from tgnn_fraud.models.tgn import TemporalFraudModel


def replay(
    model: TemporalFraudModel,
    edges: pd.DataFrame,
    features: np.ndarray,
    chunk_size: int,
    attention_path: Path | None = None,
) -> pd.DataFrame:
    """Rebuild history using one checkpoint and save val/test scores and local attention."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not np.array_equal(edges.edge_id.to_numpy(), np.arange(len(edges))):
        raise ValueError("Replay requires canonical chronological edge IDs starting at zero")
    if len(features) != len(edges) or not np.isfinite(features).all():
        raise ValueError("Features must be finite and aligned with canonical edges")
    model.eval()
    model.reset_history()
    device = model.memory.device
    times = torch.as_tensor(edges.ts.to_numpy(copy=True), dtype=torch.long, device=device)
    messages = torch.as_tensor(features, dtype=torch.float32, device=device)
    src = torch.as_tensor(edges.src.to_numpy(copy=True), dtype=torch.long, device=device)
    dst = torch.as_tensor(edges.dst.to_numpy(copy=True), dtype=torch.long, device=device)
    scores = np.empty(len(edges), dtype=np.float32)
    handle = attention_path.open("w", encoding="utf-8") if attention_path else None
    try:
        with torch.no_grad():
            for group in time_groups(edges.ts.to_numpy()):
                for start in range(0, len(group), chunk_size):
                    rows = group[start : start + chunk_size]
                    logits, attention = model.score(
                        src[rows], dst[rows], messages[rows], messages, times
                    )
                    scores[rows] = logits.sigmoid().cpu().numpy()
                    if handle:
                        for row in rows:
                            if edges.iloc[row].split == "train":
                                continue
                            endpoints = {int(edges.iloc[row].src), int(edges.iloc[row].dst)}
                            pairs = {}
                            for eid, target, weight in zip(
                                attention["edge_ids"],
                                attention["targets"],
                                attention["weights"],
                                strict=True,
                            ):
                                if target in endpoints:
                                    pairs.setdefault(int(eid), []).append(float(weight))
                            record = {
                                "edge_id": int(row),
                                "weights": {str(e): float(np.mean(w)) for e, w in pairs.items()},
                                "interpretation": "embedding attention, not causal attribution",
                            }
                            handle.write(json.dumps(record) + "\n")
                model.observe(src[group], dst[group], times[group], messages[group])
    finally:
        if handle:
            handle.close()
    result = edges.loc[edges.split != "train", ["edge_id", "split", "label"]].copy()
    result["score"] = scores[edges.split != "train"]
    return result


def load_checkpoint(path: Path, device: str = "cpu") -> TemporalFraudModel:
    """Load a trusted local prototype checkpoint; never load untrusted pickle files."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TemporalFraudModel(
        checkpoint["num_nodes"], checkpoint["feature_dim"], checkpoint["model_config"]
    )
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.reset_history()
    return model
