"""Context JSON grounded in observed graph facts."""

import pandas as pd

from tgnn_fraud.explain.context import historical_context
from tgnn_fraud.explain.motifs import detect_motifs


def build_context(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    prediction: pd.Series,
    threshold: float,
    settings: dict,
    attention: dict | None = None,
) -> dict:
    """Produce context for one prediction, joining by stable edge ID."""
    target = edges.loc[edges.edge_id == int(prediction.edge_id)].iloc[0]
    history, truncated = historical_context(edges, int(prediction.edge_id), settings)
    motifs = detect_motifs(history, target, settings)
    involved = set(history.src.astype(int)) | set(history.dst.astype(int))
    involved.update([int(target.src), int(target.dst)])
    weights = attention.get("weights", {}) if attention else {}

    def edge_record(row: pd.Series) -> dict:
        return {
            "edge_id": int(row.edge_id),
            "src": int(row.src),
            "dst": int(row.dst),
            "ts": int(row.ts),
            "timestamp_utc": row.timestamp.isoformat(),
            "amount": float(row.amount_paid),
            "amount_received": float(row.amount_received),
            "currency": str(row.currency_paid),
            "currency_received": str(row.currency_received),
            "payment_format": str(row.payment_format),
            "attn": weights.get(str(int(row.edge_id))),
        }

    descriptions = []
    for motif in motifs:
        if motif["type"] == "cycle":
            descriptions.append(
                f"Target closes a {len(motif['node_ids'])}-account temporal cycle "
                f"spanning {motif['span_seconds']} seconds."
            )
        else:
            descriptions.append(
                f"Sender has transfers to {len(motif['node_ids']) - 1} distinct accounts "
                f"in the displayed window."
            )
    return {
        "edge_id": int(prediction.edge_id),
        "run_id": str(prediction.run_id),
        "model": str(prediction.model),
        "score": float(prediction.score),
        "threshold": float(threshold),
        "window_seconds": settings["window_seconds"],
        "context_limits": {key: settings[key] for key in ["hops", "max_edges", "max_motifs"]},
        "history_policy": "strictly earlier timestamps; target shown separately",
        "context_truncated": truncated,
        "attention_status": "available" if attention else "unavailable for this model/run",
        "target": edge_record(target),
        "subgraph": {
            "nodes": [
                {"node_id": int(r.node_id), "label": f"{r.bank}:{r.account}"}
                for r in nodes.loc[nodes.node_id.isin(involved)].itertuples(index=False)
            ],
            "edges": [edge_record(row) for _, row in history.iterrows()],
        },
        "motifs": motifs,
        "summary": " ".join(descriptions) if descriptions else "No supported motif found.",
        "interpretation": "Descriptive context only; patterns and attention do not prove fraud.",
    }
