"""Bounded historical graph extraction without future events."""

import pandas as pd


def historical_context(
    edges: pd.DataFrame, edge_id: int, settings: dict
) -> tuple[pd.DataFrame, bool]:
    """Return earlier edges within seconds/hops/size limits; target is not history."""
    target = edges.loc[edges.edge_id == edge_id]
    if len(target) != 1:
        raise ValueError("Target edge must identify exactly one transaction")
    target = target.iloc[0]
    ts = int(target.ts)
    history = edges.loc[(edges.ts < ts) & (edges.ts >= ts - settings["window_seconds"])]
    frontier = {int(target.src), int(target.dst)}
    selected = set()
    for _ in range(settings["hops"]):
        incident = history.loc[history.src.isin(frontier) | history.dst.isin(frontier)]
        selected.update(incident.edge_id.tolist())
        frontier.update(incident.src.astype(int).tolist())
        frontier.update(incident.dst.astype(int).tolist())
    result = history.loc[history.edge_id.isin(selected)].sort_values(["ts", "edge_id"])
    truncated = len(result) > settings["max_edges"]
    return result.tail(settings["max_edges"]).copy(), truncated
