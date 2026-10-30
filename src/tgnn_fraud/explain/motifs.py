"""Temporal motif descriptions, not fraud verdicts or causal explanations."""

import pandas as pd


def detect_motifs(history: pd.DataFrame, target: pd.Series, settings: dict) -> list[dict]:
    """Detect time-increasing cycles closed by target and fan-out at its sender."""
    if not (history.ts < target.ts).all():
        raise ValueError("Motif history must contain strictly earlier timestamps")
    motifs = []
    # A directed temporal path dst -> ... -> src, plus current src -> dst, closes a cycle.
    # DFS scans a bounded subgraph rather than enumerating all full-graph cycles.
    outgoing = {}
    for row in history.itertuples(index=False):
        outgoing.setdefault(int(row.src), []).append(row)

    def search(node: int, last_ts: int, nodes: list[int], rows: list) -> None:
        if len(motifs) >= settings["max_motifs"]:
            return
        if node == int(target.src) and rows:
            motifs.append(
                {
                    "type": "cycle",
                    "node_ids": nodes,
                    "edge_ids": [int(r.edge_id) for r in rows] + [int(target.edge_id)],
                    "span_seconds": int(target.ts) - int(rows[0].ts),
                    "includes_target": True,
                }
            )
            return
        if len(rows) >= settings["max_cycle_length"] - 1:
            return
        for row in outgoing.get(node, []):
            if int(row.ts) <= last_ts or int(row.dst) in nodes:
                continue
            search(int(row.dst), int(row.ts), nodes + [int(row.dst)], rows + [row])

    if int(target.src) != int(target.dst):
        search(int(target.dst), -1, [int(target.dst)], [])
    transfers = pd.concat([history.loc[history.src == target.src], target.to_frame().T])
    transfers = transfers.loc[transfers.dst != target.src]
    destinations = transfers.dst.astype(int).unique()
    if len(destinations) >= settings["fanout_min_neighbors"]:
        motifs.append(
            {
                "type": "fan_out",
                "node_ids": [int(target.src), *map(int, destinations)],
                "edge_ids": transfers.edge_id.astype(int).tolist(),
                "span_seconds": int(transfers.ts.max()) - int(transfers.ts.min()),
                "includes_target": True,
            }
        )
    return motifs[: settings["max_motifs"]]
