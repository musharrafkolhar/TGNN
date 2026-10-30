"""FastAPI backend serving offline AML transaction runs and graph alerts."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "runs"
DEMO_DIR = BASE_DIR / "data" / "demo"
DATA_DIR = BASE_DIR / "data" / "ibm_pilot_processed"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data" / "processed"

app = FastAPI(title="TGNN Fraud Intelligence Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_available_runs() -> List[Dict[str, Any]]:
    """Scan both runs/ and data/demo/ for complete runs."""
    seen = {}
    search_dirs = [RUNS_DIR, DEMO_DIR]
    required_files = ["metrics.json", "data_manifest.json", "alert_export.json", "alerts.db"]

    for root in search_dirs:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("*/alerts.db")):
            run_folder = candidate.parent
            if all((run_folder / fname).is_file() for fname in required_files):
                if run_folder.name not in seen:
                    seen[run_folder.name] = run_folder

    result = []
    for run_name, path in sorted(seen.items(), reverse=True):
        try:
            metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((path / "data_manifest.json").read_text(encoding="utf-8"))
            export_data = json.loads((path / "alert_export.json").read_text(encoding="utf-8"))
            result.append(
                {
                    "run_id": run_name,
                    "model": metrics.get("model", "unknown").upper(),
                    "path": str(path),
                    "threshold": metrics.get("threshold", 0.0),
                    "source_kind": manifest.get("source_kind", "ibm_aml"),
                    "test_ap": metrics.get("test", {}).get("average_precision", 0.0),
                    "test_precision": metrics.get("test", {}).get("precision", 0.0),
                    "test_recall": metrics.get("test", {}).get("recall", 0.0),
                    "test_f1": metrics.get("test", {}).get("f1", 0.0),
                    "total_alerts": export_data.get("exported_alerts", 0),
                }
            )
        except Exception:
            continue
    return result


def get_run_path(run_id: str) -> Path:
    candidates = [RUNS_DIR / run_id, DEMO_DIR / run_id]
    for p in candidates:
        if p.exists() and (p / "alerts.db").is_file():
            return p
    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")


@app.get("/api/runs")
def list_runs():
    runs = get_available_runs()
    return {"runs": runs}


@app.get("/api/run/{run_id}/metrics")
def get_metrics(run_id: str):
    path = get_run_path(run_id)
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "data_manifest.json").read_text(encoding="utf-8"))
    return {"metrics": metrics, "manifest": manifest}


@app.get("/api/run/{run_id}/alerts")
def get_alerts(
    run_id: str,
    q: Optional[str] = Query(None, description="Search account or edge ID"),
    status: Optional[str] = Query("all"),
    min_score: Optional[float] = Query(0.0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    path = get_run_path(run_id)
    db_file = path / "alerts.db"

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM alerts WHERE score >= ?"
    params: List[Any] = [min_score]

    if status and status.lower() != "all":
        query += " AND status = ?"
        params.append(status.lower())

    if q:
        q_term = f"%{q.strip()}%"
        query += (
            " AND (CAST(edge_id AS TEXT) LIKE ? OR CAST(src AS TEXT) LIKE ?"
            " OR CAST(dst AS TEXT) LIKE ? OR currency LIKE ?)"
        )
        params.extend([q_term, q_term, q_term, q_term])

    count_cursor = conn.cursor()
    count_cursor.execute(f"SELECT COUNT(*) FROM ({query})", params)
    total_count = count_cursor.fetchone()[0]

    query += " ORDER BY score DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()

    # Load node index if available to augment account strings
    node_idx_file = DATA_DIR / "node_index.parquet"
    if node_idx_file.exists():
        try:
            ndf = pd.read_parquet(node_idx_file).set_index("node_id")
            for r in rows:
                src_id = r["src"]
                dst_id = r["dst"]
                if src_id in ndf.index:
                    r["src_label"] = f"{ndf.loc[src_id, 'bank']}:{ndf.loc[src_id, 'account']}"
                else:
                    r["src_label"] = f"acct_{src_id}"
                if dst_id in ndf.index:
                    r["dst_label"] = f"{ndf.loc[dst_id, 'bank']}:{ndf.loc[dst_id, 'account']}"
                else:
                    r["dst_label"] = f"acct_{dst_id}"
        except Exception:
            pass

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "alerts": rows,
    }


@app.get("/api/run/{run_id}/alert/{edge_id}")
def get_alert_detail(run_id: str, edge_id: int):
    path = get_run_path(run_id)
    ctx_file = path / "contexts" / f"{edge_id}.json"

    if ctx_file.exists():
        data = json.loads(ctx_file.read_text(encoding="utf-8"))
        return data

    # Fallback to query database if specific context file not saved
    db_file = path / "alerts.db"
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row = conn.cursor().execute("SELECT * FROM alerts WHERE edge_id = ?", (edge_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Alert edge {edge_id} not found.")

    r = dict(row)
    return {
        "edge_id": edge_id,
        "run_id": run_id,
        "model": r["model"],
        "score": r["score"],
        "threshold": r["threshold"],
        "target": {
            "edge_id": edge_id,
            "src": r["src"],
            "dst": r["dst"],
            "ts": r["ts"],
            "amount": r["amount"],
            "currency": r["currency"],
        },
        "subgraph": {"nodes": [], "edges": []},
        "motifs": [],
        "summary": "Historical alert detail loaded from database index.",
        "interpretation": "Descriptive context only; patterns do not prove fraud.",
    }


class StatusUpdate(BaseModel):
    status: str


@app.post("/api/run/{run_id}/alert/{alert_id}/status")
def update_alert_status(run_id: str, alert_id: int, body: StatusUpdate):
    path = get_run_path(run_id)
    db_file = path / "alerts.db"

    allowed = ["new", "under_review", "escalated", "dismissed", "closed"]
    st = body.status.lower()
    if st not in allowed:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed}")

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute("UPDATE alerts SET status = ? WHERE alert_id = ?", (st, alert_id))
    conn.commit()
    conn.close()

    return {"status": "success", "alert_id": alert_id, "new_status": st}


@app.get("/api/run/{run_id}/pr-curve")
def get_pr_curve(run_id: str):
    path = get_run_path(run_id)
    preds_file = path / "predictions.parquet"
    if not preds_file.exists():
        raise HTTPException(status_code=404, detail="Predictions parquet not found for run.")

    df = pd.read_parquet(preds_file)
    test_df = df[df["split"] == "test"]
    if test_df.empty:
        test_df = df

    y_true = test_df["label"].to_numpy()
    y_score = test_df["score"].to_numpy()

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Downsample for web rendering (100 points)
    step = max(1, len(precision) // 100)
    sampled_p = precision[::step].tolist()
    sampled_r = recall[::step].tolist()
    sampled_t = thresholds[::step].tolist() if len(thresholds) > 0 else [0.0] * len(sampled_p)

    return {
        "precision": sampled_p,
        "recall": sampled_r,
        "thresholds": sampled_t,
        "positive_count": int(np.sum(y_true)),
        "total_count": len(y_true),
    }


@app.get("/api/comparison")
def get_comparison():
    comp_file = RUNS_DIR / "comparison.json"
    if not comp_file.exists():
        comp_file = DEMO_DIR / "comparison.json"
    if not comp_file.exists():
        raise HTTPException(status_code=404, detail="Comparison file not found.")

    return json.loads(comp_file.read_text(encoding="utf-8"))


@app.get("/api/dataset/summary")
def get_dataset_summary():
    report_file = DATA_DIR / "preprocess_report.json"
    if not report_file.exists():
        return {
            "source_kind": "ibm_aml",
            "raw_rows": 5078345,
            "nodes": 157344,
            "selected_rows": 199583,
            "prevalence": 0.001019,
        }
    return json.loads(report_file.read_text(encoding="utf-8"))


# Static files mount for UI
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"status": "TGNN API running. Place index.html in app/static/."}
    return FileResponse(index_file)
