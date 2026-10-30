"""SQLite persistence with idempotent run/model/edge imports."""

import json
import sqlite3
from pathlib import Path

import pandas as pd

STATUSES = {"new", "reviewed", "dismissed"}


def initialize(path: Path) -> None:
    """Create alert/context schema and enable foreign keys on writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY, edge_id INTEGER NOT NULL,
                run_id TEXT NOT NULL, model TEXT NOT NULL, ts INTEGER NOT NULL,
                src INTEGER NOT NULL, dst INTEGER NOT NULL, amount REAL, currency TEXT,
                score REAL NOT NULL, threshold REAL NOT NULL, label INTEGER,
                status TEXT NOT NULL DEFAULT 'new', UNIQUE(run_id, model, edge_id)
            );
            CREATE TABLE IF NOT EXISTS explanations (
                alert_id INTEGER PRIMARY KEY REFERENCES alerts(alert_id), json TEXT NOT NULL
            );
        """)


def save_alert(path: Path, context: dict, label: int) -> None:
    """Upsert one saved output while preserving an investigator's review status."""
    target = context["target"]
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """INSERT INTO alerts(edge_id,run_id,model,ts,src,dst,amount,currency,
               score,threshold,label) VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(run_id,model,edge_id) DO UPDATE SET score=excluded.score,
               threshold=excluded.threshold,label=excluded.label""",
            (
                context["edge_id"],
                context["run_id"],
                context["model"],
                target["ts"],
                target["src"],
                target["dst"],
                target["amount"],
                target["currency"],
                context["score"],
                context["threshold"],
                int(label),
            ),
        )
        alert_id = connection.execute(
            "SELECT alert_id FROM alerts WHERE run_id=? AND model=? AND edge_id=?",
            (context["run_id"], context["model"], context["edge_id"]),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO explanations VALUES(?,?) "
            "ON CONFLICT(alert_id) DO UPDATE SET json=excluded.json",
            (alert_id, json.dumps(context, allow_nan=False)),
        )


def read_alerts(path: Path) -> pd.DataFrame:
    """Read ranked alerts without creating a missing database."""
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        return pd.read_sql_query("SELECT * FROM alerts ORDER BY score DESC, edge_id", connection)


def read_context(path: Path, alert_id: int) -> dict:
    """Load the JSON associated with a selected alert."""
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        row = connection.execute(
            "SELECT json FROM explanations WHERE alert_id=?", (alert_id,)
        ).fetchone()
    if row is None:
        raise ValueError("No context found for alert")
    return json.loads(row[0])


def update_status(path: Path, alert_id: int, status: str) -> None:
    """Record analyst workflow status without treating it as a fraud label."""
    if status not in STATUSES:
        raise ValueError("Unsupported review status")
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE alerts SET status=? WHERE alert_id=?", (status, alert_id))
