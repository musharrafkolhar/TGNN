"""Saved-output investigator dashboard; no model imports or training."""

import json
import sqlite3
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TGNN Fraud Investigation", page_icon="🔎", layout="wide")
st.title("TGNN · Transaction investigation")
st.caption("Saved model scores · Historical transaction context · IBM synthetic AML")

roots = [Path("runs"), Path("data/demo")]
required = ["metrics.json", "data_manifest.json", "alert_export.json"]
candidates = {p.parent for root in roots if root.exists() for p in root.glob("*/alerts.db")}
runs = sorted(
    [p for p in candidates if all((p / name).is_file() for name in required)],
    key=lambda p: p.name,
    reverse=True,
)
if len(runs) < len(candidates):
    st.sidebar.warning("Incomplete saved runs were excluded. Re-run evaluation and alert export.")
# Prefer the full run when an identical portable demo bundle is also present.
unique_runs = {}
for candidate in sorted(runs, key=lambda path: path.parts[0] == "runs"):
    unique_runs[candidate.name] = candidate
runs = sorted(unique_runs.values(), key=lambda path: path.name, reverse=True)
if not runs:
    st.info(
        "No saved alerts yet. Prepare data, train a model, evaluate it and export alerts first."
    )
    st.code("python scripts/run_prototype.py --fixture --model xgboost")
    st.stop()

run_name = st.sidebar.selectbox("Saved run", [str(p) for p in runs])
run = Path(run_name)
metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "data_manifest.json").read_text(encoding="utf-8"))
export = json.loads((run / "alert_export.json").read_text(encoding="utf-8"))
database = run / "alerts.db"
if manifest["source_kind"] == "test_fixture":
    st.warning(
        "TEST FIXTURE ONLY: planted patterns test the software. These are not IBM AML results."
    )
else:
    st.info("IBM synthetic AML data. Results apply to the recorded chronological subset.")
st.caption(f"Model: {metrics['model']} · Run: {metrics['run_id']}")
cols = st.columns(4)
for col, name, label in zip(
    cols,
    ["average_precision", "precision", "recall", "f1"],
    ["Test AP", "Precision", "Recall", "F1"],
    strict=True,
):
    value = metrics["test"][name]
    col.metric(label, f"{value:.3f}" if value is not None else "Unavailable")
st.caption(
    f"Validation-selected threshold: {metrics['threshold']:.4f}. "
    f"Displaying up to {export['top_k_limit']} of "
    f"{export['threshold_passing_test_events']} threshold-passing test events."
)
if metrics["test"]["positives"] and metrics["test"]["confusion_matrix"]["tp"] == 0:
    st.warning(
        "This run recovered no labelled laundering transactions in test at its frozen "
        "validation threshold. These pilot scores do not demonstrate useful detection."
    )
with st.expander("Dataset and evaluation provenance"):
    st.json(
        {
            "source_kind": manifest["source_kind"],
            "source_sha256": manifest["source_sha256"],
            "selected_rows": manifest["selected_rows"],
            "subset": manifest["subset"],
            "evaluation": metrics,
        }
    )

with sqlite3.connect(database) as connection:
    alerts = pd.read_sql_query("SELECT * FROM alerts ORDER BY score DESC, edge_id", connection)
origin = manifest.get("splits", {}).get("dataset_start_utc")
if origin and not alerts.empty:
    alerts["timestamp_utc"] = pd.Timestamp(origin) + pd.to_timedelta(alerts.ts, unit="s")
evaluation_mode = st.sidebar.checkbox("Show evaluation ground truth", value=False)
if evaluation_mode:
    st.info("Evaluation mode: labels are ground truth for assessment, not model inputs.")
page = st.sidebar.radio("View", ["Alert queue", "Alert detail"])
if alerts.empty:
    st.info("No alerts passed the saved threshold. This is a valid model output.")
    st.stop()

if page == "Alert queue":
    floor = st.slider("Minimum displayed risk score", 0.0, 1.0, 0.0, 0.01)
    account = st.text_input("Account node ID filter", placeholder="Optional integer ID")
    status = st.selectbox("Review status", ["all", "new", "reviewed", "dismissed"])
    view = alerts.loc[alerts.score >= floor].copy()
    first, last = int(alerts.ts.min()), int(alerts.ts.max())
    if first < last:
        lower, upper = st.slider(
            "Transaction time range (seconds from dataset start)", first, last, (first, last)
        )
        view = view.loc[view.ts.between(lower, upper)]
    if account:
        if not account.isdigit():
            st.error("Enter a numeric node ID")
        else:
            view = view.loc[(view.src == int(account)) | (view.dst == int(account))]
    if status != "all":
        view = view.loc[view.status == status]
    columns = ["alert_id", "edge_id", "ts", "src", "dst", "amount", "currency", "score", "status"]
    if "timestamp_utc" in view:
        columns.insert(2, "timestamp_utc")
    if evaluation_mode:
        columns.append("label")
    st.dataframe(view[columns], use_container_width=True, hide_index=True)
    st.caption("Choose Alert detail in the sidebar to inspect a transaction.")
    st.caption(f"{len(view)} alerts match your filters. Scores are not calibrated probabilities.")
else:
    alert_labels = {
        int(row.alert_id): f"Alert {int(row.alert_id)} | edge {int(row.edge_id)}"
        for row in alerts.itertuples(index=False)
    }
    selection = st.selectbox("Alert", list(alert_labels.values()))
    alert_id = next(key for key, label in alert_labels.items() if label == selection)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT json FROM explanations WHERE alert_id=?", (int(alert_id),)
        ).fetchone()
    context = json.loads(row[0])
    st.subheader(f"Transaction {context['edge_id']}")
    st.write(context["summary"])
    st.caption(context["interpretation"])
    account_labels = {node["node_id"]: node["label"] for node in context["subgraph"]["nodes"]}
    target = context["target"]
    st.write(
        f"**Sender:** {account_labels[target['src']]} → "
        f"**Recipient:** {account_labels[target['dst']]}"
    )
    st.write(
        f"**Amount paid:** {target['amount']:,.2f} {target['currency']} · "
        f"**Time:** {target.get('timestamp_utc', str(target['ts']) + ' dataset seconds')}"
    )
    st.caption(
        f"Saved score {context['score']:.6f} · Frozen threshold "
        f"{context['threshold']:.6f} · Attention: {context['attention_status']}"
    )
    with st.expander("Saved transaction and scoring record"):
        st.json(
            {
                "target": target,
                "score": context["score"],
                "threshold": context["threshold"],
                "attention_status": context["attention_status"],
            }
        )
    if context["context_truncated"]:
        st.caption("Context is capped at the configured maximum number of recent edges.")
    graph = nx.DiGraph()
    labels = {n["node_id"]: n["label"] for n in context["subgraph"]["nodes"]}
    graph.add_nodes_from(labels)
    historical = context["subgraph"]["edges"]
    target = context["target"]
    graph.add_edges_from((e["src"], e["dst"]) for e in [*historical, target])
    positions = nx.spring_layout(graph, seed=42)
    figure = go.Figure()
    for name, records, color in [
        ("Earlier transactions", historical, "#94a3b8"),
        ("Target transaction", [target], "#dc2626"),
    ]:
        x, y = [], []
        for edge in records:
            a, b = positions[edge["src"]], positions[edge["dst"]]
            x.extend([a[0], b[0], None])
            y.extend([a[1], b[1], None])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line={"color": color, "width": 2 if name.startswith("Earlier") else 4},
            )
        )
        for edge in records:
            a, b = positions[edge["src"]], positions[edge["dst"]]
            if edge["src"] != edge["dst"]:
                figure.add_annotation(
                    x=float(b[0] * 0.85 + a[0] * 0.15),
                    y=float(b[1] * 0.85 + a[1] * 0.15),
                    ax=float(b[0] * 0.65 + a[0] * 0.35),
                    ay=float(b[1] * 0.65 + a[1] * 0.35),
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                )
    figure.add_trace(
        go.Scatter(
            x=[positions[n][0] for n in graph],
            y=[positions[n][1] for n in graph],
            mode="markers+text",
            text=[str(n) for n in graph],
            textposition="top center",
            hovertext=[labels.get(n, str(n)) for n in graph],
            marker={"size": 14, "color": "#2563eb"},
            name="Accounts",
        )
    )
    figure.update_layout(
        height=450,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Lines show account relationships; the transaction table identifies transfer direction."
    )
    st.dataframe(pd.DataFrame([*historical, target]), use_container_width=True, hide_index=True)
    st.json(context["motifs"])
    statuses = ["new", "reviewed", "dismissed"]
    saved_status = str(alerts.loc[alerts.alert_id == alert_id, "status"].iloc[0])
    new_status = st.selectbox("Set review status", statuses, index=statuses.index(saved_status))
    if st.button("Save review status"):
        with sqlite3.connect(database) as connection:
            connection.execute(
                "UPDATE alerts SET status=? WHERE alert_id=?", (new_status, int(alert_id))
            )
        st.success("Review status saved. This does not change the model or ground-truth label.")
    if evaluation_mode:
        label = int(alerts.loc[alerts.alert_id == alert_id, "label"].iloc[0])
        st.write(f"Evaluation ground truth: {'laundering' if label else 'normal'}")
