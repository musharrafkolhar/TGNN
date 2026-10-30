"""Saved-output investigator dashboard; no model imports or training."""

import json
import sqlite3
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TGNN Fraud Investigation", page_icon="≡ƒöÄ", layout="wide")

# --------------------------------------------------------------------------------------
# Visual theme (presentation only; no data or model logic lives here)
# --------------------------------------------------------------------------------------
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --brand: #4f46e5;
  --brand-2: #7c3aed;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #e6e8f0;
  --card: #ffffff;
  --bg: #f5f6fb;
  --danger: #ef4444;
  --warn: #f59e0b;
  --ok: #10b981;
  --radius: 16px;
  --shadow: 0 1px 2px rgba(15, 23, 42, .04), 0 8px 24px rgba(79, 70, 229, .06);
}

/* Base ------------------------------------------------------------------ */
.stApp {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
  background:
    radial-gradient(1200px 400px at 85% -10%, rgba(124, 58, 237, .10), transparent 60%),
    radial-gradient(900px 380px at -10% 0%, rgba(79, 70, 229, .08), transparent 60%),
    var(--bg);
}
.stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1280px; }
footer, .stDeployButton, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
hr { border-color: var(--line); }

/* Hero ------------------------------------------------------------------ */
.hero {
  position: relative; overflow: hidden;
  padding: 30px 34px; border-radius: 22px; color: #fff; margin-bottom: 18px;
  background: linear-gradient(120deg, #4338ca 0%, #6d28d9 55%, #9333ea 100%);
  box-shadow: 0 18px 40px rgba(79, 70, 229, .28);
}
.hero::before, .hero::after {
  content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,.10);
}
.hero::before { width: 260px; height: 260px; right: -60px; top: -110px; }
.hero::after  { width: 160px; height: 160px; right: 160px; bottom: -90px; background: rgba(255,255,255,.07); }
.hero .eyebrow {
  display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; padding: 5px 11px; border-radius: 999px;
  background: rgba(255,255,255,.16); margin-bottom: 12px;
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.02em; color: #fff; padding: 0; }
.hero p  { margin: 8px 0 0; font-size: .98rem; color: rgba(255,255,255,.82); }

/* Chips ----------------------------------------------------------------- */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 16px; }
.chip {
  display: inline-flex; align-items: center; gap: 7px; padding: 6px 13px;
  background: var(--card); border: 1px solid var(--line); border-radius: 999px;
  font-size: 12.5px; color: var(--muted); box-shadow: 0 1px 2px rgba(15,23,42,.03);
}
.chip b { color: var(--ink); font-weight: 600; }
.chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
.chip .dot.warn { background: var(--warn); }

/* Section titles -------------------------------------------------------- */
.section { margin: 26px 0 12px; }
.section h3 { margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: -.01em; color: var(--ink); padding: 0; }
.section span { font-size: .86rem; color: var(--muted); }

/* Metric cards ---------------------------------------------------------- */
[data-testid="stMetric"] {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px; box-shadow: var(--shadow); position: relative; overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, var(--brand), var(--brand-2));
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(79,70,229,.14); }
[data-testid="stMetricLabel"] p {
  font-size: 12px !important; font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted);
}
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -.02em; color: var(--ink); }

/* Sidebar --------------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: #ffffff; border-right: 1px solid var(--line);
}
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 14px; }
.brand .logo {
  width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center; font-size: 20px;
  background: linear-gradient(135deg, var(--brand), var(--brand-2)); box-shadow: 0 8px 18px rgba(79,70,229,.3);
}
.brand b { display: block; font-size: 15px; color: var(--ink); letter-spacing: -.01em; }
.brand small { color: var(--muted); font-size: 12px; }
.side-note { font-size: 11.5px; color: var(--muted); line-height: 1.5; padding-top: 14px; border-top: 1px solid var(--line); margin-top: 18px; }

/* Inputs ---------------------------------------------------------------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input {
  border-radius: 12px !important; border-color: var(--line) !important; background: #fff !important;
}
[data-baseweb="select"] > div:hover, [data-baseweb="input"] > div:hover { border-color: #c7c9f5 !important; }
.stTextInput input:focus { box-shadow: 0 0 0 3px rgba(79,70,229,.15) !important; }
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: 13px; color: #334155; }

/* Buttons --------------------------------------------------------------- */
.stButton > button {
  border: 0; border-radius: 12px; padding: .6rem 1.3rem; font-weight: 600; color: #fff;
  background: linear-gradient(120deg, var(--brand), var(--brand-2));
  box-shadow: 0 8px 20px rgba(79,70,229,.28); transition: transform .15s ease, box-shadow .15s ease;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(79,70,229,.36); color: #fff; }
.stButton > button:active { transform: translateY(0); color: #fff; }

/* Tables, alerts, expanders, tabs -------------------------------------- */
[data-testid="stDataFrame"] {
  border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden;
  background: var(--card); box-shadow: var(--shadow);
}
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important; border-radius: var(--radius) !important;
  background: var(--card); box-shadow: var(--shadow); overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600; }
[data-testid="stPlotlyChart"] {
  background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 8px; box-shadow: var(--shadow);
}
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { border-radius: 10px 10px 0 0; font-weight: 600; padding: 10px 16px; }
[data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--brand), var(--brand-2)); height: 3px; }
[data-testid="stCaptionContainer"] { color: var(--muted); }

/* KPI strip ------------------------------------------------------------- */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 6px 0 14px; }
.kpi { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--shadow); }
.kpi small { display: block; font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.kpi b { display: block; margin-top: 4px; font-size: 1.45rem; font-weight: 800; letter-spacing: -.02em; color: var(--ink); }

/* Detail cards ---------------------------------------------------------- */
.card { background: var(--card); border: 1px solid var(--line); border-radius: var(--radius); padding: 20px 22px; box-shadow: var(--shadow); height: 100%; }
.card .title { font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
.flow { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.acct { flex: 1 1 140px; min-width: 130px; padding: 12px 14px; border-radius: 12px; background: #f8f9fe; border: 1px solid var(--line); }
.acct small { display: block; font-size: 10.5px; font-weight: 700; letter-spacing: .1em; color: var(--muted); }
.acct b { display: block; margin-top: 3px; font-size: .95rem; color: var(--ink); word-break: break-all; }
.acct.from { border-left: 4px solid var(--warn); }
.acct.to   { border-left: 4px solid var(--danger); }
.pipe { flex: 1 1 120px; text-align: center; }
.pipe .amt { font-weight: 800; font-size: 1.05rem; color: var(--brand); letter-spacing: -.01em; }
.pipe .line { height: 2px; margin: 8px 0 0; position: relative; background: linear-gradient(90deg, var(--warn), var(--danger)); }
.pipe .line::after { content: ""; position: absolute; right: -1px; top: -4px; border-left: 8px solid var(--danger); border-top: 5px solid transparent; border-bottom: 5px solid transparent; }
.meta { margin-top: 14px; font-size: 13px; color: var(--muted); }
.meta b { color: var(--ink); font-weight: 600; }

.score-num { font-size: 2.6rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; }
.score-num.hot { color: var(--danger); } .score-num.cool { color: var(--ok); }
.pill { display: inline-block; margin-left: 10px; padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 700; vertical-align: middle; }
.pill.hot { background: #fee2e2; color: #b91c1c; } .pill.cool { background: #d1fae5; color: #047857; }
.meter { position: relative; height: 10px; border-radius: 999px; background: #eceefa; margin: 18px 0 8px; }
.meter .fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #10b981, #f59e0b 55%, #ef4444); }
.meter .tick { position: absolute; top: -6px; width: 3px; height: 22px; border-radius: 2px; background: var(--ink); }
.meter-legend { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); }

.summary-box {
  background: linear-gradient(120deg, rgba(79,70,229,.07), rgba(124,58,237,.07));
  border: 1px solid #dcdcfb; border-radius: 14px; padding: 14px 18px; margin: 4px 0 16px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<span>{escape(subtitle)}</span>" if subtitle else ""
    st.markdown(f'<div class="section"><h3>{escape(title)}</h3>{sub}</div>', unsafe_allow_html=True)


def score_card(score: float, threshold: float) -> str:
    hot = score >= threshold
    state, label = ("hot", "Above threshold") if hot else ("cool", "Below threshold")
    fill = max(0.0, min(1.0, score)) * 100
    tick = max(0.0, min(1.0, threshold)) * 100
    return (
        '<div class="card"><div class="title">Model risk score</div>'
        f'<div><span class="score-num {state}">{score:.4f}</span>'
        f'<span class="pill {state}">{label}</span></div>'
        f'<div class="meter"><div class="fill" style="width:{fill:.1f}%"></div>'
        f'<div class="tick" style="left:calc({tick:.1f}% - 1px)"></div></div>'
        f'<div class="meter-legend"><span>0</span><span>Frozen threshold {threshold:.4f}</span>'
        "<span>1</span></div></div>"
    )


# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.sidebar.markdown(
    '<div class="brand"><div class="logo">≡ƒöÄ</div><div><b>TGNN Fraud</b>'
    "<small>Investigation console</small></div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><span class="eyebrow">Temporal Graph Network ┬╖ AML</span>'
    "<h1>Transaction Investigation</h1>"
    "<p>Saved model scores ┬╖ Historical transaction context ┬╖ IBM synthetic AML</p></div>",
    unsafe_allow_html=True,
)

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
source_label = "Test fixture" if manifest["source_kind"] == "test_fixture" else "IBM synthetic AML"
st.markdown(
    '<div class="chips">'
    f'<span class="chip"><span class="dot"></span>Model <b>{escape(str(metrics["model"]))}</b></span>'
    f'<span class="chip">Run <b>{escape(str(metrics["run_id"]))}</b></span>'
    f'<span class="chip"><span class="dot warn"></span>Data <b>{source_label}</b></span>'
    "</div>",
    unsafe_allow_html=True,
)

section("Model performance", "Held-out test split")
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
st.sidebar.markdown(
    '<div class="side-note">Scores come from saved model outputs. This console never trains '
    "or changes a model.</div>",
    unsafe_allow_html=True,
)
if alerts.empty:
    st.info("No alerts passed the saved threshold. This is a valid model output.")
    st.stop()

if page == "Alert queue":
    section("Alert queue", "Filter and prioritise transactions flagged by the model")
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        floor = st.slider("Minimum displayed risk score", 0.0, 1.0, 0.0, 0.01)
    with f2:
        account = st.text_input("Account node ID filter", placeholder="Optional integer ID")
    with f3:
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
    top_score = f"{view.score.max():.3f}" if len(view) else "ΓÇô"
    pending = int((view.status == "new").sum()) if len(view) else 0
    st.markdown(
        '<div class="kpis">'
        f'<div class="kpi"><small>Matching alerts</small><b>{len(view):,}</b></div>'
        f'<div class="kpi"><small>Highest score</small><b>{top_score}</b></div>'
        f'<div class="kpi"><small>Awaiting review</small><b>{pending:,}</b></div>'
        f'<div class="kpi"><small>Threshold</small><b>{metrics["threshold"]:.3f}</b></div>'
        "</div>",
        unsafe_allow_html=True,
    )
    column_config = {
        "score": st.column_config.ProgressColumn(
            "Risk score", min_value=0.0, max_value=1.0, format="%.3f"
        ),
        "alert_id": st.column_config.NumberColumn("Alert", format="%d"),
        "edge_id": st.column_config.NumberColumn("Edge", format="%d"),
        "amount": st.column_config.NumberColumn("Amount", format="%.2f"),
    }
    column_config = {k: v for k, v in column_config.items() if k in columns}
    st.dataframe(
        view[columns], use_container_width=True, hide_index=True, column_config=column_config
    )
    st.caption("Choose Alert detail in the sidebar to inspect a transaction.")
    st.caption(f"{len(view)} alerts match your filters. Scores are not calibrated probabilities.")
else:
    alert_labels = {
        int(row.alert_id): f"Alert {int(row.alert_id)} | edge {int(row.edge_id)}"
        for row in alerts.itertuples(index=False)
    }
    section("Alert detail", "Inspect one flagged transaction and its surrounding network")
    selection = st.selectbox("Alert", list(alert_labels.values()))
    alert_id = next(key for key, label in alert_labels.items() if label == selection)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT json FROM explanations WHERE alert_id=?", (int(alert_id),)
        ).fetchone()
    context = json.loads(row[0])
    account_labels = {node["node_id"]: node["label"] for node in context["subgraph"]["nodes"]}
    target = context["target"]

    st.markdown(
        f'<div class="section"><h3>Transaction {escape(str(context["edge_id"]))}</h3></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="summary-box"><b>{escape(str(context["summary"]))}</b><br>'
        f'<span style="color:#64748b;font-size:.9rem">{escape(str(context["interpretation"]))}'
        "</span></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2])
    with left:
        when = escape(str(target.get("timestamp_utc", str(target["ts"]) + " dataset seconds")))
        st.markdown(
            '<div class="card"><div class="title">Money flow</div><div class="flow">'
            f'<div class="acct from"><small>SENDER</small><b>{escape(str(account_labels[target["src"]]))}</b></div>'
            f'<div class="pipe"><div class="amt">{target["amount"]:,.2f} {escape(str(target["currency"]))}</div>'
            '<div class="line"></div></div>'
            f'<div class="acct to"><small>RECIPIENT</small><b>{escape(str(account_labels[target["dst"]]))}</b></div>'
            f'</div><div class="meta"><b>Time:</b> {when} &nbsp;┬╖&nbsp; '
            f"<b>Attention:</b> {escape(str(context['attention_status']))}</div></div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(score_card(context["score"], context["threshold"]), unsafe_allow_html=True)
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
        ("Earlier transactions", historical, "#b6bdd0"),
        ("Target transaction", [target], "#ef4444"),
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
                line={"color": color, "width": 1.6 if name.startswith("Earlier") else 4},
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
    node_colors = [
        "#ef4444" if n == target["dst"] else "#f59e0b" if n == target["src"] else "#6366f1"
        for n in graph
    ]
    figure.add_trace(
        go.Scatter(
            x=[positions[n][0] for n in graph],
            y=[positions[n][1] for n in graph],
            mode="markers+text",
            text=[str(n) for n in graph],
            textposition="top center",
            textfont={"size": 11, "color": "#475569"},
            hovertext=[labels.get(n, str(n)) for n in graph],
            marker={
                "size": 20,
                "color": node_colors,
                "line": {"color": "#ffffff", "width": 2.5},
            },
            name="Accounts (amber = sender, red = recipient)",
        )
    )
    figure.update_layout(
        height=480,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#334155"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0},
        hoverlabel={"bgcolor": "#0f172a", "font_color": "#ffffff"},
    )
    graph_tab, table_tab, motif_tab = st.tabs(["Network graph", "Transactions", "Motifs"])
    with graph_tab:
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            "Lines show account relationships; the transaction table identifies transfer direction."
        )
    with table_tab:
        st.dataframe(pd.DataFrame([*historical, target]), use_container_width=True, hide_index=True)
    with motif_tab:
        st.json(context["motifs"])

    section("Analyst review", "Record your decision for this alert")
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
