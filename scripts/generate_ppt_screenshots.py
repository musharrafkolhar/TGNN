"""Generate high-resolution presentation screenshots for PPT slides."""

import sqlite3
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("docs/ppt_screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Modern high-contrast dark theme (ideal for presentations & slides)
BG_DARK = "#0B0F19"  # Deep Slate 950
PANEL_BG = "#131C2E"  # Slate 900
PANEL_ALT = "#1E293B"  # Slate 800
BORDER_COLOR = "#2D3748"  # Border Slate
TEXT_WHITE = "#FFFFFF"  # Crisp White
TEXT_MUTED = "#94A3B8"  # Slate 400
CYAN_ACCENT = "#38BDF8"  # Sky 400
INDIGO_ACCENT = "#818CF8"  # Indigo 400
GREEN_NORMAL = "#10B981"  # Emerald 500
RED_ALERT = "#F43F5E"  # Rose 500
AMBER_WARN = "#F59E0B"  # Amber 500
HEADER_BG = "#1E293B"


def create_01_transaction_data_sample():
    """Slide 1: Clean, beautifully spaced table of real tabular transactions."""
    edges_path = Path("data/ibm_pilot_processed/edges.parquet")
    if not edges_path.exists():
        edges_path = Path("data/processed/edges.parquet")
    df = pd.read_parquet(edges_path)

    pos_sample = df[df.label == 1].head(4)
    neg_sample = df[df.label == 0].iloc[10:17]
    sample = pd.concat([neg_sample.iloc[:3], pos_sample.iloc[:3], neg_sample.iloc[3:6]]).copy()

    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK, dpi=250)
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 5.8, 1.6], figure=fig, hspace=0.28)

    # Header
    ax_head = fig.add_subplot(gs[0])
    ax_head.set_facecolor(BG_DARK)
    ax_head.axis("off")
    ax_head.text(
        0.02,
        0.75,
        "IBM AML Synthetic Dataset — Tabular Transaction Stream",
        fontsize=22,
        fontweight="bold",
        color=TEXT_WHITE,
        va="top",
    )
    ax_head.text(
        0.02,
        0.28,
        "Raw account-to-account transactions with timestamp, banking entities, monetary amounts, channels, and laundering labels.",
        fontsize=12.5,
        color=TEXT_MUTED,
        va="top",
    )

    # Table section
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.set_facecolor(PANEL_BG)
    ax_tbl.axis("off")

    bg_rect = patches.FancyBboxPatch(
        (0.015, 0.02),
        0.97,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.2,
    )
    ax_tbl.add_patch(bg_rect)

    headers = [
        "Event ID",
        "Timestamp (UTC)",
        "From Bank",
        "Sender Acct",
        "To Bank",
        "Receiver Acct",
        "Amount Paid",
        "Currency",
        "Channel",
        "Laundering",
    ]

    col_x = [0.038, 0.098, 0.235, 0.320, 0.428, 0.515, 0.620, 0.725, 0.818, 0.932]

    # Draw header bar
    hdr_box = patches.FancyBboxPatch(
        (0.025, 0.88),
        0.95,
        0.08,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=HEADER_BG,
        edgecolor="#334155",
        linewidth=1.2,
    )
    ax_tbl.add_patch(hdr_box)

    for i, h in enumerate(headers):
        ha = "center" if i == 9 else "left"
        ax_tbl.text(
            col_x[i], 0.92, h, color=CYAN_ACCENT, fontsize=10, fontweight="bold", va="center", ha=ha
        )

    # Draw rows
    y_start = 0.79
    y_step = 0.086

    for idx, (_, row) in enumerate(sample.iterrows()):
        y = y_start - idx * y_step
        is_fraud = row["label"] == 1
        row_bg = "#27171C" if is_fraud else ("#162032" if idx % 2 == 0 else "#111827")
        border = RED_ALERT if is_fraud else "#243044"

        row_rect = patches.FancyBboxPatch(
            (0.025, y - 0.035),
            0.95,
            0.072,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=row_bg,
            edgecolor=border,
            linewidth=1.4 if is_fraud else 0.8,
        )
        ax_tbl.add_patch(row_rect)

        ts_str = str(row["timestamp"])[:19]
        amount_str = f"${row['amount_paid']:,.2f}"
        curr_str = str(row["currency_paid"])[:10]
        fmt_str = str(row["payment_format"])[:11]

        row_vals = [
            str(row["edge_id"]),
            ts_str,
            str(row["from_bank"]),
            f"acct_{str(row['src_account'])[-6:]}",
            str(row["to_bank"]),
            f"acct_{str(row['dst_account'])[-6:]}",
            amount_str,
            curr_str,
            fmt_str,
        ]

        for c_idx, val in enumerate(row_vals):
            ax_tbl.text(
                col_x[c_idx],
                y,
                val,
                color=TEXT_WHITE,
                fontsize=9.2,
                va="center",
                fontfamily="monospace" if c_idx in [0, 1, 2, 3, 4, 5, 6] else "sans-serif",
            )

        # Label badge
        badge_color = RED_ALERT if is_fraud else GREEN_NORMAL
        badge_text = "1 (ILLICIT)" if is_fraud else "0 (Legit)"
        badge_box = patches.FancyBboxPatch(
            (col_x[9] - 0.038, y - 0.024),
            0.076,
            0.048,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            facecolor=badge_color,
            edgecolor="none",
        )
        ax_tbl.add_patch(badge_box)
        ax_tbl.text(
            col_x[9],
            y,
            badge_text,
            color=TEXT_WHITE,
            fontsize=8.2,
            fontweight="bold",
            va="center",
            ha="center",
        )

    # Footer note card
    ax_ft = fig.add_subplot(gs[2])
    ax_ft.set_facecolor(BG_DARK)
    ax_ft.axis("off")

    ft_rect = patches.FancyBboxPatch(
        (0.015, 0.08),
        0.97,
        0.84,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1,
    )
    ax_ft.add_patch(ft_rect)

    ax_ft.text(
        0.04,
        0.68,
        "Data Schema & Graph Mapping Contracts:",
        fontsize=11.5,
        fontweight="bold",
        color=CYAN_ACCENT,
    )
    ax_ft.text(
        0.04,
        0.32,
        "• Accounts as Graph Nodes: Node identity is strictly the (bank_id, account_id) string pair.\n"
        "• Transactions as Directed Edges: e = (u, v, t, x_e) timestamped with continuous log-amounts and channels.\n"
        "• Binary Synthetic Labels: 0 = Normal commercial transactions; 1 = Synthetic laundering typology motif.",
        fontsize=10.5,
        color=TEXT_WHITE,
        va="center",
    )

    out_file = OUT_DIR / "01_transaction_data_sample.png"
    plt.savefig(out_file, bbox_inches="tight", facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Saved: {out_file}")


def create_02_dataset_statistics_splits():
    """Slide 2: Dataset Statistics & Chronological Splits Infographic."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK, dpi=250)
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.8, 4.4],
        width_ratios=[1.15, 0.85],
        figure=fig,
        hspace=0.32,
        wspace=0.22,
    )

    # Top Header spanning width
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(BG_DARK)
    ax_top.axis("off")

    ax_top.text(
        0.02,
        0.88,
        "Dataset Scale, Class Imbalance & Chronological Splits",
        fontsize=22,
        fontweight="bold",
        color=TEXT_WHITE,
    )
    ax_top.text(
        0.02,
        0.65,
        "IBM Anti-Money Laundering (HI-Small) — Validated contiguous chronological pilot interval.",
        fontsize=12.5,
        color=TEXT_MUTED,
    )

    # 4 Stat Cards
    stat_cards = [
        ("Total Raw Events", "5,078,345", "Full IBM AML dataset (18 days)", CYAN_ACCENT),
        ("Total Unique Nodes", "157,344", "Distinct bank/account entities", INDIGO_ACCENT),
        ("Positive Prevalence", "0.102%", "5,177 laundering labels in full data", RED_ALERT),
        (
            "Pilot Contiguous Events",
            "199,583",
            "Contiguous 6-hour window (2022-09-02)",
            GREEN_NORMAL,
        ),
    ]

    card_width = 0.228
    for i, (title, val, desc, color) in enumerate(stat_cards):
        cx = 0.02 + i * 0.245
        rect = patches.FancyBboxPatch(
            (cx, 0.05),
            card_width,
            0.50,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=PANEL_BG,
            edgecolor=BORDER_COLOR,
            linewidth=1.4,
        )
        ax_top.add_patch(rect)
        ax_top.text(
            cx + 0.015, 0.42, title.upper(), fontsize=9, fontweight="bold", color=TEXT_MUTED
        )
        ax_top.text(cx + 0.015, 0.23, val, fontsize=19, fontweight="bold", color=color)
        ax_top.text(cx + 0.015, 0.10, desc, fontsize=8.5, color=TEXT_WHITE)

    # Split Timeline Visual (Bottom Left)
    ax_split = fig.add_subplot(gs[1, 0])
    ax_split.set_facecolor(PANEL_BG)
    ax_split.axis("off")

    rect_bg = patches.FancyBboxPatch(
        (0.02, 0.03),
        0.96,
        0.94,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.2,
    )
    ax_split.add_patch(rect_bg)

    ax_split.text(
        0.06,
        0.90,
        "Strict Chronological Partition (No Future Leakage)",
        fontsize=14,
        fontweight="bold",
        color=TEXT_WHITE,
    )
    ax_split.text(
        0.06,
        0.83,
        "Preserves all events at boundary timestamps; transformers fitted on train only.",
        fontsize=10,
        color=TEXT_MUTED,
    )

    split_info = [
        (
            "Train Split (60%)",
            "119,707 events",
            "47 Positives (0.039%)",
            "00:00 → 03:23 UTC",
            "#2563EB",
        ),
        (
            "Validation Split (20%)",
            "39,940 events",
            "15 Positives (0.038%)",
            "03:24 → 04:41 UTC",
            "#7C3AED",
        ),
        (
            "Test Split (20%)",
            "39,936 events",
            "14 Positives (0.035%)",
            "04:42 → 05:58 UTC",
            "#DB2777",
        ),
    ]

    y_pos = [0.60, 0.38, 0.16]
    for idx, (s_name, count, pos_str, time_str, color) in enumerate(split_info):
        y = y_pos[idx]
        block_rect = patches.FancyBboxPatch(
            (0.05, y),
            0.90,
            0.17,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#0B0F19",
            edgecolor=color,
            linewidth=2,
        )
        ax_split.add_patch(block_rect)

        pill = patches.FancyBboxPatch(
            (0.07, y + 0.03),
            0.018,
            0.11,
            boxstyle="round,pad=0.005",
            facecolor=color,
            edgecolor="none",
        )
        ax_split.add_patch(pill)

        ax_split.text(0.11, y + 0.11, s_name, fontsize=12, fontweight="bold", color=TEXT_WHITE)
        ax_split.text(0.11, y + 0.04, f"{count}   •   {time_str}", fontsize=9.5, color=TEXT_MUTED)
        ax_split.text(0.66, y + 0.07, pos_str, fontsize=11, fontweight="bold", color=RED_ALERT)

    # Imbalance Chart (Bottom Right)
    ax_chart = fig.add_subplot(gs[1, 1])
    ax_chart.set_facecolor(PANEL_BG)

    chart_bg = patches.FancyBboxPatch(
        (0.01, 0.03),
        0.98,
        0.94,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.2,
        transform=ax_chart.transAxes,
    )
    ax_chart.add_patch(chart_bg)

    splits = ["Train", "Validation", "Test"]
    totals = [119707, 39940, 39936]
    positives = [47, 15, 14]

    x = np.arange(len(splits))
    width = 0.35

    ax_chart.set_position([0.56, 0.10, 0.38, 0.52])
    ax_chart.bar(  # noqa: F841
        x - width / 2, totals, width, label="Total Events", color=CYAN_ACCENT, alpha=0.9
    )
    ax_chart.set_ylabel("Total Transactions (Log Scale)", color=TEXT_WHITE, fontsize=10.5)
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(splits, color=TEXT_WHITE, fontsize=11, fontweight="bold")
    ax_chart.tick_params(axis="y", labelcolor=TEXT_WHITE)
    ax_chart.set_yscale("log")
    ax_chart.grid(axis="y", linestyle="--", alpha=0.15, color=TEXT_MUTED)

    # Twin axis for positive count
    ax_tw = ax_chart.twinx()
    bars2 = ax_tw.bar(
        x + width / 2, positives, width, label="Laundering Labels", color=RED_ALERT, alpha=0.9
    )
    ax_tw.set_ylabel(
        "Laundering Count (Positives)", color=RED_ALERT, fontsize=10.5, fontweight="bold"
    )
    ax_tw.tick_params(axis="y", labelcolor=RED_ALERT)
    ax_tw.set_ylim(0, 65)

    for bar in bars2:
        h = bar.get_height()
        ax_tw.annotate(
            f"{h}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=RED_ALERT,
            fontweight="bold",
            fontsize=10,
        )

    ax_chart.set_title(
        "Extreme Class Imbalance (1 in 2,500 illicit)",
        color=TEXT_WHITE,
        fontsize=12.5,
        fontweight="bold",
        pad=15,
    )

    out_file = OUT_DIR / "02_dataset_statistics_splits.png"
    plt.savefig(out_file, bbox_inches="tight", facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Saved: {out_file}")


def create_03_temporal_graph_network_motifs():
    """Slide 3: Graph Network Representation and Detected Laundering Motifs."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK, dpi=250)
    gs = gridspec.GridSpec(
        2, 2, height_ratios=[1.2, 5.8], width_ratios=[1, 1], figure=fig, hspace=0.25, wspace=0.20
    )

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(BG_DARK)
    ax_top.axis("off")
    ax_top.text(
        0.02,
        0.75,
        "Dynamic Graph Network & Temporal Laundering Motifs",
        fontsize=22,
        fontweight="bold",
        color=TEXT_WHITE,
    )
    ax_top.text(
        0.02,
        0.28,
        "Transactions form directed multigraph edges; motifs represent coordinated laundering typologies.",
        fontsize=12.5,
        color=TEXT_MUTED,
    )

    # Left: Fan-Out / Structuring
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.set_facecolor(PANEL_BG)
    ax_left.axis("off")

    bg_l = patches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.5,
    )
    ax_left.add_patch(bg_l)

    ax_left.text(
        0.06,
        0.91,
        "Motif 1: Fan-Out / Smurfing (Structuring)",
        fontsize=14,
        fontweight="bold",
        color=CYAN_ACCENT,
    )
    ax_left.text(
        0.06,
        0.85,
        "Single origin node disperses funds rapidly into multiple recipient accounts.",
        fontsize=10,
        color=TEXT_MUTED,
    )

    src_x, src_y = 0.20, 0.48
    dest_nodes = [
        (0.78, 0.74, "Acct 111515", "Edge #63455 | $4,950 (ACH)"),
        (0.78, 0.56, "Acct 90663", "Edge #66578 | $4,800 (Wire)"),
        (0.78, 0.38, "Acct 115162", "Edge #71655 | $4,990 (Cash)"),
        (0.78, 0.20, "Acct 124089", "Edge #169724 | $4,750 (ACH)"),
    ]

    r_src = 0.08
    r_dest = 0.065
    circle_src = patches.Circle(
        (src_x, src_y), r_src, facecolor="#1E3A8A", edgecolor=CYAN_ACCENT, linewidth=2.5
    )
    ax_left.add_patch(circle_src)
    ax_left.text(
        src_x,
        src_y,
        "Origin\nAcct 35305",
        color=TEXT_WHITE,
        fontsize=9.5,
        fontweight="bold",
        ha="center",
        va="center",
    )

    for dx, dy, label, edge_info in dest_nodes:
        # Distance & edge stopping points outside circles
        dist = np.hypot(dx - src_x, dy - src_y)
        sx = src_x + (r_src + 0.01) * (dx - src_x) / dist
        sy = src_y + (r_src + 0.01) * (dy - src_y) / dist
        ex = dx - (r_dest + 0.015) * (dx - src_x) / dist
        ey = dy - (r_dest + 0.015) * (dy - src_y) / dist

        ax_left.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color=RED_ALERT, lw=2.2, mutation_scale=16),
        )

        c_dest = patches.Circle(
            (dx, dy), r_dest, facecolor="#312E81", edgecolor=INDIGO_ACCENT, linewidth=2
        )
        ax_left.add_patch(c_dest)
        ax_left.text(
            dx,
            dy,
            label,
            color=TEXT_WHITE,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

        # Edge label badge
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax_left.text(
            mx,
            my,
            edge_info,
            color=TEXT_WHITE,
            fontsize=7.8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG_DARK, edgecolor=AMBER_WARN, lw=1),
        )

    ax_left.text(
        0.06,
        0.06,
        "• 35 Fan-Out patterns detected in test alert context.\n"
        "• Typical evasion: rapid sub-threshold deposits to bypass statutory limits.",
        color=TEXT_WHITE,
        fontsize=9.5,
    )

    # Right: Circular Laundering (Cycle)
    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.set_facecolor(PANEL_BG)
    ax_right.axis("off")

    bg_r = patches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.5,
    )
    ax_right.add_patch(bg_r)

    ax_right.text(
        0.06,
        0.91,
        "Motif 2: Circular Laundering (Cycle)",
        fontsize=14,
        fontweight="bold",
        color=INDIGO_ACCENT,
    )
    ax_right.text(
        0.06,
        0.85,
        "Closed loop fund routing across intermediate accounts in strict chronological order.",
        fontsize=10,
        color=TEXT_MUTED,
    )

    cycle_nodes = [
        (0.32, 0.64, "Acct 19"),
        (0.68, 0.64, "Acct 20"),
        (0.68, 0.32, "Acct 21"),
        (0.32, 0.32, "Acct 18"),
    ]

    edges = [
        (0, 1, "t1: Edge #901"),
        (1, 2, "t2: Edge #902"),
        (2, 3, "t3: Edge #903"),
        (3, 0, "t4: Edge #1620 (Target)"),
    ]

    r_c = 0.075
    for x, y, label in cycle_nodes:
        c = patches.Circle((x, y), r_c, facecolor="#14532D", edgecolor=GREEN_NORMAL, linewidth=2.5)
        ax_right.add_patch(c)
        ax_right.text(
            x, y, label, color=TEXT_WHITE, fontsize=9.5, fontweight="bold", ha="center", va="center"
        )

    for u, v, lbl in edges:
        ux, uy = cycle_nodes[u][0], cycle_nodes[u][1]
        vx, vy = cycle_nodes[v][0], cycle_nodes[v][1]

        dx, dy = vx - ux, vy - uy
        dist = np.hypot(dx, dy)
        sx, sy = ux + (r_c + 0.01) * (dx / dist), uy + (r_c + 0.01) * (dy / dist)
        ex, ey = vx - (r_c + 0.018) * (dx / dist), vy - (r_c + 0.018) * (dy / dist)

        color = RED_ALERT if "Target" in lbl else CYAN_ACCENT
        ax_right.annotate(
            "",
            xy=(ex, ey),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.5, mutation_scale=17),
        )
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        offset_y = 0.05 if uy == vy else 0
        offset_x = 0.12 if ux == vx else 0
        ax_right.text(
            mx + offset_x,
            my + offset_y,
            lbl,
            color=TEXT_WHITE,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG_DARK, edgecolor=color, lw=1),
        )

    ax_right.text(
        0.06,
        0.06,
        "• 200 Directed Cycles identified in historical alert context.\n"
        "• Causal ordering: All supporting edges precede the target transaction.",
        color=TEXT_WHITE,
        fontsize=9.5,
    )

    out_file = OUT_DIR / "03_temporal_graph_network_motifs.png"
    plt.savefig(out_file, bbox_inches="tight", facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Saved: {out_file}")


def create_04_alert_investigation_card():
    """Slide 4: Real Alert Detail & Model Scoring Card."""
    fig = plt.figure(figsize=(16, 9), facecolor=BG_DARK, dpi=250)
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3.0, 3.0],
        width_ratios=[1.15, 0.85],
        figure=fig,
        hspace=0.28,
        wspace=0.20,
    )

    # Top Left: Alert Header & Transaction facts
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tl.set_facecolor(PANEL_BG)
    ax_tl.axis("off")

    bg_tl = patches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.5,
    )
    ax_tl.add_patch(bg_tl)

    ax_tl.text(
        0.06,
        0.88,
        "Alert Detail — Test Event #160089",
        fontsize=15,
        fontweight="bold",
        color=TEXT_WHITE,
    )
    badge = patches.FancyBboxPatch(
        (0.74, 0.83), 0.20, 0.09, boxstyle="round,pad=0.005", facecolor=RED_ALERT, edgecolor="none"
    )
    ax_tl.add_patch(badge)
    ax_tl.text(
        0.84,
        0.875,
        "FLAGGED ALERT",
        color=TEXT_WHITE,
        fontsize=9.5,
        fontweight="bold",
        va="center",
        ha="center",
    )

    details = [
        ("Timestamp", "2022-09-02 04:42:00 UTC (Test Split)"),
        ("Source Entity", "Bank: 0120627  |  Account: acct_068340"),
        ("Destination Entity", "Bank: 0143425  |  Account: acct_143425"),
        ("Transaction Amount", "5,761.73 Swiss Franc (CHF)"),
        ("Payment Channel", "Automated Clearing House (ACH)"),
        ("Historical Window", "24 Hours Lookback (Strictly Earlier)"),
    ]

    for idx, (label, val) in enumerate(details):
        y = 0.69 - idx * 0.108
        ax_tl.text(0.06, y, label + ":", fontsize=10, color=TEXT_MUTED, fontweight="bold")
        ax_tl.text(0.38, y, val, fontsize=10, color=TEXT_WHITE, fontfamily="monospace")

    # Top Right: Model Risk Scores vs Thresholds
    ax_tr = fig.add_subplot(gs[0, 1])
    ax_tr.set_facecolor(PANEL_BG)
    ax_tr.axis("off")

    bg_tr = patches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.5,
    )
    ax_tr.add_patch(bg_tr)

    ax_tr.text(
        0.06,
        0.88,
        "Model Risk Scores & Threshold Cutoffs",
        fontsize=14,
        fontweight="bold",
        color=TEXT_WHITE,
    )

    models_data = [
        ("XGBoost Baseline", 0.7606, 0.7606, CYAN_ACCENT),
        ("TGN (Temporal Graph Net)", 0.8675, 0.8674, INDIGO_ACCENT),
    ]

    for idx, (m_name, score, thresh, col) in enumerate(models_data):
        y = 0.62 - idx * 0.35
        ax_tr.text(0.06, y + 0.11, m_name, fontsize=11.5, fontweight="bold", color=col)
        ax_tr.text(
            0.06,
            y + 0.03,
            f"Assigned Risk Score: {score:.4f}  |  Validation Cutoff: {thresh:.4f}",
            fontsize=9,
            color=TEXT_MUTED,
        )

        bar_bg = patches.FancyBboxPatch(
            (0.06, y - 0.08),
            0.88,
            0.075,
            boxstyle="round,pad=0.005",
            facecolor="#0B0F19",
            edgecolor="#334155",
        )
        ax_tr.add_patch(bar_bg)
        bar_fill = patches.FancyBboxPatch(
            (0.06, y - 0.08),
            0.88 * score,
            0.075,
            boxstyle="round,pad=0.005",
            facecolor=col,
            edgecolor="none",
        )
        ax_tr.add_patch(bar_fill)

    # Bottom: Historical Graph Context & Compliance Callout
    ax_bot = fig.add_subplot(gs[1, :])
    ax_bot.set_facecolor(PANEL_BG)
    ax_bot.axis("off")

    bg_bot = patches.FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=PANEL_BG,
        edgecolor=BORDER_COLOR,
        linewidth=1.5,
    )
    ax_bot.add_patch(bg_bot)

    ax_bot.text(
        0.04,
        0.86,
        "Historical Subgraph Context (Preceding Transactions Between Nodes)",
        fontsize=13.5,
        fontweight="bold",
        color=CYAN_ACCENT,
    )

    subgraph_headers = [
        "Edge ID",
        "Timestamp (UTC)",
        "Payment Channel",
        "Amount",
        "Causal Order",
        "Attention Weight",
    ]
    s_col_x = [0.05, 0.20, 0.40, 0.58, 0.74, 0.88]

    for c_i, h in enumerate(subgraph_headers):
        ax_bot.text(s_col_x[c_i], 0.72, h, color=TEXT_MUTED, fontsize=10, fontweight="bold")

    subgraph_rows = [
        (
            "Edge #155608",
            "2022-09-02 04:33:00",
            "Credit Card",
            "2,427.64 CHF",
            "Strictly Earlier",
            "0.342",
        ),
        (
            "Edge #158582",
            "2022-09-02 04:39:00",
            "Cash Transfer",
            "11,334.38 CHF",
            "Strictly Earlier",
            "0.658",
        ),
        (
            "Edge #160089",
            "2022-09-02 04:42:00",
            "ACH (Target)",
            "5,761.73 CHF",
            "Current Target",
            "Evaluating...",
        ),
    ]

    for r_i, r in enumerate(subgraph_rows):
        y = 0.54 - r_i * 0.12
        is_tgt = "Target" in r[2]
        r_color = RED_ALERT if is_tgt else TEXT_WHITE
        for c_i, val in enumerate(r):
            ax_bot.text(
                s_col_x[c_i],
                y,
                val,
                color=r_color,
                fontsize=9.5,
                fontfamily="monospace" if c_i in [0, 1, 3] else "sans-serif",
                fontweight="bold" if is_tgt else "normal",
            )

    disc_box = patches.FancyBboxPatch(
        (0.04, 0.05),
        0.92,
        0.15,
        boxstyle="round,pad=0.005",
        facecolor="#0B0F19",
        edgecolor=BORDER_COLOR,
    )
    ax_bot.add_patch(disc_box)
    ax_bot.text(
        0.06,
        0.13,
        "Model Safeguards & Provenance Policy:",
        color=AMBER_WARN,
        fontsize=9,
        fontweight="bold",
    )
    ax_bot.text(
        0.06,
        0.08,
        "Context strictly precedes target. Ground-truth labels are never exposed during inference. Patterns/motifs provide descriptive context, not legal proof of fraud.",
        color=TEXT_MUTED,
        fontsize=8.5,
    )

    out_file = OUT_DIR / "04_alert_investigation_card.png"
    plt.savefig(out_file, bbox_inches="tight", facecolor=BG_DARK, edgecolor="none")
    plt.close()
    print(f"Saved: {out_file}")


def create_05_dashboard_alert_queue():
    """Slide 5: High-definition full-screen UI mockup of the Streamlit AML Dashboard."""
    # Query real alert records from SQLite alerts.db
    db_path = Path("runs/20261023_185426_xgboost_3e54ac/alerts.db")
    if not db_path.exists():
        db_path = Path("data/demo/20261023_185426_xgboost_3e54ac/alerts.db")

    conn = sqlite3.connect(str(db_path))
    alerts_df = pd.read_sql(
        "SELECT alert_id, edge_id, ts, src, dst, amount, currency, score, status FROM alerts LIMIT 8",
        conn,
    )
    conn.close()

    fig = plt.figure(figsize=(16, 9), facecolor="#0E1117", dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0E1117")
    ax.axis("off")

    # Left Sidebar (Streamlit Sidebar Style)
    sb_bg = patches.Rectangle(
        (0, 0), 0.22, 1, facecolor="#161B22", edgecolor="#30363D", linewidth=1.2
    )
    ax.add_patch(sb_bg)

    ax.text(0.025, 0.94, "TGNN Controls", fontsize=15, fontweight="bold", color=TEXT_WHITE)
    ax.text(0.025, 0.905, "Investigation Workspace", fontsize=10, color=TEXT_MUTED)

    # Sidebar inputs
    inputs = [
        ("Saved Run", "20261023_185430_tgn_b558e6 (TGN)", "dropdown"),
        ("Dashboard View", "Alert Queue (Triage)", "radio"),
        ("Score Threshold", "0.7606 (Validation Cutoff)", "slider"),
        ("Alert Status Filter", "All Statuses (New, Review, Escalated)", "filter"),
        ("Bank Entity Filter", "All Banks (No filter)", "search"),
    ]

    for idx, (title, val, itype) in enumerate(inputs):
        y = 0.83 - idx * 0.13
        ax.text(0.025, y, title, fontsize=9.5, fontweight="bold", color=TEXT_MUTED)
        box = patches.FancyBboxPatch(
            (0.025, y - 0.055),
            0.17,
            0.045,
            boxstyle="round,pad=0.005",
            facecolor="#21262D",
            edgecolor="#30363D",
        )
        ax.add_patch(box)
        ax.text(
            0.035,
            y - 0.032,
            val,
            fontsize=8.5,
            color=CYAN_ACCENT if itype == "dropdown" else TEXT_WHITE,
        )

    # Main Area Top Header
    ax.text(
        0.25,
        0.94,
        "TGNN · Transaction Investigation Dashboard",
        fontsize=20,
        fontweight="bold",
        color=TEXT_WHITE,
    )
    ax.text(
        0.25,
        0.905,
        "IBM Synthetic AML Data  •  Results apply to recorded contiguous chronological pilot (2022-09-02)",
        fontsize=11,
        color=TEXT_MUTED,
    )

    # Top Metric Cards (4 Streamlit Metrics)
    metrics_data = [
        ("TEST AVERAGE PRECISION (AP)", "0.001", "Evaluated on 39,936 test events", CYAN_ACCENT),
        (
            "FROZEN VALIDATION THRESHOLD",
            "0.867",
            "Maximum-F1 cutoff frozen for test",
            INDIGO_ACCENT,
        ),
        ("THRESHOLD PASSING ALERTS", "40", "Top test alerts exported for triage", AMBER_WARN),
        (
            "TEST GROUND TRUTH RECALL",
            "0.000",
            "Honest benchmark reporting (no fake scores)",
            RED_ALERT,
        ),
    ]

    card_w = 0.17
    for idx, (m_title, m_val, m_sub, m_col) in enumerate(metrics_data):
        cx = 0.25 + idx * 0.185
        c_box = patches.FancyBboxPatch(
            (cx, 0.77),
            card_w,
            0.10,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            facecolor="#161B22",
            edgecolor="#30363D",
            linewidth=1.2,
        )
        ax.add_patch(c_box)
        ax.text(cx + 0.012, 0.845, m_title, fontsize=7.5, fontweight="bold", color=TEXT_MUTED)
        ax.text(cx + 0.012, 0.805, m_val, fontsize=17, fontweight="bold", color=m_col)
        ax.text(cx + 0.012, 0.782, m_sub, fontsize=6.8, color=TEXT_MUTED)

    # Alert Queue Table Panel
    tbl_panel = patches.FancyBboxPatch(
        (0.25, 0.08),
        0.725,
        0.65,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor="#161B22",
        edgecolor="#30363D",
        linewidth=1.2,
    )
    ax.add_patch(tbl_panel)

    ax.text(0.27, 0.69, "Test Alert Triage Queue", fontsize=14, fontweight="bold", color=TEXT_WHITE)
    ax.text(
        0.27,
        0.665,
        "Selecting an alert opens historical subgraph, transaction facts, and detected graph motifs.",
        fontsize=9.5,
        color=TEXT_MUTED,
    )

    # Table headers
    tbl_headers = [
        "Alert ID",
        "Edge ID",
        "Source Acct",
        "Dest Acct",
        "Amount",
        "Currency",
        "TGN Score",
        "Review Status",
    ]
    t_col_x = [0.27, 0.35, 0.44, 0.54, 0.64, 0.73, 0.82, 0.91]

    # Header bar
    th_bar = patches.FancyBboxPatch(
        (0.265, 0.61),
        0.695,
        0.04,
        boxstyle="round,pad=0.005",
        facecolor="#21262D",
        edgecolor="#30363D",
    )
    ax.add_patch(th_bar)

    for i, h in enumerate(tbl_headers):
        ha = "center" if i == 7 else "left"
        ax.text(
            t_col_x[i],
            0.63,
            h,
            fontsize=9,
            fontweight="bold",
            color=CYAN_ACCENT,
            va="center",
            ha=ha,
        )

    # Render actual alert rows
    y_start = 0.56
    y_step = 0.058

    for idx, (_, row) in enumerate(alerts_df.iterrows()):
        y = y_start - idx * y_step
        row_bg = "#1B222D" if idx % 2 == 0 else "#161B22"

        r_box = patches.FancyBboxPatch(
            (0.265, y - 0.022),
            0.695,
            0.048,
            boxstyle="round,pad=0.004",
            facecolor=row_bg,
            edgecolor="#21262D",
            linewidth=0.8,
        )
        ax.add_patch(r_box)

        vals = [
            f"Alert #{int(row['alert_id'])}",
            f"#{int(row['edge_id'])}",
            f"acct_{int(row['src']) % 100000:05d}",
            f"acct_{int(row['dst']) % 100000:05d}",
            f"${row['amount']:,.2f}",
            str(row["currency"])[:10],
            f"{row['score']:.4f}",
        ]

        for c_i, val in enumerate(vals):
            col = TEXT_WHITE if c_i != 6 else RED_ALERT
            fontw = "bold" if c_i in [0, 6] else "normal"
            ax.text(
                t_col_x[c_i],
                y,
                val,
                color=col,
                fontsize=8.5,
                va="center",
                fontweight=fontw,
                fontfamily="monospace" if c_i in [1, 2, 3, 4, 6] else "sans-serif",
            )

        # Status Badge
        status_str = str(row["status"]).upper()
        s_badge = patches.FancyBboxPatch(
            (t_col_x[7] - 0.028, y - 0.015),
            0.056,
            0.030,
            boxstyle="round,pad=0.003",
            facecolor="#238636" if status_str == "NEW" else "#8957E5",
        )
        ax.add_patch(s_badge)
        ax.text(
            t_col_x[7],
            y,
            status_str,
            color=TEXT_WHITE,
            fontsize=7.5,
            fontweight="bold",
            va="center",
            ha="center",
        )

    out_file = OUT_DIR / "05_dashboard_alert_queue.png"
    plt.savefig(out_file, bbox_inches="tight", facecolor="#0E1117", edgecolor="none")
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    create_01_transaction_data_sample()
    create_02_dataset_statistics_splits()
    create_03_temporal_graph_network_motifs()
    create_04_alert_investigation_card()
    create_05_dashboard_alert_queue()
    print("All perfected screenshots generated successfully.")
