#!/usr/bin/env python3
"""
Plot BWE logs for a given participant.

Usage:
    python3 plot-bwe.py <logs.txt> <participant-identity>
    python3 plot-bwe.py ../sfu/logs.txt webrtcperf-3

Generates an HTML file with interactive Plotly charts.
"""

import sys
import json
import os
import re
from collections import defaultdict
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Install plotly first: pip install plotly")
    sys.exit(1)


def parse_timestamp(line: str) -> float | None:
    """Extract timestamp from the beginning of a log line."""
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})", line)
    if not m:
        return None
    dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%f")
    return dt.timestamp()


def extract_json(line: str) -> dict | None:
    """Extract the JSON object from a log line."""
    idx = line.find("{")
    if idx == -1:
        return None
    try:
        return json.loads(line[idx:])
    except json.JSONDecodeError:
        return None


def build_pid_to_identity(lines: list[str]) -> dict[str, str]:
    """Scan all log lines to map participant IDs (PA_xxx) to identities."""
    mapping = {}
    for line in lines:
        d = extract_json(line)
        if not d:
            continue
        pid = d.get("pID") or d.get("participantID")
        identity = d.get("participant")
        if pid and identity and pid.startswith("PA_"):
            mapping[pid] = identity
    return mapping


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    log_file = sys.argv[1]
    target_participant = sys.argv[2]

    with open(log_file, "r", errors="replace") as f:
        lines = f.readlines()

    # Build pID → identity mapping
    pid_map = build_pid_to_identity(lines)

    # Parse BWE logs for the target participant
    subscriber_data = []  # (ts, channelCapacity, expectedUsage)
    track_data = defaultdict(list)  # stream_label → [(ts, spatial, temporal, bw)]

    bwe_lines = [l for l in lines if "bwe-log" in l and target_participant in l]

    for line in bwe_lines:
        ts = parse_timestamp(line)
        if ts is None:
            continue
        d = extract_json(line)
        if not d:
            continue

        if "channelCapacityBps" in d:
            subscriber_data.append((
                ts,
                d["channelCapacityBps"] / 1000,
                d["expectedUsageBps"] / 1000,
            ))
        elif "currentSpatial" in d:
            pub_id = d.get("publisherID", "?")
            pub_name = pid_map.get(pub_id, pub_id)
            track_id_short = d.get("trackID", "?")[-8:]
            label = f"{pub_name} ({track_id_short})"

            track_data[label].append((
                ts,
                d["currentSpatial"],
                d["currentTemporal"],
                d["bandwidthRequestedBps"] / 1000,
            ))

    if not subscriber_data:
        print(f"No BWE logs found for participant '{target_participant}'")
        sys.exit(1)

    # Normalize timestamps to start from 0
    t0 = subscriber_data[0][0]
    subscriber_data = [(t - t0, cap, usage) for t, cap, usage in subscriber_data]
    for label in track_data:
        track_data[label] = [(t - t0, s, tp, bw) for t, s, tp, bw in track_data[label]]

    # Sort stream labels consistently
    stream_labels = sorted(track_data.keys())
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(stream_labels)}

    # Layout: 2 shared rows (BW overview + per-stream bitrate) + 1 row per stream (spatial+temporal)
    n_streams = len(stream_labels)
    n_rows = 2 + n_streams
    subplot_titles = [
        "Estimated BW & Total Usage (Kbps)",
        "Per-Stream Bitrate (Kbps)",
    ] + [f"Layers: {label}" for label in stream_labels]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False}]] * n_rows,
    )

    # --- Plot 1: Estimated BW & Total Usage ---
    ts_sub = [d[0] for d in subscriber_data]
    capacities = [d[1] for d in subscriber_data]
    usages = [d[2] for d in subscriber_data]

    fig.add_trace(go.Scatter(
        x=ts_sub, y=capacities,
        name="Estimated BW (BWE)",
        line=dict(color="red", width=2, dash="dash"),
        legendgroup="overview",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ts_sub, y=usages,
        name="Total Usage",
        line=dict(color="black", width=2),
        legendgroup="overview",
    ), row=1, col=1)

    # --- Plot 2: Per-stream bitrate ---
    for label in stream_labels:
        data = track_data[label]
        fig.add_trace(go.Scatter(
            x=[d[0] for d in data],
            y=[d[3] for d in data],
            name=label,
            line=dict(color=color_map[label]),
            legendgroup=label,
        ), row=2, col=1)

    # --- Per-stream layer plots (one row each) ---
    for i, label in enumerate(stream_labels):
        row = 3 + i
        data = track_data[label]
        ts = [d[0] for d in data]

        fig.add_trace(go.Scatter(
            x=ts,
            y=[d[1] for d in data],
            name="Spatial",
            line=dict(color=color_map[label], width=2, shape="hv"),
            legendgroup=f"layers-{i}",
            showlegend=(i == 0),
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=ts,
            y=[d[2] for d in data],
            name="Temporal",
            line=dict(color=color_map[label], width=2, dash="dot", shape="hv"),
            legendgroup=f"layers-{i}",
            showlegend=(i == 0),
        ), row=row, col=1)

        fig.update_yaxes(dtick=1, range=[-0.5, 3.5], row=row, col=1)

    # Layout
    row_height = 180
    fig.update_layout(
        title=f"BWE Analysis — {target_participant}",
        height=max(800, 350 + n_streams * row_height),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Time (s)", row=n_rows, col=1)
    fig.update_yaxes(title_text="Kbps", row=1, col=1)
    fig.update_yaxes(title_text="Kbps", row=2, col=1)

    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"bwe-{target_participant}.html")
    fig.write_html(out_file)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
