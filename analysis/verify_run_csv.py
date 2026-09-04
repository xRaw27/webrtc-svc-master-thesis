#!/usr/bin/env python3
"""
Cross-source validation, file-based variant: SFU bwe-log vs webrtcperf
detailedStatsPath CSV. No Prometheus, no scrape-interval limits — the CSV
is written at the scenario's statsInterval (set to 0.1 for 100ms samples
matching SFU BWE_LOG_INTERVAL).

Usage:
    python3 verify_run_csv.py <sfu_log> <csv> <participant>
    python3 verify_run_csv.py logs/sfu-20260602-...log results/results.csv webrtcperf-0

CSV format (from webrtcperf StatsWriter):
    datetime,participantName,trackId,<statName1>,<statName2>,...
    1748880480123,Participant-000000,r-abc...-v,553942,1280,360,24.5,...
"""

import sys
import os
import re
import json
import csv
from datetime import datetime
from collections import defaultdict

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Install deps: pip install plotly")
    sys.exit(1)


def parse_timestamp(line: str):
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})", line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S.%f").timestamp()


def extract_json(line: str):
    idx = line.find("{")
    if idx < 0:
        return None
    try:
        return json.loads(line[idx:])
    except json.JSONDecodeError:
        return None


def parse_sfu_log(path: str, participant: str):
    sub = []
    tracks = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            if "bwe-log" not in line or participant not in line:
                continue
            ts = parse_timestamp(line)
            d = extract_json(line)
            if ts is None or not d:
                continue
            if "channelCapacityBps" in d:
                sub.append((
                    ts,
                    d["channelCapacityBps"] / 1000,
                    d["expectedUsageBps"] / 1000,
                ))
            elif "currentSpatial" in d:
                pub = d.get("publisherID", "?")
                tid = d.get("trackID", "?")[-8:]
                label = f"{pub} ({tid})"
                tracks[label].append((
                    ts,
                    d["bandwidthRequestedBps"] / 1000,
                    d.get("forwardedWidth", 0),
                    d.get("forwardedHeight", 0),
                    d.get("forwardedFps", 0.0),
                ))
    return sub, tracks


def parse_csv(path: str, prom_participant: str, t_start: float, t_end: float):
    """Returns {trackId: [(ts_sec, bitrate_bps, width, height, fps)]}, filtered
    to the SFU log window with small padding."""
    by_track = defaultdict(list)
    pad = 5.0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("participantName") != prom_participant:
                continue
            try:
                ts = float(row["datetime"]) / 1000.0
            except (KeyError, ValueError):
                continue
            if ts < t_start - pad or ts > t_end + pad:
                continue

            def num(field):
                v = row.get(field, "")
                if v == "" or v is None:
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            tid = row.get("trackId", "?")
            by_track[tid].append((
                ts,
                num("videoRecvBitrates"),
                num("videoRecvWidth"),
                num("videoRecvHeight"),
                num("videoRecvFps"),
            ))
    return by_track


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    log_path = sys.argv[1]
    csv_path = sys.argv[2]
    participant = sys.argv[3]

    sub, tracks = parse_sfu_log(log_path, participant)
    if not sub:
        print(f"No bwe-log entries for {participant} in {log_path}")
        sys.exit(1)

    t_start, t_end = sub[0][0], sub[-1][0]
    t0 = t_start

    m = re.match(r"webrtcperf-(\d+)$", participant)
    prom_participant = f"Participant-{int(m.group(1)):06d}" if m else participant

    print(f"participant={participant} (csv={prom_participant}) window="
          f"{datetime.fromtimestamp(t_start):%H:%M:%S}–{datetime.fromtimestamp(t_end):%H:%M:%S} "
          f"sfu-samples={len(sub)} sfu-track-labels={len(tracks)}")

    by_track = parse_csv(csv_path, prom_participant, t_start, t_end)
    print(f"csv tracks for {prom_participant}: {len(by_track)}")
    for tid, rows in by_track.items():
        nonnull_w = sum(1 for r in rows if r[2] is not None)
        nonnull_fps = sum(1 for r in rows if r[4] is not None)
        print(f"  {tid[-8:]}: n={len(rows)} width_samples={nonnull_w} fps_samples={nonnull_fps}")

    if not by_track:
        print("No matching CSV rows. Check participantName mapping and time window.")
        sys.exit(2)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "BWE & total throughput (Kbps) — SFU model vs client-received Σ",
            "Decoded resolution (px) — client frameWidth (solid) vs SFU forwardedWidth (dotted)",
            "Decoded FPS — client framesPerSecond (solid) vs SFU forwardedFps (dotted)",
        ],
    )

    # --- Panel 1: BWE + expected + sum of client bitrates ---
    ts_sub = [t - t0 for t, _, _ in sub]
    fig.add_trace(go.Scatter(
        x=ts_sub, y=[c for _, c, _ in sub],
        name="BWE estimate (SFU)",
        line=dict(color="red", width=2, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts_sub, y=[u for _, _, u in sub],
        name="Expected usage (SFU Σ bandwidthRequested)",
        line=dict(color="black", width=2),
    ), row=1, col=1)

    # Sum client bitrates across tracks by aligning timestamps to nearest 100ms.
    by_bin = defaultdict(float)
    for rows in by_track.values():
        for ts, br, _, _, _ in rows:
            if br is None:
                continue
            by_bin[round(ts, 1)] += br
    xs_sum = sorted(by_bin.keys())
    fig.add_trace(go.Scatter(
        x=[x - t0 for x in xs_sum],
        y=[by_bin[x] / 1000 for x in xs_sum],
        name="Total received (client Σ)",
        line=dict(color="green", width=2),
    ), row=1, col=1)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    # --- Panel 2: width ---
    for i, (tid, rows) in enumerate(sorted(by_track.items())):
        xs = [r[0] - t0 for r in rows if r[2] is not None]
        ys = [r[2] for r in rows if r[2] is not None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"client width {tid[-8:]}",
            line=dict(color=colors[i % len(colors)], width=2),
        ), row=2, col=1)
    sfu_offset = len(by_track)
    for i, (label, data) in enumerate(sorted(tracks.items())):
        xs = [t - t0 for t, _, _, _, _ in data]
        ys = [w for _, _, w, _, _ in data]
        c = colors[(sfu_offset + i) % len(colors)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"SFU fwdW {label}",
            line=dict(color=c, width=1, dash="dot"),
        ), row=2, col=1)

    # --- Panel 3: fps ---
    for i, (tid, rows) in enumerate(sorted(by_track.items())):
        xs = [r[0] - t0 for r in rows if r[4] is not None]
        ys = [r[4] for r in rows if r[4] is not None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"client fps {tid[-8:]}",
            line=dict(color=colors[i % len(colors)], width=2),
        ), row=3, col=1)
    for i, (label, data) in enumerate(sorted(tracks.items())):
        xs = [t - t0 for t, _, _, _, _ in data]
        ys = [f for _, _, _, _, f in data]
        c = colors[(sfu_offset + i) % len(colors)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"SFU fwdFps {label}",
            line=dict(color=c, width=1, dash="dot"),
        ), row=3, col=1)

    fig.update_layout(
        title=f"Cross-source validation (CSV) — {participant}",
        height=950,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Kbps", row=1, col=1)
    fig.update_yaxes(title_text="px", row=2, col=1)
    fig.update_yaxes(title_text="fps", row=3, col=1)

    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f"verify-csv-{participant}.html")
    fig.write_html(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
