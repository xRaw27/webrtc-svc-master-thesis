#!/usr/bin/env python3
"""
Cross-source validation: SFU bwe-log (what the allocator says it forwards) vs
client-side WebRTC inboundRtp metrics pushed to Prometheus by webrtcperf
(what the subscriber actually received and decoded).

Usage:
    python3 verify_run.py <sfu_log> <participant>
    python3 verify_run.py logs/sfu-20260602-...log webrtcperf-0

Environment:
    PROM_URL  Prometheus base URL (default http://localhost:9090).

What it shows (per subscriber):

  1. BWE & total throughput (Kbps):
     - SFU BWE estimate (channelCapacityBps)
     - SFU expected usage (Σ bandwidthRequestedBps)
     - Client total received (Σ videoRecvBitrates across this sub's tracks)
     If expected ≈ received and ≤ BWE, the model is sound.

  2. Decoded resolution (px) per stream:
     - Solid: client frameWidth per trackId (one series per remote video)
     - Dotted: SFU forwardedWidth per (publisher, track) — what allocator
       claims it forwards, taken from publisher's currentVideoSize[currentSpatial].
     Match within ±30ms = SFU intent matches client reality.

  3. Decoded FPS per stream:
     - Solid: client framesPerSecond per trackId
     - Dotted: SFU forwardedFps from publisher's GetTemporalLayerFpsForSpatial.

Per-stream coloring on panels 2/3 isn't matched between client trackId and
SFU (publisher, track) — those are unrelated ID spaces. The dotted SFU lines
serve as a reference set; visually you should see one client solid line
per dotted SFU line.
"""

import sys
import json
import os
import re
from datetime import datetime
from collections import defaultdict

try:
    import requests
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Install deps: pip install requests plotly")
    sys.exit(1)


PROM_URL = os.environ.get("PROM_URL", "http://localhost:9090")


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
    sub = []           # (ts, channelCapacity_kbps, expectedUsage_kbps)
    tracks = defaultdict(list)  # label → [(ts, requested_kbps, fwdW, fwdH, fwdFps)]
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


def prom_query_range(query: str, start: float, end: float, step: str = "5s"):
    r = requests.get(f"{PROM_URL}/api/v1/query_range", params={
        "query": query,
        "start": int(start),
        "end": int(end),
        "step": step,
    }, timeout=10)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {payload}")
    return payload["data"]["result"]


def series_xy(series, t0):
    xs = [float(v[0]) - t0 for v in series["values"]]
    ys = [float(v[1]) for v in series["values"]]
    return xs, ys


def sum_by_timestamp(series_list):
    by_ts = defaultdict(float)
    for s in series_list:
        for ts, v in s["values"]:
            by_ts[float(ts)] += float(v)
    xs = sorted(by_ts.keys())
    return xs, [by_ts[t] for t in xs]


def parse_dt_label(s: str) -> float:
    """Webrtcperf's datetime label format: 2026-06-02T01:44:35.796Z."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def filter_current_run(series_list, t_start, max_age_sec=180):
    """Drop series from prior webrtcperf runs.

    Pushgateway never auto-expires metrics with a different label set, so when
    webrtcperf is restarted with a fresh `datetime` (its startTimestamp), all
    the previous-run series stay around forever — Prometheus dutifully keeps
    scraping their frozen last value into new time samples. Our SFU window
    then catches BOTH the live run and arbitrarily many stale ones.

    Webrtcperf's session starts up to ~max_age_sec before SFU sees first
    packets (browser spawn + connect + first frames), so we keep only series
    whose datetime label sits in [t_start - max_age, t_start + 30].
    """
    kept = []
    dropped = defaultdict(int)
    for s in series_list:
        dt_str = s["metric"].get("datetime")
        if not dt_str:
            kept.append(s)
            continue
        try:
            dt = parse_dt_label(dt_str)
        except ValueError:
            kept.append(s)
            continue
        if t_start - max_age_sec <= dt <= t_start + 30:
            kept.append(s)
        else:
            dropped[dt_str] += 1
    return kept, dict(dropped)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    log_path = sys.argv[1]
    participant = sys.argv[2]

    sub, tracks = parse_sfu_log(log_path, participant)
    if not sub:
        print(f"No bwe-log entries for {participant} in {log_path}")
        sys.exit(1)

    t_start, t_end = sub[0][0], sub[-1][0]
    t0 = t_start

    # SFU log uses the LiveKit room identity ("webrtcperf-N"); webrtcperf's
    # own metrics tag every series with its INTERNAL sessionId-based name —
    # zero-padded to six digits, e.g. "Participant-000000". Same person, two
    # label spaces. Translate so the Prometheus query finds the series.
    m = re.match(r"webrtcperf-(\d+)$", participant)
    if m:
        prom_participant = f"Participant-{int(m.group(1)):06d}"
    else:
        prom_participant = participant
    print(f"participant={participant} (prom={prom_participant}) window="
          f"{datetime.fromtimestamp(t_start):%H:%M:%S}–{datetime.fromtimestamp(t_end):%H:%M:%S} "
          f"sfu-samples={len(sub)} sfu-track-labels={len(tracks)}")
    print(f"querying {PROM_URL} ...")

    # Pad start a bit — pushgateway scrape may lag by one interval.
    q_start = t_start - 10
    q_end = t_end + 10

    # Webrtcperf per-track gauges are bare metric names (no _value suffix) —
    # the `_value` suffix only exists conceptually inside JS as a key on the
    # metric dict; the actual Prometheus name is `wst_<statName>` with labels
    # participantName + trackId. Suffixed names (_mean, _p95, _length, etc.)
    # are aggregates across all sessions, not per-track.
    bitrates = prom_query_range(
        f'wst_videoRecvBitrates{{participantName="{prom_participant}"}}',
        q_start, q_end,
    )
    # Fallback: derive bps from cumulative videoRecvBytes ourselves. Robust
    # even if webrtcperf's inboundRtp.bitrate field is flat-zero or scaled
    # weirdly — bytes are bytes.
    bitrates_from_bytes = prom_query_range(
        f'rate(wst_videoRecvBytes{{participantName="{prom_participant}"}}[10s]) * 8',
        q_start, q_end,
    )
    widths = prom_query_range(
        f'wst_videoRecvWidth{{participantName="{prom_participant}"}}',
        q_start, q_end,
    )
    heights = prom_query_range(
        f'wst_videoRecvHeight{{participantName="{prom_participant}"}}',
        q_start, q_end,
    )
    recv_fps = prom_query_range(
        f'wst_videoRecvFps{{participantName="{prom_participant}"}}',
        q_start, q_end,
    )
    print(f"  raw: bitrates={len(bitrates)} bitrates_from_bytes={len(bitrates_from_bytes)} "
          f"widths={len(widths)} heights={len(heights)} fps={len(recv_fps)} series")

    # Strip stale runs (see filter_current_run docstring).
    bitrates,            dropped_b   = filter_current_run(bitrates, t_start)
    bitrates_from_bytes, dropped_bfb = filter_current_run(bitrates_from_bytes, t_start)
    widths,              dropped_w   = filter_current_run(widths, t_start)
    heights,             dropped_h   = filter_current_run(heights, t_start)
    recv_fps,            dropped_f   = filter_current_run(recv_fps, t_start)
    all_dropped = defaultdict(int)
    for d in (dropped_b, dropped_bfb, dropped_w, dropped_h, dropped_f):
        for k, v in d.items():
            all_dropped[k] += v
    if all_dropped:
        print(f"  dropped stale runs:")
        for dt_str, count in sorted(all_dropped.items()):
            print(f"    datetime={dt_str}: {count} series")
    print(f"  kept: bitrates={len(bitrates)} bitrates_from_bytes={len(bitrates_from_bytes)} "
          f"widths={len(widths)} heights={len(heights)} fps={len(recv_fps)} series")

    # Diagnostic dumps: first sample per series so we can sanity-check units
    # and coverage. If client-side numbers look wrong on the plot, this tells
    # you whether the issue is at the Prometheus layer or downstream.
    def dump(name, series_list, limit=10):
        for s in series_list[:limit]:
            labels = ", ".join(f"{k}={v}" for k, v in s["metric"].items()
                               if k not in ("__name__", "exported_job", "instance", "job"))
            vals = s.get("values", [])
            if vals:
                first = float(vals[0][1])
                last = float(vals[-1][1])
                print(f"  {name}[{labels}] n={len(vals)} first={first:g} last={last:g}")
            else:
                print(f"  {name}[{labels}] empty")

    print("--- Prometheus diagnostic ---")
    dump("videoRecvBitrates", bitrates)
    dump("bytes->bps", bitrates_from_bytes)
    dump("videoRecvWidth", widths)
    dump("videoRecvHeight", heights)
    dump("videoRecvFps", recv_fps)

    # SFU side: first / last sample per track for sanity
    print("--- SFU diagnostic ---")
    for label, data in sorted(tracks.items())[:3]:
        if not data:
            continue
        _, req0, w0, h0, fps0 = data[0]
        _, reqN, wN, hN, fpsN = data[-1]
        print(f"  {label}: req {req0:.0f}→{reqN:.0f} kbps, "
              f"fwd {w0}x{h0}@{fps0:.1f}fps → {wN}x{hN}@{fpsN:.1f}fps, n={len(data)}")
    print("-----")

    if not bitrates and not widths and not recv_fps and not bitrates_from_bytes:
        print("NO client-side series at all.")
        print("Likely causes:")
        print("  1. webrtcperf could not reach the Mac Pushgateway (firewall, wrong IP)")
        print("  2. scenario has prometheusPushgateway commented out")
        print("  3. participant name in Prometheus doesn't match (try 'wst_*' query in"
              " Prometheus UI: http://localhost:9090/graph)")
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

    # --- Panel 1 ---
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
    if bitrates:
        xs, ys = sum_by_timestamp(bitrates)
        # Webrtcperf reports inboundRtp.bitrate in bps (rate of bytesReceived
        # over the stats interval). Normalise to Kbps.
        xs = [x - t0 for x in xs]
        ys_kbps = [y / 1000 for y in ys]
        fig.add_trace(go.Scatter(
            x=xs, y=ys_kbps,
            name="Total received (client Σ, native bitrate)",
            line=dict(color="green", width=2),
            mode="lines+markers",
        ), row=1, col=1)
    if bitrates_from_bytes:
        # Independent fallback: derive bps from cumulative videoRecvBytes —
        # works even if the native bitrate field is empty/wrong.
        xs, ys = sum_by_timestamp(bitrates_from_bytes)
        xs = [x - t0 for x in xs]
        ys_kbps = [y / 1000 for y in ys]
        fig.add_trace(go.Scatter(
            x=xs, y=ys_kbps,
            name="Total received (client Σ, rate from bytesReceived)",
            line=dict(color="darkgreen", width=2, dash="dot"),
            mode="lines+markers",
        ), row=1, col=1)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    # --- Panel 2: width ---
    for i, s in enumerate(widths):
        tid = s["metric"].get("trackId", "?")[-8:]
        xs, ys = series_xy(s, t0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"client width {tid}",
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines+markers",
        ), row=2, col=1)
    sfu_color_start = len(widths)
    for i, (label, data) in enumerate(sorted(tracks.items())):
        xs = [t - t0 for t, _, _, _, _ in data]
        ys = [w for _, _, w, _, _ in data]
        c = colors[(sfu_color_start + i) % len(colors)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"SFU fwdW {label}",
            line=dict(color=c, width=1, dash="dot"),
        ), row=2, col=1)

    # --- Panel 3: fps ---
    for i, s in enumerate(recv_fps):
        tid = s["metric"].get("trackId", "?")[-8:]
        xs, ys = series_xy(s, t0)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"client fps {tid}",
            line=dict(color=colors[i % len(colors)], width=2),
            mode="lines+markers",
        ), row=3, col=1)
    sfu_color_start = len(recv_fps)
    for i, (label, data) in enumerate(sorted(tracks.items())):
        xs = [t - t0 for t, _, _, _, _ in data]
        ys = [f for _, _, _, _, f in data]
        c = colors[(sfu_color_start + i) % len(colors)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            name=f"SFU fwdFps {label}",
            line=dict(color=c, width=1, dash="dot"),
        ), row=3, col=1)

    fig.update_layout(
        title=f"Cross-source validation — {participant}",
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
    out = os.path.join(results_dir, f"verify-{participant}.html")
    fig.write_html(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
