"""
Summarize a single SFU log run.

Parses rba-log: tick (room-level) and bwe-log: subscriber state / track allocation
(per-subscriber, from LiveKit StreamAllocator) entries and produces a structured
report:

  - Run metadata: duration, budget, participants seen
  - Room-level: total bandwidth distribution, budget violations, utilization,
    Jain's fairness across subscribers
  - Per-subscriber: BWE distribution, allocation distribution, congestion state
    breakdown, per-track layer occupancy, layer switch frequency
  - Anomaly flags: BWE drift below throttle, large intra-sub disparity,
    sustained over-budget, excessive layer churn
  - Optional CSV export of time series for further plotting

Usage:
    poetry run python analysis/summarize_run.py <log_file> [--scenario <yaml>] [--csv-out <dir>]

If --scenario is given, the throttleConfig section is parsed and per-subscriber
utilisation (BWE / throttle, allocation / throttle) is added to the report.
"""

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml
from tabulate import tabulate


EVENT_RBA_TICK = "rba-log: tick"
EVENT_RBA_START = "rba-log: start"
EVENT_RBA_STOP = "rba-log: stop"
EVENT_BWE_STATE = "bwe-log: subscriber state"
EVENT_BWE_TRACK = "bwe-log: track allocation"

# In LiveKit's allocator, this is the "infinity" sentinel.
CHANNEL_CAPACITY_INFINITY_BPS = 100_000_000


@dataclass
class ParsedLog:
    start: dict | None = None
    ticks: list[dict] = field(default_factory=list)
    bwe_states: list[dict] = field(default_factory=list)
    bwe_tracks: list[dict] = field(default_factory=list)
    stop_count: int = 0


# Each zap line ends with "<EVENT_NAME>\t{...json...}".
# Capture the event name (free-text, may contain ": ") and the JSON tail.
LINE_RE = re.compile(r"^(?P<ts>\S+)\s+\S+\s+\S+\s+\S+\s+(?P<event>[^\t]+?)\s*\t\s*(?P<json>\{.*\})\s*$")


def _parse_ts(s: str) -> datetime:
    # zap timestamp ends with timezone offset like +0200; Python wants +02:00
    if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":
        s = s[:-2] + ":" + s[-2:]
    return datetime.fromisoformat(s)


def parse_log(path: Path) -> ParsedLog:
    out = ParsedLog()
    with path.open() as f:
        for line in f:
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            event = m.group("event").strip()
            try:
                data = json.loads(m.group("json"))
            except json.JSONDecodeError:
                continue
            data["_ts"] = m.group("ts")
            if event == EVENT_RBA_TICK:
                out.ticks.append(data)
            elif event == EVENT_RBA_START:
                out.start = data
            elif event == EVENT_RBA_STOP:
                out.stop_count += 1
            elif event == EVENT_BWE_STATE:
                out.bwe_states.append(data)
            elif event == EVENT_BWE_TRACK:
                out.bwe_tracks.append(data)
    return out


def load_throttles(scenario_path: Path) -> dict[str, int]:
    """Parse scenario YAML throttleConfig into {identity_index_str: down_rate_kbps}.

    Returns dict keyed by participant identity 'webrtcperf-<i>'. Sessions not
    listed are absent (unconstrained).
    """
    cfg = yaml.safe_load(scenario_path.read_text())
    raw = cfg.get("throttleConfig")
    if raw is None:
        return {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    result: dict[str, int] = {}
    for entry in raw:
        sessions = entry.get("sessions")
        down = entry.get("down")
        if down is None or sessions is None:
            continue
        rate_kbps = None
        if isinstance(down, dict):
            rate_kbps = down.get("rate")
        elif isinstance(down, list) and down:
            # array form: take last "rate" before run starts (best-effort)
            rate_kbps = down[-1].get("rate")
        if rate_kbps is None or rate_kbps <= 0:
            continue
        for s in str(sessions).split(","):
            s = s.strip()
            if "-" in s:
                lo, hi = (int(x) for x in s.split("-", 1))
                for i in range(lo, hi + 1):
                    result[f"webrtcperf-{i}"] = int(rate_kbps) * 1000
            else:
                result[f"webrtcperf-{int(s)}"] = int(rate_kbps) * 1000
    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _stats(values: Iterable[float]) -> dict[str, float]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return {"count": 0, "min": math.nan, "p50": math.nan, "mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": len(v),
        "min": min(v),
        "p50": statistics.median(v),
        "mean": statistics.mean(v),
        "p95": _percentile(v, 95),
        "max": max(v),
    }


def _bps_to_kbps(x: float) -> float:
    return x / 1000.0 if x is not None and not math.isnan(x) else math.nan


def _jain(values: list[float]) -> float:
    """Jain's fairness index. 1.0 = perfectly fair, 1/n = single-winner."""
    nonzero = [v for v in values if v > 0]
    if not nonzero:
        return math.nan
    total = sum(nonzero)
    sq = sum(v * v for v in nonzero)
    return (total * total) / (len(nonzero) * sq) if sq > 0 else math.nan


def _format_kbps_row(name: str, st: dict[str, float]) -> list[str]:
    if st["count"] == 0:
        return [name, "—", "—", "—", "—", "—", "—"]
    return [
        name,
        f"{int(st['count'])}",
        f"{st['min']/1000:.0f}",
        f"{st['p50']/1000:.0f}",
        f"{st['mean']/1000:.0f}",
        f"{st['p95']/1000:.0f}",
        f"{st['max']/1000:.0f}",
    ]


def build_room_df(ticks: list[dict]) -> pd.DataFrame:
    rows = []
    for t in ticks:
        rows.append({
            "ts": _parse_ts(t["_ts"]),
            "budget_bps": t.get("budgetBps", 0),
            "algo_total_bps": t.get("algorithmTotalBps", 0),
            "participants": t.get("participants", 0),
            "infeasible": t.get("infeasible", False),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["utilisation"] = df["algo_total_bps"] / df["budget_bps"].where(df["budget_bps"] > 0, math.nan)
        df["over_budget"] = df["algo_total_bps"] > df["budget_bps"]
    return df


def build_sub_df(ticks: list[dict]) -> pd.DataFrame:
    """One row per (tick, subscriber)."""
    rows = []
    for t in ticks:
        ts = _parse_ts(t["_ts"])
        for s in t.get("subscribers", []):
            rows.append({
                "ts": ts,
                "identity": s.get("identity"),
                "bwe_bps": s.get("channelCapacityBps", 0),
                "congestion": s.get("congestionState", ""),
                "num_tracks": s.get("numTracks", 0),
                "greedy_bps": s.get("greedyRequestedBps", 0),
                "algo_bps": s.get("algorithmAllocatedBps", 0),
            })
    return pd.DataFrame(rows)


def build_track_df(ticks: list[dict]) -> pd.DataFrame:
    """One row per (tick, subscriber, track) where a layer was assigned."""
    rows = []
    for t in ticks:
        ts = _parse_ts(t["_ts"])
        for s in t.get("subscribers", []):
            for trk in s.get("tracks", []) or []:
                rows.append({
                    "ts": ts,
                    "identity": s.get("identity"),
                    "track_id": trk.get("trackID"),
                    "layer": trk.get("layer"),
                    "bitrate_bps": trk.get("bitrateBps", 0),
                })
    return pd.DataFrame(rows)


def section_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def render_run_meta(parsed: ParsedLog, room_df: pd.DataFrame, log_path: Path, scenario_path: Path | None) -> None:
    section_header("RUN METADATA")
    start = parsed.start or {}
    rows = [
        ["log_file", str(log_path)],
        ["scenario", str(scenario_path) if scenario_path else "—"],
        ["budget", f"{start.get('budgetBps', 0)/1_000_000:.2f} Mbps"],
        ["epoch", start.get("epoch", "?")],
        ["rba ticks", len(parsed.ticks)],
        ["bwe-log subscriber state", len(parsed.bwe_states)],
        ["bwe-log track allocation", len(parsed.bwe_tracks)],
        ["rba-log: stop", parsed.stop_count],
    ]
    if not room_df.empty:
        duration = (room_df["ts"].max() - room_df["ts"].min()).total_seconds()
        rows.append(["wall-clock duration", f"{duration:.1f} s"])
        rows.append(["unique participant counts seen", sorted(room_df["participants"].unique().tolist())])
    print(tabulate(rows, tablefmt="plain"))


def render_room(room_df: pd.DataFrame) -> None:
    section_header("ROOM-LEVEL")
    if room_df.empty:
        print("(no rba-log: tick lines found)")
        return

    # Restrict steady-state analysis to ticks where at least one participant present.
    active = room_df[room_df["participants"] > 0]
    if active.empty:
        active = room_df

    over_budget = active[active["over_budget"]]
    infeasible = active[active["infeasible"]]

    headers = ["metric", "count", "min", "p50", "mean", "p95", "max"]
    rows = [
        _format_kbps_row("budget (kbps)", _stats(active["budget_bps"].tolist())),
        _format_kbps_row("algorithmTotal (kbps)", _stats(active["algo_total_bps"].tolist())),
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    print()
    util = active["utilisation"].dropna()
    print(f"utilisation (algo_total / budget): mean={util.mean()*100:.1f}%  p95={util.quantile(0.95)*100:.1f}%  max={util.max()*100:.1f}%")
    print(f"over-budget ticks: {len(over_budget)} / {len(active)} ({len(over_budget)/len(active)*100:.2f}%)")
    print(f"infeasible ticks: {len(infeasible)} / {len(active)} ({len(infeasible)/len(active)*100:.2f}%)")


def render_fairness(sub_df: pd.DataFrame) -> None:
    section_header("FAIRNESS (Jain's index across subscribers, per tick)")
    if sub_df.empty:
        print("(no subscriber data)")
        return
    # Only ticks where >= 2 subscribers active with at least one track
    grouped = sub_df.groupby("ts")["algo_bps"].apply(list)
    indices = []
    for vals in grouped:
        if sum(1 for v in vals if v > 0) >= 2:
            indices.append(_jain(vals))
    if not indices:
        print("(not enough multi-sub ticks for fairness analysis)")
        return
    print(f"ticks analyzed: {len(indices)}")
    print(f"Jain's index — min={min(indices):.3f}  p50={statistics.median(indices):.3f}  mean={statistics.mean(indices):.3f}  max={max(indices):.3f}")
    print("(1.0 = perfectly equal allocation, 1/n = single-winner. n is number of subs in the tick.)")


def render_per_subscriber(sub_df: pd.DataFrame, track_df: pd.DataFrame, throttles: dict[str, int]) -> None:
    section_header("PER-SUBSCRIBER")
    if sub_df.empty:
        print("(no subscriber data)")
        return

    identities = sorted(sub_df["identity"].dropna().unique())

    bwe_headers = ["identity", "throttle (kbps)", "active ticks", "min", "p50", "mean", "p95", "max"]
    bwe_rows = []
    alloc_rows = []
    for identity in identities:
        s = sub_df[(sub_df["identity"] == identity) & (sub_df["num_tracks"] > 0)]
        throttle = throttles.get(identity)
        throttle_str = f"{throttle/1000:.0f}" if throttle else "—"

        # BWE only counts ticks with a real estimate — exclude both 0 (no estimate yet)
        # and the LiveKit infinity sentinel (~100 Mbps).
        real_bwe = s["bwe_bps"][(s["bwe_bps"] > 0) & (s["bwe_bps"] < CHANNEL_CAPACITY_INFINITY_BPS)].tolist()
        bwe_st = _stats(real_bwe)
        bwe_rows.append([identity, throttle_str, len(s), *_format_kbps_row("", bwe_st)[2:]])

        algo = s["algo_bps"].tolist()
        algo_st = _stats(algo)
        alloc_rows.append([identity, throttle_str, len(s), *_format_kbps_row("", algo_st)[2:]])

    print("BWE estimate (kbps) — zero/missing estimates excluded so the stats reflect what BWE actually told us:")
    print(tabulate(bwe_rows, headers=bwe_headers, tablefmt="github"))
    print()
    print("Algorithm allocation (kbps) — what RBA decided per subscriber:")
    print(tabulate(alloc_rows, headers=bwe_headers, tablefmt="github"))

    print()
    if throttles:
        # Compute mean BWE / throttle and mean alloc / throttle (utilisation vs known cap)
        util_rows = []
        for identity in identities:
            if identity not in throttles:
                continue
            throttle = throttles[identity]
            s = sub_df[(sub_df["identity"] == identity) & (sub_df["num_tracks"] > 0)]
            real_bwe = s["bwe_bps"][(s["bwe_bps"] > 0) & (s["bwe_bps"] < CHANNEL_CAPACITY_INFINITY_BPS)]
            mean_bwe = real_bwe.mean() if not real_bwe.empty else math.nan
            mean_alloc = s["algo_bps"].mean() if not s.empty else math.nan
            util_rows.append([
                identity,
                f"{throttle/1000:.0f}",
                f"{mean_bwe/1000:.0f}" if not math.isnan(mean_bwe) else "—",
                f"{(mean_bwe/throttle)*100:.0f}%" if not math.isnan(mean_bwe) else "—",
                f"{mean_alloc/1000:.0f}" if not math.isnan(mean_alloc) else "—",
                f"{(mean_alloc/throttle)*100:.0f}%" if not math.isnan(mean_alloc) else "—",
            ])
        if util_rows:
            print("Throttle utilisation (mean values vs known down-throttle):")
            print(tabulate(util_rows, headers=["identity", "throttle (kbps)", "mean BWE (kbps)", "BWE / throttle", "mean alloc (kbps)", "alloc / throttle"], tablefmt="github"))
            print()

    # Congestion state distribution
    state_rows = []
    for identity in identities:
        s = sub_df[(sub_df["identity"] == identity) & (sub_df["num_tracks"] > 0)]
        if s.empty:
            continue
        c = Counter(s["congestion"].tolist())
        total = sum(c.values())
        state_rows.append([
            identity,
            *(f"{c.get(state, 0) / total * 100:.0f}%" for state in ("NONE", "EARLY_WARNING", "CONGESTED")),
        ])
    if state_rows:
        print("Congestion state distribution (% of active ticks):")
        print(tabulate(state_rows, headers=["identity", "NONE", "EARLY_WARNING", "CONGESTED"], tablefmt="github"))


def render_per_track(track_df: pd.DataFrame) -> None:
    section_header("PER-TRACK LAYER OCCUPANCY")
    if track_df.empty:
        print("(no track decisions logged)")
        return

    # Layer time distribution per (identity, track)
    grouped = track_df.groupby(["identity", "track_id"])
    rows = []
    for (identity, track_id), g in grouped:
        layer_counts = Counter(g["layer"].tolist())
        total = sum(layer_counts.values())
        layer_dist = ", ".join(f"{l}:{c/total*100:.0f}%" for l, c in sorted(layer_counts.items(), key=lambda kv: -kv[1]))
        # Layer switches
        switches = (g["layer"] != g["layer"].shift()).sum() - 1  # subtract initial "switch from NaN"
        switches = max(0, int(switches))
        duration_min = (g["ts"].max() - g["ts"].min()).total_seconds() / 60.0
        sw_rate = switches / duration_min if duration_min > 0 else 0.0
        rows.append([
            identity,
            track_id[-12:] if track_id else "?",
            total,
            f"{g['bitrate_bps'].mean()/1000:.0f}",
            layer_dist,
            switches,
            f"{sw_rate:.1f}/min",
        ])
    print(tabulate(rows, headers=["identity", "track (last 12)", "ticks", "mean kbps", "layer mix", "switches", "switch rate"], tablefmt="github"))


def render_anomalies(sub_df: pd.DataFrame, track_df: pd.DataFrame, throttles: dict[str, int], room_df: pd.DataFrame) -> None:
    section_header("FLAGGED ANOMALIES")
    flags: list[str] = []

    # 1. Over-budget ticks
    if not room_df.empty:
        ob = room_df[room_df["over_budget"]]
        if not ob.empty:
            flags.append(f"OVER-BUDGET: {len(ob)} ticks exceeded B_SFU (max excess {(ob['algo_total_bps'] - ob['budget_bps']).max()/1000:.0f} kbps)")

    # 2. BWE drift below throttle
    if throttles and not sub_df.empty:
        for identity, throttle in throttles.items():
            s = sub_df[(sub_df["identity"] == identity) & (sub_df["bwe_bps"] > 0) & (sub_df["bwe_bps"] < CHANNEL_CAPACITY_INFINITY_BPS)]
            if s.empty:
                continue
            ratio = (s["bwe_bps"] / throttle).mean()
            if ratio < 0.75:
                flags.append(f"BWE DRIFT: {identity} mean BWE was {ratio*100:.0f}% of throttle ({throttle/1000:.0f} kbps) — likely under-probing")

    # 3. Large intra-sub disparity between tracks (algo_bps)
    if not track_df.empty:
        per_sub = track_df.groupby(["identity", "ts"])["bitrate_bps"].agg(["min", "max", "count"])
        # only ticks with >= 2 tracks
        multi = per_sub[per_sub["count"] >= 2]
        if not multi.empty:
            ratios = multi["max"] / multi["min"].replace(0, math.nan)
            disparate = ratios[ratios > 4]  # max > 4x min
            if not disparate.empty:
                # group by identity, count
                disp_per_sub = disparate.groupby(level=0).size()
                for identity, count in disp_per_sub.items():
                    pct = count / multi.xs(identity, level=0).shape[0] * 100
                    flags.append(f"INTRA-SUB DISPARITY: {identity} had >4× ratio between tracks in {count} ticks ({pct:.0f}% of multi-track ticks)")

    # 4. Excessive layer churn
    if not track_df.empty:
        grouped = track_df.groupby(["identity", "track_id"])
        for (identity, track_id), g in grouped:
            switches = (g["layer"] != g["layer"].shift()).sum() - 1
            switches = max(0, int(switches))
            duration_min = (g["ts"].max() - g["ts"].min()).total_seconds() / 60.0
            if duration_min > 0 and switches / duration_min > 20:
                flags.append(f"LAYER CHURN: {identity} track …{(track_id or '')[-12:]} switched {switches/duration_min:.0f}×/min")

    if not flags:
        print("(no anomalies flagged)")
    else:
        for f in flags:
            print(f" ⚠  {f}")


def write_csvs(csv_dir: Path, room_df: pd.DataFrame, sub_df: pd.DataFrame, track_df: pd.DataFrame) -> None:
    csv_dir.mkdir(parents=True, exist_ok=True)
    room_df.to_csv(csv_dir / "room.csv", index=False)
    sub_df.to_csv(csv_dir / "subscribers.csv", index=False)
    track_df.to_csv(csv_dir / "tracks.csv", index=False)
    print()
    print(f"CSVs written to {csv_dir}/")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("log", type=Path, help="SFU log file (sfu-<ts>.log)")
    p.add_argument("--scenario", type=Path, default=None, help="optional scenario YAML for per-session throttle info")
    p.add_argument("--csv-out", type=Path, default=None, help="write parsed time series to this directory")
    args = p.parse_args(argv)

    if not args.log.exists():
        print(f"log not found: {args.log}", file=sys.stderr)
        return 2

    parsed = parse_log(args.log)
    throttles = load_throttles(args.scenario) if args.scenario else {}
    room_df = build_room_df(parsed.ticks)
    sub_df = build_sub_df(parsed.ticks)
    track_df = build_track_df(parsed.ticks)

    render_run_meta(parsed, room_df, args.log, args.scenario)
    render_room(room_df)
    render_fairness(sub_df)
    render_per_subscriber(sub_df, track_df, throttles)
    render_per_track(track_df)
    render_anomalies(sub_df, track_df, throttles, room_df)

    if args.csv_out:
        write_csvs(args.csv_out, room_df, sub_df, track_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
