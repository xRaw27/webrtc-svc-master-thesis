#!/usr/bin/env python3
"""Parse the SFU log of one run into machine-readable evaluation artifacts.

Reads the structured `rba-log:` / `probe-log:` (and optionally `bwe-log:`)
lines emitted by the modified LiveKit SFU (see docs/04 and docs/09 in
kod/docs/) and writes, into one directory per run:

    run.json        everything as JSON: meta + params + the full epoch,
                    bwe-change and probe records of the chosen room
    epochs.csv      one row per decision epoch (room level)
    receivers.csv   one row per (epoch, receiver)
    pairs.csv       one row per (epoch, sender->receiver pair)
    senders.csv     one row per (epoch, sender/track)
    events.csv      bwe-change and probe events
    bwe_samples.csv only with --include-bwe-log: the 100 ms `bwe-log:`
                    samples (kind=sub|track), incl. forwarded width/height/fps

Derived columns computed here (so charts stay dumb):
  - pair `bitrateKbps` = the sender's ladder bitrate at the chosen level
    (level 0 / paused = 0), receiver `allocatedKbps` = sum over its pairs;
  - pair `changed` = had history and level != lambda_prev (first allocation
    of a pair is NOT counted as a change for this churn metric);
    `penalized` = changed and tau < T_stab (the switching penalty was paid);
  - epoch `utilization` = totalKbps / budgetKbps, `demandKbps` = sum of
    stock-greedy requested bandwidth, `jainQ` = Jain's fairness index over
    the per-receiver Q_corr (NaN when any Q < 0 — Jain assumes non-negative
    values — or when there are no receivers).

A log may contain several rooms (a rejoin recreates the room); by default the
room with the most epoch lines is chosen, `--room` overrides. Consistency
checks (sum of pair bitrates vs totalKbps, level <= maxLevel, monotone
ladders, sortedQ == sorted per-receiver Q) are reported as warnings, never
fatal — a run that trips them is itself a finding.

Usage:
    .venv/bin/python analysis/parse_epochs.py logs/sfu-<ts>.log
        [--room RM_xxx] [--out results/<stem>] [--include-bwe-log]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

SF_LEVEL_LABELS = ["paused", "S0T0", "S1T0", "S1T1", "S2T1", "S2T2"]

EPOCH_MSG = "rba-log: epoch"
START_MSG = "rba-log: start"
BWE_CHANGE_MSG = "rba-log: bwe-change"
PROBE_STARTED_MSG = "probe-log: cluster started"
PROBE_FINALIZED_MSG = "probe-log: cluster finalized"
BWE_SUB_MSG = "bwe-log: subscriber state"
BWE_TRACK_MSG = "bwe-log: track allocation"


def parse_ts(line: str) -> float | None:
    """Timestamp (epoch seconds) from the beginning of a zap log line."""
    try:
        return datetime.fromisoformat(line[: line.index("\t")]).timestamp()
    except (ValueError, IndexError):
        return None


def extract_json(line: str) -> dict | None:
    idx = line.find("{")
    if idx == -1:
        return None
    try:
        return json.loads(line[idx:])
    except json.JSONDecodeError:
        return None


def jain_index(values: list[float]) -> float:
    if not values or any(v < 0 for v in values):
        return float("nan")
    sq_sum = sum(v * v for v in values)
    if sq_sum == 0:
        return float("nan")
    total = sum(values)
    return (total * total) / (len(values) * sq_sum)


def scan_log(path: str, include_bwe: bool) -> tuple[dict[str, list], Counter, int]:
    """Single pass over the log; returns records grouped by message type."""
    records: dict[str, list] = defaultdict(list)
    room_epochs: Counter = Counter()
    skipped = 0
    wanted = {EPOCH_MSG, START_MSG, BWE_CHANGE_MSG, PROBE_STARTED_MSG, PROBE_FINALIZED_MSG}
    if include_bwe:
        wanted |= {BWE_SUB_MSG, BWE_TRACK_MSG}

    with open(path, errors="replace") as fh:
        for line in fh:
            msg = next((m for m in wanted if m in line), None)
            if msg is None:
                continue
            ts = parse_ts(line)
            data = extract_json(line)
            if ts is None or data is None:
                skipped += 1
                continue
            data["_ts"] = ts
            records[msg].append(data)
            if msg == EPOCH_MSG:
                room_epochs[data.get("roomID", "?")] += 1
    return records, room_epochs, skipped


def enrich_epoch(epoch: dict, t0: float, t_stab: int, warnings: list[str]) -> dict:
    """Attach derived pair/receiver/summary data to one raw epoch record."""
    ts = epoch["_ts"]
    seq = epoch.get("epochSeq")
    senders = epoch.get("senders") or []
    receivers = epoch.get("receivers") or []
    pairs = epoch.get("pairs") or []
    result = epoch.get("result") or {}

    sender_by_track = {s["trackID"]: s for s in senders}
    for s in senders:
        ladder = s.get("ladderKbps") or []
        if any(ladder[i] > ladder[i + 1] for i in range(len(ladder) - 1)):
            warnings.append(f"epoch {seq}: non-monotone ladder for {s['trackID']}")

    q_corr_by_identity = result.get("qCorr") or {}

    pair_rows = []
    alloc_by_sub: dict[str, float] = defaultdict(float)
    paused_by_sub: dict[str, int] = defaultdict(int)
    pairs_by_sub: dict[str, int] = defaultdict(int)
    level_changes = 0
    penalized_changes = 0
    for p in pairs:
        sender = sender_by_track.get(p["trackID"], {})
        ladder = sender.get("ladderKbps") or []
        level = p["level"]
        if level > p["maxLevel"]:
            warnings.append(f"epoch {seq}: level {level} > maxLevel {p['maxLevel']} for {p['trackID']}")
        bitrate = ladder[level] if 0 <= level < len(ladder) else float("nan")
        changed = p["lambdaPrev"] >= 0 and level != p["lambdaPrev"]
        penalized = changed and p["tau"] < t_stab
        level_changes += changed
        penalized_changes += penalized
        sub = p["subIdentity"]
        alloc_by_sub[sub] += bitrate if not math.isnan(bitrate) else 0.0
        pairs_by_sub[sub] += 1
        paused_by_sub[sub] += level == 0
        pair_rows.append(
            {
                "ts": ts,
                "t": ts - t0,
                "epochSeq": seq,
                "subIdentity": sub,
                "trackID": p["trackID"],
                "publisherIdentity": sender.get("publisherIdentity", ""),
                "level": level,
                "layer": p["layer"],
                "bitrateKbps": bitrate,
                "maxLevel": p["maxLevel"],
                "lambdaPrev": p["lambdaPrev"],
                "tau": p["tau"],
                "w": p["w"],
                "qCorr": p["qCorr"],
                "changed": changed,
                "penalized": penalized,
            }
        )

    receiver_rows = []
    for r in receivers:
        ident = r["identity"]
        receiver_rows.append(
            {
                "ts": ts,
                "t": ts - t0,
                "epochSeq": seq,
                "identity": ident,
                "bweRawBps": r["bweRawBps"],
                "bHatKbps": r["bHatKbps"],
                "congestionState": r["congestionState"],
                "greedyRequestedBps": r["greedyRequestedBps"],
                "allocatedKbps": alloc_by_sub.get(ident, 0.0),
                "qCorr": q_corr_by_identity.get(ident, float("nan")),
                "nPairs": pairs_by_sub.get(ident, 0),
                "nPaused": paused_by_sub.get(ident, 0),
            }
        )

    sender_rows = []
    for s in senders:
        ladder = s.get("ladderKbps") or []
        row = {
            "ts": ts,
            "t": ts - t0,
            "epochSeq": seq,
            "trackID": s["trackID"],
            "publisherIdentity": s.get("publisherIdentity", ""),
            "L": s["L"],
        }
        for lvl in range(1, 6):
            row[f"r{lvl}"] = ladder[lvl] if lvl < len(ladder) else float("nan")
        sender_rows.append(row)

    # consistency: sum of pair bitrates vs the solver's totalKbps
    total = result.get("totalKbps", 0.0)
    pair_sum = sum(r["bitrateKbps"] for r in pair_rows if not math.isnan(r["bitrateKbps"]))
    if pairs and abs(pair_sum - total) > 0.5:
        warnings.append(f"epoch {seq}: sum of pair bitrates {pair_sum:.3f} != totalKbps {total:.3f}")
    sorted_q = result.get("sortedQ") or []
    q_vals = sorted(q_corr_by_identity.values())
    if sorted_q and any(abs(a - b) > 1e-9 for a, b in zip(q_vals, sorted(sorted_q), strict=True)):
        warnings.append(f"epoch {seq}: sortedQ inconsistent with per-receiver qCorr")

    q_list = list(q_corr_by_identity.values())
    params = epoch.get("params") or {}
    budget = params.get("budgetKbps", float("nan"))
    epoch_row = {
        "ts": ts,
        "t": ts - t0,
        "epochSeq": seq,
        "nReceivers": len(receivers),
        "nSenders": len(senders),
        "nPairs": len(pairs),
        "nPaused": sum(paused_by_sub.values()),
        "totalKbps": total,
        "budgetKbps": budget,
        "utilization": (total / budget) if budget else float("nan"),
        "demandKbps": sum(r["greedyRequestedBps"] for r in receivers) / 1000.0,
        "minQ": min(q_list) if q_list else float("nan"),
        "meanQ": (sum(q_list) / len(q_list)) if q_list else float("nan"),
        "maxQ": max(q_list) if q_list else float("nan"),
        "jainQ": jain_index(q_list),
        "levelChanges": level_changes,
        "penalizedChanges": penalized_changes,
        "spentCells": result.get("spentCells", float("nan")),
        "budgetCells": result.get("budgetCells", float("nan")),
        "breakpoints": result.get("breakpoints", float("nan")),
        "dpMicros": result.get("dpMicros", float("nan")),
        "masterMicros": result.get("masterMicros", float("nan")),
        "nSkippedTracks": len(epoch.get("skippedTracks") or []),
        "nSkippedReceivers": len(epoch.get("skippedReceivers") or []),
        "solveErr": epoch.get("solveErr", ""),
        "elapsedMs": epoch.get("elapsedMs", float("nan")),
    }
    return {
        "epoch_row": epoch_row,
        "pair_rows": pair_rows,
        "receiver_rows": receiver_rows,
        "sender_rows": sender_rows,
    }


def layer_label(spatial: int, temporal: int) -> str:
    if spatial < 0 or temporal < 0:
        return "paused"
    return f"S{spatial}T{temporal}"


def bwe_sample_rows(records: dict[str, list], room_id: str, t0: float) -> list[dict]:
    rows = []
    for d in records.get(BWE_SUB_MSG, []):
        if d.get("roomID") != room_id:
            continue
        rows.append(
            {
                "ts": d["_ts"],
                "t": d["_ts"] - t0,
                "kind": "sub",
                "identity": d.get("participant", ""),
                "trackID": "",
                "currentLayer": "",
                "targetLayer": "",
                "forwardedWidth": float("nan"),
                "forwardedHeight": float("nan"),
                "forwardedFps": float("nan"),
                "bandwidthRequestedBps": float("nan"),
                "expectedUsageBps": d.get("expectedUsageBps", float("nan")),
                "channelCapacityBps": d.get("channelCapacityBps", float("nan")),
                "congestionState": d.get("congestionState", ""),
            }
        )
    for d in records.get(BWE_TRACK_MSG, []):
        if d.get("roomID") != room_id:
            continue
        rows.append(
            {
                "ts": d["_ts"],
                "t": d["_ts"] - t0,
                "kind": "track",
                "identity": d.get("participant", ""),
                "trackID": d.get("trackID", ""),
                "currentLayer": layer_label(d.get("currentSpatial", -1), d.get("currentTemporal", -1)),
                "targetLayer": layer_label(d.get("targetSpatial", -1), d.get("targetTemporal", -1)),
                "forwardedWidth": d.get("forwardedWidth", float("nan")),
                "forwardedHeight": d.get("forwardedHeight", float("nan")),
                "forwardedFps": d.get("forwardedFps", float("nan")),
                "bandwidthRequestedBps": d.get("bandwidthRequestedBps", float("nan")),
                "expectedUsageBps": float("nan"),
                "channelCapacityBps": float("nan"),
                "congestionState": "",
            }
        )
    rows.sort(key=lambda r: r["ts"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="SFU log file (logs/sfu-<ts>.log)")
    ap.add_argument("--room", help="roomID to extract (default: the room with the most epochs)")
    ap.add_argument("--out", help="output directory (default: results/<log-stem>)")
    ap.add_argument("--include-bwe-log", action="store_true", help="also parse the 100 ms bwe-log samples")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.log))[0]
    out_dir = args.out or os.path.join("results", stem)

    records, room_epochs, skipped = scan_log(args.log, args.include_bwe_log)
    if not room_epochs:
        print(f"error: no '{EPOCH_MSG}' lines found in {args.log}", file=sys.stderr)
        return 1

    if args.room:
        room_id = args.room
        if room_id not in room_epochs:
            print(f"error: room {room_id} not in log; rooms: {dict(room_epochs)}", file=sys.stderr)
            return 1
    else:
        room_id = room_epochs.most_common(1)[0][0]

    epochs = [e for e in records[EPOCH_MSG] if e.get("roomID") == room_id]
    epochs.sort(key=lambda e: e["_ts"])
    t0 = epochs[0]["_ts"]

    start_rec = next((s for s in records.get(START_MSG, []) if s.get("roomID") == room_id), None)
    params = (epochs[0].get("params") or {}) if epochs else {}
    t_stab = int(params.get("tStab", 2))

    warnings: list[str] = []
    epoch_rows, pair_rows, receiver_rows, sender_rows = [], [], [], []
    for e in epochs:
        enriched = enrich_epoch(e, t0, t_stab, warnings)
        epoch_rows.append(enriched["epoch_row"])
        pair_rows.extend(enriched["pair_rows"])
        receiver_rows.extend(enriched["receiver_rows"])
        sender_rows.extend(enriched["sender_rows"])

    event_rows = []
    for d in records.get(BWE_CHANGE_MSG, []):
        if d.get("roomID") != room_id:
            continue
        payload = {k: v for k, v in d.items() if not k.startswith("_") and k not in ("room", "roomID")}
        event_rows.append(
            {"ts": d["_ts"], "t": d["_ts"] - t0, "type": "bwe-change",
             "identity": d.get("identity", ""), "fields": json.dumps(payload)}
        )
    for msg, kind in ((PROBE_STARTED_MSG, "probe-started"), (PROBE_FINALIZED_MSG, "probe-finalized")):
        for d in records.get(msg, []):
            if d.get("roomID") != room_id:
                continue
            payload = {k: v for k, v in d.items()
                       if not k.startswith("_") and k not in ("room", "roomID", "participant", "pID", "remote")}
            event_rows.append(
                {"ts": d["_ts"], "t": d["_ts"] - t0, "type": kind,
                 "identity": d.get("participant", ""), "fields": json.dumps(payload)}
            )
    event_rows.sort(key=lambda r: r["ts"])

    os.makedirs(out_dir, exist_ok=True)

    def strip_private(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    run = {
        "meta": {
            "log": os.path.abspath(args.log),
            "roomID": room_id,
            "room": epochs[0].get("room", ""),
            "rooms": dict(room_epochs),
            "epochs": len(epochs),
            "firstTs": t0,
            "lastTs": epochs[-1]["_ts"],
            "skippedLines": skipped,
            "warnings": warnings,
        },
        "params": params,
        "start": strip_private(start_rec) if start_rec else None,
        "epochs": [dict(strip_private(e), ts=e["_ts"], t=e["_ts"] - t0) for e in epochs],
        "bweChanges": [dict(strip_private(d), ts=d["_ts"], t=d["_ts"] - t0)
                       for d in records.get(BWE_CHANGE_MSG, []) if d.get("roomID") == room_id],
        "probes": [dict(strip_private(d), ts=d["_ts"], t=d["_ts"] - t0)
                   for m in (PROBE_STARTED_MSG, PROBE_FINALIZED_MSG)
                   for d in records.get(m, []) if d.get("roomID") == room_id],
    }
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(run, fh, indent=1)
        fh.write("\n")

    tables = {
        "epochs.csv": epoch_rows,
        "receivers.csv": receiver_rows,
        "pairs.csv": pair_rows,
        "senders.csv": sender_rows,
        "events.csv": event_rows,
    }
    if args.include_bwe_log:
        tables["bwe_samples.csv"] = bwe_sample_rows(records, room_id, t0)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, name), index=False)

    dur = epochs[-1]["_ts"] - t0
    print(f"room {room_id} ({epochs[0].get('room', '?')}): {len(epochs)} epochs over {dur:.0f}s"
          + (f"; other rooms in log: { {k: v for k, v in room_epochs.items() if k != room_id} }"
             if len(room_epochs) > 1 else ""))
    print(f"rows: epochs={len(epoch_rows)} receivers={len(receiver_rows)} pairs={len(pair_rows)}"
          f" senders={len(sender_rows)} events={len(event_rows)}"
          + (f" bwe_samples={len(tables.get('bwe_samples.csv', []))}" if args.include_bwe_log else ""))
    if skipped:
        print(f"skipped {skipped} malformed lines")
    if warnings:
        print(f"{len(warnings)} consistency warnings (first 5):")
        for w in warnings[:5]:
            print(f"  - {w}")
    else:
        print("consistency checks: OK")
    print(f"wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
