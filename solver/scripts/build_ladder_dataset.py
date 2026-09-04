"""Build the ladder dataset from browser SVC captures (PLAN2).

Reads ``captures/capture_{source}_{config}.json`` files produced by
``scripts/svc_capture/index.html`` (one JSON per source x encoder config;
each frame record carries spatialIndex, temporalIndex, bytes and the RTP
timestamp) and writes to ``dataset/``:

- one CSV per source x ladder variant (eight files named like the PNGs in
  ``plots/``: ``bbb_L3T3_tf.csv``, ..., ``johnny_L2T2_sf.csv``) in long
  format: source, config, second, level, r_raw [kbps] — raw per-second
  values only (smoothing, if ever needed, is a consumer-side concern);
  both ladder variants of a config come from the same capture (the full
  layer grid is recorded, the ladder is a choice of read-out points);
- ``manifest.json`` — sources and mezzanine checksums, capture conditions,
  ladder definitions, summary stats, monotonicity check results;
- ``report.md`` + ``plots/*.png`` — the human-readable verification report.

Usage: uv run python scripts/build_ladder_dataset.py [--captures DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RTP_CLOCK = 90_000
SOURCES = ("bbb", "johnny")
CONFIGS = ("L3T3", "L2T2")

# ladder variant -> list of (S, T) decoding points, level 1..L (PLAN2)
LADDERS: dict[str, list[tuple[int, int]]] = {
    "L3T3_tf": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)],
    "L3T3_sf": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    "L2T2_tf": [(0, 0), (0, 1), (1, 1)],
    "L2T2_sf": [(0, 0), (1, 0), (1, 1)],
}
MEZZANINES = {"bbb": "bbb_720p24_180s.mp4", "johnny": "johnny_720p60_x6.mp4"}
MEZZANINE_NOTES = {
    "bbb": "first 180 s of the 596 s mezzanine (lossless stream-copy cut); "
    "the calm intro segment — the heavy scenes at ~7:25+ and ~9:05+ of the "
    "full movie are outside this cut",
    "johnny": "pre-looped 6x (lossless concat of the 10 s original); "
    "content seam every ~10 s",
}


def unwrap_rtp(rtp: np.ndarray) -> np.ndarray:
    """Undo 32-bit RTP timestamp wrap-around (frames are in arrival order)."""
    rtp = rtp.astype(np.int64)
    jumps = np.diff(rtp) < -(2**31)
    return rtp + 2**32 * np.concatenate([[0], np.cumsum(jumps)])


def load_capture(path: Path) -> tuple[dict, pd.DataFrame]:
    """Load one capture JSON into (meta, per-frame DataFrame with 'sec')."""
    doc = json.loads(path.read_text())
    meta, frames = doc["meta"], pd.DataFrame(doc["frames"])
    if frames.empty:
        raise ValueError(f"{path}: no frames")
    if frames["s"].isna().any() or frames["t"].isna().any():
        raise ValueError(f"{path}: missing spatial/temporal indices")
    rtp = unwrap_rtp(frames["rtp"].to_numpy())
    media_s = (rtp - rtp.min()) / RTP_CLOCK
    frames["sec"] = media_s.astype(np.int64)
    full_seconds = int(media_s.max())  # keep only complete 1 s windows
    frames = frames[frames["sec"] < full_seconds]
    if full_seconds < 5:
        raise ValueError(f"{path}: capture too short ({full_seconds} s)")
    return meta, frames


def ladder_series(frames: pd.DataFrame, points: list[tuple[int, int]]) -> pd.DataFrame:
    """Per-second cumulative kbps for each ladder level (PLAN2 Measurement)."""
    seconds = np.arange(frames["sec"].max() + 1)
    out = []
    for level, (cap_s, cap_t) in enumerate(points, start=1):
        sub = frames[(frames["s"] <= cap_s) & (frames["t"] <= cap_t)]
        kbps = (sub.groupby("sec")["bytes"].sum() * 8 / 1000).reindex(
            seconds, fill_value=0.0
        )
        out.append(
            pd.DataFrame(
                {
                    "second": seconds,
                    "level": level,
                    "r_raw": kbps.to_numpy(),
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def monotonicity_violations(df: pd.DataFrame) -> int:
    """Count seconds where r_raw is not strictly increasing across levels."""
    wide = df.pivot(index="second", columns="level", values="r_raw")
    return int((wide.diff(axis=1).iloc[:, 1:] <= 0).any(axis=1).sum())


def keyframe_stats(frames: pd.DataFrame) -> dict:
    key_secs = sorted(frames.loc[frames["type"] == "key", "sec"].unique().tolist())
    total = frames.groupby("sec")["bytes"].sum() * 8 / 1000
    med = float(total.median())
    spike = float(total.loc[key_secs].mean() / med) if key_secs and med > 0 else None
    return {
        "keyframe_seconds": [int(s) for s in key_secs],
        "spike_ratio_vs_median_second": spike,
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Best-effort ffprobe of a mezzanine (missing ffprobe is not fatal)."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "stream=width,height,r_frame_rate,duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        stream = json.loads(out.stdout)["streams"][0]
        return {
            "width": stream["width"],
            "height": stream["height"],
            "fps": stream["r_frame_rate"],
            "duration_s": float(stream["duration"]),
        }
    except Exception as exc:
        return {"error": str(exc)}


def plot_variant(df: pd.DataFrame, source: str, variant: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for level, group in df.groupby("level"):
        ax.plot(group["second"], group["r_raw"], lw=1.0, label=f"l{level}")
    ax.set_xlabel("sekunda")
    ax.set_ylabel("kbps")
    ax.set_title(f"{source} / {variant}: r_raw per sekunda")
    ax.legend(ncols=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--captures", default="captures", type=Path)
    ap.add_argument("--out", default="dataset", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plots").mkdir(exist_ok=True)

    manifest: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rtp_clock_hz": RTP_CLOCK,
        "values": "raw per-second kbps; smoothing is a consumer-side concern",
        "ladders": {v: [f"S{s}T{t}" for (s, t) in pts] for v, pts in LADDERS.items()},
        "captures": {},
        "sources": {},
        "summary": {},
        "monotonicity_violations": {},
    }

    found_any = False
    for source in SOURCES:
        for config in CONFIGS:
            path = args.captures / f"capture_{source}_{config}.json"
            if not path.exists():
                print(f"UWAGA: brak {path} - pomijam {source}x{config}")
                continue
            found_any = True
            meta, frames = load_capture(path)
            key = f"{source}_{config}"
            manifest["captures"][key] = {
                k: meta.get(k)
                for k in (
                    "sourceFile",
                    "config",
                    "maxKbps",
                    "loops",
                    "api",
                    "ua",
                    "startedAt",
                    "finishedAt",
                    "mediaDurS",
                    "hiddenDuringRun",
                    "lastStats",
                )
            }
            manifest["captures"][key]["frames"] = len(frames)
            manifest["captures"][key]["keyframes"] = keyframe_stats(frames)
            for variant, points in LADDERS.items():
                if not variant.startswith(config):
                    continue
                df = ladder_series(frames, points)
                df.insert(0, "config", variant)
                df.insert(0, "source", source)
                df.to_csv(args.out / f"{source}_{variant}.csv", index=False)
                viol = monotonicity_violations(df)
                manifest["monotonicity_violations"][f"{source}_{variant}"] = viol
                stats = (
                    df.groupby("level")["r_raw"]
                    .agg(["mean", "median", lambda s: s.quantile(0.95)])
                    .rename(columns={"<lambda_0>": "p95"})
                    .round(1)
                )
                manifest["summary"][f"{source}_{variant}"] = {
                    f"l{lvl}": row.to_dict() for lvl, row in stats.iterrows()
                }
                plot_variant(
                    df, source, variant, args.out / "plots" / f"{source}_{variant}.png"
                )

    if not found_any:
        print("Nie znaleziono żadnych plików capture_*.json - nic do zrobienia.")
        return 1

    media = Path("media")
    for source, mezz in MEZZANINES.items():
        mezz_path = media / mezz
        manifest["sources"][source] = {
            "mezzanine": mezz,
            "note": MEZZANINE_NOTES.get(source),
            "mezzanine_sha256": sha256(mezz_path) if mezz_path.exists() else None,
            "probe": probe(mezz_path) if mezz_path.exists() else None,
        }
    originals = media / "sources.sha256"
    if originals.exists():
        manifest["sources"]["originals_sha256"] = {
            line.split()[1]: line.split()[0]
            for line in originals.read_text().splitlines()
            if line.strip()
        }
    manifest["sources"]["johnny_url"] = (
        "https://media.xiph.org/video/derf/y4m/Johnny_1280x720_60.y4m"
    )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_report(args.out, manifest)
    print(f"OK: dataset zapisany w {args.out}/")
    return 0


def write_report(out: Path, manifest: dict) -> None:
    lines = [
        "# Ladder dataset — raport weryfikacyjny",
        "",
        f"Wygenerowano: {manifest['generated_at']}  ",
        "Wartości: surowe kbps per sekunda (bez wygładzania).",
        "",
        "## Drabinki (poziom -> punkt dekodowania)",
        "",
    ]
    for variant, points in manifest["ladders"].items():
        lines.append(
            f"- **{variant}**: "
            + " | ".join(f"l{i + 1}={p}" for i, p in enumerate(points))
        )
    lines += ["", "## Statystyki per poziom [kbps]", ""]
    for key, levels in manifest["summary"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| poziom | mean | median | p95 |")
        lines.append("|---|---|---|---|")
        for lvl, s in levels.items():
            lines.append(f"| {lvl} | {s['mean']} | {s['median']} | {s['p95']} |")
        viol = manifest["monotonicity_violations"].get(key, "?")
        lines.append("")
        lines.append(
            f"Naruszenia monotoniczności (r_l < r_l+1 per sekunda): **{viol}**"
        )
        lines.append("")
    lines += ["## Nagrania", ""]
    for key, cap in manifest["captures"].items():
        kf = cap["keyframes"]
        lines.append(
            f"- **{key}**: {cap['frames']} ramek, {cap['mediaDurS']:.1f} s, "
            f"maxBitrate {cap['maxKbps']} kbps, pętle {cap.get('loops')}, "
            f"API {cap['api']}, karta w tle: {cap.get('hiddenDuringRun')}; "
            f"klatki kluczowe w sekundach {kf['keyframe_seconds']}, "
            f"skok vs mediana sekundy: {kf['spike_ratio_vs_median_second']}"
        )
    lines += [
        "",
        "## Wykresy",
        "",
        "Po jednym PNG na źródło x wariant w `plots/` (surowe r_raw, linia na poziom).",
        "",
        "Uwaga: pierwsze sekundy każdego nagrania zawierają rozbieg BWE "
        "(bitrate rośnie do targetu) oraz skok klatki kluczowej — to celowo "
        "zachowane, surowe zachowanie enkodera.",
    ]
    (out / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
