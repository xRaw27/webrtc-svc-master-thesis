"""Multi-epoch evaluation driver (docs/EVAL.md).

Subcommands:
  timeline   one figure per (scenario, algorithm): two rows per receiver
             (stacked per-sender bars of received kbps per epoch under the
             receiver's dashed b_hat trajectory, then the chosen LEVEL per
             sender as step lines), plus a bridge row (total uplink vs B)
             and a bottom row with every participant's Q̃ per epoch.
             Scenario generation and seeding are identical to
             benchmark/epochs.py, so a timeline with the same parameters
             reproduces the harness run (re-solved here).
  scenario   JSON + CSV dumps of the raw drawn scenario data, no figures
             and no solving: scenario_s{s}.json holds everything drawn
             once (per sender: dataset file, starting second, scale; per
             receiver: b_hat trajectory kind and preference row p; plus
             B, alpha, T_stab), the time-varying parts land in CSVs next
             to it — sender ladders per epoch (_ladders.csv) and
             receiver b_hat per epoch (_bhat.csv).
  exact      E3 (docs/EVAL.md): naive vs pruned vs lex epoch by epoch on
             a SHARED history (fed from --history-from, default lex), so
             all three solve the same instance every epoch. Emits
             exact_raw.csv (per scenario/epoch/algorithm) and
             exact_bins.csv (times + allocation agreement per bin of
             --bin epochs). Single --n, small enough for naive.
  runs       E6 raw phase (docs/EVAL.md): full INDEPENDENT closed-loop
             runs — MILP (lex) and every heuristic delta-variant each
             solve the whole scenario with its OWN allocation history
             (the M8 harness execution model). One CSV row per
             (scenario, algorithm, epoch) with wall time, the level
             matrix and the independently re-evaluated sorted quality
             vector; processing of this data is specified later.
  runs_qmeans
             E6 processing step: takes --raw (a runs_raw.csv) and emits
             one row per (scenario, algorithm) with the mean over the
             run's epochs of every coordinate of sorted_q
             (k1_mean..k{n}_mean).
  accuracy   E7 (docs/EVAL.md): ONE heuristic delta-variant vs MILP
             solving EXACTLY the same task every epoch — the shared
             history is fed from the HEURISTIC's allocation, so MILP
             acts as a per-epoch oracle on the heuristic's trajectory.
             Per epoch: absolute per-coordinate relative errors of
             sorted_q vs MILP, exact-allocation agreement and the number
             of differing streams; aggregated per scenario like
             eval_static accuracy (abs only, no signed). Single --n.

Every figure X.png has a twin X.csv with exactly the plotted numbers;
run_info.json records the command line and parsed parameters. Default
output directory: results/eval/<timestamp>_<subcommand>/.

Usage examples (matching the epochs_demo harness run):
  uv run python scripts/eval_epochs.py timeline --scenarios 2 --n 4 6 \\
      --epochs 30 --trajectory mix --algorithms lex heuristic --seed 42
  uv run python scripts/eval_epochs.py scenario --scenarios 2 --n 4 6 \\
      --epochs 30 --trajectory mix --seed 42
  uv run python scripts/eval_epochs.py exact --scenarios 100 --n 3 \\
      --epochs 100 --bin 10 --seed 0
  uv run python scripts/eval_epochs.py runs --scenarios 100 --n 5 \\
      --epochs 100 --deltas 1 5 10 20 50 100 --seed 0
  uv run python scripts/eval_epochs.py runs_qmeans \\
      --raw results/eval/<ts>_runs/runs_raw.csv
  uv run python scripts/eval_epochs.py accuracy --scenarios 1 --n 5 \\
      --epochs 180 --delta 1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.benchmark.runner import ALGORITHMS
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.scenarios import (
    EpochScenario,
    dataset_files,
    random_epoch_scenario,
    scenario_instance,
)
from sfu_alloc.simulator import run_epochs, update_history

EXACT_ALGORITHMS = ("naive", "pruned", "lex")


def _participant_label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"P{i + 1}"


def _ladder_at(scenario: EpochScenario, i: int, t: int) -> np.ndarray:
    """Participant i's scaled ladder in epoch t (levels 0..L_i)."""
    table = scenario.ladder_tables[i]
    row = (
        scenario.start_seconds[i]
        if scenario.freeze_ladder
        else (scenario.start_seconds[i] + t) % len(table)
    )
    return table[row] * scenario.scales[i]


def _epoch_rates(scenario: EpochScenario, records: list[dict]) -> np.ndarray:
    """``rates[t, i, j]`` = kbps sender i's stream takes at receiver j."""
    n, epochs = scenario.n, scenario.epochs
    rates = np.zeros((epochs, n, n))
    for rec in records:
        t = rec["epoch"]
        level = np.asarray(rec["level"])
        for i in range(n):
            rates[t, i, :] = _ladder_at(scenario, i, t)[level[i, :]]
        np.fill_diagonal(rates[t], 0.0)
    return rates


def _make_scenarios(args) -> list[tuple[int, int, EpochScenario]]:
    """(index, seed, scenario) triples, seeded exactly as benchmark/epochs.py."""
    if args.B is not None and len(args.B) not in (1, args.scenarios):
        raise ValueError(
            f"--B takes 1 value or exactly --scenarios={args.scenarios} values, "
            f"got {len(args.B)}"
        )
    ladders = load_ladders(args.dataset)
    files = (
        list(args.ladder_files)
        if args.ladder_files is not None
        else dataset_files(ladders)
    )
    scenarios = []
    for s in range(args.scenarios):
        seed = args.seed + s
        rng = np.random.default_rng(seed)
        n = int(args.n[int(rng.integers(len(args.n)))])
        scenario = random_epoch_scenario(
            rng,
            n,
            ladders,
            files,
            epochs=args.epochs,
            trajectory=args.trajectory,
            freeze_ladder=args.freeze_ladder,
            scale_range=(args.scale_range[0], args.scale_range[1]),
        )
        if args.B is not None:
            # override AFTER generation: the rng sequence is untouched, so
            # ladders/preferences/trajectories match the non-overridden run
            scenario = replace(
                scenario, B=float(args.B[0] if len(args.B) == 1 else args.B[s])
            )
        scenarios.append((s, seed, scenario))
    return scenarios


def _timeline_figure(
    scenario: EpochScenario,
    rates: np.ndarray,
    levels: np.ndarray,
    qs: np.ndarray,
    title: str,
    png_path: Path,
) -> None:
    n, epochs = scenario.n, scenario.epochs
    names = [_participant_label(i) for i in range(n)]
    colors = plt.get_cmap("tab10").colors
    xs = np.arange(epochs)
    max_level = max(table.shape[1] for table in scenario.ladder_tables) - 1

    heights = [1.0, 0.55] * n + [0.7, 0.85]
    fig, axes = plt.subplots(
        2 * n + 2,
        1,
        figsize=(max(9.0, 0.28 * epochs + 3.0), 1.35 * sum(heights) + 1.6),
        sharex=True,
        gridspec_kw={"height_ratios": heights},
    )
    for j in range(n):
        ax = axes[2 * j]
        bottom = np.zeros(epochs)
        for i in range(n):
            if i == j:
                continue
            bars = rates[:, i, j]
            ax.bar(
                xs,
                bars,
                bottom=bottom,
                width=1.0,
                color=colors[i % 10],
                linewidth=0,
            )
            bottom += bars
        ax.plot(
            xs,
            scenario.b_hat_traj[:, j],
            color="red",
            ls="--",
            lw=1.3,
            drawstyle="steps-mid",
        )
        ax.set_ylabel(f"{names[j]}\n[kbps]")
        ax.margins(x=0)

        # chosen level per sender; tiny vertical offsets keep coinciding
        # lines visible without hiding the integer they sit on
        ax = axes[2 * j + 1]
        senders = [i for i in range(n) if i != j]
        offsets = np.linspace(-0.1, 0.1, len(senders)) if len(senders) > 1 else [0.0]
        for off, i in zip(offsets, senders, strict=True):
            ax.plot(
                xs,
                levels[:, i, j] + off,
                color=colors[i % 10],
                lw=1.4,
                drawstyle="steps-mid",
            )
        ax.set_ylabel(f"{names[j]}\npoziom")
        ax.set_ylim(-0.45, max_level + 0.45)
        ax.set_yticks(range(max_level + 1))
        ax.grid(axis="y", lw=0.3, alpha=0.5)
        ax.margins(x=0)

    ax = axes[2 * n]
    uplink = rates.sum(axis=(1, 2))
    ax.bar(xs, uplink, width=1.0, color="0.55", linewidth=0)
    ax.axhline(scenario.B, color="red", ls="--", lw=1.3)
    ax.set_ylabel("SFU\n[kbps]")
    ax.margins(x=0)

    ax = axes[2 * n + 1]
    for j in range(n):
        ax.plot(xs, qs[:, j], color=colors[j % 10], lw=1.4, drawstyle="steps-mid")
    ax.set_ylabel("Q̃")
    ax.set_xlabel("epoka (= sekunda)")
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.margins(x=0)

    handles = [Patch(color=colors[i % 10], label=names[i]) for i in range(n)]
    handles.append(plt.Line2D([], [], color="red", ls="--", label="limit (b̂ / B)"))
    fig.legend(
        handles=handles,
        ncols=n + 1,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        fontsize=8,
    )
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def run_timeline(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(args.algorithms) - ALGORITHMS.keys())
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    for s, seed, scenario in _make_scenarios(args):
        n = scenario.n
        names = [_participant_label(i) for i in range(n)]
        for name in args.algorithms:
            records = run_epochs(scenario, ALGORITHMS[name])
            rates = _epoch_rates(scenario, records)
            levels = np.array([rec["level"] for rec in records])
            qs = np.array([rec["q"] for rec in records])
            stem = f"timeline_s{s}_{name}"
            title = (
                f"{name}, scenariusz {s} (seed {seed}), n={n}, "
                f"E={args.epochs}, trajektoria={args.trajectory}, "
                f"B={scenario.B:.0f} kbps"
            )
            _timeline_figure(scenario, rates, levels, qs, title, out / f"{stem}.png")

            rows = []
            for rec in records:
                t = rec["epoch"]
                level = np.asarray(rec["level"])
                uplink = float(rates[t].sum())
                for j in range(n):
                    for i in range(n):
                        if i == j:
                            continue
                        rows.append(
                            {
                                "epoch": t,
                                "receiver": names[j],
                                "sender": names[i],
                                "level": int(level[i, j]),
                                "bitrate_kbps": round(float(rates[t, i, j]), 1),
                                "b_hat_kbps": round(
                                    float(scenario.b_hat_traj[t, j]), 1
                                ),
                                "Q_corr": round(float(qs[t, j]), 4),
                                "B_kbps": round(scenario.B, 1),
                                "uplink_total_kbps": round(uplink, 1),
                            }
                        )
            pd.DataFrame(rows).to_csv(out / f"{stem}.csv", index=False)
            print(f"  zapisano {stem}.png/.csv")


def run_scenario(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for s, seed, scenario in _make_scenarios(args):
        n = scenario.n
        names = [_participant_label(i) for i in range(n)]
        payload = {
            "scenario": s,
            "seed": seed,
            "n": n,
            "epochs": scenario.epochs,
            "trajectory": scenario.trajectory,
            "freeze_ladder": scenario.freeze_ladder,
            "senders": [
                {
                    "participant": names[i],
                    "file": scenario.files[i],
                    "start_second": scenario.start_seconds[i],
                    "scale": round(scenario.scales[i], 4),
                }
                for i in range(n)
            ],
            "receivers": [
                {
                    "participant": names[j],
                    "trajectory_kind": scenario.trajectory_kinds[j],
                    "p": [
                        None if i == j else round(float(scenario.p[i, j]), 4)
                        for i in range(n)
                    ],
                }
                for j in range(n)
            ],
            "B_kbps": round(float(scenario.B), 1),
            "alpha": round(float(scenario.alpha), 4),
            "T_stab": int(scenario.T_stab),
        }
        stem = f"scenario_s{s}"
        (out / f"{stem}.json").write_text(json.dumps(payload, indent=2) + "\n")

        ladder_rows = []
        for t in range(scenario.epochs):
            for i in range(n):
                ladder = _ladder_at(scenario, i, t)
                for lvl in range(1, len(ladder)):
                    ladder_rows.append(
                        {
                            "epoch": t,
                            "sender": names[i],
                            "level": lvl,
                            "bitrate_kbps": round(float(ladder[lvl]), 1),
                        }
                    )
        pd.DataFrame(ladder_rows).to_csv(out / f"{stem}_ladders.csv", index=False)

        bhat_rows = [
            {
                "epoch": t,
                "receiver": names[j],
                "b_hat_kbps": round(float(scenario.b_hat_traj[t, j]), 1),
            }
            for t in range(scenario.epochs)
            for j in range(n)
        ]
        pd.DataFrame(bhat_rows).to_csv(out / f"{stem}_bhat.csv", index=False)
        print(f"  zapisano {stem}.json + _ladders.csv + _bhat.csv")


def run_exact(args) -> None:
    """E3: the three exact algorithms epoch by epoch on a shared history."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.n) != 1:
        raise ValueError(f"exact expects exactly one --n value, got {args.n}")
    if args.bin < 1:
        raise ValueError(f"--bin must be >= 1, got {args.bin}")

    rows: list[dict] = []
    for s, seed, scenario in _make_scenarios(args):
        history = None
        for t in range(scenario.epochs):
            inst = scenario_instance(scenario, t, history)
            allocs: dict[str, np.ndarray] = {}
            for name in EXACT_ALGORITHMS:
                t0 = time.perf_counter()
                res = ALGORITHMS[name](inst)
                wall = time.perf_counter() - t0
                level = np.asarray(res.allocation, dtype=np.int64)
                allocs[name] = level
                rows.append(
                    {
                        "scenario": s,
                        "seed": seed,
                        "n": scenario.n,
                        "epoch": t,
                        "algorithm": name,
                        "wall_time_s": wall,
                        "q_min": float(res.sorted_q[0]),
                        "level": json.dumps(level.tolist(), separators=(",", ":")),
                    }
                )
            ref = allocs[EXACT_ALGORITHMS[0]]
            agree = all(
                np.array_equal(ref, allocs[name]) for name in EXACT_ALGORITHMS[1:]
            )
            for row in rows[-len(EXACT_ALGORITHMS) :]:
                row["same_alloc"] = agree
            history = update_history(history, allocs[args.history_from])
        means = {
            name: np.mean(
                [
                    r["wall_time_s"]
                    for r in rows
                    if r["scenario"] == s and r["algorithm"] == name
                ]
            )
            for name in EXACT_ALGORITHMS
        }
        print(
            f"  scenariusz {s} (seed {seed}): "
            + ", ".join(f"{k} {v * 1000:.0f} ms śr." for k, v in means.items())
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "exact_raw.csv", index=False)

    binned = raw.copy()
    binned["bin"] = binned["epoch"] // args.bin
    records = []
    for b in sorted(binned["bin"].unique()):
        sub = binned[binned["bin"] == b]
        rec: dict = {
            "epoch_from": int(b * args.bin + 1),
            "epoch_to": int(min((b + 1) * args.bin, args.epochs)),
        }
        for name in EXACT_ALGORITHMS:
            walls = sub[sub["algorithm"] == name]["wall_time_s"]
            rec[f"{name}_mean_s"] = float(walls.mean())
            rec[f"{name}_std_s"] = float(walls.std()) if len(walls) > 1 else 0.0
            rec[f"{name}_max_s"] = float(walls.max())
        first = sub[sub["algorithm"] == EXACT_ALGORITHMS[0]]
        rec["same_alloc"] = int(first["same_alloc"].sum())
        rec["comparisons"] = len(first)
        records.append(rec)
    bins = pd.DataFrame(records)
    bins.to_csv(out / "exact_bins.csv", index=False)
    print("\nZbiorczo (czasy w s, zgodność = identyczne alokacje):")
    print(bins.to_string(index=False))


def run_runs(args) -> None:
    """E6 raw phase: independent closed-loop runs of lex + heuristic variants."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.n) != 1:
        raise ValueError(f"runs expects exactly one --n value, got {args.n}")

    def heuristic_with(delta: float):
        return lambda inst: solve_heuristic(inst, delta_kbps=delta)

    algorithms = [("lex", None, ALGORITHMS["lex"])] + [
        (f"heuristic_d{d:g}", float(d), heuristic_with(float(d))) for d in args.deltas
    ]

    rows: list[dict] = []
    for s, seed, scenario in _make_scenarios(args):
        for label, delta, solver in algorithms:
            records = run_epochs(scenario, solver)
            for rec in records:
                rows.append(
                    {
                        "scenario": s,
                        "seed": seed,
                        "n": scenario.n,
                        "trajectory": scenario.trajectory,
                        "algorithm": label,
                        "delta_kbps": delta,
                        "epoch": rec["epoch"],
                        "wall_time_s": rec["wall_time_s"],
                        "level": json.dumps(rec["level"], separators=(",", ":")),
                        "sorted_q": json.dumps(rec["sorted_q"], separators=(",", ":")),
                        "q": json.dumps(rec["q"], separators=(",", ":")),
                        "changes_penalized": rec["changes_penalized"],
                        "changes_free": rec["changes_free"],
                    }
                )
            mean = float(np.mean([r["wall_time_s"] for r in records]))
            print(
                f"  scenariusz {s} (seed {seed}): {label:16s} "
                f"{mean * 1000:8.1f} ms śr./epokę"
            )
    pd.DataFrame(rows).to_csv(out / "runs_raw.csv", index=False)
    print(f"zapisano runs_raw.csv ({len(rows)} wierszy)")


def run_runs_qmeans(args) -> None:
    """E6 processing: per-run means of every sorted_q coordinate."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.raw)
    if raw["n"].nunique() != 1:
        raise ValueError(f"expected a single n in {args.raw}")
    n = int(raw["n"].iloc[0])

    # lex first, then deltas ascending (not the alphabetical d1, d10, d100...)
    order = {
        label: -1.0 if pd.isna(delta) else float(delta)
        for label, delta in raw[["algorithm", "delta_kbps"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    records = []
    for (s, alg), group in raw.groupby(["scenario", "algorithm"]):
        qs = np.array([json.loads(x) for x in group.sort_values("epoch")["sorted_q"]])
        rec = {"scenario": int(s), "seed": int(group["seed"].iloc[0]), "algorithm": alg}
        for k in range(n):
            rec[f"k{k + 1}_mean"] = float(qs[:, k].mean())
        records.append(rec)
    records.sort(key=lambda r: (r["scenario"], order[r["algorithm"]]))
    df = pd.DataFrame(
        records,
        columns=["scenario", "seed", "algorithm"]
        + [f"k{k}_mean" for k in range(1, n + 1)],
    )
    df.to_csv(out / "runs_qmeans.csv", index=False)
    print(f"zapisano runs_qmeans.csv ({len(df)} wierszy)")
    print(df.head(len(order)).to_string(index=False))


def run_accuracy(args) -> None:
    """E7: heuristic vs MILP on the same task every epoch (heuristic history)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.n) != 1:
        raise ValueError(f"accuracy expects exactly one --n value, got {args.n}")
    delta = float(args.delta)
    label = f"heuristic_d{delta:g}"

    rows: list[dict] = []
    for s, seed, scenario in _make_scenarios(args):
        n = scenario.n
        off = ~np.eye(n, dtype=bool)
        history = None
        for t in range(scenario.epochs):
            inst = scenario_instance(scenario, t, history)
            t0 = time.perf_counter()
            res_h = solve_heuristic(inst, delta_kbps=delta)
            wall_h = time.perf_counter() - t0
            h_level = np.asarray(res_h.allocation, dtype=np.int64)
            h_q = np.asarray(res_h.sorted_q, dtype=float)
            t0 = time.perf_counter()
            res_m = ALGORITHMS["lex"](inst)
            wall_m = time.perf_counter() - t0
            m_level = np.asarray(res_m.allocation, dtype=np.int64)
            m_q = np.asarray(res_m.sorted_q, dtype=float)

            abs_err = np.abs(h_q - m_q) / m_q
            diff = int((h_level != m_level)[off].sum())
            base = {"scenario": s, "seed": seed, "n": n, "epoch": t}
            rows.append(
                base
                | {
                    "algorithm": "lex",
                    "wall_time_s": wall_m,
                    "sorted_q": json.dumps(m_q.tolist(), separators=(",", ":")),
                    "level": json.dumps(m_level.tolist(), separators=(",", ":")),
                }
            )
            rows.append(
                base
                | {
                    "algorithm": label,
                    "wall_time_s": wall_h,
                    "sorted_q": json.dumps(h_q.tolist(), separators=(",", ":")),
                    "level": json.dumps(h_level.tolist(), separators=(",", ":")),
                    "same_alloc": diff == 0,
                    "diff_streams": diff,
                    "rel_err_abs": json.dumps(abs_err.tolist(), separators=(",", ":")),
                }
            )
            # the HEURISTIC's allocation drives the shared history: MILP is
            # an oracle on the heuristic's trajectory, not the other way
            history = update_history(history, h_level)
        walls = [
            r["wall_time_s"]
            for r in rows
            if r["scenario"] == s and r["algorithm"] == "lex"
        ]
        print(
            f"  scenariusz {s} (seed {seed}): lex "
            f"{float(np.mean(walls)) * 1000:.0f} ms śr./epokę"
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "accuracy_raw.csv", index=False)

    n = int(args.n[0])
    columns = (
        ["scenario", "seed", "delta"]
        + [f"abs_k{k}_{stat}" for k in range(1, n + 1) for stat in ("mean", "std")]
        + ["same_alloc", "trials", "diff_mean", "diff_std", "diff_max"]
    )
    records = []
    for s, group in raw[raw["algorithm"] == label].groupby("scenario"):
        errs = np.array([json.loads(x) for x in group["rel_err_abs"]])
        rec: dict = {
            "scenario": int(s),
            "seed": int(group["seed"].iloc[0]),
            "delta": delta,
        }
        for k in range(n):
            rec[f"abs_k{k + 1}_mean"] = float(np.mean(errs[:, k]))
            rec[f"abs_k{k + 1}_std"] = float(np.std(errs[:, k], ddof=1))
        rec["same_alloc"] = int(group["same_alloc"].sum())
        rec["trials"] = len(group)
        diffs = group["diff_streams"]
        rec["diff_mean"] = float(diffs.mean())
        rec["diff_std"] = float(diffs.std()) if len(diffs) > 1 else 0.0
        rec["diff_max"] = int(diffs.max())
        records.append(rec)
    table = pd.DataFrame(records, columns=columns)
    table = table.astype(
        {"same_alloc": "Int64", "trials": "Int64", "diff_max": "Int64"}
    )
    table.to_csv(out / "accuracy.csv", index=False)
    print("\nDokładność względem MILP na trajektorii heurystyki (ułamki; x100 dla %):")
    print(table.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default="dataset")
    common.add_argument("--ladder-files", nargs="+", default=None)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument(
        "--out",
        default=None,
        help="katalog wynikowy; domyślnie results/eval/<timestamp>_<subkomenda>/",
    )

    scen = argparse.ArgumentParser(add_help=False)
    scen.add_argument("--scenarios", type=int, required=True)
    scen.add_argument("--n", type=int, nargs="+", required=True)
    scen.add_argument("--epochs", type=int, default=100)
    scen.add_argument(
        "--trajectory",
        default="mix",
        choices=["constant", "smooth", "steps", "mix"],
    )
    scen.add_argument("--scale-range", type=float, nargs=2, default=(1.0, 1.0))
    scen.add_argument("--freeze-ladder", action="store_true")
    scen.add_argument(
        "--B",
        type=float,
        nargs="+",
        default=None,
        help="nadpisz wylosowane B [kbps]: jedna wartość wspólna albo po "
        "jednej na scenariusz; reszta scenariusza (drabinki, preferencje, "
        "trajektorie b_hat) pozostaje jak przy losowaniu",
    )

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_t = sub.add_parser(
        "timeline", parents=[common, scen], help="przebiegi w czasie per algorytm"
    )
    p_t.add_argument(
        "--algorithms", nargs="+", required=True, choices=sorted(ALGORITHMS)
    )
    sub.add_parser(
        "scenario",
        parents=[common, scen],
        help="zrzut wylosowanych danych scenariuszy do JSON + CSV (bez wykresów)",
    )
    p_e = sub.add_parser(
        "exact",
        parents=[common, scen],
        help="E3: naive vs pruned vs lex epoka po epoce na wspólnej historii",
    )
    p_e.add_argument("--bin", type=int, default=10)
    p_e.add_argument(
        "--history-from",
        default="lex",
        choices=list(EXACT_ALGORITHMS),
        help="czyja alokacja karmi wspólną historię (domyślnie lex)",
    )
    p_r = sub.add_parser(
        "runs",
        parents=[common, scen],
        help="E6 (faza surowa): niezależne pełne przebiegi MILP i heurystyk",
    )
    p_r.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    )
    p_q = sub.add_parser(
        "runs_qmeans",
        parents=[common],
        help="E6 (przetwarzanie): średnie pozycji sorted_q per przebieg",
    )
    p_q.add_argument("--raw", required=True, help="ścieżka do runs_raw.csv")
    p_a = sub.add_parser(
        "accuracy",
        parents=[common, scen],
        help="E7: heurystyka vs MILP na tym samym zadaniu w każdej epoce",
    )
    p_a.add_argument("--delta", type=float, default=1.0, help="delta_kbps heurystyki")

    args = ap.parse_args(argv)
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if args.out is None:
        args.out = f"results/eval/{datetime.now():%Y%m%d_%H%M%S}_{args.cmd}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_info.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "command": "uv run python scripts/eval_epochs.py "
                + shlex.join(raw_argv),
                "parsed_args": vars(args),
            },
            indent=2,
            default=str,
        )
    )
    if args.cmd == "timeline":
        run_timeline(args)
    elif args.cmd == "scenario":
        run_scenario(args)
    elif args.cmd == "exact":
        run_exact(args)
    elif args.cmd == "runs":
        run_runs(args)
    elif args.cmd == "runs_qmeans":
        run_runs_qmeans(args)
    else:
        run_accuracy(args)
    print(f"OK: artefakty w {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
