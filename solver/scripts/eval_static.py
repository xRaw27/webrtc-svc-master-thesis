"""Static evaluation experiments E1/E2 + allocation snapshots (docs/EVAL.md).

Subcommands:
  exact      E1: naive vs pruned vs lex — agreement + runtime scaling
  heuristic  E2: lex vs heuristic(delta...) — runtime + accuracy vs lex
  heuristic_times
             E4: times of the heuristic ALONE per (n, delta) — no lex,
             no accuracy, so n can go far beyond lex's reach; wide
             per-n table with mean/std/max per delta
  accuracy   E5: heuristic accuracy vs MILP (lex) on single-epoch
             instances at ONE fixed n — per-coordinate relative errors
             of sorted_q (signed and absolute), exact-allocation
             agreement and the number of differing streams
  alloc      allocation snapshot figures for a few scenarios
  scenario   JSON dumps of the raw drawn data behind single instances
             (dataset file + second + per-level bitrates per sender,
             b_hat + preferences per receiver, bridge B) — no figures

Shared protocol (docs/EVAL.md): instances are drawn with the M7 generator
(instance seed = ``default_rng([base_seed, n, trial])``, identical for every
algorithm at a given (n, trial)); every solve runs in a killable worker
process; if any trial of an algorithm at some ``n`` exceeds the time limit
or errors, that algorithm's results for this ``n`` are discarded, it is not
tested at larger ``n``, and the cutoff lands in ``cutoffs.csv``. Every
chart ``X.png`` has a twin ``X.csv`` with exactly the plotted numbers.

Usage examples:
  uv run python scripts/eval_static.py exact --ns 2 3 4 5 6 8 10 --trials 10
  uv run python scripts/eval_static.py heuristic --ns 3 4 6 8 10 --trials 10
  uv run python scripts/eval_static.py heuristic_times \\
      --ns 2 3 4 5 6 7 8 9 10 20 50 100 --trials 100 --deltas 1 10 100
  uv run python scripts/eval_static.py accuracy --n 6 --trials 100 \\
      --deltas 1 5 10 20 50 100
  uv run python scripts/eval_static.py alloc --n 6 --seeds 3
  uv run python scripts/eval_static.py scenario --n 4 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shlex
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfu_alloc.algorithms.brute_force import solve_naive, solve_pruned
from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.algorithms.milp import solve_lex_maxmin
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.evaluator import lex_compare, q_corr_vector
from sfu_alloc.instance import precompute, weights_from_preferences
from sfu_alloc.scenarios import random_static_case, random_static_instance

SOLVERS = {
    "naive": solve_naive,
    "pruned": solve_pruned,
    "lex": solve_lex_maxmin,
    "heuristic": solve_heuristic,
}


def _worker(name: str, kwargs: dict, inst) -> dict[str, Any]:
    """Executed in a child process; returns plain picklable data."""
    t0 = time.perf_counter()
    res = SOLVERS[name](inst, **kwargs)
    return {
        "wall": time.perf_counter() - t0,
        "sorted_q": [float(x) for x in res.sorted_q],
        "allocation": np.asarray(res.allocation).tolist(),
    }


class TimedRunner:
    """Runs solves in a single reusable worker; kills it on timeout."""

    def __init__(self) -> None:
        self._pool: mp.pool.Pool | None = None

    def _ensure(self) -> mp.pool.Pool:
        if self._pool is None:
            self._pool = mp.get_context("spawn").Pool(1)
        return self._pool

    def run(self, name: str, kwargs: dict, inst, limit_s: float) -> dict[str, Any]:
        pool = self._ensure()
        handle = pool.apply_async(_worker, (name, kwargs, inst))
        try:
            return handle.get(limit_s)
        except mp.TimeoutError:
            pool.terminate()
            pool.join()
            self._pool = None
            return {"failed": f"timeout > {limit_s:g}s"}
        except Exception as exc:  # solver error inside the worker
            return {"failed": f"error: {exc}"}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


def make_instances(args, ns: Sequence[int], trials: int) -> dict:
    ladders = load_ladders(args.dataset)
    files = args.ladder_files
    return {
        (n, t): random_static_instance(
            np.random.default_rng([args.seed, n, t]), n, ladders, files
        )
        for n in ns
        for t in range(trials)
    }


def case_json(inst, level) -> str:
    """One-line JSON snapshot of a case: inputs, allocation and utilisation."""
    level = np.asarray(level)
    n = inst.n
    rates = inst.r[np.arange(n)[:, None], level]
    np.fill_diagonal(rates, 0.0)
    recv_used = rates.sum(axis=0)
    w_struct = weights_from_preferences(np.ones((n, n)), inst.r, inst.L)
    custom_prefs = [
        j
        for j in range(n)
        if not np.allclose(inst.w[:, j], w_struct[:, j], rtol=1e-9, atol=1e-12)
    ]
    return json.dumps(
        {
            "B": round(float(inst.B), 1),
            "b_hat": [round(float(x), 1) for x in inst.b_hat],
            "r": [
                [round(float(x), 1) for x in inst.r[i, : int(inst.L[i]) + 1]]
                for i in range(n)
            ],
            "w": [[round(float(x), 4) for x in row] for row in inst.w],
            "custom_prefs_receivers": custom_prefs,
            "level": level.tolist(),
            "recv_used": [round(float(x), 1) for x in recv_used],
            "uplink_used": round(float(recv_used.sum()), 1),
        },
        separators=(",", ":"),
    )


def sweep(
    algorithms: list[tuple[str, str, dict]],
    instances: dict,
    ns: Sequence[int],
    trials: int,
    limit_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the cutoff sweep; returns (per-trial rows, cutoffs)."""
    runner = TimedRunner()
    alive = dict.fromkeys((label for label, _, _ in algorithms), True)
    rows: list[dict] = []
    cutoffs: list[dict] = []
    try:
        for n in ns:
            for label, name, kwargs in algorithms:
                if not alive[label]:
                    continue
                pending: list[dict] = []
                for t in range(trials):
                    out = runner.run(name, kwargs, instances[(n, t)], limit_s)
                    if "failed" in out:
                        alive[label] = False
                        cutoffs.append(
                            {"algorithm": label, "n": n, "reason": out["failed"]}
                        )
                        print(f"  [cutoff] {label} przy n={n}: {out['failed']}")
                        pending.clear()
                        break
                    pending.append(
                        {
                            "algorithm": label,
                            "n": n,
                            "trial": t,
                            "wall_time_s": out["wall"],
                            "sorted_q": out["sorted_q"],
                            "allocation": out["allocation"],
                            "case": case_json(instances[(n, t)], out["allocation"]),
                        }
                    )
                rows.extend(pending)
                if pending:
                    mean = float(np.mean([r["wall_time_s"] for r in pending]))
                    print(f"  {label:16s} n={n:3d}: {mean * 1000:9.1f} ms śr.")
    finally:
        runner.close()
    return pd.DataFrame(rows), pd.DataFrame(cutoffs)


def same_alloc_counts(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (algorithm, n): trials with an allocation bitwise identical to
    every other algorithm alive at that (n, trial).

    Only trials where at least two algorithms returned a result are
    compared; an algorithm alone at some ``n`` gets no row (NaN after the
    merge). When everything agrees the count equals ``trials``.
    """
    counts: dict[tuple[str, int], int] = {}
    compared: set[tuple[str, int]] = set()
    for (n, _t), group in rows.groupby(["n", "trial"]):
        if len(group) < 2:
            continue
        allocs = {
            r["algorithm"]: np.asarray(r["allocation"]) for _, r in group.iterrows()
        }
        for label, alloc in allocs.items():
            compared.add((label, int(n)))
            ok = all(np.array_equal(alloc, other) for other in allocs.values())
            counts[(label, int(n))] = counts.get((label, int(n)), 0) + int(ok)
    records = [
        {"algorithm": label, "n": n, "same_alloc_trials": counts.get((label, n), 0)}
        for (label, n) in sorted(compared)
    ]
    frame = pd.DataFrame(records, columns=["algorithm", "n", "same_alloc_trials"])
    return frame.astype({"same_alloc_trials": "Int64"})


def timing_chart(
    rows: pd.DataFrame, out: Path, stem: str, extra: pd.DataFrame | None = None
) -> None:
    agg = (
        rows.groupby(["algorithm", "n"])["wall_time_s"]
        .agg(mean_s="mean", std_s="std", max_s="max", trials="count")
        .reset_index()
        .fillna({"std_s": 0.0})
    )
    if extra is not None:
        agg = agg.merge(extra, on=["algorithm", "n"], how="left")
    agg.to_csv(out / f"{stem}.csv", index=False)
    for suffix, log_scale in (("", True), ("_linear", False)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in agg.groupby("algorithm"):
            ax.errorbar(
                group["n"],
                group["mean_s"],
                yerr=group["std_s"],
                marker="o",
                capsize=3,
                label=label,
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("n (liczba uczestników)")
        ax.set_ylabel("czas rozwiązania [s]")
        ax.set_title(f"{stem}: średni czas ± odchylenie standardowe")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{stem}{suffix}.png", dpi=120)
        plt.close(fig)


def save_raw(rows: pd.DataFrame, path: Path) -> None:
    raw = rows.copy()
    raw["q_min"] = raw["sorted_q"].map(lambda q: q[0])
    cols = ["algorithm", "n", "trial", "wall_time_s", "q_min", "case"]
    raw[cols].to_csv(path, index=False)


def run_exact(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algorithms = [("naive", "naive", {}), ("pruned", "pruned", {}), ("lex", "lex", {})]
    instances = make_instances(args, args.ns, args.trials)
    rows, cutoffs = sweep(algorithms, instances, args.ns, args.trials, args.time_limit)
    if rows.empty:
        print("Brak wyników (wszystko ucięte?)")
        return
    save_raw(rows, out / "exact_times_raw.csv")
    cutoffs.to_csv(out / "exact_cutoffs.csv", index=False)
    timing_chart(rows, out, "exact_times", extra=same_alloc_counts(rows))

    # agreement among algorithms alive at each n
    agreement = []
    for n, group in rows.groupby("n"):
        labels = sorted(group["algorithm"].unique())
        lexeq = same_alloc = total = 0
        for _, trial_rows in group.groupby("trial"):
            by = {r["algorithm"]: r for _, r in trial_rows.iterrows()}
            if len(by) < 2:
                continue
            total += 1
            qs = [by[label]["sorted_q"] for label in labels if label in by]
            allocs = [by[label]["allocation"] for label in labels if label in by]
            lexeq += all(lex_compare(qs[0], q) == 0 for q in qs[1:])
            same_alloc += all(np.array_equal(allocs[0], a) for a in allocs[1:])
        if total:
            agreement.append(
                {
                    "n": n,
                    "algorithms": "+".join(labels),
                    "trials_compared": total,
                    "lex_equal_rate": lexeq / total,
                    "identical_allocation_rate": same_alloc / total,
                }
            )
    agree = pd.DataFrame(agreement)
    agree.to_csv(out / "exact_agreement.csv", index=False)
    print("\nZgodność (sorted_q lex-equal / identyczna alokacja):")
    print(agree.to_string(index=False))


def run_heuristic(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algorithms = [("lex", "lex", {})] + [
        (f"heuristic_d{d:g}", "heuristic", {"delta_kbps": float(d)})
        for d in args.deltas
    ]
    instances = make_instances(args, args.ns, args.trials)
    rows, cutoffs = sweep(algorithms, instances, args.ns, args.trials, args.time_limit)
    if rows.empty:
        print("Brak wyników (wszystko ucięte?)")
        return
    save_raw(rows, out / "heuristic_raw.csv")
    cutoffs.to_csv(out / "heuristic_cutoffs.csv", index=False)
    timing_chart(rows, out, "heuristic_times")

    # accuracy vs lex, only where lex is present
    lex_rows = {
        (r["n"], r["trial"]): r for _, r in rows[rows["algorithm"] == "lex"].iterrows()
    }
    gap_records = []
    for _, r in rows[rows["algorithm"] != "lex"].iterrows():
        ref = lex_rows.get((r["n"], r["trial"]))
        if ref is None:
            continue
        d = np.array(ref["sorted_q"]) - np.array(r["sorted_q"])
        alloc_ref = np.array(ref["allocation"])
        alloc_h = np.array(r["allocation"])
        n = int(r["n"])
        off = ~np.eye(n, dtype=bool)
        gap_records.append(
            {
                "algorithm": r["algorithm"],
                "n": n,
                "trial": r["trial"],
                "mean_abs_q_diff": float(np.mean(np.abs(d))),
                "q_min_diff": float(d[0]),
                "frac_pairs_diff": float(np.mean(alloc_ref[off] != alloc_h[off])),
                "per_position_diff": d.tolist(),
            }
        )
    gaps = pd.DataFrame(gap_records)
    if gaps.empty:
        print("lex nie ukończył żadnego n — brak metryk dokładności")
        return

    qgap = (
        gaps.groupby(["algorithm", "n"])
        .agg(
            mean_abs_q_diff=("mean_abs_q_diff", "mean"),
            std_abs_q_diff=("mean_abs_q_diff", "std"),
            q_min_diff_mean=("q_min_diff", "mean"),
            trials=("trial", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    qgap.to_csv(out / "heuristic_qgap.csv", index=False)
    _line_chart(
        qgap,
        "n",
        "mean_abs_q_diff",
        "std_abs_q_diff",
        "średnia |Q_lex - Q_heur| po pozycjach",
        out / "heuristic_qgap.png",
    )

    allocdiff = (
        gaps.groupby(["algorithm", "n"])
        .agg(
            frac_pairs_diff_mean=("frac_pairs_diff", "mean"),
            frac_pairs_diff_std=("frac_pairs_diff", "std"),
            trials=("trial", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    allocdiff.to_csv(out / "heuristic_allocdiff.csv", index=False)
    _line_chart(
        allocdiff,
        "n",
        "frac_pairs_diff_mean",
        "frac_pairs_diff_std",
        "odsetek par (i,j) z innym poziomem niż lex",
        out / "heuristic_allocdiff.png",
    )

    # per-position chart at the largest n covered by every heuristic variant
    counts = gaps.groupby("n")["algorithm"].nunique()
    full_ns = counts[counts == len(args.deltas)].index
    if len(full_ns):
        n_star = int(full_ns.max())
        sub = gaps[gaps["n"] == n_star]
        pos_records = []
        for label, group in sub.groupby("algorithm"):
            per_pos = np.array(group["per_position_diff"].tolist())
            for k in range(n_star):
                pos_records.append(
                    {
                        "algorithm": label,
                        "n": n_star,
                        "position": k + 1,
                        "mean_diff": float(per_pos[:, k].mean()),
                        "mean_abs_diff": float(np.abs(per_pos[:, k]).mean()),
                    }
                )
        pos = pd.DataFrame(pos_records)
        pos.to_csv(out / "heuristic_qgap_positions.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in pos.groupby("algorithm"):
            ax.plot(group["position"], group["mean_diff"], marker="o", label=label)
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_xlabel(f"pozycja k w sorted_q (n = {n_star})")
        ax.set_ylabel("średnia (Q_lex[k] - Q_heur[k])")
        ax.set_title("różnica jakości per pozycja wektora")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "heuristic_qgap_positions.png", dpi=120)
        plt.close(fig)


def run_heuristic_times(args) -> None:
    """E4: times of the heuristic alone per (n, delta) — wide per-n table."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    algorithms = [
        (f"heuristic_d{d:g}", "heuristic", {"delta_kbps": float(d)})
        for d in args.deltas
    ]
    instances = make_instances(args, args.ns, args.trials)
    rows, cutoffs = sweep(algorithms, instances, args.ns, args.trials, args.time_limit)
    cutoffs.to_csv(out / "heuristic_times_cutoffs.csv", index=False)
    if rows.empty:
        print("Brak wyników (wszystko ucięte?)")
        return

    raw = rows.copy()
    raw["q_min"] = raw["sorted_q"].map(lambda q: q[0])
    raw[["algorithm", "n", "trial", "wall_time_s", "q_min"]].to_csv(
        out / "heuristic_times_raw.csv", index=False
    )

    keys = [label.removeprefix("heuristic_") for label, _, _ in algorithms]
    columns = ["n"] + [
        f"{key}_{stat}"
        for key in keys
        for stat in ("mean_s", "std_s", "max_s", "trials")
    ]
    records = []
    for n in args.ns:
        sub = rows[rows["n"] == n]
        if sub.empty:
            continue
        rec: dict = {"n": n}
        for (label, _, _), key in zip(algorithms, keys, strict=True):
            walls = sub[sub["algorithm"] == label]["wall_time_s"]
            if walls.empty:  # variant cut off at this n — cells stay empty
                continue
            rec[f"{key}_mean_s"] = float(walls.mean())
            rec[f"{key}_std_s"] = float(walls.std()) if len(walls) > 1 else 0.0
            rec[f"{key}_max_s"] = float(walls.max())
            rec[f"{key}_trials"] = len(walls)
        records.append(rec)
    table = pd.DataFrame(records, columns=columns)
    table = table.astype({f"{key}_trials": "Int64" for key in keys})
    table.to_csv(out / "heuristic_times.csv", index=False)
    print("\nCzasy heurystyki (s):")
    print(table.to_string(index=False))


def run_accuracy(args) -> None:
    """E5: heuristic accuracy vs MILP on single-epoch instances, fixed n."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = int(args.n)
    variants = [(f"heuristic_d{d:g}", float(d)) for d in args.deltas]
    ladders = load_ladders(args.dataset)
    off = ~np.eye(n, dtype=bool)

    rows: list[dict] = []
    for trial in range(args.trials):
        rng = np.random.default_rng([args.seed, n, trial])
        inst = random_static_instance(rng, n, ladders, args.ladder_files)
        t0 = time.perf_counter()
        res_lex = SOLVERS["lex"](inst)
        wall_lex = time.perf_counter() - t0
        lex_level = np.asarray(res_lex.allocation, dtype=np.int64)
        lex_q = np.asarray(res_lex.sorted_q, dtype=float)
        rows.append(
            {
                "trial": trial,
                "n": n,
                "algorithm": "lex",
                "wall_time_s": wall_lex,
                "sorted_q": json.dumps(lex_q.tolist(), separators=(",", ":")),
                "level": json.dumps(lex_level.tolist(), separators=(",", ":")),
            }
        )
        for label, delta in variants:
            t0 = time.perf_counter()
            res = solve_heuristic(inst, delta_kbps=delta)
            wall = time.perf_counter() - t0
            level = np.asarray(res.allocation, dtype=np.int64)
            q = np.asarray(res.sorted_q, dtype=float)
            signed = (q - lex_q) / lex_q
            diff = int((level != lex_level)[off].sum())
            rows.append(
                {
                    "trial": trial,
                    "n": n,
                    "algorithm": label,
                    "wall_time_s": wall,
                    "sorted_q": json.dumps(q.tolist(), separators=(",", ":")),
                    "level": json.dumps(level.tolist(), separators=(",", ":")),
                    "same_alloc": diff == 0,
                    "diff_streams": diff,
                    "rel_err_signed": json.dumps(
                        signed.tolist(), separators=(",", ":")
                    ),
                    "rel_err_abs": json.dumps(
                        np.abs(signed).tolist(), separators=(",", ":")
                    ),
                }
            )
        if (trial + 1) % 10 == 0 or trial + 1 == args.trials:
            print(f"  {trial + 1}/{args.trials} instancji (lex {wall_lex:.1f} s)")

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "accuracy_raw.csv", index=False)

    table = accuracy_table(raw, n, args.deltas)
    table.to_csv(out / "accuracy.csv", index=False)
    print("\nDokładność względem MILP (błędy jako ułamki; x100 dla %):")
    print(table.to_string(index=False))


def accuracy_table(raw: pd.DataFrame, n: int, deltas: Sequence[float]) -> pd.DataFrame:
    """Aggregate E5 raw rows into the per-delta accuracy table.

    ``diff_mean`` / ``diff_std`` are computed over ALL trials (an
    identical allocation counts as 0 differing streams); ``diff_max`` is
    the overall maximum.
    """
    columns = (
        ["delta"]
        + [f"abs_k{k}_{stat}" for k in range(1, n + 1) for stat in ("mean", "std")]
        + [f"signed_k{k}_{stat}" for k in range(1, n + 1) for stat in ("mean", "std")]
        + ["same_alloc", "trials", "diff_mean", "diff_std", "diff_max"]
    )
    records = []
    for delta in deltas:
        sub = raw[raw["algorithm"] == f"heuristic_d{delta:g}"]
        signed = np.array([json.loads(x) for x in sub["rel_err_signed"]])
        rec: dict = {"delta": float(delta)}
        for k in range(n):
            rec[f"abs_k{k + 1}_mean"] = float(np.mean(np.abs(signed[:, k])))
            rec[f"abs_k{k + 1}_std"] = float(np.std(np.abs(signed[:, k]), ddof=1))
            rec[f"signed_k{k + 1}_mean"] = float(np.mean(signed[:, k]))
            rec[f"signed_k{k + 1}_std"] = float(np.std(signed[:, k], ddof=1))
        rec["same_alloc"] = int(sub["same_alloc"].sum())
        rec["trials"] = len(sub)
        diffs = sub["diff_streams"]
        rec["diff_mean"] = float(diffs.mean())
        rec["diff_std"] = float(diffs.std()) if len(diffs) > 1 else 0.0
        rec["diff_max"] = int(diffs.max())
        records.append(rec)
    table = pd.DataFrame(records, columns=columns)
    return table.astype({"same_alloc": "Int64", "trials": "Int64", "diff_max": "Int64"})


def _line_chart(df, x, y, yerr, ylabel, path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby("algorithm"):
        ax.errorbar(
            group[x], group[y], yerr=group[yerr], marker="o", capsize=3, label=label
        )
    ax.set_xlabel("n (liczba uczestników)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _participant_label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"P{i + 1}"


def run_alloc(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ladders = load_ladders(args.dataset)
    for seed_idx in range(args.seeds):
        rng = np.random.default_rng([args.seed, args.n, seed_idx])
        inst = random_static_instance(
            rng, args.n, ladders, args.ladder_files, p_default_prefs=1.0
        )
        res = SOLVERS[args.algorithm](inst)
        level = res.allocation
        q = q_corr_vector(inst, level)
        pre = precompute(inst)
        n = inst.n
        names = [_participant_label(i) for i in range(n)]
        colors = plt.get_cmap("tab10").colors

        fig, (ax_lad, ax_all) = plt.subplots(
            1, 2, figsize=(15, 1.2 * n + 2), width_ratios=[1.0, 1.7], sharey=True
        )

        # left panel: the ladders every participant transmits
        ladder_records = []
        for i in range(n):
            xs = [float(inst.r[i, lvl]) for lvl in range(int(inst.L[i]) + 1)]
            ax_lad.plot(xs, [i] * len(xs), "o-", color=colors[i % 10], ms=6, lw=1.2)
            for lvl, x in enumerate(xs):
                ladder_records.append(
                    {"sender": names[i], "level": lvl, "bitrate_kbps": round(x, 1)}
                )
                ax_lad.annotate(
                    str(lvl),
                    (x, i),
                    textcoords="offset points",
                    xytext=(0, 8 if lvl % 2 == 0 else -14),
                    ha="center",
                    fontsize=7,
                )
        ax_lad.set_yticks(range(n))
        ax_lad.set_yticklabels(names)
        ax_lad.invert_yaxis()
        ax_lad.set_xlabel("kbps")
        ax_lad.set_title("nadawane drabinki (etykiety = poziomy)")
        ax_lad.grid(True, axis="x", alpha=0.3)

        # right panel: the allocation with per-segment w*q contributions
        alloc_records = []
        totals = []
        for j in range(n):
            totals.append(
                sum(float(inst.r[i, level[i, j]]) for i in range(n) if i != j)
            )
        xmax = max(float(inst.b_hat.max()), max(totals)) * 1.08
        uplink = 0.0
        for j in range(n):
            left = 0.0
            for i in range(n):
                if i == j:
                    continue
                lvl = int(level[i, j])
                rate = float(inst.r[i, lvl])
                contrib = float(inst.w[i, j] * pre.q_corr[i, j, lvl])
                uplink += rate
                alloc_records.append(
                    {
                        "receiver": names[j],
                        "sender": names[i],
                        "level": lvl,
                        "bitrate_kbps": round(rate, 1),
                        "w": round(float(inst.w[i, j]), 4),
                        "wq_contribution": round(contrib, 4),
                        "b_hat_kbps": round(float(inst.b_hat[j]), 1),
                        "Q_corr": round(float(q[j]), 4),
                    }
                )
                if rate > 0:
                    ax_all.barh(j, rate, left=left, color=colors[i % 10], height=0.6)
                    if rate >= 0.055 * xmax:
                        label = f"{names[i]}:L{lvl}\n{contrib:.3f}"
                    elif rate >= 0.02 * xmax:
                        label = names[i]
                    else:
                        label = ""
                    if label:
                        ax_all.text(
                            left + rate / 2,
                            j,
                            label,
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                        )
                    left += rate
            ax_all.plot([inst.b_hat[j]], [j], marker="|", ms=26, color="red", zorder=5)
            ax_all.text(
                inst.b_hat[j],
                j + 0.38,
                f"b\u0302={inst.b_hat[j]:.0f}",
                fontsize=7,
                color="red",
                ha="center",
            )
            ax_all.text(
                xmax * 0.005,
                j - 0.42,
                f"{names[j]}: Q\u0303={q[j]:.3f}",
                fontsize=8,
            )
        ax_all.set_xlim(0, xmax)
        ax_all.set_xlabel(
            "kbps (segment = strumień nadawcy, w środku wkład w\u00b7q\u0303; "
            "| = limit b\u0302)"
        )
        ax_all.set_title("alokacja u odbiorców")
        fig.suptitle(
            f"{args.algorithm}, n={n}, seed={seed_idx}; "
            f"uplink {uplink:.0f} / B={inst.B:.0f} kbps"
        )
        fig.tight_layout()
        stem = f"alloc_{args.algorithm}_n{n}_seed{seed_idx}"
        fig.savefig(out / f"{stem}.png", dpi=120)
        plt.close(fig)

        alloc_df = pd.DataFrame(alloc_records)
        alloc_df["B_kbps"] = round(inst.B, 1)
        alloc_df["uplink_total_kbps"] = round(uplink, 1)
        alloc_df.to_csv(out / f"{stem}_alloc.csv", index=False)
        pd.DataFrame(ladder_records).to_csv(out / f"{stem}_ladders.csv", index=False)
        print(
            f"  zapisano {stem}.png + _alloc.csv + _ladders.csv "
            f"(uplink {uplink:.0f} / B {inst.B:.0f})"
        )


def run_scenario(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ladders = load_ladders(args.dataset)
    for seed_idx in range(args.seeds):
        rng = np.random.default_rng([args.seed, args.n, seed_idx])
        case = random_static_case(rng, args.n, ladders, args.ladder_files)
        inst = case.instance
        n = inst.n
        names = [_participant_label(i) for i in range(n)]
        payload = {
            "seed_key": [args.seed, args.n, seed_idx],
            "n": n,
            "senders": [
                {
                    "participant": names[i],
                    "file": case.files[i],
                    "second": case.seconds[i],
                    "scale": round(case.scales[i], 4),
                    "r_kbps": [
                        round(float(inst.r[i, lvl]), 1)
                        for lvl in range(1, int(inst.L[i]) + 1)
                    ],
                }
                for i in range(n)
            ],
            "receivers": [
                {
                    "participant": names[j],
                    "b_hat_kbps": round(float(inst.b_hat[j]), 1),
                    "p": [
                        None if i == j else round(float(case.p[i, j]), 4)
                        for i in range(n)
                    ],
                }
                for j in range(n)
            ],
            "B_kbps": round(float(inst.B), 1),
        }
        path = out / f"scenario_n{n}_seed{seed_idx}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  zapisano {path.name}")


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

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exact = sub.add_parser(
        "exact", parents=[common], help="E1: naive vs pruned vs lex"
    )
    p_exact.add_argument("--ns", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10])
    p_exact.add_argument("--trials", type=int, default=10)
    p_exact.add_argument("--time-limit", type=float, default=60.0)

    p_h = sub.add_parser(
        "heuristic", parents=[common], help="E2: lex vs heuristic(delta)"
    )
    p_h.add_argument("--ns", type=int, nargs="+", default=[3, 4, 6, 8, 10])
    p_h.add_argument("--trials", type=int, default=10)
    p_h.add_argument("--deltas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p_h.add_argument("--time-limit", type=float, default=60.0)

    p_ht = sub.add_parser(
        "heuristic_times",
        parents=[common],
        help="E4: czasy samej heurystyki per (n, delta), bez lex",
    )
    p_ht.add_argument(
        "--ns", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
    )
    p_ht.add_argument("--trials", type=int, default=100)
    p_ht.add_argument("--deltas", type=float, nargs="+", default=[1.0, 10.0, 100.0])
    p_ht.add_argument("--time-limit", type=float, default=60.0)

    p_acc = sub.add_parser(
        "accuracy",
        parents=[common],
        help="E5: dokładność heurystyki vs MILP na instancjach jednoepokowych",
    )
    p_acc.add_argument("--n", type=int, default=6)
    p_acc.add_argument("--trials", type=int, default=100)
    p_acc.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    )

    p_a = sub.add_parser("alloc", parents=[common], help="allocation snapshot figures")
    p_a.add_argument("--n", type=int, default=6)
    p_a.add_argument("--seeds", type=int, default=3)
    p_a.add_argument("--algorithm", default="lex", choices=sorted(SOLVERS))

    p_s = sub.add_parser(
        "scenario",
        parents=[common],
        help="zrzut wylosowanych danych scenariusza do JSON (bez wykresów)",
    )
    p_s.add_argument("--n", type=int, default=4)
    p_s.add_argument("--seeds", type=int, default=3)

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
                "command": "uv run python scripts/eval_static.py "
                + shlex.join(raw_argv),
                "parsed_args": vars(args),
            },
            indent=2,
            default=str,
        )
    )
    if args.cmd == "exact":
        run_exact(args)
    elif args.cmd == "heuristic":
        run_heuristic(args)
    elif args.cmd == "heuristic_times":
        run_heuristic_times(args)
    elif args.cmd == "accuracy":
        run_accuracy(args)
    elif args.cmd == "alloc":
        run_alloc(args)
    else:
        run_scenario(args)
    print(f"OK: artefakty w {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
