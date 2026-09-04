"""Static scenario harness CLI (PLAN.md M7).

Runs a configured number of dataset-driven single-epoch scenarios. Per
scenario the instance is generated ONCE and every selected algorithm
solves the SAME instance; one CSV row is appended per (scenario,
algorithm). Failures of a single algorithm (e.g. ``naive`` above its state
limit) are recorded in the ``status`` column and do not stop the run.

CSV schema (header row; one data row per scenario x algorithm):
- ``scenario``      0-based scenario index within this run
- ``seed``          the scenario's RNG seed (``base_seed + scenario``)
- ``n``             number of participants (first draw of the scenario RNG
                    when several values were given)
- ``ladder_files``  semicolon-joined dataset file stems the scenario drew
                    from
- ``scale_range``   ladder scale-factor range as ``"lo:hi"``
- ``algorithm``     registry name (pruned | naive | maxmin | lex | heuristic)
- ``status``        ``"ok"`` or ``"error: <message>"``
- ``wall_time_s``   wall-clock solve time (empty on error)
- ``q_min``         first coordinate of the evaluator's ``sorted_q``
- ``q_mean``        mean of ``sorted_q``
- ``sorted_q``      the full ascending vector as a JSON list
- ``stats``         solver-specific ``SolveResult.stats`` as a JSON object

Usage:
  uv run python -m sfu_alloc.benchmark.runner --scenarios 20 --n 4 6 8 \\
      --algorithms lex maxmin heuristic [--seed 0] [--dataset dataset] \\
      [--ladder-files bbb_L3T3_tf ...] [--scale-range 1.0 1.0] [--out CSV]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sfu_alloc.algorithms import SolveResult
from sfu_alloc.algorithms.brute_force import solve_naive, solve_pruned
from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.algorithms.milp import solve_lex_maxmin, solve_maxmin
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.instance import Instance
from sfu_alloc.scenarios import dataset_files, random_static_instance

__all__ = ["ALGORITHMS", "FIELDS", "main", "run_static"]

ALGORITHMS: dict[str, Callable[[Instance], SolveResult]] = {
    "pruned": solve_pruned,
    "naive": solve_naive,
    "maxmin": solve_maxmin,
    "lex": solve_lex_maxmin,
    "heuristic": solve_heuristic,
}

FIELDS = [
    "scenario",
    "seed",
    "n",
    "ladder_files",
    "scale_range",
    "algorithm",
    "status",
    "wall_time_s",
    "q_min",
    "q_mean",
    "sorted_q",
    "stats",
]


def run_static(
    scenarios: int,
    n_values: Sequence[int],
    algorithms: Sequence[str],
    out_path: str | Path,
    *,
    base_seed: int = 0,
    dataset_dir: str | Path = "dataset",
    ladder_files: Sequence[str] | None = None,
    scale_range: tuple[float, float] = (1.0, 1.0),
) -> Path:
    """Run the static harness and return the output CSV path.

    See the module docstring for the CSV schema. Deterministic: scenario
    ``s`` uses ``numpy.random.default_rng(base_seed + s)`` and draws ``n``
    first, then the instance.
    """
    unknown = sorted(set(algorithms) - ALGORITHMS.keys())
    if unknown:
        raise ValueError(f"unknown algorithms: {unknown}")
    if scenarios < 1 or not n_values:
        raise ValueError("need at least one scenario and one n value")
    ladders = load_ladders(dataset_dir)
    files = list(ladder_files) if ladder_files is not None else dataset_files(ladders)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, restval="")
        writer.writeheader()
        for s in range(scenarios):
            seed = base_seed + s
            rng = np.random.default_rng(seed)
            n = int(n_values[int(rng.integers(len(n_values)))])
            inst = random_static_instance(
                rng, n, ladders, files, scale_range=scale_range
            )
            for name in algorithms:
                row: dict[str, Any] = {
                    "scenario": s,
                    "seed": seed,
                    "n": n,
                    "ladder_files": ";".join(files),
                    "scale_range": f"{scale_range[0]}:{scale_range[1]}",
                    "algorithm": name,
                }
                t0 = time.perf_counter()
                try:
                    res = ALGORITHMS[name](inst)
                except Exception as exc:  # recorded, run continues
                    row["status"] = f"error: {exc}"
                else:
                    q = res.sorted_q
                    row.update(
                        status="ok",
                        wall_time_s=time.perf_counter() - t0,
                        q_min=float(q[0]),
                        q_mean=float(np.mean(q)),
                        sorted_q=json.dumps([float(x) for x in q]),
                        stats=json.dumps(res.stats, default=float),
                    )
                writer.writerow(row)
            fh.flush()
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static scenario harness (PLAN.md M7).")
    ap.add_argument("--scenarios", type=int, required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument(
        "--algorithms", nargs="+", required=True, choices=sorted(ALGORITHMS)
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--ladder-files", nargs="+", default=None)
    ap.add_argument("--scale-range", type=float, nargs=2, default=(1.0, 1.0))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or f"results/static_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path = run_static(
        args.scenarios,
        args.n,
        args.algorithms,
        out,
        base_seed=args.seed,
        dataset_dir=args.dataset,
        ladder_files=args.ladder_files,
        scale_range=(args.scale_range[0], args.scale_range[1]),
    )
    print(f"OK: wyniki w {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
