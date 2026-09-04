"""Multi-epoch harness CLI (PLAN.md M8).

Runs a configured number of dataset-driven multi-epoch scenarios. Every
selected algorithm runs its OWN independent closed loop over the same
scenario data (identical ladders, preferences, ``B`` and ``b_hat``
trajectories) with the history fed from its own previous allocation, so
trajectories legitimately diverge between algorithms. One CSV row is
appended per (scenario, algorithm, epoch); an algorithm failure mid-run is
recorded as a single ``error`` row and does not stop the other algorithms.

CSV schema (header row):
- ``scenario``           0-based scenario index within this run
- ``seed``               the scenario's RNG seed (``base_seed + scenario``)
- ``n``                  number of participants (first draw of the scenario
                         RNG when several values were given)
- ``epochs``             configured number of epochs E (epoch = 1 s, fixed)
- ``trajectory``         b_hat trajectory kind (constant|smooth|steps|mix)
- ``ladder_files``       semicolon-joined dataset file stems drawn from
- ``algorithm``          registry name (pruned|naive|maxmin|lex|heuristic)
- ``epoch``              0-based epoch index
- ``status``             "ok" or "error: <message>"
- ``wall_time_s``        wall-clock solve time of this epoch
- ``q_min``, ``q_mean``  first coordinate / mean of the evaluator's sorted_q
- ``sorted_q``           the full ascending vector as a JSON list
- ``changes_penalized``  level changes vs the previous epoch with
                         ``tau < T_stab`` at change time
- ``changes_free``       remaining level changes (``tau >= T_stab``)

Usage:
  uv run python -m sfu_alloc.benchmark.epochs --scenarios 3 --n 4 6 \\
      --algorithms lex heuristic [--epochs 100] [--trajectory mix] \\
      [--seed 0] [--dataset dataset] [--ladder-files ...] \\
      [--scale-range 1.0 1.0] [--freeze-ladder] [--out CSV]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sfu_alloc.benchmark.runner import ALGORITHMS
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.scenarios import dataset_files, random_epoch_scenario
from sfu_alloc.simulator import run_epochs

__all__ = ["FIELDS", "main", "run_epoch_bench"]

FIELDS = [
    "scenario",
    "seed",
    "n",
    "epochs",
    "trajectory",
    "ladder_files",
    "algorithm",
    "epoch",
    "status",
    "wall_time_s",
    "q_min",
    "q_mean",
    "sorted_q",
    "changes_penalized",
    "changes_free",
]


def run_epoch_bench(
    scenarios: int,
    n_values: Sequence[int],
    algorithms: Sequence[str],
    out_path: str | Path,
    *,
    epochs: int = 100,
    trajectory: str = "mix",
    base_seed: int = 0,
    dataset_dir: str | Path = "dataset",
    ladder_files: Sequence[str] | None = None,
    scale_range: tuple[float, float] = (1.0, 1.0),
    freeze_ladder: bool = False,
) -> Path:
    """Run the multi-epoch harness and return the output CSV path.

    Deterministic: scenario ``s`` uses
    ``numpy.random.default_rng(base_seed + s)`` and draws ``n`` first,
    then the whole scenario. See the module docstring for the CSV schema.
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
            scenario = random_epoch_scenario(
                rng,
                n,
                ladders,
                files,
                epochs=epochs,
                trajectory=trajectory,
                freeze_ladder=freeze_ladder,
                scale_range=scale_range,
            )
            for name in algorithms:
                base_row: dict[str, Any] = {
                    "scenario": s,
                    "seed": seed,
                    "n": n,
                    "epochs": epochs,
                    "trajectory": trajectory,
                    "ladder_files": ";".join(files),
                    "algorithm": name,
                }
                try:
                    records = run_epochs(scenario, ALGORITHMS[name])
                except Exception as exc:  # recorded, other algorithms continue
                    writer.writerow(base_row | {"status": f"error: {exc}"})
                else:
                    for rec in records:
                        writer.writerow(
                            base_row
                            | {
                                "epoch": rec["epoch"],
                                "status": "ok",
                                "wall_time_s": rec["wall_time_s"],
                                "q_min": rec["q_min"],
                                "q_mean": rec["q_mean"],
                                "sorted_q": json.dumps(rec["sorted_q"]),
                                "changes_penalized": rec["changes_penalized"],
                                "changes_free": rec["changes_free"],
                            }
                        )
                fh.flush()
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-epoch harness (PLAN.md M8).")
    ap.add_argument("--scenarios", type=int, required=True)
    ap.add_argument("--n", type=int, nargs="+", required=True)
    ap.add_argument(
        "--algorithms", nargs="+", required=True, choices=sorted(ALGORITHMS)
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--trajectory",
        default="mix",
        choices=["constant", "smooth", "steps", "mix"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--ladder-files", nargs="+", default=None)
    ap.add_argument("--scale-range", type=float, nargs=2, default=(1.0, 1.0))
    ap.add_argument("--freeze-ladder", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = args.out or f"results/epochs_{datetime.now():%Y%m%d_%H%M%S}.csv"
    path = run_epoch_bench(
        args.scenarios,
        args.n,
        args.algorithms,
        out,
        epochs=args.epochs,
        trajectory=args.trajectory,
        base_seed=args.seed,
        dataset_dir=args.dataset,
        ladder_files=args.ladder_files,
        scale_range=(args.scale_range[0], args.scale_range[1]),
        freeze_ladder=args.freeze_ladder,
    )
    print(f"OK: wyniki w {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
