"""Export random instances + ``solve_heuristic`` results as JSON fixtures for
the Go port (``livekit/pkg/rtc/heuristicalloc``, cross-validation tests).

Instances are built with DEFAULT preferences only (``p ≡ 1``): the Go
implementation derives canonical weights internally from the ladders, so any
exported instance must be fully described by (ladders, b_hat, B, params,
history). The Go test rebuilds each instance as a full mesh (receiver ``j``
subscribes to every sender ``i != j`` in ascending order, ``MaxLevel = L_i``)
and asserts its sorted quality vector is lex-equal to ``expected_sorted_q``
within ``TOL = 1e-6``; ``expected_levels`` / ``expected_spent_cells`` are
compared exactly as a stricter secondary check.

Usage::

    uv run python scripts/export_go_fixtures.py [out_path]

Deterministic: fixed seed, one documented realization committed with the Go
tests.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.instance import History, Instance, weights_from_preferences

SEED = 20260829
COUNT = 50


def make_instance(rng: np.random.Generator) -> tuple[Instance, float]:
    n = int(rng.integers(2, 7))
    L = rng.integers(1, 6, size=n)
    r = []
    for i in range(n):
        levels = np.sort(rng.uniform(50.0, 2500.0, size=int(L[i])))
        r.append([0.0, *levels.tolist()])
    p = np.ones((n, n))
    w = weights_from_preferences(p, r, L)

    top = np.array([r[i][int(L[i])] for i in range(n)])
    sum_top = top.sum() - top  # per receiver j: sum_{i != j} r[i][L_i]
    b_hat = rng.uniform(0.2, 1.2, size=n) * sum_top
    B = float(rng.uniform(0.3, 1.1) * np.minimum(b_hat, sum_top).sum())

    T_stab = int(rng.integers(2, 7))
    history = None
    if rng.random() > 0.3:
        lam = np.zeros((n, n), dtype=np.int64)
        tau = np.ones((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                lam[i, j] = int(rng.integers(0, int(L[i]) + 1))
                tau[i, j] = int(rng.integers(1, T_stab + 3))
        history = History(lambda_prev=lam, tau=tau)

    inst = Instance(
        n=n,
        L=L,
        r=r,
        w=w,
        b_hat=b_hat,
        B=B,
        phi_k=float(rng.choice([1.0, 0.5, 0.25])),
        alpha=float(rng.choice([0.0, 0.1, 0.3])),
        T_stab=T_stab,
        history=history,
    )
    delta = float(rng.choice([1.0, 50.0, 100.0]))
    return inst, delta


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "go_fixtures.json"
    rng = np.random.default_rng(SEED)
    fixtures = []
    for k in range(COUNT):
        inst, delta = make_instance(rng)
        res = solve_heuristic(inst, delta_kbps=delta)
        fixtures.append(
            {
                "name": f"random-{k}",
                "n": inst.n,
                "L": inst.L.tolist(),
                "r": [
                    [float(x) for x in inst.r[i, : int(inst.L[i]) + 1]]
                    for i in range(inst.n)
                ],
                "b_hat": inst.b_hat.tolist(),
                "B": inst.B,
                "phi_k": inst.phi_k,
                "alpha": inst.alpha,
                "t_stab": inst.T_stab,
                "delta_kbps": delta,
                "history": None
                if inst.history is None
                else {
                    "lambda_prev": inst.history.lambda_prev.tolist(),
                    "tau": inst.history.tau.tolist(),
                },
                "expected_sorted_q": res.sorted_q.tolist(),
                "expected_levels": res.allocation.tolist(),
                "expected_spent_cells": int(res.stats["spent_cells"]),
            }
        )
    with open(out, "w") as fh:
        json.dump(
            {"seed": SEED, "count": len(fixtures), "fixtures": fixtures}, fh, indent=1
        )
        fh.write("\n")
    print(f"wrote {len(fixtures)} fixtures to {out}")


if __name__ == "__main__":
    main()
