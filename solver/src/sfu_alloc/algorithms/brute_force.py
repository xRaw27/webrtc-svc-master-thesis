"""Brute-force reference solvers (M3, PLAN.md).

``solve_naive`` enumerates the full Cartesian product of levels over all
ordered pairs with ``itertools.product`` and judges every candidate with the
evaluator. Deliberately dumb: its value is obvious correctness.

``solve_pruned`` searches the same space with a DFS over pairs in
receiver-major order. Levels are tried in increasing order, so as soon as a
partial downlink or uplink sum exceeds its budget (plus ``FEAS_TOL``, to
stay consistent with the evaluator) the remaining levels of that pair are
skipped (bitrates are non-decreasing in the level index). Per-receiver
``Q_corr`` sums are maintained incrementally. The optional per-receiver
Pareto-front reduction mentioned in PLAN.md is NOT implemented.

Both solvers re-evaluate their final allocation with the evaluator — the
``sorted_q`` in the returned ``SolveResult`` never comes from the search
itself — and raise ``RuntimeError`` if that re-evaluation fails.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np

from sfu_alloc.algorithms import SolveResult, finalize_result
from sfu_alloc.constants import FEAS_TOL, TOL
from sfu_alloc.evaluator import check_feasibility, lex_compare, sorted_q
from sfu_alloc.instance import Instance, precompute

__all__ = ["solve_naive", "solve_pruned"]


def _pairs_receiver_major(n: int) -> list[tuple[int, int]]:
    """Ordered pairs (i, j), i != j, grouped by receiver j."""
    return [(i, j) for j in range(n) for i in range(n) if i != j]


def solve_naive(inst: Instance, max_states: int = 2_000_000) -> SolveResult:
    """Exhaustive reference solver: full Cartesian product over pairs.

    Keeps the lexicographically best sorted quality vector (judged entirely
    by the evaluator) and one witnessing allocation. Raises ``ValueError``
    when the state count ``prod over pairs of (L[i] + 1)`` exceeds
    ``max_states``. ``stats``: ``wall_time_s``, ``states_total``,
    ``states_feasible``.
    """
    t0 = time.perf_counter()
    pairs = _pairs_receiver_major(inst.n)
    states_total = math.prod(int(inst.L[i]) + 1 for i, _ in pairs)
    if states_total > max_states:
        raise ValueError(
            f"naive enumeration needs {states_total:,} states, more than "
            f"max_states={max_states:,}; use solve_pruned or a smaller instance"
        )
    pre = precompute(inst)
    level = np.zeros((inst.n, inst.n), dtype=np.int64)
    best_level = level.copy()
    best_sorted = sorted_q(inst, level, pre=pre)  # all-zero is always feasible
    feasible = 0
    level_ranges = [range(int(inst.L[i]) + 1) for i, _ in pairs]
    for combo in itertools.product(*level_ranges):
        for (i, j), lvl in zip(pairs, combo, strict=True):
            level[i, j] = lvl
        if not check_feasibility(inst, level).ok:
            continue
        feasible += 1
        cand = sorted_q(inst, level, pre=pre)
        if lex_compare(cand, best_sorted) > 0:
            best_sorted = cand
            best_level = level.copy()
    stats: dict[str, Any] = {
        "states_total": states_total,
        "states_feasible": feasible,
    }
    result = finalize_result(inst, best_level, stats, "solve_naive")
    stats["wall_time_s"] = time.perf_counter() - t0
    return result


def solve_pruned(inst: Instance) -> SolveResult:
    """DFS over pairs with feasibility pruning; same optimum as ``solve_naive``.

    See the module docstring for the search scheme. ``stats``:
    ``wall_time_s``, ``nodes`` (level assignments tried), ``leaves``
    (complete allocations compared), ``pruned_branches`` (loop breaks on a
    budget violation).
    """
    t0 = time.perf_counter()
    pre = precompute(inst)
    n = inst.n
    pairs = _pairs_receiver_major(n)
    m = len(pairs)

    # Static per-pair data as plain Python lists: the DFS hot loop stays in
    # pure float arithmetic, no numpy per node.
    pair_rates: list[list[float]] = []
    pair_gains: list[list[float]] = []  # w[i][j] * q_corr[i][j][l]
    for i, j in pairs:
        width = int(inst.L[i]) + 1
        pair_rates.append([float(x) for x in inst.r[i, :width]])
        pair_gains.append(
            [float(inst.w[i, j] * pre.q_corr[i, j, lvl]) for lvl in range(width)]
        )
    b_limit = [float(x) + FEAS_TOL for x in inst.b_hat]
    B_limit = float(inst.B) + FEAS_TOL

    Q = [0.0] * n  # incremental Q_corr per receiver
    load = [0.0] * n  # incremental downlink load per receiver
    lev_flat = [0] * m
    best_flat = [0] * m
    best_sorted: list[float] | None = None
    counters = {"nodes": 0, "leaves": 0, "pruned_branches": 0}

    def lex_greater(u: list[float], v: list[float]) -> bool:
        """Same semantics as evaluator.lex_compare(u, v) == 1, but cheap."""
        for a, b in zip(u, v, strict=True):
            d = a - b
            if d > TOL:
                return True
            if d < -TOL:
                return False
        return False

    def dfs(k: int, uplink: float) -> None:
        nonlocal best_sorted
        if k == m:
            counters["leaves"] += 1
            cand = sorted(Q)
            if best_sorted is None or lex_greater(cand, best_sorted):
                best_sorted = cand
                best_flat[:] = lev_flat
            return
        _, j = pairs[k]
        rates = pair_rates[k]
        gains = pair_gains[k]
        base_load = load[j]
        base_q = Q[j]
        limit = b_limit[j]
        for lvl, rate in enumerate(rates):
            if base_load + rate > limit or uplink + rate > B_limit:
                counters["pruned_branches"] += 1
                break  # bitrates are non-decreasing in lvl
            counters["nodes"] += 1
            load[j] = base_load + rate
            Q[j] = base_q + gains[lvl]
            lev_flat[k] = lvl
            dfs(k + 1, uplink + rate)
        load[j] = base_load
        Q[j] = base_q

    dfs(0, 0.0)  # level 0 is never pruned, so the all-zero leaf is always seen

    level = np.zeros((n, n), dtype=np.int64)
    for (i, j), lvl in zip(pairs, best_flat, strict=True):
        level[i, j] = lvl
    stats: dict[str, Any] = dict(counters)
    result = finalize_result(inst, level, stats, "solve_pruned")
    stats["wall_time_s"] = time.perf_counter() - t0
    return result
