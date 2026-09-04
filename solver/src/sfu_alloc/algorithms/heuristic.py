"""Quantized decomposition + progressive filling heuristic (M7, "Algorithm 3").

The fast approximate algorithm implementable in a real SFU. Design per
PLAN.md M7, implemented exactly:

1. Bitrates are quantized on a grid of ``delta_kbps``: level ``l`` of stream
   ``i`` weighs ``cells(i, l) = ceil(r[i][l] / delta)`` cells (0 for level 0).
   The conservative rounding guarantees true bitrates fit whenever cell
   budgets fit.
2. Per-receiver cap in cells:
   ``cap[j] = floor(min(b_hat[j], sum_{i != j} r[i][L_i], B) / delta)``.
3. Per-receiver DP (multiple-choice knapsack): ``V_j[m]`` is the best
   weighted corrected quality of receiver ``j`` using at most ``m`` cells,
   with backpointers; the inner loop over ``m`` is vectorized with numpy.
4. Each ``V_j`` is reduced to breakpoints: strictly increasing values at
   minimal cell cost; ``m = 0`` is always kept (``V_j[0]`` can be negative
   when history is present).
5. Master — greedy progressive filling: a heap keyed by the current value;
   the lowest receiver is repeatedly advanced to its next breakpoint if the
   incremental cell cost fits the remaining global budget
   ``floor(B / delta) - spent``; a receiver that cannot advance is saturated
   and removed permanently (the global budget only shrinks, so this is
   safe). The loop stops when the heap is empty.
6. ``level`` is reconstructed from the DP backpointers at each receiver's
   final cell budget, re-evaluated with the evaluator, and feasibility is
   asserted.

The greedy master is not always lex-optimal even on the quantized problem.
Counterexample: receivers A and B both at 0.4; A's next breakpoint costs
300 kbps and lifts it to 0.5, B's costs 100 kbps and lifts it to 0.45; with
300 kbps left, lex-optimal gives everything to A, while greedy with a
cheapest-step tie-break picks B. This is an accepted cost of the heuristic.

The optional ``master="exact"`` (PLAN.md M7 step 7) is NOT implemented.
"""

from __future__ import annotations

import heapq
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from sfu_alloc.algorithms import SolveResult, finalize_result
from sfu_alloc.instance import Instance, precompute

__all__ = ["solve_heuristic"]

_CELL_SENTINEL = 2**31
"""Cell weight stored in the NaN padding; larger than any real cap."""


def _receiver_dp(
    senders: list[int],
    cap: int,
    cells: npt.NDArray[np.int64],
    values: npt.NDArray[np.float64],
    L: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int8]]:
    """Multiple-choice knapsack DP for one receiver (PLAN.md M7 step 3).

    ``V[m]`` = best total weighted quality over this receiver's senders
    using at most ``m`` cells; ``bp[g, m]`` = the level chosen for group
    ``g`` (sender ``senders[g]``) in that optimum. Level 0 costs 0 cells,
    so the DP is never infeasible. Ties keep the lowest level.
    """
    dp = np.zeros(cap + 1)
    bp = np.zeros((len(senders), cap + 1), dtype=np.int8)
    for g, i in enumerate(senders):
        best = np.full(cap + 1, -np.inf)
        best_lvl = np.zeros(cap + 1, dtype=np.int8)
        for lvl in range(int(L[i]) + 1):
            cost = int(cells[i, lvl])
            if cost > cap:
                break  # cells are non-decreasing in lvl
            cand = dp[: cap + 1 - cost] + values[i, lvl]
            view = best[cost:]
            mask = cand > view
            view[mask] = cand[mask]
            best_lvl[cost:][mask] = lvl
        dp = best
        bp[g] = best_lvl
    return dp, bp


def _breakpoints(
    V: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Strictly-increasing value breakpoints at minimal cell cost (step 4).

    ``V`` is non-decreasing in ``m``; the first occurrence of every strictly
    larger value is kept. ``m = 0`` is always kept.
    """
    keep = np.empty(V.shape[0], dtype=bool)
    keep[0] = True
    keep[1:] = V[1:] > V[:-1]
    idx = np.nonzero(keep)[0].astype(np.int64)
    return idx, V[idx]


def solve_heuristic(
    inst: Instance,
    delta_kbps: float = 1.0,
    master: str = "greedy",
) -> SolveResult:
    """Quantized decomposition + progressive filling (PLAN.md M7).

    ``delta_kbps`` is the quantization grid; smaller values track the true
    problem more closely at higher DP cost. ``master`` selects the budget
    split across receivers; only ``"greedy"`` is implemented
    (``"exact"`` — PLAN.md M7 step 7 — raises ``NotImplementedError``).
    ``stats`` records the cell budget and spend, DP/master times and the
    breakpoint count. The result is re-evaluated with the evaluator and
    feasibility is asserted (``RuntimeError`` if violated).
    """
    t0 = time.perf_counter()
    if master == "exact":
        raise NotImplementedError(
            "master='exact' is the optional PLAN.md M7 step 7 (not implemented)"
        )
    if master != "greedy":
        raise ValueError(f"unknown master {master!r}; expected 'greedy'")
    delta = float(delta_kbps)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta_kbps must be a finite float > 0, got {delta}")

    pre = precompute(inst)
    n = inst.n
    L = inst.L
    r = inst.r

    # Step 1: conservative cell weights (NaN padding -> huge sentinel).
    with np.errstate(invalid="ignore"):
        cells_f = np.ceil(r / delta)
    cells = np.where(np.isnan(cells_f), _CELL_SENTINEL, cells_f).astype(np.int64)

    # Step 2: per-receiver caps and the global budget, all in cells.
    top = r[np.arange(n), L]
    sum_top = top.sum() - top  # sum_{i != j} r[i][L_i], per receiver j
    budget = int(np.floor(inst.B / delta))
    cap = np.floor(np.minimum(np.minimum(inst.b_hat, sum_top), inst.B) / delta).astype(
        np.int64
    )

    # Steps 3-4: per-receiver DP and breakpoint reduction.
    values = inst.w[:, :, None] * pre.q_corr  # values[i, j, lvl]

    sender_lists: list[list[int]] = []
    bp_tables: list[npt.NDArray[np.int8]] = []
    bp_m: list[npt.NDArray[np.int64]] = []
    bp_v: list[npt.NDArray[np.float64]] = []
    for j in range(n):
        senders = [i for i in range(n) if i != j]
        V, bp = _receiver_dp(senders, int(cap[j]), cells, values[:, j, :], L)
        m_idx, vals = _breakpoints(V)
        sender_lists.append(senders)
        bp_tables.append(bp)
        bp_m.append(m_idx)
        bp_v.append(vals)
    dp_time = time.perf_counter() - t0

    # Step 5: greedy progressive filling over breakpoints.
    t_master = time.perf_counter()
    current = [0] * n  # breakpoint index per receiver
    spent = 0
    heap = [(float(bp_v[j][0]), j) for j in range(n)]
    heapq.heapify(heap)
    while heap:
        _, j = heapq.heappop(heap)
        nxt = current[j] + 1
        if nxt < bp_m[j].size:
            inc = int(bp_m[j][nxt] - bp_m[j][current[j]])
            if spent + inc <= budget:
                spent += inc
                current[j] = nxt
                heapq.heappush(heap, (float(bp_v[j][nxt]), j))
                continue
        # saturated: no next breakpoint, or it does not fit -> removed
    master_time = time.perf_counter() - t_master

    # Step 6: reconstruct the levels from the DP backpointers.
    level = np.zeros((n, n), dtype=np.int64)
    for j in range(n):
        m = int(bp_m[j][current[j]])
        senders = sender_lists[j]
        bp = bp_tables[j]
        for g in range(len(senders) - 1, -1, -1):
            lvl = int(bp[g, m])
            level[senders[g], j] = lvl
            m -= int(cells[senders[g], lvl])
        if m < 0:
            raise RuntimeError(f"backpointer walk for receiver {j} went below budget 0")

    stats: dict[str, Any] = {
        "master": master,
        "delta_kbps": delta,
        "budget_cells": budget,
        "spent_cells": spent,
        "breakpoints_total": int(sum(b.size for b in bp_m)),
        "dp_time_s": dp_time,
        "master_time_s": master_time,
    }
    result = finalize_result(inst, level, stats, "solve_heuristic")
    stats["wall_time_s"] = time.perf_counter() - t0
    return result
