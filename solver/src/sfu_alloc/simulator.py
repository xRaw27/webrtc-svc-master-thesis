"""Multi-epoch closed loop (PLAN.md M8).

``update_history`` implements the state update of MODEL.md §8 as a pure
function; ``run_epochs`` executes ONE algorithm's independent closed loop
over an ``EpochScenario``: build the epoch-``t`` instance, solve, record,
update the history from the algorithm's OWN allocation, repeat. Running
several algorithms over the same scenario therefore yields legitimately
diverging trajectories (each carries its own history); the harness in
``benchmark/epochs.py`` does exactly that.

Per-epoch record schema (one dict per epoch, consumed by the harness):
``epoch`` (0-based), ``wall_time_s``, ``q_min``, ``q_mean``, ``sorted_q``
(list), ``changes_penalized``, ``changes_free`` — level changes vs the
previous epoch, split by whether the pair's ``tau`` at change time was
below ``T_stab`` (penalized) or not (free); epoch 0 reports 0/0 — plus
``level`` (the allocation matrix as nested lists) and ``q`` (the
per-receiver Q̃ vector, unsorted, so participant identity is kept); the
harness CSV skips these two, the timeline visualisation consumes them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from sfu_alloc.algorithms import SolveResult
from sfu_alloc.evaluator import q_corr_vector
from sfu_alloc.instance import History, Instance
from sfu_alloc.scenarios import EpochScenario, scenario_instance

__all__ = ["count_changes", "run_epochs", "update_history"]


def update_history(history: History | None, level: np.ndarray) -> History:
    """MODEL.md §8 state update as a pure function.

    After epoch 1 (``history is None``): ``lambda_prev := level`` and
    ``tau := 1`` everywhere (the first allocation counts as a change).
    Afterwards: ``tau := 1`` where the level changed, else ``tau + 1``;
    ``lambda_prev := level``.
    """
    level = np.asarray(level, dtype=np.int64)
    if history is None:
        return History(lambda_prev=level.copy(), tau=np.ones_like(level))
    changed = level != history.lambda_prev
    tau = np.where(changed, 1, history.tau + 1).astype(np.int64)
    return History(lambda_prev=level.copy(), tau=tau)


def count_changes(
    history: History | None, level: np.ndarray, T_stab: int
) -> tuple[int, int]:
    """Off-diagonal level changes vs the previous epoch (penalized, free).

    A change of pair ``(i, j)`` is penalized when its ``tau`` at change
    time (the tau the epoch was solved with) is below ``T_stab``. With no
    previous epoch the result is ``(0, 0)``.
    """
    if history is None:
        return 0, 0
    level = np.asarray(level)
    off = ~np.eye(level.shape[0], dtype=bool)
    changed = (level != history.lambda_prev) & off
    penalized = changed & (history.tau < T_stab)
    return int(penalized.sum()), int((changed & ~penalized).sum())


def run_epochs(
    scenario: EpochScenario,
    algorithm: Callable[[Instance], SolveResult],
) -> list[dict[str, Any]]:
    """Run one algorithm's closed loop over the scenario (record schema above).

    Epoch 0 solves with ``history = None``; every later epoch carries the
    history produced by this algorithm's previous allocation (MODEL.md §8).
    """
    history: History | None = None
    records: list[dict[str, Any]] = []
    for t in range(scenario.epochs):
        inst = scenario_instance(scenario, t, history)
        t0 = time.perf_counter()
        res = algorithm(inst)
        wall = time.perf_counter() - t0
        level = np.asarray(res.allocation, dtype=np.int64)
        penalized, free = count_changes(history, level, scenario.T_stab)
        q = res.sorted_q
        qv = q_corr_vector(inst, level)
        records.append(
            {
                "epoch": t,
                "wall_time_s": wall,
                "q_min": float(q[0]),
                "q_mean": float(np.mean(q)),
                "sorted_q": [float(x) for x in q],
                "q": [float(x) for x in qv],
                "changes_penalized": penalized,
                "changes_free": free,
                "level": level.tolist(),
            }
        )
        history = update_history(history, level)
    return records
