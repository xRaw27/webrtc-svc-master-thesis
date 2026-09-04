"""Allocation algorithms (M3-M8), their common result type and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from sfu_alloc.evaluator import check_feasibility, sorted_q
from sfu_alloc.instance import Instance

__all__ = ["SolveResult", "finalize_result"]


@dataclass(frozen=True, eq=False)
class SolveResult:
    """Common result type returned by every algorithm (PLAN.md global decision).

    ``allocation`` is the chosen ``level`` matrix; ``sorted_q`` is recomputed
    from ``allocation`` by the evaluator (never taken from the solver);
    ``stats`` holds wall times and solver-specific diagnostics.
    """

    allocation: npt.NDArray[np.int64]
    sorted_q: npt.NDArray[np.float64]
    stats: dict[str, Any]


def finalize_result(
    inst: Instance, level: np.ndarray, stats: dict[str, Any], solver: str
) -> SolveResult:
    """Re-evaluate a solver's allocation before returning it (CLAUDE.md rule).

    Checks feasibility with the evaluator and recomputes ``sorted_q`` from
    the allocation — the returned vector never comes from the search or the
    MILP solver. Raises ``RuntimeError`` if the allocation is infeasible.
    """
    report = check_feasibility(inst, level)
    if not report.ok:
        raise RuntimeError(
            f"{solver} produced an infeasible allocation ({report.violated})"
        )
    level = level.copy()
    level.setflags(write=False)
    return SolveResult(allocation=level, sorted_q=sorted_q(inst, level), stats=stats)
