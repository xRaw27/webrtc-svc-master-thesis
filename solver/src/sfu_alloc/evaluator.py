"""Ground-truth evaluation of allocations (MODEL.md §5-§6).

The functions here are the single source of truth used to judge every
algorithm in this repo: feasibility checking, the corrected reception
quality vector ``Q_corr``, its ascending sort, and lexicographic comparison
with tolerance. Feasibility uses an absolute slack tolerance of ``FEAS_TOL``
kbps to absorb float summation noise and MILP boundary solutions. This
module performs no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sfu_alloc.constants import FEAS_TOL, TOL
from sfu_alloc.instance import Instance, Precomputed, precompute

__all__ = [
    "FeasibilityReport",
    "check_feasibility",
    "lex_compare",
    "q_corr_vector",
    "sorted_q",
]


@dataclass(frozen=True, eq=False)
class FeasibilityReport:
    """Result of ``check_feasibility``.

    ``downlink_slack[j] = b_hat[j] - (bitrate forwarded to j)`` and
    ``uplink_slack = B - (total forwarded bitrate)``. ``violated`` names the
    first violated constraint — ``"downlink[j]"`` for the smallest such
    ``j``, then ``"uplink"`` — or is None when the allocation is feasible.
    """

    ok: bool
    downlink_slack: npt.NDArray[np.float64]
    uplink_slack: float
    violated: str | None


def _validated_level(inst: Instance, level: object) -> npt.NDArray[np.int64]:
    """Validate an allocation matrix per MODEL.md §5.

    Returns an int64 copy with the (ignored) diagonal zeroed. Raises
    ``ValueError`` for a wrong shape, a non-integer dtype, or any
    off-diagonal entry outside ``{0, ..., L[i]}``.
    """
    arr = np.asarray(level)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"level must be an integer array, got dtype {arr.dtype}")
    if arr.shape != (inst.n, inst.n):
        raise ValueError(f"level must have shape ({inst.n}, {inst.n}), got {arr.shape}")
    lev = arr.astype(np.int64)
    np.fill_diagonal(lev, 0)
    out_of_range = (lev < 0) | (lev > inst.L[:, None])
    if np.any(out_of_range):
        i, j = map(int, np.argwhere(out_of_range)[0])
        raise ValueError(
            f"level[{i}][{j}] = {int(arr[i, j])} is outside "
            f"{{0, ..., L[{i}] = {int(inst.L[i])}}}"
        )
    return lev


def check_feasibility(inst: Instance, level: object) -> FeasibilityReport:
    """Check the downlink and uplink constraints of MODEL.md §5.

    A constraint counts as satisfied when its slack is ``>= -FEAS_TOL``
    (absolute, kbps). Downlinks are checked for ``j = 0..n-1`` in order,
    then the uplink; the first violation is reported in ``violated``.
    """
    lev = _validated_level(inst, level)
    n = inst.n
    rates = inst.r[np.arange(n)[:, None], lev]
    np.fill_diagonal(rates, 0.0)
    load = rates.sum(axis=0)
    downlink_slack = inst.b_hat - load
    uplink_slack = float(inst.B - load.sum())
    violated = None
    bad = np.nonzero(downlink_slack < -FEAS_TOL)[0]
    if bad.size:
        violated = f"downlink[{int(bad[0])}]"
    elif uplink_slack < -FEAS_TOL:
        violated = "uplink"
    return FeasibilityReport(
        ok=violated is None,
        downlink_slack=downlink_slack,
        uplink_slack=uplink_slack,
        violated=violated,
    )


def q_corr_vector(
    inst: Instance,
    level: object,
    pre: Precomputed | None = None,
) -> npt.NDArray[np.float64]:
    """Corrected total reception quality ``Q_corr`` per receiver (§6).

    ``pre`` must be the result of ``precompute(inst)`` for the *same*
    instance (a shape mismatch is rejected; deeper mismatches cannot be
    detected); when None it is computed here. Feasibility is *not* checked —
    use ``check_feasibility`` for that.
    """
    lev = _validated_level(inst, level)
    if pre is None:
        pre = precompute(inst)

    n = inst.n
    expected = (n, n, inst.r.shape[1])
    if pre.q_corr.shape != expected:
        raise ValueError(
            f"pre does not match the instance: q_corr has shape "
            f"{pre.q_corr.shape}, expected {expected}"
        )
    vals = pre.q_corr[np.arange(n)[:, None], np.arange(n)[None, :], lev]
    contrib = inst.w * vals
    np.fill_diagonal(contrib, 0.0)
    return contrib.sum(axis=0)


def sorted_q(
    inst: Instance,
    level: object,
    pre: Precomputed | None = None,
) -> npt.NDArray[np.float64]:
    """The ``Q_corr`` vector sorted ascending (MODEL.md §6)."""
    return np.sort(q_corr_vector(inst, level, pre=pre))


def lex_compare(u: object, v: object, tol: float = TOL) -> int:
    """Lexicographic comparison with tolerance per MODEL.md §6.

    Scans both vectors from index 0; at the first index where the entries
    differ by more than ``tol``, the vector with the larger entry wins.
    Returns 1 if ``u`` is lex-greater, -1 if ``v`` is, 0 if lex-equal.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.ndim != 1 or u_arr.shape != v_arr.shape:
        raise ValueError(
            f"lex_compare expects two 1-D vectors of equal length, "
            f"got shapes {u_arr.shape} and {v_arr.shape}"
        )
    diff = u_arr - v_arr
    decisive = np.nonzero(np.abs(diff) > tol)[0]
    if decisive.size == 0:
        return 0
    return 1 if diff[decisive[0]] > 0 else -1
