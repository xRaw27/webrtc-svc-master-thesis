"""Problem data model: ``History``, ``Instance``, validation, precomputation.

Implements MODEL.md §3 (problem data, validation) and §4 (derived
coefficients). Level-indexed arrays are rectangular with shape
``(n, L_max + 1)`` and padded with NaN above ``L[i]``, so accidental use of a
nonexistent level propagates loudly instead of failing silently. Diagonal
entries of pair-indexed ``(n, n)`` arrays are unused: validation ignores them
and ``precompute`` sets them to NaN in ``c`` and ``q_corr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "History",
    "Instance",
    "Precomputed",
    "instances_equal",
    "precompute",
    "weights_from_preferences",
]

_WEIGHT_TOL = 1e-9
"""Tolerance for per-receiver weight sums (MODEL.md §3)."""


def _readonly(arr: np.ndarray) -> np.ndarray:
    arr.setflags(write=False)
    return arr


def _int_square(value: object, name: str) -> npt.NDArray[np.int64]:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array, got dtype {arr.dtype}")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square (n, n) matrix, got {arr.shape}")
    return _readonly(arr.astype(np.int64))


def _ladders(value: object, L: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Build and validate the rectangular ``(n, L_max + 1)`` ladder array.

    Accepts ragged per-sender rows of length ``L[i] + 1`` or a rectangular
    array of width ``L_max + 1`` (entries above ``L[i]`` are ignored). The
    result is NaN above ``L[i]`` and read-only.
    """
    n = L.shape[0]
    L_max = int(L.max())
    out = np.full((n, L_max + 1), np.nan)
    try:
        rows = list(value)
    except TypeError as exc:
        raise ValueError("r must be a sequence of per-sender ladders") from exc
    if len(rows) != n:
        raise ValueError(f"r must have {n} rows, got {len(rows)}")
    for i, row in enumerate(rows):
        row_arr = np.asarray(row, dtype=np.float64)
        if row_arr.ndim != 1:
            raise ValueError(f"r[{i}] must be one-dimensional")
        width = int(L[i]) + 1
        if row_arr.shape[0] not in (width, L_max + 1):
            raise ValueError(
                f"r[{i}] must have length L[i]+1={width} "
                f"(or L_max+1={L_max + 1}), got {row_arr.shape[0]}"
            )
        out[i, :width] = row_arr[:width]
        ladder = out[i, :width]
        if not np.all(np.isfinite(ladder)):
            raise ValueError(f"r[{i}] contains non-finite values")
        if ladder[0] != 0.0:
            raise ValueError(f"r[{i}][0] must be 0, got {ladder[0]}")
        if np.any(np.diff(ladder) < 0):
            raise ValueError(f"r[{i}] must be non-decreasing")
        if ladder[width - 1] <= 0.0:
            raise ValueError(f"r[{i}][L[i]] must be > 0")
    return _readonly(out)


def weights_from_preferences(
    p: object, r: object, L: object
) -> npt.NDArray[np.float64]:
    """Canonical receiver weights from preferences and max bitrates (MODEL.md §3).

    ``w[i][j] = p[i][j] * r[i][L[i]] / sum_{k != j} p[k][j] * r[k][L[k]]``
    for ``i != j``; diagonal entries are 0. ``p[i][j] > 0`` is the preference
    of receiver ``j`` for stream ``i`` (all-ones ``p`` gives purely
    structural weights, proportional to maximum bitrates). ``r`` accepts the
    same ragged or rectangular form as ``Instance``; ``p`` diagonal entries
    are ignored. This function is the only home of the formula — the
    generator and the multi-epoch simulator (M10) both call it.
    """
    L_arr = np.asarray(L)
    if not np.issubdtype(L_arr.dtype, np.integer) or L_arr.ndim != 1:
        raise ValueError("L must be a 1-D integer array")
    L_arr = L_arr.astype(np.int64)
    n = L_arr.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 participants, got {n}")
    if np.any(L_arr < 1):
        raise ValueError("every L[i] must be >= 1")
    ladders = _ladders(r, L_arr)

    p_arr = np.array(p, dtype=np.float64)
    if p_arr.shape != (n, n):
        raise ValueError(f"p must have shape ({n}, {n}), got {p_arr.shape}")
    off = ~np.eye(n, dtype=bool)
    if not np.all(np.isfinite(p_arr[off])) or np.any(p_arr[off] <= 0):
        raise ValueError("p[i][j] must be finite and > 0 for every pair i != j")

    top = ladders[np.arange(n), L_arr]  # r[i][L[i]], validated > 0
    scores = np.where(off, p_arr * top[:, None], 0.0)
    return _readonly(scores / scores.sum(axis=0, keepdims=True))


@dataclass(frozen=True, eq=False)
class History:
    """Per-pair history state (MODEL.md §3): λ^{t-1} and τ^t.

    ``lambda_prev[i, j]`` is the level of pair ``(i, j)`` in the previous
    epoch; ``tau[i, j] >= 1`` counts epochs since its last level change and
    is already valid for the current epoch. Diagonal entries are ignored.
    The upper bound ``lambda_prev[i, j] <= L[i]`` is validated by
    ``Instance`` (it needs ``L``).
    """

    lambda_prev: npt.NDArray[np.int64]
    tau: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        lam = _int_square(self.lambda_prev, "lambda_prev")
        tau = _int_square(self.tau, "tau")
        if lam.shape != tau.shape:
            raise ValueError(
                f"lambda_prev and tau must have the same shape, "
                f"got {lam.shape} and {tau.shape}"
            )
        off = ~np.eye(lam.shape[0], dtype=bool)
        if np.any(lam[off] < 0):
            raise ValueError("lambda_prev entries must be >= 0")
        if np.any(tau[off] < 1):
            raise ValueError("tau entries must be >= 1 (MODEL.md §3)")
        object.__setattr__(self, "lambda_prev", lam)
        object.__setattr__(self, "tau", tau)


@dataclass(frozen=True, eq=False)
class Instance:
    """Immutable single-epoch problem instance (MODEL.md §3).

    Fields follow the thesis-to-code mapping of MODEL.md §2. ``r`` may be
    given ragged (row ``i`` of length ``L[i] + 1``); it is stored rectangular
    with NaN above ``L[i]``. All array fields are normalized to read-only
    numpy arrays. Construction raises ``ValueError`` on any violation of §3.
    """

    n: int
    L: npt.NDArray[np.int64]
    r: npt.NDArray[np.float64]
    w: npt.NDArray[np.float64]
    b_hat: npt.NDArray[np.float64]
    B: float
    phi_k: float
    alpha: float
    T_stab: int
    history: History | None

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int | np.integer):
            raise ValueError(f"n must be an integer, got {self.n!r}")
        n = int(self.n)
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")

        L = np.asarray(self.L)
        if not np.issubdtype(L.dtype, np.integer):
            raise ValueError(f"L must be an integer array, got dtype {L.dtype}")
        if L.shape != (n,):
            raise ValueError(f"L must have shape ({n},), got {L.shape}")
        L = _readonly(L.astype(np.int64))
        if np.any(L < 1):
            raise ValueError("every L[i] must be >= 1")

        r = _ladders(self.r, L)

        w = np.array(self.w, dtype=np.float64)
        if w.shape != (n, n):
            raise ValueError(f"w must have shape ({n}, {n}), got {w.shape}")
        off = ~np.eye(n, dtype=bool)
        if not np.all(np.isfinite(w[off])):
            raise ValueError("off-diagonal w entries must be finite")
        if np.any(w[off] <= 0):
            raise ValueError("w[i][j] must be > 0 for every pair i != j")
        recv_sums = np.where(off, w, 0.0).sum(axis=0)
        if np.any(np.abs(recv_sums - 1.0) > _WEIGHT_TOL):
            raise ValueError(
                f"weights of every receiver must sum to 1 within "
                f"{_WEIGHT_TOL}, got sums {recv_sums}"
            )

        b_hat = np.array(self.b_hat, dtype=np.float64)
        if b_hat.shape != (n,):
            raise ValueError(f"b_hat must have shape ({n},), got {b_hat.shape}")
        if not np.all(np.isfinite(b_hat)) or np.any(b_hat < 0):
            raise ValueError("b_hat entries must be finite and >= 0")

        B = float(self.B)
        if not np.isfinite(B) or B < 0:
            raise ValueError(f"B must be finite and >= 0, got {B}")

        phi_k = float(self.phi_k)
        if not 0.0 < phi_k <= 1.0:
            raise ValueError(f"phi_k must be in (0, 1], got {phi_k}")

        alpha = float(self.alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if isinstance(self.T_stab, bool) or not isinstance(
            self.T_stab, int | np.integer
        ):
            raise ValueError(f"T_stab must be an integer, got {self.T_stab!r}")
        T_stab = int(self.T_stab)
        if T_stab < 2:
            raise ValueError(f"T_stab must be >= 2, got {T_stab}")

        if self.history is not None:
            if not isinstance(self.history, History):
                raise ValueError("history must be a History instance or None")
            if self.history.lambda_prev.shape != (n, n):
                raise ValueError(
                    f"history arrays must have shape ({n}, {n}), "
                    f"got {self.history.lambda_prev.shape}"
                )
            too_high = (self.history.lambda_prev > L[:, None]) & off
            if np.any(too_high):
                i, j = map(int, np.argwhere(too_high)[0])
                raise ValueError(
                    f"lambda_prev[{i}][{j}] = "
                    f"{int(self.history.lambda_prev[i, j])} "
                    f"exceeds L[{i}] = {int(L[i])}"
                )

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "L", L)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "w", _readonly(w))
        object.__setattr__(self, "b_hat", _readonly(b_hat))
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "phi_k", phi_k)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "T_stab", T_stab)


@dataclass(frozen=True, eq=False)
class Precomputed:
    """Derived coefficient arrays (MODEL.md §4), constants during a solve.

    Shapes: ``q`` is ``(n, L_max + 1)``; ``c`` and ``q_corr`` are
    ``(n, n, L_max + 1)``. Entries above ``L[i]`` and on the ``i == j``
    diagonal of ``c`` / ``q_corr`` are NaN (unused).
    """

    q: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    q_corr: npt.NDArray[np.float64]


def precompute(inst: Instance) -> Precomputed:
    """Compute ``q``, ``c`` and ``q_corr`` exactly per MODEL.md §4.

    ``q[i][l] = (r[i][l] / r[i][L[i]]) ** phi_k``. The change penalty ``c``
    is zero when ``history is None`` (epoch 1); otherwise it is
    ``alpha * (T_stab - min(tau, T_stab)) / (T_stab - 1)`` for any level
    different from ``lambda_prev`` and zero for keeping it.
    ``q_corr = q - c`` (may be negative).
    """
    n, L, r = inst.n, inst.L, inst.r
    L_max = r.shape[1] - 1
    top = r[np.arange(n), L]
    q = (r / top[:, None]) ** inst.phi_k

    levels = np.arange(L_max + 1)
    if inst.history is None:
        c = np.zeros((n, n, L_max + 1))
    else:
        penalty = (
            inst.alpha
            * (inst.T_stab - np.minimum(inst.history.tau, inst.T_stab))
            / (inst.T_stab - 1)
        )
        keep = levels[None, None, :] == inst.history.lambda_prev[:, :, None]
        c = np.where(keep, 0.0, penalty[:, :, None])
    pad = levels[None, :] > L[:, None]
    c = np.where(pad[:, None, :], np.nan, c)
    diag = np.arange(n)
    c[diag, diag, :] = np.nan
    q_corr = q[:, None, :] - c

    return Precomputed(q=_readonly(q), c=_readonly(c), q_corr=_readonly(q_corr))


def instances_equal(a: Instance, b: Instance) -> bool:
    """Exact (bitwise) semantic equality of two instances.

    Compares every field with exact float equality (the dataclasses disable
    auto ``__eq__`` because numpy fields break it). The unused ``w`` diagonal
    is ignored; NaN ladder padding compares equal via ``equal_nan``.
    """
    if a.n != b.n or a.T_stab != b.T_stab:
        return False
    if (a.B, a.phi_k, a.alpha) != (b.B, b.phi_k, b.alpha):
        return False
    if not np.array_equal(a.L, b.L):
        return False
    if not np.array_equal(a.r, b.r, equal_nan=True):
        return False
    off = ~np.eye(a.n, dtype=bool)
    if not np.array_equal(a.w[off], b.w[off]):
        return False
    if not np.array_equal(a.b_hat, b.b_hat):
        return False
    if (a.history is None) != (b.history is None):
        return False
    if a.history is not None and b.history is not None:
        if not np.array_equal(a.history.lambda_prev, b.history.lambda_prev):
            return False
        if not np.array_equal(a.history.tau, b.history.tau):
            return False
    return True
