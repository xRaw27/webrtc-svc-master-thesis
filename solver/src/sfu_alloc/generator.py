"""Random instance generator and JSON (de)serialization (M2, PLAN.md).

``random_instance`` draws diverse, reproducible instances: realistic bitrate
ladders (scaled, truncated, jittered), canonical preference-based weights
(MODEL.md §3, always via ``weights_from_preferences``), a mix of tight and
loose downlinks and uplinks, and optional history. Determinism
contract: a single ``numpy.random.Generator`` seeded once drives every draw
in a fixed order, so the same seed yields the same instance bit for bit
(changing the draw order would change every seeded instance).

``to_json`` / ``from_json`` serialize instances exactly: floats survive the
round trip bit for bit (JSON uses shortest-repr), ladders are stored ragged
(only levels ``0..L[i]``, no NaN padding), and the unused ``w`` diagonal is
stored as 0. ``from_json`` re-runs full ``Instance`` validation.
"""

from __future__ import annotations

import json

import numpy as np

from sfu_alloc.instance import History, Instance, weights_from_preferences

__all__ = [
    "BASE_LADDERS",
    "SVC_L3T3_SPATIAL_FIRST",
    "SVC_L3T3_TEMPORAL_FIRST",
    "from_json",
    "random_instance",
    "to_json",
]

SVC_L3T3_TEMPORAL_FIRST: tuple[float, ...] = (90.0, 120.0, 150.0, 500.0, 1250.0)
"""L3T3 SVC stream, temporal-first path: S0T0, S0T1, S0T2, S1T2, S2T2 (kbps).

Cumulative bitrates of the 3 spatial layers (180p/360p/720p at full frame
rate: 150 / 500 / 1250 kbps) with temporal rate factors 0.6 / 0.8 / 1.0 for
T0/T1/T2: frame-rate steps come first, then the spatial layers.
"""

SVC_L3T3_SPATIAL_FIRST: tuple[float, ...] = (90.0, 300.0, 750.0, 1000.0, 1250.0)
"""L3T3 SVC stream, spatial-first path: S0T0, S1T0, S2T0, S2T1, S2T2 (kbps).

Same layer bitrates as ``SVC_L3T3_TEMPORAL_FIRST``, but the spatial layers
are added first at the lowest frame rate (factor 0.6), then the temporal
layers on top.
"""

BASE_LADDERS: tuple[tuple[float, ...], ...] = (
    (150.0, 250.0, 800.0),
    (100.0, 300.0, 600.0, 1200.0),
    SVC_L3T3_TEMPORAL_FIRST,
    SVC_L3T3_SPATIAL_FIRST,
)
"""Nonzero levels of realistic ladders (kbps), lowest first.

Ladders have mixed lengths: for sender ``i`` only ladders with at least
``L[i]`` nonzero levels are eligible.
"""


def _as_rng(seed_or_rng: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def random_instance(
    seed_or_rng: int | np.random.Generator,
    n: int,
    L_choices: tuple[int, ...] = (1, 2, 3),
    *,
    base_ladders: tuple[tuple[float, ...], ...] = BASE_LADDERS,
    scale_range: tuple[float, float] = (0.5, 2.5),
    jitter: float = 0.1,
    p_default_prefs: float = 0.5,
    u_range: tuple[float, float] = (0.2, 1.2),
    f_range: tuple[float, float] = (0.3, 1.1),
    p_no_history: float = 0.3,
    phi_k_choices: tuple[float, ...] = (1.0, 0.5, 0.25),
    alpha_choices: tuple[float, ...] = (0.0, 0.1, 0.3),
    T_stab_choices: tuple[int, ...] = (2, 3, 4, 5, 6),
) -> Instance:
    """Draw a random valid instance (defaults exactly per PLAN.md M2).

    ``seed_or_rng`` is an integer seed or a ready ``numpy.random.Generator``.
    Per sender: a base ladder is picked uniformly among those with at least
    ``L[i]`` nonzero levels, scaled by a factor from
    ``scale_range``, truncated to the drawn ``L[i]`` lowest levels, jittered
    by ``+-jitter`` per level, re-sorted to stay monotone, with ``r[i][0]=0``
    forced. Weights always come from ``weights_from_preferences`` (canonical
    construction, MODEL.md §3): with probability ``p_default_prefs`` all
    preferences are 1 (one coin per instance), otherwise each pair
    independently keeps ``p = 1`` with probability 1/2 or draws
    ``p ~ Uniform(1, 2)``. Downlinks are
    ``b_hat[j] = u_j * sum_{i != j} r[i][L_i]``; the uplink budget is
    ``B = f * sum_j min(b_hat[j], sum_{i != j} r[i][L_i])``. History is None
    with probability ``p_no_history``, otherwise ``lambda_prev`` is uniform
    per pair and ``tau`` uniform in ``{1, ..., T_stab + 2}``.

    Raises ``ValueError`` when ``n < 2`` or when ``L_choices`` exceeds the
    length of the longest base ladder. The result always passes validation
    (it is constructed through ``Instance``).
    """
    rng = _as_rng(seed_or_rng)
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    longest = max(len(b) for b in base_ladders)
    if max(L_choices) > longest:
        raise ValueError(
            f"L_choices {L_choices} exceed the longest base ladder ({longest} levels)"
        )

    L = rng.choice(np.asarray(L_choices), size=n).astype(np.int64)

    r_rows: list[list[float]] = []
    for i in range(n):
        eligible = [b for b in base_ladders if len(b) >= int(L[i])]
        base = np.asarray(eligible[int(rng.integers(len(eligible)))])
        scale = rng.uniform(*scale_range)
        levels = base[: int(L[i])] * scale
        levels = levels * (1.0 + rng.uniform(-jitter, jitter, size=levels.size))
        levels = np.sort(levels)  # re-enforce monotonicity after jitter
        r_rows.append([0.0, *levels.tolist()])

    if rng.random() < p_default_prefs:
        p = np.ones((n, n))
    else:
        keep_default = rng.random(size=(n, n)) < 0.5
        p = np.where(keep_default, 1.0, rng.uniform(1.0, 2.0, size=(n, n)))
    w = weights_from_preferences(p, r_rows, L)

    top = np.array([row[-1] for row in r_rows])
    sum_top = top.sum() - top  # sum_{i != j} r[i][L_i], per receiver j
    u = rng.uniform(*u_range, size=n)
    b_hat = u * sum_top
    f = rng.uniform(*f_range)
    B = float(f * np.minimum(b_hat, sum_top).sum())

    phi_k = float(rng.choice(np.asarray(phi_k_choices)))
    alpha = float(rng.choice(np.asarray(alpha_choices)))
    T_stab = int(rng.choice(np.asarray(T_stab_choices)))

    history = None
    if rng.random() >= p_no_history:
        lambda_prev = np.stack(
            [rng.integers(0, int(L[i]) + 1, size=n) for i in range(n)]
        )
        tau = rng.integers(1, T_stab + 3, size=(n, n))
        history = History(lambda_prev=lambda_prev, tau=tau)

    return Instance(
        n=n,
        L=L,
        r=r_rows,
        w=w,
        b_hat=b_hat,
        B=B,
        phi_k=phi_k,
        alpha=alpha,
        T_stab=T_stab,
        history=history,
    )


def to_json(inst: Instance) -> str:
    """Serialize an instance to a strict-JSON string (exact float round trip).

    Ladders are stored ragged (levels ``0..L[i]`` only, so no NaN appears);
    the unused ``w`` diagonal is stored as 0.
    """
    eye = np.eye(inst.n, dtype=bool)
    doc = {
        "version": 1,
        "n": inst.n,
        "L": inst.L.tolist(),
        "r": [inst.r[i, : int(inst.L[i]) + 1].tolist() for i in range(inst.n)],
        "w": np.where(eye, 0.0, inst.w).tolist(),
        "b_hat": inst.b_hat.tolist(),
        "B": inst.B,
        "phi_k": inst.phi_k,
        "alpha": inst.alpha,
        "T_stab": inst.T_stab,
        "history": None
        if inst.history is None
        else {
            "lambda_prev": inst.history.lambda_prev.tolist(),
            "tau": inst.history.tau.tolist(),
        },
    }
    return json.dumps(doc, allow_nan=False)


def from_json(s: str) -> Instance:
    """Parse ``to_json`` output back into a fully validated ``Instance``."""
    doc = json.loads(s)
    version = doc.get("version")
    if version != 1:
        raise ValueError(f"unsupported instance JSON version: {version!r}")
    hist_doc = doc["history"]
    history = (
        None
        if hist_doc is None
        else History(
            lambda_prev=np.asarray(hist_doc["lambda_prev"], dtype=np.int64),
            tau=np.asarray(hist_doc["tau"], dtype=np.int64),
        )
    )
    return Instance(
        n=doc["n"],
        L=np.asarray(doc["L"], dtype=np.int64),
        r=doc["r"],
        w=doc["w"],
        b_hat=doc["b_hat"],
        B=doc["B"],
        phi_k=doc["phi_k"],
        alpha=doc["alpha"],
        T_stab=doc["T_stab"],
        history=history,
    )
