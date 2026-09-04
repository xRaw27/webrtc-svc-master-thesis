"""Dataset-driven scenario generation (PLAN.md M7 static, M8 multi-epoch).

``random_static_instance`` builds a single-epoch instance
(``history = None``, penalties all zero) from the ladder dataset: every
participant sends the ladder of one uniformly random second of one
uniformly random dataset file, optionally scaled by a per-participant
factor. Bandwidth limits are drawn around a nominal per-stream estimate
(``EST_KBPS``) independently of the drawn ladders, so both loose and tight
constraints occur. Preferences follow the 80/20 default/custom rule of
PLAN.md M7; weights always go through ``weights_from_preferences``.

``random_epoch_scenario`` draws a fixed M8 scenario (everything decided
once: per-participant dataset file + starting second + ladder scale,
preferences, the bridge budget ``B``, and per-participant ``b_hat``
trajectories); ``scenario_instance`` then materializes the epoch-``t``
instance — the ladder walks the file second by second (wrap-around at the
end) and weights are recomputed every epoch from the fixed ``p`` and the
current ``r^t`` (MODEL.md §8).

Determinism contract: one ``numpy.random.Generator`` drives every draw in
a fixed order — static: per participant (file, second, scale), then
preferences per receiver, then ``b_hat``, then ``B``; epoch scenario: per
participant (file, start second, scale), then preferences, then ``B``,
then per-participant trajectories (kind first when ``trajectory="mix"``),
then ``alpha``, then ``T_stab`` (appended last, so scenario shapes seeded
before these draws existed stay intact). Changing the order would change
every seeded scenario.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfu_alloc.data.ladders import sample_ladder_row
from sfu_alloc.instance import History, Instance, weights_from_preferences

__all__ = [
    "EST_KBPS",
    "TRAJECTORY_KINDS",
    "TRAJECTORY_MIX_PROBS",
    "EpochScenario",
    "StaticDraw",
    "dataset_files",
    "random_epoch_scenario",
    "random_static_case",
    "random_static_instance",
    "scenario_instance",
]

TRAJECTORY_KINDS = ("constant", "smooth", "steps")
"""Per-participant b_hat trajectory kinds; "mix" draws one kind per participant."""

TRAJECTORY_MIX_PROBS = (0.2, 0.4, 0.4)
"""Kind probabilities (constant, smooth, steps) used by ``trajectory="mix"``."""

EST_KBPS = 2000.0
"""Nominal per-stream bitrate estimate (kbps) used by the bandwidth draws."""


def dataset_files(ladders: pd.DataFrame) -> list[str]:
    """All file stems ``{source}_{config}`` present in a loaded dataset."""
    pairs = ladders[["source", "config"]].drop_duplicates().itertuples(index=False)
    return sorted(f"{p.source}_{p.config}" for p in pairs)


def _split_stem(stem: str) -> tuple[str, str]:
    """``"bbb_L3T3_tf"`` -> ``("bbb", "L3T3_tf")``."""
    source, _, config = stem.partition("_")
    if not source or not config:
        raise ValueError(f"invalid ladder file stem: {stem!r}")
    return source, config


@dataclass(frozen=True, eq=False)
class StaticDraw:
    """The raw draws behind one static instance (scenario dumps, EVAL.md).

    ``files[i]`` / ``seconds[i]`` / ``scales[i]`` identify participant i's
    drawn ladder (dataset file stem, second of the recording, scale
    factor); ``p`` is the drawn preference matrix — the instance itself
    carries only the derived weights ``w``.
    """

    files: tuple[str, ...]
    seconds: tuple[int, ...]
    scales: tuple[float, ...]
    p: np.ndarray
    instance: Instance


def random_static_case(
    rng: np.random.Generator,
    n: int,
    ladders: pd.DataFrame,
    ladder_files: Sequence[str] | None = None,
    *,
    scale_range: tuple[float, float] = (1.0, 1.0),
    est_kbps: float = EST_KBPS,
    p_default_prefs: float = 0.8,
    pref_range: tuple[float, float] = (1.0, 2.0),
    phi_k: float = 0.5,
    alpha: float = 0.2,
    T_stab: int = 4,
) -> StaticDraw:
    """Draw one static (epoch-1) instance exactly per PLAN.md M7.

    - ladder per participant: uniform random file stem from
      ``ladder_files`` (default: every file in ``ladders``), uniform random
      second of it, scaled by a factor ``~ Uniform(scale_range)``;
    - ``b_hat[j] ~ Uniform(0.25 * E, 1.25 * E)`` with ``E = (n-1) * est_kbps``
      per receiver (the same shape as the ``B`` draw);
    - ``B ~ Uniform(0.25 * S, S)`` with ``S = n * (n-1) * est_kbps``;
    - preferences: with probability ``p_default_prefs`` a participant keeps
      all incoming preferences at 1, otherwise it draws one
      ``~ Uniform(pref_range)`` per incoming stream;
    - ``history = None`` always (``phi_k`` / ``alpha`` / ``T_stab`` are
      plain parameters and do not affect the objective here).

    Returns the instance together with the raw draws behind it
    (``StaticDraw``); ``random_static_instance`` is the instance-only
    wrapper.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    files = list(ladder_files) if ladder_files is not None else dataset_files(ladders)
    if not files:
        raise ValueError("ladder_files must not be empty")

    chosen: list[str] = []
    seconds: list[int] = []
    scales: list[float] = []
    r_rows: list[list[float]] = []
    for _ in range(n):
        stem = files[int(rng.integers(len(files)))]
        source, config = _split_stem(stem)
        _, second, ladder = sample_ladder_row(
            ladders, rng, config=config, source=source
        )
        scale = float(rng.uniform(*scale_range))
        chosen.append(stem)
        seconds.append(second)
        scales.append(scale)
        r_rows.append([value * scale for value in ladder])
    L = [len(row) - 1 for row in r_rows]

    p = np.ones((n, n))
    for j in range(n):
        if rng.random() >= p_default_prefs:
            senders = [i for i in range(n) if i != j]
            p[senders, j] = rng.uniform(*pref_range, size=n - 1)
    w = weights_from_preferences(p, r_rows, L)

    est = (n - 1) * est_kbps
    b_hat = rng.uniform(0.25 * est, 1.25 * est, size=n)
    total = n * (n - 1) * est_kbps
    B = float(rng.uniform(0.25 * total, total))

    inst = Instance(
        n=n,
        L=L,
        r=r_rows,
        w=w,
        b_hat=b_hat,
        B=B,
        phi_k=phi_k,
        alpha=alpha,
        T_stab=T_stab,
        history=None,
    )
    return StaticDraw(
        files=tuple(chosen),
        seconds=tuple(seconds),
        scales=tuple(scales),
        p=p,
        instance=inst,
    )


def random_static_instance(
    rng: np.random.Generator,
    n: int,
    ladders: pd.DataFrame,
    ladder_files: Sequence[str] | None = None,
    *,
    scale_range: tuple[float, float] = (1.0, 1.0),
    est_kbps: float = EST_KBPS,
    p_default_prefs: float = 0.8,
    pref_range: tuple[float, float] = (1.0, 2.0),
    phi_k: float = 0.5,
    alpha: float = 0.2,
    T_stab: int = 4,
) -> Instance:
    """Instance-only wrapper around ``random_static_case`` (same draws)."""
    return random_static_case(
        rng,
        n,
        ladders,
        ladder_files,
        scale_range=scale_range,
        est_kbps=est_kbps,
        p_default_prefs=p_default_prefs,
        pref_range=pref_range,
        phi_k=phi_k,
        alpha=alpha,
        T_stab=T_stab,
    ).instance


@dataclass(frozen=True, eq=False)
class EpochScenario:
    """A fixed multi-epoch scenario (PLAN.md M8): everything drawn once.

    ``ladder_tables[i]`` is participant i's full per-second ladder table
    from its dataset file, shape ``(seconds, L_i + 1)`` including the empty
    level 0. The epoch-``t`` ladder is row
    ``(start_seconds[i] + t) % seconds`` (or the pinned start row when
    ``freeze_ladder`` — the stability-test switch), scaled by
    ``scales[i]``. ``b_hat_traj`` has shape ``(epochs, n)``; ``B`` is one
    draw held for the whole run; ``p`` are the fixed preferences.
    ``trajectory_kinds[j]`` is the drawn kind of receiver j's ``b_hat``
    trajectory (all equal to ``trajectory`` unless it is ``"mix"``).
    """

    n: int
    epochs: int
    files: tuple[str, ...]
    start_seconds: tuple[int, ...]
    scales: tuple[float, ...]
    ladder_tables: tuple[np.ndarray, ...]
    p: np.ndarray
    B: float
    b_hat_traj: np.ndarray
    trajectory: str
    trajectory_kinds: tuple[str, ...]
    freeze_ladder: bool
    phi_k: float
    alpha: float
    T_stab: int


def _ladder_table(ladders: pd.DataFrame, source: str, config: str) -> np.ndarray:
    """Per-second ladder table ``(seconds, L + 1)`` with level 0 prepended."""
    sub = ladders[(ladders["source"] == source) & (ladders["config"] == config)]
    if sub.empty:
        raise ValueError(f"no dataset rows for {source}_{config}")
    wide = sub.pivot(index="second", columns="level", values="r_raw").sort_index()
    return np.column_stack([np.zeros(len(wide)), wide.to_numpy()])


def _draw_trajectory(
    rng: np.random.Generator,
    kind: str,
    epochs: int,
    low: float,
    high: float,
    waypoint_every: int,
    jump_prob: float,
) -> np.ndarray:
    """One participant's b_hat trajectory, values within ``[low, high]``."""
    if kind == "constant":
        return np.full(epochs, float(rng.uniform(low, high)))
    if kind == "smooth":
        xs = np.arange(0, epochs + waypoint_every, waypoint_every)
        ys = rng.uniform(low, high, size=len(xs))
        return np.interp(np.arange(epochs), xs, ys)
    if kind == "steps":
        values = np.empty(epochs)
        current = float(rng.uniform(low, high))
        for t in range(epochs):
            if t > 0 and rng.random() < jump_prob:
                current = float(rng.uniform(low, high))
            values[t] = current
        return values
    raise ValueError(f"unknown trajectory kind {kind!r}")


def random_epoch_scenario(
    rng: np.random.Generator,
    n: int,
    ladders: pd.DataFrame,
    ladder_files: Sequence[str] | None = None,
    *,
    epochs: int = 200,
    trajectory: str = "mix",
    freeze_ladder: bool = False,
    scale_range: tuple[float, float] = (1.0, 1.0),
    est_kbps: float = EST_KBPS,
    p_default_prefs: float = 0.8,
    pref_range: tuple[float, float] = (1.0, 2.0),
    phi_k: float = 0.5,
    alpha_range: tuple[float, float] = (0.1, 0.5),
    T_stab_range: tuple[int, int] = (10, 20),
    smooth_waypoint_every: int = 5,
    steps_jump_prob: float = 0.05,
) -> EpochScenario:
    """Draw one fixed multi-epoch scenario exactly per PLAN.md M8.

    Drawn once and held for the run: per participant (dataset file,
    starting second, ladder scale from ``scale_range``), the preferences
    (the same 80/20 rule as in M7), ``B ~ Uniform(0.25 * S, S)`` with
    ``S = n * (n-1) * est_kbps``, and per-participant ``b_hat``
    trajectories of the requested kind, each within
    ``[0.25, 1.25] * (n-1) * est_kbps`` — the same range as the static
    ``b_hat`` draw (``smooth`` interpolates waypoints drawn every
    ``smooth_waypoint_every`` epochs; ``steps`` jumps with per-epoch
    probability ``steps_jump_prob``, i.e. a mean holding time of
    ``1 / steps_jump_prob`` epochs; ``mix`` draws a kind per participant
    with probabilities 20% constant / 40% smooth / 40% steps). Finally
    ``alpha ~ Uniform(alpha_range)`` and ``T_stab ~ integer
    Uniform(T_stab_range)`` (both ends inclusive; the model requires
    ``T_stab >= 2``) — drawn LAST, so scenarios seeded before these draws
    existed keep their shape.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if trajectory != "mix" and trajectory not in TRAJECTORY_KINDS:
        raise ValueError(f"unknown trajectory {trajectory!r}")
    if T_stab_range[0] < 2:
        raise ValueError(
            f"T_stab_range must start at >= 2 (MODEL.md), got {T_stab_range}"
        )
    files = list(ladder_files) if ladder_files is not None else dataset_files(ladders)
    if not files:
        raise ValueError("ladder_files must not be empty")

    chosen: list[str] = []
    starts: list[int] = []
    scales: list[float] = []
    tables: list[np.ndarray] = []
    for _ in range(n):
        stem = files[int(rng.integers(len(files)))]
        source, config = _split_stem(stem)
        table = _ladder_table(ladders, source, config)
        chosen.append(stem)
        starts.append(int(rng.integers(len(table))))
        scales.append(float(rng.uniform(*scale_range)))
        tables.append(table)

    p = np.ones((n, n))
    for j in range(n):
        if rng.random() >= p_default_prefs:
            senders = [i for i in range(n) if i != j]
            p[senders, j] = rng.uniform(*pref_range, size=n - 1)

    total = n * (n - 1) * est_kbps
    B = float(rng.uniform(0.25 * total, total))

    est = (n - 1) * est_kbps
    low, high = 0.25 * est, 1.25 * est
    kinds: list[str] = []
    b_hat_traj = np.empty((epochs, n))
    for j in range(n):
        kind = (
            trajectory
            if trajectory != "mix"
            else str(rng.choice(TRAJECTORY_KINDS, p=TRAJECTORY_MIX_PROBS))
        )
        kinds.append(kind)
        b_hat_traj[:, j] = _draw_trajectory(
            rng, kind, epochs, low, high, smooth_waypoint_every, steps_jump_prob
        )

    alpha = float(rng.uniform(*alpha_range))
    T_stab = int(rng.integers(T_stab_range[0], T_stab_range[1] + 1))

    return EpochScenario(
        n=n,
        epochs=epochs,
        files=tuple(chosen),
        start_seconds=tuple(starts),
        scales=tuple(scales),
        ladder_tables=tuple(tables),
        p=p,
        B=B,
        b_hat_traj=b_hat_traj,
        trajectory=trajectory,
        trajectory_kinds=tuple(kinds),
        freeze_ladder=freeze_ladder,
        phi_k=phi_k,
        alpha=alpha,
        T_stab=T_stab,
    )


def scenario_instance(
    scenario: EpochScenario, epoch: int, history: History | None
) -> Instance:
    """Materialize the epoch-``t`` instance of a scenario (PLAN.md M8).

    The ladder of participant ``i`` is second
    ``(start_i + epoch) % file_length`` of its file (pinned to the start
    second when ``freeze_ladder``), scaled by its fixed factor; weights
    are recomputed from the fixed ``p`` and the current ``r`` (MODEL.md
    §8); ``b_hat`` comes from the trajectory, ``B`` is constant.
    """
    if not 0 <= epoch < scenario.epochs:
        raise ValueError(f"epoch {epoch} outside 0..{scenario.epochs - 1}")
    r_rows = []
    for i in range(scenario.n):
        table = scenario.ladder_tables[i]
        row = (
            scenario.start_seconds[i]
            if scenario.freeze_ladder
            else (scenario.start_seconds[i] + epoch) % len(table)
        )
        r_rows.append(table[row] * scenario.scales[i])
    L = [row.shape[0] - 1 for row in r_rows]
    w = weights_from_preferences(scenario.p, r_rows, L)
    return Instance(
        n=scenario.n,
        L=L,
        r=r_rows,
        w=w,
        b_hat=scenario.b_hat_traj[epoch],
        B=scenario.B,
        phi_k=scenario.phi_k,
        alpha=scenario.alpha,
        T_stab=scenario.T_stab,
        history=history,
    )
