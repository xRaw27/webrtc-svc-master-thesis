"""Hypothesis strategies producing valid instances.

Test-local helpers, deliberately independent of the M2 ``generator`` module:
they aim for structural coverage of the validation rules, not for realism.
"""

import numpy as np
from hypothesis import strategies as st

from sfu_alloc.instance import History, Instance


@st.composite
def instances(draw, max_n: int = 5, max_L: int = 3) -> Instance:
    """Draw a valid ``Instance`` (optionally with history) per MODEL.md §3."""
    n = draw(st.integers(2, max_n))
    L = [draw(st.integers(1, max_L)) for _ in range(n)]

    first_step = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False)
    step = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)
    r = []
    for i in range(n):
        steps = [draw(first_step)] + [draw(step) for _ in range(L[i] - 1)]
        r.append([0.0, *np.cumsum(steps)])

    weight = st.floats(min_value=0.05, max_value=10.0, allow_nan=False)
    w = np.zeros((n, n))
    for j in range(n):
        senders = [i for i in range(n) if i != j]
        raw = np.array([draw(weight) for _ in senders])
        w[senders, j] = raw / raw.sum()

    bandwidth = st.floats(min_value=0.0, max_value=5000.0, allow_nan=False)
    T_stab = draw(st.integers(2, 6))

    history = None
    if draw(st.booleans()):
        lam = [[draw(st.integers(0, L[i])) for _ in range(n)] for i in range(n)]
        tau = [[draw(st.integers(1, T_stab + 2)) for _ in range(n)] for i in range(n)]
        history = History(lambda_prev=np.array(lam), tau=np.array(tau))

    return Instance(
        n=n,
        L=L,
        r=r,
        w=w,
        b_hat=[draw(bandwidth) for _ in range(n)],
        B=draw(st.floats(min_value=0.0, max_value=20000.0, allow_nan=False)),
        phi_k=draw(st.sampled_from([1.0, 0.5, 0.25])),
        alpha=draw(st.sampled_from([0.0, 0.1, 0.3, 1.0])),
        T_stab=T_stab,
        history=history,
    )


@st.composite
def instances_with_preferences(draw) -> tuple[Instance, np.ndarray]:
    """Draw a valid instance plus a positive per-pair preference matrix."""
    inst = draw(instances())
    pref = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)
    p = np.ones((inst.n, inst.n))
    for i in range(inst.n):
        for j in range(inst.n):
            if i != j:
                p[i, j] = draw(pref)
    return inst, p


@st.composite
def instances_with_allocation(draw) -> tuple[Instance, np.ndarray]:
    """Draw a valid instance plus an arbitrary in-range allocation matrix."""
    inst = draw(instances())
    level = np.zeros((inst.n, inst.n), dtype=np.int64)
    for i in range(inst.n):
        for j in range(inst.n):
            if i != j:
                level[i, j] = draw(st.integers(0, int(inst.L[i])))
    return inst, level
