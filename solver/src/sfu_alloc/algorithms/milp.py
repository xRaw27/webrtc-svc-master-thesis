"""MILP formulations (M4-M5): plain and lexicographic max-min via Pyomo.

``build_base_model`` constructs the shared base MILP of MODEL.md §5-§6:
binary one-hot level choices, downlink/uplink budget constraints, and the
per-receiver corrected-quality expressions ``Qexpr``. ``solve_maxmin`` (M4)
adds a continuous ``z ∈ [-alpha, 1]`` with ``z <= Qexpr[j]`` and maximizes
``z`` — the first stage only; it remains available on its own.

``solve_lex_maxmin`` (M5) is the exact reference algorithm ("Algorithm 2"
of the thesis): the Ogryczak-Śliwiński cumulative method. Rationale:
``theta_k(x)`` — the sum of the ``k`` smallest ``Q_corr`` values — has the
linear characterization

    theta_k = max over (t, d >= 0) of [ k*t - sum_j d_j ]
              subject to d_j >= t - Q_corr[j] for every receiver j,

and lex-maximizing the ascending-sorted quality vector is equivalent to
lex-maximizing ``(theta_1, ..., theta_n)`` — so each stage is a MILP. Stage
``k`` maximizes ``k*t_k - sum_j d[k, j]`` subject to the base constraints,
the linearization above for every ``k' <= k``, and the locks
``k'*t_k' - sum_j d[k', j] >= theta_star[k'] - eps`` for ``k' < k``. The
``eps`` slack prevents later stages from becoming infeasible due to solver
tolerances; keep it small and configurable.

The solver name is a parameter (default ``"appsi_highs"``) so that e.g.
``"gurobi"`` can be swapped in later. The MIP gap is forced to 0 for
appsi-style solvers: the default relative gap (1e-4) is far above the
repo-wide ``TOL = 1e-6`` used when comparing against brute force. Every
returned allocation is re-evaluated with the evaluator — ``sorted_q`` never
comes from the solver.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pyomo.environ as pyo

from sfu_alloc.algorithms import SolveResult, finalize_result
from sfu_alloc.instance import Instance, precompute

__all__ = ["build_base_model", "extract_level", "solve_lex_maxmin", "solve_maxmin"]


def build_base_model(inst: Instance) -> pyo.ConcreteModel:
    """Build the base MILP (no objective) exactly per PLAN.md M4.

    Components: binary ``x[i, j, l]`` for every pair ``i != j`` and level
    ``l ∈ {0, ..., L[i]}``; ``one_hot`` equality per pair; ``downlink`` and
    ``uplink`` budget constraints; linear expressions
    ``Qexpr[j] = sum_{i != j} w[i][j] * sum_l q_corr[i][j][l] * x[i, j, l]``.
    """
    pre = precompute(inst)
    n = inst.n
    pairs = [(i, j) for j in range(n) for i in range(n) if i != j]
    x_index = [(i, j, lvl) for (i, j) in pairs for lvl in range(int(inst.L[i]) + 1)]

    m = pyo.ConcreteModel(name="sfu_alloc_base")
    m.x = pyo.Var(x_index, domain=pyo.Binary)

    def one_hot_rule(m: pyo.ConcreteModel, i: int, j: int):
        return sum(m.x[i, j, lvl] for lvl in range(int(inst.L[i]) + 1)) == 1

    m.one_hot = pyo.Constraint(pairs, rule=one_hot_rule)

    def downlink_rule(m: pyo.ConcreteModel, j: int):
        load = sum(
            float(inst.r[i, lvl]) * m.x[i, j, lvl]
            for i in range(n)
            if i != j
            for lvl in range(int(inst.L[i]) + 1)
        )
        return load <= float(inst.b_hat[j])

    m.downlink = pyo.Constraint(range(n), rule=downlink_rule)

    m.uplink = pyo.Constraint(
        expr=sum(float(inst.r[i, lvl]) * m.x[i, j, lvl] for (i, j, lvl) in x_index)
        <= float(inst.B)
    )

    def qexpr_rule(m: pyo.ConcreteModel, j: int):
        return sum(
            float(inst.w[i, j] * pre.q_corr[i, j, lvl]) * m.x[i, j, lvl]
            for i in range(n)
            if i != j
            for lvl in range(int(inst.L[i]) + 1)
        )

    m.Qexpr = pyo.Expression(range(n), rule=qexpr_rule)
    return m


def extract_level(inst: Instance, m: pyo.ConcreteModel) -> npt.NDArray[np.int64]:
    """Read the solved one-hot ``x`` back into a ``level`` matrix.

    Takes the argmax per pair and guards against numerical junk: the one-hot
    values must sum to 1 within 1e-4 and the winning entry must exceed 0.9,
    otherwise ``RuntimeError`` is raised.
    """
    n = inst.n
    level = np.zeros((n, n), dtype=np.int64)
    for j in range(n):
        for i in range(n):
            if i == j:
                continue
            vals = np.array(
                [pyo.value(m.x[i, j, lvl]) for lvl in range(int(inst.L[i]) + 1)]
            )
            total = float(vals.sum())
            if abs(total - 1.0) > 1e-4 or float(vals.max()) < 0.9:
                raise RuntimeError(
                    f"one-hot for pair ({i}, {j}) is numerical junk: "
                    f"sum={total:.6f}, max={float(vals.max()):.6f}"
                )
            level[i, j] = int(vals.argmax())
    return level


def _make_solver(solver: str, time_limit: float | None) -> Any:
    """Instantiate and configure the MILP solver.

    APPSI-style interfaces (``config`` attribute) get ``mip_gap = 0`` so
    that optima are exact to solver tolerances rather than the default
    relative gap; classic interfaces are returned as-is (only a generic
    time-limit option is attempted).
    """
    opt = pyo.SolverFactory(solver)
    if hasattr(opt, "config"):
        if time_limit is not None:
            opt.config.time_limit = float(time_limit)
        if hasattr(opt.config, "mip_gap"):
            opt.config.mip_gap = 0.0
    elif time_limit is not None:
        opt.options["time_limit"] = float(time_limit)
    return opt


def _termination_str(results: object) -> str:
    """Termination condition as text, for APPSI and classic result objects."""
    tc = getattr(results, "termination_condition", None)
    if tc is None:
        tc = results.solver.termination_condition  # type: ignore[attr-defined]
    return str(tc)


def solve_maxmin(
    inst: Instance,
    solver: str = "appsi_highs",
    time_limit: float | None = None,
) -> SolveResult:
    """Maximize the minimum corrected quality (PLAN.md M4, stage 1 only).

    Adds ``z ∈ [-alpha, 1]`` with ``z <= Qexpr[j]`` for every receiver and
    solves ``max z``. Only the first coordinate of the sorted quality vector
    is optimized — full lexicographic optimality is M5. ``stats`` records
    build/solve wall times, the termination condition and the solver's
    reported ``z`` (the returned ``sorted_q`` is recomputed by the
    evaluator). Raises ``RuntimeError`` on solver failure, junk one-hots, or
    an infeasible extracted allocation.
    """
    t0 = time.perf_counter()
    m = build_base_model(inst)
    m.z = pyo.Var(domain=pyo.Reals, bounds=(-float(inst.alpha), 1.0))

    def z_rule(m: pyo.ConcreteModel, j: int):
        return m.z <= m.Qexpr[j]

    m.z_le_q = pyo.Constraint(range(inst.n), rule=z_rule)
    m.obj = pyo.Objective(expr=m.z, sense=pyo.maximize)
    build_time = time.perf_counter() - t0

    opt = _make_solver(solver, time_limit)
    results = opt.solve(m)
    solve_time = time.perf_counter() - t0 - build_time

    level = extract_level(inst, m)
    stats: dict[str, Any] = {
        "solver": solver,
        "termination": _termination_str(results),
        "z_reported": float(pyo.value(m.z)),
        "build_time_s": build_time,
        "solve_time_s": solve_time,
    }
    result = finalize_result(inst, level, stats, "solve_maxmin")
    stats["wall_time_s"] = time.perf_counter() - t0
    return result


def _attach_stage(m: pyo.ConcreteModel, inst: Instance, k: int) -> None:
    """Add the stage-k linearization of theta_k (module docstring).

    Creates ``t{k}`` ∈ [-alpha, 1], ``d{k}[j] >= 0`` and the cuts
    ``d{k}[j] >= t{k} - Qexpr[j]`` for every receiver ``j``.
    """
    m.add_component(f"t{k}", pyo.Var(bounds=(-float(inst.alpha), 1.0)))
    m.add_component(f"d{k}", pyo.Var(range(inst.n), bounds=(0.0, None)))

    def cut_rule(m: pyo.ConcreteModel, j: int, k: int = k):
        return getattr(m, f"d{k}")[j] >= getattr(m, f"t{k}") - m.Qexpr[j]

    m.add_component(f"cut{k}", pyo.Constraint(range(inst.n), rule=cut_rule))


def _attach_lock(
    m: pyo.ConcreteModel, k: int, theta: float, eps: float, n: int
) -> None:
    """Lock stage k at its found optimum: k*t_k - sum_j d_k[j] >= theta - eps."""
    t_k = getattr(m, f"t{k}")
    d_k = getattr(m, f"d{k}")
    m.add_component(
        f"lock{k}",
        pyo.Constraint(expr=k * t_k - sum(d_k[j] for j in range(n)) >= theta - eps),
    )


def solve_lex_maxmin(
    inst: Instance,
    solver: str = "appsi_highs",
    eps: float = 1e-5,
    rebuild: bool = True,
) -> SolveResult:
    """Exact lexicographic max-min (PLAN.md M5, Ogryczak-Śliwiński method).

    Solves ``n`` sequential MILP stages; stage ``k`` maximizes ``theta_k``
    (the sum of the ``k`` smallest corrected qualities) with every earlier
    stage locked at its optimum within ``eps``. The default ``eps = 1e-5``
    stays 100x above HiGHS's primal feasibility tolerance (1e-7): with
    ``eps = 1e-6`` stacked locks made HiGHS falsely declare a late stage
    infeasible on a dataset instance (mixed kbps/quality coefficient
    scales), while on 30 control instances 1e-5 and 1e-6 gave
    lex-identical results. ``rebuild=True`` builds a
    fresh model for each stage (correctness-first default);
    ``rebuild=False`` incrementally extends a single model — the same
    formulation, less model-building time. ``stats`` carries the full
    ``theta_star`` sequence, per-stage wall times and terminations. Raises
    ``RuntimeError`` on solver failure, junk one-hots, or an infeasible
    extracted allocation.
    """
    t0 = time.perf_counter()
    n = inst.n
    theta_star: list[float] = []
    stage_times: list[float] = []
    stage_terminations: list[str] = []

    m: pyo.ConcreteModel | None = None
    for k in range(1, n + 1):
        stage_start = time.perf_counter()
        if rebuild or m is None:
            m = build_base_model(inst)
            for kp in range(1, k + 1):
                _attach_stage(m, inst, kp)
            for kp in range(1, k):
                _attach_lock(m, kp, theta_star[kp - 1], eps, n)
        else:
            _attach_stage(m, inst, k)
            _attach_lock(m, k - 1, theta_star[-1], eps, n)
            m.del_component(m.obj)
        t_k = getattr(m, f"t{k}")
        d_k = getattr(m, f"d{k}")
        m.obj = pyo.Objective(
            expr=k * t_k - sum(d_k[j] for j in range(n)), sense=pyo.maximize
        )
        results = _make_solver(solver, None).solve(m)
        theta_star.append(float(pyo.value(m.obj)))
        stage_terminations.append(_termination_str(results))
        stage_times.append(time.perf_counter() - stage_start)

    level = extract_level(inst, m)
    stats: dict[str, Any] = {
        "solver": solver,
        "eps": eps,
        "rebuild": rebuild,
        "theta_star": theta_star,
        "stage_times_s": stage_times,
        "stage_terminations": stage_terminations,
    }
    result = finalize_result(inst, level, stats, "solve_lex_maxmin")
    stats["wall_time_s"] = time.perf_counter() - t0
    return result
