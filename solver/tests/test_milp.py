"""M4-M5 tests: max-min and lexicographic max-min MILP vs brute force.

PLAN.md M4 acceptance: on the M3 instance families the first coordinate of
``solve_maxmin``'s sorted vector equals the brute-force first coordinate
within TOL; full vectors may legitimately differ. PLAN.md M5 acceptance:
``solve_lex_maxmin`` is lex-equal (TOL) to ``solve_pruned`` on >= 300 tiny
instances, an n=10/L=3 instance solves in seconds, and the diagnostics carry
the full theta_star sequence with per-stage times.
"""

import numpy as np
import pytest
from hypothesis import given

from sfu_alloc.algorithms.brute_force import solve_pruned
from sfu_alloc.algorithms.milp import solve_lex_maxmin, solve_maxmin
from sfu_alloc.constants import TOL
from sfu_alloc.evaluator import check_feasibility, lex_compare
from sfu_alloc.generator import random_instance
from sfu_alloc.instance import History, Instance
from strategies import instances


def assert_first_coordinate_matches(inst):
    res_milp = solve_maxmin(inst)
    res_bf = solve_pruned(inst)
    assert check_feasibility(inst, res_milp.allocation).ok
    assert abs(res_milp.sorted_q[0] - res_bf.sorted_q[0]) <= TOL


@given(inst=instances(max_n=3, max_L=2))
def test_maxmin_first_coordinate_random(inst):
    """Acceptance on the hypothesis family (n <= 3, L_i <= 2)."""
    assert_first_coordinate_matches(inst)


def test_maxmin_first_coordinate_fixed_seed_batch():
    """Acceptance on the M3 fixed-seed families (n=3 mixed L, n=4 L=1)."""
    for seed in range(40):
        assert_first_coordinate_matches(random_instance(seed, n=3, L_choices=(1, 2)))
    for seed in range(10):
        assert_first_coordinate_matches(
            random_instance(2_000 + seed, n=4, L_choices=(1,))
        )


def test_known_optimum_n2():
    """Symmetric n=2 instance where the max-min optimum is hand-checkable."""
    inst = Instance(
        n=2,
        L=[3, 3],
        r=[[0.0, 150.0, 250.0, 800.0]] * 2,
        w=[[0.0, 1.0], [1.0, 0.0]],
        b_hat=[200.0, 200.0],
        B=1000.0,
        phi_k=0.5,
        alpha=0.0,
        T_stab=2,
        history=None,
    )
    res = solve_maxmin(inst)
    expected = np.sqrt(150.0 / 800.0)
    np.testing.assert_allclose(res.sorted_q, [expected, expected], atol=1e-6)
    assert res.stats["z_reported"] == pytest.approx(expected, abs=1e-5)
    assert "optimal" in res.stats["termination"]


def test_scales_beyond_brute_force():
    """n=8 (far beyond naive enumeration) solves to optimality in one go."""
    inst = random_instance(99, n=8)
    res = solve_maxmin(inst, time_limit=60.0)
    assert check_feasibility(inst, res.allocation).ok
    assert "optimal" in res.stats["termination"]
    # at a max-min optimum z equals the smallest re-evaluated Q_corr
    assert res.stats["z_reported"] == pytest.approx(res.sorted_q[0], abs=1e-5)


def test_history_penalties_enter_objective():
    """With fresh history and alpha > 0 the optimum keeps previous levels.

    Both pairs were just switched (tau = 1) to level 1; b_hat still allows
    level 2, but switching again costs alpha = 0.3 while the quality gain
    q(250/800) - q(150/800) is smaller, so max-min keeps level 1.
    """
    ones = np.ones((2, 2), dtype=np.int64)
    inst = Instance(
        n=2,
        L=[3, 3],
        r=[[0.0, 150.0, 250.0, 800.0]] * 2,
        w=[[0.0, 1.0], [1.0, 0.0]],
        b_hat=[250.0, 250.0],
        B=1000.0,
        phi_k=0.5,
        alpha=0.3,
        T_stab=4,
        history=History(lambda_prev=ones, tau=ones),
    )
    res = solve_maxmin(inst)
    assert res.allocation[0, 1] == 1
    assert res.allocation[1, 0] == 1
    expected = np.sqrt(150.0 / 800.0)  # keeping lambda_prev is penalty-free
    np.testing.assert_allclose(res.sorted_q, [expected, expected], atol=1e-6)


# --- M5: lexicographic max-min ------------------------------------------


def assert_lex_equal_to_pruned(inst):
    res = solve_lex_maxmin(inst)
    ref = solve_pruned(inst)
    assert check_feasibility(inst, res.allocation).ok
    assert lex_compare(res.sorted_q, ref.sorted_q) == 0


@given(inst=instances(max_n=3, max_L=2))
def test_lex_maxmin_matches_brute_force_random(inst):
    """M5 acceptance on the hypothesis family (n <= 3, L_i <= 2)."""
    assert_lex_equal_to_pruned(inst)


def test_lex_maxmin_matches_brute_force_fixed_seed_batch():
    """M5 acceptance: >= 300 tiny instances, with required coverage.

    Verifies that the batch really contains instances with history, unequal
    weights, alpha = 0 and phi_k = 1 (PLAN.md M5 acceptance list).
    """
    insts = [random_instance(seed, n=3, L_choices=(1, 2)) for seed in range(250)]
    insts += [random_instance(3_000 + seed, n=4, L_choices=(1,)) for seed in range(50)]
    covered = {"history", "no_history", "unequal_w", "alpha0", "phik1"}
    seen = set()
    for inst in insts:
        assert_lex_equal_to_pruned(inst)
        seen.add("history" if inst.history is not None else "no_history")
        off = ~np.eye(inst.n, dtype=bool)
        if not np.allclose(inst.w[off], 1.0 / (inst.n - 1)):
            seen.add("unequal_w")
        if inst.alpha == 0.0:
            seen.add("alpha0")
        if inst.phi_k == 1.0:
            seen.add("phik1")
    assert seen == covered, f"missing coverage: {covered - seen}"


def test_incremental_matches_rebuild():
    """rebuild=False must produce the same optima as the rebuild path."""
    for seed in range(10):
        inst = random_instance(seed, n=3)
        a = solve_lex_maxmin(inst, rebuild=True)
        b = solve_lex_maxmin(inst, rebuild=False)
        assert lex_compare(a.sorted_q, b.sorted_q) == 0
        np.testing.assert_allclose(
            a.stats["theta_star"], b.stats["theta_star"], atol=1e-6
        )


def test_stage_one_equals_maxmin():
    """theta_star[0] must equal the plain M4 max-min optimum."""
    for seed in range(10):
        inst = random_instance(seed, n=4)
        lex = solve_lex_maxmin(inst)
        ref = solve_maxmin(inst)
        assert lex.stats["theta_star"][0] == pytest.approx(
            ref.stats["z_reported"], abs=1e-6
        )
        assert abs(lex.sorted_q[0] - ref.sorted_q[0]) <= TOL


def test_n10_L3_solves_in_seconds():
    """M5 acceptance: n=10, L=3 to optimality in seconds; full diagnostics."""
    inst = random_instance(42, n=10, L_choices=(3,))
    res = solve_lex_maxmin(inst)
    stats = res.stats
    assert len(stats["theta_star"]) == 10
    assert len(stats["stage_times_s"]) == 10
    assert all("optimal" in t for t in stats["stage_terminations"])
    assert np.all(np.diff(res.sorted_q) >= -TOL)
    assert stats["wall_time_s"] < 30.0
    print(
        f"\nn=10 lex: {stats['wall_time_s']:.2f}s total, "
        f"stages {np.round(stats['stage_times_s'], 3).tolist()}"
    )
