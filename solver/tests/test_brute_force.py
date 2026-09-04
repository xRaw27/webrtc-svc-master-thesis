"""M3 tests: naive vs pruned agreement, guards, known optimum, timing."""

import numpy as np
import pytest
from hypothesis import given

from sfu_alloc.algorithms.brute_force import solve_naive, solve_pruned
from sfu_alloc.evaluator import check_feasibility, lex_compare
from sfu_alloc.generator import random_instance
from sfu_alloc.instance import Instance
from strategies import instances


def assert_solvers_agree(inst):
    res_naive = solve_naive(inst)
    res_pruned = solve_pruned(inst)
    assert lex_compare(res_naive.sorted_q, res_pruned.sorted_q) == 0
    assert check_feasibility(inst, res_naive.allocation).ok
    assert check_feasibility(inst, res_pruned.allocation).ok


@given(inst=instances(max_n=3, max_L=2))
def test_naive_equals_pruned_random(inst):
    """PLAN.md M3: lex-equal sorted vectors on random tiny instances."""
    assert_solvers_agree(inst)


def test_naive_equals_pruned_fixed_seed_batch():
    """PLAN.md M3: >= 200 instances total; here 170 n=3 plus 30 n=4 (L=1)."""
    for seed in range(170):
        assert_solvers_agree(random_instance(seed, n=3, L_choices=(1, 2)))
    for seed in range(30):
        assert_solvers_agree(random_instance(1_000 + seed, n=4, L_choices=(1,)))


def test_known_optimum_n2():
    """Hand-checkable optimum: b_hat forces level 1 (150 kbps) on both pairs."""
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
    expected = np.sqrt(150.0 / 800.0)
    for solve in (solve_naive, solve_pruned):
        res = solve(inst)
        np.testing.assert_allclose(res.sorted_q, [expected, expected], atol=1e-6)
        assert res.allocation[0, 1] == 1
        assert res.allocation[1, 0] == 1


def test_naive_max_states_guard():
    inst = random_instance(0, n=4, L_choices=(3,))  # 4^12 = 16.8M states
    with pytest.raises(ValueError, match="max_states"):
        solve_naive(inst)


def test_timing_sanity_n4_L2():
    """PLAN.md M3: pruned must finish in seconds on n=4, L=2 (both times printed)."""
    inst = Instance(
        n=4,
        L=[2] * 4,
        r=[[0.0, 150.0, 250.0]] * 4,
        w=(np.ones((4, 4)) - np.eye(4)) / 3,
        b_hat=[400.0] * 4,
        B=1200.0,
        phi_k=0.5,
        alpha=0.0,
        T_stab=2,
        history=None,
    )
    res_naive = solve_naive(inst)  # 3^12 = 531,441 states
    res_pruned = solve_pruned(inst)
    print(
        f"\nnaive:  {res_naive.stats['wall_time_s']:8.3f} s "
        f"({res_naive.stats['states_total']:,} states, "
        f"{res_naive.stats['states_feasible']:,} feasible)\n"
        f"pruned: {res_pruned.stats['wall_time_s']:8.3f} s "
        f"({res_pruned.stats['leaves']:,} leaves, "
        f"{res_pruned.stats['pruned_branches']:,} pruned branches)"
    )
    assert lex_compare(res_naive.sorted_q, res_pruned.sorted_q) == 0
    assert res_pruned.stats["wall_time_s"] < 30.0
