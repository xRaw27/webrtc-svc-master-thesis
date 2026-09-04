"""M7 tests: feasibility everywhere, gap report vs exact optimum, runtime."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.algorithms.milp import solve_lex_maxmin
from sfu_alloc.constants import TOL
from sfu_alloc.evaluator import check_feasibility, lex_compare
from sfu_alloc.generator import random_instance
from sfu_alloc.instance import Instance
from strategies import instances


@given(inst=instances(), delta=st.sampled_from([25.0, 100.0, 250.0]))
def test_always_feasible(inst, delta):
    """M7 acceptance: feasibility on 100% of random instances."""
    res = solve_heuristic(inst, delta_kbps=delta)
    rep = check_feasibility(inst, res.allocation)
    assert rep.ok
    assert res.stats["spent_cells"] <= res.stats["budget_cells"]


@pytest.mark.parametrize("n", [5, 10, 20, 50])
def test_feasible_on_larger_sizes(n):
    """Feasibility also at sizes far beyond the exact solvers' reach."""
    for seed in range(3):
        inst = random_instance(seed, n=n)
        res = solve_heuristic(inst)
        assert check_feasibility(inst, res.allocation).ok


def test_matches_exact_when_quantization_is_lossless():
    """Bitrates that are exact multiples of delta lose nothing to rounding."""
    inst = Instance(
        n=2,
        L=[3, 3],
        r=[[0.0, 100.0, 200.0, 800.0]] * 2,
        w=[[0.0, 1.0], [1.0, 0.0]],
        b_hat=[200.0, 200.0],
        B=1000.0,
        phi_k=0.5,
        alpha=0.0,
        T_stab=2,
        history=None,
    )
    res = solve_heuristic(inst, delta_kbps=100.0)
    expected = np.sqrt(200.0 / 800.0)  # both pairs at level 2 (200 kbps)
    np.testing.assert_allclose(res.sorted_q, [expected, expected], atol=1e-9)


def test_report_gap_vs_exact_optimum():
    """M7 acceptance: comparison vs the M5 optimum — recorded and printed.

    No hard quality threshold (per PLAN.md); the only hard assertions are
    validity: feasibility, and that the heuristic never beats the exact
    optimum's first coordinate (which would indicate an evaluator/solver
    bug, not a heuristic property).
    """
    insts = [random_instance(seed, n=3, L_choices=(1, 2)) for seed in range(80)]
    insts += [random_instance(5_000 + seed, n=4, L_choices=(1,)) for seed in range(20)]
    matches = 0
    first_diffs = []
    mean_diffs = []
    for inst in insts:
        exact = solve_lex_maxmin(inst)
        heur = solve_heuristic(inst)
        assert check_feasibility(inst, heur.allocation).ok
        assert heur.sorted_q[0] <= exact.sorted_q[0] + TOL
        if lex_compare(exact.sorted_q, heur.sorted_q) == 0:
            matches += 1
        first_diffs.append(float(exact.sorted_q[0] - heur.sorted_q[0]))
        mean_diffs.append(float(np.mean(exact.sorted_q - heur.sorted_q)))
    first = np.array(first_diffs)
    print(f"\nheuristic (delta=100) vs exact optimum, {len(insts)} tiny instances:")
    print(f"  exact sorted-vector match rate: {matches}/{len(insts)}")
    print(
        f"  first-coordinate diff: min={first.min():.4f} "
        f"median={np.median(first):.4f} mean={first.mean():.4f} "
        f"max={first.max():.4f}"
    )
    print(f"  mean Q_corr diff: {np.mean(mean_diffs):.4f}")


def test_runtime_n100_L3_under_one_second():
    """M7 acceptance: n=100, L=3, default delta in < 1 s single-threaded."""
    inst = random_instance(7, n=100, L_choices=(3,))
    res = solve_heuristic(inst)
    assert check_feasibility(inst, res.allocation).ok
    assert res.stats["wall_time_s"] < 1.0
    print(
        f"\nn=100 heuristic: {res.stats['wall_time_s'] * 1000:.0f} ms total "
        f"(dp {res.stats['dp_time_s'] * 1000:.0f} ms, "
        f"master {res.stats['master_time_s'] * 1000:.0f} ms)"
    )


def test_rejects_bad_arguments():
    inst = random_instance(0, n=3)
    with pytest.raises(ValueError, match="delta_kbps"):
        solve_heuristic(inst, delta_kbps=0.0)
    with pytest.raises(NotImplementedError, match="exact"):
        solve_heuristic(inst, master="exact")
    with pytest.raises(ValueError, match="unknown master"):
        solve_heuristic(inst, master="bogus")
