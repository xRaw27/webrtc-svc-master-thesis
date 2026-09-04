"""Hypothesis properties over randomly generated valid instances."""

import numpy as np
import pytest
from hypothesis import given

from sfu_alloc.constants import TOL
from sfu_alloc.evaluator import check_feasibility, q_corr_vector, sorted_q
from sfu_alloc.instance import weights_from_preferences
from strategies import instances, instances_with_allocation, instances_with_preferences


@given(inst=instances())
def test_zero_allocation_always_feasible(inst):
    """MODEL.md §5: the all-zero allocation is feasible for every instance."""
    level = np.zeros((inst.n, inst.n), dtype=np.int64)
    rep = check_feasibility(inst, level)
    assert rep.ok
    assert rep.violated is None
    np.testing.assert_allclose(rep.downlink_slack, inst.b_hat)
    assert rep.uplink_slack == pytest.approx(inst.B)


@given(pair=instances_with_preferences())
def test_canonical_weights_positive_and_sum_to_one(pair):
    """MODEL.md §9 Example D property: canonical w is positive, sums to 1."""
    inst, p = pair
    w = weights_from_preferences(p, inst.r, inst.L)
    off = ~np.eye(inst.n, dtype=bool)
    assert np.all(w[off] > 0.0)
    np.testing.assert_allclose(np.where(off, w, 0.0).sum(axis=0), 1.0, atol=1e-9)
    assert np.all(w[np.eye(inst.n, dtype=bool)] == 0.0)


@given(pair=instances_with_allocation())
def test_q_corr_range_and_sorting(pair):
    """MODEL.md §6: Q_corr[j] in [-alpha, 1]; sorted_q sorts Q_corr ascending."""
    inst, level = pair
    qv = q_corr_vector(inst, level)
    assert np.all(qv >= -inst.alpha - TOL)
    assert np.all(qv <= 1.0 + TOL)
    sq = sorted_q(inst, level)
    assert np.all(np.diff(sq) >= 0)
    np.testing.assert_allclose(np.sort(qv), sq)
