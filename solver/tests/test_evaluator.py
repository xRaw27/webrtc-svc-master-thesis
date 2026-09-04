"""Evaluator tests: MODEL.md §9 Examples A and C, feasibility, lex_compare."""

import numpy as np
import pytest

from sfu_alloc.evaluator import (
    check_feasibility,
    lex_compare,
    q_corr_vector,
    sorted_q,
)
from sfu_alloc.instance import History, Instance, precompute

RECV = 3
"""The observed receiver in Example A (thesis Fig. 3.3, "receiver C")."""


def example_a_instance(**overrides) -> Instance:
    """MODEL.md §9 Example A embedded as n=4: 3 senders + 1 observed receiver.

    The observed receiver has b_hat = 1000 kbps; everyone else has effectively
    unlimited downlink so only the observed receiver's constraint matters.
    """
    n = 4
    kwargs = dict(
        n=n,
        L=[3] * n,
        r=[[0.0, 150.0, 250.0, 800.0]] * n,
        w=(np.ones((n, n)) - np.eye(n)) / 3,
        b_hat=[1e6, 1e6, 1e6, 1000.0],
        B=1e6,
        phi_k=0.5,
        alpha=0.2,
        T_stab=4,
        history=None,
    )
    kwargs.update(overrides)
    return Instance(**kwargs)


def alloc_to_receiver(inst: Instance, levels: tuple[int, ...]) -> np.ndarray:
    """All-zero allocation except the given levels sent to receiver RECV."""
    lev = np.zeros((inst.n, inst.n), dtype=np.int64)
    senders = [i for i in range(inst.n) if i != RECV]
    for i, level in zip(senders, levels, strict=True):
        lev[i, RECV] = level
    return lev


class TestExampleA:
    def test_levels_2_2_2(self):
        inst = example_a_instance()
        lev = alloc_to_receiver(inst, (2, 2, 2))
        rep = check_feasibility(inst, lev)
        assert rep.ok and rep.violated is None
        assert rep.downlink_slack[RECV] == pytest.approx(1000.0 - 750.0)
        assert q_corr_vector(inst, lev)[RECV] == pytest.approx(0.5590170, abs=1e-6)

    def test_levels_3_1_0(self):
        inst = example_a_instance()
        lev = alloc_to_receiver(inst, (3, 1, 0))
        rep = check_feasibility(inst, lev)
        assert rep.ok
        assert rep.downlink_slack[RECV] == pytest.approx(1000.0 - 950.0)
        assert q_corr_vector(inst, lev)[RECV] == pytest.approx(0.4776709, abs=1e-6)

    def test_sorted_q_is_ascending(self):
        inst = example_a_instance()
        lev = alloc_to_receiver(inst, (2, 2, 2))
        np.testing.assert_allclose(
            sorted_q(inst, lev), [0.0, 0.0, 0.0, 0.5590170], atol=1e-6
        )

    def test_downlink_violation(self):
        inst = example_a_instance()
        lev = alloc_to_receiver(inst, (3, 3, 3))  # 2400 kbps > b_hat = 1000
        rep = check_feasibility(inst, lev)
        assert not rep.ok
        assert rep.violated == f"downlink[{RECV}]"
        assert rep.downlink_slack[RECV] == pytest.approx(-1400.0)

    def test_uplink_violation(self):
        inst = example_a_instance(B=500.0)
        lev = alloc_to_receiver(inst, (2, 2, 2))  # 750 kbps total > B = 500
        rep = check_feasibility(inst, lev)
        assert not rep.ok
        assert rep.violated == "uplink"
        assert rep.uplink_slack == pytest.approx(-250.0)

    def test_downlink_reported_before_uplink(self):
        inst = example_a_instance(B=500.0)
        rep = check_feasibility(inst, alloc_to_receiver(inst, (3, 3, 3)))
        assert rep.violated == f"downlink[{RECV}]"


class TestExampleC:
    def test_zero_allocation_feasible_with_negative_quality(self):
        """MODEL.md §9 Example C: level ≡ 0 feasible, Q_corr can be < 0."""
        hist = History(
            lambda_prev=np.array([[0, 1], [1, 0]]),
            tau=np.ones((2, 2), dtype=np.int64),
        )
        inst = Instance(
            n=2,
            L=[1, 1],
            r=[[0.0, 300.0]] * 2,
            w=[[0.0, 1.0], [1.0, 0.0]],
            b_hat=[0.0, 0.0],
            B=0.0,
            phi_k=1.0,
            alpha=0.3,
            T_stab=4,
            history=hist,
        )
        lev = np.zeros((2, 2), dtype=np.int64)
        rep = check_feasibility(inst, lev)
        assert rep.ok  # all-zero is feasible even with zero budgets
        np.testing.assert_allclose(q_corr_vector(inst, lev), [-0.3, -0.3])


class TestLevelValidation:
    def test_rejects_level_above_L(self):
        inst = example_a_instance()
        lev = np.zeros((4, 4), dtype=np.int64)
        lev[0, 1] = 4  # L[0] = 3
        with pytest.raises(ValueError, match=r"level\[0\]\[1\]"):
            check_feasibility(inst, lev)

    def test_rejects_negative_level(self):
        inst = example_a_instance()
        lev = np.zeros((4, 4), dtype=np.int64)
        lev[2, 0] = -1
        with pytest.raises(ValueError, match=r"level\[2\]\[0\]"):
            q_corr_vector(inst, lev)

    def test_rejects_float_dtype_and_wrong_shape(self):
        inst = example_a_instance()
        with pytest.raises(ValueError, match="integer"):
            check_feasibility(inst, np.zeros((4, 4)))
        with pytest.raises(ValueError, match="shape"):
            check_feasibility(inst, np.zeros((3, 3), dtype=np.int64))

    def test_diagonal_is_ignored(self):
        inst = example_a_instance()
        lev = alloc_to_receiver(inst, (2, 2, 2))
        noisy = lev.copy()
        np.fill_diagonal(noisy, 99)  # out of range but on the unused diagonal
        np.testing.assert_allclose(q_corr_vector(inst, noisy), q_corr_vector(inst, lev))
        assert check_feasibility(inst, noisy).ok


class TestPrecomputedParameter:
    def test_pre_gives_identical_results(self):
        inst = example_a_instance()
        pre = precompute(inst)
        lev = alloc_to_receiver(inst, (3, 1, 0))
        np.testing.assert_array_equal(
            q_corr_vector(inst, lev, pre=pre), q_corr_vector(inst, lev)
        )
        np.testing.assert_array_equal(sorted_q(inst, lev, pre=pre), sorted_q(inst, lev))

    def test_pre_shape_mismatch_is_rejected(self):
        inst = example_a_instance()
        other = Instance(
            n=2,
            L=[1, 1],
            r=[[0.0, 300.0]] * 2,
            w=[[0.0, 1.0], [1.0, 0.0]],
            b_hat=[500.0, 500.0],
            B=1000.0,
            phi_k=1.0,
            alpha=0.0,
            T_stab=2,
            history=None,
        )
        with pytest.raises(ValueError, match="pre does not match"):
            q_corr_vector(inst, np.zeros((4, 4), dtype=np.int64), pre=precompute(other))


class TestLexCompare:
    def test_identical_vectors(self):
        assert lex_compare([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0

    def test_equal_within_tolerance(self):
        assert lex_compare([0.5, 0.7], [0.5 + 5e-7, 0.7 - 5e-7]) == 0

    def test_first_differing_coordinate_decides(self):
        assert lex_compare([0.2, 0.0], [0.1, 100.0]) == 1
        assert lex_compare([0.1, 100.0], [0.2, 0.0]) == -1

    def test_tie_within_tol_then_decide_later(self):
        assert lex_compare([0.5, 0.3], [0.5 + 1e-8, 0.2]) == 1
        assert lex_compare([0.5, 0.2], [0.5 + 1e-8, 0.3]) == -1

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            lex_compare([0.1, 0.2], [0.1])
