"""Validation and precomputation tests (MODEL.md §3-§4, Example B)."""

import numpy as np
import pytest

from sfu_alloc.instance import (
    History,
    Instance,
    precompute,
    weights_from_preferences,
)


def make_instance(**overrides) -> Instance:
    """Valid n=3 baseline: shared ladder (0, 150, 250, 800), equal weights."""
    kwargs = dict(
        n=3,
        L=[3, 3, 3],
        r=[[0.0, 150.0, 250.0, 800.0]] * 3,
        w=[[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        b_hat=[1000.0, 1000.0, 1000.0],
        B=3000.0,
        phi_k=0.5,
        alpha=0.2,
        T_stab=4,
        history=None,
    )
    kwargs.update(overrides)
    return Instance(**kwargs)


class TestValidation:
    def test_valid_instance_constructs(self):
        inst = make_instance()
        assert inst.n == 3
        assert inst.r.shape == (3, 4)
        assert inst.history is None

    def test_arrays_are_read_only(self):
        inst = make_instance()
        with pytest.raises(ValueError, match="read-only"):
            inst.r[0, 1] = 5.0

    def test_ragged_ladders_are_nan_padded(self):
        inst = make_instance(
            L=[3, 1, 2],
            r=[[0.0, 150.0, 250.0, 800.0], [0.0, 300.0], [0.0, 100.0, 400.0]],
        )
        assert inst.r.shape == (3, 4)
        assert np.isnan(inst.r[1, 2:]).all()
        assert np.isnan(inst.r[2, 3])
        assert inst.r[1, 1] == 300.0

    def test_rejects_n_below_2(self):
        with pytest.raises(ValueError, match="n must be >= 2"):
            Instance(
                n=1,
                L=[1],
                r=[[0.0, 100.0]],
                w=[[0.0]],
                b_hat=[100.0],
                B=100.0,
                phi_k=1.0,
                alpha=0.0,
                T_stab=2,
                history=None,
            )

    def test_rejects_non_monotone_ladder(self):
        with pytest.raises(ValueError, match="non-decreasing"):
            make_instance(r=[[0.0, 250.0, 150.0, 800.0]] * 3)

    def test_rejects_nonzero_level_zero(self):
        with pytest.raises(ValueError, match=r"r\[0\]\[0\]"):
            make_instance(r=[[10.0, 150.0, 250.0, 800.0]] * 3)

    def test_rejects_all_zero_ladder(self):
        with pytest.raises(ValueError, match="> 0"):
            make_instance(L=[1, 1, 1], r=[[0.0, 0.0]] * 3)

    def test_rejects_wrong_row_length(self):
        with pytest.raises(ValueError, match=r"r\[1\]"):
            make_instance(
                r=[
                    [0.0, 150.0, 250.0, 800.0],
                    [0.0, 150.0],
                    [0.0, 150.0, 250.0, 800.0],
                ]
            )

    def test_rejects_weights_not_summing_to_one(self):
        w = [[0.0, 0.5, 0.4], [0.5, 0.0, 0.5], [0.4, 0.5, 0.0]]
        with pytest.raises(ValueError, match="sum to 1"):
            make_instance(w=w)

    def test_rejects_nonpositive_weight(self):
        w = [[0.0, 0.5, 0.5], [1.0, 0.0, 0.5], [0.0, 0.5, 0.0]]
        with pytest.raises(ValueError, match="> 0"):
            make_instance(w=w)

    @pytest.mark.parametrize("phi_k", [0.0, 1.5, -0.2])
    def test_rejects_bad_phi_k(self, phi_k):
        with pytest.raises(ValueError, match="phi_k"):
            make_instance(phi_k=phi_k)

    @pytest.mark.parametrize("alpha", [-0.1, 1.0001])
    def test_rejects_bad_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            make_instance(alpha=alpha)

    @pytest.mark.parametrize("T_stab", [1, 2.5])
    def test_rejects_bad_T_stab(self, T_stab):
        with pytest.raises(ValueError, match="T_stab"):
            make_instance(T_stab=T_stab)

    def test_rejects_negative_b_hat_and_B(self):
        with pytest.raises(ValueError, match="b_hat"):
            make_instance(b_hat=[1000.0, -1.0, 1000.0])
        with pytest.raises(ValueError, match="B"):
            make_instance(B=-5.0)

    def test_rejects_wrong_shapes(self):
        with pytest.raises(ValueError, match="L"):
            make_instance(L=[3, 3])
        with pytest.raises(ValueError, match="w"):
            make_instance(w=np.full((2, 2), 0.5))
        with pytest.raises(ValueError, match="b_hat"):
            make_instance(b_hat=[1000.0, 1000.0])

    def test_rejects_tau_below_one(self):
        with pytest.raises(ValueError, match="tau"):
            History(
                lambda_prev=np.zeros((3, 3), dtype=np.int64),
                tau=np.zeros((3, 3), dtype=np.int64),
            )

    def test_rejects_non_integer_history(self):
        with pytest.raises(ValueError, match="integer"):
            History(lambda_prev=np.zeros((3, 3)), tau=np.ones((3, 3)))

    def test_rejects_lambda_prev_above_L(self):
        hist = History(
            lambda_prev=np.full((3, 3), 4, dtype=np.int64),
            tau=np.ones((3, 3), dtype=np.int64),
        )
        with pytest.raises(ValueError, match="lambda_prev"):
            make_instance(history=hist)

    def test_rejects_history_shape_mismatch(self):
        hist = History(
            lambda_prev=np.zeros((2, 2), dtype=np.int64),
            tau=np.ones((2, 2), dtype=np.int64),
        )
        with pytest.raises(ValueError, match="history"):
            make_instance(history=hist)


class TestPrecompute:
    def test_q_invariants(self):
        inst = make_instance(
            L=[3, 1, 2],
            r=[[0.0, 150.0, 250.0, 800.0], [0.0, 300.0], [0.0, 100.0, 400.0]],
        )
        pre = precompute(inst)
        for i in range(3):
            top = int(inst.L[i])
            q_i = pre.q[i, : top + 1]
            assert q_i[0] == pytest.approx(0.0)
            assert q_i[top] == pytest.approx(1.0)
            assert np.all(np.diff(q_i) >= 0)
            assert np.isnan(pre.q[i, top + 1 :]).all()

    def test_q_values_on_example_ladder(self):
        pre = precompute(make_instance())
        assert pre.q[0, 1] == pytest.approx(np.sqrt(150 / 800))
        assert pre.q[0, 2] == pytest.approx(np.sqrt(250 / 800))

    def test_no_history_means_zero_penalty(self):
        pre = precompute(make_instance())
        off = ~np.eye(3, dtype=bool)
        assert np.all(pre.c[off] == 0.0)
        assert np.isnan(pre.c[np.arange(3), np.arange(3), :]).all()

    @pytest.mark.parametrize(
        ("tau", "expected"),
        [(1, 0.2), (2, 0.1333333), (4, 0.0), (6, 0.0)],
    )
    def test_example_b_switch_penalty(self, tau, expected):
        """MODEL.md §9 Example B: alpha = 0.2, T_stab = 4."""
        hist = History(
            lambda_prev=np.zeros((3, 3), dtype=np.int64),
            tau=np.full((3, 3), tau, dtype=np.int64),
        )
        pre = precompute(make_instance(alpha=0.2, T_stab=4, history=hist))
        assert pre.c[0, 1, 1] == pytest.approx(expected, abs=1e-6)
        assert pre.c[0, 1, 0] == 0.0  # keeping lambda_prev is always free

    def test_q_corr_negative_when_dropping_previous_level(self):
        hist = History(
            lambda_prev=np.ones((3, 3), dtype=np.int64),
            tau=np.ones((3, 3), dtype=np.int64),
        )
        pre = precompute(make_instance(alpha=0.2, T_stab=4, history=hist))
        assert pre.q_corr[0, 1, 0] == pytest.approx(-0.2)


class TestWeightsFromPreferences:
    """MODEL.md §9 Example D: canonical weights, exact fractions."""

    # Receiver 2 (n=3) gets streams 0 and 1 with r_max = (500, 1000);
    # the receiver's own ladder (700) must not influence its weights.
    TWO_R = ((0.0, 500.0), (0.0, 1000.0), (0.0, 700.0))
    # Receiver 3 (n=4) gets streams 0..2 with r_max = (400, 800, 1600).
    THREE_R = ((0.0, 400.0), (0.0, 800.0), (0.0, 1600.0), (0.0, 500.0))

    def test_two_streams_default_preferences(self):
        w = weights_from_preferences(np.ones((3, 3)), self.TWO_R, [1, 1, 1])
        assert w[0, 2] == 1 / 3
        assert w[1, 2] == 2 / 3

    def test_two_streams_with_preferences(self):
        p = np.ones((3, 3))
        p[0, 2] = 2.0  # receiver 2 prefers stream 0 twice as much
        w = weights_from_preferences(p, self.TWO_R, [1, 1, 1])
        assert w[0, 2] == 1 / 2
        assert w[1, 2] == 1 / 2

    def test_three_streams_default_preferences(self):
        w = weights_from_preferences(np.ones((4, 4)), self.THREE_R, [1, 1, 1, 1])
        assert w[0, 3] == 1 / 7
        assert w[1, 3] == 2 / 7
        assert w[2, 3] == 4 / 7

    def test_three_streams_with_preferences(self):
        p = np.ones((4, 4))
        p[0, 3] = 2.0
        w = weights_from_preferences(p, self.THREE_R, [1, 1, 1, 1])
        assert w[0, 3] == 0.25
        assert w[1, 3] == 0.25
        assert w[2, 3] == 0.5

    def test_instance_accepts_canonical_weights(self):
        """Shared ladder + default p gives equal weights (Example A setup)."""
        w = weights_from_preferences(
            np.ones((3, 3)), [[0.0, 150.0, 250.0, 800.0]] * 3, [3, 3, 3]
        )
        inst = make_instance(w=w)
        assert inst.w[0, 1] == 1 / 2  # two senders per receiver, equal tops

    def test_diagonal_preferences_ignored(self):
        p = np.ones((3, 3))
        np.fill_diagonal(p, np.nan)
        w = weights_from_preferences(p, self.TWO_R, [1, 1, 1])
        assert w[0, 2] == 1 / 3
        assert np.all(w[np.eye(3, dtype=bool)] == 0.0)

    def test_rejects_nonpositive_preference(self):
        p = np.ones((3, 3))
        p[0, 1] = 0.0
        with pytest.raises(ValueError, match=r"p\[i\]\[j\]"):
            weights_from_preferences(p, self.TWO_R, [1, 1, 1])

    def test_rejects_wrong_p_shape(self):
        with pytest.raises(ValueError, match="p must have shape"):
            weights_from_preferences(np.ones((2, 2)), self.TWO_R, [1, 1, 1])
