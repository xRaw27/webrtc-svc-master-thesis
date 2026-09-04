"""M8 tests: history update, wrap-around, stability theorem, harness."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfu_alloc.algorithms.heuristic import solve_heuristic
from sfu_alloc.algorithms.milp import solve_lex_maxmin
from sfu_alloc.benchmark.epochs import FIELDS, run_epoch_bench
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.evaluator import lex_compare
from sfu_alloc.instance import weights_from_preferences
from sfu_alloc.scenarios import (
    TRAJECTORY_KINDS,
    random_epoch_scenario,
    scenario_instance,
)
from sfu_alloc.simulator import count_changes, run_epochs, update_history

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
needs_dataset = pytest.mark.skipif(
    not any(DATASET_DIR.glob("*.csv")),
    reason="dataset not built yet (run scripts/build_ladder_dataset.py)",
)


class TestUpdateHistory:
    def test_first_epoch_sets_tau_to_one_everywhere(self):
        level = np.array([[0, 2], [1, 0]])
        hist = update_history(None, level)
        assert np.array_equal(hist.lambda_prev, level)
        assert np.all(hist.tau == 1)

    def test_reset_on_change_increment_otherwise(self):
        level1 = np.array([[0, 2], [1, 0]])
        hist = update_history(None, level1)
        level2 = np.array([[0, 1], [1, 0]])  # pair (0, 1) changes
        hist = update_history(hist, level2)
        assert np.array_equal(hist.lambda_prev, level2)
        assert hist.tau[0, 1] == 1  # reset on change
        assert hist.tau[1, 0] == 2  # incremented
        hist = update_history(hist, level2)  # nothing changes
        assert hist.tau[0, 1] == 2
        assert hist.tau[1, 0] == 3

    def test_count_changes_split(self):
        level1 = np.array([[0, 2, 1], [1, 0, 2], [1, 2, 0]])
        hist = update_history(None, level1)
        for _ in range(4):  # tau grows to 5 everywhere
            hist = update_history(hist, level1)
        assert np.all(hist.tau[~np.eye(3, dtype=bool)] == 5)
        level2 = level1.copy()
        level2[0, 1] = 1  # free change (tau = 5 >= T_stab = 4)
        hist_mixed = update_history(hist, level2)  # tau[0,1]=1, others 6
        level3 = level2.copy()
        level3[0, 1] = 2  # penalized (tau = 1 < 4)
        level3[1, 2] = 1  # free (tau = 6 >= 4)
        penalized, free = count_changes(hist_mixed, level3, T_stab=4)
        assert (penalized, free) == (1, 1)
        assert count_changes(None, level1, T_stab=4) == (0, 0)


@needs_dataset
class TestEpochScenario:
    @pytest.fixture(scope="class")
    def ladders(self):
        return load_ladders(DATASET_DIR)

    def test_ladder_walk_wraps_around(self, ladders):
        """Epoch t uses second (start + t) mod file length; wraps at the end."""
        sc = random_epoch_scenario(
            np.random.default_rng(1), 3, ladders, ["johnny_L3T3_tf"], epochs=130
        )
        length = len(sc.ladder_tables[0])  # johnny: 60 seconds
        assert length == 60
        for i in range(sc.n):
            for epoch in (0, 1, length - sc.start_seconds[i], 129):
                inst = scenario_instance(sc, epoch, None)
                expected = (
                    sc.ladder_tables[i][(sc.start_seconds[i] + epoch) % length]
                    * sc.scales[i]
                )
                np.testing.assert_allclose(inst.r[i], expected)

    def test_freeze_ladder_pins_the_start_second(self, ladders):
        sc = random_epoch_scenario(
            np.random.default_rng(2), 3, ladders, epochs=10, freeze_ladder=True
        )
        first = scenario_instance(sc, 0, None)
        last = scenario_instance(sc, 9, None)
        np.testing.assert_array_equal(first.r, last.r)
        np.testing.assert_array_equal(first.w, last.w)

    def test_weights_follow_the_ladder_each_epoch(self, ladders):
        """MODEL.md §8: w recomputed per epoch from fixed p and current r."""
        sc = random_epoch_scenario(np.random.default_rng(3), 4, ladders, epochs=5)
        inst0 = scenario_instance(sc, 0, None)
        inst1 = scenario_instance(sc, 1, None)
        for inst in (inst0, inst1):
            expected = weights_from_preferences(sc.p, inst.r, inst.L)
            np.testing.assert_array_equal(inst.w, expected)
        assert not np.array_equal(inst0.w, inst1.w)  # r moved, so w moved

    def test_draw_bounds_and_determinism(self, ladders):
        n = 4
        total = n * (n - 1) * 2000.0
        for seed in range(5):
            sc = random_epoch_scenario(
                np.random.default_rng(seed), n, ladders, epochs=40
            )
            assert 0.25 * total <= sc.B <= total
            est = (n - 1) * 2000.0
            assert np.all(sc.b_hat_traj >= 0.25 * est)
            assert np.all(sc.b_hat_traj <= 1.25 * est)
            assert sc.b_hat_traj.shape == (40, n)
            assert 0.1 <= sc.alpha <= 0.5
            assert 10 <= sc.T_stab <= 20
            assert len(sc.trajectory_kinds) == n
            assert set(sc.trajectory_kinds) <= set(TRAJECTORY_KINDS)
        a = random_epoch_scenario(np.random.default_rng(7), n, ladders, epochs=20)
        b = random_epoch_scenario(np.random.default_rng(7), n, ladders, epochs=20)
        assert a.files == b.files and a.start_seconds == b.start_seconds
        assert a.scales == b.scales and a.B == b.B
        assert a.alpha == b.alpha and a.T_stab == b.T_stab
        assert a.trajectory_kinds == b.trajectory_kinds
        np.testing.assert_array_equal(a.b_hat_traj, b.b_hat_traj)
        np.testing.assert_array_equal(a.p, b.p)
        c = random_epoch_scenario(
            np.random.default_rng(7), n, ladders, epochs=20, trajectory="steps"
        )
        assert c.trajectory_kinds == ("steps",) * n


@needs_dataset
class TestStability:
    @pytest.fixture(scope="class")
    def constant_scenario(self):
        ladders = load_ladders(DATASET_DIR)
        return random_epoch_scenario(
            np.random.default_rng(5),
            4,
            ladders,
            epochs=7,  # >= T_stab + 3
            trajectory="constant",
            freeze_ladder=True,
            alpha_range=(0.3, 0.3),
            T_stab_range=(4, 4),
        )

    def test_stability_theorem_exact_solver(self, constant_scenario):
        """PLAN.md M8 stability theorem (hard assert, exact solver).

        Proof sketch: on a fully constant scenario the epoch-1 optimum stays
        available in every later epoch at zero penalty. While
        ``tau < T_stab``, any allocation deviating on some pair pays a
        positive penalty ``alpha * (T_stab - tau) / (T_stab - 1)`` on at
        least one receiver, so its sorted vector is strictly lex-worse than
        keeping the allocation — the solver must keep it (zero changed
        pairs in epochs 2..T_stab). Once ``tau >= T_stab`` switching
        between tied optima is free, which may change the allocation but
        leaves ``sorted_q`` unchanged. Parameters satisfy
        ``alpha * min(w) >> max(TOL, eps)`` (0.3 * ~0.1 >> 1e-6).
        """
        records = run_epochs(constant_scenario, solve_lex_maxmin)
        for rec in records[1 : constant_scenario.T_stab]:  # epochs 2..T_stab
            assert rec["changes_penalized"] == 0
            assert rec["changes_free"] == 0
        base = records[0]["sorted_q"]
        for rec in records[1:]:
            assert lex_compare(rec["sorted_q"], base) == 0

    def test_records_expose_per_receiver_q(self, constant_scenario):
        """``q`` keeps participant identity; sorting it gives ``sorted_q``."""
        records = run_epochs(constant_scenario, solve_heuristic)
        for rec in records:
            assert len(rec["q"]) == constant_scenario.n
            assert sorted(rec["q"]) == rec["sorted_q"]

    def test_heuristic_on_constant_scenario_report_only(self, constant_scenario):
        """Report-only per PLAN.md: a stability violation is a finding."""
        records = run_epochs(constant_scenario, solve_heuristic)
        changes = [
            (rec["changes_penalized"], rec["changes_free"]) for rec in records[1:]
        ]
        drift = max(
            abs(np.array(rec["sorted_q"]) - np.array(records[0]["sorted_q"])).max()
            for rec in records[1:]
        )
        print(
            f"\nheurystyka na stałym scenariuszu: zmiany={changes}, "
            f"max |drift sorted_q|={drift:.4f}"
        )


@needs_dataset
def test_penalties_reduce_changes():
    """PLAN.md M8: alpha=0.3 run has no more changes than alpha=0 (same seed)."""
    ladders = load_ladders(DATASET_DIR)
    totals = {}
    for alpha in (0.0, 0.3):
        sc = random_epoch_scenario(
            np.random.default_rng(11),
            4,
            ladders,
            epochs=20,
            trajectory="steps",
            alpha_range=(alpha, alpha),
            T_stab_range=(4, 4),
        )
        records = run_epochs(sc, solve_lex_maxmin)
        totals[alpha] = sum(
            rec["changes_penalized"] + rec["changes_free"] for rec in records
        )
    assert totals[0.3] <= totals[0.0]


@needs_dataset
class TestEpochBench:
    def test_smoke_run_writes_valid_csv(self, tmp_path):
        """PLAN.md M8 smoke: 2 scenarios x 50 epochs x lex + heuristic."""
        out = run_epoch_bench(
            2,
            [4],
            ["lex", "heuristic"],
            tmp_path / "epochs.csv",
            epochs=50,
            trajectory="mix",
            base_seed=200,
            dataset_dir=DATASET_DIR,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == FIELDS
        assert len(df) == 2 * 2 * 50
        assert (df["status"] == "ok").all()
        for (_, _), group in df.groupby(["scenario", "algorithm"]):
            assert sorted(group["epoch"]) == list(range(50))
            first = group.sort_values("epoch").iloc[0]
            assert first["changes_penalized"] == 0 and first["changes_free"] == 0
        for _, row in df.head(8).iterrows():
            assert len(json.loads(row["sorted_q"])) == row["n"]

    def test_deterministic(self, tmp_path):
        frames = []
        for name in ("a", "b"):
            out = run_epoch_bench(
                1,
                [4],
                ["heuristic"],
                tmp_path / f"{name}.csv",
                epochs=10,
                base_seed=9,
                dataset_dir=DATASET_DIR,
            )
            frames.append(pd.read_csv(out))
        stable = ["epoch", "q_min", "q_mean", "sorted_q", "changes_penalized"]
        pd.testing.assert_frame_equal(frames[0][stable], frames[1][stable])

    def test_unknown_algorithm_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown algorithms"):
            run_epoch_bench(1, [4], ["bogus"], tmp_path / "x.csv")
