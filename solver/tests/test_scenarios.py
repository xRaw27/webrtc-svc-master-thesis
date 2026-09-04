"""M7 tests: static scenario generation and the runner harness.

All tests need the shipped dataset; they skip when it has not been built.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfu_alloc.algorithms import finalize_result
from sfu_alloc.benchmark.runner import ALGORITHMS, FIELDS, run_static
from sfu_alloc.data.ladders import load_ladders
from sfu_alloc.evaluator import lex_compare
from sfu_alloc.instance import instances_equal, weights_from_preferences
from sfu_alloc.scenarios import (
    dataset_files,
    random_static_case,
    random_static_instance,
)

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"

pytestmark = pytest.mark.skipif(
    not any(DATASET_DIR.glob("*.csv")),
    reason="dataset not built yet (run scripts/build_ladder_dataset.py)",
)


@pytest.fixture(scope="module")
def ladders() -> pd.DataFrame:
    return load_ladders(DATASET_DIR)


class TestRandomStaticInstance:
    def test_every_dataset_file_yields_valid_instances(self, ladders):
        """M7 acceptance: instances from every file pass Instance validation."""
        for stem in dataset_files(ladders):
            for seed in range(3):
                rng = np.random.default_rng(seed)
                inst = random_static_instance(rng, 4, ladders, [stem])
                assert inst.history is None
                expected_L = 5 if "L3T3" in stem else 3
                assert all(int(x) == expected_L for x in inst.L)

    def test_deterministic_for_same_seed(self, ladders):
        a = random_static_instance(np.random.default_rng(7), 5, ladders)
        b = random_static_instance(np.random.default_rng(7), 5, ladders)
        assert instances_equal(a, b)
        c = random_static_instance(np.random.default_rng(8), 5, ladders)
        assert not instances_equal(a, c)

    def test_scale_range_scales_ladders_only(self, ladders):
        """Doubling the scale doubles r; weights are scale-invariant."""
        base = random_static_instance(
            np.random.default_rng(3), 4, ladders, scale_range=(1.0, 1.0)
        )
        doubled = random_static_instance(
            np.random.default_rng(3), 4, ladders, scale_range=(2.0, 2.0)
        )
        np.testing.assert_allclose(doubled.r, base.r * 2.0)
        np.testing.assert_allclose(doubled.w, base.w)
        np.testing.assert_allclose(doubled.b_hat, base.b_hat)
        assert doubled.B == base.B

    def test_preference_split(self, ladders):
        """p_default_prefs=1.0 gives purely structural weights."""
        inst = random_static_instance(
            np.random.default_rng(1), 5, ladders, p_default_prefs=1.0
        )
        expected = weights_from_preferences(np.ones((inst.n, inst.n)), inst.r, inst.L)
        np.testing.assert_array_equal(inst.w, expected)

    def test_bandwidth_draw_bounds(self, ladders):
        for seed in range(10):
            inst = random_static_instance(np.random.default_rng(seed), 4, ladders)
            est = 3 * 2000.0
            assert np.all((0.25 * est <= inst.b_hat) & (inst.b_hat <= 1.25 * est))
            total = 4 * 3 * 2000.0
            assert 0.25 * total <= inst.B <= total

    def test_case_exposes_the_draws_behind_the_instance(self, ladders):
        """random_static_case = the same instance + the raw draws behind it."""
        case = random_static_case(np.random.default_rng(7), 5, ladders)
        inst = random_static_instance(np.random.default_rng(7), 5, ladders)
        assert instances_equal(case.instance, inst)
        stems = set(dataset_files(ladders))
        for i in range(5):
            assert case.files[i] in stems
            source, _, config = case.files[i].partition("_")
            sub = ladders[(ladders["source"] == source) & (ladders["config"] == config)]
            row = sub[sub["second"] == case.seconds[i]].sort_values("level")
            assert len(row) == int(inst.L[i])
            expected = np.array([0.0, *row["r_raw"]]) * case.scales[i]
            np.testing.assert_allclose(inst.r[i, : int(inst.L[i]) + 1], expected)
        np.testing.assert_array_equal(
            inst.w, weights_from_preferences(case.p, inst.r, inst.L)
        )

    def test_rejects_bad_arguments(self, ladders):
        with pytest.raises(ValueError, match="n must be >= 2"):
            random_static_instance(np.random.default_rng(0), 1, ladders)
        with pytest.raises(ValueError, match="must not be empty"):
            random_static_instance(np.random.default_rng(0), 3, ladders, [])


class TestRunner:
    def test_lex_matches_pruned_on_tiny_scenarios(self, ladders):
        """M7 acceptance: harness-level sanity on tiny n."""
        for seed in range(5):
            rng = np.random.default_rng(seed)
            inst = random_static_instance(rng, 3, ladders)
            res_lex = ALGORITHMS["lex"](inst)
            res_pruned = ALGORITHMS["pruned"](inst)
            assert lex_compare(res_lex.sorted_q, res_pruned.sorted_q) == 0

    def test_smoke_run_writes_valid_csv(self, tmp_path):
        """M7 acceptance: end-to-end smoke config with a valid CSV."""
        out = run_static(
            4,
            [4, 6],
            ["lex", "maxmin", "heuristic"],
            tmp_path / "static.csv",
            base_seed=100,
            dataset_dir=DATASET_DIR,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == FIELDS
        assert len(df) == 4 * 3
        assert (df["status"] == "ok").all()
        assert (df["q_min"] <= df["q_mean"] + 1e-12).all()
        for _, row in df.iterrows():
            assert len(json.loads(row["sorted_q"])) == row["n"]
            assert isinstance(json.loads(row["stats"]), dict)

    def test_runner_deterministic(self, tmp_path):
        """M7 acceptance: same config + seed reproduces identical results."""
        a = pd.read_csv(
            run_static(
                3,
                [4],
                ["heuristic", "maxmin"],
                tmp_path / "a.csv",
                base_seed=5,
                dataset_dir=DATASET_DIR,
            )
        )
        b = pd.read_csv(
            run_static(
                3,
                [4],
                ["heuristic", "maxmin"],
                tmp_path / "b.csv",
                base_seed=5,
                dataset_dir=DATASET_DIR,
            )
        )
        stable = ["scenario", "seed", "n", "algorithm", "q_min", "q_mean", "sorted_q"]
        pd.testing.assert_frame_equal(a[stable], b[stable])

    def test_all_algorithms_get_the_same_instance(self, tmp_path, monkeypatch):
        """M7 acceptance: the instance is generated once per scenario."""
        captured = []

        def spy(inst):
            captured.append(inst)
            level = np.zeros((inst.n, inst.n), dtype=np.int64)
            return finalize_result(inst, level, {}, "spy")

        monkeypatch.setitem(ALGORITHMS, "spy1", spy)
        monkeypatch.setitem(ALGORITHMS, "spy2", spy)
        run_static(
            1,
            [4],
            ["spy1", "spy2"],
            tmp_path / "spy.csv",
            base_seed=0,
            dataset_dir=DATASET_DIR,
        )
        assert len(captured) == 2
        assert captured[0] is captured[1]

    def test_error_status_recorded_and_run_continues(self, tmp_path):
        """A failing algorithm yields an error row, not a crashed run."""
        out = run_static(
            1,
            [8],
            ["naive", "heuristic"],
            tmp_path / "err.csv",
            base_seed=0,
            dataset_dir=DATASET_DIR,
        )
        df = pd.read_csv(out)
        naive = df[df["algorithm"] == "naive"].iloc[0]
        assert naive["status"].startswith("error: ")
        heur = df[df["algorithm"] == "heuristic"].iloc[0]
        assert heur["status"] == "ok"

    def test_unknown_algorithm_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown algorithms"):
            run_static(1, [4], ["bogus"], tmp_path / "x.csv")
