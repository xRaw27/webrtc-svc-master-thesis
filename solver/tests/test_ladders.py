"""Loader tests (PLAN2): synthetic frame + sanity checks on a real dataset."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sfu_alloc.data.ladders import LADDER_CONFIGS, load_ladders, sample_ladder

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


def tiny_df() -> pd.DataFrame:
    rows = []
    for source, base in [("bbb", 100.0), ("johnny", 50.0)]:
        for sec in range(3):
            for level, r in enumerate([1.0, 3.0, 9.0, 27.0, 81.0], start=1):
                rows.append(
                    {
                        "source": source,
                        "config": "L3T3_tf",
                        "second": sec,
                        "level": level,
                        "r_raw": base * r + sec,
                    }
                )
    return pd.DataFrame(rows)


class TestSampleLadder:
    def test_prepends_level_zero_and_sorts_levels(self):
        rng = np.random.default_rng(0)
        ladder = sample_ladder(tiny_df(), rng, "L3T3_tf")
        assert len(ladder) == 6
        assert ladder[0] == 0.0
        assert all(a < b for a, b in pairwise(ladder))

    def test_deterministic_for_same_rng_seed(self):
        a = sample_ladder(tiny_df(), np.random.default_rng(7), "L3T3_tf")
        b = sample_ladder(tiny_df(), np.random.default_rng(7), "L3T3_tf")
        assert a == b

    def test_source_filter(self):
        rng = np.random.default_rng(1)
        ladder = sample_ladder(tiny_df(), rng, "L3T3_tf", source="johnny")
        sec = ladder[1] - 50.0  # all levels must come from the same second
        assert sec in (0.0, 1.0, 2.0)
        assert ladder[1:] == [50.0 * r + sec for r in [1.0, 3.0, 9.0, 27.0, 81.0]]

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="no dataset rows"):
            sample_ladder(tiny_df(), np.random.default_rng(0), "L2T2_tf")


@pytest.mark.skipif(
    not any(DATASET_DIR.glob("*.csv")),
    reason="dataset not built yet (run scripts/build_ladder_dataset.py)",
)
class TestShippedDataset:
    def test_loads_and_has_complete_levels(self):
        df = load_ladders(DATASET_DIR)
        assert set(df["config"]).issubset(set(LADDER_CONFIGS))
        for (config, source, _sec), group in df.groupby(["config", "source", "second"]):
            expected = 5 if config.startswith("L3T3") else 3
            assert sorted(group["level"]) == list(range(1, expected + 1)), (
                f"{config}/{source}"
            )

    def test_bitrates_positive_and_sampling_works(self):
        df = load_ladders(DATASET_DIR)
        assert (df["r_raw"] >= 0).all()
        assert (df.loc[df["level"] == df["level"].max(), "r_raw"] > 0).any()
        rng = np.random.default_rng(42)
        for config in sorted(set(df["config"])):
            ladder = sample_ladder(df, rng, config)
            assert ladder[0] == 0.0
            assert len(ladder) == (6 if config.startswith("L3T3") else 4)
