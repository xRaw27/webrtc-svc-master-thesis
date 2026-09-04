"""Ladder dataset loader (PLAN2 loader stub).

Reads the CSVs produced by ``scripts/build_ladder_dataset.py`` and samples
single-epoch ladders from them. Per PLAN2 this is the only package code
that touches the dataset; nothing else consumes it yet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["LADDER_CONFIGS", "load_ladders", "sample_ladder", "sample_ladder_row"]

LADDER_CONFIGS = ("L2T2_tf", "L2T2_sf", "L3T3_tf", "L3T3_sf")
_COLUMNS = ["source", "config", "second", "level", "r_raw"]


def load_ladders(dataset_dir: str | Path) -> pd.DataFrame:
    """Read all ladder CSVs from ``dataset_dir`` into one long DataFrame.

    The dataset ships one CSV per source x ladder variant
    (``{source}_{variant}.csv``, e.g. ``bbb_L3T3_tf.csv``); every ``*.csv``
    in the directory is read and concatenated. Raises ``FileNotFoundError``
    when no CSV is found and ``ValueError`` when required columns are
    missing. Columns: source, config, second, level, r_raw (raw per-second
    kbps; any smoothing is up to the consumer).
    """
    dataset_dir = Path(dataset_dir)
    paths = sorted(dataset_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no ladder CSVs found in {dataset_dir}")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    missing = [c for c in _COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing columns: {missing}")
    return df


def sample_ladder_row(
    df: pd.DataFrame,
    rng: np.random.Generator,
    config: str,
    source: str | None = None,
) -> tuple[str, int, list[float]]:
    """Like ``sample_ladder``, but also report the drawn (source, second).

    Consumes exactly one ``rng`` draw, identical to ``sample_ladder``, so
    the two are interchangeable inside a seeded draw sequence.
    """
    sel = df[df["config"] == config]
    if source is not None:
        sel = sel[sel["source"] == source]
    if sel.empty:
        raise ValueError(f"no dataset rows for config={config!r}, source={source!r}")
    keys = sel[["source", "second"]].drop_duplicates().sort_values(["source", "second"])
    picked = keys.iloc[int(rng.integers(len(keys)))]
    rows = sel[
        (sel["source"] == picked["source"]) & (sel["second"] == picked["second"])
    ].sort_values("level")
    ladder = [0.0, *rows["r_raw"].astype(float).tolist()]
    return str(picked["source"]), int(picked["second"]), ladder


def sample_ladder(
    df: pd.DataFrame,
    rng: np.random.Generator,
    config: str,
    source: str | None = None,
) -> list[float]:
    """Return ``r`` for one uniformly random second of one ladder variant.

    The empty level 0 (bitrate 0) is prepended, so the result has length
    ``L + 1`` and plugs directly into ``Instance`` ladder rows.
    Deterministic given the ``rng`` state.
    """
    return sample_ladder_row(df, rng, config, source)[2]
