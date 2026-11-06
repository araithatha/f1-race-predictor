"""Leakage tests for the pre-race feature table.

The key guarantee: a race's features may only use information from before that
race. We check it directly by building features on the full dataset and again
with every race after a cut-off removed. Rows before the cut-off must come out
identical; if any future race leaked into them, they would differ.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from f1model import features as F

PROCESSED = F.PROCESSED
needs_data = pytest.mark.skipif(
    not (PROCESSED / "fact_practice.csv").exists(), reason="data/processed is missing (see data/README.md)"
)


def _truncate(tables: dict, last_race_order: int) -> dict:
    races = tables["dim_race"]
    keep = set(races.loc[races["season"] * 100 + races["round"] <= last_race_order, "race_id"])
    return {k: (v[v["race_id"].isin(keep)] if "race_id" in v else v) for k, v in tables.items()}


def _build(tables: dict) -> pd.DataFrame:
    df = F.build_base(tables)
    df = F.add_context(df)
    df = F.add_weekend_features(df, tables)
    df = F.add_driver_form(df)
    df = F.add_team_form(df, tables)
    df = F.add_reliability_trend(df)
    df = F.add_circuit_history(df)
    df = F.add_regulation_interactions(df)
    return df


@needs_data
@pytest.mark.parametrize("cutoff", [201810, 202203, 202312, 202505, 202602])
def test_no_future_information_leaks(cutoff):
    tables = F.load_tables(PROCESSED)
    full = _build(tables).set_index(["race_id", "driver_id"])
    trunc = _build(_truncate(tables, cutoff)).set_index(["race_id", "driver_id"])
    # The race at the cut-off itself must also be unaffected by anything after it.
    cols = [c for c in F.ALL_FEATURES if c in trunc.columns]
    a = full.loc[trunc.index, cols].astype(float)
    b = trunc[cols].astype(float)
    diff = ~np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True)
    leaking = [c for c, d in zip(cols, diff.any(axis=0)) if d]
    assert not leaking, f"features change when future races are removed: {leaking}"


@needs_data
def test_feature_table_shape_and_targets():
    df = F.build_feature_table(PROCESSED)
    assert not df.duplicated(["race_id", "driver_id"]).any()
    # Finisher ranks run 1..n within each race, with no gaps.
    fin = df.dropna(subset=["finisher_rank"])
    ok = fin.groupby("race_id")["finisher_rank"].apply(lambda s: sorted(s) == list(range(1, len(s) + 1)))
    assert ok.all()
    assert (df.groupby("race_id")["is_win"].sum() == 1).all()


def test_rolling_prior_excludes_current_row():
    df = pd.DataFrame({"g": ["a"] * 4, "x": [1.0, 2.0, 3.0, 4.0]})
    out = F._rolling_prior(df, "g", "x", window=2)
    assert np.isnan(out.iloc[0])
    assert out.tolist()[1:] == [1.0, 1.5, 2.5]
