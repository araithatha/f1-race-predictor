"""Build the pre-race feature table: one row per driver per race.

Every feature uses only information available before lights out:

* the race weekend itself: grid, qualifying and practice pace
* the driver's and team's form in earlier races (rolling windows, shifted by one race)
* team strategy history: pit-crew speed, number of stops, positions gained in races
* reliability: driver and team DNF rates in earlier races
* circuit history from earlier editions: DNF rate and how much the grid order holds
* race-day weather at the circuit
* regulation changes: races since the last rule reset, team and circuit history that restarts
  at each reset, last-era form faded in over the first races, and this season's retirement rates

Rolling features are computed on races strictly before the current one
(``shift(1)``), so a race never sees its own result. ``tests/test_features.py``
checks this.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Input tables come from the f1-data-pipeline project. A snapshot ships in data/processed/;
# point F1_DATA_DIR at a fresh pipeline output to use newer data.
PROCESSED = Path(os.environ.get("F1_DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "processed"))

# Teams rename and change owners; follow one continuous team through those changes.
TEAM_LINEAGE = {
    "toro_rosso": "racing_bulls", "alphatauri": "racing_bulls", "rb": "racing_bulls",
    "force_india": "aston_martin", "racing_point": "aston_martin",
    "lotus_f1": "alpine", "renault": "alpine",
    "alfa": "sauber", "audi": "sauber",
    "marussia": "manor",
}

# Seasons that started under major new technical regulations: past form is a weaker guide.
NEW_REGULATION_SEASONS = {2014, 2017, 2022, 2026}
FORM_FADE_IN_RACES = 5  # races after a reset before last-era form is trusted fully

FINISHED_STATUSES = {"Finished", "Lapped"}
NON_STARTERS = {"Did not start", "Withdrew", "Did not qualify", "Did not prequalify"}


def _is_finished(status: str) -> bool:
    return status in FINISHED_STATUSES or str(status).startswith("+")


def load_tables(path: Path = PROCESSED) -> dict[str, pd.DataFrame]:
    names = ["dim_race", "fact_result", "fact_qualifying", "fact_pit_stop", "fact_practice"]
    return {n: pd.read_csv(path / f"{n}.csv") for n in names if (path / f"{n}.csv").exists()}


def _rolling_prior(df: pd.DataFrame, by: str | list[str], col: str, window: int | None, how: str = "mean") -> pd.Series:
    """Rolling statistic over a group's previous rows only (current row excluded)."""
    shifted = df.groupby(by, sort=False)[col].shift(1)
    grouped = shifted.groupby([df[b] for b in ([by] if isinstance(by, str) else by)], sort=False)
    if window is None:
        roll = grouped.expanding(min_periods=1)
    else:
        roll = grouped.rolling(window, min_periods=1)
    out = getattr(roll, how)()
    return out.reset_index(level=list(range(out.index.nlevels - 1)), drop=True).reindex(df.index)


def build_base(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Starters table with targets, in chronological order."""
    races = t["dim_race"].copy()
    races["race_order"] = races["season"] * 100 + races["round"]
    fr = t["fact_result"].copy()
    fr = fr[~fr["status"].isin(NON_STARTERS)]

    df = fr.merge(races, on="race_id", how="left")
    df["team_id"] = df["constructor_id"].replace(TEAM_LINEAGE)
    df["is_dnf"] = ~df["status"].map(_is_finished) & (df["status"] != "Disqualified")
    df["is_dsq"] = df["status"] == "Disqualified"

    # Target for the pace model: finishing order among cars that saw the flag.
    fin = df[~df["is_dnf"] & ~df["is_dsq"]]
    df["finisher_rank"] = fin.groupby("race_id")["finish_position"].rank(method="first")
    df["n_starters"] = df.groupby("race_id")["driver_id"].transform("count")
    df["n_finishers"] = df.groupby("race_id")["finisher_rank"].transform("max")
    return df.sort_values(["race_order", "finish_position"]).reset_index(drop=True)


def add_weekend_features(df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    q = t["fact_qualifying"][["race_id", "driver_id", "quali_position", "gap_to_pole_pct", "reached_q3"]]
    df = df.merge(q, on=["race_id", "driver_id"], how="left")
    df["pit_lane_start"] = df["grid"] == 0
    df["grid_pct"] = (df["grid_adj"] - 1) / (df["n_starters"] - 1)
    # Track position is worth far more at the front: pole vs P2 matters more than P12 vs P13.
    df["grid_log"] = np.log(df["grid_adj"])
    df["is_pole"] = df["grid_adj"] == 1
    df["is_front_row"] = df["grid_adj"] <= 2

    # Qualifying gap to the team-mate: the cleanest read of driver vs car.
    team_gap = df.groupby(["race_id", "constructor_id"])["gap_to_pole_pct"]
    df["quali_gap_to_teammate_pct"] = df["gap_to_pole_pct"] - (team_gap.transform("sum") - df["gap_to_pole_pct"]) / (
        team_gap.transform("count") - 1
    ).replace(0, np.nan)

    fp = t.get("fact_practice")
    if fp is not None and not fp.empty:
        best = fp.groupby(["race_id", "driver_id"]).agg(
            fp_best_gap_pct=("best_lap_gap_pct", "min"),
            fp_laps=("laps_completed", "sum"),
        )
        # Long runs: FP2 is the usual race-simulation session; sprint weekends only have FP1.
        lr = fp.sort_values("session_name", key=lambda s: s.map({"Practice 2": 0, "Practice 1": 1, "Practice 3": 2}))
        lr = lr.dropna(subset=["longrun_gap_pct"]).drop_duplicates(["race_id", "driver_id"])
        lr = lr.set_index(["race_id", "driver_id"])["longrun_gap_pct"].rename("fp_longrun_gap_pct")
        df = df.merge(best.join(lr, how="left").reset_index(), on=["race_id", "driver_id"], how="left")
    else:
        df["fp_best_gap_pct"] = df["fp_longrun_gap_pct"] = df["fp_laps"] = np.nan
    return df


def add_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["race_order", "finish_position"]).reset_index(drop=True)
    df["_gain"] = np.where(df["is_dnf"] | df["is_dsq"], np.nan, df["positions_gained"])
    df["_dnf"] = df["is_dnf"].astype(float)
    df["_finish_pct"] = (df["finish_position"] - 1) / (df["n_starters"] - 1)

    df["drv_avg_finish_pct_5"] = _rolling_prior(df, "driver_id", "_finish_pct", 5)
    df["drv_avg_points_5"] = _rolling_prior(df, "driver_id", "points", 5)
    df["drv_avg_gain_10"] = _rolling_prior(df, "driver_id", "_gain", 10)
    df["drv_dnf_rate_20"] = _rolling_prior(df, "driver_id", "_dnf", 20)
    df["drv_starts"] = df.groupby("driver_id").cumcount()
    df["drv_circuit_avg_finish_pct"] = _rolling_prior(df, ["driver_id", "circuit_id"], "_finish_pct", None)

    # Wet-weather racecraft: positions gained in earlier wet races only.
    wet = df["wx_wet_race"].fillna(False).astype(bool)
    df["_wet_gain"] = np.where(wet, df["_gain"], np.nan)
    df["drv_wet_gain_hist"] = _rolling_prior(df, "driver_id", "_wet_gain", None)

    # Championship position before the race.
    df["drv_season_points_before"] = df.groupby(["driver_id", "season"])["points"].cumsum() - df["points"]
    df["drv_champ_rank_before"] = df.groupby("race_id")["drv_season_points_before"].rank(ascending=False, method="min")
    return df


def add_team_form(df: pd.DataFrame, t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Collapse to one row per team per race, then roll over the team's earlier races.
    per_race = df.groupby(["team_id", "race_id", "race_order"], as_index=False).agg(
        _t_finish_pct=("_finish_pct", "mean"),
        _t_dnf=("_dnf", "mean"),
        _t_gain=("_gain", "mean"),
        _t_points=("points", "sum"),
        season=("season", "first"),
        era=("era", "first"),
    )

    ps = t.get("fact_pit_stop")
    if ps is not None and not ps.empty:
        ps = ps.merge(df[["race_id", "driver_id", "team_id"]], on=["race_id", "driver_id"], how="inner")
        normal = ps[ps["is_normal_stop"]]
        crew = normal.groupby(["team_id", "race_id"])["pit_time_vs_race_median"].mean().rename("_t_pit_speed")
        stops_per_car = (
            normal.groupby(["team_id", "race_id", "driver_id"]).size().groupby(["team_id", "race_id"]).mean()
        )
        race_avg = stops_per_car.groupby("race_id").transform("mean")
        stops_rel = (stops_per_car - race_avg).rename("_t_stops_rel")
        per_race = per_race.merge(pd.concat([crew, stops_rel], axis=1).reset_index(), on=["team_id", "race_id"], how="left")
    else:
        per_race["_t_pit_speed"] = per_race["_t_stops_rel"] = np.nan

    per_race = per_race.sort_values("race_order").reset_index(drop=True)
    per_race["team_avg_finish_pct_5"] = _rolling_prior(per_race, "team_id", "_t_finish_pct", 5)
    # Same, but restarting at each regulation reset: under new rules, last era's car says little.
    per_race["team_avg_finish_pct_5_era"] = _rolling_prior(per_race, ["team_id", "era"], "_t_finish_pct", 5)
    per_race["team_dnf_rate_10"] = _rolling_prior(per_race, "team_id", "_t_dnf", 10)
    per_race["team_avg_gain_10"] = _rolling_prior(per_race, "team_id", "_t_gain", 10)
    per_race["team_pit_speed_10"] = _rolling_prior(per_race, "team_id", "_t_pit_speed", 10)
    per_race["team_stops_rel_10"] = _rolling_prior(per_race, "team_id", "_t_stops_rel", 10)
    per_race["team_season_points_before"] = per_race.groupby(["team_id", "season"])["_t_points"].cumsum() - per_race["_t_points"]
    per_race["team_champ_rank_before"] = per_race.groupby("race_id")["team_season_points_before"].rank(ascending=False, method="min")

    keep = ["team_id", "race_id", "team_avg_finish_pct_5", "team_avg_finish_pct_5_era", "team_dnf_rate_10", "team_avg_gain_10",
            "team_pit_speed_10", "team_stops_rel_10", "team_season_points_before", "team_champ_rank_before"]
    return df.merge(per_race[keep], on=["team_id", "race_id"], how="left")


def add_circuit_history(df: pd.DataFrame) -> pd.DataFrame:
    def grid_holds(g: pd.DataFrame) -> float:
        g = g[~g["is_dnf"] & ~g["is_dsq"]]
        return g["grid_adj"].corr(g["finish_position"], method="spearman") if len(g) > 5 else np.nan

    per_race = df.groupby(["circuit_id", "era", "race_id", "race_order"]).apply(
        lambda g: pd.Series({"_c_dnf": g["_dnf"].mean(), "_c_grid_corr": grid_holds(g)}), include_groups=False
    ).reset_index().sort_values("race_order").reset_index(drop=True)
    per_race["circ_dnf_rate_hist"] = _rolling_prior(per_race, "circuit_id", "_c_dnf", None)
    per_race["circ_grid_corr_hist"] = _rolling_prior(per_race, "circuit_id", "_c_grid_corr", None)
    per_race["circ_editions_before"] = per_race.groupby("circuit_id").cumcount()
    # Overtaking rules change between eras (ground effect from 2022, no DRS from 2026), so also track
    # how well the grid order holds at this circuit within the current era only.
    per_race["circ_grid_corr_era"] = _rolling_prior(per_race, ["circuit_id", "era"], "_c_grid_corr", None)
    cols = ["race_id", "circ_dnf_rate_hist", "circ_grid_corr_hist", "circ_editions_before", "circ_grid_corr_era"]
    return df.merge(per_race[cols], on="race_id", how="left")


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Regulation eras: 2014-16, 2017-21, 2022-25, 2026-. Count races since the last reset."""
    df["new_regs_season"] = df["season"].isin(NEW_REGULATION_SEASONS)
    df["wx_wet_race"] = df["wx_wet_race"].fillna(False).astype(bool)
    starts = sorted(NEW_REGULATION_SEASONS)
    df["era"] = np.searchsorted(starts, df["season"].to_numpy(), side="right")
    races = df.drop_duplicates("race_id")[["race_id", "race_order", "era"]].sort_values("race_order")
    races["races_into_era"] = races.groupby("era").cumcount()
    df = df.merge(races[["race_id", "races_into_era"]], on="race_id", how="left")
    # 0 at the first race under new rules, rising to 1 after FORM_FADE_IN_RACES races.
    df["era_form_weight"] = (df["races_into_era"] / FORM_FADE_IN_RACES).clip(upper=1.0)
    return df


def add_reliability_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Retirement rates so far this season, so a sudden change (e.g. new 2026 power units) shows up fast."""
    race = df.groupby(["race_id", "race_order", "season"], as_index=False)["_dnf"].mean().sort_values("race_order")
    race = race.reset_index(drop=True)
    race["field_dnf_rate_season"] = _rolling_prior(race, "season", "_dnf", None)
    df = df.merge(race[["race_id", "field_dnf_rate_season"]], on="race_id", how="left")
    team = df.groupby(["team_id", "season", "race_id", "race_order"], as_index=False)["_dnf"].mean()
    team = team.sort_values("race_order").reset_index(drop=True)
    team["team_dnf_rate_season"] = _rolling_prior(team, ["team_id", "season"], "_dnf", None)
    return df.merge(team[["team_id", "race_id", "team_dnf_rate_season"]], on=["team_id", "race_id"], how="left")


def add_regulation_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Let the model discount last-era form right after a reset and lean on this weekend's pace instead."""
    stale = 1.0 - df["era_form_weight"]
    df["team_form_x_stale"] = df["team_avg_finish_pct_5"] * stale
    df["drv_form_x_stale"] = df["drv_avg_finish_pct_5"] * stale
    df["quali_gap_x_stale"] = df["gap_to_pole_pct"] * stale
    return df


# Feature groups (used by the models and for ablation studies).
FEATURES = {
    "weekend": ["grid_adj", "grid_pct", "grid_log", "is_pole", "is_front_row", "pit_lane_start", "quali_position", "gap_to_pole_pct",
                "quali_gap_to_teammate_pct", "reached_q3"],
    "practice": ["fp_best_gap_pct", "fp_longrun_gap_pct", "fp_laps"],
    "driver_form": ["drv_avg_finish_pct_5", "drv_avg_points_5", "drv_avg_gain_10", "drv_starts",
                    "drv_circuit_avg_finish_pct", "drv_season_points_before", "drv_champ_rank_before",
                    "drv_wet_gain_hist"],
    "team_form": ["team_avg_finish_pct_5", "team_season_points_before", "team_champ_rank_before"],
    "team_strategy": ["team_avg_gain_10", "team_pit_speed_10", "team_stops_rel_10"],
    "reliability": ["drv_dnf_rate_20", "team_dnf_rate_10", "team_dnf_rate_season", "field_dnf_rate_season"],
    "circuit": ["circ_dnf_rate_hist", "circ_grid_corr_hist", "circ_editions_before"],
    "weather": ["wx_air_temp_c", "wx_precip_mm", "wx_humidity_pct", "wx_wind_kph", "wx_wet_race"],
    "context": ["new_regs_season", "round"],
    "regulation": ["races_into_era", "era_form_weight", "team_avg_finish_pct_5_era", "circ_grid_corr_era",
                   "team_form_x_stale", "drv_form_x_stale", "quali_gap_x_stale"],
}
ALL_FEATURES = [f for group in FEATURES.values() for f in group]

# Regulation-aware features are built and tested as a candidate add-on. run.py adopts them only if
# they improve the validation seasons (on the current data they don't; see the project README).
REGULATION_AWARE = FEATURES["regulation"] + ["team_dnf_rate_season", "field_dnf_rate_season"]
CORE_FEATURES = [f for f in ALL_FEATURES if f not in REGULATION_AWARE]

TARGETS = ["finish_position", "finisher_rank", "n_finishers", "n_starters", "is_dnf", "is_dsq",
           "is_win", "is_podium", "is_top10", "status"]
IDS = ["race_id", "season", "round", "race_order", "race_name", "circuit_id", "driver_id", "constructor_id", "team_id", "era"]


def build_feature_table(path: Path = PROCESSED) -> pd.DataFrame:
    t = load_tables(path)
    df = build_base(t)
    df = add_context(df)
    df = add_weekend_features(df, t)
    df = add_driver_form(df)
    df = add_team_form(df, t)
    df = add_reliability_trend(df)
    df = add_circuit_history(df)
    df = add_regulation_interactions(df)
    for col in ALL_FEATURES:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
    df["reached_q3"] = df["reached_q3"].astype(float)
    cols = list(dict.fromkeys(IDS + TARGETS + ALL_FEATURES))
    return df[cols].sort_values(["race_order", "finish_position"]).reset_index(drop=True)
