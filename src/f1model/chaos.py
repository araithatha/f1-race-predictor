"""Stage 2: the things nobody controls.

A pace model on its own gives one predicted order, as if every race ran
cleanly. Real races don't: cars break down, drivers crash, safety cars bunch
the field, rain turns up. This layer adds two kinds of chaos and turns the pace
prediction into honest probabilities.

1. **Retirement risk.** A classifier estimates each car's chance of not
   finishing, from team and driver reliability history, how often the circuit
   eats cars, grid position (midfield first-lap pile-ups) and weather.

2. **Race-day randomness.** Each car's pace is perturbed with random noise:
   strategy calls, safety-car timing, a bad pit stop. The noise is larger in
   wet races. Its size is not guessed; it is fitted so the simulated win, podium
   and top-10 probabilities best match what happened in the validation seasons.

Each race is then simulated 10,000 times. A driver's win probability is the
share of simulations in which they won.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import FIRST_TRAIN_SEASON, SEED

log = logging.getLogger(__name__)

DNF_FEATURES = [
    "team_dnf_rate_10", "drv_dnf_rate_20", "circ_dnf_rate_hist", "grid_pct", "pit_lane_start",
    "wx_wet_race", "wx_precip_mm", "new_regs_season", "round", "drv_starts", "team_avg_finish_pct_5",
]
# Candidate add-on: this season's retirement rates, which react within a few races when new rules
# change reliability (2026: 19% of starts vs 11% in 2025). Compared like any other DNF model.
DNF_FEATURES_WITH_SEASON_RATES = DNF_FEATURES + ["team_dnf_rate_season", "field_dnf_rate_season", "races_into_era"]
EPS = 1e-6


# --------------------------------------------------------------------------- #
# 1. Retirement (DNF) risk
# --------------------------------------------------------------------------- #
class BaseRate:
    """Every car gets the historical average DNF rate. The bar to beat."""

    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def _logit():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.3, max_iter=2000))


def _lgbm_clf():
    return lgb.LGBMClassifier(num_leaves=7, n_estimators=200, learning_rate=0.03, min_child_samples=50, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)


# (model factory, training window in seasons or None for all earlier seasons, feature list or None for default).
# Cars have become far more reliable since the mid-2010s, so a recent window may track today's rates better.
DNF_CANDIDATES = {
    "Base rate (baseline)": (BaseRate, None, None),
    "Base rate, last 3 seasons": (BaseRate, 3, None),
    "Logistic regression": (_logit, None, None),
    "Logistic regression, last 4 seasons": (_logit, 4, None),
    "Logistic regression, last 4 seasons + this season's rates": (_logit, 4, DNF_FEATURES_WITH_SEASON_RATES),
    "LightGBM classifier": (_lgbm_clf, None, None),
}


def dnf_walk_forward(df: pd.DataFrame, spec, seasons: list[int]) -> pd.Series:
    make_model, window, features = spec
    features = features or DNF_FEATURES
    preds = []
    for season in seasons:
        first = FIRST_TRAIN_SEASON if window is None else max(FIRST_TRAIN_SEASON, season - window)
        train = df[(df["season"] >= first) & (df["season"] < season)]
        test = df[df["season"] == season]
        model = make_model().fit(train[features], train["is_dnf"].astype(int))
        preds.append(pd.Series(model.predict_proba(test[features])[:, 1], index=test.index))
    return pd.concat(preds)


def compare_dnf_models(df: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, str]:
    rows = []
    for name, spec in DNF_CANDIDATES.items():
        p = dnf_walk_forward(df, spec, seasons)
        y = df.loc[p.index, "is_dnf"].astype(int)
        rows.append({"model": name, "log_loss": log_loss(y, p.clip(EPS, 1 - EPS)), "brier": brier_score_loss(y, p),
                     "mean_predicted": p.mean(), "actual_rate": y.mean()})
    board = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    return board, board.loc[0, "model"]


# --------------------------------------------------------------------------- #
# 2. Monte Carlo race simulation
# --------------------------------------------------------------------------- #
def standardise_scores(df: pd.DataFrame, score_col: str = "score") -> pd.Series:
    """Put pace scores on a common scale within each race (mean 0, sd 1; lower = faster)."""
    g = df.groupby("race_id")[score_col]
    return (df[score_col] - g.transform("mean")) / g.transform("std").replace(0, 1)


def simulate_race(mu: np.ndarray, p_dnf: np.ndarray, sigma: float, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (n_sims, n_cars) matrix of finishing positions (1 = win)."""
    n = len(mu)
    perf = mu[None, :] + sigma * rng.standard_normal((n_sims, n))
    dnf = rng.random((n_sims, n)) < p_dnf[None, :]
    # Retired cars are classified behind every finisher, in random order among themselves.
    perf = np.where(dnf, 1e6 + rng.random((n_sims, n)), perf)
    return perf.argsort(axis=1).argsort(axis=1) + 1


def race_probabilities(race: pd.DataFrame, sigma: float, wet_mult: float, n_sims: int = 10_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s = sigma * (wet_mult if bool(race["wx_wet_race"].iloc[0]) else 1.0)
    pos = simulate_race(race["mu"].to_numpy(), race["p_dnf"].to_numpy(), s, n_sims, rng)
    return pd.DataFrame(
        {
            "p_win": (pos == 1).mean(axis=0),
            "p_podium": (pos <= 3).mean(axis=0),
            "p_top10": (pos <= 10).mean(axis=0),
            "expected_position": pos.mean(axis=0),
            "p10_position": np.percentile(pos, 10, axis=0),
            "p90_position": np.percentile(pos, 90, axis=0),
        },
        index=race.index,
    )


def simulate_all(df: pd.DataFrame, sigma: float, wet_mult: float, n_sims: int = 10_000) -> pd.DataFrame:
    parts = [race_probabilities(g, sigma, wet_mult, n_sims) for _, g in df.groupby("race_id")]
    return df.join(pd.concat(parts))


def probabilistic_scores(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for event, p_col in [("win", "p_win"), ("podium", "p_podium"), ("top10", "p_top10")]:
        y = df[f"is_{event}"].astype(int)
        p = df[p_col].clip(EPS, 1 - EPS)
        out[f"log_loss_{event}"] = log_loss(y, p)
        out[f"brier_{event}"] = brier_score_loss(y, p)
    out["log_loss_mean"] = np.mean([out["log_loss_win"], out["log_loss_podium"], out["log_loss_top10"]])
    return out


def fit_noise(df: pd.DataFrame, sigmas=None, wet_mults=(1.0, 1.25, 1.5, 2.0), n_sims: int = 3000) -> tuple[float, float, pd.DataFrame]:
    """Grid-search the noise size that makes the simulated probabilities best match reality."""
    sigmas = sigmas if sigmas is not None else np.round(np.arange(0.2, 1.61, 0.1), 2)
    rows = []
    for sigma in sigmas:
        for wm in wet_mults:
            sim = simulate_all(df, sigma, wm, n_sims)
            rows.append({"sigma": sigma, "wet_mult": wm, **probabilistic_scores(sim)})
    grid = pd.DataFrame(rows).sort_values("log_loss_mean").reset_index(drop=True)
    return float(grid.loc[0, "sigma"]), float(grid.loc[0, "wet_mult"]), grid
