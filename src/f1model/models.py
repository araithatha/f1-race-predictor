"""Stage 1: the pace model. Predicts the order the cars would finish in a clean race.

It is trained on cars that saw the chequered flag, ranked among themselves
(``finisher_rank``). Retirements are deliberately left out: whether a car breaks
down is a separate question, handled by the DNF model in ``chaos.py``.

Models are compared with walk-forward validation: for each validation season,
train on every earlier season and predict that season. This mimics how the
model would be used for real and never trains on the future.
"""

from __future__ import annotations

import itertools
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

FIRST_TRAIN_SEASON = 2015  # 2014 is a warm-up year for the rolling-history features
VALIDATION_SEASONS = [2021, 2022, 2023, 2024]
TEST_SEASONS = [2025, 2026]
SEED = 42


# --------------------------------------------------------------------------- #
# Metrics (computed per race, then averaged over races)
# --------------------------------------------------------------------------- #
def race_metrics(df: pd.DataFrame, score_col: str, truth_col: str = "finisher_rank") -> dict[str, float]:
    """Lower score = predicted to finish higher. Rows without a truth value are dropped.

    With the default truth (``finisher_rank``) this scores the clean-race order among
    finishers; ``top_pick_won`` is then conditional on the pick finishing. Headline
    winner numbers use the official result instead (see ``run.py``).
    """
    rows = []
    for _, g in df.dropna(subset=[truth_col]).groupby("race_id"):
        if len(g) < 5:
            continue
        pred_rank = g[score_col].rank(method="first")
        truth = g[truth_col]
        rows.append(
            {
                "spearman": spearmanr(pred_rank, truth).statistic,
                "mae_positions": float(np.abs(pred_rank - truth).mean()),
                "top_pick_won": float(g.loc[pred_rank.idxmin(), truth_col] == 1),
                "podium_overlap": len(set(g.index[pred_rank <= 3]) & set(g.index[truth <= 3])) / 3,
                "top10_overlap": len(set(g.index[pred_rank <= 10]) & set(g.index[truth <= 10])) / 10,
            }
        )
    return pd.DataFrame(rows).mean().to_dict()


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    name: str
    build: Callable[..., object] | None  # None = rule-based baseline
    grid: dict[str, list] = field(default_factory=dict)
    ranker: bool = False
    note: str = ""

    def param_sets(self) -> list[dict]:
        if not self.grid:
            return [{}]
        keys = list(self.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*self.grid.values())]


def _ridge(alpha):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge(alpha=alpha))


def _rf(min_samples_leaf, max_features):
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        RandomForestRegressor(n_estimators=400, min_samples_leaf=min_samples_leaf, max_features=max_features,
                              n_jobs=-1, random_state=SEED),
    )


def _hgb(learning_rate, max_leaf_nodes, max_iter):
    return HistGradientBoostingRegressor(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=SEED)


def _xgb(max_depth, n_estimators, min_child_weight):
    return XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, min_child_weight=min_child_weight,
                        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=SEED, n_jobs=4, verbosity=0)


def _lgbm(num_leaves, n_estimators, min_child_samples):
    return lgb.LGBMRegressor(num_leaves=num_leaves, n_estimators=n_estimators, min_child_samples=min_child_samples,
                             learning_rate=0.03, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                             random_state=SEED, verbose=-1)


def _lgbm_rank(num_leaves, n_estimators, min_child_samples):
    return lgb.LGBMRanker(objective="lambdarank", num_leaves=num_leaves, n_estimators=n_estimators,
                          min_child_samples=min_child_samples, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1)


CANDIDATES = [
    Candidate("Grid order (baseline)", None, note="finish where you start"),
    Candidate("Ridge regression", _ridge, {"alpha": [1.0, 10.0, 100.0], "target": ["rank", "log_rank"]}),
    Candidate("Random forest", _rf, {"min_samples_leaf": [5, 20], "max_features": [0.33, 0.6], "target": ["rank", "log_rank"]}),
    Candidate("Hist gradient boosting", _hgb, {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15], "max_iter": [200], "target": ["rank", "log_rank"]}),
    Candidate("XGBoost", _xgb, {"max_depth": [3, 5], "n_estimators": [300, 600], "min_child_weight": [10], "target": ["rank", "log_rank"]}),
    Candidate("LightGBM", _lgbm, {"num_leaves": [7, 15], "n_estimators": [300, 600], "min_child_samples": [30], "target": ["rank", "log_rank"]}),
    Candidate("LightGBM LambdaRank", _lgbm_rank, {"num_leaves": [7, 15], "n_estimators": [200, 500], "min_child_samples": [30]},
              ranker=True, note="learning-to-rank objective, trained per race"),
]


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #
def fit_model(cand: Candidate, params: dict, train: pd.DataFrame, features: list[str]) -> Callable[[pd.DataFrame], np.ndarray]:
    """Fit a candidate and return a function that scores new rows (lower = faster)."""
    if cand.build is None:
        return lambda X: X["grid_adj"].to_numpy(dtype=float)

    tr = train.dropna(subset=["finisher_rank"]).sort_values(["race_order", "finisher_rank"])
    # A feature with no data in the training seasons (e.g. practice pace before 2023) can't be learned.
    features = [f for f in features if tr[f].notna().any()]
    if cand.ranker:
        model = cand.build(**params)
        # Relevance label: higher is better, so invert the finishing rank within each race.
        relevance = (tr["n_finishers"] - tr["finisher_rank"]).clip(lower=0).astype(int)
        groups = tr.groupby("race_order", sort=False).size().to_numpy()
        model.fit(tr[features], relevance, group=groups)
        return lambda X: -model.predict(X[features])
    params = dict(params)
    target = params.pop("target", "rank")
    model = cand.build(**params)
    # "log_rank" stretches the front of the field: P1 vs P2 counts for more than P15 vs P16.
    y = np.log(tr["finisher_rank"]) if target == "log_rank" else tr["finisher_rank"]
    model.fit(tr[features], y)
    return lambda X: model.predict(X[features])


def fit_predict(cand: Candidate, params: dict, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Return a pace score for every row of ``test`` (lower = faster)."""
    return fit_model(cand, params, train, features)(test)


def walk_forward(df: pd.DataFrame, cand: Candidate, params: dict, features: list[str], seasons: list[int]) -> pd.DataFrame:
    """Out-of-fold predictions: each season predicted by a model trained on all earlier seasons."""
    out = []
    for season in seasons:
        train = df[(df["season"] >= FIRST_TRAIN_SEASON) & (df["season"] < season)]
        test = df[df["season"] == season].copy()
        test["score"] = fit_predict(cand, params, train, test, features)
        out.append(test)
    return pd.concat(out)


def compare_models(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Tune each candidate's hyper-parameters on the validation seasons (by rank correlation).

    Returns the leaderboard, the best params per model and their out-of-fold predictions;
    the final choice between models is made on probability accuracy in ``run.py``.
    """
    rows, best_params, oofs = [], {}, {}
    for cand in CANDIDATES:
        best = None
        for params in cand.param_sets():
            oof = walk_forward(df, cand, params, features, VALIDATION_SEASONS)
            m = race_metrics(oof, "score")
            if best is None or m["spearman"] > best[1]["spearman"]:
                best = (params, m, oof)
        best_params[cand.name] = best[0]
        oofs[cand.name] = best[2]
        rows.append({"model": cand.name, "best_params": best[0], **best[1]})
        log.info("%-24s spearman=%.3f  podium=%.2f  params=%s", cand.name, best[1]["spearman"],
                 best[1]["podium_overlap"], best[0])
    board = pd.DataFrame(rows).sort_values("spearman", ascending=False).reset_index(drop=True)
    return board, best_params, oofs
