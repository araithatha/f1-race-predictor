"""Train, compare, select and evaluate the race prediction system.

    python -m f1model.run

Protocol
--------
* Validation seasons 2021-2024 (walk-forward): every decision is made here —
  hyper-parameters, which pace model, which DNF model, and the size of the
  race-day noise.
* Test seasons 2025-2026 (walk-forward, each season predicted by a model trained
  only on earlier seasons): reported once, after all choices are fixed.

Writes leaderboards, test results and per-driver probabilities to
results/. Charts are made by f1model.figures.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import chaos, features, models
from .models import CANDIDATES, SEED, TEST_SEASONS, VALIDATION_SEASONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
log = logging.getLogger("f1model")


def favourite_and_pole(sim: pd.DataFrame) -> dict[str, float]:
    """Official-result winner hit rates: the model's favourite vs simply backing the pole-sitter."""
    fav = sim.loc[sim.groupby("race_id")["p_win"].idxmax()]
    pole = sim.loc[sim.groupby("race_id")["grid_adj"].idxmin()]
    top3 = sim.sort_values("p_podium", ascending=False).groupby("race_id").head(3)
    grid3 = sim.sort_values("grid_adj").groupby("race_id").head(3)
    return {
        "favourite_won": float(fav["is_win"].mean()),
        "pole_sitter_won": float(pole["is_win"].mean()),
        "favourite_podium": float(fav["is_podium"].mean()),
        "podium_picks_correct": float(top3.groupby("race_id")["is_podium"].mean().mean()),
        "grid_top3_podium_correct": float(grid3.groupby("race_id")["is_podium"].mean().mean()),
    }


def race_level_logloss(sim: pd.DataFrame) -> pd.Series:
    """Mean of win/podium/top-10 log loss for each race (for bootstrapping)."""
    eps = chaos.EPS
    parts = []
    for event in ["win", "podium", "top10"]:
        p = sim[f"p_{event}"].clip(eps, 1 - eps)
        y = sim[f"is_{event}"].astype(int)
        parts.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return pd.concat(parts, axis=1).mean(axis=1).groupby(sim["race_id"]).mean()


def bootstrap_diff(a: pd.Series, b: pd.Series, n: int = 5000, higher_is_better: bool = False) -> dict[str, float]:
    """Paired bootstrap over races of mean(a - b), with a 95% interval."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(SEED)
    boots = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
            "ci95_high": float(np.percentile(boots, 97.5)),
            "share_of_races_better": float((d > 0).mean() if higher_is_better else (d < 0).mean())}


def per_race_spearman(df: pd.DataFrame, score_col: str, truth_col: str) -> pd.Series:
    out = {}
    for rid, g in df.dropna(subset=[truth_col]).groupby("race_id"):
        if len(g) >= 5:
            out[rid] = g[score_col].rank(method="first").corr(g[truth_col], method="spearman")
    return pd.Series(out)


def system_on(df_scored: pd.DataFrame, p_dnf: pd.Series, sigma: float | None = None, wet: float | None = None,
              n_sims: int = 10_000):
    """Attach DNF risk, standardise pace, fit noise if not given, and simulate."""
    v = df_scored.assign(p_dnf=p_dnf.loc[df_scored.index])
    v["mu"] = chaos.standardise_scores(v)
    grid = None
    if sigma is None:
        sigma, wet, grid = chaos.fit_noise(v)
    return chaos.simulate_all(v, sigma, wet, n_sims), sigma, wet, grid


def group_importance(df, cand, params, feats, season: int, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance by feature group: drop in test Spearman when a group is scrambled within each race."""
    rng = np.random.default_rng(SEED)
    train = df[(df["season"] >= models.FIRST_TRAIN_SEASON) & (df["season"] < season)]
    test = df[df["season"] == season].copy()
    predict = models.fit_model(cand, params, train, feats)
    base = models.race_metrics(test.assign(score=predict(test)), "score")["spearman"]
    rows = []
    for group, cols in features.FEATURES.items():
        cols = [c for c in cols if c in feats]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = test.copy()
            for c in cols:
                shuffled[c] = shuffled.groupby("race_id")[c].transform(
                    lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
            drops.append(base - models.race_metrics(shuffled.assign(score=predict(shuffled)), "score")["spearman"])
        rows.append({"feature_group": group, "spearman_drop": float(np.mean(drops)), "n_features": len(cols)})
    return pd.DataFrame(rows).sort_values("spearman_drop", ascending=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    df = features.build_feature_table()
    df.to_csv(OUT / "feature_table.csv", index=False)
    feats = features.CORE_FEATURES
    log.info("feature table: %d driver-races, %d races, %d features", len(df), df["race_id"].nunique(), len(feats))

    # ---- Stage 2a: DNF model (needed to score whole systems) -----------------
    dnf_board, dnf_name = chaos.compare_dnf_models(df, VALIDATION_SEASONS)
    dnf_board.to_csv(OUT / "dnf_model_comparison_validation.csv", index=False)
    dnf_spec = chaos.DNF_CANDIDATES[dnf_name]
    log.info("selected DNF model: %s", dnf_name)
    p_dnf_val = chaos.dnf_walk_forward(df, dnf_spec, VALIDATION_SEASONS)
    base_spec = chaos.DNF_CANDIDATES["Base rate (baseline)"]

    # ---- Stage 1: tune every pace model, then pick the most accurate system ---
    board, best_params, oofs = models.compare_models(df, feats)
    sys_rows = []
    for name, oof in oofs.items():
        p = p_dnf_val if name != "Grid order (baseline)" else chaos.dnf_walk_forward(df, base_spec, VALIDATION_SEASONS)
        sim, s, w, _ = system_on(oof, p, n_sims=4000)
        sys_rows.append({"model": name, "system_log_loss": chaos.probabilistic_scores(sim)["log_loss_mean"],
                         "sigma": s, "wet_mult": w, **favourite_and_pole(sim)})
        log.info("%-24s system log loss=%.4f  favourite won=%.1f%%", name, sys_rows[-1]["system_log_loss"],
                 sys_rows[-1]["favourite_won"] * 100)
    board = board.merge(pd.DataFrame(sys_rows), on="model").sort_values("system_log_loss").reset_index(drop=True)
    board.to_csv(OUT / "model_comparison_validation.csv", index=False)
    best_name = board.loc[0, "model"]
    best = next(c for c in CANDIDATES if c.name == best_name)
    params = best_params[best_name]
    log.info("selected pace model: %s %s", best_name, params)

    # Ablation: drop one feature group at a time (validation seasons, rank correlation).
    base_val = board.loc[0, "spearman"]
    ablation = []
    for group, cols in features.FEATURES.items():
        if not set(cols) & set(feats):
            continue
        kept = [f for f in feats if f not in cols]
        oof = models.walk_forward(df, best, params, kept, VALIDATION_SEASONS)
        ablation.append({"dropped_group": group, "spearman": models.race_metrics(oof, "score")["spearman"]})
    ablation = pd.DataFrame(ablation).assign(change=lambda d: d["spearman"] - base_val).sort_values("change")
    ablation.to_csv(OUT / "ablation_validation.csv", index=False)

    # ---- Stage 2b: fit race-day noise on validation for the final system -----
    val = oofs[best_name]
    val_sim, sigma, wet_mult, noise_grid = system_on(val, p_dnf_val)
    noise_grid.to_csv(OUT / "noise_fit_validation.csv", index=False)
    log.info("fitted noise: sigma=%.2f, wet multiplier=%.2f", sigma, wet_mult)
    # Comparison systems, fitted the same way on validation.
    p_base_val = chaos.dnf_walk_forward(df, base_spec, VALIDATION_SEASONS)
    _, g_sigma, g_wet, _ = system_on(oofs["Grid order (baseline)"], p_base_val)

    # ---- Regulation changes: is the regulation-aware add-on worth adopting? --
    # Decided on validation only, before the test seasons are scored.
    reg_feats = feats + features.REGULATION_AWARE
    reg_val = models.walk_forward(df, best, params, reg_feats, VALIDATION_SEASONS)
    reg_val_sim, r_sigma, r_wet, _ = system_on(reg_val, p_dnf_val)
    adopt = chaos.probabilistic_scores(reg_val_sim)["log_loss_mean"] < chaos.probabilistic_scores(val_sim)["log_loss_mean"]
    log.info("regulation-aware add-on: %s", "ADOPTED (better on validation)" if adopt else "not adopted (no better on validation)")
    core = dict(feats=feats, val=val, val_sim=val_sim, sigma=sigma, wet=wet_mult)
    reg = dict(feats=reg_feats, val=reg_val, val_sim=reg_val_sim, sigma=r_sigma, wet=r_wet)
    final, alt = (reg, core) if adopt else (core, reg)
    feats, val, val_sim, sigma, wet_mult = final["feats"], final["val"], final["val_sim"], final["sigma"], final["wet"]
    alt_feats, alt_sigma, alt_wet = alt["feats"], alt["sigma"], alt["wet"]
    _, n_sigma, n_wet, _ = system_on(val, p_dnf_val * 0)

    # ---- Held-out test: 2025 and 2026 ----------------------------------------
    p_dnf_test = chaos.dnf_walk_forward(df, dnf_spec, TEST_SEASONS)
    p_base_test = chaos.dnf_walk_forward(df, base_spec, TEST_SEASONS)
    test = models.walk_forward(df, best, params, feats, TEST_SEASONS)
    test_sim, *_ = system_on(test, p_dnf_test, sigma, wet_mult)
    grid_test = test.assign(score=test["grid_adj"])
    grid_sim, *_ = system_on(grid_test, p_base_test, g_sigma, g_wet)
    nodnf_sim, *_ = system_on(test, p_dnf_test * 0, n_sigma, n_wet)

    test_board = []
    for cand in CANDIDATES:
        oof = models.walk_forward(df, cand, best_params[cand.name], feats, TEST_SEASONS)
        test_board.append({"model": cand.name, **models.race_metrics(oof, "score")})
    pd.DataFrame(test_board).sort_values("spearman", ascending=False).to_csv(OUT / "model_comparison_test.csv", index=False)

    no_fp = [f for f in feats if f not in features.FEATURES["practice"]]
    test_nofp = models.walk_forward(df, best, params, no_fp, TEST_SEASONS)

    # Test-season predictions from the variant that was NOT adopted, for the before/after scorecard.
    alt_test = models.walk_forward(df, best, params, alt_feats, TEST_SEASONS)
    alt_test_sim, *_ = system_on(alt_test, p_dnf_test, alt_sigma, alt_wet)
    rates_spec = chaos.DNF_CANDIDATES["Logistic regression, last 4 seasons + this season's rates"]
    rates_p_val = chaos.dnf_walk_forward(df, rates_spec, VALIDATION_SEASONS)
    rates_p_test = chaos.dnf_walk_forward(df, rates_spec, TEST_SEASONS)

    def reset_card(core_scored, core_sim, reg_scored, reg_sim, rates_p, mask_season):
        """Metrics on the first races after a rule reset vs the rest of that season."""
        out = {}
        for label, cond in [("first_5_races_after_reset", lambda d: d["races_into_era"] < 5),
                            ("rest_of_season", lambda d: d["races_into_era"] >= 5)]:
            sel = lambda d: d[(d["season"] == mask_season) & cond(d)]
            out[label] = {
                "races": int(sel(core_scored)["race_id"].nunique()),
                "spearman_core": models.race_metrics(sel(core_scored), "score")["spearman"],
                "spearman_with_regulation_features": models.race_metrics(sel(reg_scored), "score")["spearman"],
                "spearman_grid": models.race_metrics(sel(core_scored).assign(score=lambda d: d["grid_adj"]), "score")["spearman"],
                "system_log_loss_core": chaos.probabilistic_scores(sel(core_sim))["log_loss_mean"],
                "system_log_loss_with_regulation_features": chaos.probabilistic_scores(sel(reg_sim))["log_loss_mean"],
                "dnf_predicted_selected_model": float(sel(core_sim)["p_dnf"].mean()),
                "dnf_predicted_with_season_rates": float(rates_p.loc[sel(core_sim).index].mean()),
                "dnf_actual": float(sel(core_sim)["is_dnf"].mean()),
            }
        return out

    core_val, core_val_sim = core["val"], core["val_sim"]
    reg_val_sim_ = reg["val_sim"]
    core_test, core_test_sim = (alt_test, alt_test_sim) if adopt else (test, test_sim)
    reg_test, reg_test_sim = (test, test_sim) if adopt else (alt_test, alt_test_sim)
    regulation = {
        "adopted": bool(adopt),
        "2022_validation": reset_card(core_val, core_val_sim, reg["val"], reg_val_sim_, rates_p_val, 2022),
        "2026_test": reset_card(core_test, core_test_sim, reg_test, reg_test_sim, rates_p_test, 2026),
        "overall_validation": {
            "spearman_core": models.race_metrics(core_val, "score")["spearman"],
            "spearman_with_regulation_features": models.race_metrics(reg["val"], "score")["spearman"],
            "system_log_loss_core": chaos.probabilistic_scores(core_val_sim)["log_loss_mean"],
            "system_log_loss_with_regulation_features": chaos.probabilistic_scores(reg_val_sim_)["log_loss_mean"],
        },
        "overall_test": {
            "spearman_core": models.race_metrics(core_test, "score")["spearman"],
            "spearman_with_regulation_features": models.race_metrics(reg_test, "score")["spearman"],
            "system_log_loss_core": chaos.probabilistic_scores(core_test_sim)["log_loss_mean"],
            "system_log_loss_with_regulation_features": chaos.probabilistic_scores(reg_test_sim)["log_loss_mean"],
            "bootstrap_logloss_regulation_vs_core": bootstrap_diff(race_level_logloss(reg_test_sim), race_level_logloss(core_test_sim)),
        },
    }

    test_sim["official_order_score"] = test_sim["expected_position"]
    summary = {
        "selected_pace_model": best_name,
        "pace_model_params": params,
        "selected_dnf_model": dnf_name,
        "noise": {"sigma": sigma, "wet_multiplier": wet_mult},
        "validation_seasons": VALIDATION_SEASONS,
        "test_seasons": TEST_SEASONS,
        "test_races": int(test["race_id"].nunique()),
        "validation_headline": favourite_and_pole(val_sim),
        "test_headline": favourite_and_pole(test_sim),
        "pace_order_test": {
            "model": models.race_metrics(test, "score"),
            "model_without_practice": models.race_metrics(test_nofp, "score"),
            "grid": models.race_metrics(grid_test, "score"),
            "bootstrap_spearman_gain_vs_grid": bootstrap_diff(
                per_race_spearman(test, "score", "finisher_rank"), per_race_spearman(grid_test, "score", "finisher_rank"),
                higher_is_better=True),
        },
        "official_order_test": {
            "model_expected_position": models.race_metrics(test_sim.assign(_t=test_sim["finish_position"]), "expected_position", "_t"),
            "grid": models.race_metrics(test_sim.assign(_t=test_sim["finish_position"]), "grid_adj", "_t"),
        },
        "probabilistic_test": {
            "full_system": chaos.probabilistic_scores(test_sim),
            "grid_baseline_system": chaos.probabilistic_scores(grid_sim),
            "without_dnf_layer": chaos.probabilistic_scores(nodnf_sim),
            "bootstrap_logloss_vs_grid": bootstrap_diff(race_level_logloss(test_sim), race_level_logloss(grid_sim)),
            "bootstrap_logloss_vs_no_dnf_layer": bootstrap_diff(race_level_logloss(test_sim), race_level_logloss(nodnf_sim)),
        },
        "by_season_test": {
            int(s): {
                "races": int(g["race_id"].nunique()),
                "spearman": models.race_metrics(test[test["season"] == s], "score")["spearman"],
                "grid_spearman": models.race_metrics(grid_test[grid_test["season"] == s], "score")["spearman"],
                **favourite_and_pole(g),
                "system_log_loss": chaos.probabilistic_scores(g)["log_loss_mean"],
                "grid_system_log_loss": chaos.probabilistic_scores(grid_sim[grid_sim["season"] == s])["log_loss_mean"],
            }
            for s, g in test_sim.groupby("season")
        },
        "regulation_changes": regulation,
        "dnf_test": {
            "mean_predicted": float(test_sim["p_dnf"].mean()),
            "actual_rate": float(test_sim["is_dnf"].mean()),
        },
    }

    calib = []
    for event in ["win", "podium", "top10"]:
        bins = pd.cut(test_sim[f"p_{event}"], [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0], include_lowest=True)
        c = test_sim.groupby(bins, observed=True).agg(predicted=(f"p_{event}", "mean"), observed=(f"is_{event}", "mean"),
                                                       n=(f"is_{event}", "size")).reset_index()
        calib.append(c.assign(event=event, bin=c[f"p_{event}"].astype(str)).drop(columns=f"p_{event}"))
    pd.concat(calib).to_csv(OUT / "calibration_test.csv", index=False)

    group_importance(df, best, params, feats, season=2025).to_csv(OUT / "feature_group_importance_test2025.csv", index=False)

    cols = ["race_id", "season", "round", "race_name", "driver_id", "team_id", "grid_adj", "finish_position", "status",
            "p_dnf", "p_win", "p_podium", "p_top10", "expected_position", "p10_position", "p90_position"]
    test_sim[cols].sort_values(["race_id", "expected_position"]).to_csv(OUT / "test_predictions.csv", index=False)

    summary["runtime_s"] = round(time.monotonic() - started, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("done in %.0fs", summary["runtime_s"])
    print(json.dumps({k: summary[k] for k in ["selected_pace_model", "pace_model_params", "selected_dnf_model",
                                               "validation_headline", "test_headline", "regulation_changes"]},
                     indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
