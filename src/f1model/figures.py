"""Charts for the README (static PNGs that render on GitHub).

    python -m f1model.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#c9c8c2"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left",
})


def _hbar(ax, labels, values, highlight=None, fmt="{:.3f}"):
    colors = [BLUE if (highlight is None or lab == highlight) else MUTED for lab in labels]
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + max(values) * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def model_comparison(summary: dict):
    val = pd.read_csv(RESULTS / "model_comparison_validation.csv")
    grid_ll = val.loc[val["model"] == "Grid order (baseline)", "system_log_loss"].iloc[0]
    val = val[val["model"] != "Grid order (baseline)"].copy()
    val["gain_pct"] = (grid_ll - val["system_log_loss"]) / grid_ll * 100
    val = val.sort_values("gain_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _hbar(ax, val["model"], val["gain_pct"], highlight=summary["selected_pace_model"], fmt="{:.1f}%")
    ax.set_xlim(0, val["gain_pct"].max() * 1.18)
    ax.set_xlabel("Improvement in probability accuracy over the grid-order baseline (log loss, %)")
    ax.set_title("Seven models compared on 2021–2024 (walk-forward validation)")
    fig.text(0.01, 0.01, "Each season is predicted by a model trained only on earlier seasons. Selected model in blue.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "model_comparison.png", dpi=160)
    plt.close(fig)


def feature_importance():
    imp = pd.read_csv(RESULTS / "feature_group_importance_test2025.csv")
    names = {
        "weekend": "Grid & qualifying", "practice": "Practice pace", "driver_form": "Driver form",
        "team_form": "Team form", "team_strategy": "Team strategy & pit crew", "reliability": "Reliability history",
        "circuit": "Circuit history", "weather": "Weather", "context": "Season context",
    }
    imp["label"] = imp["feature_group"].map(names)
    imp = imp.sort_values("spearman_drop", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    _hbar(ax, imp["label"], imp["spearman_drop"].clip(lower=0), fmt="{:.3f}")
    ax.set_xlabel("Drop in rank correlation when the group is scrambled (2025 season)")
    ax.set_title("What drives the pace prediction")
    fig.text(0.01, 0.01, "Weather and circuit history act through the race simulation and DNF model, not the pace order.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "feature_importance.png", dpi=160)
    plt.close(fig)


def calibration():
    cal = pd.read_csv(RESULTS / "calibration_test.csv")
    titles = {"win": "Win", "podium": "Podium", "top10": "Top 10 (points)"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, event in zip(axes, ["win", "podium", "top10"]):
        c = cal[cal["event"] == event]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", zorder=1)
        ax.plot(c["predicted"], c["observed"], color=BLUE, linewidth=2, zorder=2)
        ax.scatter(c["predicted"], c["observed"], s=[max(18, n / 3) for n in c["n"]], color=BLUE,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_title(titles[event], fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("How often it happened")
    axes[0].text(0.62, 0.52, "perfect\ncalibration", color=INK_2, fontsize=8, rotation=0)
    fig.suptitle("Are the probabilities honest? Held-out 2025–2026 races", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.text(0.01, 0.01, "Dot size = number of driver-races in the bin. On the dashed line, a 30% chance happens 30% of the time.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIGS / "calibration.png", dpi=160)
    plt.close(fig)


def race_example(race_id: str | None = None):
    preds = pd.read_csv(RESULTS / "test_predictions.csv")
    race_id = race_id or preds["race_id"].max()
    r = preds[preds["race_id"] == race_id].sort_values("p_podium", ascending=False).head(10)
    from .features import PROCESSED
    drivers = pd.read_csv(PROCESSED / "dim_driver.csv").set_index("driver_id")["last_name"]
    labels = [f"{drivers.get(d, d)}  (P{int(g)} grid)" for d, g in zip(r["driver_id"], r["grid_adj"])]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(labels, r["p_podium"] * 100, color=BLUE, height=0.6)
    for b, pw, fin, status in zip(bars, r["p_win"], r["finish_position"], r["status"]):
        actual = f"finished P{int(fin)}" if status in ("Finished", "Lapped") or str(status).startswith("+") else "DNF"
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{b.get_width():.0f}%  ·  win {pw*100:.0f}%  ·  {actual}",
                va="center", fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Chance of a podium finish (%)")
    ax.set_title(f"{r['race_name'].iloc[0]} {r['season'].iloc[0]}")
    fig.text(0.01, 0.01, "Probabilities made before the race from 10,000 simulations; result shown after each bar.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGS / "race_example.png", dpi=160)
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "summary.json").read_text())
    model_comparison(summary)
    feature_importance()
    calibration()
    race_example()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
