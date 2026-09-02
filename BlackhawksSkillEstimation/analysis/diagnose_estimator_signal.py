"""Confound and signal-content checks for Blackhawks JEEDS / MCSE estimates.

Answers three questions the raw stability numbers cannot:

1. Is apparent season-to-season stability just shot-volume leakage? (partial
   correlation of the estimate controlling for shot count)
2. Do JEEDS and MCSE agree with each other on the same data?
3. Has the estimate converged internally? (estimate at half the shots vs the
   final estimate, within the same run)

Usage:
    python -m BlackhawksSkillEstimation.analysis.diagnose_estimator_signal
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from BlackhawksSkillEstimation.analysis.analyze_mcse_run import (
    DATA_ROOT,
    ESTIMATORS,
    SEASONS,
    _iter_csvs,
)


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Pearson correlation of x and y after linearly removing z from both."""
    def residual(v: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones_like(z), z])
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        return v - design @ beta

    rx, ry = residual(x), residual(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def volume_leakage(frame: pd.DataFrame, min_shots: int) -> pd.DataFrame:
    rows = []
    subset = frame[frame["shots"] >= min_shots]
    for metric in ["exec_skill", "log10_eps"]:
        for estimator in ESTIMATORS:
            group = subset[subset["estimator"] == estimator]
            if group.empty:
                continue
            corr_with_shots = stats.spearmanr(group[metric], group["shots"]).statistic
            pivot_metric = group.pivot_table(index="player_id", columns="season", values=metric)
            pivot_shots = group.pivot_table(index="player_id", columns="season", values="shots")
            raw_vals, partial_vals, ns = [], [], []
            for season_a, season_b in zip(SEASONS, SEASONS[1:]):
                if season_a not in pivot_metric.columns or season_b not in pivot_metric.columns:
                    continue
                joined = pd.concat(
                    [
                        pivot_metric[[season_a, season_b]].rename(columns={season_a: "a", season_b: "b"}),
                        pivot_shots[[season_a, season_b]].rename(columns={season_a: "sa", season_b: "sb"}),
                    ],
                    axis=1,
                ).dropna()
                if len(joined) < 10:
                    continue
                a = joined["a"].to_numpy(float)
                b = joined["b"].to_numpy(float)
                shots_mean = ((joined["sa"] + joined["sb"]) / 2).to_numpy(float)
                raw_vals.append(float(np.corrcoef(a, b)[0, 1]))
                partial_vals.append(partial_corr(a, b, shots_mean))
                ns.append(len(joined))
            if not raw_vals:
                continue
            rows.append(
                {
                    "metric": metric,
                    "estimator": estimator,
                    "spearman_metric_vs_shots": corr_with_shots,
                    "mean_raw_pearson": float(np.mean(raw_vals)),
                    "mean_partial_pearson_given_shots": float(np.nanmean(partial_vals)),
                    "n_pairs_used": int(np.sum(ns)),
                }
            )
    return pd.DataFrame(rows)


def cross_estimator_agreement(frame: pd.DataFrame, min_shots: int) -> pd.DataFrame:
    """JEEDS vs MCSE, same player-season."""
    rows = []
    subset = frame[frame["shots"] >= min_shots]
    for metric in ["exec_skill", "log10_eps"]:
        pivot = subset.pivot_table(index=["player_id", "season"], columns="estimator", values=metric)
        pair = pivot.dropna()
        if len(pair) < 10:
            continue
        rows.append(
            {
                "comparison": "jeeds_vs_mcse",
                "metric": metric,
                "n": len(pair),
                "pearson_r": float(stats.pearsonr(pair["jeeds"], pair["mcse"]).statistic),
                "spearman_rho": float(stats.spearmanr(pair["jeeds"], pair["mcse"]).statistic),
            }
        )
    return pd.DataFrame(rows)


def internal_convergence(min_shots: int) -> pd.DataFrame:
    """Correlate the estimate at 50% of shots with the estimate at 100% of shots.

    A converged estimator should show near-unity correlation: the last half of
    the data should barely move the answer. A random walk shows a much lower
    value, and the gap is a direct read on how much of the final number is
    accumulated evidence versus wherever the walk happened to end.
    """
    rows = []
    specs = [
        ("jeeds", None, "expected_execution_skill", "log10_expected_rationality"),
        ("mcse", "mcse", None, "log10_expected_rationality"),
    ]
    for estimator, logs_subdir, exec_col, rat_col in specs:
        half_exec, final_exec, half_rat, final_rat = [], [], [], []
        for _player_id, _season, csv_path in _iter_csvs(DATA_ROOT, logs_subdir):
            data = pd.read_csv(csv_path)
            if len(data) < min_shots:
                continue
            mid = len(data) // 2
            if estimator == "mcse":
                exec_series = np.sqrt(data["ees_y"] * data["ees_z"])
            else:
                exec_series = data[exec_col]
            half_exec.append(float(exec_series.iloc[mid]))
            final_exec.append(float(exec_series.iloc[-1]))
            half_rat.append(float(data[rat_col].iloc[mid]))
            final_rat.append(float(data[rat_col].iloc[-1]))
        if len(half_exec) < 10:
            continue
        rows.append(
            {
                "estimator": estimator,
                "n_runs": len(half_exec),
                "exec_half_vs_final_r": float(np.corrcoef(half_exec, final_exec)[0, 1]),
                "rationality_half_vs_final_r": float(np.corrcoef(half_rat, final_rat)[0, 1]),
            }
        )
    return pd.DataFrame(rows)


def mcse_axis_report(frame: pd.DataFrame, out_path: Path, min_shots: int) -> pd.DataFrame:
    subset = frame[(frame["estimator"] == "mcse") & (frame["shots"] >= min_shots)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(subset["exec_skill_y"], subset["exec_skill_z"], s=10, alpha=0.5, color="#1f77b4")
    axes[1].hist(subset["rho"].dropna(), bins=40, alpha=0.85, color="#1f77b4")
    axes[0].plot([0.0, 0.25], [0.0, 0.25], ls="--", color="grey", lw=1)
    axes[0].set_xlabel("ees_y (horizontal aim sd, rad)")
    axes[0].set_ylabel("ees_z (vertical aim sd, rad)")
    axes[0].set_title("MCSE: are the two aim axes distinguished?")
    axes[1].axvline(0.0, color="black", ls=":", lw=1)
    axes[1].set_xlabel("rho_ees")
    axes[1].set_ylabel("player-seasons")
    axes[1].set_title("MCSE: estimated aim-error correlation")

    jeeds = frame[(frame["estimator"] == "jeeds") & (frame["shots"] >= min_shots)]
    axes[2].hist(jeeds["exec_skill"].dropna(), bins=40, alpha=0.85, color="#1f77b4")
    axes[2].axvline(0.004, color="black", ls=":", lw=1)
    axes[2].axvline(0.25, color="black", ls=":", lw=1, label="grid bounds")
    axes[2].set_xlabel("expected execution skill (rad)")
    axes[2].set_title("JEEDS execution skill vs grid bounds")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    if subset.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n": int(len(subset)),
                "median_ees_y": float(subset["exec_skill_y"].median()),
                "median_ees_z": float(subset["exec_skill_z"].median()),
                "corr_y_z": float(np.corrcoef(subset["exec_skill_y"], subset["exec_skill_z"])[0, 1]),
                "median_rho": float(subset["rho"].median()),
                "sd_rho": float(subset["rho"].std()),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="mcse_analysis")
    parser.add_argument("--min-shots", type=int, default=100)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    frame = pd.read_csv(in_dir / "final_estimates_all.csv")

    pd.set_option("display.width", 220)

    leakage = volume_leakage(frame, args.min_shots)
    leakage.to_csv(in_dir / "shot_volume_leakage.csv", index=False)
    print("=== SHOT-VOLUME LEAKAGE (does stability survive controlling for shot count?) ===")
    print(leakage.to_string(index=False))

    cross_est = cross_estimator_agreement(frame, args.min_shots)
    cross_est.to_csv(in_dir / "cross_estimator_agreement.csv", index=False)
    print("\n=== JEEDS vs MCSE ON THE SAME DATA ===")
    print(cross_est.to_string(index=False))

    convergence = internal_convergence(args.min_shots)
    convergence.to_csv(in_dir / "internal_convergence.csv", index=False)
    print("\n=== INTERNAL CONVERGENCE (estimate at 50% of shots vs final) ===")
    print(convergence.to_string(index=False))

    axis_report = mcse_axis_report(frame, in_dir / "plots" / "mcse_axis_and_bounds.png", args.min_shots)
    axis_report.to_csv(in_dir / "mcse_axis_report.csv", index=False)
    print("\n=== MCSE AXIS / RHO REPORT ===")
    print(axis_report.to_string(index=False))


if __name__ == "__main__":
    main()
