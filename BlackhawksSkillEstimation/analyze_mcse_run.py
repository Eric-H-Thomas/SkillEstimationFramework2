"""Aggregate and compare MCSE vs JEEDS Blackhawks runs across xG models.

Reads final-row estimates from every per-player/per-season intermediate CSV
under both data roots for both estimators, then emits tidy CSVs, cross-season
stability statistics, and diagnostic plots.

Usage:
    python BlackhawksSkillEstimation/analyze_mcse_run.py \
        --output-dir mcse_analysis
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from Estimators.joint import hockey_rationality_log10_bounds

SEASONS = [20212022, 20222023, 20232024, 20242025]
SEASON_LABEL = {s: f"{str(s)[:4]}-{str(s)[6:]}" for s in SEASONS}
SHOT_GROUP = "wristshot_snapshot"

REPO_ROOT = Path(__file__).resolve().parent.parent

ROOTS = {
    "legacy": REPO_ROOT / "Data" / "Hockey",
    "new": REPO_ROOT / "Data" / "Hockey_xg_new",
}

# Rationality grid is log10-uniform on the hockey-multi JEEDS/MCSE bounds
# (default [-1, 3] when BH_EV_NORMALIZE is on), so a posterior that has
# learned nothing reports E[lambda] equal to the grid mean.
RATIONALITY_LOG10_MIN, RATIONALITY_LOG10_MAX = hockey_rationality_log10_bounds()
PRIOR_MEAN_LAMBDA = float(
    np.mean(np.power(10.0, np.linspace(RATIONALITY_LOG10_MIN, RATIONALITY_LOG10_MAX, 500)))
)

SEASON_RE = re.compile(r"intermediate_estimates_(\d{8})\.csv$")


@dataclass(frozen=True)
class Record:
    estimator: str
    xg_model: str
    player_id: int
    season: int
    shots: int
    exec_skill: float
    exec_skill_y: float
    exec_skill_z: float
    rho: float
    log10_eps: float
    log10_map_rationality: float
    # Stability-of-the-estimate-within-a-run diagnostics.
    tail_cv_exec: float
    tail_cv_log10_eps: float


def _tail_slice(frame: pd.DataFrame, frac: float = 0.2) -> pd.DataFrame:
    n = len(frame)
    start = max(0, n - max(1, int(round(n * frac))))
    return frame.iloc[start:]


def _cv(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size < 2:
        return math.nan
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return math.nan
    return float(np.std(arr) / abs(mean))


def _iter_csvs(root: Path, logs_subdir: str | None) -> list[tuple[int, int, Path]]:
    players_dir = root / "players"
    if not players_dir.is_dir():
        return []
    found: list[tuple[int, int, Path]] = []
    for player_dir in sorted(players_dir.glob("player_*")):
        try:
            player_id = int(player_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        logs_dir = player_dir / "logs"
        if logs_subdir:
            logs_dir = logs_dir / logs_subdir
        logs_dir = logs_dir / SHOT_GROUP
        if not logs_dir.is_dir():
            continue
        for csv_path in sorted(logs_dir.glob("intermediate_estimates_*.csv")):
            match = SEASON_RE.search(csv_path.name)
            if not match:
                continue
            season = int(match.group(1))
            if season not in SEASONS:
                continue
            found.append((player_id, season, csv_path))
    return found


def load_mcse(root: Path, xg_model: str) -> list[Record]:
    records: list[Record] = []
    for player_id, season, csv_path in _iter_csvs(root, "mcse"):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        last = frame.iloc[-1]
        tail = _tail_slice(frame)
        ees_y = float(last["ees_y"])
        ees_z = float(last["ees_z"])
        records.append(
            Record(
                estimator="mcse",
                xg_model=xg_model,
                player_id=player_id,
                season=season,
                shots=int(last["shot_count"]),
                exec_skill=float(np.sqrt(ees_y * ees_z)),
                exec_skill_y=ees_y,
                exec_skill_z=ees_z,
                rho=float(last["rho_ees"]),
                log10_eps=float(last["log10_expected_rationality"]),
                log10_map_rationality=float(last["log10_map_rationality"]),
                tail_cv_exec=_cv(np.sqrt(tail["ees_y"] * tail["ees_z"])),
                tail_cv_log10_eps=_cv(tail["log10_expected_rationality"]),
            )
        )
    return records


def load_jeeds(root: Path, xg_model: str) -> list[Record]:
    records: list[Record] = []
    for player_id, season, csv_path in _iter_csvs(root, None):
        frame = pd.read_csv(csv_path)
        if frame.empty or "expected_execution_skill" not in frame.columns:
            continue
        last = frame.iloc[-1]
        tail = _tail_slice(frame)
        skill = float(last["expected_execution_skill"])
        records.append(
            Record(
                estimator="jeeds",
                xg_model=xg_model,
                player_id=player_id,
                season=season,
                shots=int(last["shot_count"]),
                exec_skill=skill,
                exec_skill_y=skill,
                exec_skill_z=skill,
                rho=0.0,
                log10_eps=float(last["log10_expected_rationality"]),
                log10_map_rationality=float(last["log10_map_rationality"]),
                tail_cv_exec=_cv(tail["expected_execution_skill"]),
                tail_cv_log10_eps=_cv(tail["log10_expected_rationality"]),
            )
        )
    return records


def build_table() -> pd.DataFrame:
    records: list[Record] = []
    for xg_model, root in ROOTS.items():
        records.extend(load_mcse(root, xg_model))
        records.extend(load_jeeds(root, xg_model))
    frame = pd.DataFrame([r.__dict__ for r in records])
    return frame.sort_values(["estimator", "xg_model", "player_id", "season"]).reset_index(drop=True)


def stability_stats(frame: pd.DataFrame, metric: str, min_shots: int) -> pd.DataFrame:
    rows = []
    subset = frame[frame["shots"] >= min_shots]
    for (estimator, xg_model), group in subset.groupby(["estimator", "xg_model"]):
        pivot = group.pivot_table(index="player_id", columns="season", values=metric)
        for season_a, season_b in zip(SEASONS, SEASONS[1:]):
            if season_a not in pivot.columns or season_b not in pivot.columns:
                continue
            pair = pivot[[season_a, season_b]].dropna()
            if len(pair) < 5:
                continue
            xs = pair[season_a].to_numpy(dtype=float)
            ys = pair[season_b].to_numpy(dtype=float)
            pearson = stats.pearsonr(xs, ys)
            spearman = stats.spearmanr(xs, ys)
            rows.append(
                {
                    "estimator": estimator,
                    "xg_model": xg_model,
                    "metric": metric,
                    "season_a": season_a,
                    "season_b": season_b,
                    "n_players": len(pair),
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_rho": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                }
            )
    return pd.DataFrame(rows)


def variance_decomposition(frame: pd.DataFrame, metric: str, min_shots: int) -> pd.DataFrame:
    """Share of total variance attributable to stable between-player differences.

    This is a one-way ICC(1): players are the grouping factor and seasons are
    repeated measures. High ICC means a player's estimate is reproducible.
    """
    rows = []
    subset = frame[frame["shots"] >= min_shots]
    for (estimator, xg_model), group in subset.groupby(["estimator", "xg_model"]):
        counts = group.groupby("player_id")[metric].count()
        eligible = counts[counts >= 2].index
        data = group[group["player_id"].isin(eligible)]
        if data["player_id"].nunique() < 5:
            continue
        grand_mean = data[metric].mean()
        group_means = data.groupby("player_id")[metric].mean()
        group_sizes = data.groupby("player_id")[metric].count()
        k = group_sizes.mean()
        n_groups = len(group_means)
        n_total = len(data)
        ss_between = float(((group_means - grand_mean) ** 2 * group_sizes).sum())
        ss_within = float(
            sum(
                ((sub[metric] - group_means[pid]) ** 2).sum()
                for pid, sub in data.groupby("player_id")
            )
        )
        df_between = n_groups - 1
        df_within = n_total - n_groups
        if df_between <= 0 or df_within <= 0:
            continue
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        rows.append(
            {
                "estimator": estimator,
                "xg_model": xg_model,
                "metric": metric,
                "n_players": n_groups,
                "n_observations": n_total,
                "icc1": icc,
                "between_player_sd": math.sqrt(max(ms_between, 0.0)),
                "within_player_sd": math.sqrt(max(ms_within, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_stability_scatter(frame: pd.DataFrame, metric: str, out_path: Path, min_shots: int, label: str) -> None:
    combos = [("jeeds", "legacy"), ("jeeds", "new"), ("mcse", "legacy"), ("mcse", "new")]
    pairs = list(zip(SEASONS, SEASONS[1:]))
    fig, axes = plt.subplots(len(combos), len(pairs), figsize=(4 * len(pairs), 3.6 * len(combos)))
    subset = frame[frame["shots"] >= min_shots]
    for row_idx, (estimator, xg_model) in enumerate(combos):
        group = subset[(subset["estimator"] == estimator) & (subset["xg_model"] == xg_model)]
        pivot = group.pivot_table(index="player_id", columns="season", values=metric)
        for col_idx, (season_a, season_b) in enumerate(pairs):
            ax = axes[row_idx][col_idx]
            if season_a not in pivot.columns or season_b not in pivot.columns:
                ax.set_visible(False)
                continue
            pair = pivot[[season_a, season_b]].dropna()
            if pair.empty:
                ax.set_visible(False)
                continue
            xs = pair[season_a].to_numpy(dtype=float)
            ys = pair[season_b].to_numpy(dtype=float)
            ax.scatter(xs, ys, s=14, alpha=0.6, color="#1f77b4")
            lo = float(min(xs.min(), ys.min()))
            hi = float(max(xs.max(), ys.max()))
            ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--")
            # Calculate Spearman rank correlation coefficient. This is how we actually represent stability.
            rho = stats.spearmanr(xs, ys).statistic if len(pair) >= 5 else math.nan
            ax.set_title(
                f"{estimator.upper()} / {xg_model} xG\n"
                f"{SEASON_LABEL[season_a]} vs {SEASON_LABEL[season_b]}  "
                f"(rho={rho:.2f}, n={len(pair)})",
                fontsize=9,
            )
            ax.set_xlabel(SEASON_LABEL[season_a], fontsize=8)
            ax.set_ylabel(SEASON_LABEL[season_b], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Season-to-season stability of {label} (>= {min_shots} shots)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rationality_distributions(frame: pd.DataFrame, out_path: Path, min_shots: int) -> None:
    combos = [("jeeds", "legacy"), ("jeeds", "new"), ("mcse", "legacy"), ("mcse", "new")]
    subset = frame[frame["shots"] >= min_shots]
    fig, axes = plt.subplots(1, len(combos), figsize=(4 * len(combos), 4), sharey=True)
    prior_log10 = math.log10(PRIOR_MEAN_LAMBDA)
    for ax, (estimator, xg_model) in zip(axes, combos):
        values = subset[(subset["estimator"] == estimator) & (subset["xg_model"] == xg_model)][
            "log10_eps"
        ].dropna()
        if values.empty:
            ax.set_visible(False)
            continue
        ax.hist(
            values,
            bins=40,
            range=(RATIONALITY_LOG10_MIN, RATIONALITY_LOG10_MAX),
            color="#4c72b0",
            alpha=0.85,
        )
        ax.axvline(prior_log10, color="red", ls="--", lw=1.4, label=f"prior mean ({prior_log10:.2f})")
        ax.axvline(
            RATIONALITY_LOG10_MAX,
            color="black",
            ls=":",
            lw=1.2,
            label=f"grid cap ({RATIONALITY_LOG10_MAX:g})",
        )
        ax.set_title(
            f"{estimator.upper()} / {xg_model} xG\nmedian={values.median():.2f}  sd={values.std():.2f}",
            fontsize=10,
        )
        ax.set_xlabel("log10 expected rationality")
        ax.set_xlim(RATIONALITY_LOG10_MIN, RATIONALITY_LOG10_MAX)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("player-seasons")
    fig.suptitle(
        f"Where the rationality posterior lands on its "
        f"[{RATIONALITY_LOG10_MIN:g}, {RATIONALITY_LOG10_MAX:g}] log10 grid",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_player_trajectories(frame: pd.DataFrame, metric: str, out_path: Path, min_shots: int, label: str) -> None:
    combos = [("jeeds", "legacy"), ("jeeds", "new"), ("mcse", "legacy"), ("mcse", "new")]
    subset = frame[frame["shots"] >= min_shots]
    fig, axes = plt.subplots(1, len(combos), figsize=(4.2 * len(combos), 4.2), sharey=True)
    for ax, (estimator, xg_model) in zip(axes, combos):
        group = subset[(subset["estimator"] == estimator) & (subset["xg_model"] == xg_model)]
        pivot = group.pivot_table(index="player_id", columns="season", values=metric)
        pivot = pivot.dropna(thresh=3)
        for _, series in pivot.iterrows():
            seasons = [s for s in SEASONS if s in series.index and not pd.isna(series[s])]
            ax.plot(
                [SEASON_LABEL[s] for s in seasons],
                [series[s] for s in seasons],
                color="#1f77b4",
                alpha=0.25,
                lw=1,
                marker="o",
                ms=2.5,
            )
        ax.set_title(f"{estimator.upper()} / {xg_model} xG\n(n={len(pivot)} players)", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    axes[0].set_ylabel(label)
    fig.suptitle(f"Per-player {label} across seasons (players with >= 3 seasons)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_convergence_traces(out_path: Path, n_players: int = 6) -> None:
    """Show raw within-run traces so the reader can see whether estimates settle."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("mcse", "legacy", "expected_rationality", axes[0][0]),
        ("mcse", "new", "expected_rationality", axes[0][1]),
        ("mcse", "legacy", "ees_y", axes[1][0]),
        ("mcse", "new", "ees_y", axes[1][1]),
    ]
    for estimator, xg_model, column, ax in panels:
        root = ROOTS[xg_model]
        csvs = [p for _, season, p in _iter_csvs(root, "mcse") if season == 20242025]
        for csv_path in csvs[:n_players]:
            frame = pd.read_csv(csv_path)
            if column not in frame.columns or frame.empty:
                continue
            ax.plot(frame["shot_count"], frame[column], lw=0.9, alpha=0.8)
        ax.set_title(f"MCSE / {xg_model} xG - {column} (2024-25, {n_players} players)", fontsize=10)
        ax.set_xlabel("shots processed")
        ax.set_ylabel(column)
        if column == "expected_rationality":
            ax.axhline(PRIOR_MEAN_LAMBDA, color="red", ls="--", lw=1.3, label="prior mean E[lambda]")
            ax.set_yscale("log")
            ax.set_ylim(10 ** RATIONALITY_LOG10_MIN, 10 ** RATIONALITY_LOG10_MAX)
            ax.legend(fontsize=8)
    fig.suptitle("MCSE within-run convergence traces", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="mcse_analysis")
    parser.add_argument("--min-shots", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    frame = build_table()
    frame.to_csv(out_dir / "final_estimates_all.csv", index=False)

    coverage = (
        frame.groupby(["estimator", "xg_model", "season"])
        .agg(player_seasons=("player_id", "nunique"), median_shots=("shots", "median"))
        .reset_index()
    )
    coverage.to_csv(out_dir / "coverage.csv", index=False)

    stability_frames = []
    icc_frames = []
    for metric, label in [("exec_skill", "execution skill"), ("log10_eps", "log10 rationality")]:
        stability_frames.append(stability_stats(frame, metric, args.min_shots))
        icc_frames.append(variance_decomposition(frame, metric, args.min_shots))
        plot_stability_scatter(
            frame, metric, out_dir / "plots" / f"stability_scatter_{metric}.png", args.min_shots, label
        )
        plot_player_trajectories(
            frame, metric, out_dir / "plots" / f"player_trajectories_{metric}.png", args.min_shots, label
        )

    stability = pd.concat(stability_frames, ignore_index=True)
    stability.to_csv(out_dir / "season_pair_stability.csv", index=False)
    icc = pd.concat(icc_frames, ignore_index=True)
    icc.to_csv(out_dir / "variance_decomposition.csv", index=False)

    plot_rationality_distributions(frame, out_dir / "plots" / "rationality_distributions.png", args.min_shots)
    plot_convergence_traces(out_dir / "plots" / "mcse_convergence_traces.png")

    summary = (
        frame[frame["shots"] >= args.min_shots]
        .groupby(["estimator", "xg_model"])
        .agg(
            n=("player_id", "size"),
            exec_median=("exec_skill", "median"),
            exec_sd=("exec_skill", "std"),
            log10_eps_median=("log10_eps", "median"),
            log10_eps_sd=("log10_eps", "std"),
            tail_cv_exec_median=("tail_cv_exec", "median"),
            tail_cv_rat_median=("tail_cv_log10_eps", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_by_estimator.csv", index=False)

    pd.set_option("display.width", 200)
    print(
        f"Prior mean E[lambda] on log10-uniform "
        f"[{RATIONALITY_LOG10_MIN:g}, {RATIONALITY_LOG10_MAX:g}] grid = {PRIOR_MEAN_LAMBDA:.1f} "
        f"(log10 = {math.log10(PRIOR_MEAN_LAMBDA):.3f})\n"
    )
    print("=== COVERAGE ===")
    print(coverage.to_string(index=False))
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== SEASON-PAIR STABILITY ===")
    print(stability.to_string(index=False))
    print("\n=== VARIANCE DECOMPOSITION (ICC1) ===")
    print(icc.to_string(index=False))
    print(f"\nWrote outputs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
