"""How many synthetic shots it takes to move log-lambda off a stated prior.

Situations are sampled with replacement from one donor's cached xG maps.
Each seed reuses that situation sequence for every true ``(sigma, lambda)``.
The uniform prior is flat on the HJEEDS grid. Each Normal prior is independent
in ``(log sigma, log lambda)`` and recentered on the requested lambda.

Pilot, from the cluster repo root::

    sbatch run_hockey_rationality_sample_size.sbatch

Locally, the same entry point is::

    conda run -n skill-estimation python -m HJEEDS.run_hockey_rationality_sample_size \\
        --player-id 950160 --num-shots 200 --num-seeds 10
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from BlackhawksSkillEstimation.player_subsample_stability import DEFAULT_DATA_ROOT, DEFAULT_PLAYER_ID
from HJEEDS.estimation import build_discrete_hierarchical_prior
from HJEEDS.hockey_pipeline import (
    DEFAULT_HOCKEY_NUM_LOG_LAMBDA_GRID,
    DEFAULT_HOCKEY_NUM_SIGMA_GRID,
    HOCKEY_PRELIMINARY_HYPERPRIORS,
    HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA,
    HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA,
    fit_hockey_population_hyperparameters,
    make_hockey_log_lambda_grid,
    make_hockey_sigma_grid,
)
from HJEEDS.hockey_synthetic_player import (
    departure_flags,
    evaluate_prefixes,
    execution_rng,
    load_donor_situations,
    normal_skill_prior,
    prefix_log_likelihoods,
    prior_log_lambda_marginal,
    simulate_truth_sequence,
    situation_rng,
    summarize_posterior,
    uniform_skill_prior,
    draw_situation_indices,
)


DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "experiments" / "rationality_sample_size"
DEFAULT_PREFIXES = (10, 25, 50, 100, 200, 400)
DEFAULT_LAMBDAS = (1.0, 10.0, 100.0)
DEFAULT_SIGMA = 0.075
DEFAULT_BASE_SEED = 20261023
DEFAULT_NUM_SHOTS = 200
DEFAULT_NUM_SEEDS = 10
HJEEDS_PRIOR_NAME = "hjeeds_population"

CSV_FIELDS = [
    "player_id",
    "shot_group",
    "donor_pool_size",
    "seed",
    "true_sigma",
    "true_lambda",
    "true_log_lambda",
    "prior_name",
    "prior_lambda_center",
    "prior_mean_log_lambda",
    "prior_sd_log_lambda",
    "n_shots",
    "num_unique_situations",
    "posterior_mean_sigma",
    "posterior_mean_log_lambda",
    "posterior_sd_log_lambda",
    "ci_low_log_lambda",
    "ci_high_log_lambda",
    "shifted",
    "excluded_prior_mean",
    "recovered",
    "status",
]

FLAG_NAMES = ("shifted", "excluded_prior_mean", "recovered")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_prefixes(num_shots: int, requested: list[int] | None) -> list[int]:
    if requested:
        prefixes = sorted({int(value) for value in requested})
    else:
        prefixes = [value for value in DEFAULT_PREFIXES if value <= num_shots]
        if num_shots not in prefixes:
            prefixes.append(num_shots)
    if not prefixes:
        raise ValueError("At least one prefix length is required.")
    if prefixes[0] < 1 or prefixes[-1] > num_shots:
        raise ValueError(f"Prefix lengths must lie in 1..{num_shots}. Received {prefixes}.")
    return prefixes


def build_priors(
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    normal_lambda_centers: list[float],
    *,
    normal_sigma_center: float,
    normal_log_sigma_sd: float,
    normal_log_lambda_sd: float,
) -> dict[str, np.ndarray]:
    priors: dict[str, np.ndarray] = {
        "uniform": uniform_skill_prior(sigma_grid, log_lambda_grid),
    }
    for center in normal_lambda_centers:
        name = f"normal_lambda_{center:g}"
        if name in priors:
            raise ValueError(f"Repeated Normal prior name {name}.")
        priors[name] = normal_skill_prior(
            sigma_grid,
            log_lambda_grid,
            mean_log_sigma=float(np.log(normal_sigma_center)),
            mean_log_lambda=float(np.log(center)),
            sd_log_sigma=normal_log_sigma_sd,
            sd_log_lambda=normal_log_lambda_sd,
        )
    return priors


def prior_lambda_center(prior_name: str) -> str:
    """Requested Normal center, or blank for the uniform grid prior."""

    if prior_name == "uniform":
        return ""
    prefix = "normal_lambda_"
    if not prior_name.startswith(prefix):
        raise ValueError(f"Unrecognized prior name {prior_name}.")
    return f"{float(prior_name[len(prefix) :]):g}"


def run_experiment(args: argparse.Namespace) -> tuple[list[dict[str, object]], tuple[int, ...]]:
    donor = load_donor_situations(
        args.player_id,
        seasons=args.seasons,
        data_dir=args.data_dir,
        shot_group=args.shot_group,
        max_situations=args.max_situations,
    )
    situations = donor.situations
    prefixes = resolve_prefixes(args.num_shots, args.prefix_ns)
    priors = build_priors(
        args.sigma_grid,
        args.log_lambda_grid,
        args.normal_lambda_centers,
        normal_sigma_center=args.normal_sigma_center,
        normal_log_sigma_sd=args.normal_log_sigma_sd,
        normal_log_lambda_sd=args.normal_log_lambda_sd,
    )
    print(
        f"player {args.player_id}: {len(situations)} usable situations "
        f"from {donor.rows_in} {args.shot_group} rows "
        f"in seasons {list(donor.seasons)}"
    )
    print(
        f"sampling {args.num_shots} shots with replacement, "
        f"seeds 0..{args.num_seeds - 1}, prefixes {prefixes}"
    )

    rows: list[dict[str, object]] = []
    for seed in range(args.num_seeds):
        indices = draw_situation_indices(len(situations), args.num_shots, situation_rng(args.base_seed, seed))
        for sigma_index, true_sigma in enumerate(args.sigmas):
            for lambda_index, true_lambda in enumerate(args.lambdas):
                print(
                    f"seed {seed} sigma {true_sigma:g} lambda {true_lambda:g}",
                    flush=True,
                )
                per_shot = simulate_truth_sequence(
                    situations,
                    indices,
                    sigma_true=float(true_sigma),
                    lambda_true=float(true_lambda),
                    sigma_grid=args.sigma_grid,
                    log_lambda_grid=args.log_lambda_grid,
                    rng=execution_rng(args.base_seed, seed, sigma_index, lambda_index),
                )
                evaluated = evaluate_prefixes(
                    per_shot,
                    indices,
                    prefixes,
                    priors,
                    args.sigma_grid,
                    args.log_lambda_grid,
                    true_sigma=float(true_sigma),
                    true_lambda=float(true_lambda),
                )
                if args.save_prefix_grids:
                    _save_prefix_grids(
                        args.output_dir,
                        player_id=int(args.player_id),
                        seed=int(seed),
                        true_sigma=float(true_sigma),
                        true_lambda=float(true_lambda),
                        per_shot=per_shot,
                        prefixes=prefixes,
                        sigma_grid=args.sigma_grid,
                        log_lambda_grid=args.log_lambda_grid,
                    )
                for row in evaluated:
                    row["player_id"] = int(args.player_id)
                    row["shot_group"] = args.shot_group
                    row["donor_pool_size"] = len(situations)
                    row["seed"] = int(seed)
                    row["prior_lambda_center"] = prior_lambda_center(str(row["prior_name"]))
                rows.extend(evaluated)
    return rows, donor.seasons


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            for flag in FLAG_NAMES:
                encoded[flag] = int(bool(row[flag]))
            writer.writerow(encoded)


def majority_crossings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Smallest prefix at which a flag holds for more than half of the seeds."""

    counts: dict[tuple[float, float, str, int], int] = {}
    flag_hits: dict[str, dict[tuple[float, float, str, int], int]] = {flag: {} for flag in FLAG_NAMES}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (
            float(row["true_sigma"]),
            float(row["true_lambda"]),
            str(row["prior_name"]),
            int(row["n_shots"]),
        )
        counts[key] = counts.get(key, 0) + 1
        for flag in FLAG_NAMES:
            flag_hits[flag][key] = flag_hits[flag].get(key, 0) + int(bool(row[flag]))

    conditions = sorted({key[:3] for key in counts})
    summary: list[dict[str, object]] = []
    for sigma, true_lambda, prior_name in conditions:
        record: dict[str, object] = {
            "true_sigma": sigma,
            "true_lambda": true_lambda,
            "prior_name": prior_name,
        }
        lengths = sorted(key[3] for key in counts if key[:3] == (sigma, true_lambda, prior_name))
        for flag in FLAG_NAMES:
            crossing = None
            for length in lengths:
                key = (sigma, true_lambda, prior_name, length)
                if flag_hits[flag].get(key, 0) / counts[key] > 0.5:
                    crossing = length
                    break
            record[flag] = crossing
        summary.append(record)
    return summary


def print_summary(summary: list[dict[str, object]]) -> None:
    print("Smallest n where a majority of seeds meet each flag (blank = not within the simulated prefixes):")
    header = f"{'sigma':>8} {'lambda':>8} {'prior':<22} {'shifted':>8} {'excluded':>8} {'recovered':>8}"
    print(header)
    for record in summary:
        def cell(flag: str) -> str:
            value = record[flag]
            return "" if value is None else str(value)

        print(
            f"{record['true_sigma']:8.3f} {record['true_lambda']:8.3g} "
            f"{str(record['prior_name']):<22} {cell('shifted'):>8} {cell('excluded_prior_mean'):>8} "
            f"{cell('recovered'):>8}"
        )


def plot_rationality_curves(rows: list[dict[str, object]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        return
    sigmas = sorted({float(row["true_sigma"]) for row in ok_rows})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for sigma_index, sigma in enumerate(sigmas):
        subset = [row for row in ok_rows if float(row["true_sigma"]) == sigma]
        lambdas = sorted({float(row["true_lambda"]) for row in subset})
        priors = list(dict.fromkeys(str(row["prior_name"]) for row in subset))
        fig, axes = plt.subplots(
            1,
            len(lambdas),
            figsize=(4.4 * len(lambdas), 4.4),
            sharey=True,
            squeeze=False,
        )
        for panel, true_lambda in enumerate(lambdas):
            axis = axes[0, panel]
            for prior_index, prior_name in enumerate(priors):
                color = f"C{prior_index}"
                matched = [
                    row
                    for row in subset
                    if float(row["true_lambda"]) == true_lambda and str(row["prior_name"]) == prior_name
                ]
                lengths = sorted({int(row["n_shots"]) for row in matched})
                seeds = sorted({int(row["seed"]) for row in matched})
                for seed in seeds:
                    seed_rows = [row for row in matched if int(row["seed"]) == seed]
                    seed_rows.sort(key=lambda row: int(row["n_shots"]))
                    axis.plot(
                        [int(row["n_shots"]) for row in seed_rows],
                        [float(row["posterior_mean_log_lambda"]) for row in seed_rows],
                        color=color,
                        alpha=0.25,
                        linewidth=1.0,
                    )
                means = []
                lows = []
                highs = []
                for length in lengths:
                    at_n = [row for row in matched if int(row["n_shots"]) == length]
                    means.append(float(np.mean([row["posterior_mean_log_lambda"] for row in at_n])))
                    lows.append(float(np.mean([row["ci_low_log_lambda"] for row in at_n])))
                    highs.append(float(np.mean([row["ci_high_log_lambda"] for row in at_n])))
                axis.fill_between(lengths, lows, highs, color=color, alpha=0.15)
                axis.plot(lengths, means, color=color, linewidth=2.0, label=prior_name)
                prior_mean = float(np.mean([row["prior_mean_log_lambda"] for row in matched]))
                axis.axhline(prior_mean, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
            axis.axhline(float(np.log(true_lambda)), color="black", linestyle=":", linewidth=1.2, label="truth")
            axis.set_title(f"λ* = {true_lambda:g}")
            axis.set_xlabel("shots")
            if panel == 0:
                axis.set_ylabel("log λ")
            axis.legend(fontsize=8)
        fig.suptitle(f"Synthetic shooter, σ* = {sigma:g}. Dashed lines are prior means.")
        fig.tight_layout()
        figure_path = output_path if len(sigmas) == 1 else output_path.with_name(
            f"{output_path.stem}_sigma_{sigma:g}{output_path.suffix}"
        )
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        if sigma_index == 0 or len(sigmas) > 1:
            print(f"wrote {figure_path}")


STORY_PRIOR_LABELS = {
    "uniform": "No starting belief (JEEDS)",
    "normal_lambda_10": "Start at a typical player",
    HJEEDS_PRIOR_NAME: "Fitted roster (HJEEDS)",
}
STORY_TRUTH_LABELS = {
    1.0: "Sloppy aim",
    10.0: "Typical aim",
    100.0: "Sharp aim",
}
STORY_TRUTH_COLORS = {
    1.0: "#0072B2",
    10.0: "#666666",
    100.0: "#D55E00",
}
_STORY_PRIOR_COLORS = {
    "uniform": "#0072B2",
    "normal_lambda_10": "#D55E00",
}


def _lambda_tick_labels() -> tuple[list[float], list[str]]:
    ticks = [0.1, 1.0, 10.0, 100.0, 1000.0]
    return [float(np.log(tick)) for tick in ticks], [f"{tick:g}" for tick in ticks]


def plot_rationality_story(csv_path: Path, output_dir: Path | None = None) -> list[Path]:
    """Meeting plots from a finished sample-size CSV.

    Two figures. The first is the posterior mean of log λ, drawn on a λ axis.
    The second is how often the 95% interval excludes the typical value λ = 10.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(csv_path)
    frame = frame[frame["status"] == "ok"].copy()
    if frame.empty:
        raise ValueError(f"No successful rows in {csv_path}.")

    destination = output_dir or Path(csv_path).parent
    destination.mkdir(parents=True, exist_ok=True)
    tick_positions, tick_labels = _lambda_tick_labels()
    typical_log_lambda = float(np.log(10.0))
    priors = [name for name in STORY_PRIOR_LABELS if name in set(frame["prior_name"])]
    truths = sorted(float(value) for value in frame["true_lambda"].unique())

    estimate_path = destination / "story_estimate_vs_shots.png"
    figure, axes = plt.subplots(1, len(truths), figsize=(4.3 * len(truths), 4.6), sharey=True, squeeze=False)
    for panel, true_lambda in enumerate(truths):
        axis = axes[0, panel]
        truth_label = STORY_TRUTH_LABELS.get(true_lambda, f"True λ = {true_lambda:g}")
        axis.set_title(f"{truth_label}\ntrue λ = {true_lambda:g}")
        truth_is_typical = abs(np.log(true_lambda) - typical_log_lambda) <= 0.05
        axis.axhline(
            typical_log_lambda,
            color="#444444",
            linestyle="--" if not truth_is_typical else ":",
            linewidth=1.2,
            label="Truth, and a typical player" if truth_is_typical else "Typical λ = 10",
        )
        if not truth_is_typical:
            axis.axhline(float(np.log(true_lambda)), color="black", linestyle=":", linewidth=1.3, label="Truth")
        for prior_name in priors:
            matched = frame[(frame["true_lambda"] == true_lambda) & (frame["prior_name"] == prior_name)]
            lengths = sorted(int(value) for value in matched["n_shots"].unique())
            means, lows, highs = [], [], []
            for length in lengths:
                at_n = matched[matched["n_shots"] == length]
                means.append(float(at_n["posterior_mean_log_lambda"].mean()))
                lows.append(float(at_n["ci_low_log_lambda"].mean()))
                highs.append(float(at_n["ci_high_log_lambda"].mean()))
            color = _STORY_PRIOR_COLORS.get(prior_name, "C0")
            axis.fill_between(lengths, lows, highs, color=color, alpha=0.15)
            axis.plot(
                lengths,
                means,
                color=color,
                linewidth=2.2,
                marker="o",
                label=STORY_PRIOR_LABELS.get(prior_name, prior_name),
            )
        axis.set_xlabel("Shots")
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)
        if panel == 0:
            axis.set_ylabel("Estimated aiming sharpness, λ")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Simulated aim on one forward's real shot situations", fontsize=13)
    figure.tight_layout()
    figure.savefig(estimate_path, dpi=160)
    plt.close(figure)

    decision_path = destination / "story_calls_not_typical.png"
    figure, axes = plt.subplots(1, len(priors), figsize=(5.0 * max(len(priors), 1), 4.6), sharey=True, squeeze=False)
    for panel, prior_name in enumerate(priors):
        axis = axes[0, panel]
        axis.set_title(STORY_PRIOR_LABELS.get(prior_name, prior_name))
        axis.axhline(
            50,
            color="#bbbbbb",
            linestyle="--",
            linewidth=1.0,
            label="Half of repeats" if panel == 0 else None,
        )
        for true_lambda in truths:
            matched = frame[(frame["true_lambda"] == true_lambda) & (frame["prior_name"] == prior_name)]
            lengths = sorted(int(value) for value in matched["n_shots"].unique())
            percents = []
            for length in lengths:
                at_n = matched[matched["n_shots"] == length]
                percents.append(100.0 * float(at_n["excluded_prior_mean"].mean()))
            axis.plot(
                lengths,
                percents,
                color=STORY_TRUTH_COLORS.get(true_lambda, "C0"),
                linewidth=2.2,
                marker="o",
                label=STORY_TRUTH_LABELS.get(true_lambda, f"True λ = {true_lambda:g}"),
            )
        axis.set_xlabel("Shots")
        axis.set_ylim(-5, 105)
        if panel == 0:
            axis.set_ylabel("Percent of repeats called different from λ = 10")
        axis.legend(fontsize=9, loc="best")
    figure.suptitle("When the 95% interval no longer contains a typical aimer", fontsize=13)
    figure.tight_layout()
    figure.savefig(decision_path, dpi=160)
    plt.close(figure)
    print(f"wrote {estimate_path}")
    print(f"wrote {decision_path}")
    return [estimate_path, decision_path]


def load_player_result_csvs(players_dir: Path) -> "pd.DataFrame":
    """Read every finished ``player_*/rationality_sample_size.csv`` under ``players_dir``."""

    import pandas as pd

    paths = sorted(Path(players_dir).glob("player_*/rationality_sample_size.csv"))
    if not paths:
        raise FileNotFoundError(f"No player result CSVs under {players_dir}.")
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def plot_player_replication(players_dir: Path, output_dir: Path | None = None) -> list[Path]:
    """Compare the meeting curves across donor shot pools.

    One line per player. Sloppy and sharp aim are separate rows. The two
    starting beliefs are separate columns.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = load_player_result_csvs(players_dir)
    frame = frame[frame["status"] == "ok"].copy()
    if frame.empty:
        raise ValueError(f"No successful rows under {players_dir}.")

    destination = output_dir or Path(players_dir)
    destination.mkdir(parents=True, exist_ok=True)
    player_ids = [int(value) for value in sorted(frame["player_id"].unique())]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
    player_color = {player_id: colors[index % len(colors)] for index, player_id in enumerate(player_ids)}
    priors = [name for name in STORY_PRIOR_LABELS if name in set(frame["prior_name"])]
    focus_truths = [value for value in (1.0, 100.0) if value in set(frame["true_lambda"].astype(float))]

    calls_path = destination / "story_calls_by_player.png"
    figure, axes = plt.subplots(
        len(focus_truths),
        len(priors),
        figsize=(5.1 * max(len(priors), 1), 3.8 * max(len(focus_truths), 1)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row, true_lambda in enumerate(focus_truths):
        for column, prior_name in enumerate(priors):
            axis = axes[row, column]
            if row == 0:
                axis.set_title(STORY_PRIOR_LABELS.get(prior_name, prior_name))
            axis.axhline(50, color="#bbbbbb", linestyle="--", linewidth=1.0)
            for player_id in player_ids:
                matched = frame[
                    (frame["player_id"] == player_id)
                    & (frame["true_lambda"] == true_lambda)
                    & (frame["prior_name"] == prior_name)
                ]
                lengths = sorted(int(value) for value in matched["n_shots"].unique())
                percents = [
                    100.0 * float(matched.loc[matched["n_shots"] == length, "excluded_prior_mean"].mean())
                    for length in lengths
                ]
                axis.plot(
                    lengths,
                    percents,
                    color=player_color[player_id],
                    linewidth=2.0,
                    marker="o",
                    label=str(player_id),
                )
            if column == 0:
                truth_label = STORY_TRUTH_LABELS.get(true_lambda, f"λ = {true_lambda:g}")
                axis.set_ylabel(f"{truth_label}\npercent called different from λ = 10")
            if row == len(focus_truths) - 1:
                axis.set_xlabel("Shots")
            axis.set_ylim(-5, 105)
            if row == 0 and column == len(priors) - 1:
                axis.legend(title="Player", fontsize=8, loc="best")
    figure.suptitle("Same aiming rules on four forwards' shot situations", fontsize=13)
    figure.tight_layout()
    figure.savefig(calls_path, dpi=160)
    plt.close(figure)

    tick_positions, tick_labels = _lambda_tick_labels()
    estimate_path = destination / "story_sharp_estimate_by_player.png"
    figure, axes = plt.subplots(1, len(priors), figsize=(5.2 * max(len(priors), 1), 4.6), sharey=True, squeeze=False)
    for column, prior_name in enumerate(priors):
        axis = axes[0, column]
        axis.set_title(STORY_PRIOR_LABELS.get(prior_name, prior_name))
        axis.axhline(float(np.log(100.0)), color="black", linestyle=":", linewidth=1.2, label="Truth, λ = 100")
        axis.axhline(float(np.log(10.0)), color="#444444", linestyle="--", linewidth=1.0, label="Typical λ = 10")
        for player_id in player_ids:
            matched = frame[
                (frame["player_id"] == player_id)
                & (frame["true_lambda"] == 100.0)
                & (frame["prior_name"] == prior_name)
            ]
            lengths = sorted(int(value) for value in matched["n_shots"].unique())
            means = [
                float(matched.loc[matched["n_shots"] == length, "posterior_mean_log_lambda"].mean())
                for length in lengths
            ]
            axis.plot(
                lengths,
                means,
                color=player_color[player_id],
                linewidth=2.0,
                marker="o",
                label=str(player_id),
            )
        axis.set_xlabel("Shots")
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)
        if column == 0:
            axis.set_ylabel("Estimated aiming sharpness, λ")
        axis.legend(fontsize=8, loc="best")
    figure.suptitle("Sharp aim (true λ = 100) on four shot pools", fontsize=13)
    figure.tight_layout()
    figure.savefig(estimate_path, dpi=160)
    plt.close(figure)
    print(f"wrote {calls_path}")
    print(f"wrote {estimate_path}")
    return [calls_path, estimate_path]


def agent_panel_priors(prior_names: set[str] | list[str]) -> list[tuple[str, str]]:
    """Panels for the agent figure.

    A roster fit replaces the fixed typical-player prior. The fixed prior is
    what remains when that fit has not been added yet.
    """

    names = set(prior_names)
    if "uniform" in names and HJEEDS_PRIOR_NAME in names:
        return [
            ("uniform", STORY_PRIOR_LABELS["uniform"]),
            (HJEEDS_PRIOR_NAME, STORY_PRIOR_LABELS[HJEEDS_PRIOR_NAME]),
        ]
    return [(name, STORY_PRIOR_LABELS[name]) for name in STORY_PRIOR_LABELS if name in names]


def plot_agent_trajectories(
    players_dir: Path,
    output_dir: Path | None = None,
    *,
    panel_priors: list[tuple[str, str]] | None = None,
    figure_name: str = "story_agent_estimates.png",
    suptitle: str | None = None,
) -> list[Path]:
    """One line per simulated agent, on the same axes for every planted λ.

    An agent is one ``(player_id, seed, true_lambda)`` sequence. Donors may
    repeat. Color is the planted aiming sharpness. The two starting beliefs
    are separate panels.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frame = load_player_result_csvs(players_dir)
    frame = frame[frame["status"] == "ok"].copy()
    if frame.empty:
        raise ValueError(f"No successful rows under {players_dir}.")
    if "seed" not in frame.columns:
        frame["seed"] = 0

    destination = output_dir or Path(players_dir)
    destination.mkdir(parents=True, exist_ok=True)
    priors = panel_priors if panel_priors is not None else agent_panel_priors(set(frame["prior_name"]))
    if not priors:
        raise ValueError(f"No plottable priors under {players_dir}.")
    truths = [value for value in (1.0, 10.0, 100.0) if value in set(frame["true_lambda"].astype(float))]
    for value in sorted(float(item) for item in frame["true_lambda"].unique()):
        if value not in truths:
            truths.append(value)
    agents = (
        frame[["player_id", "seed", "true_lambda"]]
        .drop_duplicates()
        .sort_values(["true_lambda", "player_id", "seed"])
    )
    tick_positions, tick_labels = _lambda_tick_labels()

    figure, axes = plt.subplots(
        1,
        len(priors),
        figsize=(5.4 * max(len(priors), 1), 5.0),
        sharey=True,
        squeeze=False,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=STORY_TRUTH_COLORS.get(true_lambda, "C0"),
            marker="o",
            linewidth=1.8,
            label=f"{STORY_TRUTH_LABELS.get(true_lambda, 'Aim')}, λ = {true_lambda:g}",
        )
        for true_lambda in truths
    ]
    for column, (prior_name, prior_label) in enumerate(priors):
        axis = axes[0, column]
        axis.set_title(prior_label)
        for true_lambda in truths:
            axis.axhline(
                float(np.log(true_lambda)),
                color=STORY_TRUTH_COLORS.get(true_lambda, "#444444"),
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
            )
        for record in agents.itertuples(index=False):
            true_lambda = float(record.true_lambda)
            matched = frame[
                (frame["player_id"] == record.player_id)
                & (frame["seed"] == record.seed)
                & (frame["true_lambda"] == record.true_lambda)
                & (frame["prior_name"] == prior_name)
            ]
            if matched.empty:
                continue
            ordered = matched.sort_values("n_shots")
            axis.plot(
                ordered["n_shots"].to_numpy(),
                ordered["posterior_mean_log_lambda"].to_numpy(),
                color=STORY_TRUTH_COLORS.get(true_lambda, "C0"),
                linewidth=1.8,
                marker="o",
                markersize=4.0,
                alpha=0.9,
            )
        axis.set_xlabel("Shots")
        axis.set_yticks(tick_positions)
        axis.set_yticklabels(tick_labels)
        axis.set_ylim(float(np.log(0.08)), float(np.log(1200.0)))
        if column == 0:
            axis.set_ylabel("Estimated aiming sharpness, λ")
        axis.legend(handles=legend_handles, title="Assigned λ", fontsize=8, loc="best")
    if suptitle is None:
        uses_roster_fit = any(name == HJEEDS_PRIOR_NAME for name, _label in priors)
        if uses_roster_fit:
            suptitle = (
                f"{len(agents)} simulated agents. HJEEDS refits one prior from this roster at each shot count."
            )
        else:
            suptitle = f"{len(agents)} simulated agents on donor shot situations"
    figure.suptitle(suptitle, fontsize=12)
    figure.tight_layout()
    estimate_path = destination / figure_name
    figure.savefig(estimate_path, dpi=160)
    plt.close(figure)
    print(f"wrote {estimate_path}")
    return [estimate_path]


def _prefix_grid_path(output_dir: Path, player_id: int, seed: int, true_sigma: float, true_lambda: float) -> Path:
    return (
        Path(output_dir)
        / "prefix_grids"
        / f"player_{int(player_id)}_seed_{int(seed)}_sigma_{true_sigma:g}_lambda_{true_lambda:g}.npz"
    )


def _save_prefix_grids(
    output_dir: Path,
    *,
    player_id: int,
    seed: int,
    true_sigma: float,
    true_lambda: float,
    per_shot: np.ndarray,
    prefixes: list[int],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> None:
    """Store cumulative likelihood grids so a roster prior can be fit later."""

    grids = prefix_log_likelihoods(per_shot, prefixes)
    lengths = [int(length) for length in prefixes]
    stacked = np.stack([grids[length] for length in lengths], axis=0)
    path = _prefix_grid_path(output_dir, player_id, seed, true_sigma, true_lambda)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n_shots=np.asarray(lengths, dtype=int),
        log_likelihoods=stacked,
        sigma_grid=np.asarray(sigma_grid, dtype=float),
        log_lambda_grid=np.asarray(log_lambda_grid, dtype=float),
        player_id=np.int64(player_id),
        seed=np.int64(seed),
        true_sigma=np.float64(true_sigma),
        true_lambda=np.float64(true_lambda),
    )


def _load_prefix_grid_catalog(players_dir: Path) -> list[dict[str, object]]:
    paths = sorted(Path(players_dir).glob("player_*/prefix_grids/*.npz"))
    if not paths:
        raise FileNotFoundError(f"No prefix_grids under {players_dir}.")
    catalog: list[dict[str, object]] = []
    sigma_grid: np.ndarray | None = None
    log_lambda_grid: np.ndarray | None = None
    for path in paths:
        with np.load(path) as artifact:
            this_sigma = np.asarray(artifact["sigma_grid"], dtype=float).copy()
            this_log_lambda = np.asarray(artifact["log_lambda_grid"], dtype=float).copy()
            if sigma_grid is None:
                sigma_grid = this_sigma
                log_lambda_grid = this_log_lambda
            elif not np.array_equal(sigma_grid, this_sigma) or not np.array_equal(log_lambda_grid, this_log_lambda):
                raise ValueError(f"{path} uses a different skill grid from the rest of the roster.")
            catalog.append(
                {
                    "path": path,
                    "player_id": int(artifact["player_id"]),
                    "seed": int(artifact["seed"]),
                    "true_sigma": float(artifact["true_sigma"]),
                    "true_lambda": float(artifact["true_lambda"]),
                    "n_shots": np.asarray(artifact["n_shots"], dtype=int).copy(),
                    "log_likelihoods": np.asarray(artifact["log_likelihoods"], dtype=float).copy(),
                    "sigma_grid": this_sigma.copy(),
                    "log_lambda_grid": this_log_lambda.copy(),
                }
            )
    catalog.sort(key=lambda record: (int(record["player_id"]), int(record["seed"]), float(record["true_lambda"])))
    return catalog


def _uniform_row_lookup(players_dir: Path) -> dict[tuple[int, int, float, float, int], dict[str, str]]:
    lookup: dict[tuple[int, int, float, float, int], dict[str, str]] = {}
    for path in sorted(Path(players_dir).glob("player_*/rationality_sample_size.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("prior_name") != "uniform":
                    continue
                key = (
                    int(float(row["player_id"])),
                    int(float(row["seed"])),
                    float(row["true_sigma"]),
                    float(row["true_lambda"]),
                    int(float(row["n_shots"])),
                )
                lookup[key] = row
    return lookup


def _rewrite_player_csv(path: Path, extra_rows: list[dict[str, object]], *, drop_prior: str) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or CSV_FIELDS)
        for key in CSV_FIELDS:
            if key not in fieldnames:
                fieldnames.append(key)
        kept = [row for row in reader if row.get("prior_name") != drop_prior]
    for extra in extra_rows:
        encoded = {key: extra.get(key, "") for key in fieldnames}
        for flag in FLAG_NAMES:
            encoded[flag] = int(bool(extra[flag]))
        kept.append(encoded)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def add_hjeeds_roster_rows(players_dir: Path) -> Path:
    """Fit one shared roster prior at each prefix and append those posteriors.

    This is the hockey population path: ``fit_hockey_population_hyperparameters``
    on every agent's prefix likelihood, using the preliminary hockey hyperpriors,
    then ``summarize_posterior`` under that discrete prior. The agent is included
    in the roster that sets their own prior, matching ``run_hockey_population``.
    """

    players_dir = Path(players_dir)
    catalog = _load_prefix_grid_catalog(players_dir)
    sigma_grid = np.asarray(catalog[0]["sigma_grid"], dtype=float)
    log_lambda_grid = np.asarray(catalog[0]["log_lambda_grid"], dtype=float)
    lengths = [int(value) for value in catalog[0]["n_shots"]]
    for record in catalog[1:]:
        if [int(value) for value in record["n_shots"]] != lengths:
            raise ValueError(f"{record['path']} does not share the roster prefix lengths {lengths}.")

    uniform_rows = _uniform_row_lookup(players_dir)
    rows_by_player: dict[int, list[dict[str, object]]] = {}
    fit_records: list[dict[str, object]] = []
    for length_index, n_shots in enumerate(lengths):
        likelihoods = [np.asarray(record["log_likelihoods"])[length_index] for record in catalog]
        print(f"fitting roster prior at {n_shots} shots from {len(likelihoods)} agents", flush=True)
        fitted = fit_hockey_population_hyperparameters(
            likelihoods,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
            hyperpriors=HOCKEY_PRELIMINARY_HYPERPRIORS,
        )
        prior = build_discrete_hierarchical_prior(fitted, sigma_grid, log_lambda_grid)
        prior_mean, prior_sd = prior_log_lambda_marginal(prior, log_lambda_grid)
        fit_records.append(
            {
                "n_shots": int(n_shots),
                "mu_log_sigma": float(np.asarray(fitted["mu"])[0]),
                "mu_log_lambda": float(np.asarray(fitted["mu"])[1]),
                "tau_eta": float(fitted["tau_eta"]),
                "tau_rho": float(fitted["tau_rho"]),
                "correlation": float(fitted["correlation"]),
                "prior_mean_log_lambda": prior_mean,
                "prior_sd_log_lambda": prior_sd,
                "converged": bool(fitted.get("converged", False)),
                "selected_solution": fitted.get("selected_solution"),
            }
        )
        for record, log_likelihood in zip(catalog, likelihoods):
            summary = summarize_posterior(log_likelihood, prior, sigma_grid, log_lambda_grid)
            flags = departure_flags(
                posterior_mean_log_lambda=float(summary["posterior_mean_log_lambda"]),
                posterior_sd_log_lambda=float(summary["posterior_sd_log_lambda"]),
                ci_low_log_lambda=float(summary["ci_low_log_lambda"]),
                ci_high_log_lambda=float(summary["ci_high_log_lambda"]),
                prior_mean_log_lambda=prior_mean,
                prior_sd_log_lambda=prior_sd,
                true_log_lambda=float(np.log(float(record["true_lambda"]))),
            )
            key = (
                int(record["player_id"]),
                int(record["seed"]),
                float(record["true_sigma"]),
                float(record["true_lambda"]),
                int(n_shots),
            )
            source = uniform_rows.get(key, {})
            rows_by_player.setdefault(int(record["player_id"]), []).append(
                {
                    "player_id": int(record["player_id"]),
                    "shot_group": source.get("shot_group", ""),
                    "donor_pool_size": source.get("donor_pool_size", ""),
                    "seed": int(record["seed"]),
                    "true_sigma": float(record["true_sigma"]),
                    "true_lambda": float(record["true_lambda"]),
                    "true_log_lambda": float(np.log(float(record["true_lambda"]))),
                    "prior_name": HJEEDS_PRIOR_NAME,
                    "prior_lambda_center": "",
                    "prior_mean_log_lambda": prior_mean,
                    "prior_sd_log_lambda": prior_sd,
                    "n_shots": int(n_shots),
                    "num_unique_situations": source.get("num_unique_situations", ""),
                    **summary,
                    **flags,
                }
            )

    for player_id, rows in rows_by_player.items():
        csv_path = players_dir / f"player_{player_id}" / "rationality_sample_size.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path} for roster posterior rows.")
        _rewrite_player_csv(csv_path, rows, drop_prior=HJEEDS_PRIOR_NAME)
        print(f"added {len(rows)} {HJEEDS_PRIOR_NAME} rows to {csv_path}")

    payload = {
        "completed_at": _now(),
        "prior_name": HJEEDS_PRIOR_NAME,
        "roster_size": len(catalog),
        "procedure": (
            "One shared prior per shot count, fit to every simulated agent in the roster, "
            "including the agent being estimated. Same hyperpriors and fitter as "
            "run_hockey_population.py."
        ),
        "by_n_shots": fit_records,
    }
    destination = players_dir / "hjeeds_population_fit.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return destination


def apply_saved_population_prior(
    players_dir: Path,
    population_fit_path: Path,
    *,
    prior_name: str,
) -> Path:
    """Reweight saved simulated agents by a population fit from another run.

    The fit is the continuous ``(mu, tau, correlation)`` from a hockey
    population JSON. It is discretized onto the simulation grid, which can be
    coarser than the grid that produced the fit. The same prior is used at
    every shot count.
    """

    players_dir = Path(players_dir)
    fitted = json.loads(Path(population_fit_path).read_text(encoding="utf-8"))
    catalog = _load_prefix_grid_catalog(players_dir)
    sigma_grid = np.asarray(catalog[0]["sigma_grid"], dtype=float)
    log_lambda_grid = np.asarray(catalog[0]["log_lambda_grid"], dtype=float)
    lengths = [int(value) for value in catalog[0]["n_shots"]]
    for record in catalog[1:]:
        if [int(value) for value in record["n_shots"]] != lengths:
            raise ValueError(f"{record['path']} does not share the roster prefix lengths {lengths}.")

    prior = build_discrete_hierarchical_prior(fitted, sigma_grid, log_lambda_grid)
    prior_mean, prior_sd = prior_log_lambda_marginal(prior, log_lambda_grid)
    lambda_marginal = np.sum(prior, axis=0)
    lambda_marginal = lambda_marginal / np.sum(lambda_marginal)
    supported = int(np.sum(lambda_marginal > 0.01))
    print(
        f"prior {prior_name}: mean λ={float(np.exp(prior_mean)):.3f}, "
        f"log-λ sd={prior_sd:.4f}, cells above 1% mass={supported}",
        flush=True,
    )

    uniform_rows = _uniform_row_lookup(players_dir)
    rows_by_player: dict[int, list[dict[str, object]]] = {}
    for length_index, n_shots in enumerate(lengths):
        for record in catalog:
            log_likelihood = np.asarray(record["log_likelihoods"])[length_index]
            summary = summarize_posterior(log_likelihood, prior, sigma_grid, log_lambda_grid)
            flags = departure_flags(
                posterior_mean_log_lambda=float(summary["posterior_mean_log_lambda"]),
                posterior_sd_log_lambda=float(summary["posterior_sd_log_lambda"]),
                ci_low_log_lambda=float(summary["ci_low_log_lambda"]),
                ci_high_log_lambda=float(summary["ci_high_log_lambda"]),
                prior_mean_log_lambda=prior_mean,
                prior_sd_log_lambda=prior_sd,
                true_log_lambda=float(np.log(float(record["true_lambda"]))),
            )
            key = (
                int(record["player_id"]),
                int(record["seed"]),
                float(record["true_sigma"]),
                float(record["true_lambda"]),
                int(n_shots),
            )
            source = uniform_rows.get(key, {})
            rows_by_player.setdefault(int(record["player_id"]), []).append(
                {
                    "player_id": int(record["player_id"]),
                    "shot_group": source.get("shot_group", ""),
                    "donor_pool_size": source.get("donor_pool_size", ""),
                    "seed": int(record["seed"]),
                    "true_sigma": float(record["true_sigma"]),
                    "true_lambda": float(record["true_lambda"]),
                    "true_log_lambda": float(np.log(float(record["true_lambda"]))),
                    "prior_name": prior_name,
                    "prior_lambda_center": "",
                    "prior_mean_log_lambda": prior_mean,
                    "prior_sd_log_lambda": prior_sd,
                    "n_shots": int(n_shots),
                    "num_unique_situations": source.get("num_unique_situations", ""),
                    **summary,
                    **flags,
                }
            )

    for player_id, rows in rows_by_player.items():
        csv_path = players_dir / f"player_{player_id}" / "rationality_sample_size.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path} for external-prior rows.")
        _rewrite_player_csv(csv_path, rows, drop_prior=prior_name)
        print(f"added {len(rows)} {prior_name} rows to {csv_path}")

    payload = {
        "completed_at": _now(),
        "prior_name": prior_name,
        "source_fit": str(population_fit_path),
        "simulation_grid": {
            "num_sigma": int(sigma_grid.size),
            "num_log_lambda": int(log_lambda_grid.size),
        },
        "source_fit_grid_note": (
            "The continuous population parameters were discretized onto the "
            "simulation grid. That grid can be coarser than the run that produced the fit."
        ),
        "prior_mean_log_lambda": prior_mean,
        "prior_sd_log_lambda": prior_sd,
        "lambda_cells_above_1_percent": supported,
        "mu_log_lambda": float(np.asarray(fitted["mu"])[1]),
        "tau_rho": float(fitted["tau_rho"]),
    }
    destination = players_dir / f"{prior_name}.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return destination


def write_metadata(args: argparse.Namespace, donor_pool_size: int, seasons: list[int], path: Path) -> None:
    payload = {
        "completed_at": _now(),
        "sampling": "with_replacement",
        "player_id": args.player_id,
        "shot_group": args.shot_group,
        "seasons": seasons,
        "donor_pool_size": donor_pool_size,
        "num_shots": args.num_shots,
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "prefix_ns": resolve_prefixes(args.num_shots, args.prefix_ns),
        "true_sigmas": args.sigmas,
        "true_lambdas": args.lambdas,
        "sigma_grid": {
            "min": float(args.sigma_grid[0]),
            "max": float(args.sigma_grid[-1]),
            "num": int(args.sigma_grid.size),
        },
        "log_lambda_grid": {
            "min": float(args.log_lambda_grid[0]),
            "max": float(args.log_lambda_grid[-1]),
            "num": int(args.log_lambda_grid.size),
        },
        "normal_sigma_center": args.normal_sigma_center,
        "normal_lambda_centers": args.normal_lambda_centers,
        "normal_log_sigma_sd": args.normal_log_sigma_sd,
        "normal_log_lambda_sd": args.normal_log_lambda_sd,
        "preliminary_mean_log_sigma": HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA,
        "preliminary_mean_log_lambda": HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-id", type=int, default=DEFAULT_PLAYER_ID)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--shot-group", default="wristshot_snapshot")
    parser.add_argument("--num-shots", type=int, default=DEFAULT_NUM_SHOTS)
    parser.add_argument("--prefix-ns", type=int, nargs="+", default=None)
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[DEFAULT_SIGMA])
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--normal-lambda-centers", type=float, nargs="+", default=[10.0])
    parser.add_argument("--normal-sigma-center", type=float, default=0.075)
    parser.add_argument("--normal-log-sigma-sd", type=float, default=0.5)
    parser.add_argument("--normal-log-lambda-sd", type=float, default=1.0)
    parser.add_argument("--num-sigma-grid", type=int, default=DEFAULT_HOCKEY_NUM_SIGMA_GRID)
    parser.add_argument("--num-log-lambda-grid", type=int, default=DEFAULT_HOCKEY_NUM_LOG_LAMBDA_GRID)
    parser.add_argument(
        "--max-situations",
        type=int,
        default=None,
        help="Use only the first N chronological donor rows. For smoke tests.",
    )
    parser.add_argument(
        "--story-from-csv",
        type=Path,
        default=None,
        help="Write the meeting plots from an existing rationality_sample_size.csv and exit.",
    )
    parser.add_argument(
        "--story-from-players",
        type=Path,
        default=None,
        help="Write the cross-player meeting plots from player_*/rationality_sample_size.csv and exit.",
    )
    parser.add_argument(
        "--agent-lines-from",
        type=Path,
        default=None,
        help="Write one-line-per-agent estimate curves from player_*/rationality_sample_size.csv and exit.",
    )
    parser.add_argument(
        "--save-prefix-grids",
        action="store_true",
        help="Write cumulative likelihood grids for a later roster-prior fit.",
    )
    parser.add_argument(
        "--fit-hjeeds-from",
        type=Path,
        default=None,
        help="Fit one HJEEDS roster prior per prefix from player_*/prefix_grids and exit.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.story_from_csv is not None:
        plot_rationality_story(args.story_from_csv)
        return 0
    if args.story_from_players is not None:
        plot_player_replication(args.story_from_players)
        return 0
    if args.agent_lines_from is not None:
        plot_agent_trajectories(args.agent_lines_from)
        return 0
    if args.fit_hjeeds_from is not None:
        add_hjeeds_roster_rows(args.fit_hjeeds_from)
        return 0
    if args.num_shots < 1 or args.num_seeds < 1:
        parser.error("--num-shots and --num-seeds must be positive.")
    if any(value <= 0.0 for value in args.sigmas) or any(value < 0.0 for value in args.lambdas):
        parser.error("True sigma must be positive and true lambda must be nonnegative.")
    if any(value <= 0.0 for value in args.normal_lambda_centers):
        parser.error("Normal lambda centers must be positive.")
    try:
        resolve_prefixes(args.num_shots, args.prefix_ns)
    except ValueError as exc:
        parser.error(str(exc))

    args.sigma_grid = make_hockey_sigma_grid(args.num_sigma_grid)
    args.log_lambda_grid = make_hockey_log_lambda_grid(args.num_log_lambda_grid)
    rows, seasons = run_experiment(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "rationality_sample_size.csv"
    write_csv(rows, csv_path)
    print(f"wrote {csv_path}")
    donor_pool_size = int(rows[0]["donor_pool_size"]) if rows else 0
    write_metadata(args, donor_pool_size, list(seasons), args.output_dir / "run_metadata.json")
    summary = majority_crossings(rows)
    print_summary(summary)
    (args.output_dir / "majority_crossings.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_rationality_curves(rows, args.output_dir / "rationality_vs_n.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
