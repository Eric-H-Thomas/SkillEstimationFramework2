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
from HJEEDS.hockey_pipeline import (
    DEFAULT_HOCKEY_NUM_LOG_LAMBDA_GRID,
    DEFAULT_HOCKEY_NUM_SIGMA_GRID,
    HOCKEY_PRELIMINARY_MEAN_LOG_LAMBDA,
    HOCKEY_PRELIMINARY_MEAN_LOG_SIGMA,
    make_hockey_log_lambda_grid,
    make_hockey_sigma_grid,
)
from HJEEDS.hockey_synthetic_player import (
    evaluate_prefixes,
    execution_rng,
    load_donor_situations,
    normal_skill_prior,
    simulate_truth_sequence,
    situation_rng,
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
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
