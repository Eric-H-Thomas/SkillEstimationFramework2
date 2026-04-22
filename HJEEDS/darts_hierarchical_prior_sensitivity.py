# This file still requires human verification. Delete this comment when done.
"""Run a 3x3 hyperprior-sensitivity sweep for hierarchical 1D darts.

This script wraps ``HJEEDS/darts_hierarchical_vs_jeeds.py`` and reruns the
same simulated experiment under nine empirical-Bayes hyperprior conditions:

- confidence: weak, default, strong
- bias: unbiased, slightly biased, significantly biased

The goal is to answer a simple paper question: does the hierarchical method
still help when the population-level hyperpriors are weaker, stronger, or
miscentered?  Each condition gets its own normal experiment output directory,
and this script also writes combined CSVs plus a compact 3x3 heatmap for the
lowest-count bucket, where prior sensitivity should be easiest to see.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Ensure the repository root is importable when this file is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment


# This script answers the paper's robustness question: if H-JEEDS benefits from
# a shared prior in low-data settings, how sensitive is that benefit to the
# hyperprior being too weak, too strong, or miscentered?
#
# The implementation works by reusing the main experiment pipeline and swapping
# in nine alternative hyperprior settings arranged in a 3x3 grid.


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_prior_sensitivity")
DEFAULT_COUNT_BUCKETS = (5, 10, 25, 100, 1000)
DEFAULT_WEAK_STD_MULTIPLIER = 3.0
DEFAULT_STRONG_STD_MULTIPLIER = 1.0 / 3.0
DEFAULT_SLIGHT_BIAS_SD_UNITS = 0.5
DEFAULT_SIGNIFICANT_BIAS_SD_UNITS = 1.5

CONDITION_METADATA_HEADER = [
    "condition_slug",
    "confidence_level",
    "bias_level",
    "confidence_std_multiplier",
    "bias_sd_units",
    "hyperprior_mu_eta",
    "hyperprior_mu_rho",
    "hyperprior_mu_sigma",
    "hyperprior_mu_lambda",
    "hyperprior_mu_eta_sd",
    "hyperprior_mu_rho_sd",
    "hyperprior_log_tau_eta_mean",
    "hyperprior_log_tau_eta_sd",
    "hyperprior_log_tau_rho_mean",
    "hyperprior_log_tau_rho_sd",
    "hyperprior_tau_eta_median",
    "hyperprior_tau_rho_median",
    "hyperprior_m_r",
    "hyperprior_r_center",
    "hyperprior_s_r",
]

COMBINED_AGENT_LEVEL_FILENAME = "prior_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "prior_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "prior_sensitivity_summary_overall.csv"
CONDITION_METADATA_FILENAME = "prior_sensitivity_conditions.csv"
LOWEST_BUCKET_HEATMAP_FILENAME = "prior_sensitivity_lowest_bucket_heatmap.png"


@dataclass(frozen=True)
class PriorSensitivityCondition:
    """One cell in the prior confidence x prior bias sensitivity grid."""

    confidence_label: str
    confidence_slug: str
    bias_label: str
    bias_slug: str
    confidence_std_multiplier: float
    bias_sd_units: float

    @property
    def condition_slug(self) -> str:
        """Return a filename-safe identifier for this sensitivity condition."""

        return f"{self.confidence_slug}__{self.bias_slug}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the 3x3 sensitivity sweep."""

    # Most arguments simply forward through to the base experiment so the
    # sensitivity runner stays aligned with the main H-JEEDS script.
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=12345, help="Base seed used to derive per-run seeds.")
    parser.add_argument(
        "--num-seeds",
        type=int,
        required=True,
        help="Number of random seeds to run for each prior condition. This argument is required.",
    )
    parser.add_argument(
        "--count-buckets",
        type=str,
        default=",".join(str(bucket) for bucket in DEFAULT_COUNT_BUCKETS),
        help="Comma-separated observation-count buckets assigned across demonstrators.",
    )
    parser.add_argument(
        "--agents-per-bucket",
        type=int,
        default=base_experiment.DEFAULT_AGENTS_PER_BUCKET,
        help="How many demonstrators receive each observation-count bucket.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=None,
        help=(
            "Total number of demonstrators. Defaults to len(count_buckets) * "
            "agents_per_bucket so the 3x3 runner can use a five-bucket setup."
        ),
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=base_experiment.DEFAULT_DELTA,
        help="Target-grid resolution used by the darts environment and likelihood code.",
    )
    parser.add_argument(
        "--num-sigma-grid",
        type=int,
        default=base_experiment.DEFAULT_NUM_SIGMA_GRID,
        help="Number of execution-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument(
        "--num-lambda-grid",
        type=int,
        default=base_experiment.DEFAULT_NUM_LAMBDA_GRID,
        help="Number of decision-skill hypotheses on the JEEDS grid.",
    )
    parser.add_argument("--sigma-min", type=float, default=base_experiment.DEFAULT_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=base_experiment.DEFAULT_SIGMA_MAX)
    parser.add_argument("--lambda-min", type=float, default=base_experiment.DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=base_experiment.DEFAULT_LAMBDA_MAX)
    parser.add_argument("--min-regions", type=int, default=base_experiment.DEFAULT_MIN_REGIONS)
    parser.add_argument("--max-regions", type=int, default=base_experiment.DEFAULT_MAX_REGIONS)
    parser.add_argument("--min-region-width", type=float, default=base_experiment.DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for combined sensitivity artifacts and condition subdirectories.",
    )
    parser.add_argument(
        "--weak-std-multiplier",
        type=float,
        default=DEFAULT_WEAK_STD_MULTIPLIER,
        help=(
            "Multiplier applied to hyperprior standard deviations for the weak/underconfident condition."
        ),
    )
    parser.add_argument(
        "--strong-std-multiplier",
        type=float,
        default=DEFAULT_STRONG_STD_MULTIPLIER,
        help=(
            "Multiplier applied to hyperprior standard deviations for the strong/overconfident condition."
        ),
    )
    parser.add_argument(
        "--slight-bias-sd-units",
        type=float,
        default=DEFAULT_SLIGHT_BIAS_SD_UNITS,
        help="Mean shift for the slightly biased condition, measured in default prior-SD units.",
    )
    parser.add_argument(
        "--significant-bias-sd-units",
        type=float,
        default=DEFAULT_SIGNIFICANT_BIAS_SD_UNITS,
        help="Mean shift for the significantly biased condition, measured in default prior-SD units.",
    )
    parser.add_argument(
        "--condition-slugs",
        type=str,
        default="",
        help=(
            "Optional comma-separated subset of condition slugs to run, for smoke tests. "
            "Leave blank to run the full 3x3 grid."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned 3x3 workload and stop before simulation/inference.",
    )

    return parser.parse_args(argv)


def _parse_count_buckets(raw_value: str) -> tuple[int, ...]:
    """Parse comma-separated count buckets for deriving the default agent count."""

    pieces = [piece.strip() for piece in raw_value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("At least one count bucket must be provided.")

    buckets = tuple(int(piece) for piece in pieces)
    if any(bucket <= 0 for bucket in buckets):
        raise ValueError(f"Count buckets must all be positive integers. Received: {buckets}")
    return buckets


def build_base_config_from_args(args: argparse.Namespace) -> base_experiment.ExperimentConfig:
    """Build the underlying darts experiment config shared by all conditions."""

    # The sensitivity sweep shares one common experimental design; only the
    # hyperpriors change from condition to condition.
    count_buckets = _parse_count_buckets(args.count_buckets)
    num_agents = args.num_agents
    if num_agents is None:
        num_agents = len(count_buckets) * args.agents_per_bucket

    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=num_agents,
        count_buckets=args.count_buckets,
        agents_per_bucket=args.agents_per_bucket,
        delta=args.delta,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        min_regions=args.min_regions,
        max_regions=args.max_regions,
        min_region_width=args.min_region_width,
    )
    return base_experiment.build_config_from_args(base_args)


def build_sensitivity_conditions(args: argparse.Namespace) -> tuple[PriorSensitivityCondition, ...]:
    """Return the full 3x3 grid, optionally filtered to requested slugs."""

    # The 3x3 grid is generated mechanically from one set of confidence levels
    # crossed with one set of bias levels.
    if args.weak_std_multiplier <= 0.0:
        raise ValueError("weak_std_multiplier must be positive.")
    if args.strong_std_multiplier <= 0.0:
        raise ValueError("strong_std_multiplier must be positive.")
    if args.slight_bias_sd_units < 0.0:
        raise ValueError("slight_bias_sd_units must be non-negative.")
    if args.significant_bias_sd_units < 0.0:
        raise ValueError("significant_bias_sd_units must be non-negative.")

    confidence_levels = [
        ("weak", "weak", float(args.weak_std_multiplier)),
        ("default", "default", 1.0),
        ("strong", "strong", float(args.strong_std_multiplier)),
    ]
    bias_levels = [
        ("unbiased", "unbiased", 0.0),
        ("slightly biased", "slightly_biased", float(args.slight_bias_sd_units)),
        ("significantly biased", "significantly_biased", float(args.significant_bias_sd_units)),
    ]

    conditions = tuple(
        PriorSensitivityCondition(
            confidence_label=confidence_label,
            confidence_slug=confidence_slug,
            bias_label=bias_label,
            bias_slug=bias_slug,
            confidence_std_multiplier=confidence_std_multiplier,
            bias_sd_units=bias_sd_units,
        )
        for confidence_label, confidence_slug, confidence_std_multiplier in confidence_levels
        for bias_label, bias_slug, bias_sd_units in bias_levels
    )

    # Optional filtering makes smoke tests cheap without changing the logic of
    # how conditions are defined.
    requested_slugs = {piece.strip() for piece in args.condition_slugs.split(",") if piece.strip()}
    if not requested_slugs:
        return conditions

    known_slugs = {condition.condition_slug for condition in conditions}
    unknown_slugs = sorted(requested_slugs - known_slugs)
    if unknown_slugs:
        raise ValueError(
            f"Unknown condition slugs {unknown_slugs}. Known slugs are: {sorted(known_slugs)}"
        )

    return tuple(condition for condition in conditions if condition.condition_slug in requested_slugs)


def build_condition_hyperpriors(
    base_hyperpriors: base_experiment.HyperpriorConfig,
    condition: PriorSensitivityCondition,
) -> base_experiment.HyperpriorConfig:
    """Create the hyperprior settings for one sensitivity condition.

    Confidence is represented by scaling hyperprior standard deviations while
    leaving their centers fixed. Bias is represented by shifting the hyperprior
    centers. For the population mean, biased conditions expect weaker agents:
    larger log-sigma and smaller log-lambda. For the population spread,
    biased conditions expect too much between-demonstrator variation by
    shifting the log-tau prior means upward.
    """

    # Start from the base hyperpriors used in the main experiment and then
    # perturb them along two interpretable axes:
    # - confidence: scale the prior standard deviations
    # - bias: shift the centers of the priors
    base_mu_eta, base_mu_rho = base_hyperpriors.mean_vector
    base_mu_eta_sd = math.sqrt(float(base_hyperpriors.covariance_diagonal[0]))
    base_mu_rho_sd = math.sqrt(float(base_hyperpriors.covariance_diagonal[1]))
    base_log_tau_eta_mean = float(base_hyperpriors.log_tau_eta_mean)
    base_log_tau_rho_mean = float(base_hyperpriors.log_tau_rho_mean)
    base_log_tau_eta_sd = float(base_hyperpriors.log_tau_eta_sd)
    base_log_tau_rho_sd = float(base_hyperpriors.log_tau_rho_sd)

    # The biased conditions intentionally assume weaker demonstrators on
    # average: higher sigma (worse execution) and lower lambda
    # (less rational decision-making).
    mean_shift_eta = condition.bias_sd_units * base_mu_eta_sd
    mean_shift_rho = -condition.bias_sd_units * base_mu_rho_sd
    log_tau_eta_shift = condition.bias_sd_units * base_log_tau_eta_sd
    log_tau_rho_shift = condition.bias_sd_units * base_log_tau_rho_sd
    scaled_mu_eta_sd = max(base_mu_eta_sd * condition.confidence_std_multiplier, 1e-9)
    scaled_mu_rho_sd = max(base_mu_rho_sd * condition.confidence_std_multiplier, 1e-9)
    scaled_log_tau_eta_sd = max(base_log_tau_eta_sd * condition.confidence_std_multiplier, 1e-9)
    scaled_log_tau_rho_sd = max(base_log_tau_rho_sd * condition.confidence_std_multiplier, 1e-9)

    return base_experiment.HyperpriorConfig(
        mean_vector=(base_mu_eta + mean_shift_eta, base_mu_rho + mean_shift_rho),
        covariance_diagonal=(scaled_mu_eta_sd**2, scaled_mu_rho_sd**2),
        log_tau_eta_mean=base_log_tau_eta_mean + log_tau_eta_shift,
        log_tau_eta_sd=scaled_log_tau_eta_sd,
        log_tau_rho_mean=base_log_tau_rho_mean + log_tau_rho_shift,
        log_tau_rho_sd=scaled_log_tau_rho_sd,
        m_r=base_hyperpriors.m_r,
        s_r=max(base_hyperpriors.s_r * condition.confidence_std_multiplier, 1e-9),
    )


def condition_metadata_row(
    condition: PriorSensitivityCondition,
    hyperpriors: base_experiment.HyperpriorConfig,
) -> dict[str, Any]:
    """Return CSV metadata describing one condition's concrete hyperpriors."""

    mu_eta, mu_rho = hyperpriors.mean_vector
    mu_eta_sd = math.sqrt(float(hyperpriors.covariance_diagonal[0]))
    mu_rho_sd = math.sqrt(float(hyperpriors.covariance_diagonal[1]))

    # This row records the actual numeric hyperpriors used for the condition so
    # later CSVs/plots can be traced back to concrete assumptions.
    return {
        "condition_slug": condition.condition_slug,
        "confidence_level": condition.confidence_label,
        "bias_level": condition.bias_label,
        "confidence_std_multiplier": condition.confidence_std_multiplier,
        "bias_sd_units": condition.bias_sd_units,
        "hyperprior_mu_eta": mu_eta,
        "hyperprior_mu_rho": mu_rho,
        "hyperprior_mu_sigma": math.exp(mu_eta),
        "hyperprior_mu_lambda": math.exp(mu_rho),
        "hyperprior_mu_eta_sd": mu_eta_sd,
        "hyperprior_mu_rho_sd": mu_rho_sd,
        "hyperprior_log_tau_eta_mean": hyperpriors.log_tau_eta_mean,
        "hyperprior_log_tau_eta_sd": hyperpriors.log_tau_eta_sd,
        "hyperprior_log_tau_rho_mean": hyperpriors.log_tau_rho_mean,
        "hyperprior_log_tau_rho_sd": hyperpriors.log_tau_rho_sd,
        "hyperprior_tau_eta_median": math.exp(hyperpriors.log_tau_eta_mean),
        "hyperprior_tau_rho_median": math.exp(hyperpriors.log_tau_rho_mean),
        "hyperprior_m_r": hyperpriors.m_r,
        "hyperprior_r_center": math.tanh(hyperpriors.m_r),
        "hyperprior_s_r": hyperpriors.s_r,
    }


def _write_dict_rows(output_path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries with a fixed header, leaving missing fields blank."""

    # This local writer mirrors the base experiment helper but keeps the
    # sensitivity script self-contained for its combined artifacts.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def condition_prefix(
    condition: PriorSensitivityCondition,
    hyperpriors: base_experiment.HyperpriorConfig,
) -> dict[str, Any]:
    """Return the condition metadata fields prepended to combined result rows."""

    return condition_metadata_row(condition, hyperpriors)


def run_condition(
    base_config: base_experiment.ExperimentConfig,
    condition: PriorSensitivityCondition,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run all seeds for one sensitivity condition and write per-condition artifacts."""

    # Clone the shared experiment config and swap in this condition's concrete
    # hyperprior settings.  Everything else about the experiment remains fixed.
    hyperpriors = build_condition_hyperpriors(base_config.hyperpriors, condition)
    condition_config = replace(
        base_config,
        output_dir=base_config.output_dir / condition.condition_slug,
        hyperpriors=hyperpriors,
        dry_run=False,
    )

    seed_results: list[base_experiment.SeedResult] = []
    for seed_index, seed in enumerate(condition_config.seed_values, start=1):
        print(
            "[prior-sensitivity] "
            f"{condition.condition_slug}: seed {seed_index}/{condition_config.num_seeds} ({seed})",
            flush=True,
        )
        seed_results.append(base_experiment.run_single_seed(condition_config, seed))

    # Each condition gets its own normal per-condition artifact directory in
    # addition to contributing rows to the combined CSVs.
    output_paths = base_experiment.planned_output_paths(condition_config.output_dir)
    all_agent_results = [
        result
        for seed_result in seed_results
        for result in seed_result.agent_results
    ]
    summary_by_bucket_rows, summary_overall_rows = base_experiment.aggregate_results_across_seeds(seed_results)

    base_experiment.write_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
    base_experiment.write_summary_csvs(
        condition_config.output_dir,
        summary_by_bucket_rows,
        summary_overall_rows,
    )
    base_experiment.plot_error_by_bucket(output_paths["error_plot"], summary_by_bucket_rows)

    # Combined output files prepend each result row with the condition metadata
    # so one CSV can hold the full sweep without losing provenance.
    prefix = condition_prefix(condition, hyperpriors)
    combined_agent_rows = [
        {**prefix, **base_experiment._agent_result_to_row(result)}
        for result in all_agent_results
    ]
    combined_bucket_rows = [{**prefix, **row} for row in summary_by_bucket_rows]
    combined_overall_rows = [{**prefix, **row} for row in summary_overall_rows]

    return combined_agent_rows, combined_bucket_rows, combined_overall_rows, prefix


def plot_lowest_bucket_heatmap(
    output_path: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    conditions: Sequence[PriorSensitivityCondition],
) -> None:
    """Plot hierarchical error as a 3x3 grid for the lowest observation bucket."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # The heatmap focuses on the lowest-count bucket because that is where the
    # hierarchical prior should matter most according to the paper's claim.
    if not summary_by_bucket_rows:
        return

    bucket_values = sorted(
        {
            int(row["count_bucket"])
            for row in summary_by_bucket_rows
            if row.get("method") == "hierarchical" and row.get("count_bucket") not in (None, "")
        }
    )
    if not bucket_values:
        return
    selected_bucket = bucket_values[0]

    confidence_labels = []
    bias_labels = []
    for condition in conditions:
        if condition.confidence_label not in confidence_labels:
            confidence_labels.append(condition.confidence_label)
        if condition.bias_label not in bias_labels:
            bias_labels.append(condition.bias_label)

    confidence_positions = {label: index for index, label in enumerate(confidence_labels)}
    bias_positions = {label: index for index, label in enumerate(bias_labels)}

    metric_panels = [
        ("abs_sigma_error", "Execution Skill Error"),
        ("abs_log10_lambda_error", "Decision Skill Error"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for axis, (metric_name, title) in zip(axes, metric_panels):
        heatmap = np.full((len(confidence_labels), len(bias_labels)), np.nan, dtype=float)
        for row in summary_by_bucket_rows:
            if row.get("method") != "hierarchical":
                continue
            if row.get("metric") != metric_name:
                continue
            if int(row["count_bucket"]) != selected_bucket:
                continue

            confidence = str(row["confidence_level"])
            bias = str(row["bias_level"])
            heatmap[confidence_positions[confidence], bias_positions[bias]] = float(row["mean"])

        image = axis.imshow(heatmap, cmap="viridis")
        axis.set_title(f"{title}\nBucket {selected_bucket}")
        axis.set_xticks(range(len(bias_labels)))
        axis.set_xticklabels(bias_labels, rotation=25, ha="right")
        axis.set_yticks(range(len(confidence_labels)))
        axis.set_yticklabels(confidence_labels)
        axis.set_xlabel("Prior bias")
        axis.set_ylabel("Prior confidence")

        for row_index in range(heatmap.shape[0]):
            for col_index in range(heatmap.shape[1]):
                value = heatmap[row_index, col_index]
                if np.isfinite(value):
                    axis.text(
                        col_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color="white" if value > np.nanmean(heatmap) else "black",
                    )

        figure.colorbar(image, ax=axis, shrink=0.85)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def print_dry_run_summary(
    config: base_experiment.ExperimentConfig,
    conditions: Sequence[PriorSensitivityCondition],
) -> None:
    """Report the planned sensitivity workload without running inference."""

    # This summary is intentionally rich because a full 3x3 x many-seed sweep
    # can be expensive to run; the user should be able to inspect the planned
    # hyperprior shifts before launching it.
    print("=== DRY RUN: 1D Darts Prior Sensitivity ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Seeds per condition: {config.seed_values}")
    print(f"Agents per seed: {config.num_agents}")
    print(f"Count buckets: {config.count_buckets}")
    print(f"Agents per bucket: {config.agents_per_bucket}")
    print(f"Output directory: {config.output_dir.resolve()}")
    print(f"Conditions: {len(conditions)}")
    for condition in conditions:
        hyperpriors = build_condition_hyperpriors(config.hyperpriors, condition)
        metadata = condition_metadata_row(condition, hyperpriors)
        print(
            "  - "
            f"{condition.condition_slug}: confidence={condition.confidence_label}, "
            f"bias={condition.bias_label}, "
            f"mu=({metadata['hyperprior_mu_eta']:.3f}, {metadata['hyperprior_mu_rho']:.3f}), "
            f"mu_sd=({metadata['hyperprior_mu_eta_sd']:.3f}, {metadata['hyperprior_mu_rho_sd']:.3f}), "
            f"tau_median=({metadata['hyperprior_tau_eta_median']:.3f}, "
            f"{metadata['hyperprior_tau_rho_median']:.3f}), "
            f"log_tau_sd=({metadata['hyperprior_log_tau_eta_sd']:.3f}, "
            f"{metadata['hyperprior_log_tau_rho_sd']:.3f})"
        )
    print()
    print("Planned combined artifacts:")
    print(f"  - {config.output_dir / CONDITION_METADATA_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    print(f"  - {config.output_dir / LOWEST_BUCKET_HEATMAP_FILENAME}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 3x3 prior-sensitivity sweep."""

    # Parse and build the shared experiment design once.  The loop below only
    # varies the hyperprior assumptions, not the underlying darts setup.
    args = parse_args(argv)
    config = build_base_config_from_args(args)
    conditions = build_sensitivity_conditions(args)

    if config.dry_run:
        print_dry_run_summary(config, conditions)
        return 0

    # These lists accumulate rows from all nine conditions into one set of
    # combined CSVs that are easier to analyze in a spreadsheet or notebook.
    all_agent_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    all_overall_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for condition in conditions:
        agent_rows, bucket_rows, overall_rows, metadata_row = run_condition(config, condition)
        all_agent_rows.extend(agent_rows)
        all_bucket_rows.extend(bucket_rows)
        all_overall_rows.extend(overall_rows)
        condition_rows.append(metadata_row)

    output_dir = config.output_dir
    condition_columns = CONDITION_METADATA_HEADER
    _write_dict_rows(output_dir / CONDITION_METADATA_FILENAME, condition_columns, condition_rows)
    _write_dict_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        condition_columns + base_experiment.AGENT_LEVEL_CSV_HEADER,
        all_agent_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        condition_columns + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        all_bucket_rows,
    )
    _write_dict_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        condition_columns + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        all_overall_rows,
    )
    plot_lowest_bucket_heatmap(output_dir / LOWEST_BUCKET_HEATMAP_FILENAME, all_bucket_rows, conditions)

    print(f"[prior-sensitivity] Wrote combined results to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
