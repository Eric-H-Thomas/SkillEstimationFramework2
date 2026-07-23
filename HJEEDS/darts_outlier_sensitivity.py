# This file was AI-generated and still requires human review. Remove this comment when done.
"""Run the H-JEEDS outlier-contamination sensitivity experiment.

Each seed contains the default 25 agents, with five agents in each observation-
count bucket. The study replaces 0, 1, or 5 ordinary Gaussian draws with
injected outliers at a fixed Mahalanobis radius in log-skill space.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/outlier_sensitivity")
DEFAULT_CONTAMINATION_COUNTS = (0, 1, 5)
DEFAULT_OUTLIER_MAHALANOBIS_RADIUS = 3.0
OUTLIER_RANDOM_STREAM_SALT = 0x4F55544C

SCENARIOS_FILENAME = "outlier_sensitivity_scenarios.csv"
COMBINED_AGENT_LEVEL_FILENAME = "outlier_sensitivity_agent_level_results.csv"
COMBINED_SUMMARY_BY_BUCKET_FILENAME = "outlier_sensitivity_summary_by_bucket.csv"
COMBINED_SUMMARY_OVERALL_FILENAME = "outlier_sensitivity_summary_overall.csv"

CONTAMINATION_METADATA_HEADER = [
    "contamination_slug",
    "contamination_label",
    "contamination_count",
    "contamination_rate",
    "outlier_mahalanobis_radius",
    "contamination_description",
]
DESIGN_METADATA_HEADER = [
    "agents_per_bucket",
    "scenario_num_agents",
    "count_buckets",
]
SCENARIO_METADATA_HEADER = [
    "scenario_index",
    "scenario_output_dir",
    "scenario_error_plot",
]
AGENT_OUTLIER_HEADER = ["is_injected_outlier"]


@dataclass(frozen=True)
class OutlierScenario:
    """One contamination-count condition."""

    scenario_index: int
    contamination_count: int
    outlier_mahalanobis_radius: float
    config: base_experiment.ExperimentConfig

    @property
    def contamination_slug(self) -> str:
        return f"outliers_{self.contamination_count:03d}"

    @property
    def contamination_label(self) -> str:
        return "No outliers" if self.contamination_count == 0 else f"{self.contamination_count} outlier" + (
            "" if self.contamination_count == 1 else "s"
        )


def _parse_contamination_counts(raw_value: str) -> tuple[int, ...]:
    """Parse and validate the requested contamination counts."""

    counts = tuple(int(piece.strip()) for piece in raw_value.split(",") if piece.strip())
    if not counts:
        raise argparse.ArgumentTypeError("At least one contamination count is required.")
    allowed = set(DEFAULT_CONTAMINATION_COUNTS)
    if any(count not in allowed for count in counts):
        raise argparse.ArgumentTypeError(f"Contamination counts must come from {sorted(allowed)}.")
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=base_experiment.parse_seed_argument,
        required=True,
        help="Base seed used to derive per-run seeds. Use 'default' for 12345.",
    )
    parser.add_argument("--num-seeds", type=int, default=base_experiment.DEFAULT_NUM_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--contamination-counts",
        type=_parse_contamination_counts,
        default=DEFAULT_CONTAMINATION_COUNTS,
        help="Comma-separated subset of 0,1,5 (default: 0,1,5).",
    )
    parser.add_argument(
        "--outlier-mahalanobis-radius",
        type=float,
        default=DEFAULT_OUTLIER_MAHALANOBIS_RADIUS,
    )
    parser.add_argument("--aggregate-results", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-raw-rationality-error", action="store_true")
    return parser.parse_args(argv)


def scenario_index_from_environment() -> int | None:
    """Read a local/Slurm scenario index when one is provided."""

    raw_value = os.environ.get("SCENARIO_INDEX") or os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_value in (None, ""):
        return None
    scenario_index = int(raw_value)
    if scenario_index < 0:
        raise ValueError("Scenario index must be nonnegative.")
    return scenario_index


def build_config(args: argparse.Namespace, contamination_count: int) -> base_experiment.ExperimentConfig:
    """Build the unchanged default 25-agent experiment configuration."""

    base_args = argparse.Namespace(
        seed=args.seed,
        num_seeds=args.num_seeds,
        count_buckets=",".join(str(bucket) for bucket in base_experiment.DEFAULT_COUNT_BUCKETS),
        agents_per_bucket=base_experiment.DEFAULT_AGENTS_PER_BUCKET,
        num_agents=base_experiment.DEFAULT_NUM_AGENTS,
        delta=base_experiment.DEFAULT_DELTA,
        num_sigma_grid=base_experiment.DEFAULT_NUM_SIGMA_GRID,
        num_lambda_grid=base_experiment.DEFAULT_NUM_LAMBDA_GRID,
        sigma_min=base_experiment.DEFAULT_SIGMA_MIN,
        sigma_max=base_experiment.DEFAULT_SIGMA_MAX,
        lambda_min=base_experiment.DEFAULT_LAMBDA_MIN,
        lambda_max=base_experiment.DEFAULT_LAMBDA_MAX,
        output_dir=str(args.output_dir / f"outliers_{contamination_count:03d}"),
        dry_run=args.dry_run,
        min_success_regions=base_experiment.DEFAULT_MIN_SUCCESS_REGIONS,
        max_success_regions=base_experiment.DEFAULT_MAX_SUCCESS_REGIONS,
        min_region_width=base_experiment.DEFAULT_MIN_REGION_WIDTH,
    )
    return base_experiment.build_config_from_args(base_args)


def build_scenarios(args: argparse.Namespace) -> tuple[OutlierScenario, ...]:
    """Return the requested contamination scenarios."""

    if args.num_seeds <= 0:
        raise ValueError("num-seeds must be positive.")
    if args.outlier_mahalanobis_radius <= 0.0:
        raise ValueError("outlier-mahalanobis-radius must be positive.")
    return tuple(
        OutlierScenario(
            scenario_index=index,
            contamination_count=count,
            outlier_mahalanobis_radius=args.outlier_mahalanobis_radius,
            config=build_config(args, count),
        )
        for index, count in enumerate(args.contamination_counts)
    )


def injected_outlier_agent_ids(
    seed: int,
    config: base_experiment.ExperimentConfig,
    contamination_count: int,
) -> tuple[int, ...]:
    """Choose balanced, nested outlier IDs for one seed.

    Five-outlier runs replace one agent in every observation-count bucket. The
    one-outlier bucket rotates by seed, and its selected agent is also an outlier
    in the corresponding five-outlier run.
    """

    num_buckets = len(config.count_buckets)
    if config.agents_per_bucket != 5 or num_buckets != 5 or config.num_agents != 25:
        raise ValueError("The outlier study requires five agents in each of five buckets.")
    if contamination_count not in (0, 1, num_buckets):
        raise ValueError(f"Unsupported contamination count: {contamination_count}")
    if contamination_count == 0:
        return ()

    within_bucket_offset = (seed // num_buckets) % config.agents_per_bucket
    candidates = tuple(
        bucket_index * config.agents_per_bucket + within_bucket_offset
        for bucket_index in range(num_buckets)
    )
    if contamination_count == 1:
        return (candidates[seed % num_buckets],)
    return candidates


def sample_contaminated_population(
    rng: np.random.Generator,
    config: base_experiment.ExperimentConfig,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    *,
    seed: int,
    contamination_count: int,
    outlier_mahalanobis_radius: float,
) -> list[base_experiment.AgentTruth]:
    """Draw the default population, then replace selected agents with outliers."""

    truths = base_experiment.sample_true_population_params(rng, config, sigma_grid, log_lambda_grid)
    outlier_ids = injected_outlier_agent_ids(seed, config, contamination_count)
    if not outlier_ids:
        return truths

    mean = config.true_population.mean_vector
    factor = np.linalg.cholesky(config.true_population.covariance_matrix)
    log_sigma_bounds = (math.log(float(np.min(sigma_grid))), math.log(float(np.max(sigma_grid))))
    log_lambda_bounds = (float(np.min(log_lambda_grid)), float(np.max(log_lambda_grid)))

    for agent_id in outlier_ids:
        outlier_rng = np.random.default_rng(
            np.random.SeedSequence([seed % (2**32), agent_id, OUTLIER_RANDOM_STREAM_SALT])
        )
        for _attempt in range(10_000):
            angle = outlier_rng.uniform(0.0, 2.0 * math.pi)
            unit_direction = np.array([math.cos(angle), math.sin(angle)])
            eta, rho = mean + factor @ (outlier_mahalanobis_radius * unit_direction)
            if log_sigma_bounds[0] <= eta <= log_sigma_bounds[1] and log_lambda_bounds[0] <= rho <= log_lambda_bounds[1]:
                truths[agent_id] = base_experiment.AgentTruth(
                    agent_id=agent_id,
                    log_sigma_true=float(eta),
                    log_lambda_true=float(rho),
                )
                break
        else:
            raise RuntimeError("Failed to place an injected outlier inside the estimator support.")

    return truths


def _scenario_prefix(scenario: OutlierScenario) -> dict[str, Any]:
    """Return auditable metadata for one scenario."""

    config = scenario.config
    count = scenario.contamination_count
    return {
        "contamination_slug": scenario.contamination_slug,
        "contamination_label": scenario.contamination_label,
        "contamination_count": count,
        "contamination_rate": count / config.num_agents,
        "outlier_mahalanobis_radius": scenario.outlier_mahalanobis_radius,
        "contamination_description": (
            "Default Gaussian population"
            if count == 0
            else f"Replace {count} default draw(s) with fixed-radius log-skill outliers"
        ),
        "agents_per_bucket": config.agents_per_bucket,
        "scenario_num_agents": config.num_agents,
        "count_buckets": ",".join(str(bucket) for bucket in config.count_buckets),
        "scenario_index": scenario.scenario_index,
        "scenario_output_dir": str(config.output_dir),
        "scenario_error_plot": str(config.output_dir / base_experiment.ERROR_PLOT_FILENAME),
    }


def _write_rows(path: Path, header: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    """Write dictionaries under a fixed, reviewable schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in header} for row in rows)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected scenario artifact: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def run_scenario(scenario: OutlierScenario, include_raw_rationality_error: bool = False) -> None:
    """Run and write one contamination scenario."""

    config = scenario.config
    seed_results = []
    for index, seed in enumerate(config.seed_values, start=1):
        print(
            f"[outlier] {scenario.contamination_slug}: seed {index}/{config.num_seeds}: {seed}",
            flush=True,
        )

        def truth_sampler(rng, callback_config, sigma_grid, log_lambda_grid, *, current_seed=seed):
            return sample_contaminated_population(
                rng,
                callback_config,
                sigma_grid,
                log_lambda_grid,
                seed=current_seed,
                contamination_count=scenario.contamination_count,
                outlier_mahalanobis_radius=scenario.outlier_mahalanobis_radius,
            )

        seed_results.append(base_experiment.run_single_seed(config, seed, truth_sampler=truth_sampler))

    paths = base_experiment.planned_output_paths(config.output_dir)
    agent_results = [row for seed_result in seed_results for row in seed_result.agent_results]
    bucket_rows, overall_rows = base_experiment.aggregate_results_across_seeds(seed_results)
    base_experiment.write_agent_level_csv(paths["agent_level_csv"], agent_results)
    base_experiment.write_summary_csvs(config.output_dir, bucket_rows, overall_rows)
    base_experiment.plot_error_by_bucket(
        paths["error_plot"],
        bucket_rows,
        include_raw_rationality_error=include_raw_rationality_error,
    )


def aggregate_existing_results(scenarios: Sequence[OutlierScenario], output_dir: Path) -> None:
    """Combine scenario artifacts and add deterministic outlier labels."""

    scenario_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    overall_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        prefix = _scenario_prefix(scenario)
        paths = base_experiment.planned_output_paths(scenario.config.output_dir)
        scenario_rows.append(prefix)
        for row in _read_rows(paths["agent_level_csv"]):
            outlier_ids = injected_outlier_agent_ids(
                int(row["seed"]),
                scenario.config,
                scenario.contamination_count,
            )
            agent_rows.append(
                {
                    **prefix,
                    "is_injected_outlier": int(int(row["agent_id"]) in outlier_ids),
                    **row,
                }
            )
        bucket_rows.extend({**prefix, **row} for row in _read_rows(paths["summary_by_bucket_csv"]))
        overall_rows.extend({**prefix, **row} for row in _read_rows(paths["summary_overall_csv"]))

    prefix_header = CONTAMINATION_METADATA_HEADER + DESIGN_METADATA_HEADER + SCENARIO_METADATA_HEADER
    _write_rows(output_dir / SCENARIOS_FILENAME, prefix_header, scenario_rows)
    _write_rows(
        output_dir / COMBINED_AGENT_LEVEL_FILENAME,
        prefix_header + AGENT_OUTLIER_HEADER + base_experiment.AGENT_LEVEL_CSV_HEADER,
        agent_rows,
    )
    _write_rows(
        output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME,
        prefix_header + base_experiment.SUMMARY_BY_BUCKET_CSV_HEADER,
        bucket_rows,
    )
    _write_rows(
        output_dir / COMBINED_SUMMARY_OVERALL_FILENAME,
        prefix_header + base_experiment.SUMMARY_OVERALL_CSV_HEADER,
        overall_rows,
    )
    print(f"[outlier] Aggregated results into {output_dir.resolve()}", flush=True)


def render_publication_plots(output_dir: Path) -> None:
    """Render the all-agent and subgroup figures from combined results."""

    from HJEEDS.plot_outlier_sensitivity import compute_rows, render

    agent_csv = output_dir / COMBINED_AGENT_LEVEL_FILENAME
    render(compute_rows(agent_csv), output_dir, dpi=450, hide_negative_bars=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run, aggregate, or preview the outlier study."""

    args = parse_args(argv)
    scenarios = build_scenarios(args)
    scenario_index = scenario_index_from_environment()

    if args.dry_run:
        print("=== DRY RUN: Outlier Contamination Sensitivity ===")
        for scenario in scenarios:
            print(
                f"[{scenario.scenario_index}] {scenario.contamination_label}: "
                f"{scenario.config.num_seeds} seeds -> {scenario.config.output_dir}"
            )
        return 0
    if args.aggregate_results or args.plot_only:
        if scenario_index is not None:
            raise ValueError("Scenario-index mode cannot be combined with aggregation or plot-only mode.")
        aggregate_existing_results(scenarios, args.output_dir)
        render_publication_plots(args.output_dir)
        return 0
    if scenario_index is not None:
        if scenario_index >= len(scenarios):
            raise ValueError(f"Scenario index must be below {len(scenarios)}.")
        run_scenario(scenarios[scenario_index], args.include_raw_rationality_error)
        return 0

    for scenario in scenarios:
        run_scenario(scenario, args.include_raw_rationality_error)
    aggregate_existing_results(scenarios, args.output_dir)
    render_publication_plots(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
