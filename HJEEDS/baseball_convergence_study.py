# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""CLI entry point for the Statcast baseball convergence study (Phase 2)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.artifacts import _plot_bucket_error_panels, error_metric_figure_size
from HJEEDS.baseball_convergence import (
    DEFAULT_CONVERGENCE_NS,
    DEFAULT_LAMBDA_MAX,
    DEFAULT_LAMBDA_MIN,
    DEFAULT_NUM_LAMBDA_GRID,
    DEFAULT_NUM_SIGMA_GRID,
    DEFAULT_OUTPUT_DIR_CONVERGENCE,
    DEFAULT_PITCH_TYPES,
    DEFAULT_PITCHER_IDS,
    aggregate_convergence_across_seeds,
    aggregate_convergence_results,
    build_baseball_convergence_config_from_args,
    planned_convergence_output_paths,
    prepare_convergence_roster,
    print_baseball_convergence_dry_run_summary,
    required_min_pitches_for_convergence,
    run_convergence_agent_index,
    run_single_baseball_convergence_seed,
)
from HJEEDS.baseball_pitch import DEFAULT_DELTA, DEFAULT_EXECUTION_SKILL_MAX, DEFAULT_EXECUTION_SKILL_MIN
from HJEEDS.baseball_roster import add_common_roster_arguments, add_hyperprior_arguments, print_eligible_agents
from HJEEDS.config import DEFAULT_SEED, SUMMARY_BY_BUCKET_CSV_HEADER, SUMMARY_OVERALL_CSV_HEADER, parse_seed_argument
from HJEEDS.models import StatcastConvergenceAgentResult

CONVERGENCE_AGENT_LEVEL_HEADER = [
    "seed",
    "environment",
    "agent_id",
    "pitcher_id",
    "pitch_type",
    "convergence_n",
    "num_observations",
    "num_reference_observations",
    "reference_posterior_mean_sigma",
    "reference_posterior_mean_log_lambda",
    "reference_status",
    "jeeds_posterior_mean_sigma",
    "jeeds_posterior_mean_log_lambda",
    "jeeds_abs_sigma_drift_vs_full",
    "jeeds_abs_log_lambda_drift_vs_full",
    "jeeds_status",
    "hierarchical_posterior_mean_sigma",
    "hierarchical_posterior_mean_log_lambda",
    "hierarchical_abs_sigma_drift_vs_full",
    "hierarchical_abs_log_lambda_drift_vs_full",
    "hierarchical_closer_sigma",
    "hierarchical_closer_log_lambda",
    "hierarchical_status",
    "notes",
]

DRIFT_METRIC_PANELS = (
    (
        "abs_sigma_drift_vs_full",
        "Execution Skill Drift vs Full-Data JEEDS",
        r"Absolute drift ($|\hat{\sigma}_N - \hat{\sigma}_{\mathrm{full}}|$)",
        "No rows for execution skill drift",
    ),
    (
        "abs_log_lambda_drift_vs_full",
        "Log Decision Skill Drift vs Full-Data JEEDS",
        r"Absolute drift ($|\widehat{\log\lambda}_N - \widehat{\log\lambda}_{\mathrm{full}}|$)",
        "No rows for log decision skill drift",
    ),
)


def parse_convergence_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence study on Statcast baseball: compare JEEDS and HJEEDS at "
            "reduced pitch counts against a full-data independent JEEDS reference."
        )
    )
    parser.add_argument("--seed", type=parse_seed_argument, required=False, default=None)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--convergence-ns",
        type=str,
        default=",".join(str(value) for value in DEFAULT_CONVERGENCE_NS),
        help="Comma-separated cumulative pitch-count values (newest-first prefix per agent).",
    )
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--num-sigma-grid", type=int, default=DEFAULT_NUM_SIGMA_GRID)
    parser.add_argument("--num-lambda-grid", type=int, default=DEFAULT_NUM_LAMBDA_GRID)
    parser.add_argument("--sigma-min", type=float, default=DEFAULT_EXECUTION_SKILL_MIN)
    parser.add_argument("--sigma-max", type=float, default=DEFAULT_EXECUTION_SKILL_MAX)
    parser.add_argument("--lambda-min", type=float, default=DEFAULT_LAMBDA_MIN)
    parser.add_argument("--lambda-max", type=float, default=DEFAULT_LAMBDA_MAX)
    parser.add_argument("--pitcher-ids", type=str, default=",".join(str(pid) for pid in DEFAULT_PITCHER_IDS))
    parser.add_argument("--pitch-types", type=str, default=",".join(DEFAULT_PITCH_TYPES))
    add_common_roster_arguments(parser)
    add_hyperprior_arguments(parser)
    parser.add_argument(
        "--max-reference-pitches",
        type=int,
        default=None,
        help="Cap full-data reference pitches per agent (smoke tests). Default: all available.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR_CONVERGENCE))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate drift_by_N.png from an existing summary_by_N.csv.",
    )
    parser.add_argument(
        "--prepare-roster",
        action="store_true",
        help="Write convergence_roster.json and exit (no RNN inference).",
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        default=None,
        help="Build per-agent convergence cache for one roster agent (0-based Slurm array index).",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Combine per-agent caches and write convergence CSVs and drift_by_N.png.",
    )
    parser.add_argument(
        "--use-prepared-roster",
        action="store_true",
        help="Load convergence_roster.json from --output-dir instead of re-resolving roster selectors.",
    )
    return parser.parse_args(argv)


def _value_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _convergence_result_to_row(result: StatcastConvergenceAgentResult) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "environment": "baseball",
        "agent_id": result.agent_id,
        "pitcher_id": result.pitcher_id,
        "pitch_type": result.pitch_type,
        "convergence_n": result.convergence_n,
        "num_observations": result.num_observations,
        "num_reference_observations": result.num_reference_observations,
        "reference_posterior_mean_sigma": _value_or_blank(result.reference.posterior_mean_sigma),
        "reference_posterior_mean_log_lambda": _value_or_blank(result.reference.posterior_mean_log_lambda),
        "reference_status": result.reference.status,
        "jeeds_posterior_mean_sigma": _value_or_blank(result.jeeds.posterior_mean_sigma),
        "jeeds_posterior_mean_log_lambda": _value_or_blank(result.jeeds.posterior_mean_log_lambda),
        "jeeds_abs_sigma_drift_vs_full": _value_or_blank(result.abs_sigma_drift_vs_full_jeeds),
        "jeeds_abs_log_lambda_drift_vs_full": _value_or_blank(result.abs_log_lambda_drift_vs_full_jeeds),
        "jeeds_status": result.jeeds.status,
        "hierarchical_posterior_mean_sigma": _value_or_blank(result.hierarchical.posterior_mean_sigma),
        "hierarchical_posterior_mean_log_lambda": _value_or_blank(
            result.hierarchical.posterior_mean_log_lambda
        ),
        "hierarchical_abs_sigma_drift_vs_full": _value_or_blank(
            result.abs_sigma_drift_vs_full_hierarchical
        ),
        "hierarchical_abs_log_lambda_drift_vs_full": _value_or_blank(
            result.abs_log_lambda_drift_vs_full_hierarchical
        ),
        "hierarchical_closer_sigma": _value_or_blank(result.hierarchical_closer_sigma),
        "hierarchical_closer_log_lambda": _value_or_blank(result.hierarchical_closer_log_lambda),
        "hierarchical_status": result.hierarchical.status,
        "notes": result.notes,
    }


def write_convergence_agent_level_csv(
    output_path: Path,
    agent_results: Sequence[StatcastConvergenceAgentResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONVERGENCE_AGENT_LEVEL_HEADER)
        writer.writeheader()
        for result in agent_results:
            writer.writerow(_convergence_result_to_row(result))


def write_convergence_summary_csvs(
    output_dir: Path,
    summary_by_n_rows: Sequence[dict[str, Any]],
    summary_overall_rows: Sequence[dict[str, Any]],
) -> None:
    paths = planned_convergence_output_paths(output_dir)
    paths["summary_by_n_csv"].parent.mkdir(parents=True, exist_ok=True)
    with paths["summary_by_n_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_BY_BUCKET_CSV_HEADER)
        writer.writeheader()
        writer.writerows(summary_by_n_rows)
    with paths["summary_overall_csv"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_OVERALL_CSV_HEADER)
        writer.writeheader()
        writer.writerows(summary_overall_rows)


def plot_drift_by_n(output_path: Path, summary_by_n_rows: Sequence[dict[str, Any]]) -> None:
    _plot_bucket_error_panels(
        output_path,
        summary_by_n_rows,
        DRIFT_METRIC_PANELS,
        figure_size=error_metric_figure_size(len(DRIFT_METRIC_PANELS)),
    )


def regenerate_plot_from_existing_results(output_dir: Path) -> None:
    paths = planned_convergence_output_paths(output_dir)
    summary_path = paths["summary_by_n_csv"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot regenerate plot because summary CSV is missing: {summary_path}")
    with summary_path.open("r", newline="") as handle:
        summary_by_n_rows = list(csv.DictReader(handle))
    plot_drift_by_n(paths["drift_plot"], summary_by_n_rows)
    print(f"[baseball-convergence] Regenerated plot at {paths['drift_plot'].resolve()}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_convergence_args(argv)

    if args.list_eligible_pitchers:
        from HJEEDS.config import _parse_count_buckets

        convergence_ns = _parse_count_buckets(args.convergence_ns)
        pitch_types = tuple(piece.strip() for piece in args.pitch_types.split(",") if piece.strip())
        min_pitches = (
            args.min_pitches_per_agent
            if args.min_pitches_per_agent is not None
            else required_min_pitches_for_convergence(convergence_ns, args.max_reference_pitches)
        )
        print_eligible_agents(
            season_year=args.season_year,
            pitch_types=pitch_types,
            min_pitches=min_pitches,
            limit=args.list_eligible_limit,
        )
        return 0

    if args.seed is None:
        if (
            args.dry_run
            or args.plot_only
            or args.prepare_roster
            or args.aggregate_results
            or args.agent_index is not None
        ):
            args.seed = DEFAULT_SEED
        else:
            raise SystemExit(
                "error: --seed is required unless using --list-eligible-pitchers, --dry-run, "
                "--plot-only, --prepare-roster, --agent-index, or --aggregate-results"
            )

    if args.prepare_roster:
        prepare_convergence_roster(args)
        return 0

    config = build_baseball_convergence_config_from_args(args)

    if config.base.dry_run:
        print_baseball_convergence_dry_run_summary(config)
        return 0

    if args.plot_only:
        regenerate_plot_from_existing_results(config.base.output_dir)
        return 0

    if args.aggregate_results:
        all_agent_results, summary_by_n_rows, summary_overall_rows = aggregate_convergence_results(args)
        output_paths = planned_convergence_output_paths(config.base.output_dir)
        write_convergence_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
        write_convergence_summary_csvs(config.base.output_dir, summary_by_n_rows, summary_overall_rows)
        plot_drift_by_n(output_paths["drift_plot"], summary_by_n_rows)
        print(f"[baseball-convergence] Wrote aggregated results to {config.base.output_dir.resolve()}", flush=True)
        return 0

    if args.agent_index is not None:
        run_convergence_agent_index(args, args.agent_index)
        return 0

    seed_results = []
    for seed_index, seed in enumerate(config.seed_values, start=1):
        print(
            f"[baseball-convergence] Running seed {seed_index}/{config.base.num_seeds}: {seed}",
            flush=True,
        )
        seed_results.append(run_single_baseball_convergence_seed(config, seed))

    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_n_rows, summary_overall_rows = aggregate_convergence_across_seeds(seed_results)

    output_paths = planned_convergence_output_paths(config.base.output_dir)
    write_convergence_agent_level_csv(output_paths["agent_level_csv"], all_agent_results)
    write_convergence_summary_csvs(config.base.output_dir, summary_by_n_rows, summary_overall_rows)
    plot_drift_by_n(output_paths["drift_plot"], summary_by_n_rows)
    print(f"[baseball-convergence] Wrote results to {config.base.output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
