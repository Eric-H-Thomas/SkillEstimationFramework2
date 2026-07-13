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

from HJEEDS.artifacts import error_metric_figure_size
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
        r"Convergence of $\hat{\sigma}$ toward JEEDS reference",
        r"$|\hat{\sigma}_N - \hat{\sigma}_{\mathrm{ref}}|$",
        "No rows for execution skill drift",
    ),
    (
        "abs_log_lambda_drift_vs_full",
        r"Convergence of $\widehat{\log\lambda}$ toward JEEDS reference",
        r"$|\widehat{\log\lambda}_N - \widehat{\log\lambda}_{\mathrm{ref}}|$",
        "No rows for log decision skill drift",
    ),
)

# Distinct from shared METHOD_COLORS (teal/magenta error-vs-truth figures).
CONVERGENCE_METHOD_STYLES = {
    "jeeds": {
        "label": "Independent JEEDS",
        "color": "#1B4F72",
        "marker": "^",
        "linestyle": "--",
    },
    "hierarchical": {
        "label": "H-JEEDS",
        "color": "#B9770E",
        "marker": "D",
        "linestyle": "-",
    },
}

# Solid lines for per-agent intermediate-estimate plots (checkpoints only).
AGENT_ESTIMATE_METHOD_STYLES = {
    "jeeds": {
        "label": "Independent JEEDS",
        "color": "#002D72",
        "marker": "^",
    },
    "hierarchical": {
        "label": "H-JEEDS",
        "color": "#D50032",
        "marker": "D",
    },
}


def parse_convergence_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convergence study on Statcast baseball: compare JEEDS and HJEEDS at "
            "reduced pitch counts against a full-data independent JEEDS reference."
        )
    )
    parser.add_argument(
        "--seed",
        type=parse_seed_argument,
        required=False,
        default=None,
        help=(
            "Base seed (or 'default' for 12345). Kept for CLI compatibility with darts; "
            "Statcast likelihoods are deterministic in seed, so varying it does not change estimates."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help=(
            "Number of consecutive seeds to run (default: 1). For baseball, keep at 1: "
            "additional seeds repeat the same numerics and do not produce uncertainty bands."
        ),
    )
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
        "--plot-agent-estimates",
        action="store_true",
        help=(
            "Write per-agent intermediate-estimate PNGs under output_dir/agents/ "
            "from an existing convergence_agent_level_results.csv."
        ),
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
    """Plot prefix-N drift toward the full-window JEEDS reference (not error-vs-truth)."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    from HJEEDS.artifacts import METHOD_ORDER, _parse_bucket_summary_rows

    parsed_rows = _parse_bucket_summary_rows(summary_by_n_rows)
    bucket_values = sorted({row["count_bucket"] for row in parsed_rows})
    bucket_positions = {bucket: index for index, bucket in enumerate(bucket_values)}
    figure_size = error_metric_figure_size(len(DRIFT_METRIC_PANELS))
    figure, axes = plt.subplots(1, len(DRIFT_METRIC_PANELS), figsize=figure_size, sharex=True)
    axes_array = np.atleast_1d(axes)

    for axis, (metric_name, title, ylabel, missing_message) in zip(axes_array, DRIFT_METRIC_PANELS):
        axis.set_title(title)
        axis.set_xlabel("Prefix pitch-count checkpoint")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)

        metric_rows = [row for row in parsed_rows if row["metric"] == metric_name]
        methods = sorted(
            {row["method"] for row in metric_rows},
            key=lambda method: (METHOD_ORDER.get(method, len(METHOD_ORDER)), method),
        )

        if metric_rows and bucket_values:
            for method in methods:
                style = CONVERGENCE_METHOD_STYLES.get(
                    method,
                    {"label": method, "color": "#555555", "marker": "o", "linestyle": "-"},
                )
                method_rows = sorted(
                    [row for row in metric_rows if row["method"] == method],
                    key=lambda row: bucket_positions[row["count_bucket"]],
                )
                x_values: list[int] = []
                y_values: list[float] = []
                lower_errors: list[float] = []
                upper_errors: list[float] = []
                for row in method_rows:
                    mean = row["mean"]
                    ci_lower = row["ci_lower"]
                    ci_upper = row["ci_upper"]
                    x_values.append(bucket_positions[row["count_bucket"]])
                    y_values.append(mean)
                    lower_errors.append(max(0.0, mean - ci_lower) if ci_lower is not None else 0.0)
                    upper_errors.append(max(0.0, ci_upper - mean) if ci_upper is not None else 0.0)

                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=np.array([lower_errors, upper_errors], dtype=float),
                    color=style["color"],
                    marker=style["marker"],
                    markersize=5.5,
                    linestyle=style["linestyle"],
                    capsize=4,
                    linewidth=2.0,
                    label=style["label"],
                )

            axis.set_xticks(list(bucket_positions.values()))
            axis.set_xticklabels([str(bucket) for bucket in bucket_values])
            axis.legend(title="Estimator", loc="best", fontsize=8)
            if bucket_values:
                ref_n = max(bucket_values)
                axis.annotate(
                    f"ref = JEEDS @ {ref_n}",
                    xy=(0.98, 0.02),
                    xycoords="axes fraction",
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="0.35",
                )
        else:
            axis.text(
                0.5,
                0.5,
                missing_message,
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    figure.suptitle(
        "Prefix-N estimates vs fixed full-window independent JEEDS reference "
        "(not ground-truth error; JEEDS at the final checkpoint is the reference by construction)",
        fontsize=9.5,
        y=1.02,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def regenerate_plot_from_existing_results(output_dir: Path) -> None:
    paths = planned_convergence_output_paths(output_dir)
    summary_path = paths["summary_by_n_csv"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Cannot regenerate plot because summary CSV is missing: {summary_path}")
    with summary_path.open("r", newline="") as handle:
        summary_by_n_rows = list(csv.DictReader(handle))
    plot_drift_by_n(paths["drift_plot"], summary_by_n_rows)
    print(f"[baseball-convergence] Regenerated plot at {paths['drift_plot'].resolve()}", flush=True)


def _load_pitcher_name_lookup(output_dir: Path) -> dict[int, str]:
    import json

    metadata_path = output_dir / "convergence_roster_metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text())
    names: dict[int, str] = {}
    for row in payload.get("bbip_selection", []):
        try:
            names[int(row["pitcher_id"])] = str(row["player_name"])
        except (KeyError, TypeError, ValueError):
            continue
    return names


def plot_agent_intermediate_estimates(output_dir: Path) -> list[Path]:
    """Plot JEEDS/H-JEEDS checkpoint estimates per agent into ``agents/*.png``."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    paths = planned_convergence_output_paths(output_dir)
    agent_csv = paths["agent_level_csv"]
    if not agent_csv.is_file():
        raise FileNotFoundError(f"Missing agent-level CSV: {agent_csv}")

    frame = pd.read_csv(agent_csv)
    if frame.empty:
        raise ValueError(f"No rows in {agent_csv}")

    agents_dir = output_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    name_lookup = _load_pitcher_name_lookup(output_dir)
    written: list[Path] = []

    metric_panels = (
        ("jeeds_posterior_mean_sigma", "hierarchical_posterior_mean_sigma", r"$\hat{\sigma}$", "sigma"),
        (
            "jeeds_posterior_mean_log_lambda",
            "hierarchical_posterior_mean_log_lambda",
            r"$\widehat{\log\lambda}$",
            "log_lambda",
        ),
    )

    for agent_id, agent_frame in frame.groupby("agent_id", sort=True):
        agent_frame = agent_frame.sort_values("convergence_n")
        ns = agent_frame["convergence_n"].astype(int).tolist()
        pitcher_id = int(agent_frame["pitcher_id"].iloc[0])
        pitch_type = str(agent_frame["pitch_type"].iloc[0])
        player_name = name_lookup.get(pitcher_id, f"pitcher {pitcher_id}")
        final_n = max(ns)

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for axis, (jeeds_col, hier_col, ylabel, _key) in zip(axes, metric_panels):
            jeeds_y = agent_frame[jeeds_col].astype(float).tolist()
            hier_y = agent_frame[hier_col].astype(float).tolist()
            jeeds_style = AGENT_ESTIMATE_METHOD_STYLES["jeeds"]
            hier_style = AGENT_ESTIMATE_METHOD_STYLES["hierarchical"]

            axis.plot(
                ns,
                jeeds_y,
                color=jeeds_style["color"],
                marker=jeeds_style["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=jeeds_style["label"],
            )
            axis.plot(
                ns,
                hier_y,
                color=hier_style["color"],
                marker=hier_style["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=hier_style["label"],
            )

            final_row = agent_frame.loc[agent_frame["convergence_n"] == final_n].iloc[0]
            axis.axhline(
                float(final_row[jeeds_col]),
                color=jeeds_style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"JEEDS @ {final_n}",
            )
            axis.axhline(
                float(final_row[hier_col]),
                color=hier_style["color"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.85,
                label=f"H-JEEDS @ {final_n}",
            )
            axis.set_xlabel("Prefix pitch count $N$")
            axis.set_ylabel(ylabel)
            axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
            axis.set_xticks(ns)

        axes[0].legend(loc="best", fontsize=8)
        figure.suptitle(f"{player_name} ({pitcher_id}, {pitch_type})", fontsize=11)
        figure.tight_layout()
        output_path = agents_dir / f"agent_{int(agent_id):04d}.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        written.append(output_path)

    return written


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
            or args.plot_agent_estimates
            or args.prepare_roster
            or args.aggregate_results
            or args.agent_index is not None
        ):
            args.seed = DEFAULT_SEED
        else:
            raise SystemExit(
                "error: --seed is required unless using --list-eligible-pitchers, --dry-run, "
                "--plot-only, --plot-agent-estimates, --prepare-roster, --agent-index, or --aggregate-results"
            )

    if args.prepare_roster:
        prepare_convergence_roster(args)
        return 0

    if args.plot_agent_estimates:
        written = plot_agent_intermediate_estimates(Path(args.output_dir))
        print(
            f"[baseball-convergence] Wrote {len(written)} agent estimate plots under "
            f"{(Path(args.output_dir) / 'agents').resolve()}",
            flush=True,
        )
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
