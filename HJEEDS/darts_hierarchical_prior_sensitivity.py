# This file has been fully verified by a human researcher as of 05/20/26 at 10:14 AM MT.
"""Run hyperprior-robustness sweeps for hierarchical 1D darts.

This script wraps ``HJEEDS/darts_hierarchical_vs_jeeds.py`` and reruns the
same simulated experiment under misspecified empirical-Bayes hyperpriors. The
default standalone study is a 60-condition robustness grid:

- focus area: average skill, population spread, correlation, combined
- bias: strong reverse, moderate reverse, unbiased, moderate adverse, strong adverse
- confidence: weak, default, strong

Downstream ablations can request a smaller representative preset containing
the default, moderate combined misspecification, and strong combined
misspecification conditions. Each condition gets its own ordinary experiment
output directory, and this script also writes combined CSVs plus a compact plot
for the lowest-count bucket, where prior sensitivity should be easiest to see.
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


# Ensure the repository root is importable when this file is executed directly
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS import darts_hierarchical_vs_jeeds as base_experiment


# This script answers the paper's robustness question: if H-JEEDS benefits from
# a shared prior in low-data settings, how sensitive is that benefit to
# misspecification in the mean skill, population spread, and correlation
# hyperpriors?


DEFAULT_OUTPUT_DIR = Path("HJEEDS/results/hierarchical_darts_prior_sensitivity")
DEFAULT_COUNT_BUCKETS = (5, 10, 25, 100, 1000)

CONDITION_PRESET_FULL_60 = "full_60"
CONDITION_PRESET_REPRESENTATIVE = "representative"
CONDITION_PRESETS = (CONDITION_PRESET_FULL_60, CONDITION_PRESET_REPRESENTATIVE)
DEFAULT_PRIOR_SENSITIVITY_CONDITION_PRESET = CONDITION_PRESET_FULL_60

DEFAULT_WEAK_STD_MULTIPLIER = 3.0
DEFAULT_STRONG_STD_MULTIPLIER = 1.0 / 3.0
DEFAULT_MODERATE_BIAS_SD_UNITS = 0.5
DEFAULT_STRONG_BIAS_SD_UNITS = 1.5

DEFAULT_STRONG_REVERSE_CORRELATION_R_CENTER = -0.9
DEFAULT_MODERATE_REVERSE_CORRELATION_R_CENTER = -0.75
DEFAULT_UNBIASED_CORRELATION_R_CENTER = -0.5
DEFAULT_MODERATE_ADVERSE_CORRELATION_R_CENTER = 0.0
DEFAULT_STRONG_ADVERSE_CORRELATION_R_CENTER = 0.5

CONDITION_METADATA_HEADER = [
    "condition_preset",
    "condition_slug",
    "condition_label",
    "focus_slug",
    "focus_label",
    "bias_slug",
    "bias_label",
    "confidence_slug",
    "confidence_label",
    "confidence_std_multiplier",
    "average_skill_bias_sd_units",
    "population_spread_bias_sd_units",
    "correlation_r_center",
    "scale_average_skill_confidence",
    "scale_population_spread_confidence",
    "scale_correlation_confidence",
    "hyperprior_mu_eta",
    "hyperprior_mu_rho",
    "hyperprior_mu_sigma",
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
class _ConfidenceLevel:
    """One confidence level used to scale a scoped set of prior SDs."""

    label: str
    slug: str
    std_multiplier: float


@dataclass(frozen=True)
class _BiasLevel:
    """One signed bias level shared by the robustness grid."""

    label: str
    slug: str
    signed_bias_sd_units: float
    correlation_r_center: float


@dataclass(frozen=True)
class _FocusArea:
    """One group of hyperprior components being stressed."""

    label: str
    slug: str


@dataclass(frozen=True)
class PriorSensitivityCondition:
    """One concrete hyperprior-robustness condition."""

    condition_preset: str
    focus_label: str
    focus_slug: str
    bias_label: str
    bias_slug: str
    confidence_label: str
    confidence_slug: str
    confidence_std_multiplier: float
    average_skill_bias_sd_units: float
    population_spread_bias_sd_units: float
    correlation_r_center: float
    scale_average_skill_confidence: bool
    scale_population_spread_confidence: bool
    scale_correlation_confidence: bool
    condition_slug_override: str | None = None
    condition_label_override: str | None = None

    @property
    def condition_slug(self) -> str:
        """Return a filename-safe identifier for this sensitivity condition."""

        if self.condition_slug_override is not None:
            return self.condition_slug_override
        return f"{self.focus_slug}__{self.bias_slug}__{self.confidence_slug}"

    @property
    def condition_label(self) -> str:
        """Return a compact human-facing condition label."""

        if self.condition_label_override is not None:
            return self.condition_label_override
        return f"{self.focus_label}: {self.bias_label}, {self.confidence_label} confidence"


@dataclass(frozen=True)
class PriorSensitivityGridResult:
    """Combined rows produced by one prior-sensitivity grid run."""

    output_dir: Path
    conditions: tuple[PriorSensitivityCondition, ...]
    condition_rows: list[dict[str, Any]]
    agent_rows: list[dict[str, Any]]
    bucket_rows: list[dict[str, Any]]
    overall_rows: list[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the hyperprior-robustness sweep."""

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--seed",
        type=base_experiment.parse_seed_argument,
        required=True,
        help="Base seed used to derive per-run seeds. Use 'default' for 12345.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=base_experiment.DEFAULT_NUM_SEEDS,
        help=f"Number of random seeds to run for each prior condition (default: {base_experiment.DEFAULT_NUM_SEEDS}).",
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
            "agents_per_bucket so the runner can use a five-bucket setup."
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
    parser.add_argument(
        "--min-success-regions",
        "--min-regions",
        dest="min_success_regions",
        type=int,
        default=base_experiment.DEFAULT_MIN_SUCCESS_REGIONS,
    )
    parser.add_argument(
        "--max-success-regions",
        "--max-regions",
        dest="max_success_regions",
        type=int,
        default=base_experiment.DEFAULT_MAX_SUCCESS_REGIONS,
    )
    parser.add_argument("--min-region-width", type=float, default=base_experiment.DEFAULT_MIN_REGION_WIDTH)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for combined sensitivity artifacts and condition subdirectories.",
    )
    parser.add_argument(
        "--condition-preset",
        choices=CONDITION_PRESETS,
        default=DEFAULT_PRIOR_SENSITIVITY_CONDITION_PRESET,
        help=(
            "Condition set to run. full_60 is the standalone robustness study; "
            "representative is the three-condition downstream-ablation preset."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned robustness workload and stop before simulation/inference.",
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
        min_success_regions=args.min_success_regions,
        max_success_regions=args.max_success_regions,
        min_region_width=args.min_region_width,
    )
    return base_experiment.build_config_from_args(base_args)


def _confidence_levels() -> tuple[_ConfidenceLevel, ...]:
    """Return weak/default/strong confidence settings."""

    return (
        _ConfidenceLevel("weak", "weak", DEFAULT_WEAK_STD_MULTIPLIER),
        _ConfidenceLevel("default", "default", 1.0),
        _ConfidenceLevel("strong", "strong", DEFAULT_STRONG_STD_MULTIPLIER),
    )


def _bias_levels() -> tuple[_BiasLevel, ...]:
    """Return the signed five-level bias grid."""

    moderate_bias = DEFAULT_MODERATE_BIAS_SD_UNITS
    strong_bias = DEFAULT_STRONG_BIAS_SD_UNITS
    return (
        _BiasLevel(
            "strong reverse misspecification",
            "strong_reverse_misspecification",
            -strong_bias,
            DEFAULT_STRONG_REVERSE_CORRELATION_R_CENTER,
        ),
        _BiasLevel(
            "moderate reverse misspecification",
            "moderate_reverse_misspecification",
            -moderate_bias,
            DEFAULT_MODERATE_REVERSE_CORRELATION_R_CENTER,
        ),
        _BiasLevel(
            "unbiased",
            "unbiased",
            0.0,
            DEFAULT_UNBIASED_CORRELATION_R_CENTER,
        ),
        _BiasLevel(
            "moderate adverse misspecification",
            "moderate_adverse_misspecification",
            moderate_bias,
            DEFAULT_MODERATE_ADVERSE_CORRELATION_R_CENTER,
        ),
        _BiasLevel(
            "strong adverse misspecification",
            "strong_adverse_misspecification",
            strong_bias,
            DEFAULT_STRONG_ADVERSE_CORRELATION_R_CENTER,
        ),
    )


def _focus_areas() -> tuple[_FocusArea, ...]:
    """Return the four hyperprior component groups used by the full study."""

    return (
        _FocusArea("average skill", "average_skill"),
        _FocusArea("population spread", "population_spread"),
        _FocusArea("correlation", "correlation"),
        _FocusArea("combined", "combined"),
    )


def _validate_correlation_center(field_name: str, value: float) -> None:
    """Fail early when a requested correlation center cannot be transformed."""

    if not -1.0 < value < 1.0:
        raise ValueError(f"{field_name} must be strictly between -1 and 1. Received: {value}.")


def _condition_for_focus(
    *,
    preset: str,
    focus: _FocusArea,
    bias: _BiasLevel,
    confidence: _ConfidenceLevel,
    unbiased_correlation_r_center: float,
) -> PriorSensitivityCondition:
    """Build one full-grid condition from focus, bias, and confidence levels."""

    if focus.slug == "average_skill":
        average_skill_bias = bias.signed_bias_sd_units
        population_spread_bias = 0.0
        correlation_r_center = unbiased_correlation_r_center
        scale_average_skill_confidence = True
        scale_population_spread_confidence = False
        scale_correlation_confidence = False
    elif focus.slug == "population_spread":
        average_skill_bias = 0.0
        population_spread_bias = bias.signed_bias_sd_units
        correlation_r_center = unbiased_correlation_r_center
        scale_average_skill_confidence = False
        scale_population_spread_confidence = True
        scale_correlation_confidence = False
    elif focus.slug == "correlation":
        average_skill_bias = 0.0
        population_spread_bias = 0.0
        correlation_r_center = bias.correlation_r_center
        scale_average_skill_confidence = False
        scale_population_spread_confidence = False
        scale_correlation_confidence = True
    elif focus.slug == "combined":
        average_skill_bias = bias.signed_bias_sd_units
        population_spread_bias = bias.signed_bias_sd_units
        correlation_r_center = bias.correlation_r_center
        scale_average_skill_confidence = True
        scale_population_spread_confidence = True
        scale_correlation_confidence = True
    else:
        raise ValueError(f"Unknown focus area: {focus.slug}")

    return PriorSensitivityCondition(
        condition_preset=preset,
        focus_label=focus.label,
        focus_slug=focus.slug,
        bias_label=bias.label,
        bias_slug=bias.slug,
        confidence_label=confidence.label,
        confidence_slug=confidence.slug,
        confidence_std_multiplier=confidence.std_multiplier,
        average_skill_bias_sd_units=average_skill_bias,
        population_spread_bias_sd_units=population_spread_bias,
        correlation_r_center=correlation_r_center,
        scale_average_skill_confidence=scale_average_skill_confidence,
        scale_population_spread_confidence=scale_population_spread_confidence,
        scale_correlation_confidence=scale_correlation_confidence,
    )


def build_full_60_conditions() -> tuple[PriorSensitivityCondition, ...]:
    """Return the full 4 x 5 x 3 robustness grid."""

    conditions: list[PriorSensitivityCondition] = []
    unbiased_correlation_r_center = DEFAULT_UNBIASED_CORRELATION_R_CENTER
    for focus in _focus_areas():
        for bias in _bias_levels():
            for confidence in _confidence_levels():
                conditions.append(
                    _condition_for_focus(
                        preset=CONDITION_PRESET_FULL_60,
                        focus=focus,
                        bias=bias,
                        confidence=confidence,
                        unbiased_correlation_r_center=unbiased_correlation_r_center,
                    )
                )
    return tuple(conditions)


def build_representative_conditions() -> tuple[PriorSensitivityCondition, ...]:
    """Return the three-condition preset used by downstream ablations."""

    focus = _FocusArea("combined", "combined")
    default_confidence = _ConfidenceLevel("default", "default", 1.0)
    strong_confidence = _ConfidenceLevel("strong", "strong", DEFAULT_STRONG_STD_MULTIPLIER)
    moderate_bias = DEFAULT_MODERATE_BIAS_SD_UNITS
    strong_bias = DEFAULT_STRONG_BIAS_SD_UNITS

    return (
        PriorSensitivityCondition(
            condition_preset=CONDITION_PRESET_REPRESENTATIVE,
            focus_label=focus.label,
            focus_slug=focus.slug,
            bias_label="unbiased",
            bias_slug="unbiased",
            confidence_label=default_confidence.label,
            confidence_slug=default_confidence.slug,
            confidence_std_multiplier=default_confidence.std_multiplier,
            average_skill_bias_sd_units=0.0,
            population_spread_bias_sd_units=0.0,
            correlation_r_center=DEFAULT_UNBIASED_CORRELATION_R_CENTER,
            scale_average_skill_confidence=True,
            scale_population_spread_confidence=True,
            scale_correlation_confidence=True,
            condition_slug_override="default",
            condition_label_override="default",
        ),
        PriorSensitivityCondition(
            condition_preset=CONDITION_PRESET_REPRESENTATIVE,
            focus_label=focus.label,
            focus_slug=focus.slug,
            bias_label="moderate adverse misspecification",
            bias_slug="moderate_adverse_misspecification",
            confidence_label=default_confidence.label,
            confidence_slug=default_confidence.slug,
            confidence_std_multiplier=default_confidence.std_multiplier,
            average_skill_bias_sd_units=moderate_bias,
            population_spread_bias_sd_units=moderate_bias,
            correlation_r_center=DEFAULT_MODERATE_ADVERSE_CORRELATION_R_CENTER,
            scale_average_skill_confidence=True,
            scale_population_spread_confidence=True,
            scale_correlation_confidence=True,
            condition_slug_override="moderate_combined_misspecification",
            condition_label_override="moderate combined misspecification",
        ),
        PriorSensitivityCondition(
            condition_preset=CONDITION_PRESET_REPRESENTATIVE,
            focus_label=focus.label,
            focus_slug=focus.slug,
            bias_label="strong adverse misspecification",
            bias_slug="strong_adverse_misspecification",
            confidence_label=strong_confidence.label,
            confidence_slug=strong_confidence.slug,
            confidence_std_multiplier=strong_confidence.std_multiplier,
            average_skill_bias_sd_units=strong_bias,
            population_spread_bias_sd_units=strong_bias,
            correlation_r_center=DEFAULT_STRONG_ADVERSE_CORRELATION_R_CENTER,
            scale_average_skill_confidence=True,
            scale_population_spread_confidence=True,
            scale_correlation_confidence=True,
            condition_slug_override="strong_combined_misspecification",
            condition_label_override="strong combined misspecification",
        ),
    )


def build_sensitivity_conditions(args: argparse.Namespace) -> tuple[PriorSensitivityCondition, ...]:
    """Return the requested condition preset."""

    preset = getattr(args, "condition_preset", DEFAULT_PRIOR_SENSITIVITY_CONDITION_PRESET)
    if preset == CONDITION_PRESET_FULL_60:
        conditions = build_full_60_conditions()
    elif preset == CONDITION_PRESET_REPRESENTATIVE:
        return build_representative_conditions()
    else:
        raise ValueError(f"Unknown condition preset: {preset}")
    return conditions


def build_condition_hyperpriors(
    base_hyperpriors: base_experiment.HyperpriorConfig,
    condition: PriorSensitivityCondition,
) -> base_experiment.HyperpriorConfig:
    """Create the hyperprior settings for one robustness condition.

    Average-skill bias shifts the population mean prior toward worse or better
    average skill. Population-spread bias shifts the log-tau prior centers
    toward more or less between-agent variation. Correlation bias directly
    moves the prior center for the execution/decision-skill correlation. The
    confidence multiplier is scoped to the focus area instead of always scaling
    every hyperprior SD.
    """

    base_mu_eta, base_mu_rho = base_hyperpriors.mean_vector
    base_mu_eta_sd = math.sqrt(float(base_hyperpriors.covariance_diagonal[0]))
    base_mu_rho_sd = math.sqrt(float(base_hyperpriors.covariance_diagonal[1]))
    base_log_tau_eta_mean = float(base_hyperpriors.log_tau_eta_mean)
    base_log_tau_rho_mean = float(base_hyperpriors.log_tau_rho_mean)
    base_log_tau_eta_sd = float(base_hyperpriors.log_tau_eta_sd)
    base_log_tau_rho_sd = float(base_hyperpriors.log_tau_rho_sd)

    average_skill_bias = condition.average_skill_bias_sd_units
    population_spread_bias = condition.population_spread_bias_sd_units
    confidence_multiplier = condition.confidence_std_multiplier

    mean_shift_eta = average_skill_bias * base_mu_eta_sd
    mean_shift_rho = -average_skill_bias * base_mu_rho_sd
    log_tau_eta_shift = population_spread_bias * base_log_tau_eta_sd
    log_tau_rho_shift = population_spread_bias * base_log_tau_rho_sd

    if condition.scale_average_skill_confidence:
        mu_eta_sd = base_mu_eta_sd * confidence_multiplier
        mu_rho_sd = base_mu_rho_sd * confidence_multiplier
    else:
        mu_eta_sd = base_mu_eta_sd
        mu_rho_sd = base_mu_rho_sd

    if condition.scale_population_spread_confidence:
        log_tau_eta_sd = base_log_tau_eta_sd * confidence_multiplier
        log_tau_rho_sd = base_log_tau_rho_sd * confidence_multiplier
    else:
        log_tau_eta_sd = base_log_tau_eta_sd
        log_tau_rho_sd = base_log_tau_rho_sd

    if condition.scale_correlation_confidence:
        s_r = base_hyperpriors.s_r * confidence_multiplier
    else:
        s_r = base_hyperpriors.s_r

    _validate_correlation_center("condition.correlation_r_center", condition.correlation_r_center)

    return base_experiment.HyperpriorConfig(
        mean_vector=(base_mu_eta + mean_shift_eta, base_mu_rho + mean_shift_rho),
        covariance_diagonal=(max(mu_eta_sd, 1e-9) ** 2, max(mu_rho_sd, 1e-9) ** 2),
        log_tau_eta_mean=base_log_tau_eta_mean + log_tau_eta_shift,
        log_tau_eta_sd=max(log_tau_eta_sd, 1e-9),
        log_tau_rho_mean=base_log_tau_rho_mean + log_tau_rho_shift,
        log_tau_rho_sd=max(log_tau_rho_sd, 1e-9),
        m_r=math.atanh(condition.correlation_r_center),
        s_r=max(s_r, 1e-9),
    )


def condition_metadata_row(
    condition: PriorSensitivityCondition,
    hyperpriors: base_experiment.HyperpriorConfig,
) -> dict[str, Any]:
    """Return CSV metadata describing one condition's concrete hyperpriors."""

    mu_eta, mu_rho = hyperpriors.mean_vector
    mu_eta_sd = math.sqrt(float(hyperpriors.covariance_diagonal[0]))
    mu_rho_sd = math.sqrt(float(hyperpriors.covariance_diagonal[1]))

    return {
        "condition_preset": condition.condition_preset,
        "condition_slug": condition.condition_slug,
        "condition_label": condition.condition_label,
        "focus_slug": condition.focus_slug,
        "focus_label": condition.focus_label,
        "bias_slug": condition.bias_slug,
        "bias_label": condition.bias_label,
        "confidence_slug": condition.confidence_slug,
        "confidence_label": condition.confidence_label,
        "confidence_std_multiplier": condition.confidence_std_multiplier,
        "average_skill_bias_sd_units": condition.average_skill_bias_sd_units,
        "population_spread_bias_sd_units": condition.population_spread_bias_sd_units,
        "correlation_r_center": condition.correlation_r_center,
        "scale_average_skill_confidence": condition.scale_average_skill_confidence,
        "scale_population_spread_confidence": condition.scale_population_spread_confidence,
        "scale_correlation_confidence": condition.scale_correlation_confidence,
        "hyperprior_mu_eta": mu_eta,
        "hyperprior_mu_rho": mu_rho,
        "hyperprior_mu_sigma": math.exp(mu_eta),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})


def run_condition(
    base_config: base_experiment.ExperimentConfig,
    condition: PriorSensitivityCondition,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run all seeds for one sensitivity condition and write per-condition artifacts."""

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

    prefix = condition_metadata_row(condition, hyperpriors)
    combined_agent_rows = [
        {**prefix, **base_experiment._agent_result_to_row(result)}
        for result in all_agent_results
    ]
    combined_bucket_rows = [{**prefix, **row} for row in summary_by_bucket_rows]
    combined_overall_rows = [{**prefix, **row} for row in summary_overall_rows]

    return combined_agent_rows, combined_bucket_rows, combined_overall_rows, prefix


def run_sensitivity_grid(
    config: base_experiment.ExperimentConfig,
    conditions: Sequence[PriorSensitivityCondition],
) -> PriorSensitivityGridResult:
    """Run a prior-sensitivity grid and write its combined artifacts."""

    condition_tuple = tuple(conditions)
    all_agent_rows: list[dict[str, Any]] = []
    all_bucket_rows: list[dict[str, Any]] = []
    all_overall_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    for condition in condition_tuple:
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
    plot_lowest_bucket_heatmap(output_dir / LOWEST_BUCKET_HEATMAP_FILENAME, all_bucket_rows, condition_tuple)

    print(f"[prior-sensitivity] Wrote combined results to {output_dir.resolve()}", flush=True)
    return PriorSensitivityGridResult(
        output_dir=output_dir,
        conditions=condition_tuple,
        condition_rows=condition_rows,
        agent_rows=all_agent_rows,
        bucket_rows=all_bucket_rows,
        overall_rows=all_overall_rows,
    )


def _unique_in_order(values: Sequence[str]) -> list[str]:
    """Return unique strings in first-seen order."""

    unique_values: list[str] = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def _condition_lookup_by_slug(
    conditions: Sequence[PriorSensitivityCondition],
) -> dict[str, PriorSensitivityCondition]:
    """Return a condition lookup for plot labels and ordering."""

    return {condition.condition_slug: condition for condition in conditions}


def plot_lowest_bucket_heatmap(
    output_path: Path,
    summary_by_bucket_rows: Sequence[dict[str, Any]],
    conditions: Sequence[PriorSensitivityCondition],
) -> None:
    """Plot hierarchical error by focus, bias, and confidence for the lowest bucket."""

    import matplotlib

    # Force noninteractive backend; Import pyplot only after the backend has been set
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # If there are no summary rows, there is nothing to plot
    if not summary_by_bucket_rows:
        return

    # Collect all count buckets that have hierarchical rows and valid bucket labels
    bucket_values = sorted(
        {
            int(row["count_bucket"])
            for row in summary_by_bucket_rows
            if row.get("method") == "hierarchical" and row.get("count_bucket") not in (None, "")
        }
    )
    # If no hierarchical bucket rows were found, there is nothing useful to plot
    if not bucket_values:
        return

    # Choose the smallest observation-count bucket because prior sensitivity should be most visible there
    selected_bucket = bucket_values[0]

    # Build a dictionary so a CSV row's condition slug can recover the original condition object
    condition_by_slug = _condition_lookup_by_slug(conditions)

    # Preserve the first-seen order of focus areas so subplot rows follow the condition definition order
    focus_slugs = _unique_in_order([condition.focus_slug for condition in conditions])

    # Map focus slugs such as "average_skill" to display labels such as "average skill"
    focus_labels = {
        condition.focus_slug: condition.focus_label
        for condition in conditions
    }

    # Define the two error metrics shown as columns in the figure
    metric_panels = [
        ("abs_sigma_error", "Execution Skill Error"),
        ("abs_log_lambda_error", "Log-Decision Skill Error"),
    ]

    # Create a subplot grid with one row per focus area and one column per metric
    figure, axes = plt.subplots(
        len(focus_slugs), # One row per focus area
        len(metric_panels), # One column per metric
        figsize=(14, max(3.8, 3.2 * len(focus_slugs))),
        squeeze=False, # Always return a 2D axes array, even if there is only one row or one column
        constrained_layout=True, # Automatically reduce label/title overlap
    )

    # Iterate through focus areas; focus_index selects the subplot row
    for focus_index, focus_slug in enumerate(focus_slugs):

        # Keep only the conditions belonging to this focus area
        focus_conditions = [
            condition
            for condition in conditions
            if condition.focus_slug == focus_slug
        ]

        # Preserve confidence-label order for the heatmap's y-axis
        confidence_labels = _unique_in_order([condition.confidence_label for condition in focus_conditions])

        # Preserve bias-label order for the heatmap's x-axis
        bias_labels = _unique_in_order([condition.bias_label for condition in focus_conditions])

        # Map each confidence label to its row index in the heatmap array
        confidence_positions = {label: index for index, label in enumerate(confidence_labels)}

        # Map each bias label to its column index in the heatmap array
        bias_positions = {label: index for index, label in enumerate(bias_labels)}

        # Iterate through metrics; metric_index selects the subplot column
        for metric_index, (metric_name, title) in enumerate(metric_panels):

            # Select the axis for this focus x metric panel
            axis = axes[focus_index][metric_index]

            # Allocate the heatmap matrix; NaN marks cells that have not received data
            heatmap = np.full((len(confidence_labels), len(bias_labels)), np.nan, dtype=float)

            # Scan the combined summary rows and place matching means into the heatmap
            for row in summary_by_bucket_rows:
                # Ignore independent-JEEDS rows; this plot is about hierarchical prior robustness
                if row.get("method") != "hierarchical":
                    continue
                # Ignore rows for the other metric
                if row.get("metric") != metric_name:
                    continue
                # Ignore rows for count buckets other than the lowest bucket selected above
                if int(row["count_bucket"]) != selected_bucket:
                    continue

                # Read the condition slug from the row; missing slugs become an empty string
                condition_slug = str(row.get("condition_slug", ""))
                # Recover the condition object for label fallback and focus checking
                condition = condition_by_slug.get(condition_slug)
                # Prefer the row's focus slug, but fall back to the condition object if needed
                row_focus_slug = str(row.get("focus_slug", condition.focus_slug if condition else ""))
                # Skip rows that belong to another focus area
                if row_focus_slug != focus_slug:
                    continue

                # Prefer the row's confidence label, but fall back to the condition object if needed
                confidence = str(row.get("confidence_label", condition.confidence_label if condition else ""))
                # Prefer the row's bias label, but fall back to the condition object if needed
                bias = str(row.get("bias_label", condition.bias_label if condition else ""))
                # Skip rows whose labels are not part of this focus area's heatmap layout
                if confidence not in confidence_positions or bias not in bias_positions:
                    continue

                # Place the mean error into the correct confidence x bias heatmap cell
                heatmap[confidence_positions[confidence], bias_positions[bias]] = float(row["mean"])

            # Render the numeric heatmap as an image
            image = axis.imshow(heatmap, cmap="viridis")
            # Title the panel with focus area, metric name, and selected bucket
            axis.set_title(f"{focus_labels[focus_slug].title()} - {title}\nBucket {selected_bucket}")
            # Put x-axis ticks at each bias column
            axis.set_xticks(range(len(bias_labels)))
            # Label x-axis ticks with bias labels and rotate them so long labels fit
            axis.set_xticklabels(bias_labels, rotation=30, ha="right", fontsize=8)
            # Put y-axis ticks at each confidence row
            axis.set_yticks(range(len(confidence_labels)))
            # Label y-axis ticks with confidence labels
            axis.set_yticklabels(confidence_labels)
            # Label the x-axis as the prior-bias dimension
            axis.set_xlabel("Prior bias")
            # Label the y-axis as the prior-confidence dimension
            axis.set_ylabel("Prior confidence")

            # Extract only real values so empty NaN cells do not affect text color
            finite_values = heatmap[np.isfinite(heatmap)]
            # Use the mean finite value as a simple threshold for light/dark annotation text
            text_threshold = float(np.mean(finite_values)) if finite_values.size else 0.0
            # Iterate over heatmap row indices
            for row_index in range(heatmap.shape[0]):
                # Iterate over heatmap column indices
                for col_index in range(heatmap.shape[1]):
                    # Read the plotted value for this heatmap cell
                    value = heatmap[row_index, col_index]
                    # Only annotate cells that contain real data
                    if np.isfinite(value):
                        # Write the mean error value into the center of the heatmap cell
                        axis.text(
                            # x-position is the column index
                            col_index,
                            # y-position is the row index
                            row_index,
                            # Format the number to three decimal places
                            f"{value:.3f}",
                            # Center the text horizontally in the cell
                            ha="center",
                            # Center the text vertically in the cell
                            va="center",
                            # Use white text on darker/higher cells and black text otherwise
                            color="white" if value > text_threshold else "black",
                            # Keep the annotation small enough for dense condition labels
                            fontsize=8,
                        )

            # Add a colorbar for this panel so colors can be read quantitatively
            figure.colorbar(image, ax=axis, shrink=0.85)

    # Ensure the output directory exists before saving the figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write the final heatmap image to disk
    figure.savefig(output_path, dpi=300)
    # Close the figure so batch runs do not accumulate Matplotlib state
    plt.close(figure)


def _seed_values_label(seed_values: Sequence[int]) -> str:
    """Return a readable seed description for dry-run output."""

    if len(seed_values) <= 10:
        return str(tuple(seed_values))
    return f"{len(seed_values)} seeds ({seed_values[0]} through {seed_values[-1]})"


def print_dry_run_summary(
    config: base_experiment.ExperimentConfig,
    conditions: Sequence[PriorSensitivityCondition],
) -> None:
    """Report the planned sensitivity workload without running inference."""

    print("=== DRY RUN: 1D Darts Hyperprior Robustness ===")
    print("No simulation or inference functions will be executed.")
    print()
    print(f"Seeds per condition: {_seed_values_label(config.seed_values)}")
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
            f"{condition.condition_slug}: focus={condition.focus_label}, "
            f"bias={condition.bias_label}, confidence={condition.confidence_label}, "
            f"avg_bias_sd={condition.average_skill_bias_sd_units:g}, "
            f"spread_bias_sd={condition.population_spread_bias_sd_units:g}, "
            f"r_center={condition.correlation_r_center:g}, "
            f"mu=({metadata['hyperprior_mu_eta']:.3f}, {metadata['hyperprior_mu_rho']:.3f}), "
            f"mu_sd=({metadata['hyperprior_mu_eta_sd']:.3f}, {metadata['hyperprior_mu_rho_sd']:.3f}), "
            f"tau_median=({metadata['hyperprior_tau_eta_median']:.3f}, "
            f"{metadata['hyperprior_tau_rho_median']:.3f}), "
            f"log_tau_sd=({metadata['hyperprior_log_tau_eta_sd']:.3f}, "
            f"{metadata['hyperprior_log_tau_rho_sd']:.3f}), "
            f"s_r={metadata['hyperprior_s_r']:.3f}"
        )
    print()
    print("Planned combined artifacts:")
    print(f"  - {config.output_dir / CONDITION_METADATA_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_AGENT_LEVEL_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_SUMMARY_BY_BUCKET_FILENAME}")
    print(f"  - {config.output_dir / COMBINED_SUMMARY_OVERALL_FILENAME}")
    print(f"  - {config.output_dir / LOWEST_BUCKET_HEATMAP_FILENAME}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested prior-sensitivity sweep."""

    args = parse_args(argv)
    config = build_base_config_from_args(args)
    conditions = build_sensitivity_conditions(args)

    if config.dry_run:
        print_dry_run_summary(config, conditions)
        return 0

    run_sensitivity_grid(config, conditions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
