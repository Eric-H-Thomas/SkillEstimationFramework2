# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Convergence study: JEEDS vs HJEEDS drift toward full-data independent JEEDS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baseball_config import BaseballExperimentConfig, build_baseball_skill_grids
from .baseball_likelihood import compute_baseball_log_likelihood_grid
from .baseball_pitch import (
    PitchObservation,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observation,
    count_agent_pitch_rows,
    filter_roster_by_min_pitches,
    get_agent_pitch_rows,
    list_eligible_pitcher_counts,
    load_processed_statcast,
    resolve_agent_roster,
    select_top_pitchers_by_pitch_count,
)
from .config import DEFAULT_HYPERPRIORS, TruePopulationConfig, _parse_count_buckets
from .estimation import (
    build_discrete_hierarchical_prior,
    fit_population_hyperparameters_map,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from .models import (
    BaseballConvergenceSeedResult,
    ExperimentConfig,
    MethodEstimate,
    StatcastConvergenceAgentResult,
)


DEFAULT_OUTPUT_DIR_CONVERGENCE = Path("HJEEDS/results/baseball_convergence")
DEFAULT_CONVERGENCE_NS = (5, 10)
DEFAULT_NUM_SIGMA_GRID = 21
DEFAULT_NUM_LAMBDA_GRID = 21
DEFAULT_LAMBDA_MIN = 1e-3
DEFAULT_LAMBDA_MAX = 10**3.6
DEFAULT_PITCHER_IDS = (623433, 543037)
DEFAULT_PITCH_TYPES = ("FF",)


def required_min_pitches_for_convergence(
    convergence_ns: Sequence[int],
    max_reference_pitches: int | None,
) -> int:
    """Minimum pitches an agent needs for the largest N and reference cap."""

    required = max(convergence_ns)
    if max_reference_pitches is not None:
        required = max(required, max_reference_pitches)
    return required


@dataclass(frozen=True)
class BaseballConvergenceConfig:
    """Configuration for the Statcast convergence study."""

    base: ExperimentConfig
    pitcher_ids: tuple[int, ...]
    pitch_types: tuple[str, ...]
    convergence_ns: tuple[int, ...]
    max_reference_pitches: int | None
    min_pitches_per_agent: int
    agents: tuple[tuple[int, str], ...]
    agent_pitch_counts: tuple[tuple[int, str, int], ...] = ()
    excluded_agents: tuple[tuple[int, str, int], ...] = ()

    @property
    def environment(self) -> str:
        return self.base.environment

    @property
    def seed_values(self) -> tuple[int, ...]:
        return self.base.seed_values


def _abs_drift(estimate: MethodEstimate, reference: MethodEstimate) -> tuple[float | None, float | None]:
    if estimate.status != "ok" or reference.status != "ok":
        return None, None
    if estimate.posterior_mean_sigma is None or reference.posterior_mean_sigma is None:
        return None, None
    if estimate.posterior_mean_log_lambda is None or reference.posterior_mean_log_lambda is None:
        return None, None
    return (
        abs(estimate.posterior_mean_sigma - reference.posterior_mean_sigma),
        abs(estimate.posterior_mean_log_lambda - reference.posterior_mean_log_lambda),
    )


def _build_pitch_observations(
    agent_rows,
    runtime,
    execution_skills: Sequence[float],
) -> list[PitchObservation]:
    return [
        build_pitch_observation(row, runtime, execution_skills) for _, row in agent_rows.iterrows()
    ]


def _compute_log_likelihood_grid(
    pitch_observations: Sequence[PitchObservation],
    runtime,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
    delta: float,
) -> np.ndarray:
    return compute_baseball_log_likelihood_grid(
        pitch_observations=pitch_observations,
        possible_targets_feet=runtime.grids.possible_targets_feet,
        all_covs=runtime.all_covs,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        delta=delta,
    )


def summarize_convergence_seed(
    seed_result: BaseballConvergenceSeedResult,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize one seed into bucketed (by N) and overall drift rows."""

    bucket_metrics: dict[tuple[str, str, int], list[float]] = {}
    overall_metrics: dict[tuple[str, str], list[float]] = {}

    def add_metric(method: str, metric: str, convergence_n: int, value: float) -> None:
        bucket_metrics.setdefault((method, metric, convergence_n), []).append(value)
        overall_metrics.setdefault((method, metric), []).append(value)

    for result in seed_result.agent_results:
        method_drifts = {
            "jeeds": (
                result.abs_sigma_drift_vs_full_jeeds,
                result.abs_log_lambda_drift_vs_full_jeeds,
            ),
            "hierarchical": (
                result.abs_sigma_drift_vs_full_hierarchical,
                result.abs_log_lambda_drift_vs_full_hierarchical,
            ),
        }
        for method_name, (sigma_drift, lambda_drift) in method_drifts.items():
            if sigma_drift is not None:
                add_metric(method_name, "abs_sigma_drift_vs_full", result.convergence_n, sigma_drift)
            if lambda_drift is not None:
                add_metric(
                    method_name,
                    "abs_log_lambda_drift_vs_full",
                    result.convergence_n,
                    lambda_drift,
                )

    summary_by_n_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, convergence_n), values in sorted(bucket_metrics.items()):
        summary_by_n_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": convergence_n,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level mean over agents with valid drift metrics. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), values in sorted(overall_metrics.items()):
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": (
                    "Seed-level overall mean over agents with valid drift metrics. "
                    "Confidence intervals are added during across-seed aggregation."
                ),
            }
        )

    return summary_by_n_rows, summary_overall_rows


def aggregate_convergence_across_seeds(
    seed_results: Sequence[BaseballConvergenceSeedResult],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate seed-level convergence summaries with normal-approximation 95% CIs."""

    if not seed_results:
        return [], []

    def mean_confidence_interval(values: Sequence[float]) -> tuple[float, float]:
        values_array = np.asarray(values, dtype=float)
        if values_array.size == 0:
            raise ValueError("Cannot compute a confidence interval from an empty set of values.")
        if values_array.size == 1:
            scalar_value = float(values_array[0])
            return scalar_value, scalar_value
        mean_value = float(np.mean(values_array))
        sample_std = float(np.std(values_array, ddof=1))
        standard_error = sample_std / math.sqrt(values_array.size)
        half_width = 1.96 * standard_error
        return mean_value - half_width, mean_value + half_width

    bucket_groups: dict[tuple[str, str, int], dict[str, Any]] = {}
    overall_groups: dict[tuple[str, str], dict[str, Any]] = {}

    for seed_result in seed_results:
        for row in seed_result.summary_by_n_rows:
            key = (str(row["method"]), str(row["metric"]), int(row["count_bucket"]))
            bucket_groups.setdefault(key, {"means": [], "num_agents": 0, "num_seeds": 0})
            bucket_groups[key]["means"].append(float(row["mean"]))
            bucket_groups[key]["num_agents"] += int(row["num_agents"])
            bucket_groups[key]["num_seeds"] += 1

        for row in seed_result.summary_overall_rows:
            key = (str(row["method"]), str(row["metric"]))
            overall_groups.setdefault(key, {"means": [], "num_agents": 0, "num_seeds": 0})
            overall_groups[key]["means"].append(float(row["mean"]))
            overall_groups[key]["num_agents"] += int(row["num_agents"])
            overall_groups[key]["num_seeds"] += 1

    summary_by_n_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, convergence_n), info in sorted(bucket_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        if metric_name.startswith("abs_"):
            ci_lower = max(0.0, ci_lower)
        summary_by_n_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": convergence_n,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), info in sorted(overall_groups.items()):
        mean_values = info["means"]
        ci_lower, ci_upper = mean_confidence_interval(mean_values)
        if metric_name.startswith("abs_"):
            ci_lower = max(0.0, ci_lower)
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": info["num_agents"],
                "mean": float(np.mean(mean_values)),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": (
                    "Across-seed mean of seed-level summary means with a normal-approximation "
                    f"95% CI over {info['num_seeds']} seeds."
                ),
            }
        )

    return summary_by_n_rows, summary_overall_rows


def run_single_baseball_convergence_seed(
    config: BaseballConvergenceConfig,
    seed: int,
) -> BaseballConvergenceSeedResult:
    """Run the convergence sweep for one seed."""

    rng = np.random.default_rng(seed)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    all_data = load_processed_statcast()
    roster = resolve_agent_roster(config.pitcher_ids, config.pitch_types)

    max_build_n = max(config.convergence_ns)
    agent_cache: list[
        tuple[
            StatcastAgentSpec,
            list[PitchObservation],
            list[PitchObservation],
            MethodEstimate,
            int,
        ]
    ] = []

    for agent_spec in roster:
        agent_rows = get_agent_pitch_rows(
            all_data,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
        )
        if len(agent_rows) == 0:
            continue

        if config.max_reference_pitches is None:
            reference_row_count = len(agent_rows)
        else:
            reference_row_count = min(len(agent_rows), config.max_reference_pitches)

        build_n = max(max_build_n, reference_row_count)
        built_rows = agent_rows.iloc[:build_n, :]
        reference_rows = agent_rows.iloc[:reference_row_count, :]

        all_observations = _build_pitch_observations(built_rows, runtime, execution_skills)
        reference_observations = all_observations[:reference_row_count]
        prefix_observations = all_observations
        reference_log_likelihood = _compute_log_likelihood_grid(
            reference_observations,
            runtime,
            sigma_grid,
            log_lambda_grid,
            config.base.delta,
        )
        reference_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=reference_log_likelihood,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        agent_cache.append(
            (
                agent_spec,
                prefix_observations,
                reference_observations,
                reference_estimate,
                len(reference_observations),
            )
        )

    seed_result = BaseballConvergenceSeedResult(
        seed=seed,
        notes="Statcast baseball convergence study (full-data JEEDS reference).",
    )

    for convergence_n in config.convergence_ns:
        agent_records: list[
            tuple[
                StatcastAgentSpec,
                int,
                list[PitchObservation],
                np.ndarray,
                MethodEstimate,
                MethodEstimate,
                int,
            ]
        ] = []

        for (
            agent_spec,
            prefix_observations,
            _reference_observations,
            reference_estimate,
            num_reference_observations,
        ) in agent_cache:
            take_n = min(convergence_n, len(prefix_observations))
            pitch_observations = prefix_observations[:take_n]
            log_likelihood_grid = _compute_log_likelihood_grid(
                pitch_observations,
                runtime,
                sigma_grid,
                log_lambda_grid,
                config.base.delta,
            )
            jeeds_estimate = run_independent_jeeds_baseline(
                log_likelihood_grid=log_likelihood_grid,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )
            agent_records.append(
                (
                    agent_spec,
                    take_n,
                    pitch_observations,
                    log_likelihood_grid,
                    jeeds_estimate,
                    reference_estimate,
                    num_reference_observations,
                )
            )

        discrete_hierarchical_prior = None
        if agent_records:
            fitted_hyperparameters = fit_population_hyperparameters_map(
                config=config.base,
                agent_log_likelihoods=[record[3] for record in agent_records],
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )
            discrete_hierarchical_prior = build_discrete_hierarchical_prior(
                fitted_hyperparameters=fitted_hyperparameters,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )

        for (
            agent_spec,
            take_n,
            pitch_observations,
            log_likelihood_grid,
            jeeds_estimate,
            reference_estimate,
            num_reference_observations,
        ) in agent_records:
            hierarchical_estimate = run_hierarchical_estimator(
                log_likelihood_grid=log_likelihood_grid,
                discrete_prior=discrete_hierarchical_prior,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )
            jeeds_sigma_drift, jeeds_lambda_drift = _abs_drift(jeeds_estimate, reference_estimate)
            hier_sigma_drift, hier_lambda_drift = _abs_drift(hierarchical_estimate, reference_estimate)
            hierarchical_closer_sigma = None
            hierarchical_closer_log_lambda = None
            if jeeds_sigma_drift is not None and hier_sigma_drift is not None:
                hierarchical_closer_sigma = hier_sigma_drift < jeeds_sigma_drift
            if jeeds_lambda_drift is not None and hier_lambda_drift is not None:
                hierarchical_closer_log_lambda = hier_lambda_drift < jeeds_lambda_drift

            seed_result.agent_results.append(
                StatcastConvergenceAgentResult(
                    seed=seed,
                    agent_id=agent_spec.agent_id,
                    pitcher_id=agent_spec.pitcher_id,
                    pitch_type=agent_spec.pitch_type,
                    convergence_n=convergence_n,
                    num_observations=len(pitch_observations),
                    num_reference_observations=num_reference_observations,
                    reference=reference_estimate,
                    jeeds=jeeds_estimate,
                    hierarchical=hierarchical_estimate,
                    abs_sigma_drift_vs_full_jeeds=jeeds_sigma_drift,
                    abs_log_lambda_drift_vs_full_jeeds=jeeds_lambda_drift,
                    abs_sigma_drift_vs_full_hierarchical=hier_sigma_drift,
                    abs_log_lambda_drift_vs_full_hierarchical=hier_lambda_drift,
                    hierarchical_closer_sigma=hierarchical_closer_sigma,
                    hierarchical_closer_log_lambda=hierarchical_closer_log_lambda,
                    notes=(
                        f"Pitcher {agent_spec.pitcher_id} {agent_spec.pitch_type}; "
                        f"N={take_n}; reference pitches={num_reference_observations}."
                    ),
                )
            )

    seed_result.summary_by_n_rows, seed_result.summary_overall_rows = summarize_convergence_seed(
        seed_result
    )
    return seed_result


def build_baseball_skill_grids_from_convergence(
    config: BaseballConvergenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build skill grids using the shared baseball grid helpers."""

    from .baseball_config import build_baseball_skill_grids as _build
    from .baseball_config import BaseballExperimentConfig

    shim = BaseballExperimentConfig(
        base=config.base,
        pitcher_ids=config.pitcher_ids,
        pitch_types=config.pitch_types,
        max_pitches_per_agent=None,
        agents=config.agents,
    )
    return _build(shim)


def build_baseball_convergence_config_from_args(args) -> BaseballConvergenceConfig:
    convergence_ns = _parse_count_buckets(args.convergence_ns)
    pitch_types = tuple(piece.strip() for piece in args.pitch_types.split(",") if piece.strip())
    if not pitch_types:
        raise ValueError("At least one pitch type is required.")

    min_pitches_per_agent = (
        args.min_pitches_per_agent
        if args.min_pitches_per_agent is not None
        else required_min_pitches_for_convergence(convergence_ns, args.max_reference_pitches)
    )

    all_data = load_processed_statcast()
    if args.top_pitchers is not None:
        pitcher_ids = select_top_pitchers_by_pitch_count(
            all_data,
            pitch_types,
            min_pitches=min_pitches_per_agent,
            count=args.top_pitchers,
        )
    else:
        pitcher_ids = tuple(int(piece.strip()) for piece in args.pitcher_ids.split(",") if piece.strip())
        if not pitcher_ids:
            raise ValueError("At least one pitcher id is required (or use --top-pitchers).")

    roster = resolve_agent_roster(pitcher_ids, pitch_types)
    roster, excluded_agents = filter_roster_by_min_pitches(
        roster,
        all_data,
        min_pitches_per_agent,
    )
    if not roster:
        raise ValueError(
            "No agents meet min_pitches_per_agent="
            f"{min_pitches_per_agent}. Excluded: {excluded_agents}. "
            "Use --list-eligible-pitchers to find alternatives or lower the threshold."
        )

    agent_pitch_counts = tuple(
        (spec.pitcher_id, spec.pitch_type, count_agent_pitch_rows(all_data, spec.pitcher_id, spec.pitch_type))
        for spec in roster
    )
    output_dir = Path(args.output_dir)
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=len(roster),
        count_buckets=convergence_ns,
        agents_per_bucket=1,
        delta=args.delta,
        num_sigma_grid=args.num_sigma_grid,
        num_lambda_grid=args.num_lambda_grid,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        output_dir=output_dir,
        dry_run=args.dry_run,
        min_success_regions=2,
        max_success_regions=6,
        min_region_width=0.25,
        hyperpriors=DEFAULT_HYPERPRIORS,
        true_population=TruePopulationConfig(
            mean_log_sigma=math.log(0.5),
            mean_log_lambda=math.log(1.0),
            tau_eta=0.35,
            tau_rho=1.0,
            correlation=-0.5,
        ),
    )
    return BaseballConvergenceConfig(
        base=base,
        pitcher_ids=pitcher_ids,
        pitch_types=pitch_types,
        convergence_ns=convergence_ns,
        max_reference_pitches=args.max_reference_pitches,
        min_pitches_per_agent=min_pitches_per_agent,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster),
        agent_pitch_counts=agent_pitch_counts,
        excluded_agents=excluded_agents,
    )


def print_eligible_pitchers(
    pitch_types: Sequence[str],
    *,
    min_pitches: int,
    limit: int,
) -> None:
    all_data = load_processed_statcast()
    rows = list_eligible_pitcher_counts(
        all_data,
        pitch_types,
        min_pitches=min_pitches,
        limit=limit,
    )
    print(f"Eligible agents with >= {min_pitches} pitches (top {limit}):")
    if not rows:
        print("  (none)")
        return
    for pitcher_id, pitch_type, pitch_count in rows:
        print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")


def planned_convergence_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "agent_level_csv": output_dir / "convergence_agent_level_results.csv",
        "summary_by_n_csv": output_dir / "summary_by_N.csv",
        "summary_overall_csv": output_dir / "summary_overall.csv",
        "drift_plot": output_dir / "drift_by_N.png",
    }


def print_baseball_convergence_dry_run_summary(config: BaseballConvergenceConfig) -> None:
    sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    paths = planned_convergence_output_paths(config.base.output_dir)
    max_n = max(config.convergence_ns)
    pitch_builds = len(config.agents) * max(
        max_n,
        config.max_reference_pitches or 0,
    )

    print("=== DRY RUN: Baseball HJEEDS Convergence Study (Phase 2) ===")
    print("No RNN inference or estimation will run.")
    print()
    print(f"Environment: {config.environment}")
    print(f"Seeds: {config.seed_values}")
    print(f"Agents (pitcher, pitch type): {config.agents}")
    if config.agent_pitch_counts:
        print("Agent pitch counts:")
        for pitcher_id, pitch_type, pitch_count in config.agent_pitch_counts:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
    if config.excluded_agents:
        print(f"Excluded (below min_pitches_per_agent={config.min_pitches_per_agent}):")
        for pitcher_id, pitch_type, pitch_count in config.excluded_agents:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
    print(f"Min pitches per agent: {config.min_pitches_per_agent}")
    print(f"Convergence N values (cumulative newest-first prefix): {config.convergence_ns}")
    print(
        "Reference: independent JEEDS posterior mean on "
        f"{'all available' if config.max_reference_pitches is None else config.max_reference_pitches} "
        "pitches per agent."
    )
    print(f"Delta: {config.base.delta}")
    print(f"Execution skill grid: {len(sigma_grid)} points [{sigma_grid[0]:.3f}, {sigma_grid[-1]:.3f}]")
    print(f"Log-lambda grid: {len(log_lambda_grid)} points")
    print(f"Output directory: {config.base.output_dir.resolve()}")
    print("Planned artifacts:")
    for label, path in paths.items():
        print(f"  - {label}: {path}")
    print()
    print("Planned pipeline:")
    print("  1. Load ProcessedData-From-GivenFiles.pkl")
    print("  2. Per agent, build pitch surfaces once for max(N) (and reference cap if set)")
    print("  3. Fit full-data independent JEEDS reference per agent")
    print("  4. For each N, subsample prefix-N pitches (game_date descending)")
    print("  5. Fit independent JEEDS + hierarchical HJEEDS on the same prefix")
    print("  6. Record drift |estimate_N - reference_full| for sigma and log-lambda")
    print("  7. Write convergence_agent_level_results.csv, summary_by_N.csv, drift_by_N.png")
    print()
    print(
        f"Upper-bound pitch-surface builds (no cache reuse): ~{pitch_builds} "
        f"({len(config.agents)} agents x max build depth)."
    )
