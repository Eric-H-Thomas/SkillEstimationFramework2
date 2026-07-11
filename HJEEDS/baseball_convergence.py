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
from .baseball_hyperpriors import resolve_baseball_hyperpriors
from .baseball_pitch import (
    PitchObservation,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observation,
    count_agent_pitch_rows,
    filter_roster_by_min_pitches,
    get_agent_pitch_rows,
)
from .baseball_roster import (
    BaseballRosterSelection,
    load_statcast_for_roster,
    parse_pitch_types,
    resolve_baseball_roster,
    validate_roster_selection,
)
from .config import TruePopulationConfig, _parse_count_buckets
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
    season_year: int | None
    pitcher_ids: tuple[int, ...]
    pitch_types: tuple[str, ...]
    convergence_ns: tuple[int, ...]
    max_reference_pitches: int | None
    min_pitches_per_agent: int
    agent_specs: tuple[StatcastAgentSpec, ...]
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
    """Run the convergence sweep for one seed.

    Note: ``seed`` is threaded into ``build_baseball_runtime`` for API compatibility with
    the darts pipeline, but Statcast JEEDS/H-JEEDS likelihoods are unaffected. Execution
    noise PDFs are evaluated via ``multivariate_normal.pdf`` (deterministic given mean/cov);
    the RNG would only matter for ``.rvs()`` sampling, which this path never uses. Pitch
    selection is newest-first by ``game_date``, not random. Prefer ``num_seeds=1``.
    """

    rng = np.random.default_rng(seed)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    all_data = load_statcast_for_roster(config.season_year)

    agent_caches = [
        build_agent_convergence_cache(
            config=config,
            seed=seed,
            agent_spec=agent_spec,
            all_data=all_data,
            runtime=runtime,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        for agent_spec in config.agent_specs
    ]
    return run_convergence_from_agent_caches(config, seed, agent_caches)


def build_baseball_skill_grids_from_convergence(
    config: BaseballConvergenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build skill grids using the shared baseball grid helpers."""

    from .baseball_config import build_baseball_skill_grids as _build
    from .baseball_config import BaseballExperimentConfig

    shim = BaseballExperimentConfig(
        base=config.base,
        season_year=config.season_year,
        pitcher_ids=config.pitcher_ids,
        pitch_types=config.pitch_types,
        max_pitches_per_agent=None,
        use_natural_pitch_counts=True,
        agent_specs=config.agent_specs,
        agents=config.agents,
    )
    return _build(shim)


def _roster_selector_from_args(args) -> dict[str, Any]:
    validate_roster_selection(
        all_eligible_agents=args.all_eligible_agents,
        top_pitchers=args.top_pitchers,
        bbip_extremes=getattr(args, "bbip_extremes", None),
    )
    if getattr(args, "bbip_extremes", None) is not None:
        return dict(
            all_eligible_agents=False,
            pitcher_ids=None,
            top_pitchers=None,
            bbip_extremes=args.bbip_extremes,
        )
    if args.all_eligible_agents:
        return dict(all_eligible_agents=True, pitcher_ids=None, top_pitchers=None, bbip_extremes=None)
    if args.top_pitchers is not None:
        return dict(all_eligible_agents=False, pitcher_ids=None, top_pitchers=args.top_pitchers, bbip_extremes=None)
    pitcher_ids = tuple(int(piece.strip()) for piece in args.pitcher_ids.split(",") if piece.strip())
    return dict(all_eligible_agents=False, pitcher_ids=pitcher_ids, top_pitchers=None, bbip_extremes=None)


def _load_prepared_roster_selection(
    output_dir: Path,
    *,
    all_data,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches_per_agent: int,
) -> BaseballRosterSelection:
    agent_specs = load_convergence_roster(output_dir)
    agent_specs, excluded_agents = filter_roster_by_min_pitches(
        agent_specs,
        all_data,
        min_pitches_per_agent,
    )
    if not agent_specs:
        raise ValueError(
            f"No prepared roster agents meet min_pitches_per_agent={min_pitches_per_agent}."
        )
    agent_pitch_counts = tuple(
        (spec.pitcher_id, spec.pitch_type, count_agent_pitch_rows(all_data, spec.pitcher_id, spec.pitch_type))
        for spec in agent_specs
    )
    pitcher_ids_resolved = tuple(dict.fromkeys(spec.pitcher_id for spec in agent_specs))
    return BaseballRosterSelection(
        season_year=season_year,
        pitch_types=tuple(pitch_types),
        pitcher_ids=pitcher_ids_resolved,
        agent_specs=agent_specs,
        agent_pitch_counts=agent_pitch_counts,
        excluded_agents=excluded_agents,
    )


def _ensure_bbip_cache_for_args(args) -> None:
    if getattr(args, "bbip_extremes", None) is None or args.season_year is None:
        return
    output_dir = Path(args.output_dir)
    from .baseball_bbip import bbip_cache_path_for, resolve_bbip_cache_path

    if resolve_bbip_cache_path(output_dir=output_dir) is not None:
        return
    raise FileNotFoundError(
        f"Missing BB/IP innings cache for season_year={args.season_year}. "
        f"Expected {bbip_cache_path_for(output_dir)}. "
        "Write it on the login node before submitting Slurm jobs "
        "(submit scripts and --prepare-roster do this automatically)."
    )


def build_baseball_convergence_config_from_args(args) -> BaseballConvergenceConfig:
    convergence_ns = _parse_count_buckets(args.convergence_ns)
    pitch_types = parse_pitch_types(args.pitch_types)
    min_pitches_per_agent = (
        args.min_pitches_per_agent
        if args.min_pitches_per_agent is not None
        else required_min_pitches_for_convergence(convergence_ns, args.max_reference_pitches)
    )

    all_data = load_statcast_for_roster(args.season_year)
    output_dir = Path(args.output_dir)

    if getattr(args, "use_prepared_roster", False):
        roster = _load_prepared_roster_selection(
            output_dir,
            all_data=all_data,
            season_year=args.season_year,
            pitch_types=pitch_types,
            min_pitches_per_agent=min_pitches_per_agent,
        )
    else:
        if getattr(args, "bbip_extremes", None) is not None:
            _ensure_bbip_cache_for_args(args)
        roster_selector = _roster_selector_from_args(args)
        roster = resolve_baseball_roster(
            all_data=all_data,
            season_year=args.season_year,
            pitch_types=pitch_types,
            min_pitches_per_agent=min_pitches_per_agent,
            max_agents=args.max_agents,
            output_dir=output_dir,
            **roster_selector,
        )

    hyperpriors = resolve_baseball_hyperpriors(
        preset=args.hyperprior_preset,
        calibrated_path=Path(args.hyperprior_config) if args.hyperprior_config else None,
    )
    base = ExperimentConfig(
        environment="baseball",
        seed=args.seed,
        num_seeds=args.num_seeds,
        num_agents=len(roster.agent_specs),
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
        hyperpriors=hyperpriors,
        true_population=TruePopulationConfig(
            mean_log_sigma=hyperpriors.mean_vector[0],
            mean_log_lambda=hyperpriors.mean_vector[1],
            tau_eta=math.exp(hyperpriors.log_tau_eta_mean),
            tau_rho=math.exp(hyperpriors.log_tau_rho_mean),
            correlation=math.tanh(hyperpriors.m_r),
        ),
    )
    return BaseballConvergenceConfig(
        base=base,
        season_year=args.season_year,
        pitcher_ids=roster.pitcher_ids,
        pitch_types=roster.pitch_types,
        convergence_ns=convergence_ns,
        max_reference_pitches=args.max_reference_pitches,
        min_pitches_per_agent=min_pitches_per_agent,
        agent_specs=roster.agent_specs,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster.agent_specs),
        agent_pitch_counts=roster.agent_pitch_counts,
        excluded_agents=roster.excluded_agents,
    )


def print_eligible_pitchers(
    pitch_types: Sequence[str],
    *,
    min_pitches: int,
    limit: int,
    season_year: int | None = None,
) -> None:
    from .baseball_roster import print_eligible_agents

    print_eligible_agents(
        season_year=season_year,
        pitch_types=pitch_types,
        min_pitches=min_pitches,
        limit=limit,
    )


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
    pitch_builds = len(config.agent_specs) * max(
        max_n,
        config.max_reference_pitches or 0,
    )

    print("=== DRY RUN: Baseball HJEEDS Convergence Study (Phase 2) ===")
    print("No RNN inference or estimation will run.")
    print()
    print(f"Environment: {config.environment}")
    print(f"Season year: {config.season_year or 'all seasons in pickle'}")
    print(f"Seeds: {config.seed_values}")
    print(
        "Note: seed does not change Statcast likelihoods (execution PDFs use "
        "multivariate_normal.pdf; pitch order is newest-first by game_date)."
    )
    print(f"Agents: {len(config.agent_specs)}")
    if config.agent_pitch_counts:
        print("Agent pitch counts:")
        for pitcher_id, pitch_type, pitch_count in config.agent_pitch_counts:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
    if config.excluded_agents:
        print(f"Excluded (below min_pitches_per_agent={config.min_pitches_per_agent}):")
        for pitcher_id, pitch_type, pitch_count in config.excluded_agents:
            print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
    print(f"Min pitches per agent: {config.min_pitches_per_agent}")
    print(f"Convergence N values (prefix pitch counts): {config.convergence_ns}")
    print(f"Hyperprior centers (log sigma, log lambda): {config.base.hyperpriors.mean_vector}")
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
        f"({len(config.agent_specs)} agents x max build depth)."
    )


CONVERGENCE_ROSTER_FILENAME = "convergence_roster.json"
CONVERGENCE_ROSTER_METADATA_FILENAME = "convergence_roster_metadata.json"
AGENT_CACHE_SUBDIR = "agents"


def _estimate_to_dict(estimate: MethodEstimate) -> dict[str, Any]:
    return {
        "method_name": estimate.method_name,
        "posterior_mean_sigma": estimate.posterior_mean_sigma,
        "posterior_mean_log_lambda": estimate.posterior_mean_log_lambda,
        "map_sigma": estimate.map_sigma,
        "map_log_lambda": estimate.map_log_lambda,
        "rationality_percent": estimate.rationality_percent,
        "status": estimate.status,
        "notes": estimate.notes,
    }


def _estimate_from_dict(payload: dict[str, Any]) -> MethodEstimate:
    return MethodEstimate(
        method_name=str(payload.get("method_name", "jeeds")),
        posterior_mean_sigma=_optional_float(payload.get("posterior_mean_sigma")),
        posterior_mean_log_lambda=_optional_float(payload.get("posterior_mean_log_lambda")),
        map_sigma=_optional_float(payload.get("map_sigma")),
        map_log_lambda=_optional_float(payload.get("map_log_lambda")),
        rationality_percent=_optional_float(payload.get("rationality_percent")),
        status=str(payload.get("status", "")),
        notes=str(payload.get("notes", "")),
    )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def convergence_roster_path_for(output_dir: Path) -> Path:
    return output_dir / CONVERGENCE_ROSTER_FILENAME


def agent_cache_path_for(output_dir: Path, agent_id: int) -> Path:
    return output_dir / AGENT_CACHE_SUBDIR / f"agent_{agent_id:04d}.pkl"


def write_convergence_roster(
    output_dir: Path,
    roster: BaseballRosterSelection,
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches_per_agent: int,
    max_reference_pitches: int | None,
    convergence_ns: Sequence[int],
    roster_selector: dict[str, Any],
    all_data=None,
) -> Path:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    path = convergence_roster_path_for(output_dir)
    payload = [
        {
            "agent_id": spec.agent_id,
            "pitcher_id": spec.pitcher_id,
            "pitch_type": spec.pitch_type,
        }
        for spec in roster.agent_specs
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    metadata: dict[str, Any] = {
        "season_year": season_year,
        "pitch_types": list(pitch_types),
        "min_pitches_per_agent": min_pitches_per_agent,
        "max_reference_pitches": max_reference_pitches,
        "convergence_ns": list(convergence_ns),
        "num_agents": len(payload),
        "roster_selector": roster_selector,
        "pitcher_ids": list(roster.pitcher_ids),
    }
    bbip_extremes = roster_selector.get("bbip_extremes")
    if bbip_extremes is not None and all_data is not None and season_year is not None:
        from .baseball_bbip import build_bbip_manifest

        metadata["bbip_extremes"] = bbip_extremes
        metadata["bbip_selection"] = build_bbip_manifest(
            all_data,
            season_year=season_year,
            pitcher_ids=roster.pitcher_ids,
            extremes_count=bbip_extremes,
            output_dir=output_dir,
        )

    with (output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


def load_convergence_roster(output_dir: Path) -> tuple[StatcastAgentSpec, ...]:
    import json

    path = convergence_roster_path_for(output_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Convergence roster not found: {path}. Run with --prepare-roster first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(
        StatcastAgentSpec(
            agent_id=int(row["agent_id"]),
            pitcher_id=int(row["pitcher_id"]),
            pitch_type=str(row["pitch_type"]),
        )
        for row in payload
    )


def build_agent_convergence_cache(
    *,
    config: BaseballConvergenceConfig,
    seed: int,
    agent_spec: StatcastAgentSpec,
    all_data,
    runtime,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> dict[str, Any]:
    """Build per-agent reference JEEDS and prefix-N log-likelihood grids."""

    max_build_n = max(config.convergence_ns)
    agent_rows = get_agent_pitch_rows(
        all_data,
        agent_spec.pitcher_id,
        agent_spec.pitch_type,
    )
    if len(agent_rows) == 0:
        raise ValueError(f"No pitches for agent pitcher={agent_spec.pitcher_id} pitch_type={agent_spec.pitch_type}.")

    if config.max_reference_pitches is None:
        reference_row_count = len(agent_rows)
    else:
        reference_row_count = min(len(agent_rows), config.max_reference_pitches)

    build_n = max(max_build_n, reference_row_count)
    built_rows = agent_rows.iloc[:build_n, :]
    execution_skills = tuple(float(value) for value in sigma_grid)
    all_observations = _build_pitch_observations(built_rows, runtime, execution_skills)
    reference_observations = all_observations[:reference_row_count]
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

    log_likelihood_grids_by_n: dict[str, list[list[float]]] = {}
    jeeds_estimates_by_n: dict[str, dict[str, Any]] = {}
    take_n_by_n: dict[str, int] = {}
    for convergence_n in config.convergence_ns:
        take_n = min(convergence_n, len(all_observations))
        pitch_observations = all_observations[:take_n]
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
        key = str(convergence_n)
        log_likelihood_grids_by_n[key] = log_likelihood_grid.tolist()
        jeeds_estimates_by_n[key] = _estimate_to_dict(jeeds_estimate)
        take_n_by_n[key] = take_n

    return {
        "seed": seed,
        "agent_id": agent_spec.agent_id,
        "pitcher_id": agent_spec.pitcher_id,
        "pitch_type": agent_spec.pitch_type,
        "convergence_ns": list(config.convergence_ns),
        "num_reference_observations": len(reference_observations),
        "reference_estimate": _estimate_to_dict(reference_estimate),
        "log_likelihood_grids_by_n": log_likelihood_grids_by_n,
        "jeeds_estimates_by_n": jeeds_estimates_by_n,
        "take_n_by_n": take_n_by_n,
    }


def write_agent_convergence_cache(path: Path, payload: dict[str, Any]) -> None:
    import pickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_agent_convergence_cache(path: Path) -> dict[str, Any]:
    import pickle

    with path.open("rb") as handle:
        return pickle.load(handle)


def load_agent_convergence_caches(
    output_dir: Path,
    roster: Sequence[StatcastAgentSpec],
) -> tuple[list[dict[str, Any]], list[int]]:
    caches: list[dict[str, Any]] = []
    missing_agent_ids: list[int] = []
    for agent_spec in roster:
        path = agent_cache_path_for(output_dir, agent_spec.agent_id)
        if not path.is_file():
            missing_agent_ids.append(agent_spec.agent_id)
            continue
        caches.append(load_agent_convergence_cache(path))
    caches.sort(key=lambda payload: int(payload["agent_id"]))
    return caches, missing_agent_ids


def run_convergence_from_agent_caches(
    config: BaseballConvergenceConfig,
    seed: int,
    agent_caches: Sequence[dict[str, Any]],
) -> BaseballConvergenceSeedResult:
    """Run population MAP + hierarchical passes from per-agent cached grids."""

    sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    seed_result = BaseballConvergenceSeedResult(
        seed=seed,
        notes="Statcast baseball convergence study (full-data JEEDS reference).",
    )

    for convergence_n in config.convergence_ns:
        key = str(convergence_n)
        agent_records: list[
            tuple[
                StatcastAgentSpec,
                int,
                np.ndarray,
                MethodEstimate,
                MethodEstimate,
                int,
            ]
        ] = []

        for cache in agent_caches:
            if key not in cache["log_likelihood_grids_by_n"]:
                continue
            log_likelihood_grid = np.asarray(cache["log_likelihood_grids_by_n"][key], dtype=float)
            jeeds_estimate = _estimate_from_dict(cache["jeeds_estimates_by_n"][key])
            reference_estimate = _estimate_from_dict(cache["reference_estimate"])
            take_n = int(cache.get("take_n_by_n", {}).get(key, convergence_n))
            agent_spec = StatcastAgentSpec(
                agent_id=int(cache["agent_id"]),
                pitcher_id=int(cache["pitcher_id"]),
                pitch_type=str(cache["pitch_type"]),
            )
            agent_records.append(
                (
                    agent_spec,
                    take_n,
                    log_likelihood_grid,
                    jeeds_estimate,
                    reference_estimate,
                    int(cache["num_reference_observations"]),
                )
            )

        discrete_hierarchical_prior = None
        if agent_records:
            fitted_hyperparameters = fit_population_hyperparameters_map(
                config=config.base,
                agent_log_likelihoods=[record[2] for record in agent_records],
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
                    num_observations=take_n,
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


def run_single_agent_convergence_cache(
    config: BaseballConvergenceConfig,
    seed: int,
    agent_spec: StatcastAgentSpec,
    *,
    all_data=None,
) -> dict[str, Any]:
    # Seed is unused for Statcast likelihood numerics; see run_single_baseball_convergence_seed.
    rng = np.random.default_rng(seed)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    if all_data is None:
        all_data = load_statcast_for_roster(config.season_year)
    return build_agent_convergence_cache(
        config=config,
        seed=seed,
        agent_spec=agent_spec,
        all_data=all_data,
        runtime=runtime,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )


def prepare_convergence_roster(args) -> Path:
    if getattr(args, "bbip_extremes", None) is not None:
        if args.season_year is None:
            raise ValueError("--bbip-extremes requires --season-year.")
        from .baseball_bbip import write_bbip_innings_cache

        cache_path = write_bbip_innings_cache(Path(args.output_dir), season_year=args.season_year)
        print(f"[baseball-convergence] Wrote BB/IP innings cache to {cache_path.resolve()}", flush=True)

    config = build_baseball_convergence_config_from_args(args)
    all_data = load_statcast_for_roster(args.season_year)
    roster = resolve_baseball_roster(
        all_data=all_data,
        season_year=args.season_year,
        pitch_types=config.pitch_types,
        min_pitches_per_agent=config.min_pitches_per_agent,
        max_agents=args.max_agents,
        **_roster_selector_from_args(args),
    )
    path = write_convergence_roster(
        config.base.output_dir,
        roster,
        season_year=args.season_year,
        pitch_types=config.pitch_types,
        min_pitches_per_agent=config.min_pitches_per_agent,
        max_reference_pitches=config.max_reference_pitches,
        convergence_ns=config.convergence_ns,
        roster_selector=_roster_selector_from_args(args),
        all_data=all_data,
    )
    print(
        f"[baseball-convergence] Wrote roster with {len(roster.agent_specs)} agents to {path.resolve()}",
        flush=True,
    )
    return path


def run_convergence_agent_index(args, agent_index: int) -> Path:
    output_dir = Path(args.output_dir)
    roster = load_convergence_roster(output_dir)
    if agent_index < 0 or agent_index >= len(roster):
        raise IndexError(
            f"agent_index={agent_index} is out of range for roster size {len(roster)}."
        )

    agent_spec = roster[agent_index]
    config = build_baseball_convergence_config_from_args(args)
    config = BaseballConvergenceConfig(
        base=config.base,
        season_year=config.season_year,
        pitcher_ids=config.pitcher_ids,
        pitch_types=config.pitch_types,
        convergence_ns=config.convergence_ns,
        max_reference_pitches=config.max_reference_pitches,
        min_pitches_per_agent=config.min_pitches_per_agent,
        agent_specs=roster,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster),
        agent_pitch_counts=config.agent_pitch_counts,
        excluded_agents=config.excluded_agents,
    )
    seed = config.seed_values[0]
    cache = run_single_agent_convergence_cache(config, seed, agent_spec)
    out_path = agent_cache_path_for(output_dir, agent_spec.agent_id)
    write_agent_convergence_cache(out_path, cache)
    print(
        f"[baseball-convergence] agent_index={agent_index} "
        f"pitcher={agent_spec.pitcher_id} pitch_type={agent_spec.pitch_type} "
        f"-> {out_path.resolve()}",
        flush=True,
    )
    return out_path


def aggregate_convergence_results(args) -> tuple[list[StatcastConvergenceAgentResult], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(args.output_dir)
    roster = load_convergence_roster(output_dir)
    caches, missing_agent_ids = load_agent_convergence_caches(output_dir, roster)
    if missing_agent_ids:
        print(
            f"[baseball-convergence] Warning: missing {len(missing_agent_ids)} agent cache files.",
            flush=True,
        )
    if not caches:
        raise FileNotFoundError("No agent convergence caches found for aggregation.")

    config = build_baseball_convergence_config_from_args(args)
    config = BaseballConvergenceConfig(
        base=config.base,
        season_year=config.season_year,
        pitcher_ids=config.pitcher_ids,
        pitch_types=config.pitch_types,
        convergence_ns=config.convergence_ns,
        max_reference_pitches=config.max_reference_pitches,
        min_pitches_per_agent=config.min_pitches_per_agent,
        agent_specs=roster,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in roster),
        agent_pitch_counts=config.agent_pitch_counts,
        excluded_agents=config.excluded_agents,
    )

    seed_results = []
    for seed in config.seed_values:
        seed_results.append(run_convergence_from_agent_caches(config, seed, caches))

    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_n_rows, summary_overall_rows = aggregate_convergence_across_seeds(seed_results)
    return all_agent_results, summary_by_n_rows, summary_overall_rows
