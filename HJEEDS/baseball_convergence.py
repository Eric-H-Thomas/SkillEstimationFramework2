# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_sigma_gap`.
"""Statcast baseball convergence: JEEDS vs H-JEEDS drift vs each method at max N.

Core library for Phase 2 / paper walk/IP-proxy convergence. CLI lives in
``baseball_convergence_study``; Slurm entry is
``submit_hjeeds_baseball_convergence_paper_bbip.sh``.

Paper walk/IP-proxy knobs (do not change lightly — regenerates results):
``--bbip-extremes 10``, ``--season-year 2021``, ``--pitch-types FF``,
``min_pitches_per_agent=100``, ``convergence_ns=5,10,25,50,100``,
``max_reference_pitches=100``,
``--hyperprior-preset baseball-literature-informed`` (committed
``HJEEDS/data/baseball_hyperpriors_literature_informed.json``).

Modes (``submit_hjeeds_baseball_convergence_array.sh``):
  ``--prepare-roster`` → per-agent ``--agent-index`` array → ``--aggregate-results``.
Workers pass ``--use-prepared-roster`` so roster selection stays frozen.
Local sequential: omit those flags and call ``run_single_baseball_convergence_seed``.

Statcast likelihoods are deterministic in seed; prefer ``num_seeds=1``.
"""

from __future__ import annotations

import json
import hashlib
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import _optional_float
from .baseball_config import BaseballExperimentConfig, build_baseball_skill_grids
from .baseball_likelihood import compute_baseball_log_likelihood_grids_by_prefix
from .baseball_hyperpriors import (
    DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH,
    DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH,
    hyperprior_config_to_dict,
    resolve_baseball_hyperpriors,
    true_population_from_hyperpriors,
)
from .baseball_provenance import (
    PAPER_BBIP20_ROSTER_SHA256,
    PAPER_BBIP_2021_INNINGS_SHA256,
    build_baseball_run_provenance,
    require_matching_fingerprint,
    roster_fingerprint,
)
from .baseball_pitch import (
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observations_for_rows,
    count_agent_pitch_rows,
    filter_roster_by_min_pitches,
    get_agent_pitch_rows,
)
from .baseball_roster import (
    BaseballRosterSelection,
    load_statcast_for_roster,
    parse_pitch_types,
    resolve_baseball_roster,
    roster_selector_kwargs_from_args,
)
from .config import _parse_count_buckets, paper_config_required
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
# Smoke-demo pitcher pair for the convergence CLI (distinct from baseball_config defaults).
DEFAULT_PITCHER_IDS = (623433, 543037)

CONVERGENCE_ROSTER_FILENAME = "convergence_roster.json"
CONVERGENCE_ROSTER_METADATA_FILENAME = "convergence_roster_metadata.json"
AGENT_CACHE_SUBDIR = "agents"
POPULATION_FIT_DIAGNOSTICS_FILENAME = "population_fit_diagnostics.json"
CONVERGENCE_RUN_METADATA_FILENAME = "convergence_run_metadata.json"
CONVERGENCE_RUN_METADATA_SCHEMA_VERSION = 2

BBIP_METRIC_DEFINITION: dict[str, str] = {
    "name": "processed-data walks-per-inning proxy",
    "numerator": (
        "count of all retained processed Statcast rows with game_year == 2021 and "
        "events == 'walk'; the artifact spans 2021-03-15 through 2021-11-02, "
        "including spring and postseason rows"
    ),
    "denominator": "official 2021 season innings pitched from the tracked bundled artifact",
    "interpretation": "selection proxy used by this study; not official league BB/IP",
}

PAPER_BBIP20_ROSTER_SELECTOR: dict[str, Any] = {
    "all_eligible_agents": False,
    "pitcher_ids": None,
    "top_pitchers": None,
    "bbip_extremes": 10,
}

_CORE_CONVERGENCE_ARTIFACT_FILENAMES = (
    CONVERGENCE_ROSTER_FILENAME,
    CONVERGENCE_ROSTER_METADATA_FILENAME,
    "convergence_agent_level_results.csv",
    "summary_by_N.csv",
    "summary_overall.csv",
    "drift_by_N.png",
    "drift_by_N_proportional.png",
    POPULATION_FIT_DIAGNOSTICS_FILENAME,
)
_BBIP_CONVERGENCE_ARTIFACT_FILENAMES = (
    "bbip_innings_cache.json",
    "bbip_tiers_corrected.csv",
    "separability_by_N.csv",
    "separability_by_N.png",
    "separability_by_N_proportional.png",
    "separability_summary.json",
)


def _bbip_metric_definition_for(season_year: int) -> dict[str, str]:
    if int(season_year) == 2021:
        return dict(BBIP_METRIC_DEFINITION)
    return {
        "name": "processed-data walks-per-inning proxy",
        "numerator": (
            f"count of all retained processed Statcast rows with game_year == "
            f"{int(season_year)} and events == 'walk'"
        ),
        "denominator": (
            f"official {int(season_year)} season innings pitched from the resolved source artifact"
        ),
        "interpretation": "selection proxy used by this study; not official league BB/IP",
    }


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


def build_drift_summary_tables(
    bucket_metrics: dict[tuple[str, str, int], list[float]],
    overall_metrics: dict[tuple[str, str], list[float]],
    *,
    by_n_notes: str,
    overall_notes: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn collected drift values into summary_by_N / summary_overall row dicts."""

    summary_by_n_rows: list[dict[str, Any]] = []
    for (method_name, metric_name, convergence_n), values in sorted(bucket_metrics.items()):
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.size == 0 or np.any(~np.isfinite(numeric_values)) or np.any(
            numeric_values < 0.0
        ):
            raise ValueError(
                f"Invalid nonnegative drift values for {method_name}/{metric_name}/"
                f"N={convergence_n}: {values}."
            )
        summary_by_n_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "count_bucket": convergence_n,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": by_n_notes,
            }
        )

    summary_overall_rows: list[dict[str, Any]] = []
    for (method_name, metric_name), values in sorted(overall_metrics.items()):
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.size == 0 or np.any(~np.isfinite(numeric_values)) or np.any(
            numeric_values < 0.0
        ):
            raise ValueError(
                f"Invalid nonnegative overall drift values for {method_name}/{metric_name}: "
                f"{values}."
            )
        summary_overall_rows.append(
            {
                "method": method_name,
                "metric": metric_name,
                "num_agents": len(values),
                "mean": float(np.mean(values)),
                "ci_lower": "",
                "ci_upper": "",
                "notes": overall_notes,
            }
        )

    return summary_by_n_rows, summary_overall_rows


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

    return build_drift_summary_tables(
        bucket_metrics,
        overall_metrics,
        by_n_notes=(
            "Seed-level mean over agents with valid drift metrics. "
            "Confidence intervals are added during across-seed aggregation."
        ),
        overall_notes=(
            "Seed-level overall mean over agents with valid drift metrics. "
            "Confidence intervals are added during across-seed aggregation."
        ),
    )


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
    return run_convergence_from_agent_caches(
        config,
        seed,
        agent_caches,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )


def build_baseball_skill_grids_from_convergence(
    config: BaseballConvergenceConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Build skill grids via the shared baseball grid helper (config shim)."""

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
    return build_baseball_skill_grids(shim)


def _config_with_agent_specs(
    config: BaseballConvergenceConfig,
    agent_specs: Sequence[StatcastAgentSpec],
) -> BaseballConvergenceConfig:
    """Copy config fields but swap in an explicit agent roster (prepared JSON)."""

    specs = tuple(agent_specs)
    return BaseballConvergenceConfig(
        base=config.base,
        season_year=config.season_year,
        pitcher_ids=config.pitcher_ids,
        pitch_types=config.pitch_types,
        convergence_ns=config.convergence_ns,
        max_reference_pitches=config.max_reference_pitches,
        min_pitches_per_agent=config.min_pitches_per_agent,
        agent_specs=specs,
        agents=tuple((spec.pitcher_id, spec.pitch_type) for spec in specs),
        agent_pitch_counts=config.agent_pitch_counts,
        excluded_agents=config.excluded_agents,
    )


def _load_prepared_roster_selection(
    output_dir: Path,
    *,
    all_data: pd.DataFrame,
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
        f"Missing walk/IP-proxy innings cache for season_year={args.season_year}. "
        f"Expected {bbip_cache_path_for(output_dir)}. "
        "Write it on the login node before submitting Slurm jobs "
        "(submit scripts and --prepare-roster do this automatically)."
    )


def _build_convergence_config_and_data(
    args,
) -> tuple[BaseballConvergenceConfig, pd.DataFrame]:
    """Resolve roster + Statcast frame once; shared by config builders and prepare-roster."""

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
        roster_selector = roster_selector_kwargs_from_args(args)
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
        true_population=true_population_from_hyperpriors(hyperpriors),
    )
    config = BaseballConvergenceConfig(
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
    return config, all_data


def build_baseball_convergence_config_from_args(args) -> BaseballConvergenceConfig:
    config, _all_data = _build_convergence_config_and_data(args)
    return config


def planned_convergence_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "agent_level_csv": output_dir / "convergence_agent_level_results.csv",
        "summary_by_n_csv": output_dir / "summary_by_N.csv",
        "summary_overall_csv": output_dir / "summary_overall.csv",
        "drift_plot": output_dir / "drift_by_N.png",
        "drift_plot_proportional": output_dir / "drift_by_N_proportional.png",
        "population_fit_diagnostics": output_dir / POPULATION_FIT_DIAGNOSTICS_FILENAME,
        "run_metadata": output_dir / CONVERGENCE_RUN_METADATA_FILENAME,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _spec_roster_fingerprint(agent_specs: Sequence[StatcastAgentSpec]) -> str:
    return roster_fingerprint(
        [(spec.agent_id, spec.pitcher_id, spec.pitch_type) for spec in agent_specs]
    )


def build_convergence_run_provenance(
    config: BaseballConvergenceConfig,
    *,
    sigma_grid: Sequence[float] | None = None,
    log_lambda_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fingerprint every input that changes a per-agent convergence cache."""

    if (sigma_grid is None) != (log_lambda_grid is None):
        raise ValueError("Pass both convergence skill grids, or neither.")
    if sigma_grid is None:
        sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    return build_baseball_run_provenance(
        season_year=config.season_year,
        pitch_types=config.pitch_types,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        configuration={
            "workflow": "baseball-convergence-agent-cache",
            "delta": float(config.base.delta),
            "convergence_ns": [int(value) for value in config.convergence_ns],
            "max_reference_pitches": config.max_reference_pitches,
            "min_pitches_per_agent": int(config.min_pitches_per_agent),
            "roster_fingerprint": _spec_roster_fingerprint(config.agent_specs),
            "num_agents": len(config.agent_specs),
        },
    )


def load_convergence_roster_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing convergence roster metadata: {path}. Run --prepare-roster again."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed convergence roster metadata: {path}")
    return payload


def validate_convergence_metadata_against_config(
    metadata: dict[str, Any],
    config: BaseballConvergenceConfig,
    expected_provenance: dict[str, Any],
    *,
    require_paper_configuration: bool | None = None,
) -> None:
    required = {
        "season_year": config.season_year,
        "pitch_types": list(config.pitch_types),
        "min_pitches_per_agent": int(config.min_pitches_per_agent),
        "max_reference_pitches": config.max_reference_pitches,
        "convergence_ns": list(config.convergence_ns),
        "num_agents": len(config.agent_specs),
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Convergence arguments do not match the frozen roster metadata: "
            f"{mismatches}. Re-submit with the original arguments."
        )
    require_matching_fingerprint(
        metadata.get("run_provenance"),
        expected_provenance,
        label="convergence roster metadata",
    )
    is_paper_bbip20 = (
        metadata.get("season_year") == 2021
        and metadata.get("pitch_types") == ["FF"]
        and metadata.get("roster_selector") == PAPER_BBIP20_ROSTER_SELECTOR
        and metadata.get("min_pitches_per_agent") == 100
        and metadata.get("max_reference_pitches") == 100
        and metadata.get("convergence_ns") == [5, 10, 25, 50, 100]
    )
    if is_paper_bbip20 and paper_config_required(require_paper_configuration):
        actual_roster_hash = _spec_roster_fingerprint(config.agent_specs)
        if actual_roster_hash != PAPER_BBIP20_ROSTER_SHA256:
            raise ValueError(
                f"Prepared BBIP20 roster hash {actual_roster_hash} does not match the "
                f"pinned paper roster {PAPER_BBIP20_ROSTER_SHA256}."
            )
        if metadata.get("bbip_metric_definition") != BBIP_METRIC_DEFINITION:
            raise ValueError("Prepared BBIP20 metadata has a stale W/IP proxy definition.")
        innings = metadata.get("bbip_innings_provenance") or {}
        expected_artifact = {
            "filename": "baseball_innings_pitched_2021.json",
            "sha256": PAPER_BBIP_2021_INNINGS_SHA256,
        }
        if innings.get("bundled_artifact") != expected_artifact:
            raise ValueError("Prepared BBIP20 metadata has a stale bundled innings hash.")
        if innings.get("cache_source_artifact") != expected_artifact:
            raise ValueError("Prepared BBIP20 cache is not linked to the bundled innings artifact.")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def _load_convergence_run_metadata(output_dir: Path) -> dict[str, Any]:
    path = output_dir / CONVERGENCE_RUN_METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing convergence run metadata: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed convergence run metadata: {path}.")
    return payload


def _convergence_hyperprior_metadata(args, config: BaseballConvergenceConfig) -> dict[str, Any]:
    preset = str(args.hyperprior_preset)
    hyperprior_path: Path | None = None
    if preset == "baseball-literature-informed":
        hyperprior_path = DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH
    elif preset == "baseball-2021-ff":
        hyperprior_path = DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH
    elif preset == "calibrated" and args.hyperprior_config:
        hyperprior_path = Path(args.hyperprior_config)
    return {
        "preset": preset,
        "config": hyperprior_config_to_dict(config.base.hyperpriors),
        "file": str(hyperprior_path.resolve()) if hyperprior_path is not None else None,
        "file_sha256": (
            _file_sha256(hyperprior_path)
            if hyperprior_path is not None and hyperprior_path.is_file()
            else None
        ),
    }


def _convergence_cohort_snapshot(output_dir: Path) -> dict[str, Any]:
    roster_path = convergence_roster_path_for(output_dir)
    metadata_path = output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME
    if not roster_path.is_file() or not metadata_path.is_file():
        return {"frozen_roster": False}

    roster = load_convergence_roster(output_dir)
    metadata = load_convergence_roster_metadata(output_dir)
    selection = metadata.get("bbip_selection") or []
    return {
        "frozen_roster": True,
        "roster_selector": metadata.get("roster_selector"),
        "roster_fingerprint": _spec_roster_fingerprint(roster),
        "roster_json_sha256": _file_sha256(roster_path),
        "roster_metadata_sha256": _file_sha256(metadata_path),
        "bbip_manifest_sha256": hashlib.sha256(
            json.dumps(selection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest(),
        "bbip_metric_definition": metadata.get("bbip_metric_definition"),
        "bbip_innings_provenance": metadata.get("bbip_innings_provenance"),
    }


def begin_convergence_run_metadata(
    output_dir: Path,
    *,
    args,
    config: BaseballConvergenceConfig,
    run_provenance: dict[str, Any],
) -> Path:
    """Atomically mark an output generation incomplete before any result rewrite."""

    payload: dict[str, Any] = {
        "schema_version": CONVERGENCE_RUN_METADATA_SCHEMA_VERSION,
        "completion_status": "incomplete",
        "run_provenance": run_provenance,
        "hyperprior": _convergence_hyperprior_metadata(args, config),
        "cohort": _convergence_cohort_snapshot(output_dir),
        "nonpaper_overrides": {
            "allow_partial_caches": bool(getattr(args, "allow_partial_caches", False)),
            "allow_separability_failure": bool(
                getattr(args, "allow_separability_failure", False)
            ),
        },
        "artifact_sha256": {},
        "artifact_sizes_bytes": {},
    }
    path = output_dir / CONVERGENCE_RUN_METADATA_FILENAME
    return _atomic_write_json(path, payload)


def _expected_convergence_artifact_filenames(payload: dict[str, Any]) -> tuple[str, ...]:
    cohort = payload.get("cohort") or {}
    filenames = list(_CORE_CONVERGENCE_ARTIFACT_FILENAMES)
    if not bool(cohort.get("frozen_roster")):
        filenames = [
            name
            for name in filenames
            if name not in {CONVERGENCE_ROSTER_FILENAME, CONVERGENCE_ROSTER_METADATA_FILENAME}
        ]
    selector = cohort.get("roster_selector") or {}
    if selector.get("bbip_extremes") is not None:
        filenames.extend(_BBIP_CONVERGENCE_ARTIFACT_FILENAMES)
    return tuple(filenames)


def finalize_convergence_run_metadata(output_dir: Path) -> Path:
    """Hash every result artifact and atomically write the completion marker last."""

    payload = _load_convergence_run_metadata(output_dir)
    if payload.get("completion_status") != "incomplete":
        raise ValueError(
            "Convergence metadata must be marked incomplete before it can be finalized."
        )
    payload["cohort"] = _convergence_cohort_snapshot(output_dir)
    filenames = _expected_convergence_artifact_filenames(payload)
    missing = [name for name in filenames if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot mark convergence outputs complete; required artifacts are missing: "
            f"{missing}."
        )
    payload["artifact_sha256"] = {
        name: _file_sha256(output_dir / name) for name in filenames
    }
    payload["artifact_sizes_bytes"] = {
        name: int((output_dir / name).stat().st_size) for name in filenames
    }
    payload["completion_status"] = "complete"
    return _atomic_write_json(output_dir / CONVERGENCE_RUN_METADATA_FILENAME, payload)


def validate_complete_convergence_run_metadata(output_dir: Path) -> dict[str, Any]:
    """Reject incomplete, stale, missing, or modified convergence output bundles."""

    payload = _load_convergence_run_metadata(output_dir)
    if payload.get("schema_version") != CONVERGENCE_RUN_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported convergence metadata schema {payload.get('schema_version')!r}; "
            f"expected {CONVERGENCE_RUN_METADATA_SCHEMA_VERSION}."
        )
    if payload.get("completion_status") != "complete":
        raise ValueError(
            "Convergence outputs are incomplete; rerun aggregation or plot-only regeneration."
        )
    expected_names = set(_expected_convergence_artifact_filenames(payload))
    recorded_hashes = payload.get("artifact_sha256") or {}
    recorded_sizes = payload.get("artifact_sizes_bytes") or {}
    if set(recorded_hashes) != expected_names or set(recorded_sizes) != expected_names:
        raise ValueError(
            "Convergence completion metadata does not enumerate the exact required artifact set."
        )
    for name in sorted(expected_names):
        if Path(name).name != name:
            raise ValueError(f"Unsafe convergence artifact name in metadata: {name!r}.")
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Completed convergence artifact is missing: {path}.")
        actual_size = int(path.stat().st_size)
        if actual_size != int(recorded_sizes[name]):
            raise ValueError(
                f"Completed convergence artifact size changed for {name}: "
                f"{actual_size} != {recorded_sizes[name]}."
            )
        actual_hash = _file_sha256(path)
        if actual_hash != recorded_hashes[name]:
            raise ValueError(
                f"Completed convergence artifact hash changed for {name}: "
                f"{actual_hash} != {recorded_hashes[name]}."
            )
    cohort = payload.get("cohort") or {}
    if bool(cohort.get("frozen_roster")):
        current = _convergence_cohort_snapshot(output_dir)
        if current != cohort:
            raise ValueError("Frozen convergence cohort files no longer match run metadata.")
    return payload


def restamp_convergence_run_incomplete(output_dir: Path) -> Path:
    """Validate a complete bundle, then atomically invalidate it before plot-only writes."""

    payload = validate_complete_convergence_run_metadata(output_dir)
    payload["completion_status"] = "incomplete"
    payload["artifact_sha256"] = {}
    payload["artifact_sizes_bytes"] = {}
    return _atomic_write_json(output_dir / CONVERGENCE_RUN_METADATA_FILENAME, payload)


def validate_paper_bbip20_cohort(
    output_dir: Path,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Require the exact disclosed 2021 FF processed-data W/IP-proxy cohort."""

    if payload is None:
        payload = validate_complete_convergence_run_metadata(output_dir)
    overrides = payload.get("nonpaper_overrides") or {}
    if any(bool(value) for value in overrides.values()):
        raise ValueError(f"MLB paper results used non-paper overrides: {overrides}.")

    metadata = load_convergence_roster_metadata(output_dir)
    roster = load_convergence_roster(output_dir)
    selector = metadata.get("roster_selector")
    if selector != PAPER_BBIP20_ROSTER_SELECTOR:
        raise ValueError(
            f"MLB paper roster selector is {selector!r}; expected "
            f"{PAPER_BBIP20_ROSTER_SELECTOR!r}."
        )
    actual_roster_hash = _spec_roster_fingerprint(roster)
    if actual_roster_hash != PAPER_BBIP20_ROSTER_SHA256:
        raise ValueError(
            f"MLB paper roster hash {actual_roster_hash} does not match the pinned "
            f"BBIP20 roster {PAPER_BBIP20_ROSTER_SHA256}."
        )
    run_roster_hash = (
        ((payload.get("run_provenance") or {}).get("configuration") or {}).get(
            "roster_fingerprint"
        )
    )
    cohort = payload.get("cohort") or {}
    if run_roster_hash != actual_roster_hash or cohort.get("roster_fingerprint") != actual_roster_hash:
        raise ValueError("MLB run/cohort provenance is not linked to the pinned BBIP20 roster.")

    if metadata.get("bbip_metric_definition") != BBIP_METRIC_DEFINITION:
        raise ValueError("MLB roster metadata lacks the exact disclosed W/IP proxy definition.")
    if cohort.get("bbip_metric_definition") != BBIP_METRIC_DEFINITION:
        raise ValueError("MLB run metadata lacks the exact disclosed W/IP proxy definition.")

    selection = metadata.get("bbip_selection") or []
    if len(selection) != 20:
        raise ValueError(f"MLB paper BBIP selection has {len(selection)} rows; expected 20.")
    roster_ids = {int(spec.pitcher_id) for spec in roster}
    selection_ids = {int(row["pitcher_id"]) for row in selection}
    if selection_ids != roster_ids:
        raise ValueError("MLB BBIP manifest pitcher IDs do not match the frozen roster.")
    tier_counts = {
        tier: sum(str(row.get("tier")) == tier for row in selection)
        for tier in ("bottom", "top")
    }
    if tier_counts != {"bottom": 10, "top": 10}:
        raise ValueError(f"MLB BBIP manifest tier counts are {tier_counts}; expected 10/10.")

    innings = metadata.get("bbip_innings_provenance") or {}
    expected_artifact = {
        "filename": "baseball_innings_pitched_2021.json",
        "sha256": PAPER_BBIP_2021_INNINGS_SHA256,
    }
    if int(innings.get("season_year", -1)) != 2021:
        raise ValueError("MLB BBIP innings provenance is not for 2021.")
    if innings.get("bundled_artifact") != expected_artifact:
        raise ValueError("MLB BBIP metadata is not linked to the pinned bundled innings artifact.")
    if innings.get("cache_source_artifact") != expected_artifact:
        raise ValueError("MLB BBIP cache did not declare the pinned bundled artifact as its source.")
    cache_path = output_dir / "bbip_innings_cache.json"
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache_payload.get("source_artifact") != expected_artifact:
        raise ValueError("MLB BBIP innings cache is not linked to the pinned bundled artifact.")
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _persist_population_fit_diagnostics(
    output_dir: Path,
    *,
    seed: int,
    records: Sequence[dict[str, Any]],
) -> Path:
    """Persist optimizer diagnostics before a failed fit can terminate aggregation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / POPULATION_FIT_DIAGNOSTICS_FILENAME
    payload: dict[str, Any] = {"fits_by_seed": {}}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            payload = loaded
            payload.setdefault("fits_by_seed", {})
    payload["fits_by_seed"][str(seed)] = _json_safe(list(records))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


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
    print(f"Convergence N values (pitch counts): {config.convergence_ns}")
    print(f"Hyperprior centers (log sigma, log lambda): {config.base.hyperpriors.mean_vector}")
    print(
        f"References: each method's own estimate at N={max_n} "
        f"(JEEDS→JEEDS@{max_n}, H-JEEDS→H-JEEDS@{max_n}). "
        "Optional independent JEEDS full-window fit uses "
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
    print("  3. Fit full-window independent JEEDS per agent (compat / metadata)")
    print("  4. For each N, take newest-N pitches (game_date descending)")
    print("  5. Fit independent JEEDS + hierarchical HJEEDS on the same N pitches")
    print(
        f"  6. Record drift |method_N - method_{max_n}| for sigma and log-lambda "
        "(per-method self-reference)"
    )
    print(
        "  7. Write convergence_agent_level_results.csv, summary_by_N.csv, "
        "drift_by_N.png (+ proportional companion); if bbip_selection is present, "
        "also write separability_by_N.png / CSV / summary"
    )
    print()
    print(
        f"Upper-bound pitch-surface builds (no cache reuse): ~{pitch_builds} "
        f"({len(config.agent_specs)} agents x max build depth)."
    )


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
    run_provenance: dict[str, Any],
    all_data: pd.DataFrame | None = None,
    require_paper_configuration: bool | None = None,
) -> Path:
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
    roster_temporary = path.with_suffix(path.suffix + ".tmp")
    roster_temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    roster_temporary.replace(path)

    metadata: dict[str, Any] = {
        "season_year": season_year,
        "pitch_types": list(pitch_types),
        "min_pitches_per_agent": min_pitches_per_agent,
        "max_reference_pitches": max_reference_pitches,
        "convergence_ns": list(convergence_ns),
        "num_agents": len(payload),
        "roster_selector": roster_selector,
        "pitcher_ids": list(roster.pitcher_ids),
        "run_provenance": run_provenance,
        "agent_pitch_counts": [
            {
                "pitcher_id": int(pitcher_id),
                "pitch_type": str(pitch_type),
                "num_available_pitches": int(pitch_count),
            }
            for pitcher_id, pitch_type, pitch_count in roster.agent_pitch_counts
        ],
    }
    bbip_extremes = roster_selector.get("bbip_extremes")
    if bbip_extremes is not None and all_data is not None and season_year is not None:
        from .baseball_bbip import (
            build_bbip_manifest,
            bundled_innings_path_for,
            file_sha256,
            load_bbip_innings_cache,
            resolve_bbip_cache_path,
        )

        metadata["bbip_extremes"] = bbip_extremes
        metadata["bbip_metric_definition"] = _bbip_metric_definition_for(season_year)
        # Rank tiers within the selected roster (eligible extremes), not the full league table.
        metadata["bbip_selection"] = build_bbip_manifest(
            all_data,
            season_year=season_year,
            pitcher_ids=roster.pitcher_ids,
            extremes_count=bbip_extremes,
            output_dir=output_dir,
            eligible_pitcher_ids=roster.pitcher_ids,
        )
        cache_path = resolve_bbip_cache_path(output_dir=output_dir)
        if cache_path is None:
            raise FileNotFoundError(
                "Walk/IP-proxy roster metadata requires a resolved innings cache."
            )
        cached_year, _innings, source = load_bbip_innings_cache(cache_path)
        cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata["bbip_innings_provenance"] = {
            "season_year": cached_year,
            "source": source,
            "cache_filename": cache_path.name,
            "cache_sha256": file_sha256(cache_path),
            "cache_source_artifact": cache_payload.get("source_artifact"),
        }
        bundled_path = bundled_innings_path_for(season_year)
        if bundled_path.is_file():
            metadata["bbip_innings_provenance"]["bundled_artifact"] = {
                "filename": bundled_path.name,
                "sha256": file_sha256(bundled_path),
            }

        is_paper_bbip20 = (
            season_year == 2021
            and list(pitch_types) == ["FF"]
            and roster_selector == PAPER_BBIP20_ROSTER_SELECTOR
            and int(min_pitches_per_agent) == 100
            and max_reference_pitches == 100
            and list(convergence_ns) == [5, 10, 25, 50, 100]
        )
        if is_paper_bbip20 and paper_config_required(require_paper_configuration):
            actual_roster_hash = _spec_roster_fingerprint(roster.agent_specs)
            if actual_roster_hash != PAPER_BBIP20_ROSTER_SHA256:
                raise ValueError(
                    f"Resolved paper BBIP20 roster hash {actual_roster_hash} does not match "
                    f"the pinned roster {PAPER_BBIP20_ROSTER_SHA256}."
                )
            expected_artifact = {
                "filename": "baseball_innings_pitched_2021.json",
                "sha256": PAPER_BBIP_2021_INNINGS_SHA256,
            }
            innings_provenance = metadata["bbip_innings_provenance"]
            if innings_provenance.get("bundled_artifact") != expected_artifact:
                raise ValueError("Paper BBIP20 bundled innings artifact hash is stale.")
            if innings_provenance.get("cache_source_artifact") != expected_artifact:
                raise ValueError("Paper BBIP20 cache is not linked to the bundled innings artifact.")

    _atomic_write_json(output_dir / CONVERGENCE_ROSTER_METADATA_FILENAME, metadata)
    return path


def load_convergence_roster(output_dir: Path) -> tuple[StatcastAgentSpec, ...]:
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
    all_data: pd.DataFrame,
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
    all_observations = build_pitch_observations_for_rows(built_rows, runtime, execution_skills)
    take_n_by_n = {
        str(convergence_n): min(convergence_n, len(all_observations))
        for convergence_n in config.convergence_ns
    }
    requested_prefixes = sorted(
        {reference_row_count, *(int(value) for value in take_n_by_n.values())}
    )
    log_likelihood_grids = compute_baseball_log_likelihood_grids_by_prefix(
        pitch_observations=all_observations,
        prefix_lengths=requested_prefixes,
        possible_targets_feet=runtime.grids.possible_targets_feet,
        all_covs=runtime.all_covs,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
        delta=config.base.delta,
    )
    reference_log_likelihood = log_likelihood_grids[reference_row_count]
    reference_estimate = run_independent_jeeds_baseline(
        log_likelihood_grid=reference_log_likelihood,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )

    log_likelihood_grids_by_n: dict[str, list[list[float]]] = {}
    jeeds_estimates_by_n: dict[str, dict[str, Any]] = {}
    for convergence_n in config.convergence_ns:
        take_n = take_n_by_n[str(convergence_n)]
        log_likelihood_grid = log_likelihood_grids[take_n]
        jeeds_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        key = str(convergence_n)
        log_likelihood_grids_by_n[key] = log_likelihood_grid.tolist()
        jeeds_estimates_by_n[key] = _estimate_to_dict(jeeds_estimate)

    return {
        "seed": seed,
        "agent_id": agent_spec.agent_id,
        "pitcher_id": agent_spec.pitcher_id,
        "pitch_type": agent_spec.pitch_type,
        "convergence_ns": list(config.convergence_ns),
        "num_reference_observations": reference_row_count,
        "reference_estimate": _estimate_to_dict(reference_estimate),
        "log_likelihood_grids_by_n": log_likelihood_grids_by_n,
        "jeeds_estimates_by_n": jeeds_estimates_by_n,
        "take_n_by_n": take_n_by_n,
        "provenance": build_convergence_run_provenance(
            config,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        ),
    }


def write_agent_convergence_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_agent_convergence_cache(path: Path) -> dict[str, Any]:
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
        payload = load_agent_convergence_cache(path)
        identity = (
            int(payload.get("agent_id", -1)),
            int(payload.get("pitcher_id", -1)),
            str(payload.get("pitch_type", "")),
        )
        expected_identity = (
            agent_spec.agent_id,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
        )
        if identity != expected_identity:
            raise ValueError(
                f"Cache {path} identifies agent {identity}; expected {expected_identity}."
            )
        caches.append(payload)
    caches.sort(key=lambda payload: int(payload["agent_id"]))
    return caches, missing_agent_ids


def run_convergence_from_agent_caches(
    config: BaseballConvergenceConfig,
    seed: int,
    agent_caches: Sequence[dict[str, Any]],
    *,
    sigma_grid: np.ndarray | None = None,
    log_lambda_grid: np.ndarray | None = None,
) -> BaseballConvergenceSeedResult:
    """Run population MAP + hierarchical passes from per-agent cached grids.

    Drift for each method is relative to that method's own estimate at
    ``max(convergence_ns)`` (JEEDS→JEEDS@N_max, H-JEEDS→H-JEEDS@N_max).

    Optional ``sigma_grid`` / ``log_lambda_grid`` avoid rebuilding grids when the
    caller already has them (local sequential seed path). Pass both or neither.
    """

    if (sigma_grid is None) ^ (log_lambda_grid is None):
        raise ValueError("Pass both sigma_grid and log_lambda_grid, or neither.")
    if sigma_grid is None:
        sigma_grid, log_lambda_grid = build_baseball_skill_grids_from_convergence(config)
    max_n = max(config.convergence_ns)
    seed_result = BaseballConvergenceSeedResult(
        seed=seed,
        notes=(
            "Statcast baseball convergence study "
            f"(per-method self-reference at N={max_n})."
        ),
    )

    # (agent_id, N) -> packed estimate metadata used after both methods are known at max N.
    pending: list[
        tuple[
            StatcastAgentSpec,
            int,
            int,
            MethodEstimate,
            MethodEstimate,
            MethodEstimate,
            int,
        ]
    ] = []
    final_estimates: dict[int, tuple[MethodEstimate, MethodEstimate]] = {}
    population_fit_diagnostics: list[dict[str, Any]] = []

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
            diagnostic = {
                "convergence_n": int(convergence_n),
                **_json_safe(fitted_hyperparameters),
            }
            population_fit_diagnostics.append(diagnostic)
            diagnostics_path = _persist_population_fit_diagnostics(
                config.base.output_dir,
                seed=seed,
                records=population_fit_diagnostics,
            )
            if not bool(fitted_hyperparameters.get("converged", False)):
                raise RuntimeError(
                    f"Population MAP fit did not converge for seed={seed}, N={convergence_n}. "
                    f"Diagnostics were written to {diagnostics_path}."
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
            pending.append(
                (
                    agent_spec,
                    convergence_n,
                    take_n,
                    jeeds_estimate,
                    hierarchical_estimate,
                    reference_estimate,
                    num_reference_observations,
                )
            )
            if convergence_n == max_n:
                final_estimates[agent_spec.agent_id] = (jeeds_estimate, hierarchical_estimate)

    for (
        agent_spec,
        convergence_n,
        take_n,
        jeeds_estimate,
        hierarchical_estimate,
        reference_estimate,
        num_reference_observations,
    ) in pending:
        finals = final_estimates.get(agent_spec.agent_id)
        if finals is None:
            jeeds_sigma_drift = jeeds_lambda_drift = None
            hier_sigma_drift = hier_lambda_drift = None
            hierarchical_closer_sigma = hierarchical_closer_log_lambda = None
        else:
            jeeds_at_max, hierarchical_at_max = finals
            jeeds_sigma_drift, jeeds_lambda_drift = _abs_drift(jeeds_estimate, jeeds_at_max)
            hier_sigma_drift, hier_lambda_drift = _abs_drift(
                hierarchical_estimate, hierarchical_at_max
            )
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
                    f"N={take_n}; self-reference at N={max_n}."
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
    all_data: pd.DataFrame | None = None,
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
    """Write the proxy innings cache (if needed) and freeze the roster.

    Reuses the roster already resolved inside
    ``_build_convergence_config_and_data`` so prepare-roster and later
    ``--use-prepared-roster`` workers share one selection path (including
    ``output_dir`` for the walk/IP-proxy innings cache).
    """

    if getattr(args, "bbip_extremes", None) is not None:
        if args.season_year is None:
            raise ValueError("--bbip-extremes requires --season-year.")
        from .baseball_bbip import write_bbip_innings_cache

        cache_path = write_bbip_innings_cache(Path(args.output_dir), season_year=args.season_year)
        print(
            f"[baseball-convergence] Wrote walk/IP-proxy innings cache to "
            f"{cache_path.resolve()}",
            flush=True,
        )

    config, all_data = _build_convergence_config_and_data(args)
    roster = BaseballRosterSelection(
        season_year=config.season_year,
        pitch_types=config.pitch_types,
        pitcher_ids=config.pitcher_ids,
        agent_specs=config.agent_specs,
        agent_pitch_counts=config.agent_pitch_counts,
        excluded_agents=config.excluded_agents,
    )
    run_provenance = build_convergence_run_provenance(config)
    path = write_convergence_roster(
        config.base.output_dir,
        roster,
        season_year=args.season_year,
        pitch_types=config.pitch_types,
        min_pitches_per_agent=config.min_pitches_per_agent,
        max_reference_pitches=config.max_reference_pitches,
        convergence_ns=config.convergence_ns,
        roster_selector=roster_selector_kwargs_from_args(args),
        run_provenance=run_provenance,
        all_data=all_data,
        require_paper_configuration=getattr(args, "require_paper_config", False),
    )
    # Preparing a new distributed run must invalidate any older complete bundle
    # immediately. Otherwise an identical frozen roster could leave stale paper
    # outputs temporarily plot-valid while replacement agent tasks are running.
    begin_convergence_run_metadata(
        config.base.output_dir,
        args=args,
        config=config,
        run_provenance=run_provenance,
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
    config = _config_with_agent_specs(build_baseball_convergence_config_from_args(args), roster)
    metadata = load_convergence_roster_metadata(output_dir)
    expected_provenance = build_convergence_run_provenance(config)
    validate_convergence_metadata_against_config(
        metadata,
        config,
        expected_provenance,
        require_paper_configuration=getattr(args, "require_paper_config", False),
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
    config = _config_with_agent_specs(build_baseball_convergence_config_from_args(args), roster)
    metadata = load_convergence_roster_metadata(output_dir)
    expected_provenance = build_convergence_run_provenance(config)
    validate_convergence_metadata_against_config(
        metadata,
        config,
        expected_provenance,
        require_paper_configuration=getattr(args, "require_paper_config", False),
    )
    begin_convergence_run_metadata(
        output_dir,
        args=args,
        config=config,
        run_provenance=expected_provenance,
    )
    caches, missing_agent_ids = load_agent_convergence_caches(output_dir, roster)
    if missing_agent_ids:
        message = (
            f"Missing {len(missing_agent_ids)} agent cache files for IDs {missing_agent_ids}; "
            "aggregating them would change the paper cohort."
        )
        if not bool(getattr(args, "allow_partial_caches", False)):
            raise FileNotFoundError(message)
        print(f"[baseball-convergence] NON-PAPER WARNING: {message}", flush=True)
    if not caches:
        raise FileNotFoundError("No agent convergence caches found for aggregation.")

    expected_ns = tuple(config.convergence_ns)
    available_counts = {
        (int(row["pitcher_id"]), str(row["pitch_type"])): int(row["num_available_pitches"])
        for row in metadata.get("agent_pitch_counts", [])
    }
    if len(available_counts) != len(roster):
        raise ValueError(
            "Convergence roster metadata has incomplete agent pitch counts; prepare the roster again."
        )
    for cache in caches:
        require_matching_fingerprint(
            cache.get("provenance"),
            expected_provenance,
            label=f"agent {cache.get('agent_id')} convergence cache",
        )
        cached_ns = tuple(int(value) for value in cache.get("convergence_ns", ()))
        if cached_ns != expected_ns:
            raise ValueError(
                f"Agent {cache['agent_id']} cache checkpoints {cached_ns} do not match "
                f"the aggregation config {expected_ns}."
            )
        expected_keys = {str(value) for value in expected_ns}
        for field in ("log_likelihood_grids_by_n", "jeeds_estimates_by_n", "take_n_by_n"):
            actual_keys = set(cache.get(field, {}))
            if actual_keys != expected_keys:
                raise ValueError(
                    f"Agent {cache['agent_id']} cache field {field} has checkpoints "
                    f"{sorted(actual_keys)}; expected {sorted(expected_keys)}."
                )

        cache_key = (int(cache["pitcher_id"]), str(cache["pitch_type"]))
        available = available_counts[cache_key]
        expected_reference = (
            available
            if config.max_reference_pitches is None
            else min(available, int(config.max_reference_pitches))
        )
        if int(cache.get("num_reference_observations", -1)) != expected_reference:
            raise ValueError(
                f"Agent {cache['agent_id']} cache has "
                f"num_reference_observations={cache.get('num_reference_observations')}; "
                f"expected {expected_reference}."
            )
        expected_take_n = {str(value): min(int(value), available) for value in expected_ns}
        actual_take_n = {
            str(key): int(value) for key, value in cache.get("take_n_by_n", {}).items()
        }
        if actual_take_n != expected_take_n:
            raise ValueError(
                f"Agent {cache['agent_id']} cache observation counts {actual_take_n}; "
                f"expected {expected_take_n}."
            )

        expected_shape = (len(expected_provenance["sigma_grid"]), len(expected_provenance["log_lambda_grid"]))
        for checkpoint, values in cache.get("log_likelihood_grids_by_n", {}).items():
            grid = np.asarray(values, dtype=float)
            if grid.shape != expected_shape or not np.all(np.isfinite(grid)):
                raise ValueError(
                    f"Agent {cache['agent_id']} checkpoint {checkpoint} likelihood grid "
                    f"has shape {grid.shape} / finite={bool(np.all(np.isfinite(grid)))}; "
                    f"expected finite {expected_shape}."
                )

    seed_results = []
    for seed in config.seed_values:
        seed_results.append(run_convergence_from_agent_caches(config, seed, caches))

    all_agent_results = [result for seed_result in seed_results for result in seed_result.agent_results]
    summary_by_n_rows, summary_overall_rows = aggregate_convergence_across_seeds(seed_results)
    return all_agent_results, summary_by_n_rows, summary_overall_rows
