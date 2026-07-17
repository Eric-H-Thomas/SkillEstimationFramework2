# This file has been fully reviewed by a human researcher as of 07/17/26 at 1:40 PM MDT.
"""Statcast baseball seed pipeline for hierarchical vs independent JEEDS.

Used by ``baseball_hierarchical_vs_jeeds`` (Phase 1 / exploratory Statcast runs).
Paper BBIP convergence uses ``baseball_convergence`` /
``submit_hjeeds_baseball_convergence_paper_bbip.sh`` instead — this module does
not drive that path.

Per seed: load Statcast, build per-agent likelihood grids and JEEDS baselines,
fit one shared hierarchical prior, then write hierarchical posteriors.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

from .baseball_config import BaseballExperimentConfig, build_baseball_skill_grids
from .baseball_likelihood import compute_baseball_log_likelihood_grid
from .baseball_pitch import (
    PitchObservation,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observations_for_rows,
    get_agent_pitch_rows,
)
from .baseball_roster import load_statcast_for_roster
from .estimation import (
    build_discrete_hierarchical_prior,
    fit_population_hyperparameters_map,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from .models import BaseballSeedResult, MethodEstimate, StatcastAgentResult
from .sampling import assign_observation_counts


class _AgentInferenceRecord(NamedTuple):
    """Per-agent JEEDS pass held until the shared hierarchical prior is fit."""

    agent_spec: StatcastAgentSpec
    count_bucket: int
    pitch_observations: list[PitchObservation]
    log_likelihood_grid: np.ndarray
    jeeds_estimate: MethodEstimate


def _no_data_estimates() -> tuple[MethodEstimate, MethodEstimate]:
    note = "No Statcast rows for this agent."
    return (
        MethodEstimate(method_name="jeeds", status="no_data", notes=note),
        MethodEstimate(method_name="hierarchical", status="no_data", notes=note),
    )


def _pitch_take_limit(
    config: BaseballExperimentConfig,
    agent_spec: StatcastAgentSpec,
    observation_counts: Sequence[int],
) -> int | None:
    """Newest-first row cap for one agent, or ``None`` to keep all available pitches."""

    limit: int | None = (
        None if config.use_natural_pitch_counts else int(observation_counts[agent_spec.agent_id])
    )
    cap = config.max_pitches_per_agent
    if cap is None:
        return limit
    cap_i = int(cap)
    return cap_i if limit is None else min(limit, cap_i)


def _count_bucket_for_agent(
    config: BaseballExperimentConfig,
    agent_spec: StatcastAgentSpec,
    observation_counts: Sequence[int],
) -> int:
    """Design bucket label for CSV/reporting (not the realized pitch count).

    Natural-count runs use ``0`` to match ``count_buckets=(0,)`` in config.
    Realized pitch totals belong in ``num_observations``.
    """

    if config.use_natural_pitch_counts:
        return 0
    return int(observation_counts[agent_spec.agent_id])


def run_single_baseball_seed(
    config: BaseballExperimentConfig,
    seed: int,
) -> BaseballSeedResult:
    """Run hierarchical-vs-JEEDS inference for one seed on Statcast data.

    Note: ``seed`` does not change Statcast estimates. Execution-noise PDFs are built via
    ``multivariate_normal.pdf`` (deterministic given mean/cov); pitch rows are selected by
    newest ``game_date``, not by RNG. Kept only for shared ExperimentConfig / darts CLI shape.
    Prefer ``num_seeds=1``.
    """

    rng = np.random.default_rng(seed)
    sigma_grid, log_lambda_grid = build_baseball_skill_grids(config)
    execution_skills = tuple(float(value) for value in sigma_grid)
    runtime = build_baseball_runtime(rng, execution_skills, delta=config.base.delta)
    all_data = load_statcast_for_roster(config.season_year)
    observation_counts = assign_observation_counts(config.base)

    seed_result = BaseballSeedResult(
        seed=seed,
        notes="Statcast baseball HJEEDS seed result (grid JEEDS baseline).",
    )
    agent_records: list[_AgentInferenceRecord] = []

    for agent_spec in config.agent_specs:
        count_bucket = _count_bucket_for_agent(config, agent_spec, observation_counts)
        agent_rows = get_agent_pitch_rows(
            all_data,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
            max_rows=_pitch_take_limit(config, agent_spec, observation_counts),
        )
        if len(agent_rows) == 0:
            jeeds_estimate, hierarchical_estimate = _no_data_estimates()
            seed_result.agent_results.append(
                StatcastAgentResult(
                    seed=seed,
                    agent_id=agent_spec.agent_id,
                    pitcher_id=agent_spec.pitcher_id,
                    pitch_type=agent_spec.pitch_type,
                    count_bucket=count_bucket,
                    num_observations=0,
                    jeeds=jeeds_estimate,
                    hierarchical=hierarchical_estimate,
                    notes="Agent skipped due to missing data.",
                )
            )
            continue

        pitch_observations = build_pitch_observations_for_rows(
            agent_rows,
            runtime,
            execution_skills,
        )
        log_likelihood_grid = compute_baseball_log_likelihood_grid(
            pitch_observations=pitch_observations,
            possible_targets_feet=runtime.grids.possible_targets_feet,
            all_covs=runtime.all_covs,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
            delta=config.base.delta,
        )
        jeeds_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        agent_records.append(
            _AgentInferenceRecord(
                agent_spec=agent_spec,
                count_bucket=count_bucket,
                pitch_observations=pitch_observations,
                log_likelihood_grid=log_likelihood_grid,
                jeeds_estimate=jeeds_estimate,
            )
        )

    if not agent_records:
        return seed_result

    fitted_hyperparameters = fit_population_hyperparameters_map(
        config=config.base,
        agent_log_likelihoods=[record.log_likelihood_grid for record in agent_records],
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )
    discrete_hierarchical_prior = build_discrete_hierarchical_prior(
        fitted_hyperparameters=fitted_hyperparameters,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )
    seed_result.notes = (
        f"Hierarchical fit converged={fitted_hyperparameters.get('converged')}; "
        f"objective={fitted_hyperparameters.get('objective_value')}."
    )

    for record in agent_records:
        hierarchical_estimate = run_hierarchical_estimator(
            log_likelihood_grid=record.log_likelihood_grid,
            discrete_prior=discrete_hierarchical_prior,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        seed_result.agent_results.append(
            StatcastAgentResult(
                seed=seed,
                agent_id=record.agent_spec.agent_id,
                pitcher_id=record.agent_spec.pitcher_id,
                pitch_type=record.agent_spec.pitch_type,
                count_bucket=record.count_bucket,
                num_observations=len(record.pitch_observations),
                jeeds=record.jeeds_estimate,
                hierarchical=hierarchical_estimate,
                notes=(
                    f"Pitcher {record.agent_spec.pitcher_id} {record.agent_spec.pitch_type}; "
                    f"{len(record.pitch_observations)} pitches."
                ),
            )
        )

    return seed_result
