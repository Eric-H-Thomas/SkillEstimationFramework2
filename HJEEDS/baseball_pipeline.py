# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Statcast baseball seed pipeline for HJEEDS."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .baseball_config import BaseballExperimentConfig, build_baseball_skill_grids
from .baseball_likelihood import compute_baseball_log_likelihood_grid
from .baseball_pitch import (
    PitchObservation,
    StatcastAgentSpec,
    build_baseball_runtime,
    build_pitch_observation,
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


def _build_pitch_observations_for_agent(
    agent_rows,
    runtime,
    execution_skills: Sequence[float],
) -> list[PitchObservation]:
    observations: list[PitchObservation] = []
    for _, row in agent_rows.iterrows():
        observations.append(build_pitch_observation(row, runtime, execution_skills))
    return observations


def _observation_target_for_agent(
    config: BaseballExperimentConfig,
    agent_spec: StatcastAgentSpec,
    observation_counts: Sequence[int],
) -> int:
    if config.use_natural_pitch_counts:
        return 10**9
    return observation_counts[agent_spec.agent_id]


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
    agent_records: list[
        tuple[StatcastAgentSpec, int, list[PitchObservation], np.ndarray, MethodEstimate]
    ] = []

    for agent_spec in config.agent_specs:
        target_count = _observation_target_for_agent(config, agent_spec, observation_counts)
        agent_rows = get_agent_pitch_rows(
            all_data,
            agent_spec.pitcher_id,
            agent_spec.pitch_type,
        )
        if len(agent_rows) == 0:
            seed_result.agent_results.append(
                StatcastAgentResult(
                    seed=seed,
                    agent_id=agent_spec.agent_id,
                    pitcher_id=agent_spec.pitcher_id,
                    pitch_type=agent_spec.pitch_type,
                    count_bucket=0,
                    num_observations=0,
                    jeeds=MethodEstimate(
                        method_name="jeeds",
                        status="no_data",
                        notes="No Statcast rows for this agent.",
                    ),
                    hierarchical=MethodEstimate(
                        method_name="hierarchical",
                        status="no_data",
                        notes="No Statcast rows for this agent.",
                    ),
                    notes="Agent skipped due to missing data.",
                )
            )
            continue

        take_n = min(target_count, len(agent_rows))
        if config.max_pitches_per_agent is not None:
            take_n = min(take_n, config.max_pitches_per_agent)
        agent_rows = agent_rows.iloc[:take_n, :]

        pitch_observations = _build_pitch_observations_for_agent(
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
            (agent_spec, len(pitch_observations), pitch_observations, log_likelihood_grid, jeeds_estimate)
        )

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
        seed_result.notes = (
            f"Hierarchical fit converged={fitted_hyperparameters.get('converged')}; "
            f"objective={fitted_hyperparameters.get('objective_value')}."
        )
    else:
        discrete_hierarchical_prior = None

    for agent_spec, count_bucket, pitch_observations, log_likelihood_grid, jeeds_estimate in agent_records:
        hierarchical_estimate = run_hierarchical_estimator(
            log_likelihood_grid=log_likelihood_grid,
            discrete_prior=discrete_hierarchical_prior,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        seed_result.agent_results.append(
            StatcastAgentResult(
                seed=seed,
                agent_id=agent_spec.agent_id,
                pitcher_id=agent_spec.pitcher_id,
                pitch_type=agent_spec.pitch_type,
                count_bucket=count_bucket,
                num_observations=len(pitch_observations),
                jeeds=jeeds_estimate,
                hierarchical=hierarchical_estimate,
                notes=(
                    f"Pitcher {agent_spec.pitcher_id} {agent_spec.pitch_type}; "
                    f"{len(pitch_observations)} pitches."
                ),
            )
        )

    return seed_result
