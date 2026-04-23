# This file has been fully verified by a human researcher as of 04/23/26 at 8:57 AM MT.
from __future__ import annotations

import numpy as np

from .aggregation import summarize_seed_results
from .estimation import (
    build_discrete_hierarchical_prior,
    fit_population_hyperparameters_map,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from .likelihood import compute_agent_log_likelihood_grid
from .models import AgentDataset, AgentResult, AgentTruth, ExperimentConfig, MethodEstimate, SeedResult
from .sampling import (
    assign_observation_counts,
    build_skill_grids,
    sample_reward_surface,
    sample_true_population_params,
    simulate_agent_dataset,
)


# This module is the concrete seed-level recipe for the experiment.  The public
# entry script loops over seeds, but each seed follows the exact same sequence
# implemented here: simulate the environment, simulate all demonstrators, fit
# independent JEEDS, fit the hierarchical population prior, and then rerun
# agent-level posteriors under that fitted prior.


def run_single_seed(config: ExperimentConfig, seed: int) -> SeedResult:
    """Run the experiment pipeline for one seed.

    The structure below is intentionally explicit so it is easy to see the
    order in which each seed's simulation and inference steps execute.
    """

    # Every random choice for this seed hangs off one generator so rerunning the
    # same seed reproduces the same reward surface, demonstrators, and throws.
    rng = np.random.default_rng(seed)
    reward_surface = sample_reward_surface(rng, config)
    sigma_grid, log_lambda_grid = build_skill_grids(config)
    agent_truths = sample_true_population_params(rng, config, sigma_grid, log_lambda_grid)
    observation_counts = assign_observation_counts(config)

    seed_result = SeedResult(
        seed=seed,
        reward_surface=reward_surface,
        agent_truths=agent_truths,
        notes="Seed result includes standalone JEEDS estimates and hierarchical empirical-Bayes estimates.",
    )

    # Temporary cache of the agent's true skill profile, dataset, log likelihood grid, and JEEDS estimate
    agent_records: list[tuple[AgentTruth, AgentDataset, np.ndarray, MethodEstimate]] = []

    for agent_truth, num_observations in zip(agent_truths, observation_counts):
        # Stage 1: simulate one demonstrator's observed actions from the true
        # latent skill profile.
        dataset = simulate_agent_dataset(
            rng=rng,
            seed=seed,
            config=config,
            reward_surface=reward_surface,
            agent_truth=agent_truth,
            num_observations=num_observations,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        seed_result.agent_datasets.append(dataset)

        # Stage 2: turn those observations into the JEEDS log-likelihood grid
        # over all candidate ``(sigma, log lambda)`` pairs.
        log_likelihood_grid = compute_agent_log_likelihood_grid(
            config=config,
            reward_surface=reward_surface,
            agent_dataset=dataset,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )

        # Stage 3: compute the independent JEEDS posterior using a uniform
        # prior.  This is the non-hierarchical baseline from the paper.
        jeeds_estimate = run_independent_jeeds_baseline(
            log_likelihood_grid=log_likelihood_grid,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )

        agent_records.append((agent_truth, dataset, log_likelihood_grid, jeeds_estimate))

    # Only after we have all per-agent likelihood grids can we fit the shared
    # population prior.  That is what allows H-JEEDS to borrow strength across
    # demonstrators in low-data settings.
    fitted_hyperparameters = fit_population_hyperparameters_map(
        config=config,
        agent_log_likelihoods=[record[2] for record in agent_records],
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )
    discrete_hierarchical_prior = build_discrete_hierarchical_prior(
        fitted_hyperparameters=fitted_hyperparameters,
        sigma_grid=sigma_grid,
        log_lambda_grid=log_lambda_grid,
    )

    seed_result.notes = (
        "Seed result includes standalone JEEDS estimates and hierarchical empirical-Bayes estimates. "
        f"Hierarchical population fit status: converged={fitted_hyperparameters.get('converged')}; "
        f"objective={fitted_hyperparameters.get('objective_value')}."
    )

    for agent_truth, dataset, log_likelihood_grid, jeeds_estimate in agent_records:
        # Reuse the already-computed likelihood grid and swap in the fitted
        # population prior.  This mirrors the paper's "same likelihood,
        # different prior" comparison between JEEDS and H-JEEDS.
        hierarchical_estimate = run_hierarchical_estimator(
            log_likelihood_grid=log_likelihood_grid,
            discrete_prior=discrete_hierarchical_prior,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )

        seed_result.agent_results.append(
            AgentResult(
                seed=seed,
                agent_id=agent_truth.agent_id,
                count_bucket=dataset.count_bucket,
                num_observations=dataset.num_observations,
                sigma_true=agent_truth.sigma_true,
                log_lambda_true=agent_truth.log_lambda_true,
                jeeds=jeeds_estimate,
                hierarchical=hierarchical_estimate,
                notes=(
                    "Agent result contains standalone JEEDS and hierarchical empirical-Bayes estimates."
                ),
            )
        )

    # The seed summary is what later gets aggregated across random seeds to
    # produce the final means and confidence intervals reported in tables/plots.
    seed_result.summary_by_bucket_rows, seed_result.summary_overall_rows = summarize_seed_results(seed_result)
    return seed_result
