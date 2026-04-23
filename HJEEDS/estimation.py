# This file still requires human verification. Delete this comment when done.

from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np
from .models import ExperimentConfig, MethodEstimate


# This module contains the heart of the statistical comparison:
#
# - ``run_independent_jeeds_baseline`` applies a uniform prior to one agent's
#   likelihood grid, reproducing the non-hierarchical JEEDS baseline.
# - ``fit_population_hyperparameters_map`` fits the shared population prior used
#   by H-JEEDS via an empirical-Bayes MAP objective.
# - ``build_discrete_hierarchical_prior`` converts that continuous log-space
#   population model into the discrete grid used by JEEDS.
# - ``run_hierarchical_estimator`` recomputes agent posteriors under the fitted
#   shared prior.


def run_independent_jeeds_baseline(
    log_likelihood_grid: np.ndarray,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> MethodEstimate:
    """Infer one agent's skill under the independent JEEDS baseline.

    Use a uniform prior over the ``(sigma, log lambda)`` grid cells, so the
    posterior is proportional to the likelihood table computed for this agent.
    The returned estimate reports the posterior mean of sigma on its original
    scale and the posterior mean of log(lambda) on the canonical decision-skill
    axis, plus the MAP grid cell.
    """

    sigma_grid = np.asarray(sigma_grid, dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    expected_shape = (len(sigma_grid), len(log_lambda_grid))
    if log_likelihood_grid.shape != expected_shape:
        raise ValueError(
            "log_likelihood_grid has the wrong shape for the provided skill grids: "
            f"{log_likelihood_grid.shape} vs {expected_shape}."
        )

    # Unsupported or numerically invalid cells stay excluded from the posterior
    # by keeping them at ``-inf`` weight.
    finite_mask = np.isfinite(log_likelihood_grid)
    if not np.any(finite_mask):
        return MethodEstimate(
            method_name="jeeds",
            status="no_finite_likelihood",
            notes="All entries in the standalone JEEDS log-likelihood grid were non-finite.",
        )

    # Under a uniform prior, the log posterior differs from the log likelihood
    # by an additive constant. We subtract the global maximum before
    # exponentiating so the normalization is stable.
    stabilized_log_posterior = np.full_like(log_likelihood_grid, -np.inf, dtype=float)
    max_log_likelihood = float(np.max(log_likelihood_grid[finite_mask]))
    stabilized_log_posterior[finite_mask] = log_likelihood_grid[finite_mask] - max_log_likelihood

    posterior_unnormalized = np.zeros_like(log_likelihood_grid, dtype=float)
    posterior_unnormalized[finite_mask] = np.exp(stabilized_log_posterior[finite_mask])
    normalization = float(np.sum(posterior_unnormalized))

    if normalization <= 0.0 or not np.isfinite(normalization):
        return MethodEstimate(
            method_name="jeeds",
            status="invalid_posterior_normalization",
            notes="Failed to normalize the standalone JEEDS posterior under a uniform prior.",
        )

    posterior = posterior_unnormalized / normalization  # shape: (S, L)

    # Marginalize the 2D posterior down to one distribution over sigma and one
    # over log(lambda) so we can report posterior means on each axis separately.
    sigma_marginal = np.sum(posterior, axis=1)  # shape: (S,)
    lambda_marginal = np.sum(posterior, axis=0)  # shape: (L,)

    posterior_mean_sigma = float(np.dot(sigma_marginal, sigma_grid))
    posterior_mean_log_lambda = float(np.dot(lambda_marginal, log_lambda_grid))

    # Also record the MAP cell because it is a useful diagnostic when reviewing
    # how concentrated or multimodal the posterior may be.
    map_index = int(np.argmax(posterior))
    sigma_map_index, lambda_map_index = np.unravel_index(map_index, posterior.shape)
    map_sigma = float(sigma_grid[sigma_map_index])
    map_log_lambda = float(log_lambda_grid[lambda_map_index])

    return MethodEstimate(
        method_name="jeeds",
        posterior_mean_sigma=posterior_mean_sigma,
        posterior_mean_log_lambda=posterior_mean_log_lambda,
        map_sigma=map_sigma,
        map_log_lambda=map_log_lambda,
        status="ok",
        notes="Standalone JEEDS posterior computed with a uniform prior over the sigma x log-lambda grid.",
    )


def fit_population_hyperparameters_map(
    config: ExperimentConfig,
    agent_log_likelihoods: Sequence[np.ndarray],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> dict[str, Any]:
    """Fit the population hyperparameters for the hierarchical model.

    This implementation performs a lightweight empirical-Bayes MAP fit over the
    population parameters by maximizing:

    1. The marginal likelihood of each agent's JEEDS grid under the current
       population prior, plus
    2. The paper's hyperpriors over ``mu``, ``log_tau_eta``,
       ``log_tau_rho``, and ``zeta_r``.

    We optimize in an unconstrained parameter space with SciPy:

    - ``tau_eta = exp(log_tau_eta)``
    - ``tau_rho = exp(log_tau_rho)``
    - ``r = tanh(zeta_r)``

    The objective is still approximate relative to the full hierarchical model
    in the paper draft because it relies on the discretized prior described in
    ``build_discrete_hierarchical_prior``. But this gives us a concrete,
    testable MAP fit that can drive the first end-to-end experiments.
    """
    from scipy import optimize

    # Convert everything to NumPy arrays immediately so the rest of the code can
    # rely on consistent shapes and dtypes.
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    expected_shape = (len(sigma_grid), len(log_lambda_grid))

    if sigma_grid.ndim != 1 or log_lambda_grid.ndim != 1:
        raise ValueError("sigma_grid and log_lambda_grid must both be one-dimensional arrays.")
    if sigma_grid.size == 0 or log_lambda_grid.size == 0:
        raise ValueError("sigma_grid and log_lambda_grid must both be non-empty.")

    # Validate the per-agent likelihood grids before optimization starts.  This
    # keeps the optimizer from failing later with a shape error that would be
    # much harder to interpret.
    validated_log_likelihoods: list[np.ndarray] = []
    for agent_index, grid in enumerate(agent_log_likelihoods):
        grid = np.asarray(grid, dtype=float)
        if grid.shape != expected_shape:
            raise ValueError(
                "Each agent log-likelihood grid must match the JEEDS skill-grid shape. "
                f"Agent {agent_index} had shape {grid.shape}; expected {expected_shape}."
            )
        validated_log_likelihoods.append(grid)

    # The paper's hyperpriors live in log-skill space, so these are the priors
    # over the population mean and spread parameters before seeing any agents.
    m0 = np.asarray(config.hyperpriors.mean_vector, dtype=float)  # shape: (2,)
    s0 = np.asarray(config.hyperpriors.covariance_matrix, dtype=float)  # shape: (2, 2)
    s0_inverse = np.linalg.inv(s0)
    log_tau_eta_mean = float(config.hyperpriors.log_tau_eta_mean)
    log_tau_eta_sd = float(config.hyperpriors.log_tau_eta_sd)
    log_tau_rho_mean = float(config.hyperpriors.log_tau_rho_mean)
    log_tau_rho_sd = float(config.hyperpriors.log_tau_rho_sd)
    m_r = float(config.hyperpriors.m_r)
    s_r = float(config.hyperpriors.s_r)

    if log_tau_eta_sd <= 0.0 or log_tau_rho_sd <= 0.0:
        raise ValueError("Log-tau prior standard deviations must be positive.")
    if s_r <= 0.0:
        raise ValueError("Correlation prior standard deviation must be positive.")

    def unpack_parameters(parameter_vector: np.ndarray) -> dict[str, Any]:
        """Map unconstrained search coordinates into model parameters."""

        # Powell works best in an unconstrained space, so the optimizer searches
        # over log-variances and Fisher-z-like correlation coordinates, then we
        # transform those back to the constrained statistical parameters here.
        mu_eta, mu_rho, log_tau_eta, log_tau_rho, zeta_r = parameter_vector.tolist()
        tau_eta = math.exp(log_tau_eta)
        tau_rho = math.exp(log_tau_rho)
        correlation = math.tanh(zeta_r)
        covariance = np.array(
            [
                [tau_eta**2, correlation * tau_eta * tau_rho],
                [correlation * tau_eta * tau_rho, tau_rho**2],
            ],
            dtype=float,
        )
        return {
            "mu_eta": float(mu_eta),
            "mu_rho": float(mu_rho),
            "mu": np.array([mu_eta, mu_rho], dtype=float),
            "tau_eta": float(tau_eta),
            "tau_rho": float(tau_rho),
            "zeta_r": float(zeta_r),
            "r": float(correlation),
            "correlation": float(correlation),
            "covariance_matrix": covariance,
        }

    def evaluate_log_posterior(parameter_vector: np.ndarray) -> float:
        """Return the approximate population log posterior at one parameter point."""

        unpacked = unpack_parameters(parameter_vector)

        try:
            # This is where the continuous population model gets projected onto
            # the discrete JEEDS grid used by every downstream posterior.
            discrete_prior = build_discrete_hierarchical_prior(
                unpacked,
                sigma_grid=sigma_grid,
                log_lambda_grid=log_lambda_grid,
            )
        except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
            return -np.inf

        positive_prior_mask = discrete_prior > 0.0  # shape: (S, L)
        if not np.any(positive_prior_mask):
            return -np.inf

        log_prior_grid = np.full_like(discrete_prior, -np.inf, dtype=float)
        log_prior_grid[positive_prior_mask] = np.log(discrete_prior[positive_prior_mask])

        marginal_log_likelihood = 0.0
        for log_likelihood_grid in validated_log_likelihoods:
            # For each agent, integrate over the latent grid cell by combining
            # the per-agent likelihood with the current population prior.
            supported_cells = np.isfinite(log_likelihood_grid) & positive_prior_mask  # shape: (S, L)
            if not np.any(supported_cells):
                return -np.inf

            log_joint_grid = log_likelihood_grid[supported_cells] + log_prior_grid[supported_cells]
            max_log_joint = float(np.max(log_joint_grid))
            if not np.isfinite(max_log_joint):
                return -np.inf

            marginal_log_likelihood += max_log_joint + math.log(
                float(np.sum(np.exp(log_joint_grid - max_log_joint)))
            )

        # Add the hyperprior contributions after the data likelihood term.  The
        # resulting objective is the empirical-Bayes analogue of a posterior
        # over population parameters.
        mu_centered = unpacked["mu"] - m0  # shape: (2,)
        mu_log_prior = -0.5 * float(mu_centered @ s0_inverse @ mu_centered)
        tau_log_prior = -0.5 * ((parameter_vector[2] - log_tau_eta_mean) / log_tau_eta_sd) ** 2
        tau_log_prior += -0.5 * ((parameter_vector[3] - log_tau_rho_mean) / log_tau_rho_sd) ** 2
        zeta_log_prior = -0.5 * ((unpacked["zeta_r"] - m_r) / s_r) ** 2

        total_log_posterior = marginal_log_likelihood + mu_log_prior + tau_log_prior + zeta_log_prior
        if not np.isfinite(total_log_posterior):
            return -np.inf
        return float(total_log_posterior)

    # Start the optimizer at the centers of the paper's hyperpriors so the
    # default condition begins from a reasonable, interpretable point.
    initial_parameter_vector = np.array(
        [
            m0[0],
            m0[1],
            log_tau_eta_mean,
            log_tau_rho_mean,
            m_r,
        ],
        dtype=float,
    )

    initial_score = evaluate_log_posterior(initial_parameter_vector)
    objective_evaluations = 0

    def objective(parameter_vector: np.ndarray) -> float:
        """Return the negative log posterior for SciPy minimization."""

        nonlocal objective_evaluations
        objective_evaluations += 1

        score = evaluate_log_posterior(parameter_vector)
        if not np.isfinite(score):
            # Powell's method can do awkward arithmetic with infinities while
            # bracketing one-dimensional searches, so use a very large finite
            # penalty for invalid population-parameter proposals.
            return 1e100
        return -float(score)

    optimization_result = optimize.minimize(
        objective,
        initial_parameter_vector,
        method="Powell",
        options={
            "maxiter": 300,
            "xtol": 1e-3,
            "ftol": 1e-3,
            "disp": False,
        },
    )

    final_parameter_vector = np.asarray(optimization_result.x, dtype=float)
    final_score = evaluate_log_posterior(final_parameter_vector)
    objective_evaluations += 1

    # If Powell returns an invalid point, keep the initialization rather than
    # silently emitting unusable hyperparameters.
    if not np.isfinite(final_score):
        final_parameter_vector = initial_parameter_vector.copy()
        final_score = initial_score

    # Package both the fitted statistical parameters and optimizer diagnostics
    # because both are useful when manually checking experimental runs.
    fitted = unpack_parameters(final_parameter_vector)
    fitted["objective_value"] = float(final_score)
    fitted["initial_objective_value"] = float(initial_score)
    fitted["num_objective_evaluations"] = int(max(objective_evaluations, getattr(optimization_result, "nfev", 0)))
    fitted["num_optimizer_iterations"] = int(getattr(optimization_result, "nit", 0))
    fitted["optimization_method"] = "scipy.optimize.minimize(Powell)"
    fitted["converged"] = bool(optimization_result.success)
    fitted["optimizer_message"] = str(optimization_result.message)
    fitted["num_agents"] = len(validated_log_likelihoods)
    fitted["notes"] = (
        "Approximate empirical-Bayes MAP fit over a discretized hierarchical prior "
        "using scipy.optimize.minimize with Powell search in unconstrained coordinates."
    )
    return fitted


def build_discrete_hierarchical_prior(
    fitted_hyperparameters: dict[str, Any],
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> np.ndarray:
    """Discretize the fitted log-space population prior onto the JEEDS grid.

    The fitted population model lives in ``(log sigma, log lambda)`` space, and
    this helper projects it onto the experiment's discrete ``(sigma,
    log lambda)`` grid cells. This bridges the continuous population model to
    the same discrete support used by the agent-level posteriors.

    Implementation note:
    We use a cell-mass approximation rather than exact rectangle integration.
    For each grid point, we:

    1. Construct the surrounding log-space cell boundaries using adjacent grid
       points.
    2. Evaluate the bivariate Normal density at the JEEDS grid point in log
       space.
    3. Multiply by the surrounding cell area in log space.

    This keeps the discretization fully NumPy-based and easy to inspect. If we
    later decide we need exact multivariate Normal rectangle probabilities, we
    can swap this approximation out without changing downstream code.
    """
    # The hierarchical prior is defined continuously in log-skill space, but
    # inference happens on the same discrete grid used by JEEDS.  This function
    # is the bridge between those two representations.
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)

    if sigma_grid.ndim != 1 or log_lambda_grid.ndim != 1:
        raise ValueError("sigma_grid and log_lambda_grid must both be one-dimensional arrays.")
    if sigma_grid.size == 0 or log_lambda_grid.size == 0:
        raise ValueError("sigma_grid and log_lambda_grid must both be non-empty.")
    if np.any(~np.isfinite(sigma_grid)) or np.any(~np.isfinite(log_lambda_grid)):
        raise ValueError("sigma_grid and log_lambda_grid must contain only finite values.")
    if np.any(sigma_grid <= 0.0):
        raise ValueError("sigma_grid must be strictly positive for log-space discretization.")
    if np.any(np.diff(sigma_grid) <= 0.0) or np.any(np.diff(log_lambda_grid) <= 0.0):
        raise ValueError("sigma_grid and log_lambda_grid must be strictly increasing.")

    # Accept a couple of equivalent key layouts so this function stays easy to
    # reuse once the MAP-fitting step is implemented.
    # Accept several equivalent key names so this helper is reusable with both
    # the package's optimizer output and any future external experiments.
    if "mu" in fitted_hyperparameters:
        mu = np.asarray(fitted_hyperparameters["mu"], dtype=float)
    elif "mean_vector" in fitted_hyperparameters:
        mu = np.asarray(fitted_hyperparameters["mean_vector"], dtype=float)
    else:
        mu = np.array(
            [
                fitted_hyperparameters["mu_eta"],
                fitted_hyperparameters["mu_rho"],
            ],
            dtype=float,
        )

    if mu.shape != (2,):
        raise ValueError(f"Population mean must have shape (2,), received {mu.shape}.")

    if "covariance_matrix" in fitted_hyperparameters:
        covariance = np.asarray(fitted_hyperparameters["covariance_matrix"], dtype=float)
    elif "Sigma_z" in fitted_hyperparameters:
        covariance = np.asarray(fitted_hyperparameters["Sigma_z"], dtype=float)
    else:
        tau_eta = float(fitted_hyperparameters["tau_eta"])
        tau_rho = float(fitted_hyperparameters["tau_rho"])
        if "correlation" in fitted_hyperparameters:
            correlation = float(fitted_hyperparameters["correlation"])
        elif "r" in fitted_hyperparameters:
            correlation = float(fitted_hyperparameters["r"])
        elif "zeta_r" in fitted_hyperparameters:
            correlation = math.tanh(float(fitted_hyperparameters["zeta_r"]))
        else:
            raise KeyError(
                "fitted_hyperparameters must include one of: correlation, r, or zeta_r."
            )

        if tau_eta <= 0.0 or tau_rho <= 0.0:
            raise ValueError("tau_eta and tau_rho must both be strictly positive.")
        if not np.isfinite(correlation) or correlation <= -1.0 or correlation >= 1.0:
            raise ValueError("Correlation must be finite and lie strictly between -1 and 1.")

        covariance = np.array(
            [
                [tau_eta**2, correlation * tau_eta * tau_rho],
                [correlation * tau_eta * tau_rho, tau_rho**2],
            ],
            dtype=float,
        )

    if covariance.shape != (2, 2):
        raise ValueError(f"Population covariance must have shape (2, 2), received {covariance.shape}.")
    if np.any(~np.isfinite(covariance)):
        raise ValueError("Population covariance must contain only finite values.")

    determinant = float(np.linalg.det(covariance))
    if determinant <= 0.0 or not np.isfinite(determinant):
        raise ValueError("Population covariance must be positive definite.")

    inverse_covariance = np.linalg.inv(covariance)
    normalization_constant = 1.0 / (2.0 * math.pi * math.sqrt(determinant))

    def log_grid_edges(log_grid: np.ndarray) -> np.ndarray:
        """Construct surrounding cell boundaries in log space."""

        if log_grid.size == 1:
            return np.array([log_grid[0] - 0.5, log_grid[0] + 0.5], dtype=float)

        edges = np.empty(log_grid.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (log_grid[:-1] + log_grid[1:])
        edges[0] = log_grid[0] - 0.5 * (log_grid[1] - log_grid[0])
        edges[-1] = log_grid[-1] + 0.5 * (log_grid[-1] - log_grid[-2])
        return edges

    log_sigma_grid = np.log(sigma_grid)  # shape: (S,)

    # Each discrete JEEDS grid point is treated as representing the local
    # log-space cell around it.  The mass of that cell becomes the prior weight
    # assigned to the corresponding grid point.
    sigma_edges = log_grid_edges(log_sigma_grid)  # shape: (S + 1,)
    lambda_edges = log_grid_edges(log_lambda_grid)  # shape: (L + 1,)
    sigma_cell_widths = np.diff(sigma_edges)  # shape: (S,)
    lambda_cell_widths = np.diff(lambda_edges)  # shape: (L,)

    log_sigma_mesh, log_lambda_mesh = np.meshgrid(
        log_sigma_grid,
        log_lambda_grid,
        indexing="ij",
    )
    log_points = np.stack([log_sigma_mesh, log_lambda_mesh], axis=-1)  # shape: (S, L, 2)
    centered_points = log_points - mu  # shape: (S, L, 2)

    # Evaluate the bivariate Normal density at each JEEDS grid point in
    # log-skill space.
    quadratic_form = np.einsum(
        "...i,ij,...j->...",
        centered_points,
        inverse_covariance,
        centered_points,
    )  # shape: (S, L)
    log_density = math.log(normalization_constant) - 0.5 * quadratic_form  # shape: (S, L)
    density = np.exp(log_density)  # shape: (S, L)

    cell_areas = sigma_cell_widths[:, None] * lambda_cell_widths[None, :]  # shape: (S, L)
    prior_mass = density * cell_areas  # shape: (S, L)

    if np.any(~np.isfinite(prior_mass)) or np.any(prior_mass < 0.0):
        raise RuntimeError("Discrete hierarchical prior contained invalid mass values.")

    total_mass = float(np.sum(prior_mass))
    if total_mass <= 0.0 or not np.isfinite(total_mass):
        raise RuntimeError("Discrete hierarchical prior had zero or non-finite total mass.")

    return prior_mass / total_mass  # shape: (S, L)


def run_hierarchical_estimator(
    log_likelihood_grid: np.ndarray,
    discrete_prior: np.ndarray,
    sigma_grid: np.ndarray,
    log_lambda_grid: np.ndarray,
) -> MethodEstimate:
    """Infer one agent's skill using the fitted hierarchical prior.

    Once the population-level MAP fit has been discretized onto the JEEDS
    hypothesis grid, posterior inference is almost identical to the
    independent JEEDS baseline. The only difference is that we now combine the
    agent-specific log likelihood table with a non-uniform prior over grid
    cells before normalizing.

    This function intentionally mirrors ``run_independent_jeeds_baseline`` so
    the two posterior computations stay easy to compare.
    """
    sigma_grid = np.asarray(sigma_grid, dtype=float)
    log_lambda_grid = np.asarray(log_lambda_grid, dtype=float)
    expected_shape = (len(sigma_grid), len(log_lambda_grid))
    if log_likelihood_grid.shape != expected_shape:
        raise ValueError(
            "log_likelihood_grid has the wrong shape for the provided skill grids: "
            f"{log_likelihood_grid.shape} vs {expected_shape}."
        )
    if discrete_prior.shape != expected_shape:
        raise ValueError(
            "discrete_prior has the wrong shape for the provided skill grids: "
            f"{discrete_prior.shape} vs {expected_shape}."
        )

    # Treat the supplied grid as a mass table rather than assuming it is already
    # perfectly normalized.
    prior = np.asarray(discrete_prior, dtype=float)
    if np.any(~np.isfinite(prior)) or np.any(prior < 0.0):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_prior_values",
            notes="Discrete hierarchical prior contained negative or non-finite values.",
        )

    prior_total = float(np.sum(prior))
    if prior_total <= 0.0 or not np.isfinite(prior_total):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_prior_normalization",
            notes="Discrete hierarchical prior had zero or non-finite total mass.",
        )

    # Normalize defensively so the function can accept either a raw mass table
    # from the discretization step or an already normalized prior.
    prior = prior / prior_total  # shape: (S, L)
    positive_prior_mask = prior > 0.0  # shape: (S, L)

    if not np.any(positive_prior_mask):
        return MethodEstimate(
            method_name="hierarchical",
            status="no_prior_support",
            notes="Discrete hierarchical prior assigned zero mass to every grid cell.",
        )

    finite_likelihood_mask = np.isfinite(log_likelihood_grid)  # shape: (S, L)
    posterior_support_mask = finite_likelihood_mask & positive_prior_mask  # shape: (S, L)

    if not np.any(posterior_support_mask):
        return MethodEstimate(
            method_name="hierarchical",
            status="no_finite_posterior_support",
            notes=(
                "No grid cell had both positive hierarchical prior mass and a finite "
                "agent log likelihood."
            ),
        )

    log_prior = np.full_like(prior, -np.inf, dtype=float)
    log_prior[positive_prior_mask] = np.log(prior[positive_prior_mask])

    log_posterior = np.full_like(log_likelihood_grid, -np.inf, dtype=float)
    log_posterior[posterior_support_mask] = (
        log_likelihood_grid[posterior_support_mask] + log_prior[posterior_support_mask]
    )

    # As in the JEEDS baseline path, subtract the largest supported log weight
    # before exponentiating so the posterior normalization stays numerically
    # stable even when likelihoods are very sharp.
    stabilized_log_posterior = np.full_like(log_posterior, -np.inf, dtype=float)
    max_log_posterior = float(np.max(log_posterior[posterior_support_mask]))
    stabilized_log_posterior[posterior_support_mask] = (
        log_posterior[posterior_support_mask] - max_log_posterior
    )

    posterior_unnormalized = np.zeros_like(log_posterior, dtype=float)
    posterior_unnormalized[posterior_support_mask] = np.exp(
        stabilized_log_posterior[posterior_support_mask]
    )
    normalization = float(np.sum(posterior_unnormalized))

    if normalization <= 0.0 or not np.isfinite(normalization):
        return MethodEstimate(
            method_name="hierarchical",
            status="invalid_posterior_normalization",
            notes="Failed to normalize the hierarchical posterior after combining prior and likelihood.",
        )

    posterior = posterior_unnormalized / normalization  # shape: (S, L)

    # As with the independent baseline, report sigma on its original scale and
    # decision skill on the canonical log-lambda axis.
    sigma_marginal = np.sum(posterior, axis=1)  # shape: (S,)
    lambda_marginal = np.sum(posterior, axis=0)  # shape: (L,)

    posterior_mean_sigma = float(np.dot(sigma_marginal, sigma_grid))
    posterior_mean_log_lambda = float(np.dot(lambda_marginal, log_lambda_grid))

    map_index = int(np.argmax(posterior))
    sigma_map_index, lambda_map_index = np.unravel_index(map_index, posterior.shape)
    map_sigma = float(sigma_grid[sigma_map_index])
    map_log_lambda = float(log_lambda_grid[lambda_map_index])

    return MethodEstimate(
        method_name="hierarchical",
        posterior_mean_sigma=posterior_mean_sigma,
        posterior_mean_log_lambda=posterior_mean_log_lambda,
        map_sigma=map_sigma,
        map_log_lambda=map_log_lambda,
        status="ok",
        notes=(
            "Hierarchical posterior computed by combining the agent log-likelihood grid "
            "with the supplied discrete population prior."
        ),
    )
