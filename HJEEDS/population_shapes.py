# This file has been fully verified by a human researcher as of 05/19/26 at 11:34 AM MT.
"""Population-shape definitions for H-JEEDS simulator misspecification studies."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import TruePopulationConfig

DEFAULT_POPULATION_SHAPE_SLUG = "default"
UNIFORM_POPULATION_SHAPE_SLUG = "uniform"
BIMODAL_POPULATION_SHAPE_SLUG = "bimodal"
OUTLIER_HEAVY_POPULATION_SHAPE_SLUG = "outlier_heavy"

BIMODAL_BETWEEN_VARIANCE_FRACTION = 0.45
OUTLIER_HEAVY_OUTLIER_WEIGHT = 0.20
OUTLIER_HEAVY_OUTLIER_SD_MULTIPLIER = 2.0


@dataclass(frozen=True)
class PopulationShapeSpec:
    """Metadata for one true population-shape condition."""

    slug: str
    label: str
    description: str


POPULATION_SHAPE_SPECS = (
    PopulationShapeSpec(
        slug=DEFAULT_POPULATION_SHAPE_SLUG,
        label="Default Gaussian",
        description="Matched Gaussian population used by the baseline H-JEEDS simulator",
    ),
    PopulationShapeSpec(
        slug=UNIFORM_POPULATION_SHAPE_SLUG,
        label="Uniform",
        description="Moment-matched uniform population over a log-skill parallelogram",
    ),
    PopulationShapeSpec(
        slug=BIMODAL_POPULATION_SHAPE_SLUG,
        label="Bimodal",
        description="Moment-matched equal-weight two-cluster Gaussian mixture",
    ),
    PopulationShapeSpec(
        slug=OUTLIER_HEAVY_POPULATION_SHAPE_SLUG,
        label="Outlier-heavy",
        description="Moment-matched 80/20 same-center Gaussian scale mixture with broad outliers",
    ),
)


POPULATION_SHAPE_BY_SLUG = {shape.slug: shape for shape in POPULATION_SHAPE_SPECS}
DEFAULT_POPULATION_SHAPE_SLUGS = tuple(shape.slug for shape in POPULATION_SHAPE_SPECS)


def get_population_shape_spec(shape_slug: str) -> PopulationShapeSpec:
    """Return metadata for one population shape."""

    try:
        return POPULATION_SHAPE_BY_SLUG[shape_slug]
    except KeyError as exc:
        allowed = ", ".join(DEFAULT_POPULATION_SHAPE_SLUGS)
        raise ValueError(f"Unknown population shape '{shape_slug}'. Expected one of: {allowed}.") from exc


def population_shape_folder_slug(shape_slug: str) -> str:
    """Return a stable folder slug for one population shape."""

    _ = get_population_shape_spec(shape_slug) # Validate shape slug
    return f"population_shape_{shape_slug}"


def population_shape_metadata_row(shape_slug: str) -> dict[str, str]:
    """Return CSV metadata for one population shape."""

    shape = get_population_shape_spec(shape_slug)
    return {
        "population_shape_slug": shape.slug,
        "population_shape_label": shape.label,
        "population_shape_description": shape.description,
    }


def _covariance_factor(covariance: np.ndarray) -> np.ndarray:
    """Return a lower-triangular covariance factor."""

    # Convert the input to a floating-point array so linear algebra is well-defined
    covariance = np.asarray(covariance, dtype=float)
    try:
        # Find L such that covariance = L @ L.T
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        # Fail loudly so we can inspect the covariance matrix rather than silently repairing it
        raise ValueError(f"Covariance matrix is not positive definite: {covariance.tolist()}") from exc


def _sample_default_gaussian(
    rng: np.random.Generator,
    mean_vector: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample from the matched Gaussian true population."""

    return rng.multivariate_normal(mean_vector, covariance)


def _sample_uniform(
    rng: np.random.Generator,
    mean_vector: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample from a moment-matched uniform distribution in log-skill space."""

    # Compute L such that covariance = L @ L.T
    factor = _covariance_factor(covariance)
    # Draw u from a square with independent coordinates and unit variance
    # (Setting a = sqrt(3.0) gives rise to such a distribution)
    unit_draw = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=2)
    # Transform the square into a parallelogram with mean_vector as its center and covariance as its covariance
    return mean_vector + factor @ unit_draw


def _sample_bimodal(
    rng: np.random.Generator,
    mean_vector: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample from a moment-matched two-component Gaussian mixture.

    In this function, we take (BIMODAL_BETWEEN_VARIANCE_FRACTION)% of the original
    variance along the widest axis and re-express it as distance between two
    symmetric Gaussian cluster centers.
    """

    # Decompose covariance into eigenvalues and eigenvectors so we can identify the widest direction
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    # Select the eigenvector associated with the largest variance direction
    largest_index = int(np.argmax(eigenvalues))
    # Store the largest variance eigenvector as the axis separating the two modes
    principal_axis = eigenvectors[:, largest_index]
    # Allocate this fraction of the largest variance to between-cluster separation
    between_variance = BIMODAL_BETWEEN_VARIANCE_FRACTION * float(eigenvalues[largest_index])
    # Convert between-cluster variance into the distance from the global mean to each component mean
    offset_magnitude = math.sqrt(between_variance)
    # Build the vector that shifts one mode above the mean and the other below the mean
    offset = offset_magnitude * principal_axis
    # Subtract the between-cluster covariance so within-cluster plus between-cluster covariance equals the target
    component_covariance = covariance - np.outer(offset, offset)
    # Re-symmetrize after floating-point arithmetic so NumPy sees a symmetric covariance matrix
    component_covariance = (component_covariance + component_covariance.T) / 2.0
    # Choose one of the two modes with equal probability, centered at mean_vector plus or minus offset
    component_mean = mean_vector + offset if rng.random() < 0.5 else mean_vector - offset
    # Draw from the selected Gaussian component
    return rng.multivariate_normal(component_mean, component_covariance)


def _sample_outlier_heavy(
    rng: np.random.Generator,
    mean_vector: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    """Sample from a moment-matched same-center Gaussian scale mixture."""

    # Set w, the probability of drawing from the broad outlier component
    outlier_weight = OUTLIER_HEAVY_OUTLIER_WEIGHT
    # Set k, the outlier component's standard-deviation multiplier
    outlier_scale = OUTLIER_HEAVY_OUTLIER_SD_MULTIPLIER

    # Solve (1 - w) * a^2 + w * k^2 = 1 so the mixture covariance still equals covariance
    # (The weighted average variance scaling factor needs to be 1)
    inlier_scale_squared = (1.0 - outlier_weight * outlier_scale**2) / (1.0 - outlier_weight)

    # Fail early if w and k imply an impossible negative inlier variance
    if inlier_scale_squared <= 0.0:
        raise ValueError("Outlier-heavy mixture constants must leave positive inlier variance.")

    # Draw from the outlier component with probability w, otherwise from the tighter inlier component
    scale = outlier_scale if rng.random() < outlier_weight else math.sqrt(inlier_scale_squared)

    # Scale the covariance by scale^2 because covariance scales with squared standard deviation
    return rng.multivariate_normal(mean_vector, covariance * scale**2)


def sample_log_skill_profile(
    rng: np.random.Generator,
    true_population: TruePopulationConfig,
) -> tuple[float, float]:
    """Draw one true log-skill profile from the configured population shape."""

    mean_vector = true_population.mean_vector
    covariance = true_population.covariance_matrix
    shape_slug = true_population.population_shape_slug

    if shape_slug == DEFAULT_POPULATION_SHAPE_SLUG:
        eta, rho = _sample_default_gaussian(rng, mean_vector, covariance)
    elif shape_slug == UNIFORM_POPULATION_SHAPE_SLUG:
        eta, rho = _sample_uniform(rng, mean_vector, covariance)
    elif shape_slug == BIMODAL_POPULATION_SHAPE_SLUG:
        eta, rho = _sample_bimodal(rng, mean_vector, covariance)
    elif shape_slug == OUTLIER_HEAVY_POPULATION_SHAPE_SLUG:
        eta, rho = _sample_outlier_heavy(rng, mean_vector, covariance)
    else:
        get_population_shape_spec(shape_slug)
        raise AssertionError("Population shape validation should have failed earlier")

    return float(eta), float(rho)
