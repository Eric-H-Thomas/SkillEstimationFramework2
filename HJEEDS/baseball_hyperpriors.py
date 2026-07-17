# This file has been fully reviewed by a human researcher as of 07/17/26 at 12:30 PM MDT.
"""Baseball-specific hyperprior presets and calibration helpers.

Paper BBIP convergence (``submit_hjeeds_baseball_convergence_paper_bbip.sh``)
loads JEEDS-calibrated centers with low-confidence prior widths via
``--hyperprior-preset baseball-2021-ff``, which reads the committed file
``HJEEDS/data/baseball_hyperpriors_2021_ff.json``. Use ``calibrated`` only when
pointing ``--hyperprior-config`` at a freshly aggregated calibration JSON
under ``HJEEDS/results/`` (gitignored).

``true_population_from_hyperpriors`` is a Statcast-only shim: there is no
simulated ground truth, but ``ExperimentConfig`` still requires a
``TruePopulationConfig``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DEFAULT_HYPERPRIORS
from .models import HyperpriorConfig, MethodEstimate, TruePopulationConfig

# Wider than darts defaults so empirical Bayes can move on real Statcast data.
LOW_CONFIDENCE_COVARIANCE_DIAGONAL = (1.5**2, 4.0**2)
LOW_CONFIDENCE_LOG_TAU_SD = 1.0
LOW_CONFIDENCE_S_R = 1.25

DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH = (
    Path(__file__).resolve().parent / "data" / "baseball_hyperpriors_2021_ff.json"
)

HYPERPRIOR_PRESET_CHOICES = ("darts", "low-confidence", "baseball-2021-ff", "calibrated")

# Confidence presets only change prior widths / correlation concentration.
# Centers and log-tau means are always supplied by the caller.
# Values: (covariance_diagonal, log_tau_eta_sd, log_tau_rho_sd, s_r)
_CONFIDENCE_WIDTHS: dict[str, tuple[tuple[float, float], float, float, float]] = {
    "low": (
        LOW_CONFIDENCE_COVARIANCE_DIAGONAL,
        LOW_CONFIDENCE_LOG_TAU_SD,
        LOW_CONFIDENCE_LOG_TAU_SD,
        LOW_CONFIDENCE_S_R,
    ),
    "darts": (
        DEFAULT_HYPERPRIORS.covariance_diagonal,
        DEFAULT_HYPERPRIORS.log_tau_eta_sd,
        DEFAULT_HYPERPRIORS.log_tau_rho_sd,
        DEFAULT_HYPERPRIORS.s_r,
    ),
}


def _apply_confidence_widths(
    *,
    mean_vector: tuple[float, float],
    log_tau_eta_mean: float,
    log_tau_rho_mean: float,
    m_r: float,
    confidence: str,
) -> HyperpriorConfig:
    """Attach prior widths for a confidence preset; centers stay caller-chosen."""

    try:
        covariance_diagonal, log_tau_eta_sd, log_tau_rho_sd, s_r = _CONFIDENCE_WIDTHS[confidence]
    except KeyError as exc:
        raise ValueError(
            f"Unknown confidence preset '{confidence}'. "
            f"Expected one of {tuple(_CONFIDENCE_WIDTHS)}."
        ) from exc

    return HyperpriorConfig(
        mean_vector=mean_vector,
        covariance_diagonal=covariance_diagonal,
        log_tau_eta_mean=log_tau_eta_mean,
        log_tau_eta_sd=log_tau_eta_sd,
        log_tau_rho_mean=log_tau_rho_mean,
        log_tau_rho_sd=log_tau_rho_sd,
        m_r=m_r,
        s_r=s_r,
    )


def build_low_confidence_hyperpriors(
    *,
    mean_vector: tuple[float, float] | None = None,
) -> HyperpriorConfig:
    """Return darts-aligned centers with deliberately weak baseball confidence."""

    return _apply_confidence_widths(
        mean_vector=mean_vector or DEFAULT_HYPERPRIORS.mean_vector,
        log_tau_eta_mean=DEFAULT_HYPERPRIORS.log_tau_eta_mean,
        log_tau_rho_mean=DEFAULT_HYPERPRIORS.log_tau_rho_mean,
        m_r=DEFAULT_HYPERPRIORS.m_r,
        confidence="low",
    )


def build_hyperpriors_from_jeeds_estimates(
    estimates: Sequence[MethodEstimate],
    *,
    confidence: str = "low",
) -> HyperpriorConfig:
    """Build hyperpriors from independent JEEDS posterior means across agents."""

    log_sigmas: list[float] = []
    log_lambdas: list[float] = []
    for estimate in estimates:
        if estimate.status != "ok":
            continue
        if estimate.posterior_mean_sigma is None or estimate.posterior_mean_log_lambda is None:
            continue
        if estimate.posterior_mean_sigma <= 0:
            continue
        log_sigmas.append(math.log(float(estimate.posterior_mean_sigma)))
        log_lambdas.append(float(estimate.posterior_mean_log_lambda))

    if len(log_sigmas) < 2:
        raise ValueError(
            "Need at least two successful independent JEEDS estimates to calibrate hyperpriors. "
            f"Received {len(log_sigmas)}."
        )

    log_sigma_array = np.asarray(log_sigmas, dtype=float)
    log_lambda_array = np.asarray(log_lambdas, dtype=float)
    mean_vector = (float(np.mean(log_sigma_array)), float(np.mean(log_lambda_array)))

    # Floor sample SDs so log(tau) stays defined and EB can still move a little.
    tau_eta = max(float(np.std(log_sigma_array, ddof=1)), 0.05)
    tau_rho = max(float(np.std(log_lambda_array, ddof=1)), 0.05)
    correlation = float(np.corrcoef(log_sigma_array, log_lambda_array)[0, 1])
    if not math.isfinite(correlation):
        correlation = math.tanh(float(DEFAULT_HYPERPRIORS.m_r))
    correlation = float(np.clip(correlation, -0.95, 0.95))

    return _apply_confidence_widths(
        mean_vector=mean_vector,
        log_tau_eta_mean=math.log(tau_eta),
        log_tau_rho_mean=math.log(tau_rho),
        m_r=math.atanh(correlation),
        confidence=confidence,
    )


def hyperprior_config_to_dict(config: HyperpriorConfig) -> dict[str, Any]:
    """Serialize for JSON; tuples become lists to match on-disk artifacts."""

    return {
        "mean_vector": list(config.mean_vector),
        "covariance_diagonal": list(config.covariance_diagonal),
        "log_tau_eta_mean": config.log_tau_eta_mean,
        "log_tau_eta_sd": config.log_tau_eta_sd,
        "log_tau_rho_mean": config.log_tau_rho_mean,
        "log_tau_rho_sd": config.log_tau_rho_sd,
        "m_r": config.m_r,
        "s_r": config.s_r,
    }


def hyperprior_config_from_dict(payload: dict[str, Any]) -> HyperpriorConfig:
    return HyperpriorConfig(
        mean_vector=tuple(float(value) for value in payload["mean_vector"]),
        covariance_diagonal=tuple(float(value) for value in payload["covariance_diagonal"]),
        log_tau_eta_mean=float(payload["log_tau_eta_mean"]),
        log_tau_eta_sd=float(payload["log_tau_eta_sd"]),
        log_tau_rho_mean=float(payload["log_tau_rho_mean"]),
        log_tau_rho_sd=float(payload["log_tau_rho_sd"]),
        m_r=float(payload["m_r"]),
        s_r=float(payload["s_r"]),
    )


def load_hyperprior_config(path: Path) -> HyperpriorConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return hyperprior_config_from_dict(payload)


def load_default_baseball_hyperpriors_2021_ff() -> HyperpriorConfig:
    """Bundled JEEDS-calibrated centers with low-confidence prior widths (2021 FF)."""

    return load_hyperprior_config(DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH)


DEFAULT_BASEBALL_HYPERPRIORS_2021_FF = load_default_baseball_hyperpriors_2021_ff()


def write_hyperprior_config(path: Path, config: HyperpriorConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(hyperprior_config_to_dict(config), handle, indent=2)
        handle.write("\n")


def resolve_baseball_hyperpriors(
    *,
    preset: str,
    calibrated_path: Path | None = None,
) -> HyperpriorConfig:
    if preset == "darts":
        return DEFAULT_HYPERPRIORS
    if preset == "low-confidence":
        return build_low_confidence_hyperpriors()
    if preset == "baseball-2021-ff":
        return DEFAULT_BASEBALL_HYPERPRIORS_2021_FF
    if preset == "calibrated":
        if calibrated_path is None:
            raise ValueError("--hyperprior-config is required when --hyperprior-preset calibrated.")
        if not calibrated_path.is_file():
            raise FileNotFoundError(f"Hyperprior config not found: {calibrated_path}")
        return load_hyperprior_config(calibrated_path)
    raise ValueError(
        f"Unknown hyperprior preset '{preset}'. Expected one of {HYPERPRIOR_PRESET_CHOICES}."
    )


def true_population_from_hyperpriors(hyperpriors: HyperpriorConfig) -> TruePopulationConfig:
    """Shim TruePopulationConfig from hyperprior centers for ExperimentConfig.

    Statcast runs have no simulated ground truth; this only satisfies the shared
    ``ExperimentConfig`` / hierarchical estimation API.
    """

    return TruePopulationConfig(
        mean_log_sigma=hyperpriors.mean_vector[0],
        mean_log_lambda=hyperpriors.mean_vector[1],
        tau_eta=math.exp(hyperpriors.log_tau_eta_mean),
        tau_rho=math.exp(hyperpriors.log_tau_rho_mean),
        correlation=math.tanh(hyperpriors.m_r),
    )
