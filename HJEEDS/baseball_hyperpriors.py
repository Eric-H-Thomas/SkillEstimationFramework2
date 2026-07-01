# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Baseball-specific hyperprior presets and calibration helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DEFAULT_HYPERPRIORS
from .models import HyperpriorConfig, MethodEstimate

# Wider than darts defaults so empirical Bayes can move on real Statcast data.
LOW_CONFIDENCE_COVARIANCE_DIAGONAL = (1.5**2, 4.0**2)
LOW_CONFIDENCE_LOG_TAU_SD = 1.0
LOW_CONFIDENCE_S_R = 1.25


def build_low_confidence_hyperpriors(
    *,
    mean_vector: tuple[float, float] | None = None,
) -> HyperpriorConfig:
    """Return darts-aligned centers with deliberately weak baseball confidence."""

    centers = mean_vector or DEFAULT_HYPERPRIORS.mean_vector
    return HyperpriorConfig(
        mean_vector=centers,
        covariance_diagonal=LOW_CONFIDENCE_COVARIANCE_DIAGONAL,
        log_tau_eta_mean=DEFAULT_HYPERPRIORS.log_tau_eta_mean,
        log_tau_eta_sd=LOW_CONFIDENCE_LOG_TAU_SD,
        log_tau_rho_mean=DEFAULT_HYPERPRIORS.log_tau_rho_mean,
        log_tau_rho_sd=LOW_CONFIDENCE_LOG_TAU_SD,
        m_r=DEFAULT_HYPERPRIORS.m_r,
        s_r=LOW_CONFIDENCE_S_R,
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

    tau_eta = max(float(np.std(log_sigma_array, ddof=1)), 0.05)
    tau_rho = max(float(np.std(log_lambda_array, ddof=1)), 0.05)
    correlation_matrix = np.corrcoef(log_sigma_array, log_lambda_array)
    correlation = float(correlation_matrix[0, 1])
    if not math.isfinite(correlation):
        correlation = math.tanh(float(DEFAULT_HYPERPRIORS.m_r))

    correlation = float(np.clip(correlation, -0.95, 0.95))

    if confidence == "low":
        return HyperpriorConfig(
            mean_vector=mean_vector,
            covariance_diagonal=LOW_CONFIDENCE_COVARIANCE_DIAGONAL,
            log_tau_eta_mean=math.log(tau_eta),
            log_tau_eta_sd=LOW_CONFIDENCE_LOG_TAU_SD,
            log_tau_rho_mean=math.log(tau_rho),
            log_tau_rho_sd=LOW_CONFIDENCE_LOG_TAU_SD,
            m_r=math.atanh(correlation),
            s_r=LOW_CONFIDENCE_S_R,
        )

    if confidence == "darts":
        return HyperpriorConfig(
            mean_vector=mean_vector,
            covariance_diagonal=DEFAULT_HYPERPRIORS.covariance_diagonal,
            log_tau_eta_mean=math.log(tau_eta),
            log_tau_eta_sd=DEFAULT_HYPERPRIORS.log_tau_eta_sd,
            log_tau_rho_mean=math.log(tau_rho),
            log_tau_rho_sd=DEFAULT_HYPERPRIORS.log_tau_rho_sd,
            m_r=math.atanh(correlation),
            s_r=DEFAULT_HYPERPRIORS.s_r,
        )

    raise ValueError(f"Unknown confidence preset '{confidence}'. Expected 'low' or 'darts'.")


def hyperprior_config_to_dict(config: HyperpriorConfig) -> dict[str, Any]:
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
    if preset == "calibrated":
        if calibrated_path is None:
            raise ValueError("--hyperprior-config is required when --hyperprior-preset calibrated.")
        if not calibrated_path.is_file():
            raise FileNotFoundError(f"Hyperprior config not found: {calibrated_path}")
        return load_hyperprior_config(calibrated_path)
    raise ValueError(
        f"Unknown hyperprior preset '{preset}'. Expected darts, low-confidence, or calibrated."
    )
