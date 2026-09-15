# Paper correspondence: Supplement `app:baseball_hyperpriors`.
"""Baseball-specific hyperprior presets and calibration helpers.

Paper BBIP convergence (``submit_hjeeds_baseball_convergence_paper_bbip.sh``)
uses a fixed, outcome-independent weak prior via
``--hyperprior-preset baseball-literature-informed``. The committed JSON records
the external execution-error evidence and prior-predictive decision-skill check
used to choose its centers. ``baseball-2021-ff`` is retained only to fail closed
on the superseded, stale JEEDS-calibrated prior. Use ``calibrated`` only for
explicit exploratory runs with a current, user-supplied calibration JSON.

``true_population_from_hyperpriors`` is a Statcast-only shim: there is no
simulated ground truth, but ``ExperimentConfig`` still requires a
``TruePopulationConfig``.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import DEFAULT_HYPERPRIORS
from .models import HyperpriorConfig, MethodEstimate, TruePopulationConfig
from .baseball_provenance import (
    EXECUTION_KERNEL_VERSION,
    PAPER_CALIBRATION_ROSTER_SHA256,
    PITCH_ORDER_VERSION,
    canonical_json_sha256,
    load_processed_artifact_reference,
)

# Wider than darts defaults so empirical Bayes can move on real Statcast data.
LOW_CONFIDENCE_COVARIANCE_DIAGONAL = (1.5**2, 4.0**2)
LOW_CONFIDENCE_LOG_TAU_SD = 1.0
LOW_CONFIDENCE_S_R = 1.25

DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH = (
    Path(__file__).resolve().parent / "data" / "baseball_hyperpriors_2021_ff.json"
)
DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "baseball_hyperpriors_literature_informed.json"
)

HYPERPRIOR_PRESET_CHOICES = (
    "darts",
    "low-confidence",
    "baseball-literature-informed",
    "baseball-2021-ff",
    "calibrated",
)

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


def load_hyperprior_config(
    path: Path,
    *,
    require_current_execution_kernel: bool = False,
    require_paper_2021_ff_calibration: bool = False,
) -> HyperpriorConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if require_current_execution_kernel or require_paper_2021_ff_calibration:
        provenance = payload.get("_provenance") or {}
        reference = load_processed_artifact_reference()
        required_current = {
            "execution_kernel_version": EXECUTION_KERNEL_VERSION,
            "pitch_order_version": PITCH_ORDER_VERSION,
            "processed_pickle_sha256": reference["pickle_sha256"],
            "model_weights_sha256": reference["model_weights_sha256"],
        }
        mismatches = {
            key: (provenance.get(key), expected)
            for key, expected in required_current.items()
            if provenance.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"Hyperprior file {path} has stale or incomplete MLB provenance: {mismatches}. "
                "Regenerate or replace this preset before using it."
            )
    if require_paper_2021_ff_calibration:
        provenance = payload.get("_provenance") or {}
        expected_grid = {
            "delta": 0.0417,
            "requested_num_sigma_grid": 21,
            "actual_num_sigma_hypotheses": 66,
            "sigma_min": 0.17,
            "sigma_max": 2.81,
            "requested_num_lambda_grid": 21,
            "actual_num_log_lambda_hypotheses": 21,
            "lambda_min": 1e-3,
            "lambda_max": 10**3.6,
        }
        required_paper = {
            "season_year": 2021,
            "pitch_types": ["FF"],
            "num_agents": 528,
            "num_completed_agent_results": 528,
            "num_missing_agent_results": 0,
            "num_successful_estimates": 528,
            "confidence": "low",
            "min_pitches_per_agent": 100,
            "max_pitches_per_agent": None,
            "max_agents": None,
            "skill_grid": expected_grid,
            "roster_fingerprint_sha256": PAPER_CALIBRATION_ROSTER_SHA256,
        }
        mismatches = {
            key: (provenance.get(key), expected)
            for key, expected in required_paper.items()
            if provenance.get(key) != expected
        }
        selector = provenance.get("roster_selector") or {}
        if selector.get("all_eligible_agents") is not True:
            mismatches["roster_selector.all_eligible_agents"] = (
                selector.get("all_eligible_agents"),
                True,
            )
        for key in ("pitcher_ids", "top_pitchers", "bbip_extremes"):
            if selector.get(key) not in (None, []):
                mismatches[f"roster_selector.{key}"] = (selector.get(key), None)
        if mismatches:
            raise RuntimeError(
                f"Hyperprior file {path} is not the complete all-eligible 2021 FF paper "
                f"calibration: {mismatches}."
            )
        run_provenance = provenance.get("run_provenance")
        if not isinstance(run_provenance, dict):
            raise RuntimeError(f"Hyperprior file {path} lacks calibration run_provenance.")
        fingerprint_payload = dict(run_provenance)
        recorded_fingerprint = fingerprint_payload.pop("fingerprint_sha256", None)
        if recorded_fingerprint != canonical_json_sha256(fingerprint_payload):
            raise RuntimeError(f"Hyperprior file {path} has an invalid calibration fingerprint.")
        canonical_sigma = np.concatenate(
            (
                np.linspace(0.17, 1.0, num=60, dtype=float),
                np.linspace(1.0 + 0.0417, 2.81, num=6, dtype=float),
            )
        )
        canonical_log_lambda = np.log(
            np.logspace(math.log10(1e-3), math.log10(10**3.6), num=21, dtype=float)
        )
        run_configuration = run_provenance.get("configuration") or {}
        run_mismatches: dict[str, tuple[Any, Any]] = {}
        for key, expected in {
            "execution_kernel_version": EXECUTION_KERNEL_VERSION,
            "pitch_order_version": PITCH_ORDER_VERSION,
            "processed_pickle_sha256": reference["pickle_sha256"],
            "model_weights_sha256": reference["model_weights_sha256"],
            "season_year": 2021,
            "pitch_types": ["FF"],
        }.items():
            if run_provenance.get(key) != expected:
                run_mismatches[key] = (run_provenance.get(key), expected)
        if not np.array_equal(np.asarray(run_provenance.get("sigma_grid")), canonical_sigma):
            run_mismatches["sigma_grid"] = ("noncanonical", "canonical 66-point grid")
        if not np.array_equal(
            np.asarray(run_provenance.get("log_lambda_grid")), canonical_log_lambda
        ):
            run_mismatches["log_lambda_grid"] = ("noncanonical", "canonical 21-point grid")
        for key, expected in {
            "workflow": "baseball-hyperprior-calibration",
            "min_pitches_per_agent": 100,
            "max_pitches_per_agent": None,
            "max_agents": None,
            "confidence": "low",
            "num_agents": 528,
        }.items():
            if run_configuration.get(key) != expected:
                run_mismatches[f"configuration.{key}"] = (
                    run_configuration.get(key),
                    expected,
                )
        if run_configuration.get("roster_fingerprint") != PAPER_CALIBRATION_ROSTER_SHA256:
            run_mismatches["configuration.roster_fingerprint"] = (
                run_configuration.get("roster_fingerprint"),
                PAPER_CALIBRATION_ROSTER_SHA256,
            )
        if run_mismatches:
            raise RuntimeError(
                f"Hyperprior file {path} has noncanonical calibration run provenance: "
                f"{run_mismatches}."
            )
    return hyperprior_config_from_dict(payload)


def load_default_baseball_hyperpriors_2021_ff() -> HyperpriorConfig:
    """Bundled JEEDS-calibrated centers with low-confidence prior widths (2021 FF)."""

    return load_hyperprior_config(
        DEFAULT_BASEBALL_HYPERPRIORS_2021_FF_PATH,
        require_paper_2021_ff_calibration=True,
    )


def load_literature_informed_baseball_hyperpriors() -> HyperpriorConfig:
    """Fixed weak MLB prior selected without using corrected experiment outcomes."""

    config = load_hyperprior_config(
        DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH,
        require_current_execution_kernel=True,
    )
    expected = HyperpriorConfig(
        mean_vector=(math.log(0.5), math.log(100.0)),
        covariance_diagonal=LOW_CONFIDENCE_COVARIANCE_DIAGONAL,
        log_tau_eta_mean=math.log(0.35),
        log_tau_eta_sd=LOW_CONFIDENCE_LOG_TAU_SD,
        log_tau_rho_mean=0.0,
        log_tau_rho_sd=LOW_CONFIDENCE_LOG_TAU_SD,
        m_r=0.0,
        s_r=LOW_CONFIDENCE_S_R,
    )
    if config != expected:
        raise RuntimeError(
            "The committed literature-informed MLB hyperprior no longer matches the "
            "frozen pre-evaluation values. Review any proposed change explicitly."
        )
    with DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH.open("r", encoding="utf-8") as handle:
        provenance = (json.load(handle).get("_provenance") or {})
    required_selection = {
        "status": "fixed-before-corrected-MLB-evaluation",
        "selection_method": (
            "external-execution-evidence-and-prior-predictive-decision-check-v1"
        ),
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in required_selection.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            "The committed literature-informed MLB hyperprior lacks its frozen-selection "
            f"provenance: {mismatches}."
        )
    return config


def write_hyperprior_config(
    path: Path,
    config: HyperpriorConfig,
    *,
    provenance: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = hyperprior_config_to_dict(config)
    reference = load_processed_artifact_reference()
    payload["_provenance"] = {
        **(provenance or {}),
        "execution_kernel_version": EXECUTION_KERNEL_VERSION,
        "pitch_order_version": PITCH_ORDER_VERSION,
        "processed_pickle_sha256": reference["pickle_sha256"],
        "model_weights_sha256": reference["model_weights_sha256"],
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def resolve_baseball_hyperpriors(
    *,
    preset: str,
    calibrated_path: Path | None = None,
) -> HyperpriorConfig:
    if preset == "darts":
        return DEFAULT_HYPERPRIORS
    if preset == "low-confidence":
        return build_low_confidence_hyperpriors()
    if preset == "baseball-literature-informed":
        return load_literature_informed_baseball_hyperpriors()
    if preset == "baseball-2021-ff":
        return load_default_baseball_hyperpriors_2021_ff()
    if preset == "calibrated":
        if calibrated_path is None:
            raise ValueError("--hyperprior-config is required when --hyperprior-preset calibrated.")
        if not calibrated_path.is_file():
            raise FileNotFoundError(f"Hyperprior config not found: {calibrated_path}")
        return load_hyperprior_config(
            calibrated_path,
            require_current_execution_kernel=True,
        )
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
