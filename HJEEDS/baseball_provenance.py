# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_hyperpriors`.
"""Canonical artifact and configuration fingerprints for MLB publication runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ARTIFACT_REFERENCE = (
    REPO_ROOT / "HJEEDS" / "data" / "baseball_processed_artifact_reference.json"
)
EXECUTION_KERNEL_VERSION = "full-difference-grid-min-utility-fft-v3"
PITCH_ORDER_VERSION = "game-date-game-pk-at-bat-pitch-desc-v2"
NUMERICAL_IMPLEMENTATION_VERSION = "batched-rnn-cumulative-prefix-v1"
PAPER_BBIP20_ROSTER_SHA256 = (
    "43b7b59a526a3e9151903f0001c64565d7a1917bbd09debf568d9a8e396542e7"
)
PAPER_BBIP_2021_INNINGS_SHA256 = (
    "9b70036ac41c73e329a09a01890c8b85d694859b1668449e236d761e6fdde313"
)
# Ordered all-eligible 2021 FF roster from the pinned processed Statcast
# artifact, with at least 100 pitches per pitcher (528 agents total).
PAPER_CALIBRATION_ROSTER_SHA256 = (
    "75b9badba2bec2a6da84179da574747277576a7b23d85b41661bf16ecc1d3a00"
)


def load_processed_artifact_reference() -> dict[str, Any]:
    with PROCESSED_ARTIFACT_REFERENCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonical_float_sequence(values: Sequence[float]) -> list[float]:
    """Normalize generated grids across platforms before provenance hashing.

    NumPy/libm can differ by one unit in the last place across cluster nodes
    for the same ``linspace``/``logspace`` request. Fifteen significant digits
    preserve meaningful grid changes while removing that non-scientific
    machine-level variation.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Expected a one-dimensional skill grid; received shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Skill grids must contain only finite values.")
    return [float(format(float(value), ".15g")) for value in array]


def roster_fingerprint(rows: Sequence[tuple[int, int, str]]) -> str:
    """Hash an ordered ``(agent_id, pitcher_id, pitch_type)`` roster."""

    return canonical_json_sha256(
        {
            "agents": [
                {
                    "agent_id": int(agent_id),
                    "pitcher_id": int(pitcher_id),
                    "pitch_type": str(pitch_type),
                }
                for agent_id, pitcher_id, pitch_type in rows
            ]
        }
    )


def build_baseball_run_provenance(
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    sigma_grid: Sequence[float],
    log_lambda_grid: Sequence[float],
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = load_processed_artifact_reference()
    payload: dict[str, Any] = {
        "execution_kernel_version": EXECUTION_KERNEL_VERSION,
        "numerical_implementation_version": NUMERICAL_IMPLEMENTATION_VERSION,
        "processed_pickle_sha256": reference["pickle_sha256"],
        "model_weights_sha256": reference["model_weights_sha256"],
        "season_year": season_year,
        "pitch_types": [str(value) for value in pitch_types],
        "sigma_grid": _canonical_float_sequence(sigma_grid),
        "log_lambda_grid": _canonical_float_sequence(log_lambda_grid),
        "pitch_order_version": PITCH_ORDER_VERSION,
        "configuration": configuration or {},
    }
    payload["fingerprint_sha256"] = canonical_json_sha256(payload)
    return payload


def require_matching_fingerprint(
    actual: dict[str, Any] | None,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    actual_fingerprint = (actual or {}).get("fingerprint_sha256")
    expected_fingerprint = expected["fingerprint_sha256"]
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"{label} provenance fingerprint {actual_fingerprint!r} does not match "
            f"the current publication configuration {expected_fingerprint!r}."
        )
