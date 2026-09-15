# Paper correspondence: Main `subsec:baseball`; Supplement `app:baseball_hyperpriors`.
"""Baseball pitch surfaces and strike-zone grids for HJEEDS.

Wraps the classic baseball reward path (RNN outcomes + ``getUtility`` +
``convolve2d`` EV surfaces) without the full ``BaseballExp`` experiment loop.

Plate grids come from the processed Statcast pickle (built by
``SpacesBaseball.getAllData``); ``delta`` must match that pickle's spacing.
"""

from __future__ import annotations

import math
import hashlib
import json
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from scipy.stats import multivariate_normal

from .baseball_provenance import PROCESSED_ARTIFACT_REFERENCE
from .config import paper_config_required

REPO_ROOT = Path(__file__).resolve().parent.parent
STATCAST_DIR = REPO_ROOT / "Data" / "Baseball" / "StatcastData"
PROCESSED_PICKLE = STATCAST_DIR / "ProcessedData-From-GivenFiles.pkl"
MODEL_WEIGHTS = REPO_ROOT / "Environments" / "Baseball" / "final_OP"
# Matches SpacesBaseball / estimators.py baseball-multi defaults.
DEFAULT_DELTA = 0.0417
DEFAULT_EXECUTION_SKILL_MIN = 0.17
DEFAULT_EXECUTION_SKILL_MAX = 2.81
DEFAULT_EXECUTION_RHO = 0.0
DEFAULT_RNN_INFERENCE_BATCH_SIZE = 4096
_DELTA_ABS_TOL = 1e-6


@dataclass(frozen=True)
class StrikeZoneGrids:
    """Target grids shared by every pitch in an experiment."""

    delta: float
    targets_plate_x_feet: np.ndarray
    targets_plate_z_feet: np.ndarray
    model_targets_plate_x: np.ndarray
    model_targets_plate_z: np.ndarray
    possible_targets_feet: np.ndarray
    possible_targets_for_model: np.ndarray


@dataclass(frozen=True)
class PitchObservation:
    """One pitch with precomputed EV surfaces per execution-skill key."""

    executed_action: tuple[float, float]
    observed_reward: float
    evs_per_execution_skill: dict[str, np.ndarray]
    min_utility: float


@dataclass(frozen=True)
class BaseballRuntime:
    """Loaded model, geometry, and execution-noise PDFs."""

    grids: StrikeZoneGrids
    model: Any
    pdfs_per_execution_skill: dict[str, np.ndarray]
    all_covs: dict[str, np.ndarray]


@dataclass(frozen=True)
class StatcastAgentSpec:
    """One HJEEDS agent: pitcher ID + pitch type."""

    agent_id: int
    pitcher_id: int
    pitch_type: str


def baseball_execution_skill_key(execution_skill: float, rho: float = DEFAULT_EXECUTION_RHO) -> str:
    """Match ``SpacesBaseball.get_key([skill, skill], rho)``."""

    return f"{execution_skill}|{execution_skill}|{rho}"


def build_execution_skill_grid(
    delta: float,
    *,
    skill_min: float = DEFAULT_EXECUTION_SKILL_MIN,
    skill_max: float = DEFAULT_EXECUTION_SKILL_MAX,
    num_dense: int = 60,
    num_tail: int = 6,
) -> np.ndarray:
    """Classic ``baseball-multi`` symmetric σ grid (60 dense knots to 1.0 + 6 tail)."""

    dense = np.linspace(skill_min, 1.0, num=num_dense, dtype=float)
    tail = np.linspace(1.0 + delta, skill_max, num=num_tail, dtype=float)
    return np.concatenate((dense, tail))


def build_log_lambda_grid(
    *,
    lambda_min: float = 1e-3,
    lambda_max: float = 10**3.6,
    num_lambda_grid: int = 21,
) -> np.ndarray:
    """Log-λ grid spanning ``[lambda_min, lambda_max]`` (default ends match ``JointMethodQRE``)."""

    if lambda_min <= 0.0 or lambda_max <= 0.0:
        raise ValueError("lambda_min and lambda_max must be positive.")
    if lambda_min >= lambda_max:
        raise ValueError("lambda_min must be strictly less than lambda_max.")
    if num_lambda_grid < 1:
        raise ValueError("num_lambda_grid must be at least 1.")

    raw = np.logspace(
        math.log10(lambda_min),
        math.log10(lambda_max),
        num=num_lambda_grid,
        dtype=float,
    )
    return np.log(raw)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_processed_artifact_reference() -> dict:
    if not PROCESSED_ARTIFACT_REFERENCE.is_file():
        raise FileNotFoundError(
            f"Missing canonical baseball artifact metadata: {PROCESSED_ARTIFACT_REFERENCE}"
        )
    with PROCESSED_ARTIFACT_REFERENCE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_processed_payload(payload: tuple, reference: dict) -> None:
    if len(payload) != 6:
        raise ValueError(f"Processed Statcast payload must contain six items; received {len(payload)}.")
    all_data, batter_indices = payload[:2]
    expected_shape = tuple(int(value) for value in reference["dataframe_shape"])
    if tuple(all_data.shape) != expected_shape:
        raise ValueError(
            f"Processed Statcast dataframe shape {all_data.shape} does not match {expected_shape}."
        )
    expected_years = tuple(int(value) for value in reference["season_years"])
    actual_years = tuple(sorted(int(value) for value in all_data["game_year"].dropna().unique()))
    if actual_years != expected_years:
        raise ValueError(f"Processed Statcast years {actual_years} do not match {expected_years}.")
    required_columns = tuple(reference["required_columns"])
    missing_columns = [column for column in required_columns if column not in all_data.columns]
    if missing_columns:
        raise ValueError(f"Processed Statcast data is missing columns: {missing_columns}.")
    if tuple(batter_indices.shape) != tuple(reference["batter_mapping_shape"]):
        raise ValueError(
            f"Batter-index mapping shape {batter_indices.shape} does not match "
            f"{tuple(reference['batter_mapping_shape'])}."
        )
    expected_indices = np.arange(int(reference["batter_embedding_count"]), dtype=int)
    actual_indices = np.sort(batter_indices["batter_index"].to_numpy(dtype=int))
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("Batter indices are not the complete range required by final_OP.")
    expected_grid_shape = tuple(int(value) for value in reference["target_grid_shape"])
    target_axes_shape = (len(payload[2]), len(payload[3]))
    if target_axes_shape != expected_grid_shape:
        raise ValueError(
            f"Target grid shape {target_axes_shape} does not match {expected_grid_shape}."
        )
    if tuple(np.asarray(payload[4]).shape) != (math.prod(expected_grid_shape), 2):
        raise ValueError("Physical target pairs do not match the canonical target grid.")
    if tuple(np.asarray(payload[5]).shape) != (math.prod(expected_grid_shape), 2):
        raise ValueError("Model target pairs do not match the canonical target grid.")


@lru_cache(maxsize=1)
def _load_processed_pickle_raw() -> tuple:
    """Unpickle the Statcast artifact without publication hash checks."""

    if not PROCESSED_PICKLE.is_file():
        raise FileNotFoundError(
            f"Processed Statcast pickle not found: {PROCESSED_PICKLE}. "
            "Place ProcessedData-From-GivenFiles.pkl under Data/Baseball/StatcastData/. "
            "The paper workflow additionally requires the canonical hash-pinned artifact "
            "documented in Data/Baseball/StatcastData/README.md."
        )
    with PROCESSED_PICKLE.open("rb") as handle:
        loaded = pickle.load(handle)
    try:
        payload = tuple(loaded[0])
    except (IndexError, TypeError) as exc:
        raise ValueError(f"Malformed processed Statcast pickle: {PROCESSED_PICKLE}") from exc
    if len(payload) != 6:
        raise ValueError(
            f"Processed Statcast payload must contain six items; received {len(payload)}."
        )
    return payload


def _validate_canonical_processed_artifact(payload: tuple) -> None:
    """Reject pickle/model files that are not the frozen paper artifact."""

    reference = _load_processed_artifact_reference()
    expected_bytes = int(reference["pickle_bytes"])
    if PROCESSED_PICKLE.stat().st_size != expected_bytes:
        raise ValueError(
            f"Processed Statcast pickle has {PROCESSED_PICKLE.stat().st_size} bytes; "
            f"expected {expected_bytes}."
        )
    actual_hash = _sha256(PROCESSED_PICKLE)
    if actual_hash != reference["pickle_sha256"]:
        raise ValueError(
            f"Processed Statcast pickle hash {actual_hash} does not match the canonical paper "
            f"artifact {reference['pickle_sha256']}."
        )
    if _sha256(MODEL_WEIGHTS) != reference["model_weights_sha256"]:
        raise ValueError("Tracked final_OP weights do not match the canonical artifact metadata.")
    _validate_processed_payload(payload, reference)


def _load_processed_pickle(*, validate_artifact: bool | None = None) -> tuple:
    """Load the processed Statcast pickle, optionally checking the paper hashes."""

    payload = _load_processed_pickle_raw()
    if paper_config_required(validate_artifact):
        _validate_canonical_processed_artifact(payload)
    return payload


def load_processed_statcast(*, validate_artifact: bool | None = None) -> pd.DataFrame:
    """Load the merged Statcast dataframe from the processed pickle."""

    return _load_processed_pickle(validate_artifact=validate_artifact)[0]


def filter_statcast_by_season(all_data: pd.DataFrame, season_year: int | None) -> pd.DataFrame:
    """Return rows for one ``game_year``, or the full dataframe when ``season_year`` is None."""

    if season_year is None:
        return all_data
    if "game_year" not in all_data.columns:
        raise ValueError("Processed Statcast data is missing a game_year column.")
    filtered = all_data.loc[all_data["game_year"] == int(season_year)]
    if filtered.empty:
        raise ValueError(f"No Statcast rows found for season_year={season_year}.")
    return filtered


def _infer_grid_delta(targets: np.ndarray) -> float:
    diffs = np.diff(np.asarray(targets, dtype=float))
    positive = diffs[diffs > 0.0]
    if positive.size == 0:
        raise ValueError("Cannot infer grid delta: target axis has no positive spacing.")
    return float(np.median(positive))


def build_strike_zone_grids(
    delta: float = DEFAULT_DELTA,
    *,
    validate_artifact: bool | None = None,
) -> tuple[StrikeZoneGrids, np.ndarray]:
    """Load strike-zone grids from the processed Statcast pickle; require matching ``delta``."""

    (
        _all_data,
        batter_indices,
        model_targets_plate_x,
        model_targets_plate_z,
        possible_targets_feet,
        possible_targets_for_model,
    ) = _load_processed_pickle(validate_artifact=validate_artifact)
    possible_targets_feet = np.asarray(possible_targets_feet, dtype=float)
    possible_targets_for_model = np.asarray(possible_targets_for_model, dtype=float)
    targets_plate_x_feet = np.unique(possible_targets_feet[:, 0])
    targets_plate_z_feet = np.unique(possible_targets_feet[:, 1])

    inferred_x = _infer_grid_delta(targets_plate_x_feet)
    inferred_z = _infer_grid_delta(targets_plate_z_feet)
    if not math.isclose(inferred_x, inferred_z, rel_tol=0.0, abs_tol=_DELTA_ABS_TOL):
        raise ValueError(
            f"Pickle plate axes disagree on spacing: dx={inferred_x}, dz={inferred_z}."
        )
    if not math.isclose(float(delta), inferred_x, rel_tol=0.0, abs_tol=_DELTA_ABS_TOL):
        raise ValueError(
            f"delta={delta} does not match pickle grid spacing {inferred_x}. "
            "The canonical publication artifact fixes this resolution; pass its matching delta."
        )

    grids = StrikeZoneGrids(
        delta=float(delta),
        targets_plate_x_feet=targets_plate_x_feet,
        targets_plate_z_feet=targets_plate_z_feet,
        model_targets_plate_x=np.asarray(model_targets_plate_x, dtype=float),
        model_targets_plate_z=np.asarray(model_targets_plate_z, dtype=float),
        possible_targets_feet=possible_targets_feet,
        possible_targets_for_model=possible_targets_for_model,
    )
    return grids, np.asarray(batter_indices)


def _load_model(batter_indices: np.ndarray):
    import torch
    from Environments.Baseball import modelTake2

    # Embedding size uses ``.shape[0]`` (num batters), matching SpacesBaseball.
    modelTake2.batter_indices = np.asarray(batter_indices)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = modelTake2.RNN(hidden_size=32, output_size=9).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    return model


def _predict_single_pitch_contexts(
    model,
    batch_x: np.ndarray,
    batch_y: np.ndarray,
    *,
    inference_batch_size: int | None,
):
    """Evaluate independent one-pitch contexts, optionally in vectorized batches.

    ``modelTake2.prediction_func`` loops over every candidate target even though
    this publication path presents each candidate as an independent sequence of
    length one. A single batched forward pass has the same zero hidden state and
    target for every row while avoiding tens of thousands of Python-level model
    calls per observed pitch. Passing ``None`` retains the legacy implementation
    for equivalence testing.
    """

    import torch
    from Environments.Baseball import modelTake2

    inputs = np.asarray(batch_x)
    targets = np.asarray(batch_y)
    if inputs.ndim != 3 or inputs.shape[1] != 1:
        raise ValueError(
            "Batched baseball inference requires inputs shaped "
            f"(num_contexts, 1, num_features); received {inputs.shape}."
        )
    if targets.shape != inputs.shape[:2]:
        raise ValueError(
            f"Target shape {targets.shape} does not match input prefixes {inputs.shape[:2]}."
        )

    input_tensor = torch.as_tensor(inputs, dtype=torch.float)
    target_tensor = torch.as_tensor(targets, dtype=torch.long)
    if inference_batch_size is None:
        return modelTake2.prediction_func(model, input_tensor, target_tensor)
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be positive or None for legacy inference.")

    device = next(model.parameters()).device
    flat_inputs = input_tensor[:, 0, :]
    flat_targets = target_tensor[:, 0]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(flat_inputs), inference_batch_size):
            stop = min(start + inference_batch_size, len(flat_inputs))
            x_chunk = flat_inputs[start:stop].to(device)
            y_chunk = flat_targets[start:stop].to(device)
            hidden = torch.zeros(
                (len(x_chunk), int(model.hidden_size)),
                dtype=x_chunk.dtype,
                device=device,
            )
            logits, _hidden = model(x_chunk, hidden, y_chunk)
            outputs.append(logits)
    return torch.cat(outputs, dim=0)


def _utility_grid_from_row(
    pitch_row: pd.Series,
    grids: StrikeZoneGrids,
    model,
    *,
    inference_batch_size: int | None = DEFAULT_RNN_INFERENCE_BATCH_SIZE,
) -> tuple[np.ndarray, float, float]:
    """Return (utility board Zs, min utility, observed utility for the actual pitch).

    Device placement follows ``modelTake2.prediction_func`` (module-global device),
    matching the classic SpacesBaseball path.
    """

    from Environments.Baseball import modelTake2, utilsBaseball
    import torch.nn as nn

    possible_targets_len = len(grids.possible_targets_for_model)
    all_temp_data = pd.concat([pitch_row.to_frame().T] * possible_targets_len, ignore_index=True)
    all_temp_data["plate_x"] = np.copy(grids.possible_targets_for_model[:, 0])
    all_temp_data["plate_z"] = np.copy(grids.possible_targets_for_model[:, 1])
    all_temp_data.loc[len(all_temp_data.index)] = pitch_row

    feature_frame = all_temp_data[modelTake2.features].apply(pd.to_numeric, errors="coerce")
    batch_x = feature_frame.to_numpy(dtype=np.float64)
    batch_y = pd.to_numeric(all_temp_data["outcome"], errors="coerce").to_numpy(dtype=np.int64)
    batch_x = batch_x.reshape((len(batch_x), 1, len(modelTake2.features)))
    batch_y = batch_y.reshape((len(batch_y), 1))

    ypred = _predict_single_pitch_contexts(
        model,
        batch_x,
        batch_y,
        inference_batch_size=inference_batch_size,
    )
    probabilities = nn.functional.softmax(ypred, dim=1).detach().cpu().numpy()
    for outcome_index in range(9):
        all_temp_data[f"o{outcome_index}"] = probabilities[:, outcome_index]

    with_utilities = utilsBaseball.getUtility(all_temp_data)
    actual_row = with_utilities.iloc[-1]
    utility_board = with_utilities.iloc[:-1]
    min_utility = float(np.min(utility_board["utility"].values))
    zs = utility_board["utility"].values.reshape(
        (len(grids.targets_plate_x_feet), len(grids.targets_plate_z_feet)),
    )
    return zs, min_utility, float(actual_row["utility"])


def _execution_covariance(skill: float):
    skill = float(skill)
    variance = skill**2
    covariance = DEFAULT_EXECUTION_RHO * variance
    return np.asarray([[variance, covariance], [covariance, variance]], dtype=float)


def build_execution_displacement_distribution(
    covariance: np.ndarray,
    resolution: float,
    x_offsets: np.ndarray,
    z_offsets: np.ndarray,
) -> np.ndarray:
    """Discretize zero-centered Gaussian mass over all valid pairwise offsets."""

    x_offsets = np.asarray(x_offsets, dtype=float)
    z_offsets = np.asarray(z_offsets, dtype=float)
    if x_offsets.ndim != 1 or z_offsets.ndim != 1:
        raise ValueError("Displacement axes must be one-dimensional.")
    if resolution <= 0.0:
        raise ValueError("resolution must be positive.")
    x_grid, z_grid = np.meshgrid(x_offsets, z_offsets, indexing="ij")
    points = np.column_stack((x_grid.ravel(), z_grid.ravel()))
    density = multivariate_normal(mean=(0.0, 0.0), cov=np.asarray(covariance, dtype=float)).pdf(
        points
    )
    mass = np.asarray(density, dtype=float).reshape((len(x_offsets), len(z_offsets)))
    mass *= float(resolution) ** 2
    if np.any(~np.isfinite(mass)) or np.any(mass < 0.0):
        raise ValueError("Execution displacement kernel contains invalid probability mass.")
    total_mass = float(np.sum(mass))
    if total_mass > 1.0 + 1e-6:
        raise ValueError(f"Execution displacement mass sums to {total_mass:.9f}, above one.")
    return mass


def expected_utility_with_outside_floor(
    utility_grid: np.ndarray,
    displacement_mass: np.ndarray,
    min_utility: float,
) -> np.ndarray:
    """Convolve valid transitions while assigning all other mass ``min_utility``."""

    utility_grid = np.asarray(utility_grid, dtype=float)
    displacement_mass = np.asarray(displacement_mass, dtype=float)
    expected_kernel_shape = (
        2 * utility_grid.shape[0] - 1,
        2 * utility_grid.shape[1] - 1,
    )
    if displacement_mass.shape != expected_kernel_shape:
        raise ValueError(
            f"Execution kernel shape {displacement_mass.shape} does not match "
            f"{expected_kernel_shape}."
        )
    return np.asarray(
        min_utility
        + fftconvolve(
            utility_grid - min_utility,
            displacement_mass,
            mode="same",
        ),
        dtype=float,
    )


def build_baseball_runtime(
    rng: np.random.Generator,
    execution_skills: Sequence[float],
    *,
    delta: float = DEFAULT_DELTA,
    validate_artifact: bool | None = None,
) -> BaseballRuntime:
    """Load model/geometry and precompute execution-noise PDFs."""

    grids, batter_indices = build_strike_zone_grids(delta, validate_artifact=validate_artifact)
    model = _load_model(batter_indices)

    pdfs_per_execution_skill: dict[str, np.ndarray] = {}
    all_covs: dict[str, np.ndarray] = {}
    for execution_skill in execution_skills:
        skill = float(execution_skill)
        key = baseball_execution_skill_key(skill)
        cov = _execution_covariance(skill)
        all_covs[key] = cov
        x_offsets = np.arange(
            -(len(grids.targets_plate_x_feet) - 1),
            len(grids.targets_plate_x_feet),
            dtype=float,
        ) * grids.delta
        z_offsets = np.arange(
            -(len(grids.targets_plate_z_feet) - 1),
            len(grids.targets_plate_z_feet),
            dtype=float,
        ) * grids.delta
        pdfs_per_execution_skill[key] = build_execution_displacement_distribution(
            cov,
            grids.delta,
            x_offsets,
            z_offsets,
        )

    return BaseballRuntime(
        grids=grids,
        model=model,
        pdfs_per_execution_skill=pdfs_per_execution_skill,
        all_covs=all_covs,
    )


def build_pitch_observation(
    pitch_row: pd.Series,
    runtime: BaseballRuntime,
    execution_skills: Sequence[float],
    executed_action: tuple[float, float] | None = None,
    *,
    inference_batch_size: int | None = DEFAULT_RNN_INFERENCE_BATCH_SIZE,
) -> PitchObservation:
    """Build one pitch observation with EV surfaces for all execution-skill keys."""

    zs, min_utility, observed_reward = _utility_grid_from_row(
        pitch_row,
        runtime.grids,
        runtime.model,
        inference_batch_size=inference_batch_size,
    )
    evs_per_execution_skill: dict[str, np.ndarray] = {}
    for skill in execution_skills:
        key = baseball_execution_skill_key(float(skill))
        evs_per_execution_skill[key] = expected_utility_with_outside_floor(
            zs,
            runtime.pdfs_per_execution_skill[key],
            min_utility,
        )

    if executed_action is None:
        executed_action = (float(pitch_row["plate_x_feet"]), float(pitch_row["plate_z_feet"]))

    return PitchObservation(
        executed_action=(float(executed_action[0]), float(executed_action[1])),
        observed_reward=observed_reward,
        evs_per_execution_skill=evs_per_execution_skill,
        min_utility=min_utility,
    )


def build_pitch_observations_for_rows(
    agent_rows: pd.DataFrame,
    runtime: BaseballRuntime,
    execution_skills: Sequence[float],
    *,
    inference_batch_size: int | None = DEFAULT_RNN_INFERENCE_BATCH_SIZE,
) -> list[PitchObservation]:
    """Build pitch observations for every row in a Statcast agent slice."""

    return [
        build_pitch_observation(
            row,
            runtime,
            execution_skills,
            inference_batch_size=inference_batch_size,
        )
        for _, row in agent_rows.iterrows()
    ]


def _agent_pitch_subset(
    all_data: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
) -> pd.DataFrame:
    return all_data[(all_data["pitcher"] == pitcher_id) & (all_data["pitch_type"] == pitch_type)]


def get_agent_pitch_rows(
    all_data: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Return newest-first pitches for one (pitcher, pitch type) agent."""

    subset = _agent_pitch_subset(all_data, pitcher_id, pitch_type)
    ordering_columns = [
        column
        for column in ("game_date", "game_pk", "at_bat_number", "pitch_number")
        if column in subset.columns
    ]
    if "game_date" not in ordering_columns:
        raise ValueError("Processed Statcast data is missing game_date.")
    agent_data = subset.sort_values(
        by=ordering_columns,
        ascending=[False] * len(ordering_columns),
        kind="mergesort",
    )
    if max_rows is not None and len(agent_data) > max_rows:
        agent_data = agent_data.iloc[:max_rows, :]
    return agent_data


def count_agent_pitch_rows(
    all_data: pd.DataFrame,
    pitcher_id: int,
    pitch_type: str,
) -> int:
    """Return how many pitches exist for one (pitcher, pitch type) agent."""

    return int(len(_agent_pitch_subset(all_data, pitcher_id, pitch_type)))


def _renumber_agents(agents: Sequence[StatcastAgentSpec]) -> tuple[StatcastAgentSpec, ...]:
    return tuple(
        StatcastAgentSpec(agent_id=index, pitcher_id=spec.pitcher_id, pitch_type=spec.pitch_type)
        for index, spec in enumerate(agents)
    )


def filter_roster_by_min_pitches(
    roster: Sequence[StatcastAgentSpec],
    all_data: pd.DataFrame,
    min_pitches: int,
) -> tuple[tuple[StatcastAgentSpec, ...], tuple[tuple[int, str, int], ...]]:
    """Keep agents with at least ``min_pitches`` rows; return excluded (id, type, count)."""

    if min_pitches <= 0:
        raise ValueError(f"min_pitches must be positive. Received {min_pitches}.")

    kept: list[StatcastAgentSpec] = []
    excluded: list[tuple[int, str, int]] = []
    for agent_spec in roster:
        pitch_count = count_agent_pitch_rows(all_data, agent_spec.pitcher_id, agent_spec.pitch_type)
        if pitch_count >= min_pitches:
            kept.append(agent_spec)
        else:
            excluded.append((agent_spec.pitcher_id, agent_spec.pitch_type, pitch_count))
    return _renumber_agents(kept), tuple(excluded)


def list_eligible_pitcher_counts(
    all_data: pd.DataFrame,
    pitch_types: Sequence[str],
    *,
    min_pitches: int,
    limit: int | None = 20,
) -> list[tuple[int, str, int]]:
    """Return (pitcher_id, pitch_type, count) rows meeting ``min_pitches``."""

    rows: list[tuple[int, str, int]] = []
    for pitch_type in pitch_types:
        counts = all_data[all_data["pitch_type"] == pitch_type].groupby("pitcher").size()
        for pitcher_id, pitch_count in counts[counts >= min_pitches].items():
            rows.append((int(pitcher_id), str(pitch_type), int(pitch_count)))
    rows.sort(key=lambda item: (-item[2], item[1], item[0]))
    return rows if limit is None else rows[:limit]


def build_eligible_agent_roster(
    all_data: pd.DataFrame,
    pitch_types: Sequence[str],
    *,
    min_pitches: int,
    max_agents: int | None = None,
) -> tuple[StatcastAgentSpec, ...]:
    """Return all (pitcher, pitchType) agents meeting ``min_pitches``, sorted by pitch count."""

    if min_pitches <= 0:
        raise ValueError(f"min_pitches must be positive. Received {min_pitches}.")

    rows = list_eligible_pitcher_counts(
        all_data,
        pitch_types,
        min_pitches=min_pitches,
        limit=None,
    )
    if max_agents is not None:
        rows = rows[: max(0, max_agents)]
    return tuple(
        StatcastAgentSpec(agent_id=index, pitcher_id=pitcher_id, pitch_type=pitch_type)
        for index, (pitcher_id, pitch_type, _pitch_count) in enumerate(rows)
    )


def select_top_pitchers_by_pitch_count(
    all_data: pd.DataFrame,
    pitch_types: Sequence[str],
    *,
    min_pitches: int,
    count: int,
) -> tuple[int, ...]:
    """Return the top ``count`` pitcher IDs by total pitch count across ``pitch_types``."""

    if count <= 0:
        raise ValueError(f"count must be positive. Received {count}.")

    totals = all_data[all_data["pitch_type"].isin(tuple(pitch_types))].groupby("pitcher").size()
    eligible = totals[totals >= min_pitches]
    if eligible.empty:
        raise ValueError(
            f"No pitchers have at least {min_pitches} pitches for pitch types {tuple(pitch_types)}."
        )
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} pitchers but only {len(eligible)} meet min_pitches={min_pitches} "
            f"for pitch types {tuple(pitch_types)}."
        )
    ordered = sorted(
        ((int(pitcher_id), int(pitch_count)) for pitcher_id, pitch_count in eligible.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(pitcher_id for pitcher_id, _pitch_count in ordered[:count])


def resolve_agent_roster(
    pitcher_ids: Sequence[int],
    pitch_types: Sequence[str],
) -> tuple[StatcastAgentSpec, ...]:
    """Expand pitcher IDs × pitch types into numbered agent specs."""

    agents: list[StatcastAgentSpec] = []
    agent_id = 0
    for pitcher_id in pitcher_ids:
        for pitch_type in pitch_types:
            agents.append(
                StatcastAgentSpec(
                    agent_id=agent_id,
                    pitcher_id=int(pitcher_id),
                    pitch_type=str(pitch_type),
                )
            )
            agent_id += 1
    return tuple(agents)


def sample_noisy_action(
    rng: np.random.Generator,
    intended_action: tuple[float, float],
    execution_skill: float,
) -> tuple[float, float]:
    noise = rng.multivariate_normal(
        mean=np.zeros(2, dtype=float),
        cov=_execution_covariance(execution_skill),
    )
    return (
        float(intended_action[0]) + float(noise[0]),
        float(intended_action[1]) + float(noise[1]),
    )
