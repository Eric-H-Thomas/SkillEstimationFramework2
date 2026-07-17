# This file has been fully reviewed by a human researcher as of 07/16/26 at 12:32 PM MDT.
"""Baseball pitch surfaces and strike-zone grids for HJEEDS.

Wraps the classic baseball reward path (RNN outcomes + ``getUtility`` +
``convolve2d`` EV surfaces) without the full ``BaseballExp`` experiment loop.

Plate grids come from the processed Statcast pickle (built by
``SpacesBaseball.getAllData``); ``delta`` must match that pickle's spacing.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import convolve2d

REPO_ROOT = Path(__file__).resolve().parent.parent
STATCAST_DIR = REPO_ROOT / "Data" / "Baseball" / "StatcastData"
PROCESSED_PICKLE = STATCAST_DIR / "ProcessedData-From-GivenFiles.pkl"
MODEL_WEIGHTS = REPO_ROOT / "Environments" / "Baseball" / "final_OP"

# Matches SpacesBaseball / estimators.py baseball-multi defaults.
DEFAULT_DELTA = 0.0417
DEFAULT_EXECUTION_SKILL_MIN = 0.17
DEFAULT_EXECUTION_SKILL_MAX = 2.81
DEFAULT_EXECUTION_RHO = 0.0
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
    model: nn.Module
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


def _load_processed_pickle() -> tuple:
    """Load the six-item GivenFiles bundle written by ``SpacesBaseball.getAllData``."""

    if not PROCESSED_PICKLE.is_file():
        raise FileNotFoundError(
            f"Processed Statcast pickle not found: {PROCESSED_PICKLE}. "
            "Build it via SpacesBaseball.getAllData (dataTake2.manageData + target grids)."
        )
    with PROCESSED_PICKLE.open("rb") as handle:
        loaded = pickle.load(handle)
    return tuple(loaded[0])


def load_processed_statcast() -> pd.DataFrame:
    """Load the merged Statcast dataframe from the processed pickle."""

    return _load_processed_pickle()[0]


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


def build_strike_zone_grids(delta: float = DEFAULT_DELTA) -> tuple[StrikeZoneGrids, np.ndarray]:
    """Load strike-zone grids from the processed Statcast pickle; require matching ``delta``."""

    (
        _all_data,
        batter_indices,
        model_targets_plate_x,
        model_targets_plate_z,
        possible_targets_feet,
        possible_targets_for_model,
    ) = _load_processed_pickle()
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
            "Rebuild ProcessedData-From-GivenFiles.pkl at the desired resolution, "
            "or pass the matching delta."
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


def _load_model(batter_indices: np.ndarray) -> nn.Module:
    from Environments.Baseball import modelTake2

    # Embedding size uses ``.shape[0]`` (num batters), matching SpacesBaseball.
    modelTake2.batter_indices = np.asarray(batter_indices)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = modelTake2.RNN(hidden_size=32, output_size=9).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True))
    model.eval()
    return model


def _utility_grid_from_row(
    pitch_row: pd.Series,
    grids: StrikeZoneGrids,
    model: nn.Module,
) -> tuple[np.ndarray, float, float]:
    """Return (utility board Zs, min utility, observed utility for the actual pitch).

    Device placement follows ``modelTake2.prediction_func`` (module-global device),
    matching the classic SpacesBaseball path.
    """

    from Environments.Baseball import modelTake2, utilsBaseball

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

    ypred = modelTake2.prediction_func(
        model,
        torch.tensor(batch_x, dtype=torch.float),
        torch.tensor(batch_y, dtype=torch.long),
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
    from Environments.Baseball import baseball_multi as domain

    skill = float(skill)
    return domain.getCovMatrix([skill, skill], DEFAULT_EXECUTION_RHO)


def build_baseball_runtime(
    rng: np.random.Generator,
    execution_skills: Sequence[float],
    *,
    delta: float = DEFAULT_DELTA,
) -> BaseballRuntime:
    """Load model/geometry and precompute execution-noise PDFs."""

    from Environments.Baseball import baseball_multi as domain

    grids, batter_indices = build_strike_zone_grids(delta)
    model = _load_model(batter_indices)

    pdfs_per_execution_skill: dict[str, np.ndarray] = {}
    all_covs: dict[str, np.ndarray] = {}
    for execution_skill in execution_skills:
        skill = float(execution_skill)
        key = baseball_execution_skill_key(skill)
        cov = _execution_covariance(skill)
        all_covs[key] = cov
        pdfs_per_execution_skill[key] = domain.getNormalDistribution(
            rng,
            cov,
            grids.delta,
            grids.targets_plate_x_feet,
            grids.targets_plate_z_feet,
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
) -> PitchObservation:
    """Build one pitch observation with EV surfaces for all execution-skill keys."""

    zs, min_utility, observed_reward = _utility_grid_from_row(
        pitch_row,
        runtime.grids,
        runtime.model,
    )
    evs_per_execution_skill: dict[str, np.ndarray] = {}
    for skill in execution_skills:
        key = baseball_execution_skill_key(float(skill))
        evs_per_execution_skill[key] = np.asarray(
            convolve2d(
                zs,
                runtime.pdfs_per_execution_skill[key],
                mode="same",
                fillvalue=min_utility,
            ),
            dtype=float,
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
) -> list[PitchObservation]:
    """Build pitch observations for every row in a Statcast agent slice."""

    return [
        build_pitch_observation(row, runtime, execution_skills)
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

    agent_data = _agent_pitch_subset(all_data, pitcher_id, pitch_type).sort_values(
        by=["game_date"],
        ascending=False,
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
        for pitcher_id, pitch_count in counts[counts >= min_pitches].sort_values(ascending=False).items():
            rows.append((int(pitcher_id), str(pitch_type), int(pitch_count)))
    rows.sort(key=lambda item: item[2], reverse=True)
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
    eligible = totals[totals >= min_pitches].sort_values(ascending=False)
    if eligible.empty:
        raise ValueError(
            f"No pitchers have at least {min_pitches} pitches for pitch types {tuple(pitch_types)}."
        )
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} pitchers but only {len(eligible)} meet min_pitches={min_pitches} "
            f"for pitch types {tuple(pitch_types)}."
        )
    return tuple(int(pitcher_id) for pitcher_id in eligible.head(count).index)


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
    from Environments.Baseball import baseball_multi as domain

    noisy = domain.sample_noisy_action(
        rng,
        [0.0, 0.0],
        _execution_covariance(execution_skill),
        [float(intended_action[0]), float(intended_action[1])],
    )
    return float(noisy[0]), float(noisy[1])
