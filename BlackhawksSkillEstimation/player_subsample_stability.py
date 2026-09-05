"""Shared pieces of the player subsample-stability experiment.

The experiment holds one player fixed, pools every cached season, and re-estimates
skill on many random N-shot subsets. Comparing the spread of those subsample
estimates against the player's season-to-season spread separates sampling noise at
season-sized N from genuine season-over-season change.

This module owns the parts the builder, the cluster worker, and the plotting script
all have to agree on: which shots are in the pool, how a subset is drawn, what a
sample is called, and where results land.

Nothing here touches Snowflake. All shots come from the cached parquet/npz written
by :func:`BlackhawksSkillEstimation.BlackhawksJEEDS.save_player_data`.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import SHOT_TYPE_GROUPS

EXPERIMENT_NAME = "player_subsample_stability"

DEFAULT_PLAYER_ID = 950160
DEFAULT_SHOT_GROUP = "wristshot_snapshot"
DEFAULT_DATA_ROOT = Path("Data/Hockey")
DEFAULT_EXPERIMENT_ROOT = DEFAULT_DATA_ROOT / "experiments" / EXPERIMENT_NAME

DEFAULT_N_SHOTS: tuple[int, ...] = (100, 200, 400)
DEFAULT_NUM_SEEDS = 50
DEFAULT_BASE_SEED = 20260904

ESTIMATORS: tuple[str, ...] = ("jeeds", "mcse")
BASELINE_SAMPLE_KEY = "baseline"

# Grids matching the production cluster configs, so subsample estimates are directly
# comparable to the existing per-season numbers:
#   JEEDS 250x250  -> Data/Hockey/jobs/jeeds_forwards_all_seasons.json ("estimator" block)
#   MCSE 500       -> Data/Hockey/jobs/mcse_forwards_per_season.json
# The local CLI defaults (50 execution x 100 planning, 1000 particles) are NOT
# comparable and must not be used for runs that get plotted against season dots.
PRODUCTION_NUM_EXECUTION_SKILLS = 250
PRODUCTION_NUM_PLANNING_SKILLS = 250
PRODUCTION_NUM_PARTICLES = 500

SMOKE_NUM_EXECUTION_SKILLS = 50
SMOKE_NUM_PLANNING_SKILLS = 100
SMOKE_NUM_PARTICLES = 200
SMOKE_NUM_SEEDS = 3

EXECUTION_SKILL_MIN = 0.004
EXECUTION_SKILL_MAX = 0.25

_SEASON_FILE_RE = re.compile(r"^shots_(\d{8})\.parquet$")
_SAMPLE_KEY_RE = re.compile(r"^n(\d+)_seed(\d+)$")

# Sorting the pool this way puts every subset in a single canonical order. JEEDS
# posteriors are order-independent, but MCSE is a particle filter and is not, so the
# order a subset is fed in has to be a property of the shots themselves rather than
# of whichever seed happened to draw them.
_CHRONOLOGICAL_KEYS = ["season", "game_id", "event_id"]


def filter_shot_group(df: pd.DataFrame, shot_group: str) -> pd.DataFrame:
    """Apply the estimator's shot-type filter.

    Mirrors the filtering inside ``_run_jeeds_estimation`` and
    ``_filter_df_by_shot_group`` so the sampled pool size matches what the
    estimators will actually consume.
    """
    if shot_group and shot_group not in SHOT_TYPE_GROUPS:
        valid = ", ".join(sorted(SHOT_TYPE_GROUPS))
        raise ValueError(f"Unknown shot_group '{shot_group}'. Valid groups: {valid}")

    _, allowed_types, include_null = SHOT_TYPE_GROUPS[shot_group or DEFAULT_SHOT_GROUP]

    df_lc = df.rename(columns=str.lower)
    if "shot_type" not in df_lc.columns:
        return df_lc

    shot_series = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
    shot_lower = shot_series.astype(str).str.lower()
    mask = shot_lower.isin(allowed_types)
    if include_null:
        mask = shot_series.isna() | mask
    return df_lc[mask]


def discover_cached_seasons(player_id: int, data_root: Path | str = DEFAULT_DATA_ROOT) -> list[int]:
    """Return every season with both a shots parquet and a shot-maps npz on disk."""
    data_dir = Path(data_root) / "players" / f"player_{player_id}" / "data"
    if not data_dir.exists():
        return []

    seasons: list[int] = []
    for path in sorted(data_dir.glob("shots_*.parquet")):
        match = _SEASON_FILE_RE.match(path.name)
        if not match:
            continue
        season = int(match.group(1))
        if (data_dir / f"shot_maps_{season}.npz").exists():
            seasons.append(season)
    return seasons


def load_pool_metadata(
    player_id: int,
    seasons: Sequence[int],
    *,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    shot_group: str = DEFAULT_SHOT_GROUP,
) -> pd.DataFrame:
    """Read the eligible shot pool from cached parquet only.

    The builder needs shot identity, not reward surfaces, so this deliberately skips
    the much larger ``shot_maps_<season>.npz`` files. The cached parquet has already
    had the proximity and angular-transform rejections applied at save time, so every
    surviving row is one the estimators can actually consume.
    """
    data_dir = Path(data_root) / "players" / f"player_{player_id}" / "data"

    frames: list[pd.DataFrame] = []
    for season in seasons:
        shots_path = data_dir / f"shots_{season}.parquet"
        if not shots_path.exists():
            raise FileNotFoundError(f"Missing shots file: {shots_path}")
        frames.append(pd.read_parquet(shots_path))

    if not frames:
        raise ValueError(f"No seasons requested for player {player_id}.")

    pooled = pd.concat(frames, ignore_index=True)
    return sort_chronologically(filter_shot_group(pooled, shot_group))


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in _CHRONOLOGICAL_KEYS if k in df.columns]
    if not keys:
        return df.reset_index(drop=True)
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True)


def season_counts(df: pd.DataFrame) -> dict[str, int]:
    """Shots per season, keyed by season string so the result survives JSON."""
    if "season" not in df.columns:
        return {}
    counts = df["season"].value_counts().to_dict()
    return {str(int(season)): int(count) for season, count in sorted(counts.items())}


def season_fractions(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {season: float("nan") for season in counts}
    return {season: count / total for season, count in counts.items()}


def sample_key_for(n_shots: int, seed: int) -> str:
    """Stable, sortable, filename-safe identifier for one draw."""
    return f"n{int(n_shots):04d}_seed{int(seed):04d}"


def parse_sample_key(sample_key: str) -> tuple[int | None, int | None]:
    """Return ``(n_requested, seed)``; both ``None`` for the baseline."""
    if sample_key == BASELINE_SAMPLE_KEY:
        return None, None
    match = _SAMPLE_KEY_RE.match(sample_key)
    if not match:
        raise ValueError(f"Unrecognized sample key: {sample_key!r}")
    return int(match.group(1)), int(match.group(2))


def draw_event_ids(
    pool_event_ids: Sequence[int],
    *,
    n_shots: int,
    seed: int,
    base_seed: int = DEFAULT_BASE_SEED,
) -> list[int]:
    """Draw ``n_shots`` event ids without replacement.

    Each ``(n_shots, seed)`` pair gets its own RNG stream, so adding an N value or
    more seeds later never perturbs draws that already ran on the cluster.
    """
    if n_shots > len(pool_event_ids):
        raise ValueError(f"Cannot draw {n_shots} shots from a pool of {len(pool_event_ids)}.")

    rng = np.random.default_rng([int(base_seed), int(n_shots), int(seed)])
    drawn = rng.choice(np.asarray(pool_event_ids, dtype=np.int64), size=int(n_shots), replace=False)
    return [int(x) for x in drawn]


def subset_pool(pool_df: pd.DataFrame, event_ids: Iterable[int]) -> pd.DataFrame:
    """Select the sampled shots from the pool, back in chronological order."""
    wanted = {int(x) for x in event_ids}
    subset = pool_df[pool_df["event_id"].astype("int64").isin(wanted)]
    return sort_chronologically(subset)


def build_samples(
    pool_df: pd.DataFrame,
    *,
    n_shots: Sequence[int],
    num_seeds: int,
    base_seed: int = DEFAULT_BASE_SEED,
) -> tuple[dict[str, dict[str, object]], list[int]]:
    """Build the baseline plus every ``(N, seed)`` draw.

    Returns the samples keyed by ``sample_key`` and the list of N values that were
    skipped because they exceed the pool.
    """
    pool_event_ids = [int(x) for x in sort_chronologically(pool_df)["event_id"]]
    pool_size = len(pool_event_ids)

    samples: dict[str, dict[str, object]] = {
        BASELINE_SAMPLE_KEY: _sample_record(
            pool_df,
            event_ids=pool_event_ids,
            n_requested=None,
            seed=None,
        )
    }

    skipped: list[int] = []
    for n in sorted({int(x) for x in n_shots}):
        if n <= 0:
            raise ValueError(f"n_shots values must be positive, got {n}.")
        if n > pool_size:
            skipped.append(n)
            continue
        for seed in range(int(num_seeds)):
            event_ids = draw_event_ids(pool_event_ids, n_shots=n, seed=seed, base_seed=base_seed)
            samples[sample_key_for(n, seed)] = _sample_record(
                pool_df,
                event_ids=event_ids,
                n_requested=n,
                seed=seed,
            )

    return samples, skipped


def _sample_record(
    pool_df: pd.DataFrame,
    *,
    event_ids: Sequence[int],
    n_requested: int | None,
    seed: int | None,
) -> dict[str, object]:
    subset = subset_pool(pool_df, event_ids)
    counts = season_counts(subset)
    return {
        "n_requested": n_requested,
        "seed": seed,
        "n_shots": len(subset),
        "season_counts": counts,
        "event_ids": [int(x) for x in subset["event_id"]],
    }


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


def run_dir_for(run_name: str, experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT) -> Path:
    return Path(experiment_root) / run_name


def results_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / "results"


def logs_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / "logs"


def summaries_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / "summaries"


def plots_dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / "plots"


def work_dir(run_dir: Path | str) -> Path:
    """Scratch root handed to the estimators as their ``data_dir``.

    Each job gets a private subdirectory here so concurrent array tasks cannot
    collide on the estimator's own CSV/timing filenames, which are derived from the
    season tag and would otherwise be identical for every subsample.
    """
    return Path(run_dir) / "work"


def job_tag(estimator: str, sample_key: str) -> str:
    return f"{estimator}__{sample_key}"


def result_path(run_dir: Path | str, estimator: str, sample_key: str) -> Path:
    return results_dir(run_dir) / f"{job_tag(estimator, sample_key)}.json"


def trace_csv_path(run_dir: Path | str, estimator: str, sample_key: str) -> Path:
    return logs_dir(run_dir) / f"{job_tag(estimator, sample_key)}.csv"


def load_results(run_dir: Path | str) -> list[dict[str, object]]:
    """Load every completed result JSON. Partial arrays are fine."""
    out: list[dict[str, object]] = []
    for path in sorted(results_dir(run_dir).glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"  WARNING: skipping unreadable result file {path}")
    return out


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def unified_execution_skill(estimator: str, estimates: dict[str, object]) -> float:
    """Single execution-skill scalar in radians, lower = better.

    JEEDS reports one posterior-mean skill. MCSE reports a 2D profile, collapsed to
    the geometric mean of the two axes, matching ``analysis/diagnose_estimator_signal``.
    """
    if estimator == "jeeds":
        return _as_float(estimates.get("ees"))
    if estimator == "mcse":
        ees_y = _as_float(estimates.get("ees_y"))
        ees_z = _as_float(estimates.get("ees_z"))
        if not np.isfinite(ees_y) or not np.isfinite(ees_z):
            return float("nan")
        return float(np.sqrt(ees_y * ees_z))
    raise ValueError(f"Unknown estimator: {estimator!r}")


def unified_map_execution_skill(estimator: str, estimates: dict[str, object]) -> float:
    if estimator == "jeeds":
        return _as_float(estimates.get("execution_skill"))
    if estimator == "mcse":
        map_y = _as_float(estimates.get("execution_skill_y"))
        map_z = _as_float(estimates.get("execution_skill_z"))
        if not np.isfinite(map_y) or not np.isfinite(map_z):
            return float("nan")
        return float(np.sqrt(map_y * map_z))
    raise ValueError(f"Unknown estimator: {estimator!r}")


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
