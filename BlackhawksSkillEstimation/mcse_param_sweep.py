"""Shared spec for the Blackhawks MCSE parameter sweep.

Stage 1 crosses proposal-noise divisors and the resample fraction at 500
particles. A later ranking keeps the settings whose final estimates agree
across seeds and whose particle cloud stays healthy. Stage 2 reruns that
shortlist at other particle counts, with the NEFF gate off, and on the legacy
parameter bounds.

Nothing here touches Snowflake or runs the filter. The builder, the cluster
worker, and the ranking script all import this module so a job id, a shot
prefix, and a score mean the same thing in each place.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksMCSE import (
    DEFAULT_LAMBDA_MODE,
    DEFAULT_MCSE_NOISE,
    DEFAULT_MCSE_RANGES,
    DEFAULT_RESAMPLE_PERCENT,
    DEFAULT_RESAMPLING_METHOD,
    LEGACY_MCSE_RANGES,
)
from BlackhawksSkillEstimation.player_subsample_stability import sort_chronologically

EXPERIMENT_NAME = "mcse_param_sweep"

DEFAULT_DATA_ROOT = Path("Data/Hockey")
DEFAULT_EXPERIMENT_ROOT = DEFAULT_DATA_ROOT / "experiments" / EXPERIMENT_NAME
DEFAULT_SHOT_GROUP = "wristshot_snapshot"

STAGE1_NUM_PARTICLES = 500
STAGE1_NUM_PLAYERS = 3
STAGE1_SEEDS: tuple[int, ...] = (0, 1, 2)
STAGE1_MAX_SHOTS = 400

# Length-2 noise is [state divisor, lambda divisor]. The state divisor is
# applied to execution skill and rho. A larger divisor is a tighter step:
# proposal std = prior width / divisor, and the step is added on every shot.
STATE_NOISE_GRID: tuple[int, ...] = (100, 200, 500, 1000)
LAMBDA_NOISE_GRID: tuple[int, ...] = (25, 50, 200, 400)
RESAMPLE_PERCENT_GRID: tuple[float, ...] = (0.75, 0.90, 0.95, 0.99)

STAGE2_PARTICLE_COUNTS: tuple[int, ...] = (250, 1000)
STAGE2_TOP_K = 3

SMOKE_MAX_SHOTS = 5
SMOKE_SEEDS: tuple[int, ...] = (0,)
SMOKE_NUM_PLAYERS = 1

STANDARD_SUBMIT_GROUP = "standard"
N1000_SUBMIT_GROUP = "n1000"
STANDARD_TIME = "24:00:00"
STANDARD_MEM = "16G"
N1000_TIME = "48:00:00"
N1000_MEM = "16G"
DEFAULT_MAX_CONCURRENT = 100

# A setting fails health when any seed trips one of these. They are reporting
# gates, not filter inputs, so the ranking script can retune them without
# rerunning the array.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "neff_ratio_min": 0.02,
    "bound_mass_max": 0.5,
    "late_drift_max": 0.15,
}

DIMENSIONS: tuple[str, ...] = ("y", "z", "rho", "lambda")
FINAL_COLUMNS: dict[str, str] = {
    "y": "ees_y",
    "z": "ees_z",
    "rho": "rho_ees",
    "lambda": "log10_eps",
}

TRACE_COLUMNS: tuple[str, ...] = (
    "shot_count",
    "ees_y",
    "ees_z",
    "rho_ees",
    "log10_eps",
    "map_execution_skill_y",
    "map_execution_skill_z",
    "map_rho",
    "log10_map_rationality",
    "neff",
    "neff_ratio",
    "resampled",
    "bound_mass_y",
    "bound_mass_z",
    "bound_mass_rho",
    "bound_mass_lambda",
    "bound_mass_y_low",
    "bound_mass_y_high",
    "bound_mass_z_low",
    "bound_mass_z_high",
    "bound_mass_rho_low",
    "bound_mass_rho_high",
    "bound_mass_lambda_low",
    "bound_mass_lambda_high",
)


def ranges_by_name(name: str) -> dict[str, list[float]]:
    """Copy of the current JEEDS-aligned bounds, or the pre-alignment bounds."""
    if name == "current":
        source = DEFAULT_MCSE_RANGES
    elif name == "legacy":
        source = LEGACY_MCSE_RANGES
    else:
        raise ValueError(f"ranges_name must be 'current' or 'legacy', got {name!r}")
    return {
        "start": [float(value) for value in source["start"]],
        "end": [float(value) for value in source["end"]],
    }


def prior_widths(ranges: dict[str, Sequence[float]]) -> dict[str, float]:
    """Width of each particle dimension. Lambda's width is in log10 units."""
    return {
        name: float(ranges["end"][index]) - float(ranges["start"][index])
        for index, name in enumerate(DIMENSIONS)
    }


def percent_token(resample_percent: float) -> str:
    """Stable 3-digit token for a resample fraction (0.90 -> '090')."""
    return f"{int(float(resample_percent) * 100 + 0.5):03d}"


def setting_key(setting: dict[str, Any]) -> str:
    """Identity of a filter setting, excluding player and seed."""
    noise = setting["noise"]
    neff = 1 if _as_bool(setting["resample_neff"]) else 0
    return (
        f"n{int(setting['num_particles'])}"
        f"_ns{int(noise[0])}"
        f"_nl{int(noise[1])}"
        f"_r{percent_token(float(setting['resample_percent']))}"
        f"_neff{neff}"
        f"_ranges{setting['ranges_name']}"
    )


def make_setting(
    *,
    noise_state: float,
    noise_lambda: float,
    resample_percent: float,
    num_particles: int,
    resample_neff: bool = True,
    ranges_name: str = "current",
    role: str = "stage1",
) -> dict[str, Any]:
    """One filter configuration, with the bounds copied in."""
    setting: dict[str, Any] = {
        "num_particles": int(num_particles),
        "noise": [float(noise_state), float(noise_lambda)],
        "resample_percent": float(resample_percent),
        "resample_neff": bool(resample_neff),
        "resampling_method": DEFAULT_RESAMPLING_METHOD,
        "lambda_mode": DEFAULT_LAMBDA_MODE,
        "ranges_name": str(ranges_name),
        "ranges": ranges_by_name(ranges_name),
        "role": str(role),
    }
    setting["setting_key"] = setting_key(setting)
    return setting


def stage1_settings() -> list[dict[str, Any]]:
    """Full stage-1 grid: 4 state noises x 4 lambda noises x 4 resample fractions."""
    settings: list[dict[str, Any]] = []
    for noise_state in STATE_NOISE_GRID:
        for noise_lambda in LAMBDA_NOISE_GRID:
            for resample_percent in RESAMPLE_PERCENT_GRID:
                settings.append(
                    make_setting(
                        noise_state=noise_state,
                        noise_lambda=noise_lambda,
                        resample_percent=resample_percent,
                        num_particles=STAGE1_NUM_PARTICLES,
                        resample_neff=True,
                        ranges_name="current",
                        role="stage1",
                    )
                )
    return settings


def smoke_settings() -> list[dict[str, Any]]:
    """Default cell plus one looser state-noise neighbor, for a wiring check."""
    return [
        make_setting(
            noise_state=DEFAULT_MCSE_NOISE[0],
            noise_lambda=DEFAULT_MCSE_NOISE[1],
            resample_percent=DEFAULT_RESAMPLE_PERCENT,
            num_particles=STAGE1_NUM_PARTICLES,
            role="stage1",
        ),
        make_setting(
            noise_state=STATE_NOISE_GRID[0],
            noise_lambda=DEFAULT_MCSE_NOISE[1],
            resample_percent=DEFAULT_RESAMPLE_PERCENT,
            num_particles=STAGE1_NUM_PARTICLES,
            role="stage1",
        ),
    ]


def legacy_default_setting() -> dict[str, Any]:
    """Current numeric defaults on the pre-alignment bounds."""
    return make_setting(
        noise_state=DEFAULT_MCSE_NOISE[0],
        noise_lambda=DEFAULT_MCSE_NOISE[1],
        resample_percent=DEFAULT_RESAMPLE_PERCENT,
        num_particles=STAGE1_NUM_PARTICLES,
        resample_neff=True,
        ranges_name="legacy",
        role="legacy_bounds",
    )


def submit_group_for(num_particles: int) -> str:
    """1000-particle jobs need a longer walltime than the 24h array."""
    if int(num_particles) >= 1000:
        return N1000_SUBMIT_GROUP
    return STANDARD_SUBMIT_GROUP


def job_resources(num_particles: int) -> tuple[str, str, str]:
    group = submit_group_for(num_particles)
    if group == N1000_SUBMIT_GROUP:
        return group, N1000_TIME, N1000_MEM
    return group, STANDARD_TIME, STANDARD_MEM


def job_id(player_id: int, setting: dict[str, Any], seed: int) -> str:
    return f"p{int(player_id)}_{setting['setting_key']}_seed{int(seed)}"


def expand_jobs(
    settings: Sequence[dict[str, Any]],
    *,
    player_ids: Sequence[int],
    seeds: Sequence[int],
    season: int,
    shot_group: str,
) -> list[dict[str, Any]]:
    """One eligible job per (setting, player, seed). Shot ids live on the config."""
    jobs: list[dict[str, Any]] = []
    for setting in settings:
        group, time_limit, memory = job_resources(int(setting["num_particles"]))
        for player_id in player_ids:
            for seed in seeds:
                jobs.append(
                    {
                        "job_id": job_id(player_id, setting, seed),
                        "setting_key": setting["setting_key"],
                        "player_id": int(player_id),
                        "seed": int(seed),
                        "season": int(season),
                        "shot_group": str(shot_group),
                        "num_particles": int(setting["num_particles"]),
                        "noise": [float(value) for value in setting["noise"]],
                        "resample_percent": float(setting["resample_percent"]),
                        "resample_neff": bool(setting["resample_neff"]),
                        "resampling_method": str(setting["resampling_method"]),
                        "lambda_mode": str(setting["lambda_mode"]),
                        "ranges_name": str(setting["ranges_name"]),
                        "ranges": {
                            "start": [float(value) for value in setting["ranges"]["start"]],
                            "end": [float(value) for value in setting["ranges"]["end"]],
                        },
                        "role": str(setting["role"]),
                        "submit_group": group,
                        "time": time_limit,
                        "mem": memory,
                        "eligible": True,
                    }
                )
    return jobs


def chronological_prefix(df: pd.DataFrame, max_shots: int) -> list[int]:
    """First ``max_shots`` distinct event ids in season / game / event order.

    The particle filter is order-dependent, so every seed of a player has to
    see this same prefix. Proximity and invalid-map drops still happen inside
    the estimator; they are a property of the shot, not of the seed.
    """
    if int(max_shots) <= 0:
        raise ValueError(f"max_shots must be positive, got {max_shots}.")
    ordered = sort_chronologically(df)
    if "event_id" not in ordered.columns:
        raise ValueError("Shot table has no event_id column.")

    event_ids: list[int] = []
    seen: set[int] = set()
    for raw in ordered["event_id"].tolist():
        event_id = int(raw)
        if event_id in seen:
            continue
        seen.add(event_id)
        event_ids.append(event_id)
        if len(event_ids) == int(max_shots):
            break
    if len(event_ids) < int(max_shots):
        raise ValueError(
            f"Need {int(max_shots)} shots for the prefix, found {len(event_ids)}."
        )
    return event_ids


def normalize_prefixes(prefixes: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    """JSON-ready shot prefixes keyed by player id string."""
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in prefixes.items():
        if isinstance(value, dict):
            event_ids = [int(event_id) for event_id in value["event_ids"]]
            pool_size = int(value.get("pool_size", len(event_ids)))
        else:
            event_ids = [int(event_id) for event_id in value]
            pool_size = len(event_ids)
        normalized[str(int(key))] = {
            "event_ids": event_ids,
            "n_shots": len(event_ids),
            "pool_size": pool_size,
        }
    return normalized


def event_ids_for_job(config: dict[str, Any], job: dict[str, Any]) -> list[int]:
    """Shot prefix for a job. Every seed of a player resolves to the same list."""
    prefixes = config["shot_prefixes"]
    prefix = prefixes[str(int(job["player_id"]))]
    return [int(event_id) for event_id in prefix["event_ids"]]


def select_cohort(
    counts: pd.DataFrame,
    *,
    max_shots: int,
    num_players: int,
    shot_group: str,
    season: int | None = None,
    player_ids: Sequence[int] | None = None,
) -> tuple[int, list[int]]:
    """Pick the season and the players the sweep will share.

    With no season given, choose the cached season that has the most players
    at or above ``max_shots``. Ties go to the later season. Players are then
    the ``num_players`` highest counts in that season.
    """
    required = {"player_id", "season", "count"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"Shot-count table is missing columns: {sorted(missing)}")
    if counts.empty:
        raise ValueError(f"No cached {shot_group} shots were found.")

    eligible = counts[counts["count"] >= int(max_shots)].copy()
    if season is not None:
        chosen_season = int(season)
        eligible = eligible[eligible["season"] == chosen_season]
        if eligible.empty:
            raise ValueError(
                f"Season {chosen_season} has no player with at least "
                f"{int(max_shots)} {shot_group} shots."
            )
    else:
        if eligible.empty:
            raise ValueError(
                f"No cached season has a player with at least {int(max_shots)} "
                f"{shot_group} shots."
            )
        tallies = eligible.groupby("season")["player_id"].nunique()
        best_count = int(tallies.max())
        chosen_season = int(tallies[tallies == best_count].index.max())
        eligible = eligible[eligible["season"] == chosen_season]

    eligible = eligible.sort_values("count", ascending=False).drop_duplicates("player_id")

    if player_ids is not None:
        wanted = [int(player_id) for player_id in player_ids]
        have = {int(player_id) for player_id in eligible["player_id"]}
        absent = [player_id for player_id in wanted if player_id not in have]
        if absent:
            raise ValueError(
                f"Players {absent} do not have {int(max_shots)} {shot_group} "
                f"shots in season {chosen_season}."
            )
        return chosen_season, wanted

    ranked = eligible.sort_values(["count", "player_id"], ascending=[False, True])
    chosen = [int(player_id) for player_id in ranked["player_id"].head(int(num_players))]
    if len(chosen) < int(num_players):
        raise ValueError(
            f"Season {chosen_season} has {len(chosen)} player(s) with at least "
            f"{int(max_shots)} {shot_group} shots; need {int(num_players)}."
        )
    return chosen_season, chosen


def assemble_config(
    *,
    stage: int,
    season: int,
    player_ids: Sequence[int],
    seeds: Sequence[int],
    shot_group: str,
    max_shots: int,
    settings: Sequence[dict[str, Any]],
    shot_prefixes: dict[Any, Any],
    data_root: Path | str = DEFAULT_DATA_ROOT,
    experiment_root: Path | str = DEFAULT_EXPERIMENT_ROOT,
    run_name: str | None = None,
    smoke: bool = False,
    thresholds: dict[str, float] | None = None,
    benchmark_tag: str | None = None,
    benchmark_dir: Path | str | None = None,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    notes: str = "",
) -> dict[str, Any]:
    """Config JSON body. Jobs point at ``shot_prefixes`` rather than copying ids."""
    prefixes = normalize_prefixes(dict(shot_prefixes))
    for player_id in player_ids:
        if str(int(player_id)) not in prefixes:
            raise ValueError(f"Missing shot prefix for player {player_id}.")

    jobs = expand_jobs(
        settings,
        player_ids=player_ids,
        seeds=seeds,
        season=int(season),
        shot_group=shot_group,
    )
    resolved_run = run_name or f"stage{int(stage)}"
    root = Path(data_root)
    benchmark_directory = Path(benchmark_dir) if benchmark_dir is not None else root / "benchmarks"
    return {
        "config_version": 1,
        "experiment": EXPERIMENT_NAME,
        "stage": int(stage),
        "run_name": resolved_run,
        "notes": notes,
        "smoke": bool(smoke),
        "data_root": str(root),
        "output_root": str(Path(experiment_root) / resolved_run),
        "shot_group": str(shot_group),
        "season": int(season),
        "player_ids": [int(player_id) for player_id in player_ids],
        "seeds": [int(seed) for seed in seeds],
        "max_shots": int(max_shots),
        "thresholds": dict(thresholds or DEFAULT_THRESHOLDS),
        "fixed": {
            "lambda_mode": DEFAULT_LAMBDA_MODE,
            "resampling_method": DEFAULT_RESAMPLING_METHOD,
        },
        "shot_prefixes": prefixes,
        "maxg": {
            "benchmark_tag": benchmark_tag,
            "benchmark_dir": str(benchmark_directory),
        },
        "cluster_plan": {
            "jobs": jobs,
            "total_jobs": len(jobs),
            "eligible_jobs": sum(1 for job in jobs if job["eligible"]),
            "sbatch_recommendation": {
                "time": STANDARD_TIME,
                "mem": STANDARD_MEM,
                "n1000_time": N1000_TIME,
                "n1000_mem": N1000_MEM,
                "max_concurrent": int(max_concurrent),
            },
        },
    }


def stage2_settings_from_ranking(
    rows: Sequence[dict[str, Any]],
    *,
    top_k: int = STAGE2_TOP_K,
) -> list[dict[str, Any]]:
    """Shortlist from a ranking table, plus the NEFF-off and legacy checks.

    ``rows`` are setting-level ranking records. Passing rows are ordered by
    ``rank``. The top ``top_k`` are repeated at 250 and 1000 particles. The
    rank-1 setting is also repeated with the NEFF gate off, at its stage-1
    particle count. The legacy-bounds default is always appended.
    """
    passing = [row for row in rows if _as_bool(row.get("passes"))]
    if not passing:
        raise ValueError(
            "No passing settings in the ranking. Retune the health thresholds "
            "or edit the ranking before building stage 2."
        )

    def _rank(row: dict[str, Any]) -> int:
        return int(float(row["rank"]))

    passing.sort(key=lambda row: (_rank(row), str(row.get("setting_key", ""))))
    chosen = passing[: int(top_k)]
    if not chosen:
        raise ValueError("top_k selected no settings.")

    settings: list[dict[str, Any]] = []
    for row in chosen:
        for num_particles in STAGE2_PARTICLE_COUNTS:
            settings.append(
                make_setting(
                    noise_state=float(row["noise_state"]),
                    noise_lambda=float(row["noise_lambda"]),
                    resample_percent=float(row["resample_percent"]),
                    num_particles=num_particles,
                    resample_neff=True,
                    ranges_name=str(row.get("ranges_name", "current")),
                    role="n_particles",
                )
            )

    best = chosen[0]
    settings.append(
        make_setting(
            noise_state=float(best["noise_state"]),
            noise_lambda=float(best["noise_lambda"]),
            resample_percent=float(best["resample_percent"]),
            num_particles=int(float(best["num_particles"])),
            resample_neff=False,
            ranges_name=str(best.get("ranges_name", "current")),
            role="neff_off",
        )
    )
    settings.append(legacy_default_setting())
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for setting in settings:
        if setting["setting_key"] in seen:
            continue
        seen.add(setting["setting_key"])
        deduped.append(setting)
    return deduped


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


def work_dir(run_dir: Path | str) -> Path:
    """Scratch root handed to MCSE so concurrent tasks do not share filenames."""
    return Path(run_dir) / "work"


def result_path(run_dir: Path | str, job_id_value: str) -> Path:
    return results_dir(run_dir) / f"{job_id_value}.json"


def trace_csv_path(run_dir: Path | str, job_id_value: str) -> Path:
    return logs_dir(run_dir) / f"{job_id_value}.csv"


def load_results(run_dir: Path | str) -> list[dict[str, Any]]:
    """Load every completed result JSON. A partial array is fine."""
    out: list[dict[str, Any]] = []
    directory = results_dir(run_dir)
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  WARNING: skipping unreadable result file {path}")
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


# ---------------------------------------------------------------------------
# Trace summaries and ranking
# ---------------------------------------------------------------------------


def _as_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def _late_start(n_rows: int) -> int:
    """Index where the last 20% of the trace begins."""
    if n_rows <= 1:
        return 0
    start = int(n_rows * 0.8)
    if start >= n_rows:
        start = n_rows - 1
    return start


def summarize_trace(
    rows: Sequence[dict[str, Any]],
    ranges: dict[str, Sequence[float]],
    num_particles: int,
) -> dict[str, float]:
    """Collapse one job's per-shot trace into the health numbers the rank uses.

    ``num_particles`` is accepted so callers can record the cloud size next to
    the ratios. The ratios themselves are already ``neff / N`` on each row.
    """
    del num_particles  # ratios are pre-normalized; the count is recorded by the caller
    empty = {
        "neff_mean": float("nan"),
        "neff_p10": float("nan"),
        "resample_fraction": float("nan"),
        "late_drift": float("nan"),
        "bound_mass_y": float("nan"),
        "bound_mass_z": float("nan"),
        "bound_mass_rho": float("nan"),
        "bound_mass_lambda": float("nan"),
    }
    if not rows:
        return empty

    ratios = np.array([_as_float(row.get("neff_ratio")) for row in rows], dtype=float)
    if np.isfinite(ratios).all() and ratios.size:
        neff_mean = float(np.mean(ratios))
        neff_p10 = float(np.percentile(ratios, 10))
    else:
        neff_mean = float("nan")
        neff_p10 = float("nan")

    resampled = np.array(
        [1.0 if _as_bool(row.get("resampled")) else 0.0 for row in rows],
        dtype=float,
    )
    widths = prior_widths(ranges)
    window = rows[_late_start(len(rows)) :]
    scaled: list[float] = []
    for name, column in FINAL_COLUMNS.items():
        values = np.array([_as_float(row.get(column)) for row in window], dtype=float)
        width = widths[name]
        if values.size == 0 or not np.isfinite(values).all() or width <= 0:
            scaled.append(float("nan"))
            continue
        scaled.append(float(np.std(values, ddof=0) / width))
    late = np.array(scaled, dtype=float)
    late_drift = float(np.mean(late)) if np.isfinite(late).all() else float("nan")

    last = rows[-1]
    bound: dict[str, float] = {}
    for name in DIMENSIONS:
        total = _as_float(last.get(f"bound_mass_{name}"))
        if not np.isfinite(total):
            low = _as_float(last.get(f"bound_mass_{name}_low"))
            high = _as_float(last.get(f"bound_mass_{name}_high"))
            if np.isfinite(low) and np.isfinite(high):
                total = float(low + high)
        bound[f"bound_mass_{name}"] = total

    return {
        "neff_mean": neff_mean,
        "neff_p10": neff_p10,
        "resample_fraction": float(np.mean(resampled)) if resampled.size else float("nan"),
        "late_drift": late_drift,
        **bound,
    }


def _seed_spread(group: pd.DataFrame) -> float:
    """Mean across dimensions of (sample std of the final EES) / prior width."""
    if len(group) < 2:
        return float("nan")
    ranges = group.iloc[0]["ranges"]
    if not isinstance(ranges, dict):
        return float("nan")
    widths = prior_widths(ranges)
    pieces: list[float] = []
    for name, column in FINAL_COLUMNS.items():
        values = group[column].to_numpy(dtype=float)
        width = widths[name]
        if values.size < 2 or not np.isfinite(values).all() or width <= 0:
            return float("nan")
        pieces.append(float(np.std(values, ddof=1) / width))
    return float(np.mean(pieces))


def _seed_health(row: pd.Series, thresholds: dict[str, float]) -> tuple[bool, str]:
    if str(row.get("status", "")) != "success":
        return False, "status"
    neff_mean = _as_float(row.get("neff_mean"))
    if not np.isfinite(neff_mean) or neff_mean < float(thresholds["neff_ratio_min"]):
        return False, "neff"
    for name in DIMENSIONS:
        mass = _as_float(row.get(f"bound_mass_{name}"))
        if not np.isfinite(mass) or mass > float(thresholds["bound_mass_max"]):
            return False, "bound"
    drift = _as_float(row.get("late_drift"))
    if not np.isfinite(drift) or drift > float(thresholds["late_drift_max"]):
        return False, "drift"
    return True, ""


def _mean_abs_maxg_gap(
    player_frame: pd.DataFrame,
    jeeds_maxg: dict[int, float] | None,
) -> float:
    """Mean over players of |MCSE MAXG - JEEDS MAXG|. Empty when either side is missing."""
    if not jeeds_maxg:
        return float("nan")
    column = "mean_maxg_ees" if "mean_maxg_ees" in player_frame.columns else "maxg_ees"
    if column not in player_frame.columns:
        return float("nan")
    gaps: list[float] = []
    for _, row in player_frame.iterrows():
        jeeds = jeeds_maxg.get(int(row["player_id"]))
        mcse = _as_float(row.get(column))
        if jeeds is None or not np.isfinite(jeeds) or not np.isfinite(mcse):
            continue
        gaps.append(abs(float(mcse) - float(jeeds)))
    if not gaps:
        return float("nan")
    return float(np.mean(gaps))


def score_sweep(
    records: Sequence[dict[str, Any]],
    *,
    expected_player_ids: Sequence[int] | None = None,
    expected_seeds: Sequence[int] | None = None,
    jeeds_maxg: dict[int, float] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank settings by seed spread among those that pass health on every player.

    The MAXG gap is attached when both sides are present and is not part of
    the sort. Failing settings stay in the table with an empty rank.
    """
    gates = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        gates.update(thresholds)

    setting_columns = [
        "setting_key",
        "noise_state",
        "noise_lambda",
        "resample_percent",
        "num_particles",
        "resample_neff",
        "ranges_name",
        "role",
        "mean_seed_spread",
        "mean_late_drift",
        "mean_neff",
        "max_bound_mass",
        "mean_abs_maxg_gap",
        "n_players",
        "passes",
        "fail_reasons",
        "rank",
    ]
    player_columns = [
        "setting_key",
        "player_id",
        "n_seeds",
        "seed_spread",
        "mean_late_drift",
        "min_neff_mean",
        "max_bound_mass",
        "healthy",
        "fail_reasons",
        "mean_ees_y",
        "mean_ees_z",
        "mean_rho_ees",
        "mean_log10_eps",
        "mean_maxg_ees",
    ]
    if not records:
        return (
            pd.DataFrame(columns=setting_columns),
            pd.DataFrame(columns=player_columns),
        )

    frame = pd.DataFrame(records).copy()
    if "setting_key" not in frame.columns:
        frame["setting_key"] = frame.apply(lambda row: setting_key(row.to_dict()), axis=1)
    for column in ("ees_y", "ees_z", "rho_ees", "log10_eps", "neff_mean", "late_drift", "maxg_ees"):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for name in DIMENSIONS:
        column = f"bound_mass_{name}"
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    health_flags: list[bool] = []
    health_reasons: list[str] = []
    for _, row in frame.iterrows():
        ok, reason = _seed_health(row, gates)
        health_flags.append(ok)
        health_reasons.append(reason)
    frame["seed_healthy"] = health_flags
    frame["seed_fail_reason"] = health_reasons

    wanted_players = (
        [int(player_id) for player_id in expected_player_ids]
        if expected_player_ids is not None
        else sorted({int(player_id) for player_id in frame["player_id"]})
    )
    wanted_seeds = (
        [int(seed) for seed in expected_seeds]
        if expected_seeds is not None
        else sorted({int(seed) for seed in frame["seed"]})
    )

    player_rows: list[dict[str, Any]] = []
    setting_rows: list[dict[str, Any]] = []
    for setting_value, setting_frame in frame.groupby("setting_key", sort=True):
        first = setting_frame.iloc[0]
        noise = first["noise"] if "noise" in setting_frame.columns else [np.nan, np.nan]
        per_player: list[dict[str, Any]] = []
        for player_id in wanted_players:
            group = setting_frame[setting_frame["player_id"].astype(int) == int(player_id)]
            present_seeds = {int(seed) for seed in group["seed"]} if not group.empty else set()
            missing_seeds = [seed for seed in wanted_seeds if seed not in present_seeds]
            reasons: list[str] = []
            if missing_seeds:
                reasons.append("missing_seed")
            if not group.empty and not bool(group["seed_healthy"].all()):
                reasons.extend(
                    sorted({reason for reason in group["seed_fail_reason"] if reason})
                )
            spread = _seed_spread(group) if not group.empty else float("nan")
            if not np.isfinite(spread):
                reasons.append("spread")
            bound_max = float("nan")
            if not group.empty:
                bound_values = np.array(
                    [group[f"bound_mass_{name}"].to_numpy(dtype=float) for name in DIMENSIONS]
                )
                finite_bounds = bound_values[np.isfinite(bound_values)]
                if finite_bounds.size:
                    bound_max = float(np.max(finite_bounds))
            per_player.append(
                {
                    "setting_key": setting_value,
                    "player_id": int(player_id),
                    "n_seeds": int(len(group)),
                    "seed_spread": spread,
                    "mean_late_drift": float(group["late_drift"].mean()) if not group.empty else float("nan"),
                    "min_neff_mean": float(group["neff_mean"].min()) if not group.empty else float("nan"),
                    "max_bound_mass": bound_max,
                    "healthy": not reasons,
                    "fail_reasons": ";".join(dict.fromkeys(reasons)),
                    "mean_ees_y": float(group["ees_y"].mean()) if not group.empty else float("nan"),
                    "mean_ees_z": float(group["ees_z"].mean()) if not group.empty else float("nan"),
                    "mean_rho_ees": float(group["rho_ees"].mean()) if not group.empty else float("nan"),
                    "mean_log10_eps": float(group["log10_eps"].mean()) if not group.empty else float("nan"),
                    "mean_maxg_ees": (
                        float(group["maxg_ees"].mean())
                        if not group.empty and group["maxg_ees"].notna().any()
                        else float("nan")
                    ),
                }
            )
        player_rows.extend(per_player)
        player_frame = pd.DataFrame(per_player)
        passes = bool(player_frame["healthy"].all()) if not player_frame.empty else False
        fail_reasons = sorted(
            {
                reason
                for reasons in player_frame["fail_reasons"]
                for reason in str(reasons).split(";")
                if reason
            }
        )
        setting_rows.append(
            {
                "setting_key": setting_value,
                "noise_state": float(noise[0]),
                "noise_lambda": float(noise[1]),
                "resample_percent": float(first["resample_percent"]),
                "num_particles": int(first["num_particles"]),
                "resample_neff": bool(first["resample_neff"]),
                "ranges_name": str(first["ranges_name"]),
                "role": str(first["role"]) if "role" in setting_frame.columns else "",
                "mean_seed_spread": float(player_frame["seed_spread"].mean()),
                "mean_late_drift": float(player_frame["mean_late_drift"].mean()),
                "mean_neff": float(setting_frame["neff_mean"].mean()),
                "max_bound_mass": float(player_frame["max_bound_mass"].max()),
                "mean_abs_maxg_gap": _mean_abs_maxg_gap(player_frame, jeeds_maxg),
                "n_players": int(player_frame["player_id"].nunique()),
                "passes": passes,
                "fail_reasons": ";".join(fail_reasons),
                "rank": np.nan,
            }
        )

    settings_frame = pd.DataFrame(setting_rows)
    if settings_frame.empty:
        return settings_frame, pd.DataFrame(player_rows)

    passing = settings_frame[settings_frame["passes"]].sort_values(
        ["mean_seed_spread", "mean_late_drift", "setting_key"],
        kind="mergesort",
    )
    failing = settings_frame[~settings_frame["passes"]].sort_values(
        ["setting_key"],
        kind="mergesort",
    )
    if not passing.empty:
        passing = passing.copy()
        passing["rank"] = np.arange(1, len(passing) + 1)
    ranked = pd.concat([passing, failing], ignore_index=True)
    return ranked[setting_columns], pd.DataFrame(player_rows)[player_columns]


def find_jeeds_maxg(
    data_root: Path | str,
    *,
    shot_group: str,
    season: int | None = None,
) -> dict[int, float]:
    """JEEDS MAXG sums already written by ``maxg_evaluator``, if any.

    JEEDS tables are named ``maxg_results_<tag>_<season>_<shot_group>.csv``.
    MCSE and HJEEDS tables carry an extra estimator token and are skipped.
    A missing table leaves the ranking column empty; it does not change order.
    """
    root = Path(data_root)
    if not root.exists():
        return {}
    matches: list[Path] = []
    for path in sorted(root.rglob("maxg_results_*.csv")):
        name = path.name
        if "mcse" in name or "hjeeds" in name:
            continue
        if not name.endswith(f"_{shot_group}.csv"):
            continue
        matches.append(path)
    if season is not None:
        season_matches = [path for path in matches if str(int(season)) in path.name]
        if season_matches:
            matches = season_matches
    found: dict[int, float] = {}
    for path in matches:
        try:
            table = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError):
            print(f"  WARNING: skipping unreadable JEEDS MAXG table {path}")
            continue
        if "player_id" not in table.columns or "maxg_sum" not in table.columns:
            continue
        for _, row in table.iterrows():
            player_id = int(row["player_id"])
            value = _as_float(row["maxg_sum"])
            if player_id not in found and np.isfinite(value):
                found[player_id] = value
    return found
