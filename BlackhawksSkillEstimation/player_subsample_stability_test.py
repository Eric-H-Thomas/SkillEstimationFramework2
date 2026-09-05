"""Tests for the player subsample-stability sampling and config layer.

Covers the properties the cluster run depends on but that a failed array would only
reveal after hours of compute: draws are reproducible, JEEDS and MCSE see identical
shots, and impossible N values are dropped rather than silently truncated.

No estimator is invoked.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from BlackhawksSkillEstimation.build_player_subsample_config import (
    build_config,
    build_estimator_settings,
    build_jobs,
    seasons_from_cli,
)
from BlackhawksSkillEstimation.run_player_subsample_config import existing_result_is_complete
from BlackhawksSkillEstimation.player_subsample_stability import (
    BASELINE_SAMPLE_KEY,
    PRODUCTION_NUM_EXECUTION_SKILLS,
    PRODUCTION_NUM_PARTICLES,
    PRODUCTION_NUM_PLANNING_SKILLS,
    SMOKE_NUM_EXECUTION_SKILLS,
    SMOKE_NUM_PARTICLES,
    build_samples,
    draw_event_ids,
    filter_shot_group,
    parse_sample_key,
    sample_key_for,
    season_counts,
    sort_chronologically,
    subset_pool,
    unified_execution_skill,
)

SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]


def _make_pool(per_season: int = 40) -> pd.DataFrame:
    """Synthetic pool shaped like the cached parquet, in deliberately shuffled order."""
    rows = []
    for season_idx, season in enumerate(SEASONS):
        for i in range(per_season):
            game_id = 100 + season_idx * 10 + (i % 4)
            rows.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "event_id": int(f"{game_id}{i:04d}"),
                    "shot_type": "wristshot" if i % 3 else "slapshot",
                }
            )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=7).reset_index(drop=True)


def test_filter_shot_group_matches_estimator_rules() -> None:
    pool = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "shot_type": ["wristshot", "snapshot", "slapshot", None],
        }
    )

    # wristshot_snapshot includes NULL shot types; backhand does not.
    kept = set(filter_shot_group(pool, "wristshot_snapshot")["event_id"])
    assert kept == {1, 2, 4}
    assert filter_shot_group(pool, "slapshot")["event_id"].tolist() == [3]


def test_filter_shot_group_rejects_unknown_group() -> None:
    pool = pd.DataFrame({"event_id": [1], "shot_type": ["wristshot"]})
    with pytest.raises(ValueError, match="Unknown shot_group"):
        filter_shot_group(pool, "one_timer")


def test_draw_event_ids_is_deterministic_and_stream_separated() -> None:
    pool_ids = list(range(1000))

    first = draw_event_ids(pool_ids, n_shots=100, seed=3)
    assert first == draw_event_ids(pool_ids, n_shots=100, seed=3)
    assert len(set(first)) == 100, "draws must be without replacement"

    # Each (N, seed) owns its own stream, so adding N values or seeds later cannot
    # perturb draws that already ran on the cluster.
    assert first != draw_event_ids(pool_ids, n_shots=100, seed=4)
    assert first != draw_event_ids(pool_ids, n_shots=200, seed=3)[:100]
    assert first != draw_event_ids(pool_ids, n_shots=100, seed=3, base_seed=999)


def test_draw_event_ids_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="Cannot draw"):
        draw_event_ids(list(range(10)), n_shots=11, seed=0)


def test_sample_key_roundtrip() -> None:
    assert sample_key_for(200, 42) == "n0200_seed0042"
    assert parse_sample_key("n0200_seed0042") == (200, 42)
    assert parse_sample_key(BASELINE_SAMPLE_KEY) == (None, None)
    with pytest.raises(ValueError):
        parse_sample_key("nonsense")


def test_subset_pool_restores_chronological_order() -> None:
    pool = sort_chronologically(_make_pool())
    drawn = draw_event_ids([int(x) for x in pool["event_id"]], n_shots=30, seed=1)

    subset = subset_pool(pool, drawn)
    assert len(subset) == 30

    # Order must come from the shots, not from the draw: MCSE is a particle filter
    # and is sensitive to the order observations arrive in.
    keys = subset[["season", "game_id", "event_id"]].values.tolist()
    assert keys == sorted(keys)
    assert subset_pool(pool, list(reversed(drawn)))["event_id"].tolist() == subset["event_id"].tolist()


def test_build_samples_includes_baseline_and_skips_oversized_n() -> None:
    pool = sort_chronologically(_make_pool(per_season=20))  # 200 shots total
    samples, skipped = build_samples(pool, n_shots=[50, 100, 5000], num_seeds=3)

    assert skipped == [5000]
    assert BASELINE_SAMPLE_KEY in samples
    assert samples[BASELINE_SAMPLE_KEY]["n_shots"] == len(pool)
    assert samples[BASELINE_SAMPLE_KEY]["n_requested"] is None

    assert len(samples) == 1 + 2 * 3
    for n in (50, 100):
        for seed in range(3):
            sample = samples[sample_key_for(n, seed)]
            assert sample["n_shots"] == n
            assert sum(sample["season_counts"].values()) == n


def test_build_samples_records_season_mix() -> None:
    pool = sort_chronologically(_make_pool())
    samples, _ = build_samples(pool, n_shots=[100], num_seeds=2)

    baseline_counts = samples[BASELINE_SAMPLE_KEY]["season_counts"]
    assert baseline_counts == season_counts(pool)
    assert set(baseline_counts) == {str(s) for s in SEASONS}


def test_jobs_share_sample_keys_across_estimators() -> None:
    pool = sort_chronologically(_make_pool(per_season=20))
    samples, _ = build_samples(pool, n_shots=[50], num_seeds=4)
    jobs = build_jobs(samples, ["jeeds", "mcse"])

    assert len(jobs) == 2 * len(samples)
    by_estimator: dict[str, set[str]] = {}
    for job in jobs:
        by_estimator.setdefault(job["estimator"], set()).add(job["sample_key"])

    # The whole point of drawing at build time: both estimators run identical shots.
    assert by_estimator["jeeds"] == by_estimator["mcse"] == set(samples)
    assert jobs[0]["is_baseline"] is True


def test_estimator_settings_default_to_production_grids() -> None:
    settings = build_estimator_settings(
        ["jeeds", "mcse"],
        smoke=False,
        num_execution_skills=None,
        num_planning_skills=None,
        num_particles=None,
        rng_seed=0,
        save_intermediate_csv=True,
    )

    # Anything smaller is not comparable to the existing per-season estimates.
    assert settings["jeeds"]["num_execution_skills"] == PRODUCTION_NUM_EXECUTION_SKILLS
    assert settings["jeeds"]["num_planning_skills"] == PRODUCTION_NUM_PLANNING_SKILLS
    assert settings["mcse"]["num_particles"] == PRODUCTION_NUM_PARTICLES

    # MCSE must use the current EV-normalized lambda bounds. Older per-season
    # runs used a 4.0 upper bound, and estimates from the two are not comparable.
    assert settings["mcse"]["ranges"]["end"][0] == 0.25
    assert settings["mcse"]["ranges"]["end"][-1] == pytest.approx(3.0)


def test_smoke_settings_shrink_grids() -> None:
    settings = build_estimator_settings(
        ["jeeds", "mcse"],
        smoke=True,
        num_execution_skills=None,
        num_planning_skills=None,
        num_particles=None,
        rng_seed=0,
        save_intermediate_csv=False,
    )
    assert settings["jeeds"]["num_execution_skills"] == SMOKE_NUM_EXECUTION_SKILLS
    assert settings["mcse"]["num_particles"] == SMOKE_NUM_PARTICLES


def _write_fake_cache(root: Path, player_id: int, seasons: list[int]) -> None:
    data_dir = root / "players" / f"player_{player_id}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for season_idx, season in enumerate(seasons):
        rows = []
        for i in range(60):
            game_id = 100 + season_idx * 10 + (i % 3)
            rows.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "event_id": int(f"{game_id}{i:04d}"),
                    "shot_type": "wristshot",
                }
            )
        pd.DataFrame(rows).to_parquet(data_dir / f"shots_{season}.parquet")
        (data_dir / f"shot_maps_{season}.npz").write_bytes(b"")


def test_build_config_pools_every_cached_season(tmp_path: Path) -> None:
    _write_fake_cache(tmp_path, 950160, SEASONS)

    config = build_config(
        player_id=950160,
        seasons=None,
        data_root=tmp_path,
        experiment_root=tmp_path / "experiments",
        n_shots=[100, 200],
        num_seeds=5,
    )

    # All five seasons, including 20252026, which the older forwards job JSONs omit
    # only because they predate that data.
    assert config["seasons"] == SEASONS
    assert config["sampling"]["pool_size"] == 300
    assert len(config["sampling"]["pool_season_counts"]) == 5
    assert config["cluster_plan"]["total_jobs"] == 2 * (1 + 2 * 5)
    assert config["cluster_plan"]["sbatch_recommendation"]["mem"] == "16G"
    assert config["cluster_plan"]["sbatch_recommendation"]["time"] == "24:00:00"


def test_build_config_is_json_serializable_and_round_trips(tmp_path: Path) -> None:
    _write_fake_cache(tmp_path, 950160, SEASONS[:2])
    config = build_config(
        player_id=950160,
        data_root=tmp_path,
        experiment_root=tmp_path / "experiments",
        n_shots=[50],
        num_seeds=2,
        estimators=["jeeds"],
    )

    loaded = json.loads(json.dumps(config))
    sample_key = sample_key_for(50, 0)
    assert len(loaded["samples"][sample_key]["event_ids"]) == 50
    assert loaded["output_root"].endswith("player_950160")
    assert all(job["estimator"] == "jeeds" for job in loaded["cluster_plan"]["jobs"])


def test_build_config_drops_n_larger_than_pool(tmp_path: Path) -> None:
    _write_fake_cache(tmp_path, 950160, SEASONS[:1])  # 60 shots
    config = build_config(
        player_id=950160,
        data_root=tmp_path,
        experiment_root=tmp_path / "experiments",
        n_shots=[50, 400],
        num_seeds=2,
        estimators=["jeeds"],
    )

    assert config["sampling"]["skipped_n_shots"] == [400]
    assert set(config["samples"]) == {BASELINE_SAMPLE_KEY, sample_key_for(50, 0), sample_key_for(50, 1)}


def test_seasons_from_cli_all_seasons_discovers_cache() -> None:
    assert seasons_from_cli(None, True) is None
    assert seasons_from_cli(None, False) is None
    assert seasons_from_cli([20252026], False) == [20252026]
    with pytest.raises(ValueError, match="not both"):
        seasons_from_cli([20252026], True)


def test_existing_result_is_complete_retries_failures(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert existing_result_is_complete(missing) is False

    success = tmp_path / "success.json"
    success.write_text(json.dumps({"status": "success"}), encoding="utf-8")
    assert existing_result_is_complete(success) is True

    error = tmp_path / "error.json"
    error.write_text(json.dumps({"status": "error", "error": "boom"}), encoding="utf-8")
    assert existing_result_is_complete(error) is False

    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not json", encoding="utf-8")
    assert existing_result_is_complete(garbage) is False


def test_unified_execution_skill_collapses_mcse_axes() -> None:
    assert unified_execution_skill("jeeds", {"ees": 0.08}) == pytest.approx(0.08)
    assert unified_execution_skill("mcse", {"ees_y": 0.04, "ees_z": 0.09}) == pytest.approx(0.06)
    assert unified_execution_skill("mcse", {"ees_y": None, "ees_z": 0.09}) != unified_execution_skill(
        "mcse", {"ees_y": 0.04, "ees_z": 0.09}
    )
