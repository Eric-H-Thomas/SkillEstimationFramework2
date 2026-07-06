"""Tests for MCSE cluster config builder."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from BlackhawksSkillEstimation.build_mcse_cluster_config import (
    _aggregate_jobs,
    build_mcse_cluster_config,
)


def test_aggregate_jobs_sums_season_counts() -> None:
    summary = pd.DataFrame(
        [
            {"player_id": 1, "season": 20232024, "shot_group": "wristshot_snapshot", "count": 40, "missing_local_data": False},
            {"player_id": 1, "season": 20222023, "shot_group": "wristshot_snapshot", "count": 30, "missing_local_data": False},
            {"player_id": 2, "season": 20232024, "shot_group": "wristshot_snapshot", "count": 10, "missing_local_data": False},
        ]
    )
    jobs = _aggregate_jobs(summary, min_shots_per_job=50, shot_groups=["wristshot_snapshot"])
    by_player = {job["player_id"]: job for job in jobs}
    assert by_player[1]["season"] == -1
    assert by_player[1]["count"] == 70
    assert by_player[1]["eligible"] is True
    assert by_player[2]["eligible"] is False


def test_build_mcse_cluster_config_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = pd.DataFrame(
        [
            {"player_id": 950160, "season": 20232024, "shot_group": "wristshot_snapshot", "count": 100, "missing_local_data": False},
        ]
    )

    def _fake_summary(player_ids, seasons, shot_groups, data_dir=None):
        return summary

    monkeypatch.setattr(
        "BlackhawksSkillEstimation.build_mcse_cluster_config.data_io.build_observation_summary",
        _fake_summary,
    )

    config = build_mcse_cluster_config(
        data_root=Path("Data/Hockey"),
        player_ids=[950160],
        seasons=[20232024],
        shot_groups=["wristshot_snapshot"],
        split_mode="per_season",
        min_shots_per_job=50,
        num_particles=1000,
        generate_convergence_png=False,
        sbatch_time="12:00:00",
        sbatch_mem="16G",
        max_concurrent=10,
        rng_seed=0,
    )

    assert config["estimator"]["num_particles"] == 1000
    assert config["estimator"]["ranges"]["end"][0] == 0.25
    assert config["estimator"]["ranges"]["end"][-1] == 4.0
    assert config["cluster_plan"]["eligible_jobs"] == 1
    job = config["cluster_plan"]["jobs"][0]
    assert job["player_id"] == 950160
    assert job["season"] == 20232024


def test_derive_mcse_config_for_data_root(monkeypatch: pytest.MonkeyPatch) -> None:
    base = {
        "data_filters": {
            "player_ids": [950160],
            "seasons": [20232024],
            "shot_groups": ["wristshot_snapshot"],
        },
        "validation": {"min_shots_per_job": 50},
        "cluster_plan": {"split_mode": "per_season"},
    }
    summary = pd.DataFrame(
        [
            {"player_id": 950160, "season": 20232024, "shot_group": "wristshot_snapshot", "count": 0, "missing_local_data": True},
        ]
    )

    import BlackhawksSkillEstimation.build_mcse_cluster_config as mod

    monkeypatch.setattr(mod.data_io, "build_observation_summary", lambda *a, **k: summary)
    derived = mod.derive_mcse_config_for_data_root(base, "Data/Hockey_xg_new")
    assert derived["data_root"] == "Data/Hockey_xg_new"
    assert derived["maxg"]["benchmark_tag"] == mod.NEW_XG_BENCHMARK_TAG
    assert derived["cluster_plan"]["eligible_jobs"] == 0


def test_build_mcse_cluster_config_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = pd.DataFrame(
        [
            {"player_id": 1, "season": 20232024, "shot_group": "wristshot_snapshot", "count": 60, "missing_local_data": False},
        ]
    )
    monkeypatch.setattr(
        "BlackhawksSkillEstimation.build_mcse_cluster_config.data_io.build_observation_summary",
        lambda *a, **k: summary,
    )
    out = tmp_path / "mcse.json"
    config = build_mcse_cluster_config(
        data_root=tmp_path,
        player_ids=[1],
        seasons=[20232024],
        shot_groups=["wristshot_snapshot"],
        split_mode="per_season",
        min_shots_per_job=50,
        num_particles=500,
        generate_convergence_png=False,
        sbatch_time="8:00:00",
        sbatch_mem="8G",
        max_concurrent=5,
        rng_seed=1,
    )
    out.write_text(json.dumps(config), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["cluster_plan"]["jobs"][0]["eligible"] is True
