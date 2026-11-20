"""Tests for the MCSE parameter sweep grid, shot prefixes, and ranking.

No particle filter is run. The cluster jobs are expensive, so these checks
cover the decisions a bad config would only reveal after the array finishes:
the stage-1 grid is complete, every seed of a player shares one shot prefix,
stage 2 expands the shortlist, and the rank follows seed spread rather than
the MAXG gap.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.BlackhawksMCSE import (
    LEGACY_MCSE_RANGES,
    posterior_bound_masses,
)
from BlackhawksSkillEstimation.build_mcse_param_sweep_config import (
    build_stage1_config,
    build_stage2_config,
)
from BlackhawksSkillEstimation.mcse_param_sweep import (
    LAMBDA_NOISE_GRID,
    RESAMPLE_PERCENT_GRID,
    STAGE1_NUM_PARTICLES,
    STATE_NOISE_GRID,
    assemble_config,
    chronological_prefix,
    event_ids_for_job,
    expand_jobs,
    find_jeeds_maxg,
    score_sweep,
    select_cohort,
    stage1_settings,
    stage2_settings_from_ranking,
    summarize_trace,
)
from BlackhawksSkillEstimation.player_subsample_stability import sort_chronologically

RANGES = {"start": [0.0, 0.0, -1.0, 0.0], "end": [1.0, 1.0, 1.0, 1.0]}
PLAYERS = [11, 22, 33]
SEEDS = [0, 1, 2]


def test_stage1_grid_covers_every_noise_and_resample_cell() -> None:
    settings = stage1_settings()

    assert len(settings) == 4 * 4 * 4
    assert len({setting["setting_key"] for setting in settings}) == 64
    assert {tuple(setting["noise"]) for setting in settings} == {
        (float(state), float(lam))
        for state, lam in itertools.product(STATE_NOISE_GRID, LAMBDA_NOISE_GRID)
    }
    assert {setting["resample_percent"] for setting in settings} == set(RESAMPLE_PERCENT_GRID)
    assert {setting["setting_key"].split("_")[3] for setting in settings} == {
        "r075",
        "r090",
        "r095",
        "r099",
    }
    assert all(setting["num_particles"] == STAGE1_NUM_PARTICLES for setting in settings)
    assert all(setting["resample_neff"] for setting in settings)
    assert all(setting["ranges_name"] == "current" for setting in settings)
    assert all(setting["lambda_mode"] == "estimated" for setting in settings)
    assert all(setting["resampling_method"] == "systematic" for setting in settings)

    jobs = expand_jobs(
        settings,
        player_ids=PLAYERS,
        seeds=SEEDS,
        season=20232024,
        shot_group="wristshot_snapshot",
    )
    assert len(jobs) == 576
    assert {job["submit_group"] for job in jobs} == {"standard"}
    assert all(job["time"] == "24:00:00" and job["mem"] == "16G" for job in jobs)


def test_chronological_prefix_is_shared_by_every_seed() -> None:
    shuffled = pd.DataFrame(
        {
            "season": [20232024, 20232024, 20232024, 20232024, 20232024],
            "game_id": [2, 1, 1, 2, 1],
            "event_id": [20, 11, 10, 21, 10],
        }
    ).sample(frac=1.0, random_state=3)

    prefix = chronological_prefix(shuffled, 3)
    ordered = sort_chronologically(shuffled.drop_duplicates("event_id"))
    assert prefix == [int(event_id) for event_id in ordered["event_id"].head(3)]
    assert prefix == [10, 11, 20]

    other = chronological_prefix(
        pd.DataFrame(
            {
                "season": [20232024] * 4,
                "game_id": [1, 1, 1, 1],
                "event_id": [4, 3, 2, 1],
            }
        ),
        3,
    )
    config = assemble_config(
        stage=1,
        season=20232024,
        player_ids=[1, 2],
        seeds=SEEDS,
        shot_group="wristshot_snapshot",
        max_shots=3,
        settings=stage1_settings()[:2],
        shot_prefixes={1: prefix, 2: other},
        data_root=Path("Data/Hockey"),
    )
    for player_id, expected in ((1, prefix), (2, other)):
        resolved = [
            tuple(event_ids_for_job(config, job))
            for job in config["cluster_plan"]["jobs"]
            if job["player_id"] == player_id
        ]
        assert len(resolved) == 2 * len(SEEDS)
        assert len(set(resolved)) == 1
        assert resolved[0] == tuple(expected)


def test_stage1_builder_uses_one_prefix_per_player(monkeypatch: pytest.MonkeyPatch) -> None:
    counts = pd.DataFrame(
        [
            {"player_id": 7, "season": 20232024, "count": 80},
            {"player_id": 8, "season": 20232024, "count": 50},
            {"player_id": 7, "season": 20222023, "count": 90},
        ]
    )

    def fake_pool(player_id: int, seasons: list[int], **kwargs: object) -> pd.DataFrame:
        del kwargs
        # Descending ids, so a prefix that skips the chronological sort is obvious.
        event_ids = list(range(int(player_id) * 100 + 7, int(player_id) * 100 - 1, -1))
        return pd.DataFrame(
            {
                "season": [seasons[0]] * len(event_ids),
                "game_id": [1] * len(event_ids),
                "event_id": event_ids,
                "shot_type": ["wristshot"] * len(event_ids),
            }
        )

    monkeypatch.setattr(
        "BlackhawksSkillEstimation.build_mcse_param_sweep_config.load_pool_metadata",
        fake_pool,
    )
    config = build_stage1_config(
        counts=counts,
        smoke=True,
        data_root=Path("Data/Hockey"),
        experiment_root=Path("Data/Hockey/experiments/mcse_param_sweep"),
    )

    assert config["season"] == 20232024
    assert config["player_ids"] == [7]
    assert config["seeds"] == [0]
    assert config["max_shots"] == 5
    jobs = config["cluster_plan"]["jobs"]
    assert len(jobs) == 2
    prefixes = [tuple(event_ids_for_job(config, job)) for job in jobs]
    assert len(set(prefixes)) == 1
    assert prefixes[0] == (700, 701, 702, 703, 704)


def test_select_cohort_prefers_the_later_season_on_a_tie() -> None:
    counts = pd.DataFrame(
        [
            {"player_id": 1, "season": 20222023, "count": 800},
            {"player_id": 2, "season": 20222023, "count": 800},
            {"player_id": 1, "season": 20232024, "count": 500},
            {"player_id": 2, "season": 20232024, "count": 450},
            {"player_id": 3, "season": 20232024, "count": 100},
        ]
    )

    season, players = select_cohort(
        counts,
        max_shots=400,
        num_players=2,
        shot_group="wristshot_snapshot",
    )
    assert season == 20232024
    assert players == [1, 2]

    pinned, ordered = select_cohort(
        counts,
        max_shots=400,
        num_players=2,
        shot_group="wristshot_snapshot",
        season=20222023,
        player_ids=[2, 1],
    )
    assert pinned == 20222023
    assert ordered == [2, 1]

    with pytest.raises(ValueError):
        select_cohort(
            counts,
            max_shots=400,
            num_players=2,
            shot_group="wristshot_snapshot",
            season=20232024,
            player_ids=[3],
        )


def _flat_trace(ees_y: float, neff_ratio: float, n_rows: int = 10) -> list[dict[str, object]]:
    rows = []
    for index in range(n_rows):
        rows.append(
            {
                "shot_count": index + 1,
                "ees_y": ees_y,
                "ees_z": 0.3,
                "rho_ees": 0.0,
                "log10_eps": 0.2,
                "neff_ratio": neff_ratio,
                "resampled": index % 2 == 0,
                "bound_mass_y": 0.0,
                "bound_mass_z": 0.0,
                "bound_mass_rho": 0.0,
                "bound_mass_lambda": 0.0,
            }
        )
    return rows


def _record_from_trace(
    *,
    noise: tuple[float, float],
    player_id: int,
    seed: int,
    trace: list[dict[str, object]],
    maxg_ees: float,
) -> dict[str, object]:
    health = summarize_trace(trace, RANGES, num_particles=500)
    last = trace[-1]
    return {
        "status": "success",
        "player_id": player_id,
        "seed": seed,
        "noise": [noise[0], noise[1]],
        "resample_percent": 0.90,
        "num_particles": 500,
        "resample_neff": True,
        "ranges_name": "current",
        "ranges": RANGES,
        "role": "stage1",
        "ees_y": last["ees_y"],
        "ees_z": last["ees_z"],
        "rho_ees": last["rho_ees"],
        "log10_eps": last["log10_eps"],
        "maxg_ees": maxg_ees,
        **health,
    }


def test_summarize_trace_reports_late_drift_and_final_bound_mass() -> None:
    trace = _flat_trace(0.2, 0.4)
    trace[-2]["ees_y"] = 0.0
    trace[-1]["ees_y"] = 1.0
    trace[-1]["bound_mass_y"] = 0.6

    summary = summarize_trace(trace, RANGES, num_particles=500)

    assert summary["neff_mean"] == pytest.approx(0.4)
    assert summary["neff_p10"] == pytest.approx(0.4)
    assert summary["resample_fraction"] == pytest.approx(0.5)
    assert summary["bound_mass_y"] == pytest.approx(0.6)
    # Last 20% is two ees_y values, 0 and 1. Population std is 0.5, y width is 1,
    # and the other three dimensions do not move, so the mean scaled std is 0.125.
    assert summary["late_drift"] == pytest.approx(0.125)


def test_score_ranks_stable_settings_ahead_of_a_closer_maxg_gap() -> None:
    stable_finals = (0.20, 0.21, 0.22)
    wide_finals = (0.0, 0.5, 1.0)
    records = []
    for player_id in (1, 2):
        for seed, ees_y in enumerate(stable_finals):
            records.append(
                _record_from_trace(
                    noise=(200.0, 200.0),
                    player_id=player_id,
                    seed=seed,
                    trace=_flat_trace(ees_y, 0.4),
                    maxg_ees=10.0,
                )
            )
        for seed, ees_y in enumerate(wide_finals):
            records.append(
                _record_from_trace(
                    noise=(100.0, 200.0),
                    player_id=player_id,
                    seed=seed,
                    trace=_flat_trace(ees_y, 0.4),
                    maxg_ees=1.0,
                )
            )
        for seed, ees_y in enumerate(stable_finals):
            neff = 0.01 if seed == 0 else 0.4
            records.append(
                _record_from_trace(
                    noise=(500.0, 200.0),
                    player_id=player_id,
                    seed=seed,
                    trace=_flat_trace(ees_y, neff),
                    maxg_ees=1.0,
                )
            )

    ranking, by_player = score_sweep(
        records,
        expected_player_ids=[1, 2],
        expected_seeds=SEEDS,
        jeeds_maxg={1: 1.0, 2: 1.0},
    )

    by_key = {row.setting_key: row for row in ranking.itertuples(index=False)}
    stable = next(row for row in by_key.values() if row.noise_state == 200)
    wide = next(row for row in by_key.values() if row.noise_state == 100)
    collapsed = next(row for row in by_key.values() if row.noise_state == 500)

    assert bool(stable.passes) is True
    assert bool(wide.passes) is True
    assert bool(collapsed.passes) is False
    assert stable.rank == 1
    assert wide.rank == 2
    assert pd.isna(collapsed.rank)
    assert "neff" in collapsed.fail_reasons
    assert stable.mean_abs_maxg_gap > wide.mean_abs_maxg_gap
    assert stable.mean_seed_spread < wide.mean_seed_spread
    assert set(by_player["player_id"]) == {1, 2}

    spread = float(np.std(stable_finals, ddof=1)) / 4.0
    assert stable.mean_seed_spread == pytest.approx(spread)


def test_stage2_expands_the_top_settings_and_keeps_the_shot_prefix() -> None:
    rows = [
        _ranking_row(rank=1, state=100, lam=25, percent=0.90, passes=True),
        _ranking_row(rank=2, state=200, lam=200, percent=0.95, passes=True),
        _ranking_row(rank=3, state=500, lam=50, percent=0.75, passes=True),
        _ranking_row(rank=4, state=1000, lam=400, percent=0.99, passes=True),
        _ranking_row(rank=None, state=100, lam=400, percent=0.75, passes=False),
    ]
    with pytest.raises(ValueError):
        stage2_settings_from_ranking([], top_k=3)

    stage1 = assemble_config(
        stage=1,
        season=20232024,
        player_ids=PLAYERS,
        seeds=SEEDS,
        shot_group="wristshot_snapshot",
        max_shots=3,
        settings=stage1_settings()[:1],
        shot_prefixes={11: [1, 2, 3], 22: [4, 5, 6], 33: [7, 8, 9]},
        data_root=Path("Data/Hockey"),
        experiment_root=Path("Data/Hockey/experiments/mcse_param_sweep"),
        run_name="stage1",
    )
    config = build_stage2_config(stage1, rows, top_k=3)
    jobs = config["cluster_plan"]["jobs"]

    assert len(jobs) == 72
    assert sum(job["role"] == "n_particles" for job in jobs) == 54
    assert sum(job["role"] == "neff_off" for job in jobs) == 9
    assert sum(job["role"] == "legacy_bounds" for job in jobs) == 9
    assert sum(job["submit_group"] == "n1000" for job in jobs) == 27
    assert all(job["time"] == "48:00:00" and job["mem"] == "16G" for job in jobs if job["submit_group"] == "n1000")
    assert all(job["time"] == "24:00:00" for job in jobs if job["submit_group"] == "standard")
    assert not any(job["noise"] == [1000.0, 400.0] for job in jobs)

    neff_off = next(job for job in jobs if job["role"] == "neff_off")
    assert neff_off["noise"] == [100.0, 25.0]
    assert neff_off["resample_percent"] == pytest.approx(0.90)
    assert neff_off["resample_neff"] is False
    assert neff_off["num_particles"] == 500

    legacy = next(job for job in jobs if job["role"] == "legacy_bounds")
    assert legacy["noise"] == [200.0, 200.0]
    assert legacy["resample_percent"] == pytest.approx(0.90)
    assert legacy["resample_neff"] is True
    assert legacy["num_particles"] == 500
    assert legacy["ranges"]["end"][0] == pytest.approx(float(LEGACY_MCSE_RANGES["end"][0]))
    assert legacy["ranges"]["start"][3] == pytest.approx(float(LEGACY_MCSE_RANGES["start"][3]))
    assert legacy["ranges"]["end"][3] == pytest.approx(float(LEGACY_MCSE_RANGES["end"][3]))

    for player_id in PLAYERS:
        stage1_ids = tuple(event_ids_for_job(stage1, {"player_id": player_id}))
        stage2_ids = {
            tuple(event_ids_for_job(config, job))
            for job in jobs
            if job["player_id"] == player_id
        }
        assert stage2_ids == {stage1_ids}


def test_health_snapshot_is_taken_before_noise_and_resample(tmp_path: Path) -> None:
    """The health numbers have to describe the posterior, not the cloud after the update."""
    from BlackhawksSkillEstimation.BlackhawksMCSE import _build_mcse_estimator

    estimator = _build_mcse_estimator(
        tmp_path,
        num_particles=30,
        noise=[2.0, 2.0],
        resample_percent=0.5,
        resample_neff=True,
        resampling_method="systematic",
        ranges={"start": [0.004, 0.004, -0.75, -1.0], "end": [0.25, 0.25, 0.75, 3.0]},
        rng_seed=0,
        record_filter_health=True,
    )
    rng = np.random.default_rng(0)
    before_particles = np.array(estimator.particles, copy=True)
    before_probs = np.array(estimator.probs, copy=True)
    estimator.addNoise(rng)
    snapshot = estimator.consume_health_snapshot()
    assert np.allclose(snapshot["particles"], before_particles)
    assert np.allclose(snapshot["probs"], before_probs)
    assert not np.allclose(estimator.particles, before_particles)

    estimator._health_snapshot = None
    estimator.particles[:, :] = before_particles
    weights = np.linspace(1.0, 2.0, estimator.N)
    estimator.probs = (weights / weights.sum()).reshape(-1, 1)
    pre_particles = np.array(estimator.particles, copy=True)
    pre_probs = np.array(estimator.probs, copy=True)

    class _Spaces:
        def deleteSpaceParticles(self, *_args: object, **_kwargs: object) -> None:
            return None

    estimator.resample(rng, _Spaces(), None, 3)
    snapshot = estimator.consume_health_snapshot()
    assert np.allclose(snapshot["particles"], pre_particles)
    assert np.allclose(snapshot["probs"], pre_probs)
    assert np.allclose(estimator.probs, 1.0 / estimator.N)


def test_posterior_bound_mass_uses_normalized_weights() -> None:
    particles = np.array(
        [
            [0.0, 0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0, 10.0],
        ]
    )
    probs = np.array([[1.0], [3.0]])
    masses = posterior_bound_masses(particles, probs, RANGES)

    assert masses["bound_mass_y_low"] == pytest.approx(0.25)
    assert masses["bound_mass_y_high"] == pytest.approx(0.75)
    assert masses["bound_mass_y"] == pytest.approx(1.0)
    assert masses["bound_mass_z"] == pytest.approx(0.0)
    assert masses["bound_mass_lambda"] == pytest.approx(1.0)


def test_find_jeeds_maxg_ignores_other_estimators(tmp_path: Path) -> None:
    jeeds = tmp_path / "maxg_results_tag_20232024_wristshot_snapshot.csv"
    jeeds.write_text("player_id,maxg_sum\n7,1.5\n", encoding="utf-8")
    mcse = tmp_path / "maxg_results_tag_20232024_wristshot_snapshot_mcse.csv"
    mcse.write_text("player_id,maxg_sum\n7,9.0\n", encoding="utf-8")

    found = find_jeeds_maxg(tmp_path, shot_group="wristshot_snapshot", season=20232024)
    assert found == {7: 1.5}


def _ranking_row(
    *,
    rank: int | None,
    state: int,
    lam: int,
    percent: float,
    passes: bool,
) -> dict[str, object]:
    return {
        "rank": rank,
        "passes": passes,
        "setting_key": f"n500_ns{state}_nl{lam}",
        "noise_state": state,
        "noise_lambda": lam,
        "resample_percent": percent,
        "num_particles": 500,
        "ranges_name": "current",
    }
