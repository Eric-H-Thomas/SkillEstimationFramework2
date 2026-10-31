import csv
import json

import numpy as np
import pandas as pd
import pytest

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.maxg_evaluator import load_hjeeds_sigmas
from BlackhawksSkillEstimation.shot_map_format import (
    CURRENT_SHOT_MAP_SHAPE,
    LEGACY_SHOT_MAP_SHAPE,
    ShotMapFormatError,
    check_shot_map_files,
    read_value_map_shape,
)
from HJEEDS import run_hockey_population
from HJEEDS.hockey_pipeline import HOCKEY_PRELIMINARY_HYPERPRIORS

SEASON_A = 20232024
SEASON_B = 20242025


def _write_season(data_dir, player_id, season, event_ids, shape=CURRENT_SHOT_MAP_SHAPE, seed=0):
    folder = data_dir / "players" / f"player_{player_id}" / "data"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    count = len(event_ids)
    pd.DataFrame(
        {
            "event_id": event_ids,
            "shot_type": "wristshot",
            "start_x": rng.uniform(65.0, 75.0, count),
            "start_y": rng.uniform(-2.0, 2.0, count),
            "location_y": rng.uniform(-2.5, 2.5, count),
            "location_z": rng.uniform(0.5, 3.5, count),
        }
    ).to_parquet(folder / f"shots_{season}.parquet")
    maps_path = folder / f"shot_maps_{season}.npz"
    np.savez_compressed(
        maps_path,
        event_ids=np.asarray(event_ids, dtype=np.int64),
        value_maps=rng.random((count, *shape)).astype(np.float32),
        net_covs=np.tile(np.eye(2), (count, 1, 1)),
        net_coords=np.zeros((count, 2)),
    )
    return maps_path


def _write_config(path, player_ids, seasons):
    jobs = [
        {"player_id": pid, "season": season, "eligible": True, "shot_group": "wristshot_snapshot"}
        for pid in player_ids
        for season in seasons
    ]
    path.write_text(json.dumps({"cluster_plan": {"jobs": jobs}}), encoding="utf-8")


def test_read_value_map_shape_distinguishes_current_legacy_and_empty(tmp_path):
    current = _write_season(tmp_path, 1, SEASON_A, [11, 12])
    legacy = _write_season(tmp_path, 2, SEASON_A, [21], shape=LEGACY_SHOT_MAP_SHAPE)
    empty = _write_season(tmp_path, 3, SEASON_A, [])

    assert read_value_map_shape(current) == CURRENT_SHOT_MAP_SHAPE
    assert read_value_map_shape(legacy) == LEGACY_SHOT_MAP_SHAPE
    assert read_value_map_shape(empty) is None
    assert check_shot_map_files([current, empty]) == {
        "value_map_shape": list(CURRENT_SHOT_MAP_SHAPE),
        "files_checked": 1,
        "empty_files": 1,
    }
    with pytest.raises(ShotMapFormatError, match="legacy"):
        check_shot_map_files([current, legacy])


def test_load_player_data_rejects_legacy_maps_unless_disabled(tmp_path):
    _write_season(tmp_path, 1, SEASON_A, [11, 12, 13])
    _write_season(tmp_path, 1, SEASON_B, [14], shape=LEGACY_SHOT_MAP_SHAPE)

    shots, maps = load_player_data(1, [SEASON_A], data_dir=tmp_path)
    assert len(shots) == 3
    assert sorted(maps) == [11, 12, 13]
    assert maps[12]["value_map"].shape == CURRENT_SHOT_MAP_SHAPE
    assert maps[12]["value_map"].dtype == np.float64

    with pytest.raises(ShotMapFormatError, match="legacy"):
        load_player_data(1, [SEASON_A, SEASON_B], data_dir=tmp_path)
    _, maps = load_player_data(1, [SEASON_A, SEASON_B], data_dir=tmp_path, expected_map_shape=None)
    assert maps[14]["value_map"].shape == LEGACY_SHOT_MAP_SHAPE


def test_pooled_run_uses_each_players_available_seasons(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    config = tmp_path / "config.json"
    _write_config(config, [1, 2, 3], [SEASON_A, SEASON_B])
    _write_season(data_dir, 1, SEASON_A, [101, 102, 103], seed=1)
    _write_season(data_dir, 1, SEASON_B, [104, 105], seed=2)
    _write_season(data_dir, 2, SEASON_A, [201, 202], seed=3)
    _write_season(data_dir, 2, SEASON_B, [203, 204, 205], seed=4)
    _write_season(data_dir, 3, SEASON_A, [301, 302, 303, 304], seed=5)
    common = ["--output-dir", str(output_dir)]

    assert run_hockey_population.main([
        "preflight", "--config", str(config), "--data-dir", str(data_dir),
        "--seasons", str(SEASON_A), str(SEASON_B),
        "--num-sigma-grid", "4", "--num-log-lambda-grid", "5", *common,
    ]) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["seasons"] == [SEASON_A, SEASON_B]
    assert manifest["shot_map_format"]["files_checked"] == 5
    assert manifest["grid"]["num_sigma_grid"] == 4
    assert manifest["grid"]["num_log_lambda_grid"] == 5
    assert {r["player_id"]: r["seasons"] for r in manifest["players"]} == {
        1: [SEASON_A, SEASON_B],
        2: [SEASON_A, SEASON_B],
        3: [SEASON_A],
    }

    for index in range(3):
        assert run_hockey_population.main(["worker", "--player-index", str(index), *common]) == 0
    with np.load(output_dir / "likelihoods" / "player_3.npz") as artifact:
        assert artifact["seasons"].tolist() == [SEASON_A]
        assert artifact["log_likelihood_grid"].shape == (4, 5)

    assert run_hockey_population.main(["aggregate", *common]) == 0
    with (output_dir / "agent_level_results.csv").open(newline="") as handle:
        rows = {int(row["player_id"]): row for row in csv.DictReader(handle)}
    assert {pid: (int(row["num_shots"]), int(row["num_seasons"])) for pid, row in rows.items()} == {
        1: (5, 2),
        2: (5, 2),
        3: (4, 1),
    }
    assert rows[1]["seasons"] == f"{SEASON_A};{SEASON_B}"
    for row in rows.values():
        for prefix in ("jeeds", "hierarchical"):
            assert row[f"{prefix}_status"] == "ok"
            for axis in ("sigma", "log_lambda"):
                low, mean, high = (float(row[f"{prefix}_{axis}_q05"]), float(row[f"{prefix}_mean_{axis}"]), float(row[f"{prefix}_{axis}_q95"]))
                assert low <= mean <= high
                assert float(row[f"{prefix}_sd_{axis}"]) > 0
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["seasons"] == [SEASON_A, SEASON_B]
    assert metadata["failed"] == 0
    assert metadata["hyperprior"]["mean_sigma"] == pytest.approx(np.exp(HOCKEY_PRELIMINARY_HYPERPRIORS.mean_vector[0]))

    results = load_hjeeds_sigmas(output_dir / "agent_level_results.csv", estimate="jeeds")
    assert results["mean_sigma"].tolist() == pytest.approx([float(rows[pid]["jeeds_mean_sigma"]) for pid in (1, 2, 3)])
    assert set(results["hjeeds_estimate"]) == {"jeeds"}
    results = load_hjeeds_sigmas(output_dir / "agent_level_results.csv")
    assert results["mean_sigma"].tolist() == pytest.approx([float(rows[pid]["hierarchical_mean_sigma"]) for pid in (1, 2, 3)])


def test_marginal_spread_spreads_cell_mass_between_midpoints():
    grid = np.arange(11, dtype=float)
    point_mass = np.zeros(11)
    point_mass[5] = 1.0
    assert run_hockey_population._marginal_spread(point_mass, grid) == pytest.approx((0.0, 4.55, 5.45))

    uniform = np.full(11, 1.0 / 11.0)
    sd, low, high = run_hockey_population._marginal_spread(uniform, grid)
    assert sd == pytest.approx(np.sqrt(10.0))
    assert (low, high) == pytest.approx((0.275, 9.725))


def test_preflight_stops_before_writing_a_manifest_on_legacy_maps(tmp_path, capsys):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    config = tmp_path / "config.json"
    _write_config(config, [1, 2], [SEASON_A, SEASON_B])
    _write_season(data_dir, 1, SEASON_A, [101], seed=1)
    _write_season(data_dir, 2, SEASON_A, [201], seed=2)
    _write_season(data_dir, 2, SEASON_B, [202], shape=LEGACY_SHOT_MAP_SHAPE, seed=3)

    status = run_hockey_population.main([
        "preflight", "--config", str(config), "--data-dir", str(data_dir),
        "--seasons", str(SEASON_A), str(SEASON_B), "--output-dir", str(output_dir),
    ])

    assert status == 2
    assert not (output_dir / "manifest.json").exists()
    assert "legacy" in capsys.readouterr().err


def test_season_and_seasons_flags_are_exclusive():
    with pytest.raises(SystemExit):
        run_hockey_population.main(["preflight", "--season", str(SEASON_A), "--seasons", str(SEASON_B)])
