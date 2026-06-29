"""Lightweight pytests for the Blackhawks MCSE pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.BlackhawksMCSE import estimate_player_skill

LIGHTWEIGHT_TEST_PLAYERS = [950160, 950184]
TEST_SEASONS = [20232024]
DATA_DIR = Path("Data/Hockey")


def _player_has_offline_data(player_id: int) -> bool:
    player_dir = DATA_DIR / "players" / f"player_{player_id}" / "data"
    if not player_dir.exists():
        return False
    return any(player_dir.glob("shots_*.parquet"))


@pytest.mark.skipif(
    not all(_player_has_offline_data(pid) for pid in LIGHTWEIGHT_TEST_PLAYERS),
    reason="Offline shot data not cached for lightweight MCSE test players",
)
def test_mcse_offline_smoke() -> None:
    for player_id in LIGHTWEIGHT_TEST_PLAYERS:
        df, shot_maps = load_player_data(player_id, TEST_SEASONS, data_dir=DATA_DIR)
        if df.empty:
            pytest.skip(f"No offline data for player {player_id}")

        result = estimate_player_skill(
            player_id=player_id,
            offline_data=(df, shot_maps),
            shot_group="wristshot_snapshot",
            confirm=False,
            save_intermediate_csv=True,
            num_particles=100,
            rng_seed=0,
            data_dir=DATA_DIR,
        )

        assert result.get("status") == "success", result
        assert result["num_shots"] > 0
        assert result["ees_y"] is not None
        assert result["ees_z"] is not None
        csv_path = result.get("csv_path")
        assert csv_path is not None
        assert Path(csv_path).exists()
        assert "logs/mcse" in str(csv_path)
