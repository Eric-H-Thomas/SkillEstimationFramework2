from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    estimate_player_skill,
    estimate_multiple_players,
    save_player_data,
    load_player_data,
)

# At the bottom, set TEST_TO_RUN

def one_player():
    # 950 is Nathan MacKinnon
    estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
    print(f"JEEDS MAP execution skill: {estimates['execution_skill']:.4f} rad (lower is better)")
    print(f"JEEDS MAP rationality:     {estimates['rationality']:.2f} (higher is better)")

def three_players():
    # Test case for multiple player IDs
    # TODO: Fetch actual player IDs from snowflake. I didn't have access when I made this
    player_ids = [950160, 123456, 789012]  # Example player IDs
    game_ids = [44604, 270247]  # Example game IDs
    results = estimate_multiple_players(player_ids=player_ids, game_ids=game_ids)

    for result in results:
        player_id = result.get("player_id")
        status = result.get("status")

        if status == "success":
            print(f"\nPlayer {player_id}:")
            print(f"  Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
            print(f"  Rationality:     {result['rationality']:.2f} (higher is better, EXPERIMENTAL)")
            print(f"  Shots Used:      {result['num_shots']}")

            if "skill_log" in result and result["skill_log"]:
                print(f"  Tracked {len(result['skill_log'])} intermediate estimates")
        else:
            print(f"\nPlayer {player_id}: ERROR")
            print(f"  {result.get('error', 'Unknown error')}")

def one_player_one_season():
    # Per-season estimates (default)
    result = estimate_player_skill(
        player_id=950160,
        seasons=[20242025],
    )
    for season, data in result["per_season_results"].items():
        print(f"Season {season}: skill={data['execution_skill']:.4f}, shots={data['num_shots']}")
            
def one_player_two_seasons():
    # Per-season estimates (default)
    result = estimate_player_skill(
        player_id=950160,
        seasons=[20232024, 20242025],
    )
    for season, data in result["per_season_results"].items():
        print(f"Season {season}: skill={data['execution_skill']:.4f}, shots={data['num_shots']}")


def download_season_player_data():
    """Download and save player data to disk for offline use."""
    player_id = 950160
    # seasons = [20232024, 20242025]
    seasons = [20242025]
    
    print(f"Downloading data for player {player_id}, seasons {seasons}...")
    saved = save_player_data(
        player_id=player_id,
        seasons=seasons,
        output_dir="Data/Hockey",
        overwrite=True,
    )
    
    print("\nSaved files:")
    for season, paths in saved.items():
        print(f"  Season {season}:")
        print(f"    Shots: {paths['shots']}")
        print(f"    Shot maps: {paths['shot_maps']}")


def run_offline_estimation():
    """Run estimation using previously downloaded data (no DB access needed)."""
    player_id = 950160
    seasons = [20232024, 20242025]
    
    print(f"Loading offline data for player {player_id}...")
    offline_data = load_player_data(
        player_id=player_id,
        seasons=seasons,
        data_dir="Data/Hockey",
    )
    
    df, shot_maps = offline_data
    print(f"Loaded {len(df)} shots, {len(shot_maps)} shot maps")
    
    result = estimate_player_skill(
        player_id=player_id,
        offline_data=offline_data,
        per_season=True,
        confirm=False,  # Skip confirmation for automated runs
    )
    
    print("\nResults:")
    for season, data in result["per_season_results"].items():
        if data["status"] == "success":
            print(f"  Season {season}: skill={data['execution_skill']:.4f}, rationality={data['rationality']:.2f}, shots={data['num_shots']}")
        else:
            print(f"  Season {season}: {data['status']} - {data.get('warning', '')}")


TEST_TO_RUN = one_player
if __name__ == "__main__":
    TEST_TO_RUN()