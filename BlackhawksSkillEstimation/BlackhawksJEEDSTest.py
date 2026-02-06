from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    estimate_player_skill,
    estimate_multiple_players,
    save_player_data,
    load_player_data,
    save_player_data_by_games,
    load_player_data_by_games,
)
from BlackhawksSkillEstimation.plot_intermediate_estimates import (
    plot_intermediate_estimates,
    plot_all_intermediate_for_player,
)

# At the bottom, set TEST_TO_RUN


def _print_all_estimates(result: dict, prefix: str = "  ") -> None:
    """Helper to print all 4 estimates from a result dict."""
    print(f"{prefix}MAP Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
    print(f"{prefix}EES:                 {result['ees']:.4f} rad")
    print(f"{prefix}MAP Rationality:     {result['rationality']:.2f} (higher is better)")
    print(f"{prefix}EPS:                 {result['eps']:.2f}")
    print(f"{prefix}Shots Used:          {result['num_shots']}")


def one_player():
    """Quick test with 1 player, 2 games."""
    estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
    print("\nPlayer 950160 Estimates:")
    _print_all_estimates(estimates)


def three_players():
    """Test with 3 players, 2 games each."""
    player_ids = [950160, 123456, 789012]
    game_ids = [44604, 270247]
    results = estimate_multiple_players(player_ids=player_ids, game_ids=game_ids)

    for result in results:
        player_id = result.get("player_id")
        status = result.get("status")

        if status == "success":
            print(f"\nPlayer {player_id}:")
            _print_all_estimates(result)

            if "skill_log" in result and result["skill_log"]:
                print(f"  Tracked {len(result['skill_log'])} intermediate estimates")
        else:
            print(f"\nPlayer {player_id}: ERROR")
            print(f"  {result.get('error', 'Unknown error')}")

def one_player_one_season():
    """Per-season estimates for a single season."""
    result = estimate_player_skill(
        player_id=950160,
        seasons=[20242025],
    )
    for season, data in result["per_season_results"].items():
        print(f"\nSeason {season}:")
        _print_all_estimates(data)
            
def one_player_two_seasons():
    """Per-season estimates across two seasons."""
    result = estimate_player_skill(
        player_id=950160,
        seasons=[20232024, 20242025],
    )
    for season, data in result["per_season_results"].items():
        print(f"\nSeason {season}:")
        _print_all_estimates(data)


def download_season_test_data():
    """Download and save player data to disk for offline use."""
    player_id = 950160
    seasons = [20232024, 20242025]
    
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


def run_offline_season_estimation():
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
        confirm=False,
    )
    
    print("\nResults:")
    for season, data in result["per_season_results"].items():
        if data["status"] == "success":
            print(f"\nSeason {season}:")
            _print_all_estimates(data)
        else:
            print(f"  Season {season}: {data['status']} - {data.get('warning', '')}")


# =============================================================================
# LIGHTWEIGHT PIPELINE TESTING 
# =============================================================================

# Players and games for lightweight testing
# These player IDs are from the Hawks database (not NHL API IDs)
# Each player uses their top 2 games by shot count (via get_games_for_player)
LIGHTWEIGHT_TEST_PLAYERS = [
    { 
        "player_id": 950160,  # 37 shots: 19 + 18
        "game_ids": [270247, 44604],
        "name": "Nathan MacKinnon",
    },
    {
        "player_id": 950184,  # 31 shots: 16 + 15
        "game_ids": [271408, 44840],
        "name": "Cale Makar",
    },
    { 
        "player_id": 949352,  # 25 shots: 13 + 12
        "game_ids": [44905, 42496],
        "name": "Kris Letang",
    },
]


def download_lightweight_test_data():
    """Download a small dataset for pipeline testing.
    
    Downloads data for a few players with only 2 games each.
    This is lightweight enough to run on a laptop and creates
    files that can be transferred to a compute cluster for offline use.
    """
    print("=" * 60)
    print("DOWNLOADING LIGHTWEIGHT TEST DATA")
    print("=" * 60)
    print(f"Players: {len(LIGHTWEIGHT_TEST_PLAYERS)}")
    print(f"Games per player: 2")
    print()
    
    all_saved = {}
    for player_info in LIGHTWEIGHT_TEST_PLAYERS:
        player_id = player_info["player_id"]
        game_ids = player_info["game_ids"]
        name = player_info["name"]
        
        print(f"\n--- {name} (ID: {player_id}) ---")
        saved = save_player_data_by_games(
            player_id=player_id,
            game_ids=game_ids,
            output_dir="Data/Hockey",
            overwrite=True,
            tag="2games_test",
        )
        
        if saved:
            all_saved[player_id] = saved
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print("\nSaved files can be found in Data/Hockey/player_*/")
    print("Transfer this folder to the compute cluster for offline experiments.")
    
    return all_saved


def run_offline_lightweight_estimation():
    """Run estimation using previously downloaded lightweight test data.
    
    This requires no database/internet access - uses only local pickle files.
    Suitable for running on a compute cluster without network access.
    """
    print("=" * 60)
    print("RUNNING OFFLINE ESTIMATION (LIGHTWEIGHT TEST DATA)")
    print("=" * 60)
    
    results = []
    for player_info in LIGHTWEIGHT_TEST_PLAYERS:
        player_id = player_info["player_id"]
        name = player_info["name"]
        
        print(f"\n--- {name} (ID: {player_id}) ---")
        
        try:
            # Load offline data
            df, shot_maps = load_player_data_by_games(
                player_id=player_id,
                tag="2games_test",
                data_dir="Data/Hockey",
            )
            print(f"  Loaded {len(df)} shots, {len(shot_maps)} shot maps")
            
            # Run estimation
            result = estimate_player_skill(
                player_id=player_id,
                offline_data=(df, shot_maps),
                per_season=False,
                confirm=False,
            )
            
            if result.get("status") == "success" or "execution_skill" in result:
                _print_all_estimates(result)
                
                results.append({
                    "player_id": player_id,
                    "name": name,
                    "status": "success",
                    "execution_skill": result.get("execution_skill"),
                    "ees": result.get("ees"),
                    "rationality": result.get("rationality"),
                    "eps": result.get("eps"),
                    "num_shots": result.get("num_shots", len(df)),
                })
            else:
                print(f"  Status: {result.get('status', 'unknown')}")
                results.append({
                    "player_id": player_id,
                    "name": name,
                    "status": result.get("status", "unknown"),
                })
                
        except FileNotFoundError as e:
            print(f"  ERROR: Data not found. Run download_lightweight_test_data() first.")
            print(f"  {e}")
            results.append({
                "player_id": player_id,
                "name": name,
                "status": "file_not_found",
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "player_id": player_id,
                "name": name,
                "status": "error",
                "error": str(e),
            })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        if r["status"] == "success":
            print(f"{r['name']}:")
            _print_all_estimates(r, prefix="  ")
        else:
            print(f"{r['name']}: {r['status']}")
    
    return results


def full_lightweight_pipeline_test():
    """Run the complete lightweight pipeline: download then estimate."""
    print("\n" + "=" * 60)
    print("FULL LIGHTWEIGHT PIPELINE TEST")
    print("=" * 60)
    print("\nStep 1: Download data for a few players (2 games each)")
    print("-" * 40)
    
    download_lightweight_test_data()
    
    print("\n\nStep 2: Run offline estimation")
    print("-" * 40)
    
    run_offline_lightweight_estimation()
    
    print("\n\n" + "=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)


def test_intermediate_csv_and_plot():
    """Test intermediate estimate logging to CSV and plotting.
    
    Uses previously downloaded lightweight test data.
    Saves intermediate estimates to CSV and generates convergence plots.
    """
    print("=" * 60)
    print("TESTING INTERMEDIATE CSV EXPORT AND PLOTTING")
    print("=" * 60)
    
    player_id = 950160
    
    try:
        df, shot_maps = load_player_data_by_games(
            player_id=player_id,
            tag="2games_test",
            data_dir="Data/Hockey",
        )
        print(f"Loaded {len(df)} shots for player {player_id}")
    except FileNotFoundError:
        print("ERROR: Data not found. Run download_lightweight_test_data() first.")
        return
    
    # Run estimation with CSV export enabled
    print("\nRunning estimation with save_intermediate_csv=True...")
    result = estimate_player_skill(
        player_id=player_id,
        offline_data=(df, shot_maps),
        per_season=False,
        confirm=False,
        save_intermediate_csv=True,
    )
    
    print("\nFinal Estimates:")
    _print_all_estimates(result)
    
    if "csv_path" in result:
        csv_path = result["csv_path"]
        print(f"\nCSV saved to: {csv_path}")
        
        # Show first few rows
        print("\nFirst 5 rows of skill_log:")
        for i, row in enumerate(result["skill_log"][:5]):
            print(f"  Shot {row['shot_count']}: "
                  f"MAP skill={row['map_execution_skill']:.4f}, "
                  f"EES={row['ees']:.4f}, "
                  f"MAP rat={row['map_rationality']:.2f}, "
                  f"EPS={row['eps']:.2f}")
        
        # Generate plot
        print("\nGenerating convergence plot...")
        plot_path = plot_intermediate_estimates(csv_path)
        print(f"Plot saved to: {plot_path}")
    else:
        print("\nWARNING: No CSV path in result")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def generate_all_plots():
    """Generate convergence plots for all players with logged data."""
    print("=" * 60)
    print("GENERATING ALL INTERMEDIATE ESTIMATE PLOTS")
    print("=" * 60)
    
    for player_info in LIGHTWEIGHT_TEST_PLAYERS:
        player_id = player_info["player_id"]
        name = player_info["name"]
        
        print(f"\n--- {name} (ID: {player_id}) ---")
        plots = plot_all_intermediate_for_player(player_id)
        if plots:
            print(f"  Generated {len(plots)} plot(s)")
        else:
            print("  No intermediate estimate data found")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


# Set which test to run
# Options:
#   - one_player: Quick test with 1 player, 2 games
#   - three_players: Test with 3 players, 2 games each
#   - one_player_one_season: 1 player, full season (requires DB access)
#   - download_season_test_data: Download full season data
#   - download_lightweight_test_data: Download 3 players × 2 games (lightweight)
#   - run_offline_lightweight_estimation: Run estimation on downloaded data
#   - full_lightweight_pipeline_test: Download + estimate (full pipeline test)
#   - test_intermediate_csv_and_plot: Test CSV export and plotting
#   - generate_all_plots: Generate plots for all players with logged data

TEST_TO_RUN = test_intermediate_csv_and_plot
if __name__ == "__main__":
    TEST_TO_RUN()