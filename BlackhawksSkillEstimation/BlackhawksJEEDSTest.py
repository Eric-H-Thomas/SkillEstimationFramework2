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
from BlackhawksSkillEstimation.blackhawks_plots import (
    plot_player_shots_from_offline,
    plot_all_player_convergence,
)

# NOTE: At the bottom, set TEST_TO_RUN


def _print_all_estimates(result: dict, prefix: str = "  ") -> None:
    """Helper to print all 4 estimates from a result dict."""
    print(f"{prefix}MAP Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
    print(f"{prefix}EES:                 {result['ees']:.4f} rad")
    print(f"{prefix}MAP Rationality:     {result['rationality']:.2f} (higher is better)")
    print(f"{prefix}EPS:                 {result['eps']:.2f}")
    print(f"{prefix}Shots Used:          {result['num_shots']}")


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
                save_intermediate_csv=True,
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


# =============================================================================
# PER-SEASON MULTI-PLAYER TEST
# =============================================================================
# Edit these two variables to control which players and seasons to test.

SEASON_TEST_PLAYERS = [
    #{"player_id": 950160, "name": "Nathan MacKinnon"},
    #{"player_id": 950184, "name": "Cale Makar"},
    {"player_id": 949352, "name": "Kris Letang"},
]

SEASON_TEST_SEASONS = [20232024, 20242025]


def per_season_multi_player_test():
    """Download (if needed), estimate, and plot for each player x season independently."""
    from pathlib import Path

    players = SEASON_TEST_PLAYERS
    seasons = SEASON_TEST_SEASONS
    data_dir = Path("Data/Hockey")

    print("=" * 60)
    print("PER-SEASON MULTI-PLAYER TEST")
    print(f"Players: {len(players)}  |  Seasons: {seasons}")
    print("=" * 60)

    summary = []

    for player in players:
        pid = player["player_id"]
        name = player["name"]
        print(f"\n{'='*60}")
        print(f"{name} (ID: {pid})")
        print(f"{'='*60}")

        # Download any missing season data (skips existing files)
        save_player_data(
            player_id=pid,
            seasons=seasons,
            output_dir=data_dir,
            overwrite=False,
        )

        # Load all seasons together; the DataFrame has a "season" column
        try:
            df, shot_maps = load_player_data(
                player_id=pid,
                seasons=seasons,
                data_dir=data_dir,
            )
        except FileNotFoundError as e:
            print(f"  SKIP (no data): {e}")
            for s in seasons:
                summary.append({"name": name, "season": s, "status": "no_data"})
            continue

        if df.empty:
            print("  SKIP (0 shots)")
            for s in seasons:
                summary.append({"name": name, "season": s, "status": "no_shots"})
            continue

        print(f"  {len(df)} total shots loaded across {len(seasons)} season(s)")

        # per_season=True splits by the "season" column and estimates each independently
        result = estimate_player_skill(
            player_id=pid,
            offline_data=(df, shot_maps),
            per_season=True,
            confirm=False,
            save_intermediate_csv=True,
        )

        per_season_results = result.get("per_season_results", {})

        for season in seasons:
            data = per_season_results.get(season)
            if data is None:
                print(f"\n  Season {season}: no data returned")
                summary.append({"name": name, "season": season, "status": "no_data"})
                continue

            if data.get("status") != "success" and "execution_skill" not in data:
                print(f"\n  Season {season}: {data.get('status', 'unknown')}")
                summary.append({"name": name, "season": season, "status": "failed"})
                continue

            print(f"\n  Season {season}:")
            _print_all_estimates(data, prefix="    ")

            if "csv_path" in data:
                plot_path = plot_intermediate_estimates(data["csv_path"])
                print(f"    CSV:  {data['csv_path']}")
                print(f"    Plot: {plot_path}")

            summary.append({
                "name": name,
                "season": season,
                "status": "success",
                "execution_skill": data["execution_skill"],
                "ees": data["ees"],
                "rationality": data["rationality"],
                "eps": data["eps"],
                "num_shots": data["num_shots"],
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summary:
        if s["status"] == "success":
            print(f"  {s['name']:20s}  {s['season']}  "
                  f"MAP={s['execution_skill']:.4f}  EES={s['ees']:.4f}  "
                  f"rat={s['rationality']:.2f}  EPS={s['eps']:.2f}  "
                  f"({s['num_shots']} shots)")
        else:
            print(f"  {s['name']:20s}  {s['season']}  {s['status']}")


# =============================================================================
# VISUALIZATION TEST
# =============================================================================

def generate_all_viz():
    """Generate all visualizations (angular heatmaps, rink, convergence).

    Uses offline data from LIGHTWEIGHT_TEST_PLAYERS (2-game tag) or
    per-season data if available.  Outputs go to Data/Hockey/plots/.
    Requires download_lightweight_test_data() or download_season_test_data()
    to have been run first.
    """
    from pathlib import Path

    print("=" * 60)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 60)

    data_dir = Path("Data/Hockey")

    for player_info in LIGHTWEIGHT_TEST_PLAYERS:
        player_id = player_info["player_id"]
        name = player_info["name"]
        print(f"\n{'='*60}")
        print(f"{name} (ID: {player_id})")
        print(f"{'='*60}")

        # --- Angular heatmaps + rink scatter ---
        # Try season data first (more shots), fall back to 2-game tag
        try:
            result = plot_player_shots_from_offline(
                player_id=player_id,
                data_dir=data_dir,
                seasons=SEASON_TEST_SEASONS,
                max_shots=10,
            )
            source = "season"
        except (FileNotFoundError, ValueError):
            try:
                result = plot_player_shots_from_offline(
                    player_id=player_id,
                    data_dir=data_dir,
                    tag="2games_test",
                    max_shots=10,
                )
                source = "2games_test"
            except FileNotFoundError:
                print("  No offline data found. Skipping angular/rink plots.")
                result = None
                source = None

        if result:
            print(f"  Source: {source}")
            print(f"  Angular heatmaps: {len(result['angular'])} plot(s)")
            print(f"  Rink diagrams:    {len(result['rink'])} plot(s)")
            for p in result["angular"]:
                print(f"    {p}")
            for p in result["rink"]:
                print(f"    {p}")

        # --- Convergence plots ---
        conv_plots = plot_all_player_convergence(player_id, data_dir=data_dir)
        if conv_plots:
            print(f"  Convergence:      {len(conv_plots)} plot(s)")
            for p in conv_plots:
                print(f"    {p}")
        else:
            print("  No intermediate estimate CSVs found for convergence plots.")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


# Set which test to run
# Options:
#   - download_lightweight_test_data: Download 3 players x 2 games (lightweight)
#   - run_offline_lightweight_estimation: Run estimation on lightweight data
#   - test_intermediate_csv_and_plot: Test CSV export and convergence plotting
#   - generate_all_plots: Convergence plots only (all players with CSVs)
#   - generate_all_viz: Full visualization suite (angular, rink, convergence)
#   - per_season_multi_player_test: Download (if needed), estimate, and plot per-season

TEST_TO_RUN = per_season_multi_player_test
if __name__ == "__main__":
    TEST_TO_RUN()