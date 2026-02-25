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
from BlackhawksSkillEstimation.player_cache import lookup_player

import sys

# NOTE: At the bottom, set TEST_TO_RUN


def _fmt_log10(val):
    return f"{val:.4f}" if val is not None else "N/A"

def _print_all_estimates(result: dict, prefix: str = "  ") -> None:
    """Helper to print all 4 estimates from a result dict."""
    print(f"{prefix}MAP Execution Skill: {result['execution_skill']:.4f} rad (lower is better)")
    print(f"{prefix}EES:                 {result['ees']:.4f} rad")
    print(f"{prefix}MAP Rationality:     {_fmt_log10(result.get('log10_rationality'))}")
    print(f"{prefix}EPS:                 {_fmt_log10(result.get('log10_eps'))}")
    print(f"{prefix}Shots Used:          {result['num_shots']}")


# =============================================================================
# LIGHTWEIGHT PIPELINE TESTING 
# =============================================================================

# Players and games for lightweight testing
# These player IDs are from the Hawks database (not NHL API IDs)
# Each player uses their top 2 games by shot count (via get_games_for_player)
LIGHTWEIGHT_TEST_PLAYERS = [
    {  # Nathan MacKinnon – 37 shots: 19 + 18
        "player_id": 950160,
        "game_ids": [270247, 44604],
    },
    {  # Cale Makar – 31 shots: 16 + 15
        "player_id": 950184,
        "game_ids": [271408, 44840],
    },
    {  # Kris Letang – 25 shots: 13 + 12
        "player_id": 949352,
        "game_ids": [44905, 42496],
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
        name = lookup_player(player_id)
        
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
        
        try:
            # Load offline data
            df, shot_maps = load_player_data_by_games(
                player_id=player_id,
                tag="2games_test",
                data_dir="Data/Hockey",
            )
            name = lookup_player(player_id)
            print(f"\n--- {name} (ID: {player_id}) ---")
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
                    "log10_rationality": result.get("log10_rationality"),
                    "log10_eps": result.get("log10_eps"),
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
            name = lookup_player(player_id)
            print(f"  ERROR: Data not found. Run download_lightweight_test_data() first.")
            print(f"  {e}")
            results.append({
                "player_id": player_id,
                "name": name,
                "status": "file_not_found",
            })
        except Exception as e:
            name = lookup_player(player_id)
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


def generate_all_logs_plots():
    """Generate convergence plots for all players with logged data."""
    print("=" * 60)
    print("GENERATING ALL INTERMEDIATE ESTIMATE PLOTS")
    print("=" * 60)
    
    for player_info in LIGHTWEIGHT_TEST_PLAYERS:
        player_id = player_info["player_id"]
        name = lookup_player(player_id)
        
        print(f"\n--- {name} (ID: {player_id}) ---")
        plots = plot_all_intermediate_for_player(player_id)
        if plots:
            print(f"  Generated {len(plots)} plot(s)")
        else:
            print("  No intermediate estimate data found")
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def per_season_multi_player_test(pids: list[int] | None):
    """Download (if needed), estimate, and plot for each player x season independently."""
    from pathlib import Path

    players = pids or SEASON_TEST_PLAYERS
    seasons = SEASON_TEST_SEASONS
    data_dir = Path("Data/Hockey")

    print("=" * 60)
    print("PER-SEASON MULTI-PLAYER TEST")
    print(f"Players: {len(players)}  |  Seasons: {seasons}")
    print("=" * 60)

    summary = []

    for pid in players:
        name = lookup_player(pid)
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

        name = lookup_player(pid)

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
                "log10_rationality": data.get("log10_rationality"),
                "log10_eps": data.get("log10_eps"),
                "num_shots": data["num_shots"],
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for s in summary:
        if s["status"] == "success":
            print(f"  {s['name']:20s}  {s['season']}  "
                  f"MAP={s['execution_skill']:.4f}  EES={s['ees']:.4f}  "
                  f"rat={_fmt_log10(s.get('log10_rationality'))}  EPS={_fmt_log10(s.get('log10_eps'))}  "
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
        name = lookup_player(player_id)
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


def plot_info_players_comparison():
    """Generate a convergence comparison plot for all INFO_PLAYERS"""
    from BlackhawksSkillEstimation.plot_intermediate_estimates import plot_comparison
    from pathlib import Path

    for metric in ["execution_skill", "rationality"]:
        for season in SEASON_TEST_SEASONS:

            csvs: list[Path] = []
            labels: list[Path] = []
            for pid in INFO_PLAYERS:
                logs = Path(f"Data/Hockey/player_{pid}/logs")
                path = logs / f"intermediate_estimates_{season}.csv"
                if not path.exists():
                    print(f"warning: no file for player {pid} ({season})")
                    continue
                csvs.append(path)
                labels.append(lookup_player(pid) or str(pid))
            if not csvs:
                raise FileNotFoundError("no intermediate-estimate CSVs found")
            
            plot_comparison(
                csv_paths=csvs,
                labels=labels,
                output_path=f"Data/Hockey/general_plots/all_players_{season}_{metric}.png",
                title=f"{metric.capitalize().replace('_', ' ')} Convergence - All Players ({season})",
                metric=metric,
                estimate_type="expected",
                figsize=(12, 10),
            )


def rank_info_players():
    from BlackhawksSkillEstimation.plot_intermediate_estimates import rank_final_estimates
    for metric in ["execution_skill", "rationality"]:
        for season in SEASON_TEST_SEASONS:
            rank_final_estimates(
                season=season,
                players=INFO_PLAYERS,
                metric=metric
            )


def table_info_players():
    from BlackhawksSkillEstimation.plot_intermediate_estimates import compare_execution_rankings_two_seasons
    compare_execution_rankings_two_seasons(players=INFO_PLAYERS)


# =============================================================================
# PER-SEASON MULTI-PLAYER TEST
# =============================================================================
# Edit these two variables to control which players and seasons to test.

INFO_PLAYERS = [
    950182,
    950169,      
    950181,
    950205,
    949905,
    950148,
    950164,
    950161,
    949992,
    950014,
    949167,
    950185,
    950069,
    949759,
    949266,
    949791,
    950162,
    949962,
    950193,
    950000,
]

SEASON_TEST_PLAYERS = [
    #950160,  # Nathan MacKinnon
    #950184,  # Cale Makar
    949352,   # Kris Letang
]

SEASON_TEST_SEASONS = [20232024, 20242025]

# Set which test to run
# NOTE: Some of these tests that say "all" actually iterate through every player in LIGHTWEIGHT_TEST_PLAYERS
# Options:
#   - download_lightweight_test_data: Download 3 players x 2 games (lightweight)
#   - run_offline_lightweight_estimation: Run estimation on lightweight data
#   - generate_all_logs_plots: Convergence plots only (all players with CSVs)
#   - generate_all_viz: Full visualization suite (angular, rink, convergence)
#   - per_season_multi_player_test: Download (if needed), estimate, and plot per-season
#   - plot_info_players_comparison: Generate a convergence comparison plot for all INFO_PLAYERS
#   - rank_info_players: Generate a bar chart ranking for all INFO_PLAYERS

TEST_TO_RUN = table_info_players
if __name__ == "__main__":
    # TEST_TO_RUN([sys.argv[1]])

    # pids = INFO_PLAYERS[1:]
    # for pid in pids:
    #     save_player_data(player_id=pid, seasons=SEASON_TEST_SEASONS)
    #     print("\nThat was for " + lookup_player(pid))

    TEST_TO_RUN()