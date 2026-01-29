from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill, estimate_multiple_players

# Single player estimation
estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
print(f"JEEDS MAP execution skill: {estimates['execution_skill']:.4f} rad (lower is better)")
print(f"JEEDS MAP rationality:     {estimates['rationality']:.2f} (higher is better)")
'''
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
'''