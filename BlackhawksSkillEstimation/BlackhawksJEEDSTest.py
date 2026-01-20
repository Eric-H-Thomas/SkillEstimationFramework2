from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill
estimates = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
# print(f"Estimated execution skill: {estimates['execution_skill']:.4f}")
print(f"JEEDS MAP execution skill: {estimates['execution_skill']:.4f} rad (lower is better)")
print(f"JEEDS MAP rationality:     {estimates['rationality']:.2f} (higher is better)")
