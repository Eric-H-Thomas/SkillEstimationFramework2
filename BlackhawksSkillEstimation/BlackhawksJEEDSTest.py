from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill
estimate = estimate_player_skill(player_id=950160, game_ids=[44604, 270247])
print(f"Estimated execution skill: {estimate:.3f}")
