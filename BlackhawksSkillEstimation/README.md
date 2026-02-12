# Blackhawks JEEDS Skill Estimation

This module connects the Blackhawks Snowflake tables to the JEEDS execution-skill
estimator. It focuses on a direct, reproducible path from raw SQL rows to JEEDS
MAP estimates for both **execution skill** and **rationality** across a set of games.

## What the code does

1. **Pull shot data** – `query_player_game_info` retrieves shot-level details
   (coordinates, post-shot xG, flags) for the requested player and game IDs.
2. **Fetch Blackhawks analytics** – `get_game_shot_maps` retrieves precomputed
   reward surfaces (post-shot xG probability grids) for each shot in the game,
   including goal-line coordinates and execution noise covariances.
3. **Prepare JEEDS inputs** – `transform_shots_for_jeeds` converts the shot data
   and Blackhawks reward surfaces into the minimal structures JEEDS expects for
   the hockey domain: angular coordinate grids, per-skill covariance matrices,
   and per-shot expected value surfaces derived from the Blackhawks analytics.
4. **Estimate skills** – `estimate_player_skill` feeds every observed shot into
   the production JEEDS estimator (`JointMethodQRE`) and returns both MAP estimates:
   - **Execution skill (xskill)**: Mechanical accuracy in radians. **Lower is better** 
     (tight shot clustering). Range: [0.004, π/4].
   - **Rationality (pskill)**: Decision-making optimality. **Higher is better**
     (aims at high-value targets). **EXPERIMENTAL** - see interpretation notes below.

## Key modeling choices

- **Reward surface from Blackhawks analytics** – The EV surface for each shot comes
  directly from the precomputed `post_shot_xg_value_maps`, which incorporate
  detailed models of shooting position, angle, goalie positioning, and other
  factors. A corresponding (y, z) grid spanning ±30 meters in y and ±7.5 meters
  in z from the goal line provides the coordinate system.
- **Skill-to-variance mapping** – Candidate execution skills are standard 
  deviations in radians: larger skill values expand the covariance (wider 
  execution spread, more misses), smaller skills shrink the covariance (tighter 
  execution, fewer misses). **Lower execution skill is better.** This matches the 
  production hockey.py convention. The EV smoothing applied during the 
  transformation mirrors this: higher skills blur the reward surface more widely 
  (accounting for greater shot error), while lower skills keep probability mass 
  concentrated near the intended target.
- **Rationality interpretation (EXPERIMENTAL)** – The rationality estimate measures
  how optimally a player selects aim points given the expected value surface.
  Higher rationality means nearly always choosing the highest-value shot location.
  However, this metric may not fully account for real-world constraints like
  defender positioning, time pressure, or play development. Use with caution.
- **JEEDS compatibility** – The helper `SimpleHockeySpaces` mirrors the fields
  JEEDS reads for the hockey domain (`possibleTargets`, `delta`, `allCovs`, and
  `get_key`), allowing the official estimator to run unmodified.

## Running an estimation

Set the Snowflake environment variables required by `BlackhawksAPI` (see
`BlackhawksAPI/test.py` for the list). Then run:

```bash
python -m BlackhawksSkillEstimation.BlackhawksJEEDS \
  950160 \
  44604 270247 \
  --candidate-skills 0.004 0.1 0.2 0.3 0.4 0.5 0.6 0.785 \
  --num-planning-skills 25 \
  --data-dir Data/Hockey \
  --rng-seed 0
```

**Output interpretation:**
- **Execution skill**: Value in radians. **Lower = better shooter** (0.004 = elite, 0.785 = poor)
- **Rationality**: Dimensionless optimality measure. **Higher = better decision-maker**
  (EXPERIMENTAL - see notes above)

The command prints the MAP execution-skill estimate. All per-player outputs
(timing logs, intermediate estimate CSVs, plots) are stored under
`Data/Hockey/player_{id}/` in the following subdirectories:

```
player_{id}/
  data/    # .pkl files (shots + shot_maps)
  logs/    # intermediate estimate CSVs + convergence PNGs
  plots/   # angular heatmap and rink visualization PNGs
  times/   # JT-QRE-Times-* estimator timing logs
```

## API highlights

- `transform_shots_for_jeeds(df, shot_maps, candidate_skills)` – Build
  JEEDS-compatible inputs from a `pandas` DataFrame and Blackhawks shot maps
  dictionary.
- `estimate_player_skill(player_id, game_ids, ...)` – End-to-end helper that
  fetches shot data and Blackhawks reward surfaces, performs the transformation,
  runs JEEDS, and returns the MAP estimate.

Use these functions directly in notebooks or scripts when you already have a
`DataFrame` of shot rows or want to integrate the estimator into a larger
pipeline.
