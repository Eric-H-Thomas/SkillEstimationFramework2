# Blackhawks JEEDS Skill Estimation

This module connects the Blackhawks Snowflake tables to the JEEDS execution-skill
estimator. It focuses on a direct, reproducible path from raw SQL rows to a
single JEEDS MAP estimate for a player's execution skill across a set of games.

## What the code does

1. **Pull shot data** – `query_player_game_info` retrieves shot-level details
   (coordinates, post-shot xG, flags) for the requested player and game IDs.
2. **Prepare JEEDS inputs** – `transform_shots_for_jeeds` builds the minimal
   structures that the JEEDS hockey domain expects: a (y, z) grid of possible
   targets, per-skill covariance matrices, and per-shot expected value (EV)
   surfaces.
3. **Estimate skill** – `estimate_player_skill` feeds every observed shot into
   the production JEEDS estimator (`JointMethodQRE`) and returns the final MAP
   execution-skill value.

## Key modeling choices

- **Reward surface approximation** – The EV surface for each shot starts with a
  Gaussian bump centered on the observed target `(location_y, location_z)` and
  scaled by the shot's post-shot xG probability. A small grid step (default
  `0.25` meters) and sigma (`0.5` meters) keep the surface smooth without
  requiring the full simulation stack used by historical experiments.
- **Skill-to-variance mapping** – Candidate execution skills are interpreted as
  inverse noise levels: larger skill values shrink the covariance used when
  evaluating the likelihood of the executed target. The smoothing applied to the
  EV surface mirrors this intuition by concentrating mass around the intended
  target for higher skills and spreading it out for lower skills.
- **JEEDS compatibility** – The helper `SimpleHockeySpaces` mirrors the fields
  JEEDS reads for the hockey domain (`possibleTargets`, `delta`, `allCovs`, and
  `getKey`), allowing the official estimator to run unmodified.

## Running an estimation

Set the Snowflake environment variables required by `BlackhawksAPI` (see
`BlackhawksAPI/test.py` for the list). Then run:

```bash
python -m BlackhawksSkillEstimation.BlackhawksJEEDS \
  950160 \
  44604 270247 \
  --candidate-skills 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 \
  --num-planning-skills 25 \
  --grid-step 0.25 \
  --base-sigma 0.5 \
  --results-folder Experiments/blackhawks-jeeds \
  --rng-seed 0
```

The command prints the MAP execution-skill estimate and creates any missing
`Experiments/blackhawks-jeeds/times/estimators` folders that JEEDS expects when
writing timing metadata.

## API highlights

- `transform_shots_for_jeeds(df, candidate_skills, grid_step=0.25,
  base_sigma=0.5)` – Build JEEDS-compatible inputs from a `pandas` DataFrame.
- `estimate_player_skill(player_id, game_ids, ...)` – End-to-end helper that
  fetches data, performs the transformation, runs JEEDS, and returns the MAP
  estimate.

Use these functions directly in notebooks or scripts when you already have a
`DataFrame` of shot rows or want to integrate the estimator into a larger
pipeline.
