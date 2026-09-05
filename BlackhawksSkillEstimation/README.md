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
   - **Rationality (pskill)**: Raw λ from the JEEDS grid. **Higher is better**
     (more weight on high-EV targets). Grid endpoints follow EV normalization (see
     below). **EXPERIMENTAL** - see interpretation notes below.

## Key modeling choices

- **Reward surface from Blackhawks analytics** – The EV surface for each shot comes
  directly from the precomputed `expected_goal_values_post_shot_net_grid` maps, which
  incorporate detailed models of shooting position, angle, goalie positioning, and
  other factors. A corresponding (y, z) grid spanning Y ∈ [-5, 5] and Z ∈ [0, 6]
  from the goal line provides the coordinate system.
- **Skill-to-variance mapping** – Candidate execution skills are standard 
  deviations in radians: larger skill values expand the covariance (wider 
  execution spread, more misses), smaller skills shrink the covariance (tighter 
  execution, fewer misses). **Lower execution skill is better.** This matches the 
  production hockey.py convention. The EV smoothing applied during the 
  transformation mirrors this: higher skills blur the reward surface more widely 
  (accounting for greater shot error), while lower skills keep probability mass 
  concentrated near the intended target.
- **Rationality interpretation (EXPERIMENTAL)** – Raw λ (and `log10_eps`) summarize the
  posterior over decision skill on a fixed grid. They should be interpreted cautiously,
  especially when comparing across different xG surfaces or data regimes.
- **EV / rationality normalization (current test; model change)** – By default
  (`BH_EV_NORMALIZE=1`), each skill-blurred EV surface is divided by its
  peak-above-average (`max − mean`) before the QRE softmax. That makes λ
  dimensionless — preference per unit of extra advantage — instead of absorbing
  the raw xG scale, which otherwise shrinks as execution skill worsens (more blur).
  The λ grid then defaults to log10 in `[-1, 3]` (`λ ∈ [0.1, 1000]`) rather than
  `[0, 4]` on raw xG units. JEEDS and MCSE share this switch so their λ values
  stay comparable. Set `BH_EV_NORMALIZE=0` to recover the old units; estimates
  from the two conventions are not comparable, and this is still under test.
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
- **Rationality**: Raw λ on the JEEDS grid (log10 `[-1, 3]` with EV normalization on).
  **Higher = more weight on high-EV targets** (EXPERIMENTAL - see notes above)

The command prints the MAP execution-skill estimate. All per-player outputs
(timing logs, intermediate estimate CSVs, plots) are stored under
`Data/Hockey/player_{id}/` in the following subdirectories:

```
player_{id}/
  data/    # parquet + npz files (shots DataFrames + shot_maps dicts)
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

## Entrypoints

Run from the repo root with `PYTHONPATH=.` (or `python -m`). JEEDS/MCSE runners
stay at the package root; post-run scripts live under `analysis/`.

**Estimation and cluster runners**

| Module | Role |
|--------|------|
| `python -m BlackhawksSkillEstimation.BlackhawksJEEDS` | Single-player JEEDS |
| `python -m BlackhawksSkillEstimation.run_blackhawks_config` | JEEDS from a job JSON |
| `python -m BlackhawksSkillEstimation.BlackhawksMCSE` | Single-player MCSE |
| `python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config` | MCSE from a job JSON |
| `python -m BlackhawksSkillEstimation.maxg_evaluator` | MAXG evaluation over JEEDS/MCSE CSVs |
| `python -m BlackhawksSkillEstimation.summarize_mcse_runs` | Aggregate cluster MCSE logs |
| `python -m BlackhawksSkillEstimation.build_player_subsample_config` | Build a subsample-stability job JSON |
| `python -m BlackhawksSkillEstimation.run_player_subsample_config` | Run subsample-stability jobs from a job JSON |

**Post-run analysis** (`BlackhawksSkillEstimation/analysis/`)

| Module | Role |
|--------|------|
| `python -m BlackhawksSkillEstimation.analysis.analyze_mcse_run` | Tidy CSVs, cross-season stability, diagnostic plots |
| `python -m BlackhawksSkillEstimation.analysis.diagnose_estimator_signal` | Shot-volume leakage, JEEDS vs MCSE, internal convergence |
| `python -m BlackhawksSkillEstimation.analysis.diagnose_rationality_scale` | Rationality cap / scaling diagnostics |
| `python -m BlackhawksSkillEstimation.analysis.plot_rationality_vs_xskill` | Rationality vs execution-skill scatter |
| `python -m BlackhawksSkillEstimation.analysis.stability_plots_from_txt` | Cross-season stability plots from a player list |
| `python -m BlackhawksSkillEstimation.analysis.plot_player_subsample_stability` | Subsample-stability plots for one player |
| `python -m BlackhawksSkillEstimation.analysis.generate_bhawks_report` | Ranking tables / BYU-style report from a PID file |

## Player subsample stability

Season-to-season execution and decision estimates for the same player move more than
real skill plausibly does. This experiment separates sampling noise from genuine
change: hold one player fixed, pool every cached season, and re-estimate on many
random N-shot subsets. If 200-shot redraws already span as much as the season
estimates do, the season-to-season instability is mostly noise.

```bash
conda activate skill-estimation

# 1. Draw the samples (metadata scan only, no estimator runs)
python -m BlackhawksSkillEstimation.build_player_subsample_config \
  --player-id 950160 --all-seasons \
  --n-shots 100 200 400 --num-seeds 50 \
  --output Data/Hockey/jobs/player_subsample_950160.json --dry-run

# 2. Check the expanded job list
python -m BlackhawksSkillEstimation.run_player_subsample_config \
  --config Data/Hockey/jobs/player_subsample_950160.json --dry-run

# 3. Submit (bootstrap reads the config and sizes the array)
sbatch run_player_subsample_config.sbatch Data/Hockey/jobs/player_subsample_950160.json

# 4. Plot (local, post-hoc; a partial array still plots)
python -m BlackhawksSkillEstimation.analysis.plot_player_subsample_stability \
  --config Data/Hockey/jobs/player_subsample_950160.json
```

Defaults for 950160: **all five cached seasons** (1,937 `wristshot_snapshot` shots),
N ∈ {100, 200, 400} × 50 seeds, plus one full-sample baseline, run through **both**
JEEDS and MCSE — 302 jobs. `--smoke` shrinks the grids and seed count for a local check.

Notes:

- **Every season is pooled, including 20252026.** The per-season cluster configs
  `jeeds_forwards_all_seasons.json` and `mcse_forwards_per_season.json` cover the same
  five seasons, so the season dots they produce line up with the subsample pool.
- **Grids must match production.** Defaults are JEEDS 250×250 and MCSE 500 particles,
  matching the existing per-season cluster configs. The local CLI defaults (50×100,
  1000 particles) produce estimates that cannot be compared to the season dots, which
  is why `--smoke` output is labelled as such.
- **Draws happen once, at build time.** The config stores event ids, so JEEDS and MCSE
  run on byte-identical shot sets and reruns are reproducible. Each `(N, seed)` pair
  has its own RNG stream, so adding N values or seeds later does not disturb draws
  that already ran.
- **Outputs are isolated** under
  `Data/Hockey/experiments/player_subsample_stability/<run_name>/`. Nothing is written
  into the per-season `logs/` tree that `stability_plots_from_txt` and
  `analyze_mcse_run` glob over.
- **Reruns resume.** Successful result JSONs are skipped; failed or unreadable
  ones are retried. Pass `--overwrite` to force a successful job to run again.
- **MCSE on the full baseline is the long pole.** Cost scales with shots × particles,
  so the 1,937-shot MCSE baseline can approach the 24h default wall. If it times out,
  rebuild with `--sbatch-time 48:00:00`.
- Pooled draws mix seasons. If seasons are not exchangeable, subsample spread is not a
  pure N-shot noise floor — the `season_mix_*` figures exist to check exactly that.
