# Blackhawks MCSE Skill Estimation

MCSE (paper name for `QREMethod_Multi_Particles` / PFE) estimates a **2D execution-skill profile**
(direction y, elevation z) plus correlation `rho` and rationality `lambda` from Blackhawks shot data.

## Outputs

| Artifact | Purpose |
|----------|---------|
| `logs/mcse/<shot_group>/intermediate_estimates_*.csv` | Per-shot 2D skill traces (visualization) |
| `maxg_ees` in result dict | **Primary scalar** for reporting and JEEDS comparison |
| 2D fields `ees_y`, `ees_z`, `rho_ees` | Full MCSE profile |

Lower execution skill (rad) = better aim precision.

## Parameter bounds (JEEDS-aligned)

MCSE defaults match the Blackhawks JEEDS grids:

| Parameter | Range (current) |
|-----------|-----------------|
| Execution skill (y, z) | `[0.004, 0.25]` rad |
| Correlation rho | `[-0.75, 0.75]` |
| Rationality lambda | `log10` in `[0, 4]` (same as JEEDS `logspace(0, 4)`) |

**Historical MCSE bounds** (used in early smoke runs before 2026-07 alignment):

| Parameter | Range (legacy) |
|-----------|----------------|
| Execution skill (y, z) | `[0.004, π/4]` rad |
| Rationality lambda | `log10` in `[-3, 1.6]` (~0.001–40; `joint_pfe.py` hardcoded this grid) |

Constants `LEGACY_MCSE_RANGES` in `BlackhawksMCSE.py` preserve the old endpoints for
sensitivity reruns (`"ranges": LEGACY_MCSE_RANGES` in a job config JSON).

Particles default to **1000**. The PFE lambda particle grid is derived from the
`ranges` log10 endpoints in `joint_pfe.py`.

## Data source: legacy vs new xG

MCSE has **no `maps_source` switch**. It reads whatever is cached under `data_root`:

| `data_root` | xG maps |
|-------------|---------|
| `Data/Hockey` | **Legacy** (default download) |
| `Data/Hockey_xg_new` | **New xG** (separate cache tree) |

Use the submit scripts above to run legacy, new xG, or both. Each array writes to its
own tree under `players/player_*/logs/mcse/`.

## Quick start (offline cached data)

```bash
conda activate skill-estimation
python -m BlackhawksSkillEstimation.BlackhawksMCSE \
  950160 \
  --seasons 20232024 \
  --shot-group wristshot_snapshot \
  --data-dir Data/Hockey \
  --num-particles 1000 \
  --save-intermediate-csv
```

## Config runner

```bash
python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_smoke.json --dry-run

python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_smoke.json --job-index 0
```

Cluster worker: `sbatch run_blackhawks_mcse_config.sbatch Data/Hockey/jobs/mcse_smoke.json`

## League-wide cluster sweep

Build a config from cached local data (metadata scan only; no estimator runs):

```bash
# Preview job counts (legacy + new-xG eligibility)
python -m BlackhawksSkillEstimation.build_mcse_cluster_config \
  --player-file Data/Hockey/forwards23-25.txt \
  --all-seasons \
  --min-shots-per-job 100 \
  --also-write-xgnew \
  --output Data/Hockey/jobs/mcse_forwards_per_season.json \
  --dry-run

# Write legacy + derived new-xG configs
python -m BlackhawksSkillEstimation.build_mcse_cluster_config \
  --player-file Data/Hockey/forwards23-25.txt \
  --all-seasons \
  --min-shots-per-job 100 \
  --num-particles 500 \
  --sbatch-time 48:00:00 \
  --sbatch-mem 32G \
  --also-write-xgnew \
  --output Data/Hockey/jobs/mcse_forwards_per_season.json
```

This writes:

| File | `data_root` |
|------|-------------|
| `mcse_forwards_per_season.json` | `Data/Hockey` (legacy) |
| `mcse_forwards_per_season.xgnew.json` | `Data/Hockey_xg_new` (new xG, eligibility refreshed) |

Cluster defaults: **500 particles**, **48h**, **32G**, **100 concurrent** per array.
Production MCSE runs set `retain_history=False` (no growing particle-history lists) and clear
per-shot PDF/EV caches after each observation.

### Cluster submit

**Legacy only:**

```bash
sbatch run_blackhawks_mcse_config.sbatch Data/Hockey/jobs/mcse_forwards_per_season.json
```

**New xG only** (derives `.xgnew.json` on submit, then runs eligible jobs):

```bash
sbatch run_blackhawks_mcse_config_new_xg_root.sbatch Data/Hockey/jobs/mcse_forwards_per_season.json
```

**Both legacy and new xG** (two independent arrays):

```bash
sbatch run_blackhawks_mcse_config_both.sbatch Data/Hockey/jobs/mcse_forwards_per_season.json
```

Dry-run workers before submitting:

```bash
python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_forwards_per_season.json --dry-run

python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_forwards_per_season.xgnew.json --dry-run
```

Options:

- `--all-cached-players` instead of `--player-file` to sweep every player under a data root
- `--split-mode all_selected_seasons_together` for one job per player across all selected seasons
- `--num-particles 1000` (default), `--min-shots-per-job 100` (matches JEEDS cluster norm)
- `--generate-convergence-png` to enable PNGs (off by default on cluster configs for speed)

After jobs finish, aggregate final EES/rationality rows (run per data root):

```bash
python -m BlackhawksSkillEstimation.summarize_mcse_runs \
  --data-root Data/Hockey \
  --output Data/Hockey/jobs/mcse_summary_legacy.csv

python -m BlackhawksSkillEstimation.summarize_mcse_runs \
  --data-root Data/Hockey_xg_new \
  --output Data/Hockey/jobs/mcse_summary_xgnew.csv
```

## MAXG comparison with JEEDS

MAXG uses the EES profile `(ees_y, ees_z, rho_ees)` with anisotropic convolution (see `maxg_evaluator.py`).

```bash
python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag wristshot-snapshot_v1 \
  --season-tag 20232024 \
  --shot-group wristshot_snapshot \
  --estimator mcse \
  --data-dir Data/Hockey
```

Compare **MAXG-to-MAXG** across estimators, not raw scalar JEEDS xskill vs MCSE `(x_y, x_z)`.

## Defaults

- Particles: 1000 locally / smoke; **500** for league cluster configs
- Ranges: x ∈ [0.004, 0.25] per axis, ρ ∈ [-0.75, 0.75], log₁₀λ ∈ [0, 4]
- Legacy ranges (pre-alignment): x ∈ [0.004, π/4], log₁₀λ ∈ [-3, 1.6] — see `LEGACY_MCSE_RANGES`
- Cluster resources: 48h / 32G / 100 concurrent per array
- Data: cluster config uses legacy `Data/Hockey` only (use `*_both.sbatch` or new-xG wrapper for both)
- Resample: 90% with NEFF gate, systematic resampling
- Memory: `retain_history=False` in Blackhawks MCSE; per-shot PDF/EV caches cleared after each observation
