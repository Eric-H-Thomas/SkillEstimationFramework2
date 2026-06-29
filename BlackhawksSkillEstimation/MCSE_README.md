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

## Quick start (offline cached data)

```bash
conda activate skill-estimation
python -m BlackhawksSkillEstimation.BlackhawksMCSE \
  950160 \
  --seasons 20232024 \
  --shot-group wristshot_snapshot \
  --data-dir Data/Hockey \
  --num-particles 200 \
  --save-intermediate-csv
```

## Config runner

```bash
python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_smoke.json --dry-run

python -m BlackhawksSkillEstimation.run_blackhawks_mcse_config \
  --config Data/Hockey/jobs/mcse_smoke.json --job-index 0
```

Cluster: `sbatch run_blackhawks_mcse_config.sbatch Data/Hockey/jobs/mcse_smoke.json`

## MAXG comparison with JEEDS

MAXG uses the EES profile `(ees_y, ees_z, rho_ees)` with anisotropic convolution (see `maxg_evaluator.py`).

```bash
python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag WS_v1 \
  --season-tag 20232024 \
  --shot-group wristshot_snapshot \
  --estimator mcse \
  --smoke-test
```

Compare **MAXG-to-MAXG** across estimators, not raw scalar JEEDS xskill vs MCSE `(x_y, x_z)`.

## Defaults

- Particles: 1000 (smoke configs use 200)
- Ranges: x ∈ [0.004, π/4] per axis, ρ ∈ [-0.75, 0.75], log₁₀λ ∈ [-3, 1.6]
- Resample: 90% with NEFF gate, systematic resampling
