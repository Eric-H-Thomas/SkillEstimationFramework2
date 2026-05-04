# MAXG Benchmark (Maximum Adjusted eXpected Goals)

MAXG (pronounced "Max-G") is the maximum value of the execution-adjusted expected goals map. It keeps the familiar xG framing while making the "take the max" operation explicit.

This guide describes how to generate the 1000-shot benchmark and compute MAXG scores.

## 1) Generate benchmark shots

The benchmark tag is required for all steps and is used to name the output files.

```bash
python -m BlackhawksSkillEstimation.benchmark_generation \
  --seasons 20212022 20222023 20232024 20242025 \
  --shot-group wristshot_snapshot \
  --distance-bins 10,20,30,40,200 \
  --benchmark-size 1000 \
  --max-player-fraction 0.03 \
  --seed 42 \
  --tag WS_v1 \
  --output-dir Data/Hockey/benchmarks
```

Outputs:
- `Data/Hockey/benchmarks/benchmark_shots_WS_v1.parquet`
- `Data/Hockey/benchmarks/benchmark_shot_maps_WS_v1.npz`
- `Data/Hockey/benchmarks/benchmark_shots_WS_v1.provenance.json`

## 2) Evaluate MAXG scores

The benchmark tag must match the tag used when generating the benchmark.

```bash
python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag WS_v1 \
  --season-tag 20232024 \
  --shot-group wristshot_snapshot \
  --data-dir Data/Hockey \
  --pids-file forwards23-25.txt \
  --output-dir Data/Hockey/benchmarks/results
```

Outputs:
- `Data/Hockey/benchmarks/results/maxg_results_WS_v1_20232024_wristshot_snapshot.csv`
- `Data/Hockey/benchmarks/results/maxg_results_WS_v1_20232024_wristshot_snapshot_maxg_over_xskill.png`

## 3) Debug plots (optional)

```bash
python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag WS_v1 \
  --season-tag 20232024 \
  --shot-group wristshot_snapshot \
  --debug only \
  --debug-shots 5 \
  --debug-players 5
```

Plots are saved under:
- `Data/Hockey/benchmarks/plots/WS_v1_20232024_wristshot_snapshot/`

## 4) Smoke test (optional)

```bash
python -m BlackhawksSkillEstimation.maxg_evaluator \
  --benchmark-tag WS_v1 \
  --season-tag 20232024 \
  --shot-group wristshot_snapshot \
  --smoke-test \
  --smoke-shots 10 \
  --smoke-players 3
```
