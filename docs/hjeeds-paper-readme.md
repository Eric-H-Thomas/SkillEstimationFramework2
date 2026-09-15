
# H-JEEDS paper reproduction guide

This is the end-to-end guide for reproducing the H-JEEDS paper's 1D-Darts, 2D-Darts, and Major League Baseball experiments, publication plots, and model/inference schematic. Generated results are ignored by Git: rerunning the workflows below creates them from scratch. Any pre-existing `HJEEDS/results/publication_bench_3_seeds/` predates the 2026-07-28 non-wrapped-environment correction documented in `docs/hjeeds-source-provenance.md` and does not validate the corrected 1D model. Every new bench run must use a fresh output directory.

## Paper cross-reference

`docs/hjeeds-paper-code-map.md` links manuscript sections to implementation files, and `docs/hjeeds-source-provenance.md` records the scientific corrections behind the current numbers.

## Environment

Python 3.10 and the versions used for the final paper code are pinned in `environment-hjeeds.yml` and `requirements-hjeeds.txt`. The development environment for this repository is `skill-estimation`, which already satisfies these pins; the pin files exist to reconstruct the exact paper stack.

```bash
conda activate skill-estimation
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
```

Run every command below from the repository root. Full 500-seed suites are computationally expensive; Slurm is the expected route for the complete 1D and 2D experiments.

## Relevant layout

- `HJEEDS/`: estimator, experiment, aggregation, and plotting modules.
- `Environments/` and `Estimators/`: environment and legacy estimator dependencies.
- Repository root: cluster submit scripts (`submit_hjeeds_*.sh`), Slurm workers (`run_hjeeds_*.sbatch`), their shared helper `hjeeds_baseball_slurm_common.sh`, and the unified 1D suite runner `run_hjeeds_paper_experiments.py`.
- `scripts/runners/`: the local publication-bench entry point.
- `scripts/`: standalone validation, comparison, and data-preparation utilities; see `scripts/README.md` for the entry-point guide.
- `docs/`: experiment-design and correction-history notes.

## Local publication-bench audit runner

`scripts/runners/run_publication_bench.py` is a sequential local entry point for the 1D suite and 2D benchmark. It is useful for smoke tests and small-seed reproducibility checks; it is not the recommended way to launch the full 500-seed workload. The default components are therefore the two synthetic studies. MLB is deliberately audit-only in this runner: selecting `baseball` validates an already-completed Slurm result bundle and its exact frozen 20-pitcher cohort, but never launches or reports completion of that paper workflow. Use the component-specific Slurm launchers below for every publication run.

```bash
python3 scripts/runners/run_publication_bench.py \
  --components synthetic \
  --num-seeds 3 \
  --output-root HJEEDS/results/publication_bench_3_seeds_corrected
```

For a fast three-seed reproducibility audit against an existing 500-seed 1D result tree:

```bash
python3 scripts/runners/run_publication_bench.py \
  --components synthetic \
  --num-seeds 3 \
  --output-root HJEEDS/results/publication_bench_3_seeds_comparison \
  --reference-one-d-root /path/to/hjeeds_paper_500_seeds
```

This uses 1D seeds 12345--12347 and the original 2D cluster convention, seeds 1000--1002. The comparison checks every cell of every scenario-level `agent_level_results.csv`. It always reports byte-level differences and, separately, tests numerical equivalence with default absolute and relative tolerances of $10^{-7}$ and $10^{-12}$; use `--comparison-atol` and `--comparison-rtol` to change those thresholds. It writes a machine-readable report under `<output-root>/verification/`. Use `--dry-run` to inspect all commands without creating output.

The reference 1D tree must have been generated with the same corrected code. A comparison against the pre-correction 500-seed tree is expected to report scientific differences because intended-target generation changed.

After the MLB Slurm workflow in Section 3 finishes, its completed endpoint can be audited from the same interface:

```bash
python3 scripts/runners/run_publication_bench.py \
  --components baseball \
  --baseball-results-dir HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed \
  --output-root HJEEDS/results/publication_bench_mlb_audit
```

This command runs no MLB inference. A missing, incomplete, modified, or non-paper cohort fails the audit and is recorded as unsupported by the local bench; a passing endpoint is labeled `validated external Slurm paper endpoint`, never `complete`.

## 1. Reproduce all 1D-Darts experiments

The local bench's 1D component covers the baseline and all nine sensitivity families (93 scenarios before aggregation). Except for the deliberately combined compound-stress conditions, each sensitivity study varies one focal factor while holding the others at their defaults. The paper uses base seed 12345 (`default`) and 500 seeds per scenario.

The canonical paper environment is `HJEEDS/darts_environment.py`: rewards alternate inside `[-10, 10]`, actions do not wrap across the board edges, and executions outside the board receive reward zero. `HJEEDS/environment_adapters.py` routes 1D simulation through that same implementation so data generation, likelihood evaluation, and percentage-rationality scoring use identical geometry. The older circular-board utilities under `Environments/Darts/RandomDarts/darts.py` are retained only as legacy dependencies and are not the paper's 1D data-generating model.

```bash
python3 run_hjeeds_paper_experiments.py \
  --mode slurm \
  --seed default \
  --num-seeds 500 \
  --output-root HJEEDS/results/hjeeds_paper_500_seeds \
  --python-bin "$(which python)" \
  --qos normal \
  --time 23:00:00 \
  --mem 16G
```

For a sequential workstation run, replace `--mode slurm` with `--mode local` and omit the Slurm arguments. To inspect all exact commands without launching work:

```bash
python3 run_hjeeds_paper_experiments.py \
  --mode local --seed default --num-seeds 1 \
  --output-root /tmp/hjeeds-paper-dry-run --dry-run
```

The suite writes a design manifest and status CSV at the output root, one directory per experiment family, aggregated CSVs, plots, and a reproducibility archive. Before plotting or packaging, it also verifies the exact 93-scenario/seed/agent design, finite metrics, successful estimators, and converged population fits and writes `publication_result_validation.json`. More detail is in `docs/hjeeds-paper-experiments.md`.

## 2. Reproduce 2D-Darts

The paper result is one 500-seed group, divided into ten Slurm tasks and aggregated after all tasks succeed:

```bash
./submit_hjeeds_2d_cluster_tests.sh \
  --group-count 1 \
  --parts-per-group 10 \
  --seeds-per-group 500 \
  --base-seed-start 1000
```

The canonical paper run is exactly seeds 1000--1499 with 25 agents per seed: five agents in each 5, 10, 25, 100, and 1000-observation bucket. It uses the fixed 2D configuration declared in `HJEEDS/two_d_completion.py` (21-by-21 skill grid, execution-noise grid 8--60, decision-skill grid $10^{-3}$--$10^2$, and target-grid spacing 5.0). The resulting table is `HJEEDS/results/2d_cluster_tests/cluster_0/summary_by_bucket.csv`.

Before submitting replacement workers, the launcher atomically marks `cluster_0/two_d_run_metadata.json` incomplete so the previous aggregate cannot be plotted during a failed or partial rerun. Each worker is launched through one Python `run-paper-part` entry point that constructs, resolves, records, and executes the same authoritative argument list; a changed grid, delta, bucket, seed range, or agent count is rejected before simulation. The worker writes an incomplete-to-complete `two_d_part_metadata.json` that binds its exact seed partition, resolved 25-agent configuration, Python/NumPy/SciPy runtime, corrected-source fingerprint, and agent-CSV hash. Aggregation rejects any missing, incomplete, modified, or pre-fix partition; it invalidates the cluster record again before combining parts and marks it complete only after validating the exact seed-agent grid. The final completion record binds the same source/configuration fingerprint to SHA-256 hashes and byte sizes for the agent CSV, both summary CSVs, and diagnostic plot. The main-paper plotter fails closed on missing, incomplete, modified, stale-code, or noncanonical 2D bundles; therefore rerun all 500 seeds and aggregate them after any listed computational source changes.

## 3. Reproduce the MLB study

The paper uses an untracked 794,830,284-byte processed Statcast artifact whose
standardized features and 2,313-row batter-index mapping are paired with the
tracked `Environments/Baseball/final_OP` neural-network weights. Place it at
`Data/Baseball/StatcastData/ProcessedData-From-GivenFiles.pkl` and verify:

```bash
shasum -a 256 Data/Baseball/StatcastData/ProcessedData-From-GivenFiles.pkl
# 4e1bb7e5412b1efce7f0ec08079164a5a85f3ff89bcabaa02a3ee847201a392c

# Linux equivalent:
sha256sum Data/Baseball/StatcastData/ProcessedData-From-GivenFiles.pkl
```

`HJEEDS/data/baseball_processed_artifact_reference.json` records the complete
schema and model hash, and the runtime fails closed if they do not match. A fresh
Statcast download cannot be substituted: refitting the scaler or rebuilding
batter indices would feed `final_OP` inputs with different semantics. The
downloader and `scripts/prepare_baseball_data.py` are retained only for explicit
non-paper exploration and provenance inspection.

The corrected paper run uses the committed
`HJEEDS/data/baseball_hyperpriors_literature_informed.json`. Its centers were
fixed before viewing corrected MLB outcomes: execution error is centered at
$0.5$ ft using external pitch-location evidence, and decision skill is centered
at $\lambda=100$ using a prior-predictive check on 25 deterministic, broadly
spaced 2021 four-seam-fastball contexts. Broad mean and population-scale widths
and a neutral correlation center make the prior weak. The JSON records those
choices and is cryptographically bound to the corrected execution kernel,
canonical processed pickle, paired model, and deterministic pitch ordering.
Reproduce the decision-skill interpretability check and its machine-readable
record with:

```bash
python -m HJEEDS.baseball_prior_predictive_check
```

This writes `HJEEDS/data/baseball_prior_predictive_check.json`, including the
exact 25 context rows and normalized expected utilities for every candidate
$\lambda$ (zero denotes uniform target selection and one denotes optimal target
selection). This audit record also corrects the older metric label in the
hash-bound hyperprior JSON: the approximately 80\% quantity is normalized
expected utility, not the literal probability assigned to one optimal grid
target. The numerical prior and scientific provenance fields are unchanged.
The renderer accepts either recorded hash for this file and independently checks
the parsed hyperprior values.

The superseded `baseball-2021-ff` calibration preset remains committed only for
provenance and intentionally fails closed. The 528-agent calibration launcher
is retained as an optional exploratory analysis, but it is neither required nor
used by the publication workflow.

Submit the exact 2021 four-seam-fastball, high/low walk-IP-proxy convergence study:

```bash
./submit_hjeeds_baseball_convergence_paper_bbip.sh
```

This uses at least 100 eligible pitches per pitcher, checkpoints
5/10/25/50/100, at most 100 reference pitches, and the
`baseball-literature-informed` hyperprior preset. Results are written to
`HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed/`.

The grouping variable is a processed-data walks-per-inning proxy, not official
regular-season BB/IP. Its numerator counts every retained 2021 Statcast row
with `events == "walk"` in the canonical artifact (March 15--November 2,
including spring-training and postseason rows); its denominator is official
2021 season innings pitched from the tracked bundled artifact. This exact
definition and the selected cohort are embedded in the run metadata.

## 4. Render publication figures

The publication renderers read the aggregated CSVs directly:

```bash
python3 -m HJEEDS.plot_main_paper_baseline \
  --summary-by-bucket-csv HJEEDS/results/hjeeds_paper_500_seeds/baseline/summary_by_bucket.csv

python3 -m HJEEDS.plot_hyperprior_robustness_appendix --single-column
python3 -m HJEEDS.plot_population_shape_distributions
python3 -m HJEEDS.plot_population_shape_robustness \
  --single-column-output-stem HJEEDS/results/hjeeds_paper_500_seeds/population_shape/population_shape_lowest_bucket
python3 -m HJEEDS.plot_population_shape_robustness
python3 -m HJEEDS.plot_remaining_sensitivity_robustness \
  --results-root HJEEDS/results/hjeeds_paper_500_seeds \
  --experiments decision_model,compound_stress --single-column
python3 -m HJEEDS.plot_agents_per_bucket_sensitivity_panels
python3 -m HJEEDS.plot_outlier_sensitivity
python3 -m HJEEDS.plot_anchor_availability_robustness
python3 -m HJEEDS.plot_remaining_sensitivity_robustness \
  --results-root HJEEDS/results/hjeeds_paper_500_seeds \
  --experiments decision_model,true_correlation,grid_resolution,compound_stress \
  --group-by-count-bucket
python3 -m HJEEDS.plot_one_d_darts_reward_function

python3 -m HJEEDS.plot_main_paper_higher_dimensional \
  --two-d-summary-csv HJEEDS/results/2d_cluster_tests/cluster_0/summary_by_bucket.csv \
  --baseball-results-dir HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed \
  --output-dir figures
```

The baseline renderer also recomputes every plotted mean and confidence interval from the adjacent `agent_level_results.csv` (or the path supplied with `--agent-level-csv`) and rejects partial, duplicate, non-finite, noncanonical, or stale summaries. Its defaults require the exact 500-seed paper range beginning at seed 12345 and five agents in each observation bucket; `--expected-first-seed` and `--expected-num-seeds` are available only for explicitly labeled smaller audits.

All plotted numerical values are also written to CSV or read from the experiment summary CSVs, so headline claims can be audited independently of raster images.

Compile the combined notation/inference schematic with a TeX installation that includes TikZ:

```bash
cd HJEEDS/schematics
pdflatex -interaction=nonstopmode -halt-on-error hjeeds-model-overview.tex
```

## Verification

After intentional source changes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q
bash -n *.sh *.sbatch
```

The complete 500-seed simulations and the multi-year Statcast download are intentionally not run as part of the fast verification step.

Related documentation:

- `docs/hjeeds-source-provenance.md`: scientific correction history
- `docs/hjeeds-paper-code-map.md`: manuscript-to-code cross-reference
- `docs/hjeeds-paper-experiments.md`: unified 1D suite design and scenario table
- `docs/hjeeds.md`: per-experiment commands outside the paper workflow
