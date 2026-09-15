
# H-JEEDS paper correction history

This is the consolidated changelog of the scientific and infrastructure
corrections made to the H-JEEDS workflows while preparing the paper. It records
what changed, why, and which historical result artifacts each correction
invalidates. Two portability changes are worth calling out because they affect
how the entry points resolve paths:

- `run_hjeeds_2d_cluster_tests.sbatch` resolves the repository root from the environment or its own location instead of a researcher-specific home-directory path, and writes into the path used by the paper plotter.
- `HJEEDS/plot_main_paper_higher_dimensional.py` no longer reads any cross-project global. The renderer corrections documented below route every panel through its CSV-loaded arguments.

`scripts/prepare_baseball_data.py` is a compact version of the historical data-preparation shape used by `BaseballSpaces.getAllData`. It is retained for explicit non-paper exploration only: refitting a scaler and regenerating batter indices from a fresh download does not reproduce the training-time inputs paired with `final_OP`.

`scripts/runners/run_publication_bench.py` and `scripts/compare_seeded_results.py`
provide reproducibility infrastructure. They orchestrate the retained scientific
runners and compare selected seed-level CSV rows; they do not alter the
estimators or experiment definitions.

Post-extraction correction on 2026-07-28:

- `HJEEDS/environment_adapters.py` was corrected so the 1D simulation path uses the paper's non-wrapped environment in `HJEEDS/darts_environment.py`, matching the likelihood and percentage-rationality paths. The previous adapter called a legacy circular-board expected-value helper and supplied a signed wrapped difference to the deceptive decision model.
- `tests/test_one_d_environment_consistency.py` locks down the shared non-wrapped expected-value curve, execution model, and linear action distance.
- This correction changes simulated 1D actions. Historical 1D result CSVs and their derived plots must therefore be regenerated. That specific 1D adapter correction did not alter the 2D-Darts or MLB pipelines; separate 2D and MLB corrections are documented below.

Publication-bench simplification on 2026-07-28:

- The agents-per-bucket study now varies only population size under the default hyperpriors.
- Population-shape, decision-model, and true-correlation studies now run only at the default five agents per bucket.
- The compound-stress study retains its three deliberate combinations of hyperprior, population-shape, decision-model, and correlation stressors, but fixes population size at five agents per bucket.
- The anchor-availability study now holds the population at 50 agents: 25 fixed one-sample evaluation agents plus 25 context agents, with only the number of 25-sample context anchors varying. This removes the previous population-size confound.
- The default 1D suite therefore contains 93 rather than 163 scenarios.

Non-baseball correctness hardening on 2026-07-28:

- The 1D expected-reward curve now integrates every piecewise-constant reward interval with Normal CDF differences. This removes the endpoint and truncated-tail error from the former sampled convolution while preserving the same legal target grid.
- The empirical-Bayes fit now removes arbitrary per-agent log-likelihood offsets, rejects invalid grids and numerical proposals, retains the initialization if optimization would reduce the objective, and records convergence/objective diagnostics in every agent-level row. The unified runner refuses to plot or package a result tree unless all 93 scenarios, all expected seeds and agents, finite publication metrics, successful estimators, and converged population fits are present.
- Rational-policy simulations now record their behavioral ground truth as exactly 100% rationality. Softmax, flip, and deceptive simulations retain the existing lambda-based truth conversion.
- The compound-stress correlation settings are now genuinely misspecified: default uses prior/truth $(-0.5,-0.5)$, moderate uses $(-0.5,0)$, and strong uses $(-0.9,+0.9)$.
- Anchor-availability publication summaries now evaluate the same first 25 one-observation agents in every condition. Separate all-agent summaries retain the full 50-agent population for diagnostics.
- The 2D environment keys cached geometry by resolution. Cluster aggregation requires exact seed/agent coverage, successful finite outputs, corrected-source fingerprints, and complete hashed worker metadata; its Slurm launcher uses contiguous seed ranges and validates the selected Python environment.
- Remaining sensitivity plots can now be pointed at an explicit result root, preventing a corrected run from being silently mixed with stale default paths.

2D execution-noise correction:

- `HJEEDS/environment_adapters.py` now draws each 2D-Darts execution error from the generator supplied to the simulator. The former path rebuilt `Environments/Darts/RandomDarts/two_d_darts.draw_noise_sample` for every observation; that helper seeded a frozen `multivariate_normal` from `rng.bit_generator._seed_seq.entropy`, which is fixed for a generator's lifetime. Consequently every throw within a seed reused one displacement (scaled by the agent's sigma), yielding zero within-agent execution-noise variance. The replacement advances the live generator with an independent isotropic draw at the same per-axis standard deviation.
- The 2D likelihood was unaffected because it evaluates `multivariate_normal.pdf` rather than sampling from the execution generator. The Statcast pipeline was also unaffected because it does not simulate execution noise.
- `tests/test_two_d_environment_consistency.py` pins per-observation independence, isotropic scale, and Euclidean action distance; `tests/test_two_d_environment_and_aggregation.py` additionally pins RNG replay, resolution-specific geometry, complete cluster coverage, and completion provenance.
- This correction changes simulated 2D actions. Historical 2D result CSVs and derived plots must therefore be regenerated; `submit_hjeeds_2d_cluster_tests.sh` is the canonical provenance-sealed launcher.

Publication-figure renderer corrections:

- `HJEEDS/plot_main_paper_higher_dimensional.py`: removing the parent-project reference initially left the baseball panels raising `NameError` and the separability panel reading removed globals instead of its CSV-loaded arguments. The panels now use their explicit CSV-derived arguments, so all four obtain their numerical values solely from result artifacts.
- `HJEEDS/plot_main_paper_higher_dimensional.py`: hand-tuned axis windows were replaced with limits derived from plotted values and confidence intervals because the fixed windows clipped regenerated 2D results. Drift panels label the reference checkpoint using the largest observed pitch count instead of assuming 100, PNG `Description` provenance was restored, and both 2D and MLB loaders now reject incomplete or stale completion metadata before reading plot values.
- `HJEEDS/plot_main_paper_baseline.py`: the renderer now requires the exact canonical method/metric/bucket grid, finite ordered confidence intervals, unique rows, and the complete 500-seed paper cohort by default. It validates corrected optimizer diagnostics and recomputes plotted cells from the linked agent-level CSV, preventing partial, stale, or structurally consistent pre-correction summaries from rendering.

MLB correctness hardening on 2026-07-28:

- The canonical processed Statcast pickle is intentionally untracked because it is 794,830,284 bytes. `HJEEDS/data/baseball_processed_artifact_reference.json` pins its SHA-256 (`4e1bb7e5412b1efce7f0ec08079164a5a85f3ff89bcabaa02a3ee847201a392c`), schema, 2,313-entry batter mapping, and paired `final_OP` hash. Runtime loading is cached per process and fails on any mismatch.
- MLB expected utility now uses the full `(2N_x-1) x (2N_z-1)` pairwise displacement kernel and explicitly assigns all outside-board Gaussian mass the pitch's minimum utility. The former same-size convolution omitted valid large displacements and mishandled truncated mass.
- Within-day pitches are ordered deterministically by game date, game ID, at-bat number, and pitch number. Monthly Statcast downloads fail on query gaps and require completion manifests rather than silently reusing partial CSVs.
- The MLB grouping statistic is explicitly a processed-data walks-per-inning proxy, not official regular-season BB/IP. Its numerator counts all retained `events == "walk"` rows in the canonical 2021 processed slice (March 15--November 2, including spring-training and postseason rows); its denominator is official 2021 season innings pitched from the tracked bundled artifact. The computation includes eligible zero-walk pitchers, repairs historical escaped/punctuated/particle names with collision-checked token keys, and matches all 528 eligible 2021 FF pitchers. The corrected high-proxy extreme replaces Shelby Miller with Deivi Garcia.
- Paper aggregation fails on missing caches, unsuccessful/non-finite agent checkpoints, population-fit nonconvergence, or failed/incomplete walk/IP-proxy separability. Explicitly named non-paper flags are required to relax cache or separability checks; optimizer diagnostics are persisted before failure.
- The full-kernel correction invalidates the committed legacy hyperprior centers; that preset remains marked stale and fails closed. The corrected publication run instead uses a fixed weak prior selected before viewing corrected MLB outcomes: $\sigma=0.5$ ft from external pitch-location evidence, $\lambda=100$ from a documented prior-predictive check, broad widths, and neutral correlation. `HJEEDS/data/baseball_hyperpriors_literature_informed.json` records these choices and binds them to the current kernel, data, model, and ordering hashes. The all-eligible calibration workflow remains available only for optional exploration.

These corrections invalidate historical 1D-Darts, 2D-Darts, and MLB result CSVs and their derived plots. No 528-agent calibration is required before the fixed-prior MLB convergence run.

`docs/hjeeds-paper-code-map.md` maps the manuscript to the implementation, and
`docs/hjeeds-paper-readme.md` is the end-to-end reproduction guide.
