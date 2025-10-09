# Reading experiment results for darts domains

This document explains how to locate and interpret the `.results` files that
`runExpGiven.py` produces for the 1D darts domain (the same structure applies to
the other darts variants).

## Folder layout

Running `python runExpGiven.py -domain 1d -resultsFolder YOUR_FOLDER` creates
the directory tree `Experiments/1d/YOUR_FOLDER/` (the script also maintains
other helper folders alongside the `results` directory).【F:runExpGiven.py†L456-L514】
Each agent/skill configuration inside an iteration produces a single pickle file
under `Experiments/1d/YOUR_FOLDER/results/` whose name encodes the iteration
metadata and agent name.【F:runExpGiven.py†L289-L314】  You will also find a
matching `status/*.txt` marker for each finished run.

## File format

The darts experiments save their outputs as pickled Python dictionaries.  During
initialisation the base `Experiment` class records run metadata such as the
result filename, seed, agent name, x-skill, number of observations, domain,
`delta`, and estimator configuration.【F:expTypes.py†L158-L194】  After the run
completes `RandomDartsExp.getResults` adds the per-state measurements before the
parent process merges in timing information when it writes the file back to
Disk.【F:expTypes.py†L200-L519】【F:runExpGiven.py†L404-L417】  You can therefore
recover both configuration and measurement data by unpickling the file.

Key fields you will see in a 1D darts results dictionary include:

| Key | Description |
| --- | --- |
| `agent_name` | Human-readable name for the policy that played the episode.【F:expTypes.py†L163-L166】 |
| `xskill` | Execution-skill level used for this run.【F:expTypes.py†L169-L170】 |
| `numObservations` | Number of states evaluated for the agent.【F:expTypes.py†L169-L171】 |
| `delta` | Observation noise level provided by the environment.【F:expTypes.py†L172-L173】 |
| `estimators_list` | Estimators that produced skill/rationality posteriors.【F:expTypes.py†L175-L178】 |
| `observed_rewards` | Reward obtained after noise for each state.【F:expTypes.py†L200-L205】【F:expTypes.py†L362-L392】 |
| `true_diffs` | Absolute difference between intended and noisy actions per state (useful for accuracy diagnostics).【F:expTypes.py†L200-L205】【F:expTypes.py†L367-L371】 |
| `intended_actions` / `noisy_actions` | Planned versus executed actions before/after noise.【F:expTypes.py†L200-L205】【F:expTypes.py†L348-L370】 |
| `exp_rewards` | Deterministic reward of the intended action (i.e., without noise).【F:expTypes.py†L492-L498】 |
| `rs_rewards` | List of resampled rewards drawn from repeated noisy executions; entry `i` corresponds to state `i`.【F:expTypes.py†L492-L494】【F:expTypes.py†L374-L392】 |
| `valueIntendedActions` | Value estimates returned by the agent when planning the intended action.【F:expTypes.py†L495-L498】 |
| `meanAllVsPerState` | Baseline expected reward used for comparison during estimator updates.【F:expTypes.py†L495-L498】【F:expTypes.py†L349-L353】 |
| `true_rewards` | Ground-truth reward sequence tracked internally by the agent implementation.【F:expTypes.py†L495-L499】 |
| `states` | Dictionary containing the exact states that were evaluated so you can reproduce lookups.【F:expTypes.py†L506-L509】 |
| `expTotalTime` | Runtime (seconds) for the experiment appended after the parent process finishes persisting results.【F:runExpGiven.py†L404-L417】 |
| `lastEdited` | Timestamp of when the file was last written.【F:runExpGiven.py†L404-L417】 |

Estimates that come from each estimator (posterior distributions, MAP skill
values, etc.) are also merged into the same dictionary; the exact keys depend on
the estimator implementation.

## Inspecting results interactively

You can open a result file in Python with:

```python
import pickle
from pathlib import Path

with Path("Experiments/1d/testResultsFolder/results/<file>.results").open("rb") as fh:
    data = pickle.load(fh)
```

To make the inspection easier, the repository now includes a helper script:
`python Processing/read_results_summary.py Experiments/1d/testResultsFolder/results`
which prints a short summary for each `.results` file and highlights the most
useful arrays.  See the script itself for more details on the output format.
