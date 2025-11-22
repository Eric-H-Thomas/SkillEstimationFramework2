# Running experiments

This repository provides several entry points for launching experiments across the darts, billiards, baseball, soccer, and hockey domains. Each script sets sensible defaults for its target workflow and writes outputs under `Experiments/<domain>/<resultsFolder>/`.

The commands below assume you have followed the environment setup instructions in `docs/environment-setup.md` and that you run them from the repository root.

## Common behavior
- Results, status markers (`*-DONE.txt`), and timing information are created automatically under the `Experiments/` directory.
- Use `-resultsFolder <name>` to separate runs. Many scripts also hard-code or prepend the domain to this folder internally.
- `-seed` controls deterministic random number generation. Some scripts additionally save seeds to disk for reproducibility.
- Baseball, soccer, and hockey runs often load and cache external data. When reusing IDs/types, keep `-maxRows` and `-dataBy` consistent across reruns to avoid mismatched checkpoints.

## `runExpDynamic.py`
Runs darts-style experiments where agent execution skill changes over time. Defaults are tuned for cluster runs (500 iterations, 100 observations, particle filters enabled).

**Key arguments**
- `-domain` (default: `2d-multi`): darts domains (`1d`, `2d`, `2d-multi`, `sequentialDarts`, or `billiards`).
- `-mode`: darts launch mode (`normal`, `rand_pos`, `rand_v`).
- `-dynamic`: enables dynamic execution skills (on by default in this script).
- Particle filter toggles: `-particles`, `-numParticles`, `-resampleNEFF`, `-resample`, `-noise`, `-resamplingMethod`.
- Estimator selection: `-jeeds`, `-pfe`, `-pfeNeff`.
- Iteration controls: `-iters`, `-numObservations`, `-seed`, `-rerun`.

**Example**
```bash
python runExpDynamic.py -domain 2d -resultsFolder Dynamic2D -iters 50 -numObservations 200 -dynamic
```

## `runExpRandom.py`
Draws execution and planning skill parameters randomly each iteration. Suitable for broad Monte Carlo sweeps across darts, billiards, baseball, and soccer.

**Key arguments**
- `-domain`: choose among darts variants (`1d`, `2d`, `2d-multi`, `sequentialDarts`), `billiards`, `baseball`, `baseball-multi`, or `soccer`.
- Skill sampling: `-numXskillsPerExp`, `-numPskillsPerExp`, `-numRhosPerExp` control how many random tuples are tested per iteration.
- Data windows for persistent domains: `-startYear`, `-endYear`, `-startMonth`, `-endMonth`, `-startDay`, `-endDay`, `-ids`, `-types`, `-every`, `-maxRows`, `-dataBy`, `-reload`.
- Particle filter and estimator flags mirror `runExpDynamic.py` (`-particles`, `-numParticles`, `-resampleNEFF`, `-resample`, `-noise`, `-resamplingMethod`, `-jeeds`, `-pfe`, `-pfeNeff`).
- Rerun controls: `-rerun` reprocesses previous runs where applicable, and `-folderSeedNums` reloads prior seeds.

**Example**
```bash
python runExpRandom.py -domain baseball -ids 642232 621237 -types FF SL -every 10 -resultsFolder RandomBaseball
```

## `runExpGiven.py`
Executes experiments for explicitly provided skill configurations. Use this when you want deterministic combinations instead of random draws.

**Key arguments**
- Same core arguments as `runExpRandom.py` for domains, data ranges, and estimator toggles.
- Provide skill tuples: `-xSkillsGiven`/`-pSkillsGiven` force the use of supplied execution/planning skills instead of sampled values. For multi-dimensional darts, pass `-agent` values per dimension.
- Iteration sizing: `-iters`, `-numObservations`, and `-numXskillsPerExp`/`-numPskillsPerExp`/`-numRhosPerExp` dictate how many given skills are paired per iteration.
- Reproducibility: `-seed` sets RNG seeds; seeds may be written to `Experiments/<domain>/<resultsFolder>/times/` unless `-rerun` is used.

**Example**
```bash
python runExpGiven.py -domain 2d -xSkillsGiven -pSkillsGiven -numXskillsPerExp 2 -numPskillsPerExp 2 \
  -resultsFolder Given2D -seed 123
```

## `runExpBaseball.py`
Specialized entry point for baseball-only batch runs. The script overrides several defaults for the `baseball-multi` domain (e.g., `-domain baseball-multi`, `-maxRows 100`, `-every 20`).

**Key arguments**
- Data filters: `-startYear`, `-endYear`, `-startMonth`, `-endMonth`, `-startDay`, `-endDay`.
- Pitch selection: `-ids` (pitcher IDs), `-types` (pitch types), `-maxRows` (most recent pitches), `-dataBy` (`recent`, `chunks`, or `pitchNum`), and bucket range `-b1`/`-b2` when chunking.
- Checkpointing: `-every` sets how often to write intermediate backups; use `-reload` to resume from previous checkpoints.
- Estimator and particle filter controls match `runExpRandom.py`.

**Example**
```bash
python runExpBaseball.py -ids 642232 -types FF CH -startYear 2022 -endYear 2023 -resultsFolder Pitcher642232
```

## `runExpHockey.py`
Runs hockey multi-agent experiments using shot data. Defaults set `-domain hockey-multi` and enable hockey-specific agent labels.

**Key arguments**
- Player and shot selection: `-id` (player ID), `-type` (shot type), plus baseball-style data filters (`-startYear`/`-endYear` etc.) reused for consistency.
- Skill configuration: same set of `-xSkillsGiven`, `-pSkillsGiven`, and per-experiment counts used in other runners.
- Particle filter and estimator flags: `-particles`, `-numParticles`, `-resampleNEFF`, `-resample`, `-noise`, `-resamplingMethod`, `-jeeds`, `-pfe`, `-pfeNeff`.
- Rerun and seeding options: `-seed`, `-rerun`, `-folderSeedNums`.

**Example**
```bash
python runExpHockey.py -id 8478402 -type Wrist -numObservations 150 -iters 20 -resultsFolder HockeyWristShots
```
