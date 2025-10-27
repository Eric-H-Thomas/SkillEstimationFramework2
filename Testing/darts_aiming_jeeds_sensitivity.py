"""Explore how JEEDS skill estimates react to suboptimal aiming choices.

This script implements the experiment described in the accompanying task:

1. Generate a single 1-D darts reward surface.
2. For every potential aiming location on a fixed grid, simulate an agent that
   always *intends* to hit that location. The agent's execution skill is
   modeled as Gaussian noise around the intended aim (using the domain's
   wrapping behaviour).
3. Collect samples of noisy dart landings and estimate execution skill using the
   production JEEDS implementation (via the joint QRE estimator). Even though
   the simulated agents always choose what they believe is the optimal action,
   we still rely on the full JEEDS machinery so the sensitivity study mirrors
   the behaviour seen elsewhere in the codebase.
4. Compare the resulting skill estimates to ground truth as a function of the
   agent's true skill and the optimality of its chosen aim.

Running the script will create a CSV file with the underlying data and a
scatter plot that visualizes absolute estimation error versus aiming
optimality and true skill.

Example:
    python Testing/darts_aiming_jeeds_sensitivity.py --seed 7 --num-samples 150
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from contextlib import nullcontext, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np


# Ensure the repository root is importable when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Environments.Darts.RandomDarts import darts
from Estimators.joint import JointMethodQRE
from setupSpaces import SpacesRandomDarts


CSV_HEADER = [
    "true_skill",
    "aim",
    "percent_optimality",
    "estimated_skill",
    "absolute_error",
]


def generate_reward_surface(
    rng: np.random.Generator,
    min_regions: int,
    max_regions: int,
    min_width: float,
) -> Sequence[float]:
    """Sample a single 1-D darts reward surface.

    The underlying environment represents reward as a set of boundary points
    that alternate between losing and winning regions. We request one such
    state from the environment helper.
    """

    # ``generate_random_states`` returns a list of randomly drawn reward surfaces. Each
    # surface is represented by the boundary locations of alternating reward
    # regions. We request a single sample and unwrap it below.
    states = darts.generate_random_states(rng, min_regions, max_regions, 1, min_width=min_width)
    if not states:
        raise RuntimeError("Failed to sample a reward surface from the darts environment.")
    return states[0]


def simulate_executions(
    rng: np.random.Generator,
    state: Sequence[float],
    true_skill: float,
    aim: float,
    num_samples: int,
) -> np.ndarray:
    """Generate noisy dart throws using the environment's sampling helper."""

    # ``sample_noisy_action`` handles the wraparound board geometry and applies
    # Gaussian execution noise whose standard deviation equals ``true_skill``.
    # Repeating the call ``num_samples`` times produces an i.i.d. sample of
    # noisy dart landings.
    return np.array(
        [darts.sample_noisy_action(rng, state, true_skill, aim) for _ in range(num_samples)],
        dtype=float,
    )


def convolve_expected_values(
    state: Sequence[float],
    skill: float,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around the environment's convolution utility."""

    # ``compute_expected_value_curve`` numerically integrates the reward surface under the
    # wrapped Gaussian induced by ``skill``. It returns expected values and the
    # action grid on which they were evaluated.
    evs, actions = darts.compute_expected_value_curve(state, skill, delta)
    return np.asarray(evs, dtype=float), np.asarray(actions, dtype=float)


def percent_optimality(
    aim: float,
    evs: np.ndarray,
    actions: np.ndarray,
) -> float:
    """Compute the percent optimality of aiming at ``aim`` for the given EV surface."""

    # ``actions`` is the grid of aims the EVs were computed on. Interpolating
    # ensures that aims not exactly on the grid still receive a meaningful
    # expected value estimate.
    ev_at_aim = float(np.interp(aim, actions, evs))
    ev_min = float(np.min(evs))
    ev_max = float(np.max(evs))

    if np.isclose(ev_max, ev_min):
        return 1.0

    return (ev_at_aim - ev_min) / (ev_max - ev_min)


def ensure_jeeds_logging_dirs(results_folder: str) -> None:
    """Create the directories expected by the JEEDS implementation.

    The production estimator records timing data under ``Experiments/<folder>``.
    When reusing the estimator in a standalone analysis we create the
    directories up-front so the write calls succeed even though we do not
    inspect the generated files.
    """

    base_path = Path("Experiments") / results_folder / "times" / "estimators"
    base_path.mkdir(parents=True, exist_ok=True)


def create_jeeds_components(
    candidate_skills: Iterable[float],
    delta: float,
    num_planning_skills: int,
) -> Tuple[JointMethodQRE, SpacesRandomDarts]:
    """Instantiate the JEEDS estimator and the supporting spaces object."""

    spaces = SpacesRandomDarts(
        numObservations=1,
        domain=darts,
        mode="",
        delta=delta,
    )
    estimator = JointMethodQRE(list(candidate_skills), num_planning_skills, darts.get_domain_name())
    return estimator, spaces


def estimate_skill_with_jeeds(
    rng: np.random.Generator,
    samples: np.ndarray,
    state: Sequence[float],
    estimator: JointMethodQRE,
    spaces: SpacesRandomDarts,
    results_folder: str,
    tag: str,
    suppress_output: bool = True,
) -> float:
    """Run the official JEEDS estimator on the provided samples."""

    stream = io.StringIO()
    ctx = redirect_stdout(stream) if suppress_output else nullcontext()

    with ctx:
        estimator.reset()
        state_key = str(state)
        for sample in samples:
            estimator.addObservation(
                rng,
                spaces,
                state,
                float(sample),
                resultsFolder=results_folder,
                tag=tag,
                s=state_key,
            )

        results = estimator.getResults()
    map_key = f"{estimator.methodType}-MAP-{estimator.numXskills}-{estimator.numPskills}-xSkills"
    estimates = results.get(map_key, [])
    if not estimates:
        raise RuntimeError("JEEDS estimator returned no MAP execution skill estimate.")

    return float(estimates[-1])


def determine_job_index(job_index: int | None, num_jobs: int) -> int:
    """Resolve the job index, defaulting to Slurm's array task ID when present."""

    if num_jobs < 1:
        raise ValueError("num_jobs must be at least 1 to partition work.")

    if job_index is None:
        env_value = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_value is not None:
            job_index = int(env_value)
        else:
            job_index = 0

    if not 0 <= job_index < num_jobs:
        raise ValueError(
            f"Job index {job_index} is outside the valid range [0, {num_jobs - 1}]."
        )

    return job_index


def compute_chunk_bounds(total_items: int, num_jobs: int, job_index: int) -> Tuple[int, int]:
    """Return the half-open interval [start, end) processed by a job."""

    if total_items < 0:
        raise ValueError("total_items must be non-negative.")
    if num_jobs < 1:
        raise ValueError("num_jobs must be at least 1.")

    chunk_size = math.ceil(total_items / num_jobs) if total_items else 0
    start = job_index * chunk_size
    end = min(start + chunk_size, total_items)
    return start, end


def partial_filename(job_index: int, num_jobs: int) -> str:
    """Return the filename for a shard produced by ``job_index``."""

    width = max(2, len(str(max(num_jobs - 1, 0))))
    return f"jeeds_skill_vs_aim_part_{job_index:0{width}d}-of-{num_jobs:0{width}d}.csv"


def write_records_to_csv(path: Path, records: Sequence[Tuple[float, float, float, float, float]]) -> None:
    """Persist experiment records to ``path`` with the standard header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for record in records:
            writer.writerow(record)


def create_scatter_plot(records: Sequence[Tuple[float, float, float, float, float]], output_path: Path) -> None:
    """Create the visualization summarizing JEEDS estimation error."""

    if not records:
        raise ValueError("At least one record is required to create the scatter plot.")

    percents = [record[2] for record in records]
    skills = [record[0] for record in records]
    errors = [record[4] for record in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(percents, skills, c=errors, cmap="viridis", s=50)
    ax.set_xlabel("Percent optimality of chosen aim")
    ax.set_ylabel("True execution skill (sigma)")
    ax.set_title("JEEDS skill estimation error vs. aiming optimality")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Absolute error |$\hat{\sigma} - \sigma$|")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def produce_final_artifacts(records: Sequence[Tuple[float, float, float, float, float]], output_dir: Path) -> None:
    """Write the combined CSV and scatter plot to ``output_dir``."""

    if not records:
        raise RuntimeError("No experiment records available to write final artifacts.")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "jeeds_skill_vs_aim.csv"
    write_records_to_csv(csv_path, records)

    figure_path = output_dir / "jeeds_skill_vs_aim.png"
    create_scatter_plot(records, figure_path)

    print(f"Saved CSV results to {csv_path}")
    print(f"Saved visualization to {figure_path}")


def aggregate_partial_results(args: argparse.Namespace) -> None:
    """Load per-job CSV shards and generate the final artifacts."""

    if args.num_jobs < 1:
        raise ValueError("num_jobs must be at least 1 when aggregating results.")

    output_dir = Path(args.output_dir)
    partial_dir = output_dir / args.partial_subdir

    expected_files = [partial_dir / partial_filename(idx, args.num_jobs) for idx in range(args.num_jobs)]
    if not expected_files:
        raise RuntimeError("No partial results were expected; nothing to aggregate.")

    aggregated: List[Tuple[float, float, float, float, float]] = []
    for path in expected_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing partial results file '{path}'. Ensure all jobs completed successfully."
            )

        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != CSV_HEADER:
                raise ValueError(f"Unexpected CSV header in {path}: {header}")
            for row in reader:
                if not row:
                    continue
                aggregated.append(tuple(float(value) for value in row))

    if not aggregated:
        raise RuntimeError("Partial results files contained no experiment rows to aggregate.")

    aggregated.sort(key=lambda record: (record[0], record[1], record[2]))

    print(f"Aggregating {len(aggregated)} records from {len(expected_files)} partial files.")
    produce_final_artifacts(aggregated, output_dir)


def run_experiment(args: argparse.Namespace) -> None:
    if args.aggregate_results:
        aggregate_partial_results(args)
        return

    num_jobs = args.num_jobs
    job_index = determine_job_index(args.job_index, num_jobs)

    # Use a dedicated RNG so repeated runs with the same seed are reproducible.
    rng = np.random.default_rng(args.seed)

    # Draw a single random reward surface that remains fixed for the full
    # experiment, ensuring comparability across different aim/skill combos.
    state = list(generate_reward_surface(rng, args.min_regions, args.max_regions, args.min_region_width))

    # ``aims`` spans the possible target angles, ``true_skills`` enumerates the
    # ground-truth execution noise levels we want to test, and
    # ``candidate_skills`` defines the hypothesis space used by the JEEDS
    # estimator.
    aims = np.linspace(-darts.m, darts.m, num=args.num_aim_points)
    true_skills = np.linspace(args.min_true_skill, args.max_true_skill, num=args.num_true_skills)
    candidate_skills = np.linspace(args.grid_min_skill, args.grid_max_skill, num=args.num_grid_skills)

    ensure_jeeds_logging_dirs(args.jeeds_results_folder)
    jeeds_estimator, spaces = create_jeeds_components(candidate_skills, args.delta, args.num_planning_skills)

    total_combinations = len(true_skills) * len(aims)

    # When partitioning work across multiple jobs we ensure that every
    # aim/skill combination consumes a deterministic stream of random numbers
    # derived from the global seed. This guarantees reproducible results
    # regardless of how combinations are sharded across jobs.
    if total_combinations:
        combination_seeds = np.random.SeedSequence(args.seed + 1).spawn(total_combinations)
    else:
        combination_seeds = []
    start_idx, end_idx = compute_chunk_bounds(total_combinations, num_jobs, job_index)

    # Each record stores (true skill, aim, percent optimality, estimated skill,
    # absolute error). Keeping the raw data allows downstream analysis without
    # re-running the simulation.
    records: List[Tuple[float, float, float, float, float]] = []

    completed_iterations = 0
    total_iterations = end_idx - start_idx

    def update_progress() -> None:
        if total_iterations:
            percent = (completed_iterations / total_iterations) * 100
            print(
                f"\rJob {job_index + 1}/{num_jobs}: {percent:5.1f}% "
                f"({completed_iterations}/{total_iterations})",
                end="",
                flush=True,
            )

    # Group the chunk's combinations by true-skill index so we only convolve
    # expected values for the skills needed in this partition.
    combos_by_skill: dict[int, List[int]] = defaultdict(list)
    num_aims = len(aims)
    for linear_idx in range(start_idx, end_idx):
        skill_idx, aim_idx = divmod(linear_idx, num_aims)
        combos_by_skill[skill_idx].append(aim_idx)

    for skill_idx in sorted(combos_by_skill):
        true_skill = float(true_skills[skill_idx])

        # Pre-compute the EV surface under the true skill to avoid redundant
        # convolutions inside the aiming loop.
        evs_true, actions_true = convolve_expected_values(state, true_skill, args.delta)

        for aim_idx in combos_by_skill[skill_idx]:
            aim_value = float(aims[aim_idx])

            linear_idx = skill_idx * num_aims + aim_idx
            combo_rng = np.random.default_rng(combination_seeds[linear_idx])

            samples = simulate_executions(combo_rng, state, true_skill, aim_value, args.num_samples)

            # Measure how much value the chosen aim sacrifices relative to the
            # optimal aim at this true skill level.
            percent = percent_optimality(aim_value, evs_true, actions_true)

            estimated_skill = estimate_skill_with_jeeds(
                combo_rng,
                samples,
                state,
                jeeds_estimator,
                spaces,
                args.jeeds_results_folder,
                args.jeeds_time_tag,
            )

            # Quantify deviation between JEEDS estimate and ground truth.
            abs_error = abs(estimated_skill - true_skill)

            # Collect a row of results that will later be written to disk.
            records.append(
                (
                    true_skill,
                    aim_value,
                    float(percent),
                    float(estimated_skill),
                    float(abs_error),
                )
            )

            completed_iterations += 1
            update_progress()

    if total_iterations:
        print()

    output_dir = Path(args.output_dir)

    if args.num_jobs == 1:
        produce_final_artifacts(records, output_dir)
    else:
        partial_dir = output_dir / args.partial_subdir
        partial_path = partial_dir / partial_filename(job_index, num_jobs)
        write_records_to_csv(partial_path, records)
        print(f"Saved partial results to {partial_path}")
        if not records:
            print(
                "Warning: this job's assignment did not include any aim/skill "
                "combinations."
            )
        print(
            "Run the aggregation step once all jobs have finished using the "
            "--aggregate-results flag."
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Expose the key experiment knobs via the command line so the sensitivity
    # analysis can be rerun with different configurations.
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--seed", type=int, default=12345, help="Random seed for reproducibility.")
    parser.add_argument("--num-samples", type=int, default=100, help="Noisy executions per aim point.")
    parser.add_argument("--num-aim-points", type=int, default=41, help="Resolution of the aiming grid.")
    parser.add_argument("--num-true-skills", type=int, default=10, help="Number of true execution skills to test.")
    parser.add_argument(
        "--min-true-skill",
        type=float,
        default=0.2,
        help="Lower bound of the true skill range (Gaussian sigma).",
    )
    parser.add_argument(
        "--max-true-skill",
        type=float,
        default=2.0,
        help="Upper bound of the true skill range (Gaussian sigma).",
    )
    parser.add_argument(
        "--num-grid-skills",
        type=int,
        default=30,
        help="Number of candidate skills for the JEEDS estimator.",
    )
    parser.add_argument(
        "--num-planning-skills",
        type=int,
        default=33,
        help="Number of planning-skill hypotheses passed to the JEEDS estimator.",
    )
    parser.add_argument(
        "--grid-min-skill",
        type=float,
        default=0.1,
        help="Minimum candidate skill considered by JEEDS.",
    )
    parser.add_argument(
        "--grid-max-skill",
        type=float,
        default=3.0,
        help="Maximum candidate skill considered by JEEDS.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-2,
        help="Resolution of the convolution used to compute expected values.",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=2,
        help="Minimum number of reward regions to sample when generating the surface.",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=6,
        help="Maximum number of reward regions to sample when generating the surface.",
    )
    parser.add_argument(
        "--min-region-width",
        type=float,
        default=0.25,
        help="Minimum width enforced between reward boundaries (in board units).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Testing/results",
        help="Directory where artifacts (CSV + plot) will be stored.",
    )
    parser.add_argument(
        "--jeeds-results-folder",
        type=str,
        default="Testing/darts_aiming_jeeds_sensitivity",
        help="Folder name (under Experiments/) used for JEEDS timing logs.",
    )
    parser.add_argument(
        "--jeeds-time-tag",
        type=str,
        default="darts_aiming_jeeds_sensitivity",
        help="Tag appended to JEEDS timing log filenames.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="Total number of jobs into which the experiment is partitioned.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Zero-based index of the current job when running in parallel.",
    )
    parser.add_argument(
        "--partial-subdir",
        type=str,
        default="partials",
        help="Subdirectory (under output-dir) where per-job CSV shards are stored.",
    )
    parser.add_argument(
        "--aggregate-results",
        action="store_true",
        help="Aggregate per-job CSV shards and recreate the final plot.",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Delegate to the core experiment runner once configuration is parsed.
    run_experiment(args)


if __name__ == "__main__":
    main()
