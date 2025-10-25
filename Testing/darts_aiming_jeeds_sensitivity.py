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
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from contextlib import nullcontext, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

from Environments.Darts.RandomDarts import darts
from Estimators.joint import JointMethodQRE
from setupSpaces import SpacesRandomDarts


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


def run_experiment(args: argparse.Namespace) -> None:
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

    # Each record stores (true skill, aim, percent optimality, estimated skill,
    # absolute error). Keeping the raw data allows downstream analysis without
    # re-running the simulation.
    records: List[Tuple[float, float, float, float, float]] = []

    # Ensure the requested output directory exists before writing artifacts.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_iterations = len(true_skills) * len(aims)
    completed_iterations = 0

    def update_progress() -> None:
        if total_iterations:
            percent = (completed_iterations / total_iterations) * 100
            print(
                f"\rProgress: {percent:5.1f}% ({completed_iterations}/{total_iterations})",
                end="",
                flush=True,
            )

    for true_skill in true_skills:
        # Pre-compute the EV surface under the true skill to avoid redundant
        # convolutions inside the aiming loop.
        evs_true, actions_true = convolve_expected_values(state, true_skill, args.delta)

        for aim in aims:
            samples = simulate_executions(rng, state, true_skill, float(aim), args.num_samples)

            # Measure how much value the chosen aim sacrifices relative to the
            # optimal aim at this true skill level.
            percent = percent_optimality(float(aim), evs_true, actions_true)

            estimated_skill = estimate_skill_with_jeeds(
                rng,
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
                    float(true_skill),
                    float(aim),
                    float(percent),
                    float(estimated_skill),
                    float(abs_error),
                )
            )

            completed_iterations += 1
            update_progress()

    if total_iterations:
        print()

    # Persist the numerical results for subsequent analysis in spreadsheets or
    # notebooks.
    csv_path = output_dir / "jeeds_skill_vs_aim.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "true_skill",
            "aim",
            "percent_optimality",
            "estimated_skill",
            "absolute_error",
        ])
        # Dump all experiment rows in one go.
        writer.writerows(records)

    # Prepare scatter plot summarizing how estimation error varies with aim and
    # true skill.
    # Extract columns for plotting: aim optimality (x-axis), true skill (y-axis),
    # and estimation error (colour map).
    percents = [record[2] for record in records]
    skills = [record[0] for record in records]
    errors = [record[4] for record in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Encode estimation error as colour so the plot simultaneously conveys how
    # aim optimality and true skill interact to influence the JEEDS estimator.
    scatter = ax.scatter(percents, skills, c=errors, cmap="viridis", s=50)
    ax.set_xlabel("Percent optimality of chosen aim")
    ax.set_ylabel("True execution skill (sigma)")
    ax.set_title("JEEDS skill estimation error vs. aiming optimality")
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    cbar = fig.colorbar(scatter, ax=ax)
    # Annotate the colour bar so viewers can interpret estimation errors.
    cbar.set_label(r"Absolute error |$\hat{\sigma} - \sigma$|")

    # Reduce whitespace and prevent label clipping in the saved figure.
    fig.tight_layout()

    # Persist the visualization alongside the CSV for quick inspection.
    figure_path = output_dir / "jeeds_skill_vs_aim.png"
    fig.savefig(figure_path, dpi=300)
    # Explicitly close the figure to free memory when running headless batches.
    plt.close(fig)

    # Provide paths so users can quickly locate generated artifacts.
    print(f"Saved CSV results to {csv_path}")
    print(f"Saved visualization to {figure_path}")


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

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Delegate to the core experiment runner once configuration is parsed.
    run_experiment(args)


if __name__ == "__main__":
    main()
