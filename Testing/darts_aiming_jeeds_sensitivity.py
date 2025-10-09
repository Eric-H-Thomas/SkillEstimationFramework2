"""Explore how JEEDS skill estimates react to sub-optimal aiming choices.

This script implements the experiment described in the accompanying task:

1. Generate a single 1-D darts reward surface.
2. For every potential aiming location on a fixed grid, simulate an agent that
   always *intends* to hit that location. The agent's execution skill is
   modelled as Gaussian noise around the intended aim (using the domain's
   wrapping behaviour).
3. Collect samples of noisy dart landings and estimate execution skill using a
   simplified JEEDS procedure that assumes the agent is optimizing its aim for
   every candidate skill level (i.e., the estimator does **not** know where the
   agent is truly aiming).
4. Compare the resulting skill estimates to ground truth as a function of the
   agent's true skill and the optimality of its chosen aim.

Running the script will create a CSV file with the underlying data and a
scatter plot that visualizes absolute estimation error versus aiming
optimality and true skill.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Environments.Darts.RandomDarts import darts


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

    states = darts.get_N_states(rng, min_regions, max_regions, 1, min_width=min_width)
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

    return np.array(
        [darts.sample_action(rng, state, true_skill, aim) for _ in range(num_samples)],
        dtype=float,
    )


def convolve_expected_values(
    state: Sequence[float],
    skill: float,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around the environment's convolution utility."""

    evs, actions = darts.convolve_ev(None, state, skill, delta)
    return np.asarray(evs, dtype=float), np.asarray(actions, dtype=float)


def percent_optimality(
    aim: float,
    evs: np.ndarray,
    actions: np.ndarray,
) -> float:
    """Compute the percent optimality of aiming at ``aim`` for the given EV surface."""

    ev_at_aim = float(np.interp(aim, actions, evs))
    ev_min = float(np.min(evs))
    ev_max = float(np.max(evs))

    if np.isclose(ev_max, ev_min):
        return 1.0

    return (ev_at_aim - ev_min) / (ev_max - ev_min)


def log_likelihood_on_circle(samples: np.ndarray, aim: float, sigma: float) -> float:
    """Compute the log-likelihood for wrapped 1-D darts observations.

    The darts board wraps from ``-m`` to ``m``. The helper ``actionDiff`` returns
    the shortest signed distance on this circle, which lets us evaluate the
    Gaussian likelihood as if everything were centred on zero.
    """

    if sigma <= 0:
        return -np.inf

    diffs = np.array([darts.actionDiff(sample, aim) for sample in samples], dtype=float)
    var = sigma * sigma
    norm_const = -0.5 * len(diffs) * np.log(2.0 * np.pi * var)
    quad = -0.5 * np.sum(diffs * diffs) / var
    return float(norm_const + quad)


def jeeds_estimate_skill(
    samples: np.ndarray,
    state: Sequence[float],
    skill_grid: Iterable[float],
    delta: float,
) -> float:
    """Estimate execution skill using the JEEDS assumption about aiming.

    For each candidate skill we assume the agent would choose the aim that
    maximizes expected reward. We then evaluate the likelihood of the observed
    samples under a wrapped Gaussian centred at that aim.
    """

    best_skill = None
    best_log_like = -np.inf

    for skill in skill_grid:
        evs, actions = convolve_expected_values(state, skill, delta)
        aim_idx = int(np.argmax(evs))
        implied_aim = float(actions[aim_idx])
        log_like = log_likelihood_on_circle(samples, implied_aim, float(skill))

        if log_like > best_log_like:
            best_log_like = log_like
            best_skill = float(skill)

    if best_skill is None:
        raise RuntimeError("JEEDS estimation failed to evaluate any candidate skills.")

    return best_skill


def run_experiment(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    state = generate_reward_surface(rng, args.min_regions, args.max_regions, args.min_region_width)

    aims = np.linspace(-darts.m, darts.m, num=args.num_aim_points)
    true_skills = np.linspace(args.min_true_skill, args.max_true_skill, num=args.num_true_skills)
    candidate_skills = np.linspace(args.grid_min_skill, args.grid_max_skill, num=args.num_grid_skills)

    records: List[Tuple[float, float, float, float, float]] = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for true_skill in true_skills:
        evs_true, actions_true = convolve_expected_values(state, true_skill, args.delta)

        for aim in aims:
            samples = simulate_executions(rng, state, true_skill, float(aim), args.num_samples)

            percent = percent_optimality(float(aim), evs_true, actions_true)

            estimated_skill = jeeds_estimate_skill(
                samples,
                state,
                candidate_skills,
                args.delta,
            )

            abs_error = abs(estimated_skill - true_skill)

            records.append(
                (
                    float(true_skill),
                    float(aim),
                    float(percent),
                    float(estimated_skill),
                    float(abs_error),
                )
            )

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
        writer.writerows(records)

    # Prepare scatter plot
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

    figure_path = output_dir / "jeeds_skill_vs_aim.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    print(f"Saved CSV results to {csv_path}")
    print(f"Saved visualization to {figure_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_experiment(args)


if __name__ == "__main__":
    main()
