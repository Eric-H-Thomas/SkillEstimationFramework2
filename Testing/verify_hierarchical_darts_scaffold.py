"""Structural verification for the hierarchical darts experiment script.

This verification intentionally checks the lightweight parts of the experiment
script:

- the module imports correctly,
- the default parser/config values look sensible,
- a small custom configuration builds successfully, and
- the top-level ``main`` function completes in ``--dry-run`` mode.

The point is to catch wiring and orchestration issues quickly without launching
the simulation and inference path.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Testing import darts_hierarchical_vs_jeeds as experiment


def verify_default_arguments() -> None:
    """Check a few key parser defaults so accidental drift is visible."""

    args = experiment.parse_args([])
    assert args.num_agents == experiment.DEFAULT_NUM_AGENTS
    assert args.num_seeds == experiment.DEFAULT_NUM_SEEDS
    assert args.count_buckets == "5,10,25,100"
    assert args.dry_run is False


def verify_tiny_config_build() -> None:
    """Build a small custom config and verify the main derived fields."""

    args = experiment.parse_args(
        [
            "--seed",
            "7",
            "--num-seeds",
            "2",
            "--num-agents",
            "4",
            "--count-buckets",
            "3,10",
            "--agents-per-bucket",
            "2",
            "--delta",
            "0.2",
            "--output-dir",
            "Testing/results/hierarchical_darts_verification",
        ]
    )
    config = experiment.build_config_from_args(args)

    assert config.seed_values == (7, 8)
    assert config.count_buckets == (3, 10)
    assert config.expected_agent_count == 4
    assert config.num_agents == 4
    assert math_is_close(config.delta, 0.2)


def verify_dry_run_path() -> None:
    """Run ``main`` in dry-run mode and confirm it exits cleanly."""

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        exit_code = experiment.main(
            [
                "--seed",
                "9",
                "--num-seeds",
                "1",
                "--num-agents",
                "4",
                "--count-buckets",
                "3,10",
                "--agents-per-bucket",
                "2",
                "--dry-run",
            ]
        )

    output = stdout.getvalue()
    assert exit_code == 0
    assert "DRY RUN" in output
    assert "No simulation or inference functions will be executed." in output
    assert "Planned artifacts:" in output


def math_is_close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    """Tiny helper so we do not need to import ``math`` just for one assert."""

    return abs(left - right) <= tolerance


def main() -> None:
    """Execute all structural verification checks."""

    verify_default_arguments()
    verify_tiny_config_build()
    verify_dry_run_path()
    print("Hierarchical darts experiment verification succeeded.")


if __name__ == "__main__":
    main()
