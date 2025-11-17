"""Verify that JEEDS darts sensitivity experiment matches when parallelized."""

from __future__ import annotations

import copy
import difflib
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from Testing import darts_aiming_jeeds_sensitivity as darts_script


def prepare_args() -> darts_script.argparse.Namespace:  # type: ignore[attr-defined]
    """Create a baseline ``Namespace`` with experiment defaults."""

    # Reuse the module's argument parser so we stay in sync with future updates.
    args = darts_script.parse_args([]) # Here, we pass in an empty list of args to get the darts script's default values

    # Tighten the configuration so the verification runs quickly.
    args.seed = 0
    args.num_samples = 8
    args.num_aim_points = 5
    args.num_true_skills = 4
    args.num_grid_skills = 10
    args.num_planning_skills = 6
    args.partial_subdir = "partials"

    return args


def clean_output_directories(*paths: Path) -> None:
    """Remove any pre-existing artifacts from previous runs."""

    for path in paths:
        if path.exists():
            shutil.rmtree(path)


def run_serial_experiment(args: darts_script.argparse.Namespace, output_dir: Path, jeeds_folder: str) -> None:  # type: ignore[attr-defined]
    """Execute the experiment without parallelization."""

    serial_args = copy.deepcopy(args)
    serial_args.output_dir = str(output_dir)
    serial_args.jeeds_results_folder = jeeds_folder
    serial_args.jeeds_time_tag = "serial_sensitivity_check"
    serial_args.num_jobs = 1
    serial_args.job_index = None
    serial_args.aggregate_results = False

    darts_script.run_experiment(serial_args)


def run_parallel_experiment(args: darts_script.argparse.Namespace, output_dir: Path, jeeds_folder: str, num_jobs: int) -> None:  # type: ignore[attr-defined]
    """Execute the experiment using ``num_jobs`` parallel shards."""

    parallel_args = copy.deepcopy(args)
    parallel_args.output_dir = str(output_dir)
    parallel_args.jeeds_results_folder = jeeds_folder
    parallel_args.jeeds_time_tag = "parallel_sensitivity_check"
    parallel_args.num_jobs = num_jobs
    parallel_args.aggregate_results = False

    for job_index in range(num_jobs):
        shard_args = copy.deepcopy(parallel_args)
        shard_args.job_index = job_index
        darts_script.run_experiment(shard_args)

    aggregate_args = copy.deepcopy(parallel_args)
    aggregate_args.aggregate_results = True
    darts_script.run_experiment(aggregate_args)


def compare_csv_outputs(serial_csv: Path, parallel_csv: Path) -> None:
    """Confirm that both experiment modes produced identical results."""

    serial_contents = serial_csv.read_text().splitlines(keepends=True)
    parallel_contents = parallel_csv.read_text().splitlines(keepends=True)

    if serial_contents != parallel_contents:
        diff = "".join(
            difflib.unified_diff(
                serial_contents,
                parallel_contents,
                fromfile=str(serial_csv),
                tofile=str(parallel_csv),
            )
        )
        raise SystemExit(f"CSV outputs differ between serial and parallel runs:\n{diff}")


def main() -> None:
    # Define output directory paths
    base_dir = Path("Testing/parallel_verification")
    serial_dir = base_dir / "serial"
    parallel_dir = base_dir / "parallel"

    experiments_dir = Path("Experiments")
    serial_jeeds = experiments_dir / "Testing" / "parallel_verification_serial"
    parallel_jeeds = experiments_dir / "Testing" / "parallel_verification_parallel"

    # Recursively clean out the output directories
    clean_output_directories(serial_dir, parallel_dir, serial_jeeds, parallel_jeeds)

    # Import the default arguments from darts_aiming_jeeds_sensitivity.py and adjust the hyperparameters to make
    # the test space smaller
    args = prepare_args()

    # Run the experiment once serially and once in parallel
    run_serial_experiment(args, serial_dir, "Testing/parallel_verification_serial")
    run_parallel_experiment(args, parallel_dir, "Testing/parallel_verification_parallel", num_jobs=10)

    # Check that the output files were created
    serial_csv = serial_dir / "jeeds_skill_vs_aim.csv"
    parallel_csv = parallel_dir / "jeeds_skill_vs_aim.csv"

    if not serial_csv.exists():
        raise SystemExit(f"Expected serial CSV at {serial_csv} was not created.")
    if not parallel_csv.exists():
        raise SystemExit(f"Expected parallel CSV at {parallel_csv} was not created.")

    # Check that the two output files match
    compare_csv_outputs(serial_csv, parallel_csv)

    print("Parallelization check succeeded: CSV outputs match.")


if __name__ == "__main__":
    main()
