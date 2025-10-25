#!/usr/bin/env python3
"""Submit ``darts_aiming_jeeds_sensitivity.py`` to a Slurm scheduler."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from Testing import darts_aiming_jeeds_sensitivity as darts_script


def build_slurm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit darts_aiming_jeeds_sensitivity.py to Slurm",
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--job-name",
        default="darts-jeeds-sensitivity",
        help="Job name shown in the Slurm queue.",
    )
    parser.add_argument(
        "--qos",
        default="normal",
        help="Quality of service (QOS) to use when submitting the job.",
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional Slurm partition to run the job on.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Optional Slurm account to charge for the job.",
    )
    parser.add_argument(
        "--time",
        default="02:00:00",
        help="Wall clock time limit for the job (HH:MM:SS).",
    )
    parser.add_argument(
        "--mem",
        default="16G",
        help="Memory request for the job (e.g., 16G).",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=None,
        help="Number of CPU cores to allocate per task.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="GPU resource specification to pass to sbatch (e.g., 1, 'type:1').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path where Slurm should write stdout for the job.",
    )
    parser.add_argument(
        "--error",
        default=None,
        help="Optional path where Slurm should write stderr for the job.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=10,
        help="Number of parallel jobs to launch for the experiment.",
    )
    return parser


def print_help_and_exit(slurm_parser: argparse.ArgumentParser) -> None:
    print(slurm_parser.format_help())
    print("\nArguments forwarded to darts_aiming_jeeds_sensitivity.py:\n")
    try:
        darts_script.parse_args(["--help"])
    except SystemExit:
        pass
    raise SystemExit(0)


def build_sbatch_command(
    slurm_args: argparse.Namespace,
    forwarded_args: Sequence[str],
    extra_directives: Sequence[str] | None = None,
) -> list[str]:
    script_path = Path(__file__).resolve().parent / "Testing" / "darts_aiming_jeeds_sensitivity.py"

    python_executable = shlex.quote(sys.executable)
    script_str = shlex.quote(str(script_path))
    forwarded = " ".join(shlex.quote(arg) for arg in forwarded_args)
    wrapped_cmd = f"{python_executable} {script_str}"
    if forwarded:
        wrapped_cmd = f"{wrapped_cmd} {forwarded}"

    sbatch_cmd = [
        "sbatch",
        f"--job-name={slurm_args.job_name}",
        f"--qos={slurm_args.qos}",
        f"--time={slurm_args.time}",
        f"--mem={slurm_args.mem}",
    ]

    if slurm_args.partition:
        sbatch_cmd.append(f"--partition={slurm_args.partition}")
    if slurm_args.account:
        sbatch_cmd.append(f"--account={slurm_args.account}")
    if slurm_args.cpus_per_task is not None:
        sbatch_cmd.append(f"--cpus-per-task={slurm_args.cpus_per_task}")
    if slurm_args.gpus:
        sbatch_cmd.append(f"--gpus={slurm_args.gpus}")
    if slurm_args.output:
        sbatch_cmd.append(f"--output={slurm_args.output}")
    if slurm_args.error:
        sbatch_cmd.append(f"--error={slurm_args.error}")

    if extra_directives:
        sbatch_cmd.extend(extra_directives)

    sbatch_cmd.append(f"--wrap={wrapped_cmd}")
    return sbatch_cmd


def format_command(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_job_id(output: str) -> str:
    match = re.search(r"Submitted batch job (\d+)", output)
    if not match:
        raise RuntimeError(f"Unable to parse Slurm job ID from output: {output.strip()!r}")
    return match.group(1)


def main(argv: Sequence[str] | None = None) -> None:
    slurm_parser = build_slurm_parser()
    slurm_args, remaining = slurm_parser.parse_known_args(argv)

    if slurm_args.help:
        print_help_and_exit(slurm_parser)

    if slurm_args.num_jobs < 1:
        raise SystemExit("--num-jobs must be at least 1.")

    forwarded_args = list(remaining)
    if any(arg == "--num-jobs" or arg.startswith("--num-jobs=") for arg in forwarded_args):
        raise SystemExit("Pass --num-jobs to the submission script rather than forwarding it.")

    forwarded_args.extend(["--num-jobs", str(slurm_args.num_jobs)])

    # Validate the forwarded arguments using the original parser.
    _ = darts_script.parse_args(list(forwarded_args))

    if slurm_args.num_jobs == 1:
        sbatch_cmd = build_sbatch_command(slurm_args, forwarded_args)
        print("Submitting job with command:\n  " + format_command(sbatch_cmd))
        result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.strip())
        return

    array_directives = [f"--array=0-{slurm_args.num_jobs - 1}"]
    array_cmd = build_sbatch_command(slurm_args, forwarded_args, extra_directives=array_directives)
    print("Submitting job array with command:\n  " + format_command(array_cmd))
    array_result = subprocess.run(array_cmd, check=True, capture_output=True, text=True)
    if array_result.stdout:
        print(array_result.stdout.strip())
    array_job_id = parse_job_id(array_result.stdout)

    aggregator_forwarded = list(forwarded_args)
    aggregator_forwarded.append("--aggregate-results")

    aggregator_args = argparse.Namespace(**vars(slurm_args))
    aggregator_args.job_name = f"{slurm_args.job_name}-agg"

    dependency = [f"--dependency=afterok:{array_job_id}"]
    aggregator_cmd = build_sbatch_command(aggregator_args, aggregator_forwarded, extra_directives=dependency)
    print("Submitting aggregation job with command:\n  " + format_command(aggregator_cmd))
    aggregator_result = subprocess.run(aggregator_cmd, check=True, capture_output=True, text=True)
    if aggregator_result.stdout:
        print(aggregator_result.stdout.strip())


if __name__ == "__main__":
    main()
