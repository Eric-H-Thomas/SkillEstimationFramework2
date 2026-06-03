# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""Run the full H-JEEDS paper experiment suite locally or on Slurm."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import stat
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from HJEEDS.config import DEFAULT_NUM_SEEDS, parse_seed_argument


# Default location for the all-paper output tree when the caller does not override it
DEFAULT_OUTPUT_ROOT = Path("HJEEDS/results/hjeeds_paper_experiments")


# Files written at the top of the paper output tree
MANIFEST_FILENAME = "paper_experiment_manifest.json" # Record of experimental design
STATUS_FILENAME = "paper_experiment_status.csv" # Log of experiments started, finished, failed, etc.


# Private subdirectory for runner-owned cache/config files used by local and Slurm child processes
RUNNER_DIRNAME = "_runner"

# Private subdirectory for generated Slurm scripts and Slurm log paths
SLURM_DIRNAME = "_slurm"


# Synthetic status-table slug used for the final zip step, which is not a normal experiment
ZIP_ONLY_EXPERIMENT_SLUG = "__zip_only__"

# Placeholder job id used only when printing dry-run Slurm dependency commands
DRY_RUN_JOB_ID_PLACEHOLDER = 1234567


# Scientific Python libraries sometimes default to many native threads per process
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


# Fixed schema for the status CSV so interrupted runs are easy to inspect
STATUS_HEADER = [
    "timestamp",
    "experiment_slug",
    "phase",
    "status",
    "detail",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment family in the H-JEEDS paper suite."""

    # Short stable identifier used by manifests and status rows
    slug: str

    # Human-readable name printed in summaries and written to the manifest
    label: str

    # Python module invoked with ``python -m`` for this experiment family
    module: str

    # Subdirectory under the paper output root where this experiment writes artifacts
    output_subdir: str

    # Number of independent scenarios in this experiment family
    scenario_count: int

    # Whether Slurm mode can run this experiment as an array of scenario tasks
    supports_scenario_array: bool

    # Whether a separate aggregation job must collect scenario folders afterward
    needs_aggregation: bool


# The order of this tuple is the canonical paper-suite order used everywhere below
EXPERIMENT_SPECS = (
    # Single all-in-one baseline job
    ExperimentSpec(
        slug="baseline",
        label="Baseline H-JEEDS vs JEEDS",
        module="HJEEDS.darts_hierarchical_vs_jeeds",
        output_subdir="baseline",
        scenario_count=1,
        supports_scenario_array=False,
        needs_aggregation=False,
    ),
    # 4 focus areas x 5 bias levels x 3 confidence levels
    ExperimentSpec(
        slug="hyperprior_robustness",
        label="Hyperprior robustness",
        module="HJEEDS.darts_hierarchical_prior_sensitivity",
        output_subdir="hyperprior_robustness",
        scenario_count=60,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 5 agents-per-bucket values x 3 representative hyperprior conditions
    ExperimentSpec(
        slug="agents_per_bucket",
        label="Agents per bucket",
        module="HJEEDS.darts_agents_per_bucket_sensitivity",
        output_subdir="agents_per_bucket",
        scenario_count=15,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 4 population shapes x 5 agents-per-bucket values
    ExperimentSpec(
        slug="population_shape",
        label="Population shape",
        module="HJEEDS.darts_population_shape_sensitivity",
        output_subdir="population_shape",
        scenario_count=20,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 6 high-data anchor availability settings
    ExperimentSpec(
        slug="anchor_availability",
        label="Anchor availability",
        module="HJEEDS.darts_anchor_availability_sensitivity",
        output_subdir="anchor_availability",
        scenario_count=6,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 4 true decision models x 5 agents-per-bucket values
    ExperimentSpec(
        slug="decision_model",
        label="Decision model",
        module="HJEEDS.darts_decision_model_sensitivity",
        output_subdir="decision_model",
        scenario_count=20,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 5 true correlations x 5 agents-per-bucket values
    ExperimentSpec(
        slug="true_correlation",
        label="True population correlation",
        module="HJEEDS.darts_true_correlation_sensitivity",
        output_subdir="true_correlation",
        scenario_count=25,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 3 estimator grid sizes
    ExperimentSpec(
        slug="grid_resolution",
        label="Grid resolution",
        module="HJEEDS.darts_grid_resolution_sensitivity",
        output_subdir="grid_resolution",
        scenario_count=3,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
    # 3 compound stress settings x 5 agents-per-bucket values
    ExperimentSpec(
        slug="compound_stress",
        label="Compound stress",
        module="HJEEDS.darts_compound_stress_sensitivity",
        output_subdir="compound_stress",
        scenario_count=15,
        supports_scenario_array=True,
        needs_aggregation=True,
    ),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the unified paper runner."""

    # argparse owns user-facing validation for enum-like options such as --mode
    parser = argparse.ArgumentParser(description=__doc__)

    # local runs everything sequentially, slurm submits jobs, zip-only packages existing results
    parser.add_argument(
        "--mode",
        choices=("local", "slurm", "zip-only"),
        default="local",
        help="Execution mode for the full experiment suite.",
    )

    # Reuse the same seed parser as the experiment runners so "default" means 12345 everywhere
    parser.add_argument(
        "--seed",
        type=parse_seed_argument,
        required=True,
        help="Base seed used by every experiment. Use 'default' for 12345.",
    )

    # This controls the replicate count passed through to every experiment runner
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of random seeds per scenario (default: {DEFAULT_NUM_SEEDS}).",
    )

    # All experiment-specific output directories are created under this root
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder that will contain every paper experiment output.",
    )

    # Dry-run prints the planned commands or sbatch submissions without creating work
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands or Slurm submissions without launching work.",
    )

    # Allows cluster users to choose a module-loaded Python without editing this script
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used for local runs and generated Slurm scripts.",
    )

    # Slurm-specific resource knobs are only used in --mode slurm
    parser.add_argument("--qos", type=str, default=None, help="Optional Slurm QOS.")
    parser.add_argument("--partition", type=str, default=None, help="Optional Slurm partition.")
    parser.add_argument("--account", type=str, default=None, help="Optional Slurm account.")
    parser.add_argument("--time", type=str, default="23:00:00", help="Slurm wall time per job.")
    parser.add_argument("--mem", type=str, default="16G", help="Slurm memory per task.")
    parser.add_argument("--cpus-per-task", type=int, default=1, help="Slurm CPUs per task.")

    # By default, Slurm logs stay inside the paper output tree next to generated scripts
    parser.add_argument(
        "--slurm-output-dir",
        type=str,
        default=None,
        help="Directory for Slurm stdout/stderr logs. Defaults to <output-root>/_slurm/logs.",
    )

    # Lets the final archive be written somewhere other than the default sibling zip
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Optional zip path. Defaults to <output-root>.zip.",
    )
    return parser.parse_args(argv)


def zip_path_for_output_root(output_root: Path, raw_zip_path: str | None) -> Path:
    """Return the final zip path for an output root."""

    # An explicit --zip-path wins when the caller wants the archive somewhere special
    if raw_zip_path:
        return Path(raw_zip_path)

    # Otherwise place the archive next to the output root, e.g. results/foo.zip
    return output_root.with_suffix(".zip")


def experiment_output_dir(output_root: Path, spec: ExperimentSpec) -> Path:
    """Return one experiment's output directory under the paper root."""

    # The subdirectory names are part of the documented output layout
    return output_root / spec.output_subdir


def experiment_command(
    python_bin: str,
    spec: ExperimentSpec,
    *,
    seed: int,
    num_seeds: int,
    output_root: Path,
    aggregate_results: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Build the Python command for one experiment runner."""

    # Every runner in the suite accepts the same core trio: seed, num-seeds, output-dir
    command = [
        python_bin,
        "-m",
        spec.module,
        "--seed",
        str(seed),
        "--num-seeds",
        str(num_seeds),
        "--output-dir",
        str(experiment_output_dir(output_root, spec)),
    ]

    # Aggregation mode tells multi-scenario runners to read scenario folders rather than recompute
    if aggregate_results:
        command.append("--aggregate-results")

    # Dry-run mode is passed through when this top-level runner is previewing child commands
    if dry_run:
        command.append("--dry-run")
    return command


def status_path(output_root: Path) -> Path:
    """Return the status CSV path for the paper run."""

    # Keeping this as a helper prevents small path mismatches across status readers/writers
    return output_root / STATUS_FILENAME


def now_iso() -> str:
    """Return a compact local timestamp for manifests and status rows."""

    # Include the local timezone so cluster and local logs can be compared later
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_status(
    output_root: Path,
    *,
    experiment_slug: str,
    phase: str,
    status: str,
    detail: str,
) -> None:
    """Append one row to the paper experiment status CSV."""

    # Status rows live at the paper root and are appended throughout local/Slurm orchestration
    output_root.mkdir(parents=True, exist_ok=True)
    path = status_path(output_root)

    # A brand-new status file needs its header before the first row
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_HEADER)
        if write_header:
            writer.writeheader()

        # Keep status records simple enough to inspect with a spreadsheet or grep
        writer.writerow(
            {
                "timestamp": now_iso(),
                "experiment_slug": experiment_slug,
                "phase": phase,
                "status": status,
                "detail": detail,
            }
        )


def write_manifest(
    output_root: Path,
    experiment_specs: Sequence[ExperimentSpec],
    *,
    mode: str,
    seed: int,
    num_seeds: int,
    zip_path: Path,
) -> None:
    """Write a JSON manifest describing the full paper experiment run."""

    # The manifest is the stable record of what was requested at launch time
    output_root.mkdir(parents=True, exist_ok=True)

    # Store both human-facing labels and machine-facing module/output details
    manifest = {
        "created_at": now_iso(),
        "mode": mode,
        "seed": seed,
        "num_seeds": num_seeds,
        "output_root": str(output_root),
        "zip_path": str(zip_path),
        "experiment_count": len(experiment_specs),
        "slurm_scenario_tasks": sum(spec.scenario_count for spec in experiment_specs),
        "experiments": [
            {
                "slug": spec.slug,
                "label": spec.label,
                "module": spec.module,
                "output_dir": str(experiment_output_dir(output_root, spec)),
                "scenario_count": spec.scenario_count,
                "supports_scenario_array": spec.supports_scenario_array,
                "needs_aggregation": spec.needs_aggregation,
            }
            for spec in experiment_specs
        ],
    }

    # Pretty JSON keeps the manifest easy to review and diff
    with (output_root / MANIFEST_FILENAME).open("w", newline="") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def output_files_under(output_root: Path) -> list[Path]:
    """Return files already present under an output root."""

    # A missing output root cannot contain output
    if not output_root.exists():
        return []

    # The policy treats any file as existing output; empty directories are allowed
    return sorted(path for path in output_root.rglob("*") if path.is_file())


def ensure_output_root_policy(output_root: Path, *, mode: str) -> None:
    """Prevent accidental mixing of expensive paper experiment outputs."""

    # Zip-only is allowed only after a real output root already exists
    if mode == "zip-only":
        if not output_root.exists():
            raise FileNotFoundError(f"Cannot zip missing output root: {output_root}")
        if not output_files_under(output_root):
            raise FileNotFoundError(f"Cannot zip output root because it contains no files: {output_root}")
        return

    existing_output_files = output_files_under(output_root)
    if existing_output_files:
        preview = "\n".join(f"  - {path}" for path in existing_output_files[:5])
        if len(existing_output_files) > 5:
            preview += f"\n  - ... and {len(existing_output_files) - 5} more files"
        raise FileExistsError(
            "Output root already contains output files, so the paper experiment runner will not "
            f"write into it: {output_root}\n"
            "Choose a new --output-root or move/delete the existing outputs first. "
            "Existing empty directories are allowed, but any file is treated as existing output.\n"
            f"Existing files:\n{preview}"
        )


def print_suite_summary(
    experiment_specs: Sequence[ExperimentSpec],
    *,
    mode: str,
    output_root: Path,
    zip_path: Path,
    num_seeds: int,
    seed: int,
) -> None:
    """Print a compact summary of the full paper workload."""

    # The total counts Slurm array tasks, not aggregation or final zip jobs
    total_scenario_tasks = sum(spec.scenario_count for spec in experiment_specs)
    print("=== H-JEEDS Paper Experiment Suite ===")
    print(f"Mode: {mode}")
    print(f"Seed: {seed}")
    print(f"Seeds per scenario: {num_seeds}")
    print(f"Output root: {output_root.resolve()}")
    print(f"Zip path: {zip_path.resolve()}")
    print(f"Experiments: {len(experiment_specs)}")
    print(f"Slurm scenario tasks: {total_scenario_tasks}")
    print()

    # Print one line per experiment family so dry-runs can be checked at a glance
    for spec in experiment_specs:
        print(
            "  - "
            f"{spec.slug}: {spec.label}, scenarios={spec.scenario_count}, "
            f"output={experiment_output_dir(output_root, spec)}"
        )


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str] | None = None,
) -> None:
    """Run or print one subprocess command."""

    # shlex.join produces a shell-readable display string without actually using shell=True
    printable_command = shlex.join(command)
    if dry_run:
        print(f"[dry-run] {printable_command}")
        return

    # Commands are passed as argument lists to avoid shell quoting surprises
    subprocess.run(list(command), cwd=cwd, check=True, env=env)


def zip_output_root(output_root: Path, zip_path: Path) -> None:
    """Create a zip archive containing the whole paper output root."""

    # Ensure the archive destination exists, even when it is outside the output root
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_zip_path = zip_path.resolve()

    # Store paths relative to output_root.parent so the zip contains the top-level folder name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.rglob("*")):

            # Avoid recursively adding the zip file if someone places it inside the output tree
            if path.resolve() == resolved_zip_path:
                continue

            # Directories are implicit in zip archives, so only files need entries
            if path.is_file():
                archive.write(path, path.relative_to(output_root.parent))


def run_local_suite(
    experiment_specs: Sequence[ExperimentSpec],
    *,
    seed: int,
    num_seeds: int,
    output_root: Path,
    zip_path: Path,
    python_bin: str,
    dry_run: bool,
    repo_root: Path,
) -> None:
    """Run all paper experiments sequentially in the current process environment."""

    # Matplotlib tries to write a cache; point it inside the output tree for sandbox/cluster friendliness
    local_env = os.environ.copy()
    local_env["MPLBACKEND"] = "Agg"
    local_env["MPLCONFIGDIR"] = str(output_root / RUNNER_DIRNAME / "matplotlib")
    for variable in THREAD_LIMIT_ENV_VARS:
        local_env.setdefault(variable, "1")

    # Create the Matplotlib cache directory only for real runs, keeping dry-run read-only
    if not dry_run:
        Path(local_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    # Local mode intentionally runs in manifest order so failures are easy to locate
    for spec in experiment_specs:

        # Build the direct python -m command for this experiment family
        command = experiment_command(
            python_bin,
            spec,
            seed=seed,
            num_seeds=num_seeds,
            output_root=output_root,
        )

        # Record the start before launching an expensive child process
        if not dry_run:
            append_status(
                output_root,
                experiment_slug=spec.slug,
                phase="experiment",
                status="started",
                detail=shlex.join(command),
            )
        try:
            # Let the child runner perform its normal sequential experiment workflow
            run_command(command, cwd=repo_root, dry_run=dry_run, env=local_env)
        except subprocess.CalledProcessError as exc:

            # Preserve a failed status row before re-raising so the caller sees the real error
            if not dry_run:
                append_status(
                    output_root,
                    experiment_slug=spec.slug,
                    phase="experiment",
                    status="failed",
                    detail=str(exc),
                )
            raise

        # Mark this experiment complete only after the child process exits successfully
        if not dry_run:
            append_status(
                output_root,
                experiment_slug=spec.slug,
                phase="experiment",
                status="complete",
                detail=str(experiment_output_dir(output_root, spec)),
            )

    # Dry-run stops before writing the archive
    if dry_run:
        print(f"[dry-run] zip {output_root} -> {zip_path}")
        return

    # The final local step packages every experiment into one export-ready zip
    append_status(
        output_root,
        experiment_slug=ZIP_ONLY_EXPERIMENT_SLUG,
        phase="zip",
        status="started",
        detail=str(zip_path),
    )
    zip_output_root(output_root, zip_path)
    append_status(
        output_root,
        experiment_slug=ZIP_ONLY_EXPERIMENT_SLUG,
        phase="zip",
        status="complete",
        detail=str(zip_path),
    )


def slurm_output_dir(output_root: Path, raw_slurm_output_dir: str | None) -> Path:
    """Return the Slurm log directory for generated jobs."""

    # A caller-provided log directory is useful on clusters with preferred scratch/log locations
    if raw_slurm_output_dir:
        return Path(raw_slurm_output_dir)

    # The default keeps Slurm output bundled with the paper experiment root
    return output_root / SLURM_DIRNAME / "logs"


def slurm_script_path(output_root: Path, script_slug: str) -> Path:
    """Return a generated Slurm helper script path."""

    # Generated scripts are not repo files; they are artifacts of a specific paper run
    return output_root / SLURM_DIRNAME / f"{script_slug}.sh"


def slurm_common_args(args: argparse.Namespace, log_dir: Path, job_name: str) -> list[str]:
    """Build common sbatch options shared by all generated jobs."""

    # --parsable makes sbatch print a plain job id that can be reused in dependencies
    sbatch_args = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--time={args.time}",
        f"--mem={args.mem}",
        f"--cpus-per-task={args.cpus_per_task}",
        f"--output={log_dir / '%x-%A_%a.out'}",
    ]

    # Optional Slurm knobs are included only when the caller supplies them
    if args.qos:
        sbatch_args.append(f"--qos={args.qos}")
    if args.partition:
        sbatch_args.append(f"--partition={args.partition}")
    if args.account:
        sbatch_args.append(f"--account={args.account}")
    return sbatch_args


def write_slurm_script(script_path: Path, command: Sequence[str], *, repo_root: Path, scenario_array: bool) -> None:
    """Write one generated shell script executed by sbatch."""

    # Each generated script is tiny: set environment, optionally set SCENARIO_INDEX, run command
    script_path.parent.mkdir(parents=True, exist_ok=True)
    mpl_config_base_dir = script_path.parent.parent / RUNNER_DIRNAME / "matplotlib"

    # These lines make generated scripts runnable from Slurm regardless of the submit directory
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
        f"export PYTHONPATH=\"{repo_root}${{PYTHONPATH:+:$PYTHONPATH}}\"",
        "export MPLBACKEND=Agg",
        f"export HJEEDS_MPLCONFIG_BASE={shlex.quote(str(mpl_config_base_dir))}",
        'export HJEEDS_MPLCONFIG_JOB="${SLURM_JOB_ID:-local}"',
        'if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then',
        '  export HJEEDS_MPLCONFIG_JOB="${HJEEDS_MPLCONFIG_JOB}_${SLURM_ARRAY_TASK_ID}"',
        "fi",
        'export MPLCONFIGDIR="${MPLCONFIGDIR:-${HJEEDS_MPLCONFIG_BASE}/${HJEEDS_MPLCONFIG_JOB}}"',
        "mkdir -p \"$MPLCONFIGDIR\"",
        'export HJEEDS_THREAD_COUNT="${SLURM_CPUS_PER_TASK:-1}"',
        'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$HJEEDS_THREAD_COUNT}"',
        'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$HJEEDS_THREAD_COUNT}"',
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$HJEEDS_THREAD_COUNT}"',
        'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$HJEEDS_THREAD_COUNT}"',
        'export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$HJEEDS_THREAD_COUNT}"',
    ]

    # Multi-scenario runners read SCENARIO_INDEX to decide which scenario this array task owns
    if scenario_array:
        lines.append('export SCENARIO_INDEX="${SLURM_ARRAY_TASK_ID}"')

    # The actual experiment command is appended last after environment setup
    lines.append(shlex.join(command))
    lines.append("")

    # Mark the script executable so users can inspect or rerun it manually if needed
    script_path.write_text("\n".join(lines))
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)


def submit_sbatch(command: Sequence[str], *, dry_run: bool) -> str:
    """Submit or print one sbatch command and return its job id."""

    # Dry-run uses a fixed placeholder so dependency strings remain readable without pretending to be real ids
    printable_command = shlex.join(command)
    if dry_run:
        print(f"[dry-run] {printable_command}")
        return str(DRY_RUN_JOB_ID_PLACEHOLDER)

    # Real Slurm submissions rely on --parsable, so stdout should be the job id
    result = subprocess.run(command, check=True, capture_output=True, text=True)

    stdout_lines = result.stdout.strip().splitlines()
    if len(stdout_lines) != 1:
        raise RuntimeError(f"Expected one sbatch --parsable stdout line, got: {result.stdout!r}")

    # --parsable may return "job_id;cluster_name"; dependency strings only need the job id
    return stdout_lines[0].split(";", 1)[0]


def build_slurm_zip_command(
    args: argparse.Namespace,
    *,
    seed: int,
    num_seeds: int,
    output_root: Path,
    zip_path: Path,
) -> list[str]:
    """Build the command run by the final Slurm zip job."""

    # The final Slurm job calls this same script in zip-only mode
    return [
        args.python_bin,
        str(Path(__file__).resolve()),
        "--mode",
        "zip-only",
        "--seed",
        str(seed),
        "--num-seeds",
        str(num_seeds),
        "--output-root",
        str(output_root),
        "--zip-path",
        str(zip_path),
    ]


def submit_slurm_single_experiment(
    spec: ExperimentSpec,
    *,
    args: argparse.Namespace,
    log_dir: Path,
    seed: int,
    num_seeds: int,
    output_root: Path,
    repo_root: Path,
    dry_run: bool,
) -> str:
    """Submit one non-array Slurm experiment and return its job id."""

    # Base command computes one full experiment locally, or one scenario when SCENARIO_INDEX is set
    base_command = experiment_command(
        args.python_bin,
        spec,
        seed=seed,
        num_seeds=num_seeds,
        output_root=output_root,
    )

    # Single-job experiments such as the baseline do not need an array or aggregation step
    script_path = slurm_script_path(output_root, spec.slug)
    if not dry_run:
        write_slurm_script(script_path, base_command, repo_root=repo_root, scenario_array=False)

    # Submit the ordinary job and add it directly to final-zip dependencies
    sbatch_command = slurm_common_args(args, log_dir, f"hjeeds-{spec.slug}")
    sbatch_command.append(str(script_path))
    job_id = submit_sbatch(sbatch_command, dry_run=dry_run)

    # Record the submitted single job
    if not dry_run:
        append_status(
            output_root,
            experiment_slug=spec.slug,
            phase="experiment",
            status="submitted",
            detail=f"job_id={job_id}",
        )
    return job_id


def submit_slurm_array_experiment(
    spec: ExperimentSpec,
    *,
    args: argparse.Namespace,
    log_dir: Path,
    seed: int,
    num_seeds: int,
    output_root: Path,
    repo_root: Path,
    dry_run: bool,
) -> str:
    """Submit one Slurm scenario array plus its dependent aggregation job."""

    # Build the Python command shared by every scenario task in this experiment family
    # This command does not include --aggregate-results, so it runs computation rather than collection
    # In the generated Slurm script, SCENARIO_INDEX will make this command run exactly one scenario
    base_command = experiment_command(
        args.python_bin,
        spec,
        seed=seed, # Pass the global reproducibility seed through to the child runner
        num_seeds=num_seeds, # Number of seeds each scenario task should run sequentially
        output_root=output_root,
    )

    # Choose the path for the generated shell script that scenario array tasks will execute
    # For example: <output-root>/_slurm/population_shape_scenario.sh
    """
    There are 18 total scripts: 1 baselines, 8 scenario-array scripts, 8 aggregation scripts,
    and 1 final zip script. The scenario-array scripts are reused by all tasks in that array.
    So when you run something like "sbatch --array=0-59 _slurm/hyperprior_robustness_scenario.sh",
    Slurm runs that same script 60 times, once per array task, with a different SLURM_ARRAY_TASK_ID,
    which is used to set the SCENARIO_INDEX.
    """
    scenario_script_path = slurm_script_path(output_root, f"{spec.slug}_scenario")

    # If a real run, write the scenario shell script to disk
    if not dry_run:
        write_slurm_script(scenario_script_path, base_command, repo_root=repo_root, scenario_array=True)

    # Build the common sbatch options for this scenario array job
    # This includes sbatch, --parsable, job name, time, memory, CPUs, output log path, and optional QOS fields
    scenario_sbatch = slurm_common_args(args, log_dir, f"hjeeds-{spec.slug}")

    # Add the array range and generated script path to the sbatch command
    # If scenario_count is 20, this becomes --array=0-19
    # Each task gets a different SLURM_ARRAY_TASK_ID, which becomes SCENARIO_INDEX inside the script
    scenario_sbatch.extend([f"--array=0-{spec.scenario_count - 1}", str(scenario_script_path)])

    # Submit the scenario array job, or print it in dry-run mode
    # The returned id is the Slurm array job id in real mode, or the placeholder id in dry-run mode
    scenario_job_id = submit_sbatch(scenario_sbatch, dry_run=dry_run)

    # Record the array job id in the status CSV so the cluster submission can be audited later
    if not dry_run:
        append_status(
            output_root,
            experiment_slug=spec.slug,
            phase="scenario-array",
            status="submitted",
            detail=f"job_id={scenario_job_id}",
        )

    # Build the Python command for the aggregation job
    # This uses the same experiment runner, seed, num-seeds, and output root as the scenario array
    # The difference is aggregate_results=True, which appends --aggregate-results
    aggregate_command = experiment_command(
        args.python_bin,
        spec,
        seed=seed,
        num_seeds=num_seeds,
        output_root=output_root,
        aggregate_results=True,
    )

    # Choose the path for the generated aggregation shell script
    # For example: <output-root>/_slurm/population_shape_aggregate.sh
    aggregate_script_path = slurm_script_path(output_root, f"{spec.slug}_aggregate")

    # If a real run, write the aggregation script to disk
    if not dry_run:
        write_slurm_script(aggregate_script_path, aggregate_command, repo_root=repo_root, scenario_array=False)

    # Build the common sbatch options for the aggregation job
    aggregate_sbatch = slurm_common_args(args, log_dir, f"hjeeds-{spec.slug}-agg")

    # Add the dependency and script path to the aggregation sbatch command
    # afterok:<scenario_job_id> means this job runs only after the scenario array succeeds
    # For Slurm arrays, afterok waits for all tasks in that array job to finish successfully
    aggregate_sbatch.extend([f"--dependency=afterok:{scenario_job_id}", str(aggregate_script_path)])

    # Submit the aggregation job, or print it in dry-run mode
    # This job id is what the final zip job should depend on for this experiment family
    aggregate_job_id = submit_sbatch(aggregate_sbatch, dry_run=dry_run)

    # Record aggregation submission in the status CSV
    # Include both job ids so it is clear which array the aggregation waited for
    if not dry_run:
        append_status(
            output_root,
            experiment_slug=spec.slug,
            phase="aggregation",
            status="submitted",
            detail=f"job_id={aggregate_job_id}, afterok={scenario_job_id}",
        )

    # Return the aggregation job id because that is the terminal job for this experiment family
    # The final zip job waits on this id, not directly on the scenario array id
    return aggregate_job_id


def submit_slurm_final_zip_job(
    *,
    args: argparse.Namespace,
    log_dir: Path,
    seed: int,
    num_seeds: int,
    output_root: Path,
    zip_path: Path,
    repo_root: Path,
    dependency_job_ids: Sequence[str],
    dry_run: bool,
) -> None:
    """Submit the final Slurm job that zips the completed paper output root."""

    zip_command = build_slurm_zip_command(
        args,
        seed=seed,
        num_seeds=num_seeds,
        output_root=output_root,
        zip_path=zip_path,
    )

    # Write the zip script after every experiment submission has been prepared
    zip_script_path = slurm_script_path(output_root, "final_zip")
    if not dry_run:
        write_slurm_script(zip_script_path, zip_command, repo_root=repo_root, scenario_array=False)

    # The final zip job waits for every experiment's terminal job
    zip_sbatch = slurm_common_args(args, log_dir, "hjeeds-final-zip")
    if dependency_job_ids:
        zip_sbatch.append(f"--dependency=afterok:{':'.join(dependency_job_ids)}")
    zip_sbatch.append(str(zip_script_path))
    zip_job_id = submit_sbatch(zip_sbatch, dry_run=dry_run)

    # Record final zip submission only after sbatch returns successfully
    if not dry_run:
        append_status(
            output_root,
            experiment_slug=ZIP_ONLY_EXPERIMENT_SLUG,
            phase="zip",
            status="submitted",
            detail=f"job_id={zip_job_id}, afterok={':'.join(dependency_job_ids)}",
        )


def run_slurm_suite(
    experiment_specs: Sequence[ExperimentSpec],
    *,
    args: argparse.Namespace,
    seed: int,
    num_seeds: int,
    output_root: Path,
    zip_path: Path,
    repo_root: Path,
    dry_run: bool,
) -> None:
    """Submit all paper experiments as Slurm jobs with dependent aggregation and zip jobs."""

    # Resolve and create the Slurm log directory before submitting real jobs
    log_dir = slurm_output_dir(output_root, args.slurm_output_dir)
    if not dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)

    # The final zip job depends on these ids: baseline jobs and aggregation jobs
    dependency_job_ids: list[str] = []

    # Submit each experiment family in manifest order
    for spec in experiment_specs:
        if spec.supports_scenario_array:
            dependency_job_id = submit_slurm_array_experiment(
                spec,
                args=args,
                log_dir=log_dir,
                seed=seed,
                num_seeds=num_seeds,
                output_root=output_root,
                repo_root=repo_root,
                dry_run=dry_run,
            )
        else:
            dependency_job_id = submit_slurm_single_experiment(
                spec,
                args=args,
                log_dir=log_dir,
                seed=seed,
                num_seeds=num_seeds,
                output_root=output_root,
                repo_root=repo_root,
                dry_run=dry_run,
            )
        dependency_job_ids.append(dependency_job_id)

    submit_slurm_final_zip_job(
        args=args,
        log_dir=log_dir,
        seed=seed,
        num_seeds=num_seeds,
        output_root=output_root,
        zip_path=zip_path,
        repo_root=repo_root,
        dependency_job_ids=dependency_job_ids,
        dry_run=dry_run,
    )


def run_zip_only(output_root: Path, zip_path: Path) -> None:
    """Zip an already-computed paper output root."""

    # zip-only is used by the final Slurm dependency job after all experiments finish
    append_status(
        output_root,
        experiment_slug=ZIP_ONLY_EXPERIMENT_SLUG,
        phase="zip",
        status="started",
        detail=str(zip_path),
    )

    # Package whatever is currently in the output root without rerunning experiments
    zip_output_root(output_root, zip_path)

    # Record completion so the status CSV shows the archive was produced
    append_status(
        output_root,
        experiment_slug=ZIP_ONLY_EXPERIMENT_SLUG,
        phase="zip",
        status="complete",
        detail=str(zip_path),
    )
    print(f"[paper-runner] Wrote zip archive to {zip_path.resolve()}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run or submit the full H-JEEDS paper experiment suite."""

    # Parse CLI options first; parse_seed_argument has already normalized "default" to 12345
    args = parse_args(argv)

    # Normalize user-provided path strings into pathlib objects
    output_root = Path(args.output_root)
    zip_path = zip_path_for_output_root(output_root, args.zip_path)

    # The repository root is the directory containing this launcher
    repo_root = Path(__file__).resolve().parent

    # Basic validation happens here before writing manifests or submitting jobs
    if args.num_seeds <= 0:
        raise ValueError(f"num_seeds must be positive. Received {args.num_seeds}.")

    # Protect existing outputs while still allowing pre-created empty directory trees
    ensure_output_root_policy(output_root, mode=args.mode)

    # Always print the high-level plan, even for real runs
    print_suite_summary(
        EXPERIMENT_SPECS,
        mode=args.mode,
        output_root=output_root,
        zip_path=zip_path,
        num_seeds=args.num_seeds,
        seed=args.seed,
    )

    # Flush before launching child processes so the summary appears before child logs
    sys.stdout.flush()

    # Dry-run intentionally avoids writing the manifest or status files
    if args.dry_run:
        print()
        print("Planned commands:")

    # zip-only should preserve the original launch manifest instead of rewriting it
    elif args.mode != "zip-only":
        write_manifest(
            output_root,
            EXPERIMENT_SPECS,
            mode=args.mode,
            seed=args.seed,
            num_seeds=args.num_seeds,
            zip_path=zip_path,
        )

    # zip-only is a packaging mode for already-computed results
    if args.mode == "zip-only":
        if args.dry_run:
            print(f"[dry-run] zip {output_root} -> {zip_path}")
        else:
            run_zip_only(output_root, zip_path)
        return 0

    # Local mode runs the child runners sequentially and then zips the output root
    if args.mode == "local":
        run_local_suite(
            EXPERIMENT_SPECS,
            seed=args.seed,
            num_seeds=args.num_seeds,
            output_root=output_root,
            zip_path=zip_path,
            python_bin=args.python_bin,
            dry_run=args.dry_run,
            repo_root=repo_root,
        )
        return 0

    # The only remaining mode is slurm, which submits jobs and exits after submission
    run_slurm_suite(
        EXPERIMENT_SPECS,
        args=args,
        seed=args.seed,
        num_seeds=args.num_seeds,
        output_root=output_root,
        zip_path=zip_path,
        repo_root=repo_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    # Return main's integer exit code to the shell
    raise SystemExit(main())
