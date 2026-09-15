#!/usr/bin/env python3
# Paper correspondence: Main `sec:experiments`; reproducibility smoke-test entry point.
"""Run the synthetic H-JEEDS publication experiments from one local entry point.

The seeded synthetic bench consists of the complete 1D-Darts suite and the
2D-Darts baseline. MLB is intentionally audit-only here: its paper endpoint is
the frozen 20-pitcher convergence workflow using the committed, fixed weak
hyperprior, launched through the documented Slurm script. Selecting the MLB
component validates an already-completed paper output bundle; this runner never
launches or claims to complete that endpoint locally.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_publication_results import validate_agent_csv
from HJEEDS.baseball_convergence import (
    validate_complete_convergence_run_metadata,
    validate_paper_bbip20_cohort,
)


DEFAULT_OUTPUT_ROOT = Path("HJEEDS/results/publication_bench")
DEFAULT_ONE_D_SEED = 12345
DEFAULT_TWO_D_SEED = 1000
DEFAULT_ONE_D_EXPECTED_SCENARIOS = 93
COMPONENT_ORDER = ("1d", "2d", "baseball")
DEFAULT_COMPONENTS = ("1d", "2d")
DEFAULT_BASEBALL_RESULTS_DIR = Path(
    "HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed"
)
BASEBALL_AUDIT_ONLY_MESSAGE = (
    "not supported by the local bench; use the Slurm paper convergence "
    "workflow documented in README.md"
)


def parse_seed(value: str) -> int:
    if value.strip().lower() == "default":
        return DEFAULT_ONE_D_SEED
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("Seeds must be nonnegative integers.")
    return seed


def parse_components(raw: str) -> tuple[str, ...]:
    normalized = raw.strip().lower()
    if normalized == "all":
        return COMPONENT_ORDER
    if normalized == "synthetic":
        return ("1d", "2d")
    requested = tuple(piece.strip().lower() for piece in raw.split(",") if piece.strip())
    if not requested:
        raise argparse.ArgumentTypeError("Select at least one component.")
    unknown = set(requested) - set(COMPONENT_ORDER)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown component(s): {', '.join(sorted(unknown))}."
        )
    return tuple(component for component in COMPONENT_ORDER if component in requested)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        type=parse_components,
        default=DEFAULT_COMPONENTS,
        help=(
            "all, synthetic, or a comma-separated subset of 1d,2d,baseball. "
            "Baseball only audits an existing Slurm-produced paper endpoint."
        ),
    )
    parser.add_argument("--num-seeds", type=int, default=500)
    parser.add_argument("--one-d-seed", type=parse_seed, default=DEFAULT_ONE_D_SEED)
    parser.add_argument("--two-d-seed", type=parse_seed, default=DEFAULT_TWO_D_SEED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--baseball-results-dir",
        type=Path,
        default=DEFAULT_BASEBALL_RESULTS_DIR,
        help=(
            "Existing Slurm-produced paper MLB convergence directory to audit when "
            "the baseball component is selected; no MLB computation is launched."
        ),
    )
    parser.add_argument(
        "--reference-one-d-root",
        type=Path,
        help="Optional 500-seed 1D result root to compare against after the run.",
    )
    parser.add_argument(
        "--reference-two-d-root",
        type=Path,
        help="Optional 500-seed 2D result root to compare against after the run.",
    )
    parser.add_argument("--comparison-atol", type=float, default=1e-7)
    parser.add_argument("--comparison-rtol", type=float, default=1e-12)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(piece) for piece in command)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def validate_two_d_output(
    output_root: Path,
    *,
    first_seed: int,
    num_seeds: int,
    expected_agents_per_seed: int = 25,
) -> None:
    """Reject incomplete, failed, or nonconverged local 2D bench output."""

    agent_csv = output_root / "two_d" / "agent_level_results.csv"
    validation = validate_agent_csv(
        agent_csv,
        set(range(first_seed, first_seed + num_seeds)),
        expected_environment="2d",
    )
    if validation.agents_per_seed != expected_agents_per_seed:
        raise ValueError(
            f"{agent_csv}: expected {expected_agents_per_seed} agents per seed, "
            f"found {validation.agents_per_seed}."
        )


def validate_baseball_paper_endpoint(results_dir: Path) -> dict[str, object]:
    """Validate, but never run, the exact external Slurm paper MLB endpoint."""

    resolved_results_dir = results_dir.resolve()
    completion = validate_complete_convergence_run_metadata(resolved_results_dir)
    validate_paper_bbip20_cohort(resolved_results_dir, completion)
    return completion


def build_commands(args: argparse.Namespace, output_root: Path) -> dict[str, list[list[str]]]:
    python_bin = str(args.python_bin.resolve())
    commands: dict[str, list[list[str]]] = {}

    if "1d" in args.components:
        commands["1d"] = [
            [
                python_bin,
                str(REPO_ROOT / "run_hjeeds_paper_experiments.py"),
                "--mode",
                "local",
                "--seed",
                str(args.one_d_seed),
                "--num-seeds",
                str(args.num_seeds),
                "--output-root",
                str(output_root / "one_d"),
                "--python-bin",
                python_bin,
            ]
        ]

    if "2d" in args.components:
        commands["2d"] = [
            [
                python_bin,
                "-m",
                "HJEEDS.darts_hierarchical_vs_jeeds",
                "--seed",
                str(args.two_d_seed),
                "--num-seeds",
                str(args.num_seeds),
                "--environment",
                "2d",
                "--output-dir",
                str(output_root / "two_d"),
            ]
        ]

    if "baseball" in args.components:
        # Deliberately empty: the local bench must never substitute a one-stage,
        # partial run for the paper's fixed-prior convergence workflow.
        commands["baseball"] = []

    return commands


def comparison_command(
    *,
    python_bin: Path,
    candidate_root: Path,
    reference_root: Path,
    output_dir: Path,
    first_seed: int,
    num_seeds: int,
    atol: float,
    rtol: float,
    expected_files: int | None,
) -> list[str]:
    command = [
        str(python_bin.resolve()),
        str(REPO_ROOT / "scripts/compare_seeded_results.py"),
        "--candidate-root",
        str(candidate_root),
        "--reference-root",
        str(reference_root.resolve()),
        "--first-seed",
        str(first_seed),
        "--num-seeds",
        str(num_seeds),
        "--output-dir",
        str(output_dir),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]
    if expected_files is not None:
        command.extend(("--expected-files", str(expected_files)))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive.")
    if args.comparison_atol < 0 or args.comparison_rtol < 0:
        raise ValueError("Comparison tolerances must be nonnegative.")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.dry_run:
        raise FileExistsError(
            f"Output root is nonempty: {output_root}. Use a fresh path for each bench run."
        )
    commands = build_commands(args, output_root)

    comparisons: list[tuple[str, list[str]]] = []
    if "1d" in args.components and args.reference_one_d_root is not None:
        comparisons.append(
            (
                "1d",
                comparison_command(
                    python_bin=args.python_bin,
                    candidate_root=output_root / "one_d",
                    reference_root=args.reference_one_d_root,
                    output_dir=output_root / "verification/one_d",
                    first_seed=args.one_d_seed,
                    num_seeds=args.num_seeds,
                    atol=args.comparison_atol,
                    rtol=args.comparison_rtol,
                    expected_files=DEFAULT_ONE_D_EXPECTED_SCENARIOS,
                ),
            )
        )
    if "2d" in args.components and args.reference_two_d_root is not None:
        comparisons.append(
            (
                "2d",
                comparison_command(
                    python_bin=args.python_bin,
                    candidate_root=output_root / "two_d",
                    reference_root=args.reference_two_d_root,
                    output_dir=output_root / "verification/two_d",
                    first_seed=args.two_d_seed,
                    num_seeds=args.num_seeds,
                    atol=args.comparison_atol,
                    rtol=args.comparison_rtol,
                    expected_files=1,
                ),
            )
        )

    print("=== H-JEEDS publication bench ===", flush=True)
    print(f"Components: {', '.join(args.components)}", flush=True)
    if {"1d", "2d"}.intersection(args.components):
        print(f"Synthetic seeds per scenario: {args.num_seeds}", flush=True)
    if "1d" in args.components:
        print(f"1D seeds: {args.one_d_seed}..{args.one_d_seed + args.num_seeds - 1}", flush=True)
    if "2d" in args.components:
        print(f"2D seeds: {args.two_d_seed}..{args.two_d_seed + args.num_seeds - 1}", flush=True)
    print(f"Output root: {output_root}", flush=True)
    for component in args.components:
        if component == "baseball":
            print(
                "[baseball] audit-only: "
                f"{args.baseball_results_dir.resolve()} ({BASEBALL_AUDIT_ONLY_MESSAGE})",
                flush=True,
            )
        for command in commands[component]:
            print(f"[{component}] {command_text(command)}", flush=True)
    for label, command in comparisons:
        print(f"[compare-{label}] {command_text(command)}", flush=True)
    if args.dry_run:
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest = {
        "started_utc": started,
        "completed_utc": None,
        "components": list(args.components),
        "num_synthetic_seeds": (
            args.num_seeds if {"1d", "2d"}.intersection(args.components) else None
        ),
        "one_d_seeds": (
            list(range(args.one_d_seed, args.one_d_seed + args.num_seeds))
            if "1d" in args.components
            else []
        ),
        "two_d_seeds": (
            list(range(args.two_d_seed, args.two_d_seed + args.num_seeds))
            if "2d" in args.components
            else []
        ),
        "baseball_results_dir": (
            str(args.baseball_results_dir.resolve()) if "baseball" in args.components else None
        ),
        "commands": {
            component: [command_text(command) for command in commands[component]]
            for component in args.components
        },
        "component_mode": {
            component: (
                "external-paper-endpoint-audit" if component == "baseball" else "local-run"
            )
            for component in args.components
        },
        "component_status": {component: "pending" for component in args.components},
        "comparison_status": {label: "pending" for label, _command in comparisons},
    }
    manifest_path = output_root / "publication_bench_manifest.json"
    write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("MPLCONFIGDIR", str(output_root / "_matplotlib"))
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment.setdefault(variable, "1")

    failed = False
    for component in args.components:
        manifest["component_status"][component] = (
            "auditing external Slurm endpoint" if component == "baseball" else "running"
        )
        write_json(manifest_path, manifest)
        try:
            if component == "baseball":
                validate_baseball_paper_endpoint(args.baseball_results_dir)
            else:
                for command in commands[component]:
                    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
            if component == "2d":
                validate_two_d_output(
                    output_root,
                    first_seed=args.two_d_seed,
                    num_seeds=args.num_seeds,
                )
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as error:
            return_code = getattr(error, "returncode", "validation")
            if component == "baseball":
                manifest["component_status"][component] = (
                    f"{BASEBALL_AUDIT_ONLY_MESSAGE}; validation failed: {error}"
                )
            else:
                manifest["component_status"][component] = (
                    f"failed ({return_code}): {error}"
                )
            print(f"[{component}] failed: {error}", file=sys.stderr, flush=True)
            failed = True
            write_json(manifest_path, manifest)
            break
        manifest["component_status"][component] = (
            "validated external Slurm paper endpoint"
            if component == "baseball"
            else "complete"
        )
        write_json(manifest_path, manifest)

    if not failed:
        for label, command in comparisons:
            manifest["comparison_status"][label] = "running"
            write_json(manifest_path, manifest)
            completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
            if completed.returncode == 0:
                manifest["comparison_status"][label] = "matched"
            else:
                manifest["comparison_status"][label] = f"differences ({completed.returncode})"
                failed = True
            write_json(manifest_path, manifest)

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"Bench manifest: {manifest_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
