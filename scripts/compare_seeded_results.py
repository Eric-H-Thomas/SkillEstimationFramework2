#!/usr/bin/env python3
"""Compare candidate seed-level H-JEEDS CSVs with a larger reference run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence


IDENTITY_FIELDS = ("seed", "environment", "agent_id", "count_bucket", "num_observations")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--num-seeds", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--expected-files", type=int)
    parser.add_argument("--max-reported-differences", type=int, default=500)
    return parser.parse_args(argv)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in IDENTITY_FIELDS)


def read_rows(path: Path, seeds: set[int], *, stop_after_seed_range: bool) -> tuple[list[str], dict[tuple[str, ...], dict[str, str]]]:
    rows: dict[tuple[str, ...], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        max_seed = max(seeds)
        previous_seed: int | None = None
        for row in reader:
            seed = int(row["seed"])
            if previous_seed is not None and seed < previous_seed:
                raise ValueError(f"Seed rows are not sorted in {path}")
            previous_seed = seed
            if seed in seeds:
                key = row_key(row)
                if key in rows:
                    raise ValueError(f"Duplicate row identity {key} in {path}")
                rows[key] = row
            elif stop_after_seed_range and seed > max_seed:
                break
    return fieldnames, rows


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def compare_value(candidate: str, reference: str, *, atol: float, rtol: float) -> tuple[bool, float | None]:
    if candidate == reference:
        return True, 0.0
    candidate_number = parse_number(candidate)
    reference_number = parse_number(reference)
    if candidate_number is None or reference_number is None:
        return False, None
    if math.isnan(candidate_number) and math.isnan(reference_number):
        return True, 0.0
    if math.isinf(candidate_number) or math.isinf(reference_number):
        return candidate_number == reference_number, math.inf
    difference = abs(candidate_number - reference_number)
    return math.isclose(candidate_number, reference_number, rel_tol=rtol, abs_tol=atol), difference


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("Tolerances must be nonnegative")

    candidate_root = args.candidate_root.resolve()
    reference_root = args.reference_root.resolve()
    seeds = set(range(args.first_seed, args.first_seed + args.num_seeds))
    candidate_files = sorted(candidate_root.rglob("agent_level_results.csv"))

    report: dict[str, object] = {
        "candidate_root": str(candidate_root),
        "reference_root": str(reference_root),
        "seeds": sorted(seeds),
        "atol": args.atol,
        "rtol": args.rtol,
        "expected_files": args.expected_files,
        "candidate_files": len(candidate_files),
        "files_compared": 0,
        "rows_compared": 0,
        "cells_compared": 0,
        "exact_string_differences": 0,
        "tolerance_failures": 0,
        "missing_reference_files": [],
        "header_mismatches": [],
        "row_set_mismatches": [],
        "max_numeric_absolute_difference": 0.0,
        "max_numeric_difference_location": None,
        "status": "pending",
    }
    differences: list[dict[str, str]] = []
    exact_differences_by_field: Counter[str] = Counter()
    exact_differences_by_seed: Counter[str] = Counter()
    tolerance_failures_by_field: Counter[str] = Counter()
    tolerance_failures_by_seed: Counter[str] = Counter()
    max_absolute_difference_by_field: dict[str, float] = {}

    if args.expected_files is not None and len(candidate_files) != args.expected_files:
        report["row_set_mismatches"].append(
            {
                "file": "<bench>",
                "candidate": f"{len(candidate_files)} files",
                "reference": f"expected {args.expected_files} files",
            }
        )

    for candidate_path in candidate_files:
        relative = candidate_path.relative_to(candidate_root)
        reference_path = reference_root / relative
        if not reference_path.is_file():
            report["missing_reference_files"].append(relative.as_posix())
            continue

        candidate_fields, candidate_rows = read_rows(
            candidate_path, seeds, stop_after_seed_range=False
        )
        reference_fields, reference_rows = read_rows(
            reference_path, seeds, stop_after_seed_range=True
        )
        report["files_compared"] += 1
        if candidate_fields != reference_fields:
            report["header_mismatches"].append(
                {
                    "file": relative.as_posix(),
                    "candidate": candidate_fields,
                    "reference": reference_fields,
                }
            )
            continue

        candidate_keys = set(candidate_rows)
        reference_keys = set(reference_rows)
        if candidate_keys != reference_keys:
            report["row_set_mismatches"].append(
                {
                    "file": relative.as_posix(),
                    "candidate_only": [list(key) for key in sorted(candidate_keys - reference_keys)[:20]],
                    "reference_only": [list(key) for key in sorted(reference_keys - candidate_keys)[:20]],
                    "candidate_rows": len(candidate_keys),
                    "reference_rows": len(reference_keys),
                }
            )

        for key in sorted(candidate_keys & reference_keys):
            candidate_row = candidate_rows[key]
            reference_row = reference_rows[key]
            report["rows_compared"] += 1
            for field in candidate_fields:
                report["cells_compared"] += 1
                candidate_value = candidate_row[field]
                reference_value = reference_row[field]
                if candidate_value != reference_value:
                    report["exact_string_differences"] += 1
                    exact_differences_by_field[field] += 1
                    exact_differences_by_seed[key[0]] += 1
                matches, absolute_difference = compare_value(
                    candidate_value,
                    reference_value,
                    atol=args.atol,
                    rtol=args.rtol,
                )
                if absolute_difference is not None and absolute_difference > report["max_numeric_absolute_difference"]:
                    report["max_numeric_absolute_difference"] = absolute_difference
                    report["max_numeric_difference_location"] = {
                        "file": relative.as_posix(),
                        "row": list(key),
                        "field": field,
                        "candidate": candidate_value,
                        "reference": reference_value,
                    }
                if absolute_difference is not None:
                    max_absolute_difference_by_field[field] = max(
                        max_absolute_difference_by_field.get(field, 0.0),
                        absolute_difference,
                    )
                if not matches:
                    report["tolerance_failures"] += 1
                    tolerance_failures_by_field[field] += 1
                    tolerance_failures_by_seed[key[0]] += 1
                    if len(differences) < args.max_reported_differences:
                        differences.append(
                            {
                                "file": relative.as_posix(),
                                "row_identity": "|".join(key),
                                "field": field,
                                "candidate": candidate_value,
                                "reference": reference_value,
                                "absolute_difference": "" if absolute_difference is None else repr(absolute_difference),
                            }
                        )

    failure_count = (
        len(report["missing_reference_files"])
        + len(report["header_mismatches"])
        + len(report["row_set_mismatches"])
        + int(report["tolerance_failures"])
    )
    report["exact_string_differences_by_field"] = dict(sorted(exact_differences_by_field.items()))
    report["exact_string_differences_by_seed"] = dict(sorted(exact_differences_by_seed.items()))
    report["tolerance_failures_by_field"] = dict(sorted(tolerance_failures_by_field.items()))
    report["tolerance_failures_by_seed"] = dict(sorted(tolerance_failures_by_seed.items()))
    report["max_numeric_absolute_difference_by_field"] = dict(
        sorted(max_absolute_difference_by_field.items())
    )
    report["status"] = "match" if failure_count == 0 else "differences"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    difference_path = args.output_dir / "comparison_differences.csv"
    with difference_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "file",
                "row_identity",
                "field",
                "candidate",
                "reference",
                "absolute_difference",
            ),
        )
        writer.writeheader()
        writer.writerows(differences)

    print(json.dumps(report, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Differences: {difference_path}", flush=True)
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
