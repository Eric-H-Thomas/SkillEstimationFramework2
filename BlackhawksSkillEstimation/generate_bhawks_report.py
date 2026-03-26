"""Generate BYU-style Blackhawks report artifacts from a PID file.

This script reads player IDs from a text file (one ID per line) and exports:
- byu_results_with_shot_type.csv
- season-comparison ranking tables (PNG)

Default output directory:
Data/Hockey/_bhawks_reports

Examples:
python -m BlackhawksSkillEstimation.generate_bhawks_report \
    --pids-file Data/Hockey/forwards23-25.txt \
    --output-dir Data/Hockey/_bhawks_reports/forwards23-25 \
    --shot-types wristshot_snapshot

python -m BlackhawksSkillEstimation.generate_bhawks_report \
    --pids-file Data/Hockey/infoPlayers.txt \
    --output-dir Data/Hockey/_bhawks_reports/infoPlayers
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from BlackhawksSkillEstimation.plot_intermediate_estimates import (
    compare_execution_rankings_two_seasons_by_shot_type,
)


def _parse_pid_file(path: Path) -> list[int]:
    pids: list[int] = []
    for idx, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            pids.append(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid player ID on line {idx} in {path}: {line}") from exc
    return pids


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate BYU-style Blackhawks report artifacts from a PID file.",
    )
    p.add_argument(
        "--pids-file",
        type=Path,
        required=True,
        help="Path to text file containing one player ID per line.",
    )
    p.add_argument(
        "--season-a",
        type=int,
        default=20232024,
        help="First season used for side-by-side ranking tables.",
    )
    p.add_argument(
        "--season-b",
        type=int,
        default=20242025,
        help="Second season used for side-by-side ranking tables.",
    )
    p.add_argument(
        "--shot-types",
        nargs="+",
        default=["wristshot_snapshot", "slapshot"],
        help="Shot type groups to include (e.g., wristshot_snapshot slapshot).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/Hockey"),
        help="Root data directory where player logs are stored.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey/_bhawks_reports"),
        help="Directory where CSV and PNG table outputs will be saved.",
    )
    p.add_argument(
        "--csv-filename",
        default="byu_results_with_shot_type.csv",
        help="Output CSV filename.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show figures while generating PNGs.",
    )
    return p


def _count_rows(path: Path) -> int:
    with open(path, newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def main() -> None:
    args = _build_parser().parse_args()

    if not args.pids_file.exists():
        raise FileNotFoundError(f"PID file not found: {args.pids_file}")

    pids = _parse_pid_file(args.pids_file)
    if not pids:
        raise ValueError(f"No valid player IDs found in: {args.pids_file}")

    shot_types: Sequence[str] = tuple(args.shot_types)

    outputs = compare_execution_rankings_two_seasons_by_shot_type(
        players=pids,
        shot_types=shot_types,
        season_a=args.season_a,
        season_b=args.season_b,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        csv_filename=args.csv_filename,
        show=args.show,
    )

    csv_path = Path(outputs["csv"])
    png_paths = [Path(p) for p in outputs["pngs"]]

    print("=" * 80)
    print("Blackhawks report generation complete")
    print("=" * 80)
    print(f"PID file:        {args.pids_file}")
    print(f"Players loaded:  {len(pids)}")
    print(f"Seasons:         {args.season_a}, {args.season_b}")
    print(f"Shot types:      {', '.join(shot_types)}")
    print(f"CSV:             {csv_path}")
    print(f"CSV rows:        {_count_rows(csv_path)}")
    for p in png_paths:
        print(f"PNG:             {p}")


if __name__ == "__main__":
    main()
