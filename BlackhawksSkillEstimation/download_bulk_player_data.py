"""Batch downloader for Blackhawks offline player data.

Reads player IDs from a text file (one ID per line) and saves per-season
parquet/npz artifacts via ``save_player_data`` so runs can be executed offline
on a cluster.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from BlackhawksSkillEstimation.BlackhawksJEEDS import save_player_data
from BlackhawksSkillEstimation.player_cache import lookup_player


def _parse_pid_file(path: Path) -> list[int]:
    """Parse a newline-delimited PID file.

    Ignores blank lines and lines starting with '#'.
    """
    pids: list[int] = []
    for idx, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            pids.append(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid PID on line {idx} in {path}: {line}") from exc
    return pids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Blackhawks player data (parquet + npz) for many players."
    )
    parser.add_argument(
        "--pids-file",
        type=Path,
        default=Path("Data/Hockey/forwards23-25.txt"),
        help="Path to text file containing one PID per line.",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[20232024, 20242025],
        help="Season IDs to persist for each player.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey"),
        help="Output data directory (default: Data/Hockey).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based PID index to start from after file parsing.",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=None,
        help="Optional cap on number of players to process from start-index.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between players to reduce API/DB pressure.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite season artifacts if they already exist.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if not args.pids_file.exists():
        raise FileNotFoundError(f"PID file not found: {args.pids_file}")

    all_pids = _parse_pid_file(args.pids_file)
    if args.start_index < 0 or args.start_index >= len(all_pids):
        raise ValueError(
            f"start-index must be in [0, {len(all_pids) - 1}], got {args.start_index}"
        )

    selected = all_pids[args.start_index :]
    if args.max_players is not None:
        if args.max_players <= 0:
            raise ValueError("max-players must be positive when provided")
        selected = selected[: args.max_players]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Bulk Blackhawks Data Download")
    print("=" * 80)
    print(f"PID file:       {args.pids_file}")
    print(f"Total IDs:      {len(all_pids)}")
    print(f"Start index:    {args.start_index}")
    print(f"Selected IDs:   {len(selected)}")
    print(f"Seasons:        {args.seasons}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Overwrite:      {args.overwrite}")
    print("=" * 80)

    success_count = 0
    no_data_count = 0
    error_count = 0

    total = len(selected)
    for i, pid in enumerate(selected, start=1):
        player_name = lookup_player(player_id=pid)
        display_name = player_name if isinstance(player_name, str) and player_name else "Unknown"
        print(f"\n[{i}/{total}] Player {pid} ({display_name})")
        try:
            saved = save_player_data(
                player_id=pid,
                seasons=args.seasons,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            seasons_saved = sorted(int(s) for s in saved.keys())
            if not seasons_saved:
                no_data_count += 1
                print("  -> no season files saved")
            else:
                success_count += 1
                print(f"  -> saved seasons: {seasons_saved}")
        except Exception as exc:  # continue batch on per-player failures
            error_count += 1
            print(f"  -> ERROR: {exc}")

        if args.sleep_seconds > 0 and i < total:
            time.sleep(args.sleep_seconds)

    print("\n" + "=" * 80)
    print("Download complete")
    print("=" * 80)
    print(f"Successes: {success_count}")
    print(f"No data:   {no_data_count}")
    print(f"Errors:    {error_count}")


if __name__ == "__main__":
    main()
