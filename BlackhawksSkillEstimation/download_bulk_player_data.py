"""Batch downloader for Blackhawks offline player data.

Reads player IDs from a text file (one ID per line) and saves per-season
parquet/npz artifacts via ``save_player_data`` so runs can be executed offline
on a cluster.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from BlackhawksSkillEstimation.BlackhawksJEEDS import save_player_data


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


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    parser.add_argument(
        "--manifest-file",
        type=Path,
        default=Path("Data/Hockey/download_manifest_forwards23-25.json"),
        help="Where to write JSON run summary and failures.",
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
    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"Manifest file:  {args.manifest_file}")
    print("=" * 80)

    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total = len(selected)
    for i, pid in enumerate(selected, start=1):
        print(f"\n[{i}/{total}] Player {pid}")
        try:
            saved = save_player_data(
                player_id=pid,
                seasons=args.seasons,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            seasons_saved = sorted(int(s) for s in saved.keys())
            if not seasons_saved:
                failures.append(
                    {
                        "pid": pid,
                        "status": "no_data",
                        "error": "No season files were saved.",
                        "timestamp": _timestamp(),
                    }
                )
                print("  -> no season files saved")
            else:
                successes.append(
                    {
                        "pid": pid,
                        "status": "ok",
                        "seasons_saved": seasons_saved,
                        "timestamp": _timestamp(),
                    }
                )
                print(f"  -> saved seasons: {seasons_saved}")
        except Exception as exc:  # continue batch on per-player failures
            failures.append(
                {
                    "pid": pid,
                    "status": "error",
                    "error": str(exc),
                    "timestamp": _timestamp(),
                }
            )
            print(f"  -> ERROR: {exc}")

        if args.sleep_seconds > 0 and i < total:
            time.sleep(args.sleep_seconds)

    manifest = {
        "run_started_utc": _timestamp(),
        "pids_file": str(args.pids_file),
        "total_ids_in_file": len(all_pids),
        "start_index": args.start_index,
        "selected_ids": len(selected),
        "seasons": args.seasons,
        "output_dir": str(args.output_dir),
        "overwrite": args.overwrite,
        "success_count": len(successes),
        "failure_count": len(failures),
        "successes": successes,
        "failures": failures,
        "run_finished_utc": _timestamp(),
    }
    args.manifest_file.write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 80)
    print("Download complete")
    print("=" * 80)
    print(f"Successes: {len(successes)}")
    print(f"Failures:  {len(failures)}")
    print(f"Manifest:  {args.manifest_file}")


if __name__ == "__main__":
    main()
