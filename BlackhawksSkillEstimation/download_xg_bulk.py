"""Resumable bulk downloader for new-model xG offline data.

Usage:
    python -m BlackhawksSkillEstimation.download_xg_bulk [args]

This script downloads shots parquet + shot_maps npz for players listed in a
PID file (one per line). It is resumable via a JSON checkpoint file.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import sys

from BlackhawksSkillEstimation.BlackhawksJEEDS import save_player_data


DEFAULT_SEASONS = [20212022, 20222023, 20232024, 20242025, 20252026]


def load_pids(pids_file: Path) -> List[int]:
    pids: List[int] = []
    with open(pids_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            try:
                p = int(s)
            except Exception:
                # tolerate lines like "950160  # comment"
                first = s.split()[0]
                try:
                    p = int(first)
                except Exception:
                    raise ValueError(f"Could not parse PID from line: {line!r}")
            pids.append(p)
    return pids


def read_state(state_file: Path):
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception:
        return None


def write_state(state_file: Path, state: dict):
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def make_checkpoint(
    pids_file: str,
    output_dir: str,
    seasons: List[int],
    maps_source: str,
    value_column: str,
    total_pids: int,
    next_index: int,
    last_completed_index: int | None,
    last_completed_pid: int | None,
    successes: int,
    errors: int,
    no_data: int,
):
    return {
        "pids_file": pids_file,
        "output_dir": output_dir,
        "seasons": seasons,
        "maps_source": maps_source,
        "value_column": value_column,
        "total_pids": total_pids,
        "next_index": next_index,
        "last_completed_index": last_completed_index,
        "last_completed_pid": last_completed_pid,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "successes": successes,
        "errors": errors,
        "no_data": no_data,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resumable downloader for new xG player data")
    parser.add_argument("--pids-file", default="Data/Hockey/forwards23-25.txt")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_SEASONS,
        help="Seasons to download (e.g. 20232024)",
    )
    parser.add_argument("--output-dir", default="Hockey", help="Output root under Data/")
    parser.add_argument(
        "--start-index", type=int, default=None, help="0-based index to start from (overrides checkpoint)")
    parser.add_argument("--max-players", type=int, default=None, help="Max players to process (smoke-test)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing season files")
    parser.add_argument(
        "--state-file",
        default=str(Path("Data") / "Hockey_xg_new" / "download_state.json"),
        help="Checkpoint JSON file path",
    )
    parser.add_argument("--value-column", default="expected_goals")
    args = parser.parse_args(argv)

    pids_file = Path(args.pids_file)
    if not pids_file.exists():
        print(f"PID file not found: {pids_file}")
        return 2

    pids = load_pids(pids_file)
    total = len(pids)
    if total == 0:
        print("No PIDs found in file.")
        return 0

    state_file = Path(args.state_file)
    state = read_state(state_file) if args.start_index is None else None

    if args.start_index is not None:
        start_index = args.start_index
    else:
        if state and isinstance(state.get("next_index"), int):
            start_index = int(state["next_index"])
        else:
            start_index = 0

    if start_index < 0 or start_index >= total:
        print(f"start-index {start_index} out of range for {total} PIDs")
        return 2

    output_root = Path("Data") / args.output_dir

    # Counters and checkpoint bookkeeping
    successes = state.get("successes", 0) if state else 0
    errors = state.get("errors", 0) if state else 0
    no_data = state.get("no_data", 0) if state else 0
    last_completed_index = state.get("last_completed_index") if state else None
    last_completed_pid = state.get("last_completed_pid") if state else None

    maps_source = "new"
    value_column = args.value_column

    try:
        for idx in range(start_index, total):
            if args.max_players is not None and (idx - start_index) >= args.max_players:
                break

            pid = pids[idx]
            print(f"({idx}/{total}) player {pid} — resume with --start-index {idx}")

            try:
                res = save_player_data(
                    player_id=pid,
                    seasons=list(args.seasons),
                    output_dir=output_root,
                    overwrite=args.overwrite,
                    maps_source=maps_source,
                    value_column=value_column,
                )
            except Exception as e:
                print(f"  ERROR downloading player {pid}: {e}")
                errors += 1
                # Update checkpoint to next index (skip this player on resume)
                next_index = idx + 1
                chk = make_checkpoint(
                    pids_file=str(pids_file),
                    output_dir=str(args.output_dir),
                    seasons=list(args.seasons),
                    maps_source=maps_source,
                    value_column=value_column,
                    total_pids=total,
                    next_index=next_index,
                    last_completed_index=last_completed_index,
                    last_completed_pid=last_completed_pid,
                    successes=successes,
                    errors=errors,
                    no_data=no_data,
                )
                write_state(state_file, chk)
                # continue with next player
                continue

            # Interpret result
            if not res:
                print(f"  No data saved for player {pid}.")
                no_data += 1
            else:
                successes += 1

            last_completed_index = idx
            last_completed_pid = pid
            next_index = idx + 1

            chk = make_checkpoint(
                pids_file=str(pids_file),
                output_dir=str(args.output_dir),
                seasons=list(args.seasons),
                maps_source=maps_source,
                value_column=value_column,
                total_pids=total,
                next_index=next_index,
                last_completed_index=last_completed_index,
                last_completed_pid=last_completed_pid,
                successes=successes,
                errors=errors,
                no_data=no_data,
            )
            write_state(state_file, chk)

    except KeyboardInterrupt:
        # On clean interrupt, set next_index to current idx (retry current player)
        cur = locals().get("idx", start_index)
        resume_index = cur
        chk = make_checkpoint(
            pids_file=str(pids_file),
            output_dir=str(args.output_dir),
            seasons=list(args.seasons),
            maps_source=maps_source,
            value_column=value_column,
            total_pids=total,
            next_index=resume_index,
            last_completed_index=last_completed_index,
            last_completed_pid=last_completed_pid,
            successes=successes,
            errors=errors,
            no_data=no_data,
        )
        write_state(state_file, chk)
        print(f"Interrupted. Resume with --start-index {resume_index}")
        return 1

    print("Download run complete.")
    print(f"  successes={successes}, errors={errors}, no_data={no_data}")
    print(f"  checkpoint written to: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())