#!/usr/bin/env python3
"""Download Statcast CSVs expected by SpacesBaseball.getAllData()."""

from __future__ import annotations

import calendar
import concurrent.futures
import sys
from pathlib import Path

import pandas as pd
from pybaseball import cache, statcast

cache.enable()

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "Data" / "Baseball" / "StatcastData"

# (output filename, list of calendar years to pull)
TARGETS = [
    ("raw21.csv", [2021]),
    ("raw22.csv", [2022]),
    ("raw18_19_20.csv", [2018, 2019, 2020]),
]


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def fetch_range(start: str, end: str) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(statcast, start_dt=start, end_dt=end)
        return future.result()


def fetch_years(years: list[int], outfile: Path) -> None:
    if outfile.is_file():
        print(f"Skip {outfile.name} (already exists)", flush=True)
        return

    frames: list[pd.DataFrame] = []
    for year in years:
        print(f"\n=== {outfile.name}: year {year} ===", flush=True)
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month:02d}-{last_day(year, month):02d}"
            print(f"  {start} -> {end} ...", end=" ", flush=True)
            try:
                df = fetch_range(start, end)
            except Exception as exc:
                print(f"ERROR: {exc}", flush=True)
                continue
            if df is None or df.empty:
                print("empty", flush=True)
                continue
            print(f"{len(df):,} rows", flush=True)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No Statcast rows downloaded for {outfile.name}")

    combined = pd.concat(frames, ignore_index=True)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outfile, index=False)
    print(f"\nWrote {outfile} ({len(combined):,} rows)", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, years in TARGETS:
        fetch_years(years, OUT_DIR / name)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
