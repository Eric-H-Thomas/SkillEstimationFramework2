#!/usr/bin/env python3
"""Download Statcast CSVs expected by SpacesBaseball.getAllData()."""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import hashlib
import json
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

# Regular-season months that must contain Statcast rows. Other queried months may
# legitimately be empty, but every monthly request must still complete successfully.
EXPECTED_NONEMPTY_MONTHS = {
    2018: set(range(3, 11)),
    2019: set(range(3, 11)),
    2020: set(range(7, 11)),
    2021: set(range(4, 11)),
    2022: set(range(4, 11)),
}


def last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def fetch_range(start: str, end: str) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(statcast, start_dt=start, end_dt=end)
        return future.result()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_path_for(outfile: Path) -> Path:
    return outfile.with_suffix(".download_manifest.json")


def _validate_month_frame(df: pd.DataFrame, *, year: int, month: int) -> None:
    if df is None:
        raise RuntimeError(f"Statcast returned None for {year}-{month:02d}.")
    if df.empty:
        if month in EXPECTED_NONEMPTY_MONTHS[year]:
            raise RuntimeError(
                f"Statcast returned no rows for expected in-season month {year}-{month:02d}."
            )
        return
    if "game_date" not in df.columns:
        raise ValueError(f"Statcast response for {year}-{month:02d} lacks game_date.")
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Statcast response for {year}-{month:02d} contains invalid game_date values.")
    if not ((dates.dt.year == year) & (dates.dt.month == month)).all():
        bad = dates.loc[~((dates.dt.year == year) & (dates.dt.month == month))].iloc[0]
        raise ValueError(
            f"Statcast response for {year}-{month:02d} contains out-of-range date {bad}."
        )


def _validate_existing_download(outfile: Path, years: list[int]) -> bool:
    manifest_path = manifest_path_for(outfile)
    if not outfile.is_file() or not manifest_path.is_file():
        return False
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_queries = {(year, month) for year in years for month in range(1, 13)}
    actual_queries = {
        (int(row["year"]), int(row["month"])) for row in payload.get("monthly_queries", [])
    }
    if actual_queries != expected_queries:
        return False
    if payload.get("csv_sha256") != sha256(outfile):
        return False
    if int(payload.get("csv_bytes", -1)) != outfile.stat().st_size:
        return False
    for row in payload["monthly_queries"]:
        year, month, rows = int(row["year"]), int(row["month"]), int(row["rows"])
        if month in EXPECTED_NONEMPTY_MONTHS[year] and rows <= 0:
            return False
    return True


def fetch_years(years: list[int], outfile: Path, *, overwrite: bool = False) -> None:
    if outfile.is_file():
        if not overwrite and _validate_existing_download(outfile, years):
            print(f"Skip {outfile.name} (validated complete download)", flush=True)
            return
        if not overwrite:
            raise FileExistsError(
                f"{outfile} exists without a valid complete-download manifest. "
                "Inspect it, then rerun with --overwrite."
            )

    frames: list[pd.DataFrame] = []
    query_records: list[dict[str, object]] = []
    for year in years:
        print(f"\n=== {outfile.name}: year {year} ===", flush=True)
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month:02d}-{last_day(year, month):02d}"
            print(f"  {start} -> {end} ...", end=" ", flush=True)
            try:
                df = fetch_range(start, end)
            except Exception as exc:
                raise RuntimeError(f"Statcast download failed for {start} through {end}: {exc}") from exc
            _validate_month_frame(df, year=year, month=month)
            row_count = int(len(df))
            query_records.append(
                {"year": year, "month": month, "start": start, "end": end, "rows": row_count}
            )
            if df.empty:
                print("empty", flush=True)
                continue
            print(f"{len(df):,} rows", flush=True)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"No Statcast rows downloaded for {outfile.name}")

    combined = pd.concat(frames, ignore_index=True)
    dates = pd.to_datetime(combined["game_date"], errors="raise")
    actual_years = set(int(value) for value in dates.dt.year.unique())
    if actual_years != set(years):
        raise ValueError(f"Combined {outfile.name} contains years {actual_years}; expected {set(years)}.")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_suffix(outfile.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(outfile)
    manifest = {
        "outfile": outfile.name,
        "years": years,
        "monthly_queries": query_records,
        "rows": int(len(combined)),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "csv_bytes": outfile.stat().st_size,
        "csv_sha256": sha256(outfile),
    }
    manifest_path_for(outfile).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {outfile} ({len(combined):,} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing CSVs and their completion manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, years in TARGETS:
        fetch_years(years, OUT_DIR / name, overwrite=args.overwrite)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
