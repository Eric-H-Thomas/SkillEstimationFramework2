"""Aggregate final MCSE estimates from intermediate CSVs into one table."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

_MCSE_CSV_RE = re.compile(r"^intermediate_estimates_(\d{8})(.*)\.csv$")
_METRIC_COLS = [
    "shot_count",
    "ees_y",
    "ees_z",
    "rho_ees",
    "map_execution_skill_y",
    "map_execution_skill_z",
    "expected_rationality",
    "map_rationality",
]


def _parse_mcse_csv_path(path: Path) -> dict[str, object] | None:
    m = _MCSE_CSV_RE.match(path.name)
    if not m:
        return None
    shot_group = path.parent.name
    season = int(m.group(1))
    suffix = m.group(2)
    return {"season": season, "shot_group": shot_group, "suffix": suffix}


def _read_final_row(csv_path: Path) -> dict[str, object]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    last = rows[-1]
    out: dict[str, object] = {"csv_path": str(csv_path)}
    for col in _METRIC_COLS:
        raw = last.get(col, "")
        if col == "shot_count":
            out[col] = int(float(raw)) if raw not in ("", None) else None
        else:
            try:
                out[col] = float(raw) if raw not in ("", None) else None
            except ValueError:
                out[col] = None
    return out


def discover_mcse_csvs(data_root: Path) -> list[Path]:
    logs_root = data_root / "players"
    if not logs_root.exists():
        return []
    return sorted(logs_root.glob("player_*/logs/mcse/*/intermediate_estimates_*.csv"))


def summarize_mcse_runs(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for csv_path in discover_mcse_csvs(data_root):
        meta = _parse_mcse_csv_path(csv_path)
        if meta is None:
            continue
        player_match = re.search(r"player_(\d+)", str(csv_path))
        if not player_match:
            continue
        payload = _read_final_row(csv_path)
        if not payload:
            continue
        rows.append(
            {
                "player_id": int(player_match.group(1)),
                "season": meta["season"],
                "shot_group": meta["shot_group"],
                "partition_suffix": meta["suffix"] or None,
                **{k: v for k, v in payload.items() if k != "csv_path"},
                "csv_path": payload["csv_path"],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MCSE intermediate CSV final rows")
    parser.add_argument("--data-root", type=Path, default=Path("Data/Hockey"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    df = summarize_mcse_runs(args.data_root.resolve())
    if df.empty:
        print("No MCSE intermediate CSVs found.")
        return

    print(f"Found {len(df)} MCSE result(s).")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".parquet":
            df.to_parquet(args.output, index=False)
        else:
            df.to_csv(args.output, index=False)
        print(f"Wrote {args.output}")
    else:
        print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
