"""Summarize player-level pooled-resampling stability for the HJEEDS run.

The CSV has pooled random redraws, not season-labeled fits.  This therefore reports
resampling stability, not a seasonal ICC: at a chosen N, lower CV means that a
player's estimate moves less across redraws.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = Path("Data/Hockey/experiments/hjeeds_subsample/agent_level_results.csv")
DEFAULT_N = 400

METRICS = {
    "jeeds": ("jeeds_status", "jeeds_mean_log_lambda"),
    "hierarchical": ("hierarchical_status", "hierarchical_mean_sigma"),
}


def summarize(source: pd.DataFrame, n_shots: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for player_id, player_rows in source.groupby("player_id", sort=True):
        row: dict[str, float | int] = {"player_id": int(player_id)}
        successful = player_rows[player_rows["jeeds_status"].eq("ok") & player_rows["hierarchical_status"].eq("ok")]
        row["successful_rows"] = int(len(successful))
        for label, (status_column, value_column) in METRICS.items():
            values = player_rows[
                player_rows["n_shots"].eq(n_shots) & player_rows[status_column].eq("ok")
            ][value_column].dropna().astype(float)
            count = int(values.size)
            mean = float(values.mean()) if count else float("nan")
            standard_deviation = float(values.std(ddof=1)) if count >= 2 else float("nan")
            coefficient_of_variation = (
                standard_deviation / abs(mean)
                if count >= 2 and np.isfinite(mean) and abs(mean) > 0
                else float("nan")
            )
            row[f"{label}_n{n_shots}_successes"] = count
            row[f"{label}_n{n_shots}_mean"] = mean
            row[f"{label}_n{n_shots}_sd"] = standard_deviation
            row[f"{label}_n{n_shots}_cv"] = coefficient_of_variation
            row[f"{label}_n{n_shots}_stability_score"] = (
                1.0 / (1.0 + coefficient_of_variation)
                if np.isfinite(coefficient_of_variation)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--n-shots", type=int, default=DEFAULT_N)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = summarize(pd.read_csv(args.csv), args.n_shots)
    output = args.output or args.csv.parent / "summaries" / "player_resampling_stability.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"wrote {output} ({len(summary)} players, N={args.n_shots})")
    print(summary.sort_values("successful_rows", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()