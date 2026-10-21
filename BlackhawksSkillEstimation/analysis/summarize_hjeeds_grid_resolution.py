"""Compare subsample stability across HJEEDS hockey likelihood-grid resolutions.

Each finished cell is a directory with ``manifest.json`` and
``agent_level_results.csv``. The default root is the 3x3 sweep. The completed
3-season 50x100 run is included when its CSV is still present, so season-pool
and resolution effects can be read from one table.

Stability uses the same coefficient of variation as
``summarize_hjeeds_subsample_stability.py``: at a fixed N, lower CV means the
player's estimate moves less across redraws. The score is ``1 / (1 + CV)``.
Both JEEDS and hierarchical posterior means are reported on sigma and on
log-lambda. Player 950160 is one row; the roster median is the median of the
other players' CVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("Data/Hockey/experiments/hjeeds_grid_resolution")
DEFAULT_LEGACY = Path("Data/Hockey/experiments/hjeeds_subsample")
DEFAULT_TARGET_PLAYER_ID = 950160

METRICS = {
    "jeeds_sigma": ("jeeds_status", "jeeds_mean_sigma"),
    "jeeds_log_lambda": ("jeeds_status", "jeeds_mean_log_lambda"),
    "hierarchical_sigma": ("hierarchical_status", "hierarchical_mean_sigma"),
    "hierarchical_log_lambda": ("hierarchical_status", "hierarchical_mean_log_lambda"),
}


def coefficient_of_variation(values: pd.Series) -> dict[str, float]:
    """Mean, sample SD, CV, and stability score for one player's redraws."""

    clean = values.dropna().astype(float)
    count = int(clean.size)
    mean = float(clean.mean()) if count else float("nan")
    standard_deviation = float(clean.std(ddof=1)) if count >= 2 else float("nan")
    if count >= 2 and np.isfinite(mean) and abs(mean) > 0.0 and np.isfinite(standard_deviation):
        cv = float(standard_deviation / abs(mean))
        stability = float(1.0 / (1.0 + cv))
    else:
        cv = float("nan")
        stability = float("nan")
    return {
        "successes": count,
        "mean": mean,
        "sd": standard_deviation,
        "cv": cv,
        "stability_score": stability,
    }


def player_stability(source: pd.DataFrame, n_shots: int) -> pd.DataFrame:
    """Per-player stability at one subsample size."""

    rows: list[dict[str, float | int | str]] = []
    for player_id, player_rows in source.groupby("player_id", sort=True):
        for metric, (status_column, value_column) in METRICS.items():
            selected = player_rows[
                player_rows["n_shots"].eq(n_shots) & player_rows[status_column].eq("ok")
            ][value_column]
            stats = coefficient_of_variation(selected)
            rows.append({
                "player_id": int(player_id),
                "n_shots": int(n_shots),
                "metric": metric,
                **stats,
            })
    return pd.DataFrame(rows)


def _median(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.median())


def comparison_rows(
    source: pd.DataFrame,
    *,
    season_pool: int,
    seasons: list[int],
    n_sigma: int,
    n_lambda: int,
    target_player_id: int,
    n_shots_values: list[int],
) -> list[dict[str, object]]:
    """Target-player and roster-median stability rows for one finished cell."""

    rows: list[dict[str, object]] = []
    season_label = " ".join(str(season) for season in seasons)
    for n_shots in n_shots_values:
        per_player = player_stability(source, n_shots)
        if per_player.empty:
            continue
        for metric in METRICS:
            metric_rows = per_player[per_player["metric"].eq(metric)]
            target = metric_rows[metric_rows["player_id"].eq(target_player_id)]
            roster = metric_rows[~metric_rows["player_id"].eq(target_player_id)]
            roster = roster[roster["successes"].ge(2)]
            if not target.empty:
                target_row = target.iloc[0]
                rows.append({
                    "season_pool": season_pool,
                    "seasons": season_label,
                    "n_sigma": n_sigma,
                    "n_lambda": n_lambda,
                    "n_shots": int(n_shots),
                    "scope": "target",
                    "player_id": int(target_player_id),
                    "n_players": 1,
                    "metric": metric,
                    "successes": int(target_row["successes"]),
                    "mean": float(target_row["mean"]),
                    "sd": float(target_row["sd"]),
                    "cv": float(target_row["cv"]),
                    "stability_score": float(target_row["stability_score"]),
                })
            median_cv = _median(roster["cv"]) if not roster.empty else float("nan")
            rows.append({
                "season_pool": season_pool,
                "seasons": season_label,
                "n_sigma": n_sigma,
                "n_lambda": n_lambda,
                "n_shots": int(n_shots),
                "scope": "roster_median",
                "player_id": "",
                "n_players": int(len(roster)),
                "metric": metric,
                "successes": int(roster["successes"].median()) if not roster.empty else 0,
                "mean": _median(roster["mean"]) if not roster.empty else float("nan"),
                "sd": _median(roster["sd"]) if not roster.empty else float("nan"),
                "cv": median_cv,
                "stability_score": (
                    float(1.0 / (1.0 + median_cv)) if np.isfinite(median_cv) else float("nan")
                ),
            })
    return rows


def run_metadata(run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    sigma_grid = manifest["grid"]["sigma_grid"]
    log_lambda_grid = manifest["grid"]["log_lambda_grid"]
    seasons = [int(season) for season in manifest["seasons"]]
    return {
        "season_pool": len(seasons),
        "seasons": seasons,
        "n_sigma": int(manifest["grid"].get("num_sigma_grid", len(sigma_grid))),
        "n_lambda": int(manifest["grid"].get("num_log_lambda_grid", len(log_lambda_grid))),
        "target_player_id": int(manifest.get("target_player_id", DEFAULT_TARGET_PLAYER_ID)),
    }


def discover_runs(root: Path, legacy: Path | None) -> list[Path]:
    runs: list[Path] = []
    if legacy is not None and (legacy / "agent_level_results.csv").exists() and (legacy / "manifest.json").exists():
        runs.append(legacy)
    if root.exists():
        runs.extend(
            path.parent
            for path in sorted(root.glob("sigma*_lambda*/agent_level_results.csv"))
            if (path.parent / "manifest.json").exists()
        )
    return runs


def summarize_runs(
    runs: list[Path],
    n_shots_values: list[int] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in runs:
        metadata = run_metadata(run_dir)
        source = pd.read_csv(run_dir / "agent_level_results.csv")
        available = sorted(int(value) for value in source["n_shots"].dropna().unique())
        selected = available if n_shots_values is None else [n for n in n_shots_values if n in available]
        rows.extend(comparison_rows(source, n_shots_values=selected, **metadata))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--n-shots", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = discover_runs(args.root, args.legacy)
    if not runs:
        raise SystemExit(f"No finished cells under {args.root} or {args.legacy}.")
    summary = summarize_runs(runs, args.n_shots)
    output = args.output or args.root / "summaries" / "grid_resolution_stability.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"wrote {output} ({len(summary)} rows from {len(runs)} cells)")
    expected = [
        args.root / f"sigma{num_sigma}_lambda{num_lambda}"
        for num_sigma in (50, 100, 250)
        for num_lambda in (50, 100, 250)
    ]
    missing = [str(path) for path in expected if path not in runs and not (path / "agent_level_results.csv").exists()]
    if missing:
        print("cells not finished yet:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
