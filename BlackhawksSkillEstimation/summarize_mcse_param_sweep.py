"""Rank a finished MCSE parameter sweep.

Reads the per-job JSON files and writes a setting-level ranking plus a
player-level table. A setting is ranked only when every player passes the
health gates. The absolute JEEDS MAXG gap is added when that table is already
on disk, and it does not change the order.

Thresholds default to the values stored in the sweep config. Override them
here to retune the ranking without rerunning the array.

Examples
--------

    python -m BlackhawksSkillEstimation.summarize_mcse_param_sweep \\
      --config Data/Hockey/jobs/mcse_param_sweep_stage1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from BlackhawksSkillEstimation.mcse_param_sweep import (
    DEFAULT_THRESHOLDS,
    find_jeeds_maxg,
    load_results,
    score_sweep,
    summaries_dir,
)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Wrote {path} ({len(frame)} row(s))")


def summarize_run(
    run_dir: Path,
    *,
    data_root: Path,
    shot_group: str,
    season: int | None,
    expected_player_ids: list[int] | None,
    expected_seeds: list[int] | None,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = load_results(run_dir)
    jeeds = find_jeeds_maxg(data_root, shot_group=shot_group, season=season)
    if jeeds:
        print(f"JEEDS MAXG rows: {len(jeeds)}")
    else:
        print("No JEEDS MAXG table found; the gap column will be empty.")
    return score_sweep(
        records,
        expected_player_ids=expected_player_ids,
        expected_seeds=expected_seeds,
        jeeds_maxg=jeeds,
        thresholds=thresholds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank MCSE parameter-sweep results.")
    parser.add_argument("--config", type=Path, default=None, help="Sweep config JSON. Supplies the run directory, players, seeds, and thresholds.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Directory with results/. Overrides the config output root.")
    parser.add_argument("--data-root", type=Path, default=None, help="Where to look for an existing JEEDS MAXG table.")
    parser.add_argument("--output", type=Path, default=None, help="Ranking CSV. Default: <run-dir>/summaries/ranking.csv")
    parser.add_argument("--neff-ratio-min", type=float, default=None)
    parser.add_argument("--bound-mass-max", type=float, default=None)
    parser.add_argument("--late-drift-max", type=float, default=None)
    args = parser.parse_args()

    config: dict = {}
    if args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))

    run_dir = args.run_dir or (Path(config["output_root"]) if config else None)
    if run_dir is None:
        raise SystemExit("Pass --config or --run-dir.")

    data_root = args.data_root or Path(config.get("data_root", "Data/Hockey"))
    shot_group = str(config.get("shot_group", "wristshot_snapshot"))
    season = config.get("season")
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(config.get("thresholds") or {})
    if args.neff_ratio_min is not None:
        thresholds["neff_ratio_min"] = float(args.neff_ratio_min)
    if args.bound_mass_max is not None:
        thresholds["bound_mass_max"] = float(args.bound_mass_max)
    if args.late_drift_max is not None:
        thresholds["late_drift_max"] = float(args.late_drift_max)

    ranking, by_player = summarize_run(
        Path(run_dir),
        data_root=Path(data_root),
        shot_group=shot_group,
        season=int(season) if season is not None else None,
        expected_player_ids=config.get("player_ids"),
        expected_seeds=config.get("seeds"),
        thresholds=thresholds,
    )

    output = args.output or (summaries_dir(run_dir) / "ranking.csv")
    player_output = output.with_name("ranking_by_player.csv")
    _write_table(ranking, output)
    _write_table(by_player, player_output)

    passing = ranking[ranking["passes"]] if not ranking.empty else ranking
    print(f"Passing settings: {len(passing)} / {len(ranking)}")
    if not passing.empty:
        preview = passing.head(10)[
            ["rank", "setting_key", "mean_seed_spread", "mean_late_drift", "mean_abs_maxg_gap"]
        ]
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
