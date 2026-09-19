"""Preflight, worker, and aggregation entry point for cached Blackhawks HJEEDS.

The preliminary hockey hyperprior is manually selected from descriptive prior
Blackhawks JEEDS summaries, not fitted from this run. The descriptive centers
are sigma=0.075 and lambda=10; HJEEDS stores natural logs, so the recorded
means are ln(0.075)=-2.590267 and ln(10)=2.302585. Historical Blackhawks
JEEDS used log10(lambda) in [-1, 3], represented here as ln(lambda) in
[ln(0.1), ln(1000)].
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from HJEEDS.estimation import (
    build_discrete_hierarchical_prior,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from HJEEDS.hockey_pipeline import (
    DEFAULT_HOCKEY_LOG_LAMBDA_GRID,
    DEFAULT_HOCKEY_SIGMA_GRID,
    HOCKEY_PRELIMINARY_HYPERPRIORS,
    _filter_shot_group,
    compute_hockey_log_likelihood_grid,
    fit_hockey_population_hyperparameters,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _player_ids(config: dict[str, Any], season: int, eligible_only: bool) -> list[int]:
    jobs = config.get("cluster_plan", {}).get("jobs", [])
    selected = []
    for job in jobs:
        if int(job.get("season", -1)) != season:
            continue
        if eligible_only and not bool(job.get("eligible", False)):
            continue
        selected.append(int(job["player_id"]))
    return list(dict.fromkeys(selected))


def _grid_metadata(sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> dict[str, Any]:
    return {
        "sigma_grid": sigma_grid.tolist(),
        "log_lambda_grid": log_lambda_grid.tolist(),
        "lambda_bounds": [float(np.exp(log_lambda_grid[0])), float(np.exp(log_lambda_grid[-1]))],
        "log10_lambda_bounds": [float(log_lambda_grid[0] / np.log(10.0)), float(log_lambda_grid[-1] / np.log(10.0))],
    }


def _hyperprior_metadata() -> dict[str, Any]:
    mean_sigma, mean_lambda = HOCKEY_PRELIMINARY_HYPERPRIORS.mean_vector
    return {
        "source": "manual preliminary centers based on descriptive Blackhawks JEEDS summaries; not fitted from this HJEEDS run",
        "mean_sigma": 0.075,
        "mean_lambda": 10.0,
        "mean_log_sigma": float(mean_sigma),
        "mean_log_lambda": float(mean_lambda),
        "log_standard_deviations": [0.5, 1.0],
        "covariance_diagonal": list(HOCKEY_PRELIMINARY_HYPERPRIORS.covariance_diagonal),
        "coordinate_note": "Stored means use natural logs; historical hockey bounds are log10(lambda)=[-1,3].",
    }


def _preflight_player(player_id: int, season: int, data_dir: Path, shot_group: str) -> dict[str, Any]:
    try:
        shots, maps = load_player_data(player_id, [season], data_dir=data_dir)
        filtered = _filter_shot_group(shots, shot_group)
        event_ids = {int(value) for value in filtered["event_id"].dropna()}
        map_ids = {int(value) for value in maps}
        usable = event_ids & map_ids
        finite_maps = 0
        for event_id in usable:
            value_map = np.asarray(maps[event_id].get("value_map"), dtype=float)
            if value_map.ndim == 2 and value_map.size and np.isfinite(value_map).all():
                finite_maps += 1
        return {"player_id": player_id, "season": season, "status": "ready" if finite_maps else "no_usable_maps", "num_shots": int(len(filtered)), "event_ids": len(event_ids), "map_ids": len(map_ids), "usable_map_ids": len(usable), "finite_maps": finite_maps}
    except Exception as exc:
        return {"player_id": player_id, "season": season, "status": "error", "error": str(exc)}


def preflight(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    player_ids = _player_ids(config, args.season, args.eligible_only)
    records = [_preflight_player(pid, args.season, args.data_dir, args.shot_group) for pid in player_ids]
    ready = [record for record in records if record["status"] == "ready"]
    manifest = {"created_at": _now(), "config": str(args.config), "season": args.season, "shot_group": args.shot_group, "eligible_only": args.eligible_only, "data_dir": str(args.data_dir), "output_dir": str(args.output_dir), "players": records, "ready_player_ids": [record["player_id"] for record in ready], "grid": _grid_metadata(args.sigma_grid, args.log_lambda_grid), "hyperprior": _hyperprior_metadata()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"players={len(player_ids)} ready={len(ready)} output={args.output_dir}")
    return 0 if ready else 1


def worker(args: argparse.Namespace) -> int:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    player_ids = manifest["ready_player_ids"]
    index = args.player_index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1")) - 1
    if index < 0 or index >= len(player_ids):
        raise IndexError(f"player index {index} outside 0..{len(player_ids) - 1}")
    player_id = int(player_ids[index])
    output_path = args.output_dir / "likelihoods" / f"player_{player_id}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        print(f"exists player={player_id}")
        return 0
    try:
        shots, maps = load_player_data(player_id, [args.season], data_dir=args.data_dir)
        shots = _filter_shot_group(shots, args.shot_group)
        grid = compute_hockey_log_likelihood_grid(shots, maps, args.sigma_grid, args.log_lambda_grid)
        np.savez_compressed(
            output_path,
            player_id=player_id,
            season=args.season,
            shot_group=args.shot_group,
            num_shots=len(shots),
            log_likelihood_grid=grid,
            sigma_grid=args.sigma_grid,
            log_lambda_grid=args.log_lambda_grid,
        )
        print(f"completed player={player_id} shots={len(shots)}")
        return 0
    except Exception as exc:
        failure_dir = args.output_dir / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / f"player_{player_id}.json").write_text(json.dumps({"player_id": player_id, "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise


def aggregate(args: argparse.Namespace) -> int:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    expected_sigma_grid = np.asarray(manifest["grid"]["sigma_grid"], dtype=float)
    expected_log_lambda_grid = np.asarray(manifest["grid"]["log_lambda_grid"], dtype=float)
    expected_season = int(manifest["season"])
    expected_shot_group = str(manifest["shot_group"])
    retained: list[dict[str, Any]] = []
    grids: list[np.ndarray] = []
    failed: list[dict[str, Any]] = []
    for player_id in manifest["ready_player_ids"]:
        artifact_path = args.output_dir / "likelihoods" / f"player_{int(player_id)}.npz"
        if not artifact_path.exists():
            failed.append({"player_id": int(player_id), "error": "missing likelihood artifact"})
            continue
        try:
            with np.load(artifact_path) as artifact:
                artifact_player_id = int(artifact["player_id"])
                artifact_season = int(artifact["season"])
                artifact_shot_group = str(artifact["shot_group"])
                artifact_sigma_grid = np.asarray(artifact["sigma_grid"], dtype=float)
                artifact_log_lambda_grid = np.asarray(artifact["log_lambda_grid"], dtype=float)
                artifact_grid = np.asarray(artifact["log_likelihood_grid"], dtype=float)
                artifact_num_shots = int(artifact["num_shots"])
        except (KeyError, TypeError, ValueError, OSError) as exc:
            failed.append({"player_id": int(player_id), "error": f"invalid artifact metadata: {exc}"})
            continue

        mismatches = []
        if artifact_player_id != int(player_id):
            mismatches.append(f"player_id={artifact_player_id}")
        if artifact_season != expected_season:
            mismatches.append(f"season={artifact_season}")
        if artifact_shot_group != expected_shot_group:
            mismatches.append(f"shot_group={artifact_shot_group}")
        if not np.array_equal(artifact_sigma_grid, expected_sigma_grid):
            mismatches.append("sigma_grid")
        if not np.array_equal(artifact_log_lambda_grid, expected_log_lambda_grid):
            mismatches.append("log_lambda_grid")
        if artifact_grid.shape != (expected_sigma_grid.size, expected_log_lambda_grid.size):
            mismatches.append(f"grid_shape={artifact_grid.shape}")
        if mismatches:
            failed.append({"player_id": int(player_id), "error": "artifact does not match manifest: " + ", ".join(mismatches)})
            continue

        grids.append(artifact_grid)
        retained.append({"player_id": int(player_id), "num_shots": artifact_num_shots})
    if not grids:
        raise RuntimeError("No likelihood artifacts available for aggregation.")

    fitted = fit_hockey_population_hyperparameters(grids, sigma_grid=expected_sigma_grid, log_lambda_grid=expected_log_lambda_grid, hyperpriors=HOCKEY_PRELIMINARY_HYPERPRIORS)
    prior = build_discrete_hierarchical_prior(fitted, expected_sigma_grid, expected_log_lambda_grid)
    np.save(args.output_dir / "discrete_prior.npy", prior)
    (args.output_dir / "population_fit.json").write_text(json.dumps(_jsonable(fitted), indent=2) + "\n", encoding="utf-8")

    rows = []
    for record, grid in zip(retained, grids):
        jeeds = run_independent_jeeds_baseline(grid, expected_sigma_grid, expected_log_lambda_grid)
        hierarchical = run_hierarchical_estimator(grid, prior, expected_sigma_grid, expected_log_lambda_grid)
        rows.append({**record, "jeeds_status": jeeds.status, "jeeds_mean_sigma": jeeds.posterior_mean_sigma, "jeeds_mean_log_lambda": jeeds.posterior_mean_log_lambda, "hierarchical_status": hierarchical.status, "hierarchical_mean_sigma": hierarchical.posterior_mean_sigma, "hierarchical_mean_log_lambda": hierarchical.posterior_mean_log_lambda})
    with (args.output_dir / "agent_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "failed_players.json").write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
    metadata = {"completed_at": _now(), "retained": len(retained), "failed": len(failed), "hyperprior": manifest["hyperprior"], "grid": manifest["grid"]}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated retained={len(retained)} failed={len(failed)} output={args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "worker", "aggregate"))
    parser.add_argument("--config", type=Path, default=Path("Data/Hockey/jobs/jeeds_forwards_20252026.json"))
    parser.add_argument("--season", type=int, default=20252026)
    parser.add_argument("--shot-group", default="wristshot_snapshot")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Hockey"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/Hockey/experiments/hjeeds_20252026"))
    parser.add_argument("--eligible-only", action="store_true", help="Use the existing >=100-shot eligibility filter.")
    parser.add_argument("--player-index", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sigma-grid", type=float, nargs="+", default=DEFAULT_HOCKEY_SIGMA_GRID.tolist())
    parser.add_argument("--log-lambda-grid", type=float, nargs="+", default=DEFAULT_HOCKEY_LOG_LAMBDA_GRID.tolist())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.sigma_grid = np.asarray(args.sigma_grid, dtype=float)
    args.log_lambda_grid = np.asarray(args.log_lambda_grid, dtype=float)
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "worker":
        return worker(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())