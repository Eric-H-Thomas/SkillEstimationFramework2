"""Run synchronized Blackhawks subsamples through JEEDS and HJEEDS.

The worker tasks use the same deterministic draw function and base seed as the
existing Blackhawks JEEDS subsample experiment. Aggregation fits one HJEEDS
population prior for each ``(n_shots, seed)`` from the other players, then
applies that prior to every player's likelihood grid, including player 950160.

Typical Slurm workflow::

    python -m HJEEDS.run_hockey_subsample preflight --output-dir OUT
    sbatch run_hjeeds_hockey_subsample.sbatch OUT preflight
    sbatch --dependency=afterany:JOB run_hjeeds_hockey_subsample.sbatch OUT aggregate
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
from BlackhawksSkillEstimation.player_subsample_stability import (
    DEFAULT_BASE_SEED,
    DEFAULT_DATA_ROOT,
    draw_event_ids,
    filter_shot_group,
    sort_chronologically,
    subset_pool,
)
from HJEEDS.estimation import build_discrete_hierarchical_prior, run_hierarchical_estimator, run_independent_jeeds_baseline
from HJEEDS.hockey_pipeline import (
    DEFAULT_HOCKEY_LOG_LAMBDA_GRID,
    DEFAULT_HOCKEY_SIGMA_GRID,
    HOCKEY_PRELIMINARY_HYPERPRIORS,
    fit_hockey_population_hyperparameters,
)
from HJEEDS.hockey_adapter import compute_hockey_log_likelihood_grid


DEFAULT_PLAYER_FILE = Path("Data/Hockey/forwards23-25.txt")
DEFAULT_OUTPUT_DIR = Path("Data/Hockey/experiments/hjeeds_subsample")
DEFAULT_PLAYER_ID = 950160
DEFAULT_N_SHOTS = (100, 200, 400)
DEFAULT_NUM_SEEDS = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _read_player_ids(path: Path) -> list[int]:
    player_ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value:
            player_ids.append(int(value))
    return list(dict.fromkeys(player_ids))


def _load_pool(player_id: int, seasons: list[int], data_dir: Path, shot_group: str):
    shots, maps = load_player_data(player_id, seasons, data_dir=data_dir)
    pool = sort_chronologically(filter_shot_group(shots, shot_group))
    return pool, maps


def _task_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return list(manifest["tasks"])


def _task_path(output_dir: Path, task: dict[str, Any]) -> Path:
    return output_dir / "likelihoods" / f"n{int(task['n_shots']):04d}" / f"seed{int(task['seed']):04d}" / f"player_{int(task['player_id'])}.npz"


def _grid_metadata(sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> dict[str, Any]:
    return {
        "sigma_grid": sigma_grid.tolist(),
        "log_lambda_grid": log_lambda_grid.tolist(),
        "lambda_bounds": [float(np.exp(log_lambda_grid[0])), float(np.exp(log_lambda_grid[-1]))],
    }


def preflight(args: argparse.Namespace) -> int:
    player_ids = _read_player_ids(args.players_file)
    available: dict[str, int] = {}
    skipped: list[dict[str, Any]] = []
    for player_id in player_ids:
        try:
            pool, _ = _load_pool(player_id, args.seasons, args.data_dir, args.shot_group)
            available[str(player_id)] = int(len(pool))
        except Exception as exc:
            skipped.append({"player_id": player_id, "reason": str(exc)})

    tasks: list[dict[str, Any]] = []
    skipped_n: dict[str, list[int]] = {}
    for n_shots in args.n_shots:
        for seed in range(args.num_seeds):
            for player_id in player_ids:
                count = available.get(str(player_id), 0)
                if count < n_shots:
                    skipped_n.setdefault(str(player_id), []).append(int(n_shots))
                    continue
                tasks.append({"player_id": player_id, "n_shots": int(n_shots), "seed": int(seed)})

    manifest = {
        "created_at": _now(),
        "players_file": str(args.players_file),
        "player_ids": player_ids,
        "target_player_id": args.target_player_id,
        "seasons": args.seasons,
        "shot_group": args.shot_group,
        "data_dir": str(args.data_dir),
        "base_seed": args.base_seed,
        "n_shots": args.n_shots,
        "num_seeds": args.num_seeds,
        "exclude_target_from_prior": not args.include_target_in_prior,
        "available_pool_sizes": available,
        "skipped_players": skipped,
        "skipped_n_shots": skipped_n,
        "grid": _grid_metadata(args.sigma_grid, args.log_lambda_grid),
        "tasks": tasks,
        "hyperprior_source": "HOCKEY_PRELIMINARY_HYPERPRIORS from HJEEDS.hockey_pipeline",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    print(f"players={len(player_ids)} available={len(available)} tasks={len(tasks)} output={args.output_dir}")
    return 0 if tasks else 1


def worker(args: argparse.Namespace) -> int:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks = _task_records(manifest)
    index = args.task_index
    if index is None:
        index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1")) - 1
    if index < 0 or index >= len(tasks):
        raise IndexError(f"task index {index} outside 0..{len(tasks) - 1}")
    task = tasks[index]
    output_path = _task_path(args.output_dir, task)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force:
        print(f"exists task={index} path={output_path}")
        return 0

    pool, maps = _load_pool(int(task["player_id"]), manifest["seasons"], Path(manifest["data_dir"]), manifest["shot_group"])
    event_ids = draw_event_ids(
        [int(value) for value in pool["event_id"]],
        n_shots=int(task["n_shots"]),
        seed=int(task["seed"]),
        base_seed=int(manifest["base_seed"]),
    )
    subset = subset_pool(pool, event_ids)
    grid = compute_hockey_log_likelihood_grid(
        subset,
        maps,
        np.asarray(manifest["grid"]["sigma_grid"], dtype=float),
        np.asarray(manifest["grid"]["log_lambda_grid"], dtype=float),
    )
    np.savez_compressed(
        output_path,
        player_id=int(task["player_id"]),
        n_shots=int(task["n_shots"]),
        seed=int(task["seed"]),
        event_ids=np.asarray(event_ids, dtype=np.int64),
        log_likelihood_grid=grid,
    )
    print(f"completed task={index} player={task['player_id']} n={task['n_shots']} seed={task['seed']}")
    return 0


def _load_artifact(path: Path, sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> tuple[np.ndarray, int]:
    with np.load(path) as artifact:
        grid = np.asarray(artifact["log_likelihood_grid"], dtype=float)
        num_shots = int(artifact["n_shots"])
    expected_shape = (len(sigma_grid), len(log_lambda_grid))
    if grid.shape != expected_shape:
        raise ValueError(f"{path} has grid shape {grid.shape}; expected {expected_shape}")
    return grid, num_shots


def aggregate(args: argparse.Namespace) -> int:
    manifest = json.loads((args.output_dir / "manifest.json").read_text(encoding="utf-8"))
    sigma_grid = np.asarray(manifest["grid"]["sigma_grid"], dtype=float)
    log_lambda_grid = np.asarray(manifest["grid"]["log_lambda_grid"], dtype=float)
    target_id = int(manifest["target_player_id"])
    player_ids = [int(value) for value in manifest["player_ids"]]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for n_shots in manifest["n_shots"]:
        for seed in range(int(manifest["num_seeds"])):
            grids: dict[int, np.ndarray] = {}
            counts: dict[int, int] = {}
            for player_id in player_ids:
                task = {"player_id": player_id, "n_shots": n_shots, "seed": seed}
                path = _task_path(args.output_dir, task)
                if not path.exists():
                    missing.append({"player_id": player_id, "n_shots": n_shots, "seed": seed})
                    continue
                try:
                    grids[player_id], counts[player_id] = _load_artifact(path, sigma_grid, log_lambda_grid)
                except Exception as exc:
                    missing.append({"player_id": player_id, "n_shots": n_shots, "seed": seed, "error": str(exc)})

            prior_ids = [
                pid
                for pid in grids
                if not manifest["exclude_target_from_prior"] or pid != target_id
            ]
            if not prior_ids or target_id not in grids:
                continue
            try:
                fitted = fit_hockey_population_hyperparameters(
                    [grids[pid] for pid in prior_ids],
                    sigma_grid=sigma_grid,
                    log_lambda_grid=log_lambda_grid,
                    hyperpriors=HOCKEY_PRELIMINARY_HYPERPRIORS,
                )
                prior = build_discrete_hierarchical_prior(fitted, sigma_grid, log_lambda_grid)
            except Exception as exc:
                missing.append({"n_shots": n_shots, "seed": seed, "error": f"prior fit: {exc}"})
                continue

            for player_id, grid in grids.items():
                jeeds = run_independent_jeeds_baseline(grid, sigma_grid, log_lambda_grid)
                hierarchical = run_hierarchical_estimator(grid, prior, sigma_grid, log_lambda_grid)
                rows.append({
                    "player_id": player_id,
                    "n_shots": int(n_shots),
                    "seed": int(seed),
                    "num_shots": counts[player_id],
                    "is_target": player_id == target_id,
                    "prior_player_count": len(prior_ids),
                    "jeeds_status": jeeds.status,
                    "jeeds_mean_sigma": jeeds.posterior_mean_sigma,
                    "jeeds_mean_log_lambda": jeeds.posterior_mean_log_lambda,
                    "hierarchical_status": hierarchical.status,
                    "hierarchical_mean_sigma": hierarchical.posterior_mean_sigma,
                    "hierarchical_mean_log_lambda": hierarchical.posterior_mean_log_lambda,
                    "population_mu_log_sigma": float(fitted["mu"][0]),
                    "population_mu_log_lambda": float(fitted["mu"][1]),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "agent_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["player_id", "n_shots", "seed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "missing_artifacts.json").write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "run_metadata.json").write_text(json.dumps({"completed_at": _now(), "rows": len(rows), "missing": len(missing), "target_player_id": target_id}, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated rows={len(rows)} missing={len(missing)} output={args.output_dir}")
    return 0 if rows else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "worker", "aggregate"))
    parser.add_argument("--players-file", type=Path, default=DEFAULT_PLAYER_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seasons", type=int, nargs="+", default=[20232024, 20242025, 20252026])
    parser.add_argument("--shot-group", default="wristshot_snapshot")
    parser.add_argument("--target-player-id", type=int, default=DEFAULT_PLAYER_ID)
    parser.add_argument("--n-shots", type=int, nargs="+", default=list(DEFAULT_N_SHOTS))
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--include-target-in-prior", action="store_true")
    parser.add_argument("--task-index", type=int, default=None)
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
