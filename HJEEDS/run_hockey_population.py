"""Preflight, worker, and aggregation entry point for cached Blackhawks HJEEDS.

One run fits a single population prior over a roster. With ``--season`` each
player's likelihood uses that season's shots; with ``--seasons`` it pools every
listed season the player has data for into one likelihood per player.

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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.shot_map_format import (
    ShotMapFormatError,
    check_shot_map_files,
    player_shot_map_paths,
)
from HJEEDS.estimation import (
    build_discrete_hierarchical_prior,
    run_hierarchical_estimator,
    run_independent_jeeds_baseline,
)
from HJEEDS.hockey_pipeline import (
    HOCKEY_PRELIMINARY_HYPERPRIORS,
    _filter_shot_group,
    compute_hockey_log_likelihood_grid,
    fit_hockey_population_hyperparameters,
)
from HJEEDS.run_hockey_subsample import resolve_skill_grids


DEFAULT_SEASON = 20252026


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


def _player_ids(config: dict[str, Any], seasons: list[int], eligible_only: bool) -> list[int]:
    wanted = set(seasons)
    jobs = config.get("cluster_plan", {}).get("jobs", [])
    selected = []
    for job in jobs:
        if int(job.get("season", -1)) not in wanted:
            continue
        if eligible_only and not bool(job.get("eligible", False)):
            continue
        selected.append(int(job["player_id"]))
    return list(dict.fromkeys(selected))


def _grid_metadata(sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> dict[str, Any]:
    return {
        "num_sigma_grid": int(len(sigma_grid)),
        "num_log_lambda_grid": int(len(log_lambda_grid)),
        "sigma_grid": sigma_grid.tolist(),
        "log_lambda_grid": log_lambda_grid.tolist(),
        "lambda_bounds": [float(np.exp(log_lambda_grid[0])), float(np.exp(log_lambda_grid[-1]))],
        "log10_lambda_bounds": [float(log_lambda_grid[0] / np.log(10.0)), float(log_lambda_grid[-1] / np.log(10.0))],
    }


def _hyperprior_metadata() -> dict[str, Any]:
    hyperpriors = HOCKEY_PRELIMINARY_HYPERPRIORS
    mean_log_sigma, mean_log_lambda = hyperpriors.mean_vector
    return {
        "source": "manual hockey-wide hyperprior in HJEEDS.hockey_pipeline; not fitted from this HJEEDS run",
        "mean_sigma": float(np.exp(mean_log_sigma)),
        "mean_lambda": float(np.exp(mean_log_lambda)),
        "mean_log_sigma": float(mean_log_sigma),
        "mean_log_lambda": float(mean_log_lambda),
        "log_standard_deviations": [float(np.sqrt(value)) for value in hyperpriors.covariance_diagonal],
        "covariance_diagonal": list(hyperpriors.covariance_diagonal),
        "tau_eta_median": float(np.exp(hyperpriors.log_tau_eta_mean)),
        "log_tau_eta_sd": float(hyperpriors.log_tau_eta_sd),
        "tau_rho_median": float(np.exp(hyperpriors.log_tau_rho_mean)),
        "log_tau_rho_sd": float(hyperpriors.log_tau_rho_sd),
        "coordinate_note": "Stored means use natural logs; historical hockey bounds are log10(lambda)=[-1,3].",
    }


def _normalized_posterior(log_weights: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(log_weights)
    if not finite.any():
        return None
    weights = np.zeros_like(log_weights, dtype=float)
    weights[finite] = np.exp(log_weights[finite] - log_weights[finite].max())
    return weights / weights.sum()


def _marginal_spread(marginal: np.ndarray, grid: np.ndarray) -> tuple[float, float, float]:
    """Posterior SD and 90% interval, spreading each cell's mass evenly between grid midpoints."""
    mean = float(marginal @ grid)
    sd = float(np.sqrt(marginal @ (grid - mean) ** 2))
    edges = np.concatenate([grid[:1], 0.5 * (grid[:-1] + grid[1:]), grid[-1:]])
    cdf = np.concatenate([[0.0], np.cumsum(marginal)])
    q05, q95 = np.interp([0.05, 0.95], cdf, edges)
    return sd, float(q05), float(q95)


def _estimate_columns(prefix: str, estimate: Any, log_weights: np.ndarray, sigma_grid: np.ndarray, log_lambda_grid: np.ndarray) -> dict[str, Any]:
    posterior = _normalized_posterior(log_weights) if estimate.status == "ok" else None
    sd_sigma = sigma_q05 = sigma_q95 = sd_log_lambda = log_lambda_q05 = log_lambda_q95 = None
    if posterior is not None:
        sd_sigma, sigma_q05, sigma_q95 = _marginal_spread(posterior.sum(axis=1), sigma_grid)
        sd_log_lambda, log_lambda_q05, log_lambda_q95 = _marginal_spread(posterior.sum(axis=0), log_lambda_grid)
    return {
        f"{prefix}_status": estimate.status,
        f"{prefix}_mean_sigma": estimate.posterior_mean_sigma,
        f"{prefix}_sd_sigma": sd_sigma,
        f"{prefix}_sigma_q05": sigma_q05,
        f"{prefix}_sigma_q95": sigma_q95,
        f"{prefix}_map_sigma": estimate.map_sigma,
        f"{prefix}_mean_log_lambda": estimate.posterior_mean_log_lambda,
        f"{prefix}_sd_log_lambda": sd_log_lambda,
        f"{prefix}_log_lambda_q05": log_lambda_q05,
        f"{prefix}_log_lambda_q95": log_lambda_q95,
        f"{prefix}_map_log_lambda": estimate.map_log_lambda,
    }


def _available_seasons(player_id: int, seasons: list[int], data_dir: Path) -> list[int]:
    data_subdir = data_dir / "players" / f"player_{player_id}" / "data"
    return [
        season
        for season in seasons
        if (data_subdir / f"shots_{season}.parquet").exists() and (data_subdir / f"shot_maps_{season}.npz").exists()
    ]


def _preflight_player(player_id: int, seasons: list[int], data_dir: Path, shot_group: str) -> dict[str, Any]:
    available = _available_seasons(player_id, seasons, data_dir)
    if not available:
        return {"player_id": player_id, "seasons": [], "status": "no_data", "error": "no requested season has both shots and shot-map files"}
    try:
        shots, maps = load_player_data(player_id, available, data_dir=data_dir)
        filtered = _filter_shot_group(shots, shot_group)
        event_ids = {int(value) for value in filtered["event_id"].dropna()}
        map_ids = {int(value) for value in maps}
        usable = event_ids & map_ids
        finite_maps = 0
        for event_id in usable:
            value_map = np.asarray(maps[event_id].get("value_map"), dtype=float)
            if value_map.ndim == 2 and value_map.size and np.isfinite(value_map).all():
                finite_maps += 1
        return {"player_id": player_id, "seasons": available, "status": "ready" if finite_maps else "no_usable_maps", "num_shots": int(len(filtered)), "event_ids": len(event_ids), "map_ids": len(map_ids), "usable_map_ids": len(usable), "finite_maps": finite_maps}
    except ShotMapFormatError:
        raise
    except Exception as exc:
        return {"player_id": player_id, "seasons": available, "status": "error", "error": str(exc)}


def _manifest_seasons(manifest: dict[str, Any]) -> list[int]:
    if "seasons" in manifest:
        return [int(season) for season in manifest["seasons"]]
    return [int(manifest["season"])]


def _seasons_by_player(manifest: dict[str, Any]) -> dict[int, list[int]]:
    default = _manifest_seasons(manifest)
    return {
        int(record["player_id"]): [int(season) for season in record.get("seasons", default)]
        for record in manifest["players"]
    }


def _artifact_seasons(artifact: Any) -> list[int]:
    if "seasons" in artifact.files:
        return [int(season) for season in np.atleast_1d(artifact["seasons"])]
    return [int(artifact["season"])]


def preflight(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    player_ids = _player_ids(config, args.seasons, args.eligible_only)
    try:
        map_format = check_shot_map_files(
            path for player_id in player_ids for path in player_shot_map_paths(args.data_dir, player_id, args.seasons)
        )
    except ShotMapFormatError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2
    records = [_preflight_player(pid, args.seasons, args.data_dir, args.shot_group) for pid in player_ids]
    ready = [record for record in records if record["status"] == "ready"]
    manifest = {
        "created_at": _now(),
        "config": str(args.config),
        "seasons": args.seasons,
        "shot_group": args.shot_group,
        "eligible_only": args.eligible_only,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "shot_map_format": map_format,
        "players": records,
        "ready_player_ids": [record["player_id"] for record in ready],
        "grid": _grid_metadata(args.sigma_grid, args.log_lambda_grid),
        "hyperprior": _hyperprior_metadata(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"players={len(player_ids)} ready={len(ready)} seasons={args.seasons} output={args.output_dir}")
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
    seasons = _seasons_by_player(manifest)[player_id]
    shot_group = str(manifest["shot_group"])
    sigma_grid = np.asarray(manifest["grid"]["sigma_grid"], dtype=float)
    log_lambda_grid = np.asarray(manifest["grid"]["log_lambda_grid"], dtype=float)
    try:
        shots, maps = load_player_data(player_id, seasons, data_dir=Path(manifest["data_dir"]))
        shots = _filter_shot_group(shots, shot_group)
        grid = compute_hockey_log_likelihood_grid(shots, maps, sigma_grid, log_lambda_grid)
        np.savez_compressed(
            output_path,
            player_id=player_id,
            seasons=np.asarray(seasons, dtype=np.int64),
            shot_group=shot_group,
            num_shots=len(shots),
            log_likelihood_grid=grid,
            sigma_grid=sigma_grid,
            log_lambda_grid=log_lambda_grid,
        )
        print(f"completed player={player_id} seasons={seasons} shots={len(shots)}")
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
    expected_shot_group = str(manifest["shot_group"])
    seasons_by_player = _seasons_by_player(manifest)
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
                artifact_seasons = _artifact_seasons(artifact)
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
        if sorted(artifact_seasons) != sorted(seasons_by_player[int(player_id)]):
            mismatches.append(f"seasons={artifact_seasons}")
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
        retained.append({
            "player_id": int(player_id),
            "num_shots": artifact_num_shots,
            "num_seasons": len(artifact_seasons),
            "seasons": ";".join(str(season) for season in sorted(artifact_seasons)),
        })
    if not grids:
        raise RuntimeError("No likelihood artifacts available for aggregation.")

    fitted = fit_hockey_population_hyperparameters(grids, sigma_grid=expected_sigma_grid, log_lambda_grid=expected_log_lambda_grid, hyperpriors=HOCKEY_PRELIMINARY_HYPERPRIORS)
    prior = build_discrete_hierarchical_prior(fitted, expected_sigma_grid, expected_log_lambda_grid)
    np.save(args.output_dir / "discrete_prior.npy", prior)
    (args.output_dir / "population_fit.json").write_text(json.dumps(_jsonable(fitted), indent=2) + "\n", encoding="utf-8")

    with np.errstate(divide="ignore"):
        log_prior = np.log(prior)
    rows = []
    for record, grid in zip(retained, grids):
        jeeds = run_independent_jeeds_baseline(grid, expected_sigma_grid, expected_log_lambda_grid)
        hierarchical = run_hierarchical_estimator(grid, prior, expected_sigma_grid, expected_log_lambda_grid)
        rows.append({
            **record,
            **_estimate_columns("jeeds", jeeds, grid, expected_sigma_grid, expected_log_lambda_grid),
            **_estimate_columns("hierarchical", hierarchical, grid + log_prior, expected_sigma_grid, expected_log_lambda_grid),
        })
    with (args.output_dir / "agent_level_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "failed_players.json").write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "completed_at": _now(),
        "retained": len(retained),
        "failed": len(failed),
        "seasons": _manifest_seasons(manifest),
        "shot_map_format": manifest.get("shot_map_format"),
        "hyperprior": _hyperprior_metadata(),
        "grid": manifest["grid"],
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"aggregated retained={len(retained)} failed={len(failed)} output={args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "worker", "aggregate"))
    parser.add_argument("--config", type=Path, default=Path("Data/Hockey/jobs/jeeds_forwards_20252026.json"))
    parser.add_argument("--season", type=int, default=None, help=f"Single season (default {DEFAULT_SEASON} when --seasons is not given).")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Pool these seasons into one likelihood per player.")
    parser.add_argument("--shot-group", default="wristshot_snapshot")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/Hockey"))
    parser.add_argument("--output-dir", type=Path, default=Path("Data/Hockey/experiments/hjeeds_20252026"))
    parser.add_argument("--eligible-only", action="store_true", help="Use the existing >=100-shot eligibility filter.")
    parser.add_argument("--player-index", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-sigma-grid", type=int, default=None)
    parser.add_argument("--num-log-lambda-grid", type=int, default=None)
    parser.add_argument("--sigma-grid", type=float, nargs="+", default=None)
    parser.add_argument("--log-lambda-grid", type=float, nargs="+", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.season is not None and args.seasons is not None:
        parser.error("Pass only one of --season and --seasons.")
    if args.seasons is None:
        args.seasons = [args.season if args.season is not None else DEFAULT_SEASON]
    args.seasons = list(dict.fromkeys(args.seasons))
    try:
        args.sigma_grid, args.log_lambda_grid = resolve_skill_grids(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "worker":
        return worker(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
