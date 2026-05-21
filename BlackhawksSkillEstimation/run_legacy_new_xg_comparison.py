"""Offline legacy vs new xG JEEDS comparison runner.

Runs per-player, per-season JEEDS estimation using only local cache data,
restricting to shots present in both offline datasets. Outputs model-suffixed
player directories under a single comparison root and writes per-job JSON
partials plus a summary aggregation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.BlackhawksJEEDS import estimate_player_skill, load_player_data
from BlackhawksSkillEstimation.phase3_select_ab_players import read_player_ids


DEFAULT_SEASONS: list[int] = [20212022, 20222023, 20232024, 20242025]
DEFAULT_LEGACY_ROOT = Path("Data/Hockey")
DEFAULT_NEW_ROOT = Path("Data/Hockey_xg_new")
DEFAULT_OUTPUT_ROOT = Path("Data/Hockey_model_comparison")
DEFAULT_SHOT_TYPES: tuple[str, ...] = ("wristshot", "snapshot")


def _season_paths(root: Path, player_id: int, season: int) -> tuple[Path, Path]:
    data_dir = root / "players" / f"player_{player_id}" / "data"
    return data_dir / f"shots_{season}.parquet", data_dir / f"shot_maps_{season}.npz"


def _season_has_cache(root: Path, player_id: int, season: int) -> bool:
    shots_path, maps_path = _season_paths(root, player_id, season)
    return shots_path.exists() and maps_path.exists()


def _filter_shots_by_types(df: pd.DataFrame, shot_types: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df_lc = df.rename(columns=str.lower)
    if "shot_type" not in df_lc.columns:
        return df_lc
    shot = df_lc["shot_type"].where(pd.notna(df_lc["shot_type"]))
    shot_lower = shot.astype(str).str.lower()
    mask = shot_lower.isin(shot_types)
    return df_lc[mask].reset_index(drop=True)


def _event_id_set(df: pd.DataFrame) -> set[int]:
    if df.empty:
        return set()
    if "event_id" not in df.columns:
        return set()
    event_ids = pd.to_numeric(df["event_id"], errors="coerce").dropna().astype(int)
    return set(event_ids.tolist())


def _prune_shot_maps(
    shot_maps: dict[int, dict[str, object]],
    event_ids: set[int],
) -> dict[int, dict[str, object]]:
    if not shot_maps:
        return {}
    keep = {int(eid) for eid in event_ids}
    return {int(eid): payload for eid, payload in shot_maps.items() if int(eid) in keep}


def _prune_df_to_maps(df: pd.DataFrame, shot_maps: dict[int, dict[str, object]]) -> pd.DataFrame:
    if df.empty:
        return df
    if not shot_maps:
        return df.iloc[0:0].copy()
    df_lc = df.rename(columns=str.lower)
    if "event_id" not in df_lc.columns:
        return df_lc.iloc[0:0].copy()
    keep_ids = set(shot_maps.keys())
    event_ids = pd.to_numeric(df_lc["event_id"], errors="coerce")
    return df_lc[event_ids.isin(keep_ids)].reset_index(drop=True)


def _load_season_data(root: Path, player_id: int, season: int) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    return load_player_data(player_id=player_id, seasons=[season], data_dir=root)


def _candidate_skills(num_execution_skills: int) -> list[float]:
    return np.linspace(0.004, 0.25, int(num_execution_skills)).tolist()


def _run_worker(
    *,
    player_id: int,
    model: str,
    seasons: list[int],
    legacy_root: Path,
    new_root: Path,
    output_root: Path,
    shot_types: set[str],
    min_shots: int,
    num_execution_skills: int,
    num_planning_skills: int,
    rng_seed: int,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    partials_dir = output_root / "reports" / "partials"
    partials_dir.mkdir(parents=True, exist_ok=True)

    per_season: dict[str, dict[str, Any]] = {}
    combined_frames: list[pd.DataFrame] = []
    combined_shot_maps: dict[int, dict[str, object]] = {}
    eligible_seasons: list[int] = []

    for season in seasons:
        season_key = str(season)
        season_entry: dict[str, Any] = {
            "status": "pending",
            "shots_legacy": 0,
            "shots_new": 0,
            "intersection_shots": 0,
            "shots_model": 0,
        }

        if not _season_has_cache(legacy_root, player_id, season) or not _season_has_cache(new_root, player_id, season):
            season_entry["status"] = "missing_files"
            shots_path, maps_path = _season_paths(legacy_root, player_id, season)
            if not shots_path.exists() or not maps_path.exists():
                season_entry["missing_legacy"] = [
                    str(p) for p in (shots_path, maps_path) if not p.exists()
                ]
            shots_path, maps_path = _season_paths(new_root, player_id, season)
            if not shots_path.exists() or not maps_path.exists():
                season_entry["missing_new"] = [
                    str(p) for p in (shots_path, maps_path) if not p.exists()
                ]
            per_season[season_key] = season_entry
            continue

        try:
            legacy_df, legacy_maps = _load_season_data(legacy_root, player_id, season)
            new_df, new_maps = _load_season_data(new_root, player_id, season)
        except Exception as exc:
            season_entry["status"] = "load_error"
            season_entry["error"] = str(exc)
            per_season[season_key] = season_entry
            continue

        legacy_df = _filter_shots_by_types(legacy_df, shot_types)
        new_df = _filter_shots_by_types(new_df, shot_types)

        legacy_ids = _event_id_set(legacy_df)
        new_ids = _event_id_set(new_df)
        intersection_ids = legacy_ids.intersection(new_ids)

        season_entry["shots_legacy"] = len(legacy_ids)
        season_entry["shots_new"] = len(new_ids)
        season_entry["intersection_shots"] = len(intersection_ids)

        if not intersection_ids:
            season_entry["status"] = "no_overlap"
            per_season[season_key] = season_entry
            continue

        if len(intersection_ids) < min_shots:
            season_entry["status"] = "below_min_shots"
            per_season[season_key] = season_entry
            continue

        legacy_df = legacy_df[legacy_df["event_id"].isin(intersection_ids)].reset_index(drop=True)
        new_df = new_df[new_df["event_id"].isin(intersection_ids)].reset_index(drop=True)
        legacy_maps = _prune_shot_maps(legacy_maps, intersection_ids)
        new_maps = _prune_shot_maps(new_maps, intersection_ids)

        if model == "legacy":
            season_df = legacy_df
            season_maps = legacy_maps
        else:
            season_df = new_df
            season_maps = new_maps

        season_df = _prune_df_to_maps(season_df, season_maps)
        season_entry["shots_model"] = len(season_df)
        if len(season_df) < min_shots:
            season_entry["status"] = "below_min_shots_after_prune"
            per_season[season_key] = season_entry
            continue

        combined_frames.append(season_df)
        combined_shot_maps.update(season_maps)
        eligible_seasons.append(int(season))
        season_entry["status"] = "eligible"
        per_season[season_key] = season_entry

    estimation_error: str | None = None
    estimation_result: dict[str, Any] | None = None

    if eligible_seasons:
        combined_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
        try:
            estimation_result = estimate_player_skill(
                player_id=player_id,
                seasons=eligible_seasons,
                per_season=True,
                candidate_skills=_candidate_skills(num_execution_skills),
                num_planning_skills=num_planning_skills,
                rng_seed=rng_seed,
                return_intermediate_estimates=False,
                save_intermediate_csv=True,
                data_dir=output_root,
                player_dir_name=f"player_{player_id}__{model}",
                confirm=False,
                offline_data=(combined_df, combined_shot_maps),
                shot_group="wristshot_snapshot",
            )
        except Exception as exc:
            estimation_error = str(exc)

    if estimation_result and isinstance(estimation_result.get("per_season_results"), dict):
        for season, season_result in estimation_result["per_season_results"].items():
            season_key = str(season)
            entry = per_season.setdefault(season_key, {})
            entry["eligibility_status"] = entry.get("status")
            entry["status"] = season_result.get("status")
            entry["execution_skill"] = season_result.get("execution_skill")
            entry["rationality"] = season_result.get("rationality")
            entry["ees"] = season_result.get("ees")
            entry["eps"] = season_result.get("eps")
            entry["num_shots"] = season_result.get("num_shots")
            entry["csv_path"] = season_result.get("csv_path")
            entry["error"] = season_result.get("error")

    overall_status = "success"
    if estimation_error:
        overall_status = "estimation_error"
    elif not eligible_seasons:
        overall_status = "no_eligible_seasons"

    payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "player_id": int(player_id),
            "model": model,
            "seasons_requested": [int(s) for s in seasons],
            "seasons_estimated": eligible_seasons,
            "legacy_root": str(legacy_root),
            "new_root": str(new_root),
            "output_root": str(output_root),
            "shot_types": sorted(shot_types),
            "min_shots": int(min_shots),
            "num_execution_skills": int(num_execution_skills),
            "num_planning_skills": int(num_planning_skills),
            "rng_seed": int(rng_seed),
        },
        "player_id": int(player_id),
        "model": model,
        "overall_status": overall_status,
        "estimation_error": estimation_error,
        "per_season": per_season,
    }

    output_path = partials_dir / f"player_{player_id}__{model}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return output_path


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def _summary_stats(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": int(len(df)),
        "execution_skill": {},
        "rationality": {},
    }

    for metric in ("execution_skill", "rationality"):
        x = df[f"{metric}_legacy"].to_numpy(dtype=float)
        y = df[f"{metric}_new"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            summary[metric] = {
                "count": 0,
                "pearson": float("nan"),
                "spearman": float("nan"),
                "mean_delta": float("nan"),
                "mean_abs_delta": float("nan"),
                "weighted_mean_delta": float("nan"),
                "weighted_mean_abs_delta": float("nan"),
            }
            continue

        delta = y[mask] - x[mask]
        abs_delta = np.abs(delta)
        weights = df.loc[mask, "weight"].to_numpy(dtype=float)
        summary[metric] = {
            "count": int(mask.sum()),
            "pearson": _pearson(x[mask], y[mask]),
            "spearman": _spearman(x[mask], y[mask]),
            "mean_delta": float(np.mean(delta)) if delta.size else float("nan"),
            "mean_abs_delta": float(np.mean(abs_delta)) if abs_delta.size else float("nan"),
            "weighted_mean_delta": _weighted_mean(delta, weights),
            "weighted_mean_abs_delta": _weighted_mean(abs_delta, weights),
        }

    return summary


def _choose_weight(row: pd.Series) -> float:
    for key in (
        "intersection_shots_legacy",
        "intersection_shots_new",
        "num_shots_legacy",
        "num_shots_new",
    ):
        value = row.get(key)
        if pd.notna(value) and float(value) > 0:
            return float(value)
    return float("nan")


def _run_summary(
    *,
    output_root: Path,
) -> tuple[Path, Path]:
    reports_dir = output_root / "reports"
    partials_dir = reports_dir / "partials"
    if not partials_dir.exists():
        raise FileNotFoundError(f"Missing partials directory: {partials_dir}")

    rows: list[dict[str, Any]] = []
    for json_path in sorted(partials_dir.glob("player_*__*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        model = payload.get("model") or payload.get("run_metadata", {}).get("model")
        player_id = payload.get("player_id") or payload.get("run_metadata", {}).get("player_id")
        if model is None or player_id is None:
            continue
        per_season = payload.get("per_season", {})
        for season_key, season_data in per_season.items():
            try:
                season = int(season_key)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "player_id": int(player_id),
                    "season": season,
                    "model": str(model),
                    "status": season_data.get("status"),
                    "eligibility_status": season_data.get("eligibility_status"),
                    "execution_skill": season_data.get("execution_skill"),
                    "rationality": season_data.get("rationality"),
                    "ees": season_data.get("ees"),
                    "eps": season_data.get("eps"),
                    "num_shots": season_data.get("num_shots"),
                    "intersection_shots": season_data.get("intersection_shots"),
                    "shots_legacy": season_data.get("shots_legacy"),
                    "shots_new": season_data.get("shots_new"),
                    "shots_model": season_data.get("shots_model"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No partial rows found under {partials_dir}")

    legacy_df = df[df["model"] == "legacy"].drop(columns=["model"]).add_suffix("_legacy")
    legacy_df = legacy_df.rename(columns={"player_id_legacy": "player_id", "season_legacy": "season"})
    new_df = df[df["model"] == "new"].drop(columns=["model"]).add_suffix("_new")
    new_df = new_df.rename(columns={"player_id_new": "player_id", "season_new": "season"})

    merged = legacy_df.merge(new_df, on=["player_id", "season"], how="outer")
    merged["delta_execution_skill"] = (
        merged["execution_skill_new"] - merged["execution_skill_legacy"]
    )
    merged["abs_delta_execution_skill"] = merged["delta_execution_skill"].abs()
    merged["delta_rationality"] = merged["rationality_new"] - merged["rationality_legacy"]
    merged["abs_delta_rationality"] = merged["delta_rationality"].abs()
    merged["weight"] = merged.apply(_choose_weight, axis=1)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"legacy_new_xg_summary_{ts}.csv"
    json_path = reports_dir / f"legacy_new_xg_summary_{ts}.json"

    merged.sort_values(["season", "player_id"], inplace=True, ignore_index=True)
    merged.to_csv(csv_path, index=False)

    paired_mask = merged["execution_skill_legacy"].notna() & merged["execution_skill_new"].notna()
    summary_payload: dict[str, Any] = {
        "run_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "partials_dir": str(partials_dir),
        },
        "counts": {
            "partial_files": int(len(list(partials_dir.glob("player_*__*.json")))),
            "total_rows": int(len(merged)),
            "paired_rows": int(paired_mask.sum()),
        },
        "overall": _summary_stats(merged),
        "per_season": {},
    }

    for season, group in merged.groupby("season"):
        summary_payload["per_season"][str(int(season))] = _summary_stats(group)

    json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    return csv_path, json_path


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Offline legacy vs new xG JEEDS comparison")
    parser.add_argument("--players-file", type=Path, required=True)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Seasons to evaluate (default: 20212022 20222023 20232024 20242025)",
    )
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--shot-types",
        type=str,
        nargs="+",
        default=list(DEFAULT_SHOT_TYPES),
    )
    parser.add_argument("--min-shots", type=int, default=75)
    parser.add_argument("--model", type=str, choices=["legacy", "new"], default="legacy")
    parser.add_argument("--num-execution-skills", type=int, default=250)
    parser.add_argument("--num-planning-skills", type=int, default=250)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["worker", "summary"],
        default="worker",
        help="Run per-player estimation (worker) or aggregate summary (summary)",
    )
    parser.add_argument(
        "--player-id",
        type=int,
        default=None,
        help="Optional single player id for worker mode. Defaults to all players in file.",
    )

    args = parser.parse_args()

    shot_types = {s.strip().lower() for s in args.shot_types if str(s).strip()}
    if not shot_types:
        raise ValueError("At least one shot type is required.")

    if args.mode == "summary":
        csv_path, json_path = _run_summary(output_root=args.output_root)
        print(f"Wrote summary CSV:  {csv_path}")
        print(f"Wrote summary JSON: {json_path}")
        return

    player_ids = read_player_ids(args.players_file)
    if args.player_id is not None:
        if int(args.player_id) not in player_ids:
            raise ValueError(f"Player id {args.player_id} not found in {args.players_file}")
        player_ids = [int(args.player_id)]

    for player_id in player_ids:
        output_path = _run_worker(
            player_id=int(player_id),
            model=str(args.model),
            seasons=list(args.seasons),
            legacy_root=args.legacy_root,
            new_root=args.new_root,
            output_root=args.output_root,
            shot_types=shot_types,
            min_shots=int(args.min_shots),
            num_execution_skills=int(args.num_execution_skills),
            num_planning_skills=int(args.num_planning_skills),
            rng_seed=int(args.rng_seed),
        )
        print(f"Wrote partial JSON: {output_path}")


if __name__ == "__main__":
    _cli()
