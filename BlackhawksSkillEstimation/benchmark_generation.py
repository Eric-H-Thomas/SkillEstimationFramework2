"""Benchmark shot generation for MAXG evaluation."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from BlackhawksAPI import get_shot_maps_by_event_ids, query_season_shots
from BlackhawksSkillEstimation.BlackhawksJEEDS import (
    MIN_DISTANCE_FROM_NET_FT,
    SHOT_TYPE_GROUPS,
    _BH_Y,
    _BH_Z,
    _NET_CENTER,
    _save_shot_maps_npz,
    add_partition_columns,
)
from Environments.Hockey import getAngularHeatmapsPerPlayer as angular_heatmaps


@dataclass(frozen=True)
class BenchmarkConfig:
    seasons: list[int]
    shot_group: str
    distance_bins: list[float]
    benchmark_size: int
    max_player_fraction: float
    seed: int
    tag: str
    output_dir: Path
    oversample_factor: int = 3


@dataclass
class BenchmarkStats:
    pool_size: int = 0
    kept_after_distance: int = 0
    rejected_player_cap: int = 0
    rejected_angular_skip: int = 0
    rejected_missing_map: int = 0


def _parse_distance_bins(text: str) -> list[float]:
    bins = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(bins) < 2:
        raise ValueError("distance bins must include at least two edges")
    if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError("distance bins must be strictly increasing")
    return bins


def _distance_bin_labels(bins: list[float]) -> list[str]:
    return [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)]


def _add_distance_bins(df: pd.DataFrame, bins: list[float]) -> pd.DataFrame:
    labels = _distance_bin_labels(bins)
    out = df.copy()
    distances = np.hypot(out["start_x"] - _NET_CENTER[0], out["start_y"] - _NET_CENTER[1])
    out["distance_to_net"] = distances
    out["distance_bin"] = pd.cut(
        distances,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    out = out[out["distance_bin"].notna()].copy()
    out["distance_bin"] = out["distance_bin"].astype(str)
    return out


def _allocate_proportional_counts(counts: pd.Series, total: int) -> dict[tuple, int]:
    if counts.empty:
        raise ValueError("No pool counts available for allocation")
    raw = counts / counts.sum() * total
    base = np.floor(raw).astype(int)
    remainder = raw - base
    leftover = int(total - base.sum())

    if leftover > 0:
        top = remainder.sort_values(ascending=False).index[:leftover]
        base.loc[top] += 1

    return {idx: int(val) for idx, val in base.items() if val > 0}


def _is_angular_skip(
    base_ev: np.ndarray,
    player_location: np.ndarray,
    executed_action: np.ndarray,
) -> bool:
    angular_out = angular_heatmaps.getAngularHeatmap(
        base_ev,
        player_location,
        executed_action,
        grid_y=_BH_Y,
        grid_z=_BH_Z,
    )
    return bool(angular_out[9])


def _build_pool(config: BenchmarkConfig, stats: BenchmarkStats) -> pd.DataFrame:
    _, allowed_types, include_null = SHOT_TYPE_GROUPS[config.shot_group]
    pool = query_season_shots(
        seasons=config.seasons,
        shot_types=allowed_types,
        include_null_shot_type=include_null,
    )
    if pool.empty:
        raise RuntimeError("No shots returned for the requested seasons")

    # query_season_shots already returns lowercased columns and filtered shot types.
    pool = add_partition_columns(pool)
    pool = _add_distance_bins(pool, config.distance_bins)

    stats.pool_size = len(pool)

    distances = pool["distance_to_net"]
    pool = pool[distances >= MIN_DISTANCE_FROM_NET_FT].copy()
    stats.kept_after_distance = len(pool)
    return pool


def _sample_benchmark(
    pool: pd.DataFrame,
    config: BenchmarkConfig,
    stats: BenchmarkStats,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], dict[tuple, int]]:
    rng = np.random.default_rng(config.seed)
    stratum_cols = ["partition_geometry", "distance_bin", "season"]
    counts = pool.groupby(stratum_cols).size()
    targets = _allocate_proportional_counts(counts, config.benchmark_size)

    max_player_shots = max(1, int(np.floor(config.max_player_fraction * config.benchmark_size)))
    # pool sizes tracked in _build_pool

    selected_records: list[dict[str, object]] = []
    selected_event_ids: set[int] = set()
    selected_maps: dict[int, dict[str, object]] = {}
    player_counts: dict[int, int] = {}

    grouped = {
        stratum: group.reset_index(drop=True)
        for stratum, group in pool.groupby(stratum_cols)
    }

    for stratum, target in targets.items():
        if target <= 0:
            continue
        group = grouped.get(stratum)
        if group is None or group.empty:
            raise RuntimeError(f"No available shots for stratum {stratum}")

        order = rng.permutation(len(group))
        group = group.iloc[order].reset_index(drop=True)
        idx = 0
        accepted = 0

        while accepted < target:
            remaining = target - accepted
            if idx >= len(group):
                raise RuntimeError(
                    f"Unable to fill stratum {stratum} (needed {target}, got {accepted})."
                )

            batch_size = min(
                len(group) - idx,
                max(remaining * config.oversample_factor, remaining),
            )
            batch = group.iloc[idx : idx + batch_size]
            idx += batch_size

            event_ids = [int(eid) for eid in batch["event_id"].tolist()]
            maps = get_shot_maps_by_event_ids(event_ids)

            for record in batch.to_dict("records"):
                event_id = int(record["event_id"])
                player_id = int(record["player_id"])

                if event_id in selected_event_ids:
                    continue
                if event_id not in maps:
                    stats.rejected_missing_map += 1
                    continue
                if player_counts.get(player_id, 0) >= max_player_shots:
                    stats.rejected_player_cap += 1
                    continue

                base_ev = maps[event_id]["value_map"]
                player_location = np.array([float(record["start_x"]), float(record["start_y"])])
                executed_action = np.array([float(record["location_y"]), float(record["location_z"])])
                if _is_angular_skip(base_ev, player_location, executed_action):
                    stats.rejected_angular_skip += 1
                    continue

                selected_records.append(record)
                selected_event_ids.add(event_id)
                selected_maps[event_id] = maps[event_id]
                player_counts[player_id] = player_counts.get(player_id, 0) + 1
                accepted += 1

                if accepted >= target:
                    break

    if len(selected_records) != config.benchmark_size:
        raise RuntimeError(
            f"Benchmark size mismatch (expected {config.benchmark_size}, got {len(selected_records)})."
        )

    selected_df = pd.DataFrame(selected_records)
    return selected_df, selected_maps, targets


def generate_benchmark(config: BenchmarkConfig) -> dict[str, Path]:
    stats = BenchmarkStats()
    pool = _build_pool(config, stats)

    selected_df, shot_maps, targets = _sample_benchmark(pool, config, stats)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shots_path = output_dir / f"benchmark_shots_{config.tag}.parquet"
    maps_path = output_dir / f"benchmark_shot_maps_{config.tag}.npz"
    meta_path = output_dir / f"benchmark_shots_{config.tag}.provenance.json"

    selected_df.to_parquet(shots_path, engine="pyarrow", compression="snappy")
    _save_shot_maps_npz(shot_maps, maps_path)

    max_player_shots = max(1, int(np.floor(config.max_player_fraction * config.benchmark_size)))
    metadata = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "stats": asdict(stats),
        "max_player_shots": max_player_shots,
        "strata_targets": {str(k): int(v) for k, v in targets.items()},
        "shot_count": len(selected_df),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "shots": shots_path,
        "shot_maps": maps_path,
        "metadata": meta_path,
    }


def _parse_seasons(values: str | list[str]) -> list[int]:
    if isinstance(values, str):
        parts = values.replace(",", " ").split()
    else:
        parts = []
        for value in values:
            parts.extend(value.replace(",", " ").split())
    return [int(part) for part in parts]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate benchmark shots for MAXG evaluation.")
    parser.add_argument(
        "--seasons",
        required=True,
        nargs="+",
        help="Seasons as space- or comma-separated values (e.g., 20212022 20222023 or 20212022,20222023).",
    )
    parser.add_argument(
        "--shot-group",
        default="wristshot_snapshot",
        help="Shot group tag from SHOT_TYPE_GROUPS.",
    )
    parser.add_argument(
        "--distance-bins",
        default="10,20,30,40,200",
        help="Comma-separated distance bin edges in feet.",
    )
    parser.add_argument(
        "--benchmark-size",
        type=int,
        default=1000,
        help="Number of benchmark shots to sample.",
    )
    parser.add_argument(
        "--max-player-fraction",
        type=float,
        default=0.03,
        help="Max fraction of benchmark shots from a single player.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Required benchmark tag used in filenames (e.g., WS_v1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey/benchmarks"),
        help="Output directory for benchmark files.",
    )
    parser.add_argument(
        "--oversample-factor",
        type=int,
        default=3,
        help="Oversample factor per stratum to offset rejection filtering.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = BenchmarkConfig(
        seasons=_parse_seasons(args.seasons),
        shot_group=args.shot_group,
        distance_bins=_parse_distance_bins(args.distance_bins),
        benchmark_size=args.benchmark_size,
        max_player_fraction=args.max_player_fraction,
        seed=args.seed,
        tag=args.tag,
        output_dir=args.output_dir,
        oversample_factor=args.oversample_factor,
    )

    paths = generate_benchmark(config)
    print("Benchmark saved:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
