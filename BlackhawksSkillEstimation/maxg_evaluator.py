"""MAXG evaluator for benchmark shots."""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.signal import convolve2d

from BlackhawksSkillEstimation.BlackhawksJEEDS import _BH_Y, _BH_Z, _load_shot_maps_npz
from Environments.Hockey import getAngularHeatmapsPerPlayer as angular_heatmaps
from Environments.Hockey.hockey import getCovMatrix, getNormalDistribution


@dataclass(frozen=True)
class AngularBenchmarkShot:
    event_id: int
    grid_targets_angular: np.ndarray
    grid_utilities: np.ndarray
    mean: list[float]
    delta: list[float]


def load_benchmark(benchmark_dir: Path, tag: str) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    shots_path = benchmark_dir / f"benchmark_shots_{tag}.parquet"
    maps_path = benchmark_dir / f"benchmark_shot_maps_{tag}.npz"

    if not shots_path.exists():
        raise FileNotFoundError(f"Missing benchmark shots file: {shots_path}")
    if not maps_path.exists():
        raise FileNotFoundError(f"Missing benchmark shot maps file: {maps_path}")

    df = pd.read_parquet(shots_path)
    shot_maps = _load_shot_maps_npz(maps_path)
    return df, shot_maps


def _extract_player_id(path: Path) -> int:
    for part in path.parts:
        match = re.match(r"player_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unable to parse player_id from path: {path}")


def discover_ees_csvs(data_dir: Path, season_tag: str, shot_group: str) -> list[Path]:
    if shot_group:
        pattern = f"players/player_*/logs/{shot_group}/intermediate_estimates_{season_tag}.csv"
    else:
        pattern = f"players/player_*/logs/intermediate_estimates_{season_tag}.csv"
    return sorted(data_dir.glob(pattern))


def _extract_model(path: Path) -> str | None:
    """Return 'legacy' or 'new' if path contains the model suffix, else None."""
    for part in path.parts:
        if part.startswith("player_"):
            if "__" in part:
                suffix = part.split("__", 1)[1]
                if suffix in {"legacy", "new"}:
                    return suffix
            return None
    return None


def load_ees_xskills(
    data_dir: Path,
    season_tag: str,
    shot_group: str,
    player_ids: Sequence[int] | None = None,
    model_filter: str | None = None,
) -> pd.DataFrame:
    csv_paths = discover_ees_csvs(data_dir, season_tag, shot_group)
    rows: list[dict[str, object]] = []

    for path in csv_paths:
        pid = _extract_player_id(path)
        model = _extract_model(path)
        if player_ids is not None and pid not in player_ids:
            continue
        if model_filter is not None and model != model_filter:
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        if "expected_execution_skill" in df.columns:
            xskill = float(df["expected_execution_skill"].iloc[-1])
        elif "ees" in df.columns:
            xskill = float(df["ees"].iloc[-1])
        else:
            raise ValueError(f"Expected execution skill column missing in {path}")

        rows.append({"player_id": pid, "xskill_ees": xskill, "csv_path": str(path), "model": model})

    if not rows:
        raise RuntimeError("No EES CSVs found for the requested season/shot group")

    return pd.DataFrame(rows).sort_values("xskill_ees").reset_index(drop=True)


def build_angular_cache(
    shots_df: pd.DataFrame,
    shot_maps: dict[int, dict[str, object]],
) -> list[AngularBenchmarkShot]:
    angular_shots: list[AngularBenchmarkShot] = []

    for _, row in shots_df.iterrows():
        event_id = int(row["event_id"])
        if event_id not in shot_maps:
            raise RuntimeError(f"Missing shot map for event_id {event_id}")

        base_ev = shot_maps[event_id]["value_map"]
        grid_y = shot_maps[event_id].get("grid_y")
        grid_z = shot_maps[event_id].get("grid_z")
        player_location = np.array([float(row["start_x"]), float(row["start_y"])])
        executed_action = np.array([float(row["location_y"]), float(row["location_z"])])

        angular_out = angular_heatmaps.getAngularHeatmap(
            base_ev,
            player_location,
            executed_action,
            grid_y=_BH_Y if grid_y is None else grid_y,
            grid_z=_BH_Z if grid_z is None else grid_z,
        )

        dirs = np.array(angular_out[0])
        elevations = np.array(angular_out[1])
        grid_targets_angular = angular_out[3]
        grid_utilities = angular_out[7]
        skip = bool(angular_out[9])

        if skip:
            raise RuntimeError(f"Benchmark shot {event_id} flagged as angular skip")

        middle = max(0, int(len(dirs) / 2) - 1)
        mean = [float(dirs[middle]), float(elevations[middle])]
        delta = [
            float(abs(dirs[1] - dirs[0])) if len(dirs) > 1 else 0.01,
            float(abs(elevations[1] - elevations[0])) if len(elevations) > 1 else 0.01,
        ]

        angular_shots.append(
            AngularBenchmarkShot(
                event_id=event_id,
                grid_targets_angular=grid_targets_angular,
                grid_utilities=grid_utilities,
                mean=mean,
                delta=delta,
            )
        )

    return angular_shots


def compute_convolved_evs(
    shot: AngularBenchmarkShot,
    xskill: float,
) -> np.ndarray:
    rng = np.random.default_rng(0)

    cov = getCovMatrix([xskill, xskill], 0.0)
    pdf = getNormalDistribution(
        rng,
        cov,
        shot.delta,
        shot.mean,
        shot.grid_targets_angular,
    )
    evs = convolve2d(shot.grid_utilities, pdf, mode="same", fillvalue=0.0)
    return evs


def compute_maxg_sum(
    angular_shots: Iterable[AngularBenchmarkShot],
    xskill: float,
) -> float:
    total = 0.0
    for shot in angular_shots:
        evs = compute_convolved_evs(shot, xskill)
        total += float(np.max(evs))
    return total


def evaluate_maxg(
    angular_shots: list[AngularBenchmarkShot],
    xskill_table: pd.DataFrame,
    benchmark_tag: str,
    season_tag: str,
    shot_group: str,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []

    for _, row in xskill_table.iterrows():
        player_id = int(row["player_id"])
        xskill = float(row["xskill_ees"])
        model = row.get("model") if hasattr(row, "get") else None
        maxg_sum = compute_maxg_sum(angular_shots, xskill)
        print(f"MAXG finished: player {player_id} | maxg_sum={maxg_sum:.4f}")

        results.append(
            {
                "player_id": player_id,
                "xskill_ees": xskill,
                "model": model,
                "maxg_sum": maxg_sum,
                "benchmark_tag": benchmark_tag,
                "season_tag": season_tag,
                "shot_group": shot_group,
                "num_benchmark_shots": len(angular_shots),
            }
        )

    return pd.DataFrame(results)


def _plot_maxg_over_xskill(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "xskill_ees" not in results.columns or "maxg_sum" not in results.columns:
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(results["xskill_ees"], results["maxg_sum"], alpha=0.6, s=20)
    plt.xlabel("xskill (EES)")
    plt.ylabel("MAXG sum")
    plt.title("MAXG vs xskill")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _parse_player_ids(text: str | None) -> list[int] | None:
    if not text:
        return None
    parts = text.replace(",", " ").split()
    return [int(part.strip()) for part in parts if part.strip()]


def _load_player_ids_file(path: Path) -> list[int]:
    raw = path.read_text().replace(",", " ")
    ids: list[int] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.extend(part for part in stripped.replace(",", " ").split() if part)
    return [int(pid) for pid in ids]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MAXG scores on benchmark shots.")
    parser.add_argument(
        "--benchmark-tag",
        required=True,
        help="Required benchmark tag used in filenames (e.g., WS_v1).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("Data/Hockey/benchmarks"),
        help="Directory containing benchmark files.",
    )
    parser.add_argument(
        "--season-tag",
        default="20232024",
        help="Season tag used in intermediate estimates CSVs.",
    )
    parser.add_argument(
        "--shot-group",
        default="wristshot_snapshot",
        help="Shot group tag used in intermediate estimates CSVs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data/Hockey"),
        help="Base directory containing player estimate logs.",
    )
    parser.add_argument(
        "--player-ids",
        default=None,
        help="Comma-separated player IDs to evaluate (default: discover from logs).",
    )
    parser.add_argument(
        "--pids-file",
        type=Path,
        default=None,
        help="Path to a file containing player IDs (whitespace/comma-separated, # comments allowed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey/benchmarks/results"),
        help="Directory to write results CSV.",
    )
    parser.add_argument(
        "--debug",
        choices=("off", "plots", "only"),
        default="off",
        help="Debug mode: off (no plots), plots (plots + evaluation), only (plots only).",
    )
    parser.add_argument(
        "--debug-shots",
        type=int,
        default=5,
        help="Number of benchmark shots to plot.",
    )
    parser.add_argument(
        "--debug-players",
        type=int,
        default=5,
        help="Number of players to plot.",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=Path("Data/Hockey/benchmarks/plots"),
        help="Directory for debug plots.",
    )
    parser.add_argument(
        "--debug-run-tag",
        default="",
        help="Optional suffix for debug plot folder (e.g., debug-run-1).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test on a small subset.",
    )
    parser.add_argument(
        "--smoke-shots",
        type=int,
        default=10,
        help="Number of shots for smoke test.",
    )
    parser.add_argument(
        "--smoke-players",
        type=int,
        default=3,
        help="Number of players for smoke test.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for debug and smoke-test sampling.",
    )
    parser.add_argument(
        "--model-filter",
        choices=("legacy", "new"),
        default=None,
        help="When running against comparison data, filter to 'legacy' or 'new' suffixed player folders.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    shots_df, shot_maps = load_benchmark(args.benchmark_dir, args.benchmark_tag)

    player_ids = _parse_player_ids(args.player_ids)
    if args.pids_file:
        file_ids = _load_player_ids_file(args.pids_file)
        if player_ids is None:
            player_ids = file_ids
        else:
            player_ids = sorted(set(player_ids).union(file_ids))
    xskill_table = load_ees_xskills(
        data_dir=args.data_dir,
        season_tag=args.season_tag,
        shot_group=args.shot_group,
        player_ids=player_ids,
        model_filter=args.model_filter,
    )

    rng = np.random.default_rng(args.rng_seed)
    if args.debug == "only" and args.debug_shots > 0:
        sample_count = min(args.debug_shots, len(shots_df))
        sampled = shots_df.sample(n=sample_count, random_state=args.rng_seed)
        angular_shots = build_angular_cache(sampled, shot_maps)
    else:
        angular_shots = build_angular_cache(shots_df, shot_maps)

    if args.smoke_test:
        if len(angular_shots) > args.smoke_shots:
            pick = rng.choice(len(angular_shots), size=args.smoke_shots, replace=False)
            angular_shots = [angular_shots[i] for i in pick]
        if len(xskill_table) > args.smoke_players:
            xskill_table = xskill_table.sample(
                n=args.smoke_players,
                random_state=args.rng_seed,
            )

    if args.debug in {"plots", "only"}:
        from BlackhawksSkillEstimation import maxg_plots

        suffix = f"_{args.debug_run_tag}" if args.debug_run_tag else ""
        debug_dir = args.debug_output / f"{args.benchmark_tag}_{args.season_tag}_{args.shot_group}{suffix}"
        maxg_plots.generate_debug_plots(
            angular_shots=angular_shots,
            xskill_table=xskill_table,
            output_dir=debug_dir,
            num_shots=args.debug_shots,
            num_players=args.debug_players,
            seed=args.rng_seed,
        )

    if args.debug == "only":
        return

    results = evaluate_maxg(
        angular_shots,
        xskill_table,
        benchmark_tag=args.benchmark_tag,
        season_tag=args.season_tag,
        shot_group=args.shot_group,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"maxg_results_{args.benchmark_tag}_{args.season_tag}_{args.shot_group}.csv"
    results.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    plot_path = output_path.with_name(output_path.stem + "_maxg_over_xskill.png")
    _plot_maxg_over_xskill(results, plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
