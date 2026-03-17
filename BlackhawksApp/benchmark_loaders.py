"""Benchmark Blackhawks app data loading performance.

Run from repo root:
    python -m BlackhawksApp.benchmark_loaders --player 950160 --seasons 20242025
"""
from __future__ import annotations

import argparse
import statistics
import time

from BlackhawksApp import data_io


def _time_call(fn, repeats: int) -> list[float]:
    samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def _fmt_stats(label: str, samples: list[float]) -> str:
    mean_s = statistics.mean(samples)
    med_s = statistics.median(samples)
    min_s = min(samples)
    max_s = max(samples)
    return (
        f"{label}: n={len(samples)} "
        f"mean={mean_s:.3f}s median={med_s:.3f}s min={min_s:.3f}s max={max_s:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parquet+npz app loaders")
    parser.add_argument("--player", type=int, required=True, help="Player ID")
    parser.add_argument("--seasons", type=int, nargs="*", default=None, help="Season tags, e.g. 20242025")
    parser.add_argument("--data-dir", type=str, default="Data/Hockey", help="Data root")
    parser.add_argument("--repeats", type=int, default=3, help="How many timed runs")
    args = parser.parse_args()

    print(f"Benchmarking player={args.player}, seasons={args.seasons}, data_dir={args.data_dir}")

    players_t = _time_call(lambda: data_io.get_players(args.data_dir), args.repeats)
    print(_fmt_stats("get_players", players_t))

    seasons_t = _time_call(
        lambda: data_io.get_seasons(player_id=args.player, data_dir=args.data_dir),
        args.repeats,
    )
    print(_fmt_stats("get_seasons", seasons_t))

    heatmaps_t = _time_call(
        lambda: data_io.load_heatmaps(
            player_id=args.player,
            seasons=args.seasons,
            data_dir=args.data_dir,
        ),
        args.repeats,
    )
    print(_fmt_stats("load_heatmaps(parquet+npz)", heatmaps_t))

    csvs = data_io.get_intermediate_csvs(player_id=args.player, data_dir=args.data_dir)
    if csvs:
        first_csv = csvs[0]
        estimates_t = _time_call(lambda: data_io.load_estimates(first_csv), args.repeats)
        print(_fmt_stats(f"load_estimates({first_csv.name})", estimates_t))
    else:
        print("No intermediate_estimates CSVs found; skipping estimate-loader benchmark.")


if __name__ == "__main__":
    main()
