"""Create exploratory plots from hockey H-JEEDS agent-level result CSVs.

The report is descriptive: these CSVs contain point estimates and shot counts,
but not posterior intervals or observed outcomes.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHODS = (("jeeds", "JEEDS", "#1768AC"), ("hierarchical", "H-JEEDS", "#D1495B"))


def season_label(season: str) -> str:
    return f"{season[:4]}-{season[6:]}" if len(season) == 8 else season


def read_results(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "player_id": row["player_id"],
            "num_shots": int(row["num_shots"]),
            "jeeds_sigma": float(row["jeeds_mean_sigma"]),
            "hierarchical_sigma": float(row["hierarchical_mean_sigma"]),
            "jeeds_log_lambda": float(row["jeeds_mean_log_lambda"]),
            "hierarchical_log_lambda": float(row["hierarchical_mean_log_lambda"]),
        }
        for row in rows
    ]


def values(rows: list[dict], method: str, metric: str) -> np.ndarray:
    return np.asarray([row[f"{method}_{metric}"] for row in rows], dtype=float)


def finish(fig: plt.Figure, output: Path) -> None:
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


METRICS = (
    ("log_lambda", "Decision skill", "log lambda"),
    ("sigma", "Xskill", "sigma"),
)


def plot_season_metric(
    rows: list[dict], season: str, output_dir: Path, metric: str, metric_label: str, axis_label: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    label = season_label(season)
    for method, method_label, color in METHODS:
        axes[0].hist(values(rows, method, metric), bins=24, alpha=0.55, label=method_label, color=color)
    axes[0].set_title(f"{metric_label} estimates: {label}")
    axes[0].set_xlabel(axis_label)
    axes[0].set_ylabel("Players")
    axes[0].legend(frameon=False)

    axes[1].scatter(values(rows, "jeeds", metric), values(rows, "hierarchical", metric), s=15, alpha=0.65, color="#3C3C3C")
    axes[1].set_title("JEEDS vs H-JEEDS")
    axes[1].set_xlabel(f"JEEDS {axis_label}")
    axes[1].set_ylabel(f"H-JEEDS {axis_label}")
    low = min(axes[1].get_xlim()[0], axes[1].get_ylim()[0])
    high = max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])
    axes[1].plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
    axes[1].set_xlim(low, high)
    axes[1].set_ylim(low, high)

    delta = values(rows, "hierarchical", metric) - values(rows, "jeeds", metric)
    axes[2].scatter([row["num_shots"] for row in rows], delta, s=15, alpha=0.65, color="#D1495B")
    axes[2].axhline(0, color="#777777", linestyle="--", linewidth=1)
    axes[2].set_title("H-JEEDS minus JEEDS")
    axes[2].set_xlabel("Shots")
    axes[2].set_ylabel(f"Delta {axis_label}")
    if metric == "sigma":
        axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)
    finish(fig, output_dir / f"season_{season}_{metric}.png")


def plot_season_comparison_metric(
    frames: dict[str, list[dict]], output_dir: Path, metric: str, metric_label: str, axis_label: str
) -> None:
    seasons = sorted(frames)
    if len(seasons) < 2:
        return
    first, second = seasons[:2]
    first_by_id = {row["player_id"]: row for row in frames[first]}
    second_by_id = {row["player_id"]: row for row in frames[second]}
    common = sorted(set(first_by_id) & set(second_by_id))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (method, method_label, color) in zip(axes[0], METHODS):
        first_values = np.asarray([first_by_id[player][f"{method}_{metric}"] for player in common])
        second_values = np.asarray([second_by_id[player][f"{method}_{metric}"] for player in common])
        axis.scatter(first_values, second_values, s=16, alpha=0.7, color=color)
        low = min(first_values.min(), second_values.min())
        high = max(first_values.max(), second_values.max())
        axis.plot([low, high], [low, high], color="#777777", linestyle="--", linewidth=1)
        axis.set_title(f"{method_label}: season-to-season")
        axis.set_xlabel(f"{season_label(first)} {axis_label}")
        axis.set_ylabel(f"{season_label(second)} {axis_label}")
    for axis, (method, method_label, color) in zip(axes[1], METHODS):
        change = np.asarray([second_by_id[player][f"{method}_{metric}"] - first_by_id[player][f"{method}_{metric}"] for player in common])
        shots = np.asarray([second_by_id[player]["num_shots"] for player in common])
        axis.scatter(shots, change, s=16, alpha=0.7, color=color)
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(f"{method_label}: change vs current shots")
        axis.set_xlabel(f"{season_label(second)} shots")
        axis.set_ylabel(f"Change in {axis_label}")
    fig.suptitle(f"{metric_label}: {len(common)} players present in both seasons", y=1.02)
    finish(fig, output_dir / f"season_to_season_{metric}.png")

    with (output_dir / f"top_season_movers_{metric}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "player_id", "first_season", "second_season", f"change_{metric}"])
        for method, _, _ in METHODS:
            changes = sorted(((second_by_id[player][f"{method}_{metric}"] - first_by_id[player][f"{method}_{metric}"], player) for player in common), reverse=True)
            for change, player in changes[:10] + changes[-10:]:
                writer.writerow([method, player, first, second, change])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", default="Data/Hockey/experiments/hjeeds_*/agent_level_results.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("Data/Hockey/experiments/hjeeds_plots"))
    args = parser.parse_args()
    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        parser.error(f"No result CSVs matched {args.input_glob!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = {path.parent.name.removeprefix("hjeeds_"): read_results(path) for path in paths}
    for season, rows in frames.items():
        for metric, metric_label, axis_label in METRICS:
            plot_season_metric(rows, season, args.output_dir, metric, metric_label, axis_label)
    for metric, metric_label, axis_label in METRICS:
        plot_season_comparison_metric(frames, args.output_dir, metric, metric_label, axis_label)
    print(f"Wrote exploratory plots to {args.output_dir} for {len(frames)} seasons.")


if __name__ == "__main__":
    main()