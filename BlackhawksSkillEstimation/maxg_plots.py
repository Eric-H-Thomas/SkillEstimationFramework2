"""Debug plotting for MAXG convolution checks."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.maxg_evaluator import AngularBenchmarkShot, compute_convolved_evs


def _select_players_by_quantiles(xskill_table: pd.DataFrame, num_players: int) -> pd.DataFrame:
    if xskill_table.empty:
        return xskill_table

    quantiles = np.linspace(0.05, 0.95, num_players)
    xskills = xskill_table["xskill_ees"].to_numpy()
    indices = []
    for q in quantiles:
        target = np.quantile(xskills, q)
        idx = int(np.argmin(np.abs(xskills - target)))
        indices.append(idx)

    indices = sorted(set(indices))
    return xskill_table.iloc[indices]


def _select_shots(
    angular_shots: Sequence[AngularBenchmarkShot],
    num_shots: int,
    rng: np.random.Generator,
) -> list[AngularBenchmarkShot]:
    if len(angular_shots) <= num_shots:
        return list(angular_shots)
    indices = rng.choice(len(angular_shots), size=num_shots, replace=False)
    return [angular_shots[i] for i in indices]


def _plot_convolution_pair(
    shot: AngularBenchmarkShot,
    pdf_convolved: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = shot.grid_utilities
    vmax = float(max(np.max(original), np.max(pdf_convolved)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        shot.grid_targets_angular[:, :, 0].ravel(),
        shot.grid_targets_angular[:, :, 1].ravel(),
        c=original.ravel(),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=8,
    )
    axes[0].set_title("Original EV")
    axes[0].set_xlabel("Direction")
    axes[0].set_ylabel("Elevation")

    max_idx_pdf = int(np.argmax(pdf_convolved))
    iy_pdf, iz_pdf = np.unravel_index(max_idx_pdf, pdf_convolved.shape)

    axes[1].scatter(
        shot.grid_targets_angular[:, :, 0].ravel(),
        shot.grid_targets_angular[:, :, 1].ravel(),
        c=pdf_convolved.ravel(),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
        s=8,
    )
    axes[1].scatter(
        shot.grid_targets_angular[iy_pdf, iz_pdf, 0],
        shot.grid_targets_angular[iy_pdf, iz_pdf, 1],
        color="black",
        marker="X",
        s=60,
        edgecolors="black",
        label="Max",
    )
    axes[1].set_title("PDF Convolved EV")
    axes[1].set_xlabel("Direction")
    axes[1].set_ylabel("Elevation")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def generate_debug_plots(
    angular_shots: Sequence[AngularBenchmarkShot],
    xskill_table: pd.DataFrame,
    output_dir: Path,
    num_shots: int = 5,
    num_players: int = 5,
    seed: int = 42,
) -> None:
    if not angular_shots or xskill_table.empty:
        return

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = _select_shots(angular_shots, num_shots, rng)
    players = _select_players_by_quantiles(xskill_table, num_players)

    for shot in shots:
        for _, row in players.iterrows():
            player_id = int(row["player_id"])
            xskill = float(row["xskill_ees"])
            pdf_convolved = compute_convolved_evs(shot, xskill)

            title = f"event {shot.event_id} | player {player_id} | xskill {xskill:.4f}"
            filename = f"event_{shot.event_id}_player_{player_id}.png"
            output_path = output_dir / f"event_{shot.event_id}" / filename

            _plot_convolution_pair(shot, pdf_convolved, output_path, title)
