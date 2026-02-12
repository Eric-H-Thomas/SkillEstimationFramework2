"""Plot intermediate JEEDS estimates over shots.

Visualizes how execution skill and rationality estimates evolve as the
JEEDS estimator processes more shots.  Rationality (lambda) is always
plotted on a **logarithmic** y-axis because the underlying hypothesis
grid is built with ``np.logspace`` (see ``Estimators/joint.py``).

Public API
----------
load_intermediate_estimates
    Parse a per-shot CSV into a dict of float arrays.
plot_intermediate_estimates
    Dual-axis convergence plot (skill left, rationality right) for one CSV.
plot_all_intermediate_for_player
    Batch-plot every CSV found under a player's ``logs/`` directory.
plot_comparison
    Overlay one metric from several CSVs (e.g. cross-player or cross-season).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intermediate_estimates(csv_path: Path | str) -> dict[str, list[float]]:
    """Load an intermediate-estimates CSV into a dict of float lists.

    Expected columns: ``shot_count``, ``expected_execution_skill``,
    ``map_execution_skill``, ``expected_rationality``, ``map_rationality``.
    """
    csv_path = Path(csv_path)
    data: dict[str, list[float]] = {
        "shot_count": [],
        "expected_execution_skill": [],
        "map_execution_skill": [],
        "expected_rationality": [],
        "map_rationality": [],
    }
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data["shot_count"].append(int(row["shot_count"]))
            data["expected_execution_skill"].append(float(row["expected_execution_skill"]))
            data["map_execution_skill"].append(float(row["map_execution_skill"]))
            data["expected_rationality"].append(float(row["expected_rationality"]))
            data["map_rationality"].append(float(row["map_rationality"]))
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_title(csv_path: Path) -> str:
    """Derive a human-readable title from the CSV path."""
    parts = csv_path.stem.replace("intermediate_estimates", "").strip("_")
    player_id = csv_path.parent.parent.name.replace("player_", "")

    if parts.isdigit() and len(parts) == 8:
        tag = f"{parts[:4]}-{parts[4:]}"
    elif parts:
        tag = parts
    else:
        tag = None

    base = f"JEEDS Convergence – Player {player_id}"
    return f"{base} ({tag})" if tag else base


# ---------------------------------------------------------------------------
# Single-CSV convergence plot
# ---------------------------------------------------------------------------

def plot_intermediate_estimates(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    title: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> Path:
    """Dual-axis convergence plot of execution skill and rationality.

    Left y-axis (linear): execution skill in radians (lower = better).
    Right y-axis (**log**): rationality / lambda (higher = better).

    Parameters
    ----------
    csv_path : Path | str
        Intermediate-estimates CSV produced by ``BlackhawksJEEDS``.
    output_path : Path | str | None
        Destination PNG.  Defaults to the CSV path with a ``.png`` suffix
        (i.e. next to the CSV in ``logs/``).
    title, show, figsize
        Standard matplotlib knobs.

    Returns
    -------
    Path
        Path to the saved image.
    """
    csv_path = Path(csv_path)
    data = load_intermediate_estimates(csv_path)
    if not data["shot_count"]:
        raise ValueError(f"No data in {csv_path}")

    if output_path is not None:
        output_path = Path(output_path)
    else:
        # Keep convergence PNGs next to their CSVs in logs/
        output_path = csv_path.with_suffix(".png")
    title = title or _auto_title(csv_path)

    fig, ax_skill = plt.subplots(figsize=figsize)
    ax_rat = ax_skill.twinx()

    shots = data["shot_count"]

    # Execution skill – warm colours, left axis
    l1 = ax_skill.plot(shots, data["expected_execution_skill"],
                       color="#FF7F50", lw=2, label="EES (skill)")
    l2 = ax_skill.plot(shots, data["map_execution_skill"],
                       color="#DC143C", lw=2, ls="--", label="MAP (skill)")

    # Rationality – cool colours, right axis, LOG scale
    l3 = ax_rat.plot(shots, data["expected_rationality"],
                     color="#40E0D0", lw=2, label="EPS (rationality)")
    l4 = ax_rat.plot(shots, data["map_rationality"],
                     color="#4169E1", lw=2, ls="--", label="MAP (rationality)")
    ax_rat.set_yscale("log")

    ax_skill.set_xlabel("Shot Count", fontsize=12)
    ax_skill.set_ylabel("Execution Skill (rad, lower = better)",
                        color="#DC143C", fontsize=11)
    ax_rat.set_ylabel("Rationality (higher = better)",
                      color="#4169E1", fontsize=11)
    ax_skill.tick_params(axis="y", labelcolor="#DC143C")
    ax_rat.tick_params(axis="y", labelcolor="#4169E1")

    lines = l1 + l2 + l3 + l4
    ax_skill.legend(lines, [l.get_label() for l in lines],
                    loc="upper right", fontsize=10, framealpha=0.7)

    plt.title(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Batch: all CSVs for one player
# ---------------------------------------------------------------------------

def plot_all_intermediate_for_player(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    show: bool = False,
) -> list[Path]:
    """Generate convergence plots for every intermediate CSV of *player_id*.

    Looks in ``<data_dir>/player_<id>/logs/intermediate_estimates*.csv``.
    """
    logs_dir = Path(data_dir) / f"player_{player_id}" / "logs"
    if not logs_dir.exists():
        print(f"No logs directory: {logs_dir}")
        return []

    csvs = sorted(logs_dir.glob("intermediate_estimates*.csv"))
    if not csvs:
        print(f"No CSVs in {logs_dir}")
        return []

    paths: list[Path] = []
    for csv_file in csvs:
        try:
            p = plot_intermediate_estimates(csv_file, show=show)
            paths.append(p)
            print(f"  {csv_file.name} → {p.name}")
        except Exception as exc:
            print(f"  {csv_file.name}: {exc}")
    return paths


# ---------------------------------------------------------------------------
# Multi-CSV comparison
# ---------------------------------------------------------------------------

def plot_comparison(
    csv_paths: Sequence[Path | str],
    labels: Sequence[str] | None = None,
    output_path: Path | str | None = None,
    title: str = "JEEDS Estimate Comparison",
    metric: str = "execution_skill",
    estimate_type: str = "map",
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> Path:
    """Overlay one metric from several CSVs.

    Parameters
    ----------
    metric : str
        ``"execution_skill"`` or ``"rationality"``.
    estimate_type : str
        ``"map"``, ``"expected"``, or ``"both"``.
    """
    csv_paths = [Path(p) for p in csv_paths]
    labels = list(labels) if labels else [p.stem for p in csv_paths]
    output_path = (
        Path(output_path) if output_path
        else csv_paths[0].parent / f"comparison_{metric}_{estimate_type}.png"
    )

    map_key = "map_execution_skill" if metric == "execution_skill" else "map_rationality"
    exp_key = "expected_execution_skill" if metric == "execution_skill" else "expected_rationality"
    exp_label = "EES" if metric == "execution_skill" else "EPS"

    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))

    for i, (cp, label) in enumerate(zip(csv_paths, labels)):
        data = load_intermediate_estimates(cp)
        shots = data["shot_count"]
        if estimate_type in ("map", "both"):
            ax.plot(shots, data[map_key], color=colours[i], lw=2,
                    label=f"{label} (MAP)" if estimate_type == "both" else label,
                    ls="--" if estimate_type == "both" else "-")
        if estimate_type in ("expected", "both"):
            ax.plot(shots, data[exp_key], color=colours[i], lw=2,
                    label=f"{label} ({exp_label})" if estimate_type == "both" else label,
                    alpha=0.7 if estimate_type == "both" else 1.0)

    ylabel = "Execution Skill (rad)" if metric == "execution_skill" else "Rationality"
    ax.set_xlabel("Shot Count", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if metric == "rationality":
        ax.set_yscale("log")

    ax.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_intermediate_estimates.py <csv_path> [--show]")
        print("       python plot_intermediate_estimates.py --player <player_id> [--show]")
        sys.exit(1)

    show = "--show" in sys.argv
    if "--player" in sys.argv:
        idx = sys.argv.index("--player")
        plot_all_intermediate_for_player(int(sys.argv[idx + 1]), show=show)
    else:
        out = plot_intermediate_estimates(sys.argv[1], show=show)
        print(f"Saved: {out}")
