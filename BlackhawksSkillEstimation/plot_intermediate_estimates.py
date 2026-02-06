"""Plot intermediate JEEDS estimates over shots.

This module provides functions to visualize how execution skill and rationality
estimates evolve as the JEEDS estimator observes more shots. Useful for
understanding convergence behavior and comparing MAP vs. expected estimates.

Example
-------
from BlackhawksSkillEstimation.plot_intermediate_estimates import plot_intermediate_estimates
plot_intermediate_estimates("Data/Hockey/player_950160/logs/intermediate_estimates_20242025.csv")
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_intermediate_estimates(csv_path: Path | str) -> dict[str, list[float]]:
    """Load intermediate estimates from a CSV file.
    
    Parameters
    ----------
    csv_path : Path | str
        Path to the CSV file with columns: shot_count, expected_execution_skill,
        map_execution_skill, expected_rationality, map_rationality.
    
    Returns
    -------
    dict
        Dictionary with keys matching column names, values as lists of floats.
    """
    csv_path = Path(csv_path)
    data: dict[str, list[float]] = {
        "shot_count": [],
        "expected_execution_skill": [],
        "map_execution_skill": [],
        "expected_rationality": [],
        "map_rationality": [],
    }
    
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["shot_count"].append(int(row["shot_count"]))
            data["expected_execution_skill"].append(float(row["expected_execution_skill"]))
            data["map_execution_skill"].append(float(row["map_execution_skill"]))
            data["expected_rationality"].append(float(row["expected_rationality"]))
            data["map_rationality"].append(float(row["map_rationality"]))
    
    return data


def plot_intermediate_estimates(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    title: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> Path:
    """Plot intermediate estimates from a CSV file.
    
    Creates a dual-axis line plot showing how estimates evolve over shots:
    - Left y-axis: Execution skill estimates (in radians)
    - Right y-axis: Rationality estimates (unitless)
    
    Parameters
    ----------
    csv_path : Path | str
        Path to the intermediate estimates CSV file.
    output_path : Path | str | None
        Where to save the plot. If None, saves alongside the CSV with .png extension.
    title : str | None
        Plot title. If None, auto-generates from filename.
    show : bool
        If True, display the plot interactively.
    figsize : tuple[float, float]
        Figure size in inches (width, height).
    
    Returns
    -------
    Path
        Path to the saved plot image.
    """
    csv_path = Path(csv_path)
    data = load_intermediate_estimates(csv_path)
    
    if not data["shot_count"]:
        raise ValueError(f"No data found in {csv_path}")
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.with_suffix(".png")
    else:
        output_path = Path(output_path)
    
    # Auto-generate title from filename if not provided
    if title is None:
        # Extract player ID and tag from path
        # e.g., "Data/Hockey/player_950160/logs/intermediate_estimates_20242025.csv"
        parts = csv_path.stem.replace("intermediate_estimates", "").strip("_")
        player_dir = csv_path.parent.parent.name  # "player_950160"
        player_id = player_dir.replace("player_", "")
        
        # Format tag nicely - check if it's a season (8-digit number like 20242025)
        if parts and parts.isdigit() and len(parts) == 8:
            # Format as "2024-2025" season
            tag = f"{parts[:4]}-{parts[4:]}"
        elif parts:
            tag = parts
        else:
            tag = None
        
        if tag:
            title = f"JEEDS Estimate Convergence - Player {player_id} ({tag})"
        else:
            title = f"JEEDS Estimate Convergence - Player {player_id}"
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    shots = data["shot_count"]
    
    # Plot execution skill on left axis (warm colors: orange/red)
    line_ees_skill = ax1.plot(
        shots, data["expected_execution_skill"],
        color="#FF7F50",  # coral
        linewidth=2,
        label="EES",
        linestyle="-",
    )
    line_map_skill = ax1.plot(
        shots, data["map_execution_skill"],
        color="#DC143C",  # crimson
        linewidth=2,
        label="MAP Skill",
        linestyle="--",
    )
    
    # Plot rationality on right axis (cool colors: blue/cyan)
    line_ees_rat = ax2.plot(
        shots, data["expected_rationality"],
        color="#40E0D0",  # turquoise
        linewidth=2,
        label="EPS",
        linestyle="-",
    )
    line_map_rat = ax2.plot(
        shots, data["map_rationality"],
        color="#4169E1",  # royal blue
        linewidth=2,
        label="MAP Rationality",
        linestyle="--",
    )
    
    # Labels and formatting
    ax1.set_xlabel("Shot Count", fontsize=12)
    ax1.set_ylabel("Execution Skill (radians, lower = better)", color="#DC143C", fontsize=11)
    ax2.set_ylabel("Rationality (higher = better)", color="#4169E1", fontsize=11)
    
    ax1.tick_params(axis="y", labelcolor="#DC143C")
    ax2.tick_params(axis="y", labelcolor="#4169E1")
    
    # Combine legends from both axes with transparency
    lines = line_ees_skill + line_map_skill + line_ees_rat + line_map_rat
    labels = [line.get_label() for line in lines]
    legend = ax1.legend(
        lines, labels,
        loc="upper right",
        fontsize=10,
        framealpha=0.7,
        fancybox=True,
    )
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


def plot_all_intermediate_for_player(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    show: bool = False,
) -> list[Path]:
    """Generate plots for all intermediate estimate CSVs for a player.
    
    Parameters
    ----------
    player_id : int
        Player ID to process.
    data_dir : Path | str
        Base data directory containing player folders.
    show : bool
        If True, display each plot interactively.
    
    Returns
    -------
    list[Path]
        Paths to all generated plot images.
    """
    data_dir = Path(data_dir)
    logs_dir = data_dir / f"player_{player_id}" / "logs"
    
    if not logs_dir.exists():
        print(f"No logs directory found at {logs_dir}")
        return []
    
    csv_files = list(logs_dir.glob("intermediate_estimates*.csv"))
    
    if not csv_files:
        print(f"No intermediate estimate CSVs found in {logs_dir}")
        return []
    
    output_paths = []
    for csv_file in csv_files:
        print(f"Plotting {csv_file.name}...")
        try:
            output_path = plot_intermediate_estimates(csv_file, show=show)
            output_paths.append(output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error: {e}")
    
    return output_paths


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
    """Compare estimates across multiple CSV files (e.g., different players or seasons).
    
    Parameters
    ----------
    csv_paths : Sequence[Path | str]
        List of CSV file paths to compare.
    labels : Sequence[str] | None
        Labels for each CSV file. If None, uses filenames.
    output_path : Path | str | None
        Where to save the plot.
    title : str
        Plot title.
    metric : str
        Which metric to plot: "execution_skill" or "rationality".
    estimate_type : str
        Which estimate type: "map", "expected" (EES/EPS), or "both".
    show : bool
        If True, display the plot interactively.
    figsize : tuple[float, float]
        Figure size in inches.
    
    Returns
    -------
    Path
        Path to the saved plot image.
    """
    csv_paths = [Path(p) for p in csv_paths]
    
    if labels is None:
        labels = [p.stem for p in csv_paths]
    
    if output_path is None:
        output_path = csv_paths[0].parent / f"comparison_{metric}_{estimate_type}.png"
    else:
        output_path = Path(output_path)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))
    
    # Map metric to column names
    map_key = "map_execution_skill" if metric == "execution_skill" else "map_rationality"
    expected_key = "expected_execution_skill" if metric == "execution_skill" else "expected_rationality"
    expected_label = "EES" if metric == "execution_skill" else "EPS"
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        data = load_intermediate_estimates(csv_path)
        shots = data["shot_count"]
        
        if estimate_type in ("map", "both"):
            ax.plot(
                shots, data[map_key],
                color=colors[i],
                linewidth=2,
                label=f"{label} (MAP)" if estimate_type == "both" else label,
                linestyle="--" if estimate_type == "both" else "-",
            )
        
        if estimate_type in ("expected", "both"):
            ax.plot(
                shots, data[expected_key],
                color=colors[i],
                linewidth=2,
                label=f"{label} ({expected_label})" if estimate_type == "both" else label,
                linestyle="-",
                alpha=0.7 if estimate_type == "both" else 1.0,
            )
    
    ylabel = "Execution Skill (radians)" if metric == "execution_skill" else "Rationality"
    ax.set_xlabel("Shot Count", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python plot_intermediate_estimates.py <csv_path> [--show]")
        print("       python plot_intermediate_estimates.py --player <player_id> [--show]")
        sys.exit(1)
    
    show = "--show" in sys.argv
    
    if "--player" in sys.argv:
        idx = sys.argv.index("--player")
        player_id = int(sys.argv[idx + 1])
        plot_all_intermediate_for_player(player_id, show=show)
    else:
        csv_path = sys.argv[1]
        output = plot_intermediate_estimates(csv_path, show=show)
        print(f"Plot saved to: {output}")
