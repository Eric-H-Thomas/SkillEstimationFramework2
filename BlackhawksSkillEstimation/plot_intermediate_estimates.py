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
import matplotlib.ticker as mticker
import numpy as np

from BlackhawksSkillEstimation.player_cache import lookup_player


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_intermediate_estimates(csv_path: Path | str) -> dict[str, list[float]]:
    """Load an intermediate-estimates CSV into a dict of float lists.

    Expected columns: ``shot_count``, ``expected_execution_skill``,
    ``map_execution_skill``, ``expected_rationality``, ``map_rationality``.
    Also reads ``log10_expected_rationality`` and ``log10_map_rationality``
    when present.
    """
    csv_path = Path(csv_path)
    data: dict[str, list[float]] = {
        "shot_count": [],
        "expected_execution_skill": [],
        "map_execution_skill": [],
        "expected_rationality": [],
        "map_rationality": [],
        "log10_expected_rationality": [],
        "log10_map_rationality": [],
    }
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            data["shot_count"].append(int(row["shot_count"]))
            data["expected_execution_skill"].append(float(row["expected_execution_skill"]))
            data["map_execution_skill"].append(float(row["map_execution_skill"]))
            er = float(row["expected_rationality"])
            mr = float(row["map_rationality"])
            data["expected_rationality"].append(er)
            data["map_rationality"].append(mr)
            # Derive log10 if not in CSV (backward compatibility)
            if "log10_expected_rationality" in row and row["log10_expected_rationality"]:
                data["log10_expected_rationality"].append(float(row["log10_expected_rationality"]))
            else:
                data["log10_expected_rationality"].append(np.log10(er) if er > 0 else float("nan"))
            if "log10_map_rationality" in row and row["log10_map_rationality"]:
                data["log10_map_rationality"].append(float(row["log10_map_rationality"]))
            else:
                data["log10_map_rationality"].append(np.log10(mr) if mr > 0 else float("nan"))
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
    
    player_name = lookup_player(player_id=int(player_id))

    base = f"JEEDS Convergence – Player {player_id}"
    if player_name:
        base += f" ({player_name})"
    return f"{base} - {tag}" if tag else base


# ---------------------------------------------------------------------------
# Single-CSV convergence plot
# ---------------------------------------------------------------------------

def plot_intermediate_estimates(
    csv_path: Path | str,
    output_path: Path | str | None = None,
    title: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
    burnin: int = 5,
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
    burnin : int
        Number of initial shots to omit from the plot.  The prior dominates
        these early updates, producing misleading spikes.  Set to 0 to show
        everything.  Default is 5.

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

    shots = data["shot_count"][burnin:]
    ees_data = data["expected_execution_skill"][burnin:]
    map_skill_data = data["map_execution_skill"][burnin:]
    eps_data = data["expected_rationality"][burnin:]
    map_rat_data = data["map_rationality"][burnin:]

    # Execution skill – warm colours, left axis
    l1 = ax_skill.plot(shots, ees_data,
                       color="#FF7F50", lw=2, label="EES (skill)")
    l2 = ax_skill.plot(shots, map_skill_data,
                       color="#DC143C", lw=2, ls="--", label="MAP (skill)")

    # Rationality – cool colours, right axis, LOG scale
    l3 = ax_rat.plot(shots, eps_data,
                     color="#40E0D0", lw=2, label="EPS (rationality)")
    l4 = ax_rat.plot(shots, map_rat_data,
                     color="#4169E1", lw=2, ls="--", label="MAP (rationality)")
    ax_rat.set_yscale("log")
    ax_rat.set_ylim(10, 10 ** 3.5)

    # Show ticks at each half order of magnitude (10^1, 10^1.5, 10^2, ...)
    _half_log_ticks = [10 ** (x / 2) for x in range(2, 8)]  # 10^1 to 10^3.5
    ax_rat.yaxis.set_major_locator(mticker.FixedLocator(_half_log_ticks))
    ax_rat.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"10^{np.log10(val):.1f}" if val > 0 else ""
    ))
    ax_rat.yaxis.set_minor_locator(mticker.NullLocator())

    ax_skill.set_xlabel("Shot Count", fontsize=12)
    ax_skill.set_ylabel("Execution Skill (rad, lower = better)",
                        color="#DC143C", fontsize=11)
    ax_rat.set_ylabel("Rationality (higher = better)",
                      color="#4169E1", fontsize=11)
    ax_skill.tick_params(axis="y", labelcolor="#DC143C")
    ax_rat.tick_params(axis="y", labelcolor="#4169E1", which="both", labelsize=9)

    # Burn-in annotation
    if burnin > 0:
        ax_skill.annotate(
            f"first {burnin} shots omitted",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="grey", fontstyle="italic",
        )

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

#TODO: Can we "zoom in" the y axis throughout the plot so we can see their values better once they've converged?
def plot_comparison(
    csv_paths: Sequence[Path | str],
    labels: Sequence[str] | None = None,
    output_path: Path | str | None = None,
    title: str = "JEEDS Estimate Comparison",
    metric: str = "execution_skill",
    estimate_type: str = "expected",
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
    burnin: int = 5,
) -> Path:
    """Overlay one metric from several CSVs.

    Parameters
    ----------
    metric : str
        ``"execution_skill"`` or ``"rationality"``.
    estimate_type : str
        ``"map"``, ``"expected"``, or ``"both"``.
    burnin : int
        Number of initial shots to omit (burn-in).  Default 5.
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
        shots = data["shot_count"][burnin:]
        if estimate_type in ("map", "both"):
            ax.plot(shots, data[map_key][burnin:], color=colours[i], lw=2,
                    label=f"{label} (MAP)" if estimate_type == "both" else label,
                    ls="--" if estimate_type == "both" else "-")
        if estimate_type in ("expected", "both"):
            ax.plot(shots, data[exp_key][burnin:], color=colours[i], lw=2,
                    label=f"{label} ({exp_label})" if estimate_type == "both" else label,
                    alpha=0.7 if estimate_type == "both" else 1.0)

    ylabel = "Execution Skill (rad)" if metric == "execution_skill" else "Rationality"
    ax.set_xlabel("Shot Count", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if metric == "rationality":
        ax.set_yscale("log")
        ax.set_ylim(10, 10 ** 3.5)
        _half_log_ticks = [10 ** (x / 2) for x in range(2, 8)]  # 10^1 to 10^3.5
        ax.yaxis.set_major_locator(mticker.FixedLocator(_half_log_ticks))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda val, _: f"10^{np.log10(val):.1f}" if val > 0 else ""
        ))
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    # Burn-in annotation
    if burnin > 0:
        ax.annotate(
            f"first {burnin} shots omitted",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="grey", fontstyle="italic",
        )

    ax.legend(fontsize=10)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show() if show else plt.close(fig)
    return output_path


def rank_final_estimates(
        season: str | int = "20242025",
        players: list[int] | None = None,
        metric: str = "execution_skill",
        data_dir: str | Path = "Data/Hockey",
        output_dir: str | Path = "Data/Hockey/general_plots",
        show: bool = False,
        figsize: tuple[float, float] = (8, 6),
) -> Path:
    """Rank final *expected* estimates for a set of players as a horizontal bar chart.

    Parameters
    ----------
    season : str | int
        Season tag used in intermediate CSV filenames.
    players : list[int] | None
        List of player IDs to include. This must be provided.
    metric : str
        Either "execution_skill" or "rationality". Rankings use the expected
        estimate only (e.g. `expected_execution_skill` or `expected_rationality`).
    data_dir, output_dir : str | Path
        Paths for reading player logs and writing the output image.
    show : bool
        If True, display the figure instead of closing it.
    figsize : tuple[float, float]
        Figure size passed to `plt.subplots`.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if players is None:
        raise ValueError("`players` must be provided (no auto-discovery)")

    if metric not in ("execution_skill", "rationality"):
        raise ValueError("metric must be 'execution_skill' or 'rationality'")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    values: list[float] = []

    for pid in players:
        logs = data_dir / f"player_{pid}" / "logs"
        csv = logs / f"intermediate_estimates_{season}.csv"
        if not csv.exists():
            print(f"warning: {pid} has no CSV for {season}")
            continue
        data = load_intermediate_estimates(csv)
        if not data["shot_count"]:
            continue
        if metric == "execution_skill":
            val = data["expected_execution_skill"][-1]
        else:
            # Use log10 expected rationality (backwards-compatible key exists)
            val = data.get("log10_expected_rationality", [None])[-1]
            if val is None:
                # fall back to raw expected_rationality then take log10 if present
                raw = data["expected_rationality"][-1]
                val = np.log10(raw) if raw > 0 else float("nan")
        names.append(lookup_player(pid) or str(pid))
        values.append(val)

    if not values:
        raise ValueError("No data collected for provided players")

    # Sorting: execution_skill -> ascending (lower = better); rationality -> descending
    reverse = metric == "rationality"
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=reverse)
    sorted_names = [names[i] for i in order]
    sorted_values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_values, color="C0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.invert_yaxis()  # best at the top

    if metric == "execution_skill":
        ax.set_xlabel("Final expected execution skill (rad, lower = better)")
        fmt = "{:.3f}"
    else:
        ax.set_xlabel("Final expected log10(rationality) (log10(lambda))")
        fmt = "{:.2f}"

    # Annotate values at end of bars
    max_v = max(sorted_values)
    min_v = min(sorted_values)
    rng = max_v - min_v if max_v != min_v else abs(max_v) if max_v != 0 else 1.0
    offset = rng * 0.01
    for i, v in enumerate(sorted_values):
        ax.text(v + offset, i, fmt.format(v), va="center", fontsize=8)

    ax.set_title(f"Final expected {metric.replace('_', ' ')} rankings – season {season}")
    out = output_dir / f"final_{metric}_rankings_{season}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    else:
        plt.close(fig)
    return out


def compare_execution_rankings_two_seasons(
    players: list[int] | None,
    season_a: str | int = "20232024",
    season_b: str | int = "20242025",
    data_dir: str | Path = "Data/Hockey",
    output_dir: str | Path = "Data/Hockey/general_plots",
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> Path:
    """Render two adjacent ranking tables (by execution skill) for two seasons.

    Each table has three columns: player name, execution skill in season A,
    execution skill in season B. The left table is sorted by season A
    (ascending), the right table by season B (ascending).

    `players` must be provided.
    """
    if players is None:
        raise ValueError("`players` must be provided (no auto-discovery)")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    vals_a: list[float] = []
    vals_b: list[float] = []

    for pid in players:
        name = lookup_player(pid) or str(pid)
        csv_a = data_dir / f"player_{pid}" / "logs" / f"intermediate_estimates_{season_a}.csv"
        csv_b = data_dir / f"player_{pid}" / "logs" / f"intermediate_estimates_{season_b}.csv"

        def _load_exec(csvp):
            if not csvp.exists():
                return float("nan")
            d = load_intermediate_estimates(csvp)
            return d["expected_execution_skill"][-1] if d["shot_count"] else float("nan")

        va = _load_exec(csv_a)
        vb = _load_exec(csv_b)
        names.append(name)
        vals_a.append(va)
        vals_b.append(vb)

    # Build rows as (name, a, b)
    rows = list(zip(names, vals_a, vals_b))

    # Helper to produce display value
    def _fmt(v: float) -> str:
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"

    # Sort copies for each table
    def _nan_key(x):
        v = x[1]
        return (np.isnan(v), v if not np.isnan(v) else float("inf"))

    left_rows = sorted(rows, key=_nan_key)
    # For right table sort by season B value
    def _nan_key_b(x):
        v = x[2]
        return (np.isnan(v), v if not np.isnan(v) else float("inf"))

    right_rows = sorted(rows, key=_nan_key_b)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace':0.1})
    fig.suptitle("Execution skill estimates (σ in radians, lower=better)", fontsize=14)
    for ax in (axl, axr):
        ax.axis("off")

    # Prepare ranked tables showing final values only. Both tables list the
    # player name and execution scores for both seasons; sort order differs.
    left_cell = [[r[0], _fmt(r[1]), _fmt(r[2])] for r in left_rows]
    right_cell = [[r[0], _fmt(r[1]), _fmt(r[2])] for r in right_rows]

    col_labels = ["Player", str(season_a), str(season_b)]

    tbl_l = axl.table(cellText=left_cell, colLabels=col_labels, cellLoc="left", loc="center")
    tbl_r = axr.table(cellText=right_cell, colLabels=col_labels, cellLoc="left", loc="center")

    # narrow columns since execution skill values are short
    cols = list(range(len(col_labels)))
    tbl_l.auto_set_column_width(col=cols)
    tbl_r.auto_set_column_width(col=cols)

    tbl_l.auto_set_font_size(False)
    tbl_r.auto_set_font_size(False)
    tbl_l.set_fontsize(9)
    tbl_r.set_fontsize(9)
    tbl_l.scale(1, 1.2)
    tbl_r.scale(1, 1.2)

    axl.set_title(f"Ranked by {season_a}")
    axr.set_title(f"Ranked by {season_b}")

    out = output_dir / f"exec_rankings_compare_{season_a}_{season_b}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    else:
        plt.close(fig)
    return out

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
