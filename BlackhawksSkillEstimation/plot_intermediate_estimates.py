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
    Also requires ``log10_expected_rationality`` and ``log10_map_rationality``.
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
            data["log10_expected_rationality"].append(float(row["log10_expected_rationality"]))
            data["log10_map_rationality"].append(float(row["log10_map_rationality"]))
    return data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _humanize_canonical_season_tag(tag: str) -> str:
    """Convert canonical season tag to human-readable label.
    
    Examples:
        "s20232024" -> "2023-2024"
        "s20222023to20242025" -> "2022-2023 to 2024-2025 (3 seasons)"
        "s20222023__20242025" -> "2022-2023 + 2024-2025 (non-adjacent)"
    """
    if not tag or not tag.startswith("s"):
        # Not a canonical season tag, return as-is
        return tag
    
    tag = tag[1:]  # Strip leading 's'
    
    if "__" in tag:
        # Non-adjacent list form
        season_strs = tag.split("__")
        humanized = []
        for s in season_strs:
            if len(s) == 8 and s.isdigit():
                humanized.append(f"{s[:4]}-{s[4:]}")
            else:
                return tag  # Fallback if parsing fails
        return " + ".join(humanized) + f" (non-adjacent, {len(humanized)} seasons)"
    
    elif "to" in tag:
        # Adjacent range form
        parts = tag.split("to")
        if len(parts) == 2 and all(len(p) == 8 and p.isdigit() for p in parts):
            start_season_start = int(parts[0][:4])
            end_season_start = int(parts[1][:4])
            num_seasons = end_season_start - start_season_start + 1
            return f"{parts[0][:4]}-{parts[0][4:]} to {parts[1][:4]}-{parts[1][4:]} ({num_seasons} seasons)"
        return tag  # Fallback
    
    elif len(tag) == 8 and tag.isdigit():
        # Single season
        return f"{tag[:4]}-{tag[4:]}"
    
    return tag  # Fallback for unrecognized format


def _humanize_partition_suffix(partition_blob: str) -> str:
    """Convert filename partition suffix into a readable value label."""
    if not partition_blob:
        return ""

    if "-" not in partition_blob:
        return partition_blob.replace("_", " ")

    _column_slug, value_slug = partition_blob.split("-", 1)

    if value_slug.startswith("multi-"):
        value_slug = value_slug[len("multi-") :]
        values = [v.replace("-", " ") for v in value_slug.split("-or-") if v]
        return ", ".join(values)
    else:
        return value_slug.replace("-", " ")


def _auto_title(csv_path: Path) -> str:
    """Derive a human-readable title from the CSV path."""
    parts = csv_path.stem.replace("intermediate_estimates", "").strip("_")

    # Detect group subdir: player_{id}/logs/{group}/file.csv vs flat logs/file.csv
    if csv_path.parent.name == "logs":
        # Flat structure: player_{id}/logs/file.csv
        player_dir_name = csv_path.parent.parent.name
        group_label = None
    elif csv_path.parent.parent.name == "logs":
        # Group subdir: player_{id}/logs/{group}/file.csv
        player_dir_name = csv_path.parent.parent.parent.name
        group_tag = csv_path.parent.name
        from BlackhawksSkillEstimation.BlackhawksJEEDS import SHOT_TYPE_GROUPS
        group_info = SHOT_TYPE_GROUPS.get(group_tag)
        group_label = group_info[0] if group_info else group_tag
    else:
        player_dir_name = csv_path.parent.parent.name
        group_label = None

    player_id = player_dir_name.replace("player_", "")

    season_part = parts
    partition_part = ""
    if "__" in parts:
        season_part, partition_part = parts.split("__", 1)

    # Parse season tag: handle canonical season tags and legacy formats.
    if season_part.startswith("s"):
        tag = _humanize_canonical_season_tag(season_part)
    elif season_part.isdigit() and len(season_part) == 8:
        tag = f"{season_part[:4]}-{season_part[4:]}"
    elif season_part:
        tag = season_part
    else:
        tag = None

    partition_label = _humanize_partition_suffix(partition_part)
    
    player_name = lookup_player(player_id=int(player_id))

    base = f"JEEDS Convergence \u2013 Player {player_id}"
    if player_name:
        base += f" ({player_name})"
    if group_label:
        base += f" \u2013 {group_label}"
    if tag:
        base += f" - {tag}"
    if partition_label:
        base += f" ({partition_label})"
    return base


def get_estimate_value_at_shot(
    estimates: dict[str, list[float]],
    shot_index: int,
    metric: str = "expected_execution_skill",
) -> float | None:
    """Return metric value at 1-based shot index, or None if unavailable."""
    values = estimates.get(metric)
    if not values:
        return None
    idx = int(shot_index) - 1
    if idx < 0 or idx >= len(values):
        return None
    return float(values[idx])


def get_estimate_before_after_delta(
    estimates: dict[str, list[float]],
    shot_index: int,
    metric: str = "expected_execution_skill",
) -> tuple[float | None, float | None, float | None]:
    """Return (before, after, delta) for a 1-based shot index.

    ``before`` is the metric at ``shot_index - 1`` and is None when the shot
    has no predecessor. ``delta`` is computed as ``after - before``.
    """
    after = get_estimate_value_at_shot(estimates, shot_index, metric)
    if after is None:
        return None, None, None
    before = get_estimate_value_at_shot(estimates, int(shot_index) - 1, metric)
    if before is None:
        return None, after, None
    return before, after, float(after - before)


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
    include_map_estimates: bool = True,
) -> plt.Figure:
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
    if include_map_estimates:
        l2 = ax_skill.plot(shots, map_skill_data,
                           color="#DC143C", lw=2, ls="--", label="MAP (skill)")
    else:
        l2 = []

    # Rationality – cool colours, right axis, LOG scale
    l3 = ax_rat.plot(shots, eps_data,
                     color="#40E0D0", lw=2, label="EPS (rationality)")
    if include_map_estimates:
        l4 = ax_rat.plot(shots, map_rat_data,
                         color="#4169E1", lw=2, ls="--", label="MAP (rationality)")
    else:
        l4 = []
    ax_rat.set_yscale("log")
    ax_rat.set_ylim(10 ** 0.95, 10 ** 3.65)

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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Batch: all CSVs for one player
# ---------------------------------------------------------------------------

def plot_all_intermediate_for_player(
    player_id: int,
    data_dir: Path | str = Path("Data/Hockey"),
    show: bool = False,
) -> list[plt.Figure]:
    """Generate convergence plots for every intermediate CSV of *player_id*.

    Looks in ``<data_dir>/players/player_<id>/logs/intermediate_estimates*.csv`` and
    also in group subdirectories ``logs/*/intermediate_estimates*.csv``.
    """
    logs_dir = Path(data_dir) / "players" / f"player_{player_id}" / "logs"
    if not logs_dir.exists():
        print(f"No logs directory: {logs_dir}")
        return []

    # Collect CSVs from flat logs/ and from group subdirs logs/*/
    csvs = sorted(
        set(logs_dir.glob("intermediate_estimates*.csv"))
        | set(logs_dir.glob("*/intermediate_estimates*.csv"))
    )
    if not csvs:
        print(f"No CSVs in {logs_dir}")
        return []

    paths: list[plt.Figure] = []
    for csv_file in csvs:
        try:
            fig = plot_intermediate_estimates(csv_file, show=show)
            paths.append(fig)
            out = csv_file.with_suffix('.png')
            print(f"  {csv_file.name} → {out.name}")
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
) -> plt.Figure:
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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        fig.show()
    else:
        plt.close(fig)
    return fig


def rank_final_estimates(
    season: str | int = "20242025",
    players: list[int] | None = None,
    metric: str = "execution_skill",
    data_dir: str | Path = "Data/Hockey",
    output_dir: str | Path = "Data/Hockey/general_plots",
    show: bool = False,
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
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
        logs = data_dir / "players" / f"player_{pid}" / "logs"
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
    return fig


def compare_execution_rankings_two_seasons_by_shot_type(
    players: list[int] | None,
    shot_types: Sequence[str] = ("wristshot_snapshot", "slapshot"),
    season_a: str | int = "20232024",
    season_b: str | int = "20242025",
    data_dir: str | Path = "Data/Hockey",
    output_dir: str | Path = "Data/Hockey/_bhawks_reports",
    csv_filename: str = "byu_results_with_shot_type.csv",
    show: bool = False,
    figsize: tuple[float, float] = (12, 6),
) -> dict[str, Path | list[Path]]:
    """Create per-shot-type season-comparison tables and export BYU-style CSV.

    CSV columns follow the BYU format with one additional column:
    ``player_id_hawks,season,shot_type,sigma_value,log10_rationality``.

    For each shot type, two PNG tables are generated (``exec_...`` and ``rat_...``),
    each with side-by-side rankings: left sorted by season A and right sorted by
    season B.
    """
    if players is None:
        raise ValueError("`players` must be provided (no auto-discovery)")

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from BlackhawksSkillEstimation.BlackhawksJEEDS import SHOT_TYPE_GROUPS

    def _load_exec(pid: int, season: str | int, shot_type: str) -> tuple[float, float, int]:
        csvp = (
            data_dir
            / "players"
            / f"player_{pid}"
            / "logs"
            / shot_type
            / f"intermediate_estimates_{season}.csv"
        )
        if not csvp.exists():
            return float("nan"), float("nan"), 0
        d = load_intermediate_estimates(csvp)
        if not d["shot_count"]:
            return float("nan"), float("nan"), 0
        return (
            d["expected_execution_skill"][-1],
            d["log10_expected_rationality"][-1],
            int(d["shot_count"][-1]),
        )

    def _fmt(v: float) -> str:
        return "-" if np.isnan(v) else f"{v:.3f}"

    def _sort_key(v: float, descending: bool = False) -> tuple[bool, float]:
        if np.isnan(v):
            return (True, float("inf"))
        return (False, -v if descending else v)

    def _add_inline_count_overlays(
        ax: plt.Axes,
        tbl: plt.Table,
        per_row_counts: list[tuple[int, int]],
    ) -> None:
        for row_idx, (count_a, count_b) in enumerate(per_row_counts, start=1):
            for col_idx, count in ((1, count_a), (2, count_b)):
                if count <= 0:
                    continue
                cell = tbl[row_idx, col_idx]
                x = cell.get_x() + cell.get_width() * 0.98
                y = cell.get_y() + cell.get_height() * 0.5
                ax.text(
                    x,
                    y,
                    f"n={count}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7,
                    color="#8A8A8A",
                    clip_on=False,
                    zorder=10,
                )

    def _render_rankings_table(
        rows: list[tuple[str, int, float, float, float, float, int, int]],
        shot_type: str,
        shot_label: str,
        value_idx_a: int,
        value_idx_b: int,
        title: str,
        out_prefix: str,
        descending: bool = False,
    ) -> Path:
        left_rows = sorted(rows, key=lambda r: _sort_key(r[value_idx_a], descending=descending))
        right_rows = sorted(rows, key=lambda r: _sort_key(r[value_idx_b], descending=descending))

        # Scale figure height with row count so table titles stay readable.
        dynamic_height = max(figsize[1], min(0.22 * max(1, len(rows)) + 2.5, 80.0))
        fig, (axl, axr) = plt.subplots(
            1,
            2,
            figsize=(figsize[0], dynamic_height),
            gridspec_kw={"wspace": 0.1},
        )
        fig.suptitle(
            f"{title} - {shot_label}",
            fontsize=14,
            y=0.995,
        )
        for ax in (axl, axr):
            ax.axis("off")

        left_cell = [[r[0], _fmt(r[value_idx_a]), _fmt(r[value_idx_b])] for r in left_rows]
        right_cell = [[r[0], _fmt(r[value_idx_a]), _fmt(r[value_idx_b])] for r in right_rows]
        left_counts = [(r[6], r[7]) for r in left_rows]
        right_counts = [(r[6], r[7]) for r in right_rows]
        col_labels = ["Player", str(season_a), str(season_b)]

        # Keep a small top margin inside each axes so the subplot title never overlaps.
        tbl_l = axl.table(
            cellText=left_cell,
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 0.965],
        )
        tbl_r = axr.table(
            cellText=right_cell,
            colLabels=col_labels,
            cellLoc="left",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 0.965],
        )

        cols = list(range(len(col_labels)))
        tbl_l.auto_set_column_width(col=cols)
        tbl_r.auto_set_column_width(col=cols)

        max_digits = 1
        all_counts = [c for pair in (left_counts + right_counts) for c in pair if c > 0]
        if all_counts:
            max_digits = max(len(str(c)) for c in all_counts)
        season_width_multiplier = 1.25 + 0.03 * max(0, max_digits - 2)

        for tbl in (tbl_l, tbl_r):
            for row_idx in range(len(left_cell) + 1):
                for col_idx in (1, 2):
                    cell = tbl[row_idx, col_idx]
                    cell.set_width(cell.get_width() * season_width_multiplier)
            for row_idx in range(1, len(left_cell) + 1):
                for col_idx in (1, 2):
                    tbl[row_idx, col_idx].get_text().set_ha("left")
                    tbl[row_idx, col_idx].PAD = 0.035

        tbl_l.auto_set_font_size(False)
        tbl_r.auto_set_font_size(False)
        tbl_l.set_fontsize(9)
        tbl_r.set_fontsize(9)
        tbl_l.scale(1, 1.2)
        tbl_r.scale(1, 1.2)

        fig.canvas.draw()
        _add_inline_count_overlays(axl, tbl_l, left_counts)
        _add_inline_count_overlays(axr, tbl_r, right_counts)

        axl.set_title(f"Ranked by {season_a}", pad=10)
        axr.set_title(f"Ranked by {season_b}", pad=10)

        out_png = output_dir / f"{out_prefix}_rankings_compare_{season_a}_{season_b}_{shot_type}.png"
        fig.subplots_adjust(top=0.97, wspace=0.1)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        if show:
            fig.show()
        else:
            plt.close(fig)
        return out_png

    csv_rows: list[tuple[int, str, str, float, float]] = []
    png_paths: list[Path] = []

    for shot_type in shot_types:
        rows: list[tuple[str, int, float, float, float, float, int, int]] = []
        for pid in players:
            name = lookup_player(pid) or str(pid)
            va, ra, count_a = _load_exec(pid, season_a, shot_type)
            vb, rb, count_b = _load_exec(pid, season_b, shot_type)
            if np.isnan(va) and np.isnan(vb):
                continue

            rows.append((name, pid, va, vb, ra, rb, count_a, count_b))

            if not np.isnan(va):
                csv_rows.append((pid, str(season_a), shot_type, float(va), float(ra)))
            if not np.isnan(vb):
                csv_rows.append((pid, str(season_b), shot_type, float(vb), float(rb)))

        if not rows:
            print(f"warning: no season results found for shot type '{shot_type}'")
            continue

        shot_label = SHOT_TYPE_GROUPS.get(shot_type, (shot_type, set(), False))[0]
        png_paths.append(
            _render_rankings_table(
                rows=rows,
                shot_type=shot_type,
                shot_label=shot_label,
                value_idx_a=2,
                value_idx_b=3,
                title="Execution skill estimates (sigma in radians, lower=better)",
                out_prefix="exec",
            )
        )
        png_paths.append(
            _render_rankings_table(
                rows=rows,
                shot_type=shot_type,
                shot_label=shot_label,
                value_idx_a=4,
                value_idx_b=5,
                title="Rationality estimates (log10(lambda), higher=more rational)",
                out_prefix="rat",
                descending=True,
            )
        )

    if not csv_rows:
        raise ValueError("No shot-type season data found; CSV was not written")

    csv_rows.sort(key=lambda r: (r[2], r[0], r[1]))
    csv_out = output_dir / csv_filename
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player_id_hawks", "season", "shot_type", "sigma_value", "log10_rationality"])
        for pid, season, shot_type, sigma, log10_rationality in csv_rows:
            writer.writerow([
                pid,
                season,
                shot_type,
                f"{sigma:.3f}",
                "" if np.isnan(log10_rationality) else f"{log10_rationality:.3f}",
            ])

    return {
        "csv": csv_out,
        "pngs": png_paths,
    }

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
