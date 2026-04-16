"""Generate JEEDS cross-season stability plots from a player-list .txt file.

This script assumes per-season intermediate CSVs already exist under:
Data/Hockey/players/player_<id>/logs/<shot_group>/intermediate_estimates_<season>.csv

Default output layout is centralized under:
Data/Hockey/general_plots/stability/<txt_stem>/
  - per_player/
  - combined/
    - population/
  - stability_summary.csv

Usage examples
--------------
py -m BlackhawksSkillEstimation.stability_plots_from_txt \
  --players-file Data/Hockey/forwards23-25.txt \
  --seasons 20212022 20222023 20232024 20242025 \
  --also-save-under-player-dirs 

python -m BlackhawksSkillEstimation.stability_plots_from_txt \
  --players-file Data/Hockey/stability9.txt \
  --jobs-config Data/Hockey/jobs/stability9.json \
  --shot-group wristshot_snapshot
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from BlackhawksSkillEstimation.player_cache import lookup_player
from BlackhawksSkillEstimation.plot_intermediate_estimates import load_intermediate_estimates


def _read_player_ids(players_file: Path) -> list[int]:
    player_ids: list[int] = []
    for raw in players_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not re.fullmatch(r"\d+", line):
            raise ValueError(f"Invalid player id line '{line}' in {players_file}")
        player_ids.append(int(line))

    if not player_ids:
        raise ValueError(f"No player IDs found in {players_file}")
    return player_ids


def _load_seasons_from_jobs_config(jobs_config: Path) -> list[int]:
    data = json.loads(jobs_config.read_text(encoding="utf-8"))
    seasons = data.get("data_filters", {}).get("seasons", [])
    out = sorted({int(s) for s in seasons})
    if not out:
        raise ValueError(f"No seasons found in jobs config: {jobs_config}")
    return out


def _auto_discover_seasons(data_root: Path, player_ids: list[int], shot_group: str) -> list[int]:
    seasons: set[int] = set()
    pattern = re.compile(r"^intermediate_estimates_(\d{8})$")

    for pid in player_ids:
        logs_dir = data_root / "players" / f"player_{pid}" / "logs" / shot_group
        if not logs_dir.exists():
            continue
        for csv_path in logs_dir.glob("intermediate_estimates_*.csv"):
            m = pattern.match(csv_path.stem)
            if m:
                seasons.add(int(m.group(1)))

    out = sorted(seasons)
    if not out:
        raise ValueError(
            "Could not auto-discover seasons from existing CSV files. "
            "Pass --seasons or --jobs-config."
        )
    return out


def _season_label(season: int) -> str:
    s = str(season)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


def _resolve_seasons(
    seasons_arg: list[int] | None,
    jobs_config: Path | None,
    data_root: Path,
    player_ids: list[int],
    shot_group: str,
) -> list[int]:
    if seasons_arg:
        return sorted({int(s) for s in seasons_arg})
    if jobs_config:
        return _load_seasons_from_jobs_config(jobs_config)
    return _auto_discover_seasons(data_root, player_ids, shot_group)


def _final_values_for_player(
    player_id: int,
    seasons: list[int],
    data_root: Path,
    shot_group: str,
) -> tuple[list[dict[str, float | int | str]], list[int]]:
    rows: list[dict[str, float | int | str]] = []
    missing: list[int] = []

    for season in seasons:
        csv_path = (
            data_root
            / "players"
            / f"player_{player_id}"
            / "logs"
            / shot_group
            / f"intermediate_estimates_{season}.csv"
        )
        if not csv_path.exists():
            missing.append(season)
            continue

        data = load_intermediate_estimates(csv_path)
        if not data["shot_count"]:
            missing.append(season)
            continue

        rows.append(
            {
                "player_id": player_id,
                "season": season,
                "execution_skill": float(data["expected_execution_skill"][-1]),
                "log10_rationality": float(data["log10_expected_rationality"][-1]),
                "shots": int(data["shot_count"][-1]),
            }
        )

    return rows, missing


def _plot_player_metric(
    player_id: int,
    player_name: str,
    rows: list[dict[str, float | int | str]],
    *,
    metric_key: str,
    y_label: str,
    title_metric: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
    annotate_variability: bool = True,
) -> None:
    ordered = sorted(rows, key=lambda r: int(r["season"]))
    xs = [int(r["season"]) for r in ordered]
    ys = [float(r[metric_key]) for r in ordered]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(xs, ys, marker="o", lw=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([_season_label(s) for s in xs], rotation=25, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_metric} Over Seasons - {player_name} ({player_id})")
    effective_limits = _expand_limits_if_needed(y_limits, ys)
    if effective_limits is not None:
        ax.set_ylim(effective_limits)

    if annotate_variability and ys:
        y_min = float(np.min(ys))
        y_max = float(np.max(ys))
        y_rng = y_max - y_min
        y_mean = float(np.mean(ys))
        pct = (100.0 * y_rng / abs(y_mean)) if not np.isclose(y_mean, 0.0) else float("nan")
        if metric_key == "execution_skill":
            detail = f"range={y_rng:.4f} rad"
        else:
            detail = f"range={y_rng:.3f} log10"
        pct_text = f", {pct:.1f}% of mean" if np.isfinite(pct) else ""
        ax.text(
            0.02,
            0.98,
            f"{detail}{pct_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="#444444",
            bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 2.0},
        )
    ax.grid(alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_metric(
    per_player_rows: dict[int, list[dict[str, float | int | str]]],
    player_names: dict[int, str],
    *,
    seasons: list[int],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    y_values_all: list[float] = []

    for pid, rows in sorted(per_player_rows.items()):
        ordered = sorted(rows, key=lambda r: int(r["season"]))
        xs = [int(r["season"]) for r in ordered]
        ys = [float(r[metric_key]) for r in ordered]
        if not xs:
            continue
        y_values_all.extend(ys)
        ax.plot(xs, ys, marker="o", lw=1.8, label=f"{player_names.get(pid, str(pid))} ({pid})")

    ax.set_xticks(seasons)
    ax.set_xticklabels([_season_label(s) for s in seasons], rotation=25, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    effective_limits = _expand_limits_if_needed(y_limits, y_values_all)
    if effective_limits is not None:
        ax.set_ylim(effective_limits)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cv(values: list[float]) -> float:
    if not values:
        return float("nan")
    mean_v = float(np.mean(values))
    if np.isclose(mean_v, 0.0):
        return float("nan")
    return float(np.std(values) / abs(mean_v))


def _write_summary_csv(
    summary_rows: list[dict[str, object]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "player_id",
        "player_name",
        "seasons_used",
        "missing_seasons",
        "mean_execution_skill",
        "std_execution_skill",
        "cv_execution_skill",
        "mean_log10_rationality",
        "std_log10_rationality",
        "cv_log10_rationality",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def _consecutive_season_pairs(seasons: list[int]) -> list[tuple[int, int]]:
    seasons_sorted = sorted(seasons)
    return [(seasons_sorted[i], seasons_sorted[i + 1]) for i in range(len(seasons_sorted) - 1)]


def _weighted_corr_and_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
) -> tuple[float, float, float]:
    """Return weighted Pearson r, weighted slope, weighted intercept."""
    if xs.size < 2:
        return float("nan"), float("nan"), float("nan")

    w_sum = float(np.sum(ws))
    if w_sum <= 0:
        return float("nan"), float("nan"), float("nan")

    mx = float(np.average(xs, weights=ws))
    my = float(np.average(ys, weights=ws))

    dx = xs - mx
    dy = ys - my
    cov = float(np.average(dx * dy, weights=ws))
    varx = float(np.average(dx * dx, weights=ws))
    vary = float(np.average(dy * dy, weights=ws))

    if varx <= 0 or vary <= 0:
        return float("nan"), float("nan"), float("nan")

    r = cov / float(np.sqrt(varx * vary))
    slope = cov / varx
    intercept = my - slope * mx
    return float(r), float(slope), float(intercept)


def _collect_population_pair_points(
    season_lookup: dict[int, dict[int, dict[str, float | int | str]]],
    season_1: int,
    season_2: int,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    n_used = 0
    n_skipped = 0

    for _pid, player_seasons in season_lookup.items():
        row_1 = player_seasons.get(season_1)
        row_2 = player_seasons.get(season_2)
        if row_1 is None or row_2 is None:
            n_skipped += 1
            continue

        x = float(row_1[metric_key])
        y = float(row_2[metric_key])
        shots_1 = int(row_1.get("shots", 0))
        shots_2 = int(row_2.get("shots", 0))
        w = float(min(shots_1, shots_2))

        if not np.isfinite(x) or not np.isfinite(y) or w <= 0:
            n_skipped += 1
            continue

        xs.append(x)
        ys.append(y)
        ws.append(w)
        n_used += 1

    return np.array(xs), np.array(ys), np.array(ws), n_used, n_skipped


def _draw_population_scatter_ax(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    *,
    season_1: int,
    season_2: int,
    metric_label: str,
) -> tuple[float, float, float]:
    label_1 = _season_label(season_1)
    label_2 = _season_label(season_2)

    # Set up axis labels and grid early.
    ax.set_title(f"{label_1} vs {label_2}")
    ax.set_xlabel(f"{label_1} {metric_label}")
    ax.set_ylabel(f"{label_2} {metric_label}")
    ax.grid(alpha=0.25)

    if xs.size == 0:
        ax.text(0.5, 0.5, "No overlapping players", ha="center", va="center", transform=ax.transAxes)
        return float("nan"), float("nan"), float("nan")

    r, slope, intercept = _weighted_corr_and_fit(xs, ys, ws)
    ax.scatter(xs, ys, s=36, alpha=0.75)

    mn = float(min(np.min(xs), np.min(ys)))
    mx = float(max(np.max(xs), np.max(ys)))
    span = max(mx - mn, 1e-12)
    pad = span * 0.05
    lo = mn - pad
    hi = mx + pad

    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.array([lo, hi])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#CC0000", lw=2.0, ls="--")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    r_text = f"weighted r={r:.3f}" if np.isfinite(r) else "weighted r=N/A"
    ax.text(
        0.02,
        0.98,
        r_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#333333",
        bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 2.0},
    )

    return r, slope, intercept


def _plot_population_pair_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    ws: np.ndarray,
    *,
    season_1: int,
    season_2: int,
    metric_label: str,
    output_path: Path,
) -> tuple[float, float, float]:
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    r, slope, intercept = _draw_population_scatter_ax(
        ax,
        xs,
        ys,
        ws,
        season_1=season_1,
        season_2=season_2,
        metric_label=metric_label,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return r, slope, intercept


def _plot_population_pairs_multipanel(
    pair_payloads: list[dict[str, object]],
    *,
    metric_label: str,
    output_path: Path,
) -> None:
    if not pair_payloads:
        return

    n = len(pair_payloads)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.8))
    if n == 1:
        axes = [axes]

    for ax, payload in zip(axes, pair_payloads):
        _draw_population_scatter_ax(
            ax,
            payload["xs"],
            payload["ys"],
            payload["ws"],
            season_1=int(payload["season_1"]),
            season_2=int(payload["season_2"]),
            metric_label=metric_label,
        )

    fig.suptitle("Population Stability: Consecutive Season Correlations", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_population_stats_csv(
    rows: list[dict[str, object]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "season_1",
        "season_2",
        "pair_label",
        "metric",
        "weighted_r",
        "weighted_slope",
        "weighted_intercept",
        "n_used",
        "n_skipped",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _expand_limits_if_needed(
    base_limits: tuple[float, float] | None,
    values: list[float],
    pad_ratio: float = 0.05,
) -> tuple[float, float] | None:
    """Expand provided y-limits when data exceeds them.

    Keeps fixed limits unchanged when all values are in range. If any value is out
    of bounds, returns expanded limits with a small margin so markers/lines are
    fully visible.
    """
    if base_limits is None or not values:
        return base_limits

    lo, hi = float(base_limits[0]), float(base_limits[1])
    data_min = float(np.min(values))
    data_max = float(np.max(values))

    if data_min >= lo and data_max <= hi:
        return (lo, hi)

    new_lo = min(lo, data_min)
    new_hi = max(hi, data_max)
    span = max(new_hi - new_lo, 1e-12)
    pad = span * pad_ratio
    return (new_lo - pad, new_hi + pad)


def run_stability_plots_from_txt(
    *,
    players_file: Path,
    data_root: Path = Path("Data/Hockey"),
    shot_group: str = "wristshot_snapshot",
    seasons: list[int] | None = None,
    jobs_config: Path | None = None,
    output_root: Path = Path("Data/Hockey/general_plots/stability"),
    include_per_player: bool = True,
    include_combined: bool = True,
    also_save_under_player_dirs: bool = False,
    annotate_variability: bool = True,
) -> dict[str, object]:
    player_ids = _read_player_ids(players_file)
    seasons_resolved = _resolve_seasons(seasons, jobs_config, data_root, player_ids, shot_group)

    run_name = players_file.stem
    run_dir = output_root / run_name
    per_player_dir = run_dir / "per_player"
    combined_dir = run_dir / "combined"
    population_dir = run_dir / "population"
    execution_limits = (0.04, 0.14)
    rationality_limits = (1.0, 2.2)

    all_rows_by_player: dict[int, list[dict[str, float | int | str]]] = {}
    player_names: dict[int, str] = {}
    summary_rows: list[dict[str, object]] = []

    for pid in player_ids:
        pname = lookup_player(pid) or str(pid)
        player_names[pid] = pname

        rows, missing = _final_values_for_player(
            player_id=pid,
            seasons=seasons_resolved,
            data_root=data_root,
            shot_group=shot_group,
        )
        all_rows_by_player[pid] = rows

        if rows and include_per_player:
            _plot_player_metric(
                player_id=pid,
                player_name=pname,
                rows=rows,
                metric_key="execution_skill",
                y_label="Final expected execution skill (rad, lower = better)",
                title_metric="Execution Skill Stability",
                output_path=per_player_dir / f"player_{pid}_execution_stability.png",
                y_limits=execution_limits,
                annotate_variability=annotate_variability,
            )
            _plot_player_metric(
                player_id=pid,
                player_name=pname,
                rows=rows,
                metric_key="log10_rationality",
                y_label="Final expected log10(rationality)",
                title_metric="Rationality Stability",
                output_path=per_player_dir / f"player_{pid}_rationality_stability.png",
                y_limits=rationality_limits,
                annotate_variability=annotate_variability,
            )

            if also_save_under_player_dirs:
                player_stability_dir = (
                    data_root
                    / "players"
                    / f"player_{pid}"
                    / "plots"
                    / "stability"
                    / run_name
                )
                _plot_player_metric(
                    player_id=pid,
                    player_name=pname,
                    rows=rows,
                    metric_key="execution_skill",
                    y_label="Final expected execution skill (rad, lower = better)",
                    title_metric="Execution Skill Stability",
                    output_path=player_stability_dir / "execution_stability.png",
                    y_limits=execution_limits,
                    annotate_variability=annotate_variability,
                )
                _plot_player_metric(
                    player_id=pid,
                    player_name=pname,
                    rows=rows,
                    metric_key="log10_rationality",
                    y_label="Final expected log10(rationality)",
                    title_metric="Rationality Stability",
                    output_path=player_stability_dir / "rationality_stability.png",
                    y_limits=rationality_limits,
                    annotate_variability=annotate_variability,
                )

        exec_values = [float(r["execution_skill"]) for r in rows]
        rat_values = [float(r["log10_rationality"]) for r in rows]

        summary_rows.append(
            {
                "player_id": pid,
                "player_name": pname,
                "seasons_used": ";".join(str(int(r["season"])) for r in sorted(rows, key=lambda x: int(x["season"]))),
                "missing_seasons": ";".join(str(s) for s in missing),
                "mean_execution_skill": float(np.mean(exec_values)) if exec_values else float("nan"),
                "std_execution_skill": float(np.std(exec_values)) if exec_values else float("nan"),
                "cv_execution_skill": _cv(exec_values),
                "mean_log10_rationality": float(np.mean(rat_values)) if rat_values else float("nan"),
                "std_log10_rationality": float(np.std(rat_values)) if rat_values else float("nan"),
                "cv_log10_rationality": _cv(rat_values),
            }
        )

    if include_combined:
        nonempty = {pid: rows for pid, rows in all_rows_by_player.items() if rows}
        if nonempty:
            _plot_combined_metric(
                per_player_rows=nonempty,
                player_names=player_names,
                seasons=seasons_resolved,
                metric_key="execution_skill",
                y_label="Final expected execution skill (rad, lower = better)",
                title=f"Execution Skill Stability - {run_name}",
                output_path=combined_dir / "combined_execution_stability.png",
                y_limits=execution_limits,
            )
            _plot_combined_metric(
                per_player_rows=nonempty,
                player_names=player_names,
                seasons=seasons_resolved,
                metric_key="log10_rationality",
                y_label="Final expected log10(rationality)",
                title=f"Rationality Stability - {run_name}",
                output_path=combined_dir / "combined_rationality_stability.png",
                y_limits=rationality_limits,
            )

    summary_csv = run_dir / "stability_summary.csv"
    _write_summary_csv(summary_rows=summary_rows, output_csv=summary_csv)

    # Population-level stability: consecutive season pair correlations.
    season_lookup = {pid: {int(r["season"]): r for r in rows} for pid, rows in all_rows_by_player.items()}
    pair_payloads: list[dict[str, object]] = []
    pair_stats_rows: list[dict[str, object]] = []
    for season_1, season_2 in _consecutive_season_pairs(seasons_resolved):
        xs, ys, ws, n_used, n_skipped = _collect_population_pair_points(
            season_lookup,
            season_1,
            season_2,
            "execution_skill",
        )

        pair_png = population_dir / f"execution_scatter_{season_1}_vs_{season_2}.png"
        weighted_r, weighted_slope, weighted_intercept = _plot_population_pair_scatter(
            xs,
            ys,
            ws,
            season_1=season_1,
            season_2=season_2,
            metric_label="Final expected execution skill (rad)",
            output_path=pair_png,
        )

        pair_payloads.append(
            {
                "season_1": season_1,
                "season_2": season_2,
                "xs": xs,
                "ys": ys,
                "ws": ws,
            }
        )

        pair_stats_rows.append(
            {
                "season_1": season_1,
                "season_2": season_2,
                "pair_label": f"{season_1}_vs_{season_2}",
                "metric": "execution_skill",
                "weighted_r": f"{weighted_r:.6f}" if np.isfinite(weighted_r) else "",
                "weighted_slope": f"{weighted_slope:.6f}" if np.isfinite(weighted_slope) else "",
                "weighted_intercept": f"{weighted_intercept:.6f}" if np.isfinite(weighted_intercept) else "",
                "n_used": n_used,
                "n_skipped": n_skipped,
            }
        )

    multipanel_png = population_dir / "execution_scatter_consecutive_pairs.png"
    _plot_population_pairs_multipanel(
        pair_payloads,
        metric_label="Final expected execution skill (rad)",
        output_path=multipanel_png,
    )

    population_stats_csv = population_dir / "population_pair_stats.csv"
    _write_population_stats_csv(pair_stats_rows, population_stats_csv)

    return {
        "players_file": str(players_file),
        "run_dir": str(run_dir),
        "per_player_dir": str(per_player_dir),
        "combined_dir": str(combined_dir),
        "population_dir": str(population_dir),
        "summary_csv": str(summary_csv),
        "population_stats_csv": str(population_stats_csv),
        "player_count": len(player_ids),
        "seasons": seasons_resolved,
        "execution_ylim": execution_limits,
        "rationality_ylim": rationality_limits,
    }


def _default_jobs_config(players_file: Path, data_root: Path) -> Path | None:
    candidate = data_root / "jobs" / f"{players_file.stem}.json"
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-season JEEDS stability plots from a player-list txt file.")
    parser.add_argument("--players-file", required=True, type=Path, help="Path to txt file containing one player ID per line.")
    parser.add_argument("--data-root", type=Path, default=Path("Data/Hockey"), help="Root hockey data directory.")
    parser.add_argument("--shot-group", default="wristshot_snapshot", help="Shot-group subdirectory under logs/.")
    parser.add_argument("--seasons", type=int, nargs="+", help="Explicit seasons, e.g. 20212022 20222023 20232024 20242025")
    parser.add_argument("--jobs-config", type=Path, help="Optional config JSON to infer seasons from data_filters.seasons.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Hockey/general_plots/stability"),
        help="Root output directory. Outputs go to <output-root>/<players-file-stem>/...",
    )
    parser.add_argument("--no-per-player", action="store_true", help="Skip per-player plots.")
    parser.add_argument("--no-combined", action="store_true", help="Skip combined plots.")
    parser.add_argument(
        "--no-variability-annotation",
        action="store_true",
        help="Disable range/% annotation text on per-player plots.",
    )
    parser.add_argument(
        "--also-save-under-player-dirs",
        action="store_true",
        help=(
            "Also mirror per-player plots to "
            "Data/Hockey/players/player_<id>/plots/stability/<players-file-stem>/."
        ),
    )

    args = parser.parse_args()

    players_file = args.players_file
    jobs_config = args.jobs_config or _default_jobs_config(players_file, args.data_root)

    result = run_stability_plots_from_txt(
        players_file=players_file,
        data_root=args.data_root,
        shot_group=args.shot_group,
        seasons=args.seasons,
        jobs_config=jobs_config,
        output_root=args.output_root,
        include_per_player=not args.no_per_player,
        include_combined=not args.no_combined,
        also_save_under_player_dirs=args.also_save_under_player_dirs,
        annotate_variability=not args.no_variability_annotation,
    )

    print("Stability plots complete")
    print(f"Run directory: {result['run_dir']}")
    print(f"Summary CSV:   {result['summary_csv']}")
    print(f"Population:    {result['population_dir']}")
    print(f"Pair stats:    {result['population_stats_csv']}")
    print(f"Players:       {result['player_count']}")
    print(f"Seasons:       {result['seasons']}")
    print(f"Skill ylim:    {result['execution_ylim']}")
    print(f"Rat ylim:      {result['rationality_ylim']}")


if __name__ == "__main__":
    main()
