"""Plot the player subsample-stability results.

Answers the question the experiment was built for: when the same player is
re-estimated on many random N-shot subsets of his pooled career, how much do the
estimates move, and is that spread big enough to explain the season-to-season jumps
we already see?

Reads whatever result JSONs exist, so a partially finished array still plots.

Figures
-------
``subsample_by_n``        Spread of subsample estimates at each N, with the
                          full-sample baseline and the player's actual season
                          estimates on the same axes.
``spread_vs_n``           How that spread shrinks with N, against the observed
                          season-to-season spread.
``jeeds_vs_mcse``         Do the two estimators react the same way to the same draw?
``season_mix_<estimator>`` Does a draw's season composition drive its estimate?

Usage
-----
    python -m BlackhawksSkillEstimation.analysis.plot_player_subsample_stability \
      --config Data/Hockey/jobs/player_subsample_950160.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.player_cache import lookup_player
from BlackhawksSkillEstimation.player_subsample_stability import (
    BASELINE_SAMPLE_KEY,
    DEFAULT_DATA_ROOT,
    load_results,
    plots_dir,
    summaries_dir,
)
from Estimators.joint import hockey_rationality_log10_bounds

# (column, axis label, short label)
METRICS: tuple[tuple[str, str, str], ...] = (
    ("exec_skill", "Execution skill (rad, lower = better)", "Execution skill"),
    ("log10_eps", "log10 rationality (EXPERIMENTAL)", "log10 rationality"),
)

ESTIMATOR_LABELS = {"jeeds": "JEEDS", "mcse": "MCSE"}
SEASON_CATEGORY = "seasons"

_BASELINE_COLOR = "#CC0000"
_SEASON_COLOR = "#1B7837"
_JITTER_SEED = 12345


def _season_label(season: int) -> str:
    text = str(season)
    return f"{text[:4]}-{text[4:]}" if len(text) == 8 else text


def results_frame(records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Flatten result JSONs into one row per (estimator, sample)."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "success":
            continue
        row: dict[str, Any] = {
            "estimator": record.get("estimator"),
            "sample_key": record.get("sample_key"),
            "is_baseline": bool(record.get("is_baseline", False)),
            "n_requested": record.get("n_requested"),
            "seed": record.get("seed"),
            "num_shots": record.get("num_shots"),
            "exec_skill": record.get("exec_skill"),
            "map_exec_skill": record.get("map_exec_skill"),
            "log10_eps": record.get("log10_eps"),
            "log10_map_rationality": record.get("log10_map_rationality"),
            "runtime_seconds": record.get("runtime_seconds"),
        }
        for season, fraction in (record.get("season_fractions") or {}).items():
            row[f"frac_{season}"] = fraction
        for season, count in (record.get("season_counts") or {}).items():
            row[f"count_{season}"] = count
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["estimator", "is_baseline", "n_requested", "seed"])
    return frame.reset_index(drop=True)


def load_season_finals(
    player_id: int,
    *,
    shot_group: str,
    data_root: Path,
) -> pd.DataFrame:
    """Final per-season estimates from the existing production runs.

    These are the numbers whose instability motivated the experiment, so they are the
    reference the subsample spread has to be compared against. Seasons that were never
    run per-season are simply absent.
    """
    logs_root = data_root / "players" / f"player_{player_id}" / "logs"
    sources = {
        "jeeds": logs_root / shot_group,
        "mcse": logs_root / "mcse" / shot_group,
    }

    rows: list[dict[str, Any]] = []
    for estimator, directory in sources.items():
        if not directory.exists():
            continue
        for csv_path in sorted(directory.glob("intermediate_estimates_*.csv")):
            season_text = csv_path.stem.removeprefix("intermediate_estimates_")
            if not (len(season_text) == 8 and season_text.isdigit()):
                continue
            try:
                data = pd.read_csv(csv_path)
            except (pd.errors.ParserError, pd.errors.EmptyDataError):
                print(f"  WARNING: could not read {csv_path}")
                continue
            if data.empty:
                continue

            last = data.iloc[-1]
            if estimator == "jeeds":
                exec_skill = float(last["expected_execution_skill"])
            else:
                exec_skill = float(np.sqrt(float(last["ees_y"]) * float(last["ees_z"])))

            rows.append(
                {
                    "estimator": estimator,
                    "season": int(season_text),
                    "num_shots": int(last["shot_count"]),
                    "exec_skill": exec_skill,
                    "log10_eps": float(last["log10_expected_rationality"]),
                }
            )

    return pd.DataFrame(rows)


def _subsample_rows(frame: pd.DataFrame, estimator: str) -> pd.DataFrame:
    return frame[(frame["estimator"] == estimator) & (~frame["is_baseline"])]


def _baseline_value(frame: pd.DataFrame, estimator: str, metric: str) -> float:
    match = frame[(frame["estimator"] == estimator) & (frame["is_baseline"])]
    if match.empty:
        return float("nan")
    return float(match.iloc[0][metric])


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _safe_pearson(xs: np.ndarray, ys: np.ndarray, *, min_n: int = 3) -> float:
    """Pearson r, or NaN when it would be undefined or meaningless.

    Guards the two cases a partially finished array produces: too few points (r on
    two points is always +/-1) and a constant axis, which makes numpy divide by zero.
    """
    if xs.size < min_n or ys.size < min_n:
        return float("nan")
    if np.std(xs) < 1e-12 or np.std(ys) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _apply_metric_limits(ax: plt.Axes, metric: str, values: Sequence[float]) -> None:
    """Keep rationality on its grid so a posterior pinned at an edge stays visible."""
    finite = _finite(values)
    if finite.size == 0:
        return
    if metric != "log10_eps":
        return

    low, high = hockey_rationality_log10_bounds()
    pad = 0.05 * max(high - low, 1e-9)
    ax.set_ylim(min(low, float(finite.min())) - pad, max(high, float(finite.max())) + pad)


def plot_subsample_by_n(
    frame: pd.DataFrame,
    season_finals: pd.DataFrame,
    *,
    estimators: Sequence[str],
    title_prefix: str,
    output_path: Path,
) -> None:
    """Subsample spread at each N, next to the observed season-to-season spread."""
    rng = np.random.default_rng(_JITTER_SEED)
    n_values = sorted({int(x) for x in frame.loc[~frame["is_baseline"], "n_requested"].dropna()})
    categories = [f"N={n}" for n in n_values] + [SEASON_CATEGORY]

    fig, axes = plt.subplots(
        len(METRICS),
        len(estimators),
        figsize=(6.4 * len(estimators), 4.8 * len(METRICS)),
        squeeze=False,
    )

    for row, (metric, y_label, _short) in enumerate(METRICS):
        for col, estimator in enumerate(estimators):
            ax = axes[row][col]
            subsamples = _subsample_rows(frame, estimator)

            groups: list[np.ndarray] = []
            for n in n_values:
                groups.append(_finite(subsamples.loc[subsamples["n_requested"] == n, metric]))

            season_values = _finite(
                season_finals.loc[season_finals["estimator"] == estimator, metric]
                if not season_finals.empty
                else []
            )
            groups.append(season_values)

            positions = np.arange(1, len(groups) + 1)
            populated = [i for i, group in enumerate(groups) if group.size]
            if populated:
                ax.boxplot(
                    [groups[i] for i in populated],
                    positions=positions[populated],
                    widths=0.55,
                    showfliers=False,
                    medianprops={"color": "#333333"},
                )

            for i, group in enumerate(groups):
                if not group.size:
                    continue
                is_season = i == len(groups) - 1
                jitter = rng.uniform(-0.16, 0.16, size=group.size)
                ax.scatter(
                    positions[i] + jitter,
                    group,
                    s=42 if is_season else 16,
                    alpha=0.95 if is_season else 0.45,
                    color=_SEASON_COLOR if is_season else "#3C6E9F",
                    marker="D" if is_season else "o",
                    zorder=3,
                    label="Season estimates" if is_season else None,
                )

            baseline = _baseline_value(frame, estimator, metric)
            if np.isfinite(baseline):
                ax.axhline(
                    baseline,
                    color=_BASELINE_COLOR,
                    ls="--",
                    lw=1.8,
                    label="Full-sample baseline",
                )

            ax.set_xticks(positions)
            ax.set_xticklabels(categories, rotation=0)
            ax.set_ylabel(y_label)
            ax.set_title(f"{ESTIMATOR_LABELS.get(estimator, estimator)}")
            ax.grid(alpha=0.25, axis="y")
            _apply_metric_limits(
                ax, metric, np.concatenate([g for g in groups if g.size] or [np.array([])])
            )
            if row == 0 and col == 0:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=8, loc="best")

    fig.suptitle(
        f"{title_prefix}: subsample spread by sample size vs actual seasons",
        fontsize=14,
    )
    _save(fig, output_path, rect=(0, 0, 1, 0.96))


def spread_table(frame: pd.DataFrame, estimators: Sequence[str]) -> pd.DataFrame:
    """Std / IQR / range of the subsample estimates at each N."""
    rows: list[dict[str, Any]] = []
    for estimator in estimators:
        subsamples = _subsample_rows(frame, estimator)
        for n in sorted({int(x) for x in subsamples["n_requested"].dropna()}):
            for metric, _label, short in METRICS:
                values = _finite(subsamples.loc[subsamples["n_requested"] == n, metric])
                if values.size < 2:
                    continue
                q75, q25 = np.percentile(values, [75, 25])
                rows.append(
                    {
                        "estimator": estimator,
                        "metric": metric,
                        "metric_label": short,
                        "n_shots": n,
                        "num_samples": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "iqr": float(q75 - q25),
                        "range": float(values.max() - values.min()),
                    }
                )
    return pd.DataFrame(rows)


def season_spread_table(season_finals: pd.DataFrame) -> pd.DataFrame:
    """Observed season-to-season spread: the number the experiment is explaining."""
    rows: list[dict[str, Any]] = []
    if season_finals.empty:
        return pd.DataFrame(rows)

    for estimator, group in season_finals.groupby("estimator"):
        for metric, _label, short in METRICS:
            values = _finite(group[metric])
            if values.size < 2:
                continue
            q75, q25 = np.percentile(values, [75, 25])
            rows.append(
                {
                    "estimator": estimator,
                    "metric": metric,
                    "metric_label": short,
                    "num_seasons": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "iqr": float(q75 - q25),
                    "range": float(values.max() - values.min()),
                }
            )
    return pd.DataFrame(rows)


def plot_spread_vs_n(
    spread: pd.DataFrame,
    season_spread: pd.DataFrame,
    *,
    estimators: Sequence[str],
    title_prefix: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(METRICS),
        len(estimators),
        figsize=(6.4 * len(estimators), 4.4 * len(METRICS)),
        squeeze=False,
    )

    for row, (metric, y_label, short) in enumerate(METRICS):
        for col, estimator in enumerate(estimators):
            ax = axes[row][col]
            subset = spread[(spread["estimator"] == estimator) & (spread["metric"] == metric)]
            subset = subset.sort_values("n_shots")

            if not subset.empty:
                for stat, marker in (("std", "o"), ("iqr", "s"), ("range", "^")):
                    ax.plot(
                        subset["n_shots"],
                        subset[stat],
                        marker=marker,
                        lw=1.8,
                        label=stat.upper() if stat == "iqr" else stat.capitalize(),
                    )

            season_row = season_spread[
                (season_spread["estimator"] == estimator) & (season_spread["metric"] == metric)
            ]
            if not season_row.empty:
                season_std = float(season_row.iloc[0]["std"])
                ax.axhline(
                    season_std,
                    color=_SEASON_COLOR,
                    ls="--",
                    lw=1.8,
                    label="Season-to-season std",
                )

            ax.set_xscale("log")
            ax.set_xlabel("Shots per subsample (N)")
            ax.set_ylabel(f"Spread of {short.lower()}")
            ax.set_title(ESTIMATOR_LABELS.get(estimator, estimator))
            ax.grid(alpha=0.25, which="both")
            if not subset.empty:
                # Label only the N values that were actually run; the log scale's
                # default minor ticks (6x10^1 and friends) are noise here.
                ax.set_xticks(subset["n_shots"])
                ax.set_xticklabels([str(int(n)) for n in subset["n_shots"]])
                ax.xaxis.set_minor_formatter(mticker.NullFormatter())
            ax.legend(fontsize=8)

    fig.suptitle(f"{title_prefix}: does estimate spread shrink with N?", fontsize=14)
    _save(fig, output_path, rect=(0, 0, 1, 0.95))


def plot_jeeds_vs_mcse(
    frame: pd.DataFrame,
    *,
    title_prefix: str,
    output_path: Path,
) -> None:
    """Same draw, both estimators: do they agree on which subsets look good?"""
    jeeds = frame[frame["estimator"] == "jeeds"].set_index("sample_key")
    mcse = frame[frame["estimator"] == "mcse"].set_index("sample_key")
    shared = sorted(set(jeeds.index) & set(mcse.index))
    if not shared:
        print("  Skipping JEEDS vs MCSE: no sample has results from both estimators.")
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.2 * len(METRICS), 5.6), squeeze=False)

    for col, (metric, label, short) in enumerate(METRICS):
        ax = axes[0][col]
        xs = jeeds.loc[shared, metric].to_numpy(dtype=float)
        ys = mcse.loc[shared, metric].to_numpy(dtype=float)
        sizes = jeeds.loc[shared, "n_requested"].to_numpy(dtype=float)
        keep = np.isfinite(xs) & np.isfinite(ys)

        if keep.sum() >= 2:
            scatter = ax.scatter(
                xs[keep],
                ys[keep],
                c=np.nan_to_num(sizes[keep], nan=0.0),
                cmap="viridis",
                s=38,
                alpha=0.85,
            )
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label("Shots per subsample (N)\n0 = full-sample baseline")

            r = _safe_pearson(xs[keep], ys[keep])
            annotation = (
                f"Pearson r = {r:.3f}  (n={int(keep.sum())})"
                if np.isfinite(r)
                else f"n={int(keep.sum())} (too few points for r)"
            )
            ax.text(
                0.02,
                0.98,
                annotation,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2.0},
            )

            low = float(min(xs[keep].min(), ys[keep].min()))
            high = float(max(xs[keep].max(), ys[keep].max()))
            pad = 0.05 * max(high - low, 1e-9)
            ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#888888", ls=":", lw=1.2)

        ax.set_xlabel(f"JEEDS {label}")
        ax.set_ylabel(f"MCSE {label}")
        ax.set_title(short)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{title_prefix}: JEEDS vs MCSE on identical subsamples", fontsize=14)
    _save(fig, output_path, rect=(0, 0, 1, 0.94))


def plot_season_mix(
    frame: pd.DataFrame,
    *,
    estimator: str,
    seasons: Sequence[int],
    title_prefix: str,
    output_path: Path,
) -> None:
    """Estimate vs each season's share of the draw.

    Pooling seasons assumes they are exchangeable. If they are not, a seed that
    happens to over-weight one year will show up here as a sloped cloud, and the
    subsample spread cannot be read as a pure N-shot noise floor.
    """
    subsamples = _subsample_rows(frame, estimator)
    available = [s for s in seasons if f"frac_{s}" in subsamples.columns]
    if subsamples.empty or not available:
        print(f"  Skipping season mix for {estimator}: no subsample rows.")
        return

    fig, axes = plt.subplots(
        len(METRICS),
        len(available),
        figsize=(3.9 * len(available), 4.2 * len(METRICS)),
        squeeze=False,
    )

    for row, (metric, y_label, _short) in enumerate(METRICS):
        for col, season in enumerate(available):
            ax = axes[row][col]
            xs = subsamples[f"frac_{season}"].to_numpy(dtype=float)
            ys = subsamples[metric].to_numpy(dtype=float)
            sizes = subsamples["n_requested"].to_numpy(dtype=float)
            keep = np.isfinite(xs) & np.isfinite(ys)

            if keep.sum() >= 2:
                ax.scatter(
                    xs[keep],
                    ys[keep],
                    c=np.nan_to_num(sizes[keep], nan=0.0),
                    cmap="viridis",
                    s=24,
                    alpha=0.8,
                )
                r = _safe_pearson(xs[keep], ys[keep])
                if np.isfinite(r):
                    ax.text(
                        0.02,
                        0.98,
                        f"r = {r:.2f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
                    )

            baseline = _baseline_value(frame, estimator, metric)
            if np.isfinite(baseline):
                ax.axhline(baseline, color=_BASELINE_COLOR, ls="--", lw=1.2)

            ax.set_title(_season_label(season), fontsize=10)
            ax.grid(alpha=0.25)
            if row == len(METRICS) - 1:
                ax.set_xlabel("Share of subsample")
            if col == 0:
                ax.set_ylabel(y_label)

        # Share the y range across seasons so the clouds are visually comparable.
        # Blank the inner tick labels only afterwards, since changing the limits
        # would otherwise leave stale fixed ticks behind.
        limits = [axes[row][c].get_ylim() for c in range(len(available))]
        low = min(limit[0] for limit in limits)
        high = max(limit[1] for limit in limits)
        for c in range(len(available)):
            axes[row][c].set_ylim(low, high)
            if c > 0:
                axes[row][c].tick_params(labelleft=False)

    label = ESTIMATOR_LABELS.get(estimator, estimator)
    fig.suptitle(
        f"{title_prefix}: {label} estimate vs season composition of each subsample",
        fontsize=13,
    )
    _save(fig, output_path, rect=(0, 0, 1, 0.94))


def _save(fig: plt.Figure, output_path: Path, *, rect: tuple[float, float, float, float]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot player subsample-stability results.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="Config JSON written by the builder.")
    source.add_argument("--run-dir", type=Path, help="Experiment run directory.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <run-dir>/plots.")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        run_dir = Path(config["output_root"])
    else:
        run_dir = args.run_dir

    records = load_results(run_dir)
    if not records:
        raise SystemExit(f"No result JSONs found under {run_dir / 'results'}.")

    frame = results_frame(records)
    if frame.empty:
        raise SystemExit(f"Found {len(records)} result(s), but none completed successfully.")

    player_id = int(config.get("player_id") or records[0]["player_id"])
    shot_group = str(config.get("shot_group") or records[0].get("shot_group", "wristshot_snapshot"))
    seasons = [int(s) for s in (config.get("seasons") or records[0].get("seasons", []))]
    estimators = [e for e in ("jeeds", "mcse") if e in set(frame["estimator"])]

    player_name = lookup_player(player_id) or str(player_id)
    title_prefix = f"{player_name} ({player_id})"

    expected = len(config.get("cluster_plan", {}).get("jobs", [])) or len(records)
    print(f"Player:      {title_prefix}")
    print(f"Run dir:     {run_dir}")
    print(f"Results:     {len(frame)} successful of {len(records)} written ({expected} planned)")
    print(f"Estimators:  {', '.join(estimators)}")

    season_finals = load_season_finals(player_id, shot_group=shot_group, data_root=args.data_root)
    missing_season_dots = sorted(set(seasons) - set(season_finals.get("season", pd.Series(dtype=int))))
    if missing_season_dots:
        print(
            "  NOTE: no per-season estimates on disk for "
            f"{missing_season_dots}; those seasons are still in the subsample pool, "
            "so this is a plotting gap, not a sampling gap."
        )

    out_dir = args.output_dir or plots_dir(run_dir)
    summary_dir = summaries_dir(run_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(summary_dir / "final_estimates.csv", index=False)
    spread = spread_table(frame, estimators)
    spread.to_csv(summary_dir / "subsample_spread.csv", index=False)
    season_spread = season_spread_table(season_finals)
    if not season_finals.empty:
        season_finals.to_csv(summary_dir / "season_finals.csv", index=False)
        season_spread.to_csv(summary_dir / "season_spread.csv", index=False)
    print(f"  wrote {summary_dir / 'final_estimates.csv'}")

    plot_subsample_by_n(
        frame,
        season_finals,
        estimators=estimators,
        title_prefix=title_prefix,
        output_path=out_dir / "subsample_by_n.png",
    )
    plot_spread_vs_n(
        spread,
        season_spread,
        estimators=estimators,
        title_prefix=title_prefix,
        output_path=out_dir / "spread_vs_n.png",
    )
    if len(estimators) > 1:
        plot_jeeds_vs_mcse(
            frame,
            title_prefix=title_prefix,
            output_path=out_dir / "jeeds_vs_mcse.png",
        )
    for estimator in estimators:
        plot_season_mix(
            frame,
            estimator=estimator,
            seasons=seasons,
            title_prefix=title_prefix,
            output_path=out_dir / f"season_mix_{estimator}.png",
        )

    if not spread.empty:
        print("\nSubsample spread:")
        print(spread.to_string(index=False))
    if not season_spread.empty:
        print("\nObserved season-to-season spread:")
        print(season_spread.to_string(index=False))


if __name__ == "__main__":
    main()
