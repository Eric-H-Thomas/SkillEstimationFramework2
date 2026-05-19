"""Correlate legacy vs new xG JEEDS estimates for a player list across seasons."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.plot_intermediate_estimates import load_intermediate_estimates


def _read_player_ids(players_file: Path) -> list[int]:
    player_ids: list[int] = []
    for raw in players_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.isdigit():
            raise ValueError(f"Invalid player id line '{line}' in {players_file}")
        player_ids.append(int(line))
    if not player_ids:
        raise ValueError(f"No player IDs found in {players_file}")
    return player_ids


def _season_label(season: int) -> str:
    s = str(season)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


def _last_finite(values: list[float]) -> float | None:
    for val in reversed(values):
        if np.isfinite(val):
            return float(val)
    return None


def _last_int(values: list[float]) -> int | None:
    for val in reversed(values):
        if np.isfinite(val):
            return int(val)
    return None


def _load_metric(
    data_root: Path,
    player_id: int,
    season: int,
    shot_group: str,
    metric: str,
) -> tuple[float | None, int | None]:
    csv_path = (
        data_root
        / "players"
        / f"player_{player_id}"
        / "logs"
        / shot_group
        / f"intermediate_estimates_{season}.csv"
    )
    if not csv_path.exists():
        return None, None

    data = load_intermediate_estimates(csv_path)
    shots = _last_int(data.get("shot_count", []))
    if metric == "execution_skill":
        return _last_finite(data.get("expected_execution_skill", [])), shots
    if metric == "log10_rationality":
        return _last_finite(data.get("log10_expected_rationality", [])), shots

    raise ValueError(f"Unsupported metric: {metric}")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


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


def _scatter_plot(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    if df.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["legacy_value"].to_numpy(dtype=float)
    y = df["new_value"].to_numpy(dtype=float)
    w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones_like(x)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(x, y, s=24, alpha=0.75)

    mn = float(min(np.min(x), np.min(y)))
    mx = float(max(np.max(x), np.max(y)))
    span = max(mx - mn, 1e-12)
    pad = span * 0.05
    lo = mn - pad
    hi = mx + pad
    r, slope, intercept = _weighted_corr_and_fit(x, y, w)
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.array([lo, hi])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#CC0000", lw=1.5, ls="--")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_multipanel(
    per_season: dict[int, pd.DataFrame],
    output_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    seasons = [s for s in sorted(per_season.keys()) if not per_season[s].empty]
    if not seasons:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(seasons)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(rows, cols)

    for idx, season in enumerate(seasons):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        df = per_season[season]
        x = df["legacy_value"].to_numpy(dtype=float)
        y = df["new_value"].to_numpy(dtype=float)
        w = df["weight"].to_numpy(dtype=float) if "weight" in df.columns else np.ones_like(x)
        ax.scatter(x, y, s=22, alpha=0.75)

        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))
        span = max(mx - mn, 1e-12)
        pad = span * 0.05
        lo = mn - pad
        hi = mx + pad
        r, slope, intercept = _weighted_corr_and_fit(x, y, w)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.array([lo, hi])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#CC0000", lw=1.2, ls="--")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_season_label(season))
        ax.grid(alpha=0.25)
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
        if r == rows - 1:
            ax.set_xlabel("")
        if c == 0:
            ax.set_ylabel("")

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle("Legacy vs New xG Estimates by Season", fontsize=14)
    fig.supxlabel(x_label, y=0.02, fontsize=10)
    fig.supylabel(y_label, x=0.02, fontsize=10)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.95))
    fig.subplots_adjust(bottom=0.12, left=0.08, wspace=0.12, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_correlations(
    *,
    players_file: Path,
    seasons: list[int],
    shot_group: str,
    data_root_legacy: Path,
    data_root_new: Path,
    output_dir: Path,
    metric: str,
) -> dict[str, object]:
    player_ids = _read_player_ids(players_file)

    rows: list[dict[str, object]] = []
    per_season: dict[int, list[dict[str, object]]] = {s: [] for s in seasons}

    for pid in player_ids:
        for season in seasons:
            legacy_value, legacy_shots = _load_metric(
                data_root_legacy,
                pid,
                season,
                shot_group,
                metric,
            )
            new_value, new_shots = _load_metric(
                data_root_new,
                pid,
                season,
                shot_group,
                metric,
            )
            if legacy_value is None or new_value is None:
                continue
            if not (np.isfinite(legacy_value) and np.isfinite(new_value)):
                continue
            if legacy_shots is None or new_shots is None:
                continue
            row = {
                "player_id": pid,
                "season": season,
                "legacy_value": float(legacy_value),
                "new_value": float(new_value),
                "weight": int(min(legacy_shots, new_shots)),
            }
            rows.append(row)
            per_season[season].append(row)

    all_df = pd.DataFrame(rows)
    per_season_df = {season: pd.DataFrame(items) for season, items in per_season.items()}

    summary_rows: list[dict[str, object]] = []
    for season, df in per_season_df.items():
        if df.empty:
            summary_rows.append({
                "season": season,
                "n_pairs": 0,
                "pearson_r": "",
                "spearman_r": "",
            })
            continue
        x = df["legacy_value"].to_numpy(dtype=float)
        y = df["new_value"].to_numpy(dtype=float)
        summary_rows.append({
            "season": season,
            "n_pairs": len(df),
            "pearson_r": f"{_pearson(x, y):.6f}",
            "spearman_r": f"{_spearman(x, y):.6f}",
        })

    if not all_df.empty:
        x = all_df["legacy_value"].to_numpy(dtype=float)
        y = all_df["new_value"].to_numpy(dtype=float)
        summary_rows.append({
            "season": "ALL",
            "n_pairs": len(all_df),
            "pearson_r": f"{_pearson(x, y):.6f}",
            "spearman_r": f"{_spearman(x, y):.6f}",
        })

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / f"legacy_vs_new_xg_{metric}_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    merged_csv = output_dir / f"legacy_vs_new_xg_{metric}_pairs.csv"
    if not all_df.empty:
        all_df.to_csv(merged_csv, index=False)

    label = "Execution skill (expected_execution_skill)" if metric == "execution_skill" else "log10(rationality)"
    _scatter_multipanel(
        per_season=per_season_df,
        output_path=output_dir / f"legacy_vs_new_xg_{metric}_seasons_multipanel.png",
        x_label=f"Legacy {label}",
        y_label=f"New {label}",
    )

    for season, df in per_season_df.items():
        if df.empty:
            continue
        _scatter_plot(
            df,
            output_dir / f"legacy_vs_new_xg_{metric}_season_{season}.png",
            title=f"{_season_label(season)}: Legacy vs New",
            x_label=f"Legacy {label}",
            y_label=f"New {label}",
        )

    if not all_df.empty:
        _scatter_plot(
            all_df,
            output_dir / f"legacy_vs_new_xg_{metric}_overall.png",
            title="All Seasons: Legacy vs New",
            x_label=f"Legacy {label}",
            y_label=f"New {label}",
        )

    return {
        "summary_csv": str(summary_csv),
        "pairs_csv": str(merged_csv),
        "multipanel_png": str(output_dir / f"legacy_vs_new_xg_{metric}_seasons_multipanel.png"),
        "overall_png": str(output_dir / f"legacy_vs_new_xg_{metric}_overall.png"),
        "rows": len(all_df),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare legacy vs new xG JEEDS estimates and compute correlations.",
    )
    parser.add_argument(
        "--players-file",
        type=Path,
        default=Path("Data/Hockey/forwards23-25.txt"),
        help="Player list (one ID per line).",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="Season tags, e.g. 20212022 20222023 20232024 20242025",
    )
    parser.add_argument(
        "--shot-group",
        default="wristshot_snapshot",
        help="Shot group subdirectory under logs/.",
    )
    parser.add_argument(
        "--data-root-legacy",
        type=Path,
        default=Path("Data/Hockey"),
        help="Root directory for legacy run logs.",
    )
    parser.add_argument(
        "--data-root-new",
        type=Path,
        default=Path("Data/Hockey_xg_new"),
        help="Root directory for new xG run logs.",
    )
    parser.add_argument(
        "--metric",
        choices=["execution_skill", "log10_rationality"],
        default="execution_skill",
        help="Metric to compare between runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey_xg_new/compare_legacy_new_xg"),
        help="Output directory for CSV and plots.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    result = run_correlations(
        players_file=args.players_file,
        seasons=[int(s) for s in args.seasons],
        shot_group=args.shot_group,
        data_root_legacy=args.data_root_legacy,
        data_root_new=args.data_root_new,
        output_dir=args.output_dir,
        metric=args.metric,
    )

    print("Legacy vs new xG correlation complete")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Pairs CSV:   {result['pairs_csv']}")
    print(f"Multipanel:  {result['multipanel_png']}")
    print(f"Overall:     {result['overall_png']}")
    print(f"Pairs used:  {result['rows']}")


if __name__ == "__main__":
    main()
