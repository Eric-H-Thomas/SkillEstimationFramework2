"""Compare legacy vs new executed post-shot xG season totals per player."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from BlackhawksApp.data_io import add_post_shot_xg_column
from BlackhawksSkillEstimation.run_legacy_new_xg_comparison import (
    _event_id_set,
    _load_season_data,
    _prune_df_to_maps,
    _prune_shot_maps,
    _season_has_cache,
)

FilterMode = Literal["all", "intersection"]

DEFAULT_LEGACY_ROOT = Path("Data/Hockey")
DEFAULT_NEW_ROOT = Path("Data/Hockey_xg_new")
DEFAULT_OUTPUT_DIR = Path("Data/Hockey_xg_new/compare_legacy_new_executed_xg")

X_LABEL = "Legacy executed xG sum (season)"
Y_LABEL = "New executed xG sum (season)"


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


def _sum_executed_xg(
    df: pd.DataFrame,
    shot_maps: dict[int, dict[str, object]],
) -> tuple[float, int]:
    enriched = add_post_shot_xg_column(df, shot_maps)
    valid = enriched["post_shot_xg"].dropna()
    return float(valid.sum()), int(len(valid))


def _apply_filter_mode(
    legacy_df: pd.DataFrame,
    legacy_maps: dict[int, dict[str, object]],
    new_df: pd.DataFrame,
    new_maps: dict[int, dict[str, object]],
    filter_mode: FilterMode,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], pd.DataFrame, dict[int, dict[str, object]]]:
    legacy_df = legacy_df.rename(columns=str.lower)
    new_df = new_df.rename(columns=str.lower)

    if filter_mode == "intersection":
        intersection_ids = _event_id_set(legacy_df) & _event_id_set(new_df)
        legacy_df = legacy_df[legacy_df["event_id"].isin(intersection_ids)].reset_index(drop=True)
        new_df = new_df[new_df["event_id"].isin(intersection_ids)].reset_index(drop=True)
        legacy_maps = _prune_shot_maps(legacy_maps, intersection_ids)
        new_maps = _prune_shot_maps(new_maps, intersection_ids)

    legacy_df = _prune_df_to_maps(legacy_df, legacy_maps)
    new_df = _prune_df_to_maps(new_df, new_maps)
    legacy_maps = _prune_shot_maps(legacy_maps, _event_id_set(legacy_df))
    new_maps = _prune_shot_maps(new_maps, _event_id_set(new_df))
    return legacy_df, legacy_maps, new_df, new_maps


def _row_from_filtered(
    *,
    player_id: int,
    season: int,
    filter_mode: FilterMode,
    legacy_df: pd.DataFrame,
    legacy_maps: dict[int, dict[str, object]],
    new_df: pd.DataFrame,
    new_maps: dict[int, dict[str, object]],
) -> dict[str, object] | None:
    legacy_sum, legacy_n = _sum_executed_xg(legacy_df, legacy_maps)
    new_sum, new_n = _sum_executed_xg(new_df, new_maps)

    if legacy_n == 0 or new_n == 0:
        return None
    if not (np.isfinite(legacy_sum) and np.isfinite(new_sum)):
        return None

    delta = new_sum - legacy_sum
    ratio = new_sum / legacy_sum if legacy_sum != 0 else float("nan")

    return {
        "player_id": player_id,
        "season": season,
        "filter_mode": filter_mode,
        "legacy_xg_sum": legacy_sum,
        "new_xg_sum": new_sum,
        "legacy_n_shots": legacy_n,
        "new_n_shots": new_n,
        "delta": delta,
        "ratio": ratio,
        "weight": int(min(legacy_n, new_n)),
    }


def _collect_player_season_rows(
    *,
    player_id: int,
    season: int,
    legacy_root: Path,
    new_root: Path,
    filter_modes: list[FilterMode],
) -> list[dict[str, object]]:
    if not _season_has_cache(legacy_root, player_id, season) or not _season_has_cache(new_root, player_id, season):
        return []

    try:
        legacy_df, legacy_maps = _load_season_data(legacy_root, player_id, season)
        new_df, new_maps = _load_season_data(new_root, player_id, season)
    except Exception:
        return []

    rows: list[dict[str, object]] = []
    for filter_mode in filter_modes:
        ldf, lmaps, ndf, nmaps = _apply_filter_mode(
            legacy_df,
            legacy_maps,
            new_df,
            new_maps,
            filter_mode,
        )
        row = _row_from_filtered(
            player_id=player_id,
            season=season,
            filter_mode=filter_mode,
            legacy_df=ldf,
            legacy_maps=lmaps,
            new_df=ndf,
            new_maps=nmaps,
        )
        if row is not None:
            rows.append(row)
    return rows


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


def _to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["legacy_value"] = out["legacy_xg_sum"]
    out["new_value"] = out["new_xg_sum"]
    return out


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
    sizes = 12.0 + 0.35 * np.sqrt(np.clip(w, 0, None))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(x, y, s=sizes, alpha=0.75)

    mn = float(min(np.min(x), np.min(y)))
    mx = float(max(np.max(x), np.max(y)))
    span = max(mx - mn, 1e-12)
    pad = span * 0.05
    lo = mn - pad
    hi = mx + pad

    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.0, ls="-", alpha=0.6)

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
    *,
    suptitle: str,
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
        sizes = 10.0 + 0.35 * np.sqrt(np.clip(w, 0, None))
        ax.scatter(x, y, s=sizes, alpha=0.75)

        mn = float(min(np.min(x), np.min(y)))
        mx = float(max(np.max(x), np.max(y)))
        span = max(mx - mn, 1e-12)
        pad = span * 0.05
        lo = mn - pad
        hi = mx + pad

        ax.plot([lo, hi], [lo, hi], color="#888888", lw=0.8, ls="-", alpha=0.6)

        corr_r, slope, intercept = _weighted_corr_and_fit(x, y, w)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.array([lo, hi])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#CC0000", lw=1.2, ls="--")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_season_label(season))
        ax.grid(alpha=0.25)
        r_text = f"weighted r={corr_r:.3f}" if np.isfinite(corr_r) else "weighted r=N/A"
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

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(suptitle, fontsize=14)
    fig.supxlabel(x_label, y=0.02, fontsize=10)
    fig.supylabel(y_label, x=0.02, fontsize=10)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.95))
    fig.subplots_adjust(bottom=0.12, left=0.08, wspace=0.12, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_summary_rows(all_df: pd.DataFrame, seasons: list[int]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    for season in seasons:
        df = all_df[all_df["season"] == season]
        if df.empty:
            summary_rows.append({
                "season": season,
                "n_pairs": 0,
                "pearson_r": "",
                "spearman_r": "",
                "mean_delta": "",
                "mean_ratio": "",
            })
            continue
        x = df["legacy_xg_sum"].to_numpy(dtype=float)
        y = df["new_xg_sum"].to_numpy(dtype=float)
        summary_rows.append({
            "season": season,
            "n_pairs": len(df),
            "pearson_r": f"{_pearson(x, y):.6f}",
            "spearman_r": f"{_spearman(x, y):.6f}",
            "mean_delta": f"{float(np.mean(df['delta'])):.6f}",
            "mean_ratio": f"{float(np.mean(df['ratio'])):.6f}",
        })

    if not all_df.empty:
        x = all_df["legacy_xg_sum"].to_numpy(dtype=float)
        y = all_df["new_xg_sum"].to_numpy(dtype=float)
        summary_rows.append({
            "season": "ALL",
            "n_pairs": len(all_df),
            "pearson_r": f"{_pearson(x, y):.6f}",
            "spearman_r": f"{_spearman(x, y):.6f}",
            "mean_delta": f"{float(np.mean(all_df['delta'])):.6f}",
            "mean_ratio": f"{float(np.mean(all_df['ratio'])):.6f}",
        })

    return summary_rows


def _write_mode_outputs(
    *,
    all_df: pd.DataFrame,
    seasons: list[int],
    output_dir: Path,
    filter_mode: FilterMode,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_csv = output_dir / f"legacy_vs_new_executed_xg_{filter_mode}_pairs.csv"
    summary_csv = output_dir / f"legacy_vs_new_executed_xg_{filter_mode}_summary.csv"
    multipanel_png = output_dir / f"legacy_vs_new_executed_xg_{filter_mode}_seasons_multipanel.png"
    overall_png = output_dir / f"legacy_vs_new_executed_xg_{filter_mode}_overall.png"

    if not all_df.empty:
        all_df.to_csv(pairs_csv, index=False)
    pd.DataFrame(_build_summary_rows(all_df, seasons)).to_csv(summary_csv, index=False)

    plot_df = _to_plot_df(all_df)
    per_season = {
        season: _to_plot_df(all_df[all_df["season"] == season])
        for season in seasons
    }

    mode_title = "All shots" if filter_mode == "all" else "Intersection shots"
    _scatter_multipanel(
        per_season=per_season,
        output_path=multipanel_png,
        x_label=X_LABEL,
        y_label=Y_LABEL,
        suptitle=f"Legacy vs New Executed xG Sums by Season ({mode_title})",
    )
    _scatter_plot(
        plot_df,
        overall_png,
        title=f"Legacy vs New Executed xG Sums — All Seasons ({mode_title})",
        x_label=X_LABEL,
        y_label=Y_LABEL,
    )

    return {
        "filter_mode": filter_mode,
        "pairs_csv": str(pairs_csv),
        "summary_csv": str(summary_csv),
        "multipanel_png": str(multipanel_png),
        "overall_png": str(overall_png),
        "rows": len(all_df),
    }


def run_executed_xg_comparison(
    *,
    players_file: Path,
    seasons: list[int],
    legacy_root: Path,
    new_root: Path,
    output_dir: Path,
    filter_modes: list[FilterMode],
) -> dict[str, object]:
    player_ids = _read_player_ids(players_file)
    rows_by_mode: dict[FilterMode, list[dict[str, object]]] = {mode: [] for mode in filter_modes}
    results: dict[str, object] = {"modes": {}}

    for idx, player_id in enumerate(player_ids, start=1):
        for season in seasons:
            for row in _collect_player_season_rows(
                player_id=player_id,
                season=season,
                legacy_root=legacy_root,
                new_root=new_root,
                filter_modes=filter_modes,
            ):
                rows_by_mode[row["filter_mode"]].append(row)

        if idx % 25 == 0 or idx == len(player_ids):
            print(f"Processed {idx}/{len(player_ids)} players...", flush=True)

    for filter_mode in filter_modes:
        all_df = pd.DataFrame(rows_by_mode[filter_mode])
        results["modes"][filter_mode] = _write_mode_outputs(
            all_df=all_df,
            seasons=seasons,
            output_dir=output_dir,
            filter_mode=filter_mode,
        )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare legacy vs new executed post-shot xG season totals per player.",
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
        help="Season tags, e.g. 20232024 20242025",
    )
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=DEFAULT_LEGACY_ROOT,
        help="Root directory for legacy offline cache.",
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=DEFAULT_NEW_ROOT,
        help="Root directory for new xG offline cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["all", "intersection", "both"],
        default="both",
        help="Which shot sets to compare (default: both).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.filter_mode == "both":
        filter_modes: list[FilterMode] = ["all", "intersection"]
    else:
        filter_modes = [args.filter_mode]

    result = run_executed_xg_comparison(
        players_file=args.players_file,
        seasons=[int(s) for s in args.seasons],
        legacy_root=args.legacy_root,
        new_root=args.new_root,
        output_dir=args.output_dir,
        filter_modes=filter_modes,
    )

    print("Legacy vs new executed xG comparison complete")
    for mode, info in result["modes"].items():
        print(f"\n[{mode}] pairs: {info['rows']}")
        print(f"  Pairs CSV:   {info['pairs_csv']}")
        print(f"  Summary CSV: {info['summary_csv']}")
        print(f"  Multipanel:  {info['multipanel_png']}")
        print(f"  Overall:     {info['overall_png']}")


if __name__ == "__main__":
    main()
