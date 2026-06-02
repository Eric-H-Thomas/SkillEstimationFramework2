"""Plot rationality versus execution skill for legacy/new JEEDS runs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from BlackhawksSkillEstimation.maxg_evaluator import discover_ees_csvs
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


def _extract_player_id(path: Path) -> int:
    for part in path.parts:
        match = re.match(r"player_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Unable to parse player_id from path: {path}")


def _extract_model(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("player_") and "__" in part:
            suffix = part.split("__", 1)[1]
            if suffix in {"legacy", "new"}:
                return suffix
    return None


def _extract_season(path: Path) -> int | None:
    match = re.match(r"intermediate_estimates_(\d{8})$", path.stem)
    if match:
        return int(match.group(1))
    return None


def _last_finite(values: list[float]) -> float | None:
    for val in reversed(values):
        if np.isfinite(val):
            return float(val)
    return None


def _spread_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "max": float("nan"),
            "iqr": float("nan"),
            "p90_p10": float("nan"),
        }

    q = lambda p: float(np.percentile(values, p))
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p10": q(10),
        "p25": q(25),
        "median": q(50),
        "p75": q(75),
        "p90": q(90),
        "max": float(np.max(values)),
        "iqr": q(75) - q(25),
        "p90_p10": q(90) - q(10),
    }


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


def _weighted_corr_and_fit(xs: np.ndarray, ys: np.ndarray, ws: np.ndarray) -> tuple[float, float, float]:
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


def _season_label(season: int) -> str:
    s = str(season)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


def _load_rows(
    *,
    players: list[int],
    seasons: list[int],
    data_root: Path,
    shot_group: str,
    model_filter: str,
    source_label: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for season in seasons:
        for path in discover_ees_csvs(data_root, str(season), shot_group):
            pid = _extract_player_id(path)
            if pid not in players:
                continue

            model = _extract_model(path)
            if model is not None and model != model_filter:
                continue

            season_id = _extract_season(path)
            if season_id is None:
                continue

            data = load_intermediate_estimates(path)
            xskill = _last_finite(data.get("expected_execution_skill", []))
            rationality = _last_finite(data.get("log10_expected_rationality", []))
            shots = _last_finite(data.get("shot_count", []))
            if xskill is None or rationality is None or shots is None:
                continue
            if not (np.isfinite(xskill) and np.isfinite(rationality)):
                continue

            rows.append(
                {
                    "player_id": pid,
                    "season": int(season_id),
                    "model": source_label if model is None else model,
                    "xskill_ees": float(xskill),
                    "log10_expected_rationality": float(rationality),
                    "shots": int(shots),
                    "csv_path": str(path),
                }
            )

    return rows


def _plot_scatter(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    color_by: str | None = None,
) -> None:
    if df.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = df["xskill_ees"].to_numpy(dtype=float)
    y = df["log10_expected_rationality"].to_numpy(dtype=float)
    w = df["shots"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    if color_by and color_by in df.columns:
        categories = sorted(df[color_by].dropna().unique())
        cmap = plt.get_cmap("tab10")
        for idx, category in enumerate(categories):
            mask = df[color_by] == category
            ax.scatter(
                df.loc[mask, "xskill_ees"],
                df.loc[mask, "log10_expected_rationality"],
                s=np.clip(df.loc[mask, "shots"].to_numpy(dtype=float) / 8.0, 12, 48),
                alpha=0.72,
                label=str(category),
                color=cmap(idx % 10),
                edgecolors="none",
            )
    else:
        ax.scatter(x, y, s=np.clip(w / 8.0, 12, 48), alpha=0.72, edgecolors="none")

    mn_x = float(np.min(x))
    mx_x = float(np.max(x))
    mn_y = float(np.min(y))
    mx_y = float(np.max(y))
    pad_x = max(mx_x - mn_x, 1e-12) * 0.05
    pad_y = max(mx_y - mn_y, 1e-12) * 0.05
    lo_x = mn_x - pad_x
    hi_x = mx_x + pad_x
    lo_y = mn_y - pad_y
    hi_y = mx_y + pad_y

    r, slope, intercept = _weighted_corr_and_fit(x, y, w)
    if np.isfinite(slope) and np.isfinite(intercept):
        x_line = np.array([lo_x, hi_x])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#CC0000", lw=1.5, ls="--")

    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(lo_y, hi_y)
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

    if color_by and color_by in df.columns:
        ax.legend(title=color_by, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_season_multipanel(per_season: dict[int, pd.DataFrame], output_path: Path, title: str) -> None:
    seasons = [season for season in sorted(per_season.keys()) if not per_season[season].empty]
    if not seasons:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(4, len(seasons))
    rows = int(np.ceil(len(seasons) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 5.0 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for idx, season in enumerate(seasons):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        df = per_season[season]
        x = df["xskill_ees"].to_numpy(dtype=float)
        y = df["log10_expected_rationality"].to_numpy(dtype=float)
        w = df["shots"].to_numpy(dtype=float)
        ax.scatter(x, y, s=np.clip(w / 8.0, 12, 48), alpha=0.72, edgecolors="none")

        mn_x = float(np.min(x))
        mx_x = float(np.max(x))
        mn_y = float(np.min(y))
        mx_y = float(np.max(y))
        pad_x = max(mx_x - mn_x, 1e-12) * 0.05
        pad_y = max(mx_y - mn_y, 1e-12) * 0.05
        lo_x = mn_x - pad_x
        hi_x = mx_x + pad_x
        lo_y = mn_y - pad_y
        hi_y = mx_y + pad_y
        r_stat, slope, intercept = _weighted_corr_and_fit(x, y, w)
        if np.isfinite(slope) and np.isfinite(intercept):
            x_line = np.array([lo_x, hi_x])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color="#CC0000", lw=1.2, ls="--")

        ax.set_xlim(lo_x, hi_x)
        ax.set_ylim(lo_y, hi_y)
        ax.set_title(_season_label(season))
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"weighted r={r_stat:.3f}" if np.isfinite(r_stat) else "weighted r=N/A",
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

    for idx in range(len(seasons), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.supxlabel("expected_execution_skill", y=0.02, fontsize=10)
    fig.supylabel("log10(expected_rationality)", x=0.02, fontsize=10)
    fig.tight_layout(rect=(0.04, 0.08, 1, 0.95))
    fig.subplots_adjust(bottom=0.12, left=0.08, wspace=0.14, hspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_diagnostic(
    *,
    players_file: Path,
    seasons: list[int],
    data_root_legacy: Path,
    data_root_new: Path,
    shot_group: str,
    output_dir: Path,
) -> dict[str, object]:
    player_ids = _read_player_ids(players_file)

    rows: list[dict[str, object]] = []
    rows.extend(
        _load_rows(
            players=player_ids,
            seasons=seasons,
            data_root=data_root_legacy,
            shot_group=shot_group,
            model_filter="legacy",
            source_label="legacy",
        )
    )
    rows.extend(
        _load_rows(
            players=player_ids,
            seasons=seasons,
            data_root=data_root_new,
            shot_group=shot_group,
            model_filter="new",
            source_label="new",
        )
    )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for model in ("legacy", "new"):
        model_df = df[df["model"] == model]
        x_vals = model_df["xskill_ees"].dropna().to_numpy(dtype=float)
        y_vals = model_df["log10_expected_rationality"].dropna().to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": model,
                "scope": "overall",
                "n_pairs": int(len(model_df)),
                "pearson_r": _pearson(x_vals, y_vals),
                "spearman_r": _spearman(x_vals, y_vals),
                "xskill_stats": str(_spread_stats(x_vals)),
                "rationality_stats": str(_spread_stats(y_vals)),
            }
        )

        for season in seasons:
            s_df = model_df[model_df["season"] == season]
            x_vals = s_df["xskill_ees"].dropna().to_numpy(dtype=float)
            y_vals = s_df["log10_expected_rationality"].dropna().to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model": model,
                    "scope": str(season),
                    "n_pairs": int(len(s_df)),
                    "pearson_r": _pearson(x_vals, y_vals),
                    "spearman_r": _spearman(x_vals, y_vals),
                    "xskill_stats": str(_spread_stats(x_vals)),
                    "rationality_stats": str(_spread_stats(y_vals)),
                }
            )

    summary_csv = output_dir / "rationality_vs_xskill_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    pairs_csv = output_dir / "rationality_vs_xskill_pairs.csv"
    if not df.empty:
        df.to_csv(pairs_csv, index=False)

    combined_plot = output_dir / "rationality_vs_xskill_combined.png"
    _plot_scatter(
        df,
        combined_plot,
        title="Rationality vs xskill - Legacy and New",
        x_label="expected_execution_skill",
        y_label="log10(expected_rationality)",
        color_by="model",
    )

    for model in ("legacy", "new"):
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue
        _plot_scatter(
            model_df,
            output_dir / f"rationality_vs_xskill_{model}.png",
            title=f"Rationality vs xskill - {model.title()}",
            x_label="expected_execution_skill",
            y_label="log10(expected_rationality)",
        )
        per_season = {season: model_df[model_df["season"] == season] for season in seasons}
        _plot_season_multipanel(
            per_season,
            output_dir / f"rationality_vs_xskill_{model}_seasons.png",
            title=f"Rationality vs xskill by season - {model.title()}",
        )

    return {
        "summary_csv": str(summary_csv),
        "pairs_csv": str(pairs_csv),
        "combined_plot": str(combined_plot),
        "rows": int(len(df)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot rationality versus xskill for legacy/new JEEDS runs.")
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
        "--output-dir",
        type=Path,
        default=Path("Data/Hockey_xg_new/rationality_vs_xskill"),
        help="Output directory for CSV and plots.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_diagnostic(
        players_file=args.players_file,
        seasons=[int(s) for s in args.seasons],
        data_root_legacy=args.data_root_legacy,
        data_root_new=args.data_root_new,
        shot_group=args.shot_group,
        output_dir=args.output_dir,
    )

    print("Rationality vs xskill diagnostic complete")
    print(f"Summary CSV: {result['summary_csv']}")
    print(f"Pairs CSV:   {result['pairs_csv']}")
    print(f"Combined:    {result['combined_plot']}")
    print(f"Rows used:   {result['rows']}")


if __name__ == "__main__":
    main()