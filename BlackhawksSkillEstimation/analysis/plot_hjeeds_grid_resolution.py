"""Plot how HJEEDS hockey estimates depend on likelihood-grid resolution.

Every cell of the sweep uses the same players, shot draws, and seeds, so each
estimate can be paired with the same ``(player, N, seed)`` estimate on the finest
grid. Resolution error is reported in units of that player's seed-to-seed SD on
the finest grid: a shift of 0.1 means the grid moves the estimate by a tenth of
the noise from redrawing shots, which adds about 0.5% to the total spread.

Figures
-------
``resolution_shift_heatmaps``  Median shift from the finest grid at one N.
``resolution_shift_by_n``      The same shift for every grid at each N.
``cost_vs_shift``              Grid points against the largest shift.
``lambda_snapping``            Where posterior-mean log-lambda lands relative
                               to the lambda grid points.
``stability_cv_heatmaps``      Roster-median CV across redraws for each grid.
``cv_vs_pool_fraction``        Redraw CV against the share of the pool drawn.
``old_vs_new_jeeds``           Old Blackhawks JEEDS against HJEEDS independent
                               JEEDS on the same draws for the target player.

Usage
-----
    python -m BlackhawksSkillEstimation.analysis.plot_hjeeds_grid_resolution
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("Data/Hockey/experiments/hjeeds_grid_resolution")
DEFAULT_OLD_RESULTS = Path("Data/Hockey/experiments/player_subsample_stability/player_950160/results")
DEFAULT_N_SHOTS = 400
DEFAULT_TARGET_PLAYER_ID = 950160
NEGLIGIBLE_SHIFT = 0.1
MIN_SEEDS = 10
SHIFT_FLOOR = 1e-3
SNAP_TOLERANCE = 0.02
KEY = ["player_id", "n_shots", "seed"]

# (estimator, parameter, CSV column, panel label)
QUANTITIES: tuple[tuple[str, str, str, str], ...] = (
    ("jeeds", "sigma", "jeeds_mean_sigma", "JEEDS execution sigma"),
    ("jeeds", "log_lambda", "jeeds_mean_log_lambda", "JEEDS log lambda"),
    ("hierarchical", "sigma", "hierarchical_mean_sigma", "H-JEEDS execution sigma"),
    ("hierarchical", "log_lambda", "hierarchical_mean_log_lambda", "H-JEEDS log lambda"),
)

LAMBDA_COLORS = {50: "#D1495B", 100: "#EDAE49", 250: "#00798C"}
SIGMA_MARKERS = {50: "o", 100: "s", 250: "^"}
SIGMA_LINESTYLES = {50: ":", 100: "--", 250: "-"}


def grid_label(n_sigma: int, n_lambda: int) -> str:
    return f"{n_sigma}x{n_lambda}"


def load_cells(root: Path) -> tuple[pd.DataFrame, dict[tuple[int, int], dict]]:
    """Stack every finished cell and keep each manifest's grid and pool sizes."""

    frames: list[pd.DataFrame] = []
    manifests: dict[tuple[int, int], dict] = {}
    for csv_path in sorted(root.glob("sigma*_lambda*/agent_level_results.csv")):
        manifest = json.loads((csv_path.parent / "manifest.json").read_text(encoding="utf-8"))
        grid = manifest["grid"]
        n_sigma = int(grid.get("num_sigma_grid", len(grid["sigma_grid"])))
        n_lambda = int(grid.get("num_log_lambda_grid", len(grid["log_lambda_grid"])))
        frame = pd.read_csv(csv_path)
        ok = frame["jeeds_status"].eq("ok") & frame["hierarchical_status"].eq("ok")
        frame = frame[ok].copy()
        frame["n_sigma"] = n_sigma
        frame["n_lambda"] = n_lambda
        frames.append(frame)
        manifests[(n_sigma, n_lambda)] = manifest
    if not frames:
        raise FileNotFoundError(f"No finished cells under {root}.")
    return pd.concat(frames, ignore_index=True), manifests


def _cell(data: pd.DataFrame, n_sigma: int, n_lambda: int) -> pd.DataFrame:
    return data[data["n_sigma"].eq(n_sigma) & data["n_lambda"].eq(n_lambda)]


def _seed_sd(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby(["player_id", "n_shots"])[column]
    sd = grouped.std(ddof=1)
    counts = grouped.size()
    return sd[(counts >= MIN_SEEDS) & (sd > 0)]


def _roster_median_cv(frame: pd.DataFrame, column: str) -> float:
    grouped = frame.groupby("player_id")[column]
    counts = grouped.size()
    means = grouped.mean()
    sds = grouped.std(ddof=1)
    keep = (counts >= MIN_SEEDS) & (means.abs() > 0)
    if not keep.any():
        return float("nan")
    return float((sds[keep] / means[keep].abs()).median())


def _snap_fraction(values: np.ndarray, grid: np.ndarray) -> float:
    """Share of estimates within ``SNAP_TOLERANCE`` grid spacings of a grid point.

    Estimates spread smoothly between grid points give about
    ``2 * SNAP_TOLERANCE``.
    """

    if values.size == 0 or grid.size < 2:
        return float("nan")
    spacing = float(np.min(np.diff(grid)))
    nearest = np.abs(values[:, None] - grid[None, :]).min(axis=1)
    return float(np.mean(nearest < SNAP_TOLERANCE * spacing))


def resolution_table(
    data: pd.DataFrame,
    manifests: dict[tuple[int, int], dict],
) -> pd.DataFrame:
    """Per grid, N, and quantity: shift from the finest grid, CV, rank agreement."""

    reference_key = (max(k[0] for k in manifests), max(k[1] for k in manifests))
    reference = _cell(data, *reference_key)
    reference_indexed = reference.set_index(KEY)
    rows: list[dict[str, object]] = []
    for (n_sigma, n_lambda), manifest in sorted(manifests.items()):
        cell = _cell(data, n_sigma, n_lambda)
        sigma_grid = np.asarray(manifest["grid"]["sigma_grid"], dtype=float)
        log_lambda_grid = np.asarray(manifest["grid"]["log_lambda_grid"], dtype=float)
        for n_shots in sorted(cell["n_shots"].unique()):
            cell_n = cell[cell["n_shots"].eq(n_shots)]
            reference_n = reference[reference["n_shots"].eq(n_shots)]
            for estimator, parameter, column, label in QUANTITIES:
                joined = cell_n.set_index(KEY)[[column]].join(
                    reference_indexed[[column]], rsuffix="_reference", how="inner"
                )
                seed_sd = _seed_sd(reference_n, column)
                index = pd.MultiIndex.from_arrays([
                    joined.index.get_level_values("player_id"),
                    joined.index.get_level_values("n_shots"),
                ])
                sd = seed_sd.reindex(index).to_numpy()
                shift = np.abs(joined[column].to_numpy() - joined[f"{column}_reference"].to_numpy())
                ratio = shift / sd
                ratio = ratio[np.isfinite(ratio)]

                cell_means = cell_n.groupby("player_id")[column].mean()
                reference_means = reference_n.groupby("player_id")[column].mean()
                paired_means = pd.concat([cell_means, reference_means], axis=1, join="inner")
                rank = (
                    float(paired_means.corr(method="spearman").iloc[0, 1])
                    if len(paired_means) > 2
                    else float("nan")
                )
                grid = sigma_grid if parameter == "sigma" else log_lambda_grid
                rows.append({
                    "n_sigma": int(n_sigma),
                    "n_lambda": int(n_lambda),
                    "grid": grid_label(n_sigma, n_lambda),
                    "grid_points": int(n_sigma * n_lambda),
                    "is_reference": (n_sigma, n_lambda) == reference_key,
                    "n_shots": int(n_shots),
                    "estimator": estimator,
                    "parameter": parameter,
                    "quantity": label,
                    "rows": int(len(cell_n)),
                    "paired_rows": int(ratio.size),
                    "median_shift_seed_sd": float(np.median(ratio)) if ratio.size else float("nan"),
                    "p90_shift_seed_sd": float(np.quantile(ratio, 0.9)) if ratio.size else float("nan"),
                    "median_abs_shift": float(np.median(shift)) if shift.size else float("nan"),
                    "roster_median_cv": _roster_median_cv(cell_n, column),
                    "spearman_player_means": rank,
                    "grid_spacing": float(np.min(np.diff(grid))),
                    "snap_fraction": _snap_fraction(cell_n[column].to_numpy(dtype=float), grid),
                })
    return pd.DataFrame(rows)


def _counts(table: pd.DataFrame) -> tuple[list[int], list[int]]:
    return sorted(table["n_sigma"].unique()), sorted(table["n_lambda"].unique())


def _quantity_matrix(table: pd.DataFrame, label: str, value: str) -> np.ndarray:
    sigma_counts, lambda_counts = _counts(table)
    matrix = np.full((len(sigma_counts), len(lambda_counts)), np.nan)
    subset = table[table["quantity"].eq(label)]
    for _, row in subset.iterrows():
        i = sigma_counts.index(int(row["n_sigma"]))
        j = lambda_counts.index(int(row["n_lambda"]))
        matrix[i, j] = float(row[value])
    return matrix


def _annotate_heatmap(ax, matrix: np.ndarray, text: list[list[str]], dark: np.ndarray) -> None:
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, text[i][j], ha="center", va="center", fontsize=10,
                color="white" if dark[i, j] else "black",
            )


def _heatmap_axes(ax, sigma_counts: list[int], lambda_counts: list[int], title: str) -> None:
    ax.set_xticks(range(len(lambda_counts)), [str(n) for n in lambda_counts])
    ax.set_yticks(range(len(sigma_counts)), [str(n) for n in sigma_counts])
    ax.set_xlabel("log-lambda grid points")
    ax.set_ylabel("sigma grid points")
    ax.set_title(title, fontsize=11)


def plot_shift_heatmaps(table: pd.DataFrame, n_shots: int, output: Path) -> None:
    subset = table[table["n_shots"].eq(n_shots)]
    sigma_counts, lambda_counts = _counts(subset)
    norm = LogNorm(vmin=SHIFT_FLOOR, vmax=3.0)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    image = None
    for ax, (_, _, _, label) in zip(axes.flat, QUANTITIES):
        matrix = _quantity_matrix(subset, label, "median_shift_seed_sd")
        reference = _quantity_matrix(subset, label, "is_reference").astype(bool)
        shown = np.clip(matrix, SHIFT_FLOOR, None)
        image = ax.imshow(shown, origin="lower", cmap="YlOrRd", norm=norm)
        text = [
            ["ref" if reference[i, j] else f"{matrix[i, j]:.3f}" for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
        _annotate_heatmap(ax, matrix, text, shown > 0.5)
        _heatmap_axes(ax, sigma_counts, lambda_counts, label)
    fig.suptitle(
        f"How far each grid moves the estimate from the finest grid (N={n_shots})\n"
        "median |estimate - 250x250| in units of seed-to-seed SD; below 0.1 is negligible",
        fontsize=12,
    )
    fig.colorbar(image, ax=axes, shrink=0.75, label="median shift (seed-to-seed SDs)")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_shift_by_n(table: pd.DataFrame, output: Path) -> None:
    subset = table[~table["is_reference"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (_, _, _, label) in zip(axes.flat, QUANTITIES):
        rows = subset[subset["quantity"].eq(label)]
        for (n_sigma, n_lambda), group in rows.groupby(["n_sigma", "n_lambda"]):
            group = group.sort_values("n_shots")
            ax.plot(
                group["n_shots"],
                np.clip(group["median_shift_seed_sd"], SHIFT_FLOOR, None),
                color=LAMBDA_COLORS.get(int(n_lambda), "gray"),
                linestyle=SIGMA_LINESTYLES.get(int(n_sigma), "-"),
                marker=SIGMA_MARKERS.get(int(n_sigma), "o"),
                label=grid_label(int(n_sigma), int(n_lambda)),
            )
        ax.axhline(NEGLIGIBLE_SHIFT, color="black", linewidth=0.8, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(SHIFT_FLOOR * 0.8, 5.0)
        ax.set_xticks(sorted(rows["n_shots"].unique()))
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("shots per draw (N)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median shift from 250x250 (seed SDs)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(
        "Grid error grows with N because the posterior narrows (dashed line: 0.1 seed SD)\n"
        "color: lambda points; marker and line style: sigma points; overlapping lines differ only in sigma",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(
        handles, labels, title="sigma x lambda points", loc="lower center",
        ncol=len(labels), bbox_to_anchor=(0.5, -0.01), fontsize=9,
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cost_vs_shift(table: pd.DataFrame, n_shots: int, output: Path) -> None:
    subset = table[table["n_shots"].eq(n_shots)]
    worst = subset.groupby(["n_sigma", "n_lambda", "grid_points"])["median_shift_seed_sd"].max()
    worst_jeeds = (
        subset[subset["estimator"].eq("jeeds")]
        .groupby(["n_sigma", "n_lambda", "grid_points"])["median_shift_seed_sd"].max()
    )
    reference_key = subset.loc[subset["is_reference"], ["n_sigma", "n_lambda"]].iloc[0]
    reference_key = (int(reference_key["n_sigma"]), int(reference_key["n_lambda"]))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for (n_sigma, n_lambda, points), value in worst.items():
        jeeds_value = worst_jeeds.loc[(n_sigma, n_lambda, points)]
        color = LAMBDA_COLORS.get(int(n_lambda), "gray")
        marker = SIGMA_MARKERS.get(int(n_sigma), "o")
        # Grids with the same point count (50x100 and 100x50) would overlap.
        x = points * (0.92 if n_sigma < n_lambda else 1.08 if n_sigma > n_lambda else 1.0)
        top = max(value, SHIFT_FLOOR)
        ax.scatter(x, top, s=90, color=color, marker=marker, zorder=3)
        ax.scatter(x, max(jeeds_value, SHIFT_FLOOR), s=90, facecolors="none", edgecolors=color, marker=marker, zorder=3)
        ax.plot([x, x], [max(jeeds_value, SHIFT_FLOOR), top], color=color, alpha=0.4)
        label = grid_label(int(n_sigma), int(n_lambda))
        if (int(n_sigma), int(n_lambda)) == reference_key:
            label += " (reference)"
        ax.annotate(
            label, (x, top), textcoords="offset points",
            xytext=(-8, 6) if n_sigma < n_lambda else (8, 6), ha="right" if n_sigma < n_lambda else "left",
            fontsize=9,
        )
    ax.axhline(NEGLIGIBLE_SHIFT, color="black", linewidth=0.8, linestyle="--")
    ax.text(0.01, NEGLIGIBLE_SHIFT * 1.12, "0.1 seed SD", fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    points_values = sorted(worst.index.get_level_values("grid_points").unique())
    ticks = [points_values[0]]
    for value in points_values[1:]:
        if value / ticks[-1] >= 1.5:
            ticks.append(value)
    ax.set_xticks(ticks, [f"{int(t):,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(points_values[0] * 0.75, points_values[-1] * 1.45)
    ax.set_ylim(SHIFT_FLOOR * 0.7, 5.0)
    ax.set_xlabel("grid points (sigma x lambda); likelihood time scales about linearly")
    ax.set_ylabel(f"largest median shift from 250x250 at N={n_shots}\n(seed-to-seed SDs)")
    ax.set_title(
        "Cost against resolution error\n"
        "filled: worst of JEEDS and H-JEEDS; hollow: JEEDS only; 250x250 is the reference",
        fontsize=11,
    )
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_snapping(
    data: pd.DataFrame,
    manifests: dict[tuple[int, int], dict],
    table: pd.DataFrame,
    n_shots: int,
    output: Path,
) -> None:
    n_sigma = max(k[0] for k in manifests)
    lambda_counts = sorted({k[1] for k in manifests if k[0] == n_sigma})
    rows = (("jeeds_mean_log_lambda", "JEEDS log lambda"), ("hierarchical_mean_log_lambda", "H-JEEDS log lambda"))
    fig, axes = plt.subplots(len(rows), len(lambda_counts), figsize=(4.2 * len(lambda_counts), 7.5), squeeze=False)
    for r, (column, label) in enumerate(rows):
        pooled = data[data["n_sigma"].eq(n_sigma) & data["n_shots"].eq(n_shots)][column].to_numpy(dtype=float)
        low, high = np.quantile(pooled, [0.005, 0.995])
        pad = 0.05 * (high - low)
        bins = np.linspace(low - pad, high + pad, 90)
        for c, n_lambda in enumerate(lambda_counts):
            ax = axes[r, c]
            values = _cell(data, n_sigma, n_lambda)
            values = values[values["n_shots"].eq(n_shots)][column].to_numpy(dtype=float)
            grid = np.asarray(manifests[(n_sigma, n_lambda)]["grid"]["log_lambda_grid"], dtype=float)
            ax.hist(values, bins=bins, color=LAMBDA_COLORS.get(n_lambda, "gray"), zorder=2)
            for point in grid[(grid >= bins[0]) & (grid <= bins[-1])]:
                ax.axvline(point, color="black", linewidth=0.8, linestyle=":", alpha=0.6, zorder=3)
            row = table[
                table["n_sigma"].eq(n_sigma) & table["n_lambda"].eq(n_lambda)
                & table["n_shots"].eq(n_shots) & table["quantity"].eq(label)
            ].iloc[0]
            ax.text(
                0.02, 0.97,
                f"on a grid point: {row['snap_fraction']:.0%}\n"
                f"roster median CV: {row['roster_median_cv']:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            ax.set_title(f"{label}, {n_lambda} lambda points", fontsize=10)
            ax.set_xlim(bins[0], bins[-1])
            if r == len(rows) - 1:
                ax.set_xlabel("posterior mean log lambda")
            if c == 0:
                ax.set_ylabel("player x seed estimates")
    fig.suptitle(
        f"Posterior means relative to the lambda grid (dotted lines), {n_sigma} sigma points, N={n_shots}\n"
        f"'on a grid point' is within {SNAP_TOLERANCE:.0%} of a grid spacing; smooth estimates give about "
        f"{2 * SNAP_TOLERANCE:.0%}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cv_heatmaps(table: pd.DataFrame, n_shots: int, output: Path) -> None:
    subset = table[table["n_shots"].eq(n_shots)]
    sigma_counts, lambda_counts = _counts(subset)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, (_, _, _, label) in zip(axes.flat, QUANTITIES):
        matrix = _quantity_matrix(subset, label, "roster_median_cv")
        reference = matrix[-1, -1]
        relative = matrix / reference
        image = ax.imshow(relative, origin="lower", cmap="RdBu", vmin=0.0, vmax=2.0)
        text = [
            [f"{matrix[i, j]:.4f}\n({relative[i, j]:.2f}x)" for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
        _annotate_heatmap(ax, matrix, text, np.abs(relative - 1.0) > 0.6)
        _heatmap_axes(ax, sigma_counts, lambda_counts, label)
        fig.colorbar(image, ax=ax, shrink=0.8, label="CV relative to 250x250")
    fig.suptitle(
        f"Roster-median coefficient of variation across redraws (N={n_shots})\n"
        "below 1x means the grid makes estimates look more stable than the finest grid",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cv_vs_pool_fraction(
    data: pd.DataFrame,
    manifests: dict[tuple[int, int], dict],
    output: Path,
) -> None:
    reference_key = (max(k[0] for k in manifests), max(k[1] for k in manifests))
    reference = _cell(data, *reference_key)
    pools = {int(k): int(v) for k, v in manifests[reference_key]["available_pool_sizes"].items()}
    columns = (("jeeds_mean_sigma", "JEEDS execution sigma"), ("hierarchical_mean_sigma", "H-JEEDS execution sigma"))
    n_values = sorted(reference["n_shots"].unique())
    fig, axes = plt.subplots(1, len(columns), figsize=(11, 4.5), sharey=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(n_values)))
    for ax, (column, label) in zip(axes, columns):
        for color, n_shots in zip(colors, n_values):
            frame = reference[reference["n_shots"].eq(n_shots)]
            grouped = frame.groupby("player_id")[column]
            counts = grouped.size()
            cv = (grouped.std(ddof=1) / grouped.mean().abs())[counts >= MIN_SEEDS]
            fraction = np.asarray([n_shots / pools[int(pid)] for pid in cv.index])
            rho = pd.Series(cv.to_numpy()).corr(pd.Series(fraction), method="spearman")
            ax.scatter(fraction, cv, s=12, alpha=0.6, color=color, label=f"N={n_shots} (Spearman {rho:.2f})")
        ax.set_xlabel("share of the player's pool drawn (N / pool size)")
        ax.set_ylabel("CV across 50 redraws")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Redraws from a small pool overlap, so they look more stable ({grid_label(*reference_key)} grid)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def load_old_blackhawks_jeeds(results_dir: Path) -> pd.DataFrame:
    """Per-draw Blackhawks JEEDS posterior-mean execution skill from result JSONs."""

    rows = []
    for path in sorted(results_dir.glob("jeeds__n*_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success" or payload.get("n_requested") is None:
            continue
        rows.append({
            "n_shots": int(payload["n_requested"]),
            "seed": int(payload["seed"]),
            "old_sigma": float(payload["exec_skill"]),
        })
    return pd.DataFrame(rows)


def plot_old_vs_new(
    data: pd.DataFrame,
    manifests: dict[tuple[int, int], dict],
    old: pd.DataFrame,
    target_player_id: int,
    output: Path,
) -> None:
    reference_key = (max(k[0] for k in manifests), max(k[1] for k in manifests))
    coarse_key = (min(k[0] for k in manifests), min(k[1] for k in manifests))
    target = data[data["player_id"].eq(target_player_id)]
    series = [
        ("Old Blackhawks JEEDS\n250x250", old.rename(columns={"old_sigma": "value"}), "#555555"),
    ]
    for key, color in ((coarse_key, LAMBDA_COLORS[50]), (reference_key, LAMBDA_COLORS[250])):
        frame = _cell(target, *key)[["n_shots", "seed", "jeeds_mean_sigma"]].rename(columns={"jeeds_mean_sigma": "value"})
        series.append((f"HJEEDS JEEDS\n{grid_label(*key)}", frame, color))

    n_values = sorted(old["n_shots"].unique())
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1.0]})
    rng = np.random.default_rng(12345)
    width = 0.8 / len(series)
    for k, (label, frame, color) in enumerate(series):
        positions = []
        values = []
        for i, n_shots in enumerate(n_values):
            subset = frame[frame["n_shots"].eq(n_shots)]["value"].to_numpy(dtype=float)
            position = i + (k - (len(series) - 1) / 2) * width
            positions.append(position)
            values.append(subset)
            left.scatter(
                position + rng.uniform(-width * 0.3, width * 0.3, subset.size), subset,
                s=10, alpha=0.5, color=color,
            )
        box = left.boxplot(values, positions=positions, widths=width * 0.8, showfliers=False, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor("none")
            patch.set_edgecolor(color)
        for element in ("whiskers", "caps", "medians"):
            for line in box[element]:
                line.set_color(color)
        left.scatter([], [], color=color, label=label.replace("\n", " "))
    left.set_xticks(range(len(n_values)), [f"N={n}" for n in n_values])
    left.set_ylabel("posterior mean execution sigma (rad)")
    left.set_title(f"Player {target_player_id}: same 50 draws per N", fontsize=11)
    left.legend(fontsize=9)
    left.grid(alpha=0.3, axis="y")

    reference_target = _cell(target, *reference_key)[["n_shots", "seed", "jeeds_mean_sigma"]]
    paired = reference_target.merge(old, on=["n_shots", "seed"], how="inner")
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(n_values)))
    for color, n_shots in zip(colors, n_values):
        subset = paired[paired["n_shots"].eq(n_shots)]
        r = subset["old_sigma"].corr(subset["jeeds_mean_sigma"])
        right.scatter(subset["old_sigma"], subset["jeeds_mean_sigma"], s=16, color=color, label=f"N={n_shots} (r={r:.2f})")
    low = float(np.nanmin(paired[["old_sigma", "jeeds_mean_sigma"]].to_numpy()))
    high = float(np.nanmax(paired[["old_sigma", "jeeds_mean_sigma"]].to_numpy()))
    right.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--", label="y = x")
    right.set_xlabel("old Blackhawks JEEDS 250x250")
    right.set_ylabel(f"HJEEDS independent JEEDS {grid_label(*reference_key)}")
    right.set_title("Same draw, two implementations", fontsize=11)
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    fig.suptitle(
        "The old-vs-new spread difference follows the implementation, not the grid",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-results", type=Path, default=DEFAULT_OLD_RESULTS)
    parser.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    parser.add_argument("--target-player-id", type=int, default=DEFAULT_TARGET_PLAYER_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    data, manifests = load_cells(args.root)
    table = resolution_table(data, manifests)
    summaries = args.root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    table_path = summaries / "resolution_shift.csv"
    table.to_csv(table_path, index=False)
    print(f"wrote {table_path} ({len(table)} rows from {len(manifests)} cells)")

    output_dir = args.output_dir or args.root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_shift_heatmaps(table, args.n_shots, output_dir / "resolution_shift_heatmaps.png")
    plot_shift_by_n(table, output_dir / "resolution_shift_by_n.png")
    plot_cost_vs_shift(table, args.n_shots, output_dir / "cost_vs_shift.png")
    plot_lambda_snapping(data, manifests, table, args.n_shots, output_dir / "lambda_snapping.png")
    plot_cv_heatmaps(table, args.n_shots, output_dir / "stability_cv_heatmaps.png")
    plot_cv_vs_pool_fraction(data, manifests, output_dir / "cv_vs_pool_fraction.png")
    if args.old_results.exists():
        old = load_old_blackhawks_jeeds(args.old_results)
        if not old.empty:
            plot_old_vs_new(data, manifests, old, args.target_player_id, output_dir / "old_vs_new_jeeds.png")
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
