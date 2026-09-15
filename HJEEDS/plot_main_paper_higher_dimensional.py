# Paper correspondence: Main `fig:two_d_baseline_results`/`fig:baseball_drift`; Supplement `app:baseball_sigma_gap`.
"""Render the higher-dimensional main-paper figures for one-column AAAI use.

Both figure families read experimental summary CSVs by default:

* 2D-Darts (10): ``HJEEDS/results/2d_cluster_tests/cluster_0/summary_by_bucket.csv``
* Baseball (11/12):
  ``HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed/``

The plotting functions derive their points, uncertainty intervals, checkpoint
labels, and axis limits directly from those CSV artifacts; no paper values are
hard-coded here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from HJEEDS.baseball_convergence import (
    validate_complete_convergence_run_metadata,
    validate_paper_bbip20_cohort,
)
from HJEEDS.baseball_plot_style import BASEBALL_METHOD_STYLES
from HJEEDS.baseball_hyperpriors import (
    DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH,
    hyperprior_config_to_dict,
    load_literature_informed_baseball_hyperpriors,
)
from HJEEDS.baseball_provenance import (
    build_baseball_run_provenance,
    require_matching_fingerprint,
)
from HJEEDS.two_d_completion import validate_complete_two_d_run_metadata


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"
PAPER_RUN_HYPERPRIOR_SHA256 = (
    "4eb52436e860034b4904b9ceb24ae327834028fc94b0af3110c5eaf477fd46e5"
)
DEFAULT_TWO_D_SUMMARY_CSV = (
    REPO_ROOT / "HJEEDS/results/2d_cluster_tests/cluster_0/summary_by_bucket.csv"
)
DEFAULT_BASEBALL_RESULTS_DIR = (
    REPO_ROOT
    / "HJEEDS/results/baseball_convergence_paper_bbip20_literature_informed"
)

# Keep Matplotlib's font cache in a writable temporary location on desktop and
# cluster runs.  This is presentation infrastructure only; it does not affect
# plotted values.
_MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "hjeeds_main_paper_plot_cache"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

FIGURE_WIDTH = 3.35
TEXT_COLOR = "#2F2C37"
CHARCOAL = "#565264"
MUTED_TEXT_COLOR = "#66616F"
GRID_COLOR = "#D9D5DF"
SPINE_COLOR = "#AAA4B3"
PAPER_BASEBALL_CHECKPOINTS = (5, 10, 25, 50, 100)

TWO_D_EXECUTION_METRIC = "abs_sigma_error"
TWO_D_DECISION_METRIC = "abs_rationality_percent_error"

DUAL_AXIS_STYLES = {
    ("jeeds", "execution"): {
        "label": "JEEDS \u00b7 execution",
        "color": "#339CFF",
        "marker": "o",
        "linestyle": "-",
    },
    ("hierarchical", "execution"): {
        "label": "H-JEEDS \u00b7 execution",
        "color": "#F3883B",
        "marker": "s",
        "linestyle": "-",
    },
    ("jeeds", "decision"): {
        "label": "JEEDS \u00b7 decision",
        "color": "#5DC977",
        "marker": "^",
        "linestyle": (0, (4.0, 2.1)),
    },
    ("hierarchical", "decision"): {
        "label": "H-JEEDS \u00b7 decision",
        "color": "#EB77B1",
        "marker": "D",
        "linestyle": (0, (4.0, 2.1)),
    },
}


@dataclass(frozen=True)
class IntervalSeries:
    """Means and 95% interval endpoints for one method."""

    means: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def asymmetric_errors(self) -> np.ndarray:
        means = np.asarray(self.means, dtype=float)
        return np.vstack(
            (
                means - np.asarray(self.lower, dtype=float),
                np.asarray(self.upper, dtype=float) - means,
            )
        )


PLOT_RC = {
    "font.family": "Helvetica",
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.2,
    "axes.titleweight": "bold",
    "axes.labelcolor": TEXT_COLOR,
    "axes.edgecolor": SPINE_COLOR,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 7.0,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

DUAL_AXIS_PLOT_RC = {
    **PLOT_RC,
    # AAAI requires labels and other text inside figures to be at least 9 pt.
    "font.size": 9.5,
    "axes.titlesize": 9.9,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "savefig.facecolor": "white",
}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_two_d_series(
    summary_csv: Path,
) -> tuple[tuple[int, ...], dict[str, IntervalSeries], dict[str, IntervalSeries]]:
    """Load 2D execution/decision error series from ``summary_by_bucket.csv``."""

    # Publication figures may only read an atomically completed canonical run.
    # This verifies the 1000--1499 seed set, all 25 agents per seed, fixed paper
    # configuration, corrected-source fingerprint, and every output hash before
    # a single plotted value is loaded.
    validate_complete_two_d_run_metadata(
        summary_csv.parent,
        require_paper_configuration=True,
    )

    if not summary_csv.is_file():
        raise FileNotFoundError(f"Missing 2D summary CSV: {summary_csv}")

    rows = _read_csv_rows(summary_csv)
    paper_rows = [
        row
        for row in rows
        if row.get("metric", "").strip()
        in {TWO_D_EXECUTION_METRIC, TWO_D_DECISION_METRIC}
    ]
    buckets = tuple(
        sorted({int(float(row["count_bucket"])) for row in paper_rows})
    )
    if not buckets:
        raise ValueError(f"No 2D paper metrics found in {summary_csv}")

    execution: dict[str, IntervalSeries] = {}
    decision: dict[str, IntervalSeries] = {}
    for method in ("jeeds", "hierarchical"):
        for metric, destination in (
            (TWO_D_EXECUTION_METRIC, execution),
            (TWO_D_DECISION_METRIC, decision),
        ):
            by_n = {
                int(float(row["count_bucket"])): row
                for row in paper_rows
                if row.get("method", "").strip() == method
                and row.get("metric", "").strip() == metric
            }
            missing = [n for n in buckets if n not in by_n]
            if missing:
                raise ValueError(
                    f"Missing {method}/{metric} rows for buckets {missing} in {summary_csv}"
                )
            means: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for n in buckets:
                mean = _optional_float(by_n[n].get("mean"))
                ci_lower = _optional_float(by_n[n].get("ci_lower"))
                ci_upper = _optional_float(by_n[n].get("ci_upper"))
                if mean is None or ci_lower is None or ci_upper is None:
                    raise ValueError(
                        f"Incomplete CI fields for {method}/{metric}/N={n} in {summary_csv}"
                    )
                if not (ci_lower <= mean <= ci_upper):
                    raise ValueError(
                        "Expected ci_lower <= mean <= ci_upper for "
                        f"{method}/{metric}/N={n}, got {ci_lower}, {mean}, {ci_upper}"
                    )
                means.append(mean)
                lower.append(ci_lower)
                upper.append(ci_upper)
            destination[method] = IntervalSeries(
                means=tuple(means),
                lower=tuple(lower),
                upper=tuple(upper),
            )
    return buckets, execution, decision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_baseball_result_provenance(results_dir: Path) -> None:
    """Reject plots not produced by the current paper kernel/data/prior run."""

    path = results_dir / "convergence_run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing MLB convergence provenance: {path}. Re-run the corrected paper experiment."
        )
    payload = validate_complete_convergence_run_metadata(results_dir)
    validate_paper_bbip20_cohort(results_dir, payload)
    actual = payload.get("run_provenance")
    if not isinstance(actual, dict):
        raise ValueError(f"Malformed run_provenance in {path}")
    expected = build_baseball_run_provenance(
        season_year=actual.get("season_year"),
        pitch_types=actual.get("pitch_types") or (),
        sigma_grid=actual.get("sigma_grid") or (),
        log_lambda_grid=actual.get("log_lambda_grid") or (),
        configuration=actual.get("configuration") or {},
    )
    require_matching_fingerprint(actual, expected, label="main-paper MLB result")
    configuration = actual.get("configuration") or {}
    required = {
        "season_year": 2021,
        "pitch_types": ["FF"],
        "configuration.workflow": "baseball-convergence-agent-cache",
        "configuration.delta": 0.0417,
        "configuration.convergence_ns": list(PAPER_BASEBALL_CHECKPOINTS),
        "configuration.max_reference_pitches": 100,
        "configuration.min_pitches_per_agent": 100,
        "configuration.num_agents": 20,
    }
    values = {
        "season_year": actual.get("season_year"),
        "pitch_types": actual.get("pitch_types"),
        "configuration.workflow": configuration.get("workflow"),
        "configuration.delta": configuration.get("delta"),
        "configuration.convergence_ns": configuration.get("convergence_ns"),
        "configuration.max_reference_pitches": configuration.get("max_reference_pitches"),
        "configuration.min_pitches_per_agent": configuration.get("min_pitches_per_agent"),
        "configuration.num_agents": configuration.get("num_agents"),
    }
    mismatches = {
        key: (values[key], expected_value)
        for key, expected_value in required.items()
        if values[key] != expected_value
    }
    if mismatches:
        raise ValueError(f"MLB result metadata is not the paper configuration: {mismatches}")
    canonical_sigma = np.concatenate(
        (
            np.linspace(0.17, 1.0, num=60, dtype=float),
            np.linspace(1.0 + 0.0417, 2.81, num=6, dtype=float),
        )
    )
    canonical_log_lambda = np.log(
        np.logspace(np.log10(1e-3), np.log10(10**3.6), num=21, dtype=float)
    )
    # Stable JSON normalization can round a coordinate by a few units in the
    # last place.  Keep this provenance check strict while admitting only that
    # serialization-scale noise.
    actual_sigma = np.asarray(actual.get("sigma_grid", ()), dtype=float)
    if actual_sigma.shape != canonical_sigma.shape or not np.allclose(
        actual_sigma, canonical_sigma, rtol=0.0, atol=5e-14
    ):
        raise ValueError("MLB result did not use the canonical 66-point execution grid.")
    actual_log_lambda = np.asarray(actual.get("log_lambda_grid", ()), dtype=float)
    if actual_log_lambda.shape != canonical_log_lambda.shape or not np.allclose(
        actual_log_lambda, canonical_log_lambda, rtol=0.0, atol=5e-14
    ):
        raise ValueError("MLB result did not use the canonical 21-point log-lambda grid.")

    # Validate the fixed, outcome-independent paper prior against the corrected
    # MLB artifact/kernel provenance. Result metadata must pin this exact file.
    paper_hyperpriors = load_literature_informed_baseball_hyperpriors()
    hyperprior = payload.get("hyperprior") or {}
    expected_prior_hash = _sha256(DEFAULT_BASEBALL_LITERATURE_HYPERPRIORS_PATH)
    if hyperprior.get("preset") != "baseball-literature-informed":
        raise ValueError(f"MLB paper result used hyperprior preset {hyperprior.get('preset')!r}.")
    if hyperprior.get("file_sha256") not in {
        expected_prior_hash,
        PAPER_RUN_HYPERPRIOR_SHA256,
    }:
        raise ValueError(
            "MLB result hyperprior hash matches neither the currently committed "
            "nor the original paper-run literature-informed prior."
        )
    if hyperprior.get("config") != hyperprior_config_to_dict(paper_hyperpriors):
        raise ValueError("MLB result hyperprior values do not match the committed paper prior.")


def load_baseball_separability(
    results_dir: Path,
) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    """Load AUC and mean-sigma-gap series from ``separability_by_N.csv``."""

    _validate_baseball_result_provenance(results_dir)
    path = results_dir / "separability_by_N.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseball separability CSV: {path}")

    expected_methods = {"jeeds", "hierarchical"}
    sigma_rows = [
        row
        for row in _read_csv_rows(path)
        if row.get("metric", "").strip() == "sigma"
        and row.get("method", "").strip() in expected_methods
    ]
    seen: set[tuple[str, int]] = set()
    for row in sigma_rows:
        key = (row.get("method", "").strip(), int(float(row["convergence_n"])))
        if key in seen:
            raise ValueError(f"Duplicate baseball separability row for {key} in {path}")
        seen.add(key)
        if int(float(row["num_bottom"])) != 10 or int(float(row["num_top"])) != 10:
            raise ValueError(
                f"Incomplete paper walk/IP-proxy cohort for {key} in {path}: "
                f"num_bottom={row.get('num_bottom')}, num_top={row.get('num_top')}"
            )
        auc = float(row["auc"])
        gap = float(row["mean_gap_top_minus_bottom"])
        mean_bottom = float(row["mean_bottom"])
        mean_top = float(row["mean_top"])
        if (
            not np.isfinite(auc)
            or not 0.0 <= auc <= 1.0
            or not np.isfinite(gap)
            or not np.isfinite(mean_bottom)
            or not np.isfinite(mean_top)
            or mean_bottom <= 0.0
            or mean_top <= 0.0
        ):
            raise ValueError(f"Invalid AUC/gap for {key} in {path}: auc={auc}, gap={gap}")
    pitch_counts = tuple(
        sorted({int(float(row["convergence_n"])) for row in sigma_rows})
    )
    if not pitch_counts:
        raise ValueError(f"No sigma rows found in {path}")
    if pitch_counts != PAPER_BASEBALL_CHECKPOINTS:
        raise ValueError(
            f"Paper baseball separability checkpoints are {pitch_counts}; "
            f"expected {PAPER_BASEBALL_CHECKPOINTS}."
        )

    separability: dict[str, tuple[float, ...]] = {}
    sigma_gap: dict[str, tuple[float, ...]] = {}
    for method in ("jeeds", "hierarchical"):
        by_n = {
            int(float(row["convergence_n"])): row
            for row in sigma_rows
            if row.get("method", "").strip() == method
        }
        missing = [n for n in pitch_counts if n not in by_n]
        if missing:
            raise ValueError(f"Missing {method} sigma rows for N={missing} in {path}")
        separability[method] = tuple(float(by_n[n]["auc"]) for n in pitch_counts)
        sigma_gap[method] = tuple(
            float(by_n[n]["mean_gap_top_minus_bottom"]) for n in pitch_counts
        )
    return pitch_counts, separability, sigma_gap


def load_baseball_drift(
    results_dir: Path,
) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]], dict[str, tuple[float, ...]]]:
    """Load execution/decision self-reference drift from ``summary_by_N.csv``."""

    _validate_baseball_result_provenance(results_dir)
    path = results_dir / "summary_by_N.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing baseball drift summary CSV: {path}")

    rows = _read_csv_rows(path)
    expected_methods = {"jeeds", "hierarchical"}
    expected_metrics = {"abs_sigma_drift_vs_full", "abs_log_lambda_drift_vs_full"}
    relevant_rows = [
        row
        for row in rows
        if row.get("method", "").strip() in expected_methods
        and row.get("metric", "").strip() in expected_metrics
    ]
    seen: set[tuple[str, str, int]] = set()
    for row in relevant_rows:
        key = (
            row.get("method", "").strip(),
            row.get("metric", "").strip(),
            int(float(row["count_bucket"])),
        )
        if key in seen:
            raise ValueError(f"Duplicate baseball drift row for {key} in {path}")
        seen.add(key)
        if int(float(row["num_agents"])) != 20:
            raise ValueError(
                f"Paper baseball drift row {key} has num_agents={row.get('num_agents')}; expected 20."
            )
        mean = float(row["mean"])
        if not np.isfinite(mean) or mean < 0.0:
            raise ValueError(f"Invalid drift mean for {key} in {path}: {mean}")
    pitch_counts = tuple(
        sorted(
            {
                int(float(row["count_bucket"]))
                for row in relevant_rows
            }
        )
    )
    if not pitch_counts:
        raise ValueError(f"No drift rows found in {path}")
    if pitch_counts != PAPER_BASEBALL_CHECKPOINTS:
        raise ValueError(
            f"Paper baseball drift checkpoints are {pitch_counts}; "
            f"expected {PAPER_BASEBALL_CHECKPOINTS}."
        )

    execution_drift: dict[str, tuple[float, ...]] = {}
    decision_drift: dict[str, tuple[float, ...]] = {}
    for method in ("jeeds", "hierarchical"):
        sigma_by_n = {
            int(float(row["count_bucket"])): float(row["mean"])
            for row in relevant_rows
            if row.get("method", "").strip() == method
            and row.get("metric", "").strip() == "abs_sigma_drift_vs_full"
        }
        lambda_by_n = {
            int(float(row["count_bucket"])): float(row["mean"])
            for row in relevant_rows
            if row.get("method", "").strip() == method
            and row.get("metric", "").strip() == "abs_log_lambda_drift_vs_full"
        }
        missing_sigma = [n for n in pitch_counts if n not in sigma_by_n]
        missing_lambda = [n for n in pitch_counts if n not in lambda_by_n]
        if missing_sigma or missing_lambda:
            raise ValueError(
                f"Missing {method} drift rows for "
                f"sigma N={missing_sigma}, log-lambda N={missing_lambda} in {path}"
            )
        execution_drift[method] = tuple(sigma_by_n[n] for n in pitch_counts)
        decision_drift[method] = tuple(lambda_by_n[n] for n in pitch_counts)
    return pitch_counts, execution_drift, decision_drift


def _padded_limits(
    values: Sequence[float],
    *,
    lower_pad_fraction: float,
    upper_pad_fraction: float,
    include_zero: bool = False,
) -> tuple[float, float]:
    """Return axis limits that enclose every supplied value with proportional padding.

    Deriving limits from the data keeps a point or interval endpoint from being
    silently clipped when the figure is regenerated from a different seed group
    or result tree.
    """

    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("Cannot derive axis limits from an empty value set.")
    low = min(numeric)
    high = max(numeric)
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    span = high - low
    if span <= 0.0:
        span = abs(high) or 1.0
    return low - lower_pad_fraction * span, high + upper_pad_fraction * span


def _style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.tick_params(axis="both", pad=2.0)
    axis.title.set_color(TEXT_COLOR)


def _style_dual_axes(left_axis, right_axis) -> None:
    """Apply a shared, neutral frame to a paired execution/decision axis."""

    left_axis.set_axisbelow(True)
    left_axis.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.78, zorder=1)
    right_axis.grid(False)

    for axis in (left_axis, right_axis):
        axis.tick_params(
            axis="both",
            colors=TEXT_COLOR,
            labelsize=9.0,
            length=2.8,
            width=0.65,
            pad=2.2,
        )
        axis.spines["top"].set_visible(False)

    left_axis.spines["right"].set_visible(False)
    right_axis.spines["left"].set_visible(False)
    right_axis.spines["bottom"].set_visible(False)
    for axis, spine_name in (
        (left_axis, "left"),
        (left_axis, "bottom"),
        (right_axis, "right"),
    ):
        axis.spines[spine_name].set_color(CHARCOAL)
        axis.spines[spine_name].set_linewidth(0.65)


def _figure_legend(figure, axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=2,
        frameon=False,
        borderaxespad=0.0,
        columnspacing=1.25,
        handlelength=1.55,
        handletextpad=0.45,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)


def _dual_axis_legend(figure, execution_axis, decision_axis) -> None:
    """Place metric-grouped legend entries in two centered rows."""

    execution_handles, execution_labels = execution_axis.get_legend_handles_labels()
    decision_handles, decision_labels = decision_axis.get_legend_handles_labels()
    for handles, labels, top in (
        (execution_handles, execution_labels, 0.995),
        (decision_handles, decision_labels, 0.925),
    ):
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, top),
            ncol=2,
            frameon=False,
            borderaxespad=0.0,
            columnspacing=1.15,
            handlelength=1.65,
            handletextpad=0.42,
        )
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)


def _save_bundle(
    figure,
    output_dir: Path,
    stem: str,
    dpi: int,
    *,
    png_description: str | None = None,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_dir / f"{stem}{suffix}"
        # Do not use bbox_inches="tight": the canvas is intentionally the exact
        # 3.35-inch AAAI column width, and all elements fit inside its margins.
        save_kwargs: dict = {}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
            if png_description is not None:
                save_kwargs["metadata"] = {"Description": png_description}
        figure.savefig(path, **save_kwargs)
        written.append(path)
    return tuple(written)


def _draw_error_series(axis, x_values, series: IntervalSeries, style: dict) -> None:
    axis.plot(
        x_values,
        series.means,
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        markersize=4.4,
        markeredgecolor="white",
        markeredgewidth=0.65,
        linewidth=1.35,
        label=style["label"],
        zorder=3,
    )
    axis.errorbar(
        x_values,
        series.means,
        yerr=series.asymmetric_errors(),
        fmt="none",
        ecolor=style["color"],
        elinewidth=0.95,
        capsize=2.9,
        capthick=0.95,
        zorder=5,
    )


def _interval_axis_limits(
    series_by_method: dict[str, IntervalSeries],
    *,
    nonnegative: bool = True,
) -> tuple[float, float]:
    """Return padded limits that contain every plotted confidence interval."""

    lower_values = [
        float(value)
        for series in series_by_method.values()
        for value in series.lower
    ]
    upper_values = [
        float(value)
        for series in series_by_method.values()
        for value in series.upper
    ]
    if not lower_values or not upper_values:
        raise ValueError("At least one interval series is required to set axis limits.")
    values = np.asarray(lower_values + upper_values, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Axis-limit inputs must all be finite.")

    data_min = min(lower_values)
    data_max = max(upper_values)
    data_span = data_max - data_min
    scale = max(abs(data_min), abs(data_max), 1.0)
    padding = max(0.08 * data_span, 0.03 * scale)
    lower = data_min - padding
    if nonnegative:
        lower = max(0.0, lower)
    upper = data_max + padding
    if not lower < upper:
        upper = lower + max(1.0, 0.1 * scale)
    return float(lower), float(upper)


def plot_two_d(
    output_dir: Path,
    dpi: int,
    *,
    observation_buckets: Sequence[int],
    execution: dict[str, IntervalSeries],
    decision: dict[str, IntervalSeries],
) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, execution_axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.60))
        decision_axis = execution_axis.twinx()
        figure.patch.set_facecolor("white")
        x_values = np.arange(len(observation_buckets), dtype=float)

        for method in ("jeeds", "hierarchical"):
            _draw_error_series(
                execution_axis,
                x_values,
                execution[method],
                DUAL_AXIS_STYLES[(method, "execution")],
            )
            _draw_error_series(
                decision_axis,
                x_values,
                decision[method],
                DUAL_AXIS_STYLES[(method, "decision")],
            )

        execution_axis.set_ylim(*_interval_axis_limits(execution))
        execution_axis.set_ylabel(
            "Execution error\n" + r"($|\hat{\sigma}-\sigma|$)",
            color=TEXT_COLOR,
            labelpad=4.5,
        )
        decision_axis.set_ylim(*_interval_axis_limits(decision))
        decision_axis.set_ylabel(
            "Decision error\n(percentage points)",
            color=TEXT_COLOR,
            labelpad=5.0,
        )
        execution_axis.set_xticks(
            x_values, [str(value) for value in observation_buckets]
        )
        execution_axis.set_xlabel("Observations per agent", color=TEXT_COLOR, labelpad=4.0)
        _style_dual_axes(execution_axis, decision_axis)
        _dual_axis_legend(figure, execution_axis, decision_axis)
        figure.subplots_adjust(left=0.205, right=0.795, top=0.785, bottom=0.19)
        written = _save_bundle(
            figure,
            output_dir,
            "10_two_d_error_by_count_bucket",
            dpi,
            png_description=(
                "Rendered from 2d_cluster_tests/cluster_0/summary_by_bucket.csv"
            ),
        )
        plt.close(figure)
    return written


def _draw_method_lines(axis, x_values, values_by_method) -> None:
    for method in ("jeeds", "hierarchical"):
        style = BASEBALL_METHOD_STYLES[method]
        values = np.asarray(values_by_method[method], dtype=float)
        if len(values) != len(x_values):
            raise ValueError(
                f"{method} has {len(values)} values for {len(x_values)} x coordinates."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError(f"{method} plotting values contain non-finite entries.")
        axis.plot(
            x_values,
            values,
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            markeredgewidth=0.55,
            linewidth=1.45,
            label=style["label"],
            zorder=3,
        )


def _set_pitch_axis(axis, pitch_counts: Sequence[int]) -> None:
    max_n = max(pitch_counts)
    axis.set_xlim(0.0, max_n * 1.05)
    axis.set_xticks(list(pitch_counts), [str(value) for value in pitch_counts])
    # The first two proportional checkpoints (5 and 10 pitches in the paper
    # run) are close together.  Direct their labels away from one another so
    # they remain distinct at one-column print size.
    if len(pitch_counts) >= 2 and pitch_counts[1] - pitch_counts[0] <= 0.07 * max_n:
        tick_labels = axis.get_xticklabels()
        tick_labels[0].set_horizontalalignment("right")
        tick_labels[1].set_horizontalalignment("left")


def plot_baseball_separability_auc(
    output_dir: Path,
    dpi: int,
    *,
    pitch_counts: Sequence[int],
    separability: dict[str, Sequence[float]],
) -> tuple[Path, ...]:
    """Main-paper walk/IP-proxy AUC panel (one-column)."""

    import matplotlib.pyplot as plt

    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.30))

        _draw_method_lines(axis, pitch_counts, separability)
        axis.axhline(0.5, color="#88838F", linestyle="--", linewidth=0.75, zorder=1)
        axis.axhline(0.8, color="#66616F", linestyle=":", linewidth=0.8, zorder=1)
        axis.text(
            0.98,
            0.5,
            "random classifier",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=MUTED_TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
        )
        axis.text(
            0.98,
            0.8,
            "0.8",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9.5,
            color=MUTED_TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
        )
        axis.set_ylabel("AUC", labelpad=4.0)
        axis.set_ylim(0.0, 1.05)
        _set_pitch_axis(axis, pitch_counts)
        _style_axis(axis)
        axis.set_xlabel(r"Pitches observed, $c$", labelpad=4.0)
        _figure_legend(figure, axis)
        figure.subplots_adjust(left=0.18, right=0.975, top=0.89, bottom=0.18)
        written = _save_bundle(
            figure,
            output_dir,
            "11_baseball_separability_by_c",
            dpi,
            png_description=(
                "Rendered from baseball_convergence_paper_bbip20_literature_informed/"
                "separability_by_N.csv (AUC panel)"
            ),
        )
        plt.close(figure)
    return written


def plot_baseball_sigma_gap(
    output_dir: Path,
    dpi: int,
    *,
    pitch_counts: Sequence[int],
    sigma_gap: dict[str, Sequence[float]],
) -> tuple[Path, ...]:
    """Supplementary mean top-minus-bottom $\\hat{\\sigma}$ gap panel."""

    import matplotlib.pyplot as plt

    gap_values = [float(value) for series in sigma_gap.values() for value in series]
    gap_min = min(gap_values)
    gap_max = max(gap_values)
    gap_pad = max(0.012, 0.08 * (gap_max - gap_min))

    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.65))
        _draw_method_lines(axis, pitch_counts, sigma_gap)
        axis.axhline(0.0, color="#88838F", linestyle="--", linewidth=0.75, zorder=1)
        axis.set_ylabel(r"Top $-$ bottom mean $\hat{\sigma}$", labelpad=4.0)
        axis.set_ylim(gap_min - gap_pad, gap_max + gap_pad)
        axis.set_xlabel(r"Pitches observed, $c$", labelpad=4.0)
        _set_pitch_axis(axis, pitch_counts)
        _style_axis(axis)
        _figure_legend(figure, axis)
        figure.subplots_adjust(left=0.20, right=0.975, top=0.88, bottom=0.18)
        written = _save_bundle(
            figure,
            output_dir,
            "13_baseball_sigma_gap_by_c",
            dpi,
            png_description=(
                "Rendered from baseball_convergence_paper_bbip20_literature_informed/"
                "separability_by_N.csv (mean sigma-gap panel)"
            ),
        )
        plt.close(figure)
    return written


def plot_baseball_drift(
    output_dir: Path,
    dpi: int,
    *,
    pitch_counts: Sequence[int],
    execution_drift: dict[str, Sequence[float]],
    decision_drift: dict[str, Sequence[float]],
) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    execution_values = np.asarray(
        [value for method in ("jeeds", "hierarchical") for value in execution_drift[method]],
        dtype=float,
    )
    decision_values = np.asarray(
        [value for method in ("jeeds", "hierarchical") for value in decision_drift[method]],
        dtype=float,
    )
    if np.any(~np.isfinite(execution_values)) or np.any(execution_values < 0.0):
        raise ValueError("Execution drift values must be finite and nonnegative.")
    if np.any(~np.isfinite(decision_values)) or np.any(decision_values < 0.0):
        raise ValueError("Decision drift values must be finite and nonnegative.")
    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, execution_axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.60))
        decision_axis = execution_axis.twinx()
        x_values = np.arange(len(pitch_counts), dtype=float)

        for method in ("jeeds", "hierarchical"):
            execution_style = DUAL_AXIS_STYLES[(method, "execution")]
            decision_style = DUAL_AXIS_STYLES[(method, "decision")]
            execution_axis.plot(
                x_values,
                execution_drift[method],
                color=execution_style["color"],
                marker=execution_style["marker"],
                linestyle=execution_style["linestyle"],
                markersize=4.4,
                markeredgecolor="white",
                markeredgewidth=0.65,
                linewidth=1.35,
                label=execution_style["label"],
                zorder=3,
            )
            decision_axis.plot(
                x_values,
                decision_drift[method],
                color=decision_style["color"],
                marker=decision_style["marker"],
                linestyle=decision_style["linestyle"],
                markersize=4.4,
                markeredgecolor="white",
                markeredgewidth=0.65,
                linewidth=1.35,
                label=decision_style["label"],
                zorder=3,
            )

        # Drift is measured against each method's own estimate at the largest
        # checkpoint, so the axis labels name that checkpoint from the data
        # rather than assuming the paper's value.
        reference_n = max(pitch_counts)
        execution_axis.set_ylim(
            *_padded_limits(
                [value for series in execution_drift.values() for value in series],
                lower_pad_fraction=0.04,
                upper_pad_fraction=0.05,
                include_zero=True,
            )
        )
        execution_axis.set_ylabel(
            "Execution drift\n"
            + rf"$|\hat{{\sigma}}_c-\hat{{\sigma}}_{{{reference_n}}}|$",
            color=TEXT_COLOR,
            labelpad=4.5,
        )
        decision_axis.set_ylim(
            *_padded_limits(
                [value for series in decision_drift.values() for value in series],
                lower_pad_fraction=0.04,
                upper_pad_fraction=0.05,
                include_zero=True,
            )
        )
        decision_axis.set_ylabel(
            "Decision drift\n"
            + rf"$|\widehat{{\log\lambda}}_c-\widehat{{\log\lambda}}_{{{reference_n}}}|$",
            color=TEXT_COLOR,
            labelpad=5.0,
        )
        execution_axis.axhline(
            0.0,
            color="#88838F",
            linestyle=(0, (4.0, 2.1)),
            linewidth=0.75,
            zorder=1,
        )
        execution_axis.set_xlim(-0.2, len(pitch_counts) - 0.8)
        execution_axis.set_xticks(x_values, [str(value) for value in pitch_counts])
        execution_axis.set_xlabel(r"Pitches observed, $c$", labelpad=4.0)
        _style_dual_axes(execution_axis, decision_axis)
        _dual_axis_legend(figure, execution_axis, decision_axis)
        figure.subplots_adjust(left=0.25, right=0.765, top=0.82, bottom=0.17)
        written = _save_bundle(
            figure,
            output_dir,
            "12_baseball_drift_by_c",
            dpi,
            png_description=(
                "Rendered from baseball_convergence_paper_bbip20_literature_informed/"
                "summary_by_N.csv"
            ),
        )
        plt.close(figure)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render higher-dimensional paper figures for one-column AAAI layout "
            "from experimental summary CSVs."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--two-d-summary-csv",
        type=Path,
        default=DEFAULT_TWO_D_SUMMARY_CSV,
        help="2D summary_by_bucket.csv used for figure 10.",
    )
    parser.add_argument(
        "--baseball-results-dir",
        type=Path,
        default=DEFAULT_BASEBALL_RESULTS_DIR,
        help="Directory containing separability_by_N.csv and summary_by_N.csv.",
    )
    parser.add_argument(
        "--figures",
        choices=("all", "baseball", "2d"),
        default="all",
        help="Which figure set to render (default: all).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, received {args.dpi}")

    written: list[Path] = []
    if args.figures in {"all", "2d"}:
        buckets, execution, decision = load_two_d_series(args.two_d_summary_csv)
        written.extend(
            plot_two_d(
                args.output_dir,
                args.dpi,
                observation_buckets=buckets,
                execution=execution,
                decision=decision,
            )
        )
        print(
            "[main-paper-higher-dimensional] 2D source CSV: "
            f"{args.two_d_summary_csv.resolve()}"
        )

    if args.figures in {"all", "baseball"}:
        pitch_counts, separability, sigma_gap = load_baseball_separability(
            args.baseball_results_dir
        )
        drift_counts, execution_drift, decision_drift = load_baseball_drift(
            args.baseball_results_dir
        )
        if pitch_counts != drift_counts:
            raise ValueError(
                "Baseball pitch-count checkpoints differ between separability "
                f"{pitch_counts} and drift {drift_counts}"
            )
        written.extend(
            plot_baseball_separability_auc(
                args.output_dir,
                args.dpi,
                pitch_counts=pitch_counts,
                separability=separability,
            )
        )
        written.extend(
            plot_baseball_sigma_gap(
                args.output_dir,
                args.dpi,
                pitch_counts=pitch_counts,
                sigma_gap=sigma_gap,
            )
        )
        written.extend(
            plot_baseball_drift(
                args.output_dir,
                args.dpi,
                pitch_counts=drift_counts,
                execution_drift=execution_drift,
                decision_drift=decision_drift,
            )
        )
        print(
            "[main-paper-higher-dimensional] Baseball source CSVs: "
            f"{args.baseball_results_dir.resolve()}"
        )

    print("[main-paper-higher-dimensional] Wrote:")
    for path in written:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
