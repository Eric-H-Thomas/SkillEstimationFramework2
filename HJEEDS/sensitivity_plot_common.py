# This file was AI-generated and still requires human review. Remove this comment when done.
"""Shared helpers for H-JEEDS sensitivity-plot scripts."""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable, Protocol, Sequence

import numpy as np


TEXT_COLOR = "#2F2C37"
CHARCOAL = "#565264"
GRID_COLOR = "#D9D5DF"
NEGATIVE_TEXT_COLOR = "#7A3651"
CATEGORICAL_COLORS = ("#C4B7CB", "#BFEDEF", "#98E2C6", "#DBB957")
NUMERIC_5_COLORS = ("#4DB8ED", "#7984D6", "#A279E4", "#C56FCA", "#E97999")
NUMERIC_3_COLORS = (NUMERIC_5_COLORS[0], NUMERIC_5_COLORS[2], NUMERIC_5_COLORS[4])
NUMERIC_6_COLORS = ("#4DB8ED", "#708EDC", "#9075DC", "#AE6DD6", "#CB72BF", "#E77F99")


class MirroredImprovementRow(Protocol):
    """Minimal row interface needed to infer mirrored x-axis limits."""

    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float


@dataclass(frozen=True)
class SeedObservation:
    """One agent-level observation after converting estimates to absolute errors."""

    key: Hashable
    seed: int
    jeeds_execution_error: float
    hjeeds_execution_error: float
    jeeds_decision_error: float
    hjeeds_decision_error: float


@dataclass(frozen=True)
class ImprovementSummary:
    """Ratio-of-mean-errors percent-improvement summary for one condition."""

    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float
    average_mean: float
    num_seeds: int
    num_agents_per_seed: int


def as_float(row: dict[str, Any], column: str) -> float | None:
    """Parse a numeric CSV field, treating blanks as missing."""

    value = row.get(column, "")
    if value in (None, ""):
        return None
    return float(value)


def rationality_error(row: dict[str, Any], method_prefix: str) -> float | None:
    """Return rationality-percentage absolute error for one method."""

    existing = as_float(row, f"{method_prefix}_abs_rationality_percent_error")
    if existing is not None:
        return existing

    estimate = as_float(row, f"{method_prefix}_rationality_percent")
    truth = as_float(row, "rationality_percent_true")
    if estimate is None or truth is None:
        return None
    return abs(estimate - truth)


def seed_observation_from_agent_row(row: dict[str, Any], key: Hashable) -> SeedObservation | None:
    """Parse one valid CSV row into the errors used for seed-level summaries."""

    sigma_true = as_float(row, "sigma_true")
    jeeds_sigma = as_float(row, "jeeds_posterior_mean_sigma")
    hjeeds_sigma = as_float(row, "hierarchical_posterior_mean_sigma")
    jeeds_rationality_error = rationality_error(row, "jeeds")
    hjeeds_rationality_error = rationality_error(row, "hierarchical")
    if (
        sigma_true is None
        or jeeds_sigma is None
        or hjeeds_sigma is None
        or jeeds_rationality_error is None
        or hjeeds_rationality_error is None
    ):
        return None

    return SeedObservation(
        key=key,
        seed=int(row["seed"]),
        jeeds_execution_error=abs(jeeds_sigma - sigma_true),
        hjeeds_execution_error=abs(hjeeds_sigma - sigma_true),
        jeeds_decision_error=jeeds_rationality_error,
        hjeeds_decision_error=hjeeds_rationality_error,
    )


def mean_ci(values: Sequence[float]) -> tuple[float, float, float]:
    """Return mean and a normal-approximation 95% CI."""

    values_array = np.asarray(values, dtype=float)
    if values_array.size == 0:
        raise ValueError("Cannot compute an interval from zero values.")
    mean = float(np.mean(values_array))
    if values_array.size == 1:
        return mean, mean, mean
    half_width = 1.96 * float(np.std(values_array, ddof=1)) / math.sqrt(values_array.size)
    return mean, mean - half_width, mean + half_width


def ratio_of_means_improvement_ci(
    jeeds_errors: Sequence[float],
    hjeeds_errors: Sequence[float],
) -> tuple[float, float, float]:
    """Return percent reduction in mean error with a paired delta-method 95% CI."""

    jeeds_array = np.asarray(jeeds_errors, dtype=float)
    hjeeds_array = np.asarray(hjeeds_errors, dtype=float)
    if jeeds_array.size == 0 or hjeeds_array.size == 0:
        raise ValueError("Cannot compute improvement from zero seed-level errors.")
    if jeeds_array.size != hjeeds_array.size:
        raise ValueError("JEEDS and H-JEEDS error arrays must have the same length.")

    jeeds_mean = float(np.mean(jeeds_array))
    hjeeds_mean = float(np.mean(hjeeds_array))
    if jeeds_mean <= 0.0:
        raise ValueError("Mean JEEDS error must be positive to compute percent improvement.")

    improvement = 100.0 * (jeeds_mean - hjeeds_mean) / jeeds_mean
    if jeeds_array.size == 1:
        return improvement, improvement, improvement

    covariance = np.cov(np.vstack([jeeds_array, hjeeds_array]), ddof=1) / jeeds_array.size
    gradient = np.asarray([100.0 * hjeeds_mean / (jeeds_mean**2), -100.0 / jeeds_mean])
    variance = float(gradient @ covariance @ gradient.T)
    half_width = 1.96 * math.sqrt(max(0.0, variance))
    return improvement, improvement - half_width, improvement + half_width


def summarize_seed_improvements(observations: Iterable[SeedObservation]) -> dict[Hashable, ImprovementSummary]:
    """Summarize agent-level errors as percent reduction between mean errors."""

    seed_sums: dict[tuple[Hashable, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for observation in observations:
        entry = seed_sums[(observation.key, observation.seed)]
        entry[0] += observation.jeeds_execution_error
        entry[1] += observation.hjeeds_execution_error
        entry[2] += observation.jeeds_decision_error
        entry[3] += observation.hjeeds_decision_error
        entry[4] += 1.0

    seed_errors: dict[Hashable, dict[str, list[float]]] = defaultdict(
        lambda: {
            "jeeds_execution": [],
            "hjeeds_execution": [],
            "jeeds_decision": [],
            "hjeeds_decision": [],
            "n": [],
        }
    )
    for (key, _seed), values in seed_sums.items():
        jeeds_sigma_sum, hjeeds_sigma_sum, jeeds_rat_sum, hjeeds_rat_sum, count = values
        if count <= 0:
            continue

        jeeds_sigma_mean = jeeds_sigma_sum / count
        hjeeds_sigma_mean = hjeeds_sigma_sum / count
        jeeds_rat_mean = jeeds_rat_sum / count
        hjeeds_rat_mean = hjeeds_rat_sum / count
        if jeeds_sigma_mean <= 0.0 or jeeds_rat_mean <= 0.0:
            continue

        seed_errors[key]["jeeds_execution"].append(jeeds_sigma_mean)
        seed_errors[key]["hjeeds_execution"].append(hjeeds_sigma_mean)
        seed_errors[key]["jeeds_decision"].append(jeeds_rat_mean)
        seed_errors[key]["hjeeds_decision"].append(hjeeds_rat_mean)
        seed_errors[key]["n"].append(count)

    summaries: dict[Hashable, ImprovementSummary] = {}
    for key, values in seed_errors.items():
        execution_mean, execution_ci_lower, execution_ci_upper = ratio_of_means_improvement_ci(
            values["jeeds_execution"],
            values["hjeeds_execution"],
        )
        decision_mean, decision_ci_lower, decision_ci_upper = ratio_of_means_improvement_ci(
            values["jeeds_decision"],
            values["hjeeds_decision"],
        )
        summaries[key] = ImprovementSummary(
            execution_mean=execution_mean,
            execution_ci_lower=execution_ci_lower,
            execution_ci_upper=execution_ci_upper,
            decision_mean=decision_mean,
            decision_ci_lower=decision_ci_lower,
            decision_ci_upper=decision_ci_upper,
            average_mean=(execution_mean + decision_mean) / 2.0,
            num_seeds=len(values["jeeds_execution"]),
            num_agents_per_seed=int(round(float(np.mean(values["n"])))),
        )
    return summaries


def summary_fields(summary: ImprovementSummary) -> dict[str, float | int]:
    """Return common dataclass fields for plotted improvement rows."""

    return {
        "execution_mean": summary.execution_mean,
        "execution_ci_lower": summary.execution_ci_lower,
        "execution_ci_upper": summary.execution_ci_upper,
        "decision_mean": summary.decision_mean,
        "decision_ci_lower": summary.decision_ci_lower,
        "decision_ci_upper": summary.decision_ci_upper,
        "average_mean": summary.average_mean,
        "num_seeds": summary.num_seeds,
        "num_agents_per_seed": summary.num_agents_per_seed,
    }


def round_down_to_five(value: float) -> float:
    """Round down to a publication-friendly five-point axis boundary."""

    return float(math.floor(value / 5.0) * 5.0)


def round_up_to_five(value: float) -> float:
    """Round up to a publication-friendly five-point axis boundary."""

    return float(math.ceil(value / 5.0) * 5.0)


def format_axis_tick(value: float, _position: int) -> str:
    """Show mirrored tick labels as positive percentages on both sides."""

    if abs(value) < 1e-9:
        return "0"
    return f"{abs(int(value))}"


def tick_step(x_min: float, x_max: float) -> int:
    """Return a tidy tick step for one mirrored x-axis."""

    return 5 if (x_max - x_min) <= 50 else 10


def grouped_y_positions(group_keys: Sequence[Hashable], group_gap: float) -> tuple[np.ndarray, list[float]]:
    """Return y positions with extra space between consecutive groups."""

    y_values: list[float] = []
    boundaries: list[float] = []
    current_y = 0.0
    previous_group: Hashable | None = None
    previous_y: float | None = None

    for group in group_keys:
        if previous_group is not None and group != previous_group:
            current_y += group_gap
            if previous_y is not None:
                boundaries.append((previous_y + current_y) / 2.0)
        y_values.append(current_y)
        previous_y = current_y
        previous_group = group
        current_y += 1.0

    return np.asarray(y_values, dtype=float), boundaries


def group_centers(
    group_keys: Sequence[Hashable],
    y_positions: Sequence[float],
    *,
    sort: bool = False,
) -> list[tuple[Hashable, float]]:
    """Return y centers for unlabeled groups."""

    positions_by_group: dict[Hashable, list[float]] = defaultdict(list)
    group_order: list[Hashable] = []
    for group, y_value in zip(group_keys, y_positions):
        if group not in positions_by_group:
            group_order.append(group)
        positions_by_group[group].append(float(y_value))

    if sort:
        group_order = sorted(group_order)

    return [
        (group, (min(positions_by_group[group]) + max(positions_by_group[group])) / 2.0)
        for group in group_order
    ]


def labeled_group_centers(
    group_keys: Sequence[Hashable],
    group_labels: Sequence[str],
    y_positions: Sequence[float],
) -> list[tuple[Hashable, str, float]]:
    """Return y centers for labeled groups."""

    labels_by_group = {group: label for group, label in zip(group_keys, group_labels)}
    return [
        (group, labels_by_group[group], center)
        for group, center in group_centers(group_keys, y_positions)
    ]


def ci_half_width(mean: float, ci_lower: float, ci_upper: float) -> float:
    """Return a compact half-width for a plotted 95% CI."""

    return max(abs(mean - ci_lower), abs(ci_upper - mean))


def missing_bar_label_latex(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format a hidden negative bar as mean plus 95% CI half-width."""

    return f"{mean:.1f}% ($\\pm${ci_half_width(mean, ci_lower, ci_upper):.1f}%)"


def format_percent(value: float) -> str:
    """Format a percent value compactly, preserving large negative labels."""

    if abs(value) >= 100.0:
        return f"{value:,.0f}%"
    return f"{value:.1f}%"


def missing_bar_label_plain(mean: float, ci_lower: float, ci_upper: float) -> str:
    """Format a hidden negative bar using plain ASCII punctuation."""

    half_width = ci_half_width(mean, ci_lower, ci_upper)
    return f"{format_percent(mean)} (+/-{format_percent(half_width)})"


def hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a hex color to RGB values on [0, 1]."""

    hex_value = hex_color.strip().lstrip("#")
    return tuple(int(hex_value[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def rgb_to_hex(rgb: Iterable[float]) -> str:
    """Convert RGB values on [0, 1] to a hex string."""

    return "#" + "".join(f"{max(0, min(255, round(channel * 255))):02X}" for channel in rgb)


def blend(hex_color: str, target_hex: str, amount: float) -> str:
    """Blend a color toward a target color."""

    rgb = np.asarray(hex_to_rgb(hex_color))
    target = np.asarray(hex_to_rgb(target_hex))
    blended = (1.0 - amount) * rgb + amount * target
    return rgb_to_hex(blended)


def relative_luminance(hex_color: str) -> float:
    """Return approximate relative luminance for choosing label color."""

    rgb = []
    for channel in hex_to_rgb(hex_color):
        if channel <= 0.03928:
            rgb.append(channel / 12.92)
        else:
            rgb.append(((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def readable_label_color(fill_color: str, *, threshold: float = 0.28) -> str:
    """Return a readable text color for one bar fill."""

    return "#FFFFFF" if relative_luminance(fill_color) < threshold else TEXT_COLOR


def compact_x_limits(
    rows: Sequence[MirroredImprovementRow],
    hide_negative_bars: bool,
    *,
    base_extent: float = 3.0,
    negative_placeholder: float = 8.0,
) -> tuple[float, float]:
    """Return tight asymmetric x-axis limits that preserve plotted intervals."""

    left_extents = [0.0, -base_extent]
    right_extents = [0.0, base_extent]
    for row in rows:
        if hide_negative_bars and row.execution_mean < 0.0:
            left_extents.append(-negative_placeholder)
        else:
            left_extents.append(-row.execution_ci_upper)
            right_extents.append(-row.execution_ci_lower)

        if hide_negative_bars and row.decision_mean < 0.0:
            right_extents.append(negative_placeholder)
        else:
            left_extents.append(row.decision_ci_lower)
            right_extents.append(row.decision_ci_upper)

    x_min = min(-5.0, round_down_to_five(min(left_extents)))
    x_max = max(5.0, round_up_to_five(max(right_extents)))
    if x_min >= 0.0 or x_max <= 0.0 or x_min >= x_max:
        raise ValueError(f"Invalid x-axis limits inferred: {x_min}, {x_max}")
    return x_min, x_max


def mirrored_bar_arrays(
    rows: Sequence[MirroredImprovementRow],
    hide_negative_bars: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mirrored bar values and x-error arrays for execution/decision panels."""

    execution_values = np.asarray(
        [0.0 if hide_negative_bars and row.execution_mean < 0.0 else -row.execution_mean for row in rows],
        dtype=float,
    )
    decision_values = np.asarray(
        [0.0 if hide_negative_bars and row.decision_mean < 0.0 else row.decision_mean for row in rows],
        dtype=float,
    )
    execution_xerr = np.asarray(
        [
            [0.0, 0.0]
            if hide_negative_bars and row.execution_mean < 0.0
            else [row.execution_ci_upper - row.execution_mean, row.execution_mean - row.execution_ci_lower]
            for row in rows
        ],
        dtype=float,
    ).T
    decision_xerr = np.asarray(
        [
            [0.0, 0.0]
            if hide_negative_bars and row.decision_mean < 0.0
            else [row.decision_mean - row.decision_ci_lower, row.decision_ci_upper - row.decision_mean]
            for row in rows
        ],
        dtype=float,
    ).T
    return execution_values, decision_values, execution_xerr, decision_xerr


def configure_matplotlib() -> None:
    """Configure Matplotlib for headless rendering."""

    matplotlib_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "hjeeds_matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    import matplotlib

    matplotlib.use("Agg", force=True)


def save_figure_bundle(figure: Any, output_stem: Path, dpi: int) -> None:
    """Save a figure as PNG, SVG, and PDF when the backend is available."""

    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi)
    figure.savefig(output_stem.with_suffix(".svg"))
    try:
        figure.savefig(output_stem.with_suffix(".pdf"))
    except ImportError as exc:
        print(f"Skipped PDF export because Matplotlib's PDF backend is unavailable: {exc}", flush=True)
