"""Render the higher-dimensional main-paper figures for one-column AAAI use.

The original result CSVs for these three figures are not available in this
checkout.  The values below were therefore recovered from the already-rendered
paper PNGs by calibrating each axis against its visible ticks and reading the
colored marker/error-bar coordinates.  They are *digitized presentation data*,
not recomputed experimental results.  Keeping that distinction explicit avoids
accidentally implying that this plotting-only script reruns either estimator.

Source PNGs used for recovery (relative to the SportsHCI project root):

* ``HJEEDSPaper/figures/2d-darts/10_two_d_error_by_count_bucket.png``
* ``HJEEDSPaper/figures/baseball/11_baseball_separability_by_N.png``
* ``HJEEDSPaper/figures/baseball/12_baseball_drift_by_N.png``

The recovered coordinates reproduce the headline values reported in the paper
at their stated precision (for example, 6.70 vs. 5.93 execution error and 21.8
vs. 15.4 decision-skill error at five 2D-Darts observations; .87/.86 AUC at
100 baseball pitches).  If the source CSVs are recovered later, replace these
constants with CSV readers while retaining the one-column rendering functions.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from HJEEDS.baseball_plot_style import BASEBALL_METHOD_STYLES


REPO_ROOT = Path(__file__).resolve().parent.parent
SPORTSHCI_ROOT = REPO_ROOT.parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "HJEEDS/results/hjeeds_paper_500_seeds/final_paper_plots"

# Keep Matplotlib's font cache in a writable temporary location on desktop and
# cluster runs.  This is presentation infrastructure only; it does not affect
# any recovered values.
_MATPLOTLIB_CACHE = Path(os.environ.get("TMPDIR", "/tmp")) / "hjeeds_main_paper_plot_cache"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

SOURCE_PNGS = {
    "10_two_d_error_by_count_bucket": (
        SPORTSHCI_ROOT / "HJEEDSPaper/figures/2d-darts/10_two_d_error_by_count_bucket.png"
    ),
    "11_baseball_separability_by_N": (
        SPORTSHCI_ROOT / "HJEEDSPaper/figures/baseball/11_baseball_separability_by_N.png"
    ),
    "12_baseball_drift_by_N": (
        SPORTSHCI_ROOT / "HJEEDSPaper/figures/baseball/12_baseball_drift_by_N.png"
    ),
}

FIGURE_WIDTH = 3.35
TEXT_COLOR = "#2F2C37"
CHARCOAL = "#565264"
MUTED_TEXT_COLOR = "#66616F"
GRID_COLOR = "#D9D5DF"
SPINE_COLOR = "#AAA4B3"

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

OBSERVATION_BUCKETS = (5, 10, 25, 100, 1000)
PITCH_COUNTS = (5, 10, 25, 50, 100)


@dataclass(frozen=True)
class IntervalSeries:
    """Digitized means and visible 95% interval endpoints for one method."""

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


# Pixel calibration for the 2D source used y=1192.5 at execution error 5.0
# with 529.5 px/unit, and y=1173.5 at decision error 12 with 94.5 px/unit.
# Interval caps were read at their visible centers; means are cap midpoints.
TWO_D_EXECUTION = {
    "jeeds": IntervalSeries(
        means=(6.704, 6.364, 6.050, 5.422, 5.180),
        lower=(6.471, 6.105, 5.778, 5.144, 4.892),
        upper=(6.938, 6.624, 6.322, 5.701, 5.468),
    ),
    "hierarchical": IntervalSeries(
        means=(5.935, 5.735, 5.574, 5.163, 5.139),
        lower=(5.672, 5.463, 5.302, 4.890, 4.855),
        upper=(6.197, 6.007, 5.846, 5.436, 5.423),
    ),
}

TWO_D_DECISION = {
    "jeeds": IntervalSeries(
        means=(21.831, 20.005, 16.582, 12.868, 11.772),
        lower=(21.005, 19.238, 15.757, 12.085, 11.101),
        upper=(22.656, 20.772, 17.407, 13.651, 12.444),
    ),
    "hierarchical": IntervalSeries(
        means=(15.407, 14.767, 13.270, 12.011, 11.646),
        lower=(14.730, 14.053, 12.540, 11.249, 10.984),
        upper=(16.085, 15.481, 14.000, 12.772, 12.307),
    ),
}

# Baseball values were recovered from marker centers after calibrating the AUC,
# mean-gap, and drift axes.  AUC has .01 resolution for the 10-vs-10 comparison.
BASEBALL_SEPARABILITY = {
    "jeeds": (0.56, 0.53, 0.74, 0.76, 0.87),
    "hierarchical": (0.44, 0.52, 0.69, 0.77, 0.86),
}

BASEBALL_SIGMA_GAP = {
    "jeeds": (-0.0059, 0.0389, 0.0706, 0.1026, 0.1309),
    "hierarchical": (0.0002, 0.0204, 0.0403, 0.0813, 0.1146),
}

BASEBALL_EXECUTION_DRIFT = {
    "jeeds": (0.1286, 0.1200, 0.0630, 0.0520, 0.0),
    "hierarchical": (0.0734, 0.0703, 0.0511, 0.0449, 0.0),
}

BASEBALL_DECISION_DRIFT = {
    "jeeds": (0.694, 0.721, 0.611, 0.369, 0.0),
    "hierarchical": (0.999, 0.927, 0.317, 0.180, 0.0),
}


PLOT_RC = {
    "font.family": "DejaVu Sans",
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
    "font.size": 9.0,
    "axes.titlesize": 9.4,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 9.0,
    "savefig.facecolor": "white",
}


def _validate_recovery() -> None:
    """Guard the reported headline values against accidental transcription edits."""

    assert f"{TWO_D_EXECUTION['jeeds'].means[0]:.2f}" == "6.70"
    assert f"{TWO_D_EXECUTION['hierarchical'].means[0]:.2f}" == "5.93"
    assert f"{TWO_D_DECISION['jeeds'].means[0]:.1f}" == "21.8"
    assert f"{TWO_D_DECISION['hierarchical'].means[0]:.1f}" == "15.4"
    assert BASEBALL_SEPARABILITY["jeeds"][-1] == 0.87
    assert BASEBALL_SEPARABILITY["hierarchical"][-1] == 0.86
    assert f"{BASEBALL_SIGMA_GAP['jeeds'][-1]:.2f}" == "0.13"
    assert f"{BASEBALL_SIGMA_GAP['hierarchical'][-1]:.2f}" == "0.11"


def _style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.48, alpha=0.9)
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


def _save_bundle(figure, output_dir: Path, stem: str, dpi: int) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_dir / f"{stem}{suffix}"
        # Do not use bbox_inches="tight": the canvas is intentionally the exact
        # 3.35-inch AAAI column width, and all elements fit inside its margins.
        save_kwargs = {}
        if suffix == ".png":
            save_kwargs["dpi"] = dpi
            save_kwargs["metadata"] = {
                "Description": "Presentation redraw from digitized coordinates in the original paper PNG; not recomputed data."
            }
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


def plot_two_d(output_dir: Path, dpi: int) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, execution_axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.92))
        decision_axis = execution_axis.twinx()
        figure.patch.set_facecolor("white")
        x_values = np.arange(len(OBSERVATION_BUCKETS), dtype=float)

        for method in ("jeeds", "hierarchical"):
            _draw_error_series(
                execution_axis,
                x_values,
                TWO_D_EXECUTION[method],
                DUAL_AXIS_STYLES[(method, "execution")],
            )
            _draw_error_series(
                decision_axis,
                x_values,
                TWO_D_DECISION[method],
                DUAL_AXIS_STYLES[(method, "decision")],
            )

        execution_axis.set_ylim(4.75, 7.05)
        execution_axis.set_ylabel(
            "Execution error\n" + r"($|\hat{\sigma}-\sigma|$)",
            color=TEXT_COLOR,
            labelpad=4.5,
        )
        decision_axis.set_ylim(10.4, 23.2)
        decision_axis.set_ylabel(
            "Decision error\n(percentage points)",
            color=TEXT_COLOR,
            labelpad=5.0,
        )
        execution_axis.set_xticks(x_values, [str(value) for value in OBSERVATION_BUCKETS])
        execution_axis.set_xlabel("Observations per agent", color=TEXT_COLOR, labelpad=4.0)
        _style_dual_axes(execution_axis, decision_axis)
        _dual_axis_legend(figure, execution_axis, decision_axis)
        figure.subplots_adjust(left=0.205, right=0.795, top=0.785, bottom=0.19)
        written = _save_bundle(figure, output_dir, "10_two_d_error_by_count_bucket", dpi)
        plt.close(figure)
    return written


def _draw_method_lines(axis, x_values, values_by_method) -> None:
    for method in ("jeeds", "hierarchical"):
        style = BASEBALL_METHOD_STYLES[method]
        axis.plot(
            x_values,
            values_by_method[method],
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            markeredgewidth=0.55,
            linewidth=1.45,
            label=style["label"],
            zorder=3,
        )


def _set_pitch_axis(axis) -> None:
    axis.set_xlim(0.0, 105.0)
    axis.set_xticks(PITCH_COUNTS, [str(value) for value in PITCH_COUNTS])


def plot_baseball_separability(output_dir: Path, dpi: int) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    with plt.rc_context(PLOT_RC):
        figure, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 4.18), sharex=True)

        _draw_method_lines(axes[0], PITCH_COUNTS, BASEBALL_SEPARABILITY)
        axes[0].axhline(0.5, color="#88838F", linestyle="--", linewidth=0.75, zorder=1)
        axes[0].axhline(0.8, color="#66616F", linestyle=":", linewidth=0.8, zorder=1)
        axes[0].text(
            0.98,
            0.5,
            "chance",
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6.1,
            color=MUTED_TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
        )
        axes[0].text(
            0.98,
            0.8,
            "0.8",
            transform=axes[0].get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=6.1,
            color=MUTED_TEXT_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
        )
        axes[0].set_title("BB/IP separability", pad=4.0)
        axes[0].set_ylabel("AUC", labelpad=4.0)
        axes[0].set_ylim(0.0, 1.05)

        _draw_method_lines(axes[1], PITCH_COUNTS, BASEBALL_SIGMA_GAP)
        axes[1].axhline(0.0, color="#88838F", linestyle="--", linewidth=0.75, zorder=1)
        axes[1].set_title("Mean execution-skill gap", pad=4.0)
        axes[1].set_ylabel(r"Top $-$ bottom mean $\hat{\sigma}$", labelpad=4.0)
        axes[1].set_ylim(-0.012, 0.138)

        for axis in axes:
            _set_pitch_axis(axis)
            _style_axis(axis)
        axes[0].tick_params(labelbottom=False)
        axes[1].set_xlabel(r"Pitches observed, $N$", labelpad=4.0)
        _figure_legend(figure, axes[0])
        figure.align_ylabels(axes)
        figure.subplots_adjust(left=0.20, right=0.975, top=0.915, bottom=0.105, hspace=0.34)
        written = _save_bundle(figure, output_dir, "11_baseball_separability_by_N", dpi)
        plt.close(figure)
    return written


def plot_baseball_drift(output_dir: Path, dpi: int) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    with plt.rc_context(DUAL_AXIS_PLOT_RC):
        figure, execution_axis = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, 2.92))
        decision_axis = execution_axis.twinx()
        x_values = np.arange(len(PITCH_COUNTS), dtype=float)

        for method in ("jeeds", "hierarchical"):
            execution_style = DUAL_AXIS_STYLES[(method, "execution")]
            decision_style = DUAL_AXIS_STYLES[(method, "decision")]
            execution_axis.plot(
                x_values,
                BASEBALL_EXECUTION_DRIFT[method],
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
                BASEBALL_DECISION_DRIFT[method],
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

        execution_axis.set_ylim(-0.005, 0.135)
        execution_axis.set_yticks((0.00, 0.03, 0.06, 0.09, 0.12))
        execution_axis.set_ylabel(
            "Execution drift\n" + r"$|\hat{\sigma}_N-\hat{\sigma}_{100}|$",
            color=TEXT_COLOR,
            labelpad=4.5,
        )
        decision_axis.set_ylim(-0.05, 1.05)
        decision_axis.set_yticks((0.0, 0.25, 0.5, 0.75, 1.0))
        decision_axis.set_ylabel(
            "Decision drift\n"
            + r"$|\widehat{\log\lambda}_N-\widehat{\log\lambda}_{100}|$",
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
        execution_axis.set_xlim(-0.2, len(PITCH_COUNTS) - 0.8)
        execution_axis.set_xticks(x_values, [str(value) for value in PITCH_COUNTS])
        execution_axis.set_xlabel(r"Pitches observed, $N$", labelpad=4.0)
        _style_dual_axes(execution_axis, decision_axis)
        _dual_axis_legend(figure, execution_axis, decision_axis)
        figure.subplots_adjust(left=0.25, right=0.765, top=0.785, bottom=0.19)
        written = _save_bundle(figure, output_dir, "12_baseball_drift_by_N", dpi)
        plt.close(figure)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render digitized higher-dimensional paper figures for one-column AAAI layout."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg", force=True)
    args = parse_args(argv)
    if args.dpi <= 0:
        raise ValueError(f"--dpi must be positive, received {args.dpi}")

    _validate_recovery()
    written = (
        *plot_two_d(args.output_dir, args.dpi),
        *plot_baseball_separability(args.output_dir, args.dpi),
        *plot_baseball_drift(args.output_dir, args.dpi),
    )
    print("[main-paper-higher-dimensional] Digitized source PNGs:")
    for path in SOURCE_PNGS.values():
        print(f"  {path}")
    print("[main-paper-higher-dimensional] Wrote:")
    for path in written:
        print(f"  {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
