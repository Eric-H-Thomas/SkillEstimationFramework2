# This file was written or edited by AI and still requires human review. Delete this comment when done.
"""BB/IP top/bottom separability of JEEDS vs H-JEEDS estimates by observation count."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from HJEEDS.baseball_bbip import label_bbip_tiers_for_pitcher_ids

DEFAULT_AUC_THRESHOLD = 0.8
AGENT_LEVEL_FILENAME = "convergence_agent_level_results.csv"
ROSTER_FILENAME = "convergence_roster.json"
METADATA_FILENAME = "convergence_roster_metadata.json"
SEPARABILITY_CSV = "separability_by_N.csv"
SEPARABILITY_PLOT = "separability_by_N.png"
SEPARABILITY_SUMMARY = "separability_summary.json"
TIER_TABLE_CSV = "bbip_tiers_corrected.csv"

SEPARABILITY_CSV_HEADER = [
    "method",
    "metric",
    "convergence_n",
    "num_bottom",
    "num_top",
    "mean_bottom",
    "mean_top",
    "mean_gap_top_minus_bottom",
    "auc",
    "notes",
]


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mann_whitney_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC for ranking ``scores`` against binary ``labels`` (1 = positive class).

    Equivalent to the Mann–Whitney U statistic normalized by n_pos * n_neg.
    """

    scores_array = np.asarray(scores, dtype=float)
    labels_array = np.asarray(labels, dtype=int)
    if scores_array.size != labels_array.size:
        raise ValueError("scores and labels must have the same length.")
    positive = scores_array[labels_array == 1]
    negative = scores_array[labels_array == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Need at least one positive and one negative label for AUC.")

    # Pairwise: fraction of (pos, neg) pairs where pos score > neg score (+ 0.5 ties).
    greater = 0.0
    for pos_score in positive:
        greater += float(np.sum(pos_score > negative))
        greater += 0.5 * float(np.sum(pos_score == negative))
    return greater / float(positive.size * negative.size)


def load_agent_level_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_bbip_tiers(
    output_dir: Path,
    *,
    extremes_count: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Return pitcher_id -> {tier, bbip, player_name, walks, innings_pitched}.

    Prefer recomputing tiers within the selected roster so mislabeled ``middle``
    rows in older metadata are corrected.
    """

    metadata_path = output_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing roster metadata: {metadata_path}")
    metadata = load_json(metadata_path)
    selection = metadata.get("bbip_selection") or []
    if not selection:
        raise ValueError(
            f"No bbip_selection in {metadata_path}. Re-run --prepare-roster with --bbip-extremes."
        )

    count = extremes_count
    if count is None:
        count = int(metadata.get("bbip_extremes") or metadata.get("roster_selector", {}).get("bbip_extremes") or 0)
    if count <= 0:
        count = max(1, len(selection) // 2)

    table_rows = [
        {
            "pitcher_id": int(row["pitcher_id"]),
            "player_name": row.get("player_name", ""),
            "walks": int(row.get("walks", 0)),
            "innings_pitched": float(row.get("innings_pitched", 0.0)),
            "bbip": float(row["bbip"]),
        }
        for row in selection
    ]
    import pandas as pd

    table = pd.DataFrame(table_rows)
    pitcher_ids = [int(row["pitcher_id"]) for row in table_rows]
    tiers = label_bbip_tiers_for_pitcher_ids(table, pitcher_ids, extremes_count=count)

    by_id: dict[int, dict[str, Any]] = {}
    for row in table_rows:
        pitcher_id = int(row["pitcher_id"])
        by_id[pitcher_id] = {
            **row,
            "tier": tiers[pitcher_id],
        }
    return by_id


def write_corrected_tier_table(path: Path, tiers: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(tiers.values(), key=lambda row: (str(row["tier"]), float(row["bbip"])))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["pitcher_id", "player_name", "walks", "innings_pitched", "bbip", "tier"],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_separability_rows(
    agent_rows: Sequence[dict[str, Any]],
    tiers: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute AUC and mean-gap rows for JEEDS and H-JEEDS at each N (sigma primary)."""

    methods = (
        ("jeeds", "jeeds_posterior_mean_sigma", "jeeds_posterior_mean_log_lambda"),
        ("hierarchical", "hierarchical_posterior_mean_sigma", "hierarchical_posterior_mean_log_lambda"),
    )
    skill_metrics = (
        ("sigma", 0, "Higher sigma = worse execution; expect top-BB/IP (walkers) higher."),
        ("log_lambda", 1, "Secondary axis; not the primary original-JEEDS separator."),
    )

    by_n: dict[int, list[dict[str, Any]]] = {}
    for row in agent_rows:
        pitcher_id = int(row["pitcher_id"])
        if pitcher_id not in tiers:
            continue
        tier = tiers[pitcher_id]["tier"]
        if tier not in ("top", "bottom"):
            continue
        convergence_n = int(row["convergence_n"])
        enriched = {**row, "tier": tier, "bbip": tiers[pitcher_id]["bbip"]}
        by_n.setdefault(convergence_n, []).append(enriched)

    output_rows: list[dict[str, Any]] = []
    for convergence_n in sorted(by_n):
        group = by_n[convergence_n]
        for method_name, sigma_key, log_lambda_key in methods:
            for metric_name, key_index, notes in skill_metrics:
                value_key = sigma_key if key_index == 0 else log_lambda_key
                bottom_scores: list[float] = []
                top_scores: list[float] = []
                for row in group:
                    value = _optional_float(row.get(value_key))
                    if value is None:
                        continue
                    if row["tier"] == "bottom":
                        bottom_scores.append(value)
                    else:
                        top_scores.append(value)
                if not bottom_scores or not top_scores:
                    continue
                scores = bottom_scores + top_scores
                # Positive class = top BB/IP (high walk rate). For sigma, higher score => top.
                labels = [0] * len(bottom_scores) + [1] * len(top_scores)
                auc = mann_whitney_auc(scores, labels)
                mean_bottom = float(np.mean(bottom_scores))
                mean_top = float(np.mean(top_scores))
                output_rows.append(
                    {
                        "method": method_name,
                        "metric": metric_name,
                        "convergence_n": convergence_n,
                        "num_bottom": len(bottom_scores),
                        "num_top": len(top_scores),
                        "mean_bottom": mean_bottom,
                        "mean_top": mean_top,
                        "mean_gap_top_minus_bottom": mean_top - mean_bottom,
                        "auc": auc,
                        "notes": notes,
                    }
                )
    return output_rows


def first_n_meeting_auc(
    rows: Sequence[dict[str, Any]],
    *,
    method: str,
    metric: str,
    threshold: float,
) -> int | None:
    candidates = [
        row
        for row in rows
        if row["method"] == method and row["metric"] == metric and float(row["auc"]) >= threshold
    ]
    if not candidates:
        return None
    return int(min(int(row["convergence_n"]) for row in candidates))


def write_separability_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEPARABILITY_CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def plot_separability_by_n(output_path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Keep visual language aligned with baseball drift / intermediate-estimate plots.
    from HJEEDS.baseball_convergence_study import CONVERGENCE_METHOD_STYLES

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    panels = (
        ("auc", "BB/IP separability (AUC)", "AUC (top vs bottom BB/IP)"),
        (
            "mean_gap_top_minus_bottom",
            r"Mean $\hat{\sigma}$ gap (top $-$ bottom)",
            r"Mean $\hat{\sigma}$ gap",
        ),
    )

    sigma_rows = [row for row in rows if row["metric"] == "sigma"]
    for axis, (y_key, title, ylabel) in zip(axes, panels):
        for method_name in ("jeeds", "hierarchical"):
            style = CONVERGENCE_METHOD_STYLES[method_name]
            method_rows = sorted(
                (row for row in sigma_rows if row["method"] == method_name),
                key=lambda row: int(row["convergence_n"]),
            )
            if not method_rows:
                continue
            xs = [int(row["convergence_n"]) for row in method_rows]
            ys = [float(row[y_key]) for row in method_rows]
            axis.plot(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=style["label"],
            )
        if y_key == "auc":
            axis.axhline(0.5, color="0.5", linestyle="--", linewidth=1.0, label="Chance")
            axis.axhline(
                DEFAULT_AUC_THRESHOLD,
                color="0.35",
                linestyle=":",
                linewidth=1.0,
                label=f"AUC={DEFAULT_AUC_THRESHOLD}",
            )
            axis.set_ylim(0.0, 1.05)
        axis.set_title(title)
        axis.set_xlabel(r"Pitch-count checkpoint $N$")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)
        axis.legend(loc="best", fontsize=8)

    figure.suptitle(
        r"Execution skill ($\hat{\sigma}$) vs BB/IP top/bottom groups",
        fontsize=11,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_summary(
    rows: Sequence[dict[str, Any]],
    *,
    auc_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "auc_threshold": auc_threshold,
        "primary_metric": "sigma",
        "methods": {},
    }
    for method in ("jeeds", "hierarchical"):
        summary["methods"][method] = {
            "first_n_auc_ge_threshold_sigma": first_n_meeting_auc(
                rows, method=method, metric="sigma", threshold=auc_threshold
            ),
            "first_n_auc_ge_threshold_log_lambda": first_n_meeting_auc(
                rows, method=method, metric="log_lambda", threshold=auc_threshold
            ),
            "auc_by_n_sigma": {
                str(row["convergence_n"]): float(row["auc"])
                for row in rows
                if row["method"] == method and row["metric"] == "sigma"
            },
            "gap_by_n_sigma": {
                str(row["convergence_n"]): float(row["mean_gap_top_minus_bottom"])
                for row in rows
                if row["method"] == method and row["metric"] == "sigma"
            },
        }
    return summary


def run_separability_analysis(
    output_dir: Path,
    *,
    auc_threshold: float = DEFAULT_AUC_THRESHOLD,
    extremes_count: int | None = None,
) -> dict[str, Any]:
    agent_path = output_dir / AGENT_LEVEL_FILENAME
    if not agent_path.is_file():
        raise FileNotFoundError(f"Missing agent-level results: {agent_path}")

    tiers = resolve_bbip_tiers(output_dir, extremes_count=extremes_count)
    write_corrected_tier_table(output_dir / TIER_TABLE_CSV, tiers)

    # Patch metadata bbip_selection tiers in place for reproducibility of later reads.
    metadata_path = output_dir / METADATA_FILENAME
    metadata = load_json(metadata_path)
    for row in metadata.get("bbip_selection") or []:
        pitcher_id = int(row["pitcher_id"])
        if pitcher_id in tiers:
            row["tier"] = tiers[pitcher_id]["tier"]
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    agent_rows = load_agent_level_rows(agent_path)
    separability_rows = compute_separability_rows(agent_rows, tiers)
    write_separability_csv(output_dir / SEPARABILITY_CSV, separability_rows)
    plot_separability_by_n(output_dir / SEPARABILITY_PLOT, separability_rows)
    summary = build_summary(separability_rows, auc_threshold=auc_threshold)
    summary["num_bottom"] = sum(1 for info in tiers.values() if info["tier"] == "bottom")
    summary["num_top"] = sum(1 for info in tiers.values() if info["tier"] == "top")
    summary["output_dir"] = str(output_dir.resolve())
    with (output_dir / SEPARABILITY_SUMMARY).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how quickly JEEDS vs H-JEEDS separate BB/IP top/bottom pitchers "
            "as a function of observation count N (post-hoc on convergence outputs)."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Convergence result directory containing convergence_agent_level_results.csv.",
    )
    parser.add_argument(
        "--auc-threshold",
        type=float,
        default=DEFAULT_AUC_THRESHOLD,
        help=f"AUC threshold for 'first N to separate' (default: {DEFAULT_AUC_THRESHOLD}).",
    )
    parser.add_argument(
        "--bbip-extremes",
        type=int,
        default=None,
        help="Override top/bottom count (default: read from roster metadata).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    summary = run_separability_analysis(
        output_dir,
        auc_threshold=args.auc_threshold,
        extremes_count=args.bbip_extremes,
    )
    print(f"[baseball-separability] Wrote artifacts under {output_dir.resolve()}", flush=True)
    for method, payload in summary["methods"].items():
        first_n = payload["first_n_auc_ge_threshold_sigma"]
        auc_by_n = payload["auc_by_n_sigma"]
        print(
            f"[baseball-separability] {method}: first N with AUC>={summary['auc_threshold']} "
            f"on sigma = {first_n}; auc_by_n={auc_by_n}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
