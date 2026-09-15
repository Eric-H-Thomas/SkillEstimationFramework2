"""Aggregate agents-per-bucket results for the supplementary figure.

Paper correspondence: Supplement, "Agents Per Observation-Count Bucket"
(`app:agents_per_bucket`).  This module computes the values shown in
Figure `fig:agents_per_bucket_sensitivity_panels`; it does not render a
standalone figure.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from HJEEDS.sensitivity_plot_common import (
    NUMERIC_5_COLORS,
    TEXT_COLOR,
    blend,
    seed_observation_from_agent_row,
    summary_fields,
    summarize_seed_improvements,
)


DEFAULT_RESULTS_DIR = Path("HJEEDS/results/hjeeds_paper_500_seeds/agents_per_bucket")
DEFAULT_AGENT_LEVEL_CSV = DEFAULT_RESULTS_DIR / "agents_per_bucket_sensitivity_agent_level_results.csv"

AGENTS_BASE_COLORS = {
    1: NUMERIC_5_COLORS[0],
    2: NUMERIC_5_COLORS[1],
    5: NUMERIC_5_COLORS[2],
    10: NUMERIC_5_COLORS[3],
    25: NUMERIC_5_COLORS[4],
}

CONDITION_ORDER = {
    "default": 0,
    "moderate_combined_misspecification": 1,
    "strong_combined_misspecification": 2,
}

CONDITION_LABELS = {
    "default": "Default",
    "moderate_combined_misspecification": "Moderate misspec.",
    "strong_combined_misspecification": "Strong misspec.",
}

CONDITION_CODES = {
    "default": "DEF",
    "moderate_combined_misspecification": "MOD",
    "strong_combined_misspecification": "STR",
}


@dataclass(frozen=True)
class AgentsCondition:
    """Metadata for one agents-per-bucket condition in the supplement."""

    agents_per_bucket: int
    condition_slug: str
    condition_label: str
    condition_code: str


@dataclass(frozen=True)
class ImprovementRow:
    """One seed-aggregated row used by the supplementary figure."""

    condition: AgentsCondition
    execution_mean: float
    execution_ci_lower: float
    execution_ci_upper: float
    decision_mean: float
    decision_ci_lower: float
    decision_ci_upper: float
    average_mean: float
    num_seeds: int
    num_agents_per_seed: int


def _selected_bucket(agent_level_csv: Path, requested_bucket: int | None) -> int:
    """Return the requested bucket, or the smallest bucket in the result file."""

    if requested_bucket is not None:
        return requested_bucket
    with agent_level_csv.open("r", newline="") as handle:
        buckets = {int(row["count_bucket"]) for row in csv.DictReader(handle)}
    if not buckets:
        raise ValueError(f"No count buckets found in {agent_level_csv}.")
    return min(buckets)


def compute_improvement_rows(
    *,
    agent_level_csv: Path,
    count_bucket: int | None,
) -> tuple[list[ImprovementRow], int]:
    """Compute the seed-level improvements shown for one observation bucket."""

    selected_bucket = _selected_bucket(agent_level_csv, count_bucket)
    observations = []
    metadata: dict[tuple[int, str], AgentsCondition] = {}
    with agent_level_csv.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["count_bucket"]) != selected_bucket:
                continue
            if row.get("jeeds_status") != "ok" or row.get("hierarchical_status") != "ok":
                continue
            condition_slug = str(row["condition_slug"])
            if condition_slug not in CONDITION_ORDER:
                continue
            agents_per_bucket = int(row["agents_per_bucket"])
            condition_key = (agents_per_bucket, condition_slug)
            observation = seed_observation_from_agent_row(row, condition_key)
            if observation is None:
                continue
            metadata[condition_key] = AgentsCondition(
                agents_per_bucket=agents_per_bucket,
                condition_slug=condition_slug,
                condition_label=CONDITION_LABELS[condition_slug],
                condition_code=CONDITION_CODES[condition_slug],
            )
            observations.append(observation)

    rows = [
        ImprovementRow(condition=metadata[key], **summary_fields(summary))
        for key, summary in summarize_seed_improvements(observations).items()
    ]
    rows.sort(
        key=lambda row: (
            row.condition.agents_per_bucket,
            CONDITION_ORDER[row.condition.condition_slug],
        )
    )
    return rows, selected_bucket


def bar_color(condition: AgentsCondition) -> str:
    """Return the hue/shade used for an agents-by-hyperprior condition."""

    base = AGENTS_BASE_COLORS.get(condition.agents_per_bucket, "#B8B8B8")
    if condition.condition_slug == "default":
        return blend(base, "#FFFFFF", 0.42)
    if condition.condition_slug == "strong_combined_misspecification":
        return blend(base, TEXT_COLOR, 0.28)
    return base
