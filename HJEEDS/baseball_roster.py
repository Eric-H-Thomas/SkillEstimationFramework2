# This file has been fully reviewed by a human researcher as of 07/17/26 at 12:08 PM MDT.
"""Shared Statcast roster selection for baseball HJEEDS entry points.

Paper BBIP convergence (``submit_hjeeds_baseball_convergence_paper_bbip.sh``)
calls ``resolve_baseball_roster`` with ``--bbip-extremes 10``, ``--season-year
2021``, ``--pitch-types FF``, and ``min_pitches_per_agent=100``. That path
selects top-10 + bottom-10 BB/IP pitchers among FF-eligible pitchers, expands
to (pitcher, pitchType) agents, then drops any agent below the pitch floor.

Exactly one roster selector is allowed among ``--all-eligible-agents``,
``--top-pitchers``, and ``--bbip-extremes``; otherwise pass ``--pitcher-ids``.
``--max-agents`` caps the resolved list after selection (smoke tests only).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .baseball_hyperpriors import HYPERPRIOR_PRESET_CHOICES
from .baseball_pitch import (
    StatcastAgentSpec,
    build_eligible_agent_roster,
    count_agent_pitch_rows,
    filter_roster_by_min_pitches,
    filter_statcast_by_season,
    list_eligible_pitcher_counts,
    load_processed_statcast,
    resolve_agent_roster,
    select_top_pitchers_by_pitch_count,
)


@dataclass(frozen=True)
class BaseballRosterSelection:
    """Resolved agents and metadata for one baseball experiment."""

    season_year: int | None
    pitch_types: tuple[str, ...]
    pitcher_ids: tuple[int, ...]
    agent_specs: tuple[StatcastAgentSpec, ...]
    agent_pitch_counts: tuple[tuple[int, str, int], ...]
    excluded_agents: tuple[tuple[int, str, int], ...]


def parse_pitch_types(raw_value: str) -> tuple[str, ...]:
    pitch_types = tuple(piece.strip() for piece in raw_value.split(",") if piece.strip())
    if not pitch_types:
        raise ValueError("At least one pitch type is required.")
    return pitch_types


def load_statcast_for_roster(season_year: int | None) -> pd.DataFrame:
    all_data = load_processed_statcast()
    return filter_statcast_by_season(all_data, season_year)


def validate_roster_selection(
    *,
    all_eligible_agents: bool = False,
    top_pitchers: int | None = None,
    bbip_extremes: int | None = None,
) -> None:
    """Reject ambiguous combinations of mutually exclusive roster selector flags."""

    active = []
    if all_eligible_agents:
        active.append("--all-eligible-agents")
    if top_pitchers is not None:
        active.append("--top-pitchers")
    if bbip_extremes is not None:
        active.append("--bbip-extremes")
    if len(active) > 1:
        raise ValueError(
            "Use exactly one roster selector. Received: " + ", ".join(active)
        )


def _cap_agent_specs(
    agent_specs: Sequence[StatcastAgentSpec],
    max_agents: int | None,
) -> tuple[StatcastAgentSpec, ...]:
    """Keep at most ``max_agents`` specs and renumber ``agent_id`` from 0."""

    capped = tuple(agent_specs)
    if max_agents is not None and len(capped) > max_agents:
        capped = capped[:max_agents]
    return tuple(
        StatcastAgentSpec(agent_id=index, pitcher_id=spec.pitcher_id, pitch_type=spec.pitch_type)
        for index, spec in enumerate(capped)
    )


def _specs_from_pitcher_ids(
    pitcher_ids: Sequence[int],
    pitch_types: Sequence[str],
    all_data: pd.DataFrame,
    *,
    min_pitches_per_agent: int,
    max_agents: int | None,
) -> tuple[tuple[StatcastAgentSpec, ...], tuple[tuple[int, str, int], ...]]:
    """Expand pitcher×pitch-type, drop agents below the pitch floor, then cap."""

    agent_specs = resolve_agent_roster(pitcher_ids, pitch_types)
    agent_specs, excluded_agents = filter_roster_by_min_pitches(
        agent_specs,
        all_data,
        min_pitches_per_agent,
    )
    return _cap_agent_specs(agent_specs, max_agents), excluded_agents


def resolve_baseball_roster(
    *,
    all_data: pd.DataFrame,
    season_year: int | None,
    pitch_types: Sequence[str],
    pitcher_ids: Sequence[int] | None = None,
    top_pitchers: int | None = None,
    bbip_extremes: int | None = None,
    all_eligible_agents: bool = False,
    min_pitches_per_agent: int,
    max_agents: int | None = None,
    output_dir: Path | None = None,
) -> BaseballRosterSelection:
    """Resolve a concrete agent roster from CLI-style selection options."""

    validate_roster_selection(
        all_eligible_agents=all_eligible_agents,
        top_pitchers=top_pitchers,
        bbip_extremes=bbip_extremes,
    )

    excluded_agents: tuple[tuple[int, str, int], ...] = ()

    if bbip_extremes is not None:
        if season_year is None:
            raise ValueError("--bbip-extremes requires --season-year.")
        from .baseball_bbip import select_bbip_extreme_pitcher_ids

        # Eligibility is per (pitcher, pitchType); BB/IP extremes are pitcher-level.
        eligible_pitcher_ids = tuple(
            pitcher_id
            for pitcher_id, _pitch_type, _pitch_count in list_eligible_pitcher_counts(
                all_data,
                pitch_types,
                min_pitches=min_pitches_per_agent,
                limit=None,
            )
        )
        pitcher_ids_resolved = select_bbip_extreme_pitcher_ids(
            all_data,
            season_year=season_year,
            count=bbip_extremes,
            output_dir=output_dir,
            eligible_pitcher_ids=eligible_pitcher_ids,
        )
        agent_specs, excluded_agents = _specs_from_pitcher_ids(
            pitcher_ids_resolved,
            pitch_types,
            all_data,
            min_pitches_per_agent=min_pitches_per_agent,
            max_agents=max_agents,
        )
    elif all_eligible_agents:
        # Already applies min_pitches and optional max_agents inside pitch helpers.
        agent_specs = build_eligible_agent_roster(
            all_data,
            pitch_types,
            min_pitches=min_pitches_per_agent,
            max_agents=max_agents,
        )
    elif top_pitchers is not None:
        pitcher_ids_resolved = select_top_pitchers_by_pitch_count(
            all_data,
            pitch_types,
            min_pitches=min_pitches_per_agent,
            count=top_pitchers,
        )
        agent_specs, excluded_agents = _specs_from_pitcher_ids(
            pitcher_ids_resolved,
            pitch_types,
            all_data,
            min_pitches_per_agent=min_pitches_per_agent,
            max_agents=max_agents,
        )
    elif pitcher_ids:
        pitcher_ids_resolved = tuple(int(pitcher_id) for pitcher_id in pitcher_ids)
        agent_specs, excluded_agents = _specs_from_pitcher_ids(
            pitcher_ids_resolved,
            pitch_types,
            all_data,
            min_pitches_per_agent=min_pitches_per_agent,
            max_agents=max_agents,
        )
    else:
        raise ValueError(
            "Specify one roster selector: --all-eligible-agents, --top-pitchers, "
            "--bbip-extremes, or --pitcher-ids."
        )

    if not agent_specs:
        raise ValueError(
            f"No agents meet min_pitches_per_agent={min_pitches_per_agent}. "
            f"Excluded: {excluded_agents}."
        )

    agent_pitch_counts = tuple(
        (spec.pitcher_id, spec.pitch_type, count_agent_pitch_rows(all_data, spec.pitcher_id, spec.pitch_type))
        for spec in agent_specs
    )
    pitcher_ids_final = tuple(dict.fromkeys(spec.pitcher_id for spec in agent_specs))
    return BaseballRosterSelection(
        season_year=season_year,
        pitch_types=tuple(pitch_types),
        pitcher_ids=pitcher_ids_final,
        agent_specs=agent_specs,
        agent_pitch_counts=agent_pitch_counts,
        excluded_agents=excluded_agents,
    )


def add_common_roster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--season-year",
        type=int,
        default=None,
        help="Restrict Statcast rows to one game_year (e.g. 2021). Default: all seasons in the pickle.",
    )
    parser.add_argument(
        "--all-eligible-agents",
        action="store_true",
        help="Use every (pitcher, pitchType) pair meeting min_pitches_per_agent.",
    )
    parser.add_argument(
        "--top-pitchers",
        type=int,
        default=None,
        help="Select top-N pitchers by pitch count (cartesian product with --pitch-types).",
    )
    parser.add_argument(
        "--bbip-extremes",
        type=int,
        default=None,
        help="Select top-N and bottom-N pitchers by season BB/IP (requires --season-year).",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Cap the resolved roster size after selection (useful for smoke tests).",
    )
    parser.add_argument(
        "--min-pitches-per-agent",
        type=int,
        default=None,
        help="Require at least this many pitches per agent.",
    )
    parser.add_argument(
        "--list-eligible-pitchers",
        action="store_true",
        help="Print eligible agents and exit.",
    )
    parser.add_argument(
        "--list-eligible-limit",
        type=int,
        default=20,
        help="How many eligible agents to show with --list-eligible-pitchers.",
    )


def add_hyperprior_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hyperprior-preset",
        choices=HYPERPRIOR_PRESET_CHOICES,
        default="low-confidence",
        help="Population hyperprior preset. Default: low-confidence (weak empirical-Bayes prior).",
    )
    parser.add_argument(
        "--hyperprior-config",
        type=str,
        default=None,
        help="JSON file with HyperpriorConfig fields (required for --hyperprior-preset calibrated).",
    )


def print_eligible_agents(
    *,
    season_year: int | None,
    pitch_types: Sequence[str],
    min_pitches: int,
    limit: int | None,
) -> None:
    all_data = load_statcast_for_roster(season_year)
    all_rows = list_eligible_pitcher_counts(
        all_data,
        pitch_types,
        min_pitches=min_pitches,
        limit=None,
    )
    season_label = f"season={season_year}" if season_year is not None else "all seasons"
    total = len(all_rows)
    print(
        f"Eligible agents with >= {min_pitches} pitches ({season_label}): "
        f"{total} total"
    )
    if not all_rows:
        print("  (none)")
        return
    display_rows = all_rows if limit is None else all_rows[:limit]
    if limit is not None and total > limit:
        print(f"Showing top {limit}:")
    for pitcher_id, pitch_type, pitch_count in display_rows:
        print(f"  pitcher={pitcher_id} pitch_type={pitch_type} count={pitch_count}")
