"""Streamlit shot inspector for Blackhawks JEEDS data.

Usage:
    pip install -r requirements.txt
    streamlit run BlackhawksApp/app.py

Config workflow:
    1) Build/export JSON under Data/Hockey/jobs from the Cluster Config Builder.
    2) On cluster submit with:
       sbatch run_blackhawks_config.sbatch Data/Hockey/jobs/<config>.json
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

from BlackhawksApp import data_io
from BlackhawksSkillEstimation.blackhawks_plots import (
    plot_shot_angular_heatmap,
    plot_shot_rink,
)
from BlackhawksSkillEstimation.plot_intermediate_estimates import (
    get_estimate_before_after_delta,
)
from BlackhawksSkillEstimation.BlackhawksJEEDS import save_player_data
from BlackhawksSkillEstimation.player_cache import lookup_player

st.set_page_config(layout="wide")
st.title("Blackhawks JEEDS Shot Inspector")

@st.cache_data(show_spinner=False)
def _cached_players(data_root: str) -> list[int]:
    return data_io.get_players(data_root)


@st.cache_data(show_spinner=False)
def _cached_seasons(player_id: int, data_root: str) -> list[int]:
    return data_io.get_seasons(player_id=player_id, data_dir=data_root)


@st.cache_data(show_spinner=False)
def _cached_shots_metadata(player_id: int, season: int, data_root: str):
    return data_io.load_shots_metadata(player_id=player_id, season=season, data_dir=data_root)


@st.cache_resource(show_spinner=False)
def _cached_shot_maps(player_id: int, season: int, data_root: str):
    return data_io.load_shot_maps_for_season(player_id=player_id, season=season, data_dir=data_root)


@st.cache_data(show_spinner=False)
def _cached_estimates(csv_path: str):
    return data_io.load_estimates(csv_path)


@st.cache_data(show_spinner=False)
def _cached_player_id_text_files(data_root: str) -> list[str]:
    return [str(p) for p in data_io.get_player_id_text_files(data_root)]


@st.cache_data(show_spinner=False)
def _cached_player_name_catalog(player_ids: tuple[int, ...]):
    return data_io.get_player_name_catalog(list(player_ids))


@st.cache_data(show_spinner=False)
def _cached_union_seasons(player_ids: tuple[int, ...], data_root: str) -> list[int]:
    return data_io.get_all_available_seasons(list(player_ids), data_root)


@st.cache_data(show_spinner=False)
def _cached_partition_values(player_ids: tuple[int, ...], seasons: tuple[int, ...], data_root: str):
    return data_io.discover_partition_values(list(player_ids), list(seasons), data_dir=data_root)


@st.cache_data(show_spinner=False)
def _cached_observation_summary(
    player_ids: tuple[int, ...],
    seasons: tuple[int, ...],
    shot_groups: tuple[str, ...],
    data_root: str,
    partition_column: str,
    partition_values: tuple[str, ...],
):
    partition_col = partition_column if partition_column else None
    part_vals = list(partition_values) if partition_values else None
    return data_io.build_observation_summary(
        list(player_ids),
        list(seasons),
        list(shot_groups),
        data_dir=data_root,
        partition_column=partition_col,
        partition_values=part_vals,
    )


def _safe_float(value: object, default: float = 0.08) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(out) or np.isinf(out):
        return default
    return out


def _format_pow10(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"10^{value:.3f}"


def _sync_shot_index(value: int) -> None:
    st.session_state["selected_shot_index"] = int(value)
    st.session_state["shot_index_slider"] = int(value)
    st.session_state["shot_index_input"] = int(value)


def _on_slider_change() -> None:
    _sync_shot_index(int(st.session_state.get("shot_index_slider", 1)))


def _on_input_change() -> None:
    _sync_shot_index(int(st.session_state.get("shot_index_input", 1)))


data_root = st.sidebar.text_input("Data root", value="Data/Hockey")
players = _cached_players(data_root)
if not players:
    st.error(f"No player folders found under: {data_root}")
    st.stop()

player_id = st.sidebar.selectbox("Player", players, index=0)
seasons = _cached_seasons(player_id=player_id, data_root=data_root)
if not seasons:
    st.warning("No seasons found for selected player.")
    st.stop()

default_season = seasons[-1]
season = st.sidebar.selectbox("Season", seasons, index=seasons.index(default_season))

shot_groups = data_io.get_shot_group_tags()
shot_group = st.sidebar.selectbox(
    "Shot type group",
    shot_groups,
    index=0,
    format_func=lambda g: f"{g} ({data_io.get_shot_group_display(g)})",
)

try:
    season_df = _cached_shots_metadata(player_id=player_id, season=season, data_root=data_root)
except Exception as exc:
    st.error(f"Failed to load season metadata: {exc}")
    st.stop()

filtered_df = data_io.filter_by_shot_group(season_df, shot_group=shot_group)
filtered_df = data_io.with_shot_index(filtered_df)

if filtered_df.empty:
    st.warning("No shots match the selected season and shot group.")
    st.stop()

csv_path = data_io.get_convergence_artifact(
    player_id=player_id,
    season=season,
    shot_group=shot_group,
    data_dir=data_root,
    suffix=".csv",
)
png_path = data_io.get_convergence_artifact(
    player_id=player_id,
    season=season,
    shot_group=shot_group,
    data_dir=data_root,
    suffix=".png",
)

estimates = None
if csv_path is not None:
    try:
        estimates = _cached_estimates(str(csv_path))
    except Exception as exc:
        st.warning(f"Could not load estimates CSV: {exc}")

csv_shot_count = 0
if estimates is not None:
    csv_shot_count = len(estimates.get("shot_count", []))

nav_max = csv_shot_count if csv_shot_count > 0 else len(filtered_df)
if nav_max <= 0:
    st.warning("No shot index available for navigation.")
    st.stop()

st.subheader("Convergence")
if png_path is None:
    st.warning("Missing pre-generated convergence PNG for this player/season/shot-group context.")
else:
    # Keep the plot centered while allowing it to occupy most of the viewport width.
    center_cols = st.columns([1, 12, 1])
    center_cols[1].image(str(png_path), caption=f"{png_path.name}", width='stretch')

selection_context = (player_id, season, shot_group, len(filtered_df), nav_max)
if st.session_state.get("selection_context") != selection_context:
    st.session_state["selection_context"] = selection_context
    _sync_shot_index(1)

if "selected_shot_index" not in st.session_state:
    _sync_shot_index(1)

max_index = nav_max
selected_idx = int(st.session_state.get("selected_shot_index", 1))
if selected_idx < 1 or selected_idx > max_index:
    _sync_shot_index(1)
    selected_idx = 1

main_cols = st.columns([1, 2])
nav_col = main_cols[0]
table_col = main_cols[1]

nav_col.subheader("Shot navigation")
controls = nav_col.columns([2, 4])

if "shot_index_input" not in st.session_state:
    st.session_state["shot_index_input"] = selected_idx
if "shot_index_slider" not in st.session_state:
    st.session_state["shot_index_slider"] = selected_idx

controls[0].number_input(
    "Shot #",
    min_value=1,
    max_value=max_index,
    step=1,
    key="shot_index_input",
    on_change=_on_input_change,
)
controls[1].slider(
    "Shot number",
    min_value=1,
    max_value=max_index,
    key="shot_index_slider",
    on_change=_on_slider_change,
)

selected_idx = int(st.session_state.get("selected_shot_index", 1))
selected_row = data_io.get_shot_row_by_index(filtered_df, selected_idx)
selected_event_id = int(selected_row["event_id"]) if selected_row is not None else None

before, after, delta = None, None, None
before_rat_log10, after_rat_log10, delta_rat_log10 = None, None, None
if estimates is not None:
    try:
        before, after, delta = get_estimate_before_after_delta(
            estimates,
            shot_index=selected_idx,
            metric="expected_execution_skill",
        )
        before_rat_log10, after_rat_log10, delta_rat_log10 = get_estimate_before_after_delta(
            estimates,
            shot_index=selected_idx,
            metric="log10_expected_rationality",
        )
    except Exception as exc:
        nav_col.warning(f"Could not load estimate deltas: {exc}")
else:
    nav_col.caption("No matching convergence CSV found for estimate delta.")

nav_col.write(f"Selected shot index: {selected_idx}")
if selected_event_id is not None:
    nav_col.write(f"Selected event_id: {selected_event_id}")
else:
    nav_col.warning("Selected shot index is outside the filtered season rows; adjust index for shot-level visuals.")

if selected_row is not None:
    nav_col.write(f"Game ID: {int(selected_row.get('game_id', -1))}")

if after is not None:
    before_str = f"{before:.5f}" if before is not None else "N/A"
    after_str = f"{after:.5f}"
    delta_str = f"{delta:.5f}" if delta is not None else "N/A"
    nav_col.write(
        f"xskill expected before/after: {before_str} -> {after_str}"
    )
    nav_col.write(f"delta xskill expected (after-before): {delta_str}")

if after_rat_log10 is not None:
    before_rat_str = _format_pow10(before_rat_log10)
    after_rat_str = _format_pow10(after_rat_log10)
    delta_rat_str = f"{delta_rat_log10:.5f}" if delta_rat_log10 is not None else "N/A"
    nav_col.write(
        f"rationality expected before/after: {before_rat_str} -> {after_rat_str}"
    )
    nav_col.write(f"delta rationality expected log10 (after-before): {delta_rat_str}")

table_col.subheader("Filtered shot table")
table_df = filtered_df.copy()
show_post_shot_xg = table_col.checkbox(
    "Show post-shot xG column",
    value=False,
    help="Loads cached shot maps and samples the xG grid at each shot location.",
)

if show_post_shot_xg:
    try:
        shot_maps_for_table = _cached_shot_maps(player_id=player_id, season=season, data_root=data_root)
        table_df = data_io.add_post_shot_xg_column(table_df, shot_maps_for_table)
    except Exception as exc:
        table_col.warning(f"Could not compute post-shot xG column: {exc}")

if estimates is not None:
    deltas: list[float | None] = []
    rationality_deltas: list[float | None] = []
    for idx in range(1, len(table_df) + 1):
        _before_i, _after_i, delta_i = get_estimate_before_after_delta(
            estimates,
            shot_index=idx,
            metric="expected_execution_skill",
        )
        _before_rat_i, _after_rat_i, delta_rat_i = get_estimate_before_after_delta(
            estimates,
            shot_index=idx,
            metric="log10_expected_rationality",
        )
        deltas.append(delta_i)
        rationality_deltas.append(delta_rat_i)
    table_df["delta_expected_execution_skill"] = deltas
    table_df["delta_expected_rationality"] = rationality_deltas

table_columns = [
    col
    for col in [
        "shot_index",
        "event_id",
        "game_id",
        "period",
        "shot_type",
        "shot_is_goal",
        "post_shot_xg",
        "delta_expected_execution_skill",
        "delta_expected_rationality",
        "start_x",
        "start_y",
        "location_y",
        "location_z",
    ]
    if col in table_df.columns
]
table_col.dataframe(table_df[table_columns], width='stretch', height=320, hide_index=True)

if selected_row is not None:
    st.write(
        f"X/Y on ice (start_x/start_y): ({_safe_float(selected_row.get('start_x')):.2f}, {_safe_float(selected_row.get('start_y')):.2f})"
    )
    st.write(
        f"Projected location on net (location_z/location_y): ({_safe_float(selected_row.get('location_z')):.2f}, {_safe_float(selected_row.get('location_y')):.2f})"
    )
else:
    st.caption("No row metadata available at this shot index for the current filtered season table.")

confirm_col = st.columns(1)[0]
if confirm_col.button("Load visuals for selected shot"):
    st.session_state["confirmed_context"] = (player_id, season, shot_group, selected_idx)

if st.session_state.get("confirmed_context") == (player_id, season, shot_group, selected_idx):
    with st.spinner("Rendering selected-shot visuals..."):
        try:
            if selected_row is None or selected_event_id is None:
                st.warning("Selected shot index cannot be mapped to an event row for visuals.")
                st.stop()

            shot_maps = _cached_shot_maps(player_id=player_id, season=season, data_root=data_root)
            payload = data_io.get_heatmap_for_shot(shot_maps, selected_event_id)
            if payload is None:
                st.warning("No shot-map payload for selected event_id.")
                st.stop()

            player_loc = [
                _safe_float(selected_row.get("start_x"), default=0.0),
                _safe_float(selected_row.get("start_y"), default=0.0),
            ]
            executed_action = [
                _safe_float(selected_row.get("location_y"), default=0.0),
                _safe_float(selected_row.get("location_z"), default=0.0),
            ]

            fig_heat = plot_shot_angular_heatmap(
                value_map=payload["value_map"],
                player_location=player_loc,
                executed_action=executed_action,
                show=False,
                title=f"Player {player_id}",
                event_id=selected_event_id,
                is_goal=bool(selected_row.get("shot_is_goal", False)),
            )
            if fig_heat is not None:
                st.pyplot(fig_heat)

            fig_rink = plot_shot_rink(
                [player_loc],
                is_goal=[bool(selected_row.get("shot_is_goal", False))],
                show=False,
                title=f"Player {player_id} - Event {selected_event_id}",
                player_xy_list=[player_loc],
            )
            rink_col, cov_col = st.columns([1, 1])
            if fig_rink is not None:
                with rink_col:
                    st.pyplot(fig_rink)
            with cov_col:
                st.caption("Net-face covariance plot placeholder (coming soon).")
        except Exception as exc:
            st.error(f"Failed to render selected-shot visuals: {exc}")
else:
    st.caption("Press 'Load visuals for selected shot' to render heatmaps and rink.")

st.divider()
st.header("Cluster Config Builder")
st.caption("Build a JSON config from local filters, preview sample counts, and save under Data/Hockey/jobs.")

_CONFIG_DEFAULTS = {
    "cfg_player_file": "",
    "cfg_players": [],
    "cfg_player_select_mode": "ID",
    "cfg_player_names": [],
    "cfg_pasted_players": "",
    "cfg_seasons": [],
    "cfg_shot_groups": data_io.get_shot_group_tags(),
    "cfg_enable_partition": False,
    "cfg_partition_column": "",
    "cfg_partition_values": [],
    "cfg_estimation_mode": "per_season",
    "cfg_min_shots": 50,
    "cfg_num_execution_skills": 50,
    "cfg_num_planning_skills": 100,
    "cfg_rng_seed": 0,
    "cfg_save_csv": True,
    "cfg_generate_png": True,
    "cfg_png_include_map": True,
    "cfg_sbatch_time": "24:00:00",
    "cfg_sbatch_mem": "16G",
    "cfg_sbatch_max_concurrent": 100,
    "cfg_write_run_summary": False,
}

if st.button("Reset config builder", help="Reset all config-builder fields to defaults."):
    for _key in list(st.session_state.keys()):
        if _key in _CONFIG_DEFAULTS or _key.startswith("cfg_ambig_name_"):
            del st.session_state[_key]
    st.rerun()

config_cols = st.columns([1, 1])

with config_cols[0]:
    st.subheader("Player + Filter Selection")
    player_select_mode = st.radio(
        "Player input mode",
        options=["ID", "Name"],
        horizontal=True,
        key="cfg_player_select_mode",
        help="ID mode is the default. Name mode stays hidden unless you switch to it.",
    )

    selected_players_for_config: list[int] = []
    selected_player_names_for_config: list[str] = []
    selected_player_file = ""
    pasted_players = ""

    if player_select_mode == "ID":
        player_file_options = [""] + _cached_player_id_text_files(data_root)
        selected_player_file = st.selectbox(
            "Optional player ID text file",
            options=player_file_options,
            index=0,
            key="cfg_player_file",
            help="Reads IDs from files like Data/Hockey/forwards23-25.txt.",
        )
        selected_players_for_config = st.multiselect(
            "Add players from local cache",
            options=players,
            default=[],
            key="cfg_players",
        )
        pasted_players = st.text_area(
            "Optional pasted player IDs",
            value="",
            height=90,
            key="cfg_pasted_players",
            placeholder="Example: 950182, 950169, 950181",
        )
    else:
        name_catalog = _cached_player_name_catalog(tuple(players))
        cached_names = list(name_catalog.keys())
        st.caption(f"Name mode uses local cache only. {len(cached_names)} cached name(s) available.")
        selected_player_names_for_config = st.multiselect(
            "Select player names",
            options=cached_names,
            default=[],
            key="cfg_player_names",
            help="Search and select names from cache. If a name is missing, switch to ID mode.",
        )

        explicit_ids_for_ambiguous_names: list[int] = []
        ambiguous_names = [name for name in selected_player_names_for_config if len(name_catalog.get(name, [])) > 1]
        if ambiguous_names:
            st.warning("Some selected names map to multiple player IDs. Choose the exact ID(s) below.")
            for name in ambiguous_names:
                candidates = name_catalog.get(name, [])
                chosen_ids = st.multiselect(
                    f"Resolve {name}",
                    options=candidates,
                    default=[],
                    key=f"cfg_ambig_name_{name}",
                    format_func=lambda pid, player_name=name: f"{player_name} ({pid})",
                    help="Select one or more IDs for this shared name.",
                )
                explicit_ids_for_ambiguous_names.extend([int(pid) for pid in chosen_ids])

            unresolved_ambiguous = [
                name
                for name in ambiguous_names
                if not st.session_state.get(f"cfg_ambig_name_{name}")
            ]
            if unresolved_ambiguous:
                st.caption(
                    f"Waiting for ID selection on {len(unresolved_ambiguous)} ambiguous name(s)."
                )

            selected_player_names_for_config = [
                name
                for name in selected_player_names_for_config
                if len(name_catalog.get(name, [])) == 1
            ]
            selected_players_for_config = explicit_ids_for_ambiguous_names

        if not cached_names:
            st.info("No cached player names available yet. Use ID mode to populate cache first.")

    resolved_players = data_io.resolve_player_ids(
        selected_players=selected_players_for_config,
        selected_player_names=selected_player_names_for_config,
        player_file=selected_player_file if selected_player_file else None,
        pasted_player_ids=pasted_players,
    )

    if resolved_players:
        if player_select_mode == "Name":
            st.caption(
                f"Resolved {len(resolved_players)} unique players from {len(selected_player_names_for_config)} unique-name selection(s)."
            )
        else:
            st.caption(f"Resolved {len(resolved_players)} unique players.")
    else:
        if player_select_mode == "Name":
            st.warning("Select at least one cached name (and resolve any ambiguous names) to build a config.")
        else:
            st.warning("Select at least one player source to build a config.")

    available_seasons = _cached_union_seasons(tuple(resolved_players), data_root) if resolved_players else []
    selected_seasons_for_config = st.multiselect(
        "Seasons",
        options=available_seasons,
        default=[],
        key="cfg_seasons",
    )

    with st.expander("Add seasons not in local cache", expanded=False):
        extra_seasons_raw = st.text_input(
            "Additional seasons (8-digit, comma/space separated)",
            value="",
            key="cfg_extra_seasons",
            placeholder="Example: 20222023 20212022",
        )

        season_tokens = [tok for tok in re.split(r"[\s,]+", extra_seasons_raw.strip()) if tok]
        extra_seasons: set[int] = set()
        invalid_tokens = 0
        for tok in season_tokens:
            if re.fullmatch(r"\d{8}", tok):
                extra_seasons.add(int(tok))
            else:
                invalid_tokens += 1

        if invalid_tokens > 0:
            st.warning(
                f"Ignored {invalid_tokens} invalid season token(s). Use 8-digit format like 20222023.",
            )

    if extra_seasons:
        selected_seasons_for_config = sorted(set(selected_seasons_for_config) | extra_seasons)

    local_season_set = set(available_seasons)
    selected_local_count = sum(1 for season_i in selected_seasons_for_config if season_i in local_season_set)
    selected_missing_count = max(0, len(selected_seasons_for_config) - selected_local_count)
    st.caption(
        f"Selected seasons: {selected_local_count} local, {selected_missing_count} to download"
    )

    all_shot_groups = data_io.get_shot_group_tags()
    selected_shot_groups = st.multiselect(
        "Shot groups",
        options=all_shot_groups,
        default=all_shot_groups,
        key="cfg_shot_groups",
        format_func=lambda g: f"{g} ({data_io.get_shot_group_display(g)})",
    )

    enable_partition_filter = st.checkbox(
        "Enable partition filter",
        value=False,
        key="cfg_enable_partition",
        help="Leave off unless you explicitly want a partition-based subset.",
    )
    partition_column = ""
    partition_values: list[str] = []
    if enable_partition_filter:
        partition_catalog = (
            _cached_partition_values(tuple(resolved_players), tuple(selected_seasons_for_config), data_root)
            if resolved_players and selected_seasons_for_config
            else {}
        )
        partition_column_options = [""] + sorted(partition_catalog.keys())
        partition_column = st.selectbox(
            "Partition column",
            options=partition_column_options,
            index=0,
            key="cfg_partition_column",
            help="Example values may include labels like Lshallow/Rshallow when present in data.",
        )
        if partition_column:
            partition_values = st.multiselect(
                "Partition values",
                options=partition_catalog.get(partition_column, []),
                default=partition_catalog.get(partition_column, []),
                key="cfg_partition_values",
            )

with config_cols[1]:
    st.subheader("Estimator + Job Settings")
    estimation_mode = st.selectbox(
        "Estimation mode",
        options=["per_season", "all_selected_seasons_together"],
        index=0,
        key="cfg_estimation_mode",
        format_func=lambda mode: (
            "Per-season"
            if mode == "per_season"
            else "Combined seasons"
        ),
        help=(
            "Determines whether to give each season its own job or to pool them all into "
            "one continuous estimate per player over the course of all selected seasons."
        ),
    )

    min_shots_per_job = st.number_input(
        "Minimum observations per job",
        min_value=1,
        max_value=10000,
        value=50,
        step=1,
        key="cfg_min_shots",
    )
    num_execution_skills = st.slider(
        "Execution skill hypotheses",
        min_value=10,
        max_value=250,
        value=50,
        step=5,
        key="cfg_num_execution_skills",
    )
    num_planning_skills = st.slider(
        "Planning skill hypotheses",
        min_value=10,
        max_value=250,
        value=100,
        step=10,
        key="cfg_num_planning_skills",
    )
    rng_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
        key="cfg_rng_seed",
    )
    save_intermediate_csv = st.checkbox(
        "Save intermediate estimates CSV",
        value=True,
        key="cfg_save_csv",
    )
    generate_convergence_png = st.checkbox(
        "Generate convergence PNG from intermediate CSV",
        value=True,
        key="cfg_generate_png",
        help="When enabled, writes convergence PNGs next to intermediate CSVs.",
    )
    png_include_map = st.checkbox(
        "Include MAP estimates in convergence PNG",
        value=True,
        key="cfg_png_include_map",
        help="Turn off to render expected estimates only (no MAP lines).",
    )
    per_season_estimation = estimation_mode == "per_season"

    st.markdown("**SBATCH settings used for array submission**")
    sbatch_time = st.text_input(
        "SBATCH time limit",
        value="24:00:00",
        key="cfg_sbatch_time",
        help=(
            "Walltime passed to sbatch --time. Format: HH:MM:SS. "
            "Example/default: 24:00:00 (24 hours)."
        ),
    )
    sbatch_mem = st.text_input(
        "SBATCH memory",
        value="16G",
        key="cfg_sbatch_mem",
        help=(
            "Memory passed to sbatch --mem per task. "
            "Use units like G or M (for example, 16G)."
        ),
    )
    sbatch_max_concurrent = st.number_input(
        "Max concurrent array jobs",
        min_value=1,
        max_value=1000,
        value=100,
        step=1,
        key="cfg_sbatch_max_concurrent",
        help=(
            "Concurrency cap used in sbatch --array as %%N. "
            "Example/default: 100 means --array=1-<jobs>%100."
        ),
    )

    st.markdown("**Run outputs**")
    write_run_summary = st.checkbox(
        "Generate run_summary JSON",
        value=False,
        key="cfg_write_run_summary",
        help="When enabled, writes a run_summary JSON after execution.",
    )

can_build_summary = bool(resolved_players and selected_seasons_for_config and selected_shot_groups)
summary_df = None
download_target_df = None
jobs_preview: list[dict[str, object]] = []

if can_build_summary:
    summary_df = _cached_observation_summary(
        tuple(resolved_players),
        tuple(selected_seasons_for_config),
        tuple(selected_shot_groups),
        data_root,
        partition_column,
        tuple(partition_values),
    )
    download_target_df = summary_df.copy()

    if estimation_mode == "all_selected_seasons_together" and not summary_df.empty:
        grouped = (
            summary_df
            .groupby(["player_id", "shot_group"], as_index=False)
            .agg(
                count=("count", "sum"),
                missing_local_data=("missing_local_data", "max"),
            )
        )
        grouped["season"] = -1
        summary_df = grouped[["player_id", "season", "shot_group", "count", "missing_local_data"]]

    jobs_preview = data_io.build_job_rows(summary_df, min_shots_per_job=int(min_shots_per_job))

    total_obs = int(summary_df["count"].sum()) if not summary_df.empty else 0
    eligible_jobs = [j for j in jobs_preview if j["eligible"]]
    skipped_jobs = [j for j in jobs_preview if not j["eligible"]]

    metric_cols = st.columns([1, 1, 1, 1])
    metric_cols[0].metric("Total matching observations", total_obs)
    metric_cols[1].metric("Total jobs", len(jobs_preview))
    metric_cols[2].metric("Eligible jobs", len(eligible_jobs))
    metric_cols[3].metric("Below threshold / missing", len(skipped_jobs))

    missing_pairs: list[dict[str, int]] = []
    missing_count = 0
    if download_target_df is not None and not download_target_df.empty:
        missing_df = (
            download_target_df.loc[
                download_target_df["missing_local_data"].astype(bool),
                ["player_id", "season"],
            ]
            .drop_duplicates()
            .sort_values(["player_id", "season"])
        )
        missing_count = int(len(missing_df))
        if missing_count > 0:
            missing_pairs = missing_df.to_dict("records")

    with st.container(border=True):
        st.markdown("**On-demand local data backfill**")
        download_flash = st.session_state.pop("cfg_download_flash", None)
        if download_flash:
            level = download_flash.get("level", "info")
            message = str(download_flash.get("message", ""))
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)

        if missing_count == 0:
            st.caption("No missing local player-season data found for the current filters.")
        else:
            st.caption(
                f"Found {missing_count} missing player-season pair(s). Local files stay default; download is only used for missing rows."
            )

        run_download = st.button(
            "Download Missing Data",
            disabled=missing_count == 0,
            help="Fetch only player-season rows marked as missing_local_data in this summary.",
        )

        if run_download and missing_pairs:
            grouped_missing: dict[int, list[int]] = {}
            for row in missing_pairs:
                pid = int(row["player_id"])
                season_i = int(row["season"])
                grouped_missing.setdefault(pid, []).append(season_i)

            downloaded_seasons = 0
            players_with_downloads = 0
            players_no_new = 0
            failed_players: list[int] = []

            with st.spinner("Downloading missing player-season data..."):
                for pid, seasons_for_pid in grouped_missing.items():
                    try:
                        # Local-first behavior: skip seasons already persisted.
                        saved = save_player_data(
                            player_id=pid,
                            seasons=sorted(set(seasons_for_pid)),
                            output_dir=Path(data_root),
                            overwrite=False,
                        )
                    except Exception:
                        failed_players.append(pid)
                        continue

                    saved_count = len(saved)
                    if saved_count > 0:
                        players_with_downloads += 1
                        downloaded_seasons += saved_count
                    else:
                        players_no_new += 1

            if failed_players:
                st.session_state["cfg_download_flash"] = {
                    "level": "warning",
                    "message": (
                        "Backfill finished with partial failures. "
                        f"Downloaded {downloaded_seasons} season(s) across {players_with_downloads} player(s); "
                        f"no-new-data for {players_no_new} player(s); failed players: {failed_players}."
                    ),
                }
            else:
                st.session_state["cfg_download_flash"] = {
                    "level": "success",
                    "message": (
                        "Backfill finished. "
                        f"Downloaded {downloaded_seasons} season(s) across {players_with_downloads} player(s); "
                        f"no-new-data for {players_no_new} player(s)."
                    ),
                }

            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    preview_df = summary_df.copy()
    if estimation_mode == "all_selected_seasons_together":
        preview_df["season"] = "ALL_SELECTED"
    
    # Add player names from cache
    preview_df["player_name"] = preview_df["player_id"].apply(
        lambda pid: lookup_player(player_id=pid) or f"Player {pid}"
    )
    
    # Display with player names first
    display_columns = ["player_id", "player_name", "season", "shot_group", "count", "missing_local_data"]
    display_df = preview_df[[col for col in display_columns if col in preview_df.columns]]
    
    st.dataframe(
        display_df.sort_values(["player_id", "season", "shot_group"]),
        width="stretch",
        height=260,
    )
else:
    st.info("Select players, seasons, and shot groups to compute observation counts.")

export_section_cols = st.columns([2, 1])
with export_section_cols[0]:
    config_filename = st.text_input(
        "JSON config filename",
        value="",
        placeholder="Leave blank for auto-generated name",
        help="Optional: specify a custom filename for the exported config (without .json extension)",
        disabled=not can_build_summary or summary_df is None,
    )

export_disabled = not can_build_summary or summary_df is None
with export_section_cols[1]:
    if st.button("Export JSON Config", disabled=export_disabled, use_container_width=True):
        exported_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        data_filters = {
            "player_ids": resolved_players,
            "seasons": selected_seasons_for_config,
            "shot_groups": selected_shot_groups,
        }
        if enable_partition_filter and partition_column:
            data_filters["partition_column"] = partition_column
            data_filters["partition_values"] = partition_values

        config = {
            "config_version": 1,
            "created_at": exported_at,
            "data_root": data_root,
            "data_filters": data_filters,
            "estimator": {
                "per_season": bool(per_season_estimation),
                "num_execution_skills": int(num_execution_skills),
                "num_planning_skills": int(num_planning_skills),
                "rng_seed": int(rng_seed),
                "save_intermediate_csv": bool(save_intermediate_csv),
                "generate_convergence_png": bool(generate_convergence_png),
                "convergence_png_include_map": bool(png_include_map),
            },
            "output": {
                "write_run_summary": bool(write_run_summary),
            },
            "validation": {
                "min_shots_per_job": int(min_shots_per_job),
                "fail_policy": "skip",
            },
            "cluster_plan": {
                "split_mode": estimation_mode,
                "total_jobs": len(jobs_preview),
                "eligible_jobs": sum(1 for j in jobs_preview if j["eligible"]),
                "sbatch_recommendation": {
                    "time": sbatch_time,
                    "mem": sbatch_mem,
                    "max_concurrent": int(sbatch_max_concurrent),
                },
                "jobs": jobs_preview,
            },
        }
        out_path = data_io.save_job_config(
            config,
            data_dir=data_root,
            output_subdir="jobs",
            custom_filename=config_filename if config_filename.strip() else None,
        )
        st.success(f"Saved config: {out_path}")
        with st.expander("View exported config JSON", expanded=False):
            st.json(config)
