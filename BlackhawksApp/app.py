"""Streamlit shot inspector for Blackhawks JEEDS data.

Usage:
    pip install -r requirements.txt
    streamlit run BlackhawksApp/app.py
"""
from __future__ import annotations

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
        "delta_expected_execution_skill",
        "delta_expected_rationality",
        "start_x",
        "start_y",
        "location_y",
        "location_z",
    ]
    if col in table_df.columns
]
table_col.dataframe(table_df[table_columns], width='stretch', height=320)

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
            if fig_rink is not None:
                visuals_cols = st.columns([1, 1])
                visuals_cols[0].pyplot(fig_rink)
                # TODO: Replace this placeholder with our custom covariance/error plot
                # driven by our own execution-error stdev model (not BH net_cov fields).
                visuals_cols[1].caption("TODO: custom covariance/error visualization (our model)")
        except Exception as exc:
            st.error(f"Failed to render selected-shot visuals: {exc}")
else:
    st.caption("Press 'Load visuals for selected shot' to render heatmaps and rink.")
