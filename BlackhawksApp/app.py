"""Streamlit app for Blackhawks JEEDS visualizations.

This lightweight scaffold demonstrates how to import the refactored
plotting functions and display them in Streamlit.

Usage:
    pip install -r requirements.txt
    streamlit run BlackhawksApp/app.py
"""
from __future__ import annotations

import streamlit as st

from BlackhawksApp import data_io
from BlackhawksSkillEstimation.blackhawks_plots import (
    plot_player_shots_from_loaded_data,
    plot_player_convergence,
)

st.title("Blackhawks JEEDS Visualizations")

@st.cache_data(show_spinner=False)
def _cached_players(data_root: str) -> list[int]:
    return data_io.get_players(data_root)


@st.cache_data(show_spinner=False)
def _cached_seasons(player_id: int, data_root: str) -> list[int]:
    return data_io.get_seasons(player_id=player_id, data_dir=data_root)


@st.cache_data(show_spinner=False)
def _cached_intermediate_csvs(player_id: int, data_root: str) -> list[str]:
    return [str(p) for p in data_io.get_intermediate_csvs(player_id=player_id, data_dir=data_root)]


@st.cache_resource(show_spinner=False)
def _cached_heatmap_blob(player_id: int, seasons: tuple[int, ...], data_root: str):
    # Kept for loader/caching preflight and future app pages.
    return data_io.load_heatmaps(player_id=player_id, seasons=list(seasons), data_dir=data_root)


data_root = st.sidebar.text_input("Data root", value="Data/Hockey")
players = _cached_players(data_root)
if not players:
    st.error(f"No player folders found under: {data_root}")
    st.stop()

player_id = st.selectbox("Player ID", players, index=0)
seasons = _cached_seasons(player_id=player_id, data_root=data_root)
default_seasons = seasons[-1:] if seasons else []
selected_seasons = st.multiselect("Seasons", seasons, default=default_seasons)
max_shots = st.slider("Max heatmaps", min_value=1, max_value=50, value=6)

preloaded_df = None
preloaded_maps = None

if selected_seasons:
    # Warm/validate resource cache before plotting.
    try:
        preloaded_df, preloaded_maps = _cached_heatmap_blob(
            player_id=player_id,
            seasons=tuple(selected_seasons),
            data_root=data_root,
        )
        st.caption(f"Loaded {preloaded_df.shape[0]} shots and {len(preloaded_maps)} shot maps (cached).")
    except Exception as exc:
        st.warning(f"Preload failed: {exc}")

st.header("Player shots (offline)")
if st.button("Load shots and plots"):
    with st.spinner("Loading player data and rendering plots..."):
        try:
            if not selected_seasons:
                st.error("Select at least one season.")
                st.stop()

            if preloaded_df is not None and preloaded_maps is not None:
                df, shot_maps = preloaded_df, preloaded_maps
            else:
                df, shot_maps = _cached_heatmap_blob(
                    player_id=player_id,
                    seasons=tuple(selected_seasons),
                    data_root=data_root,
                )

            figs = plot_player_shots_from_loaded_data(
                player_id=player_id,
                df=df,
                shot_maps=shot_maps,
                seasons=selected_seasons,
                max_shots=max_shots,
            )
            if figs["angular"]:
                st.subheader("Angular heatmaps")
                for i, f in enumerate(figs["angular"]):
                    st.pyplot(f)
            if figs["rink"]:
                st.subheader("Rink")
                for f in figs["rink"]:
                    if f:
                        st.pyplot(f)
        except Exception as e:
            st.error(f"Failed to load or render: {e}")

st.header("Convergence / Intermediate estimates")
csv_candidates = _cached_intermediate_csvs(player_id=player_id, data_root=data_root)
csv_path = st.selectbox(
    "Intermediate CSV",
    options=[""] + csv_candidates,
    index=0,
)
if csv_path:
    if st.button("Render convergence"):
        try:
            fig = plot_player_convergence(csv_path, show=False)
            if fig:
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed: {e}")
