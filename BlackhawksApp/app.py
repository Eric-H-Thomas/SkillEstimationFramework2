"""Streamlit app for Blackhawks JEEDS visualizations.

This lightweight scaffold demonstrates how to import the refactored
plotting functions and display them in Streamlit.

Usage:
    pip install -r requirements.txt
    streamlit run BlackhawksApp/app.py
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

from BlackhawksSkillEstimation.BlackhawksJEEDS import load_player_data
from BlackhawksSkillEstimation.blackhawks_plots import (
    plot_player_shots_from_offline,
    plot_player_convergence,
)

st.title("Blackhawks JEEDS Visualizations")

player_id = st.number_input("Player ID", value=950160, step=1)

st.header("Player shots (offline)")
if st.button("Load shots and plots"):
    with st.spinner("Loading player data and rendering plots..."):
        try:
            figs = plot_player_shots_from_offline(player_id=player_id, seasons=[20242025], max_shots=6)
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
csv_path = st.text_input("Intermediate CSV path (or leave blank to skip)")
if csv_path:
    if st.button("Render convergence"):
        try:
            fig = plot_player_convergence(csv_path, show=False)
            if fig:
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed: {e}")
