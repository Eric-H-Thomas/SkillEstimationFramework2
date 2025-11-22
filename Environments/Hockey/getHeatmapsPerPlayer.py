"""Split aggregated hockey heatmap data into per-player/shot-type outputs.

This script is a small data-preparation step in the hockey simulation pipeline
used throughout the repository. Experiments generate a single
``heatmap_data.pkl`` file that aggregates shot records for every player. This
utility reshapes that file into per-player, per-shot-type pickles and writes a
text summary so downstream analysis or visualization code can read smaller,
focused datasets from ``Experiments/hockey-multi/<experiment>/Data/Heatmaps``.

Functions in this module
------------------------
* ``_load_data_file`` – open the aggregated pickle and return its contents.
* ``_ensure_heatmap_dir`` – create the ``Heatmaps`` output directory if needed.
* ``_group_shots_by_player`` – organize rows by player identifier and shot
  type, returning both the grouped data and the list of shot types observed.
* ``_write_player_pickles`` – emit one pickle file for each player/shot-type
  combination.
* ``_write_summary_stats`` – generate a text file that reports counts per
  player and shot type.
* ``main`` – CLI entry point that wires the above helpers together based on the
  experiment folder passed via ``sys.argv``.
"""

import os
import pickle
import sys


def _load_data_file(folder: str, filename: str):
    """Load the aggregated heatmap pickle file."""
    # The master pickle file contains shots for all players.
    # Loading it upfront allows us to later split it by player/shot type.
    try:
        with open(os.path.join(folder, filename), "rb") as infile:
            return pickle.load(infile)
    except Exception as exc:  # pragma: no cover - passthrough error reporting
        print(exc)
        print(
            "File with data not present. Please place the data (.pkl) file inside the folder for the experiment."
        )
        sys.exit()


def _ensure_heatmap_dir(folder: str) -> str:
    """Create the Heatmaps/ directory inside the experiment folder if absent."""

    # Files for each player are emitted to Experiments/.../Data/Heatmaps/.
    # Centralizing the directory creation avoids repetition.
    heatmap_dir = os.path.join(folder, "Heatmaps")
    if not os.path.exists(heatmap_dir):
        os.mkdir(heatmap_dir)
    return heatmap_dir


def _group_shots_by_player(data):
    """Group rows by shooter and shot type."""

    player_data = {}
    shot_types = []

    for row in data:
        # Each row is keyed by a unique ID; its contents specify the player and
        # the type of shot they took.
        pid = data[row]["shooter_id"]
        type_shot = data[row]["shot_type"]

        if pid not in player_data:
            player_data[pid] = {}
        if type_shot not in player_data[pid]:
            player_data[pid][type_shot] = {}

        player_data[pid][type_shot][row] = data[row]

        if type_shot not in shot_types:
            shot_types.append(type_shot)

    return player_data, shot_types


def _write_player_pickles(folder: str, player_data: dict):
    """Write one pickle file per (player, shot type) combination."""

    heatmap_dir = _ensure_heatmap_dir(folder)
    for player_id, shots in player_data.items():
        for shot_type, entries in shots.items():
            # File names encode both identifiers so downstream scripts can
            # locate the specific dataset they require.
            file_name = f"heatmap_data_player_{player_id}_type_shot_{shot_type}.pkl"
            with open(os.path.join(heatmap_dir, file_name), "wb") as outfile:
                pickle.dump(entries, outfile)


def _write_summary_stats(folder: str, player_data: dict, shot_types: list):
    """Generate a text file with counts per player and shot type."""

    stats_path = os.path.join(folder, "stats-AllData.txt")
    with open(stats_path, "w") as outfile:
        print(f"Total number of players: {len(player_data)}", file=outfile)
        print(file=outfile)

        for player_id in player_data:
            print(f"Player: {player_id}", file=outfile)
            for shot_type in shot_types:
                # Only print counts for shot types recorded for the player.
                if shot_type in player_data[player_id]:
                    count = len(player_data[player_id][shot_type])
                    print(f"\t{shot_type}: {count}", file=outfile)


def main():
    """Entry point when executing the module as a script."""

    try:
        experiment_folder = sys.argv[1]
    except Exception:  # pragma: no cover - command line validation only
        print(
            "Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument. (Make sure the data (.pkl) file is located inside such folder too.)"
        )
        sys.exit()

    folder = os.path.join("Experiments", "hockey-multi", experiment_folder, "Data")
    file_name = "heatmap_data.pkl"

    data = _load_data_file(folder, file_name)
    player_data, shot_types = _group_shots_by_player(data)
    _write_player_pickles(folder, player_data)
    _write_summary_stats(folder, player_data, shot_types)


if __name__ == "__main__":
    main()
