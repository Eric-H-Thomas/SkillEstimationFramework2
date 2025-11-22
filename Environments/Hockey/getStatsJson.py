"""Generate summary JSON files for hockey angular heatmap experiments.

This script scans the serialized angular heatmap data for a given hockey
experiment and produces lightweight JSON summaries used by downstream
visualization scripts (for example ``makePlotsAngularHeatmapsAllPlayers.py``).
It counts the number of shot attempts per player and shot type, records the
unique player IDs present in the data, and writes out a fixed list of shot
types expected by the plotting pipeline.

Key Outputs
-----------
``statsAfterFiltering.json``
	A flat list of ``[playerID, typeShot, count]`` entries. This drives batch
	plotting scripts by indicating which player/shot combinations have data.
``playerIDs.json``
	List of numeric player IDs encountered in the angular heatmap files.
``typeShots.json``
	Canonical list of shot types to consider (currently ``snapshot`` and
	``wristshot``).

Input Requirements
------------------
The script expects angular heatmap pickle files located at
``Experiments/hockey-multi/{experimentFolder}/Data/AngularHeatmaps/`` with the
naming convention ``angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl``.
These files are produced by ``getAngularHeatmapsPerPlayer.py`` during the
data-preparation phase.

Command-Line Usage
------------------
When run as a script, expects one argument:
	1. ``experimentFolder`` – Directory name under ``Experiments/hockey-multi/``

Example:
	``python getStatsJson.py my_experiment``
"""

import json
import os
import pickle
import sys


def main() -> None:
	"""Entry point for generating stats JSON files."""

	try:
		experimentFolder = sys.argv[1]
	except Exception:
		print(
			"Need to specify the name of the folder for the experiment (located under\n"
			"'Experiments/hockey-multi/') as command line argument."
		)
		sys.exit(1)

	data_root = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"
	angular_folder = f"{data_root}AngularHeatmaps{os.sep}"

	try:
		files = os.listdir(angular_folder)
	except FileNotFoundError:
		print(f"Cannot find angular heatmap folder: {angular_folder}")
		sys.exit(1)

	json_folder = f"{data_root}JSON{os.sep}"
	os.makedirs(json_folder, exist_ok=True)

	statsInfo = {}
	typeShots = []

	for eachFile in files:
		if "player" not in eachFile:
			continue

		splitted = eachFile.split("_")
		playerID = splitted[4]
		typeShot = splitted[-1].split(".")[0]

		if playerID not in statsInfo:
			statsInfo[playerID] = {}

		if typeShot not in statsInfo[playerID]:
			statsInfo[playerID][typeShot] = 0

		if typeShot not in typeShots:
			typeShots.append(typeShot)

		fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"

		with open(angular_folder + fileName, "rb") as infile:
			data = pickle.load(infile)

		statsInfo[playerID][typeShot] = len(data)

	statsInfoFlat = []

	for eachID in statsInfo:
		for st in typeShots:
			if st in statsInfo[eachID]:
				statsInfoFlat.append([eachID, st, statsInfo[eachID][st]])

	with open(f"{json_folder}statsAfterFiltering.json", "w") as outfile:
		json.dump(statsInfoFlat, outfile)

	playerIDs = list(statsInfo.keys())
	playerIDs = list(map(lambda x: int(x), playerIDs))

	with open(f"{json_folder}playerIDs.json", "w") as outfile:
		json.dump(playerIDs, outfile)

	typeShots = ["snapshot", "wristshot"]

	with open(f"{json_folder}typeShots.json", "w") as outfile:
		json.dump(typeShots, outfile)

	print("Done.")


if __name__ == "__main__":
	main()


