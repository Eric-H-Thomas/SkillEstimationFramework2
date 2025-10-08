import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os,sys,code
import json,pickle

import urllib.request
from hockey_rink import NHLRink


def makePlot(path,experimentFolder,playerID,typeShot,label):


	rink = NHLRink()

	# ax = rink.draw(figsize=(10, 4))


	#  bdc_url = (
	#      "https://raw.githubusercontent.com/the-bucketless/bdc/refs/heads/main/data/"
	#      "pxp_womens_oly_2022_v2.csv"
	#  )

	#  bdc_pbp = pd.read_csv(bdc_url)


	#  finland_pp_passes = (
	#     bdc_pbp
	#     .query(
	#         "team_name == 'Olympic (Women) - Finland'"
	#         " and event == 'Play'"
	#         " and situation_type == '5 on 4'"
	#     )
	#     .dropna(subset=["frame_id_1"])
	# )

	xs = []
	ys = []

	xs2 = []
	ys2 = []


	for row in data:

		playerLocation = [data[row]["start_x"],data[row]["start_y"]]
		# projectedZ = data[row]["projected_z"]
		# shot_location = final_y, projected_z, start_x, start_y
		# executedAction = [data[row]["shot_location"][0],data[row]["shot_location"][1]]
		executedAction = [89,data[row]["shot_location"][0]]


		# Start = base of arrows
		xs.append(playerLocation[0])
		ys.append(playerLocation[1])


		# End = endpoint of arrows
		xs2.append(executedAction[0])
		ys2.append(executedAction[1])



	# x: array-like or key in data
	#     The x-coordinates of the base of the arrows.

	# y: array-like or key in data
	#     The y-coordinates of the base of the arrows.

	# dx: array-like or key in data (optional)
	#     The length of the arrows in the x direction.
	#     One of dx and x2 has to be specified.

	# dy: array-like or key in data (optional)
	#     The length of the arrows in the y direction.
	#     One of dy and y2 has to be specified.

	# x2: array-like or key in data (optional)
	#     The endpoint of the arrows.
	#     One of dx and x2 has to be specified.

	# y2: array-like or key in data (optional)
	#     The endpoint of the arrows.
	#     One of dy and y2 has to be specified.

	rink.arrow(
	x=xs, y=ys,
	x2=xs2, y2=ys2,
	facecolor="purple", edgecolor="black",
	head_width=4, length_includes_head=True,
	draw_kw={"figsize": (10, 4)},
	)

	plt.savefig(f"{path}{playerID}-{typeShot}{label}-{len(data)}.png")
	plt.clf()
	plt.close()


	# code.interact("...", local=dict(globals(), **locals()))



if __name__ == '__main__':
	 
	try:
		experimentFolder = sys.argv[1]
	except:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}"

	try:
		with open(f"{mainFolder}Data{os.sep}JSON{os.sep}statsAfterFiltering.json","r") as infile:
			info = json.load(infile)
	except Exception as e:
		print(e,"Need to create statsAfterFiltering.json file. Run getStatsJson.py script first.")
		exit()


	for each in info:
		playerID = each[0]
		typeShot = each[1]

		print(f"Creating plot for Player {playerID} | Shot Type: {typeShot}...")


		for i, folder in enumerate([f"{mainFolder}Data{os.sep}AngularHeatmaps{os.sep}",f"{mainFolder}Data{os.sep}AngularHeatmaps-Filtered{os.sep}"]):
	
			if i == 0:
				fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"
				label = ""
			else:

				label = "-filtered"

				files = os.listdir(folder)

				for fileName in files:
					if playerID in fileName and typeShot in fileName:
						break


			try:
				with open(folder+fileName,"rb") as infile:
					data = pickle.load(infile)
			except Exception as e:
				print(e)
				print("Can't load data for that player.")
				exit()


			path = f"Testing{os.sep}Rinks{os.sep}"

			if not os.path.exists(path):
					os.mkdir(path)


			makePlot(path,experimentFolder,playerID,typeShot,label)


	# - from overhead of the rink
	# - for each shot, plot line from where they start to where they cross the goal
	
	# Color by filtered, crazy update, etc