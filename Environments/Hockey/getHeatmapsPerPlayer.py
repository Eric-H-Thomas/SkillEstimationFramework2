import pickle
import code
import os,sys


if __name__ == '__main__':
	

	try:
		experimentFolder = sys.argv[1]
	except Exception as e:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument. (Make sure the data (.pkl) file is located inside such folder too.)")
		exit()



	folder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"
	fileName = f"heatmap_data.pkl"

	try:
		with open(folder+fileName,"rb") as infile:
			data = pickle.load(infile)
	except Exception as e:
		print(e)
		print("File with data not present. Please place the data (.pkl) file inside the folder for the experiment.")
		exit()


	if not os.path.exists(f"{folder}Heatmaps{os.sep}"):
		os.mkdir(f"{folder}Heatmaps{os.sep}")


	playerData = {}

	typeShots = []


	# Group data by player ID
	for row in data:

		pid = data[row]["shooter_id"]

		if pid not in playerData:
			playerData[pid] = {}


		typeShot = data[row]["shot_type"]

		if typeShot not in typeShots:
			typeShots.append(typeShot)

		if typeShot not in playerData[pid]:
			playerData[pid][typeShot] = {}


		if row not in playerData[pid][typeShot]:
			playerData[pid][typeShot][row] = {}

		playerData[pid][typeShot][row] = data[row]


	# Create new pickle file per player ID
	for eachID in playerData:		
		for typeShot in playerData[eachID]:
			with open(f"{folder}Heatmaps{os.sep}heatmap_data_player_{eachID}_type_shot_{typeShot}.pkl","wb") as outfile:
				pickle.dump(playerData[eachID][typeShot],outfile)


	with open(f"{folder}stats-AllData.txt","w") as outfile:
		print(f"Total number of players: {len(playerData)}",file=outfile)
		print(file=outfile)

		for eachID in playerData:
			print(f"Player: {eachID}",file=outfile)

			for st in typeShots:
				if st in playerData[eachID]:
					print(f"\t{st}: {len(playerData[eachID][st])}",file=outfile)


	# code.interact("...", local=dict(globals(), **locals()))


