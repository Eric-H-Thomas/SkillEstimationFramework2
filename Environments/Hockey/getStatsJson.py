import os,json,code
import pickle, sys


if __name__ == '__main__':

	try:
		experimentFolder = sys.argv[1]
	except Exception as e:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()

	
	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"

	
	folder = f"{mainFolder}AngularHeatmaps{os.sep}"
	files = os.listdir(folder)

	mainFolder += f"JSON{os.sep}"

	if not os.path.exists(mainFolder):
		os.mkdir(mainFolder)

	
	statsInfo = {}
	typeShots = []


	for eachFile in files:

		if "player" in eachFile:

			splitted = eachFile.split("_")
			playerID = splitted[4]
			typeShot = splitted[-1].split(".")[0]
			# code.interact("...", local=dict(globals(), **locals()))	


			if playerID not in statsInfo:
				statsInfo[playerID] = {}

			if typeShot not in statsInfo[playerID]:
				statsInfo[playerID][typeShot] = 0

			if typeShot not in typeShots:
				typeShots.append(typeShot)


			fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"


			with open(folder+fileName,"rb") as infile:
				data = pickle.load(infile)


			statsInfo[playerID][typeShot] = len(data)


	statsInfoFlat = []


	for eachID in statsInfo:
		for st in typeShots:
			if st in statsInfo[eachID]:
				statsInfoFlat.append([eachID,st,statsInfo[eachID][st]])


	with open(f"{mainFolder}statsAfterFiltering.json","w") as outfile:
		json.dump(statsInfoFlat,outfile)
		

	playerIDs = list(statsInfo.keys())
	playerIDs = list(map(lambda x: int(x),playerIDs))

	with open(f"{mainFolder}playerIDs.json","w") as outfile:
		json.dump(playerIDs,outfile)


	typeShots = ["snapshot","wristshot"]

	with open(f"{mainFolder}typeShots.json","w") as outfile:
		json.dump(typeShots,outfile)

	# code.interact("...", local=dict(globals(), **locals()))	
	print("Done.")


