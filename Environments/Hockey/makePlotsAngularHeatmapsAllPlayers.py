from makePlotsAngularHeatmaps import makePlots
import json,sys,os


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

		print(f"Creating plots for Player {playerID} | Shot Type: {typeShot}")

		makePlots(experimentFolder,playerID,typeShot)

