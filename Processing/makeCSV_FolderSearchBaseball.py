import argparse,sys,os
import json,pickle
import numpy as np
import pandas as pd
import code,csv

from utils import *


from pybaseball import playerid_reverse_lookup, cache


def makeCSV(resultsDict,methods,label,resultsFolder):
	
	makeFolder2(resultsFolder,"CSV")

	finalEstimates = []

	for pitcherID in resultsDict.keys():

		for pitchType in resultsDict[pitcherID]:

			estimatesAcrossMethods = []
			
			numObs = resultsDict[pitcherID][pitchType]["numObservations"]

			for method in methods:

				est = None

				if "BM" in method:  
					tempM, beta, tt = getInfoBM(method)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
						est = resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta]
				elif "BM" not in method:
					if resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
						est = resultsDict[pitcherID][pitchType]["estimates"][method]
						
				if est != None:
					estimatesAcrossMethods.append(est)

			finalEstimates.append([pitcherID,pitcherNames[pitcherID],pitchType,numObs] + estimatesAcrossMethods)


	saveTo = open(f"{resultsFolder}CSV{os.sep}finalEstimatesInfo-{label}.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = ["ID","Name","PitchType","NumObservations"] + methods

	csvWriter.writerow(columns)
	
	for i in range(len(finalEstimates)):
		csvWriter.writerow(finalEstimates[i])

	saveTo.close()

	# code.interact("...", local=dict(globals(), **locals()))


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Process baseball results files from experiments to create CSV')
	parser.add_argument("-folderContains", dest = "folderContains", help = "Name of folder containing the results of the experiments", type = str, default = "ForPaper-Experiment")
	args = parser.parse_args()


	experimentFolder = f"..{os.sep}Experiments{os.sep}baseball{os.sep}"

	# From Experiments folder
	folders = os.listdir(experimentFolder)


	expFolders = []

	# Filter experiments
	# Assumes folders with actual experiments have given string in their name
	# To ignore testing folders
	for f in folders:
		if args.folderContains in f:
			expFolders.append(f)


	resultsDict = {}
	pitcherNames = {}

	seenRFs = []
	methods = None


	if Path(f"{experimentFolder}CSV{os.sep}resultsDictInfo.pkl").is_file():
		print("Loading previously gathered info...")

		with open(f"{experimentFolder}CSV{os.sep}resultsDictInfo.pkl",'rb') as handle:
			resultsDict = pickle.load(handle)

		with open(f"{experimentFolder}CSV{os.sep}otherInfo.pkl",'rb') as handle:
			info = pickle.load(handle)

		pitcherNames = info["pitcherNames"]
		methods = info["methods"]
		seenRFs = info["seenRFs"]


	print("Accessing results files...")

	
	for f in expFolders:

		# Load info from folder
		subfiles = os.listdir(f"{experimentFolder}{f}{os.sep}results")

		for sf in subfiles:
			
			if ".results" not in sf:
				continue

			print(f"Looking at: {sf}")

			splitted = sf.split(".results")[0].split("_")
			pitcherID = splitted[1]
			pitchType = splitted[2]

			if sf in seenRFs and resultsDict[pitcherID][pitchType]["valid"]:
				print("Results info already available.")
				continue

			else:

				# if result file not already present on folder with compiled result files
				# And it's an actual results file (to skip backup folder if present)
				if ".results" in sf and "temp" not in sf:
	
					print("Getting results info...")
					seenRFs.append(sf)

					rf = f"{experimentFolder}{f}{os.sep}results{os.sep}{sf}"

					with open(rf) as infile:

						results = json.load(infile)

						# To initialize methods list once
						if methods == None:

							methods = []

							for m in results.keys():

								# SKIPPING NJT METHODS
								if "NJT" in m:
									continue

								if (not m.isalpha()) and "-" in m and "allProbs" not in m:
									methods.append(m)


						agent = results["agent_name"]
						pitcherID = agent[0]
						pitchType = agent[1]
						
						numObservations = results["numObservations"]


						if pitcherID not in resultsDict:
							resultsDict[pitcherID] = {}

						if pitchType not in resultsDict[pitcherID]:
							resultsDict[pitcherID][pitchType] = {}


						resultsDict[pitcherID][pitchType] = {"estimates": {},"numObservations":numObservations,"valid":False}

						totalValid = 0

						for m in methods:

							try:
								validCount = False

								# if the method exists on the results file, load
								testLoadMethod = results[m]

								if len(testLoadMethod) == numObservations:
									validCount = True
									totalValid += 1

							except:
								print(f"\t\t{m} - not present")
								# code.interact("...", local=dict(globals(), **locals()))
								continue


							# If TBA/BM method, need to account for possible different betas
							if "BM" in m:
								tempM, beta, tt = getInfoBM(m)

								# To initialize once
								if tt not in resultsDict[pitcherID][pitchType]["estimates"]:
									resultsDict[pitcherID][pitchType]["estimates"][tt] = {}

								if tempM not in resultsDict[pitcherID][pitchType]["estimates"][tt]:
									resultsDict[pitcherID][pitchType]["estimates"][tt][tempM] = {}

								if beta not in resultsDict[pitcherID][pitchType]["estimates"][tt][tempM]:
									resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = [0.0] * numObservations
									
								# Save final estimate
								if validCount:
									resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = results[m][-1]
								else:
									resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = {}

							else:
								# Save final estimates
								# Won't add info for method if there's a mismatch between
								# expected # of observations and the number of estimates produced
								if validCount:
									resultsDict[pitcherID][pitchType]["estimates"][m] = results[m][-1]
								else:
									resultsDict[pitcherID][pitchType]["estimates"][m] = {}


						# If exps has all required info, mark as valid
						if totalValid == len(methods):
							resultsDict[pitcherID][pitchType]["valid"] = True


	print(f"\nObtained information from {len(seenRFs)} results files.")


	#############################################################################
	# Store processed results
	#############################################################################

	makeFolder2(experimentFolder,"CSV")


	print("Saving info...",end=" ")

	# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
	with open(f"{experimentFolder}CSV{os.sep}resultsDictInfo.pkl","wb") as outfile:
		pickle.dump(resultsDict,outfile)


	otherInfo = {}
	otherInfo["methods"] = methods
	otherInfo["pitcherNames"] = pitcherNames
	otherInfo["seenRFs"] = seenRFs

	with open(f"{experimentFolder}CSV{os.sep}otherInfo.pkl","wb") as outfile:
		pickle.dump(otherInfo,outfile)

	print("Done.")

	#############################################################################


	pitcherNames = {}
	
	for pitcherID in resultsDict.keys():

		try:
			result = playerid_reverse_lookup([int(pitcherID)])[["name_first","name_last"]]
			pitcherNames[pitcherID] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"

		except:
			print("Error in playerid_reverse_lookup(). Loading info from json instead...\n")

			with open(f"..{os.sep}Experiments{os.sep}baseball{os.sep}pitcherNames.json","r") as infile:
				names = json.load(infile)

			tempName = names[pitcherID].split(", ")
			pitcherNames[pitcherID] = f"{tempName[1]} {tempName[0]}"



	print(f"Creating CSV files...",end=" ")
	

	forCSV = ['OR-66','BM-MAP-66-Beta-0.5', 'BM-EES-66-Beta-0.5',
			'BM-MAP-66-Beta-0.95', 'BM-EES-66-Beta-0.95',
			'BM-MAP-66-Beta-0.99', 'BM-EES-66-Beta-0.99',
			'JT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-xSkills',
			'JT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-xSkills',
			'JT-QRE-MAP-66-66-xSkills','JT-QRE-EES-66-66-xSkills']

	
	label = "AllMethods"
	makeCSV(resultsDict,methods,label,experimentFolder)

	label = "SelectedMethods"
	makeCSV(resultsDict,forCSV,label,experimentFolder)


	print("Done.")
	
	# code.interact("...", local=dict(globals(), **locals()))

