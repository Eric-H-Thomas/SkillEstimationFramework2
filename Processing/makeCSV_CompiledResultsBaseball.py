import argparse,sys,os
import json,pickle
import numpy as np
import pandas as pd
import code,csv

# from utils import *
from pathlib import Path

from pybaseball import playerid_reverse_lookup, cache



def makeFolder2(resultsFolder,folderName):

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(resultsFolder  + os.sep + folderName):
		os.mkdir(resultsFolder + os.sep + folderName)


def makeCSV(resultsDict,methods,label,resultsFolder):
	
	makeFolder2(resultsFolder,"CSV")

	finalEstimates = []

	for pitcherID in resultsDict.keys():

		for pitchType in resultsDict[pitcherID]:

			for chunk in resultsDict[pitcherID][pitchType]:

				estimatesAcrossMethods = []
				
				numObs = resultsDict[pitcherID][pitchType][chunk]["numObservations"]

				for method in methods:

					est = None

					if "BM" in method:  
						tempM, beta, tt = getInfoBM(method)
						if resultsDict[pitcherID][pitchType][chunk]["estimates"][tt][tempM][beta] != {}:						
							est = resultsDict[pitcherID][pitchType][chunk]["estimates"][tt][tempM][beta]
					elif "BM" not in method:						
						if resultsDict[pitcherID][pitchType][chunk]["estimates"][method] != {}:
							est = resultsDict[pitcherID][pitchType][chunk]["estimates"][method]
							
					if est != None:
						estimatesAcrossMethods.append(est)

				finalEstimates.append([pitcherID,pitcherNames[pitcherID],pitchType,chunk,numObs] + estimatesAcrossMethods)


	saveTo = open(f"{resultsFolder}CSV{os.sep}finalEstimatesInfo-{label}.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = ["ID","Name","PitchType","Chunk","NumObservations"] + methods

	csvWriter.writerow(columns)
	
	for i in range(len(finalEstimates)):
		csvWriter.writerow(finalEstimates[i])

	saveTo.close()

	# code.interact("...", local=dict(globals(), **locals()))


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Process baseball results files from experiments to create CSV')
	parser.add_argument("-compiledResultsFolder", dest = "compiledResultsFolder", help = "Name of folder containing the results of the experiments (already compiled)", type = str, default = "ForPaper-Experiment")
	args = parser.parse_args()


	if args.compiledResultsFolder[-1] != os.sep:
		args.compiledResultsFolder += os.sep


	experimentFolder = f"Experiments{os.sep}baseball-multi{os.sep}"
	rfsFolder = f"{experimentFolder}{args.compiledResultsFolder}"


	resultsDict = {}
	pitcherNames = {}

	seenRFs = []
	methods = None


	resultFiles = os.listdir(f"{rfsFolder}results")


	if Path(f"{rfsFolder}CSV{os.sep}resultsDictInfo.pkl").is_file():
		print("Loading previously gathered info...")

		with open(f"{rfsFolder}CSV{os.sep}resultsDictInfo.pkl",'rb') as handle:
			resultsDict = pickle.load(handle)

		with open(f"{rfsFolder}CSV{os.sep}otherInfo.pkl",'rb') as handle:
			info = pickle.load(handle)

		pitcherNames = info["pitcherNames"]
		methods = info["methods"]
		seenRFs = info["seenRFs"]


	print("Accessing results files...")

	# print(resultFiles)

	for sf in resultFiles:
		
		if ".results" not in sf:
			continue

		print(f"Looking at: {sf}")

		splitted = sf.split(".results")[0].split("_")
		pitcherID = splitted[1]
		pitchType = splitted[2]
		chunk = splitted[4]

		if sf in seenRFs and resultsDict[pitcherID][pitchType][chunk]["valid"]:
			print("Results info already available.")
			continue

		else:

			# if result file not already present on folder with compiled result files
			# And it's an actual results file (to skip backup folder if present)
			if ".results" in sf and "temp" not in sf:

				print("Getting results info...")
				seenRFs.append(sf)

				rf = f"{rfsFolder}{os.sep}results{os.sep}{sf}"

				with open(rf,"rb") as infile:

					results = pickle.load(infile)

					# To initialize methods list once
					if methods == None:

						methods = []

						for m in results.keys():

							# SKIPPING NJT METHODS
							if "NJT" in m:
								continue

							if (not m.isalpha()) and "-" in m and "allProbs" not in m and "whenResampled" not in m and "allParticles" not in m:
								methods.append(m)


					agent = results["agent_name"]
					pitcherID = agent[0]
					pitchType = agent[1]
					
					numObservations = results["numObservations"]


					if pitcherID not in resultsDict:
						resultsDict[pitcherID] = {}

					if pitchType not in resultsDict[pitcherID]:
						resultsDict[pitcherID][pitchType] = {}

					if chunk not in resultsDict[pitcherID][pitchType]:
						resultsDict[pitcherID][pitchType][chunk] = {}


					resultsDict[pitcherID][pitchType][chunk] = {"estimates": {},"numObservations":numObservations,"valid":False}

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
								resultsDict[pitcherID][pitchType][chunk]["estimates"][m] = results[m][-1]
							else:
								resultsDict[pitcherID][pitchType][chunk]["estimates"][m] = {}


					# If exps has all required info, mark as valid
					if totalValid == len(methods):
						resultsDict[pitcherID][pitchType][chunk]["valid"] = True


	print(f"\nObtained information from {len(seenRFs)} results files.")


	#############################################################################
	# Store processed results
	#############################################################################

	makeFolder2(rfsFolder,"CSV")


	print("Saving info...",end=" ")

	# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
	with open(f"{rfsFolder}CSV{os.sep}resultsDictInfo.pkl","wb") as outfile:
		pickle.dump(resultsDict,outfile)


	otherInfo = {}
	otherInfo["methods"] = methods
	otherInfo["pitcherNames"] = pitcherNames
	otherInfo["seenRFs"] = seenRFs

	with open(f"{rfsFolder}CSV{os.sep}otherInfo.pkl","wb") as outfile:
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
	

	forCSV = [
			# 'OR-66','BM-MAP-66-Beta-0.5', 'BM-EES-66-Beta-0.5',
			# 'BM-MAP-66-Beta-0.95', 'BM-EES-66-Beta-0.95',
			# 'BM-MAP-66-Beta-0.99', 'BM-EES-66-Beta-0.99',
			# 'JT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-xSkills',
			# 'JT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-xSkills',
			'JT-QRE-MAP-66-66-xSkills','JT-QRE-EES-66-66-xSkills']

	
	label = "AllMethods"
	makeCSV(resultsDict,methods,label,rfsFolder)

	label = "SelectedMethods"
	makeCSV(resultsDict,forCSV,label,rfsFolder)


	print("Done.")
	
	# code.interact("...", local=dict(globals(), **locals()))

