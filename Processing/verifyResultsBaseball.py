import argparse
import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
from pybaseball import playerid_reverse_lookup, cache
import code
import csv
import numpy as np
import json, pickle

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}verifyResultsBaseball.py")[0]

module = SourceFileLoader("dataTake2.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}dataTake2.py").load_module()
sys.modules["data"] = module


if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Validate experiments.')
	parser.add_argument('-file','-file', help='Name of the file that contains the IDs of the pitchers to use',default = "ids.txt")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "Testing")

	parser.add_argument("-startYear", dest = "startYear", help = "Desired start year.", type = str, default = "2021")
	parser.add_argument("-endYear", dest = "endYear", help = "Desired end year.", type = str, default = "2021")
	
	parser.add_argument("-startMonth", dest = "startMonth", help = "Desired start month.", type = str, default = "01")
	parser.add_argument("-endMonth", dest = "endMonth", help = "Desired end month.", type = str, default = "12")
	
	parser.add_argument("-startDay", dest = "startDay", help = "Desired start day.", type = str, default = "01")
	parser.add_argument("-endDay", dest = "endDay", help = "Desired end day.", type = str, default = "31")
	
	args = parser.parse_args()


	resultsFolder = f"Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}"

	try:
		with open(f"{resultsFolder}{args.file}","r") as inFile:
			pitcherIDs = inFile.readlines()
	except Exception as e:
		print(e)
		print("Couldn't open file. Please make sure file exists.")
		exit()


	processedDataFolder = f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"

	csvFiles = [f"{processedDataFolder}raw22.csv",f"{processedDataFolder}raw21.csv",f"{processedDataFolder}raw18_19_20.csv"]

	data,batterIndices,standardizer = sys.modules["data"].manageData(csvFiles)


	allPitchers = data.pitcher.unique()
	allPitchTypes = data.pitch_type.unique()


	print("\nFinding info for given pitchers...")

	# Convert pitcher IDs to numbers
	for i in range(len(pitcherIDs)):
		pitcherIDs[i] = int(pitcherIDs[i].strip())


	# Get pitcher's first name and last name based on IDs
	info = playerid_reverse_lookup(pitcherIDs)



	###########################################################################
	# Statcast data info
	###########################################################################

	pitchersInfo = {}

	
	saveTo = open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}pitchersInfo.txt","w")

	for pid in pitcherIDs:

		temp = info.loc[info.key_mlbam == pid]

		name = f"{temp.name_first.values[0].capitalize()} {temp.name_last.values[0].capitalize()}"

		pitchersInfo[pid] = {"name": name, "rowCount": {}}

		saveTo.write(f"Pitcher: {name} | ID: {pid} | ")


		for pitchType in allPitchTypes:
			count = len(data.query(f"pitcher == {pid} and pitch_type == '{pitchType}'" ,engine="python"))
			saveTo.write(f"{pitchType}: {count} | ")

			pitchersInfo[pid]["rowCount"][pitchType] = count

		saveTo.write("\n")

			
	saveTo.close()

	###########################################################################
	

	###########################################################################
	# Info from results
	###########################################################################

	print("\nGetting info from results...")


	resultsInfo = {}

	allResultsFiles = os.listdir(f"{resultsFolder}results")


	# Find methods used on experiments
	# Assumes all results files are using the same set of estimators
	methods = []

	with open(f"{resultsFolder}results{os.sep}{allResultsFiles[0]}") as infile:
		results = json.load(infile)

		for m in results.keys():
			if (not m.isalpha()) and "-" in m and "allProbs" not in m:
				methods.append(m)

	
	for eachRF in allResultsFiles:

		rfID = int(eachRF.split("_")[1])
		rfPT = eachRF.split("_")[2].split(".")[0]

		if rfID not in resultsInfo:
			resultsInfo[rfID] = {}

		if rfPT not in resultsInfo[rfID]:
			resultsInfo[rfID][rfPT] = {"expCount":0,"methods":{},"allProbs:":{}}

		resultsInfo[rfID][rfPT]["expCount"] += 1


		# Load info from results file
		with open(f"{resultsFolder}results{os.sep}{eachRF}") as infile:
			results = json.load(infile)

			for m in methods:

				# Get number of estimates for estimators
				try:
					resultsInfo[rfID][rfPT]["methods"][m] = len(results[m])
				except:
					# No estimates present for this method
					resultsInfo[rfID][rfPT]["methods"][m] = {}

			tempMethodsAllProbs = []

			for m in results.keys():

				if "allProbs" in m:
					tempMethodsAllProbs.append(m)

			resultsInfo[rfID][rfPT]["allProbs"] = tempMethodsAllProbs


	###########################################################################


	###########################################################################
	# Comparison between statcast & results data - experiments status in general
	###########################################################################

	print("\nGetting experiment stats...")

	stats = []
	ids = []

	# Verify there are results files for all given IDs 
	# with each possible pitch type (if info present)
	for pid in pitchersInfo:

		temp = []

		for eachPT in allPitchTypes:

			# If expected to have experiments for this pitcher-pitch type combination
			if pitchersInfo[pid]["rowCount"][eachPT] != 0:

				if pid not in resultsInfo:
					# print(f"No experiments found for ID = {pid}.")
					temp.append(0)
				elif eachPT not in resultsInfo[pid]:
					# print(f"No experiments found for ID = {pid} - Pitch Type = {eachPT}.")
					temp.append(0)
				else:
					if resultsInfo[pid][eachPT]["expCount"] > 1:
						# print(f"There are {resultsInfo[pid][eachPT]['expCount']} experiments for  ID = {pid} - Pitch Type = {eachPT}.")
						temp.append(resultsInfo[pid][eachPT]["expCount"])
					else:
						temp.append(1)

			# Not expected since no info present
			else:
				temp.append(-1)

		stats.append(temp)
		ids.append(pid)


	# 0 = not present
	# 1 = present
	# -1 = not expected
	# >1 = how many available

	allExpected = len(allPitchTypes)

	stats = np.array(stats)

	otherInfo = []

	for i in range(len(stats)):

		# How many experiments are expected?
		expected = allExpected-int(np.count_nonzero(stats[i] == -1))
		
		# How many are available?
		available = int(np.count_nonzero(stats[i] == 1))

		status = ""
		
		if available == 0:
			status = "None"
		elif available == expected:
			status = "All"
		elif available < expected:
			status = "Partial"
		else:
			status = "Over"

		otherInfo.append([expected,available,status])


	ids = np.array(ids)
	ids = ids.reshape(len(ids),1)

	otherInfo = np.array(otherInfo)
	allStats = np.hstack((ids,stats,otherInfo))

	
	saveTo = open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}StatsExperiments.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = np.insert(allPitchTypes,0,"ID")
	columns = np.insert(columns,len(columns),["Expected","Available","Status"])

	csvWriter.writerow(columns)
	
	for i in range(len(allStats)):
		csvWriter.writerow(allStats[i])

	saveTo.close()
	
	###########################################################################
	

	###########################################################################
	# Comparison between statcast & results data 
	# Verify # of observations - for each method
	###########################################################################
	
	print("\nGetting methods stats...")

	statsMethods = []
	ids = []

	for pid in pitchersInfo:

		for eachPT in allPitchTypes:

			if pid not in resultsInfo:
				statsMethods.append([0]*len(methods))
				ids.append([pid,eachPT])

			else:

				# If expected to have experiments for this pitcher-pitch type combination
				if pitchersInfo[pid]["rowCount"][eachPT] != 0:

					# If seen an experiment for this pitch type
					if eachPT in resultsInfo[pid]:

						temp = []

						for m in methods:

							if resultsInfo[pid][eachPT]["methods"][m] == {}:
								temp.append(0)
							elif resultsInfo[pid][eachPT]["methods"][m] == pitchersInfo[pid]["rowCount"][eachPT]:
								temp.append(1)
							elif resultsInfo[pid][eachPT]["methods"][m] < pitchersInfo[pid]["rowCount"][eachPT]:
								diff = pitchersInfo[pid]["rowCount"][eachPT] - resultsInfo[pid][eachPT]["methods"][m]
								temp.append(diff*-1)
							else:
								temp.append(resultsInfo[pid][eachPT]["methods"][m])

						statsMethods.append(temp)

					else:
						statsMethods.append([0]*len(methods))		

					ids.append([pid,eachPT])


	# 0 = not present
	# 1 = right amount
	# -1 = not expected
	# <1 = less than expected (difference of how many missing)(
	# 	if all methods missing expected, results file might be initial info only
	# >1 = more than expected

	statsMethods = np.array(statsMethods)

	allExpected = len(methods)

	otherInfo = []

	for i in range(len(statsMethods)):
		
		# How many are available?
		available = int(np.count_nonzero(statsMethods[i] == 1))

		status = ""
		
		if available == 0:
			status = "None"
		elif available == allExpected:
			status = "All"
		elif available < allExpected:
			status = "Partial"
		else:
			status = "Over"

		pid = ids[i][0]
		eachPT = ids[i][1]
		otherInfo.append([available,status,pitchersInfo[pid]["rowCount"][eachPT]])


	ids = np.array(ids)

	otherInfo = np.array(otherInfo)
	
	allStatsMethods = np.hstack((ids,otherInfo,statsMethods))

	saveTo = open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}StatsMethods.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = ["ID","PitchType","Available","Status","RowCount"] + methods

	csvWriter.writerow(columns)
	
	for i in range(len(allStatsMethods)):
		csvWriter.writerow(allStatsMethods[i])

	saveTo.close()


	# Find which experiments we don't have any information
	missingExps = []

	for e in range(len(otherInfo)):
		if int(otherInfo[e][0]) == 0:
			missingExps.append([int(ids[e][0]),ids[e][1]])

	with open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}IDsMissingExps-Total{len(missingExps)}.json","w") as outfile:
		json.dump({"missingExps":missingExps},outfile)

	###########################################################################
	

	###########################################################################
	# Verify all probs
	###########################################################################

	allProbsInfo = []

	for pid in resultsInfo:

		for eachPT in resultsInfo[pid]:

			# Previous version, missing allProbs for diff betas and prev JT naming convention
			if "BM-66-allProbs" in resultsInfo[pid][eachPT]["allProbs"]:
				allProbsInfo.append([pid,eachPT,-1])
			else:
				allProbsInfo.append([pid,eachPT,1])

	# 1 = ok
	# -1 = prev version, missing all probs for betas and prev naming convention for JT

	saveTo = open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}StatsAllProbs.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = ["ID","PitchType","AllProbs"]

	csvWriter.writerow(columns)
	
	for i in range(len(allProbsInfo)):
		csvWriter.writerow(allProbsInfo[i])

	saveTo.close()


	###########################################################################
	

	###########################################################################
	# Save info to file
	###########################################################################	

	infoToSave = {}

	infoToSave["allPitchers"] = allPitchers
	infoToSave["allPitchTypes"] = allPitchTypes
	infoToSave["pitchersInfo"] = pitchersInfo
	infoToSave["pitchersInfo"] = pitchersInfo
	infoToSave["resultsInfo"] = resultsInfo
	infoToSave["allStats"] = allStats
	infoToSave["allStatsMethods"] = allStatsMethods
	infoToSave["allProbsInfo"] = allProbsInfo

	with open(f"{mainFolderName}Experiments{os.sep}baseball{os.sep}{args.resultsFolder}{os.sep}info.pkl",'wb') as handle:
		pickle.dump(infoToSave,handle)

	###########################################################################	


	code.interact("...", local=dict(globals(), **locals()))

