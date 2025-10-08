import os,sys

import numpy as np
import code

import pickle
import argparse


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "2d-multi")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	args = parser.parse_args()

	if "multi" in args.domain:
		from utilsDartsMulti import getAgentInfoFromFileName
	else:
		from utilsDarts import getAgentInfoFromFileName


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	resultFiles = os.listdir(args.resultsFolder + os.path.sep + "results")


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass


	
	# agentTypes = ["Target", "Flip", "Tricker","Bounded"]
	agentTypes = ["TargetAgentAbrupt","TargetAgentGradual","BoundedAgentAbrupt","BoundedAgentGradual"]


	allExpsCount = 1

	info = {}

	skippedError = 0

	# Collate results for the methods
	for rf in resultFiles: 

		# For each file, get the information from it
		print ('\n('+str(allExpsCount)+'/'+str(len(resultFiles))+') - RF : ', rf)

		allExpsCount += 1

		aName, agentType, xStr, param = getAgentInfoFromFileName(rf)


		with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:

			try:
				results = pickle.load(infile)
			except Exception as e: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} because error. (File incomplete).\nERROR: {e}")
				skippedError += 1


			finished = True

			try:
				lastEdited = results["lastEdited"]
			except Exception as e: 
				# To skip results file for exps that are still running
				finished = False



		# If previously seen, load prev info and count it
		if aName in info:
			if finished:
				info[aName]["numExpsFinished"] += 1
			else:
				info[aName]["numExpsNotFinished"] += 1
		else:
			info[aName] = {"type":agentType,
							"x":xStr, "param":param}

			if finished:
				info[aName]["numExpsFinished"] = 1
				info[aName]["numExpsNotFinished"] = 0
			else:
				info[aName]["numExpsFinished"] = 0
				info[aName]["numExpsNotFinished"] = 1


	infoPerType = {}

	for at in agentTypes:
		infoPerType[at] = {"total":0, "xs":[],"params":[],"names":[]}


	allXs = []

	for a in info:

		infoPerType[info[a]["type"]]["total"] += info[a]["numExpsFinished"]
		infoPerType[info[a]["type"]]["names"].append(a)

		x = info[a]["x"]
		try:
			infoPerType[info[a]["type"]]["xs"].append(float(x))
		except:
			infoPerType[info[a]["type"]]["xs"].append(x)
		
		if info[a]["param"] == "NA":
			infoPerType[info[a]["type"]]["params"].append(info[a]["param"])
		else:
			infoPerType[info[a]["type"]]["params"].append(float(info[a]["param"]))

		if x not in allXs:
			allXs.append(x)



	outfile = open(f"{args.resultsFolder}{os.sep}expStats.txt","w")
	outfileV = open(f"{args.resultsFolder}{os.sep}expStatsVerbose.txt","w")
	toPrint = [sys.stdout,outfile,outfileV]

	for each in toPrint:

		print("\n"+"-"*20,file=each)
		# How many exps in total = how many different agents
		print(f"\nTotal # experiments/agents: {allExpsCount}",file=each)
		print(file=each)

		# How many exps per agent type?
		for at in agentTypes:

			print(f"\nFor {at} agent: ",file=each)
			
			# Verbose - print out all agent names
			if each == outfileV:
				
				print("\tSeen Agents: ",file=each)

				for eachName in infoPerType[at]["names"]:

					aStr = f"\t\t{eachName} | finished: {info[eachName]['numExpsFinished']} | not finished: {info[eachName]['numExpsNotFinished']}"
					print(aStr,file=each)


			print(f"\tTotal # exps: {infoPerType[at]['total']}",file=each)
			
			# How many different xskills?
			print(f"\tTotal # unique xskills: {len(np.unique(infoPerType[at]['xs']))}",file=each)
			
			# How many pskills?
			print(f"\tTotal # unique pskills: {len(np.unique(infoPerType[at]['params']))}",file=each)


		print(f"\nSeen a total of {len(allXs)} different xskills across agent types.",file=each)
		print("\n"+"-"*20,file=each)

		print(f"Skipped because of error: {skippedError}", file=each)

	outfile.close()

	# code.interact("...", local=dict(globals(), **locals()))






















