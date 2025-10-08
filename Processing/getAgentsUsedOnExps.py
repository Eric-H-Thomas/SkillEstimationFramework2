import os,sys

import numpy as np
import code

import json
import argparse

from utilsDarts import *


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	resultFiles = os.listdir(args.resultsFolder + os.path.sep + "results")


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass


	
	#agentTypes = ["Target", "Flip", "Tricker","Bounded", "TargetBelief"]
	agentTypes = ["Target", "Flip", "Tricker","Bounded"]


	allExpsCount = 1

	agents = []


	# Collate results for the methods
	for rf in resultFiles: 

		# For each file, get the information from it
		print ('\n('+str(allExpsCount)+'/'+str(len(resultFiles))+') - RF : ', rf)

		allExpsCount += 1

		aName, agentType, xStr, param = getAgentInfoFromFileName(rf)

		# OnlineExp_70-438_23_06_15_56_12_0_188317-18-AgentBoundedAgent-X70.4377-L0.002910744216446768
		splitted = rf.split("-")
		seedNum = splitted[1].split("_")[-1]


		agents.append([agentType,float(xStr),float(param),int(seedNum)])


	# Order agents by xskill?


	# Save to json file
	toSave = {"agents": agents}

	with open("agents.json","w") as outfile:
		json.dump(toSave,outfile)



	# On spawn script

	
	# Load json file
	with open("agents.json","r") as infile:
		agentsInfo = json.load(infile)
	

	agentsInfo = agentsInfo["agents"]


	# Divide agents
	numThreads = 2
	totalAgents = len(agentsInfo) / numThreads
	over = len(agentsInfo) % numThreads
	

	# Save to new file with mainSeedNum + threadNum
	
	# Save over agents on last file


	# On the actual spawning of the jobs
	# 	Pass agent file as param on instruction

	# Set 
	# 	-seed num (inner) to same used on exps
	# 	-num iters param to # of diff agents/xskills

	code.interact("...", local=dict(globals(), **locals()))

