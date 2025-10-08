import os,sys

import numpy as np
import code

import pickle
import argparse

from utilsDarts import *


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "2d")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]


	oiFile = args.resultsFolder + os.sep + "otherInfo" 

	with open(oiFile,"rb") as file:
		otherInfo = pickle.load(file)

		domain = otherInfo["domain"]
		mode = otherInfo["mode"]

		processedRFsAgentNames = otherInfo["processedRFsAgentNames"]

		try:
			wrap = otherInfo["wrap"]
		except:
			wrap = True
	
	

	domainModule,delta = getDomainInfo(args.domain,wrap)

	prdFile = args.resultsFolder + os.sep + "ProcessedResultsFilesForPlots" + os.sep + "resultsDictInfo"


	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	#############################################
	# ASSUMING EXPS HAVE BEEN PROCESSED ALREADY
	# AND PLOTS CREATED AT LEAST ONCE TO HAVE
	# PERCENT TRUE RATIONAL INFO PRESENT
	#############################################

	allExpsCount = 1
	
	allXs = []

	infoPerType = {}
	resultsDict = {}

	for a in processedRFsAgentNames:

		print ('\n('+str(allExpsCount)+'/'+str(len(processedRFsAgentNames))+') - File: ', a)

		aType, xStr, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)

		allExpsCount += 1

		x = float(xStr)
		p = float(p)

		if x not in allXs:
			allXs.append(x)


		# If previously seen, load prev info and count it
		if aType in infoPerType:
			infoPerType[aType]["xs"].append(x)
			infoPerType[aType]["ps"].append(p)
			infoPerType[aType]["percents"].append(float(resultsDict[a]["percentTrueP"]))
			infoPerType[aType]["numExps"].append(resultsDict[a]["num_exps"])
		else:
			infoPerType[aType] = {"xs":[],"ps":[],"percents":[],"numExps":[]}


		del resultsDict[a]

	# Find percent rationality distribution

	buckets = [0.25,0.50,0.75,1.00]
	infoPerBuckets = {}



	for at in infoPerType:

		if at not in infoPerBuckets:
			infoPerBuckets[at] = {}

		for b in buckets:
			infoPerBuckets[at][b] = []


	for at in infoPerType:

		for i in range(len(infoPerType[at]["xs"])):

			# Find bucket
			for b in range(len(buckets)):
				if infoPerType[at]["percents"][i] <= buckets[b]:
					bucket = buckets[b]
					break

			# Save info
			try:
				infoPerBuckets[at][bucket].append([infoPerType[at]["xs"][i],infoPerType[at]["ps"][i],infoPerType[at]["percents"][i]])
			except:
				continue


	outfile = open(f"{args.resultsFolder}{os.sep}expStats.txt","w")
	outfileV = open(f"{args.resultsFolder}{os.sep}expStatsVerbose.txt","w")
	toPrint = [sys.stdout,outfile,outfileV]

	for each in toPrint:

		print("\n"+"-"*20,file=each)
		# How many exps in total = how many different agents
		print(f"\nTotal # experiments/agents: {allExpsCount}",file=each)
		print(file=each)

		# How many exps per agent type?
		for at in infoPerBuckets:

			print(f"\nFor {at} agent: ",file=each)
			
			# Verbose - print out all agent names
			if each == outfileV:
				
				print("\tSeen Agents: ",file=each)

				for ii in range(len(infoPerType[at]['xs'])):
					print(f"\t\tx: {infoPerType[at]['xs'][ii]} | p: {infoPerType[at]['ps'][ii]} | numExps: {infoPerType[at]['numExps'][ii]}",file=each)

			print(f"\tTotal # exps: {len(infoPerType[at]['xs'])}",file=each)
			
			# How many different xskills?
			print(f"\tTotal # unique xskills: {len(np.unique(infoPerType[at]['xs']))}",file=each)
			
			# How many pskills?
			print(f"\tTotal # unique pskills: {len(np.unique(infoPerType[at]['ps']))}",file=each)

			for b in buckets:
				print(f"\t\tTotal # of agents with <= {b} % rationality: {len(infoPerBuckets[at][b])}",file=each)


		print(f"\nSeen a total of {len(allXs)} different xskills across agent types.",file=each)
		print("\n"+"-"*20,file=each)

	outfile.close()

	code.interact("...", local=dict(globals(), **locals()))


	# Find distribution of rationality percents
	# Assuming exps were processed already

