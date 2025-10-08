from matplotlib import rcParams,rc
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

import argparse,sys,os
import math 
import pickle
import numpy as np
import pandas as pd
import code,csv
from copy import deepcopy

import dataframe_image as dfi

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors

from utilsDarts import getDomainInfo, makeFolder, getBetas

import gc


from pybaseball import playerid_reverse_lookup, cache

import six

from pathlib import Path

if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	result_files = os.listdir(args.resultsFolder + os.sep + "results")


	if len(result_files) == 0:
		print("No result files present for experiment.")
		exit()


	# If the plots folder doesn't exist already, create it
	if not os.path.exists(args.resultsFolder + os.sep + "plots" + os.sep):
		os.mkdir(args.resultsFolder + os.sep + "plots" + os.sep)


	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework")) + os.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	resultsDict = {}


	cache.enable()


	namesEstimators = []
	typeTargetsList = []

	numHypsX = []
	numHypsP = []
	seenAgents = []

	methodsNames = ['OR', 'BM-MAP', 'BM-EES',
				"JT-QRE-MAP","JT-QRE-MAP-GivenPrior","JT-QRE-MAP-MinLambda","JT-QRE-MAP-GivenPrior-MinLambda",
				"JT-QRE-EES","JT-QRE-EES-GivenPrior","JT-QRE-EES-MinLambda","JT-QRE-EES-GivenPrior-MinLambda",
				"NJT-QRE-MAP","NJT-QRE-MAP-GivenPrior","NJT-QRE-MAP-MinLambda","NJT-QRE-MAP-GivenPrior-MinLambda",
				"NJT-QRE-EES","NJT-QRE-EES-GivenPrior","NJT-QRE-EES-MinLambda","NJT-QRE-EES-GivenPrior-MinLambda"]

	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Environments/"
	mainFolderName = scriptPath.split("Processing")[0] + "Environments" + os.sep

	domainModule,delta = getDomainInfo(args.domain)


	methodsAllProbs = ["BM-Beta-0.5-allProbs",
			"BM-Beta-0.75-allProbs",
			"BM-Beta-0.85-allProbs",
			"BM-Beta-0.9-allProbs",
			"BM-Beta-0.95-allProbs",
			"BM-Beta-0.99-allProbs",
			"JT-QRE-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-allProbs",
			"JT-QRE-MinLambda-1.3-allProbs",
			"JT-QRE-MinLambda-1.7-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-allProbs",
			"NJT-QRE-xSkills-allProbs",
			"NJT-QRE-pSkills-allProbs",
			"NJT-QRE-xSkills-allProbs",
			"NJT-QRE-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-pSkills-allProbs",
			"NJT-QRE-MinLambda-1.3-xSkills-allProbs",
			"NJT-QRE-MinLambda-1.3-pSkills-allProbs",
			"NJT-QRE-MinLambda-1.3-xSkills-allProbs",
			"NJT-QRE-MinLambda-1.3-pSkills-allProbs",
			"NJT-QRE-MinLambda-1.7-xSkills-allProbs",
			"NJT-QRE-MinLambda-1.7-pSkills-allProbs",
			"NJT-QRE-MinLambda-1.7-xSkills-allProbs",
			"NJT-QRE-MinLambda-1.7-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-pSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-xSkills-allProbs",
			"NJT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-pSkills-allProbs"]


	if not os.path.exists(args.resultsFolder+os.sep+"ResultsDictFiles"+os.sep):
			os.mkdir(args.resultsFolder+os.sep+"ResultsDictFiles"+os.sep)

	rdFile = args.resultsFolder + os.sep + "ResultsDictFiles" + os.sep + "resultsDictInfo"


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		oiFile = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)

			namesEstimators = otherInfo["namesEstimators"]
			methods = otherInfo["methods"]
			methodsAllProbs = otherInfo["methodsAllProbs"]
			numHypsX = otherInfo['numHypsX']
			numHypsP = otherInfo['numHypsP']
			seenAgents = otherInfo["seenAgents"]
			domain = otherInfo["domain"]
			typeTargetsList = otherInfo["typeTargetsList"]
			betas = otherInfo["betas"]
			resultFilesLoaded = otherInfo["result_files"]

		loadedInfo = True

	except:
		# Do nothing, just continue to processing files
		loadedInfo = False

		resultFilesLoaded = []


	# If wasn't able to load prev processed info, start from scratch
	if not loadedInfo:
		domain = ""

		# Open the first file to load the different number of hypothesis used for the different estimators
		i = 0
		while result_files[i][-7:] != 'results':
			i += 1

		with open(args.resultsFolder + os.sep + "results" + os.sep + result_files[i],"rb") as infile:
			results = pickle.load(infile)
			
			namesEstimators = results["namesEstimators"]
			numHypsX = results['numHypsX']
			numHypsP = results['numHypsP']
			domain = results["domain"]


			methods = []
			# methodsAllProbs = []

			for m in results.keys():

				# SKIPPING NJT METHODS
				# if "NJT" in m:
					# continue

				if (not m.isalpha()) and "-" in m and "allProbs" not in m:
					methods.append(m)

				# if "allProbs" in m:
				# 	methodsAllProbs.append(m)



		betas = []
		getBetas(results,betas,typeTargetsList)

		# methods = getMethods(args.domain,methodsNames,namesEstimators,numHypsX,numHypsP,betas,typeTargetsList)


		# code.interact("1...", local=dict(globals(), **locals()))
	

	############################################################################################################################
	############################################################################################################################
	# Use when debugging/testing - to speed up - read only specified # of results file (and not however many there are in the folder)
	#result_files = result_files[0:30]
	############################################################################################################################
	############################################################################################################################


	# Start processing results
	total_num_exps = 0

	# NOTE: EACH RESULT FILE BELONGS TO A GIVEN PITCHER AND PITCH TYPE
	# EACH COMBINATION WILL BE SEEN ONLY ONCE (MEANING JUST 1 EXP)

	for rf in result_files: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)

		param = ""

		with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:
			results = pickle.load(infile)

			agent = results["agent_name"]
			pitcherID = agent[0]
			pitchType = agent[1]
			seenAgents.append(agent)

			numObservations = results["numObservations"]


			if pitcherID not in resultsDict:
				resultsDict[pitcherID] = {}

			if pitchType not in resultsDict[pitcherID]:
				resultsDict[pitcherID][pitchType] = {}


		if rf in resultFilesLoaded or Path(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}").is_file():
			print(f"\tNot processing {rf} since already processed.")
			
			if pitcherID not in resultsDict:
				resultsDict[pitcherID] = {}

			if pitchType not in resultsDict[pitcherID]:
				resultsDict[pitcherID][pitchType] = {}

			continue

		else:

			resultsDict[pitcherID][pitchType] = {"estimates": {},"allProbs":{},"numObservations":numObservations}

			for m in methods:

				try:
					validCount = False

					# if the method exists on the results file, load
					testLoadMethod = results[m]

					if len(testLoadMethod) == numObservations:
						validCount = True

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
						
					# Save estimates
					if validCount:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = results[m]
					else:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = {}

				else:
					# Save estimates
					# Won't add info for method if there's a mismatch between
					# expected # of observations and the number of estimates produced
					if validCount:
						resultsDict[pitcherID][pitchType]["estimates"][m] = results[m]
					else:
						resultsDict[pitcherID][pitchType]["estimates"][m] = {}


			for ma in methodsAllProbs:

				try:

					validCount = False

					# if the method exists on the results file, load
					testLoadMethod = results[ma]

					# Plus 1 since prior probs included in all probs
					if len(testLoadMethod) == numObservations+1:
						validCount = True

				except:
					# print("->",m)
					# code.interact("all probs...", local=dict(globals(), **locals()))
					continue


				# Won't add info for method if there's a mismatch between
				# expected # of observations and the number of estimates produced	
				if validCount:
					resultsDict[pitcherID][pitchType]["allProbs"][ma] = testLoadMethod
				else:
					resultsDict[pitcherID][pitchType]["allProbs"][ma] = {}


			# FOR TESTING
			# if pitcherID == "425794" and pitchType == 'CH':
			# 	code.interact("test...", local=dict(globals(), **locals()))


			del results

			
			if list(resultsDict[pitcherID][pitchType]["estimates"].keys()) == []:
				print(f"\n\t\tNo results seen yet for {pitcherID}-{pitchType}. Only initial exp info present.")
				del resultsDict[pitcherID][pitchType]


			########################################
			# Save info to file
			########################################

			# # If file with info exists already, load info (to avoid overwriting)
			# if Path(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}").is_file():

			# 	with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}",'r') as handle:
			# 		loadedDictInfo = json.load(handle)


			# 	if pitcherID not in loadedDictInfo:
			# 		loadedDictInfo[pitcherID] = {}

			# 	if pitchType not in loadedDictInfo[pitcherID]:
			# 		loadedDictInfo[pitcherID][pitchType] = {}

			# 	# Update dict info
			# 	loadedDictInfo[pitcherID][pitchType] = resultsDict[pitcherID][pitchType] 


			# 	# Update file
			# 	with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}",'w') as handle:
			# 		json.dump(loadedDictInfo,handle)

			# 	del loadedDictInfo
				

			# # CASE: Creating file for the first time
			# else:
			# Update file
			with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}",'wb') as handle:
				pickle.dump(resultsDict,handle)


			# Reset dict
			del resultsDict[pitcherID][pitchType]

			resultsDict[pitcherID][pitchType] = {}

			gc.collect()

			########################################


	otherInfo = {}
	otherInfo["namesEstimators"] = namesEstimators
	otherInfo["methods"] = methods
	otherInfo["methodsAllProbs"] = methodsAllProbs
	otherInfo['numHypsX'] = numHypsX
	otherInfo['numHypsP'] = numHypsP
	otherInfo["seenAgents"] = seenAgents
	otherInfo["domain"] = domain
	otherInfo["typeTargetsList"] = typeTargetsList
	otherInfo["betas"] = betas
	otherInfo["result_files"] = result_files

	saveAs2 = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

	with open(saveAs2,"wb") as outfile:
		pickle.dump(otherInfo,outfile)


	print('\nCompiled results for', total_num_exps, 'experiments')
