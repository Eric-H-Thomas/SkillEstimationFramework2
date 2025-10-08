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

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors

import gc

from importlib.machinery import SourceFileLoader
from pathlib import Path

from pybaseball import playerid_reverse_lookup, cache

import six

import scipy


def makeFolder(resultsFolder,folderName):

	if resultsFolder[-1] == "/":
		tempFolder = resultsFolder
	else:
		tempFolder = resultsFolder + os.sep 

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(tempFolder + os.sep + folderName):
		os.mkdir(tempFolder + os.sep + folderName)


def getDomainInfo(domainName):

	if "baseball" in domainName:
		load = f"Environments{os.sep}Baseball{os.sep}"
		domainModule = SourceFileLoader("baseball",load+"baseball.py").load_module()
		
		# 0.5 inches | 0.0417 feet
		delta = 0.0417

	return domainModule,delta


def multivariateGaussianEntropy(covMatrix):

    # Compute the determinant of the covariance matrix
    detCov = np.linalg.det(covMatrix)
    
    # Dimensionality of the distribution
    dimensionality = covMatrix.shape[0]
    
    # Entropy calculation
    entropy = 0.5 * (dimensionality * (1 + np.log(2 * np.pi)) + np.log(detCov))
    
    return entropy


def getCovMatrix(stdDevs,rho):

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	return covMatrix



if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball-multi")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	result_files = os.listdir(args.resultsFolder + os.sep + "results")

	try:
		result_files.remove("backup")
	except:
		pass


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

	methodsNames = []

	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Environments/"
	mainFolderName = scriptPath.split("Processing")[0] + "Environments" + os.sep

	domainModule,delta = getDomainInfo(args.domain)


	makeFolder(args.resultsFolder,"ResultsDictFiles")
	rdFile = args.resultsFolder + os.sep + "ResultsDictFiles" + os.sep + "resultsDictInfo"


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		oiFile = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)

			namesEstimators = otherInfo["namesEstimators"]
			methods = otherInfo["methods"]
			numHypsX = otherInfo['numHypsX']
			numHypsP = otherInfo['numHypsP']
			seenAgents = otherInfo["seenAgents"]
			domain = otherInfo["domain"]
			typeTargetsList = otherInfo["typeTargetsList"]
			# betas = otherInfo["betas"]
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

			for m in results.keys():

				# SKIPPING NJT METHODS
				# if "NJT" in m:
					# continue

				if (not m.isalpha()) and "-" in m and "allProbs" not in m and "allParticles" not in m:
					methods.append(m)


		# betas = []
		# getBetas(results,betas,typeTargetsList)


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

			chunk = rf.split("_")[4].split(".results")[0]

			numObservations = results["numObservations"]


			if pitcherID not in resultsDict:
				resultsDict[pitcherID] = {}

			if pitchType not in resultsDict[pitcherID]:
				resultsDict[pitcherID][pitchType] = {}

			if chunk not in resultsDict[pitcherID][pitchType]:
				resultsDict[pitcherID][pitchType][chunk] = {}


		# code.interact("...", local=dict(globals(), **locals()))

		if rf in resultFilesLoaded or Path(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}").is_file():
			print(f"\tNot processing {rf} since already processed.")
			
			if pitcherID not in resultsDict:
				resultsDict[pitcherID] = {}

			if pitchType not in resultsDict[pitcherID]:
				resultsDict[pitcherID][pitchType] = {}

			if chunk not in resultsDict[pitcherID][pitchType]:
				resultsDict[pitcherID][pitchType][chunk] = {}

			continue

		else:

			resultsDict[pitcherID][pitchType][chunk] = {"estimates":{},"numObservations":numObservations,"entropy":{}}

			for m in methods:

				if "whenResampled" in m:
					continue

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
						resultsDict[pitcherID][pitchType][chunk]["estimates"][m] = results[m]
					else:
						resultsDict[pitcherID][pitchType][chunk]["estimates"][m] = {}	

				

				if "rho" not in m and "pSkills" not in m \
				and "whenResampled" not in m and "all" not in m \
				and "resamplingMethod" not in m:

					if m not in resultsDict[pitcherID][pitchType][chunk]["entropy"]:
						resultsDict[pitcherID][pitchType][chunk]["entropy"][m] = []


					for each in range(len(results[m])):

						if "Multi" in m:
							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = results[m][each]
							rho = results[tempM][each]
						# Case: JEEDS
						else:
							estimatedX = results[m][each]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0

						# Get Covariance Matrix
						covMatrix = getCovMatrix(stdDevs,rho)

						# Compute Generalized Variance
						entropy = multivariateGaussianEntropy(covMatrix)

						# entropy = scipy.stats.multivariate_normal(cov=covMatrix).entropy()

						resultsDict[pitcherID][pitchType][chunk]["entropy"][m].append(entropy)

					# code.interact("...", local=dict(globals(), **locals()))



			del results

			
			if list(resultsDict[pitcherID][pitchType][chunk]["estimates"].keys()) == []:
				print(f"\n\t\tNo results seen yet for {pitcherID}-{pitchType}. Only initial exp info present.")
				del resultsDict[pitcherID][pitchType][chunk]

				

			# Update file
			with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'wb') as handle:
				pickle.dump(resultsDict,handle)


			# Reset dict
			del resultsDict[pitcherID][pitchType][chunk]

			resultsDict[pitcherID][pitchType][chunk] = {}

			gc.collect()

			########################################


	otherInfo = {}
	otherInfo["namesEstimators"] = namesEstimators
	otherInfo["methods"] = methods
	otherInfo['numHypsX'] = numHypsX
	otherInfo['numHypsP'] = numHypsP
	otherInfo["seenAgents"] = seenAgents
	otherInfo["domain"] = domain
	otherInfo["typeTargetsList"] = typeTargetsList
	# otherInfo["betas"] = betas
	otherInfo["result_files"] = result_files

	saveAs2 = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

	with open(saveAs2,"wb") as outfile:
		pickle.dump(otherInfo,outfile)


	print('\nCompiled results for', total_num_exps, 'experiments')

	# code.interact("...", local=dict(globals(), **locals()))
