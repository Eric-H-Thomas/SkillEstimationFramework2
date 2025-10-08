import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)

# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
# To be used later for finding and properly loading the domains 
# Will look something like: "/home/archibald/skill-estimation/Environments/"
mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.path.sep
spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.path.sep)[0] + "setupSpaces.py").load_module()

import math
import numpy as np
import pandas as pd
import code

import pickle
import argparse
from time import time

from utilsDarts import *
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":

	# USAGE EXAMPLE:
	#  python Processing/process_results.py -resultsFolder Experiments/testingRandomAgent-AllMethods-1D-rand/ -domain 1d


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-delta", dest = "delta", help = "Delta = resolution to use when doing the convolution", type = float, default = 5.0)
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "2d")
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	parser.add_argument("-N", dest = "N", help = "", type = int, default = 1)
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	resultFiles = os.listdir(args.resultsFolder + os.path.sep + "results")


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass


	# If the plots folder doesn't exist already, create it
	if not os.path.exists(args.resultsFolder + os.path.sep + "plots" + os.path.sep):
		os.mkdir(args.resultsFolder + os.path.sep + "plots" + os.path.sep)


	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework")) + os.path.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	# methodsNames = ['OR', 'BM-MAP', 'BM-EES', "JT-FLIP-MAP", "JT-FLIP-EES","JT-QRE-MAP", "JT-QRE-EES"]
	# methodsNames = ['OR', 'BM-MAP', 'BM-EES',"JT-QRE-MAP", "JT-QRE-EES", "NJT-QRE-MAP", "NJT-QRE-EES"]

	# methodsNames = ['OR', 'BM-MAP', 'BM-EES',
	# 			"JT-QRE-MAP", #"JT-QRE-MAP-GivenPrior","JT-QRE-MAP-MinLambda","JT-QRE-MAP-GivenPrior-MinLambda",
	# 			"JT-QRE-EES",#"JT-QRE-EES-GivenPrior","JT-QRE-EES-MinLambda","JT-QRE-EES-GivenPrior-MinLambda",
	# 			"NJT-QRE-MAP",#"NJT-QRE-MAP-GivenPrior","NJT-QRE-MAP-MinLambda","NJT-QRE-MAP-GivenPrior-MinLambda",
	# 			"NJT-QRE-EES"]#"NJT-QRE-EES-GivenPrior","NJT-QRE-EES-MinLambda","NJT-QRE-EES-GivenPrior-MinLambda"]

	methodsNames = ["JT-QRE-MAP","JT-QRE-EES"]


	#agentTypes = ["Target", "Flip", "Tricker","Bounded", "TargetBelief"]
	agentTypes = ["Target", "Flip", "Tricker","Bounded"]


	resultsDict = {}

	actualMethodsNames = []
	actualMethodsOnExps = []
	typeTargetsList = []


	processedRFs = []
	processedRFsAgentNames = []

	numHypsX = []
	numHypsP = []
	numStates = 0
	seenAgents = []


	makeFolder2(args.resultsFolder,"ProcessedResultsFiles")
	
	rdFile = args.resultsFolder + os.sep + "ProcessedResultsFiles" + os.sep + "resultsDictInfo"
	oiFile = args.resultsFolder + os.path.sep + "otherInfo" 


	# For 2D will have "normal", "rand_pos" or "rand_v".
	# For 1D it will stay as none.
	mode = None

	wrap = None


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)

			actualMethodsNames = otherInfo["actualMethodsNames"]
			actualMethodsOnExps = otherInfo["actualMethodsOnExps"]
			typeTargetsList = otherInfo["typeTargetsList"]
			numHypsX = otherInfo['numHypsX']
			numHypsP = otherInfo['numHypsP']
			numStates = otherInfo["numObservations"]
			seenAgents = otherInfo["seenAgents"]
			methods = otherInfo["methods"]
			domain = otherInfo["domain"]
			mode = otherInfo["mode"]
			betas = otherInfo["betas"]

			processedRFs = otherInfo["processedRFs"]
			processedRFsAgentNames = otherInfo["processedRFsAgentNames"]

			try:
				wrap = otherInfo["wrap"]
			except:
				wrap = True

	# If wasn't able to load info, find it and save it to file
	except:

		domain = ""

		okFile = False
		i = 0

		while not okFile:

			# Open the first file to load the different number of hypothesis used for the different estimators
			while resultFiles[i][-7:] != 'results':
				i += 1

			try:
				with open(args.resultsFolder + os.path.sep + "results" + os.path.sep + resultFiles[i],"rb") as infile:
				
					results = pickle.load(infile)
					
					numHypsX = results['numHypsX']
					numHypsP = results['numHypsP']
					numStates = results["numObservations"]
					domain = results["domain"]
					mode = results["mode"]
					tempTime = results["expTotalTime"]

					try:
						wrap = results["wrap"]
					except:
						wrap = True
					
					okFile = True

			except Exception as e: 
				okFile = False


		betas = []

		for tempKey in results.keys():

			if "BM-EES" in tempKey and "Beta" in tempKey:
				b = float(tempKey.split("Beta-")[1])
				if b not in betas:
					betas.append(b)

			if "BM" in tempKey:
				if "Optimal" in tempKey and "OptimalTargets" not in typeTargetsList:
					typeTargetsList.append("OptimalTargets")
				if "Domain" in tempKey and "DomainTargets" not in typeTargetsList:
					typeTargetsList.append("DomainTargets")


		methods = ["tn"]
		
		for m in methodsNames:
			if "JT" in m:   
				for nh in range(len(numHypsX)):
					methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-pSkills")
					methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-xSkills")
			elif "OR" in m:
				for nh in range(len(numHypsX)):
					if domain == "1d" or domain == "2d":
						methods.append(m + "-" + str(numHypsX[nh]))
					else: # For sequential darts & billiards
						methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesMidGame")
						methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesFullGame")
			elif "BM" in m:
				for nh in range(len(numHypsX)):
					for tt in typeTargetsList:
						# 1D & 2D will only have TBA-"OptimalTargets"
						if (domain == "1d" or domain == "2d") and tt == "DomainTargets":
							continue
						for b in betas:
							methods.append(m + "-" + str(numHypsX[nh]) + "-" + tt + "-Beta-" + str(b))

			else:
				for nh in range(len(numHypsX)):
					# Methods left is OR & no pskill hyps is needed
					methods.append(m + "-" + str(numHypsX[nh])) # + "-" + str(numHypsP[nh]))
		


		for m in methods:

			try:
				# if the method exists on the results file, load
				testLoadMethod = results[m]

				actualMethodsOnExps.append(m)

				if "OR" in m and "-estimatesMidGame" in m:
					actualMethodsNames.append("OR-MidGame")
				elif "OR" in m and "-estimatesFullGame" in m:
					actualMethodsNames.append("OR-FullGame")
				elif "OR" in m:
					actualMethodsNames.append("OR")

				# To add just once as can possibly appear multiple 
				# times because of diff betas
				elif "BM-MAP" in m and "BM-MAP" not in actualMethodsNames:
					actualMethodsNames.append("BM-MAP")
				elif "BM-EES" in m and "BM-EES" not in actualMethodsNames:
					actualMethodsNames.append("BM-EES")

				elif "JT-FLIP-MAP" in m:
					actualMethodsNames.append("JT-FLIP-MAP")
				elif "JT-FLIP-EES" in m:
					actualMethodsNames.append("JT-FLIP-EES")

				# Must come before "JT-QRE" method comparison
				# since ("JT-QRE" in "NJT-QRE") results in true as well 
				elif "NJT-QRE-MAP" in m and "xSkills" in m:
					actualMethodsNames.append("NJT-QRE-MAP")
				elif "NJT-QRE-EES" in m and "xSkills" in m:
					actualMethodsNames.append("NJT-QRE-EES")
				
				elif "JT-QRE-MAP" in m and "xSkills" in m:
					actualMethodsNames.append("JT-QRE-MAP")
				elif "JT-QRE-EES" in m and "xSkills" in m:
					actualMethodsNames.append("JT-QRE-EES")

			except:
				continue


		actualMethodsNames.append("TN")
		actualMethodsOnExps.append("tn")

		
		otherInfo = {}
		otherInfo["actualMethodsNames"] = actualMethodsNames
		otherInfo["actualMethodsOnExps"] = actualMethodsOnExps
		otherInfo["typeTargetsList"] = typeTargetsList

		otherInfo["methods"] = methods
		otherInfo['numHypsX'] = numHypsX
		otherInfo['numHypsP'] = numHypsP
		otherInfo["numObservations"] = numStates
		otherInfo["domain"] = domain
		otherInfo["mode"] = mode
		otherInfo["wrap"] = wrap
		otherInfo["betas"] = betas

		otherInfo["resultFiles"] = resultFiles
		otherInfo["processedRFs"] = processedRFs
		otherInfo["processedRFsAgentNames"] = processedRFsAgentNames
		otherInfo["seenAgents"] = seenAgents

		with open(oiFile,"wb") as outfile:
			pickle.dump(otherInfo, outfile)

		del otherInfo
		del results

		#code.interact("!!!...", local=dict(globals(), **locals()))


	domainModule,delta = getDomainInfo(args.domain,wrap)
	
	############################################################################################################################
	############################################################################################################################
	# Use when debugging/testing - to speed up - read only specified # of results file (and not however many there are in the folder)
	#resultFiles = resultFiles[0:3000]
	############################################################################################################################
	############################################################################################################################


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args,wrap)

	####################################


	bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		minMaxX = [bucketsX[0],bucketsX[-1]] #[0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]


	# Start processing results
	total_num_exps = 0

	allExpsCount = 1

	skippedError = 0
	skippedNoLastEditted = 0

	times = []

	# Collate results for the methods
	for rf in resultFiles: 

		# For each file, get the information from it
		print ('\n('+str(allExpsCount)+'/'+str(len(resultFiles))+') - RF : ', rf)

		allExpsCount += 1

		if rf in processedRFs:
			print(f"Skipping {rf} since already processed.")
			continue


		param = ""

		with open(args.resultsFolder + os.path.sep + "results" + os.path.sep + rf,"rb") as infile:

			try:
				results = pickle.load(infile)
			except Exception as e: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} because error. (File incomplete).\nERROR: {e}")
				skippedError += 1
				continue


			try:
				lastEdited = results["lastEdited"]
			except Exception as e: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} because it doesn't contain 'lastEdited' info.")
				skippedNoLastEditted += 1
				continue



			try:
				tempTime = results["expTotalTime"]
			except: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} since not done yet.")
				continue


			aNameOriginal,aName,param = getAgentInfo(args.domain,results['agent_name'])

			# Create copy of agent type
			seenAgentType = aNameOriginal


			startTime = time()

			# If previously seen, load prev info and count it
			if aName in processedRFsAgentNames:
				print("Loading info for ", aName, " since process results file already present for it.")
	
				with open(f"{rdFile}-{aName}","rb") as infile:
					tempInfo = pickle.load(infile)

				resultsDict[aName] = tempInfo[aName]
				resultsDict[aName]["num_exps"] += 1

			# Otherwise, initialize corresponding things
			else:

				# avg_rewards - avg of observed rewards per shot - [0:shotNum]    
				resultsDict[aName] = {"plot_y": dict(), "estimates": dict(),\
									"mse_percent_pskills": dict(),\
									"percentsEstimatedPs": dict(),\
									"percentTrueP": 0.0, "avg_rewards": [],
									"num_exps": 1, "info":[]}

				resultsDict[aName]["true_rewards"]: []

				# For True Noise (will be obtained using std)
				resultsDict[aName]["plot_y"]['tn'] = [0.0]


				# For the rest of the methods
				for m in methods:

					if m == "tn":
						resultsDict[aName]["plot_y"][m] = [0.0] * numStates
						
						resultsDict[aName]["estimates"][m] = [0.0] * numStates

					else:

						try:
							# if the method exists on the results file, load
							testLoadMethod = results[m]
						
						except:
							# print("Skipping:",m)
							# code.interact(local=locals())
							continue

						# If TBA/BM method, need to account for possible different betas
						if "BM" in m:
							tempM, beta, tt = getInfoBM(m)

							# To initialize once
							if tt not in resultsDict[aName]["plot_y"]:
								resultsDict[aName]["plot_y"][tt] = {}
								resultsDict[aName]["estimates"][tt] = {}

							if tempM not in resultsDict[aName]["plot_y"][tt]:
								resultsDict[aName]["plot_y"][tt][tempM] = {}
								resultsDict[aName]["estimates"][tt][tempM] = {}

							if beta not in resultsDict[aName]["plot_y"][tt][tempM]:
								resultsDict[aName]["plot_y"][tt][tempM][beta] = [0.0] * numStates
								resultsDict[aName]["estimates"][tt][tempM][beta] = [0.0] * numStates

						else:
							resultsDict[aName]["plot_y"][m] = [0.0] * numStates
							resultsDict[aName]["estimates"][m] = [0.0] * numStates

				
				# to store the avg reward of each experiment - for each one of the agents
				resultsDict[aName]["avg_rewards"] = [0.0] * numStates


				resultsDict[aName]["ev_intendedAction"] = [0.0] * numStates
				resultsDict[aName]["percent_true_reward"] = [0.0] * numStates


				# mean observed reward of experiment
				resultsDict[aName]["mean_observed_reward"] = 0.0

				# mean true reward of experiment
				resultsDict[aName]["mean_true_reward"] = 0.0

				resultsDict[aName]["mean_rs_reward_per_exp"] = 0.0
				resultsDict[aName]["mean_rs_reward_"] = []

				# to store the true reward of each experiment - for each one of the agents
				resultsDict[aName]["true_rewards"] = [0.0] * numStates
				resultsDict[aName]["mean_value_intendedAction"] = 0.0
				resultsDict[aName]["mean_random_reward_mean_vs"] = 0.0
				
				if seenAgentType not in seenAgents:
					seenAgents.append(seenAgentType)


			####################################################################################
			# GET INFO FROM EXPERIMENT
			####################################################################################

			# mean observed reward of experiment
			resultsDict[aName]["mean_observed_reward"] += np.nanmean(results['observed_rewards'])

			# mean true reward of experiment
			resultsDict[aName]["mean_true_reward"] += np.nanmean(results['true_rewards'])
			resultsDict[aName]["mean_value_intendedAction"]  += np.nanmean(results["valueIntendedActions"])
			resultsDict[aName]["mean_random_reward_mean_vs"]  += np.nanmean(results["meanAllVsPerState"])


			#######################################################################################################
			# Computing MSE
			########################################################################################################
			
			x = float(results['xskill'])
			
			if "Random" not in results['agent_name']:
				p = float(param)


			for m in actualMethodsOnExps: 

				# skip TN method since there's not info about it in the results file
				if m == "tn":
					continue

				# Tries to load the results/info for the given method if present in results file. Otherwise, keep going
				try:
					mxs = results[m]
				except:
					continue

				if "-pSkills" in m:
					useP = True
				else:
					useP = False


				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)

					# For each observation/state
					for mxi in range(len(mxs)):	
						merr = (mxs[mxi] - x) ** 2.0  #MSE ERROR

						if len(resultsDict[aName]["plot_y"][tt][tempM][beta]) < mxi + 1: 
							# print "if: mxi: ", mxi
							resultsDict[aName]["plot_y"][tt][tempM][beta].append(merr)
						else:
							# print "else: mxi: ", mxi
							resultsDict[aName]["plot_y"][tt][tempM][beta][mxi] += merr

						# store estimate per method & per obs
						resultsDict[aName]["estimates"][tt][tempM][beta][mxi] += mxs[mxi]

					# code.interact("...", local=dict(globals(), **locals()))

				# Rest of the methods
				else:

					###################################################
					# Find MSE of actual estimate
					###################################################
					
					# For each observation/state
					for mxi in range(len(mxs)):

						if useP and "Random" not in aName:
							merr = (mxs[mxi] - p) ** 2.0  #MSE ERROR

						else:
							merr = (mxs[mxi] - x) ** 2.0  #MSE ERROR

						try:
							if len(resultsDict[aName]["plot_y"][m]) < mxi + 1: 
								# print "if: mxi: ", mxi
								resultsDict[aName]["plot_y"][m].append(merr)
							else:
								# print "else: mxi: ", mxi
								resultsDict[aName]["plot_y"][m][mxi] += merr
						except:
							pass
							#code.interact("...", local=dict(globals(), **locals()))

						# Store estimate per method & per obs
						resultsDict[aName]["estimates"][m][mxi] += mxs[mxi]
					
					###################################################
						

					###################################################
					# Convert from estimate to rationality percentage
					###################################################
					
					# Skip TN & OR & TBA
					if "pSkills" not in m:
						continue


					# To determine whether to use JT's or NJT's current xskill estimate 
					if "NJT" in m:
						mm = "NJT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"
					else:
						mm = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"


					if m not in resultsDict[aName]["percentsEstimatedPs"]:
						resultsDict[aName]["percentsEstimatedPs"][m] = {}


					resultsDict[aName]["percentsEstimatedPs"][m][resultsDict[aName]["num_exps"]] = [0.0] * numStates
					resultsDict[aName]["percentsEstimatedPs"][m]["averaged"] = [0.0] * numStates
					resultsDict[aName]["mse_percent_pskills"][m] = [0.0] * numStates
					
					# For each observation/state
					for mxi in range(len(mxs)):
						
						# Use estimated xskill and not actual true one
						# WHY? estimatedX and not trueX?? because "right" answer is not available
						# xStr = float(resultsDict[aName]["plot_y"][mm][mxi])
						xStr = float(resultsDict[aName]["estimates"][mm][mxi])

						# find proper bucket for current x
						bucket1, bucket2 = getBucket(bucketsX,minMaxX,xStr)


						# Get pskill estimate of current method - estimatedP
						estimatedP = mxs[mxi]


						# Convert estimatedP to corresponding % of rand max
						if bucket2 != None:
							prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
							prat2 = np.interp(estimatedP,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

							prat = np.interp(estimatedP, [prat1], [prat2])
							percent_estimatedP = prat
						# edges/extremes case
						else:
							# using one of the functions for now
							prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

							percent_estimatedP = prat1


						# Save info
						resultsDict[aName]["percentsEstimatedPs"][m][resultsDict[aName]["num_exps"]][mxi] = percent_estimatedP

					###################################################
							
			########################################################################################################


			# Compute avg observed reward and resampled rewards
			for mxi in range(numStates):
				# Compute the mean of the rewards received up til this point
				resultsDict[aName]["avg_rewards"][mxi] += sum(results['observed_rewards'][0:mxi]) / (1.0 * (mxi+1)) 
				resultsDict[aName]["true_rewards"][mxi] += sum(results['true_rewards'][0:mxi]) / (1.0 * (mxi+1)) 


				# resultsDict[aName]["ev_intendedAction"][mxi] += sum(results["evIntendedActions"][0:mxi])/(1.0 * (mxi + 1))
				# resultsDict[aName]["percent_true_reward"][mxi] += (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"])
				# percentOfTrueReward = (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"])

				# resultsDict[aName]["mean_rs_reward_"].append(sum(results["rs_rewards"][mxi]) / (len(results["rs_rewards"][mxi]) * 1.0))

			# Get mean rs per exp
			# resultsDict[aName]["mean_rs_reward_per_exp"] = sum( resultsDict[aName]["mean_rs_reward_"])/(len( resultsDict[aName]["mean_rs_reward_"]) * 1.0)

			  
			# Compute the mean of the actual estimates - for the different observations - EVERY
			# for m in methods:
			#     for mxi in range(numStates): 
			#         resultsDict[aName]["estimates"][m][mxi] += sum(results[m][0:mxi]) / (1.0 * (mxi+1)) 

				# if m == "OR-17" and aName == "Target":
					# print resultsDict["Target"]["plot_y"]["OR-17"][:10]


			########################################################################################################
			# TRUE NOISE
			########################################################################################################

			td = None
			size = None
			
			if args.domain == "2d" or args.domain == "sequentialDarts":
				# Computing true differences here 
				# as the one from the rf's seems incorrect
				td = []


				tempIntended = np.array(results["intended_actions"])
				tempNoisy = np.array(results["noisy_actions"])

				# Find differences
				diffsX = tempIntended[:,0] - tempNoisy[:,0]
				diffsY = tempIntended[:,1] - tempNoisy[:,1]

				td = [diffsX,diffsY]
				size = len(diffsX)

				# for ti in range(len(diffsX)):

					# Get prediction/estimate until this point
					# estXs = np.std(diffsX[:ti+1],ddof=0)
					# estYs = np.std(diffsY[:ti+1],ddof=0)

					# concatXY = np.concatenate((diffsX[:ti+1],diffsY[:ti+1])) 
					# est = np.std(concatXY)
				
			else:
		
				# Compute TN - Plot out the estimate from numpy std on the true difference
				td = results['true_diffs']
				size = len(td)

						
			for ti in range(size):

				if args.domain == "2d" or args.domain == "sequentialDarts":
					concatXY = np.concatenate((td[0][:ti+1],td[1][:ti+1])) 
					est = np.std(concatXY)
		
				else:
					# Get prediction
					est = np.std(td[:ti+1],ddof=0)

				# Get error 
				er = (est-x) ** 2.0 #MSE ERROR

				# er = mean_squared_error([x]*(ti+1),td[:ti+1],squared=False)


				if len(resultsDict[aName]["plot_y"]['tn']) < ti + 1:
					resultsDict[aName]["plot_y"]['tn'].append(er)
				else:
					resultsDict[aName]["plot_y"]['tn'][ti] += er  

				resultsDict[aName]["estimates"]["tn"][ti] += est                        

			########################################################################################################

			# code.interact("...", local=dict(globals(), **locals()))


			# resampled_rewards = results['rs_rewards']


			#############################################################################3
			# Store processed results - online manner
			#############################################################################3

			# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
			with open(f"{rdFile}-{aName}","wb") as outfile:
				pickle.dump(resultsDict,outfile)

			# code.interact("...", local=dict(globals(), **locals()))
			del resultsDict[aName]

			#############################################################################3
		

			###########################################################
			# Update info file 
			###########################################################
			
			processedRFs.append(rf)
			processedRFsAgentNames.append(aName)

			with open(oiFile,"rb") as infile:
				otherInfo = pickle.load(infile)

			otherInfo["processedRFs"] = processedRFs
			otherInfo["processedRFsAgentNames"] = processedRFsAgentNames
			otherInfo["seenAgents"] = seenAgents
			
			with open(oiFile,"wb") as outfile:
				pickle.dump(otherInfo,outfile)

			del otherInfo
			
			###########################################################

			total_num_exps += 1


			endTime = time()

			times.append(endTime-startTime)


	print('\nCompiled results for', total_num_exps, 'experiments.')
	print(f"Skipped because of error when loading file: {skippedError} experiments")
	print(f"Skipped because of not having 'last edited' timestamp info: {skippedNoLastEditted} experiments")
	
	if len(times) != 0:
		print("AVG time to process each results file: ", sum(times)/len(times))
		 
	# code.interact("...", local=dict(globals(), **locals()))
