import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)

# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
# To be used later for finding and properly loading the domains 
# Will look something like: "/home/archibald/skill-estimation/Environments/"
mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.sep
spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.sep)[0] + "setupSpaces.py").load_module()

import math
import numpy as np
import pandas as pd
import code

import pickle
import argparse
import time

from utilsDartsMulti import *
from scipy.spatial.distance import euclidean
from scipy.linalg import inv, sqrtm, norm
from scipy.interpolate import griddata


def computeCovErrorElipse(method,temp):

	covMatrix = np.zeros((k,k))

	# for eachObservation in range(len(estimatedXS)):

	# ASSUMING LAST OBSERVATION

	if "Multi" in method:
		estX = np.round(estimatedXS[-1],4)
		estR = round(estimatedR[-1],4)
	# Normal JEEDS (assuming symmetric agents)
	else:
		estX = np.round([estimatedXS[-1],estimatedXS[-1]],4)
		estR = 0.0


	#############################################
	# CREATE COVARIANCE MATRIX
	#############################################

	np.fill_diagonal(covMatrix,np.square(estX))

	# Fill the upper and lower triangles
	for i in range(k):
		for j in range(i+1,k):
			covMatrix[i,j] = np.prod(estX) * estR
			covMatrix[j,i] = covMatrix[i,j]

	# code.interact("...", local=dict(globals(), **locals()))

	#############################################


	# Calculate ellipse parameters
	eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
	major_axis = 2 * np.sqrt(5.991*eigenvalues[0])
	minor_axis = 2 * np.sqrt(5.991*eigenvalues[1])
	rotation_angle = np.arctan2(eigenvectors[1,0],eigenvectors[0,0])*(180/np.pi)

	# Save info
	temp["methods"][method]["infoElipse"] = {"majorAxis": major_axis,
														   "minorAxis": minor_axis,
														   "rotationAngle": rotation_angle}


def getBhattacharyyaDistance(cov1,cov2):
    """
    Calculate the Bhattacharyya distance between two multivariate normal distributions
    given their covariance matrices.

    Parameters:
        cov1 (ndarray): Covariance matrix of the first distribution.
        cov2 (ndarray): Covariance matrix of the second distribution.

    Returns:
        float: Bhattacharyya distance between the two distributions.
    """
    cov_avg = 0.5 * (cov1 + cov2)
    _, logdet1 = np.linalg.slogdet(cov1)
    _, logdet2 = np.linalg.slogdet(cov2)
    _, logdet_avg = np.linalg.slogdet(cov_avg)
    
    det_term = 0.5 * (logdet_avg - 0.5 * (logdet1 + logdet2))
    
    diff_mu = np.zeros_like(cov1)  # Assuming means are at the origin
    cov_inv = np.linalg.inv(cov_avg)
    mahalanobis_dist = np.trace(np.dot(np.dot(diff_mu.T, cov_inv), diff_mu))
    
    return 0.125 * mahalanobis_dist + 0.5 * det_term


def getMahalanobisDistance(mu1,mu2,cov1,cov2):
    """
    Calculate the Mahalanobis distance between two multivariate normal distributions.
    
    Parameters:
        mu1 (ndarray): Mean vector of the first distribution.
        mu2 (ndarray): Mean vector of the second distribution.
        cov1 (ndarray): Covariance matrix of the first distribution.
        cov2 (ndarray): Covariance matrix of the second distribution.
    
    Returns:
        float: Mahalanobis distance between the two distributions.
    """
    # diff_mu = mu1 - mu2
    cov_sum = cov1 + cov2
    cov_inv = np.linalg.inv(cov_sum)
    # mahalanobis_dist = np.sqrt(np.dot(np.dot(diff_mu.T, cov_inv), diff_mu))
    mahalanobis_dist = np.trace(np.dot(cov_inv,cov2))
    return mahalanobis_dist


def getNorm(e0,e1):
	a = norm(e0,2)
	b = norm(e1,2)

	return math.sqrt((a-b)**2)


def getW(e0,e1):
	sqrtE1 = sqrtm(e1)
	return np.trace(e0+e1-2*((sqrtE1@e0@sqrtE1)**(1/2)))


def getSQHD(e0,e1):

	temp = (np.linalg.det(e0)**(1/4))*(np.linalg.det(e1)**(1/4))
	t1 = temp/(np.linalg.det((e0+e1)/2)**(1/2))
	# x = (-1/8)@()
	# t2 = np.exp(x)
	return 1-t1 #@t2


def getKLD(e0,e1,k):
	return 0.5 * (np.trace((inv(e1)@e0))-k+np.log((np.linalg.det(e1))/(np.linalg.det(e0))))


def computeMetric(agentInfo,method,temp2,metric,k,trueXS,trueR):

	info = []

	for i in range(len(estimatedXS)):


		# Recreate true covMatrix based on type of agent
		if "Change" in agentInfo["name"]:
			
			if "Abrupt" in agentInfo["name"]:
				changeAt = agentInfo["changeAt"]

				if i < changeAt:
					trueXS = agentInfo["start"]
				else:
					trueXS = agentInfo["end"]

			elif "Gradual" in agentInfo["name"]:
				trueXS = agentInfo["gradualXskills"][i]

			trueCovMatrix = domainModule.getCovMatrix(trueXS,trueR)

		else:
			trueXS = agentInfo["trueXS"]
			trueCovMatrix = domainModule.getCovMatrix(trueXS,trueR)


		# Get estimated covariance matrix
		if "Multi" in method:
			estimatedCovMatrix = domainModule.getCovMatrix(estimatedXS[i],estimatedR[i])
		else:
			# Normal JEEDS (assuming symmetric agents)
			estimatedCovMatrix = domainModule.getCovMatrix([estimatedXS[i],estimatedXS[i]],0.0)

		# print(f"State #{i+1}")
		# info = f"\tX: {estimatedXS[i]} | R: {estimatedR[i]:.4f}"
		# print(info)

		# print("trueCovMatrix",trueCovMatrix)
		# print("estimatedCovMatrix",estimatedCovMatrix)

		e0, e1 = trueCovMatrix,estimatedCovMatrix


		if metric == "KLD":
			kld1 = getKLD(e0,e1,k)
			kld2 = getKLD(e1,e0,k)
			m = [kld1,kld2]

		elif metric == "JeffreysDivergence":
			kld1 = getKLD(e0,e1,k)
			kld2 = getKLD(e1,e0,k)
			m = kld1 + kld2

		elif metric == "SquaredHellingerDistance":
			d1 = getSQHD(e0,e1)
			# d2 = getSQHD(e1,e0)

			# m = [d1,d2]
			m = d1

		elif metric == "Wasserstein":
			m1 = getW(e0,e1)
			# m2 = getW(e1,e0)

			# m = [m1,m2]
			m = m1

		elif metric == "Norm":
			m = getNorm(e0,e1)
			# m = getNorm(e1,e0)

		elif metric == "MahalonobisDistance":
			mean = np.array([0.0,0.0])
			m1 = getMahalanobisDistance(mean,mean,e0,e1)
			m2 = getMahalanobisDistance(mean,mean,e1,e0)
			m = [m1,m2]

		elif metric == "BhattacharyyaDistance":
			m = getBhattacharyyaDistance(e0,e1)


		# print(f"\t{metric}: {m}")
		info.append(m)


	# Save info
	temp2["methods"][method][metric] = info


if __name__ == "__main__":


	# USAGE EXAMPLE:
	#  python Processing/process_results.py -resultsFolder Experiments/testingRandomAgent-AllMethods-1D-rand/ -domain 1d


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	resultFiles = os.listdir(f"{args.resultsFolder}{os.sep}results")


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass


	mainFolder = f"{args.resultsFolder}{os.sep}plots{os.sep}"

	# If the plots folder doesn't exist already, create it
	if not os.path.exists(mainFolder):
		os.mkdir(mainFolder)



	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework"))+os.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)



	# methodsNames = ["QRE-Multi-Particles-N-JT-MAP","QRE-Multi-Particles-N-JT-EES"]


	#agentTypes = ["Target", "Flip", "Tricker","Bounded"]
	agentTypes = ["TargetAgentAbrupt","TargetAgentGradual","BoundedAgentAbrupt","BoundedAgentGradual"]


	resultsDict = {}

	typeTargetsList = []


	processedRFs = []
	processedRFsAgentNames = []

	methods = []

	numHypsX = []
	numHypsP = []
	numHypsR = []

	numStates = 0
	seenAgents = []


	makeFolder2(args.resultsFolder,"ProcessedResultsFiles")
	
	rdFile = f"{args.resultsFolder}{os.sep}ProcessedResultsFiles{os.sep}resultsDictInfo"
	oiFile = f"{args.resultsFolder}{os.sep}otherInfo" 


	# For 2D will have "normal", "rand_pos" or "rand_v".
	mode = None


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)

			typeTargetsList = otherInfo["typeTargetsList"]
			methods = otherInfo["methods"]
			numHypsX = otherInfo['numHypsX']
			numHypsP = otherInfo['numHypsP']
			numHypsR = otherInfo['numHypsR']
			numStates = otherInfo["numObservations"]
			seenAgents = otherInfo["seenAgents"]
			domain = otherInfo["domain"]
			mode = otherInfo["mode"]
			dimensions = otherInfo["dimensions"]
			numParticles = otherInfo["numParticles"]

			processedRFs = otherInfo["processedRFs"]
			processedRFsAgentNames = otherInfo["processedRFsAgentNames"]

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
				with open(args.resultsFolder + os.sep + "results" + os.sep + resultFiles[i],"rb") as infile:
				
					results = pickle.load(infile)
					
					numHypsX = results['numHypsX']
					numHypsP = results['numHypsP']
					numHypsR = results['numHypsR']
					numStates = results["numObservations"]
					domain = results["domain"]
					mode = results["mode"]
					# tempTime = results["expTotalTime"]

					copyMethods = results["namesEstimators"]

					try:
						dimensions = results["dimensions"]
					except:
						dimensions = 2

						
					try:
						numParticles = results["numParticles"]
					except:
						numParticles = -1


					# method name in the format: f"QRE-Multi-Particles-{self.N}-Resample{int(self.percent*100)}%{ll}-NoiseDiv{noise}" 

					methods = []
					for each in copyMethods:
						
						methods.append(each+"-xSkills")
						methods.append(each+"-pSkills")
						
						if "Multi" in each:						
							methods.append(each+"-rhos")

					okFile = True

			except Exception as e: 
				okFile = False
				# code.interact("...", local=dict(globals(), **locals()))



		otherInfo = {}
		otherInfo["typeTargetsList"] = typeTargetsList

		otherInfo["methods"] = methods
		otherInfo['numHypsX'] = numHypsX
		otherInfo['numHypsP'] = numHypsP
		otherInfo['numHypsR'] = numHypsR
		otherInfo["numObservations"] = numStates
		otherInfo["domain"] = domain
		otherInfo["mode"] = mode
		otherInfo["dimensions"] = dimensions
		otherInfo["numParticles"] = numParticles

		otherInfo["resultFiles"] = resultFiles
		otherInfo["processedRFs"] = processedRFs
		otherInfo["processedRFsAgentNames"] = processedRFsAgentNames
		otherInfo["seenAgents"] = seenAgents

		with open(oiFile,"wb") as outfile:
			pickle.dump(otherInfo, outfile)

		del otherInfo
		del results

	
	# code.interact("!!!...", local=dict(globals(), **locals()))

	domainModule,delta = getDomainInfo(domain)
	
	print(methods)

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
	# args.resolution = delta
	pconfInfo, pconfPrats = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args)

	pconfInfo = np.asarray(pconfInfo)
	pconfPrats = np.asarray(pconfPrats)

	####################################

	# metrics = ["KLD","JeffreysDivergence",\
	# 			"SquaredHellingerDistance","Wasserstein",\
	# 			"MahalonobisDistance","BhattacharyyaDistance"]

	metrics = ["JeffreysDivergence"]


	# Start processing results
	total_num_exps = 0

	allExpsCount = 1

	skippedError = 0
	skippedNoLastEditted = 0

	times = []

	info = {}

	# Collate results for the methods
	for rf in resultFiles: 

		# For each file, get the information from it
		print ('\n('+str(allExpsCount)+'/'+str(len(resultFiles))+') - RF : ', rf)

		allExpsCount += 1

		if rf in processedRFs:
			print(f"Skipping {rf} since already processed.")
			continue


		with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:

			try:
				results = pickle.load(infile)
			except Exception as e: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} because error. (File incomplete).\nERROR: {e}")
				skippedError += 1
				continue



			agent = results['agent_name']
			# aName, agentType, xStr, param = getAgentInfoFromFileName(rf)
			agentType, aName, xStr, param = getAgentInfo(domain,agent,dimensions=2)


			finished = True

			try:
				lastEdited = results["lastEdited"]
			except Exception as e: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} because it doesn't contain 'lastEdited' info.")
				skippedNoLastEditted += 1
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

			if finished == False:
				continue


			try:
				tempTime = results["expTotalTime"]
			except: 
				# To skip results file for exps that are still running
				print(f"Skipping {rf} since not done yet.")
				continue


			startTime = time.perf_counter()

			# If previously seen, load prev info and count it
			if agent in processedRFsAgentNames:
				print(f"Loading info for {agent} since process results file already present for it.")
	
				with open(f"{rdFile}-{agent}","rb") as infile:
					tempInfo = pickle.load(infile)

				resultsDict[agent] = tempInfo[agent]
				resultsDict[agent]["num_exps"] += 1

			# Otherwise, initialize corresponding things
			else:

				# avg_rewards - avg of observed rewards per shot - [0:shotNum]    
				resultsDict[agent] = {"plot_y": {}, "estimates": {},\
									"mse_percent_pskills": {},\
									"percentsEstimatedPs": {},\
									"percentTrueP": 0.0, "avg_rewards": [],
									"num_exps": 1, "infoMetrics":[], "infoCovError": []}

				resultsDict[agent]["true_rewards"]: []

				# For True Noise (will be obtained using std)
				resultsDict[agent]["plot_y"]['tn'] = [0.0]


				# For the rest of the methods
				for m in methods:

					if m == "tn":
						resultsDict[agent]["plot_y"][m] = [0.0] * numStates
						
						resultsDict[agent]["estimates"][m] = [0.0] * numStates

					else:

						try:
							# if the method exists on the results file, load
							testLoadMethod = results[m]
						
						except:
							# print("Skipping:",m)
							# code.interact(local=locals())
							continue

						
						if "pSkills" in m or "rhos" in m or "Multi" not in m:
							resultsDict[agent]["plot_y"][m] = [0.0] * numStates
							resultsDict[agent]["estimates"][m] = [0.0] * numStates

						else:
							resultsDict[agent]["plot_y"][m] = np.zeros((numStates,dimensions))
							resultsDict[agent]["estimates"][m] = np.zeros((numStates,dimensions))
							
				
				# to store the avg reward of each experiment - for each one of the agents
				resultsDict[agent]["avg_rewards"] = [0.0] * numStates


				resultsDict[agent]["ev_intendedAction"] = [0.0] * numStates
				resultsDict[agent]["percent_true_reward"] = [0.0] * numStates


				# mean observed reward of experiment
				resultsDict[agent]["mean_observed_reward"] = 0.0

				# mean true reward of experiment
				resultsDict[agent]["mean_true_reward"] = 0.0

				resultsDict[agent]["mean_rs_reward_per_exp"] = 0.0
				resultsDict[agent]["mean_rs_reward_"] = []

				# to store the true reward of each experiment - for each one of the agents
				resultsDict[agent]["true_rewards"] = [0.0] * numStates
				resultsDict[agent]["mean_value_intendedAction"] = 0.0
				resultsDict[agent]["mean_random_reward_mean_vs"] = 0.0
				
				if agentType not in seenAgents:
					seenAgents.append(agentType)


			########################################################################################################
			# GET INFO FROM EXPERIMENT
			########################################################################################################

			# mean observed reward of experiment
			resultsDict[agent]["mean_observed_reward"] += np.nanmean(results['observed_rewards'])

			# mean true reward of experiment
			resultsDict[agent]["mean_true_reward"] += np.nanmean(results['true_rewards'])
			resultsDict[agent]["mean_value_intendedAction"]  += np.nanmean(results["valueIntendedActions"])
			resultsDict[agent]["mean_random_reward_mean_vs"]  += np.nanmean(results["meanAllVsPerState"])


			########################################################################################################
			# Compute MSE
			########################################################################################################
			
			# '''

			try:
				x = float(results['xskill'])
			except:
				x = results['xskill']



			p = float(param)

			if "multi" in domain:
				#code.interact("...", local=dict(globals(), **locals()))
				rho = float(aName.split("|")[-2].split("R")[1])


			# Assuming only JTM methods given
			for m in methods: 

				# Tries to load the results/info for the given method if present in results file. Otherwise, keep going
				try:
					mxs = results[m]
				except:
					# code.interact("...", local=dict(globals(), **locals()))
					continue


				###################################################
				# Find MSE of actual estimate
				###################################################
				
				# print(agent)
				# print(m)

				# For each observation/state
				for mxi in range(len(mxs)):


					if "Change" in agent:
						
						if "Abrupt" in agent:
							changeAt = results["changeAt"]

							if mxi < changeAt:
								trueXS = x[0]
							else:
								trueXS = x[1]

						elif "Gradual" in agent:
							trueXS = results["gradualXskills"][mxi]

					else:
						trueXS = x

					# print("trueXS: ",trueXS)


					if "pSkills" in m:
						merr = (mxs[mxi]-p) ** 2.0  #MSE ERROR
						resultsDict[agent]["plot_y"][m][mxi] += merr

					elif "rhos" in m:
						merr = (mxs[mxi]-rho) ** 2.0  #MSE ERROR
						resultsDict[agent]["plot_y"][m][mxi] += merr

					else:
						if "Multi" not in m:
							merr = (mxs[mxi]-trueXS[0]) ** 2.0  #MSE ERROR
							resultsDict[agent]["plot_y"][m][mxi] += merr

						else:
							for ee in range(len(trueXS)):
								merr = (mxs[mxi][ee]-trueXS[ee]) ** 2.0  #MSE ERROR
								resultsDict[agent]["plot_y"][m][mxi][ee] += merr



					if "pSkills" in m or "rhos" in m:
						resultsDict[agent]["estimates"][m][mxi] += mxs[mxi]
					else:
						# Store estimate per method & per obs
						if "Multi" in m:
							for ee in range(len(x)):
								resultsDict[agent]["estimates"][m][mxi][ee] += mxs[mxi][ee]
						else:
							resultsDict[agent]["estimates"][m][mxi] += mxs[mxi]

				# code.interact("...", local=dict(globals(), **locals()))

				###################################################
				


				###################################################
				# SKIPPING FOR NOW
				# NEED TO DEFINE HOW TO GO ABOUT IT
				###################################################

				###################################################
				# Convert from estimate to rationality percentage
				###################################################
				# '''

				if "pSkills" not in m:
					continue


				mm = m.split("-pSkills")[0]+"-xSkills"
				mmr = ""


				if "Multi" in m:
					mmr = m.split("-pSkills")[0]+"-rhos"
				else:
					mmr = "0.0"


				if m not in resultsDict[agent]["percentsEstimatedPs"]:
					resultsDict[agent]["percentsEstimatedPs"][m] = {}


				resultsDict[agent]["percentsEstimatedPs"][m][resultsDict[agent]["num_exps"]] = [0.0] * numStates
				resultsDict[agent]["percentsEstimatedPs"][m]["averaged"] = [0.0] * numStates
				resultsDict[agent]["mse_percent_pskills"][m] = [0.0] * numStates
				

				# '''

				# For each observation/state
				for mxi in range(len(mxs)):
					
					# Use estimated xskill and not actual true one
					# WHY? estimatedX and not trueX?? because "right" answer is not available
					try:
						xStr = [float(resultsDict[agent]["estimates"][mm][mxi])]
					except:
						xStr = resultsDict[agent]["estimates"][mm][mxi]


					if mmr != "0.0":
						estimatedRho = float(resultsDict[agent]["estimates"][mmr][mxi])
					else:
						estimatedRho = float(mmr)


					# Get pskill estimate of current method - estimatedP
					estimatedP = mxs[mxi]


					if len(xStr) == 1:
						xStr.append(xStr[0])
					else:
						xStr = xStr.tolist()

					estimate = xStr
					estimate += [estimatedRho]
					estimate += [estimatedP]

					percent_estimatedP = griddata(pconfInfo,pconfPrats,estimate,method="nearest")[0]
					# print(percent_estimatedP)
					# code.interact("...", local=dict(globals(), **locals()))


					# Save info
					resultsDict[agent]["percentsEstimatedPs"][m][resultsDict[agent]["num_exps"]][mxi] = percent_estimatedP

					###################################################

				# '''
			########################################################################################################



			# Compute avg observed reward and resampled rewards
			for mxi in range(numStates):
				# Compute the mean of the rewards received up til this point
				resultsDict[agent]["avg_rewards"][mxi] += sum(results['observed_rewards'][0:mxi]) / (1.0 * (mxi+1)) 
				resultsDict[agent]["true_rewards"][mxi] += sum(results['true_rewards'][0:mxi]) / (1.0 * (mxi+1)) 

                       
			########################################################################################################


			########################################################################################################
			# Get exps's info
			########################################################################################################

			trueXS,trueR = getXandR_FromAgentName(agent)
			# print(trueXS)


			agentInfo = {}
			agentInfo["name"] = agent

			if "Abrupt" in agent:
				agentInfo["changeAt"] = results["changeAt"]
				# print(agentInfo["changeAt"])
			elif "Gradual" in agent:
				agentInfo["gradualXskills"] = results["gradualXskills"]
				# print(agentInfo["gradualXskills"])


			if "Change" in agent:
				agentInfo["start"] = trueXS[0] 
				agentInfo["end"] = trueXS[1]
			else:
				agentInfo["trueXS"] = trueXS


			k = results["dimensions"]

			states = results["states"]["states"]
			noisyActions = np.array(results["noisy_actions"])
			intendedActions = np.array(results["intended_actions"])

			temp1 = {"trueXS": trueXS, "trueR": trueR,
								  "methods": {}, "states": states,
								  "noisyActions": noisyActions,
								  "intendedActions": intendedActions}

			temp2 = {"methods":{}}


			# print(results.keys())
			for method in methods:

				if "xSkills" in method:

					if method in results.keys():

						if method not in temp1["methods"]:
							temp1["methods"][method] = {}

						if method not in temp2["methods"]:
							temp2["methods"][method] = {}


						estimatedXS = results[f"{method}"]
						temp1["methods"][method]["estimatedXS"] = estimatedXS
						temp2["methods"][method]["estimatedXS"] = estimatedXS


						if "Multi" in method:
							estimatedR = results[f"{method.replace('-xSkills','')}-rhos"]
							temp1["methods"][method]["estimatedR"] = estimatedR
							temp2["methods"][method]["estimatedR"] = estimatedR


						computeCovErrorElipse(method,temp1)


						for eachM in metrics:
							computeMetric(agentInfo,method,temp2,eachM,k,trueXS,trueR)


			resultsDict[agent]["infoCovError"].append(temp1)
			resultsDict[agent]["infoMetrics"].append(temp2)

			########################################################################################################3
			


			########################################################################################################3
			# Store processed results - online manner
			########################################################################################################3

			# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
			with open(f"{rdFile}-{agent}","wb") as outfile:
				pickle.dump(resultsDict,outfile)

			# code.interact("...", local=dict(globals(), **locals()))
			del resultsDict[agent]

			########################################################################################################3
		

			###########################################################
			# Update info file 
			###########################################################
			
			processedRFs.append(rf)

			if agent not in processedRFsAgentNames:
				processedRFsAgentNames.append(agent)


			with open(oiFile,"rb") as infile:
				otherInfo = pickle.load(infile)

			otherInfo["processedRFs"] = processedRFs
			otherInfo["processedRFsAgentNames"] = processedRFsAgentNames
			otherInfo["seenAgents"] = seenAgents
			otherInfo["metrics"] = metrics
			
			with open(oiFile,"wb") as outfile:
				pickle.dump(otherInfo,outfile)

			del otherInfo
			
			###########################################################

			total_num_exps += 1


			endTime = time.perf_counter()

			print(f"Time: {endTime-startTime:.2f}")

			times.append(endTime-startTime)




	print('\nCompiled results for', total_num_exps, 'experiments.')
	print(f"Skipped because of error when loading file: {skippedError} experiments")
	print(f"Skipped because of not having 'last edited' timestamp info: {skippedNoLastEditted} experiments")
	
	if len(times) != 0:
		print("AVG time to process each results file: ", sum(times)/len(times))



	# Case no experiments were processed
	#if total_num_exps == 0:
	#	exit()


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


