import os,code
from pathlib import Path
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib

import pickle,code

from copy import deepcopy

import numpy as np

import argparse

from itertools import product

# Find location of current file
scriptPath = os.path.realpath(__file__)

# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
# To be used later for finding and properly loading the domains 
# Will look something like: "/home/archibald/skill-estimation/Environments/"
mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.sep
spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.sep)[0] + "setupSpaces.py").load_module()


def getParamsFromAgentName(a):  

	if "Bounded" in a:
		string = a.split("|L")
		aType = "Bounded"

	elif "Flip" in a:
		string = a.split("|P")
		aType = "Flip"

	elif "Tricker" in a:
		string = a.split("|Eps")
		aType = "Tricker"

	elif "TargetBelief" in a:
		string = a.split("|B")
		aType = "TargetBelief"

	elif "Target" in a:
		string = [a]
		aType = "Target"

	elif "Random" in a:
		string = a.split("|X")
		aType = "Random"

	# print("aType: ", aType)

	# Find pskill
	if aType == "Target":
		p = 100.0
	else:
		# code.interact("...", local=dict(globals(), **locals()))

		p = round(float(string[1]),4)

		# verify if number in scientific notation
		if "e" in str(p):
			# will truncate rest of decimal places
			# Just keep first 3
			# Splits on "e" to stay with first part
			# Convert back to number
			p =  float('{:0.3e}'.format(p).split("e")[0])


	string2 = string[0].split("|")[1:]

	# Find xskill
	if "Change" in a:
		x = [eval(each[1:]) if not each.isdigit() else float(each) for each in string2]
	else:
		x = [float(each[1:]) if not each.isdigit() else float(each) for each in string2]

	# Return info
	return aType, x, p


def getXandR_FromAgentName(a):
	if "Target" in a:

		# Removing pskill param
		info = a.split("|")

		info = info[1:]

	else:
		# Removing pskill param
		info = a.split("|")

		info = info[1:-1]


	trueXS = []

	for each in info[:-1]:
		# print(each)
		try:
			trueXS.append(float(each.replace("X","")))
		# Case dynamic agent
		except:
			temp = each.replace("X","").replace("[","").replace("]","").replace(" ","")
			temp = [float(each) for each in temp.split(",")]
			trueXS.append(temp)
			# print(trueXS)

	trueR = float(info[-1].strip("R"))

	return trueXS,trueR


def getMethods(domain,methodsNames,namesEstimators,numHypsX,numHypsP,betas,typeTargetsList):
	
	methods = []

	for m in methodsNames:

		#if any(m in s for s in namesEstimators):

		if "JT" in m:   
			# code.interact("inside...", local=dict(globals(), **locals()))
			for nh in range(len(numHypsX)):

				methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-pSkills")
				methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-xSkills")


				lookFor = ["-GivenPrior-MinLambda","-GivenPrior","-MinLambda"]

				for each in lookFor:
					
					if any(each in s for s in namesEstimators):
						methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + each + "-pSkills")
						methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + each + "-xSkills")

					'''
					l = ""

					if "GivenPrior" in m and "MinLambda" in m:
						l += "-GivenPrior"
						l += "-MinLambda"
					elif "GivenPrior" in m:
						l += "-GivenPrior"
					elif "MinLambda" in m:
						l += "-MinLambda"
					'''
					
				
		elif "OR" in m:
			for nh in range(len(numHypsX)):
				if domain in ["1d","2d","baseball"]:
					methods.append(m + "-" + str(numHypsX[nh]))
				else: # For sequential darts & billiards
					methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesMidGame")
					methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesFullGame")
		
		elif "BM" in m:
			for nh in range(len(numHypsX)):

				if typeTargetsList == []:
					for b in betas:
						methods.append(m + "-" + str(numHypsX[nh]) + "-Beta-" + str(b))

				else:
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

	return methods


def makeFolder(resultsFolder,folderName):

	if resultsFolder[-1] == "/":
		tempFolder = resultsFolder
	else:
		tempFolder = resultsFolder + os.sep 

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(tempFolder + "plots" + os.sep + folderName):
		os.mkdir(tempFolder + "plots" + os.sep + folderName)


def makeFolder2(resultsFolder,folderName):

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(resultsFolder  + os.sep + folderName):
		os.mkdir(resultsFolder + os.sep + folderName)


def makeFolder3(folderName):

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(folderName):
		os.mkdir(folderName)


def getDomainInfo(domainName,wrap=None):


	if domainName == "2d":		
		load = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
		domainModule = SourceFileLoader("two_d_darts",load+"two_d_darts.py").load_module()
		delta = 5.0

	elif domainName == "2d-multi":
		load = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
		domainModule = SourceFileLoader("two_d_darts_multi",load+"two_d_darts_multi.py").load_module()		
		delta = 5.0
		
	elif domainName == "sequentialDarts":
		load = f"Environments{os.sep}Darts{os.sep}SequentialDarts{os.sep}"
		domainModule = SourceFileLoader("sequential_darts",load+"sequential_darts.py").load_module()
		delta = 5.0

	elif "baseball" in domainName:
		load = f"Environments{os.sep}Baseball{os.sep}"
		domainModule = SourceFileLoader("baseball",load+"baseball.py").load_module()
		delta = 5.0

	return domainModule,delta


def getAgentInfoFromFileName(rf):

	if "Change" in rf:
		splitted = rf.split("Change")
		agentType = splitted[0].split("_")[-1][5:]
		params = splitted[1].split("_")[1:]
		params[2] = params[2].split(".results")[0]

	else:
		splitted = rf.split("Agent|")
		agentType = splitted[0].split("Agent")[1]
		params = splitted[1].split("|")

		params[-1] = params[-1].split(".results")[0]


	# code.interact("...", local=dict(globals(), **locals()))


	label = ""

	if agentType == "Flip":
		label = "P"
	elif agentType == "Tricker":
		label = "Eps"
	elif agentType == "Bounded":
		label = "L"


	if "Abrupt" in agentType or "Gradual" in agentType:
		param = "" 
		xStr = "|".join(params)

		if "Target" in agentType:
			param = "100"
		else:
			param = "NA"

	elif "Target" in agentType:
		xStr = "|".join(params)
		param = "100"
	else:
		xStr = "|".join(params[:-1])
		param = float(params[-1].split(label)[1])
	
	aName = f"{xStr}|{label}{param}"


	return aName, agentType, xStr, param


def getAgentInfo(domainName,agentName,dimensions):

	if domainName == "2d-multi":

		if "Change" in agentName:
			splitted = agentName.split("Change")
			aNameOriginal = splitted[0]
			params = splitted[1].split("|")[1:]

		else:
			aNameOriginal,params = agentName.split("Agent|")
			params = params.split("|")

		# code.interact("...", local=dict(globals(), **locals()))

		label = ""

		if "Flip" in aNameOriginal:
			label = "P"
		elif "Tricker" in aNameOriginal:
			label = "Eps"
		elif "Bounded" in aNameOriginal:
			label = "L"


		if "Target" in aNameOriginal:
			xStr = "|".join(params)
			param = "100"
		else:
			xStr = "|".join(params[:-1])
			param = float(params[-1].split(label)[1])
		
		aName = f"{xStr}|{label}{param}"


	elif domainName in ["1d","2d","sequentialDarts"]:
		
		aNameOriginal = str(agentName.split("Agent-X")[0])
		params = str(agentName.split("Agent-X")[1])


		aName = aNameOriginal

		if aName == "Flip":
			string = params.split("-P")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-P" + param

		elif aName == "Tricker":
			string = params.split("-Eps")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-Eps" + param

		elif aName == "Bounded":
			string = params.split("-L")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-L" + param

		elif aName == "Target":
			string = params.split("-X")
			xStr = string[0]
			param = "100"
			aName += "-X" + xStr

	
	# code.interact("...", local=dict(globals(), **locals()))

	'''
	print(aNameOriginal)
	print(aName)
	print(xStr)
	print(param)
	'''

	return aNameOriginal,aName,xStr,float(param)


def getBucket(buckets,minMax,param):

	# Find proper bucket for current x
	for b in range(len(buckets)):
		if param <= buckets[b]:
			break

	# Get actual bucket
	bucket1 = buckets[b]

	# Placeholder variable
	otherBucket = None
	
	bucket2 = None

	# First bucket
	if b == 0:
		# use left edge/extreme - i.e. 0
		otherBucket = minMax[0]
	# If last bucket
	elif b == len(buckets)-1:
		# use right edge/extreme - i.e. 5/100 depending on the domain
		otherBucket = minMax[1]
	# Somewhere in the middle - consider next bucket
	else:
		bucket2 = bucket1
		bucket1 = buckets[b-1]

	# print(f"B1: {bucket1} | B2: {bucket2}")
	return bucket1,bucket2


def loadProcessedInfo(f,a):

	try:

		with open(f"{f}-{a}","rb") as infile:
			tempInfo = pickle.load(infile)

		return tempInfo[a]

	except Exception as e:
		return False


def updateProcessedInfo(f,a,resultsDict):

	with open(f"{f}-{a}","wb") as outfile:
		pickle.dump(resultsDict,outfile)


def pconf(rng,resultsFolder,domain,domainModule,spaceModule,mode,args,wrap=None):

	print("\n----------------------")
	
	print("PCONF: ")
	print("Domain: ", domain)
	
	if "2d" in domain:
		print("Mode: ", mode)

	print("----------------------\n")


	numSamples = 1000

	mainFolder = "Spaces" + os.sep + "ExpectedRewards" + os.sep
	fileName = f"ExpectedRewards-{domain}-N{numSamples}"
	expectedRFolder = mainFolder + fileName


	# pconfPerXskill = []


	if "Exp" not in resultsFolder[0:3]:
		resultsFolder = "Experiments" + os.sep + resultsFolder

	if not os.path.exists(resultsFolder + os.sep + "plots" + os.sep):
		os.mkdir(resultsFolder + os.sep + "plots" + os.sep)


	tempName = resultsFolder + os.sep + "plots" + os.sep + "pconfInfo"

	if domain == "sequentialDarts":
		tempName += "-Values"

	#do per line

	# if file is not present, need to compute info
	if not os.path.exists(tempName):

		dimensions = 1

		if domain == "2d-multi":

			numSamples = 50

			# All the lambdas that we will use to generate the plot
			lambdas = np.logspace(-5,1.5,100)

			args.resolution = 5.0

			# Get the states to use for evaluation
			states = domainModule.generate_random_states(rng,numSamples,mode)

			xskills = [5, 10, 30, 50, 70, 90, 110, 130, 150]
			# xskills = np.linspace(2.5,150.5,num=33)

			rhos = np.round(np.linspace(-0.80,0.80,num=5),4)


			dimensions = 2

			xskills = [np.copy(xskills),np.copy(xskills)]

			allParams = list(product(xskills[0],xskills[1]))

			if len(xskills) >= 2:

				for di in range(2,len(xskills)):
					allParams = list(product(allParams,xskills[di]))

			allXskill = np.copy(allParams)
			allParams = list(product(allParams,rhos))

			# Convert to 1D [x1,x2,rho]
			for ii in range(len(allParams)):
				allParams[ii] = eval(str(allParams[ii]).replace(")","").replace("(",""))


			allParams = [list(each) for each in allParams]
			# code.interact("...", local=dict(globals(), **locals()))


		elif domain == "sequentialDarts":
			
			lambdas = np.logspace(-5,1.5,100)

			xskills = np.linspace(2.5,150.5,num=33)	

			args.resolution = 5.0
			
			startScore = domainModule.getPlayerStartScore()
			states = list(range(startScore + 1))

			args.N = 1


		print("\nCreating spaces...")

		if domain in ["2d-multi"]:
			spaces = spacesModule.SpacesRandomDarts(numSamples,domainModule,mode,args.resolution,numSamples,expectedRFolder)
			
			# spaces.updateSpace(rng,[allXskill,rhos],states)
			for si, s in enumerate(states):
				
				print(f"State {si+1}/{len(states)}" )
				
				for each in allParams:
					# print(each)
					# Plus [None] cause methods expects pskill too
					spaces.updateSpaceParticles(rng,each+[None],s,{},fromEstimator=False)
					# code.interact("...", local=dict(globals(), **locals()))


		elif domain == "sequentialDarts":
			spaces = spacesModule.SpacesSequentialDarts(numSamples,domainModule,args.mode,args.resolution,numSamples,expectedRFolder)
			spaces.updateSpace(xskills)

		print("\nDone spaces...")

		
		pconfInfo = []
		pconfPrats = []

		for each in allParams:

			xs, r = each[:-1], each[-1]

			key = "|".join(map(str,xs))+f"|{r}"


			if domain == "2d-multi":
				space = spaces.convolutionsPerXskill[key]
				size = len(states)
				loopInfo = states
			else:
				size = len(states)-2
				loopInfo = states[2:]



			print('Generating data for ', key, "... ")
			# print(space)


			prat = [] #This is where the probability of rational reward will be stored
			mins = [] #Store min reward possible
			maxs = [] #Store max reward possible
			means = [] #Store the mean of the possible rewards (this is the uniform random reward)
			evs = [] #Store the ev of the current agent's strategy

	
			for l in lambdas:     

				# tempKey = key + f"|{l}"

				max_rs = np.zeros(size)
				min_rs = np.zeros(size)
				exp_rs = np.zeros(size)
				mean_rs = np.zeros(size)

				si = 0
				
				for s in loopInfo:
					# print(s)

					if domain in ["2d-multi"]:
						values = np.copy(space[str(s)]["all_vs"].flatten())
					else:
						values = np.copy(space.flatEVsPerState[s])

					# print("values: ",values)

					# Get the values from the ev 
					max_rs[si] = np.max(values)
					min_rs[si] = np.min(values) 
					mean_rs[si] = np.mean(values) 

					# Bounded decision-making with lambda = l
					b = np.max(values*l)
					expev = np.exp(values*l-b)
					sumexp = np.sum(expev)
					P = expev/sumexp


					# Store bounded agent's EV
					exp_rs[si] = sum(P*values)
					# code.interact("v...", local=dict(globals(), **locals()))


					si += 1

				
				pratTemp = np.mean((exp_rs - mean_rs)/(max_rs - mean_rs))

				prat.append(pratTemp)
				mins.append(np.mean(min_rs))
				means.append(np.mean(mean_rs))
				maxs.append(np.mean(max_rs))
				evs.append(np.mean(exp_rs))

				# Store to use later
				pconfInfo.append(xs+[r,l])
				pconfPrats.append(pratTemp)


			# print(prat)
			# code.interact("...", local=dict(globals(), **locals()))


		# pconfPerXskill["xskills"] = xskills
		# pconfPerXskill["rhos"] = rhos
		# pconfPerXskill["allParams"] = allParams


		# Find mins and max per dimensions
		# mins = []
		# maxs = []

		# for each in xskills:
		# 	mins.append(np.min(each))
		# 	maxs.append(np.max(each))

		# pconfPerXskill["mins"] = mins
		# pconfPerXskill["maxs"] = maxs
		info = {}
		info["pconfInfo"] = pconfInfo
		info["pconfPrats"] = pconfPrats


		# Save dict containing all info - to be able to rerun it later
		with open(tempName,"wb") as outfile:
			pickle.dump(info,outfile)

		
	# file with the info is present, proceed to load
	else:
		print("Loading pconf info...")

		with open(tempName, "rb") as file:
			info = pickle.load(file)

	print("Finished pconf()")



	pconfInfoForPlot = np.array(info["pconfInfo"])
	pconfPratsForPlot = np.array(info["pconfPrats"])

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')

	# img = ax.scatter(pconfInfoForPlot[:,0],pconfInfoForPlot[:,1],pconfInfoForPlot[:,2], c=pconfPratsForPlot, cmap=plt.viridis())
	# fig.colorbar(img,pad=0.12)

	# ax.set_xlabel('X1')
	# ax.set_ylabel('X2')
	# ax.set_zlabel('Rho')

	# plt.show()



	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(min(pconfPratsForPlot),max(pconfPratsForPlot))
	sm = ScalarMappable(norm = norm, cmap = cmap)
	sm.set_array([])


	'''
	for x1 in np.unique(pconfInfoForPlot[:,0]):

		for x2 in np.unique(pconfInfoForPlot[:,1]):

			for rho in np.unique(pconfInfoForPlot[:,2]):
		
				fig = plt.figure()
				ax = fig.add_subplot(111)

				indexes = np.where(pconfInfoForPlot[:,0] == x1)
				subsetInfo = pconfInfoForPlot[indexes]
				subsetPrats = pconfPratsForPlot[indexes]
				# code.interact("...", local=dict(globals(), **locals()))

				indexes = np.where(subsetInfo[:,1] == x2)
				subsetInfo = subsetInfo[indexes]
				subsetPrats = subsetPrats[indexes]

				indexes = np.where(subsetInfo[:,2] == rho)

				# print("subsetInfo: ",subsetInfo)
				# print("subsetPrats: ",subsetPrats)

				img = ax.scatter(subsetInfo[:,3][indexes],subsetPrats[indexes],c=cmap(norm(subsetPrats[indexes])))

				fig.colorbar(img,pad=0.12)

				ax.set_title(f"X1: {x1} | X2: {x2} | R: {rho}")
				ax.set_xlabel('Lambdas')
				ax.set_ylabel('Rationality Percentage')

				plt.savefig(f"pconf-X1-{x1}-X2-{x2}-R{rho}.png")

				plt.clf()
				plt.close()
				
				# code.interact("...", local=dict(globals(), **locals()))

	'''


	# code.interact("...", local=dict(globals(), **locals()))

	return info["pconfInfo"],info["pconfPrats"]


def getLambdaRangeGivenPercent(pconfPerXskill,xskill,percentBuckets):

	bucketsX = sorted(pconfPerXskill.keys())

	minMaxX = [bucketsX[0],bucketsX[-1]]

	allBuckets = []
	for key in pconfPerXskill:
	
		# buckets = 
		# minMaxX = 
		# x = 

		# Find bucket where xskill lies on
		# find proper bucket for current x
		b1,b2 = getBucket(bucketsX,minMaxX,x)


	diffs = []
	diff1 = abs(b1-xskill)

	if b2 != None:
		diff2 = abs(b2-xskill)

		if diff1 < diff2:
			closest = b1
		else:
			closest = b2
	else:
		closest = b1


	info = {}

	bb = 0

	# Lambdas are sorted
	for ip in range(len(pconfPerXskill[closest]["lambdas"])):
		
		#if bb > len(percentBuckets)-1:
		#	# Stop searching
		#	break

		if pconfPerXskill[closest]["prat"][ip] <= percentBuckets[bb]:
			continue
		else:
			info[bb] = [pconfPerXskill[closest]["lambdas"][ip-1],pconfPerXskill[closest]["prat"][ip-1]]
			bb += 1

	# Add info for last bucket
	# since prats wont be >= 1, else is never seens
	# if bb not in info:
	# 	info[bb] = [pconfPerXskill[closest]["lambdas"][ip],pconfPerXskill[closest]["prat"][ip]]

	# code.interact("()...", local=dict(globals(), **locals()))

	return info

def getPercentRationals(pconfPerXskill,minMaxX,xskills,pskills):

	bucketsX = sorted(pconfPerXskill.keys())

	percents = {}

	for x in xskills:

		percents[x] = []

		for p in pskills:

			# find proper bucket for current x
			bucket1, bucket2 = getBucket(bucketsX,minMaxX,x)

			# Convert estimatedP to corresponding % of rand max
			if bucket2 != None:
				prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
				prat2 = np.interp(p,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

				prat = np.interp(p, [prat1], [prat2])
				percent_estimatedP = prat
			# edges/extremes case
			else:
				# using one of the functions for now
				prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

				percent_estimatedP = prat1

			percents[x].append(percent_estimatedP)

	return percents

def getPercentRationalGivenParams(pconfPerXskill,minMaxX,x,p):

	bucketsX = sorted(pconfPerXskill.keys())

	percent = None

	# find proper bucket for current x
	bucket1, bucket2 = getBucket(bucketsX,minMaxX,x)

	# Convert estimatedP to corresponding % of rand max
	if bucket2 != None:
		prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
		prat2 = np.interp(p,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

		prat = np.interp(p, [prat1], [prat2])
		percent_estimatedP = prat
	# edges/extremes case
	else:
		# using one of the functions for now
		prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

		percent_estimatedP = prat1

	percent = percent_estimatedP

	return percent


if __name__ == '__main__':

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "2d-multi")
	parser.add_argument("-noWrap", dest = "noWrap", help = "Flag to disable wrapping action space in 1D domain.", action = 'store_true')
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()



	domainModule,delta = getDomainInfo(args.domain)

	
	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(rng,args.resultsFolder,args.domain,domainModule,spacesModule,args.mode,args)

	####################################



	code.interact("...", local=dict(globals(), **locals()))

