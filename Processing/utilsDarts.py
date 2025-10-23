import os,code
from pathlib import Path
from importlib.machinery import SourceFileLoader

import pickle,code

from copy import deepcopy

import numpy as np
# import matplotlib.pyplot as plt

import argparse

import sys

if "setupSpaces" not in sys.modules:

	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Environments/"
	mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.sep
	spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.sep)[0] + "setupSpaces.py").load_module()
	# print("Imported module")

else:
	spacesModule = sys.modules["setupSpaces"]


def sortTwoLists(aList,bList):
	# sorts aList and find parallel on bList to "sort" - put in the same/corresponding position
	# for example: lambas & estimates

	originalList = deepcopy(aList)

	aList.sort()
	
	tempList = []

	for each in range(len(aList)):

		# Find position of element sortedList[each] in original list
		i = originalList.index(aList[each])

		# Get element at position i from bList and store
		tempList.append(bList[i])

	# return new "sorted" lists
	return aList, tempList


def getParamsFromAgentName(a):  

	if "Bounded" in a:
		string = a.split("-L")
		aType = "Bounded"

	elif "Flip" in a:
		string = a.split("-P")
		aType = "Flip"

	elif "Tricker" in a:
		string = a.split("-Eps")
		aType = "Tricker"

	elif "TargetBelief" in a:
		string = a.split("-B")
		aType = "TargetBelief"

	elif "Target" in a:
		string = [a]
		aType = "Target"

	elif "Random" in a:
		string = a.split("-X")
		aType = "Random"

	#print("aType: ", aType)

	# Find pskill
	if aType == "Target":
		p = 100.0
	elif aType == "Random": 
		string2 = string[1].split("-N")[1]
		string3 = string2.split("-K")
		p = string3[0] + "/" + string3[1] 
	else:
		p = round(float(string[1]),4)

		# verify if number in scientific notation
		if "e" in str(p):
			# will truncate rest of decimal places
			# Just keep first 3
			# Splits on "e" to stay with first part
			# Convert back to number
			p =  float('{:0.3e}'.format(p).split("e")[0])


	# Find xskill
	string2 = string[0].split("-X")
	x = round(float(string2[1]),4)

	# Return info
	return aType, x, p


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


def getBetas(results,betas,typeTargetsList):

	for tempKey in results.keys():

		if "BM-EES" in tempKey and "Beta" in tempKey and "allProbs" not in tempKey:
			b = float(tempKey.split("Beta-")[1])
			if b not in betas:
				betas.append(b)

		if "BM" in tempKey:
			if "Optimal" in tempKey and "OptimalTargets" not in typeTargetsList:
				typeTargetsList.append("OptimalTargets")
			if "Domain" in tempKey and "DomainTargets" not in typeTargetsList:
				typeTargetsList.append("DomainTargets")


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

	if domainName == "1d":

		load = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
		
		if wrap:
			domainModule = SourceFileLoader("darts",load+"darts.py").load_module()
		else:	
			domainModule = SourceFileLoader("darts",load+"darts_no_wrap.py").load_module()
		
		delta = 1e-2

	elif domainName == "2d":		
		load = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
		domainModule = SourceFileLoader("two_d_darts",load+"two_d_darts.py").load_module()
		delta = 5.0
		
	elif domainName == "sequentialDarts":
		load = f"Environments{os.sep}Darts{os.sep}SequentialDarts{os.sep}"
		domainModule = SourceFileLoader("sequential_darts",load+"sequential_darts.py").load_module()
		delta = 5.0

	elif domainName == "baseball":
		load = f"Environments{os.sep}Baseball{os.sep}"
		domainModule = SourceFileLoader("baseball",load+"baseball.py").load_module()
		delta = 5.0

	return domainModule,delta


def getInfoBM(m):

	if "MAP" in m:
		tempM = "BM-MAP"
	else:
		tempM = "BM-EES"

	beta = float(m.split("Beta-")[1])

	if "Targets" in m:
		tt =  m.split("-Beta")[0].split("-")[-1]
	else:
		tt = "None"

	return tempM, beta, tt


def getBucket(bucketsX,minMaxX,xParam):

	# Find proper bucket for current x
	for b in range(len(bucketsX)):
		if xParam <= bucketsX[b]:
			break

	# Get actual bucket
	bucket1 = bucketsX[b]

	# Placeholder variable
	otherBucket = None
	
	bucket2 = None

	# First bucket
	if b == 0:
		# use left edge/extreme - i.e. 0
		otherBucket = minMaxX[0]
	# If last bucket
	elif b == len(bucketsX)-1:
		# use right edge/extreme - i.e. 5/100 depending on the domain
		otherBucket = minMaxX[1]
	# Somewhere in the middle - consider next bucket
	else:
		bucket2 = bucket1
		bucket1 = bucketsX[b-1]

	# print(f"B1: {bucket1} | B2: {bucket2}")
	return bucket1,bucket2


def getAgentInfoFromFileName(rf):
		
	splitted = rf.split("Agent")
	agentType = splitted[1]

	splitted = splitted[2].split(".results")[0][2:]


	aName = agentType

	if agentType == "Flip":
		string = splitted.split("-P")
		xStr = string[0]
		param = str(round(float(string[1]),4))
		aName += "-X" + xStr
		aName += "-P" + param

	elif agentType == "Tricker":
		string = splitted.split("-Eps")
		xStr = string[0]
		param = str(round(float(string[1]),4))
		aName += "-X" + xStr
		aName += "-Eps" + param

	elif agentType == "Bounded":
		string = splitted.split("-L")
		xStr = string[0]
		param = str(round(float(string[1]),4))
		aName += "-X" + xStr
		aName += "-L" + param

	elif agentType == "Target":
		string = splitted.split("-X")
		xStr = string[0]
		param = "100.00"
		aName += "-X" + xStr

	elif agentType == "TargetBelief":
		string = splitted.split("-TrueX")
		param = string[0] # beliefX
		xStr = str(round(float(string[1]),4)) # trueX
		aName += "-X" + xStr
		aName += "-B" + param

	elif agentType == "Random":
		string1 = splitted.split("-N")
		string2 = string1[1].split("-K")
		xStr = string1[0]
		param1 = string2[0]
		param2 = string2[1]
		aName += "-X" + xStr
		aName += "-N" + param1
		aName += "-K" + param2

		param = {"param1":param1,"param2":param2}

	return aName, agentType, xStr, param


def getAgentInfo(domainName,agentName):

	if domainName in ["1d","2d","sequentialDarts"]:

		if "TargetAgent-BeliefX" in agentName:
			aNameOriginal = str(agentName.split("Agent-BeliefX")[0]) + "Belief"
			params = str(agentName.split("Agent-BeliefX")[1])

		else:		
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

		elif aName == "TargetBelief":
			string = params.split("-TrueX")
			param = string[0] # beliefX
			xStr = string[1] # trueX
			aName += "-X" + xStr
			aName += "-B" + param

		elif aName == "Random":
			string1 = params.split("-N")
			string2 = string1[1].split("-K")
			xStr = string1[0]
			param1 = string2[0]
			param2 = string2[1]
			aName += "-X" + xStr
			aName += "-N" + param1
			aName += "-K" + param2

			param = {"param1":param1,"param2":param2}

	
	# code.interact("...", local=dict(globals(), **locals()))
	
	if aName == "Random":
		return aNameOriginal,aName,param
	else:
		return aNameOriginal,aName,float(param)


def loadProcessedInfo(f,a):

	with open(f"{f}-{a}","rb") as infile:
		tempInfo = pickle.load(infile)

	return tempInfo[a]


def updateProcessedInfo(f,a,resultsDict):

	with open(f"{f}-{a}","wb") as outfile:
		pickle.dump(resultsDict,outfile)


def pconf(rng,resultsFolder,domain,domainModule,spaceModule,mode,args,wrap):

	print("\n----------------------")
	
	print("PCONF: ")
	print("Domain: ", domain)
	
	if domain == "1d":
		print("Wrap = ",wrap)
	elif domain == "2d":
		print("Mode: ", mode)

	print("----------------------\n")


	numSamples = 1000

	mainFolder = "Spaces" + os.sep + "ExpectedRewards" + os.sep
	fileName = f"ExpectedRewards-{args.domain}-N{numSamples}"
	expectedRFolder = mainFolder + fileName


	pconfPerXskill = {}

	if "Exp" not in resultsFolder[0:3]:
		resultsFolder = "Experiments" + os.sep + resultsFolder

	if not os.path.exists(resultsFolder + os.sep + "plots" + os.sep):
		os.mkdir(resultsFolder + os.sep + "plots" + os.sep)


	tempName = resultsFolder + os.sep + "plots" + os.sep + "pconfInfo"

	if domain == "sequentialDarts":
		tempName += "-Values"


	# if file is not present, need to compute info
	if not os.path.exists(tempName):

		if domain == "1d" or domain == "2d":

			if domain == "1d":

				numSamples = 1000

				# All the lambdas that we will use to generate the plot
				lambdas = np.logspace(-5,2,100)
				# lambdas = np.linspace(0.001, 100, 100)

				# The xskills we want to have our predictions at
				# xskills = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
				xskills = np.linspace(0.5,15.0,num=10)

				args.resolution = 1e-1

				# Get the states to use for evaluation
				states = domainModule.generate_random_states(rng,3,10,numSamples)

			else: # 2D

				numSamples = 50

				# All the lambdas that we will use to generate the plot
				lambdas = np.logspace(-5,1.5,100)

				# The xskills we want to have our predictions at
				xskills = [5, 10, 30, 50, 70, 90, 110, 130, 150]

				# xskills = np.linspace(2.5,150.5,num=33)	

				args.resolution = 5.0

				# Get the states to use for evaluation
				states = domainModule.generate_random_states(rng,numSamples,args.mode)

		elif domain == "sequentialDarts":
			
			lambdas = np.logspace(-5,1.5,100)

			xskills = np.linspace(2.5,150.5,num=33)	

			args.resolution = 5.0
			
			startScore = domainModule.getPlayerStartScore()
			states = list(range(startScore + 1))

			args.N = 1


		print("\nCreating spaces...")

		if domain == "1d" or domain == "2d":
			spaces = spacesModule.SpacesRandomDarts(numSamples,domainModule,args.mode,args.resolution,numSamples,expectedRFolder)
			spaces.updateSpace(rng,xskills,states)

		elif domain == "sequentialDarts":
			spaces = spacesModule.SpacesSequentialDarts(numSamples,domainModule,args.mode,args.resolution,numSamples,expectedRFolder)
			spaces.updateSpace(xskills)

		print("\nDone spaces...")


		# Go through all of the execution skills
		for x in xskills:
			print('Generating data for execution skill level', x)

			prat = [] #This is where the probability of rational reward will be stored
			mins = [] #Store min reward possible
			maxs = [] #Store max reward possible
			means = [] #Store the mean of the possible rewards (this is the uniform random reward)
			evs = [] #Store the ev of the current agent's strategy

			if domain in ["1d","2d"]:
				space = spaces.convolutionsPerXskill[x]
			else:
				space = spaces.spacesPerXskill[x]


			if domain in ["1d","2d"]:
				size = len(states)
				loopInfo = states

			else:
				size = len(states)-2
				loopInfo = states[2:]


			for l in lambdas:     

				# Minus 2 to not include info for state 0 and 1
				# If initialize to 0, causes NANs
				max_rs = np.zeros(size)
				min_rs = np.zeros(size)
				exp_rs = np.zeros(size)
				mean_rs = np.zeros(size)

				si = 0
				
				for s in loopInfo:

					if domain == "1d":
						values = space[str(s)]["all_vs"]
					elif domain == "2d":
						values = space[str(s)]["all_vs"].flatten()
					else:
						values = np.copy(space.flatEVsPerState[s])


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

				
				prat.append(np.mean((exp_rs - mean_rs)/(max_rs - mean_rs)))
				mins.append(np.mean(min_rs))
				means.append(np.mean(mean_rs))
				maxs.append(np.mean(max_rs))
				evs.append(np.mean(exp_rs))

			# plt.plot(lambdas, prat, label='x=' + str(x))

			# store to use later
			pconfPerXskill[x] = {"lambdas":lambdas, "prat": prat}
			#code.interact("...",local=locals())

		# plt.xlabel('Lambda')
		# plt.ylabel('% Rational Reward')
		# plt.legend()
		# plt.show()
		# code.interact("v...", local=dict(globals(), **locals()))


		# Save dict containing all info - to be able to rerun it later
		with open(tempName,"wb") as outfile:
			pickle.dump(pconfPerXskill, outfile)


		if domain in ["1d","2d"]:
			spaces.convolutionsPerXskill.clear()
		else:
			space = spaces.spacesPerXskill.clear()

		
	# file with the info is present, proceed to load
	else:
		print("Loading pconf info...")

		with open(tempName, "rb") as file:
			pconfPerXskill = pickle.load(file)

	# code.interact("pconf end",local=locals())
	print("Finished pconf()")
	return pconfPerXskill


def printInfo(percents,pskills,wpt,wx,wp,chars):

	#print("\n\t\t\t\t  pskills")
	print("-"*chars)
	print(f"pskills |\t   ", end = "")
	
	for p in pskills:
		print(f"{p:>{wpt}.3f}", end = " | ")

	print()
	print("-"*chars)

	for x in percents:
		print(f"xskill: {x:>{wx}.3f} | ", end = "")

		for pr in percents[x]:
			print(f"{pr:>{wp}.4f}", end=" | ")
		
		print()
	
	print("-"*chars)


def getLambdaRangeGivenPercent(pconfPerXskill,xskill,percentBuckets):

	bucketsX = sorted(pconfPerXskill.keys())

	minMaxX = [bucketsX[0],bucketsX[-1]]

	# Find bucket where xskill lies on
	# find proper bucket for current x
	b1,b2 = getBucket(bucketsX,minMaxX,xskill)


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
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball")
	parser.add_argument("-noWrap", dest = "noWrap", help = "Flag to disable wrapping action space in 1D domain.", action = 'store_true')
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	wrap = True

	if args.noWrap:
		wrap = False



	oiFile = args.resultsFolder + os.sep + "otherInfo" 

	with open(oiFile,"rb") as file:
		otherInfo = pickle.load(file)

		domain = otherInfo["domain"]
		mode = otherInfo["mode"]
	

	domainModule,delta = getDomainInfo(args.domain,wrap)

	
	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args,wrap)

	bucketsX = sorted(pconfPerXskill.keys())

	minMaxX = [bucketsX[0],bucketsX[-1]]

	####################################

	if args.domain == "1d":
		# xskills = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
		xskills = np.round(np.linspace(0.5,20.0,num=10),4)

		pskills = np.linspace(-3.0,2.0,6)		

		wpt = 2
		wx = 7
		wp = 5

		chars = 70

	else: # 2D or Sequential-Darts
		xskills = np.linspace(2.5,150.5,num=33)	

		pskills = np.linspace(-3.0,1.5,10)

		wpt = 5
		wx = 7
		wp = 5

		chars = 108


			
	pskills = np.power(10,pskills) # Exponentiate

	percents = getPercentRationals(pconfPerXskill,minMaxX,xskills,pskills)

	printInfo(percents,pskills,wpt,wx,wp,chars)


	##################################################################################
	
	'''
	logFlag = input("Pskills in log terms or not? (enter t/f)")
	givenPskills = input("Enter the pskills of interest (separated by commas):").split(",")


	for i in range(len(givenPskills)):
		givenPskills[i] = float(givenPskills[i])

	if logFlag == "t":
		givenPskills = np.power(10,givenPskills) # Exponentiate


	print("\n\n")
	percentsTake2 = getPercentRationals(pconfPerXskill,minMaxX,xskills,givenPskills)
	printInfo(percentsTake2,givenPskills,wpt,wx,wp,chars)
	'''

	##################################################################################


	percentBuckets = [0.25,0.50,0.75,1.0]
	xs = 3.0

	# info[bucket] = [lambda,prat]
	info = getLambdaRangeGivenPercent(pconfPerXskill,xs,percentBuckets)


	code.interact("...", local=dict(globals(), **locals()))

