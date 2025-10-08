from utilsDarts import *



def updatePercentRationality(rdFile,numStates):
	
	resultsDict = {}


	for ee in range(len(processedRFsAgentNames)):
		a = processedRFsAgentNames[ee]

		print(f"({ee+1}/{len(processedRFsAgentNames)}) - Agent: {a}")

		aType, xStr, p = getParamsFromAgentName(a)
		

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(rdFile,a)

		# For each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA
			if "pSkills" not in m:
				continue


			# To determine whether to use JT's or NJT's current xskill estimate 
			if "NJT" in m:
				mm = "NJT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"
			else:
				mm = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"


			resultsDict[a]["percentsEstimatedPs"][m][resultsDict[a]["num_exps"]] = [0.0] * numStates
			resultsDict[a]["percentsEstimatedPs"][m]["averaged"] = [0.0] * numStates
		

			# for each state
			for mxi in range(numStates):

				
				# Use estimated xskill and not actual true one
				# WHY? estimatedX and not trueX?? because "right" answer is not available
				# xStr = float(resultsDict[a]["plot_y"][mm][mxi])
				xStr = float(resultsDict[a]["estimates"][mm][mxi])

				# find proper bucket for current x
				bucket1, bucket2 = getBucket(bucketsX,minMaxX,xStr)


				# Get pskill estimate of current method - estimatedP
				estimatedP = resultsDict[a]["estimates"][m][mxi]


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
				resultsDict[a]["percentsEstimatedPs"][m][resultsDict[a]["num_exps"]][mxi] = percent_estimatedP


		# Update info on file
		updateProcessedInfo(rdFile,a,resultsDict)


		del resultsDict[a]

if __name__ == '__main__':
	
	# ASSUMES RESULTS OF EXPERIMENTS WERE PROCESSED ALREADY


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "1d")	
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	
	args = parser.parse_args()


	rdFile = args.resultsFolder + os.sep + "ProcessedResultsFiles" + os.sep + "resultsDictInfo"
	oiFile = args.resultsFolder + os.sep + "otherInfo" 


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

	try:
		actualProcessedRFs = os.listdir(args.resultsFolder + os.sep + "ProcessedResultsFiles")
	except:
		print("Folder for processed results files not present.\nNeed to process results files first.")
		exit()


	if len(actualProcessedRFs) == 0:
		print("Need to process results files first.")
		exit()



	domainModule,delta = getDomainInfo(domain,wrap)

	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args,wrap)

	####################################


	bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		minMaxX = [bucketsX[0],bucketsX[-1]]#[0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]



	updatePercentRationality(rdFile,numStates)


