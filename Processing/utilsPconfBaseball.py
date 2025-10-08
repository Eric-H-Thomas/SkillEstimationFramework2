import pickle, os

from utilsDarts import *

if __name__ == '__main__':

	print("Loading pconf info...")

	# Assuming pconf file exists already
	try:
		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"
		tempName = "pconf-NumObs970.pickle"

		with open(saveAt+tempName, "rb") as file:
			pconfPerXskill = pickle.load(file)
	except:
		print("Pconf file not present. Need to compute first.")
		exit()




	bucketsX = sorted(pconfPerXskill.keys())
	minMaxX = [bucketsX[0],bucketsX[-1]]

	xSkills = [0.50,0.63,0.75,0.88,1.0]
	lambdas = np.linspace(0.001,1300,10)

	for xs in xSkills:
		
		print(f"x: {xs}")

		for p in lambdas:
			
			prat = getPercentRationalGivenParams(pconfPerXskill,minMaxX,xs,p)

			print(f"\tp: {p} | prat: {prat} ")


# @profile
def onlineExperiment(args,xskill,agents,env,estimatorsObj,subsetEstimators,numHypsX,numHypsP,tag,counter,seedNum,rng,indexOR,rerun=False):
	
	print("\nPerforming experiments...\n")

	# For each one of the different agents
	for agent in agents:

		tempRerun = rerun

		if args.domain == "billiards":
			label = f"Agent: {agent}"
			saveAt = f"{tag}-{counter}-Agent{agent}.results"
		elif args.domain == "baseball":
			label = f"Agent -> pitcherID: {agent[0]} | pitchType: {agent[1]}"
			saveAt = f"{tag}.results"
		else:
			label = f"Agent: {agent.name}"
			saveAt = f"{tag}-{counter}-Agent{agent.getName()}.results"
			
		resultsFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}results{os.sep}{saveAt}"
		
		if args.domain == "baseball":
			statusFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}{tag}-DONE"
		else:
			statusFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}{tag}-{counter}-Agent{agent.getName()}-DONE"


		expStartTime = time.time()
		

		# Experiment already done
		if Path(f"{statusFile}.txt").is_file() and not args.rerun and env.domainName == "baseball":
			print(f"Experiment for {label} was already performed and it finished successfully.")
		
			del env.spaces.allData

		# Proceed to perform experiment (full/reload/rerun modes)
		else:

			print(f"\n{label}")

			#To handle case mode rerun but prev rf for current agent doesn't exist
			if tempRerun:
				# Verify if prev rf file exists
				try:
					with open(resultsFile,'rb') as handle:
						resultsLoaded = json.load(handle)
				except:
					tempRerun = False


			if env.domainName in ["1d","2d"]:
				exp = RandomDartsExp(env.numObservations,args.mode,env,agent,xskill,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,args.allProbs,seedNum,rng,tempRerun)
			elif env.domainName == "sequentialDarts":
				exp = SequentialDartsExp(env.numObservations,args.mode,env,agent,xskill,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,args.allProbs,seedNum,rng,tempRerun)
			elif env.domainName == "billiards":
				exp = BilliardsExp(env.numObservations,env,agent,xskill,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,seedNum,tempRerun)
			elif env.domainName == "baseball":
				exp = BaseballExp(args,env,agent,estimatorsObj,args.resultsFolder,resultsFile,indexOR,seedNum)
			

			# Experiment valid, proceed to perform exp
			if exp.getValid():
			
				exp.run(tag,counter)

				expStopTime = time.time()
				expTotalTime = expStopTime-expStartTime


				if env.domainName != "baseball":

					# Load initial info from file 
					# OR load results from prev exp
					with open(resultsFile,'rb') as handle:
						resultsLoaded = json.load(handle)

					results = exp.getResults()
					results['expTotalTime'] = expTotalTime
					results['lastEdited'] = str(datetime.datetime.now())

					# Update dict info
					resultsLoaded.update(results)

					with open(resultsFile,'w') as outfile:
						json.dump(resultsLoaded,outfile)

				else:
					# Assuming results file already exist since created when saving initial info
					# Add just exp time to file since results are saved to file within run()
					with open(resultsFile,'r') as handle:
						results = json.load(handle)

					results['expTotalTime'] = expTotalTime

					with open(resultsFile,'w') as outfile:
						json.dump(results,outfile)
					

			# If done = true means experiment finished successfully, mark as finished.
			if exp.getStatus():

				# File will be empty. The fact that it exists indicates experiment finished successfully.
				outfile = open(f"{statusFile}.txt",'w')
				outfile.close()
				
				if tempRerun == True:
	print("\n\n")


	try:
		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"
		tempName = "pitchers.txt"

		with open(saveAt+tempName, "r") as file:
			pitchersInfo = file.readlines()
	except:
		print("Pitcher file not present.")
		exit()


	with open(f"{saveAt}pitchersPrat.csv","w") as outfile:
		for i in range(len(pitchersInfo)):

			splittedInfo = pitchersInfo[i].split(",")

			name = splittedInfo[0]
			xs = float(splittedInfo[1])
			p = float(splittedInfo[2])
			
			prat = getPercentRationalGivenParams(pconfPerXskill,minMaxX,xs,p)

			print(f"x: {xs} | p: {p} | prat: {prat}")

			print(f"{name},{xs},{p},{prat},{prat*100:.2f}",file=outfile)

	
	code.interact("...", local=dict(globals(), **locals()))