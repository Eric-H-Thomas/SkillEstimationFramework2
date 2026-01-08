"""Run hockey experiments with optional particle filter estimators.

This script wires together environment setup, estimator creation, and the
iteration loop that executes experiments for a set of agents. It is invoked as
its own executable. A high-level overview of what lives here:

Functions
---------
* ``work``: Small wrapper that executes an experiment in a separate process
  when multiprocessing is enabled and returns the produced results.
* ``onlineExperiment``: Core routine that iterates through agents and runs the
  selected experiment class for each one, handling persistence and status
  tracking.
* ``createEstimators``: Builds the estimator objects used in experiments and
  records how long construction takes.
* ``main``: Entry point that parses command-line arguments, prepares
  directories, configures estimators and the environment, and executes the
  experiments.

Integration notes
-----------------
The ``main`` function ties together utilities across the repository: environment
initialization from :mod:`setupEnv`, experiment classes from :mod:`expTypes`,
and estimator implementations in ``Estimators``. Results and metadata are stored
under the ``Experiments/`` directory so downstream processing scripts can read
them.
"""

import numpy as np
import os, time, datetime
import pickle
import argparse
import code
from itertools import product

from setupEnv import *
from expTypes import *

from Estimators.estimators import *
from Estimators.utils_pfe import *

# from memory_profiler import profile
from gc import collect
from datetime import datetime

from multiprocess import Process, Queue


'''
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Determines the rank of the calling process in the communicator
# Using as id?
rank = comm.Get_rank()

# Returns the number of processes
num = comm.Get_size()
'''


def work(exp,tag,counter,domainName,q):
        """Execute a single experiment inside a worker process."""

        # Run the experiment with the provided tag/counter metadata so the
        # results files can be uniquely identified on disk.
        exp.run(tag,counter)

        # Non-baseball/soccer domains return results via the queue so the parent
        # process can persist them to disk.
        if domainName not in ["baseball","baseball-multi","soccer"]:
                q.put(exp.get_results())


# @profile
def onlineExperiment(args,xskill,agents,env,estimatorsObj,subsetEstimators,tag,counter,seedNum,rng,indexOR,rerun=False):
        """Execute an experiment run for each agent in the collection.

        This routine selects the correct experiment class based on the configured
        domain, handles persistence of results, and records execution time. When
        ``rerun`` is True the function attempts to reload prior results files
        before running anew.
        """

        print("\nPerforming experiments...\n")

        rngMain = rng
        # print("MAIN: ", rngMain.bit_generator._seed_seq.entropy)

        if args.testingBounded:
                agentInfo,agents = agents[0], agents[1]
        else:
                agentInfo = None


        originalTag = tag


        # For each one of the different agents
        for agent in agents:

                # Reset RNG so each agent uses the same seed for reproducibility
                # within the iteration.
                rng = np.random.default_rng(seedNum)
                # print("(): ",agent, rng.bit_generator._seed_seq.entropy)

                tempRerun = rerun

                # Build human-friendly labels and output filenames per domain.
                if args.domain == "billiards":
                        label = f"Agent: {agent}"
                        saveAt = f"{tag}-{counter}-Agent{agent}.results"
                elif args.domain in ["baseball","baseball-multi"]:
                        label = f"Agent -> pitcherID: {agent[0]} | pitchType: {agent[1]}"
                        saveAt = f"{tag}.results"
                elif args.domain in ["hockey-multi"]:
                        if args.testingBounded:
                                label = f"Agent -> Player: {agentInfo[0]} | Shot Type: {agentInfo[1]}\nAgent: {agent.getName()}"

                                tag = originalTag + f"_Agent{agent.getName()}"

                                if args.jeeds:
                                        tag += "_JEEDS"

                                if args.pfe:
                                        tag += "_PFE"

                                if args.pfeNeff:
                                        tag += "_PFE_NEFF"

                                saveAt = f"{tag}.results"
                        else:
                                label = f"Agent -> player: {agent[0]} | shot type: {agent[1]}"
                                saveAt = f"{tag}.results"
                elif args.domain == "soccer":
                        label = f"Agent -> playerID: {agent}"
                        saveAt = f"{tag}.results"
                else:
                        label = f"Agent: {agent.name}"
                        saveAt = f"{tag}-{counter}-Agent{agent.getName()}.results"
			
                # Determine where to persist output for this agent and the status
                # flag file used to detect completed runs.
                resultsFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}results{os.sep}{saveAt}"

                if args.domain in ["baseball","baseball-multi","hockey-multi","soccer"]:
                        statusFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}{tag}-DONE"
                else:
                        statusFile = f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}{tag}-{counter}-Agent{agent.getName()}-DONE"


		expStartTime = time.time()
		

		# # Experiment already done
		# if Path(f"{statusFile}.txt").is_file() and not args.rerun and (env.domainName in ["baseball","baseball-multi"]):
		# 	print(f"Experiment for {label} was already performed and it finished successfully.")
		
		# 	del env.spaces.allData

		# # Proceed to perform experiment (full/reload/rerun modes)
		# else:

                # Display which agent configuration is currently running.
                print(f"\n{label}")

		# To handle case mode rerun but prev rf for current agent doesn't exist
                if tempRerun:
                        # Verify if a previous results file exists; if not,
                        # rerun behaves like a fresh run.
                        try:
                                with open(resultsFile,'rb') as handle:
                                        resultsLoaded = pickle.load(handle)
                        except:
                                tempRerun = False


		if env.domain_name in ["1d", "2d", "2d-multi"]:
			exp = RandomDartsExp(env.numObservations, args.mode, env, agent, xskill, estimatorsObj, subsetEstimators, args.resultsFolder, resultsFile, indexOR, args.probs_history, seedNum, rng, tempRerun)
		elif env.domain_name == "sequentialDarts":
			exp = SequentialDartsExp(env.numObservations, args.mode, env, agent, xskill, estimatorsObj, subsetEstimators, args.resultsFolder, resultsFile, indexOR, args.probs_history, seedNum, rng, tempRerun)
		elif env.domain_name == "billiards":
			exp = BilliardsExp(env.numObservations,env,agent,xskill,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,seedNum,rng,tempRerun)
		elif env.domain_name in ["baseball", "baseball-multi"]:
			exp = BaseballExp(args,env,agent,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,seedNum,rng)
		elif env.domain_name in ["hockey-multi"]:
			exp = HockeyExp(args,env,[agentInfo,agent],estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,seedNum,rng)
		elif env.domain_name == "soccer":
			exp = SoccerExp(args,env,agent,estimatorsObj,subsetEstimators,args.resultsFolder,resultsFile,indexOR,seedNum,rng)
		

		# Experiment valid, proceed to perform exp
                # Proceed only when the experiment instance is properly
                # configured (valid dataset, states, etc.).
                if exp.getValid():

                        if env.domain_name in ["baseball", "baseball-multi", "hockey-multi", "soccer"]:
                                # Some experiment classes handle their own
                                # persistence internally, so simply run them
                                # inline and use an empty results dict here.
                                # exp.run(tag,counter,num,comm)
                                exp.run(tag,counter)
                                results = {}

                        else:
                                # For darts-like domains run the experiment in a
                                # separate process to keep memory usage contained
                                # and avoid interference between runs.
                                q = Queue()

                                process = Process(target=work, args=(exp, tag, counter, env.domain_name, q))

                                # print(f"ID of parent process: {os.getpid()}")

                                # Start the process
                                process.start()

                                results = q.get()

                                # Wait for the process to finish
                                process.join()


			expStopTime = time.time()
			expTotalTime = expStopTime-expStartTime


                        if env.domain_name not in ["baseball", "baseball-multi", "hockey-multi", "soccer"]:

                                # Load initial info from file
                                # OR load results from prev exp
                                with open(resultsFile,'rb') as handle:
                                        resultsLoaded = pickle.load(handle)

                                # Attach timing metadata and persist back to disk.
                                results['expTotalTime'] = expTotalTime
                                results['lastEdited'] = str(datetime.now())

                                # Update dict info
                                resultsLoaded.update(results)

                                with open(resultsFile,'wb') as outfile:
                                        pickle.dump(resultsLoaded,outfile)

                                del resultsLoaded

                        else:
                                # Assuming results file already exist since created when saving initial info
                                # Add just exp time to file since results are saved to file within run()
                                with open(resultsFile,'rb') as handle:
                                        results = pickle.load(handle)

                                results['expTotalTime'] = expTotalTime

                                with open(resultsFile,'wb') as outfile:
                                        pickle.dump(results,outfile)
				
				del results

			print(f"Total time for experiment: {expTotalTime:.4f}")

		# If done = true means experiment finished successfully, mark as finished.
		if exp.getStatus():

			# File will be empty. The fact that it exists indicates experiment finished successfully.
			outfile = open(f"{statusFile}.txt",'w')
			outfile.close()
			
			if tempRerun == True:
				# Create status file for rerun mode too
				currentFiles = os.listdir(f"Experiments{os.sep}{args.resultsFolder}{os.sep}status{os.sep}")
				statusFile += f"-RERUN-{len(currentFiles)}" 

				# File will be empty. The fact that it exists indicates experiment finished successfully.
				outfile = open(f"{statusFile}.txt",'w')
				outfile.close()

		# Increment counter
		counter += 1

		del exp

		# Call garbage collector
		collected = collect()
		# print(f"Garbage collector: collected {collected} objects.")

		# code.interact("...", local=dict(globals(), **locals()))

	return counter


def createEstimators(args,infoForEstimators,env):
        """Instantiate all requested estimators and log creation time."""

        print("\nCreating estimators...")

        creationStart = time.time()

        estimators = Estimators(infoForEstimators,env)

        # code.interact("...", local=dict(globals(), **locals()))

        creationStop = time.time()

        print("Done creating the estimators.\n")

        # store the time taken to create estimators to txt file
        with open("Experiments" + os.sep + args.resultsFolder  + os.sep + "times" + os.sep + f"timeToCreateEstimators{args.seedNum}.txt", "w") as file:
                file.write("\n\nTime to create all the estimators: "+str(creationStop - creationStart))

        return estimators


# @profile
def main():
        """Script entry point that prepares configuration and executes runs."""

        # gc.set_debug(gc.DEBUG_LEAK)


	# USAGE:  python runExp.py -resultsFolder Results -domain 1d

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Compute the execution skill for agent given specific number of shots')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Enter the name of the folder to store the experiments in", type = str, default = "Results")
	
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "hockey-multi")
	parser.add_argument("-delta", dest = "delta", help = "Delta = resolution to use when doing the convolution", type = float, default = 1e-2)
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "rand_pos")

	parser.add_argument("-seed", dest = "seedNum", help = "Enter a seed number", type = int, default = -1)
	parser.add_argument("-folderSeedNums", dest = "folderSeedNums", help = "Enter the name of the folder containing the desired seedNums file to rerun/read from", type = str, default = "test")
	parser.add_argument("-rerun", dest = "rerun", help = "Flag to rerun exps.", action = 'store_true')
	parser.add_argument("-noWrap", dest = "noWrap", help = "Flag to disable wrapping action space in 1D domain.", action = 'store_true')

	parser.add_argument("-someAgents", dest = "someAgents", help = "Flag to run exps with only a subset of the agents.", action = 'store_true')

	parser.add_argument("-saveStates", dest = "saveStates", help = "To enable the saving of the different set of states used within the experiments. \
																	Number will be included on file name (needed in case of multiple terminals)", type = int, default = 0)
	parser.add_argument("-allProbs", dest = "allProbs", help = "Flag to enable saving allProbs - probabilities per state - of the BoundedAgent. Info is needed for verifyBoundedAgenPlots.", action = 'store_true')
	parser.add_argument("-plotState", dest = "plotState", help = "Flag to enable plotting of states with agent's info (actions & rewards)", action = 'store_true')


	parser.add_argument("-iters", dest = "iters", help = "Enter the number of iterations to perform", type = int, default = 10)
	parser.add_argument("-numObservations", dest = "numObservations", help = "Enter the number of observations to use for each experiment.", type = int, default = 100)
	

	parser.add_argument("-xSkillsGiven", dest = "xSkillsGiven", help = "Flag to enable the use of given params (not rand ones).", action = 'store_true')
	parser.add_argument("-pSkillsGiven", dest = "pSkillsGiven", help = "Flag to enable the use of given params (not rand ones).", action = 'store_true')
	parser.add_argument("-numXskillsPerExp", dest = "numXskillsPerExp", help = "Enter the number of xskills to use per experiment.", type = int, default = 3)
	parser.add_argument("-numPskillsPerExp", dest = "numPskillsPerExp", help = "Enter the number of pskills to use per experiment.", type = int, default = 3)
	parser.add_argument("-numRhosPerExp", dest = "numRhosPerExp", help = "Enter the number of rhos to use per experiment.", type = int, default = 3)


	# FOR DARTS MULTI DOMAIN
	parser.add_argument('-agent', dest = 'agent', nargs='+', help='Specify the xskill param (per dimension / assuming 2 dimensions) to use for the agent',default = [])


	# FOR BASEBALL DOMAIN
	parser.add_argument("-startYear", dest = "startYear", help = "Desired start year for the data.", type = str, default = "2021")
	parser.add_argument("-endYear", dest = "endYear", help = "Desired end year for the data.", type = str, default = "2021")
	parser.add_argument("-startMonth", dest = "startMonth", help = "Desired start month for the data.", type = str, default = "01")
	parser.add_argument("-endMonth", dest = "endMonth", help = "Desired end month for the data.", type = str, default = "12")		
	parser.add_argument("-startDay", dest = "startDay", help = "Desired start day for the data.", type = str, default = "01")
	parser.add_argument("-endDay", dest = "endDay", help = "Desired end day for the data.", type = str, default = "31")
	parser.add_argument('-ids', dest = 'ids', nargs='+', help='List of pitcher IDs to use',default = [])
	parser.add_argument('-types', dest = 'types', nargs='+', help='List of pitch types to use',default = [])
	parser.add_argument('-every', dest = 'every', help='Create checkpoints and reset info every X number of observations.',type = int,default = 20)
	parser.add_argument("-maxRows", dest = "maxRows", help = "Max number of most recent rows to select from data.",  type = int, default = 1000)
	parser.add_argument("-reload", dest = "reload", help = "Flag to reload row info from prev exps.", action = 'store_true')
	parser.add_argument("-dataBy", dest = "dataBy", help = "Flag to specify how to get/filter the data by (recent,chunks,pitchNum).", type = str, default = "recent")
	parser.add_argument("-b1", dest = "b1", help = "Specify start of bucket.", type = int, default = 1)
	parser.add_argument("-b2", dest = "b2", help = "Specify end of bucket.", type = int, default = 2)


	# FOR HOCKEY DOMAIN
	parser.add_argument('-id', dest = 'id', help='Id of player to use',type = str, default = "1")
	parser.add_argument('-type', dest = 'type', type=str, help='Shot type to use',default = "")


	# FOR PFE
	parser.add_argument('-numParticles', dest = 'numParticles', nargs='+', help='List containing the different number of particles to test',default = [])
	parser.add_argument("-particles", dest = "particles", help = "Flag to enable the use of estimators with particle filter.", action = 'store_true')
	parser.add_argument("-resampleNEFF", dest = "resampleNEFF", help = "Flag to enable resampling based on the NEFF threshold", action = 'store_true')
	parser.add_argument("-resample", dest = "resample", help = "Specify the percent to use for the resampling",nargs='+', default = [])
	parser.add_argument('-noise', dest = 'noise', nargs='+', help='List of noises to use (W/noise)',default = [])
	parser.add_argument('-resamplingMethod', dest = 'resamplingMethod', help='Specify the resampling method to use for the resample step of the PF (Default = numpy choice).',type = str,default = "numpy")

	# For Dynamic Agents
	parser.add_argument("-dynamic", dest = "dynamic", help = "Flag to enable agents to have dynamic xskill", action = 'store_true')
	

	# FOR ESTIMATORS
	parser.add_argument("-jeeds", dest = "jeeds", help = "Flag to enable the use of jeeds estimator only.", action = 'store_true')
	parser.add_argument("-pfe", dest = "pfe", help = "Flag to enable the use of given pfe estimator only.", action = 'store_true')
	parser.add_argument("-pfeNeff", dest = "pfeNeff", help = "Flag to enable the use of pfe estimators (with neff sampling) only.", action = 'store_true')


	args = parser.parse_args()



	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# HARDCODING PARAMS FOR CLUSTER EXPS
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	args.testingBounded = False #True

	# args.xSkillsGiven = True
	# args.pSkillsGiven = True
	# args.numXskillsPerExp = 1
	# args.numPskillsPerExp = 3

	# Seed Num 
	args.seedNum = -1



	args.domain = "hockey-multi"
	# args.maxRows = 100
	args.every = 20
	args.iters = 1


	args.numParticles = [1000]
	# args.numParticles = [33] # JEEDS via PFE

	args.particles = True # False


	# if args.jeeds:
		# args.particles = False

	
	args.jeeds = True

	args.pfeNeff = True
	args.resampleNEFF = True
	# args.resamplingMethod = "numpy"
	# print(args.resamplingMethod)


	# if args.pfeNeff:
	# 	args.resampleNEFF = True
	# else:
	# 	args.resampleNEFF = False
	# '''


	if args.testingBounded:
		# args.resultsFolder = f"Experiment-Player{args.id}-{args.type}-TestingBounded"
		args.resultsFolder = f"JEEDS&JEEDSviaPFE-BoundedAgent"
	else:
		# args.resultsFolder = f"Experiment-Player{args.id}-{args.type}"
		tempInfo = datetime.now()
		args.resultsFolder = f"{tempInfo.year}-{tempInfo.month}-{tempInfo.day}"
		# args.resultsFolder = "2025-1-20"

		# args.resultsFolder = f"Players-PskillNoise10"
		# args.resultsFolder = f"Players-PskillNoise25"
		args.resultsFolder = f"Players-PskillNoChangeTake2"


		# args.resultsFolder = f"Players-Noise200-Take2"
		# args.resultsFolder = "Distribution-Student-t"
		# args.resultsFolder = "JEEDS&JEEDSviaPFE-ClusterRerun"


	# For testing purposes
	useEstimators = True

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initial Setup
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	rf = f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}Experiment{os.sep}"

	folders = ["Experiments",f"Experiments{os.sep}{args.domain}",f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}",rf,
				f"{rf}results{os.sep}","Data", f"Spaces",
				f"Spaces{os.sep}ExpectedRewards",f"{rf}status{os.sep}",
				f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}"]


	if "hockey" in args.domain:
		args.resultsFolder += os.sep + "Experiment"


	folders.append(f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}times{os.sep}")
	folders.append(f"Experiments{os.sep}{args.domain}{os.sep}{args.resultsFolder}{os.sep}times{os.sep}experiments")


	if args.domain == "sequentialDarts":
		folders += [f"Spaces{os.sep}ValueFunctions"]


	if args.domain == "billiards":
		f = f"Data{os.sep}{args.domain.capitalize()}{os.sep}"
		folders += [f"{f}ProcessedShots",
					f"Spaces{os.sep}SuccessRates"]


	if args.domain in ["baseball","baseball-multi","hockey-multi"]:
		f = f"Data{os.sep}{args.domain.capitalize()}{os.sep}"
		f2 = f"{rf}plots{os.sep}"

		folders += [f"{rf}results{os.sep}",
				f"{rf}info{os.sep}",f,f2,
				f"{rf}results{os.sep}backup{os.sep}",f"{rf}info{os.sep}backup{os.sep}"]

		if args.domain not in ["hockey-multi"]:
			folders += [f"{f2}StrikeZoneBoards{os.sep}",
				f"{f2}StrikeZoneBoards{os.sep}PickleFiles{os.sep}",
				f"{f}Models{os.sep}",
				f"{f}StatcastData{os.sep}"]

	if args.domain == "soccer":
		folders += [f"Data{os.sep}Soccer{os.sep}",f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}",
					f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}Dataset{os.sep}"]


        for folder in folders:
                # Ensure required directories exist before writing any files.
                if not os.path.exists(folder):

                        try:
                                os.mkdir(folder)
                        # To catch file exists error
                        # Will continue as assuming other process already create it
                        except FileExistsError:
                                continue

	del folders

	args.resultsFolder = args.domain + os.sep + args.resultsFolder
	

        # Record high-level metadata about the experiment run for later review.
        with open(f"Experiments{os.sep}{args.resultsFolder}{os.sep}info.txt","w") as outfile:
                print(f"Experiment Started: {datetime.now()}",file=outfile)
                print(f"Model Used: Version 1.Pending",file=outfile)



	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# FOR ESTIMATORS
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	# All available estimators
	# namesEstimators = ["OR","BM","JT-FLIP","JT-QRE","NJT-QRE"]
	# namesEstimators = ["JT-QRE"]
	# namesEstimators = ["BM"]
	# namesEstimators = ["BM","JT-QRE"]
	namesEstimators = []


	if args.jeeds:
		namesEstimators = ["JT-QRE"]


	if args.rerun:
		subsetEstimators = ["JT-QRE"]

		outfile = open(f"{rf}rerunStartTime{args.seedNum}.txt",'w')
		outfile.write(str(datetime.now()))
		outfile.close()

	else: 
		subsetEstimators = []


	# For TBA/BM Method
	# numBetas = 5 #50
	#betas = np.round(np.linspace(0.001,0.99,num=numBetas),4)
	betas = [0.50,0.75,0.85,0.90,0.95,0.99]

	########################################
	# NEED TO SET BETA ACCORDINGLY!
	########################################

	'''
	if args.domain == "billiards":
		betas = [0.55]
	elif args.domain == "baseball":
		betas = [0.99]
	else:
		betas = [0.99]
	'''
	########################################


	# Make number of hypothesis be (2n - 1) - to be subsets
	# (To help with convolutions info)

	if args.domain == "1d":
		# numHyps = [5,9,17,33,65]
		numHypsX = [17]
		# numHypsX = [33]
		numHypsP = [33]

		startX_Estimator = 0.25 
		stopX_Estimator = 15.0 #5.0


	elif args.domain == "2d" or args.domain == "sequentialDarts":
		numHypsX = [33]
		numHypsP = [33]

		startX_Estimator = 2.5
		stopX_Estimator = 150.5


		if args.particles:

			# [xi,xj,rho,p]
			s = [2.5,2.5,-0.75,-3]
			e = [150.5,150.5,0.75,1.5]


	elif args.domain == "2d-multi":

		# Row = different # of hyps to test
		# Columns = # of hyps per xskill dimension
		# numHypsX = [[45,45]]
		# numHypsP = [45]
		numHypsX = [[33,33]]
		numHypsP = [33]

		# Using single one (not list) for now
		# Make it an odd number so that 0.0 is included!
		# only used for JT-QRE-Multi (not for nom)
		numHypsRhos = 33

		# Range per dimension
		startX_Estimator = [2.5,2.5]
		stopX_Estimator = [150.5,150.5]


		if args.particles:

			# [xi,xj,rho,p]
			s = [2.5,2.5,-0.75,-3]
			e = [150.5,150.5,0.75,1.5]


	elif args.domain == "billiards":
		numHypsX = [2] #[17]
		numHypsP = [2] #[17]

		# From previous exps
		startX_Estimator = 0.010
		stopX_Estimator = 0.9

		namesEstimators = ["OR","BM","JT-QRE"]


	elif args.domain == "baseball":
		numHypsX = [66]
		numHypsP = [66]

		# 2.0 inches | 0.17 feet
		startX_Estimator = 0.17
		# 33.72 inches | 2.81 feet
		stopX_Estimator = 2.81

		# No JTM-FLIP
		# namesEstimators = ["OR","BM",
		# 				"JT-QRE","JT-QRE-GivenPrior","JT-QRE-MinLambda","JT-QRE-GivenPrior-MinLambda",
		# 				"NJT-QRE","NJT-QRE-GivenPrior","NJT-QRE-MinLambda","NJT-QRE-GivenPrior-MinLambda"]

		minLambda = [1.3, 1.7] # 1.3 -> 19.9526 since in logspace
		givenPrior = [[8,0.4,1.0]] # [0] = 'a' parameter, [1] = mean, [2] = cov
		otherArgs = {"minLambda": minLambda,"givenPrior": givenPrior}

	
	elif args.domain == "baseball-multi":
		numHypsX = [[66,66]]
		numHypsP = [66]

		# 2.0 inches | 0.17 feet
		startX_Estimator = 0.17
		# 33.72 inches | 2.81 feet
		stopX_Estimator = 2.81


		minLambda = [1.3, 1.7] # 1.3 -> 19.9526 since in logspace
		givenPrior = [[8,0.4,1.0]] # [0] = 'a' parameter, [1] = mean, [2] = cov
		otherArgs = {"minLambda": minLambda,"givenPrior": givenPrior}

		# Using single one (not list) for now
		# Make it an odd number so that 0.0 is included!
		numHypsRhos = 33 #33

		# Range per dimension
		startX_Estimator = [0.17,0.17]
		stopX_Estimator = [2.81,2.81]

		if args.particles:
			# [xi,xj,rho,p]
			s = [0.17,0.17,-0.75,-3]
			e = [2.81,2.81,0.75,1.5]


	elif args.domain == "hockey-multi":
		numHypsX = [[33,33]]
		numHypsP = [33]

		# Radians
		startX_Estimator = 0.004
		stopX_Estimator = np.pi/4


		# Using single one (not list) for now
		# Make it an odd number so that 0.0 is included!
		numHypsRhos = 33 #33

		# Range per dimension
		startX_Estimator = [0.004,0.004]
		stopX_Estimator = [np.pi/4, np.pi/4]

		if args.particles:
			# [xi,xj,rho,p]
			s = [0.004,0.004,-0.75,-3]
			e = [np.pi/4,np.pi/4,0.75,1.6]


	elif args.domain == "soccer":
		numHypsX = [2] #[17]
		numHypsP = [2] #[17]

		# TODO: NEED TO SET ACCORDINGLY
		startX_Estimator = 0.010
		stopX_Estimator = 0.9

		namesEstimators = ["OR","BM","JT-QRE"]



	# Info for Particle Filter Estimators (independent of domain)
	if args.particles:

		namesEstimators.append("JT-QRE-Multi-Particles")

		ranges = {"start":s, "end":e}


		if args.numParticles == []:
			# numParticlesList = [200,500,1000,2000]
			numParticlesList = [1000]
		else:
			numParticlesList = [int(ii) for ii in args.numParticles]


		if args.noise == []:
			# noises = [5,50,500]
			# noises = [50,500]

			# [[xskill,pskill]]
			# noises = [[200,10]]
			# noises = [[200,25]]
			noises = [[200,200]]
			
			# noises = [-1] # JEEDS via PFE
		else:
			noises = [int(ii) for ii in args.noise]


		if args.resample == []:
			# percents = [0.75,0.80,0.90]
			# percents = [0.75,0.90]
			percents = [0.90]
			# percents = [-1] # JEEDS VIA PFE
		else:
			percents = [float(ii) for ii in args.resample]



	infoForEstimators = {"namesEstimators":namesEstimators}


	infoForEstimators["numHypsX"] = numHypsX
	infoForEstimators["numHypsP"] = numHypsP
	infoForEstimators["startX_Estimator"] = startX_Estimator
	infoForEstimators["stopX_Estimator"] = stopX_Estimator
	infoForEstimators["betas"] = betas


	if args.domain in ["baseball","baseball-multi","hockey"]:
		infoForEstimators["otherArgs"] = otherArgs


	if args.domain in ["2d-multi","baseball-multi","hockey-multi"]:
		infoForEstimators["numHypsRhos"] = numHypsRhos

	if args.particles:
		infoForEstimators["diffNs"] = numParticlesList
		infoForEstimators["noises"] = noises
		infoForEstimators["percents"] = percents
		infoForEstimators["resampleNEFF"] = args.resampleNEFF
		infoForEstimators["ranges"] = ranges
		infoForEstimators["resamplingMethod"] = args.resamplingMethod
	
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # FOR EXPERIMENTS
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # To ensure seeds generated later on are different
        if args.seedNum == -1:
                args.seedNum = np.random.randint(0,100000,1)[0]


        # TESTING
        args.seedNum = 62912

        # Seed numpy for reproducibility across runs and generate per-iteration
        # seeds that will drive individual experiments.
        np.random.seed(args.seedNum)
        print(f"Seed set to: {args.seedNum}")


        # Generate seeds to use based on main seed
        seeds = np.random.randint(0,1000000,args.iters)
        print("Seeds: ",seeds)

	# FOR TESTING
	# seeds[0] = 5515
	# seeds[0] = 11149



	print("\nSetting environment...")

	env = Environment(args)


	if args.domain in ["baseball","baseball-multi","soccer"]:
		agents = env.agentGenerator.getAgents()



	# How many experiments to do?
	if args.domain not in ["baseball","baseball-multi","soccer"]:
		numExperiments = args.iters
	# For baseball domain - Will do as many experiments as agents desired
	else:
		numExperiments = len(agents)



	# code.interact("...", local=dict(globals(), **locals()))

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Create the estimators
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	estimators = createEstimators(args,infoForEstimators,env)

	if not useEstimators:
		# No estimators - Use when we don't want the estimators to produce estimates
		# Just let agent plan, take actions and obtain rewards
		estimators.estimators = []


	indexOR = None

	if useEstimators:
		tempEstimators = estimators.getCopyOfEstimators()

		# Find index of OR method (if present)
		# Do search only once since estimators always in the same order
		for e in range(len(tempEstimators)):
			if isinstance(tempEstimators[e],ObservedReward):
				indexOR = e

	
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# MEMORY MANAGEMENT
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	del namesEstimators

	if not args.particles:
		del startX_Estimator
		del stopX_Estimator

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# ONLINE EXPERIMENT
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	# To help differentiate between experiments of the same iteration
	# When xskill is random - it is okay since xskill are different each time
	# But when xSkillsGiven = true, xskills are the same every time per iteration
	# Will be included in the tag (tag = other info + iter + counter)
	counter = 0

	timesPerExps = []


	overallStart = time.time()


	# For each one of the desired number of experiments
	for i in range(numExperiments):

		print(f"{'-'*30} START ITERATION {i} {'-'*30}")

		expStart = time.time()
		
		seedNum = int(seeds[i])

		rng = np.random.default_rng(seeds[i])


		if args.domain in ["1d","2d","2d-multi","sequentialDarts"] or (args.domain == "hockey-multi" and args.testingBounded):

			# Need to reset agent generator per exp 
			# So that params are regenerated randomly
			# if xskill/pskill not given | to ensure randomness
			if not env.xSkillsGiven:
				env.agentGenerator.setXskillParams()

			if not env.pSkillsGiven:
				env.agentGenerator.setPskillParams()


		if args.domain in ["1d","2d","2d-multi"]:
			# Generate new set of states for experiment
			# States will be the same across agents
			env.setStates(rng)


		# Memory management
		#	For 1D, 2D-rand_pos & rand_v
		# 		Since different convolution info per state)
		# 	For baseball domain
		# 		Resetting exp info (per row)
		env.resetEnv()



		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Update spaces info
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		# To update spaces to account for the new states & 
		# randomly generated xskills (if xSkillsGiven = False)
		env.setSpacesAgents(rng)
				

		# Update spaces to have corresponding info
		# for xskill hyps (of estimators) and agents xskills
		# For 1D & 2D need to call every time after getting new set of states!
		if args.domain in ["1d","2d"]:
			env.setSpacesEstimators(rng,args,estimators.allXskills)
		elif args.domain in ["2d-multi","baseball-multi","hockey-multi"]:
			env.setSpacesEstimators(rng,args,[estimators.allXskills,estimators.rhos])
		# For other domains need to set spaces for estimators xskill just once (at the beginning)
		else:
			if i == 0:
				env.setSpacesEstimators(rng,args,estimators.allXskills)

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Perform experiment
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		if args.domain in ["baseball","baseball-multi","soccer"]:

			# Agent 
			# 	Baseball: [pitcherID, pitchType]
			#	Soccer: [playerID]

			agent = agents[i]

			if args.domain in ["baseball","baseball-multi"]:
				tag = f"OnlineExp_{agent[0]}_{agent[1]}"
			else:
				tag = f"OnlineExp_{agent}"

			if args.jeeds:
				tag += f"_JEEDS_Chunk_{args.seedNum}"

			if args.pfe:
				tag += f"_PFE_Chunk_{args.seedNum}"

			if args.pfeNeff:
				tag += f"_PFE_NEFF_Chunk_{args.seedNum}"

			# NOTE: ARGS.SEEDNUM indicates the chunk number
			counter = onlineExperiment(args,None,[agent],env,estimators,subsetEstimators,tag,counter,int(args.seedNum),rng,indexOR)


		elif args.domain in ["hockey-multi"]:

			if not args.testingBounded:
				agent = [args.id,args.type]
				tag = f"OnlineExp_Player_{agent[0]}_TypeShot_{agent[1]}"
			
				if args.jeeds:
					tag += f"_JEEDS"

				if args.pfe:
					tag += f"_PFE"

				if args.pfeNeff:
					tag += f"_PFE_NEFF"

				# NOTE: ARGS.SEEDNUM indicates the chunk number
				counter = onlineExperiment(args,None,[agent],env,estimators,subsetEstimators,tag,counter,int(args.seedNum),rng,indexOR)
			
			else:
				allInfo = list(product(env.agentGenerator.xSkills,env.agentGenerator.rhos))
		
				# For each one of the desired xskills
				for each in allInfo:

					print(f"{'-'*70}\nITERATION {i} - \nEXECUTION SKILL LEVEL: ",end="")
					print("each: ",each)

					aStr = ""
					justXs = ""

					for ti in range(len(each)):
						if ti == 1:
							aStr += "\nRHO:"
							aStr += f" {each[ti]} | "
							justXs += f"{each[ti]}-"

						else:	
							for t in each[ti]:
								aStr += f" {t} | "
								justXs += f"{t}-"

					print(aStr)
					print(f"{'-'*70}")



					# Create agents (with the different agent types)
					agents = env.agentGenerator.getAgents(rng,env,each)

					
					agent = [args.id,args.type]
					tag = f"OnlineExp_Player_{agent[0]}_TypeShot_{agent[1]}"
				
					# if args.jeeds:
					# 	tag += f"_JEEDS"

					# if args.pfe:
					# 	tag += f"_PFE"

					# if args.pfeNeff:
					# 	tag += f"_PFE_NEFF"

					counter = onlineExperiment(args,each,[agent,agents],env,estimators,subsetEstimators,tag,counter,seedNum,rng,indexOR,False)
					

		else: 

			if args.domain == "2d-multi":
				if args.dynamic:
					allInfo = list(product(env.agentGenerator.dynamicXskills,env.agentGenerator.rhos))
				else:
					allInfo = list(product(env.agentGenerator.xSkills,env.agentGenerator.rhos))
			else:
				allInfo = env.agentGenerator.xSkills

			# code.interact("...", local=dict(globals(), **locals()))


			# For each one of the desired xskills
			for each in allInfo:

				print(f"{'-'*70}\nITERATION {i} - \nEXECUTION SKILL LEVEL: ",end="")
				# print("each: ",each)

				aStr = ""
				justXs = ""

				if args.domain == "2d-multi":
					for ti in range(len(each)):
						if ti == 1:
							aStr += "\nRHO:"
							aStr += f" {each[ti]} | "
							justXs += f"{each[ti]}-"

						else:

							if args.dynamic:
								aStr += f" Start: {each[ti][0]} | End: {each[ti][1]}"
								justXs += f"{each[ti][0]}-{each[ti][1]}"

							else:
								for t in each[ti]:
									aStr += f" {t} | "
									justXs += f"{t}-"
				else:
					aStr += f" {each} | "
					justXs += f"{each}-"

				print(aStr)
				print(f"{'-'*70}")


				if args.domain == "2d-multi":
					# print(each)

					# Set space for current agent
					if args.dynamic:
						env.spaces.updateSpace(rng,[[each[0][0]],[each[-1]]],env.states)
						env.spaces.updateSpace(rng,[[each[0][1]],[each[-1]]],env.states)
					else:
						env.spaces.updateSpace(rng,[each[:-1],[each[-1]]],env.states)


				# Create agents (with the different agent types)
				agents = env.agentGenerator.getAgents(rng,env,each)

				tag = None
				expRerun = False

				if args.rerun:

					# Sample RF name:
					# OnlineExp_2-638_23_04_28_33_58_0_188317-0-AgentTargetAgent-X2.6377.results

					currentFiles = os.listdir(f"Experiments{os.sep}{args.resultsFolder}{os.sep}results{os.sep}")

					xStr = str(round(x,3)).replace('.','-')
					# Find respective tag name (since dependent on date)
					for eachRF in currentFiles:
						if xStr in eachRF and str(seedNum) in eachRF:
							tag = eachRF.split(f"{seedNum}")[0] + f"{seedNum}"
							expRerun = True
							print(f"Found tag: {tag}")
							break
					
					# code.interact("...", local=dict(globals(), **locals()))

					if tag == None:
						print("On rerun mode but tag not found. Proceeding to perform full experiment.")
						tag = f"OnlineExp_{str(round(x,3)).replace('.','-')}_{time.strftime('%y_%m_%d_%M_%S')}_{i}_{seedNum}"
						expRerun = False
				else:
					tag = f"OnlineExp_{justXs.replace('.','-')}_{time.strftime('%y_%m_%d_%M_%S')}_{i}_{seedNum}"
					
					if args.particles:
						tag += f"_{args.resamplingMethod}"


				counter = onlineExperiment(args,each,agents,env,estimators,subsetEstimators,tag,counter,seedNum,rng,indexOR,expRerun)
				
				# code.interact("...", local=dict(globals(), **locals()))

		
		expStop = time.time()
		timesPerExps.append(expStop-expStart)
		
		print(f"Finished Online Iteration: {i}")
		print(f"Total Time elapsed: {expStop-overallStart} seconds.  Average of {(expStop-overallStart)/float(i+1)} per iteration.\n")
		print("-"*70)
		print()


		# On every iter to save on the go
		timesResults = {"timesPerExps":timesPerExps,"avgTimePerExps":sum(timesPerExps)/len(timesPerExps)}

		with open(f"Experiments{os.sep}{args.resultsFolder}{os.sep}times{os.sep}experiments{os.sep}times-{args.seedNum}", 'wb') as outfile:
			pickle.dump(timesResults,outfile)

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# code.interact("after exp...", local=dict(globals(), **locals()))

	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	


	# Store seeds used (if not on rerun mode, as seeds will be provided if that's the case)
	if not args.rerun:
		results = {"seeds":seeds.tolist()}

		with open(f"Experiments{os.sep}{args.resultsFolder}{os.sep}seeds{args.seedNum}", 'wb') as outfile:
			pickle.dump(results,outfile)


	print('\n****\nDone with all', numExperiments, 'experiments\n****')
	# code.interact("END...", local=dict(globals(), **locals()))



if __name__ == "__main__": 

	main()


	'''

	# MASTER
	if rank == 0:
		main()

	# WORKERS
	else:

		# For each iteration?
		for M in range(1,3):

			# Receive the info (coming from master, rank = worker id?)
			# https://stackoverflow.com/questions/59559597/mpi4py-irecv-causes-segmentation-fault
			# 2<<20 = 2097152 = ~2.097152 MB
			# req = comm.irecv(bytes(2<<20),source=0,tag=rank)
			req = comm.irecv(source=0,tag=rank)


			# Get the data
			task = req.wait()
			# serializedData = req.wait()
			# print(serializedData)

			# task = req.loads(serializedData)


			print('[', M, '] WORKER ', rank, ' MESSAGE RECEIVED: ', task)
			# print('[', M, '] WORKER ', rank, ' MESSAGE RECEIVED: ', task['tid'])


			# rng = np.random.default_rng(task["seedNum"])
			# task["rng"] = rng


			# CASE: Task assigned to current worker
			# print(f"Worker {rank} is doing task {task['particles']}")

			# for each in task["particles"]:
			#	pdfsPerXskill, evsPerXskill = workUpdate(task,each,task["pdfsPerXskill"],task["evsPerXskill"])

			# print(f"Worker {wid} finished task {task['particles']}")
			
		
			# Update the data accordingly
			# reply = task*10
			reply = task["tid"] * 10
			# reply = {"pdfsPerXskill":pdfsPerXskill,"evsPerXskill":evsPerXskill}

			# Send back the resulting data
			q = comm.isend(reply,dest=0)

			# Wait til info is sent back
			q.wait()


		# comm.Disconnect()

	'''


