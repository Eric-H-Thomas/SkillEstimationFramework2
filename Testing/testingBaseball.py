from pathlib import Path
from importlib.machinery import SourceFileLoader
import os,sys,code,copy,json

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Testing{os.sep}testingBaseball.py")[0]

module = SourceFileLoader("baseball",f"{mainFolderName}{os.sep}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.interpolate import griddata

from testingBaseball_OR import *
from testingBaseball_Joint import *


class Estimators():

	def __init__(self,xskills,numHypsP,namesEstimators,domainName,env=None,betas=[],agentType=None):

		# Proceed to create the estimators for each one of the different number of hypothesis skills
		self.estimators = []

		# To remember all xskills seen (across the different number of hyps)
		self.allXskills = sorted(xskills)


		if "OR" in namesEstimators:
			self.estimators.append(ObservedReward(xskills,domainName))

		if "JT-FLIP" in namesEstimators:
			self.estimators.append(JointMethodFlip(xskills,numHypsP,domainName))

		if "JT-QRE" in namesEstimators:
			self.estimators.append(JointMethodQRE(xskills,numHypsP,domainName))

		if "NJT-QRE" in namesEstimators:
			self.estimators.append(NonJointMethodQRE(xskills,numHypsP,domainName))


		self.namesEstimators = []

		for each in self.estimators:

			names = each.getEstimatorName()
			
			if type(names) == str:			
				self.namesEstimators.append(names)
			else:
				for n in names:	
					self.namesEstimators.append(n)	

		# code.interact("...", local=dict(globals(), **locals()))

	def printEstimators(self):
		for e in self.estimators:
			print("Estimator: " + str(e.getEstimatorName()) + "\n" )
			print("\txskills: " + str(e.xskills)  + "\n")

	
	def getCopyOfEstimators(self):

		# Get copy of the estimators
		copyEstimators = copy.deepcopy(self.estimators)

		# Reset each one (will reset the attributes from within the object)
		for e in copyEstimators:
			e.reset()

		# Will return a copy of the list that contains all the estimators
		return copyEstimators


def run(infoPerRow,xskills,lambdas,delta,possibleTargets,agentFolder):

	allCovs = np.zeros((len(xskills),2,2))

	for xi in range(len(xskills)):
		x = xskills[xi]

		val = x**2
		cvs = np.zeros((2,2))
		np.fill_diagonal(cvs,val)

		allCovs[xi] = cvs


	# For hyps = 66: 
	'''
	array([0.17, 0.80461538, 1.43923077, 2.07384615, 2.70846154,
       3.34307692, 3.97769231, 4.61230769, 5.24692308, 5.88153846,
       6.51615385, 7.15076923, 7.78538462, 8.42])
	'''
	# xskillsForAgents = xskills[0:len(xskills):5]

	# array([0.17, 1.82, 3.47, 5.12, 6.77, 8.42])
	xskillsForAgents = xskills[0:len(xskills):13]


	# For hyps = 100: 
	'''
	array([1.00000000e-03, 4.43062146e-03, 1.96304065e-02, 8.69749003e-02,
	       3.85352859e-01, 1.70735265e+00, 7.56463328e+00, 3.35160265e+01,
	       1.48496826e+02, 6.57933225e+02])'''
	# pskillsForAgents = lambdas[0:len(lambdas):10]

	# array([1.00000000e-03, 1.96304065e-02, 3.85352859e-01, 7.56463328e+00,
    #  1.48496826e+02])
	pskillsForAgents = np.concatenate((lambdas[0:len(lambdas):19],[lambdas[-1]]))


	# [xskill,pskill] combinations
	agents = [] 

	for xi in xskillsForAgents:
		for pi in pskillsForAgents:
			agents.append([xi,pi])


	# numHypsX = [66]
	numHypsP = 66
	# startX_Estimator = 0.17
	# stopX_Estimator = 2.81
	namesEstimators = ["OR","JT-QRE","NJT-QRE"]
	domainName = "baseball"
	estimatorsObj = Estimators(xskills,numHypsP,namesEstimators,domainName)


	for eachAgent in agents:

		estimators = estimatorsObj.getCopyOfEstimators()
		
		x = eachAgent[0]
		p = eachAgent[1]

		N = sys.modules["domain"].draw_noise_sample(mean=[0.0,0.0],X=x**2)

		print(f"Agent -> X: {x} P: {p}")

		for bf in infoPerRow:

			# Intended Action
			ai = infoPerRow[bf]["intendedActionDiffLambdasPerXskill"][x][p][0]
			action = possibleTargets[ai]

			evIntendedAction = infoPerRow[bf]["intendedActionDiffLambdasPerXskill"][x][p][1]


			# Noisy/Executed Action
			noise = N.rvs()
			noisyAction = [action[0]+noise[0],action[1]+noise[1]]


			# Find reward for executed action
			ai = np.zeros((1,2))
			ai[0][0] = noisyAction[0] # First action dimension
			ai[0][1] = noisyAction[1] # Second action dimension

			# Using cubic interpolation since 2D
			observedReward = griddata(possibleTargets,infoPerRow[bf]["evsPerXskill"][x].flatten(),ai, method='cubic')[0]


			otherArgs = {}
			otherArgs["evsPerXskill"] = infoPerRow[bf]["evsPerXskill"]
			otherArgs["maxEVPerXskill"] = infoPerRow[bf]["maxEVPerXskill"]
			otherArgs["allCovs"] = allCovs
			otherArgs["delta"] = delta


			for e in estimators:
				if isinstance(e,JointMethodQRE):
					e.addObservation(noisyAction,possibleTargets,otherArgs)
				elif isinstance(e,NonJointMethodQRE):
					e.addObservation(noisyAction,possibleTargets,otherArgs)
				else: # OR
					e.addObservation(observedReward,otherArgs)


		results = {}
		results["xskill"] = x
		results["pskill"] = p

		for e in estimators:
			R = e.getResults()
			for en, er in R.items():
				results[en] = er

		# Store results to json file
		with open(f"BaseballTesting{os.sep}Results{os.sep}{agentFolder}{os.sep}results-{agentFolder}-Agent-X{x}-P{p}.json",'w') as outfile:
			json.dump(results, outfile)

		# code.interact("end run()...", local=dict(globals(), **locals()))


def prepInfoForExp(info):
	
	infoPerRow = {}

	for x in info.keys():
		for bf in info[x]:

			if bf not in infoPerRow:
				infoPerRow[bf] = {"evsPerXskill":{},
									"actionsPerXskill":{},
									"maxEVPerXskill":{},
									"intendedActionDiffLambdasPerXskill":{}}

			if x not in infoPerRow[bf]:
				infoPerRow[bf]["evsPerXskill"][x] = {}
				infoPerRow[bf]["actionsPerXskill"][x] = {}
				infoPerRow[bf]["maxEVPerXskill"][x] = {}
				infoPerRow[bf]["intendedActionDiffLambdasPerXskill"][x] = {}

			infoPerRow[bf]["evsPerXskill"][x] = info[x][bf]["EVs"]
			infoPerRow[bf]["maxEVPerXskill"][x] = info[x][bf]["maxEV"]
			infoPerRow[bf]["intendedActionDiffLambdasPerXskill"][x] = info[x][bf]["lambdas"]

	return infoPerRow


def pconf(resultsFolder,agentFolder):

	info = {}

	pconfPerXskill = {}

	# 0.5 inches | 0.0417 feet
	delta = 0.0417
	
	# minXskill:  2.0 inches | 0.17 feet
	# maxXskill: 101 inches | 2.81 feet
	# xskills = np.linspace(0.17,2.81,num=66)
	xskills = np.concatenate((np.linspace(0.17,1.0,num=60),np.linspace(1.00+delta,2.81,num=6)))
	# xskills = np.concatenate((np.linspace(0.17,1.0,num=1),np.linspace(1.00+delta,2.81,num=1)))
	
	lambdas = np.logspace(-3,3.4,100)
	# lambdas = np.logspace(-3,3.4,3)


	folder = f"..{os.sep}Experiments{os.sep}baseball{os.sep}{resultsFolder}{os.sep}plots{os.sep}StrikeZoneBoards{os.sep}"
	pickleFolder = f"{folder}PickleFiles{os.sep}{agentFolder}{os.sep}"

	# Load all dartboards available for agent
	boards = os.listdir(pickleFolder)
	boards.remove("info.pkl")


	with open(pickleFolder+"info.pkl","rb") as handle:
	    info2 = pickle.load(handle)

	targetsPlateX = info2["plate_x_feet"]
	targetsPlateZ = info2["plate_z_feet"]


	# Store dense matrix for actions
	possibleTargets = []

	for i in range(len(targetsPlateX)):
		for j in range(len(targetsPlateZ)):
			possibleTargets.append([targetsPlateX[i],targetsPlateZ[j]])

	possibleTargets = np.array(possibleTargets)

	idx = list(range(len(possibleTargets)))


	# Go through all of the execution skills
	for x in xskills:
		print('Generating data for execution skill level', x)

		info[x] = {}

		prat = [] # This is where the probability of rational reward will be stored
		mins = [] # Store min reward possible
		maxs = [] # Store max reward possible
		means = [] # Store the mean of the possible rewards (this is the uniform random reward)
		evs = [] # Store the ev of the current agent's strategy

		Zn = sys.modules["domain"].getSymmetricNormalDistribution(x,delta,targetsPlateX,targetsPlateZ)

		for l in lambdas:   

			size = len(boards)  

			max_rs = np.zeros(size)
			min_rs = np.zeros(size)
			exp_rs = np.zeros(size)
			mean_rs = np.zeros(size)

			si = 0
			
			for boardFile in boards:

				if boardFile not in info[x]:
					info[x][boardFile] = {}

					boardDF = pd.read_pickle(pickleFolder+boardFile)

					Zs = np.reshape(boardDF.utility.values,(len(targetsPlateX),len(targetsPlateZ)))
					minUtility = np.min(Zs)

					# Convolve to produce the EV and aiming spot
					EVs = convolve2d(Zs,Zn,mode="same",fillvalue=minUtility)

					maxEV = np.max(EVs)	
					mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
					action = [targetsPlateX[mx],targetsPlateZ[mz]]

					info[x][boardFile]["EVs"] = EVs
					info[x][boardFile]["maxEV"] = maxEV

				else:
					#maxEV = info[x][boardFile]["maxEV"]
					EVs = info[x][boardFile]["EVs"]


				if "lambdas" not in info[x][boardFile]:
					info[x][boardFile]["lambdas"] = {}


				# Get the values from the ev 
				max_rs[si] = np.max(EVs)
				min_rs[si] = np.min(EVs) 
				mean_rs[si] = np.mean(EVs) 

				# Bounded decision-making with lambda = l
				b = np.max(EVs*l)
				expev = np.exp(EVs*l-b)
				sumexp = np.sum(expev)
				P = expev/sumexp

				# Store bounded agent's EV
				boundedEVs = P*EVs
				exp_rs[si] = np.sum(boundedEVs)

				#info[x][boardFile]["lambdas"][l] = P

				# Save sampled action instead of entire probs array
				ai = np.random.choice(idx,p=P.flatten())
				info[x][boardFile]["lambdas"][l] = [ai,boundedEVs.flatten()[ai]]

				# code.interact("pconf()...", local=dict(globals(), **locals()))

				si += 1

			
			prat.append(np.mean((exp_rs - mean_rs)/(max_rs - mean_rs)))
			mins.append(np.mean(min_rs))
			means.append(np.mean(mean_rs))
			maxs.append(np.mean(max_rs))
			evs.append(np.mean(exp_rs))

		plt.plot(lambdas, prat, label='x=' + str(x))

		pconfPerXskill[x] = {"lambdas":lambdas, "prat": prat}


	plt.xlabel('Lambda')
	plt.ylabel('% Rational Reward')
	plt.legend()
	plt.savefig(f"{folder}lambdasVSprat-{agentFolder}.png")


	# Save pconf info to pickle file
	with open(f"BaseballTesting{os.sep}Results{os.sep}{agentFolder}{os.sep}pconf.pkl",'wb') as handle:
	    pickle.dump(pconfPerXskill,handle)


	# code.interact("pconf()...", local=dict(globals(), **locals()))
	return pconfPerXskill,info,xskills,lambdas,delta,possibleTargets


if __name__ == '__main__':

	if len(sys.argv) != 4:
		print("Please provide pitcher ID, pitch type and experiment folder.\nAlso make sure pickle files exist already on the experiment folder. \nUsage: testingBaseball.py pitcherID pitchType expFolder")
		exit()


	# Testing pitcher IDs: 642232, 621237, 621107, 622250
	pitcherID =  sys.argv[1]

	# Testing pitch types: FF, CU
	pitchType = sys.argv[2]

	# Testing experiment folders: test1, test2
	resultsFolder = sys.argv[3]

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
	print(f"\n{agentFolder}\n")


	folders = [f"BaseballTesting{os.sep}Results{os.sep}",f"BaseballTesting{os.sep}Results{os.sep}{agentFolder}{os.sep}"]
	
	for f in folders:
		if not Path(f).is_dir():
			os.mkdir(f)


	pconfPerXskill,info,xskills,lambdas,delta,possibleTargets = pconf(resultsFolder,agentFolder)

	infoPerRow = prepInfoForExp(info)
	
	run(infoPerRow,xskills,lambdas,delta,possibleTargets,agentFolder)
