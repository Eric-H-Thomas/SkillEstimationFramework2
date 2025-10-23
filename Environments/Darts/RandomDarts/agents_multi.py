from abc import ABCMeta, abstractmethod

import code
import warnings

from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import numpy as np

# Agents for the darts domain
# The agent will choose an action for a given state given their xskill level

class Agent():

	# make ABC class in python 2.7
	__metaclass__ = ABCMeta

	def __init__(self,rng,name,params,domain):

		self.name = name
		self.params = params

		self.true_rewards = []

		# self.convolutions_all_ts = []
		self.convolutions_all_vs = []
		self.convolutions_all_mvs = []
		self.convolutions_ts = []
		self.convolutions_vs = []

		self.domainName = domain.get_domain_name()

		if "2d" in self.domainName or "hockey" in self.domainName:
			self.mean = [0.0]*len(params['noise_level'])

			# Use covariance matrix of start xskill
			if "Change" in name:
				self.setNoiseModel(rng,domain,[params['noise_level'][0][0],params['noise_level'][0][1],params['rho']])
			else:
				self.setNoiseModel(rng,domain,[params['noise_level'][0],params['noise_level'][1],params['rho']])
			
			# code.interact("...", local=dict(globals(), **locals()))


	def setNoiseModel(self,rng,domain,given):
		self.covMatrix = domain.getCovMatrix(given[:-1],given[-1])
		self.noiseModel = domain.draw_noise_sample(rng,self.mean,self.covMatrix)


	def getInfoConvolutions(self):
		# return first 2 as a dict in order to save it to file
		return {"convolutions_all_vs" : self.convolutions_all_vs},\
				 self.convolutions_ts, self.convolutions_vs, self.convolutions_all_mvs


	@abstractmethod
	def get_action(self,convolutionSpace,returnZn=False):
		# All agents must have this method implemented
 
		# store actual true reward = max EV
		self.true_rewards.append(convolutionSpace["vs"])

		# self.convolutions_all_ts.append(convolutionSpace["all_ts"])
		self.convolutions_all_vs.append(convolutionSpace["all_vs"])
				
		self.convolutions_ts.append(convolutionSpace["ts"])
		self.convolutions_vs.append(convolutionSpace["vs"])

		# print(self.covMatrix)
		return convolutionSpace["all_vs"], convolutionSpace["ts"], convolutionSpace["vs"]


	def get_true_rewards(self):
		return self.true_rewards

	def getName(self):
		return self.name

	def print_agent(self):
		print('Agent: ', self.name)
		print('Params: ', self.params)

	def addInfoToName(self,params):

		temp = ""

		# Just Multi Dimensional Xskill
		d = len(params['noise_level'])

		for t in range(d):
			temp += f"|X{params['noise_level'][t]}"

		return f"{temp}|R{params['rho']}"


	def getEvIntendedAction(self,listedTargets,values,action):

		# For action
		ai = np.zeros((1,2))
		ai[0][0] = action[0] # First action dimension
		ai[0][1] = action[1] # Second action dimension

		# Using cubic interpolation since 2D
		# "linear" was used before and values where about the same but slightly smaller
		# Thus, resulting in a negative %Reward (when computing expected --- mean since expected < mean )
		# for some "bad" agents
		# evAction = griddata(listedTargets,listedValues,ai,method='cubic',fill_value=np.min(listedValues))[0]
		evAction = griddata(listedTargets,values.ravel(),ai,method='cubic')[0]
		# code.interact("getEvIntendedAction()", local=dict(globals(), **locals()))
		
		return evAction


class TargetAgent(Agent):

	def __init__(self,rng,params,domain,n="TargetAgent"):

		name = n + self.addInfoToName(params)
		Agent.__init__(self,rng,name,params,domain)


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False):

		all_vs,ts,vs = Agent.get_action(self,convolutionSpace,returnZn)

		# Returns action that has the highest ev
		# print(ts,vs)
		return ts, vs


class TargetAgentAbruptChange(TargetAgent):

	def __init__(self,rng,numObs,params,domain):


		name = f"TargetAgentAbruptChange"
		TargetAgent.__init__(self,rng,params,domain,name)

		step = numObs//3

		# Generate a number between 0+step and numObs-step
		# To get a number in the 2nd third of the observations
		# So that estimators have some time to learn
		# (1st third - learn, 2nd third - change, last third - learn again)
		self.changeAt = np.random.randint(0+step,numObs-step)

		self.startX = params['noise_level'][0]
		self.endX = params['noise_level'][1]
		self.rho = params['rho']

		self.x = self.startX


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,spaces,otherArgs,returnZn=False):

		# print(f"{otherArgs['i']} - Current X: {self.x} (Change at: {self.changeAt})")

		if otherArgs["i"] == self.changeAt:
			self.x = self.endX
			self.setNoiseModel(rng,domain,[self.endX[0],self.endX[1],self.rho])
			# print(f"Changing X to: {self.x}")


		info = self.x
		# print("!!!",info)
		# print("---",otherArgs["s"])

		convolutionSpace = spaces.getSpace(rng,[info,self.rho],otherArgs["actualState"])

		all_vs,ts,vs = Agent.get_action(self,convolutionSpace,returnZn)


		# Returns action that has the highest ev
		return ts, vs, convolutionSpace


class TargetAgentGradualChange(TargetAgent):

	def __init__(self,rng,numObs,params,domain):

		name = f"TargetAgentGradualChange"
		TargetAgent.__init__(self,rng,params,domain,name)

		self.startX = params['noise_level'][0]
		self.endX = params['noise_level'][1]
		self.rho = params['rho']

		temp1 = np.round(np.linspace(self.startX[0],self.endX[0],numObs),4)
		temp2 = np.round(np.linspace(self.startX[1],self.endX[1],numObs),4)
		self.xSkills = np.stack((temp1,temp2),axis=1)


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,spaces,otherArgs,returnZn=False):

		self.x = self.xSkills[otherArgs["i"]]
		self.setNoiseModel(rng,domain,[self.x[0],self.x[1],self.rho])
		# print(f"Current Skill: {self.x}")

		info = self.x.tolist()

		convolutionSpace = spaces.getSpace(rng,[info,self.rho],otherArgs["actualState"])

		all_vs,ts,vs = Agent.get_action(self,convolutionSpace,returnZn)


		# Returns action that has the highest ev
		return ts, vs, convolutionSpace


class FlipAgent(Agent):

	def __init__(self,rng,params,domain):

		name = "FlipAgent"+self.addInfoToName(params)
		name += "|P"+str(params['prob_rational'])
		super(FlipAgent, self).__init__(rng,name,params,domain)

	
	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False,ax=None):

		all_vs,ts,vs = super(FlipAgent,self).get_action(convolutionSpace,returnZn)

		P = self.params['prob_rational']
		
		# Flip a coin to see if agent will select a rational or a random action
		result = rng.binomial(1,P) # 1 = number of trials | P = probability of each trial

		# if heads, act rational
		if result == 1: 
			# print("Acting RATIONAL")
			return ts,vs
		# otherwise, act random
		else:
			# print("Acting RANDOM")

			if self.domainName == "2d-multi":
				# Generate random action - from darts board - 2d
				# formula obtained from:
				# https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409

				R = 170.0

				r = R * np.sqrt(rng.random())

				theta = rng.random() * 2 * np.pi

				# Convert to cartesian
				x = r * np.cos(theta)
				y = r * np.sin(theta)


				action = [x,y]

				evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)

				# code.interact("FLIP - getAction()", local=dict(globals(), **locals()))

				return action,evAction


class TrickerAgent(Agent):

	def __init__(self,rng,params,domain):
		
		name = "TrickerAgent" + self.addInfoToName(params)
		name += "|Eps" + str(params["eps"])
		super(TrickerAgent, self).__init__(rng,name,params,domain)

	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False,returnIndexSelected=False):

		all_vs,ts,vs = super(TrickerAgent,self).get_action(convolutionSpace,returnZn)


		eps = self.params["eps"]

		# ev to accept = max ev - ( (1 - epsilon of agent) * max ev)
		# the bigger the epsilon, the less actions possible (closer to the action to the max ev)
		# the smaller the epsilon, the more possible actions
		
		# PREV
		# epsEv = vs - ((1.0 - eps) * vs)

		minEV = np.min(all_vs)
		maxEV = np.max(all_vs)
		meanEV = np.mean(all_vs)

		# NEW
		epsEV = meanEV + eps * (maxEV-meanEV)

		# Truncating decimal places
		# Because in some case maxEV is > epsEV causing error
		# due to difference with decimal places
		# Example:  
		#	maxEV = 0.999110228611768
		# 	epsEV = 0.9991102286117681
		epsEV = float(str(epsEV)[0:10])
		# print("epsEV: ", epsEV) 


		# Find all the evs that are >= epsEv
		possibleEvsIndexes = np.where(all_vs >= epsEV)
		# print "possibleEvsIndexes: ", possibleEvsIndexes

		possibleEvs = all_vs[possibleEvsIndexes]
		# print "possibleEvs: ", possibleEvs

		# Find the target that is furthest away from the target that gives the max ev
		dist = []

		###############################################################################
		if self.domainName == "2d-multi":

			possibleTargetsX = listedTargets[:,0][possibleEvsIndexes]
			possibleTargetsY = listedTargets[:,1][possibleEvsIndexes]

			# Find distances
			for i in range(len(possibleTargetsX)):
				eachPossibleTarget = [possibleTargetsX[i],possibleTargetsY[i]]
				dist.append(domain.calculate_wrapped_action_difference(eachPossibleTarget,ts))


		# Get the max distance
		maxDist = max(dist)


		# Get the index of the max distance
		maxDistIndex = dist.index(maxDist)

		if self.domainName == "2d-multi":
			# Get the action that corresponds to the maximum distance (the farthest one)
			action = [possibleTargetsX[maxDistIndex],possibleTargetsY[maxDistIndex]]


		evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)


		# FOR TESTING
		'''
		import Environments.Darts.RandomDarts.two_d_darts_multi as domainModule
		rng = np.random.default_rng(1000)
		state = domainModule.generate_random_states(rng,1,"normal")[0]

		fig = plt.figure()
		ax = plt.gca()

		domainModule.draw_board(ax)
		domainModule.label_regions(state, color = "black")

		plt.scatter(action[0],action[1],label='Intended Action')
		plt.scatter(ts[0],ts[1],label='Max EV Action')
		plt.legend()
		plt.show()
		'''

		# code.interact("TRICKER - getAction()", local=dict(globals(), **locals()))

		if returnIndexSelected:
			return action,maxDistIndex,evAction
		else:
			return action,evAction


class BoundedAgent(Agent):

	def __init__(self,rng,params,domain,n="BoundedAgent"):

		# L = Lambda
		name = n + self.addInfoToName(params)
		name += "|L"+str(params['lambda'])
		Agent.__init__(self,rng,name,params,domain)

		self.nansInfoS = []
		self.nansInfoCounter = []
		self.nansInfoFinalL = []

		self.allProbs = []

	def getAllProbs(self):
		# return as a dict in order to save it to file
		return {"allProbs": self.allProbs}

	def getNansInfo(self):
		# return the dictionary containing the corresponding information for when the NANs occurred
		return self.nansInfoS, self.nansInfoCounter,self.nansInfoFinalL

	def computeProbs(self,L,all_vs,domain):

		aSum = 0.0


		#probs = np.zeros((len(all_vs),len(all_vs[0])))
		probs = np.zeros((len(all_vs),1))

		# To be used for exp normalization trick - find maxEV and * by L
		# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
		b = np.max(all_vs*L)


		all_vsL = np.copy(all_vs) * L
		
		# With normalization trick
		y = np.exp(all_vsL - b)

		# Without normalization trick - causes warning since probs are 0's
		# y = np.exp(all_vsL)

		# Normalize
		probs = y / np.sum(y)


		# code.interact("inside computeProbs()", local=locals())

		return probs

	def verifyProbs(self,probs,L,all_vs,nansFlag,counter,domain):

		# verify if probs have nans
		nans = np.isnan(probs)

		# if any of the probs resulted in a NAN
		if np.any(nans) or np.sum(probs) == 0.0:
			
			nansFlag = True
			# print "NAN found!"

			# keep 90% of previous lambda
			L = L * 0.90

			# print "testing new L: ", L

			# compute probs again
			probs = self.computeProbs(L,all_vs,domain)
			# print "sum probs with new L: ", np.sum(probs)

			counter += 1

			# verify probs again
			return self.verifyProbs(probs,all_vs,nansFlag,counter,domain)

		# no NANs found, return probs
		else:

			# print "Final L: ", L
			# print "sum probs with final L: ", np.sum(probs)
			#print probs

			# nans = np.isnan(probs)
			# print "nans? ", np.any(nans)
			# raw_input()

			# print "verifyProbs()"
			# code.interact(local=locals())

			return probs,L,nansFlag,counter

	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnProbs=False,returnZn=False):

		all_vs,ts,vs = super(BoundedAgent,self).get_action(convolutionSpace,returnZn)

		L = self.params['lambda']

		all_vs = all_vs.flatten()
		
		probs = self.computeProbs(L,all_vs,domain)

		# keep track of how many times probs were updated/recomputed & lambda shrunk
		counter = 0
		nansFlag = False

		finalProbs,finalL,nansFlag,counter = self.verifyProbs(probs,L,all_vs,nansFlag,counter,domain)

		self.allProbs.append(finalProbs.tolist())
		
		'''
		print "back in get_action()"
		print "Final L: ", finalL
		print "sum probs with final L: ", np.sum(finalProbs)
		print "finalProbs: ", finalProbs

		nans = np.isnan(finalProbs)
		print "nans? ", np.any(nans)
		'''
		
		'''
		nansAllProbs = np.isnan(self.allProbs[-1])
		print "allProbs nan? ", np.any(nansAllProbs)
		# print "allProbs: ", self.allProbs[-1]
		print "allProbs: ", len(self.allProbs[-1])
		# raw_input()
		if np.any(nansAllProbs) == True:
			code.interact(local=locals())
		'''
		
		
		# if recomputing of the probs happened
		if nansFlag:    
			self.nansInfoS.append(S)
			self.nansInfoCounter.append(counter)
			self.nansInfoFinalL.append(finalL)



		N = len(listedTargets)


		# select action from possible actions with probs as weight -- by index
		indexAction = rng.choice(range(N),p=finalProbs)

		# Map from index back to action
		action = list(listedTargets[indexAction])


		if returnProbs:

			# Find the action with the max prob
			maxProb = np.max(finalProbs)

			# Find the indexes of the action with the max prob
			maxProbIndex = np.where(finalProbs == maxProb)

			# Get the probs of the indexes out of the finalProbs array to use with np.choice
			#indexProbs = np.copy(finalProbs[maxProbIndex[0], maxProbIndex[1]])
			indexProbs = np.copy(finalProbs[maxProbIndex[0]])
			
			# code.interact(local=locals())

			# verify if there's just one index for the action with the highest probability
			# if there's more than one, select one at random
			# this occurs whenever L = 0.0 since the probabilities have a uniform distribution for all the different actions
			# (L = 0.0 = acting uniformly at randoms)
			# or whenever the exact same probability occurs for different targets (and must choose at random from those too)
			if len(maxProbIndex[0]) != 1:

				# Normalize the probs so that they add up to 1 - to avoid error
				indexProbs /= np.sum(indexProbs)

				# Select index at random - probs have uniform distribution
				mp = np.random.choice(range(0,len(maxProbIndex[0])), p = indexProbs)

				# "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
				#xmp, ymp = np.unravel_index(mp,finalProbs.shape)
				xmp = np.unravel_index(mp,finalProbs.shape)

				maxProbX = xmp
				#maxProbY = ymp

			else:
				maxProbX = maxProbIndex[0]
				#maxProbY = maxProbIndex[1]


			# Get action corresponding to the indices
			actionMaxProb = listedTargets[maxProbX]


		evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)

		#code.interact("BOUNDED - get_action()", local=locals())


		if returnProbs:
			return action,nansFlag,finalProbs,actionMaxProb,evAction
		else:
			return action,nansFlag,evAction


class BoundedAgentAbruptChange(BoundedAgent):

	def __init__(self,rng,numObs,params,domain):

		name = f"BoundedAgentAbruptChange"
		BoundedAgent.__init__(self,rng,params,domain,name)

		step = numObs//3

		# Generate a number between 0+step and numObs-step
		# To get a number in the 2nd third of the observations
		# So that estimators have some time to learn
		# (1st third - learn, 2nd third - change, last third - learn again)
		self.changeAt = np.random.randint(0+step,numObs-step)

		self.startX = params['noise_level'][0]
		self.endX = params['noise_level'][1]
		self.rho = params['rho']

		self.x = self.startX


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,spaces,otherArgs,returnZn=False):

		# print(f"{otherArgs['i']} - Current X: {self.x} (Change at: {self.changeAt})")

		if otherArgs["i"] == self.changeAt:
			self.x = self.endX
			self.setNoiseModel(rng,domain,[self.endX[0],self.endX[1],self.rho])
			# print(f"Changing X to: {self.x}")


		info = self.x
		# print("!!!",info)
		# print("---",otherArgs["s"])

		convolutionSpace = spaces.getSpace(rng,[info,self.rho],otherArgs["actualState"])

		all_vs,ts,vs = Agent.get_action(self,convolutionSpace,returnZn)

		action,nansFlag,evAction = BoundedAgent.get_action(self,rng,domain,listedTargets,convolutionSpace)


		# Returns action that has the highest ev
		return action, evAction, convolutionSpace


class BoundedAgentGradualChange(BoundedAgent):

	def __init__(self,rng,numObs,params,domain):

		name = f"BoundedAgentGradualChange"
		BoundedAgent.__init__(self,rng,params,domain,name)

		self.startX = params['noise_level'][0]
		self.endX = params['noise_level'][1]
		self.rho = params['rho']

		temp1 = np.round(np.linspace(self.startX[0],self.endX[0],numObs),4)
		temp2 = np.round(np.linspace(self.startX[1],self.endX[1],numObs),4)
		self.xSkills = np.stack((temp1,temp2),axis=1)


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,spaces,otherArgs,returnZn=False):

		self.x = self.xSkills[otherArgs["i"]]
		self.setNoiseModel(rng,domain,[self.x[0],self.x[1],self.rho])
		# print(f"Current Skill: {self.x}")

		info = self.x.tolist()

		convolutionSpace = spaces.getSpace(rng,[info,self.rho],otherArgs["actualState"])

		all_vs,ts,vs = Agent.get_action(self,convolutionSpace,returnZn)


		action,nansFlag,evAction = BoundedAgent.get_action(self,rng,domain,listedTargets,convolutionSpace)


		# Returns action that has the highest ev
		return action, evAction, convolutionSpace


