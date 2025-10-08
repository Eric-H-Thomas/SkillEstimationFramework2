from abc import ABCMeta, abstractmethod
from scipy import stats, integrate
from scipy.interpolate import griddata
import numpy as np
import code
import matplotlib.pyplot as plt


# Agents for the sequential darts domain
# The agent will choose an action for a given state given their xskill level

class Agent():

	# make ABC class in python 2.7
	__metaclass__ = ABCMeta

	def __init__(self,name,params,domain):

		self.name = name
		self.params = params

		self.domainName = domain.getDomainName()

		self.noiseModel = domain.getNoiseModel(params['noise_level']**2)

		self.trueRewards = []


	@abstractmethod
	def getAction(self,domain,space,currentScore):
		# All agents must have this method implemented
		pass

	def getTrueRewards(self):
	    return self.trueRewards


	def getName(self):
		return self.name

	def printAgent(self):
		print("-"*15)
		print('Agent: ', self.name)
		print('Params: ', self.params)
		print("-"*15)


	def getValueIntendedAction(self,space,currentScore,action):

		points = space.actions

		# For action
		ai = np.zeros((1,2))
		ai[0][0] = action[0] #First action dimension
		ai[0][1] = action[1] #Second action dimension

		evCopy = np.copy(space.flatEVsPerState[currentScore])
		# code.interact("...", local=dict(globals(), **locals()))

		# Using cubic interpolation since 2D
		# "linear" was used before and values where about the same but slightly smaller
		# Thus, resulting in a negative %Reward (when computing expected --- mean since expected < mean )
		# for some "bad" agents
		# valueAction = griddata(points,evCopy,ai,method='cubic',fill_value=np.min(space.flatEVsPerState[currentScore]))[0]
		valueAction = griddata(points,evCopy,ai,method='cubic')[0]
		# code.interact("...", local=dict(globals(), **locals()))

		return valueAction


class TargetAgent(Agent):

	def __init__(self,params,domain):

		name = "TargetAgent-X" + str(params['noise_level']) 
		super(TargetAgent, self).__init__(name,params,domain)

	# Agent's strategy for selecting action at a given state
	def getAction(self,rng,domain,space,currentScore):

		# Store actual true reward = max EV
		self.trueRewards.append(space.PI_EV[currentScore])

		# Return action that has the highest value for current state/score
		# Using learned value function
		# print(f"Target Agent selected: {space.PI[currentScore]} at currentScore: {currentScore}")
		return space.PI[currentScore], space.PI_EV[currentScore]


class TargetAgentWithBeliefs(Agent):

	def __init__(self,params,domain):

		name = "TargetAgent-BeliefX"+str(params['noise_level_belief'])+"-TrueX"+str(params['noise_level']) 
		super(TargetAgentWithBeliefs, self).__init__(name,params,domain)

		self.beliefX = params['noise_level_belief']
		self.actualX = params['noise_level']


	# Agent's strategy for selecting action at a given state
	def getAction(self,rng,domain,spaces,currentScore):

		# Use belief xskill for planning
		# Use actual/true xskill for executing

		# Action is selected from possible targets for given beliefX 
		#	(action = max EV action from beliefX convolution)
		# Actual EV of that action is then obtained from the true EVs 
		#	(from the trueX convolution) 



		# Select/plan action based on belief xskill 
		spaceBelief = spaces.spacesPerXskill[self.beliefX]
		actionBeliefX = spaceBelief.PI[currentScore]
		evActionBeliefX = spaceBelief.V[currentScore]

		# Actual xskill info - for testing purposes
		spaceActual = spaces.spacesPerXskill[self.actualX]
		actionActualX = spaceActual.PI[currentScore]
		evActionActualX = spaceActual.V[currentScore]

		# Value of the selected action using beliefX on actualX space
		evActionBeliefOnActualXSpace = self.getValueIntendedAction(spaceActual,currentScore,actionBeliefX)

		'''
		print(f"BeliefX: {self.beliefX} | Actual: {self.actualX}")
		print(f"actionActualX: {actionActualX} | evActionActualX: {evActionActualX}")
		print(f"actionBeliefX: {actionBeliefX} | evActionBeliefX: {evActionBeliefX}")
		print(f"evActionBeliefOnActualXSpace: {evActionBeliefOnActualXSpace}")
		'''

		# Store actual true reward = max EV
		self.trueRewards.append(evActionBeliefOnActualXSpace)

		# print(f"TargetAgentWithBeliefs selected: {actionBeliefX} at currentScore: {currentScore}")
		return actionBeliefX, evActionBeliefOnActualXSpace


class FlipAgent(Agent):

	def __init__(self, params, domain):

		name = "FlipAgent-X"+str(params['noise_level'])+"-P"+str(params['prob_rational'])
		super(FlipAgent, self).__init__(name, params, domain)
	
	# Agent's strategy for selecting action at a given state
	def getAction(self,rng,domain,space,currentScore,ax=None):

		P = self.params['prob_rational']

		# Store actual true reward = max EV
		self.trueRewards.append(space.PI_EV[currentScore])
		
		# Flip a coin to see if agent will select a rational or a random action
		result = rng.binomial(1,P) # 1 = number of trials | P = probability of each trial

		# if heads, act rational
		if result == 1: 
			# print("Acting RATIONAL")
			return space.PI[currentScore],space.PI_EV[currentScore]
		# otherwise, act random
		else:
			# print("Acting RANDOM")

			# generate random action - from darts board - 2d
			# formula obtained from:
			# https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly/50746409#50746409

			R = 170.0

			r = R * np.sqrt(rng.random())

			theta = rng.random() * 2 * np.pi

			# convert to cartesian
			x = r * np.cos(theta)
			y = r * np.sin(theta)

			action = [x,y]

			# plt.scatter(x,y)
			# slices = [25,11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]
			# domain.drawBoard(ax,slices)

			valueAction = self.getValueIntendedAction(space,currentScore,action)

			return action,valueAction


class TrickerAgent(Agent):

	def __init__(self,params,domain):
		
		name = "TrickerAgent-X" + str(params["noise_level"])+"-Eps" + str(params["eps"])
		super(TrickerAgent, self).__init__(name, params, domain)

	# Agent's strategy for selecting action at a given state
	def getAction(self,rng,domain,space,currentScore,returnIndexSelected=False):

		eps = self.params["eps"]


		ts = space.PI[currentScore]
		vs = space.PI_EV[currentScore]

		# Store actual true reward = EV best action
		self.trueRewards.append(vs)

		
		minEV = min(space.flatEVsPerState[currentScore])
		maxEV = max(space.flatEVsPerState[currentScore])
		meanEV = np.mean(space.flatEVsPerState[currentScore])

		# The bigger the epsilon, the less actions possible (closer to the action to the max ev)
		# The smaller the epsilon, the more possible actions
		
		# PREV
		# epsEV = minEV + eps * (maxEV-minEV)
		
		# NEW
		epsEV = meanEV + eps * (maxEV-meanEV)
		# print("epsEV: ", epsEV) 


		# Find all the evs that are >= epsEV
		# Note: rounding values for comparison because comparing floats
		# And in some cases, causes list of possibleEvsIndexes to be empty
		# Example: 
		# 	np.max(space.flatEVsPerState[currentScore]) = 0.12423437960304905
		# 	epsEV = -0.12423437960304895
		possibleEvsIndexes = np.where(np.round_(space.flatEVsPerState[currentScore],decimals=4) >= round(epsEV,4))[0]
		# print("possibleEvsIndexes: ", possibleEvsIndexes)

	
		possibleEvs = np.array(space.flatEVsPerState[currentScore])[possibleEvsIndexes]
		# print "possibleEvs: ", possibleEvs

		# find the target that is furthest away from the target that gives the max ev
		dist = []

		possibleTargets = np.array(space.actions)[possibleEvsIndexes]

		# Find distances
		for ti in possibleTargets:
			dist.append(domain.actionDiff(ti,ts))


		# get the max distance
		maxDist = max(dist)
		
		# code.interact("TRICKER - getAction()", local=dict(globals(), **locals()))


		# get the index of the max distance
		maxDistIndex = dist.index(maxDist)

		# get the action that corresponds to the maximum distance (the farthest one)
		action = possibleTargets[maxDistIndex]


		evAction = self.getValueIntendedAction(space,currentScore,action)

		if returnIndexSelected:
			return action, evAction, maxDistIndex
		else:
			return action,evAction


class BoundedAgent(Agent):

	def __init__(self,params,domain):

		# L = Lambda
		name = "BoundedAgent-X"+str(params['noise_level'])+"-L"+str(params['lambda'])
		super(BoundedAgent, self).__init__(name, params, domain)

		self.nansInfoS = []
		self.nansInfoCounter = []
		self.nansInfoFinalL = []

		self.allProbs = []

	def getAllProbs(self):
		# return as a dict in order to save it to json file
		return {"allProbs": self.allProbs}

	def getNansInfo(self):
		# return the dictionary containing the corresponding information for when the NANs occurred
		return self.nansInfoS, self.nansInfoCounter, self.nansInfoFinalL

	def computeProbs(self,L,allEVs,domain):

		aSum = 0.0

		allEVs = np.array(allEVs)

		probs = np.zeros((len(allEVs),1))

		# To be used for exp normalization trick - find maxEV and * by L
		# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
		b = np.max(allEVs*L)

		allEVsL = np.copy(allEVs)*L
		
		# With normalization trick
		y = np.exp(allEVsL-b)

		# Without normalization trick - causes warning since probs are 0's
		# y = np.exp(allEVsL)

		# Normalize
		probs = y/np.sum(y)

		# code.interact("inside computeProbs()", local=locals())
		return probs

	def verifyProbs(self,probs,L,allEVs,nansFlag,counter,domain):

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
			probs = self.computeProbs(L,allEVs,domain)
			# print "sum probs with new L: ", np.sum(probs)

			counter += 1

			# verify probs again
			return self.verifyProbs(probs,L,allEVs,nansFlag,counter,domain)

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
	def getAction(self,rng,domain,space,currentScore,returnProbs=False):

		L = self.params['lambda']

		# Store actual true reward = max EV
		self.trueRewards.append(space.PI_EV[currentScore])

		
		probs = self.computeProbs(L,space.flatEVsPerState[currentScore],domain)

		# keep track of how many times probs were updated/recomputed & lambda shrunk
		counter = 0
		nansFlag = False

		finalProbs,finalL,nansFlag,counter = self.verifyProbs(probs,L,space.flatEVsPerState[currentScore],nansFlag,counter,domain)

		self.allProbs.append(finalProbs.tolist())

		
		# if recomputing of the probs happened
		if nansFlag:    
			self.nansInfoS.append(S)
			self.nansInfoCounter.append(counter)
			self.nansInfoFinalL.append(finalL)


		# Since now flatten/parallel arrays

		N = len(space.actions)

		# select action from possible actions with probs as weight -- by index
		indexAction = rng.choice(range(N),p=finalProbs)

		# Map from index back to action
		action = space.actions[indexAction]


		if returnProbs:

			# Find the action with the max prob
			maxProb = np.max(finalProbs)

			# Find the indexes of the action with the max prob
			maxProbIndex = np.where(finalProbs==maxProb)

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
				mp = rng.choice(range(0,len(maxProbIndex[0])),p=indexProbs)

				# "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
				#xmp, ymp = np.unravel_index(mp,finalProbs.shape)
				xmp = np.unravel_index(mp,finalProbs.shape)

				maxProbX = xmp
				#maxProbY = ymp

			else:
				maxProbX = maxProbIndex[0]
				#maxProbY = maxProbIndex[1]


			# get action corresponding to the indices
			#actionMaxProb = [all_ts["all_ts_x"][maxProbX],all_ts["all_ts_y"][maxProbY]]
			actionMaxProb = space.actions[maxProbX]


		evAction = self.getValueIntendedAction(space,currentScore,action)

		#code.interact("BOUNDED - getAction()", local=dict(globals(), **locals()))

		if returnProbs:
			return action,evAction,nansFlag,finalProbs,actionMaxProb
		else:
			return action,evAction,nansFlag





############ YET TO UPDATE ############
class RandomAgent(Agent):

	def __init__(self, params, domain):

		# N = number of actions to select
		# K = # of times to sample each action
		# the bigger the parameters, the more rational the agent is

		name = "RandomAgent-X"+str(params['noise_level'])+"-N"+str(params['num_actions'])+"-K"+str(params['num_samples'])
		super(RandomAgent, self).__init__(name, params, domain)


	# Agent's strategy for selecting action at a given state
	def getAction(self,rng, S, delta, convolutions, domain, returnZn = False):

		all_ts, all_vs, ts, vs, convolutions, meanAllVs = super(RandomAgent,self).getAction(S, delta, convolutions, domain, returnZn)


		# select n random aimings at the darts board

		possibleActions = []

		#self.all_ts_2d = np.array([all_ts["all_ts_y"]] * len(all_ts["all_ts_x"]))
		#for xj in range(len(self.all_ts_2d)):
		#    for yj in range(len(self.all_ts_2d[0])):
		#        targetAction = [targetsC["all_ts_x"][xj],self.all_ts_2d[xj][yj]]


		for i in range(self.params["num_actions"]):

			xi = rng.randint(0,len(all_ts["all_ts_x"]))
			#yi = rng.randint(0,len(all_ts["all_ts_y"]))


			x = all_ts["all_ts_x"][xi]

			#y = all_ts["all_ts_y"][yi]
			# Since now flatten
			y = all_ts["all_ts_y"][xi]

			possibleActions.append([x,y])


		rewards = []

		# sample each one of them K times
		for a in possibleActions:

			rs = []

			for j in range(self.params["num_samples"]):
				na = domain.sample_action(S, self.params["noise_level"], a, noiseModel = None)
				rs.append(domain.get_v(S, na))

			rewards.append(sum(rs)/float(self.params["num_samples"]))

			
		# pick the action one that yields the highest reward based on its observed reward (the one with most hits)

		# find highest reward
		maxR = max(rewards)

		# find index of highest rewards
		maxRI = rewards.index(maxR)

		# fins which action yield such reward
		action = possibleActions[maxRI]

		# get ev of selected action
		evAction = self.getEvIntendedAction(all_ts, all_vs, action)

		#code.interact("getAction()", local=locals())


		# returns action selected by agent
		return action, convolutions, evAction, meanAllVs


############ YET TO UPDATE ############
class DeltaAgent(Agent):

	def __init__(self, params, domain):

		name = "DeltaAgent-X" + str(params['noise_level']) + "-Delta" + str(params["delta"]) 
		super(DeltaAgent, self).__init__(name, params, domain)

	# Agent's strategy for selecting action at a given state
	def getAction(self, rng, S, delta, convolutions, domain, returnZn = False):

		# Acts similar to Target Agent but uses the agent's delta parameter as the resolution for the convolution
		all_ts, all_vs, ts, vs, convolutions, meanAllVs = super(DeltaAgent,self).getAction(S, self.params["delta"], convolutions, domain, returnZn)

		# returns action that has the highest ev for the given resolution
		return ts, convolutions, vs, meanAllVs


#class EpsilonAgent(Agent):
#    pass