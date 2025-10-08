from abc import ABCMeta, abstractmethod

import code
import warnings

from scipy import stats
from scipy import integrate
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
import numpy as np

#Agents for the darts domain
#The agent will choose an action for a given state given their xskill level

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


		if "TargetAgent-BeliefX" in self.name:
			self.true_rewards_beliefX = []

			# self.convolutions_all_ts_beliefX = []
			self.convolutions_all_vs_beliefX = []
			self.convolutions_all_mvs_beliefX = []
			self.convolutions_ts_beliefX = []
			self.convolutions_vs_beliefX = []

		self.domainName = domain.getDomainName()

		if self.domainName == "1d":
			self.noiseModel = None # domain.getNoiseModel(L)

		elif self.domainName == "2d":
			self.noiseModel = domain.getNoiseModel(rng,params['noise_level']**2)

	def getInfoConvolutions(self):

		if "TargetAgent-BeliefX" in self.name:
			# return the one containing 'all' as a dict in order to save it to file
			return {"convolutions_all_vs" : self.convolutions_all_vs},\
					 self.convolutions_ts, self.convolutions_vs, self.convolutions_all_mvs,\
					 {"convolutions_all_ts_beliefX" : self.convolutions_all_ts_beliefX}, {"convolutions_all_vs_beliefX" : self.convolutions_all_vs_beliefX},\
					 self.convolutions_ts_beliefX, self.convolutions_vs_beliefX, self.convolutions_all_mvs_beliefX
		else:
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

		return convolutionSpace["all_vs"], convolutionSpace["ts"], convolutionSpace["vs"]


	def get_true_rewards(self):
		return self.true_rewards

	def getName(self):
		return self.name

	def print_agent(self):
		print('Agent: ', self.name)
		print(' Params: ', self.params)

	def getEvIntendedAction(self,listedTargets,values,action):

		if self.domainName == "1d":
			
			# Interpolate in order to get the EV for the intendsed action
			evAction = np.interp(action,listedTargets,values)

		elif self.domainName == "2d":

			# For action
			ai = np.zeros((1,2))
			ai[0][0] = action[0] # First action dimension
			ai[0][1] = action[1] # Second action dimension

			# Using cubic interpolation since 2D
			# "linear" was used before and values where about the same but slightly smaller
			# Thus, resulting in a negative %Reward (when computing expected --- mean since expected < mean )
			# for some "bad" agents
			# evAction = griddata(listedTargets,values.ravel(),ai,method='cubic',fill_value=np.min(listedValues))[0]
			evAction = griddata(listedTargets,values.ravel(),ai,method='cubic')[0]
			# code.interact("getEvIntendedAction()", local=dict(globals(), **locals()))
		
		return evAction


class TargetAgent(Agent):

	def __init__(self,rng,params,domain):

		name = "TargetAgent-X" + str(params['noise_level']) 
		super(TargetAgent, self).__init__(rng,name, params, domain)

	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False):

		all_vs,ts,vs = super(TargetAgent,self).get_action(convolutionSpace,returnZn)

		# Returns action that has the highest ev
		# print(ts,vs)
		return ts, vs


class FlipAgent(Agent):

	def __init__(self,rng,params,domain):

		name = "FlipAgent-X"+str(params['noise_level'])+"-P"+str(params['prob_rational'])
		super(FlipAgent, self).__init__(rng,name, params, domain)

	
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

			if self.domainName == "1d":
				action = rng.uniform(-domain.m,domain.m) 

				evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)

				return action,evAction
			
			elif self.domainName == "2d":
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


				# plt.scatter(x,y)
				# domain.draw_board(ax)
				# plt.show()

				# code.interact("FLIP - getAction()", local=dict(globals(), **locals()))

				# if self.params['noise_level'] >= 150:
					# code.interact("...", local=dict(globals(), **locals()))
				
				return action,evAction


class TrickerAgent(Agent):

	def __init__(self,rng, params, domain):
		
		name = "TrickerAgent-X" + str(params["noise_level"])+"-Eps" + str(params["eps"])
		super(TrickerAgent, self).__init__(rng,name, params, domain)

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


		# find all the evs that are >= epsEv
		possibleEvsIndexes = np.where(all_vs >= epsEV)
		# print "possibleEvsIndexes: ", possibleEvsIndexes

		possibleEvs = all_vs[possibleEvsIndexes]
		# print "possibleEvs: ", possibleEvs

		# Find the target that is furthest away from the target that gives the max ev
		dist = []

		if self.domainName == "1d":

			possibleTargets = listedTargets[possibleEvsIndexes]

			for eachPossibleTarget in possibleTargets:
				dist.append(domain.actionDiff(eachPossibleTarget,ts))

		###############################################################################
		elif self.domainName == "2d":

			possibleTargetsX = listedTargets[:,0][possibleEvsIndexes]
			possibleTargetsY = listedTargets[:,1][possibleEvsIndexes]

			# Find distances
			for i in range(len(possibleTargetsX)):
				eachPossibleTarget = [possibleTargetsX[i],possibleTargetsY[i]]
				dist.append(domain.actionDiff(eachPossibleTarget,ts))



		# Get the max distance
		maxDist = max(dist)


		# Get the index of the max distance
		maxDistIndex = dist.index(maxDist)

		if self.domainName == "1d":
			# Get the action that corresponds to the maximum distance (the farthest one)
			action = possibleTargets[maxDistIndex]

		elif self.domainName == "2d":
			# Get the action that corresponds to the maximum distance (the farthest one)
			action = [possibleTargetsX[maxDistIndex],possibleTargetsY[maxDistIndex]]


		evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)

		# code.interact("TRICKER - getAction()", local=dict(globals(), **locals()))

		if returnIndexSelected:
			return action,maxDistIndex,evAction
		else:
			return action,evAction


class BoundedAgent(Agent):

	def __init__(self,rng,params,domain):

		# L = Lambda
		name = "BoundedAgent-X"+str(params['noise_level'])+"-L"+str(params['lambda'])
		super(BoundedAgent, self).__init__(rng,name, params, domain)

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

		if self.domainName == "1d":

			# 1D array
			probs = np.ndarray(shape = (len(all_vs),1))

			probs.fill(1.0/(len(all_vs) * 1.0))


			# Exp normalization trick - find maxEV and * by p
			# To avoid "overflow encountered in exp" warning for some p's since numbers become too big for exp func when multiplied by EVs
			# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
			

			# Create copy of EVs 
			evsCP = np.copy(all_vs)


			# Bounded decision-making with lambda = l
			b = np.max(evsCP*L)
			expev = np.exp(evsCP*L-b)
			sumexp = np.sum(expev)
			upd = expev/sumexp


			# reshape array - so that multiplication results in correct shape (element-wise multiplication)
			upd.shape = (len(upd),1)
			
			# Update
			probs = np.multiply(probs,upd)

			# To remove sigle-dimensional entries from the array
			# Probs format = [[1],[2],[3],...]
			# Squeeze will change format to [1,2,3,...]
			probs = np.squeeze(probs)


			#Normalize probs
			probs /= np.sum(probs)

			#code.interact("qre ",local=locals())


		elif self.domainName == "2d":

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

		all_vs,ts,vs = super(BoundedAgent, self).get_action(convolutionSpace,returnZn)

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


		if self.domainName == "1d":
			# select action from possible actions with probs as weight
			action = np.random.choice(listedTargets,p=finalProbs)


			# Find the action with the max prob
			maxProb = np.max(finalProbs)

			# Find the indexes of the action with the max prob
			maxProbIndex = np.where(finalProbs == maxProb)


			actionMaxProb = listedTargets[maxProbIndex][0]

			#code.interact("get_action()", local=locals())

		elif self.domainName == "2d":

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


class RandomAgent(Agent):

	def __init__(self,rng, params, domain):

		# N = number of actions to select
		# K = # of times to sample each action
		# the bigger the parameters, the more rational the agent is

		name = "RandomAgent-X"+str(params['noise_level'])+"-N"+str(params['num_actions'])+"-K"+str(params['num_samples'])
		super(RandomAgent, self).__init__(rng,name, params, domain)


	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False):

		all_vs,ts,vs = super(RandomAgent,self).get_action(convolutionSpace,returnZn)


		if self.domainName == "1d":

			# randomly pick N actions from the target actions
			possibleActions = np.random.choice(listedTargets, self.params["num_actions"])

			rewards = []

			# sample each one of them K times
			for a in possibleActions:
				rewards.append(domain.sample_N(rng,S,self.params["noise_level"],self.params["num_samples"],a))


		elif self.domainName == "2d":

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
		evAction = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],action)

		#code.interact("get_action()", local=locals())


		# returns action selected by agent
		return action, evAction


###########################################
# NEED TO CHECK
###########################################

class TargetAgentWithBeliefs(Agent):

	def __init__(self,rng, params, domain):

		name = "TargetAgent-BeliefX" + str(params['noise_level_belief']) + "-TrueX" + str(params['noise_level']) 
		super(TargetAgentWithBeliefs, self).__init__(rng,name, params, domain)

	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZN=False):

		# Action is selected from possible targets for given beliefX (action = max EV action from beliefX convolution)
		# Actual EV of that action is then obtained from the true EVs (from the trueX convolution) 

		# With True xskill (4), then with Belief xskill (next 4)
		all_vs, ts, vs,\
		all_ts_beliefX, all_vs_beliefX, ts_beliefX, vs_beliefX,\
		meanAllVs, meanAllVsBeliefX = super(TargetAgentWithBeliefs,self).get_action(convolutionSpace,returnZn)
		
		# Max EV action from beliefX
		actionBeliefX = ts_beliefX

		# Find how much EV the agent will really get - from trueX (all_ts trueX, all_vs trueX) 
		evActionBeliefXOnTrueX = self.getEvIntendedAction(listedTargets,convolutionSpace["all_vs"],actionBeliefX)

		#code.interact("", local=locals())

		# returns action that has the highest ev (but from the beliefX) along with actual EV (from trueX)
		return actionBeliefX, evActionBeliefXOnTrueX, meanAllVs


class DeltaAgent(Agent):

	def __init__(self,rng, params, domain):

		name = "DeltaAgent-X" + str(params['noise_level']) + "-Delta" + str(params["delta"]) 
		super(DeltaAgent, self).__init__(rng,name, params, domain)

	# Agent's strategy for selecting action at a given state
	def get_action(self,rng,domain,listedTargets,convolutionSpace,returnZn=False):

		# Acts similar to Target Agent but uses the agent's delta parameter as the resolution for the convolution
		all_vs,ts,vs = super(DeltaAgent,self).get_action(convolutionSpace,returnZn)

		# returns action that has the highest ev for the given resolution
		return ts,vs

###########################################
