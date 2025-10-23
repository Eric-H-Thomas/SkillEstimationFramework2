import numpy as np
import scipy
import json,pickle
import os,time,sys
import code
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from scipy.signal import convolve2d


# For Sequential 2D-Darts - Given xskill - Value Iteration
class StateActionValueSpace():

	__slots__ = ["mode","numObservations","resolution","x",
				"domainName","boardStates","states","actions","delta",
				"valueIterFolder","expectedReward",
				"flatEVsPerState","meanEVsPerState","V","PI","PI_EV","gamma",
				"tolerance","iterations","allEVs"]#,"flatTargetActions",Xn","Yn","Zn"]

	def __init__(self,mode,numObservations,domain,delta,possibleTargets,x,numSamples,expectedRFolder=None,valueIterFolder=None,fromEstimator=False,testing=False):
		
		# np.random.seed(0)

		self.mode = mode
		self.numObservations = numObservations
		self.resolution = delta
		self.x = x

		self.domainName = domain.get_domain_name()


		# If mode == normal, list will have same state (normal board) * numObservations
		# Since assuming normal mode and same dartboard state for now, 
		# boardsStates will just be 1 element (list of slices)
		self.boardStates = np.array(domain.getBoardStates(numObservations,self.mode)[0])


		# States = Player's possible scores
		startScore = domain.getPlayerStartScore()
		self.states = list(range(startScore+1))


		# Get actions and their probs
		# Can get once since always the same (set of targets independent of state)

		self.actions = possibleTargets


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Perform value iteration 
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		if valueIterFolder == None:
			valueIterFolder = "Spaces" + os.sep + "ValueFunctions" + os.sep

		self.valueIterFolder = valueIterFolder


		toLoad = f"{valueIterFolder}ValueFunc-Domain{domain.get_domain_name()}-Resolution{self.resolution}-Xskill{x:.4f}"

		# Load file containing learned value function for xskill, if it exists already
		if os.path.exists(toLoad):
			print(f"\nLoading value function for xskill {x}")
			self.loadValueIterInfo(toLoad)
		# Perform value iteration if info is not present already
		else:
			print(f"\nPerforming value iter for xskill {x}")
			self.valueIteration(domain,valueIterFolder,testing)

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Find Expected Rewards (avg # of turns)
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		# Note: Finding expected rewards after value iter since the getAction agent method 
		# inside getAvgTurns needs access to the value iter info

		expectedRewardsInfo = {}


		# Load all available files from Spaces/ExpectedRewards folder
		availableFiles = os.listdir(f"Spaces{os.sep}ExpectedRewards")
		files = []

		# Keep only files that contain ExpectedRewards info for current domain
		for each in availableFiles:
			if "ExpectedRewards" in each and self.domainName in each:
				if int(each.split("-N")[1]) == numSamples:
					files.append(each)


		found = False

		# Load files that contains historical data, if present
		if len(files) != 0:
		
			print(f"Loading expected rewards for existing xskills (N = {numSamples})")

			for each in files:

				try:
					# Load info from file
					with open(f"Spaces{os.sep}ExpectedRewards{os.sep}{each}") as inFile:
						temp = json.load(inFile)

					# Update dict
					expectedRewardsInfo.update(temp)
				except:
					continue


			# Based on % times hitting the board (from 2D)
			# Setting to 1.0 for testing
			# NEED TO UPDATE ACCORDINGLY
			threshold = 1.0


			# Find if there's any xskill close enough to current one to use its 
			# expected reward info

			dist = []
			tempXs = list(expectedRewardsInfo.keys())

			for eachX in tempXs:
				# Find distance between current xskill and available ones
				dist.append(abs(self.x-float(eachX)))

			# Find smallest distance
			minDist = min(dist)


			# If within acceptable threshold
			if minDist <= threshold:

				# Find xskill with such distance
				indexClosest = dist.index(minDist)
				closestXskill = tempXs[indexClosest]
				found = True


		# Load info into expecedRewards as xskill present
		if str(x) in expectedRewardsInfo:
			print(f"Loading info for xskill {x} (N = {numSamples})")
			self.expectedReward = expectedRewardsInfo[str(x)]
	
		elif found:
			print(f"Loading info for closest xskill available: {closestXskill} (actual xskill = {self.x}) (N = {numSamples})")
			self.expectedReward = expectedRewardsInfo[str(closestXskill)]

		# Compute expected rewards and save to file
		else:
			print(f"Computing expected rewards for xskill {x} (N = {numSamples})")

			# convInfo = self.getConvInfo(domain)
			self.expectedReward = self.getAvgTurns(domain,numSamples)

			expectedRewardsInfo[str(x)] = self.expectedReward

			# Creating copy with just newly obtain info
			# To only save that to file
			# Since expectedRewardsInfo now has all available info from all files
			tempNewInfo = {}
			tempNewInfo[str(x)] = self.expectedReward


			# Load existing info from file (if any)
			try:
				with open(expectedRFolder) as inFile:
					temp = json.load(inFile)

				# Update dict
				tempNewInfo.update(temp)
				
				del temp

			except:
				pass

			
			# Update file as new info was obtained
			try:
				with open(expectedRFolder,'w') as outfile:
					json.dump(tempNewInfo,outfile)

				del tempNewInfo

			except:
				pass


		del expectedRewardsInfo
		# code.interact("...", local=dict(globals(), **locals()))
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Flattening arrays and compute mean EVs per state/score
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		#self.flatTargetActions = []
		# self.flatTargetActions = self.actions

		self.flatEVsPerState = {}
		self.meanEVsPerState = {}

		'''
		for xi in range(len(self.Xs)):
			for yi in range(len(self.Ys)):
				self.flatTargetActions.append([self.Xs[xi],self.Ys[yi]])
		'''

		'''
		# Flatten each array of EVs - for each score/state
		for s in self.allEVs:
			tempFlatEVs = []
			for xi in range(len(self.Xs)):
				for yi in range(len(self.Ys)):
					# allEVs = (69,69)
					tempFlatEVs.append(self.allEVs[s][0][xi][yi])
					#code.interact("...", local=dict(globals(), **locals()))

			# Compute random reward for EVs of current state
			tempMeanEVs = np.mean(tempFlatEVs) 

			self.flatEVsPerState[s] = tempFlatEVs
			self.meanEVsPerState[s] = tempMeanEVs
		'''

		for s in self.allEVs:
			self.flatEVsPerState[s] = self.allEVs[s]
			self.meanEVsPerState[s] = float(np.mean(self.allEVs[s]))

		# code.interact("!!!...", local=dict(globals(), **locals()))

		# Verify ok to perform this instruction
		# Make sure that it doesn't affect anything else
		del self.allEVs
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



	def valueIteration(self,domain,saveTo,testing=False):

		#################################
		# np.random.seed(0)
		#################################

		if testing:
			xSkillFolder = saveTo + "plots-xskill" + str(self.x) + os.sep

			#If there's not a folder , create it
			if not os.path.exists(xSkillFolder):
				os.mkdir(xSkillFolder)


		startTime = time.time()

		##############################################################################
		# SET PARAMS FOR VALUE ITERATION
		##############################################################################

		# Discount factor
		self.gamma = 1.0

		# Error tolerance (when do we stop?)
		self.tolerance = 0.001 * self.x

		# How much did it change this iter?
		self.delta = 10.0

		self.iterations = 0


		#######################################
		# Initialize values
		#######################################

		closestFound = False

		# Load available xskills
		availableXskills = self.findAvailableXskills()

		if availableXskills != []:

			distances = []
			# Find distance between current xskill and available ones
			for tx in availableXskills:
				distances.append(abs(self.x-tx))

			# Find smallest distance
			minDist = min(distances)

			# Find xskill with such distance
			indexClosest = distances.index(minDist)
			closestXskill = availableXskills[indexClosest]
			closestFound = True
			# code.interact("loading...", local=dict(globals(), **locals()))


		##############################
		# Used for testing purposes
		# closestFound = False
		##############################

		# Initialize values to those of the closest xskill possible 
		if closestFound:
			print(f"Initializing values to (previously learned) values of closest xskill {closestXskill}")
			
			toLoad = f"{self.valueIterFolder}ValueFunc-Domain{domain.get_domain_name()}-Resolution{self.resolution}-Xskill{closestXskill:.4f}"
			
			ok = False

			while not ok:

				try:
					with open(toLoad,"r") as infile:
						results = json.load(infile)
						self.V = np.array(results["V"])
						ok = True
				except:
					continue

			# code.interact("...", local=dict(globals(), **locals()))
		else:
			
			startScore = domain.getPlayerStartScore()
			# self.V = -1 * np.linspace(self.x,1.5*self.x,startScore+1)
			self.V = -1 * np.linspace(1,16,startScore+1)


			# 0.0 since game ends (done)
			self.V[0] = 0.0
			# 0.0 since never gets here
			self.V[1] = 0.0

		#######################################


		#######################################
		# Initialize other things
		#######################################
		self.PI = [[None,None]] * len(self.states)
		self.PI_EV = [None] * len(self.states)


		numSamples = 100_000
		singleProbs, doubleProbs = self.precalc(domain,self.actions,self.x**2,numSamples)

		#######################################

		##############################################################################


		# code.interact("Before vi...", local=dict(globals(), **locals()))

		##############################################################################
		# PERFORM VALUE ITERATION
		##############################################################################

		while self.delta > self.tolerance:

			# To remember EVs for different scores/states
			# Resets each time as interested in 
			# remembering the EVs once converged only
			allEVs = {}

			# Reset delta
			self.delta = 0.0
			
			# Skip states 0 & 1
			# for s in self.states[2:]:
			for s in range(2,len(self.V)):

				# print(s)
				# code.interact("HERE...", local=dict(globals(), **locals()))

				# To print every 5 iters and 20 states
				if self.iterations%5 == 0 and s%20 == 0:
					print("Iteration: " + str(self.iterations) + " | State: " + str(s))
				

				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Compute EVs - With convolution
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				'''

				# Get values
				Xs,Ys,Zs = domain.getValues(self.boardStates,self.resolution,s,self.V)

				# Convolve to produce the EV and aiming spot
				EV = convolve2d(Zs,self.Zn,mode="same",fillvalue=self.V[s])
				
				'''
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Compute EVs - Code from Thomas Miller (Dr. Archibald's student)
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# '''

				if s <= 61:
					score_change = singleProbs[:,:s-1] @ np.flip(self.V[2:s+1])
					bust = np.sum(singleProbs[:,s-1:],axis=1) * self.V[s]
					
					doub_change = doubleProbs[:,:s-1] @ np.flip(self.V[2:s+1])
					doub_bust = (np.sum(doubleProbs[:,s+1:],axis=1) + doubleProbs[:,s-1])  * self.V[s]

					EV = score_change+bust+doub_change+doub_bust
					
				else:
					score_change = (singleProbs + doubleProbs) @ np.flip(self.V[s-60:s+1])
					EV = score_change

				# '''
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


				# Get max EV
				bestEV = np.max(EV)	


				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Find best action
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				
				# PREV - with convolution
				#mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
				#action = [Xs[mxi],Ys[myi]]

				# Now
				mi = np.unravel_index(EV.argmax(), EV.shape)[0]
				action = [self.actions[mi][0],self.actions[mi][1]]

				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				

				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Save info
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

				#if s not in allEVs:
				#	allEVs[s] = []
				#allEVs[s].append(EV)

				allEVs[s] = EV

				self.PI_EV[s] = float(bestEV)
				self.PI[s] = action
			
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				

				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Update info
				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				
				# How much are we going to change the value?
				currentDelta = abs(self.V[s] + 1 - self.gamma*bestEV) 

				if currentDelta > self.delta:
					self.delta = currentDelta						
				
				# Update value of state with highest value
				# Value of state = direct reward of action + expected value of next states
				self.V[s] = -1 + self.gamma*bestEV

				# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


			# For testing - plot current values
			if testing:
				# '''
				fig = plt.figure()
				ax = plt.gca()
				cmap = plt.get_cmap("viridis")
				norm = plt.Normalize(min(self.V),max(self.V))
				ax.scatter(self.states,self.V, c=cmap(norm(self.V)))
				sm = ScalarMappable(norm=norm, cmap=cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm)
				cbar.ax.set_title("Values")

				fig.suptitle("xSkill: " + str(self.x) + " | Iteration: " + str(self.iterations))
				#fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad = 0.2)
				plt.savefig(xSkillFolder + "values-xSkill" + str(self.x) + "-Iteration" + str(self.iterations) + ".png")

				plt.clf()
				plt.close()
				# '''

			# print("Current Delta: ", self.delta)
			# code.interact(f"after iter {self.iterations}...", local=dict(globals(), **locals()))
			
			self.iterations += 1



		# code.interact("after vi...", local=dict(globals(), **locals()))

		stopTime = time.time()
		totalTime = stopTime - startTime
		##############################################################################


		##############################################################################
		# SAVE INFO TO JSON FILE
		##############################################################################

		results = {}

		results['xskill'] = self.x
		
		results['gamma'] = self.gamma
		results['tolerance'] = self.tolerance
		results['iterations'] = self.iterations

		results['V'] = self.V.tolist()
		results['PI'] = self.PI
		results['PI_EV'] = self.PI_EV

		results['totalTime'] = totalTime

		folder = f"{saveTo}ValueFunc-Domain{self.domainName}-Resolution{self.resolution}-Xskill{self.x:.4f}"
		# code.interact("...", local=dict(globals(), **locals()))


		ok = False

		while not ok:
			
			try:
				# Save to json file
				with open(folder,'w') as outfile:
					json.dump(results,outfile)

				self.allEVs = allEVs

				with open(folder+".pickle", "wb") as outfile:
					pickle.dump(allEVs,outfile)

				ok = True

			except:
				continue

		# code.interact("...", local=dict(globals(), **locals()))

		##############################################################################


	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# Code from: Thomas Miller (Dr. Archibald's student)
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	# VECTORIZED
	def precalc(self,domain,mus,var,sample_size=100000):

		N = scipy.stats.multivariate_normal([0,0],var)

		
		# Generate noises (sample from noise model)
		ps = N.rvs(size=sample_size)
		
		# To store probs
		# shape = (for each target, possible scores(60))
		non_doubs = np.zeros((mus.shape[0],61),dtype=np.float64) #np.longdouble)
		doub_a = np.zeros_like(non_doubs,dtype=np.float64) #np.longdouble)

		# non_doubs = np.zeros((mus.shape[0],61),dtype=np.longdouble)
		# doub_a = np.zeros_like(non_doubs,dtype=np.longdouble)
		

		# Monte-Carlo sampling

		# For a given target action
		for i,mu in enumerate(mus):
			
			# Add noises to target action 
			# p = an array that contains target action shifted by the different noises
			p = ps+mu

			# Find respective score for each noisy action
			ss,doubs = domain.npscore(p[:,0],p[:,1],return_doub=True)
			
			# Find scores of actions that don't result in doubles 
			# Indices not repeated since using np.unique
			# nonc = counts for each unique score
			nond, nonc = np.unique(ss[~doubs],return_counts=True)
			x = np.sum(nonc)
			
			# Find scores of actions that result in doubles 
			# Indices not repeated since using np.unique
			# nonc = counts for each unique score
			doub,c = np.unique(ss[doubs],return_counts=True)
			y = np.sum(c)

			# Save probs
			non_doubs[i,nond] = nonc/sample_size
			doub_a[i,doub] = c/sample_size
			
		# code.interact("precalc()...", local=dict(globals(), **locals()))
		
		########################################
		# non_doubs *= np.square(self.resolution)
		# doub_a *= np.square(self.resolution)
		########################################
		
		return non_doubs, doub_a
	
	# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	def loadValueIterInfo(self,toLoad):

		with open(toLoad, "r") as infile:
			results = json.load(infile)

			self.gamma = results["gamma"]
			self.tolerance = results["tolerance"]
			self.iterations = results["iterations"]

			self.V = results["V"]
			self.PI = results["PI"]
			self.PI_EV = results["PI_EV"]


		with open(toLoad+".pickle","rb") as file:
			info = pickle.load(file)

			self.allEVs = info
		# code.interact("...", local=dict(globals(), **locals()))

	def findAvailableXskills(self):

		availableXskills = []

		if os.path.exists(self.valueIterFolder):
	
			# Load all files on value iter folder
			valueIterFiles = os.listdir(self.valueIterFolder)

			# for each file
			for eachFile in valueIterFiles:
				# print(eachFile)

				if "pickle" not in eachFile and "plots" not in eachFile and ".DS_Store" not in eachFile:
					xskill = float(eachFile.split("Xskill")[1])

					if xskill not in availableXskills:
						availableXskills.append(xskill)

		return availableXskills

	
	def getConvInfo(self,domain):

		startScore = domain.getPlayerStartScore()

		convInfo = {}

		print("\tDoing convolution for different states...")

		Xn,Yn,Zn = domain.getSymmetricNormalDistribution(self.x,self.resolution)

		for state in range(startScore+1):

			convInfo[state] = {}

			value = self.V[state]

			Xs,Ys,Zs = domain.getValues(self.boardStates,self.x,self.resolution,state,self.V)

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,Zn,mode="same",fillvalue=value)

			convInfo[state]["EVs"] = EVs

			# Get maximum of EV
			mxi, myi = np.unravel_index(EVs.argmax(), EVs.shape)
			
			# Best aiming point
			mx = Xn[mxi]
			my = Yn[myi]

			convInfo[state]["mx"] = mx
			convInfo[state]["my"] = my

		return convInfo

	def getAvgTurns(self,domain,N=1000):

		print("\tSimulating games to find avg # of turns...")

		allNumTurns = 0.0

		numTurnsPerGame = []

		# Simulate game N times
		for i in range(N):
			# print(f"X: {x} | Simulation #{i}/{N}")

			numTurns = 0.0
			currentScore = domain.getPlayerStartScore()

			# While playing the game (0 = last possible score/state -> game done)
			while currentScore > 0:
				# print(f"X: {x} | Current score: {currentScore}")

				action = self.PI[currentScore]

				# Sample action
				noisyAction = domain.sampleAction(self.x,action)
				
				# newScore,double = domain.score(boardStates,noisyAction)
				newScore,double = domain.npscore(noisyAction[0],noisyAction[1],return_doub=True)
				numTurns += -1

				nextScore = currentScore-newScore


				#Did we bust (score too much)?
				# Less than 0 or exactly 1
				if nextScore < 0 or nextScore == 1:
					nextScore = currentScore 

				#Did we double out correctly?
				if nextScore == 0:
					if not double:
						nextScore = currentScore 

				# print("currentScore: ", currentScore)
				# print("noisyAction:", noisyAction)
				# print("newScore:", newScore)
				# print("nextScore:", nextScore)
				# code.interact("...", local=dict(globals(), **locals()))
				
				currentScore = int(nextScore)


				# To limit possible number of turns as agents with bad xskill can
				# take lots of turns to finish an actual game
				if numTurns <= -500:
					break


			allNumTurns += numTurns
			numTurnsPerGame.append(numTurns)
		
		# code.interact("...", local=dict(globals(), **locals()))

		avgTurns = allNumTurns/N

		# if abs(space.V[startScore]-avgTurns) > 5.0:
		# 	code.interact("...", local=dict(globals(), **locals()))

		return avgTurns


	def getTargets(self):
		return self.Xs, self.Ys

	def getTargetsProbs(self):
		return self.Zn

	def getV(self):
		return self.V

	def getPI(self):
		return self.PI

	def getPI_EV(self):
		return self.PI_EV


# For Billards - Given xskill - Success Rate
class StateActionSpaceBilliards():

	__slots__ = ["numObservations","resolution","agentType","agent","x","domainName","successRatesInfo"]

	def __init__(self,numObservations,domain,delta,agentType,agent,x,numSamples,successRatesFolder=None,fromEstimator=False):


		self.numObservations = numObservations
		self.resolution = delta

		self.agentType = agentType
		self.agent = agent
		self.x = x

		self.domainName = domain.get_domain_name()

		Shot.shot_list = []

		# To store success rate info for current agent(xskill), per agent type
		self.successRatesInfo = {}

		successRatesFolder += f"-{agentType}.json"

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Find Success Rates 
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		successRatesInfoAll = {}
		
		# Load file that contains historical data, if present
		if os.path.exists(successRatesFolder):
			print(f"Loading success rates for existing xskills.")

			# Check if the file has content
			if os.path.getsize(successRatesFolder) != 0:
				# Load info from file
				with open(successRatesFolder) as inFile:
					successRatesInfoAll = json.load(inFile)
			else:
				successRatesInfoAll = {}


		# Load info into expecedRewards as xskill present
		if str(x) in successRatesInfoAll:

			print(f"Loading info for xskill {x}")
			self.successRatesInfo = successRatesInfoAll[str(x)]
		
		# Compute expected rewards and save to file
		else:
			print(f"Computing expected rewards for xskill {x}")

			self.getSuccessRate(numSamples)

			successRatesInfoAll[str(x)] = self.successRatesInfo

			# Update file as new info was obtained
			#print("Updating json file")
			with open(successRatesFolder,'w') as outfile:
				json.dump(successRatesInfoAll,outfile)
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	def getSuccessRate(self,numSamples):

		# Get shots needed (numSamples of them)
		tempProcessedShotsList = getAndProcessShots(numSamples,self.agent)

		successfulCount = 0
		unsuccessfulCount = 0

		# Count how many where successful or not
		for each in range(numSamples):

			if tempProcessedShotsList[each].successfulOrNot == "Yes":
				successfulCount += 1
			else:
				unsuccessfulCount += 1

		successfulTotal = (successfulCount/float(numSamples))
		unsuccessfulTotal = (unsuccessfulCount/float(numSamples))

		self.successRatesInfo = {"successfulCount": successfulCount,
								"successfulTotal": successfulTotal,
								"unsuccessfulCount": unsuccessfulCount,
								"unsuccessfulTotal": unsuccessfulTotal}





