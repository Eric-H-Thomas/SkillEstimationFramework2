import numpy as np
import code
import os

class ObservedReward():

	__slots__ = ["xskills","numXskills","domainName","name","estimates",
				"xsvs","meanReward","i","estimatesFullGame","xsvsFullGame","xsvsMidGame",
				"estimatesMidGame","rewardsCurrentGame","rewardsPerGame",
				"meanRewardFullGame","meanRewardMidGame"]

	def __init__(self,xskills,domainName):

		self.xskills = xskills
		self.numXskills = len(xskills)

		self.domainName = domainName
		self.name = 'OR-' + str(self.numXskills)

		if self.domainName in ["1d","2d","baseball"]:
			self.estimates = []
			self.xsvs = np.array([0.0]*len(self.xskills))
			self.meanReward = 0.0
			self.i = 0

		elif self.domainName in ["sequentialDarts","billiards"]:
	
			self.estimatesFullGame = []
			self.xsvsFullGame = np.array([0.0]*len(self.xskills))

			self.estimatesMidGame = []
			self.xsvsMidGame = np.array([0.0]*len(self.xskills))

			# To keep track of total rewards for a given game
			self.rewardsCurrentGame = 0

			# To keep track of total rewards per game
			self.rewardsPerGame = []

	
			self.meanRewardFullGame = 0.0
			self.meanRewardMidGame = 0.0
		

	def getEstimatorName(self):
		return self.name


	def midReset(self):
		self.estimates = []


	def reset(self):

		if self.domainName in ["1d","2d","baseball"]:
			self.estimates = []
			self.xsvs = np.array([0.0]*len(self.xskills))
			self.meanReward = 0.0
			self.i = 0

		elif self.domainName in ["sequentialDarts","billiards"]:
			self.estimatesFullGame = []
			self.xsvsFullGame = np.array([0.0]*len(self.xskills))

			self.estimatesMidGame = []
			self.xsvsMidGame = np.array([0.0]*len(self.xskills))

			self.meanRewardFullGame = 0.0
			self.meanRewardMidGame = 0.0
			
			# To keep track of total rewards for a given game
			self.rewardsCurrentGame = 0

			# To keep track of total rewards per game
			self.rewardsPerGame = []


	def interpolateRewardSkill(self,estimateType):

		if estimateType == "mid":
			r = self.meanRewardMidGame
			rs = self.xsvsMidGame
		elif estimateType == "full":
			r = self.meanRewardFullGame
			rs = self.xsvsFullGame
		# For 1D, 2D & Baseball 
		else:
			r = self.meanReward
			rs = self.xsvs


		xs = self.xskills

		if r >= max(rs):
			return xs[np.where(rs == max(rs))[0][0]]

		if r <= min(rs):
			return xs[np.where(rs == min(rs))[0][0]]

		for i in range(len(rs)-1):
			if rs[i] >= r and r >= rs[i+1]:
				return xs[i] + (r-rs[i])*(xs[i+1] - xs[i])/(rs[i+1] - rs[i])
			
		return xs[-1]


	def updateHyps(self,xsvs,i,spaces,**otherArgs):

		if self.domainName == "billiards":
			info = spaces.successRatesPerSkill[otherArgs["agentType"]]
		elif self.domainName in ["sequentialDarts"]:
			info = spaces.expectedRewardsPerXskill
		elif self.domainName == "baseball":
			info = otherArgs["infoPerRow"]["maxEVPerXskill"]
		else: # 1D or 2D
			info = spaces.convolutionsPerXskill


		# Update expected reward for each hypothesis
		for xi in range(len(self.xskills)):

			# Get the corresponding xskill level at the given index
			x = self.xskills[xi]
			
			# Update mean expected rewards
			if self.domainName == "billiards":
				tempInfo = info[x]["successfulTotal"]
			elif self.domainName in ["sequentialDarts","baseball"]:
				tempInfo = info[x]
			# For 1D & 2D
			else:
				tempInfo = info[x][otherArgs["s"]]["vs"]

			xsvs[xi] += (1/float(i+1))*(tempInfo-xsvs[xi])

		return xsvs


	def gameFinished(self):

		###################################################################
		# For OR-FullGame
		###################################################################

		self.rewardsPerGame.append(self.rewardsCurrentGame)

		# Updated observed reward as games was completed (and can use that data)
		self.meanRewardFullGame = sum(self.rewardsPerGame)/len(self.rewardsPerGame)
		
		###################################################################
		

		###################################################################
		# For OR-MidGame
		###################################################################

		# Update reward for mid game too
		self.meanRewardMidGame = self.meanRewardFullGame

		###################################################################


		###################################################################
		# Reset info
		###################################################################
		self.rewardsCurrentGame = 0
		##################################################################

		'''
		print("meanRewardFullGame: ", self.meanRewardFullGame)
		print("meanRewardMidGame: ", self.meanRewardMidGame)
		code.interact("Game finished...", local=dict(globals(), **locals()))
		'''


	def addObservation(self,spaces,reward,**otherArgs):

		if self.domainName in ["1d","2d","baseball"]:
			self.meanReward += (1)/(float(self.i+1)) * (reward-self.meanReward)

			self.xsvs = self.updateHyps(self.xsvs,self.i,spaces,**otherArgs)
			
			# Interpolate to set prediction
			closestX = self.interpolateRewardSkill("normal")			
			self.estimates.append(closestX)
			self.i += 1

			# print(f"OR- -> Current estimate: {self.estimates[-1]} (mean reward = {self.meanReward})\n")
			# code.interact("...", local=dict(globals(), **locals()))


		elif self.domainName in ["sequentialDarts","billiards"]:

			self.rewardsCurrentGame += reward

			###################################################################
			# For OR-MidGame
			###################################################################


			# Plus one to account for current ongoing game
			tempNumGames = len(self.rewardsPerGame) + 1

			self.meanRewardMidGame = (sum(self.rewardsPerGame) + self.rewardsCurrentGame) / tempNumGames

			self.xsvsMidGame = self.updateHyps(self.xsvsMidGame,tempNumGames,spaces,**otherArgs)
			
			# Interpolate to set prediction
			closestX = self.interpolateRewardSkill("mid")
			self.estimatesMidGame.append(closestX)

			# print(f"OR-MidGame -> Current estimate: {self.estimatesMidGame[-1]} (mean reward = {self.meanRewardMidGame})\n")
			# code.interact("...", local=dict(globals(), **locals()))

			###################################################################


			###################################################################
			# For OR-FullGame
			###################################################################

			self.xsvsFullGame = self.updateHyps(self.xsvsFullGame,len(self.rewardsPerGame),spaces,**otherArgs)
			
			# Interpolate to set prediction
			closestX = self.interpolateRewardSkill("full")
			self.estimatesFullGame.append(closestX)

			# print(f"OR-FullGame -> Current estimate: {self.estimatesFullGame[-1]} (mean reward = {self.meanRewardFullGame})\n")
			#code.interact("...", local=dict(globals(), **locals()))

			###################################################################


	def getResults(self):
		
		results = dict()

		if self.domainName in ["1d","2d","baseball"]:
			  results[self.name] = self.estimates
		

		elif self.domainName in ["sequentialDarts","billiards"]:
			results[self.name+"-estimatesFullGame"] = self.estimatesFullGame
			results[self.name+"-estimatesMidGame"] = self.estimatesMidGame

			results[self.name+"-rewardsPerGame"] = self.rewardsPerGame
			results[self.name+"-meanRewardFullGame"] = self.meanRewardFullGame
			results[self.name+"-meanRewardMidGame"] = self.meanRewardMidGame
			
		return results

