import numpy as np
import code
import os
import json

class ObservedReward():

	def __init__(self,xskills,domainName):

		self.xskills = xskills
		self.numXskills = len(xskills)

		self.domainName = domainName
		self.name = 'OR-' + str(self.numXskills)

		self.estimates = []
		self.xsvs = np.array([0.0]*len(self.xskills))
		self.meanReward = 0.0
		self.i = 0
		

	def getEstimatorName(self):
		return self.name


	def reset(self):
		self.estimates = []
		self.xsvs = np.array([0.0]*len(self.xskills))
		self.meanReward = 0.0
		self.i = 0


	def interpolateRewardSkill(self):

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

	def updateHyps(self,xsvs,i,otherArgs):

		info = otherArgs["maxEVPerXskill"]

		# Update expected reward for each hypothesis
		for xi in range(len(self.xskills)):

			# Get the corresponding xskill level at the given index
			x = self.xskills[xi]
			
			# Update mean expected rewards
			tempInfo = info[x]

			xsvs[xi] += (1/float(i+1))*(tempInfo-xsvs[xi])

		return xsvs


	def addObservation(self,reward,otherArgs):

		self.meanReward += (1)/(float(self.i+1)) * (reward-self.meanReward)

		self.xsvs = self.updateHyps(self.xsvs,self.i,otherArgs)
		
		# Interpolate to set prediction
		closestX = self.interpolateRewardSkill()			
		self.estimates.append(closestX)
		self.i += 1

		# print(f"OR- -> Current estimate: {self.estimates[-1]} (mean reward = {self.meanReward})\n")
		# code.interact("...", local=dict(globals(), **locals()))


	def getResults(self):

		results = dict()
		results[self.name] = self.estimates
		return results

