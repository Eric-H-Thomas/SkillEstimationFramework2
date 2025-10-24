import numpy as np
import scipy
import code
from scipy.stats import multivariate_normal

from Estimators.utils import *

# from memory_profiler import profile

class BayesianMethod():

	__slots__ = ["estimates","xskills","numXskills","domainName",
				"typeTargets","methodType","beta","names","pxs","allProbs"]


	def __init__(self,xskills,beta,domainName,typeTargets=""):

		self.estimates = dict()
	
		self.xskills = xskills
		self.numXskills = len(xskills)

		self.domainName = domainName

		self.methodType = "BM"

		self.typeTargets = typeTargets

		self.beta = beta

		if self.typeTargets != "":
			self.names = [self.methodType+'-MAP-'+str(self.numXskills)+"-"+typeTargets+"-Beta-"+str(self.beta),
					  self.methodType+'-EES-'+str(self.numXskills)+"-"+typeTargets+"-Beta-"+str(self.beta)]
		else:
			self.names = [self.methodType+'-MAP-'+str(self.numXskills)+"-Beta-"+str(self.beta),
					  self.methodType+'-EES-'+str(self.numXskills)+"-Beta-"+str(self.beta)]


		for n in self.names:
			self.estimates[n] = []

		self.pxs = np.array([1.0/self.numXskills]*self.numXskills)

		# To append the initial probs - uniform distribution for all
		self.allProbs = [self.pxs.tolist()]


	def getEstimatorName(self):
		return self.names


	def midReset(self):
		for n in self.names:
			self.estimates[n] = []

		self.allProbs = []
		

	def reset(self):
		for n in self.names:
			self.estimates[n] = []
		self.pxs = np.array([1.0/self.numXskills]*self.numXskills)

		self.allProbs = []

		# to append the initial probs - uniform distribution for all
		self.allProbs.append(self.pxs.tolist())


	# @profile
	def addObservation(self,rng,domain,spaces,state,action,**otherArgs):

		if "multi" in self.domainName:
			# print(state)
			for each in self.xskills:
				spaces.updateSpace(rng,[[[each,each]],[0.0]],state)
				spaces.setFocalActions([each,each,0.0],state)
			
			spaces.focalActions[str(state)] = np.array(spaces.focalActions[str(state)])

			# code.interact("BM...", local=dict(globals(), **locals()))


		# print("\nprevious probs: ", self.pxs)
		# print(f"BETA: {self.beta}")

		for xi in range(self.numXskills):

			x = self.xskills[xi]

			# Get the corresponding xskill level hypothesis at the given index
			if self.domainName in ["2d-multi","baseball-multi"]:
				key = spaces.getKey([x,x],r=0.0)
			else:
				key = x


			if self.domainName in ["1d","2d","2d-multi"]:

				if self.domainName == "1d":
					
					diffs = []
					for efc in spaces.focalActions[str(otherArgs["s"])]:
						diff_fn = getattr(domain, "calculate_wrapped_action_difference", None)
						if diff_fn is None:
							diff_fn = getattr(domain, "calculate_action_difference")
						diffs.append(diff_fn(action, efc))

					size = len(diffs)
					pdfs = scipy.stats.norm.pdf(diffs,loc=0,scale=x)
					update = (self.beta*(np.sum(pdfs)/size)) + ((1-self.beta)/20.0)  

				else: # 2D
					size = len(spaces.focalActions[str(otherArgs["s"])])
					tempCovs = [np.array(spaces.convolutionsPerXskill[key]["cov"])]*size
					
					pdfs = computePDF(x=action,means=spaces.focalActions[str(otherArgs["s"])],covs=tempCovs) # covariance = std^2
					update = (self.beta*(np.sum(pdfs)/size)) + ((1-self.beta)/spaces.sizeActionSpace)	

			elif self.domainName == "sequentialDarts":

				action = np.array(action)

				if self.typeTargets == "OptimalTargets":
					pdfs = computePDF(x=action,means=spaces.allPIsForXskillsPerState[otherArgs["currentScore"]],covs=spaces.allCovsGivenXskillOptimalTargets[xi]) # covariance = std^2
					size = len(spaces.allPIsForXskillsPerState[otherArgs["currentScore"]])
				# DomainTargets
				else:
					pdfs = computePDF(x=action,means=spaces.possibleTargets,covs=spaces.allCovsGivenXskillDomainTargets[xi]) # covariance = std^2
					size = len(spaces.possibleTargets)
				# Update prob for current xskill
				update = (self.beta*(np.sum(pdfs)/size)) + ((1-self.beta)/spaces.sizeActionSpace)


			elif self.domainName == "billiards":
				# diff = estimatedNoise (target(closest to noisy) - executed)
				diff = otherArgs["diff"]

				update = (self.beta*scipy.stats.norm.pdf(x=diff,loc=0,scale=x))+((1-self.beta)*(1/360.0))


			elif self.domainName == "baseball":
				
				action = np.array(action)

				focalActions = otherArgs["infoPerRow"]["focalActions"]
				cov = spaces.allCovs[key]
				pdfs = computePDF(x=action,means=focalActions,covs=[cov]*len(focalActions)) # covariance = std^2
				
				# Update prob for current xskill
				update = (self.beta*(np.sum(pdfs)/len(focalActions)))+((1-self.beta)/spaces.sizeActionSpace)
				
				# code.interact("BM - pdfs...", local=dict(globals(), **locals()))
				

			# Update probs
			self.pxs[xi] *= update


		# Normalize probs
		self.pxs /= np.sum(self.pxs)
		# code.interact("...", local=dict(globals(), **locals()))

		self.allProbs.append(self.pxs.tolist())

		
		# Get estimate. Uses MAP estimate
		pmi = np.argmax(self.pxs)

		# self.names[0] = 'BM-MAP-'+str(num_xskills)
		self.estimates[self.names[0]].append(self.xskills[pmi])

		# Get EES Estimate
		ees = np.dot(self.pxs,self.xskills)
		
		# self.names[1] = 'BM-EES-'+str(num_xskills)
		self.estimates[self.names[1]].append(ees)

		# For testing
		'''
		print("x | target | prob")
		for xi in range(self.numXskills):
			print(f"{self.xskills[xi]} -> {spaces.spacesPerXskill[self.xskills[xi]].PI[otherArgs["currentScore"]]} -> {self.pxs[xi]}")
		'''

		# print(f"probs sum: {np.sum(self.pxs)}")
		# print(f"BM - Beta: {self.beta} -> Current estimate: X: MAP: {self.xskills[pmi]} | EES: {ees}\n")
		# code.interact("...", local=dict(globals(), **locals()))


		if "multi" in self.domainName:
			spaces.focalActions.clear() 


	def getResults(self):
		results = dict()
		
		for n in self.names:
			results[n] = self.estimates[n]

		results[f"{self.methodType}-Beta-{self.beta}-allProbs"] = self.allProbs
		
		return results
