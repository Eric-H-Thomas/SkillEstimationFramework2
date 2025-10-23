import numpy as np
import json,pickle
import os,imp,sys
import code,time

from scipy.signal import convolve2d

from pathlib import Path
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from itertools import product

# from memory_profiler import profile
from gc import collect

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split("setupSpaces.py")[0]

if "spacesUtils" not in sys.modules:
	module = SourceFileLoader("spacesUtils",f"{mainFolderName}spacesUtils.py").load_module()
	sys.modules["spacesUtils"] = module

# from unxpass.components.utils import load_model
# from unxpass.databases import SQLiteDatabase


class Spaces():

	__slots__ = ["domain","domainName","mode","delta","numObservations"]

	def __init__(self,numObservations,domain,mode,delta):

		self.domain = domain
		self.domainName = domain.get_domain_name()
		self.mode = mode
		self.delta = delta

		self.numObservations = numObservations


class SpacesRandomDarts(Spaces):

	__slots__ = ["convolutionsPerXskill","listedTargets","sizeActionSpace","focalActions","all_ts"]
	

	def __init__(self,numObservations,domain,mode,delta,numSamples=1000,expectedRFolder=None):

		super().__init__(numObservations,domain,mode,delta)
		
		self.convolutionsPerXskill = {}
		self.focalActions = {}


		if self.domainName == "1d":
			
			numPoints = int(6*self.domain.m/self.delta)
			grid = np.linspace(-3*self.domain.m,3*self.domain.m,numPoints)
			
			left = int(numPoints/3)
			right = int(2*left)

			self.listedTargets = grid[left:right]

			self.sizeActionSpace = 2*(self.domain.m)
		

		# Assuming set of listed targets is always the same 
		# (same dartboard every time, what might change is the scores based on the mode)
		elif "2d" in self.domainName:
			
			'''
			self.listedTargets = []

			ts = np.arange(-170.0, 171.0,self.delta)

			for ii in range(len(ts)):
				for jj in range(len(ts)):
					self.listedTargets.append([ts[ii],ts[jj]])
			'''

			XS = np.arange(-170.0,171.0,self.delta)
			YS = np.arange(-170.0,171.0,self.delta)

			XXS,YYS = np.meshgrid(XS,YS,indexing="ij")
			tempXYS = np.vstack([XXS.ravel(),YYS.ravel()])

			listedTargets = np.dstack(tempXYS)[0]
			self.listedTargets = np.array(listedTargets)


			# Size of action space = area of rectangle = length * width
			# self.sizeActionSpace = ((len(np.arange(-170.0,171.0,self.delta))) * 1.0)**2

			# Size of action space = area of circle = pi*R**2
			self.sizeActionSpace = np.pi*(170.0**2)


			self.all_ts = {"all_ts_x": listedTargets[:,0], "all_ts_y": listedTargets[:,1]}

		# code.interact("(): ", local=dict(globals(), **locals())) 


	def initInfoForExps(self):
		pass


	def reset(self):

		# For all domains, to redo convs at start of new exp with new seed
		self.convolutionsPerXskill.clear() 
		self.focalActions.clear() 


	# @profile
	def addSpace(self,rng,givenInfo,S,returnZn=False):

		# print("STATE: ", S)

		if type(givenInfo) == dict:
			info = givenInfo["key"]
		else:
			info = givenInfo

		# Verify if info for the convolution of the given state, xskill and delta exists
		exists = self.verifyIfExists(info,S)

		if exists:
			# print(f"Convolution present for x = {info}")
			return
		# Otherwise, do convolution
		else:
			# print(f"Doing convolution for x = {info}")
			
			if info not in self.convolutionsPerXskill:
				self.convolutionsPerXskill[info] = {}
			
			# print("addSpace() - STATES: ",S)
			self.doConvolution(rng,givenInfo,S,returnZn)


	def getKey(self,info,r):
		return "|".join(map(str,info))+f"|{r}"


	def updateSpaceParticles(self,rng,each,state,otherArgs,fromEstimator=False):
		
		if self.domainName == "2d-multi":

			info = {"mean":[0.0]*(len(each)-2)}

			temp, tempR = each[:-2],each[-2]

			info["key"] = self.getKey(temp,tempR)

			info["covMatrix"] = self.domain.getCovMatrix(temp,tempR)

				
			# if not self.verifyIfExists(info["key"],state):
			self.addSpace(rng,info,state,returnZn=False)
			# else:
				# print(f"Convolution present for x = {info['key']}")
				# pass

			del info



	# Singular (one at a time (params & state))
	def deleteSpaceParticles(self,each,state):

		if self.domainName == "2d-multi":

			temp, tempR = each[:-1],each[-1]

			k = self.getKey(temp,tempR)

			# Assuming convolution space present on dict
			# Still need try/except in case multiple particles with the same key
			try:
				# self.convolutionsPerXskill[k][str(state)].clear()
				
				# REMOVING ALL INFO 
				# Since storing 1 state at a time anyways
				self.convolutionsPerXskill[k].clear()
				del self.convolutionsPerXskill[k]
				# print(f"Removing convolution for x = {k}")	
			except:
				# code.interact("deleteSpaceParticles()...", local=dict(globals(), **locals()))	
				pass


	# Singular (one at a time (params & state))
	def deleteSpace(self,each,state):

		if self.domainName == "2d-multi":
			temp, tempR = each[:-1],each[-1]
			k = self.getKey(temp,tempR)
		else:
			k = each

		# Assuming convolution space present on dict
		self.convolutionsPerXskill[k][str(state)].clear()				
		del self.convolutionsPerXskill[k][str(state)]
		# print(f"Removing convolution for x = {k}")	

		# code.interact("deleteSpaceParticles()...", local=dict(globals(), **locals()))	


	def updateSpace(self,rng,givenXskillsStart,states,fromEstimator=False):
		
		if self.domainName == "2d-multi":
			rhos = givenXskillsStart[1]
			givenXskills = givenXskillsStart[0]

			info = {"mean":[0.0]*len(givenXskills[0])}


			for temp in givenXskills:

				for tempR in rhos:
					
					info["key"] = self.getKey(temp,tempR)

					info["covMatrix"] = self.domain.getCovMatrix(temp,tempR)

					try:
						states[0][0]
					except Exception as e:
						# print(e)
						states = [states]
						# code.interact("...", local=dict(globals(), **locals()))
						

					# print("updateSpace() - STATES: ",states,type(states))

					for s in states:
						if not self.verifyIfExists(temp+[tempR],s):
							# print(f"Doing convolution for x = {info['key']}")
							self.addSpace(rng,info,s,returnZn=False)
						else:
							# print(f"Convolution present for x = {info['key']}")
							pass
		else:

			try:
				states[0][0]
			except:
				states = [states]

			for x in givenXskillsStart:
				for s in states:
					self.addSpace(rng,x,s,returnZn=False)


	# Singular (one at a time (params & state))
	def setFocalActions(self,givenInfo,state):
		
		if str(state) not in self.focalActions:
			self.focalActions[str(state)] = []

		if self.domainName == "2d-multi":
			key = self.getKey(givenInfo[:-1],givenInfo[-1])
			self.focalActions[str(state)].append(self.convolutionsPerXskill[key][str(state)]["ts"])

		else:
			self.focalActions[str(state)].append(self.convolutionsPerXskill[givenInfo][str(state)]["ts"])

		# code.interact("Setting focal actions...", local=dict(globals(), **locals()))

	
	def get(self,rng,S,info,toSend):

		# Verify if info for the convolution of the given state, xskill and delta exists
		if self.domainName == "2d-multi":
			exists = self.verifyIfExists(toSend["params"][0]+[toSend["params"][1]],S)
		else:
			exists = self.verifyIfExists(info,S)


		# Doesn't exist -> do convolution
		if not exists:

			# print(f"Doing convolution for x = {info}")

			params = toSend["params"]
			toSend["mean"] = [0.0]*len(params[0])
			toSend["key"] = info
			toSend["covMatrix"] = self.domain.getCovMatrix(params[0],params[1])

			if self.domainName == "2d-multi":
				toCheck = toSend["key"]
			else:
				toCheck = info

			if toCheck not in self.convolutionsPerXskill:
				self.convolutionsPerXskill[toCheck] = {}

			self.doConvolution(rng,toSend,S)


		return self.getConvInfo(info,S)


	# Singular (one at a time (params & state))
	def getSpace(self,rng,params,S,returnZn=False):

		if self.domainName == "2d-multi":	
			# [x1,x2,rho]
			info = self.getKey(params[0],params[1])
			toSend = {"params": params}
			# code.interact("getSpace()...", local=dict(globals(), **locals()))	
			return self.get(rng,S,info,toSend)

		else:
			info = params
			toSend = params
			return self.get(rng,S,info,toSend)

		
	def verifyIfExists(self,X,S):

		if type(X) == list:
			info = self.getKey(X[:-1],X[-1])
		else:
			info = X

		if info in self.convolutionsPerXskill:			
			if str(S) in self.convolutionsPerXskill[info]:
				return True
			# State not present on available convolutions
			else:
				return False
		# No convolutions present for given xskill
		else:
			return False


	# @profile
	def doConvolution(self,rng,info,state,returnZn=False):

		# print("doConvolution() - STATE: ",state)
		# print(info)


		tt = time.perf_counter()

		if self.domainName == "1d":
			all_vs,ts,vs = self.domain.get_expected_values_and_optimal_action(rng,state,info,self.delta)
			
		elif self.domainName == "2d":
			if returnZn:
				all_vs,ts,vs,onBoardEVs,Zn = self.domain.get_expected_values_and_optimal_action(rng,state,info,self.delta,returnZn)
			else:
				all_vs,ts,vs,onBoardEVs = self.domain.get_expected_values_and_optimal_action(rng,state,info,self.delta,returnZn)
		elif self.domainName == "2d-multi":
			if returnZn:
				all_vs,ts,vs,onBoardEVs,Zn = self.domain.get_expected_values_and_optimal_action(rng,state,info["mean"],info["covMatrix"],self.delta,returnZn)
			else:
				all_vs,ts,vs,onBoardEVs = self.domain.get_expected_values_and_optimal_action(rng,state,info["mean"],info["covMatrix"],self.delta,returnZn)

		# print(f"Total time for get_expected_values_and_optimal_action: {time.perf_counter()-tt:.4f}")
		# print()

		tempDict = {}
		# tempDict["all_ts"] = all_ts
		tempDict["all_vs"] = all_vs
		tempDict["ts"] = ts
		tempDict["vs"] = vs


		'''
		if self.domainName in ["1d","2d-multi"]:
			listed_vs = all_vs
		
		else:
			listed_vs = []
			for ii in range(len(all_ts["all_ts_x"])):
				for jj in range(len(all_ts["all_ts_y"])):
					listed_vs.append(all_vs[ii][jj])
		# tempDict["listed_vs"] = np.array(listed_vs)
		
		'''


		if self.domainName == "1d":
			tempDict["mean_vs"] = np.mean(all_vs)
		# For 2D
		else:
			tempDict["mean_vs"] = np.mean(onBoardEVs)


		'''
		fig = plt.figure()
		ax = plt.gca()
		cmap = plt.get_cmap("viridis")
		norm = plt.Normalize(min(listed_vs),max(listed_vs))
    
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm,ax=ax)
		cbar.ax.set_title("EVs")

		if self.domainName == "1d":
			plt.scatter(self.listedTargets,[0]*len(self.listedTargets),c=cmap(norm(listed_vs)))
		else:
			plt.scatter(self.listedTargets[:,0],self.listedTargets[:,1],c=cmap(norm(listed_vs)))
		# plt.show()
		# plt.savefig(f"allTargets-xskill{X}.png")
		# plt.clf()
		# plt.close()
		'''
		
		# if X > 150:
		# if X > 2:
			# code.interact("doConvolution()...", local=dict(globals(), **locals()))	


		if returnZn:
			tempDict["Zn"] = Zn
		


		if self.domainName == "2d-multi":
			# tempDict["cov"] = info["covMatrix"]

			# Save info
			self.convolutionsPerXskill[info["key"]]["cov"] = info["covMatrix"]
			self.convolutionsPerXskill[info["key"]][str(state)] = tempDict

			# print(f"ref count convolutionsPerXskill: {sys.getrefcount(self.convolutionsPerXskill[info['key']][str(state)])}")
			# print(f"size of convolutionsPerXskill: {sys.getsizeof(self.convolutionsPerXskill[info['key']][str(state)])}")

		else:

			val = info**2
			cvs = np.zeros((2,2))
			np.fill_diagonal(cvs,val)

			# tempDict["cov"] = cvs


			# Save info
			self.convolutionsPerXskill[info]["cov"] = cvs
			self.convolutionsPerXskill[info][str(state)] = tempDict


		# code.interact("space...", local=dict(globals(), **locals()))


	# Method will be called after calling verifyIfExists()
	# Info will always exist (and keys as well) - (thus, won't cause error of info not existing)
	def getConvInfo(self,X,S):
		return self.convolutionsPerXskill[X][str(S)]


class SpacesSequentialDarts(Spaces):

	__slots__ = ["mode","numSamples","expectedRFolder","expectedRewardsPerXskill",
				"valueIterFolder","possibleTargets","spacesPerXskill",
				"estimatorXskills","allPIsForXskillsPerState",
				"allCovs","allCovsGivenXskillOptimalTargets","allCovsGivenXskillDomainTargets",
				"sizeActionSpace","totalTargetActions","possibleTargets"]

	def __init__(self,numObservations,domain,mode,delta,numSamples=1000,expectedRFolder=None,valueIterFolder=None):

		super().__init__(numObservations,domain,mode,delta)

		self.mode = mode

		self.numSamples = numSamples
		self.expectedRFolder = expectedRFolder
		self.expectedRewardsPerXskill = {}

		self.valueIterFolder = valueIterFolder

		# self.possibleTargets = np.array(domain.getActions())

		allTargets = []

		# defaultX = np.arange(-340.0,341.0,self.resolution)
		# defaultY = np.arange(-340.0,341.0,self.resolution)

		defaultX = np.arange(-170.0,171.0,self.delta)
		defaultY = np.arange(-170.0,171.0,self.delta)

		for xi in defaultX:
			for yi in defaultY:
				allTargets.append((xi,yi))
		self.possibleTargets = np.array(allTargets)


		self.spacesPerXskill = {}
		self.estimatorXskills = []
		

	def initInfoForExps(self):

		print("Initializing info for experiments...")

		# To only initialize info for xskills from estimators
		# Method called from estimators.py

		self.allPIsForXskillsPerState = {}
		
		self.allCovs = np.zeros((len(self.estimatorXskills),2,2))

		for s in range(2,self.domain.getPlayerStartScore()+1):
			self.allPIsForXskillsPerState[s] = np.zeros((len(self.estimatorXskills),2))

			for xi in range(len(self.estimatorXskills)):
				x = self.estimatorXskills[xi]
				self.allPIsForXskillsPerState[s][xi] = self.spacesPerXskill[x].PI[s]


		# Size of action space = area of rectangle = length * width
		# self.sizeActionSpace = len(np.arange(-170.0,171.0,self.delta))**2
		
		# Size of action space = area of circle = pi*R**2
		self.sizeActionSpace = np.pi*(170.0**2)

		# Assumes set of target action for all xskills across all the states
		# are of the same size
		self.totalTargetActions = len(self.allPIsForXskillsPerState[2])

		#'''
		self.allCovsGivenXskillOptimalTargets = np.zeros((len(self.estimatorXskills),self.totalTargetActions,2,2))
		self.allCovsGivenXskillDomainTargets = np.zeros((len(self.estimatorXskills),len(self.possibleTargets),2,2))

		for xi in range(len(self.estimatorXskills)):
			x = self.estimatorXskills[xi]

			val = x**2
			cvs = np.zeros((2,2))
			np.fill_diagonal(cvs,val)

			self.allCovs[xi] = cvs

			for z in range(len(self.allCovsGivenXskillOptimalTargets[xi])):
				self.allCovsGivenXskillOptimalTargets[xi][z] = cvs

			for z in range(len(self.allCovsGivenXskillDomainTargets[xi])):
				self.allCovsGivenXskillDomainTargets[xi][z] = cvs

		# '''

	
	def reset(self):

		tempXs = list(self.expectedRewardsPerXskill.keys())

		# Clear info for xskills 
		# Keep only xskill hyps for the estimators
		for tempX in tempXs:
			if tempX not in self.estimatorXskills:
				del self.expectedRewardsPerXskill[tempX]
				del self.spacesPerXskill[tempX] 


	def spacePresent(self,x):

		if x in self.spacesPerXskill:			
			return True
		else:
			return False


	def addSpace(self,x,fromEstimator=False):

		if fromEstimator and x not in self.estimatorXskills:
			self.estimatorXskills.append(x)


		present = self.spacePresent(x)
		
		if present:
			return
		else:
			newSpace = sys.modules["spacesUtils"].StateActionValueSpace(self.mode,self.numObservations,self.domain,self.delta,self.possibleTargets,x,self.numSamples,self.expectedRFolder,self.valueIterFolder)

			self.spacesPerXskill[x] = newSpace
			self.expectedRewardsPerXskill[x] = newSpace.expectedReward


	def updateSpace(self,givenXskills,fromEstimator=False):
		for x in givenXskills:
			self.addSpace(x,fromEstimator)


class SpacesBilliards(Spaces):

	__slots__ = ["numSamples","successRatesFolder","agentTypes",
				"spacesPerXskill","agentToXskillID","agentIdToXskill",
				"xskilltoAgentId","agentIdToType","successRatesPerSkill",
				"sizeActionSpace"]

	def __init__(self,numObservations,domain,delta,numSamples,successRatesFolder,agentTypes):
		
		super().__init__(numObservations,domain,delta)

		self.numSamples = numSamples
		self.successRatesFolder = successRatesFolder
		self.agentTypes = agentTypes

		self.sizeActionSpace = 360.0

		self.spacesPerXskill = {}


		self.agentToXskillID = {5:14, 7:16, 9:17, 10:18, 11:19, 19:20, 20:21, 21:22, 23:23, \
							24:24, 25:26, 26:25, 27:28, 28:29, 6:14, 12:16, 13:17, 15:18, 16:19}

		self.agentIdToXskill = {23:0.025, 26:0.05, 9:0.0625, 19:0.1, 5:0.125, 10:0.1875,\
							20:0.2, 7:0.25, 21:0.3, 11:0.375, 25:0.4, 24:0.5, 6:0.125, 12:0.25,\
							13:0.0625, 15:0.1875,16:0.375, 27:0.625, 28:0.75}

		self.xskilltoAgentId = {
			"CC":{0.025:23, 0.05:26, 0.0625:9,
				0.1:19, 0.125:5, 0.1875:10, 0.2:20,
				0.25:7, 0.3:21, 0.375:11, 0.4:25, 
				0.5:24, 0.625:27, 0.75:28},
			"MG":{0.025:29, 0.05:30, 0.0625:13,
				0.125:6, 0.1875:15,
				0.25:12, 0.375:16, 
				0.5:31, 0.75:32}}

		self.agentIdToType = {23:"CC", 26:"CC", 9:"CC",
							19:"CC", 5:"CC", 10:"CC", 20:"CC",
							7:"CC", 21:"CC", 11:"CC", 25:"CC", 
							24:"CC", 27:"CC", 28:"CC",
							29:"MG", 30:"MG", 13:"MG",
							6:"MG", 15:"MG",
							12:"MG", 16:"MG", 
							31:"MG", 32:"MG"}


		# MachineGun-0.1
		# id = 34

		# MachineGun-0.2
		# id = 35

		# MachineGun-0.3
		# id = 36

		# MachineGun-0.4
		# id = 37

		# MachineGun-0.625
		# id = 38



		self.successRatesPerSkill = {}

		for eachType in self.agentTypes:
			
			self.successRatesPerSkill[eachType] = {}

			self.spacesPerXskill[eachType] = {}


	def initInfoForExps(self):
		pass


	def reset(self):
		pass


	def addSpace(self,eachType,agent,x,fromEstimator=False):

		if x in self.spacesPerXskill[eachType]:
			print(f"Space for {eachType} - x = {x} is present already.")
			return 
		else:
			newSpace = StateActionSpaceBilliards(self.numObservations,self.domain,self.delta,eachType,agent,x,self.numSamples,self.successRatesFolder)
			
			self.spacesPerXskill[eachType][x] = newSpace

			if x not in self.successRatesPerSkill[eachType]:
				self.successRatesPerSkill[eachType][x] = newSpace.successRatesInfo

			if fromEstimator and x not in self.estimatorXskills:
				self.estimatorXskills.append(x)


	def updateSpace(self,givenXskills,eachType,fromEstimator=False):
		for x in givenXskills:
			agent = self.xskilltoAgentId[eachType][x]
			self.addSpace(eachType,agent,x,fromEstimator)


class SpacesBaseball(Spaces):

	__slots__ = ["numSamples","expectedRFolder","allData","dataLoaded",
				"minPlateX","maxPlateX","minPlateZ","maxPlateZ",
				"targetsPlateXFeet","targetsPlateZFeet",
				"targetsPlateXInches","targetsPlateZInches",
				"modelTargetsPlateX","modelTargetsPlateZ",
				"model","possibleTargetsFeet","possibleTargetsForModel",
				"expectedRewardsPerXskill","xswingFeats", "spacesPerXskill",
				"estimatorXskills","allCovs","batterIndices","sizeActionSpace",
				"defaultFocalActions","infoPerRow","focalActionMiddle","pdfsPerXskill",
				"evsPerXskill","indexes"]

	# @profile
	def __init__(self,args,numObservations,domain,delta,numSamples,expectedRFolder,learningRate,epochs,parallel=False):		

		global torch, nn, pd, StandardScaler

		# Importing here as only baseball exps need these modules
		import torch
		import torch.nn as nn 
		import pandas as pd
		from sklearn.preprocessing import StandardScaler

		from multiprocessing import Manager

		super().__init__(numObservations,domain,"",delta)

		self.numSamples = numSamples

		self.expectedRFolder = expectedRFolder

		self.estimatorXskills = []


		#####################################################
		# Init Info
		#####################################################

		# In feets
		self.minPlateX = -2.13
		self.maxPlateX = 2.13

		# In feets
		self.minPlateZ = -2.50
		self.maxPlateZ = 6.60

		self.targetsPlateXFeet = np.arange(self.minPlateX,self.maxPlateX,delta)
		self.targetsPlateZFeet = np.arange(self.minPlateZ,self.maxPlateZ,delta)

		# To include end point
		# self.targetsPlateXFeet = np.append(self.targetsPlateXFeet,self.maxPlateX)
		# self.targetsPlateZFeet = np.append(self.targetsPlateZFeet,self.maxPlateZ)


		# Size of action space = area of rectangle = length * width
		# a = abs(self.minPlateX-self.maxPlateX)
		# b = abs(self.minPlateZ-self.maxPlateZ)
		# self.sizeActionSpace = a*b
		self.sizeActionSpace = len(self.targetsPlateXFeet)*len(self.targetsPlateZFeet)

		# Store inches version of targets
		# self.targetsPlateXInches = self.targetsPlateXFeet*12
		# self.targetsPlateZInches = self.targetsPlateZFeet*12

		self.defaultFocalActions = np.array([[-0.71,1.546],[-0.71,3.412],[0.71,1.546],
									[0.71,3.412],[-0.71,2.479],[0.71,2.479],[0.0,1.546],
									[0.0,3.412],[0.0,2.479],[-0.355,2.0125],[0.355,2.0125],
									[-0.355,2.9455],[0.355,2.9455],[0.0,1.2395],[0.0,3.7185],
									[-1.0165,2.479],[1.0165,2.479],[-1.0165,1.2395],
									[-1.0165,3.7185],[1.0165,1.2395],[1.0165,3.7185]])
		
		self.focalActionMiddle =  [0.0,2.479]

		#####################################################
		

		#####################################################
		# Get Data
		#####################################################

		self.getAllData()

		#####################################################


		#####################################################
		# PER EXPERIMENT
		#####################################################

		# self.expectedRewardsPerXskill = {}

		#####################################################


		#####################################################
		# Get Model
		#####################################################

		# ASSUMES TRAINED MODEL READY TO USE

		'''
		self.model,train_losses,test_losses,train_accs,test_accs,modelFolder = \
					sys.modules["model"].getModel(learningRate,epochs,self.batterIndices,\
												train=None,test=None,withinFramework=True)
		'''

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		sys.modules["modelTake2"].batter_indices = self.batterIndices

		self.model = sys.modules["modelTake2"].RNN(hidden_size = 32, output_size = 9).to(device)		
		self.model.load_state_dict(torch.load(f'Environments{os.sep}Baseball{os.sep}final_OP',map_location=device))

		#####################################################

		
		if parallel:

			manager = Manager()

			self.pdfsPerXskill = manager.dict()
			self.evsPerXskill = manager.dict()
	
		else:
			self.pdfsPerXskill = {}
			self.evsPerXskill = {}


		# To do mapping just once
		self.indexes = {}

		for i in list(range(len(self.modelTargetsPlateX))):
			for j in list(range(len(self.modelTargetsPlateZ))):
				tempIndex = np.where((self.possibleTargetsForModel[:,0] == self.modelTargetsPlateX[i]) & (self.possibleTargetsForModel[:,1] == self.modelTargetsPlateZ[j]))[0][0]
				self.indexes[f"{self.modelTargetsPlateX[i]}|{self.modelTargetsPlateZ[j]}"] = tempIndex


		# code.interact("spaces init()...", local=dict(globals(), **locals()))


	# @profile
	def initInfoForExps(self):
	
		self.allCovs = {}


		# For baseball-multi domain
		if "multi" in self.domainName:

			# Computing for symmetric set only since for normal JTM only
			# The ones for JTM-Particles will be managed within addObservation()
			info = list(product(self.estimatorXskills,[0.0]))

			for xi in range(len(info)):
				x = info[xi]
				key = self.getKey(x[0],x[1])
				self.allCovs[key] = self.domain.getCovMatrix(x[0],x[1])
			
		else:

			for xi in range(len(self.estimatorXskills)):
				x = self.estimatorXskills[xi]

				val = x**2
				cvs = np.zeros((2,2))
				np.fill_diagonal(cvs,val)

				self.allCovs[x] = cvs


	# @profile
	def getAllData(self):

		fileName = f"ProcessedData-From-GivenFiles.pkl"	

		processedDataFolder = f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"

		# Verify if file containing proceesed data already exist
		if Path(f"{processedDataFolder}{fileName}").is_file():
			print(f"\nFile with processed data for the given csv files is present. \nLoading processed data (and other info)...")

			with open(f"{processedDataFolder}{fileName}", "rb") as aFile:
				loadedInfo = pickle.load(aFile)

			self.allData,self.batterIndices = loadedInfo[0][0],loadedInfo[0][1]
			self.modelTargetsPlateX,self.modelTargetsPlateZ = loadedInfo[0][2],loadedInfo[0][3]
			self.possibleTargetsFeet,self.possibleTargetsForModel = loadedInfo[0][4],loadedInfo[0][5]

			print("Processed data (and other info) was loaded successfully.")
			del loadedInfo
		

		# Otherwise, compute
		else:
			print("\nProcessing data...")

			csvFiles = [f"{processedDataFolder}raw22.csv",f"{processedDataFolder}raw21.csv",f"{processedDataFolder}raw18_19_20.csv"]

			self.allData,self.batterIndices,standardizer = sys.modules["dataTake2"].manageData(csvFiles)


			# Standardize possible actions for model
			copyStandardizer = StandardScaler()
			indexes = [3,4]
			i = 0

			temp = {"plate_x":self.targetsPlateXFeet,"plate_z":self.targetsPlateZFeet}
			
			for k,v in temp.items():

				tempDF = pd.DataFrame(v,columns=[k])

				copyStandardizer.mean_ = standardizer.mean_[indexes[i]]
				copyStandardizer.scale_ = standardizer.scale_[indexes[i]]
				copyStandardizer.var_ = standardizer.var_[indexes[i]]

				tempDF[[k]] = copyStandardizer.transform(tempDF[[k]].values)
				temp[k] = tempDF[k]

				i += 1

			self.modelTargetsPlateX = temp["plate_x"].values
			self.modelTargetsPlateZ = temp["plate_z"].values


			# Store dense matrix for actions
			self.possibleTargetsFeet = []
			self.possibleTargetsForModel = []

			for i in range(len(self.targetsPlateXFeet)):
				for j in range(len(self.targetsPlateZFeet)):
					self.possibleTargetsFeet.append([self.targetsPlateXFeet[i],self.targetsPlateZFeet[j]])
					self.possibleTargetsForModel.append([self.modelTargetsPlateX[i],self.modelTargetsPlateZ[j]])

			self.possibleTargetsFeet = np.array(self.possibleTargetsFeet)
			self.possibleTargetsForModel = np.array(self.possibleTargetsForModel)


			# Save processed data and other info
			with open(f"{processedDataFolder}{fileName}", "wb") as outfile:
				toSave = [self.allData,self.batterIndices,
						  self.modelTargetsPlateX,self.modelTargetsPlateZ,
						  self.possibleTargetsFeet,self.possibleTargetsForModel]
				pickle.dump([toSave],outfile)

			print("Data was processed and saved successfully.")

		self.dataLoaded = True

		# Call garbage collector
		collect()


	def getAllDataPrev(self,args):

		startDate = f"{args.startYear}-{args.startMonth}-{args.startDay}"
		endDate = f"{args.endYear}-{args.endMonth}-{args.endDay}"
		fileName = f"ProcessedData-From-{startDate}-To-{endDate}.pkl"	

		processedDataFolder = f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"

		# Verify if file containing proceesed data already exist
		if Path(f"{processedDataFolder}{fileName}").is_file():
			print(f"\nFile with processed data for the date range ({startDate} -> {endDate}) is present. \nLoading processed data (and other info)...")

			with open(f"{processedDataFolder}{fileName}", "rb") as aFile:
				loadedInfo = pickle.load(aFile)

			self.allData,self.batterIndices = loadedInfo[0][0],loadedInfo[0][1]
			self.modelTargetsPlateX,self.modelTargetsPlateZ = loadedInfo[0][2],loadedInfo[0][3]
			self.possibleTargetsFeet,self.possibleTargetsForModel = loadedInfo[0][4],loadedInfo[0][5]

			print("Processed data (and other info) was loaded successfully.")

		# Otherwise, compute
		else:
			print("\nProcessing data...")

			rawData = sys.modules["data"].getData(args.startYear,args.startMonth,args.startDay,args.endYear,args.endMonth,args.endDay,withinFramework=True)

			self.allData,self.batterIndices,standardizer = sys.modules["data"].manageData(rawData)


			# Standardize possible actions for model
			copyStandardizer = StandardScaler()
			indexes = [3,4]
			i = 0

			temp = {"plate_x":self.targetsPlateXFeet,"plate_z":self.targetsPlateZFeet}
			
			for k,v in temp.items():

				tempDF = pd.DataFrame(v,columns=[k])

				copyStandardizer.mean_ = standardizer.mean_[indexes[i]]
				copyStandardizer.scale_ = standardizer.scale_[indexes[i]]
				copyStandardizer.var_ = standardizer.var_[indexes[i]]

				tempDF[[k]] = copyStandardizer.transform(tempDF[[k]].values)
				temp[k] = tempDF[k]

				i += 1

			self.modelTargetsPlateX = temp["plate_x"].values
			self.modelTargetsPlateZ = temp["plate_z"].values


			# Store dense matrix for actions
			self.possibleTargetsFeet = []
			self.possibleTargetsForModel = []

			for i in range(len(self.targetsPlateXFeet)):
				for j in range(len(self.targetsPlateZFeet)):
					self.possibleTargetsFeet.append([self.targetsPlateXFeet[i],self.targetsPlateZFeet[j]])
					self.possibleTargetsForModel.append([self.modelTargetsPlateX[i],self.modelTargetsPlateZ[j]])

			self.possibleTargetsFeet = np.array(self.possibleTargetsFeet)
			self.possibleTargetsForModel = np.array(self.possibleTargetsForModel)


			# Save processed data and other info
			with open(f"{processedDataFolder}{fileName}", "wb") as outfile:
				toSave = [self.allData,self.batterIndices,
						  self.modelTargetsPlateX,self.modelTargetsPlateZ,
						  self.possibleTargetsFeet,self.possibleTargetsForModel]
				pickle.dump([toSave],outfile)

			print("Data was processed and saved successfully.")

		self.dataLoaded = True


	def getDataByMostRecent(self,agentData,tempArgs):

		maxRows = tempArgs[0]

		# Select only X recent rows/pitches
		if len(agentData) > maxRows:
			agentData = agentData.iloc[:maxRows,:]

		return agentData


	def getDataByChunks(self,agentData,tempArgs):

		# To divide data in chunks
		
		# totalRows = len(agentData)
		# totalChunks = totalRows / numObservations

		numObservations = tempArgs[0]
		chunkNum = tempArgs[1]

		end = numObservations*chunkNum
		start = end - numObservations
		

		agentData = agentData.iloc[start:end,:]

		return agentData


	def getDataByPitchNum(self,agentData,tempArgs):

		b1 = tempArgs[0]
		b2 = tempArgs[1]

		return agentData


	# @profile
	def getAgentData(self,dataBy,pitcherID,pitchType,tempArgs):		
		
		agentDataTemp = self.allData.query(f"pitcher == {pitcherID} and pitch_type == '{pitchType}'")

		# Sort data by game date
		# To have most recent pitches first
		agentData = agentDataTemp.sort_values(by=["game_date"],ascending=False)



		if dataBy == "recent":
			func = self.getDataByMostRecent
			# received - tempArgs = [maxRows]
		elif dataBy == "chunks":
			func = self.getDataByChunks
			# received - tempArgs = [numObservations,chunkNum]
		elif dataBy == "pitchNum":
			func = self.getDataByPitchNum
			# received - tempArgs = [b1,b2]

		agentData = func(agentData,tempArgs)

		
		# code.interact("...", local=dict(globals(), **locals()))

		
		self.dataLoaded = False
		
		del self.allData
		del agentDataTemp
		collect()

		return agentData


	def reset(self):
		self.expectedRewardsPerXskill = {}
		self.infoPerRow = {}


	def getKey(self,info,r):
		return "|".join(map(str,info))+f"|{r}"


	def setStrikeZoneBoard(self,allTempData,minUtility):
		
		# Populate Strike Zone Board - Can create once since independent of xskill
		Zs = np.zeros((len(self.modelTargetsPlateX),len(self.modelTargetsPlateZ)))

		# for i in list(range(len(self.modelTargetsPlateX))):
		# 	for j in list(range(len(self.modelTargetsPlateZ))):
		# 		tempIndex = np.where((allTempData.plate_x == self.modelTargetsPlateX[i]) & (allTempData.plate_z == self.modelTargetsPlateZ[j]))[0][0]
		# 		Zs2[i][j] = np.copy(allTempData.iloc[tempIndex]["utility"])

		for i in list(range(len(self.modelTargetsPlateX))):
			for j in list(range(len(self.modelTargetsPlateZ))):
				tempIndex = self.indexes[f"{self.modelTargetsPlateX[i]}|{self.modelTargetsPlateZ[j]}"]
				Zs[i][j] = np.copy(allTempData.iloc[tempIndex]["utility"])

		# code.interact("...", local=dict(globals(), **locals()))
		return Zs


	def updateSpace(self,rng,givenXskills,fromEstimator=False):

		if "multi" in self.domainName:
			givenXskills = givenXskills[0]

		for x in givenXskills:

			# Adding symmetric set only since for normal JTM only
			# The ones for JTM-Particles will be managed within addObservation()
			if fromEstimator and x not in self.estimatorXskills:

				if "multi" in self.domainName:
					 if x[0] == x[1]:
					 	self.estimatorXskills.append(x)
				else:
					self.estimatorXskills.append(x)


	def updateSpaceParticles(self,rng,each,state,info,wid=None):

		if "multi" in self.domainName:

			allTempData = info["allTempData"]
			minUtility = info["minUtility"]
			
			# Assuming method will get called only with multi domain
			covMatrix = self.domain.getCovMatrix(each[:-2],each[-2])
			key = self.getKey(each[:-2],each[-2])


			if key not in self.pdfsPerXskill:
				# print(f"Computing pdfs for {key}... (wid: {wid})")
				self.pdfsPerXskill[key] = self.domain.getNormalDistribution(rng,covMatrix,self.delta,self.targetsPlateXFeet,self.targetsPlateZFeet)
			else:
				# print(f"Pdfs info is present for {key}. (wid: {wid})")
				pass


			if key not in self.evsPerXskill:
				# print(f"Computing EVs for {key}... (wid: {wid})")
				# t1 = time.perf_counter()
				# Zs = self.setStrikeZoneBoard(allTempData,minUtility)
				# print(f"Total time for setting the board: {time.perf_counter()-t1:.4f}")
				
				Zs = info["Zs"]

				# t1 = time.perf_counter()
				self.evsPerXskill[key] = convolve2d(Zs,self.pdfsPerXskill[key],mode="same",fillvalue=minUtility)
				# print(f"Total time for convolve2d: {time.perf_counter()-t1:.4f}")
			else:
				# print(f"EVs info present for {key}... (wid: {wid})")
				pass

			# code.interact("...", local=dict(globals(), **locals()))


	def deleteSpaceParticles(self,each,state):

		key = self.getKey(each[:-2],each[-2])

		try:
			# print(f"Removing convolution for x = {key}")

			self.pdfsPerXskill[key].clear()
			self.evsPerXskill[key].clear()

			del self.pdfsPerXskill[key]
			del self.evsPerXskill[key]
		
		except:
			pass


	# Use function when not computing expected reward in an online manner
	# (load available info from file & compute only needed ones)
	'''
	def getExpInfo(self,xskills,pitcherID,pitchType,agentData,resultsFolder):

		#####################################################
		# Manage Expected Rewards
		#####################################################
		
		expectedRewardsInfo = {}

		toLoad = f"{self.expectedRFolder}.json"
		
		# Load file that contains historical data, if present
		if os.path.exists(toLoad):
			print(f"\nLoading expected rewards for existing xskills...")

			# Check if the file has content
			if os.path.getsize(toLoad) != 0:
				# Load info from file
				with open(toLoad) as inFile:
					expectedRewardsInfo = json.load(inFile)
			else:
				expectedRewardsInfo = {}


		xskillsToGet = []
		present = False

		# Determine if info is present
		if str(pitcherID) in expectedRewardsInfo:
			if pitchType in expectedRewardsInfo[str(pitcherID)]:

				present = True
				
				for x in xskills:
					# Load info into expecedRewards as xskill present
					if str(x) in expectedRewardsInfo[str(pitcherID)][pitchType]:
						print(f"Loading info for xskill {x}")
						self.expectedRewardsPerXskill[x] = expectedRewardsInfo[str(pitcherID)][pitchType][str(x)]
					else:
						xskillsToGet.append(x)
			else:
				expectedRewardsInfo[pitcherID][pitchType] = {}
		else:
			expectedRewardsInfo[pitcherID] = {}
			expectedRewardsInfo[pitcherID][pitchType] = {}


		if not present:
			xskillsToGet = xskills


		# Compute expected rewards for missing xskills and save to file
		if len(xskillsToGet) != 0:
			print(f"\nWill compute expected rewards for xskills {xskillsToGet}...")
	
		#####################################################
		

		#####################################################
		# Get info for experiment (expected rewards + info per row)
		#####################################################
		
		print(f"\nComputing info per row needed for experiment...")
		timesPerRow = self.computeExpInfo(xskills,xskillsToGet,pitcherID,pitchType,agentData,resultsFolder)

		#####################################################


		#####################################################
		# Update expected rewards file if needed
		#####################################################

		if len(xskillsToGet) != 0:
			for x in xskillsToGet:
				expectedRewardsInfo[pitcherID][pitchType][x] = self.expectedRewardsPerXskill[x] 

			with open(toLoad,'w') as outfile:
				json.dump(expectedRewardsInfo,outfile)

		#####################################################

		# code.interact("...", local=dict(globals(), **locals()))
		return timesPerRow
	'''

	'''
	def computeExpInfo(self,allXskills,xskillsToGet,pitcherID,pitchType,agentData,resultsFolder):
		
		agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
		infoFolder = f"Experiments{os.sep}{resultsFolder}{os.sep}info"
		plotFolder1 = f"Experiments{os.sep}{resultsFolder}{os.sep}plots{os.sep}StrikeZoneBoards{os.sep}Utilities{os.sep}{agentFolder}{os.sep}"
		plotFolder2 = f"Experiments{os.sep}{resultsFolder}{os.sep}plots{os.sep}StrikeZoneBoards{os.sep}EVsPerXskill{os.sep}{agentFolder}{os.sep}"
		pickleFolder = f"Experiments{os.sep}{resultsFolder}{os.sep}plots{os.sep}StrikeZoneBoards{os.sep}PickleFiles{os.sep}{agentFolder}{os.sep}"

		for folder in [infoFolder,plotFolder1,plotFolder2,pickleFolder]:
			if not os.path.exists(folder):
				os.mkdir(folder)

		# temp = {}
		# temp["plate_x_feet"] = self.targetsPlateXFeet
		# temp["plate_z_feet"] = self.targetsPlateZFeet
		# temp["plate_x_inches"] = self.targetsPlateXInches
		# temp["plate_z_inches"] = self.targetsPlateZInches
		# temp["plate_x_feet_standardized"] = self.modelTargetsPlateX
		# temp["plate_z_feet_standardized"] = self.modelTargetsPlateZ

		# with open(f"{pickleFolder}info.pkl", "wb") as handle:
		#	pickle.dump(temp,handle)


		timesPerRow = []

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		
		pdfsPerXskill = {}
		
		for x in allXskills:
			pdfsPerXskill[x] = self.domain.getSymmetricNormalDistribution(x,self.delta,self.targetsPlateXFeet,self.targetsPlateZFeet)
		
		# code.interact("...", local=dict(globals(), **locals()))


		it = 0
		total = len(agentData)

		for index,row in agentData.iterrows():

			it += 1
			print(f"Looking at row {it}/{total} | index: {index}...")

			startTimeRow = time.time()

			self.infoPerRow[index] = {}

			# st1 = time.time()
			allTempData = pd.DataFrame([row]*(len(self.possibleTargetsForModel)))
			# print(f"Time copy row: {time.time()-st1}")
			# code.interact("...", local=dict(globals(), **locals()))


			# Update position of each copy of the row to be that of a given possible action
			# st1 = time.time()
			allTempData["plate_x"] = np.array(self.possibleTargetsForModel)[:,0]
			allTempData["plate_z"] = np.array(self.possibleTargetsForModel)[:,1]
			# print(f"Time update targets: {time.time()-st1}")


			# Include original 'row' (df with actual pitch info) to get the probabilities 
			# for the different outcomes as well as the utility - for the actual pitch
			# st1 = time.time()
			allTempData.loc[len(allTempData.index)+1] = row
			# print(f"Time get last row: {time.time()-st1}")


			# Run model
			# st1 = time.time()
			results = nn.functional.softmax(self.model(torch.tensor(allTempData[sys.modules["data"].xswingFeats].values.astype(float),\
					 dtype = torch.float32).to(device)), dim = 1).cpu().detach().numpy()
			# print(f"Time query model: {time.time()-st1}")

			# Save info
			# st1 = time.time()
			for i in range(9):
				allTempData[f'o{i}'] = results[:,i]
			# print(f"Time copy probs: {time.time()-st1}")
			
			# Get utilities
			# st1 = time.time()
			withUtilities = sys.modules["utilsBaseball"].getUtility(allTempData)
			# print(f"Time get utilities: {time.time()-st1}")


			# Get updated info for actual pitch (actual pitch + probs + utility)
			# st1 = time.time()
			row = withUtilities.iloc[-1].copy()
			

			# Save info to reuse within exp
			# utility = observed reward
			self.infoPerRow[index]["observedReward"] = row["utility"]

			# Remove actual pitch from data
			withUtilities = withUtilities.iloc[:-1,:]
 
			#print(f"Time get new info row and remove: {time.time()-st1}")
			

			# st1 = time.time()
			
			minUtility = np.min(withUtilities["utility"].values)

			# Prepare color map
			# cmap = plt.get_cmap("viridis")
			# norm = plt.Normalize(minUtility, max(withUtilities["utility"].values))
			# sm = ScalarMappable(norm=norm,cmap=cmap)
			# sm.set_array([])

			###############################################################
			# Strike Zone Board
			###############################################################

			fileName = f"pitcherID{pitcherID}-pitchType{pitchType}-index{index}"
			
			# # for l in ["feet","inches"]:
			# l = "feet"

			# # PLOT - RAW BOARD
			# fig,ax = plt.subplots()

			# cbar = fig.colorbar(sm)
			# cbar.ax.get_yaxis().labelpad = 15
			# cbar.ax.set_ylabel("Utilities",rotation = 270)

			# if l == "inches":
			# 	ax.scatter(self.possibleTargetsFeet[:,0]*12,self.possibleTargetsFeet[:,1]*12,c = cmap(norm(withUtilities["utility"].values)))
			# else:
			# 	ax.scatter(self.possibleTargetsFeet[:,0],self.possibleTargetsFeet[:,1],c = cmap(norm(withUtilities["utility"].values)))
			
			# ax.set_xlabel(f"plate_x - {l}")
			# ax.set_ylabel(f"plate_z - {l}")

			# ax.set_title(f"pitcherID: {pitcherID} | pitchType: {pitchType} | index: {index} - {l}")

			# # Plot actual executed action & EV
			# ax.scatter(row[f"plate_x_{l}"],row[f"plate_z_{l}"],color=cmap(norm(observedReward)),marker="*",edgecolors="black")
			# plt.savefig(f"{plotFolder1}{fileName}-{l}.jpg")
			# plt.close()
			# plt.clf()

			# Save info to pickle file
			withUtilities[["plate_x","plate_z","utility"]].to_pickle(f"{pickleFolder}{fileName}.pkl")  

			# print(f"Time plot raw utility board and save to pickle file: {time.time()-st1}")
			# code.interact("...", local=dict(globals(), **locals()))
			
			###############################################################
			

			# Populate Dartboard - Can create once since independent of xskill
			# Zs = np.zeros((len(self.targetsPlateX), len(self.targetsPlateZ)))

			# for i in range(len(self.targetsPlateX)):
			# 	for j in range(len(self.targetsPlateZ)):
					
			# 		tempIndex = np.where((withUtilities.plate_x == self.targetsPlateX[i]) & (withUtilities.plate_z == self.targetsPlateZ[j]))[0][0]
			# 		Zs[i][j] = withUtilities.iloc[tempIndex]["utility"]

			# Can obtain utilies this way since possible actions
			# are assigned to dataframe in order 
			# (Thus, no need to search for corresponding index)
			# st1 = time.time()
			Zs = np.reshape(withUtilities.utility.values,(len(self.targetsPlateXFeet),len(self.targetsPlateZFeet)))
			# print(f"Time reshape utilities df: {time.time()-st1}")
	

			# Fixed set of actions based on dartboard + best actions for xskill hyps
			self.infoPerRow[index]["focalActions"] = self.defaultFocalActions
			# self.infoPerRow[index]["evsFocalActions"] = []
			self.infoPerRow[index]["evsPerXskill"] = {}
			self.infoPerRow[index]["maxEVPerXskill"] = {}


			for x in allXskills:

				#st1 = time.time()
			
				# Convolve to produce the EV and aiming spot
				EVs = convolve2d(Zs,pdfsPerXskill[x],mode="same",fillvalue=minUtility)

				maxEV = np.max(EVs)	
				mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
				action = [self.targetsPlateXFeet[mx],self.targetsPlateZFeet[mz]]
				
				#print(f"Time conv {x}: {time.time()-st1}")
				# code.interact("...", local=dict(globals(), **locals()))

				self.infoPerRow[index]["focalActions"].append(action)
				self.infoPerRow[index]["evsPerXskill"][x] = EVs	
				self.infoPerRow[index]["maxEVPerXskill"][x] = maxEV	

				# if x in xskillsToGet:
				# 	avgUtility[x] += maxEV


				#st1 = time.time()

				# PLOT - BOARD EV's given xskill
				# fig,ax = plt.subplots()

				# cbar = fig.colorbar(sm)
				# cbar.ax.get_yaxis().labelpad = 15
				# cbar.ax.set_ylabel("EVs",rotation = 270)

				# ax.scatter(self.possibleTargetsFeet[:,0],self.possibleTargetsFeet[:,1],c = cmap(norm(EVs.flatten())))
				# ax.set_xlabel("targetsPlateX")
				# ax.set_ylabel("targetsPlateZ")

				# ax.scatter(action[0],action[1],color=cmap(norm(maxEV)),marker="*",edgecolors="black")

				# ax.set_title(f"pitcherID: {pitcherID} | pitchType: {pitchType} | index: {index} | xskill: {x}")

				# # Plot best action & EV
				# ax.scatter(action[0],action[1],color=cmap(norm(maxEV)),marker="*",edgecolors="black")
				# plt.savefig(f"{plotFolder2}{fileName}-xskill{x}.jpg")
				# plt.close()
				# plt.clf()
				#print(f"Time save plot conv {x}: {time.time()-st1}")


			stopTimeRow = time.time()
			timesPerRow.append(stopTimeRow-startTimeRow)
			print(timesPerRow[-1])


		# Save one more time to ensure all seen data is stored


		# code.interact("end...", local=dict(globals(), **locals()))
		return timesPerRow
	'''


class SpacesHockey(Spaces):

	__slots__ = [
			"minY","maxY","minZ","maxZ",
			"targetsY","targetsZ","possibleTargets",
			"expectedRewardsPerXskill",
			"dirs","elevations",
			"estimatorXskills","allCovs","sizeActionSpace",
			"defaultFocalActions","focalActionMiddle","pdfsPerXskill",
			"evsPerXskill","grid","mean"]

	# @profile
	def __init__(self,args,numObservations,domain,delta):		


		super().__init__(numObservations,domain,"",delta)


		self.estimatorXskills = []


		#####################################################
		# Init Info
		#####################################################

		self.minY = -3.0
		self.maxY = 3.0

		self.minZ = 0.0
		self.maxZ = 4.0

		# self.targetsY = np.arange(self.minY,self.maxY,delta)
		# self.targetsZ = np.arange(self.minZ,self.maxZ,delta)


		# delta is now per dimension

		self.targetsY = np.linspace(self.minY,self.maxY,60)
		self.targetsZ = np.linspace(self.minZ,self.maxZ,40) 


		# To include end point
		# self.targetsY = np.append(self.targetsY,self.maxY)
		# self.targetsZ = np.append(self.targetsZ,self.maxZ)


		# Size of action space = area of rectangle = length * width
		# a = abs(self.minY-self.maxY)
		# b = abs(self.minZ-self.maxZ)
		# self.sizeActionSpace = a*b
		self.sizeActionSpace = len(self.targetsY)*len(self.targetsZ)


		self.defaultFocalActions = []
		
		# self.focalActionMiddle =  [0.0,2.479]


		# Store dense matrix for actions
		self.possibleTargets = []

		# for i in range(len(self.targetsY)):
			# for j in range(len(self.targetsZ)):
				# self.possibleTargets.append([self.targetsY[i],self.targetsZ[j]])

		
		for j in range(len(self.targetsZ)):
			for i in range(len(self.targetsY)):
				self.possibleTargets.append([self.targetsY[i],self.targetsZ[j]])

		self.possibleTargets = np.array(self.possibleTargets)
		


		#####################################################
		
		
		self.pdfsPerXskill = {}
		self.evsPerXskill = {}


		#####################################################
		# PER EXPERIMENT
		#####################################################

		# self.expectedRewardsPerXskill = {}

		#####################################################


		# code.interact("spaces init()...", local=dict(globals(), **locals()))


	# @profile
	def initInfoForExps(self):
	
		self.allCovs = {}


		# For baseball-multi domain
		if "multi" in self.domainName:

			# Computing for symmetric set only since for normal JTM only
			# The ones for JTM-Particles will be managed within addObservation()
			info = list(product(self.estimatorXskills,[0.0]))

			for xi in range(len(info)):
				x = info[xi]
				key = self.getKey(x[0],x[1])
				self.allCovs[key] = self.domain.getCovMatrix(x[0],x[1])
			
		else:

			for xi in range(len(self.estimatorXskills)):
				x = self.estimatorXskills[xi]

				val = x**2
				cvs = np.zeros((2,2))
				np.fill_diagonal(cvs,val)

				self.allCovs[x] = cvs



	# @profile
	def getAgentData(self,rf,player,typeShot,maxRows):		

		rfup = str(rf.replace(f"Experiment",""))

		folder = f"Experiments{os.sep}{rfup}{os.sep}Data{os.sep}AngularHeatmaps{os.sep}"
		fileName = f"angular_heatmap_data_player_{player}_type_shot_{typeShot}.pkl"
		
		try:
			with open(folder+fileName,"rb") as infile:
				agentData = pickle.load(infile)

			# Select only X recent rows (slice dict)
			if len(agentData) > maxRows:
				agentData = {k:agentData[k] for i,k in enumerate(agentData) if i < maxRows}


		except Exception as e:
			print(e)
			print("File not present.")
			agentData = []

		# code.interact("...", local=dict(globals(), **locals()))
		
		return agentData


	def reset(self):
		self.expectedRewardsPerXskill = {}


	def getKey(self,info,r):
		return "|".join(map(str,info))+f"|{r}"


	def updateSpace(self,rng,givenXskills,fromEstimator=False):

		if "multi" in self.domainName:
			givenXskills = givenXskills[0]

		for x in givenXskills:

			# Adding symmetric set only since for normal JTM only
			# The ones for JTM-Particles will be managed within addObservation()
			if fromEstimator and x not in self.estimatorXskills:

				if "multi" in self.domainName:
					 if x[0] == x[1]:
					 	self.estimatorXskills.append(x)
				else:
					self.estimatorXskills.append(x)


	def updateSpaceParticles(self,rng,each,state,info,wid=None):

		if "multi" in self.domainName:

			# Assuming method will get called only with multi domain
			covMatrix = self.domain.getCovMatrix(each[:-2],each[-2])
			key = self.getKey(each[:-2],each[-2])


			if key not in self.pdfsPerXskill:
				# print(f"Computing pdfs for {key}... (wid: {wid})")
				self.pdfsPerXskill[key] = self.domain.getNormalDistribution(rng,covMatrix,self.delta,self.mean,self.grid)
			else:
				# print(f"Pdfs info is present for {key}. (wid: {wid})")
				pass


			if key not in self.evsPerXskill:
				# print(f"Computing EVs for {key}... (wid: {wid})")
				# t1 = time.perf_counter()
				
				Zs = info["Zs"]
				# print(Zs)
				minUtility = np.min(Zs)

				# t1 = time.perf_counter()
				self.evsPerXskill[key] = convolve2d(Zs,self.pdfsPerXskill[key],mode="same",fillvalue=0.0)
				# print(f"Total time for convolve2d: {time.perf_counter()-t1:.4f}")
			else:
				# print(f"EVs info present for {key}... (wid: {wid})")
				pass


			# if "0.7854" in key:
				# code.interact("updateSpaceParticles()...", local=dict(globals(), **locals()))


	def deleteSpaceParticles(self,each,state):

		key = self.getKey(each[:-2],each[-2])

		try:
			# print(f"Removing convolution for x = {key}")

			self.pdfsPerXskill[key].clear()
			self.evsPerXskill[key].clear()

			del self.pdfsPerXskill[key]
			del self.evsPerXskill[key]
		
		except:
			pass


class SpacesSoccer(Spaces):

	__slots__ = ["minPitchX","maxPitchX","minPitchY","maxPitchY",
				"targetsY","targetsY","sizeActionSpace","defaultFocalActions",
				"focalActionMiddle","dataFolder","model","allCovs","estimatorXskills","db"]


	# @profile
	def __init__(self,args,numObservations,domain,delta):		

		super().__init__(numObservations,domain,"",delta)


		self.estimatorXskills = []


		#####################################################
		# Init Info
		#####################################################

		# (0,0) = top left corner

		# Pitch Coordinates = (x,y) coordinates
		self.minPitchX = 0
		self.maxPitchX = 120

		self.minPitchY = 0
		self.maxPitchY = 80

		self.targetsY = np.arange(self.minPitchX,self.maxPitchX,delta)
		self.targetsY = np.arange(self.minPitchY,self.maxPitchY,delta)

		# To include end point
		self.targetsY = np.append(self.targetsY,self.maxPitchX)
		self.targetsY = np.append(self.targetsY,self.maxPitchY)


		# Size of action space = area of rectangle = length * width
		# a = abs(self.minPitchX-self.maxPitchX)
		# b = abs(self.minPitchY-self.maxPitchY)
		# self.sizeActionSpace = a*b
		self.sizeActionSpace = len(self.targetsY)*len(self.targetsY)

		# TODO: PENDING TO DEFINE
		self.defaultFocalActions = np.array([])
		
		self.focalActionMiddle =  [60,40]

		#####################################################

		# FOR DATA
		self.dataFolder = f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}PerPlayer"		

		
		STORES_FP = Path("../un-xPass/stores")

		self.db = SQLiteDatabase(STORES_FP / "database.sqlite")


		#####################################################
		# Get Model
		#####################################################

		# ASSUMES TRAINED MODEL READY TO USE

		# LOAD VALUE MODEL
		# Test1-Filtered
		# modelFolder = "runs:/284aa3e220d84e44bb76bd7a8c9e6a46/component"

		# Test2-All
		modelFolder = "runs:/5680315123dc43d99cc37570405c554c/component"


		self.model = load_model(f"{modelFolder}")
	
		#####################################################

		# code.interact("spaces init()...", local=dict(globals(), **locals()))


	# @profile
	def initInfoForExps(self):
		
		self.allCovs = np.zeros((len(self.estimatorXskills),2,2))

		for xi in range(len(self.estimatorXskills)):
			x = self.estimatorXskills[xi]

			val = x**2
			cvs = np.zeros((2,2))
			np.fill_diagonal(cvs,val)

			self.allCovs[xi] = cvs


	def processIndex(self,k):
		temp = k.replace("(","").replace(")","").split(",")
		return (int(temp[0]),int(temp[1]))

	# @profile
	def getAgentData(self,playerID,maxRows):

		info = []

		try:
			for each in ["Actions","Features","Labels"]:
				with open(f"{self.dataFolder}-{each}{os.sep}data-Player{playerID}.pkl","rb") as infile:
					temp = pd.read_pickle(infile)

					# Select only X rows
					if len(temp) > maxRows:
						temp = temp.iloc[:maxRows,:]

					info.append(temp)

			with open(f"{self.dataFolder}-GameStates{os.sep}data-Player{playerID}.pkl","rb") as infile:
				temp = pd.read_pickle(infile)

				# Select only X rows
				if len(temp) > maxRows:
					temp = temp[:maxRows]

				info.append(temp)

		except:
			print(f"File for player {playerID} not found. Unable to perform exp.")
			# code.interact("...", local=dict(globals(), **locals()))
		
		return info


	def reset(self):
		pass

	def updateSpace(self,givenXskills,fromEstimator=False):

		for x in givenXskills:
			if fromEstimator and x not in self.estimatorXskills:
				self.estimatorXskills.append(x)





