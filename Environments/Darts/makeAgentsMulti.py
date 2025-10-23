import numpy as np
import code
import Environments.Darts
from os import sep
from itertools import product, permutations
	
from importlib.machinery import SourceFileLoader

utilsModule = SourceFileLoader("utilsDarts",f"Processing{sep}utilsDartsMulti.py").load_module()


class AgentGenerator():
	
	__slots__ = ["domain","domainName","agentTypes","xSkillsGiven","pSkillsGiven",
				"numXskillsPerExp","numPskillsPerExp","numRhosPerExp","agentsModule",
				"startX_Agent","stopX_Agent","startP","stopP","startP_Bounded","stopP_Bounded",
				"startR","stopR", "minNumActions","maxNumActions","xSkills","xSkillsBeliefs",
				"paramValues","lambdaValues","rhos","n","k","paramValuesDelta","buckets",
				"dynamicXskills","agent","dynamic","dimensions"]

	def __init__(self,domainModule,agentTypes,**infoParams):

		self.domain = domainModule
		self.domainName = self.domain.get_domain_name()
		self.agentTypes = agentTypes
		
		self.xSkillsGiven = infoParams["xSkillsGiven"]
		self.pSkillsGiven = infoParams["pSkillsGiven"]
		self.numXskillsPerExp = infoParams["numXskillsPerExp"]
		self.numPskillsPerExp = infoParams["numPskillsPerExp"]
		self.numRhosPerExp = infoParams["numRhosPerExp"]
		self.dynamic = infoParams["dynamic"]
		self.agent = infoParams["agent"]
		self.dimensions = infoParams["dimensions"]
		
		self.initXskillParamsForAgents()
		self.initPskillParamsForAgents()
		self.initRskillParamsForAgents()

		self.setXskillParams()
		self.setPskillParams()
		self.setRskillParams()

		import Environments.Darts.RandomDarts.agents_multi as module

		self.agentsModule = module

		# Plus 1 cause 1st one always 0
		# if numPskillsPerExp = 4, buckets = [0,0.25,0.50,0.75,1.0]
		# Ignore first pos (0)
		self.buckets = np.linspace(0.0,1.0,self.numPskillsPerExp+1)[1:]


	def initXskillParamsForAgents(self):

		if self.domainName == "2d-multi":
			self.startX_Agent = [3.0,3.0]
			self.stopX_Agent = [150.5,150.5]

		elif self.domainName == "hockey-multi":
			self.startX_Agent = [0.004]
			self.stopX_Agent = [np.pi/4]

			
	def initPskillParamsForAgents(self):

		self.startP = 0.001
		self.stopP = 1.0 

		self.startP_Bounded = -3 #0.001 

		if self.domainName == "1d":
			self.stopP_Bounded = 2.0 # 100.0
		else:
			self.stopP_Bounded = 1.5 # ~32.0

		# Verify init values for params
		self.minNumActions = 1.0
		self.maxNumActions = 100.0


	def initRskillParamsForAgents(self):
		
		self.startR = -0.75
		self.stopR = 0.75


	def setXskillParams(self):

		if self.dynamic:

			# Assuming using range of 1st dimension only
			# self.dynamicXskills = list(permutations([self.startX_Agent[0],self.stopX_Agent[0]],r=2))

			# Convert xskill params from tuple to list
			# self.dynamicXskills = [list(each) for each in self.dynamicXskills]

			# self.dynamicXskills = [[[3,5],[20,24]],[[13,15],[40,44]]]


			sampleGoodXskillRange = [8,15]
			sampleBadXskillsRange = [130,145]

			set1 = []
			set2 = []

			for each in range(self.numXskillsPerExp):
				goodXskill = np.round(np.random.uniform(sampleGoodXskillRange[0],sampleGoodXskillRange[1],self.dimensions),4)
				set1.append(goodXskill)
				
				badXskill = np.round(np.random.uniform(sampleBadXskillsRange[0],sampleBadXskillsRange[1],self.dimensions),4)
				set2.append(badXskill)


			set1 = np.asarray(set1)
			set2 = np.asarray(set2)
			prevTemp = np.column_stack((set1,set2))

			mid = int(self.numXskillsPerExp/2)
			flipped = np.flip(prevTemp[mid:])
			temp = np.concatenate((prevTemp[:mid],flipped))

			self.dynamicXskills = []
			for each in temp:
				x1,x2 = each[:self.dimensions].tolist(),each[self.dimensions:].tolist()
				self.dynamicXskills.append([x1,x2])

			# code.interact("...", local=dict(globals(), **locals()))

		# For exps with given or rand params
		else:


			temp = []
					
			if self.xSkillsGiven:

				for i in range(len(self.startX_Agent)):
					temp.append(np.round(np.linspace(self.startX_Agent[i],self.stopX_Agent[i],num=self.numXskillsPerExp),4))	
				
				########################################
				# TO RUN EXPS WITH GIVEN SET OF AGENTS
				if self.agent == []:
					
					# To create just symmetric agents
					# self.xSkills = [[e,e] for e in temp[0]]

					# self.xSkills = [[3.0,3.0],[3.0,76.5],[3.0,150.5],[76.5,3.0],[76.5,76.5],[76.5,150.5],[150.5,3.0],[150.5,76.5],[150.5,150.5]] 
					# self.xSkills = [[3.0,3.0],[150.5,150.5],[3.0,150.5]] 
					# self.xSkills = [[3.0,3.0],[76.75,76.75],[150.5,150.5]]
					# self.xSkills = [[3.0,3.0]]
					# self.xSkills = [[10.0,10.0],[10.0,100.0],[100.0,10.0],[100.0,100.0]]

					# self.xSkills = [[0.004,0.004],[0.004,np.pi/4],[np.pi/4,np.pi/4]]
					
					self.xSkills = [[0.004,0.004]]
					# self.xSkills = [[0.19934954,0.19934954]]
					# self.xSkills = [[np.pi/8,np.pi/8]]
					# self.xSkills = [[0.59004862,0.59004862]]
					# self.xSkills = [[np.pi/4,np.pi/4]]
					# self.xSkills = list(map(list,product(*temp)))
					# print(self.xSkills)

				else:
					self.xSkills = [[float(self.agent[0]),float(self.agent[1])]]


			else:

				x1 = np.round(np.random.uniform(self.startX_Agent[i],self.stopX_Agent[i],self.numXskillsPerExp),4)
				x2 = np.round(np.random.uniform(self.startX_Agent[i],self.stopX_Agent[i],self.numXskillsPerExp),4)
				temp.append([x1,x2])

				self.xSkills = list(map(list,product(x1,x2)))
				# print(self.xSkills)


		# code.interact("...", local=dict(globals(), **locals()))

		########################################


	def setPskillParams(self):
		
		if self.pSkillsGiven:

			# For Flip Agent & Ticker Agent
			# For now, will use the same set of values for both agents 
			self.paramValues = np.linspace(self.startP,self.stopP,num=self.numPskillsPerExp)
			
			# For Bounded Agent
			lambdaValuesLog = np.linspace(self.startP_Bounded,self.stopP_Bounded,num=self.numPskillsPerExp)
			self.lambdaValues = np.power(10,lambdaValuesLog) # Exponentiate

			# Testing very small lambdas for bounded agent
			# lambdaValues = [0.00001,0.0001,0.001,0.01,0.1,0.5,1.0]
			# lambdaValues = np.random.uniform(0.0,0.001,100)

		else:
			# For Flip Agent & Ticker Agent
			self.paramValues = np.random.uniform(self.startP,self.stopP,self.numPskillsPerExp)

			# For Bounded Agent
			lambdaValuesLog = np.random.uniform(self.startP_Bounded,self.stopP_Bounded,self.numPskillsPerExp)		
			self.lambdaValues = np.power(10,lambdaValuesLog) # Exponentiate


		##################################################################
		# For Delta Agent
		##################################################################
		# Always given - for now - not at random
		
		if self.domain == "1d":
			# 0.01 is the delta used for Target so far
			self.paramValuesDelta = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

		elif self.domain == "2d":
			# 5.0 is the delta used for Target so far
			self.paramValuesDelta = [1.0,3.0,5.0,10.0]

		##################################################################


	def setRskillParams(self):
		if self.xSkillsGiven:
			# self.rhos = np.round(np.linspace(self.startR,self.stopR,num=self.numRhosPerExp),4)
			# self.rhos = [0.0,0.75,-0.75]
			self.rhos = [0.0]#,0.75]
			
		else:
			self.rhos = np.round(np.random.uniform(self.startR,self.stopR,self.numRhosPerExp),4)


	def getAgents(self,rng,env,info):

		print("\nCreating Agents...")
		agents = []
		 
		# For each one of the diffent types of agents
		for a in range(len(self.agentTypes)):

			# Target Agent
			if self.agentTypes[a] in [self.agentsModule.TargetAgent,self.agentsModule.TargetAgentAbruptChange,self.agentsModule.TargetAgentGradualChange]:
				keys = ["num_samples","noise_level","rho"]
				pskills = [25]
			# Flip Agent
			elif self.agentTypes[a] == self.agentsModule.FlipAgent:
				keys = ["prob_rational","noise_level","rho"]
				pskills = self.paramValues
			# Tricker Agent
			elif self.agentTypes[a] == self.agentsModule.TrickerAgent:
				keys = ["eps","noise_level","rho"]
				pskills = self.paramValues
			# Bounded Agent
			elif self.agentTypes[a] in [self.agentsModule.BoundedAgent,self.agentsModule.BoundedAgentAbruptChange,self.agentsModule.BoundedAgentGradualChange]:
				keys = ["lambda","noise_level","rho"]

				if self.pSkillsGiven:
					pskills = self.lambdaValues
				else:
					'''

					# Find respective lambda values
					pskills = []
					
					# Find lambda range for this %
					info = utilsModule.getLambdaRangeGivenPercent(env.pconfPerXskill,info,self.buckets)
					#info[bucket] = [lambda,prat]


					# Sample random lambda from the ranges
					for each in range(len(self.buckets)):

						# print(f"Each: {each} | Bucket: {self.buckets[each]}")
						
						# In case pconf info didn't found lambda for that bucket
						# Example: bad xskills, may not reach self.100% rat
						if each not in info:
							# last seen bucket
							try:
								l = np.random.uniform(info[each-1][0],self.stopP_Bounded)
								# print(f"not present on info -> ({info[each-1][0]},{self.stopP_Bounded}) -> l = {l} \n\t-> prat endpoint = {info[each-1][1]}")
							# otherwise, any from range
							except:
								l = np.random.uniform(self.startP_Bounded,self.stopP_Bounded)
								# print(f"not present on info -> ({self.startP_Bounded},{self.stopP_Bounded}) -> l = {l}")

						else:
							# Uniformly random sample from range
							# Doesnt include endpoint
							if each == 0: # first
								l = np.random.uniform(self.startP_Bounded,info[each][0])
								# print(f"first -> ({self.startP_Bounded},{info[each][0]}) -> l = {l} \n\t-> prat endpoint = {info[each][1]}")
							elif each == len(self.buckets)-1: # last
								l = np.random.uniform(info[each][0],self.stopP_Bounded)
								# print(f"last -> ({info[each][0]},{self.stopP_Bounded}) -> l = {l} \n\t-> prat endpoint = {info[each][1]}")
							else: # middle buckets
								# try:
								l = np.random.uniform(info[each-1][0],info[each][0])
								# print(f"middle -> ({info[each-1][0]},{info[each][0]}) -> l = {l} \n\t-> prat endpoints = {info[each-1][1]}|{info[each][1]}")
								
						pskills.append(l)


					# need to exponentiate pskills 
					# since range in log terms!

					pskills = np.power(10,pskills)

					'''
					pskills = pskills = self.lambdaValues

						
				# code.interact("...", local=dict(globals(), **locals()))	


			for p in range(len(pskills)):
				if env.domainName == "2d-multi" or env.domainName == "hockey-multi":
					params = {keys[0]:pskills[p],keys[1]:info[0],keys[2]:info[1]}
				else:
					params = {keys[0]:pskills[p],keys[1]:info}
					
				if self.agentTypes[a] in [self.agentsModule.TargetAgentAbruptChange,
										  self.agentsModule.TargetAgentGradualChange,
										  self.agentsModule.BoundedAgentAbruptChange,
										  self.agentsModule.BoundedAgentGradualChange]:
					agents.append(self.agentTypes[a](rng,env.numObservations,params,self.domain))
				else:
					agents.append(self.agentTypes[a](rng,params,self.domain))
		

		print("Done creating Agents.\n")
		# code.interact("...", local=dict(globals(), **locals()))	

		return agents

