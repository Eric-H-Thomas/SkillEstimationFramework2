import numpy as np
import code
import Environments.Darts
from os import sep
	
from importlib.machinery import SourceFileLoader

utilsModule = SourceFileLoader("utilsDarts",f"Processing{sep}utilsDarts.py").load_module()


class AgentGenerator():
	
	__slots__ = ["domain","domainName","agentTypes","xSkillsGiven","pSkillsGiven",
				"numXskillsPerExp","numPskillsPerExp","agentsModule",
				"startX_Agent","stopX_Agent","startP","stopP","startP_Bounded","stopP_Bounded",
				"minNumActions","maxNumActions","xSkills","xSkillsBeliefs",
				"paramValues","lambdaValues","n","k","paramValuesDelta","buckets"]

	def __init__(self,domainModule,agentTypes,**infoParams):

		self.domain = domainModule
		self.domainName = self.domain.get_domain_name()
		self.agentTypes = agentTypes
		
		self.xSkillsGiven = infoParams["xSkillsGiven"]
		self.pSkillsGiven = infoParams["pSkillsGiven"]
		self.numXskillsPerExp = infoParams["numXskillsPerExp"]
		self.numPskillsPerExp = infoParams["numPskillsPerExp"]
		
		self.initXskillParamsForAgents()
		self.initPskillParamsForAgents()
		self.setXskillParams()
		self.setPskillParams()

		if self.domainName == "1d" or self.domainName == "2d":
			import Environments.Darts.RandomDarts.agents as module
			self.agentsModule = module

		# Domain == sequentialDarts
		else:
			import Environments.Darts.SequentialDarts.agents as module
			self.agentsModule = module

		# Plus 1 cause 1st one always 0
		# if numPskillsPerExp = 4, buckets = [0,0.25,0.50,0.75,1.0]
		# Ignore first pos (0)
		self.buckets = np.linspace(0.0,1.0,self.numPskillsPerExp+1)[1:]


	def initXskillParamsForAgents(self):

		if self.domainName == "1d":
			self.startX_Agent = 0.5 
			self.stopX_Agent = 14.5 #4.5

		elif self.domainName == "2d" or self.domainName == "sequentialDarts":
			self.startX_Agent = 3.0
			self.stopX_Agent = 150.5
			
	def initPskillParamsForAgents(self):

		self.startP = 0.001
		self.stopP = 1.0 

		self.startP_Bounded = -3 # 0.001

		if self.domainName == "1d":
			self.stopP_Bounded = 2.0 # 100.0
		else:
			self.stopP_Bounded = 1.5 # ~32.0

		# Verify init values for params
		self.minNumActions = 1.0
		self.maxNumActions = 100.0

	def setXskillParams(self):
		
		if self.xSkillsGiven:
			self.xSkills = np.round(np.linspace(self.startX_Agent,self.stopX_Agent,num=self.numXskillsPerExp),4)
			
			# For Target Agent with Beliefs (same set for now)
			self.xSkillsBeliefs = self.xSkills #np.round(np.linspace(self.startX_Agent,self.stopX_Agent,num=self.numXskillsPerExp),4)

		else:
			self.xSkills = np.round(np.random.uniform(self.startX_Agent,self.stopX_Agent,self.numXskillsPerExp),4)

			# For Target Agent with Beliefs (same set for now)
			self.xSkillsBeliefs = self.xSkills #np.round(np.random.uniform(self.startX_Agent,self.stopX_Agent,self.numXskillsPerExp),4)

		
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

			# For Random Agent
			self.n = np.random.randint(self.minNumActions,self.maxNumActions+1,self.numPskillsPerExp) # Number of actions
			self.k = np.random.randint(5,50,self.numPskillsPerExp) # Number of samples

		else:
			# For Flip Agent & Ticker Agent
			self.paramValues = np.random.uniform(self.startP,self.stopP,self.numPskillsPerExp)

			# For Random Agent
			self.n = np.random.randint(1.0,self.maxNumActions+1,self.numPskillsPerExp) # Number of actions
			self.k = np.random.randint(1.0,101.0,self.numPskillsPerExp) # Number of samples

			# For Bounded Agent
			# lambdaValuesLog = np.random.uniform(self.startP_Bounded,self.stopP_Bounded,self.numPskillsPerExp)		
			# self.lambdaValues = np.power(10,lambdaValuesLog) # Exponentiate

			self.lambdaValues = []



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

	def getAgents(self,rng,env,eachX):

		print("\nCreating Agents...")
		agents = []
		 
		# For each one of the diffent types of agents
		for a in range(len(self.agentTypes)):

			# Target Agent
			if self.agentTypes[a] == self.agentsModule.TargetAgent:
				keys = ["num_samples","noise_level"]
				pskills = [25]
			# Flip Agent
			elif self.agentTypes[a] == self.agentsModule.FlipAgent:
				keys = ["prob_rational","noise_level"]
				pskills = self.paramValues
			# Tricker Agent
			elif self.agentTypes[a] == self.agentsModule.TrickerAgent:
				keys = ["eps","noise_level"]
				pskills = self.paramValues
			# Bounded Agent
			elif self.agentTypes[a] == self.agentsModule.BoundedAgent:
				keys = ["lambda","noise_level"]

				if self.pSkillsGiven:
					pskills = self.lambdaValues
				else:
					# Find respective lambda values
					pskills = []
					
					# Find lambda range for this %
					info = utilsModule.getLambdaRangeGivenPercent(env.pconfPerXskill,eachX,self.buckets)
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


				# code.interact("...", local=dict(globals(), **locals()))	


			# Delta Agent
			elif self.agentTypes[a] == self.agentsModule.DeltaAgent:
				keys = ["delta","noise_level"]
				pskills = self.paramValuesDelta
			# Target Agent with Beliefs
			elif self.agentTypes[a] == self.agentsModule.TargetAgentWithBeliefs:
				keys = ["noise_level","noise_level_belief"]
				pskills = self.xSkillsBeliefs
			# Random Agent
			elif self.agentTypes[a] == self.agentsModule.RandomAgent:
				keys = ["num_actions","num_samples","noise_level"]
				pskills = self.n


			for p in range(len(pskills)):

				# For Random Agent
				if len(keys) == 3:
					# pskills == self.n in this case
					params = {keys[0]: pskills[p], keys[1]: self.k[p], keys[2]: x}
				
				# For all other agents
				else:
					params = {keys[0]: pskills[p], keys[1]: eachX}
				
				agents.append(self.agentTypes[a](rng,params,self.domain))
		
		print("Done creating Agents.\n")

		return agents
