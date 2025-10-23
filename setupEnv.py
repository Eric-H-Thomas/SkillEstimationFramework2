from os import sep
import numpy as np
import setupSpaces as spacesModule
# from memory_profiler import profile

import code
from importlib.machinery import SourceFileLoader


class Environment():

	__slots__ = ["mode","states","domain","domainName","delta",
				"numObservations","agentGenerator","spaces",
				"xSkillsGiven","pSkillsGiven",
				"numXskillsPerExp","numPskillsPerExp","numRhosPerExp",
				"wrap","pconfPerXskill","mean","dimensions",
				"resultsFolder","particles","resampleNEFF","resample","dynamic","seedNum"]

	# @profile
	def __init__(self,args):
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# FOR DOMAINS
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		# Only meaningful for 1D
		self.wrap = True

		self.dimensions = 1
		self.particles = False

		if args.domain == "1d":

			if args.noWrap:
				self.wrap = False
				import Environments.Darts.RandomDarts.darts_no_wrap as domainModule
				print("Importing 1D - NO WRAP")
			else:
				print("Importing 1D - WRAP")
				import Environments.Darts.RandomDarts.darts as domainModule
		
			args.delta = 1e-2
			self.mode = ""
			self.states = []

		elif args.domain in ["2d","2d-multi"]:
			
			if "multi" in args.domain:
				import Environments.Darts.RandomDarts.two_d_darts_multi as domainModule
			else:
				import Environments.Darts.RandomDarts.two_d_darts as domainModule

			args.delta = 5.0
			self.mode = args.mode
			self.states = []

		elif args.domain == "sequentialDarts":
			import Environments.Darts.SequentialDarts.sequential_darts as domainModule
			args.delta = 5.0
			# Assuming normal mode and same dartboard state for now
			self.mode = "normal"
			numObservations = 1
		
		elif args.domain == "billiards":
			import Environments.Billiards.billiards as domainModule
			args.delta = 0.01

		elif args.domain in ["baseball","baseball-multi"]:

			if "multi" in args.domain:
				import Environments.Baseball.baseball_multi as domainModule
			else:
				import Environments.Baseball.baseball as domainModule
			
			# 0.5 inches | 0.0417 feet
			args.delta = 0.0417

			# 1.0 inches | 0.0833333 feet
			# args.delta = 0.0833333


		elif args.domain in ["hockey-multi"]:

			import Environments.Hockey.hockey as domainModule
			
			args.delta = [0.10169491525423746,0.10256410256410256]


		elif args.domain == "soccer":
			import Environments.Soccer.soccer as domainModule

			# TODO: NEED TO SET ACCORDINGLY
			args.delta = 0.001



		if "multi" in args.domain:
			self.dimensions = 2


		self.domain = domainModule																																																																													
		self.domainName = self.domain.get_domain_name()
		self.delta = args.delta
		self.resultsFolder = args.resultsFolder


		self.seedNum = args.seedNum
		

		if args.domain == "2d-multi" or (args.domain == "hockey-multi" and args.testingBounded):
			utilsModule = SourceFileLoader("utilsDarts",f"Processing{sep}utilsDartsMulti.py").load_module()
		elif args.domain not in ["baseball","soccer"]:
			utilsModule = SourceFileLoader("utilsDarts",f"Processing{sep}utilsDarts.py").load_module()



		####################################
		# PCONF
		####################################

		if self.domainName not in ["baseball","baseball-multi","hockey-multi","soccer"]:
		
			if self.domainName != "2d-multi":

				# Pconf process will use a different seed from the experiment's ones
				tempRng = np.random.default_rng(np.random.randint(1000001,1000000000,1))

				# Compute functions - to use for conversion to % of RandMax Reward
				self.pconfPerXskill = utilsModule.pconf(tempRng,args.resultsFolder,self.domainName,self.domain,spacesModule,args.mode,args,self.wrap)

		####################################

		# How many states to use per experiment?
		# (Not applicable to baseball domain as an experiment 
		# for a given pitcherID & pitchType will use all available data)
		self.numObservations = args.numObservations

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# FOR agentsModule																																										
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		# Flag to know when specific xskills/pskills are given
		# If enabled, will test the given xskills/pskills 
		# specifically and not at random
		# WORKS FOR 1D, 2D AND SEQUENTIAL DARTS ONLY
		self.xSkillsGiven = args.xSkillsGiven
		self.pSkillsGiven = args.pSkillsGiven 

		self.dynamic = args.dynamic


		infoParams = {}


		if self.domainName in ["1d","2d","2d-multi","sequentialDarts"] or (args.domain == "hockey-multi" and args.testingBounded):
			
			if args.domain in ["1d","2d"]:
				import Environments.Darts.RandomDarts.agents as agentsModule
			elif args.domain == "2d-multi" or (args.domain == "hockey-multi" and args.testingBounded):
				import Environments.Darts.RandomDarts.agents_multi as agentsModule
			elif args.domain == "sequentialDarts":
				import Environments.Darts.SequentialDarts.agents as agentsModule


			if args.domain == "2d-multi" or (args.domain == "hockey-multi" and args.testingBounded):
				import Environments.Darts.makeAgentsMulti as module
			else:
				import Environments.Darts.makeAgents as module


			self.numXskillsPerExp = args.numXskillsPerExp
			self.numPskillsPerExp = args.numPskillsPerExp

			
			infoParams = {"xSkillsGiven": self.xSkillsGiven,
						"pSkillsGiven": self.pSkillsGiven,
						"numXskillsPerExp": self.numXskillsPerExp,
						"numPskillsPerExp": self.numPskillsPerExp,
						"dynamic": self.dynamic,
						"dimensions": self.dimensions}

			if args.domain == "2d-multi" or (args.domain == "hockey-multi" and args.testingBounded):

				# Ensure number is ODD to ensure 0.0 is included in the rhos
				# So that Normal JTM can do spaces lookup accordingly in multi space
				self.numRhosPerExp = args.numRhosPerExp

				infoParams["numRhosPerExp"] = self.numRhosPerExp
				infoParams["dimensions"] = self.dimensions

				infoParams["agent"] = args.agent


			if args.someAgents:
				agentTypes = [agentsModule.BoundedAgent]
			elif args.dynamic:
				agentTypes = [agentsModule.TargetAgentAbruptChange,agentsModule.TargetAgentGradualChange]
				agentTypes += [agentsModule.BoundedAgentAbruptChange,agentsModule.BoundedAgentGradualChange]
			else:

				if self.xSkillsGiven and self.pSkillsGiven:
					agentTypes = [agentsModule.BoundedAgent]
				else:
					agentTypes = [agentsModule.TargetAgent,agentsModule.FlipAgent,agentsModule.TrickerAgent,agentsModule.BoundedAgent]



		elif self.domainName == "billiards":

			import Environments.Billiards.makeAgents as module
			
			agentTypes = ["CC"] #["MG"]
			# agentTypes = ["CC","MG"]


			# How many different xskills to use per experiment?
			self.numXskillsPerExp = 1 #14

			# CURRENTLY: CC - 14 | MG - 9

			# How many differnt pskills to use per experiment?
			self.numPskillsPerExp = 1

			# NOTE: infoParams will be empty (will remove later if not needed) 
			# since params for agents are always the same
			# Are not created dynamically
			infoParams = {}
		
		# Baseball domain
		elif self.domainName in ["baseball","baseball-multi"]: 
			import Environments.Baseball.makeAgents as module

		elif self.domainName in ["hockey-multi"]:
			pass


		elif self.domainName == "soccer":
			import Environments.Soccer.makeAgents as module


		if self.domainName in ["baseball","baseball-multi"]:
			self.agentGenerator = module.AgentGenerator(self.domain,args.ids,args.types)
		elif self.domainName in ["hockey-multi"]:
			if args.testingBounded:
				self.agentGenerator = module.AgentGenerator(self.domain,agentTypes,**infoParams)
		elif self.domainName == "soccer":
			self.agentGenerator = module.AgentGenerator(self.domain,args.ids)
		else: # darts
			self.agentGenerator = module.AgentGenerator(self.domain,agentTypes,**infoParams)

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# FOR SPACES
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\

		numSamples = 1000

		argsForSpaces = [numSamples]

		
		if self.domainName in ["1d","2d","2d-multi","sequentialDarts","baseball","baseball-multi"]:
			mainFolder = f"Spaces{sep}ExpectedRewards{sep}"

			if self.domainName in ["baseball","baseball-multi"]:
				fileName = f"ExpectedRewards-{self.domainName}"
			else:
				fileName = f"ExpectedRewards-{self.domainName}-Seed{args.seedNum}-N{numSamples}"
			
			expectedRFolder = mainFolder + fileName
			argsForSpaces.append(expectedRFolder)

		if self.domainName == "sequentialDarts":
			argsForSpaces.append(None)
		
		if self.domainName == "billiards":
			mainFolder = "Spaces" + sep + "SuccessRates" + sep
			fileName = f"SuccessRates-{self.domainName}-N{numSamples}"
			successRatesFolder = mainFolder + fileName
			argsForSpaces.append(successRatesFolder)

		if self.domainName in ["baseball","baseball-multi"]:

			# Set hyperparameters for model
			learningRate = 1e-5
			epochs = 20 #40

			argsForSpaces.append(learningRate)
			argsForSpaces.append(epochs)


		self.initSpaces(args,argsForSpaces)

		# code.interact("...", local=dict(globals(), **locals()))


		# %%%%%%%%%%%%%%%%%%%%%%%s%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		

	# @profile
	def initSpaces(self,args,argsForSpaces):

		# Initial setup for spaces - create object
			
		# Make spaces for agents' xskills
		if self.domainName in ["1d","2d","2d-multi"]:
			self.spaces = spacesModule.SpacesRandomDarts(self.numObservations,self.domain,self.mode,self.delta,argsForSpaces[0],argsForSpaces[1])

		elif self.domainName == "sequentialDarts":
			# argsForSpaces = numSamples,expectedRFolder,valueIterFolder
			self.spaces = spacesModule.SpacesSequentialDarts(self.numObservations,self.domain,self.mode,self.delta,argsForSpaces[0],argsForSpaces[1],argsForSpaces[2])

		elif self.domainName == "billiards":
			# argsForSpaces = numSamples,successRatesFolder,agentTypes
			self.spaces = spacesModule.SpacesBilliards(self.numObservations,self.domain,self.delta,argsForSpaces[0],argsForSpaces[1],self.agentGenerator.agentTypes)
		
		elif self.domainName in ["baseball","baseball-multi"]:
			# argsForSpaces = numSamples,expectedRFolder,learningRate,epochs
			self.spaces = spacesModule.SpacesBaseball(args,self.numObservations,self.domain,self.delta,argsForSpaces[0],argsForSpaces[1],argsForSpaces[2],argsForSpaces[3])

		elif self.domainName in ["hockey-multi"]:
			# argsForSpaces = numSamples,expectedRFolder,learningRate,epochs
			self.spaces = spacesModule.SpacesHockey(args,self.numObservations,self.domain,self.delta)

		elif self.domainName == "soccer":
			self.spaces = spacesModule.SpacesSoccer(args,self.numObservations,self.domain,self.delta)
			
	# @profile
	def setSpacesEstimators(self,rng,args,info):

		print()

		# Update spaces to have corresponding info
		# for xskill hyps (for estimators)
		
		fromEstimator = True

		print("\nCreating spaces with estimators xskills...")

		if self.domainName in ["baseball","baseball-multi","hockey-multi","soccer"]:
			self.spaces.updateSpace(rng,info,fromEstimator)

		elif self.domainName == "billiards":
			for eachType in self.agentGenerator.agentTypes:
				self.spaces.updateSpace(rng,info,eachType,fromEstimator)
		
		elif self.domainName == "sequentialDarts":
			self.spaces.updateSpace(rng,info,fromEstimator)
		
		elif self.domainName == "2d-multi":
			# Won't set convolutions since setting inside exp
			pass

			# To set only once
			# if self.mode == "normal":
				# self.spaces.updateSpace(rng,info,self.states,fromEstimator)

			# If using particles, spaces will get initialized 
			# and managed inside addObservation()
			
		else: # 1D or 2D
			self.spaces.updateSpace(rng,info,self.states,fromEstimator)
		
			for s in self.states:
				for each in info:
					self.spaces.setFocalActions(each,s)

				self.spaces.focalActions[str(s)] = np.array(self.spaces.focalActions[str(s)])


		# To initialize info for exps
		self.spaces.initInfoForExps()

		print("Done creating spaces with estimators xskills...\n")


	# Can possibly merge with setSpacesEstimators
	# @profile
	def setSpacesAgents(self,rng):

		print()

		# To update information on xskills available on spaces
		# Needed for when xskills generated at random

		print("\nCreating spaces with agents xskills...")
		
		# Make spaces for agent's xskills
		if self.domainName in ["1d","2d"]:
			self.spaces.updateSpace(rng,self.agentGenerator.xSkills,self.states)
			# self.spaces.updateSpace(rng,self.agentGenerator.xSkillsBeliefs,self.states)
		
		# elif self.domainName == "2d-multi":	
			# self.spaces.updateSpace(rng,[self.agentGenerator.xSkills,self.agentGenerator.rhos],self.states)

		elif self.domainName == "sequentialDarts":		
			self.spaces.updateSpace(rng,self.agentGenerator.xSkills)
			# self.spaces.updateSpace(rng,self.agentGenerator.xSkillsBeliefs)

		elif self.domainName == "billiards":
			for eachType in self.agentGenerator.agentTypes:
				self.spaces.updateSpace(rng,self.agentGenerator.xSkills,eachType)
				self.spaces.updateSpace(rng,self.agentGenerator.xSkillsBeliefs,eachType)
		
		elif self.domainName in ["2d-multi","baseball","baseball-multi","hockey-multi","soccer"]:
			pass
		
		print("Done creating spaces with agents xskills...\n")

		# code.interact("...", local=dict(globals(), **locals()))


	def setStates(self,rng):

		if self.domainName == "1d":
			# Create the states
			self.states = self.domain.generate_random_states(rng,3,10,self.numObservations,0.0)

			# Create states with fewer regions
			# states = generate_random_states(1,2,self.numObservations,1.0)

		elif self.domainName in ["2d","2d-multi"]:
			# States will be list of slices
			# If mode is normal, slices will be always the same
			# Slices will vary for any other mode
			self.states = self.domain.generate_random_states(rng,self.numObservations,self.mode)


	def resetEnv(self):
		self.spaces.reset()


