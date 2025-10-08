import numpy as np

import Environments.Baseball
	

class AgentGenerator():

	__slots__ = ["domainModule","domain","agentTypes","xSkillsBeliefs",
				"xSkills","pitcherIDs","pitchTypes","agents"]
	
	def __init__(self,domainModule,ids=[],types=[]):

		self.domainModule = domainModule
		self.domain = domainModule.getDomainName()

		# Empty for now - might use later
		self.xSkillsBeliefs = []

		
		if ids == []:
			self.pitcherIDs = [656605]
		else:
			self.pitcherIDs = ids


		if types == []:
			self.pitchTypes = ['FF','SL','CU','CH','FS',
						   'SI','KC','FC','CS','FA',
						   'KN','EP','SC','FT'] 
		else:
			self.pitchTypes = types


	def getAgents(self):
		
		# Baseball agents are independent of xskill

		print("\nCreating Agents...")
		
		agents = []

		self.agents = []

		for pi in self.pitcherIDs:
			for pt in self.pitchTypes:
				agents.append([pi,pt])

		print("Done creating Agents.\n")

		return agents
