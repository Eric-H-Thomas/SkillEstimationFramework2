import numpy as np

import Environments.Billiards
	

class AgentGenerator():

	__slots__ = ["domainModule","domain","agentTypes","xSkillsBeliefs","xskillToAgentCC","xskillToAgentMG","xSkills"]
	
	def __init__(self,domainModule,agentTypes,**infoParams):

		# NOTE: infoParams will be empty (will remove later if not needed) 
		# since params for agents are always the same
		# Are not created dynamically

		self.domainModule = domainModule
		self.domain = domainModule.getDomainName()
		self.agentTypes = agentTypes

		# Empty for now - might use later
		self.xSkillsBeliefs = []

		
		self.xskillToAgentCC = {0.025:23, 0.05:26, 0.0625:9,
					    0.1:19, 0.125:5, 0.1875:10, 0.2:20,
					    0.25:7, 0.3:21, 0.375:11, 0.4:25, 
					    0.5:24, 0.625:27, 0.75:28}

		self.xskillToAgentMG = {0.025:29, 0.05:30, 0.0625:13,
					    0.125:6, 0.1875:15,
					    0.25:12, 0.375:16, 
					    0.5:31, 0.75:32}

		# ASSUMING ONLY 1 AGENT TYPE GIVEN AT A TIME
		# SINCE WILL NEED SAME SET OF XSKILLS FOR ALL TYPES
		# WILL UPDATE LATER, ONCE GAMES ARE AVAILABLE
		if "CC" in agentTypes:
			self.xSkills = list(self.xskillToAgentCC.keys())
		else:
			self.xSkills = list(self.xskillToAgentMG.keys())


	def getAgents(self,eachX):
		
		# ASSUMING SAME AGENTS EVERYTIME

		print("\nCreating Agents...")
		agents = []


		# ASSUMING ONLY 1 AGENT TYPE GIVEN AT A TIME
		# SINCE WILL NEED SAME SET OF XSKILLS FOR ALL TYPES
		# WILL UPDATE LATER, ONCE GAMES ARE AVAILABLE
		if "CC" in self.agentTypes:
			agents.append(self.xskillToAgentCC[self.xSkills[eachX]])
		else:
			agents.append(self.xskillToAgentMG[self.xSkills[eachX]])


		# # For each one of the diffent types of agents
		# for a in range(len(self.agentTypes)):

		# 	agents.append()


		print("Done creating Agents.\n")

		return agents
