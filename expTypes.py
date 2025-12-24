from abc import ABCMeta, abstractmethod
import numpy as np
import code

import sys,os,time
from copy import deepcopy

import Estimators
from Estimators.utils_pfe import *


from scipy.signal import convolve2d
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

from math import dist
from shutil import copy

from itertools import product

from gc import collect
# from pympler import asizeof
# from memory_profiler import profile

# from unxpass.datasets import PassesDataset
# from unxpass import features as fs
# from unxpass import labels as ls
# from unxpass.visualization import plot_action
# from functools import partial
from datetime import datetime

import multiprocessing

from memory_profiler import profile



class Experiment(metaclass=ABCMeta):

	__slots__ = ["list_of_subset_of_estimators","rerun","valid","resultsFolder","seedNum","numObservations",
				"env","agent", "execution_skill", "estimators", "estimatorsObj", "estimators_list", "indexOR", "trueDiffs",
				"observedRewards","intendedActions","noisyActions","results","done","resultsFile","rng","mode",
				"numWorkers","manager","queue","readyEvent","processes","pdfsPerXskill","evsPerXskill"]


	def __init__(self,numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng,rerun=False):
		
		self.resultsFolder = resultsFolder
		self.resultsFile = resultsFile
		self.seedNum = seedNum
		self.rng = rng

		# print("init(): ", self.rng.bit_generator._seed_seq.entropy)

		self.rerun = rerun

		self.numObservations = numObservations
		self.env = env
		
		self.agent = agent
		self.execution_skill = x


		self.estimatorsObj = estimatorsObj
		self.list_of_subset_of_estimators= list_of_subset_of_estimators
		self.estimators_list = estimatorsObj.estimators_list

		# If estimatorsObj == [] means no estimators - use when we don't want the estimators to produce estimates
		# (just let agent plan & take actions & obtain rewards)
		if estimatorsObj != []:
			# Get copy of the estimators 
			estimators = estimatorsObj.getCopyOfEstimators()
		else:
			estimators = []

		self.estimators = estimators
		self.indexOR = indexOR


		# Store diffs and rewards for processing results
		self.trueDiffs = []
		self.observedRewards = []

		self.intendedActions = []
		self.noisyActions = []
		
		self.results = {}

		self.valid = True
		self.done = False
		self.mode = ""


	def getValid(self):
		return self.valid


	def getStatus(self):
		return self.done


	# @profile
	def updateEstimators(self,noisyAction,observedReward,**otherArgs):

		# otherArgs
		# For 1D & 2D: tag,resultsFolder,delta, state
		# For Darts: currentScore,tag,resultsFolder,delta
		# For Billiards: agentType,diff
		# For Baseball: rowIndex, maxEVPerXskill 
			
		# For each estimator, update their estimates/info
		for e in self.estimators:

			# print(e.getEstimatorName())

			# if self.rerun and e not in self.list_of_subset_of_estimators:
			# 	continue

			# print("updateEstimators(): ",self.rng.bit_generator._seed_seq.entropy)
			# startTimeEstimator = time.time()

			if isinstance(e,Estimators.joint.JointMethodQRE) or isinstance(e,Estimators.joint.QREMethod_Multi):
				if "baseball" in self.env.domain_name or "hockey" in self.env.domain_name:
					e.add_observation(self.rng, self.env.spaces, None, noisyAction, **otherArgs)
				else:
					e.add_observation(self.rng, self.env.spaces, self.env.states[otherArgs["i"]], noisyAction, **otherArgs)

			elif isinstance(e,Estimators.joint_pfe.QREMethod_Multi_Particles):
				if "baseball" in self.env.domain_name or "hockey" in self.env.domain_name:
					e.add_observation(self.rng, self.env.spaces, None, noisyAction, **otherArgs)
				else:
					e.add_observation(self.rng, self.env.spaces, self.env.states[otherArgs["i"]], noisyAction, **otherArgs)

			elif isinstance(e,Estimators.joint.JointMethodFlip):
				e.add_observation(self.env.spaces, noisyAction, **otherArgs)
			
			elif isinstance(e,Estimators.joint.NonJointMethodQRE):
				e.add_observation(self.env.spaces, noisyAction, **otherArgs)
			
			elif isinstance(e,Estimators.bayesian.BayesianMethod):
				e.add_observation(self.rng, self.env.domain, self.env.spaces, self.env.states[otherArgs["i"]], noisyAction, **otherArgs)
			
			else: # ObservedReward
				e.add_observation(self.env.spaces, observedReward, **otherArgs)

			
			# print(f"Total time {e.getEstimatorName()}: {time.time()-startTimeEstimator}")

		# code.interact("updateEstimators()...", local=dict(globals(), **locals()))


	@abstractmethod
	def run(self):
		pass


	def saveInitExperimentInfo(self):

		self.results['resultsFile'] = self.resultsFile
		self.results['seedNum'] = self.seedNum

		if self.env.domain_name in ["billiards", "baseball", "baseball-multi", "hockey-multi", "soccer"]:
			self.results['agent_name'] = self.agent
		else:
			self.results['agent_name'] = self.agent.getName()

		
		self.results['xskill'] = self.execution_skill
		self.results['numObservations'] = self.numObservations

		self.results["domain"] = self.env.domain_name
		self.results["delta"] = self.env.delta

		self.results['estimators_list'] = self.estimatorsObj.estimators_list
		
		self.results['num_execution_skill_hypotheses'] = self.estimatorsObj.num_execution_skill_hypotheses
		self.results['num_rationality_hypotheses'] = self.estimatorsObj.num_rationality_hypotheses
		
		self.results['mode'] = self.mode
		
		if self.env.domain_name == "2d-multi":
			self.results['numHypsR'] = self.estimatorsObj.numHypsR
			self.results['rhos'] = self.estimatorsObj.rhos.tolist()


		if self.env.particles:
			self.results['numParticles'] = self.estimatorsObj.numParticles


		if not self.rerun:
			# Save initial experiment info to file
			with open(self.resultsFile,'wb') as outfile:
				pickle.dump(self.results,outfile)


	@abstractmethod
	def getResults(self):

		if self.env.domain_name != "baseball":
			self.results["true_diffs"] = self.trueDiffs

		self.results['intended_actions'] = self.intendedActions
		self.results['noisy_actions'] = self.noisyActions
		self.results['observed_rewards'] = self.observedRewards

		# Save the information for each estimator out to file
		for e in self.estimators:

			# if self.rerun and e not in self.list_of_subset_of_estimators:
			# 	continue

			R = e.get_results()
			for en, er in R.items():
				self.results[en] = er


	def startWorkers(self):

		self.numWorkers = 2

		self.manager = multiprocessing.Manager()

		self.queue = multiprocessing.JoinableQueue()

		self.readyEvent = multiprocessing.Event() 

		self.pdfsPerXskill = self.manager.dict()
		self.evsPerXskill = self.manager.dict()

		self.processes = []

		# Start the workers
		for wid in range(self.numWorkers):
			process = multiprocessing.Process(target=worker,args=(wid,self.queue,self.readyEvent,self.pdfsPerXskill,self.evsPerXskill))
			process.start()
			self.processes.append(process)


	def stopWorkers(self):
		print("Stopping workers...")

		# Stop the workers
		for i in range(self.numWorkers):
			self.queue.put(None)


		# Signal workers to process the termination signal
		self.readyEvent.set()


		# Wait for all worker processes to finish
		for process in self.processes:
			process.join()

		print("Done.")


class RandomDartsExp(Experiment):

	__slots__ = ["rerun","mode","trueRewards","expectedRewards",
				"resampledRewards","valueIntendedActions",
				"meanAllVsPerState","randomRewards","allProbs","nansTimes","key","params"]

	def __init__(self,numObservations,mode,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,allProbs,seedNum,rng,rerun=False):

		if env.domain_name == "2d-multi":
			super().__init__(numObservations,env,agent,x[0],estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng,rerun)
		else:
			super().__init__(numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng,rerun)
		
		self.mode = mode
		self.allProbs = allProbs

		self.trueRewards = []
		self.expectedRewards = []
		self.resampledRewards = []
		self.valueIntendedActions = []
		self.meanAllVsPerState = []
		self.randomRewards = []

		if "Bounded" in self.agent.name: 
			self.nansTimes = 0

		if self.env.domain_name == "2d-multi":

			self.params = x

			temp = ""

			for t in x[0]:
				temp += f"{t}|"

			# Add rho param
			self.key = f"{temp}{x[1]}"

		self.saveInitExperimentInfo()


	# @profile
	def run(self,tag,counter,returnZn=False):

		# print(f"AGENT: {self.agent.name} | PID: {os.getpid()}")

		otherArgs = {"tag":tag+str(counter),"resultsFolder":self.resultsFolder}


		# For each one of the different states
		for i in range(self.numObservations):

			if i % 100 == 0:
				print(f"<state {i} of {self.numObservations}>")


			self.resampledRewards.append([])

			# Get state for evaluation
			state = self.env.states[i]
			otherArgs["s"] = str(state)
			otherArgs["actualState"] = state
			otherArgs["i"] = i
			otherArgs["agent"] = self.agent.name


			# To setup space(convolutions) for current xskill and states
			# Will perform convolution if not present. Returns info either way.
			if not self.env.dynamic:

				if self.env.domain_name == "2d-multi":
					convolutionSpace = self.env.spaces.getSpace(self.rng,self.params,state)
				else:
					convolutionSpace = self.env.spaces.getSpace(self.rng, self.execution_skill, state)

			listedTargets = self.env.spaces.listedTargets


			# Get agent's action 
			if "Change" in self.agent.name:
				action, evAction, convolutionSpace = self.agent.get_action(self.rng,self.env.domain,listedTargets,self.env.spaces,otherArgs)
			elif "Bounded" in self.agent.name:
				action, nansFlag, evAction = self.agent.get_action(self.rng,self.env.domain,listedTargets,convolutionSpace,returnZn)

				if nansFlag == True:
					nansTimes += 1
			else:
				action, evAction = self.agent.get_action(self.rng,self.env.domain,listedTargets,convolutionSpace,returnZn)


			self.intendedActions.append(action)
			self.valueIntendedActions.append(evAction)

			self.meanAllVsPerState.append(convolutionSpace["mean_vs"])


			# Add noise to action + get respective reward
			if self.env.domain_name == "2d-multi":
				noisy_action = self.env.domain.sample_noisy_action(self.rng,state,self.agent.mean,self.agent.covMatrix,action,self.agent.noiseModel)
			else:
				noisy_action = self.env.domain.sample_noisy_action(self.rng, state, self.execution_skill, action, self.agent.noiseModel)
				

			if self.env.domain_name == "1d":
				observed_reward = self.env.domain.get_reward_for_action(self.rng,state,noisy_action)
			else:
				observed_reward = self.env.domain.get_reward_for_action(state,noisy_action)
			
		diff_fn = getattr(self.env.domain, "calculate_wrapped_action_difference", None)
		if diff_fn is None:
			diff_fn = getattr(self.env.domain, "calculate_action_difference")

			trueDiff = diff_fn(noisy_action, action)

			self.noisyActions.append(noisy_action)
			self.observedRewards.append(observed_reward)
			self.trueDiffs.append(trueDiff)


			# Make the first one match - (to store the actual observed reward)
			self.resampledRewards[i].append(observed_reward)


			# Resample rewards
			for j in range(20):		

				if self.env.domain_name == "2d-multi":
					na = self.env.domain.sample_noisy_action(self.rng,state,self.agent.mean,self.agent.covMatrix,action,self.agent.noiseModel)
				else:
					na = self.env.domain.sample_noisy_action(self.rng, state, self.execution_skill, action, self.agent.noiseModel)
				
				if self.env.domain_name == "1d":
					ob = self.env.domain.get_reward_for_action(self.rng,state,na)
				else:
					ob = self.env.domain.get_reward_for_action(state,na)
		
				self.resampledRewards[i].append(ob)


			#Expected Reward
			#tt, exp_reward = get_optimal_action_and_value(state, xskill)
			#tt = agent.get_action(i)

			if self.env.domain_name == "1d":
				exp_reward = self.env.domain.get_reward_for_action(self.rng,state,action)
			else:
				exp_reward = self.env.domain.get_reward_for_action(state,action)
			self.expectedRewards.append(exp_reward)


			# FOR TESTING PURPOSES
			# if self.env.domainName == "2d-multi":
			# 	otherArgs["plot"] = True


			# Update estimators
			self.updateEstimators(noisy_action,observed_reward,**otherArgs)

			# print(f"Action: {action} | Noisy Action: {noisy_action}")
			# print(f"evAction: {evAction} | observedReward: {observed_reward}")
			# code.interact("End of state...", local=dict(globals(), **locals()))
			# End of state


			# Memory Management - Delete conv space for current agent
			# Can't delete cause set of agents using same  set of states (regardless of mode)
			# Reset env will handle this once exp is done
			'''
			if "2d-multi" in self.env.domainName and self.env.mode != "normal":
				print("DELETING")
				kk = self.env.spaces.getKey(self.params[0],self.params[1])
				del self.env.spaces.convolutionsPerXskill[kk][str(state)]


			elif self.env.domain == "1d" or \
			(self.env.domainName == "2d" and self.env.mode != "normal"):
				self.env.spaces.deleteSpace(self.xskill,state)
			'''

			# code.interact("end of state...", local=dict(globals(), **locals()))


		# if "Change" in self.agent.name:
			# self.plotIntendedActions()


		# End of exp
		print(f"<Done>\n")
		self.done = True
		# code.interact("...", local=dict(globals(), **locals()))

		'''
		# if self.xskill > 150:
		if "Tricker" in self.agent.name or "Flip" in "Tricker" in self.agent.name:
			print("Mean true rewards: ",np.mean(self.agent.get_true_rewards()))
			print("Mean intended actions: ",np.mean(self.valueIntendedActions))
			print("Mean random rewards: ",np.mean(self.meanAllVsPerState))

		'''


	def getResults(self):
		
		super().getResults()

		if self.env.domain_name == "1d":
			self.results['intended_actions'] = self.intendedActions
			self.results['noisy_actions'] = self.noisyActions
			self.results['wrap'] = self.env.wrap

		elif "2d" in self.env.domain_name:

			# Will store actions separately since they are in a 2D-list ([[x,y], [x,y]]) and causes
			# 'not serializable' error when trying to save

			# Convert to numpy array in order to access respective columns easier
			intended_actions = np.array(self.intendedActions)
			noisy_actions = np.array(self.noisyActions)
			
			# Select column containing x's 
			intended_actions_x = intended_actions[:,0]
			# Select column containing y's
			intended_actions_y = intended_actions[:,1]
			
			# Select column containing x's 
			noisy_actions_x = noisy_actions[:,0]
			# Select column containing y's
			noisy_actions_y = noisy_actions[:,1]

			# Now can store
			self.results['intended_actions_x'] = list(intended_actions_x)
			self.results['intended_actions_y'] = list(intended_actions_y)
			
			self.results['noisy_actions_x'] = list(noisy_actions_x)
			self.results['noisy_actions_y'] = list(noisy_actions_y)
	

		self.results['exp_rewards'] = self.expectedRewards
		self.results['rs_rewards'] = self.resampledRewards

		self.results['valueIntendedActions'] = self.valueIntendedActions
		self.results['meanAllVsPerState'] = self.meanAllVsPerState

		self.results['true_rewards'] = self.agent.get_true_rewards()


		if "multi" in self.env.domain_name:
			self.results['mean'] = self.agent.mean
			self.results['covMatrix'] = self.agent.covMatrix.tolist()
			self.results["dimensions"] = self.env.dimensions

		# Saving the states used for the experiment - dictionary
		self.results["states"] = {"states" : self.env.states}
		self.results["domain"] = self.env.domain_name
		self.results["mode"] = self.env.mode
		

		if "Bounded" in self.agent.name:

			nansInfoS,nansInfoCounter,nansInfoFinalL = self.agent.getNansInfo()

			self.results['nansTimes'] = self.nansTimes
			self.results['nansInfoS'] = nansInfoS
			self.results['nansInfoCounter'] = nansInfoCounter
			self.results['nansInfoFinalL'] = nansInfoFinalL

			if self.allProbs:
				allProbs = self.agent.getAllProbs()

				# returned as a dict
				self.results["allProbs"] = self.allProbs


		if self.env.dynamic:

			# Save dynamic xskill info used by agent on exps
			if "AbruptChange" in self.agent.name:
				self.results['changeAt'] = self.agent.changeAt

			if "GradualChange" in self.agent.name:
				self.results['gradualXskills'] = self.agent.xSkills.tolist()


		return self.results


	def plotIntendedActions(self):

		print("Plotting intended actions...")

		if not os.path.exists(f"Experiments{os.sep}{self.resultsFolder}{os.sep}IntendedActions{os.sep}"):
			os.mkdir(f"Experiments{os.sep}{self.resultsFolder}{os.sep}IntendedActions{os.sep}")


		saveAt = f"Experiments{os.sep}{self.resultsFolder}{os.sep}IntendedActions{os.sep}{self.agent.name}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)


		for i in range(self.numObservations):

			fig = plt.figure()
			ax = plt.gca()

			self.env.domain.draw_board(ax)
			self.env.domain.label_regions(self.env.states[i])

			plt.scatter(self.intendedActions[i][0],self.intendedActions[i][1])


			label = f"Observation: {i} | "

			if "AbruptChange" in self.agent.name:
				if i < self.agent.changeAt:
					label += f"ChangeAt: {self.agent.changeAt} | Current X: {self.agent.startX}"
				else:
					label += f"ChangeAt: {self.agent.changeAt} | Current X: {self.agent.endX}"

			elif "GradualChange" in self.agent.name:
				label += f"Current X: {self.agent.xSkills[i]}"

			plt.title(label)


			plt.savefig(f"{saveAt}NumObs{i}.png",bbox_inches='tight')
			plt.clf()
			plt.close("all")

			# code.interact("...", local=dict(globals(), **locals()))


class SequentialDartsExp(Experiment):

	__slots__ = ["rerun","mode","valueIntendedActions","meanAllVsPerState",
				"space","allProbs","nansTimes"]

	def __init__(self,numObservations,mode,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,allProbs,seedNum,rng,rerun=False):

		super().__init__(numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng,rerun)
		
		self.mode = mode
		self.allProbs = allProbs

		self.valueIntendedActions = []
		self.meanAllVsPerState = []

		self.space = env.spaces.spacesPerXskill[x]

		if "Bounded" in self.agent.name: 
			self.nansTimes = 0

		self.saveInitExperimentInfo()


	def run(self,tag,counter):
		
		otherArgs = {"tag":tag+str(counter),"resultsFolder":self.resultsFolder}

		atObservation = 0


		# Will keep playing games until the desired # of observations/states is reached
		while atObservation < self.numObservations:

			# Reset setup for new game
			startScore = self.env.domain.getPlayerStartScore()
			currentScore = startScore

			# While playing the game (0 = last possible score/state -> game done)
			while currentScore > 0:
				
				print(f"<Current score: {currentScore}>")
				otherArgs["currentScore"] = currentScore

				noisyAction,observedReward,nextScore = self.interact(currentScore)

				# Update estimators
				self.updateEstimators(noisyAction,observedReward,**otherArgs)

				# End of state
				atObservation += 1
				# atObservation = num turns seen so far overall for darts

				# We've seen all the desired # of observations, can stop
				if atObservation == self.numObservations:
					break

				# Prep for next iter
				currentScore = int(nextScore)

				# code.interact("...", local=dict(globals(), **locals()))
	
				# If game finished
				if currentScore == 0 and self.indexOR != None:
					self.estimators[self.indexOR].gameFinished()
					# print("DONE!")
			
			print(f"<Current score: {currentScore}>\n<Done>\n")
			# code.interact("...", local=dict(globals(), **locals()))
			# Game finished

		# code.interact("End of experiment...", local=dict(globals(), **locals()))
		
		# if self.xskill > 150:
		'''
		if "Tricker" in self.agent.name or "Flip" in self.agent.name:
			print("Mean true rewards: ",np.mean(self.agent.getTrueRewards()))
			print("Mean intended actions: ",np.mean(self.valueIntendedActions))
			print("Mean random rewards: ",np.mean(self.meanAllVsPerState))

			code.interact("...", local=dict(globals(), **locals()))
		'''
		# code.interact("...", local=dict(globals(), **locals()))


		# End of experiment
		self.done = True


	def interact(self,currentScore):

		# Get agent's action 
		if "Bounded" in self.agent.name:
			action, valueAction, nansFlag = self.agent.getAction(self.rng,self.env.domain,self.space,currentScore)

			if nansFlag == True:
				self.nansTimes += 1
		elif "BeliefX" in self.agent.name:
			action, valueAction = self.agent.getAction(self.rng,self.env.domain,env.spaces,currentScore)
		else:
			action, valueAction = self.agent.getAction(self.rng,self.env.domain,self.space,currentScore)


		# Add noise to action
		noisyAction = self.env.domain.sampleAction(self.execution_skill, action, self.agent.noiseModel)

		# print(f"Intended Action: {action}")
		# print(f"Noisy Action: {noisyAction}")
		diff_fn = getattr(self.env.domain, "calculate_wrapped_action_difference", None)
		if diff_fn is None:
			diff_fn = getattr(self.env.domain, "calculate_action_difference")
		self.trueDiffs.append(diff_fn(noisyAction, action))

		# Save info
		self.intendedActions.append(action)
		self.valueIntendedActions.append(valueAction)
		self.noisyActions.append(noisyAction)
		self.meanAllVsPerState.append(self.space.meanEVsPerState[currentScore])

		# newScore,double = self.env.domain.score(self.space.boardStates,noisyAction)
		newScore,double = self.env.domain.npscore(noisyAction[0],noisyAction[1],return_doub=True)
	
		# Get reward (reward = turn was taken)
		observedReward = -1
		self.observedRewards.append(observedReward)

		nextScore = currentScore-newScore

		if nextScore < 0 or nextScore == 1:
			nextScore = currentScore 
		if nextScore == 0:
			if not double:
				nextScore = currentScore 
		
		return noisyAction,observedReward,nextScore

	def getResults(self):
		
		super().getResults()

		self.results['true_rewards'] = self.agent.getTrueRewards()
		self.results["mode"] = self.mode
		
		self.results["valueIntendedActions"] = self.valueIntendedActions
		self.results["meanAllVsPerState"] = self.meanAllVsPerState

		# Will store actions separately since they are in a 2D-list ([[x,y], [x,y]]) and causes
		# 'not serializable' error when trying to save
		# Convert to numpy array in order to access respective columns easier
		self.intendedActions = np.array(self.intendedActions).tolist()
		self.noisyActions = np.array(self.noisyActions).tolist()

		self.results['intended_actions'] = self.intendedActions
		self.results['noisy_actions'] = self.noisyActions


		if "Bounded" in self.agent.name:

			nansInfoS, nansInfoCounter, nansInfoFinalL = self.agent.getNansInfo()

			self.results['nansTimes'] = self.nansTimes
			self.results['nansInfoS'] = nansInfoS
			self.results['nansInfoCounter'] = nansInfoCounter
			self.results['nansInfoFinalL'] = nansInfoFinalL

			if self.allProbs:
				allProbs = agent.getAllProbs()

				# returned as a dict
				self.results["allProbs"] = allProbs

		return self.results


class BilliardsExp(Experiment):

	__slots__ = ["rerun","processedShotsList","estimatedDiffs","agent","agentType","space"]

	def __init__(self,numObservations,env,agent,x,estimatorsObj,resultsFolder,resultsFile,indexOR,seedNum,rerun=False):
		
		super().__init__(numObservations,env,agent,x,estimatorsObj,resultsFolder,resultsFile,indexOR,seedNum)
		
		self.rerun = rerun

		env.domain.Shot.shot_list = []
		
		self.processedShotsList = []

		self.estimatedDiffs = []

		# Agent ID
		self.agent = agent
		self.agentType = env.spaces.agentIdToType[agent]

		self.space = env.spaces.spacesPerXskill[self.agentType][x]

		import Environments.Billiards.utilsBilliards as utils

		self.saveInitExperimentInfo()

	def run(self,tag,counter):

		otherArgs = {"tag":tag+str(counter),"agentType": self.agentType,"resultsFolder":self.resultsFolder}
		
		self.processedShotsList = self.env.domain.getAndProcessShots(self.numObservations,self.agent,self.seedNum,self.rerun)

		currentGameID = self.processedShotsList[0]


		# For each one of the processed shots
		for i in range(len(self.processedShotsList)):

			eachShot = self.processedShotsList[i]

			# Executed shot
			phi = float(eachShot.phi)

			# Save info
			self.intendedActions.append(eachShot.nlPhi)
			self.noisyActions.append(phi)

			# Get the list containing all the possible shots for the given shot out of the object
			possibleShots = eachShot.estimatedNLPhisAll

			# diff = target(closest to noisy) - executed
			diff = float(eachShot.estimatedNoise)
			# print "Estimated Noise/diff: ",diff
			self.estimatedDiffs.append(diff)

			# Estimated True Noise
			trueDiff = float(eachShot.trueNoise)
			self.trueDiffs.append(trueDiff)

			if eachShot.successfulOrNot == "Yes":
				observedReward = 1.0
			else:
				observedReward = 0.0
			self.observedRewards.append(observedReward)


			# Save info -- Not available for billiards?
			# self.valueIntendedActions.append() # EV for intended action 
			# self.meanAllVsPerState.append()

			 # diff = estimatedNoise (target(closest to noisy) - executed)
			otherArgs["diff"] = diff
			otherArgs["shot"] = self.env.domain.Shot.shot_list[i]
			otherArgs["processedShot"] = eachShot
			otherArgs["possibleShots"] = possibleShots

			# Update estimators
			self.updateEstimators(phi,observedReward,**otherArgs)

			
			# If there are still shots to look at, determine if game finished or not
			if i+1 < len(self.processedShotsList):
				
				nextShotGameID = self.processedShotsList[i+1].gameID
				
				# To do so: Verify if the next shot to be processed is from the current game or not
				# If it is not, a new game will be starting, so need to update currentGameID and OR-FullGame estimates before then
				if currentGameID != nextShotGameID:
					
					self.estimators[self.indexOR].gameFinished()
					
					currentGameID = nextShotGameID

			# code.interact("end of shot...", local=dict(globals(), **locals()))

		self.done = True


	def getResults(self):

		super().getResults()

		self.results['rerun'] = self.rerun

		self.results['intended_actions'] = self.intendedActions
		self.results['noisy_actions'] = self.noisyActions
		self.results["estimated_diffs"] = self.estimatedDiffs

		return self.results


class BaseballExp(Experiment):

	__slots__ = ["rerun","pitcherID","pitchType","space","agentData","xSkills",
				"timesPerObservations","valid",
				"agentFolder","infoFile","estimatorsInfoFile",
				"plotFolder1","plotFolder2","pickleFolder","infoPerRow",
				"iter","every","pdfsPerXskill","pdfsFile","rerun","done","checkpointDataAvailable",
				"tempResultsFile","tempInfoFile","tempEstimatorsFile","infoFileYear",
				"jeeds","pfe","pfeNeff","comm"]


	# @profile
	def __init__(self,args,env,agent,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng):
		
		global torch, nn, pd
		
		# Importing here as only baseball exps need these modules
		import torch
		import torch.nn as nn 
		import pandas as pd


		# self.numWorkers = args.numWorkers
		# self.queue = args.queue
		# self.readyEvent = args.readyEvent


		x = "N/A"

		super().__init__(env.numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng)
		

		self.xSkills = estimatorsObj.allXskills

		# Using symmetric set since will need to init pdfsPerXskill info just for normal JTM
		# The ones for the particles will be managed within addObservation()
		if "multi" in self.env.domain_name:
			# self.xSkills = list(product(list(map(list,self.xSkills)),estimatorsObj.rhos))
			self.xSkills = estimatorsObj.xskillsNormal


		self.every = args.every
		self.rerun = args.rerun

		self.done = False
		self.iter = 1

		self.pitcherID = agent[0]
		self.pitchType = agent[1]
		self.agentData = []

		self.jeeds = args.jeeds
		self.pfe = args.pfe
		self.pfeNeff = args.pfeNeff


		print("Getting data for agent...")

		
		# To avoid loading data twice the very first experiment
		# Since spaces constructors loads it when space object created
		if not self.env.spaces.dataLoaded:
			# Load/Query all available data for given date range
			self.env.spaces.getAllData()


		toSend = None

		if args.dataBy == "recent":
			toSend = [args.maxRows]
		elif args.dataBy == "chunks":
			toSend = [self.numObservations,self.seedNum]
		elif args.dataBy == "pitchNum":
			# Adjust starting point since list slicing starts from 0
			# Endpoint is ok since list slicing not inclusive of ending point
			toSend = [args.b1-1,args.b2]

		# Grab corresponding data for agent (give pitchID & pitchType)
		# Also deletes allData (for memory optimization)
		self.agentData = self.env.spaces.getAgentData(args.dataBy,self.pitcherID,self.pitchType,toSend)

		self.timesPerObservations = []


		##################################################################################
		# VALID EXPERIMENT OR NOT?
		##################################################################################
		
		if len(self.agentData) == 0:
			print("No data is present for the given agent. Can't proceed with experiment.")
			# Experiment unsuccessful
			self.valid = False

		else:

			# Data is present for agent, can proceed
			print(f"Data was obtained successfully.")

			self.valid = True
			self.numObservations = len(self.agentData)


			otherTag = ""

			if self.jeeds:
				otherTag += f"_JEEDS_Chunk_{self.seedNum}"

			if self.pfe:
				otherTag += f"_PFE_Chunk_{self.seedNum}"

			if self.pfeNeff:
				otherTag += f"_PFE_NEFF_Chunk_{self.seedNum}"


			agentFileName = f"pitcherID{self.pitcherID}-PitchType{self.pitchType}{otherTag}"

			self.initialSetup()


			##################################################################################
			# Check if the pdfsPerXskill info is available or not
			##################################################################################
			
			loaded = False

			try:
				print("\nVerifying if pdfs for xskill hyps are present in order to load them...",end=" ")
			
				# Load pdfs previously used
				with open(self.pdfsFile,'rb') as handle:
					self.env.spaces.pdfsPerXskill = pickle.load(handle)

				loaded = True

				print("Done")
		
			except:
				print("No info was available.")
				pass


			if loaded:
				print("\nVerifying info available (computing when needed)...")
			else:
				print("\nComputing info...")


			updated = 0 

			for each in self.xSkills:

				# Using symmetric set since will need to init pdfsPerXskill info just for normal JTM
				# The ones for the particles will be managed within addObservation()
				if "multi" in self.env.domain_name:

					key = self.env.spaces.getKey([each,each],0.0)

					if key in self.env.spaces.pdfsPerXskill:
						print(f"Info available for {key}")

					# Symmetric Case (assumming 2 dims for now)
					elif each in self.env.spaces.pdfsPerXskill:
						print(f"Prev symmetric info available for {each}. Updating key to {key}.")
						# Update key
						self.env.spaces.pdfsPerXskill[key] = np.copy(self.env.spaces.pdfsPerXskill[each])
						del self.env.spaces.pdfsPerXskill[each]
						updated += 1

					else:
						print(f"Computing info for {key}.")
						covMatrix = self.env.domain.getCovMatrix([each,each],0.0)
						self.env.spaces.pdfsPerXskill[key] = self.env.domain.getNormalDistribution(self.rng,covMatrix,self.env.spaces.delta,self.env.spaces.targetsPlateXFeet,self.env.spaces.targetsPlateZFeet)
						updated += 1

				else:

					if each in self.env.spaces.pdfsPerXskill:
						print(f"Info available for {each}.")
					else:
						print(f"Computing info for {each}.")
						self.env.spaces.pdfsPerXskill[each] = self.env.domain.getSymmetricNormalDistribution(self.rng,each,self.env.spaces.delta,self.env.spaces.targetsPlateXFeet,self.env.spaces.targetsPlateZFeet)
						updated += 1

			# code.interact("()...", local=dict(globals(), **locals()))

			# Only save if new info was added to dict
			if updated > 0:
				self.saveInfoPDFs()

			print("Done.")

			##################################################################################


			##################################################################################
			# Validate checkpoint
			##################################################################################


			rf = f"Experiments{os.sep}{self.resultsFolder}{os.sep}"

			resultsFileJustName = f"OnlineExp_{self.pitcherID}_{self.pitchType}{otherTag}.results"
			estimatorsFileJustName = f"estimators-info-{agentFileName}.pkl"
			infoFileJustName = f"info-{agentFileName}.pkl"

			self.tempResultsFile = f"{rf}results{os.sep}temp-{resultsFileJustName}"
			self.tempEstimatorsFile = f"{rf}info{os.sep}temp-{estimatorsFileJustName}"
			self.tempInfoFile = f"{rf}info{os.sep}temp-{infoFileJustName}"


			validCheckpointFound = True

			# If either one of the temp files is present, 
			# something happened in the middle of a checkpoint
			if Path(self.tempResultsFile).is_file() or Path(self.tempEstimatorsFile).is_file() or Path(self.tempInfoFile).is_file():

				print("Something happened during the previous checkpoint as temp files are present.")

				# Start from previous checkpoint, if any available
				try:
					print("Checking to see if info for a previous checkpoint is available...")
					
					copy(f"{rf}results{os.sep}backup{os.sep}{resultsFileJustName}",f"{rf}results{os.sep}{resultsFileJustName}")
					copy(f"{rf}info{os.sep}backup{os.sep}{estimatorsFileJustName}",f"{rf}info{os.sep}{estimatorsFileJustName}")
					copy(f"{rf}info{os.sep}backup{os.sep}{infoFileJustName}",f"{rf}info{os.sep}{infoFileJustName}")
				
					print("Info from previous checkpoint obtained.")

				# Couldn't find backup info, proceed with full experiment
				except:
					print("Couldn't find necessary backup info, doing with full experiment.")
					
					# In case rerun flag was enabled
					self.rerun = False
					self.checkpointDataAvailable = False
					validCheckpointFound = False


				# Clean up
				if Path(self.tempResultsFile).is_file():
					os.remove(self.tempResultsFile)

				if Path(self.tempEstimatorsFile).is_file():
					os.remove(self.tempEstimatorsFile)

				if Path(self.tempInfoFile).is_file():
					os.remove(self.tempInfoFile)

			##################################################################################


			##################################################################################
			# Check if there's any info available for the estimators
			##################################################################################
				
			# Only if not in rerun mode
			if not self.rerun and validCheckpointFound:
				try:
					print("\nVerifying if estimators info is present in order to load them...",end=" ")
					# Reload info into estimators accordingly (ensure continuality)
					with open(self.estimatorsInfoFile,'rb') as handle:
						self.estimators = pickle.load(handle)

					print("Done.")

				except:
					print("No estimator info available to load.")

			# Otherwise, estimators already set at the beggining 

			##################################################################################


			##################################################################################
			# Check if there's any already data processed to start from
			##################################################################################

			self.checkpointDataAvailable = False

			self.infoFileYear = 99999999

			if validCheckpointFound:
				
				try:
					print("\nChecking if there's any processed data available...")
					
					# Load already process data from file, if any
					with open(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl",'rb') as handle:
						tempAgentData = pickle.load(handle)

					#stats = os.stat(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl")

					#self.infoFileYear = datetime.fromtimestamp(stats.st_birthtime).year
					#print(f"File found from year {self.infoFileYear}.")

					self.checkpointDataAvailable = True

					# How many rows were processed already?
					seen = len(tempAgentData)
					
					# How many rows were processed already?
					available = len(tempAgentData)
					print(f"Loaded available processed info (# rows: {available}).")
					
				except:
					print("No processed data available.")
					self.checkpointDataAvailable = False

			##################################################################################
			

			##################################################################################
			# Experiment setup
			##################################################################################

			# CASE: Reload/Rerun need to set seed accordingly
			if Path(self.resultsFile).is_file() and validCheckpointFound:

				# Load info available
				with open(self.resultsFile,'rb') as handle:
					prevResults = pickle.load(handle)

				self.seedNum = prevResults["seedNum"]
				# print("Setting seed: ",self.seedNum)

				# Set seed for current experiment to that of previous exp
				np.random.seed(self.seedNum)

			else:
				# FOR TESTING
				# self.seedNum = 599433
				# np.random.seed(self.seedNum)
				
				self.saveInitExperimentInfo()


			self.infoPerRow = {}
			
			# Either some or all processed data is available
			if self.checkpointDataAvailable:
	
				# CASE: All data available, start from beginning (rerun or reload mode)
				if len(tempAgentData) == self.numObservations or args.rerun or args.reload:
					# print("Rerun mode.")
					lookAt = self.agentData.index.values[:self.every]
					self.iter = 1

				# Case partial data is available, continuation exp
				else:

					# To enable saving of newly processed rows since reload mode
					# And not attempt to load new batch of data as it won't be present
					self.checkpointDataAvailable = False
					
					# Update rows to be seen
					lookAt = self.agentData.index.values[seen:seen+self.every]
					self.iter = seen+1

					# CASE: reload/continuation exp after checkpoint
					# Subset self.agent data to contain only data left to look at
					# Minus 1 cause iter counter starting at 1
					self.agentData = self.agentData.iloc[(self.iter-1):,:]

				
				print("Loading info accordingly...",end=" ")
				
				# Load the first "every" observations
				for index in lookAt:
					# If info for row present
					if index in tempAgentData:
						self.infoPerRow[index] = tempAgentData[index]

				print("Done.")

			
			# code.interact("...", local=dict(globals(), **locals()))
		

			# In case attempting to run exp (not in rerun mode) and all info present
			if len(self.agentData) == 0:
				print("All data has been processed", end=" ")

				# Experiment performed before
				if Path(self.resultsFile).is_file():
					print("and results file is present. Stopping.")
					self.valid = False
					self.done = True

				# Otherwise valid = True & done = False to proceed with full exp
				else:
					print("but results file is not present. Proceed with experiment.")


			# Initial backup
			if self.rerun:

				# File management
				# Move files to their respective backup folder
				# As full results/estimators file from prev exp is present at this point
				
				# Adding counter to filename to remember all prev results (won't overwrite)
				currentFiles = os.listdir(f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}backup{os.sep}")
				fileName = f"OnlineExp_{self.pitcherID}_{self.pitchType}"
				# counter = len(currentFiles)+1
				# newName = f"{fileName}-{counter}.results" 
				newName = f"{fileName}.results" 
				
				try:
					# print("Moving results file from previous experiment to backup folder...")
					os.rename(f"{self.resultsFile}", f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}backup{os.sep}{newName}")
					self.saveInitExperimentInfo()
				except:
					pass

				try:
					# print("Moving estimators file from previous experiment to backup folder...")
					newName = f"estimators-info-{agentFileName}.pkl" 
					os.rename(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}estimators-info-{agentFileName}.pkl", f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}backup{os.sep}{newName}")
				except:
					pass

			##################################################################################


		##################################################################################


	# @profile
	def initialSetup(self):

		otherTag = ""

		if self.jeeds:
			otherTag += f"_JEEDS_Chunk_{self.seedNum}"

		if self.pfe:
			otherTag += f"_PFE_Chunk_{self.seedNum}"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF_Chunk_{self.seedNum}"


		self.agentFolder = f"pitcherID{self.pitcherID}-PitchType{self.pitchType}{otherTag}"
		
		infoFolder = f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}"
		self.infoFile = f"{infoFolder}info-{self.agentFolder}.pkl"
		self.pdfsFile = f"{infoFolder}pdfsPerXskill-{self.agentFolder}.pkl"
		self.estimatorsInfoFile = f"{infoFolder}estimators-info-{self.agentFolder}.pkl"
		
		self.pickleFolder = f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}StrikeZoneBoards{os.sep}PickleFiles{os.sep}{self.agentFolder}{os.sep}"
		
		folders = [infoFolder,self.pickleFolder]


		for folder in folders:
			if not os.path.exists(folder):
				os.mkdir(folder)
		

	def saveInitExperimentInfo(self):

		self.results["iter"] = self.iter
		super().saveInitExperimentInfo()


	def saveInfoPDFs(self):
		print("Saving pdfs info to file...")

		with open(self.pdfsFile,'wb') as outfile:
			pickle.dump(self.env.spaces.pdfsPerXskill,outfile)



	# @profile
	# def run(self,tag,counter,num,comm):
	def run(self,tag,counter):

		
		# self.startWorkers()
		# self.numWorkers = num
		# self.comm = comm


		# from pympler.tracker import SummaryTracker	
		# tracker = SummaryTracker()


		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


		otherArgs = {"tag":tag+str(counter),"resultsFolder":self.resultsFolder}

		possibleTargetsLen = len(self.env.spaces.possibleTargetsForModel)


		print("\nGoing through observations...")


		# For each row in the data
		# That is, for a given pitch (observation/state)
		for row in self.agentData.itertuples():

			index = row.Index

			otherArgs["i"] = self.iter


			# If rerun mode, processed info already available
			# Just update estimators and create checkpoints
			if index in self.infoPerRow:

				print(f"Looking at row {self.iter}/{self.numObservations} | index: {index} | loaded...")

				startTime = time.time()
 
				otherArgs["infoPerRow"] = self.infoPerRow[index]

				# code.interact("...", local=dict(globals(), **locals()))

				otherArgs["allTempData"] = self.infoPerRow[index]["allTempData"]
				otherArgs["minUtility"] = self.infoPerRow[index]["minUtility"]

				# Populate Dartboard - Can create once since independent of xskill
				# t1 = time.perf_counter()
				Zs = self.env.spaces.setStrikeZoneBoard(otherArgs["allTempData"],otherArgs["minUtility"])
				otherArgs["Zs"] = Zs


				# Save info
				self.noisyActions.append([row.plate_x_feet,row.plate_z_feet])
				self.observedRewards.append(self.infoPerRow[index]["observedReward"])


				computed = 0 

				for x in self.xSkills:

					if "multi" in self.env.domain_name:

						# Getting symmetric key since pdfsPerXskill info initialized only for normal JTM
						# The ones for the particles will be managed within addObservation()
						x = self.env.spaces.getKey([x,x],0.0)


					# Compute if haven't seen before OR need to be updated
					# Need to be updated = symmetric case from prev exps since pdfs wrong
					if x not in self.infoPerRow[index]["evsPerXskill"] or (x in self.infoPerRow[index]["evsPerXskill"] and self.infoFileYear < 2024):
						
						if x not in self.infoPerRow[index]["evsPerXskill"]:
							print(f"Computing EVs for {x}.")
						elif (x in self.infoPerRow[index]["evsPerXskill"] and self.infoFileYear < 2024):
							print(f"Recomputing EVs for {x}.")

						# Recompute pdfs (with prev & new set of xskills)
						Zs = self.env.spaces.setStrikeZoneBoard(otherArgs["allTempData"],otherArgs["minUtility"])
						self.evsPerXskill[x] = convolve2d(Zs,self.pdfsPerXskill[x],mode="same",fillvalue=otherArgs["minUtility"])
						computed += 1


				# Update file if present
				if computed > 0 and self.infoFileYear != 99999999:

					# If file with info exists already, load info (to avoid overwriting)
					if Path(self.infoFile).is_file():

						loaded = True
						with open(self.infoFile,'rb') as handle:
							infoPerRowLoaded = pickle.load(handle)
							# copy = infoPerRowLoaded.copy()

						# code.interact("after...", local=dict(globals(), **locals()))

						# Update dict info
						self.infoPerRow.update(infoPerRowLoaded)


					# Update file
					with open(self.tempInfoFile,'wb') as handle:
						pickle.dump(self.infoPerRow,handle)


				# Update estimators
				self.updateEstimators(self.noisyActions[-1],self.observedRewards[-1],**otherArgs)


				# For memory management
				for x in self.xSkills:

					if "multi" in self.env.domain_name:
						x = self.env.spaces.getKey([x,x],0.0)
					

				self.timesPerObservations.append(time.time()-startTime)


				# Save info every X number of rows
				# Iters counter start from 1
				if self.iter%self.every == 0:

					# print(f"RAM infoPerRow: {asizeof.asizeof(self.infoPerRow)*0.000001} MB")

					print("\nCreating checkpoint & getting info for next set of observations...")

					self.checkpointResults()
					self.checkpointEstimators()

					# Delete infoPerRow to load new batch of data
					del self.infoPerRow
					self.infoPerRow = {}

					self.checkpointInfoPerRow()

					# If at last iteration, reloaded all available processed data when saving
					if len(self.infoPerRow) == self.numObservations and not self.checkpointDataAvailable:
						self.infoPerRow = {}


					# RESET DICT
					del self.results
					self.results = {}

					print("Verifying checkpoints...")
					self.verifyCheckpoint()

					# code.interact("...", local=dict(globals(), **locals()))


			# Proceed to process row info
			else:

				print(f"Looking at row {self.iter}/{self.numObservations} | index: {index} | computing...")

				startTime = time.time()


				self.infoPerRow[index] = {"focalActions": np.copy(self.env.spaces.defaultFocalActions),
									"evsPerXskill": {},
									"maxEVPerXskill":{},
									"maxEvTargetActions":[]}

				allTempData = pd.DataFrame([row]*(possibleTargetsLen))

				# Update position of each copy of the row to be that of a given possible action
				allTempData["plate_x"] = np.copy(self.env.spaces.possibleTargetsForModel[:,0])
				allTempData["plate_z"] = np.copy(self.env.spaces.possibleTargetsForModel[:,1])


				# Include original 'row' (df with actual pitch info) to get the probabilities 
				# for the different outcomes as well as the utility - for the actual pitch
				allTempData.loc[len(allTempData.index)+1] = row

				
				'''
				# Run model
				results = nn.functional.softmax(self.env.spaces.model(torch.tensor(allTempData[sys.modules["data"].xswingFeats].values.astype(float),\
						 dtype = torch.float32).to(device)), dim = 1).cpu().detach().numpy()
				
				# Save info
				for i in range(9):
					allTempData[f'o{i}'] = results[:,i]

				'''

				########################################
				# NEW MODEL
				########################################

				batch_x = allTempData[sys.modules["modelTake2"].features].values
				batch_y = allTempData.outcome.values.astype(int)

				# Reshape so that each pa is a separate entry in the batch
				batch_x = batch_x.reshape((len(batch_x), 1,len(sys.modules["modelTake2"].features)))
				batch_y = batch_y.reshape((len(batch_y),1))

				torch_batch_x = torch.tensor(batch_x, dtype = torch.float)
				torch_batch_y = torch.tensor(batch_y, dtype = torch.long)

				ypred = sys.modules["modelTake2"].prediction_func(self.env.spaces.model,torch_batch_x,torch_batch_y)

				allTempData[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = nn.functional.softmax(ypred,dim = 1).detach().cpu().numpy()
				
				# code.interact("after...", local=dict(globals(), **locals()))

				########################################

				
				# Get utilities
				allTempData = sys.modules["utilsBaseball"].getUtility(allTempData)


				# Get updated info for actual pitch (actual pitch + probs + utility)
				row = allTempData.iloc[-1].copy()
				

				# Save info to reuse within exp | utility = observed reward
				self.infoPerRow[index]["observedReward"] = deepcopy(row["utility"])

				# Remove actual pitch from data
				allTempData = allTempData.iloc[:-1,:]
				
				minUtility = np.min(allTempData["utility"].values)

				self.infoPerRow[index]["allTempData"] = deepcopy(allTempData)
				self.infoPerRow[index]["minUtility"] = deepcopy(minUtility)

				
				###############################################################
				# Strike Zone Board
				###############################################################

				fileName = f"{self.agentFolder}-index{index}"
				
				
				# Save info to pickle file
				# allTempData[["plate_x","plate_z","utility"]].to_pickle(f"{self.pickleFolder}{fileName}.pkl",protocol=5)  

				###############################################################
				

				# Populate Dartboard - Can create once since independent of xskill
				# t1 = time.perf_counter()
				Zs = self.env.spaces.setStrikeZoneBoard(allTempData,minUtility)
				# print(f"Total time for setStrikeZoneBoard(): {time.perf_counter()-t1:.4f}")
				
				# code.interact("...", local=dict(globals(), **locals()))


				middle = self.env.spaces.focalActionMiddle
				newFocalActions = []

				# t1 = time.perf_counter()

				for x in self.xSkills:

					# print(f"xskill: {x}")

					if "multi" in self.env.domain_name:

						# Getting symmetric key since pdfsPerXskill info initialized only for normal JTM
						# The ones for the particles will be managed within addObservation()
						x = self.env.spaces.getKey([x,x],0.0)


					# Convolve to produce the EV and aiming spot
					EVs = convolve2d(Zs,self.env.spaces.pdfsPerXskill[x],mode="same",fillvalue=minUtility)
					

					# FOR TESTING
					# EVs = np.ones(Zs.shape)


					maxEV = np.max(EVs)	
					mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
					action = [self.env.spaces.targetsPlateXFeet[mx],self.env.spaces.targetsPlateZFeet[mz]]
					self.infoPerRow[index]["maxEvTargetActions"].append(action)


					# Adding extra focal actions to default set:
					# 	- target for best xskill hyp
					#	- other targets if they are more than 0.16667 feet (or 2 inches) away 
					# 	  from middle target OR last focal target added
					if action not in newFocalActions and "multi" not in self.env.domain_name:
						if (x == np.min(self.xSkills)) or dist(action,middle) >= 0.16667 or dist(action,newFocalActions[-1]) >= 0.16667:
							newFocalActions.append(action)


					########
					# NEED TO ACCOUNT FOR MULTI DIMENSIONS ON SKILL ONCE ADDING TBA
					# Specifically, manage focal actions info
					########


					self.infoPerRow[index]["evsPerXskill"][x] = np.copy(EVs)	
					self.infoPerRow[index]["maxEVPerXskill"][x] = maxEV	


				# print(f"Total time for convolve2d for all xskills: {time.perf_counter()-t1:.4f}")
				# print(f"Before update: {time.time()-startTime}")


				# Update set of focal actions
				if "multi" not in self.env.domain_name:
					self.infoPerRow[index]["focalActions"] = np.concatenate([self.infoPerRow[index]["focalActions"],newFocalActions])			
				

				otherArgs["infoPerRow"] = self.infoPerRow[index]
				otherArgs["allTempData"] = allTempData
				otherArgs["minUtility"] = minUtility
				otherArgs["Zs"] = Zs


				# Save info
				self.noisyActions.append([row["plate_x_feet"],row["plate_z_feet"]])
				self.observedRewards.append(self.infoPerRow[index]["observedReward"])



				############################################################
				# TO DIVIDE UPDATE STEP AMONG WORKERS
				############################################################

				'''
				t1 = time.perf_counter()

				numWorkers = self.numWorkers
				# queue = self.queue
				# readyEvent = self.readyEvent

				pdfsPerXskill = {}
				evsPerXskill = {}


				for e in self.estimators:
					if isinstance(e,Estimators.joint_pfe.QREMethod_Multi_Particles):
						break

				particles = e.particles
				numParticles = len(particles)

				loadSize = int(numParticles/numWorkers)


				wid = 0
				counter = 0

				start = 0
				end = loadSize

				queue = {}

				# Put tasks into the queue
				for tid in range(numWorkers):

					toSend = {"tid":tid,"wid":wid,"seedNum":self.seedNum,
								"particles":particles[start:end,:],
								# "state":None,
								# "allTempData":otherArgs["allTempData"],
								"minUtility":otherArgs["minUtility"],
								"targetsPlateXFeet":self.env.spaces.targetsPlateXFeet,
								"targetsPlateZFeet":self.env.spaces.targetsPlateZFeet,
								"Zs":Zs
								}

					# from sys import getsizeof
					# print(getsizeof(toSend))

					# queue.put(toSend)
					# queue.append(toSend)
					queue[tid] = toSend
					wid += 1

					# Update slicing
					start += loadSize
					end += loadSize

				# Signal workers to start processing
				# readyEvent.set()

				# Wait until all tasks are done
				# queue.join()


				M = self.iter
				num = self.numWorkers

				# To keep track of the workers
				R = []

				# To keep track of the data to send
				D = [-1]*(numWorkers)


				# For each worker
				for n in range(1,numWorkers):

					# Compute dummy data to send?
					d = n*M

					# Send info (data,who gets it, worker number?)
					# r = self.comm.isend(d,dest=n,tag=n)
					r = self.comm.isend(queue[n-1],dest=n,tag=n)
					# serializedData = pickle.dumps(queue[n-1])
					# r = self.comm.isend(serializedData,dest=n,tag=n)

					# Save worker
					R.append(r)
				

				# For each worker
				for r in R:
					# Wait until they finish
					r.wait()
				

				# For each worker
				for n in range(1,numWorkers):

					# Get the info worker n needs to send back
					rq = self.comm.irecv(source=n)

					# Wait until worker returns data and save it into list
					D[n] = rq.wait()


				print('[', M, '] ', numWorkers-1, ' MESSAGES SENT and RECEIVED')
				print('       D is now: ', D)


				print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
				print("len(spaces.pdfsPerXskill): ", len(self.env.spaces.pdfsPerXskill))
				print("len(spaces.evsPerXskill): ", len(self.env.spaces.evsPerXskill))
				# print("len(pdfsPerXskill): ", len(self.pdfsPerXskill))
				# print("len(evsPerXskill): ", len(self.evsPerXskill))

				# self.env.spaces.pdfsPerXskill.update(self.pdfsPerXskill)
				# self.env.spaces.evsPerXskill.update(self.evsPerXskill)

				# print("len(spaces.pdfsPerXskill): ", len(self.env.spaces.pdfsPerXskill))
				# print("len(spaces.evsPerXskill): ", len(self.env.spaces.evsPerXskill))

				print(f"Total time: {time.perf_counter()-t1:.4f}")
				print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

				code.interact("()...", local=dict(globals(), **locals()))
				'''


				############################################################


				# Update estimators
				self.updateEstimators(self.noisyActions[-1],self.observedRewards[-1],**otherArgs)


				self.timesPerObservations.append(time.time()-startTime)


				# Save info every X number of rows
				# Iters counter start from 1
				if self.iter%self.every == 0:

					# print(f"RAM infoPerRow: {asizeof.asizeof(self.infoPerRow)*0.000001} MB")

					print("\nCreating checkpoint...")
					self.checkpointInfoPerRow()
					self.checkpointResults()
					self.checkpointEstimators()
					
					# RESET DICTS
					del self.infoPerRow
					del self.results

					self.infoPerRow = {}
					self.results = {}

					print("Verifying checkpoints...")
					self.verifyCheckpoint()

					# code.interact("...", local=dict(globals(), **locals()))


				# End of row clean
				del row
				del allTempData
				del EVs
				del ypred


				# RESET DICTS
				'''
				for k in self.pdfsPerXskill:
					del self.env.spaces.pdfsPerXskill[k]
					del self.env.spaces.evsPerXskill[k]
					
				self.pdfsPerXskill.clear()
				self.evsPerXskill.clear()
	

				# Reset the event for the next round
				# readyEvent.clear()
				'''

			
			# Call garbage collector
			collect()


			# print(f"Total time row: {self.timesPerObservations[-1]}\n")			
			# code.interact("end row...", local=dict(globals(), **locals()))
			
			self.iter += 1


			# tracker.print_diff()
			
			###############################################################

		
		# Save info for leftover rows (in case saw other rows and condition was not met (didn't reach x # of observations))
		if len(self.infoPerRow) != 0:
			print("\nCreating final checkpoint...")
			
			self.checkpointInfoPerRow()
			self.checkpointResults()
			self.checkpointEstimators()

			print("Verifying checkpoints...")
			self.verifyCheckpoint()

			del self.infoPerRow
			del self.results

			# Call garbage collector
			collect()


		# Mark experiment as done
		self.done = True


		# self.stopWorkers()

		# code.interact("end...", local=dict(globals(), **locals()))


	# @profile
	def checkpointInfoPerRow(self):

		otherTag = ""

		if self.jeeds:
			otherTag += f"_JEEDS_Chunk_{self.seedNum}"

		if self.pfe:
			otherTag += f"_PFE_Chunk_{self.seedNum}"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF_Chunk_{self.seedNum}"

		agentFileName = f"pitcherID{self.pitcherID}-PitchType{self.pitchType}{otherTag}"
		
		# Load next batch of data
		if self.checkpointDataAvailable:

			# Reset dict
			self.infoPerRow = {}
			
			# Load already process data from file
			with open(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl",'rb') as handle:
				tempAgentData = pickle.load(handle)

			# Load next "every" observations
			lookAt = self.agentData.index.values[self.iter:self.iter+self.every]

			for index in lookAt:
				if index in tempAgentData:
					self.infoPerRow[index] = tempAgentData[index]


			if len(self.infoPerRow) == 0:
				print("No more data available to load.")
				self.checkpointDataAvailable = False

			# code.interact("load - checkpointInfoPerRow()...", local=dict(globals(), **locals()))
		

		# Save info to file
		#if not self.checkpointDataAvailable:
		loaded = False

		# If file with info exists already, load info (to avoid overwriting)
		if Path(self.infoFile).is_file():

			loaded = True
			with open(self.infoFile,'rb') as handle:
				infoPerRowLoaded = pickle.load(handle)
				# copy = infoPerRowLoaded.copy()

			# Update dict info
			self.infoPerRow.update(infoPerRowLoaded)


		# Update file
		with open(self.tempInfoFile,'wb') as handle:
			pickle.dump(self.infoPerRow,handle)


		if loaded:
			del infoPerRowLoaded

			# code.interact("save - checkpointInfoPerRow()...", local=dict(globals(), **locals()))


	# @profile
	def checkpointResults(self):

		# Save results and info seen so far to file
		self.getResults()
		# copyResults = self.results.copy()

		self.results["iter"] = self.iter

		
		# Attempt to get info from temp file (if present)
		try:
			with open(self.tempResultsFile,'rb') as handle:
				prevResults = pickle.load(handle)
		# Otherwise start from initial info
		except:
			# Since results file with initial exp info created at the beginning
			with open(self.resultsFile,'rb') as handle:
				prevResults = pickle.load(handle)


		seenKeys = []

		# Will only update things that are changing: following keys & estimators
		for k in self.results:
			# print(k)

			if k in ["estimators_list","agent_name","num_execution_skill_hypotheses","num_rationality_hypotheses"]:
				continue

			if type(self.results[k]) == list:

				if k in prevResults:
					seenKeys.append(k)

					prevResults[k].extend(self.results[k])
					# print("extending...")
					self.results[k] = prevResults[k]


			# elif type(self.results[k]) not in [int,str]:
				# self.results[k]


			# if k in ["observed_rewards","noisy_actions","intended_actions","timesPerObservations"] or "-" in k:
				
				# if "resamplingMethod" in k:
					# continue

				# seenKeys.append(k)
				# seenKeys.append(k)

				'''
				if "allParticles" in k:
					if k in prevResults:
						prevResults[k].extend(self.results[k])
						print("extending...")
						self.results[k] = prevResults[k]
				else:
					if k in prevResults:
						print("concatenating...")
						self.results[k] = np.concatenate((np.array(prevResults[k],dtype="object"),np.array(self.results[k],dtype="object"))).tolist()
					# except:
						# code.interact("checkpointResults...", local=dict(globals(), **locals()))
				'''

		# Copy leftover info from prev results
		for k in prevResults:
			if k not in seenKeys:
				self.results[k] = prevResults[k]


		# Update file
		with open(self.tempResultsFile,'wb') as outfile:
			pickle.dump(self.results,outfile)


		# code.interact("end checkpointResults...", local=dict(globals(), **locals()))
		del prevResults
	

	# @profile
	def checkpointEstimators(self):

		# Reset estimators info (mainly estimates lists)
		for e in self.estimators:
			e.mid_reset()

		self.noisyActions = []
		self.observedRewards = []
		self.timesPerObservations = []

		otherTag = ""

		if self.jeeds:
			otherTag += f"_JEEDS_Chunk_{self.seedNum}"

		if self.pfe:
			otherTag += f"_PFE_Chunk_{self.seedNum}"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF_Chunk_{self.seedNum}"

		# Will save just the status of rest of params
		agentFileName = f"pitcherID{self.pitcherID}-PitchType{self.pitchType}{otherTag}"
		with open(self.tempEstimatorsFile,'wb') as handle:
			pickle.dump(self.estimators,handle)

		# code.interact("end checkpointEstimators...", local=dict(globals(), **locals()))


	# @profile
	def verifyCheckpoint(self):

		otherTag = ""

		if self.jeeds:
			otherTag += f"_JEEDS_Chunk_{self.seedNum}"

		if self.pfe:
			otherTag += f"_PFE_Chunk_{self.seedNum}"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF_Chunk_{self.seedNum}"

		agentFileName = f"pitcherID{self.pitcherID}-PitchType{self.pitchType}{otherTag}"


		rf1 = f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}"
		rf2 = f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}"

		fileName1 = f"OnlineExp_{self.pitcherID}_{self.pitchType}{otherTag}"
		tempFileName1 = f"temp-{fileName1}.results"

		fileName2 = f"estimators-info-{agentFileName}"
		tempFileName2 = f"temp-{fileName2}.pkl"

		fileName3 = f"info-{agentFileName}"
		tempFileName3 = f"temp-{fileName3}.pkl"

		# code.interact("verifyCheckpoint...", local=dict(globals(), **locals()))


		# Verify if checkpoint was successful
		if Path(f"{rf1}{tempFileName1}").is_file() \
			and Path(f"{rf2}{tempFileName2}").is_file() \
			and (Path(f"{rf2}{tempFileName3}").is_file() or Path(f"{rf2}{fileName3}.pkl").is_file()):
			# ^ either temp or full info file
			
			print("Checkpoint successful. Managing files...",end=" ")

			lookingAt = [[fileName1,tempFileName1,".results",rf1],
						[fileName2,tempFileName2,".pkl",rf2],
						[fileName3,tempFileName3,".pkl",rf2]]
			
			for i in range(3):

				# if i == 2 and Path(f"{rf2}{fileName3}.pkl").is_file():
				# 	lookingAt[i][1] = fileName3

				# newName = f"{lookingAt[i][0]}-{counter}.pkl"
				newName = f"{lookingAt[i][0]}{lookingAt[i][2]}"
				# Copy to backup
				try:
					copy(f"{lookingAt[i][3]}{lookingAt[i][0]}{lookingAt[i][2]}",f"{lookingAt[i][3]}backup{os.sep}{newName}")
				except:
					# Initial backup
					copy(f"{lookingAt[i][3]}{lookingAt[i][1]}",f"{lookingAt[i][3]}backup{os.sep}{newName}")
				
				# Only rename if not present already
				os.rename(f"{lookingAt[i][3]}{lookingAt[i][1]}",f"{lookingAt[i][3]}{lookingAt[i][0]}{lookingAt[i][2]}")

			print("Done.\n")


	def getResults(self):

		super().getResults()

		self.results["timesPerObservations"] = self.timesPerObservations

		try:
			self.results["avgTimePerObservations"] = sum(self.timesPerObservations)/len(self.timesPerObservations)
		except:	
			self.results["avgTimePerObservations"] = "N/A"


class HockeyExp(Experiment):

	__slots__ = ["rerun","space","agentData","xSkills",
				"player",
				"timesPerObservations","valid",
				"agentFolder","infoFile","estimatorsInfoFile",
				"plotFolder1","plotFolder2","pickleFolder","infoPerRow",
				"iter","every","pdfsPerXskill","pdfsFile","rerun","done","checkpointDataAvailable",
				"tempResultsFile","tempInfoFile","tempEstimatorsFile","infoFileYear",
				"jeeds","pfe","pfeNeff","comm","ids","type","testingBounded","agentBounded"]


	# @profile
	def __init__(self,args,env,agent,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng):
		

		if args.testingBounded:
			agent, agentBounded = agent
			self.agentBounded = agentBounded
			self.testingBounded = args.testingBounded
		else:
			self.testingBounded = False
			agent = agent[1]


		x = "N/A"

		super().__init__(env.numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng)
		

		self.xSkills = estimatorsObj.allXskills

		# Using symmetric set since will need to init pdfsPerXskill info just for normal JTM
		# The ones for the particles will be managed within addObservation()
		if "multi" in self.env.domain_name:
			# self.xSkills = list(product(list(map(list,self.xSkills)),estimatorsObj.rhos))
			self.xSkills = estimatorsObj.xskillsNormal


		self.every = args.every
		self.rerun = args.rerun

		self.done = False
		self.iter = 1

		self.player = agent[0]
		self.type = agent[1]

		self.agentData = []

		self.jeeds = args.jeeds
		self.pfe = args.pfe
		self.pfeNeff = args.pfeNeff

		self.ids = []


		print("Getting data for agent...")
		# code.interact("()...", local=dict(globals(), **locals()))

		
		# Grab corresponding data for agent (give pitchID & pitchType)
		# Also deletes allData (for memory optimization)
		self.agentData = self.env.spaces.getAgentData(self.resultsFolder,self.player,self.type,args.maxRows)

		self.timesPerObservations = []


		##################################################################################
		# VALID EXPERIMENT OR NOT?
		##################################################################################
		
		if len(self.agentData) == 0:
			print("No data is present for the given agent. Can't proceed with experiment.")
			# Experiment unsuccessful
			self.valid = False

		else:

			# Data is present for agent, can proceed
			print(f"Data was obtained successfully.")

			self.valid = True
			self.numObservations = len(self.agentData)


			otherTag = f"_TypeShot_{self.type}"

			if self.testingBounded:
				otherTag += f"_Agent{self.agentBounded.getName()}"

			if self.jeeds:
				otherTag += f"_JEEDS"

			if self.pfe:
				otherTag += f"_PFE"

			if self.pfeNeff:
				otherTag += f"_PFE_NEFF"


			agentFileName = f"Player_{self.player}{otherTag}"

			self.initialSetup()


			##################################################################################
			# Validate checkpoint
			##################################################################################


			rf = f"Experiments{os.sep}{self.resultsFolder}{os.sep}"

			resultsFileJustName = f"OnlineExp_Player_{self.player}{otherTag}.results"
			estimatorsFileJustName = f"estimators-info-{agentFileName}.pkl"
			infoFileJustName = f"info-{agentFileName}.pkl"

			self.tempResultsFile = f"{rf}results{os.sep}temp-{resultsFileJustName}"
			self.tempEstimatorsFile = f"{rf}info{os.sep}temp-{estimatorsFileJustName}"
			self.tempInfoFile = f"{rf}info{os.sep}temp-{infoFileJustName}"


			validCheckpointFound = True

			# If either one of the temp files is present, 
			# something happened in the middle of a checkpoint
			if Path(self.tempResultsFile).is_file() or Path(self.tempEstimatorsFile).is_file() or Path(self.tempInfoFile).is_file():

				print("Something happened during the previous checkpoint as temp files are present.")

				# Start from previous checkpoint, if any available
				try:
					print("Checking to see if info for a previous checkpoint is available...")
					
					copy(f"{rf}results{os.sep}backup{os.sep}{resultsFileJustName}",f"{rf}results{os.sep}{resultsFileJustName}")
					copy(f"{rf}info{os.sep}backup{os.sep}{estimatorsFileJustName}",f"{rf}info{os.sep}{estimatorsFileJustName}")
					copy(f"{rf}info{os.sep}backup{os.sep}{infoFileJustName}",f"{rf}info{os.sep}{infoFileJustName}")
				
					print("Info from previous checkpoint obtained.")

				# Couldn't find backup info, proceed with full experiment
				except:
					print("Couldn't find necessary backup info, doing with full experiment.")
					
					# In case rerun flag was enabled
					self.rerun = False
					self.checkpointDataAvailable = False
					validCheckpointFound = False


				# Clean up
				if Path(self.tempResultsFile).is_file():
					os.remove(self.tempResultsFile)

				if Path(self.tempEstimatorsFile).is_file():
					os.remove(self.tempEstimatorsFile)

				if Path(self.tempInfoFile).is_file():
					os.remove(self.tempInfoFile)

			##################################################################################


			##################################################################################
			# Check if there's any info available for the estimators
			##################################################################################
				
			# Only if not in rerun mode
			if not self.rerun and validCheckpointFound:
				try:
					print("\nVerifying if estimators info is present in order to load them...",end=" ")
					# Reload info into estimators accordingly (ensure continuality)
					with open(self.estimatorsInfoFile,'rb') as handle:
						self.estimators = pickle.load(handle)

					print("Done.")

				except:
					print("No estimator info available to load.")

			# Otherwise, estimators already set at the beggining 

			##################################################################################


			##################################################################################
			# Check if there's any already data processed to start from
			##################################################################################

			self.checkpointDataAvailable = False

			self.infoFileYear = 99999999

			if validCheckpointFound:
				
				try:
					print("\nChecking if there's any processed data available...")
					
					# Load already process data from file, if any
					with open(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl",'rb') as handle:
						tempAgentData = pickle.load(handle)

					#stats = os.stat(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl")

					#self.infoFileYear = datetime.fromtimestamp(stats.st_birthtime).year
					#print(f"File found from year {self.infoFileYear}.")

					self.checkpointDataAvailable = True

					# How many rows were processed already?
					seen = len(tempAgentData)
					
					# How many rows were processed already?
					available = len(tempAgentData)
					print(f"Loaded available processed info (# rows: {available}).")
					
				except:
					print("No processed data available.")
					self.checkpointDataAvailable = False

			##################################################################################
			

			##################################################################################
			# Experiment setup
			##################################################################################

			# CASE: Reload/Rerun need to set seed accordingly
			if Path(self.resultsFile).is_file() and validCheckpointFound:

				# Load info available
				with open(self.resultsFile,'rb') as handle:
					prevResults = pickle.load(handle)

				self.seedNum = prevResults["seedNum"]
				# print("Setting seed: ",self.seedNum)

				# Set seed for current experiment to that of previous exp
				np.random.seed(self.seedNum)

			else:
				# FOR TESTING
				# self.seedNum = 599433
				# np.random.seed(self.seedNum)
				
				self.saveInitExperimentInfo()


			self.infoPerRow = {}
			
			# Either some or all processed data is available
			if self.checkpointDataAvailable:
	
				# CASE: All data available, start from beginning (rerun or reload mode)
				if len(tempAgentData) == self.numObservations or args.rerun or args.reload:
					# print("Rerun mode.")
					lookAt = {k:self.agentData[k] for i,k in enumerate(self.agentData) if i < self.every}
					self.iter = 1

				# Case partial data is available, continuation exp
				else:

					# To enable saving of newly processed rows since reload mode
					# And not attempt to load new batch of data as it won't be present
					self.checkpointDataAvailable = False
					
					# Update rows to be seen
					lookAt = {k:self.agentData[k] for i,k in enumerate(self.agentData) if i >= seen and i < seen+self.every}
					self.iter = seen+1

					# CASE: reload/continuation exp after checkpoint
					# Subset self.agent data to contain only data left to look at
					self.agentData = lookAt

				
				print("Loading info accordingly...",end=" ")
				
				# Load the first "every" observations
				for index in lookAt:
					# If info for row present
					if index in tempAgentData:
						self.infoPerRow[index] = tempAgentData[index]

				print("Done.")

			
			# code.interact("...", local=dict(globals(), **locals()))
		

			# In case attempting to run exp (not in rerun mode) and all info present
			if len(self.agentData) == 0:
				print("All data has been processed", end=" ")

				# Experiment performed before
				if Path(self.resultsFile).is_file():
					print("and results file is present. Stopping.")
					self.valid = False
					self.done = True

				# Otherwise valid = True & done = False to proceed with full exp
				else:
					print("but results file is not present. Proceed with experiment.")


			# Initial backup
			if self.rerun:

				# File management
				# Move files to their respective backup folder
				# As full results/estimators file from prev exp is present at this point
				
				# Adding counter to filename to remember all prev results (won't overwrite)
				currentFiles = os.listdir(f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}backup{os.sep}")
				fileName =  f"OnlineExp_Player_{agent[0]}"
				# counter = len(currentFiles)+1
				# newName = f"{fileName}-{counter}.results" 
				newName = f"{fileName}.results" 
				
				try:
					# print("Moving results file from previous experiment to backup folder...")
					os.rename(f"{self.resultsFile}", f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}backup{os.sep}{newName}")
					self.saveInitExperimentInfo()
				except:
					pass

				try:
					# print("Moving estimators file from previous experiment to backup folder...")
					newName = f"estimators-info-{agentFileName}.pkl" 
					os.rename(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}estimators-info-{agentFileName}.pkl", f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}backup{os.sep}{newName}")
				except:
					pass

			##################################################################################


		##################################################################################


	# @profile
	def initialSetup(self):

		otherTag = f"_TypeShot_{self.type}"

		if self.testingBounded:
			otherTag += f"_Agent{self.agentBounded.getName()}"

		if self.jeeds:
			otherTag += f"_JEEDS"

		if self.pfe:
			otherTag += f"_PFE"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF"


		self.agentFolder = f"Player_{self.player}{otherTag}"
		
		infoFolder = f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}"
		self.infoFile = f"{infoFolder}info-{self.agentFolder}.pkl"
		self.pdfsFile = f"{infoFolder}pdfsPerXskill-{self.agentFolder}.pkl"
		self.estimatorsInfoFile = f"{infoFolder}estimators-info-{self.agentFolder}.pkl"
		
		
		folders = [infoFolder]

		for folder in folders:
			if not os.path.exists(folder):
				os.mkdir(folder)
		

	def saveInitExperimentInfo(self):

		self.results["iter"] = self.iter
		super().saveInitExperimentInfo()


	def saveInfoPDFs(self):
		print("Saving pdfs info to file...")

		with open(self.pdfsFile,'wb') as outfile:
			pickle.dump(self.env.spaces.pdfsPerXskill,outfile)



	# @profile
	# def run(self,tag,counter,num,comm):
	def run(self,tag,counter):


		otherArgs = {"tag":tag+str(counter),"resultsFolder":self.resultsFolder}


		print("\nGoing through observations...")


		# For each row in the data
		# That is, for a given pitch (observation/state)
		for row in self.agentData:

			otherArgs["i"] = self.iter

			# If rerun mode, processed info already available
			# Just update estimators and create checkpoints
			if row in self.infoPerRow:

				print(f"Looking at row {self.iter}/{self.numObservations} | row: {row} | loaded...")

				startTime = time.time()
 
				otherArgs["infoPerRow"] = self.infoPerRow[row]

				# code.interact("...", local=dict(globals(), **locals()))

				otherArgs["minUtility"] = self.infoPerRow[row]["minUtility"]

				# Populate Board - Can create once since independent of xskill
				Zs = self.infoPerRow[row]["row"]["gridUtilitiesComputed"]
				otherArgs["Zs"] = Zs

				self.ids.append(row)

				
				listedTargetsAngular = self.infoPerRow[row]["row"]["listedTargetsAngular"]
				gridTargetsAngular = self.infoPerRow[row]["row"]["gridTargetsAngular"]

				executedActionAngular = self.infoPerRow[row]["row"]["executedActionAngular"]
				# print("executedActionAngular: ",executedActionAngular)
				

				# For methods
				listedTargets = listedTargetsAngular


				self.env.spaces.possibleTargets = listedTargets
				self.env.spaces.grid = gridTargetsAngular

				# code.interact("...", local=dict(globals(), **locals()))


				dirs,elevations = self.infoPerRow[row]["row"]["dirs"],self.infoPerRow[row]["row"]["elevations"]
				self.env.spaces.delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

				# Assumming both same size (-1 to ofset for index-based 0)
				middle = int(len(dirs)/2) - 1
				mean = [dirs[middle],elevations[middle]]

				self.env.spaces.mean = mean

				otherArgs["domain"] = self.env.domain
				otherArgs["delta"] = self.env.spaces.delta
				otherArgs["mean"] = self.env.spaces.mean


				skip = False

				if self.testingBounded:

					otherArgs["agent"] = self.agentBounded.getName()


					convolutionSpace = {"ts":None,"vs":None,"all_vs":Zs}
					action, nansFlag, evAction = self.agentBounded.get_action(self.rng,self.env.domain,listedTargets,convolutionSpace,returnZn=False)
					# code.interact("...", local=dict(globals(), **locals()))


					# Add noise to action + get respective reward
					noisy_action = self.env.domain.sample_noisy_action(self.rng,None,None,action,self.agentBounded.noiseModel)
					# print("noisy action: ", noisy_action)

					self.noisyActions.append(noisy_action)
					self.observedRewards.append(None)


					if noisy_action[1] <= 0.0:
						print(f"Negative elevation. Setting to 0.")


					# for x in self.xSkills:
					# 	cov = self.env.domain.getCovMatrix([x,x],0.0)
					# 	pdfs = computePDF(x=noisy_action,means=self.env.spaces.possibleTargets,covs=np.array([cov]*len(self.env.spaces.possibleTargets)))
					# 	pdfs /= np.sum(pdfs)

					# 	if np.isnan(pdfs).any():
					# 		print("Skipping state because of NANs in pdfs")
					# 		skip = True
					# 		# code.interact("...", local=dict(globals(), **locals()))
					# 		break


				else:
					otherArgs["agent"] = "-".join(self.agent)

					self.noisyActions.append(executedActionAngular)
					self.observedRewards.append(self.infoPerRow[row]["observedReward"])



				for x in self.xSkills:

					if "multi" in self.env.domain_name:

						covMatrix = self.env.domain.getCovMatrix([x,x],0.0)

						# Getting symmetric key since pdfsPerXskill info initialized only for normal JTM
						# The ones for the particles will be managed within addObservation()
						x = self.env.spaces.getKey([x,x],0.0)


					# NEEDED HERE (FOR EACH XSKILL)
					# SINCE THE SET OF TARGETS (ANGULAR) WILL DIFFER PER ROW
					self.env.spaces.pdfsPerXskill[x] = self.env.domain.getNormalDistribution(self.rng,covMatrix,self.env.spaces.delta,mean,gridTargetsAngular)


					# Compute if haven't seen before OR need to be updated
					if x not in self.infoPerRow[row]["evsPerXskill"]:
						self.evsPerXskill[x] = convolve2d(Zs,self.pdfsPerXskill[x],mode="same",fillvalue=0.0)


				if not skip:

					# Update estimators
					self.updateEstimators(self.noisyActions[-1],self.observedRewards[-1],**otherArgs)


				# For memory management
				for x in self.xSkills:

					if "multi" in self.env.domain_name:
						x = self.env.spaces.getKey([x,x],0.0)
					

				self.timesPerObservations.append(time.time()-startTime)


				# Save info every X number of rows
				# Iters counter start from 1
				if self.iter%self.every == 0:

					# print(f"RAM infoPerRow: {asizeof.asizeof(self.infoPerRow)*0.000001} MB")

					print("\nCreating checkpoint & getting info for next set of observations...")

					self.checkpointResults()
					self.checkpointEstimators()

					# Delete infoPerRow to load new batch of data
					del self.infoPerRow
					self.infoPerRow = {}

					self.checkpointInfoPerRow()

					# If at last iteration, reloaded all available processed data when saving
					if len(self.infoPerRow) == self.numObservations and not self.checkpointDataAvailable:
						self.infoPerRow = {}


					# RESET DICT
					del self.results
					self.results = {}

					print("Verifying checkpoints...")
					self.verifyCheckpoint()

					# code.interact("...", local=dict(globals(), **locals()))


			# Proceed to process row info
			else:

				print(f"Looking at row {self.iter}/{self.numObservations} | row: {row} | computing...")

				startTime = time.time()


				self.infoPerRow[row] = {"focalActions": np.copy(self.env.spaces.defaultFocalActions),
									"evsPerXskill": {},
									"maxEVPerXskill":{},
									"maxEvTargetActions":[]}


				# Save info to reuse within exp | utility = observed reward
				self.infoPerRow[row]["observedReward"] = 0

				self.infoPerRow[row]["row"] = deepcopy(self.agentData[row])

				self.ids.append(row)


				###############################################################
				# Board
				###############################################################

				fileName = f"{self.agentFolder}-row{row}"
				
				
				# dirs = self.agentData[row]["dirs"]
				# elevations = self.agentData[row]["elevations"]

				# Populate Board - Can create once since independent of xskill
				Zs = self.agentData[row]["gridUtilitiesComputed"]

				listedTargetsAngular = self.agentData[row]["listedTargetsAngular"]
				gridTargetsAngular = self.agentData[row]["gridTargetsAngular"]

				executedActionAngular = self.agentData[row]["executedActionAngular"]
				# print("executedActionAngular: ",executedActionAngular)

				# For methods
				listedTargets = listedTargetsAngular

			
				minUtility = np.min(Zs)
				self.infoPerRow[row]["minUtility"] = minUtility


				self.env.spaces.possibleTargets = listedTargets
				self.env.spaces.grid = gridTargetsAngular


				dirs,elevations = self.agentData[row]["dirs"],self.agentData[row]["elevations"]
				self.env.spaces.delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

				# Assumming both same size (-1 to ofset for index-based 0)
				middle = int(len(dirs)/2) - 1
				mean = [dirs[middle],elevations[middle]]

				self.env.spaces.mean = mean


				# code.interact("...", local=dict(globals(), **locals()))

				# middle = self.env.spaces.focalActionMiddle
				# newFocalActions = []

				# t1 = time.perf_counter()

				for x in self.xSkills:

					# print(f"xskill: {x}")


					if "multi" in self.env.domain_name:

						covMatrix = self.env.domain.getCovMatrix([x,x],0.0)

						# Getting symmetric key since pdfsPerXskill info initialized only for normal JTM
						# The ones for the particles will be managed within addObservation()
						x = self.env.spaces.getKey([x,x],0.0)

					
					# NEEDED HERE (FOR EACH XSKILL)
					# SINCE THE SET OF TARGETS (ANGULAR) WILL DIFFER PER ROW
					self.env.spaces.pdfsPerXskill[x] = self.env.domain.getNormalDistribution(self.rng,covMatrix,self.env.spaces.delta,mean,gridTargetsAngular)


					# Convolve to produce the EV and aiming spot
					EVs = convolve2d(Zs,self.env.spaces.pdfsPerXskill[x],mode="same",fillvalue=0.0)
					

					# FOR TESTING
					# EVs = np.ones(Zs.shape)


					maxEV = np.max(EVs)	
					mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
					# action = [self.env.spaces.targetsPlateXFeet[mx],self.env.spaces.targetsPlateZFeet[mz]]
					# self.infoPerRow[row]["maxEvTargetActions"].append(action)


					# Adding extra focal actions to default set:
					# 	- target for best xskill hyp
					#	- other targets if they are more than 0.16667 feet (or 2 inches) away 
					# 	  from middle target OR last focal target added
					# if action not in newFocalActions and "multi" not in self.env.domainName:
					# 	if (x == np.min(self.xSkills)) or dist(action,middle) >= 0.16667 or dist(action,newFocalActions[-1]) >= 0.16667:
					# 		newFocalActions.append(action)


					########
					# NEED TO ACCOUNT FOR MULTI DIMENSIONS ON SKILL ONCE ADDING TBA
					# Specifically, manage focal actions info
					########


					self.infoPerRow[row]["evsPerXskill"][x] = np.copy(EVs)	
					self.infoPerRow[row]["maxEVPerXskill"][x] = maxEV	

				# code.interact("exp types...", local=dict(globals(), **locals()))

				# print(f"Total time for convolve2d for all xskills: {time.perf_counter()-t1:.4f}")
				# print(f"Elapsed time for observation - before update: {time.time()-startTime}")


				# Update set of focal actions
				if "multi" not in self.env.domain_name:
					self.infoPerRow[row]["focalActions"] = np.concatenate([self.infoPerRow[row]["focalActions"],newFocalActions])			
				

				otherArgs["infoPerRow"] = self.infoPerRow[row]
				otherArgs["minUtility"] = minUtility
				otherArgs["Zs"] = Zs

				otherArgs["domain"] = self.env.domain
				otherArgs["delta"] = self.env.spaces.delta
				otherArgs["mean"] = self.env.spaces.mean

				skip = False

				if self.testingBounded:

					otherArgs["agent"] = self.agentBounded.getName()

					convolutionSpace = {"ts":None,"vs":None,"all_vs":Zs}
					action, nansFlag, evAction = self.agentBounded.get_action(self.rng,self.env.domain,listedTargets,convolutionSpace,returnZn=False)

					# code.interact("...", local=dict(globals(), **locals()))

					# Add noise to action + get respective reward
					noisy_action = self.env.domain.sample_noisy_action(self.rng,None,None,action,self.agentBounded.noiseModel)
					# print("noisy action: ", noisy_action)

					self.noisyActions.append(noisy_action)
					self.observedRewards.append(None)


					if noisy_action[1] <= 0.0:
						print(f"Negative elevation. Setting to 0.")


					# for x in self.xSkills:
					# 	cov = self.env.domain.getCovMatrix([x,x],0.0)
					# 	pdfs = computePDF(x=noisy_action,means=self.env.spaces.possibleTargets,covs=np.array([cov]*len(self.env.spaces.possibleTargets)))
					# 	pdfs /= np.sum(pdfs)

					# 	if np.isnan(pdfs).any():
					# 		print(f"Skipping state because of NANs in pdfs. (caused by {x}|)")
					# 		skip = True
					# 		# code.interact("...", local=dict(globals(), **locals()))
					# 		break

				else:

					otherArgs["agent"] = "-".join(self.agent)

					# Save info (shot location = executed action)
					# self.noisyActions.append([self.agentData[row]["shot_location"][0],self.agentData[row]["shot_location"][1]])
					self.noisyActions.append(executedActionAngular)
					self.observedRewards.append(self.infoPerRow[row]["observedReward"])


				if not skip:

					# Update estimators
					self.updateEstimators(self.noisyActions[-1],self.observedRewards[-1],**otherArgs)


				self.timesPerObservations.append(time.time()-startTime)

				# print(f"Time per observation: {self.timesPerObservations[-1]}")
				# print()


				# Save info every X number of rows
				# Iters counter start from 1
				if self.iter%self.every == 0:

					# print(f"RAM infoPerRow: {asizeof.asizeof(self.infoPerRow)*0.000001} MB")

					print("\nCreating checkpoint...")
					self.checkpointInfoPerRow()
					self.checkpointResults()
					self.checkpointEstimators()
					
					# RESET DICTS
					del self.infoPerRow
					del self.results

					self.infoPerRow = {}
					self.results = {}

					print("Verifying checkpoints...")
					self.verifyCheckpoint()

					# code.interact("...", local=dict(globals(), **locals()))


				# End of row clean
				del row
				del EVs


				# RESET DICTS
				# '''
				self.env.spaces.pdfsPerXskill.clear()
				self.env.spaces.evsPerXskill.clear()
					
				# self.pdfsPerXskill.clear()
				# self.evsPerXskill.clear()
	

				# Reset the event for the next round
				# readyEvent.clear()
				# '''

			
			# Call garbage collector
			collect()


			# print(f"Total time row: {self.timesPerObservations[-1]}\n")			
			# code.interact("end row...", local=dict(globals(), **locals()))
			
			self.iter += 1


			# tracker.print_diff()
			
			###############################################################

		
		# Save info for leftover rows (in case saw other rows and condition was not met (didn't reach x # of observations))
		if len(self.infoPerRow) != 0:
			print("\nCreating final checkpoint...")
			
			self.checkpointInfoPerRow()
			self.checkpointResults()
			self.checkpointEstimators()

			print("Verifying checkpoints...")
			self.verifyCheckpoint()

			del self.infoPerRow
			del self.results

			# Call garbage collector
			collect()


		# Mark experiment as done
		self.done = True


		# self.stopWorkers()

		# code.interact("end...", local=dict(globals(), **locals()))


	# @profile
	def checkpointInfoPerRow(self):

		otherTag = f"_TypeShot_{self.type}"

		if self.testingBounded:
			otherTag += f"_Agent{self.agentBounded.getName()}"

		if self.jeeds:
			otherTag += f"_JEEDS"

		if self.pfe:
			otherTag += f"_PFE"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF"

		agentFileName = f"Player_{self.player}{otherTag}"
		
		# Load next batch of data
		if self.checkpointDataAvailable:

			# Reset dict
			self.infoPerRow = {}
			
			# Load already process data from file
			with open(f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}info-{agentFileName}.pkl",'rb') as handle:
				tempAgentData = pickle.load(handle)

			# Load next "every" observations
			lookAt = {k:self.agentData[k] for i,k in enumerate(self.agentData) if i >= self.iter and i < self.iter+self.every}

			for row in lookAt:
				if row in tempAgentData:
					self.infoPerRow[row] = tempAgentData[row]


			if len(self.infoPerRow) == 0:
				print("No more data available to load.")
				self.checkpointDataAvailable = False

			# code.interact("load - checkpointInfoPerRow()...", local=dict(globals(), **locals()))
		

		# Save info to file
		#if not self.checkpointDataAvailable:
		loaded = False

		# If file with info exists already, load info (to avoid overwriting)
		if Path(self.infoFile).is_file():

			loaded = True
			with open(self.infoFile,'rb') as handle:
				infoPerRowLoaded = pickle.load(handle)
				# copy = infoPerRowLoaded.copy()

			# Update dict info
			self.infoPerRow.update(infoPerRowLoaded)


		# Update file
		with open(self.tempInfoFile,'wb') as handle:
			pickle.dump(self.infoPerRow,handle)


		if loaded:
			del infoPerRowLoaded

			# code.interact("save - checkpointInfoPerRow()...", local=dict(globals(), **locals()))


	# @profile
	def checkpointResults(self):

		# Save results and info seen so far to file
		self.getResults()
		# copyResults = self.results.copy()

		self.results["iter"] = self.iter

		# code.interact("start checkpointResults...", local=dict(globals(), **locals()))

		
		if self.rerun:

			if self.iter == 0:
				prevResults = {}
			else:
				if Path(self.resultsFile).is_file():
					# Since results file with initial exp info created at the beginning
					with open(self.resultsFile,'rb') as handle:
						prevResults = pickle.load(handle)
				else:
					prevResults = {}

		else:
			# Attempt to get info from temp file (if present)
			try:
				with open(self.tempResultsFile,'rb') as handle:
					prevResults = pickle.load(handle)
			# Otherwise start from initial info
			except:
				if Path(self.resultsFile).is_file():
					# Since results file with initial exp info created at the beginning
					with open(self.resultsFile,'rb') as handle:
						prevResults = pickle.load(handle)
				else:
					prevResults = {}


		seenKeys = []

		# Will only update things that are changing: following keys & estimators
		for k in self.results:
			# print(k)

			if k in ["estimators_list","agent_name","num_execution_skill_hypotheses","num_rationality_hypotheses"] and not self.rerun:
				continue

			if type(self.results[k]) == list:

				if k in prevResults:
					seenKeys.append(k)

					prevResults[k].extend(self.results[k])
					# print("extending...")
					self.results[k] = prevResults[k]


			# elif type(self.results[k]) not in [int,str]:
				# self.results[k]


			# if k in ["observed_rewards","noisy_actions","intended_actions","timesPerObservations"] or "-" in k:
				
				# if "resamplingMethod" in k:
					# continue

				# seenKeys.append(k)
				# seenKeys.append(k)

				'''
				if "allParticles" in k:
					if k in prevResults:
						prevResults[k].extend(self.results[k])
						print("extending...")
						self.results[k] = prevResults[k]
				else:
					if k in prevResults:
						print("concatenating...")
						self.results[k] = np.concatenate((np.array(prevResults[k],dtype="object"),np.array(self.results[k],dtype="object"))).tolist()
					# except:
						# code.interact("checkpointResults...", local=dict(globals(), **locals()))
				'''

		# Copy leftover info from prev results
		for k in prevResults:
			if k not in seenKeys:
				self.results[k] = prevResults[k]


		# Update file
		with open(self.tempResultsFile,'wb') as outfile:
			pickle.dump(self.results,outfile)


		# code.interact("end checkpointResults...", local=dict(globals(), **locals()))
		del prevResults


		# Reset list keeping track of IDs
		print(self.ids)
		self.ids = []
	

	# @profile
	def checkpointEstimators(self):

		# Reset estimators info (mainly estimates lists)
		for e in self.estimators:
			e.mid_reset()

		self.noisyActions = []
		self.observedRewards = []
		self.timesPerObservations = []

		otherTag = f"_TypeShot_{self.type}"

		if self.testingBounded:
			otherTag += f"_Agent{self.agentBounded.getName()}"

		if self.jeeds:
			otherTag += f"_JEEDS"

		if self.pfe:
			otherTag += f"_PFE"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF"

		# Will save just the status of rest of params
		agentFileName = f"Player_{self.player}{otherTag}"

		with open(self.tempEstimatorsFile,'wb') as handle:
			pickle.dump(self.estimators,handle)

		# code.interact("end checkpointEstimators...", local=dict(globals(), **locals()))


	# @profile
	def verifyCheckpoint(self):

		otherTag = f"_TypeShot_{self.type}"

		if self.testingBounded:
			otherTag += f"_Agent{self.agentBounded.getName()}"

		if self.jeeds:
			otherTag += f"_JEEDS"

		if self.pfe:
			otherTag += f"_PFE"

		if self.pfeNeff:
			otherTag += f"_PFE_NEFF"

		agentFileName = f"Player_{self.player}{otherTag}"


		rf1 = f"Experiments{os.sep}{self.resultsFolder}{os.sep}results{os.sep}"
		rf2 = f"Experiments{os.sep}{self.resultsFolder}{os.sep}info{os.sep}"

		fileName1 = f"OnlineExp_Player_{self.player}{otherTag}"
		agentFileName = f"Player_{self.player}{otherTag}"
		tempFileName1 = f"temp-{fileName1}.results"

		fileName2 = f"estimators-info-{agentFileName}"
		tempFileName2 = f"temp-{fileName2}.pkl"

		fileName3 = f"info-{agentFileName}"
		tempFileName3 = f"temp-{fileName3}.pkl"

		# code.interact("verifyCheckpoint...", local=dict(globals(), **locals()))


		# Verify if checkpoint was successful
		if Path(f"{rf1}{tempFileName1}").is_file() \
			and Path(f"{rf2}{tempFileName2}").is_file() \
			and (Path(f"{rf2}{tempFileName3}").is_file() or Path(f"{rf2}{fileName3}.pkl").is_file()):
			# ^ either temp or full info file
			
			print("Checkpoint successful. Managing files...",end=" ")

			lookingAt = [[fileName1,tempFileName1,".results",rf1],
						[fileName2,tempFileName2,".pkl",rf2],
						[fileName3,tempFileName3,".pkl",rf2]]
			
			for i in range(3):

				# if i == 2 and Path(f"{rf2}{fileName3}.pkl").is_file():
				# 	lookingAt[i][1] = fileName3

				# newName = f"{lookingAt[i][0]}-{counter}.pkl"
				newName = f"{lookingAt[i][0]}{lookingAt[i][2]}"
				# Copy to backup
				try:
					copy(f"{lookingAt[i][3]}{lookingAt[i][0]}{lookingAt[i][2]}",f"{lookingAt[i][3]}backup{os.sep}{newName}")
				except Exception as e:
					# print(e)
					# Initial backup
					copy(f"{lookingAt[i][3]}{lookingAt[i][1]}",f"{lookingAt[i][3]}backup{os.sep}{newName}")
				
				# Only rename if not present already
				os.rename(f"{lookingAt[i][3]}{lookingAt[i][1]}",f"{lookingAt[i][3]}{lookingAt[i][0]}{lookingAt[i][2]}")

			print("Done.\n")


	def getResults(self):

		super().getResults()

		self.results["timesPerObservations"] = self.timesPerObservations
		self.results["ids"] = self.ids

		try:
			self.results["avgTimePerObservations"] = sum(self.timesPerObservations)/len(self.timesPerObservations)
		except:	
			self.results["avgTimePerObservations"] = "N/A"


class SoccerExp(Experiment):

	__slots__ = ["rerun","playerID","space","dataActions","gamestates","features","labels","xSkills",
				"timesPerObservations","valid","done","iter"]


	# @profile
	def __init__(self,args,env,agent,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng):
		
		x = "N/A"

		self.xSkills = estimatorsObj.allXskills

		super().__init__(env.numObservations,env,agent,x,estimatorsObj,list_of_subset_of_estimators,resultsFolder,resultsFile,indexOR,seedNum,rng)
		
		self.done = False
		self.iter = 1

		self.playerID = agent

	
		print("Getting data for agent...")
		
		# Grab corresponding data for agent (give pitchID & pitchType)
		# Also deletes allData (for memory optimization)
		info = self.env.spaces.getAgentData(self.playerID,args.maxRows)
		self.timesPerObservations = []


		##################################################################################
		# VALID EXPERIMENT OR NOT?
		##################################################################################
		
		# if len(info) != 3:
		if len(info) != 4:
			print("Unable to load data for the given agent. Can't proceed with experiment.")
			# Experiment unsuccessful
			self.valid = False

		else:

			# self.dataActions,self.features,self.labels = info[0],info[1],info[2]
			self.dataActions,self.features,self.labels,self.gamestates = info[0],info[1],info[2],info[3]

			# Data is present for agent, can proceed
			print(f"Data was obtained successfully.")

			self.valid = True
			self.numObservations = len(self.dataActions)
			agentFileName = f"playerID{self.playerID}"

			self.saveInitExperimentInfo()

			# code.interact("...", local=dict(globals(), **locals()))
		
		##################################################################################

		if not os.path.exists(f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}"):
			os.mkdir(f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}")

		if not os.path.exists(f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}Player{self.playerID}{os.sep}"):
			os.mkdir(f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}Player{self.playerID}{os.sep}")



	def run(self,tag,counter):
		
		# Features to use for value model (vaep_xg_360)
		subsetFeatures = [fs.actiontype_onehot,fs.result_onehot,fs.actiontype_result_onehot,
						fs.bodypart_onehot,fs.time,fs.startlocation,fs.endlocation,fs.startpolar,
						fs.endpolar,fs.movement,fs.team,fs.time_delta,fs.space_delta,fs.goalscore,
						fs.packing_rate,fs.defenders_in_3m_radius,fs.defenders_in_5m_radius]

		# Labels to use for value model (vaep_xg_360)
		subsetLabels = [ls.scores,ls.scores_xg,ls.concedes,ls.concedes_xg]


		xb = 52 #104 #26 #52 #17 #26
		yb = 34 #68 #17 #34 #11 #17


		tempFileName = f"Experiments{os.sep}{self.resultsFolder}{os.sep}plots{os.sep}Player{self.playerID}{os.sep}"

		print("\nGoing through observations...")

		dataset = PassesDataset(xfns=subsetFeatures,yfns=subsetLabels,load_cached=False)


		with open(f"{tempFileName}times-xb{xb}-yb{yb}.txt","w") as timesFile:

			indexGS = 0

			# For each row in the data
			# That is, for a given (observation/state)
			for index, row in self.dataActions.iterrows():

				print(f"Looking at row {self.iter}/{self.numObservations} | index: {index} | computing...")

				self.iter += 1


				tempFS = self.features.iloc[index].to_frame().T

				gid = tempFS.index[0][0]
				actionID = tempFS.index[0][1]

				dataset.setFeatures(tempFS)
				dataset.setLabels(self.labels.iloc[index].to_frame().T)

				'''
				tempRow = self.gamestates.iloc[index].to_frame().T
				tempRow.start_x = float(tempRow.start_x)
				tempRow.start_y = float(tempRow.start_y)
				tempRow["name"] = row.name
				tempRow = tempRow.set_index(["game_id","name"])
				'''

				# Find corresponding gs for current action
				# Since they are store in a per row manner
				# gamestate = list of dataframes with length = nb_prev_actions
				temp = []
				for each in self.gamestates[indexGS]:
					tempInfo = each.loc[(gid,actionID)].to_frame().T

					cols = ['period_id', 'time_seconds', 'team_id', 'player_id',\
							'start_x', 'start_y', 'end_x', 'end_y', 'bodypart_id',\
							'type_id', 'result_id', 'possession_team_id', \
							'under_pressure', 'in_visible_area_360']

					# To account for error with np.sqrt() when given np.float64 as type (for example)
					tempInfo[cols] = tempInfo[cols].apply(pd.to_numeric)

					temp.append(tempInfo)

				indexGS += 1

				
				modelTimeStart = time.time()
				p_value_surfaces = self.env.domain.predict_surface(self.env.spaces.model,dataset,temp,game_id=row.game_id, db=self.env.spaces.db, x_bins=xb, y_bins=yb)
				modelTime = time.time() - modelTimeStart


				# code.interact("run()...", local=dict(globals(), **locals()))


				fig, ax = plt.subplots()

				ax, im = plot_action(row, surface=p_value_surfaces[f"action_{actionID}"], surface_kwargs={"interpolation": "bilinear"})
				cbar = plt.colorbar(im)


				plt.savefig(f"{tempFileName}player{self.playerID}-gameID{gid}-xb-{xb}yb-{yb}-surface{actionID}.png", bbox_inches='tight')
				plt.clf()


				# Save times to file on exps folder
				print(modelTime,file=timesFile)

				timesFile.flush()

				continue



				startTime = time.time()


				self.infoPerRow[index] = {"focalActions": np.copy(self.env.spaces.defaultFocalActions),
									"evsPerXskill": {},
									"maxEVPerXskill":{},
									"maxEvTargetActions":[]}

				allTempData = pd.DataFrame([row]*(possibleTargetsLen))

				# Update position of each copy of the row to be that of a given possible action
				allTempData["plate_x"] = np.copy(self.env.spaces.possibleTargetsForModel[:,0])
				allTempData["plate_z"] = np.copy(self.env.spaces.possibleTargetsForModel[:,1])


				# Include original 'row' (df with actual pitch info) to get the probabilities 
				# for the different outcomes as well as the utility - for the actual pitch
				allTempData.loc[len(allTempData.index)+1] = row

			
				########################################
				# MODEL
				########################################

				# code.interact("after...", local=dict(globals(), **locals()))

				########################################

				
				# Get utilities


				# Save info to reuse within exp | utility = observed reward

				
				###############################################################
				# Pitch Board
				###############################################################

				fileName = f"{self.agentFolder}-index{index}"
				
				###############################################################
				

				# Populate Dartboard - Can create once since independent of xskill
				Zs = self.env.spaces.setStrikeZoneBoard(allTempData,minUtility)


				middle = self.env.spaces.focalActionMiddle
				newFocalActions = []


				for x in self.xSkills:

					if type(x) == tuple:
						x = self.env.spaces.getKey(x[0],x[1])

					# Convolve to produce the EV and aiming spot
					EVs = convolve2d(Zs,self.env.spaces.pdfsPerXskill[x],mode="same",fillvalue=minUtility)
					
					# FOR TESTING
					# EVs = np.ones(Zs.shape)

					maxEV = np.max(EVs)	
					mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
					action = [self.env.spaces.targetsPlateXFeet[mx],self.env.spaces.targetsPlateZFeet[mz]]
					self.infoPerRow[index]["maxEvTargetActions"].append(action)


					# Adding extra focal actions to default set:
					# 	- target for best xskill hyp
					#	- other targets if they are more than 0.16667 feet (or 2 inches) away 
					# 	  from middle target OR last focal target added
					if action not in newFocalActions:
						if (x == np.min(self.xSkills)) or dist(action,middle) >= 0.16667 or dist(action,newFocalActions[-1]) >= 0.16667:
							newFocalActions.append(action)
					

					self.infoPerRow[index]["evsPerXskill"][x] = np.copy(EVs)	
					self.infoPerRow[index]["maxEVPerXskill"][x] = maxEV	


				# print(f"Before update: {time.time()-startTime}")


				# Update set of focal actions
				self.infoPerRow[index]["focalActions"] = np.concatenate([self.infoPerRow[index]["focalActions"],newFocalActions])			
				
				otherArgs["infoPerRow"] = self.infoPerRow[index]
				otherArgs["allTempData"] = allTempData
				otherArgs["minUtility"] = minUtility


				# Save info
				self.noisyActions.append([row["plate_x_feet"],row["plate_z_feet"]])
				self.observedRewards.append(self.infoPerRow[index]["observedReward"])

				# Update estimators
				self.updateEstimators(self.noisyActions[-1],self.observedRewards[-1],**otherArgs)


				self.timesPerObservations.append(time.time()-startTime)


		# Mark experiment as done
		self.done = True


	def getResults(self):
		pass



def worker(wid,taskQueue,readyEvent,pdfsPerXskill,evsPerXskill):
	
	while True:

		# Wait for the signal to start processing tasks
		readyEvent.wait() 

		task = taskQueue.get()

		
		# CASE: Done
		if task is None:
			taskQueue.task_done()
			break

	
		rng = np.random.default_rng(task["seedNum"])
		task["rng"] = rng


		# CASE: Task assigned to current worker
		print(f"Worker {wid} is doing task {task['particles']}")

		for each in task["particles"]:
			workUpdate(task,each,pdfsPerXskill,evsPerXskill)

		print(f"Worker {wid} finished task {task['particles']}")
	
		taskQueue.task_done()

