import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)

# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
# To be used later for finding and properly loading the domains 
# Will look something like: "/home/archibald/skill-estimation/Environments/"
mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.sep
spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.sep)[0] + "setupSpaces.py").load_module()

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
import matplotlib
import matplotlib.gridspec as gridspec

# import plotly as py
# import plotly.tools as tls

import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
chart_studio.tools.set_credentials_file(username='din7@msstate.edu', api_key='1D3t7xYrB9hlKVrnqdsI')


import math
import numpy as np
import pandas as pd
import code
from copy import deepcopy

from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from scipy.signal import fftconvolve

from scipy import stats
from scipy.optimize import curve_fit

import pickle
import argparse

from utilsDarts import *

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors


from time import time

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.ticker import MaxNLocator

import matplotlib.gridspec as gridspec

from matplotlib.ticker import FormatStrFormatter


def plotRationalityParamsVsSkillEstimatePerMethod(processedRFsAgentNames,numHypsX,numHypsP,methods,resultsFolder,agentType):

	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-rationalityParamsVSxSkillEstimatePerMethod")
	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-rationalityParamsVSpSkillEstimatePerMethod")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	resultsDict = {}

	##################################### FOR XSKILLS #####################################
	# for each method
	for method in methods:

		if method == "tn":
			continue
		else:

			params = {}
			estimates = {}

			if "pSkills" not in method:

				# for each one of the different bounded agents with a given xskill
				for eachAgent in processedRFsAgentNames:

					# only look at specified type of agent
					if agentType in eachAgent:

						aType, x, p = getParamsFromAgentName(eachAgent)

						# Load processed info		
						resultsDict[eachAgent] = loadProcessedInfo(prdFile,eachAgent)


						if x not in params.keys():
							params[x] = []
							estimates[x] = []

						params[x].append(p)
						
						if "BM" in method:
							tempM, beta, tt = getInfoBM(method)
							estimates[x].append(resultsDict[eachAgent]["estimates"][tt][tempM][givenBeta][-1])
						else:						
							estimates[x].append(resultsDict[eachAgent]["estimates"][method][-1])

						del resultsDict[eachAgent]


				# for each one of the different xskills for bounded agents
				for each in params.keys():

					fig = plt.figure()
					ax = plt.subplot(111)

					# "sort" info
					paramsSorted, estimatesSorted = sortTwoLists(params[each],estimates[each])

					plt.plot(paramsSorted,estimatesSorted, marker = ".", label = method)

					plt.xlabel(r'\textbf{'+xString+'}')
					plt.ylabel(r'\textbf{XSkill Estimates}')
					plt.margins(0.05)
					plt.title(xString +" vs. Estimate | "+agentType+" | XSkill: " + str(each) + " | Method: " + method )

					# Shrink current axis by 20%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

					# Put a legend to the right of the current axis
					ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

					plt.savefig(resultsFolder + os.sep + agentType + os.sep + agentType + "-rationalityParamsVSxSkillEstimatePerMethod" + os.sep + "results-"+xString+"VSEstimate-Method-" + method + "-XSkill-" + str(each) + ".png", bbox_inches = 'tight')
					plt.clf()
					plt.close("all")
	#######################################################################################

	##################################### FOR PSKILLS #####################################
	# for each method
	for method in methods:

		if method == "tn":
			continue
		else:

			params = {}
			estimates = {}

			if "pSkills" in method:

				# for each one of the different bounded agents with a given xskill
				for eachAgent in processedRFsAgentNames:

					# only look at specified type of agent
					if agentType in eachAgent:

						aType, x, p = getParamsFromAgentName(eachAgent)

						# Load processed info		
						resultsDict[eachAgent] = loadProcessedInfo(prdFile,eachAgent)


						if x not in params.keys():
							params[x] = []
							estimates[x] = []

						params[x].append(p)
						estimates[x].append(resultsDict[eachAgent]["estimates"][method][-1])

						del resultsDict[eachAgent]


				# for each one of the different xskills for bounded agents
				for each in params.keys():

					fig = plt.figure()
					ax = plt.subplot(111)

					# "sort" info
					sortedParams, estimatesSorted = sortTwoLists(params[each],estimates[each])

					plt.plot(sortedParams,estimatesSorted, marker = ".", label = method)

					plt.xlabel(r'\textbf{'+xString+'}')
					plt.ylabel(r'\textbf{PSkill Estimates}')
					plt.margins(0.05)
					plt.title(xString + " vs. Estimate | BoundedAgent | XSkill: " + str(each) + " | Method: " + method )

					# Shrink current axis by 20%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

					# Put a legend to the right of the current axis
					ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

					plt.savefig(resultsFolder + os.sep + agentType + os.sep + agentType + "-rationalityParamsVSpSkillEstimatePerMethod" + os.sep + "results-LambdaVSEstimate-Method-" + method + "-XSkill-" + str(each) + ".png", bbox_inches = 'tight')
					plt.clf()
					plt.close("all")

	#######################################################################################

def plotMSEAllRationalityParamsPerMethods(processedRFsAgentNames,methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentType):

	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-mseAllRationalityParamsPerXSkillMethodsBoundedAgent")
	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-mseAllRationalityParamsPerPSkillMethodsBoundedAgent")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	numMethods = len(methods)

	resultsDict = {}

	for a in processedRFsAgentNames:

		# only consider specified type of agent
		if agentType not in a:
			continue
		else:

			aType, x, p = getParamsFromAgentName(a)

			# Load processed info		
			resultsDict[a] = loadProcessedInfo(prdFile,a)


			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = { "xSkills": {}, "numExps": 0 }

			# if we haven't seen this xskill for this lambda yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"mseMethods": {}}

			params[p]["numExps"] = resultsDict[a]["num_exps"]

			# update ifo
			for m in methods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][tt][tempM][givenBeta]
				else:
					params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][m]

			del resultsDict[a]


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different params - ok for now
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


	##################################### FOR XSKILLS #####################################
	ci = 0
	c = list(colors.keys())[ci]

	for m in range(len(methods)):
		
		if "pSkills" not in methods[m]:
					
			for eachXSkill in sortedXSkills:
				
				fig = plt.figure()
				ax = plt.subplot(111)

				
				for eachParam in sortedParams:

					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o", label = methods[m])
					except:
						continue

				plt.xlabel(r'\textbf{'+xString+'}',fontsize = 18)
				plt.ylabel(r'\textbf{Mean Squared Error}', fontsize = 18)
				plt.margins(0.05)
				plt.title("Method: " + str(methods[m]) + " | xSkill: " + str(eachXSkill) + " | Experiments: " + str(params[p]["numExps"]))

				# Shrink current axis by 20%
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

				elements = [Line2D([0],[0], color = colors[c], marker = "o", label = methods[m])]

				# Put a legend to the right of the current axis
				ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

				plt.savefig(resultsFolder +  os.sep + agentType + os.sep + agentType + "-mseAllRationalityParamsPerXSkillMethodsBoundedAgent" + os.sep + "results-mseVs"+xString+"-Method-"+str(methods[m])+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close("all")
			
			ci += 1
			c = list(colors.keys())[ci]

	######################################################################################


	##################################### FOR PSKILLS #####################################
	ci = 0
	c = list(colors.keys())[ci]

	for m in range(len(methods)):
		
		if "pSkills" in methods[m]:
					
			for eachXSkill in sortedXSkills:
				
				fig = plt.figure()
				ax = plt.subplot(111)

				
				for eachParam in sortedParams:

					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o", label = methods[m])
					except:
						continue

				plt.xlabel(r'\textbf{'+xString+'}',fontsize = 18)
				plt.ylabel(r'\textbf{Mean squared error}', fontsize = 18)
				plt.margins(0.05)
				plt.title("Method: " + str(methods[m]) + " | xSkill: " + str(eachXSkill) + " | Experiments: " + str(params[p]["numExps"]))

				# Shrink current axis by 20%
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

				elements = [Line2D([0],[0], color = colors[c], marker = "o", label = methods[m])]

				# Put a legend to the right of the current axis
				ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
 
				plt.savefig(resultsFolder +os.sep + agentType + os.sep + agentType + "-mseAllRationalityParamsPerPSkillMethodsBoundedAgent" + os.sep + "results-mseVs"+xString+"-Method-"+str(methods[m])+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close("all")
			
			ci += 1
			c = list(colors.keys())[ci]
			
	######################################################################################

def plotMSEAllRationalityParamsAllMethods(processedRFsAgentNames,methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentType):

	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-mseAllRationalityParamsAllXSkillMethods")
	makeFolder2(resultsFolder + os.sep + agentType, agentType + "-mseAllRationalityParamsAllPSkillMethods")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	numMethods = len(methods)

	resultsDict = {}

	for a in processedRFsAgentNames:

		# only consider specified type of agent
		if agentType not in a:
			continue
		else:
		
			aType, x, p = getParamsFromAgentName(a)

			# Load processed info		
			resultsDict[a] = loadProcessedInfo(prdFile,a)


			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = { "xSkills": {}, "numExps": 0 }

			# if we haven't seen this xskill for this param yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"mseMethods": {}}

			params[p]["numExps"] = resultsDict[a]["num_exps"]

			# update ifo
			for m in methods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][tt][tempM][givenBeta]
				else:
					params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][m]

			del resultsDict[a]


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different params - ok for now
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	# code.interact(local = locals())

	##################################### FOR XSKILLS #####################################
	for eachXSkill in sortedXSkills:
		
		fig = plt.figure()
		ax = plt.subplot(111)
		
		for eachParam in sortedParams:

			ci = 0

			# Get color
			c = list(colors.keys())[ci]

			for m in range(len(methods)):
				
				if "pSkills" not in methods[m]:
					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o")
						ci += 1
						c = list(colors.keys())[ci]
					except:
						continue

				else:
					continue
					# code.interact("...", local=dict(globals(), **locals()))


		plt.xlabel(r'\textbf{' + xString + '}', fontsize=18)
		plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)
		plt.margins(0.05)
		plt.title("xSkill: " + str(eachXSkill) + " | Experiments: " + str(params[p]["numExps"]))

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		elements = []
		ci = 0

		for m in range(len(methods)):
			if "pSkills" not in methods[m]:
				# Get color
				c = list(colors.keys())[ci]
				elements.append(Line2D([0],[0], color = colors[c], marker = "o", label = methods[m]))
				ci += 1

		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc = 'center left', bbox_to_anchor = (1, 0.5))

		plt.savefig(resultsFolder + os.sep + agentType + os.sep + agentType + "-mseAllRationalityParamsAllXSkillMethods" + os.sep + "results-mseVs"+xString+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
		plt.clf()
		plt.close("all")
	#######################################################################################

	##################################### FOR PSKILLS #####################################
	for eachXSkill in sortedXSkills:
		
		fig = plt.figure()
		ax = plt.subplot(111)
		
		for eachParam in sortedParams:
			ci = 0
			# Get color
			c = list(colors.keys())[ci]

			for m in range(len(methods)):
				
				if "pSkills" in methods[m]:
					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o")
						ci += 1
						# Get color
						c = list(colors.keys())[ci]
					except:
						continue
				else:
					continue

		plt.xlabel(r'\textbf{' + xString + '}',fontsize = 18)
		plt.ylabel(r'\textbf{Mean Squared Error}', fontsize = 18)
		plt.margins(0.05)
		plt.title("xSkill: " + str(eachXSkill) + " | Experiments: " + str(params[p]["numExps"]))

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		elements = []
		ci = 0
		c = list(colors.keys())[ci]

		for m in range(len(methods)):
			if "pSkills" in methods[m]:
				elements.append(Line2D([0],[0], color = colors[c], marker = "o", label = methods[m]))
				ci += 1
				c = list(colors.keys())[ci]

		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc = 'center left', bbox_to_anchor = (1, 0.5))

		plt.savefig(resultsFolder + os.sep + agentType + os.sep + agentType + "-mseAllRationalityParamsAllPSkillMethods" + os.sep + "results-mseVs"+xString+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
		plt.clf()
		plt.close("all")
	#######################################################################################

def plotEVintendedVSagentType(processedRFsAgentNames,numHypsX,numHypsP,resultsFolder,domain):

	makeFolder(resultsFolder, "EVsIntendedPerAgentType")

	rewardsPerAgentType = {}

	resultsDict = {}

	for a in processedRFsAgentNames:

		agentType, x, p = getParamsFromAgentName(a)


		if agentType == "Target":
			continue
		if agentType == "Random":
			continue

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)	


		if agentType not in rewardsPerAgentType.keys():
			rewardsPerAgentType[agentType] = {"x": [], "p": [], "evIntended": [], "trueReward": []}

		rewardsPerAgentType[agentType]["x"].append(x)
		rewardsPerAgentType[agentType]["p"].append(p)
		rewardsPerAgentType[agentType]["evIntended"].append(resultsDict[a]["mean_value_intendedAction"])
		rewardsPerAgentType[agentType]["trueReward"].append(resultsDict[a]["mean_true_reward"])

		del resultsDict[a]


	for at in rewardsPerAgentType.keys():

		fig = plt.figure()

		fig.suptitle("EVs intended | Domain: " + domain + " | Agent: " + at)

		# rows, cols, pos
		ax = fig.add_subplot(2, 1, 1)
		ax2 = fig.add_subplot(2, 1, 2)

		#plt.plot(eachParam,params[eachParam]["observed"], color = "blue", marker = ".")
		#plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*")

		s = ax.scatter(rewardsPerAgentType[at]["x"],rewardsPerAgentType[at]["p"], c = rewardsPerAgentType[at]["evIntended"])
		cbar = fig.colorbar(s, ax = ax)
		cbar.set_label("EV Intended", labelpad=+1)

		s2 = ax2.scatter(rewardsPerAgentType[at]["x"],rewardsPerAgentType[at]["p"], c = rewardsPerAgentType[at]["trueReward"])
		cbar = fig.colorbar(s2, ax = ax2)
		cbar.set_label("True Reward", labelpad=+1)

		
		ax.set_xlabel("xSkills")
		ax.set_ylabel("pSkills")
		
		ax2.set_xlabel("xSkills")
		ax2.set_ylabel("pSkills")
		
		plt.margins(0.05)

		plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "EVsIntendedPerAgentType" + os.sep + "evsIntended-domain-" + domain + "-agent" + at + ".png", bbox_inches = 'tight')
		plt.clf()
		plt.close("all")

def plotRewardsVSagentType(processedRFsAgentNames,numHypsX,numHypsP,resultsFolder,domain):

	makeFolder(resultsFolder, "rewardsPerAgentType")

	rewardsPerAgentType = {}

	resultsDict = {}

	for a in processedRFsAgentNames:

		agentType, x, p = getParamsFromAgentName(a)

		if agentType == "Target":
			continue
		if agentType == "Random":
			continue

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if agentType not in rewardsPerAgentType.keys():
			rewardsPerAgentType[agentType] = {"x": [], "p": [], "percentTrueP":[], "observedReward": [], "trueReward": [], "evIntended": []}

		rewardsPerAgentType[agentType]["x"].append(x)
		rewardsPerAgentType[agentType]["p"].append(p)
		rewardsPerAgentType[agentType]["percentTrueP"].append(resultsDict[a]["percentTrueP"])

		rewardsPerAgentType[agentType]["observedReward"].append(resultsDict[a]["mean_observed_reward"])
		rewardsPerAgentType[agentType]["trueReward"].append(resultsDict[a]["mean_true_reward"])
		rewardsPerAgentType[agentType]["evIntended"].append(resultsDict[a]["mean_value_intendedAction"])

		del resultsDict[a]


	for at in rewardsPerAgentType.keys():

		#################################################################
		# Observed Rewards vs True Rewards
		#################################################################

		for i in range(2):

			if i == 0:
				label = "pskills"
				toPlot = rewardsPerAgentType[at]["p"]
			else:
				label = "Rationality Percentage"
				toPlot = rewardsPerAgentType[at]["percentTrueP"]

			fig = plt.figure()

			fig.suptitle("Rewards | Domain: " + domain + " | Agent: " + at)

			# rows, cols, pos
			ax = fig.add_subplot(2, 1, 1)
			ax2 = fig.add_subplot(2, 1, 2)

			#plt.plot(eachParam,params[eachParam]["observed"], color = "blue", marker = ".")
			#plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*")


			s = ax.scatter(rewardsPerAgentType[at]["x"],toPlot, c = rewardsPerAgentType[at]["observedReward"])
			cbar = fig.colorbar(s, ax = ax)
			cbar.set_label("Observed Reward", labelpad=+1)

			s2 = ax2.scatter(rewardsPerAgentType[at]["x"],toPlot, c = rewardsPerAgentType[at]["trueReward"])
			cbar = fig.colorbar(s2, ax = ax2)
			cbar.set_label("True Reward", labelpad=+1)

			
			ax.set_xlabel("xSkills")
			ax.set_ylabel(label)
			
			ax2.set_xlabel("xSkills")
			ax2.set_ylabel(label)
			
			plt.margins(0.05)

			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "rewardsPerAgentType" + os.sep + "rewards-domain-" + domain + "-agent" + at + "-"+label+ ".png", bbox_inches = 'tight')
			plt.clf()
			plt.close("all")

		#################################################################


		#################################################################
		# EV Intended vs True Rewards
		#################################################################

		for i in range(2):

			if i == 0:
				label = "pskills"
				toPlot = rewardsPerAgentType[at]["p"]
			else:
				label = "Rationality Percentage"
				toPlot = rewardsPerAgentType[at]["percentTrueP"]


			fig = plt.figure()

			fig.suptitle("EVs intended | Domain: " + domain + " | Agent: " + at)

			# rows, cols, pos
			ax = fig.add_subplot(2, 1, 1)
			ax2 = fig.add_subplot(2, 1, 2)

			#plt.plot(eachParam,params[eachParam]["observed"], color = "blue", marker = ".")
			#plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*")

			s = ax.scatter(rewardsPerAgentType[at]["x"],toPlot, c = rewardsPerAgentType[at]["evIntended"])
			cbar = fig.colorbar(s, ax = ax)
			cbar.set_label("EV Intended", labelpad=+1)

			s2 = ax2.scatter(rewardsPerAgentType[at]["x"],toPlot, c = rewardsPerAgentType[at]["trueReward"])
			cbar = fig.colorbar(s2, ax = ax2)
			cbar.set_label("True Reward", labelpad=+1)

			
			ax.set_xlabel("xSkills")
			ax.set_ylabel(label)
			
			ax2.set_xlabel("xSkills")
			ax2.set_ylabel(label)
			
			plt.margins(0.05)

			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "rewardsPerAgentType" + os.sep + "evsIntended-domain-" + domain + "-agent" + at + "-" + label + ".png", bbox_inches = 'tight')
			plt.clf()
			plt.close("all")


def plotMeanEstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType(processedRFsAgentNames,methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,seenAgents,givenBeta):
	############################################################################################################################
	# plot all agents (of a given type) on the same plot for each one of the methods 
	# will plot mean of estimates
	# y axis = agent's true noise
	############################################################################################################################

	makeFolder(resultsFolder, "EstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType")

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	infoAgentTypes = {}

	for at in agentTypes:
		if at in seenAgents:
			infoAgentTypes[at] = {"noises": [], "pskills": [], "selectedNoises": [], "selectedPSkills": {}}


	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		infoAgentTypes[aType]["noises"].append(x)
		infoAgentTypes[aType]["pskills"].append(p)


	# Find which agents to select
	# For now min, max and somewhere in the middle
	# All noises are the same (for a given domain) for all agents but still iterating through and computing in case in the future changed to different
	# Pskills will vary according to the agent type

	for at in agentTypes:
		
		if at not in seenAgents:
			continue

		sortedNoises = sorted(infoAgentTypes[at]["noises"])

		# Find min - noises
		infoAgentTypes[at]["selectedNoises"].append(sortedNoises[0])
		
		# Find middle - noises
		infoAgentTypes[at]["selectedNoises"].append(sortedNoises[len(sortedNoises)//2])

		# Find max - noises
		infoAgentTypes[at]["selectedNoises"].append(sortedNoises[-1])



		# find pskills for given noises    
		for x in infoAgentTypes[at]["selectedNoises"]:

			if "Target" not in at:

				ixs = np.where(np.array(infoAgentTypes[at]["noises"]) == x)

				pskillsCopy = np.array(infoAgentTypes[at]["pskills"])

				ps = pskillsCopy[ixs]

				sortedPskills = sorted(ps)


				infoAgentTypes[at]["selectedPSkills"][x] = []

				# Find min - pskills 
				infoAgentTypes[at]["selectedPSkills"][x].append(sortedPskills[0])
				
				# Find middle - pskills 
				infoAgentTypes[at]["selectedPSkills"][x].append(sortedPskills[len(sortedPskills)//2])

				# Find max - pskills 
				infoAgentTypes[at]["selectedPSkills"][x].append(sortedPskills[-1])

			# for target agent
			else:
				ixs = np.where(np.array(infoAgentTypes[at]["noises"]) == x)

				pskillsCopy = np.array(infoAgentTypes[at]["pskills"])

				ps = pskillsCopy[ixs]

				sortedPskills = sorted(ps)

				infoAgentTypes[at]["selectedPSkills"][x] = []

				# Just 1 pskill since target agent 
				infoAgentTypes[at]["selectedPSkills"][x].append(sortedPskills[0])


	resultsDict = {}
	
	infoPlot = {}

	# Find the corresponding information for the selected agents
	for at in agentTypes:


		if at not in seenAgents:
			continue
		
		infoPlot[at] = {}

		for x in infoAgentTypes[at]["selectedNoises"]: 

			infoPlot[at][x] = {}

			for p in infoAgentTypes[at]["selectedPSkills"][x]: 

				infoPlot[at][x][p] = {} 

				found = False

				for a in processedRFsAgentNames:
					# code.interact(local = locals())

					if at == "Target": # since target agent doesn't have a p param
						if ("X" + str(x)[:-1]) in a and at in a:
							found = True
					else:
						if ("X" + str(x)[:-1]) in a and  str(p)[:-1] in a and at in a:
							found = True


					if found:

						# Load processed info		
						resultsDict[a] = loadProcessedInfo(prdFile,a)

						infoPlot[at][x][p]["percentTrueP"] = resultsDict[a]["percentTrueP"]


						for m in methods:
						
							if "tn" in m:
								continue

							if ("BM" in m and str(givenBeta) in m) or ("BM" not in m):

								if "BM" in m and str(givenBeta) in m:
									tempM, beta, tt = getInfoBM(m)

								if "-pSkills" in m:
									infoPlot[at][x][p][m] = {}

									infoPlot[at][x][p][m]["percentsEstimatedPs"] = resultsDict[a]["percentsEstimatedPs"][m]["averaged"]
									infoPlot[at][x][p][m]["percentTrueP"] = resultsDict[a]["percentTrueP"]
								# xskill method
								else: 

									if "BM" not in m:
										infoPlot[at][x][p][m] = resultsDict[a]["estimates"][m]
									else:
										infoPlot[at][x][p][m] = resultsDict[a]["estimates"][tt][tempM][givenBeta]

						del resultsDict[a]

						# Stop searching - assuming only 1 exp per agent since params assigned at random
						break

	
	##################################### FOR XSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs

	for at in agentTypes:

		if at not in seenAgents:
			continue

		# use as colors list index
		#i = 0

		ls = ["dotted","dashdot", "dashed"]

		for m in methods:

			# use as colors list index
			i = 0

			if "tn" in m or "-pSkills" in m:
				continue

			if "BM" in m and str(givenBeta) not in m:
				continue 


			fig = plt.figure()

			for x in infoAgentTypes[at]["selectedNoises"]:

				# get color
				color = colors[list(colors.keys())[i]]

				# increment index
				i += 1  

				mi = 0

				for p in infoAgentTypes[at]["selectedPSkills"][x]:

					prat = infoPlot[at][x][p]["percentTrueP"]

					plt.plot(range(len(infoPlot[at][x][p][m])),infoPlot[at][x][p][m], lw = 2.0, label = round(prat,4), color = color, ls = ls[mi])
				
					plt.plot(range(len(infoPlot[at][x][p][m])),[x]*len(infoPlot[at][x][p][m]), lw = 2.0, color = color)

					mi += 1


			plt.xlabel(r'\textbf{Number of observations}')
			plt.ylabel(r'\textbf{Execution Noise Level Estimate}')
			plt.margins(0.05)
			
			#fig.suptitle(r'\textbf{Method: ' + str(m) + ' | Agent Type: ' + at)

			# Put a legend to the right of the current axis
			plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			# Save png
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "EstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType" + os.sep + "estimatesXSKILLS-Method" + m + "-AgentType" + at + ".png", bbox_inches='tight')
			

			plt.clf()
			plt.close("all")


	##################################### FOR PSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs

	for at in agentTypes:

		if "Target" in at:
			continue

		if at not in seenAgents:
			continue

		# use as colors list index
		i = 0

		ls = ["solid", "dotted", "dashed"]

		for m in methods:

			if "tn" in m or "-xSkills" in m or "OR" in m or "BM" in m:
				continue

			#fig = plt.figure()

			#xs = infoAgentTypes[at]["selectedPSkills"].keys()
			#xi = 0

			# use as colors list index
			#i = 0

			#for p in infoAgentTypes[at]["selectedPSkills"][xs[xi]]:
			for x in infoAgentTypes[at]["selectedNoises"]:

				fig = plt.figure()

				# use as colors list index
				i = 0


				#for x in infoAgentTypes[at]["selectedNoises"]:
				for p in infoAgentTypes[at]["selectedPSkills"][x]:

					# get color
					color = colors[list(colors.keys())[i]]

					# increment index
					i += 1  

					mi = 0

					prat = infoPlot[at][x][p]["percentTrueP"]
				
					plt.plot(range(len(infoPlot[at][x][p][m]["percentsEstimatedPs"])),infoPlot[at][x][p][m]["percentsEstimatedPs"], lw = 2.0, label = prat, color = color, ls = ls[mi])
				
					plt.plot(range(len(infoPlot[at][x][p][m]["percentsEstimatedPs"])),[infoPlot[at][x][p][m]["percentTrueP"]]*len(infoPlot[at][x][p][m]["percentsEstimatedPs"]), lw = 2.0, color = color)

					mi += 1


				plt.xlabel(r'\textbf{Number of observations}')
				plt.ylabel(r'\textbf{Percent Reward}')
				plt.margins(0.05)
				
				fig.suptitle(r'\textbf{Method: ' + str(m) + ' | Agent Type: ' + at + " | Xskill: " + str(x))

				# Put a legend to the right of the current axis
				plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = "pskill")

				# Save png
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "EstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType" + os.sep + "estimatesPSKILLS-Method" + m + "-AgentType" + at + "-Xskills" + str(x) + ".png", bbox_inches='tight')

				plt.clf()
				plt.close("all")

def plotContourMSE_xSkillpSkillPerAgentTypePerMethod(processedRFsAgentNames,agentTypesFull,methods,resultsFolder,numStates,domain):

	makeFolder(resultsFolder, "contourMseXSkillPSkill-PerAgentTypePerMethod")

	agentTypes = deepcopy(seenAgents)

	# Remove target agent since no pskill param and thus not going to plot it
	if "Target" in agentTypesFull:
		agentTypes.remove("Target")


	if domain == "1d":
		pskillBuckets = [10, 30, 50, 70, 100]
	elif domain == "2d" or domain == "sequentialDarts":
		pskillBuckets = [5, 10, 15, 20, 32]


	if domain == "1d":
		# xskillBuckets = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
		xskillBuckets = np.linspace(0.5, 15.0, num = 100)
	elif domain == "2d" or domain == "sequentialDarts":
		xskillBuckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]


	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams.update({"axes.labelweight": "bold"})
	plt.rcParams.update({"axes.titleweight": "bold"})


	# Initialize dict to store info in
	info = {}

	for at in agentTypes:
		info[at] = {"method": {}}

		for m in methods:
			if "tn" not in m:

				info[at]["method"][m] = {"x": [], "p": [], "mse": [], "mseCategory": [], "bucketsCategory": [], \
										"pBuckets": [], "xBuckets": [], "xskillBuckets": xskillBuckets, "pskillBuckets": pskillBuckets}


	# For normalizing colorbar across methods
	'''
	minMseP = 9999
	maxMseP = -9999
	minMseX = 9999
	maxMseX = -9999
	'''

	resultsDict = {}

	probsRational = None

	# Get info 
	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if aType == "Target":
			continue

		# To normalize across all methods of same agent type
		minMseP = 9999
		maxMseP = -9999
		minMseX = 9999
		maxMseX = -9999

		
		for m in methods:

			if "tn" not in m:
				info[aType]["method"][m]["x"].append(x)
				info[aType]["method"][m]["p"].append(p)


				###############################################################
				# Needs to go inside because storing info within method dict
				###############################################################

				# find bucket for p
				for b in range(len(pskillBuckets)):
					if p <= pskillBuckets[b]:
						break

				# store 
				info[aType]["method"][m]["pBuckets"].append(b)


				# find bucket
				for b in range(len(xskillBuckets)):
					if x <= xskillBuckets[b]:
						break

				# store 
				info[aType]["method"][m]["xBuckets"].append(b)
				###############################################################


				if "-pSkills" in m:

					mse = resultsDict[a]["mse_percent_pskills"][m][numStates-1]
					info[aType]["method"][m]["mse"].append(mse)

					# to update bounds for colorbar normalization
					if mse < minMseP:
						minMseP = mse

					if mse > maxMseP:
						maxMseP = mse

				else:

					if "BM" not in m:
						mse = resultsDict[a]["plot_y"][m][numStates-1]
					else:
						tempM, beta, tt = getInfoBM(m)
						mse = resultsDict[a]["plot_y"][tt][tempM][givenBeta][numStates-1]


					info[aType]["method"][m]["mse"].append(mse)

					# to update bounds for colorbar normalization
					if mse < minMseX:
						minMseX = mse
						
					if mse > maxMseX:
						maxMseX = mse



		info[aType]["minMseP"] = minMseP
		info[aType]["maxMseP"] = maxMseP

		info[aType]["minMseX"] = minMseX
		info[aType]["maxMseX"] = maxMseX
		

		del resultsDict[a]


	# Find rationality info
	lessRationalInfo = []
	
	for eachX in pconfPerXskill:

		# Lambdas are sorted
		for ip in range(len(pconfPerXskill[eachX]["lambdas"])):
			if pconfPerXskill[eachX]["prat"][ip] >= 0.50:
				lessRationalInfo.append([eachX,pconfPerXskill[eachX]["lambdas"][ip],pconfPerXskill[eachX]["prat"][ip],ip])
				break

	lessRationalInfo = np.asarray(lessRationalInfo)
	# code.interact("...", local=dict(globals(), **locals()))


	if domain == "1d":
		# Create different execution skill levels 
	 	# xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
		xSkills = np.linspace(0.5, 15.0, num = 100) # (start, stop, num samples)

	elif domain == "2d" or domain == "sequentialDarts":
		# Create different execution skill levels 
		xSkills = np.linspace(2.5, 150.5, num = 100) # (start, stop, num samples)

	   
	# If normalizing across all agents
	'''
	minPZ = 9999
	maxPZ = -9999
	minXZ = 9999
	maxXZ = -9999
	'''

	#infoPlot = {}


	# Plot per method and per agent type
	for at in agentTypes:

		if at == "Random" or at == "Target":
			continue

		if "Bounded" in at:
			# Create different probabilities for an agent being rational
			# probsRational = np.linspace(0.0, 100.0, num = 100)
			
			# 0.001 - 100
			if domain == "1d":
				probsRational = np.logspace(-3, 2.0, num = 100)
			elif domain == "2d" or domain == "sequentialDarts":
				probsRational = np.logspace(-3, 1.5, num = 100)

		else:
			# Create different probabilities for an agent being rational
			probsRational = np.linspace(0.0, 1.0, num = 100)



		# To normalize across all methods of same agent type
		minPZ = 9999
		maxPZ = -9999
		minXZ = 9999
		maxXZ = -9999

		#infoPlot[at] = {}
			
		# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
		gx, gy = np.meshgrid(xSkills,probsRational)



		minMseP = info[at]["minMseP"]
		maxMseP = info[at]["maxMseP"]
		minMseX = info[at]["minMseX"]
		maxMseX = info[at]["maxMseX"]


		for m in methods:

			if "tn" not in m:

				#infoPlot[at][m] = {}

				####################################################################################################
				# Scatter plot
				####################################################################################################

				fig = plt.figure()

				# rows, cols, pos
				ax = fig.add_subplot(2, 1, 1)


				cmap = plt.get_cmap("viridis")

				'''
				if "-pSkills" in m:
					norm = plt.Normalize(minMseP, maxMseP)
				else:
					norm = plt.Normalize(minMseX, maxMseX)
				'''

				# Norm per method
				norm = plt.Normalize(np.nanmin(info[at]["method"][m]["mse"]),np.nanmax(info[at]["method"][m]["mse"]))

				s = ax.scatter(info[at]["method"][m]["x"],info[at]["method"][m]["p"], c = cmap(norm(info[at]["method"][m]["mse"])))

				if "Bounded" in at:
					# Add line to present where agents with 50% rationality are
					plt.plot(lessRationalInfo[:,0],lessRationalInfo[:,1],c="black",linewidth=2)


				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm,ax=ax)
				cbar.set_label("MSE", labelpad=+1)

				ax.set_xlabel("xSkills")
				ax.set_ylabel("pSkills")
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" + os.sep + "scatterPlot-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close("all")
				####################################################################################################


				N = len(info[at]["method"][m]["x"])

				# to store the different xskills & probs rational 
				POINTS = np.zeros((N,2))
				
				# to store the mean of the observed rewards
				VALUES = np.zeros((N,1))

				for i in range(N):

					POINTS[i][:] = [info[at]["method"][m]["x"][i],info[at]["method"][m]["p"][i]] #[x,p]

					VALUES[i] = info[at]["method"][m]["mse"][i] #mse


				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'cubic')

				Z = griddata(POINTS,VALUES,(gx,gy),method='linear',rescale=True)
				
				Z = Z[:,:,0]

				# remove inf's -> causes surface plot to be all of the same color 
				Z[Z == np.inf] = np.nan


				# To update colorbar norm
				aMin = np.nanmin(Z)
				aMax = np.nanmax(Z)

				if "-pSkills" in m:

					if aMin < minPZ:
						minPZ = aMin

					if aMax > maxPZ:
						maxPZ = aMax

				# xskill method
				else:
					if aMin < minXZ:
						minXZ = aMin

					if aMax > maxXZ:
						maxXZ = aMax


				# Save info to use in plot next
				info[at]["method"][m]["POINTS"] = POINTS
				info[at]["method"][m]["VALUES"] = VALUES
				info[at]["method"][m]["Z"] = Z
				info[at]["method"][m]["gx"] = gx
				info[at]["method"][m]["gy"] = gy



		info[at]["minPZ"] = minPZ
		info[at]["maxPZ"] = maxPZ

		info[at]["minXZ"] = minXZ
		info[at]["maxXZ"] = maxXZ


	#code.interact("before contours", local=locals())

	# Plotting contours separately in order to have access to the Z info before to normalize colorbar
	for at in agentTypes:

		minPZ = info[at]["minPZ"]
		maxPZ = info[at]["maxPZ"]

		minXZ = info[at]["minXZ"]
		maxXZ = info[at]["maxXZ"]

		for m in methods:

			if "tn" not in m:
				fig = plt.figure()
				ax = plt.subplot(111)

				cmap = plt.get_cmap("viridis")

				# For normalizing colormaps of individual contour plots
				# Across all
				'''
				if "-pSkills" in m:
					norm = plt.Normalize(minPZ, maxPZ)
				else:
					norm = plt.Normalize(minXZ, maxXZ)
					# Add line to present where agents with 50% rationality are
					plt.plot(lessRationalInfo[:,0],np.log10(lessRationalInfo[:,1]),c="black",linewidth=2)
				'''

				gx = info[at]["method"][m]["gx"]
				gy = info[at]["method"][m]["gy"]
				Z = info[at]["method"][m]["Z"]

				# Normalize per method
				norm = plt.Normalize(np.nanmin(Z),np.nanmax(Z))

				if "Bounded" in at:
					cs = plt.contourf(gx,np.log10(gy),Z,norm=norm)
					
					# Add line to present where agents with 50% rationality are
					plt.plot(lessRationalInfo[:,0],np.log10(lessRationalInfo[:,1]),c="black",linewidth=2)

					# ax.set_yscale('symlog')

				else:
					cs = plt.contourf(gx, gy, Z, norm = norm)


				#plt.xlabel("Execution Skills")
				#plt.ylabel("Planning Skills")
				plt.xlabel(r"\textbf{Execution Noise Levels}")
				plt.ylabel(r"\textbf{Rationality Parameters}")
				#plt.title("MSE | Domain: " + domain + " | Agent: " + at +" | Method: " + m)


				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm,ax=ax)


				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".pdf", bbox_inches='tight')
				plt.clf()
				plt.close("all")


	rows = int(len(methods)/2)

	# if odd, add +
	if len(methods) % 2 != 0:
		rows += 1

	cols = 2

	# Plot all methods together (different subplots) per agent type
	for at in agentTypes:

		if at == "Random":
			continue

		if at == "Target":
			continue


		minPZ = info[at]["minPZ"]
		maxPZ = info[at]["maxPZ"]

		minXZ = info[at]["minXZ"]
		maxXZ = info[at]["maxXZ"]


		fig = plt.figure(figsize=(10,15))

		fig.subplots_adjust(hspace=0.8, wspace=0.6)

		fig.suptitle("MSE | Domain: " + domain + " | Agent: " + at)

		cmap = plt.get_cmap("viridis")


		for mi in range(len(methods)):
			
			m = methods[mi]

			if "tn" not in m:

				'''
				if "-pSkills" in m:
					norm = plt.Normalize(minPZ, maxPZ)
				else:
					norm = plt.Normalize(minXZ, maxXZ)
				'''

				# Normalize per method
				norm = plt.Normalize(np.nanmin(Z),np.nanmax(Z))

				# (# rows, # cols, position)
				ax = fig.add_subplot(rows, cols, mi+1)

				gx = info[at]["method"][m]["gx"]
				gy = info[at]["method"][m]["gy"]
				Z = info[at]["method"][m]["Z"]

				if "Bounded" in at:
					cs = plt.contourf(gx, np.log10(gy), Z, norm = norm) # cmap = cmap(norm(Z)))
				else:
					cs = plt.contourf(gx, gy, Z, norm = norm) # cmap = cmap(norm(Z)))


				ax.set_xlabel("XSkills")
				ax.set_ylabel("PSkills")
				ax.set_title( "Method: " + m)

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm, ax = ax)

		plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-AllMethods.png", bbox_inches='tight')
		plt.clf()
		plt.close("all")



	##############################################################################################################
	# Calling function(s) here in order to make use of the info that was computed in this function
	##############################################################################################################

	# diffPlotsForMSExSkillpSkillPerAgentTypePerMethod(resultsFolder, agentTypes, methods, domain, numStates, info, xskillBuckets, pskillBuckets)

	# createContourMSECategories_xSkillpSkill(resultsFolder, resultsDict, agentTypes, methods, xSkills, probsRational, info)

def plotContourEstimates_xSkillpSkill_PerAgentTypePerMethod(processedRFsAgentNames,agentTypesFull,methods,resultsFolder,numStates,domain):

	makeFolder(resultsFolder, "contourEstimatesXSkillPSkill-PerAgentTypePerMethod")

	agentTypes = deepcopy(seenAgents)

	# Remove target agent since no pskill param and thus not going to plot it
	if "Target" in agentTypesFull:
		agentTypes.remove("Target")


	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams.update({"axes.labelweight": "bold"})
	plt.rcParams.update({"axes.titleweight": "bold"})


	# Initialize dict to store info in
	info = {}

	for at in agentTypes:
		info[at] = {"method": {}}

		for m in methods:
			if "tn" not in m:

				info[at]["method"][m] = {"x": [], "p": [], "estimates": []}


	probsRational = None

	minX = 9999
	maxX = -9999
	minP = 9999
	maxP = -9999

	# To normalize across all methods of same agent type - contour
	minPZ = 9999
	maxPZ = -9999
	minXZ = 9999
	maxXZ = -9999


	resultsDict = {}

	# Get info 
	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if aType == "Target":
			continue
		
		for m in methods:

			if "tn" not in m:
				info[aType]["method"][m]["x"].append(x)
				info[aType]["method"][m]["p"].append(p)
				
				if "BM" not in m:
					est = resultsDict[a]["estimates"][m][numStates-1]
				else:
					tempM, beta, tt = getInfoBM(m)
					est = resultsDict[a]["estimates"][tt][tempM][beta][numStates-1]
				

				info[aType]["method"][m]["estimates"].append(est)

				'''
				if "-pSkills" in m:

					if est < minP:
						minP = est
					if est > maxP:
						maxP = est
				else:

					if est < minX:
						minX = est
					if est > maxX:
						maxX = est
				'''


		del resultsDict[a]

	
	if domain == "1d":
		# 0.001 - 100
		lambdas = np.logspace(-3,2.0,num=100)

	elif domain == "2d" or domain == "sequentialDarts":
		lambdas = np.logspace(-3,1.5,num=100)


	# Find rationality info
	lessRationalInfo = []
	
	for eachX in pconfPerXskill:

		# Lambdas are sorted
		for ip in range(len(pconfPerXskill[eachX]["lambdas"])):
			if pconfPerXskill[eachX]["prat"][ip] >= 0.50:
				lessRationalInfo.append([eachX,pconfPerXskill[eachX]["lambdas"][ip],pconfPerXskill[eachX]["prat"][ip],ip])
				break

	lessRationalInfo = np.asarray(lessRationalInfo)
	# code.interact("...", local=dict(globals(), **locals()))


	if domain == "1d":
		# Create different execution skill levels 
		# xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
		xSkills = np.linspace(0.5, 15.0, num = 100) # (start, stop, num samples)

	elif domain == "2d" or domain == "sequentialDarts":
		# Create different execution skill levels 
		xSkills = np.linspace(2.5, 150.5, num = 100) # (start, stop, num samples)


	minX = xSkills[0]
	maxX = xSkills[-1]
	   
	# Plot per method and per agent type
	for at in agentTypes:

		if at == "Random" or at == "Target":
			continue

		if "Bounded" in at:
			# Create different probabilities for an agent being rational			
			if domain == "1d":
				# 0.001 - 100
				probsRational = np.logspace(-3,2.0,num=100)

			elif domain == "2d" or domain == "sequentialDarts":
				probsRational = np.logspace(-3,1.5,num=100)

		else:
			# Create different probabilities for an agent being rational
			probsRational = np.linspace(0.0,1.0,num=100)


		# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
		gx, gy = np.meshgrid(xSkills,probsRational)

		cmap = plt.get_cmap("viridis")


		for m in methods:

			if "tn" not in m:

				####################################################################################################
				# Scatter plot
				####################################################################################################

				fig = plt.figure(figsize=(10,10))

				# rows, cols, pos
				ax = fig.add_subplot(2,1,1)


				if "-pSkills" in m:
					norm = plt.Normalize(lambdas[0],lambdas[-1])
				else:
					norm = plt.Normalize(minX,maxX)


				plt.scatter(info[at]["method"][m]["x"],info[at]["method"][m]["p"], c = cmap(norm(info[at]["method"][m]["estimates"])))
				
				if "Bounded" in at:
					# Add line to present where agents with 50% rationality are
					plt.plot(lessRationalInfo[:,0],lessRationalInfo[:,1],c="black",linewidth=2)


				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm,ax=ax)			
				cbar.set_label("Estimate", labelpad=+1)

				ax.set_xlabel("Execution Skill Noise Level")
				ax.set_ylabel("Decision-Making Skill")
				
				plt.margins(0.05)
					
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" + os.sep + "scatterPlot-Estimate-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close("all")
				####################################################################################################


				N = len(info[at]["method"][m]["x"])
				# code.interact(local=locals())

				# to store the different xskills & probs rational 
				POINTS = np.zeros((N,2))
				
				# to store the mean of the observed rewards
				VALUES = np.zeros((N,1))

				for i in range(N):
					POINTS[i][:] = [info[at]["method"][m]["x"][i],info[at]["method"][m]["p"][i]] #[x,p]
					VALUES[i] = info[at]["method"][m]["estimates"][i] #estimates

				Z = griddata(POINTS,VALUES,(gx,gy),method='linear',rescale=True)
				# Z = griddata(POINTS,VALUES,(gx,gy),method='cubic',rescale=True)


				Z = Z[:,:,0]

				# remove inf's -> causes surface plot to be all of the same color 
				Z[Z == np.inf] = np.nan

				# To update colorbar norm
				aMin = np.nanmin(Z)
				aMax = np.nanmax(Z)

				if "-pSkills" in m:

					if aMin < minPZ:
						minPZ = aMin

					if aMax > maxPZ:
						maxPZ = aMax

				# xskill method
				else:
					if aMin < minXZ:
						minXZ = aMin

					if aMax > maxXZ:
						maxXZ = aMax


				# Save info to use in plot next
				info[at]["method"][m]["POINTS"] = POINTS
				info[at]["method"][m]["VALUES"] = VALUES
				info[at]["method"][m]["Z"] = Z
				info[at]["method"][m]["gx"] = gx
				info[at]["method"][m]["gy"] = gy


		info[at]["minPZ"] = minPZ
		info[at]["maxPZ"] = maxPZ

		info[at]["minXZ"] = minXZ
		info[at]["maxXZ"] = maxXZ


	#code.interact("before contours", local=locals())

	# Plotting contours separately in order to have access to the Z info before to normalize colorbar
	for at in agentTypes:

		if at == "Random":
			continue

		if at == "Target":
			continue

		'''
		if "Bounded" in at:
			# Create different probabilities for an agent being rational			
			if domain == "1d":
				# 0.001 - 100
				probsRational = np.logspace(-3,2.0,num=100)

			elif domain == "2d" or domain == "sequentialDarts":
				probsRational = np.logspace(-3,1.5,num=100)

		else:
			# Create different probabilities for an agent being rational
			probsRational = np.linspace(0.0,1.0,num=100)
		'''

		for m in methods:

			if "tn"  in m:
				continue

			fig = plt.figure(figsize=(6,5),tight_layout=True)
			    
			spec = gridspec.GridSpec(ncols=1, nrows=1,figure=fig)#,width_ratios=[1,0.05])

			ax = fig.add_subplot(spec[0])
			#ax2 = fig.add_subplot(spec[1])

			fig.tight_layout()


			#fig = plt.figure(figsize=(6,5))
			#ax = plt.subplot(111)

			gx = info[at]["method"][m]["gx"]
			gy = info[at]["method"][m]["gy"]
			Z = info[at]["method"][m]["Z"]


			minPZ = info[at]["minPZ"]
			maxPZ = info[at]["maxPZ"]

			minXZ = info[at]["minXZ"]
			maxXZ = info[at]["maxXZ"]

			#'''
			if "pSkills" in m:
				if "Bounded" in at:
					norm = plt.Normalize(np.log10(lambdas[0]),np.log10(lambdas[-1]))
				else:
					norm = plt.Normalize(lambdas[0],lambdas[-1])
			else:
				norm = plt.Normalize(minXZ,maxXZ)
			#'''
			#norm = plt.Normalize(np.nanmin(Z),np.nanmax(Z))


			if "Bounded" in at:

				if "pSkills" in m:
					cs = plt.contourf(gx,np.log10(gy),np.log10(Z),norm = norm)
				else:
					cs = plt.contourf(gx,np.log10(gy),Z,norm = norm)
				
				# Add line to present where agents with 50% rationality are
				plt.plot(lessRationalInfo[:,0],np.log10(lessRationalInfo[:,1]),c="black",linewidth=2)

				# ax.set_yscale('symlog')

			else:
				cs = plt.contourf(gx,gy,Z,norm = norm)


			plt.xlabel(r"\textbf{Execution Skill Noise Levels}")
			plt.ylabel(r"\textbf{Rationality Parameters}")

			sm.set_array([])
			sm = ScalarMappable(norm = norm, cmap = cmap)

			'''
			from mpl_toolkits.axes_grid1 import make_axes_locatable
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			'''

			'''
			if "pSkill" in m:
				ratio = 0.9
			else:
				ratio = 0.7

			x_left, x_right = ax.get_xlim()
			y_low, y_high = ax.get_ylim()
			ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
			'''


			if "pSkill" in m:

				# Shrink current axis
				# box = ax.get_position()
				# ax.set_position([box.x0, box.y0, box.width-(box.width*0.05), box.height])
				
				# divider = make_axes_locatable(ax)
				# cax = divider.append_axes("right", size="5%", pad=0.05)
				# cbar = fig.colorbar(sm,ax=ax,spacing="proportional",fraction=0.0385)
				cbar = fig.colorbar(sm,ax=ax)
				'''
				if "Bounded" in at:

					ticks = cbar.ax.get_yticks()

					newTicks = []
					
					for each in ticks:
						newTicks.append(str(np.log10(round(each,2))))
					
					cbar.ax.set_yticks(newTicks) 
				'''
				   
			else:
				# import matplotlib.ticker as ticker
				# tts = ticker.FixedLocator([2,4,6,8,10,12,14])
			

				# cax = divider.append_axes("bottom", size="5%", pad=0.05)
				cbar = fig.colorbar(sm,ax=ax,orientation="horizontal")

				'''
				from mpl_toolkits.axes_grid1.inset_locator import inset_axes
				axins = inset_axes(ax,
				                    width="100%",  
				                    height="5%",
				                    loc='lower center',
				                    borderpad=-5
				                   )
				fig.colorbar(sm, cax=axins, orientation="horizontal")
				'''

				#ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
			
			# ax.set_box_aspect(1)

	

			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" +os.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png")
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" +os.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".pdf")
			plt.clf()
			plt.close("all")

def plotContourEstimates_xSkillPercent_PerAgentTypePerMethod(processedRFsAgentNames,agentTypesFull,methods,resultsFolder,numStates,domain):

	makeFolder(resultsFolder, "contourEstimatesXSkillPercent-PerAgentTypePerMethod")

	agentTypes = deepcopy(seenAgents)

	# Remove target agent since no pskill param and thus not going to plot it
	if "Target" in agentTypesFull:
		agentTypes.remove("Target")


	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams.update({"axes.labelweight": "bold"})
	plt.rcParams.update({"axes.titleweight": "bold"})


	# Initialize dict to store info in
	info = {}

	for at in agentTypes:
		info[at] = {"method": {}}

		for m in methods:
			if "tn" not in m:

				info[at]["method"][m] = {"x": [], "p": [], "estimates": []}


	probsRational = None

	minX = 9999
	maxX = -9999
	minP = 9999
	maxP = -9999

	# To normalize across all methods of same agent type - contour
	minPZ = 9999
	maxPZ = -9999
	minXZ = 9999
	maxXZ = -9999


	resultsDict = {}

	# Get info 
	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		if aType == "Target":
			continue
		
		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)

		percentTrueP = resultsDict[a]["percentTrueP"]

		for m in methods:

			if "tn" not in m:
				info[aType]["method"][m]["x"].append(x)
				info[aType]["method"][m]["p"].append(percentTrueP)


				if "-pSkills" in m:
					est = resultsDict[a]["percentsEstimatedPs"][m]["averaged"][numStates-1]
					info[aType]["method"][m]["estimates"].append(est)

					if est < minP:
						minP = est
					if est > maxP:
						maxP = est

				else:
				
					if "BM" not in m:
						est = resultsDict[a]["estimates"][m][numStates-1]
					else:
						tempM, beta, tt = getInfoBM(m)
						est = resultsDict[a]["estimates"][tt][tempM][beta][numStates-1]
					

					info[aType]["method"][m]["estimates"].append(est)

					if est < minX:
						minX = est
					if est > maxX:
						maxX = est


		del resultsDict[a]

	
	# Find rationality info
	lessRationalInfo = []
	
	for eachX in pconfPerXskill:

		# Lambdas are sorted
		for ip in range(len(pconfPerXskill[eachX]["lambdas"])):
			if pconfPerXskill[eachX]["prat"][ip] >= 0.50:
				lessRationalInfo.append([eachX,pconfPerXskill[eachX]["lambdas"][ip],pconfPerXskill[eachX]["prat"][ip],ip])
				break

	lessRationalInfo = np.asarray(lessRationalInfo)
	# code.interact("...", local=dict(globals(), **locals()))


	# Create different execution skill levels 
	if domain == "1d":
		# xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
		xSkills = np.linspace(0.5, 15.0, num = 100) # (start, stop, num samples)

	elif domain == "2d" or domain == "sequentialDarts":
		xSkills = np.linspace(2.5, 150.5, num = 100) # (start, stop, num samples)

	
	# Since on prat terms
	probsRational = np.linspace(0.0,1.0,num=100)
	

	# Plot per method and per agent type
	for at in agentTypes:

		if at == "Random" or at == "Target":
			continue

		# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
		gx, gy = np.meshgrid(xSkills,probsRational)

		for m in methods:

			if "tn" not in m:

				####################################################################################################
				# Scatter plot
				####################################################################################################

				fig = plt.figure(figsize=(10,10))

				# rows, cols, pos
				ax = fig.add_subplot(2,1,1)

				if "pSkills" in m:
					cmap = plt.get_cmap("viridis")
					norm = plt.Normalize(probsRational[0],probsRational[-1])
				else:
					cmap = plt.get_cmap("viridis_r")
					norm = plt.Normalize(xSkills[0],xSkills[-1])

				plt.scatter(info[at]["method"][m]["x"],info[at]["method"][m]["p"], c = cmap(norm(info[at]["method"][m]["estimates"])))
				
				'''
				if "Bounded" in at:
					# Add line to present where agents with 50% rationality are
					plt.plot(lessRationalInfo[:,0],lessRationalInfo[:,2],c="black",linewidth=2)
				'''

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm,ax=ax)			
				cbar.set_label("Estimate", labelpad=+1)

				ax.set_xlabel("Execution Skill Noise Level")
				ax.set_ylabel("Rationality Percentage")
				
				plt.margins(0.05)
					
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPercent-PerAgentTypePerMethod" + os.sep + "scatterPlot-Estimate-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close("all")
				####################################################################################################


				N = len(info[at]["method"][m]["x"])
				# code.interact(local=locals())

				# to store the different xskills & probs rational 
				POINTS = np.zeros((N,2))
				
				# to store the mean of the observed rewards
				VALUES = np.zeros((N,1))

				for i in range(N):
					POINTS[i][:] = [info[at]["method"][m]["x"][i],info[at]["method"][m]["p"][i]] #[x,p]
					VALUES[i] = info[at]["method"][m]["estimates"][i] #estimates

				Z = griddata(POINTS,VALUES,(gx,gy),method='linear',rescale=True)
				# Z = griddata(POINTS,VALUES,(gx,gy),method='cubic',rescale=True)


				Z = Z[:,:,0]

				# remove inf's -> causes surface plot to be all of the same color 
				Z[Z == np.inf] = np.nan

				# To update colorbar norm
				aMin = np.nanmin(Z)
				aMax = np.nanmax(Z)

				if "-pSkills" in m:

					if aMin < minPZ:
						minPZ = aMin

					if aMax > maxPZ:
						maxPZ = aMax

				# xskill method
				else:
					if aMin < minXZ:
						minXZ = aMin

					if aMax > maxXZ:
						maxXZ = aMax


				# Save info to use in plot next
				info[at]["method"][m]["POINTS"] = POINTS
				info[at]["method"][m]["VALUES"] = VALUES
				info[at]["method"][m]["Z"] = Z
				info[at]["method"][m]["gx"] = gx
				info[at]["method"][m]["gy"] = gy


		info[at]["minPZ"] = minPZ
		info[at]["maxPZ"] = maxPZ

		info[at]["minXZ"] = minXZ
		info[at]["maxXZ"] = maxXZ


	#code.interact("before contours", local=locals())

	# Plotting contours separately in order to have access to the Z info before to normalize colorbar
	for at in agentTypes:

		if at == "Random" or at == "Target":
			continue

		for m in methods:

			if "tn"  in m:
				continue


			fig = plt.figure(figsize=(6,5),tight_layout=True)
			    
			spec = gridspec.GridSpec(ncols=1, nrows=1,figure=fig)#,width_ratios=[1,0.05])

			ax = fig.add_subplot(spec[0])
			#ax2 = fig.add_subplot(spec[1])

			fig.tight_layout()


			#fig = plt.figure(figsize=(6,5))
			#ax = plt.subplot(111)

			gx = info[at]["method"][m]["gx"]
			gy = info[at]["method"][m]["gy"]
			Z = info[at]["method"][m]["Z"]


			minPZ = info[at]["minPZ"]
			maxPZ = info[at]["maxPZ"]

			minXZ = info[at]["minXZ"]
			maxXZ = info[at]["maxXZ"]

			# '''
			if "pSkills" in m:
				cmap = plt.get_cmap("viridis")
				norm = plt.Normalize(probsRational[0],probsRational[-1])
			else:
				cmap = plt.get_cmap("viridis_r")
				norm = plt.Normalize(xSkills[0],xSkills[-1])
			# '''
			# norm = plt.Normalize(np.nanmin(Z),np.nanmax(Z))


			cs = plt.contourf(gx,gy,Z,norm = norm,cmap=cmap)
			
			'''
			if "Bounded" in at:
				
				# Add line to present where agents with 50% rationality are
				plt.plot(lessRationalInfo[:,0],lessRationalInfo[:,2],c="black",linewidth=2)

				# ax.set_yscale('symlog')
			'''

			plt.xlabel(r"\textbf{Execution Skill Noise Levels}")
			plt.ylabel(r"\textbf{Rationality Percentage}")

			sm.set_array([])
			sm = ScalarMappable(norm = norm, cmap = cmap)

			'''
			from mpl_toolkits.axes_grid1 import make_axes_locatable
			# create an axes on the right side of ax. The width of cax will be 5%
			# of ax and the padding between cax and ax will be fixed at 0.05 inch.
			divider = make_axes_locatable(ax)
			'''

			'''
			if "pSkill" in m:
				ratio = 0.9
			else:
				ratio = 0.7

			x_left, x_right = ax.get_xlim()
			y_low, y_high = ax.get_ylim()
			ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
			'''


			if "pSkill" in m:
				# Shrink current axis
				# box = ax.get_position()
				# ax.set_position([box.x0, box.y0, box.width-(box.width*0.05), box.height])
				
				# divider = make_axes_locatable(ax)
				# cax = divider.append_axes("right", size="5%", pad=0.05)
				# cbar = fig.colorbar(sm,ax=ax,spacing="proportional",fraction=0.0385)
				cbar = fig.colorbar(sm,ax=ax)
				   
			else:
				# import matplotlib.ticker as ticker
				# tts = ticker.FixedLocator([2,4,6,8,10,12,14])
			

				# cax = divider.append_axes("bottom", size="5%", pad=0.05)
				cbar = fig.colorbar(sm,ax=ax,orientation="horizontal")

				'''
				from mpl_toolkits.axes_grid1.inset_locator import inset_axes
				axins = inset_axes(ax,
				                    width="100%",  
				                    height="5%",
				                    loc='lower center',
				                    borderpad=-5
				                   )
				fig.colorbar(sm, cax=axins, orientation="horizontal")
				'''

				#ax.set_aspect(1./ax.get_data_ratio(),adjustable='box')
			
			# ax.set_box_aspect(1)
	

			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPercent-PerAgentTypePerMethod" +os.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png")
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "contourEstimatesXSkillPercent-PerAgentTypePerMethod" +os.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".pdf")
			plt.clf()
			plt.close("all")


def plotMSExSkillsPerBucketsPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP):

	# BUCKETS PER PERCENTS RAND/MAX REWARD -- SHOWING MSE FOR XSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-pSkills"

	makeFolder(resultsFolder, "mseXskills-PerBucketsPerAgentType")

	# buckets = [25,50,75,100]
	'''
	if domain == "sequentialDarts":

		# Buckets in percents terms - between 0-1
		buckets = [0.20,0,40,0.60,0.80,1.0]

		# labelsB = {0.25:"0.00-0.25",0.50:"0.25-0.50",0.75:"0.50-0.75",1.0:"0.75-1.00"}
		labelsB = {0.25:"0\%-25\%",0.50:"25\%-50\%",0.75:"50\%-75\%",1.0:"75\%-100\%"}

	else:
	'''
	
	# Buckets in percents terms - between 0-1
	buckets = [0.25,0.50,0.75,1.0]

	# labelsB = {0.25:"0.00-0.25",0.50:"0.25-0.50",0.75:"0.50-0.75",1.0:"0.75-1.00"}
	labelsB = {0.25:"0\%-25\%",0.50:"25\%-50\%",0.75:"50\%-75\%",1.0:"75\%-100\%"}


	# init dict to store info
	mseDict = {}

	stdInfoPerAgentTypePerMethod = {}
	stdPerAgentTypePerMethod = {}
	confidenceIntervals = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": {}}

		stdInfoPerAgentTypePerMethod[at] = {}
		stdPerAgentTypePerMethod[at] = {}
		confidenceIntervals[at] = {}

		for b in buckets:
			mseDict[at]["numAgents"][b] = 0.0


		for m in actualMethodsOnExps:
			# Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			mseDict[at]["perMethod"][m] = {}
			stdInfoPerAgentTypePerMethod[at][m] = {} 
			stdPerAgentTypePerMethod[at][m] = {}
			confidenceIntervals[at][m] = {}


			for b in buckets:
				mseDict[at]["perMethod"][m][b] = [0.0] * numStates # to store per state - across different exps per agent type

				stdInfoPerAgentTypePerMethod[at][m][b] = [] 
				stdPerAgentTypePerMethod[at][m][b] = 0.0
				confidenceIntervals[at][m][b] = {"low": 0.0, "high": 0.0, "value": 0.0}

	resultsDict = {}

	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# update agent count
		# mseDict[aType]["numAgents"] += resultsDict[a]["num_exps"]
		

		#estimatedP = resultsDict[a]["mse_percent_pskills"][method][numStates-1] # #### ESTIMATED %

		trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
		# using true percent and not estimated one

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break


		# get actual bucket
		b = buckets[b]


		mseDict[aType]["numAgents"][b] += 1.0
		

		# for each method
		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue


			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)
				# for each state
				for mxi in range(numStates):	

					# xskill error
					sq = resultsDict[a]["plot_y"][tt][tempM][beta][mxi]

					# store squared error
					mseDict[aType]["perMethod"][m][b][mxi] += sq


			elif "BM" not in m:

				# for each state
				for mxi in range(numStates):	

					# xskill error
					sq = resultsDict[a]["plot_y"][m][mxi]

					# store squared error
					mseDict[aType]["perMethod"][m][b][mxi] += sq


		del resultsDict[a]


	# normalize - find Mean Squared Error

	# for each agent type
	for at in seenAgents:

		# for each method
		for m in actualMethodsOnExps:

			# Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:

				# for each state
				for mxi in range(numStates):

					#find MSE  = SE / # of agents
					mseDict[at]["perMethod"][m][b][mxi] /= mseDict[aType]["numAgents"][b]


	#####################################################################################################
	# get data for standard deviation across all agents of same type --- for last state
	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue


			### Find bucket
			trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
			# using true percent and not estimated one

			# find bucket corresponding to trueP
			for b in range(len(buckets)):
				if trueP <= buckets[b]:
					break

			# get actual bucket
			b = buckets[b]

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)
				# For xskill methods since ignoring pskill
				stdInfoPerAgentTypePerMethod[aType][m][b].append(resultsDict[a]["plot_y"][tt][tempM][beta][-1])
			elif "BM" not in m:
				# For xskill methods since ignoring pskill
				stdInfoPerAgentTypePerMethod[aType][m][b].append(resultsDict[a]["plot_y"][m][-1])


		del resultsDict[a]


	# compute actual std
	for at in seenAgents:
		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:
				if len(stdInfoPerAgentTypePerMethod[at][m][b]) != 0:
					stdPerAgentTypePerMethod[at][m][b] = np.std(stdInfoPerAgentTypePerMethod[at][m][b])

	#####################################################################################################


	#####################################################################################################
	# COMPUTE CONFIDENCE INTERVALS
	#####################################################################################################
	
	ci = 0.95

	for at in seenAgents:
		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:

				if len(stdInfoPerAgentTypePerMethod[at][m][b]) != 0:

					mu = mseDict[at]["perMethod"][m][b][-1]
					sigma = stdPerAgentTypePerMethod[at][m][b]
					N = mseDict[at]["numAgents"][b]


					confidenceIntervals[at][m][b]["low"], confidenceIntervals[at][m][b]["high"] =\
					stats.norm.interval(ci, loc=mu, scale=sigma/np.sqrt(N))

					# for 95% interval
					Z = 1.960

					confidenceIntervals[at][m][b]["value"] = Z * (sigma/np.sqrt(N))
				else:
					confidenceIntervals[at][m][b]["low"] = -1
					confidenceIntervals[at][m][b]["high"] = -1
					confidenceIntervals[at][m][b]["value"] = -1

	#####################################################################################################

	xskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + "mseXskills-PerBucketsPerAgentType" + os.sep + "confidenceIntervals-xSkills.txt", "a")

	# save info to text files
	for at in seenAgents:

		for m in actualMethodsOnExps:

				#Skip pskill methods
				if "-pSkills" in m:
					continue

				if "tn" in m:
					continue

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					tempM2 = m.split("-Beta")[0]
					tempM3 = methodNamesPaper[m.split("-Beta")[0]]

				# Gather all MSE's of current method
				allMseCurrentMethodAndBucket = []

				for b in buckets:
					allMseCurrentMethodAndBucket.append(mseDict[at]["perMethod"][m][b][-1])
			

				# order from highest to lowest MSE
				orderedMSE = sorted(allMseCurrentMethodAndBucket, reverse = True)


				d_x = {"Agents":[], "Bucket": [], "MSE":[], "Low": [], "High": [], "Values": []}
			

				if "BM" not in m:
					tempM3 = m

					
				xskillsCI.write("\n--------------------------------------------------------------------------------------------------------\n\
							Agent: " + str(at) + " - Method: " + str(m) + "\n"+ \
						"--------------------------------------------------------------------------------------------------------\n ")

				# Output info to files
				for mseO in range(len(orderedMSE)):

					index = allMseCurrentMethodAndBucket.index(orderedMSE[mseO])

					b = buckets[index]


					### for xskills
					xskillsCI.write("Agent: " + at + " |   Bucket: " + str(b) + \
							" ->  Low: " + f"{confidenceIntervals[at][m][b]['low']:.4f}" +\
							" | High: " + f"{confidenceIntervals[at][m][b]['high']:.4f}" +\
							" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
							" | Value: " + f"{confidenceIntervals[at][m][b]['value']:.4f}" + "\n")

					d_x["Agents"].append(at)
					d_x["Bucket"].append(b)
					d_x["MSE"].append(round(orderedMSE[mseO],4))
					d_x["Low"].append(round(confidenceIntervals[at][m][b]["low"],4))
					d_x["High"].append(round(confidenceIntervals[at][m][b]["high"],4))
					d_x["Values"].append(round(confidenceIntervals[at][m][b]["value"],4))


					xskillsCI.write("\n")

				xskillsCI.write("\n")

				 # Convert dicts to pandas dataframe
				d_x_pd = pd.DataFrame(d_x, columns = ["Agents", "Bucket", "Low", "MSE",  "High", "Values"])

				xskillsCI.write(d_x_pd.style.to_latex())
				
				xskillsCI.write("\n")

	# code.interact("x by p - after computing confidence intervals: ", local=locals())



	# Plot - for MSE

	colors = ["red", "green", "blue", "orange"]

	# for each agent type
	for at in seenAgents:

		#code.interact("mse...",local=locals())

		# for each method
		for m in actualMethodsOnExps:

			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)

			
			# Skip pxskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			c = 0
			for b in buckets:
				if np.count_nonzero(mseDict[at]["perMethod"][m][b]) != 0:
					# print "b: ", b, "| color: ", colors[c] 
					# plt.plot(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = colors[c])
					plt.semilogx(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label=labelsB[b], c = colors[c])
				c += 1


			plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
			plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

			plt.margins(0.05)
			# plt.suptitle('Agent: ' + at + ' | MSE of Xskill Methods')

			fig.subplots_adjust(hspace= 1.0, wspace=1.0)

			# Shrink current axis by 10%
			# box = ax.get_position()
			# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

			elements = [Line2D([0],[0], color = colors[0],label = labelsB[buckets[0]]),
						Line2D([0],[0], color = colors[1], label = labelsB[buckets[1]]),
						Line2D([0],[0], color = colors[2], label = labelsB[buckets[2]]),
						Line2D([0],[0], color = colors[3], label = labelsB[buckets[3]])]
			
			# Put a legend to the right of the current axis
			plt.legend(handles = elements)#, loc='center left', bbox_to_anchor=(1, 0.5))

			# plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "mseXskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".png", bbox_inches='tight')
			# plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "mseXskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".pdf", bbox_inches='tight')
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "mseXskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".png", bbox_inches='tight')
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "mseXskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".pdf", bbox_inches='tight')

			plt.clf()
			plt.close("all")

def plotMSEpSkillsPerBucketsPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP):

	# BUCKETS PER XSKILLS -- SHOWING MSE FOR PSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0])+ "-xSkills"

	makeFolder(resultsFolder, "msePskills-PerBucketsPerAgentType")

	if domain == "1d":
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]

	cmap = plt.get_cmap("rainbow")
	norm = plt.Normalize(min(buckets),max(buckets))


	# init dict to store info
	mseDict = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": {}}

		for b in buckets:
			mseDict[at]["numAgents"][b] = 0.0


		for m in actualMethodsOnExps:
			# Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			mseDict[at]["perMethod"][m] = {}

			for b in buckets:
				mseDict[at]["perMethod"][m][b] = [0.0] * numStates # to store per state - across different exps per agent type


	resultsDict = {}


	for a in processedRFsAgentNames:

		aType, X, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if X <= buckets[b]:
				break

		# get actual bucket
		b = buckets[b]


		# update agent count
		mseDict[aType]["numAgents"][b] += 1
	

		# for each method
		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue


			# for each state
			for mxi in range(numStates):

				# mse percent
				sq = resultsDict[a]["mse_percent_pskills"][m][mxi]

				# store squared error
				mseDict[aType]["perMethod"][m][b][mxi] += sq


		del resultsDict[a]


	# normalize - find Mean Squared Error

	# for each agent type
	for at in seenAgents:

		# for each method
		for m in actualMethodsOnExps:

			# Skip pxskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:

				# for each state
				for mxi in range(numStates):

					#find MSE  = SE / # of agents
					mseDict[at]["perMethod"][m][b][mxi] /= mseDict[at]["numAgents"][b]


	# Plot - for MSE

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	# colors = ["red", "green", "blue", "orange"]


	# for each agent type
	for at in seenAgents:


		# for each method
		for m in actualMethodsOnExps:

			maxLast = -9999
			minLast = 9999
			

			# Skip pxskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			
			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)


			# ax = fig.add_subplot(1, 2, i)
			# ax.title.set_text('Method: ' + m)

			makePlot = False

			if domain == "sequentialDarts" and "Bounded" in at:
				makePlot = True


			if makePlot:
				loc = 1

				# axins = zoomed_inset_axes(ax1,zoom=1,loc=loc)
			
			c = 0
			for b in buckets:
				if np.count_nonzero(mseDict[at]["perMethod"][m][b]) != 0:
					# print "b: ", b, "| color: ", colors[c] 
					# color = colors[list(colors.keys())[c]]
					color = cmap(norm(b))
					# plt.plot(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
					plt.semilogx(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
				
					if makePlot:
						# axins.semilogx(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)

						if mseDict[at]["perMethod"][m][b][-1] > maxLast:
							maxLast = mseDict[at]["perMethod"][m][b][-1]

						if mseDict[at]["perMethod"][m][b][-1] < minLast:
							minLast = mseDict[at]["perMethod"][m][b][-1]

				c += 1

			'''
			if makePlot:

				startZoom = 900

				# subregion of the original image
				axins.set_xlim(startZoom,numStates)

				if minLast <= 0:
					minLast = -0.80
				else:
					minLast = minLast-(minLast*0.20)
				
				axins.set_ylim(minLast,maxLast+(maxLast*0.20))

				axins.set_xticklabels([])
				axins.set_xticklabels([],minor=True)
				plt.xticks(visible=False) 

				# mark_inset(ax1,axins,loc1=2,loc2=4,fc="none",ec="0.5")
			'''

			plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
			plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

			plt.margins(0.05)

			ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

			# plt.suptitle('Agent: ' + at + ' | MSE of Pskills Methods')

			fig.subplots_adjust(hspace= 1.0, wspace=1.0)

			# Shrink current axis by 10%
			# box = ax.get_position()
			# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

			elements = []

			for i in range(len(buckets)):
				# elements.append(Line2D([0],[0], color = colors[list(colors.keys())[i]],label = buckets[i]))
				elements.append(Line2D([0],[0], color = cmap(norm(buckets[i])),label = buckets[i]))

				
			# Put a legend to the right of the current axis
			plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))


			#fig.tight_layout()

			# plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".png", bbox_inches='tight')
			# plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".pdf", bbox_inches='tight')
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".png", bbox_inches='tight', pad_inches = 0)
			plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePskills-PerBucketsPerAgentType" + os.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".pdf", bbox_inches='tight', pad_inches = 0)

			plt.clf()
			plt.close("all")

def plotLastMSEAllBetasSamePlotAllAgent(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerMethod,betas):

	# Plot - For BM method
	#	X: Different betas
	# 	Y: MSE across agents last observation 

	saveAt = resultsFolder + os.sep + "plots" + os.sep + "BETAS" + os.sep + "LastMSE-AcrossAllAgents-AllBetasSamePlot" + os.sep 
	makeFolder3(saveAt)


	# code.interact("...", local=dict(globals(), **locals()))


	for tt in typeTargetsList:

		for plottingMethod in ["BM-MAP","BM-EES"]:

			try:
		
				makeFolder3(saveAt+plottingMethod)
				
				tempInfo = []

				for beta in betas:
					tempInfo.append(mseAcrossAllAgentsPerMethod[tt][plottingMethod][beta][-1])

				numObs = len(mseAcrossAllAgentsPerMethod[tt][plottingMethod][beta])

				fig = plt.figure(figsize = (10,10))
				ax = plt.subplot(111)

				plt.plot(betas,tempInfo,label=betas)
				plt.scatter(betas,tempInfo)
				
				plt.xlabel(r'\textbf{Betas}')
				plt.ylabel(r'\textbf{Mean squared error after '+str(numObs)+' observations}')
				plt.margins(0.05)
				plt.title(tt+" - "+plottingMethod)

				ax.tick_params(axis='x', labelrotation = 90)

				fileName = "results-"+tt+"-"+plottingMethod+"-AllAgents"
				plt.savefig(saveAt + os.sep + plottingMethod + os.sep + fileName + ".png", bbox_inches='tight')

				plt.clf()
				plt.close("all")

			except:
				continue

def plotLastMSEAllBetasSamePlotPerAgentType(methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,mseAcrossAllAgentsPerAgentType,betas):

	saveAt = resultsFolder + os.sep + "plots" + os.sep + "BETAS" + os.sep + "LastMSE-AllBetasSamePlotPerAgentType" + os.sep 
	makeFolder3(saveAt)

	
	# code.interact("...", local=dict(globals(), **locals()))


	for at in agentTypes:

		for tt in typeTargetsList:

			for plottingMethod in ["BM-MAP","BM-EES"]:
			
				try:
					makeFolder3(saveAt+plottingMethod)
					
					tempInfo = []

					for beta in betas:
						tempInfo.append(mseAcrossAllAgentsPerAgentType[at][tt][plottingMethod][beta][-1])

					numObs = len(mseAcrossAllAgentsPerAgentType[at][tt][plottingMethod][beta])

					fig = plt.figure(figsize = (10,10))
					ax = plt.subplot(111)

					plt.plot(betas,tempInfo,label=betas)
					plt.scatter(betas,tempInfo)
					
					plt.xlabel(r'\textbf{Betas}')
					plt.ylabel(r'\textbf{Mean squared error after '+str(numObs)+' observations}')
					plt.margins(0.05)
					plt.title(tt+" - "+plottingMethod+'- Agent: '+at)

					ax.tick_params(axis='x', labelrotation = 90)

					fileName = "results-"+tt+"-"+plottingMethod+"-Agent"+at
					plt.savefig(saveAt + os.sep + plottingMethod + os.sep + fileName + ".png", bbox_inches='tight')

					plt.clf()
					plt.close("all")

				except:
					continue

	#code.interact("after...", local=dict(globals(), **locals()))
	#######################################################################################

def func(x,a,b):
# def func(t, P0, P1, P2, P3):

	# binding curve
	# formula obtained from: https://stackoverflow.com/questions/49944018/fit-a-logarithmic-curve-to-data-points-and-extrapolate-out-in-numpy/49944478
	return (b*x)/((x+a)*1.0)

	# bezier curve
	# y = P0*(1-t)*(1-t)*(1-t) + 3*(1-t)*(1-t)*t*P1 + 3*(1-t)*t*t*P2 + t*t*t*P3
	# return y

def plotMSEAllBetasSamePlotPerAgentType(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerAgentType,betas):

	saveAt = resultsFolder + os.sep + "plots" + os.sep + "BETAS" + os.sep + "MSEAcrossAllAgents-AllBetasSamePlot-PerAgentType"+os.sep
	
	makeFolder3(saveAt)
	# makeFolder3(saveAt+"Plotly")

	for at in agentTypes:

		for xType in ["Log","NotLog"]:
		
			makeFolder3(saveAt+xType)
			# makeFolder3(saveAt+"Plotly"+os.sep+xType)

			for tt in typeTargetsList:

				for plottingMethod in ["BM-MAP","BM-EES"]:

					makeFolder3(saveAt+xType+os.sep+plottingMethod)
					# makeFolder3(saveAt+"Plotly"+os.sep+xType+os.sep+plottingMethod)
					
					fig = plt.figure(figsize = (20,30))

					ax = plt.subplot(2, 1, 1)

		 
					# cmap = plt.get_cmap("viridis")
					# norm = plt.Normalize(min(betas),max(betas))

					for method in methods:

						# only plotting BM methods
						if plottingMethod not in method:
							continue

						tempM, beta, tt = getInfoBM(method)
						#m = methodsDict[method]
						tempM2 = method.split("-Beta")[0]
						tempM3 = methodNamesPaper[method.split("-Beta")[0]]
						
						if xType == "Log":
							plt.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta], lw='2.0', label= str(beta))
						else:
							plt.plot(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta], lw='2.0', label= str(beta))

					ax.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
					ax.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
					plt.margins(0.05)

					plt.title(xType + '-- XSKILL -' + plottingMethod + ' | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

					# Put a legend to the right of the current axis
					ax.legend(loc='best')
					fig.canvas.mpl_connect("scroll_event", func)

					fileName = "results-"+xType+"-XSKILL-"+tt+"-"+plottingMethod+"-Agent"+at+"-"+domain
					plt.savefig(saveAt + xType + os.sep + plottingMethod + os.sep + fileName + ".png", bbox_inches='tight')


					# ~~~~~ PLOTLY ~~~~~
					'''
					data = []
					for method in methods:
						# only plotting BM methods
						if plottingMethod not in method:
							continue
						tempM, beta, tt = getInfoBM(method)
						#m = methodsDict[method]
						tempM2 = method.split("-Beta")[0]
						tempM3 = methodNamesPaper[method.split("-Beta")[0]]						
						if xType == "Log":
							d = go.Scatter(x=list(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta]))),y=mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta], color= str(beta))
						else:
							d = go.Scatter(x=list(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta]))),y=mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta], color= str(beta))
						data.append(d)
					# Create Plotly Plot -- Hosting offline
					# plotly_fig =  py.plot(data,auto_open=True)
					# plotly_fig['layout']['showlegend'] = True   
					# plotly_fig['layout']['autosize'] = True  
					# plotly_fig['layout']['height'] *= .80
					# plotly_fig['layout']['width'] *= .80
					# plotly_fig['layout']['margin']['t'] = 50
					# plotly_fig['layout']['margin']['l'] = 0
					fig = go.Figure(data)
					fig.update_layout(title_text=xType + '-- XSKILL -' + plottingMethod + ' | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')
					pio.write_html(fig, file=saveAt + "Plotly" + os.sep + xType + os.sep + plottingMethod + os.sep + fileName + ".html", auto_open=False)
					# plotly_fig['layout']['title'] = xType + '-- XSKILL -' + plottingMethod + ' | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments'
					# Save plotly
					# unique_url = py.offline.plot(plotly_fig, filename = saveAt + "Plotly" + os.sep + xType + os.sep + plottingMethod + os.sep + fileName + ".html", auto_open=False)
					'''
					plt.clf()
					plt.close("all")

def computeAndPlotMSEAcrossAllAgentsPerMethod(processedRFsAgentNames,methodsNames,methods,resultsFolder,agentTypes,numStates,domain,betas,givenBeta):

	# compute MSE across all agents for each method

	makeFolder(resultsFolder, "MSEAcrossAllAgents-PerMethod") 


	mseAcrossAllAgentsPerMethod = {}

	for tt in typeTargetsList:
		mseAcrossAllAgentsPerMethod[tt] = {}


	for m in methods:

		if "BM" in m:

			tempM, beta, tt = getInfoBM(m)

			if tempM not in mseAcrossAllAgentsPerMethod[tt]:
				mseAcrossAllAgentsPerMethod[tt][tempM] = {}

			if beta not in mseAcrossAllAgentsPerMethod[tt][tempM]:
				mseAcrossAllAgentsPerMethod[tt][tempM][beta] = [0.0] * numStates

		else:
			mseAcrossAllAgentsPerMethod[m] = [0.0] * numStates


	resultsDict = {}


	totalAgents = len(processedRFsAgentNames)

	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for s in range(numStates):
					mseAcrossAllAgentsPerMethod[tt][tempM][beta][s] += resultsDict[a]["plot_y"][tt][tempM][beta][s]
			else:

				for s in range(numStates):
					if "-pSkills" in m:
						mseAcrossAllAgentsPerMethod[m][s] += resultsDict[a]["mse_percent_pskills"][m][s]
					else:
						mseAcrossAllAgentsPerMethod[m][s] += resultsDict[a]["plot_y"][m][s]


		del resultsDict[a]


	# Will help set y axis limits
	maxXskillError = -99999
	maxPskillError = -99999

	# Normalize
	for m in methods:

		if m == "tn":
			continue

		if "BM" in m:
			tempM, beta, tt = getInfoBM(m)

			for s in range(numStates):
				mseAcrossAllAgentsPerMethod[tt][tempM][beta][s] /= (totalAgents*1.0)

				# Update max error seen - will use as y axis limit
				if mseAcrossAllAgentsPerMethod[tt][tempM][beta][s] > maxXskillError:
					maxXskillError = mseAcrossAllAgentsPerMethod[tt][tempM][beta][s]

		else:
			for s in range(numStates):
				mseAcrossAllAgentsPerMethod[m][s] /= (totalAgents*1.0)


				if "xSkills" in m or "OR" in m:
					# Update max error seen - will use as y axis limit
					if mseAcrossAllAgentsPerMethod[m][s] > maxXskillError:
						maxXskillError = mseAcrossAllAgentsPerMethod[m][s] 

				elif "pSkills" in m:
					# Update max error seen - will use as y axis limit
					if mseAcrossAllAgentsPerMethod[m][s] > maxPskillError:
						maxPskillError = mseAcrossAllAgentsPerMethod[m][s] 


	# For padding at the top of the y axis
	maxXskillError += (maxXskillError/10.0) 
	maxPskillError += (maxPskillError/10.0) 


	fig = plt.figure(figsize = (10,10))

	##################################### FOR XSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs

	ax1 = plt.subplot(2, 1, 1)

	for method in methods:

		# only plotting xskills methods
		if "pSkills" in method or "tn" in method:
			continue

		# Will just plot given beta
		if "BM" in method and str(givenBeta) in method:
			tempM, beta, tt = getInfoBM(method)
			#m = methodsDict[method]
			tempM2 = method.split("-Beta")[0]
			tempM3 = methodNamesPaper[method.split("-Beta")[0]]

			plt.semilogx(range(len(mseAcrossAllAgentsPerMethod[tt][tempM][givenBeta])),mseAcrossAllAgentsPerMethod[tt][tempM][givenBeta], lw='2.0', label= str(tempM3), c = methodsColors[tempM2])
			

		# other methods (and not other BM methods)         
		elif "BM" not in method:		
			#m = methodsDict[method]
			m = methodNamesPaper[method]
			plt.semilogx(range(len(mseAcrossAllAgentsPerMethod[method])),mseAcrossAllAgentsPerMethod[method], lw='2.0', label= str(m), c = methodsColors[method])


	ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
	ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
	plt.margins(0.05)
	plt.ylim(top = maxXskillError)

	#plt.title('XSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerMethod["totalNumExps"]) + ' experiments')

	# Put a legend to the right of the current axis
	ax1.legend(loc='best')
	fig.tight_layout()

	plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-PerMethod" + os.sep + "results-XSKILL"+domain+".png", bbox_inches='tight')
	plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-PerMethod" + os.sep + "results-XSKILL"+domain+".pdf", bbox_inches='tight')

	plt.clf()
	plt.close("all")


	##################################### FOR PSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs

	fig = plt.figure(figsize = (10,10))

	ax1 = plt.subplot(2, 1, 1)

	for method in methods:

		# only plotting xskills methods
		if "pSkills" not in method  or "tn" in method:
			continue


		#m = methodsDict[method]
		m = methodNamesPaper[method]

 
		# other methods          
		plt.semilogx(range(len(mseAcrossAllAgentsPerMethod[method])),mseAcrossAllAgentsPerMethod[method], lw='2.0', label= str(m), c = methodsColors[method])
		# plt.plot(range(len(mseAcrossAllAgentsPerMethod[method])),mseAcrossAllAgentsPerMethod[method], lw='2.0', label= str(m))

	ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
	ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
	plt.margins(0.05)
	plt.ylim(top = maxPskillError)

	#plt.title('PSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerMethod["totalNumExps"]) + ' experiments')

	# Put a legend to the right of the current axis
	ax1.legend(loc='best')
	fig.tight_layout()
	plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-PerMethod" + os.sep + "results-PSKILL-Agent"+"-"+domain+".png", bbox_inches='tight')
	plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-PerMethod" + os.sep + "results-PSKILL-Agent"+"-"+domain+".pdf", bbox_inches='tight')

	plt.clf()
	plt.close("all")


	plotLastMSEAllBetasSamePlotAllAgent(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerMethod,betas)
	
	# code.interact("...", local=dict(globals(), **locals()))

# MSE ACROSS ALL & SELECTED SUBSETS
def computeAndPlotMSEAcrossAllAgentsTypesAllMethods(processedRFsAgentNames,methodsNames,methods,resultsFolder,agentTypes,numStates,domain,betas,givenBeta,makeOtherPlots=False):

	makeFolder(resultsFolder, "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType")
	makeFolder(resultsFolder, "MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType")


	mseAcrossAllAgentsPerAgentType = {}
	stdInfoPerAgentTypePerMethod = {}
	stdPerAgentTypePerMethod = {}
	confidenceIntervals = {}

	mseSelectedAgents = {}
	stdInfoPerAgentTypePerMethodSelectedAgents = {}
	stdPerAgentTypePerMethodSelectedAgents = {}
	confidenceIntervalsSelectedAgents = {}

	selectedAgentTypes = ["Bounded","Flip","Tricker"]
	totalSelectedAgents = {}

	agentSubsets = ["NoRationals",
					"NearlyRational","NearlyRational","NearlyRational","NearlyRational","NearlyRational",
					"NearlyRational","NearlyRational",
					"MiddleAgents","MiddleAgents","MiddleAgents",
					"LessRational","NoBadAgents",
					"Agents","Agents","LessRational","LessRational","LessRational"]


	ops = [[">="],
			[">="],[">="],[">="],[">="],[">="],
			[">="],[">="],
			[">","<"],[">","<"],[">=","<"],
			["<="],[">"],
			[">="],["<"],["<="],["<="],["<="]]


	params = [[0.75],[0.95],[0.90],[0.85],[0.80],[0.75],
				[0.70],[0.65],
				[0.25,0.75],[0.25,0.50],[0.50,0.75],
				[0.25],[0.25],
				[0.50],[0.50],[0.15],[0.10],[0.05]]

	names = []

	for x in range(len(agentSubsets)):
		
		sub = agentSubsets[x]

		if len(ops[x]) == 1:
			name = f"{sub}{ops[x][0]}{params[x][0]}"
		else:
			name = f"{sub}{ops[x][0]}{params[x][0]}and{ops[x][1]}{params[x][1]}"

		names.append(name)


	# Set up dict to save info
	for at in agentTypes:
		mseAcrossAllAgentsPerAgentType[at] = {"numAgents": 0.0, "totalNumExps": 0.0}
		
		# Initialize to 0 for all, will ignore not selected agents
		totalSelectedAgents[at] = {}

		for x in range(len(agentSubsets)):
			totalSelectedAgents[at][names[x]] = 0


		stdInfoPerAgentTypePerMethod[at] = {}
		stdPerAgentTypePerMethod[at] = {}
		confidenceIntervals[at] = {}

		if at in selectedAgentTypes:
			mseSelectedAgents[at] = {}

			stdInfoPerAgentTypePerMethodSelectedAgents[at] = {}
			stdPerAgentTypePerMethodSelectedAgents[at] = {}
			confidenceIntervalsSelectedAgents[at] = {}

			for x in range(len(agentSubsets)):
				mseSelectedAgents[at][names[x]] = {"numAgents": 0.0, "totalNumExps": 0.0}
				stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]] = {}
				stdPerAgentTypePerMethodSelectedAgents[at][names[x]] = {}
				confidenceIntervalsSelectedAgents[at][names[x]] = {}

		for tt in typeTargetsList:
			mseAcrossAllAgentsPerAgentType[at][tt] = {}

			stdInfoPerAgentTypePerMethod[at][tt] = {}
			stdPerAgentTypePerMethod[at][tt] = {}
			confidenceIntervals[at][tt] = {}

			if at in selectedAgentTypes:
				for x in range(len(agentSubsets)):
					mseSelectedAgents[at][names[x]][tt] = {}

					stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][tt] = {}
					stdPerAgentTypePerMethodSelectedAgents[at][names[x]][tt] = {}
					confidenceIntervalsSelectedAgents[at][names[x]][tt] = {}

		for m in methods:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				if tempM not in mseAcrossAllAgentsPerAgentType[at][tt]:
					mseAcrossAllAgentsPerAgentType[at][tt][tempM] = {}
					stdInfoPerAgentTypePerMethod[at][tt][tempM] = {}
					stdPerAgentTypePerMethod[at][tt][tempM] = {}
					confidenceIntervals[at][tt][tempM] = {}
	
					if at in selectedAgentTypes:
						for x in range(len(agentSubsets)):
							mseSelectedAgents[at][names[x]][tt][tempM] = {}
							stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM] = {}
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM] = {}
							confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM] = {}

				if beta not in mseAcrossAllAgentsPerAgentType[at][tt][tempM]:
					mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta] = [0.0] * numStates
					stdInfoPerAgentTypePerMethod[at][tt][tempM][beta] = [] 
					stdPerAgentTypePerMethod[at][tt][tempM][beta] = 0.0
					confidenceIntervals[at][tt][tempM][beta] = {"low": 0.0, "high": 0.0, "value": 0.0}
	
					if at in selectedAgentTypes:
						for x in range(len(agentSubsets)):
							mseSelectedAgents[at][names[x]][tt][tempM][beta] = [0.0] * numStates	
							stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM][beta] = [] 
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM][beta] = 0.0
							confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta] = {"low": 0.0, "high": 0.0, "value": 0.0}

				
			else:
				mseAcrossAllAgentsPerAgentType[at][m] = [0.0] * numStates
				stdInfoPerAgentTypePerMethod[at][m] = [] 
				stdPerAgentTypePerMethod[at][m] = 0.0
				confidenceIntervals[at][m] = {"low": 0.0, "high": 0.0, "value": 0.0}

				if at in selectedAgentTypes:
					for x in range(len(agentSubsets)):
						mseSelectedAgents[at][names[x]][m] = [0.0] * numStates		
						stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = [] 
						stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = 0.0
						confidenceIntervalsSelectedAgents[at][names[x]][m] = {"low": 0.0, "high": 0.0, "value": 0.0}


	#####################################################################################################
	
	saveTo = resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + "info.pickle"
	loaded = False


	# Try to load info
	if os.path.exists(saveTo):

		try:
			print("Loading info...")

			with open(saveTo,"rb") as file:
				info = pickle.load(file)

			loaded = True

			mseAcrossAllAgentsPerAgentType = info["mseAcrossAllAgentsPerAgentType"]
			stdInfoPerAgentTypePerMethod = info["stdInfoPerAgentTypePerMethod"]
			stdPerAgentTypePerMethod = info["stdPerAgentTypePerMethod"]
			confidenceIntervals = info["confidenceIntervals"]

			mseSelectedAgents = info["mseSelectedAgents"]
			stdInfoPerAgentTypePerMethodSelectedAgents = info["stdInfoPerAgentTypePerMethodSelectedAgents"]
			stdPerAgentTypePerMethodSelectedAgents = info["stdPerAgentTypePerMethodSelectedAgents"]
			confidenceIntervalsSelectedAgents = info["confidenceIntervalsSelectedAgents"]

		except:
			loaded = False


	if not loaded:

		print("Computing info...")

		resultsDict = {}

		'''
		ii = 0

		allXs = {}
		allPs = {}
		allPercents = {}

		for at in selectedAgentTypes:
			allXs[at] = []
			allPs[at] = []
			allPercents[at] = []
		'''


		# Compute mse across all agents of same type
		for a in processedRFsAgentNames:

			aType, x, p = getParamsFromAgentName(a)

			# print(f"{ii}/{len(processedRFsAgentNames)} - Agent: {a}")
			# ii += 1

			# Load processed info		
			resultsDict[a] = loadProcessedInfo(prdFile,a)

			percent_trueP = resultsDict[a]["percentTrueP"]

			'''
			if aType in selectedAgentTypes:
				allXs[aType].append(x)
				allPs[aType].append(p)
				allPercents[aType].append(percent_trueP)
			'''

			#####################################################################
			# Determine if agent fits any of the subsets or not
			#####################################################################
			
			selected = []
					
			for li in range(len(agentSubsets)):

				if aType in selectedAgentTypes:

					if len(ops[li]) == 1:
						# print(f"{percent_trueP}{ops[li][0]}{params[li][0]}")
						result = eval(f"{percent_trueP}{ops[li][0]}{params[li][0]}")
					else:
						# print(f"{percent_trueP}{ops[li][0]}{params[li][0]} and {percent_trueP}{ops[li][1]}{params[li][1]}")
						result = eval(f"{percent_trueP}{ops[li][0]}{params[li][0]} and {percent_trueP}{ops[li][1]}{params[li][1]}")
					
					# print(result)
					# print()
					selected.append(result)

					if result:
						totalSelectedAgents[aType][names[li]] += 1
				else:
					selected.append(False)	

			#####################################################################	

			# del resultsDict[a]
			# continue

			for m in methods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)

					for s in range(numStates):
						mseAcrossAllAgentsPerAgentType[aType][tt][tempM][beta][s] += resultsDict[a]["plot_y"][tt][tempM][beta][s]

						for x in range(len(agentSubsets)):
							if selected[x]:
								mseSelectedAgents[aType][names[x]][tt][tempM][beta][s] += resultsDict[a]["plot_y"][tt][tempM][beta][s]

					# Get data for standard deviation across all agents of same type --- for last state
					stdInfoPerAgentTypePerMethod[aType][tt][tempM][beta].append(resultsDict[a]["plot_y"][tt][tempM][beta][-1])
					
					for x in range(len(agentSubsets)):
						if selected[x]:
							stdInfoPerAgentTypePerMethodSelectedAgents[aType][names[x]][tt][tempM][beta].append(resultsDict[a]["plot_y"][tt][tempM][beta][-1])

				else:

					for s in range(numStates):
						if "-pSkills" in m:
							mseAcrossAllAgentsPerAgentType[aType][m][s] += resultsDict[a]["mse_percent_pskills"][m][s]
							
							for x in range(len(agentSubsets)):
								if selected[x]:
									mseSelectedAgents[aType][names[x]][m][s] += resultsDict[a]["mse_percent_pskills"][m][s]							
						else:
							mseAcrossAllAgentsPerAgentType[aType][m][s] += resultsDict[a]["plot_y"][m][s]
							
							for x in range(len(agentSubsets)):
								if selected[x]:
									mseSelectedAgents[aType][names[x]][m][s] += resultsDict[a]["plot_y"][m][s]


					# Get data for standard deviation across all agents of same type --- for last state
					if "-pSkills" in m:
						stdInfoPerAgentTypePerMethod[aType][m].append(resultsDict[a]["mse_percent_pskills"][m][-1])
						for x in range(len(agentSubsets)):
							if selected[x]:
								stdInfoPerAgentTypePerMethodSelectedAgents[aType][names[x]][m].append(resultsDict[a]["mse_percent_pskills"][m][-1])
					else:
						stdInfoPerAgentTypePerMethod[aType][m].append(resultsDict[a]["plot_y"][m][-1])
						for x in range(len(agentSubsets)):
							if selected[x]:
								stdInfoPerAgentTypePerMethodSelectedAgents[aType][names[x]][m].append(resultsDict[a]["plot_y"][m][-1])


			mseAcrossAllAgentsPerAgentType[aType]["numAgents"] += 1.0
			mseAcrossAllAgentsPerAgentType[aType]["totalNumExps"] += resultsDict[a]["num_exps"]


			for x in range(len(agentSubsets)):
				if selected[x]:
					mseSelectedAgents[aType][names[x]]["numAgents"] += 1.0
					mseSelectedAgents[aType][names[x]]["totalNumExps"] += resultsDict[a]["num_exps"]


			del resultsDict[a]


		'''
		for at in selectedAgentTypes:
			allXs[at] = np.array(allXs[at])
			allPs[at] = np.array(allPs[at])
			allPercents[at] = np.array(allPercents[at])

		code.interact("...", local=dict(globals(), **locals()))
		'''

		# Normalize
		for at in agentTypes:
			for m in methods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)

					for s in range(numStates):
						mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
						
						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:
								mseSelectedAgents[at][names[x]][tt][tempM][beta][s] /= (mseSelectedAgents[at][names[x]]["numAgents"] * 1.0)

					stdPerAgentTypePerMethod[at][tt][tempM][beta] = np.std(stdInfoPerAgentTypePerMethod[at][tt][tempM][beta])
					
					for x in range(len(agentSubsets)):
						if totalSelectedAgents[at][names[x]] != 0:
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM][beta] = np.std(stdInfoPerAgentTypePerMethod[at][tt][tempM][beta])

				else:
					for s in range(numStates):
						mseAcrossAllAgentsPerAgentType[at][m][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
						
						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:
								mseSelectedAgents[at][names[x]][m][s] /= (mseSelectedAgents[at][names[x]]["numAgents"] * 1.0)

					stdPerAgentTypePerMethod[at][m] = np.std(stdInfoPerAgentTypePerMethod[at][m])
			
					for x in range(len(agentSubsets)):
						if totalSelectedAgents[at][names[x]] != 0:
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = np.std(stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][m])


		# SAVE TO FILE

		toSave = {}
		toSave["mseAcrossAllAgentsPerAgentType"] = mseAcrossAllAgentsPerAgentType
		toSave["stdInfoPerAgentTypePerMethod"] = stdInfoPerAgentTypePerMethod
		toSave["stdPerAgentTypePerMethod"] = stdPerAgentTypePerMethod
		toSave["confidenceIntervals"] = confidenceIntervals

		toSave["mseSelectedAgents"] = mseSelectedAgents
		toSave["stdInfoPerAgentTypePerMethodSelectedAgents"] = stdInfoPerAgentTypePerMethodSelectedAgents
		toSave["stdPerAgentTypePerMethodSelectedAgents"] = stdPerAgentTypePerMethodSelectedAgents
		toSave["confidenceIntervalsSelectedAgents"] = confidenceIntervalsSelectedAgents


		with open(saveTo,"wb") as outfile:
			pickle.dump(toSave, outfile)

	



	# To set y axis limits
	maxXskillError = {}
	maxPskillError = {}
	minXskillError = {}
	minPskillError = {}

	maxXskillErrorSelectedAgents = {}
	maxPskillErrorSelectedAgents = {}
	minXskillErrorSelectedAgents = {}
	minPskillErrorSelectedAgents = {}

	for x in range(len(agentSubsets)):

		maxXskillErrorSelectedAgents[names[x]] = {}
		maxPskillErrorSelectedAgents[names[x]] = {}
		
		minXskillErrorSelectedAgents[names[x]] = {}
		minPskillErrorSelectedAgents[names[x]] = {}


	for i in range(len(methodsLists)):

		tempMethods = methodsLists[i]
		label = labels[i]

		maxXskillError[label] = -999999
		maxPskillError[label] = -999999

		minXskillError[label] = 999999
		minPskillError[label] = 999999

		for x in range(len(agentSubsets)):

			maxXskillErrorSelectedAgents[names[x]][label] = -999999
			maxPskillErrorSelectedAgents[names[x]][label] = -999999

			minXskillErrorSelectedAgents[names[x]][label] = 999999
			minPskillErrorSelectedAgents[names[x]][label] = 999999


		for at in agentTypes:
			
			for m in tempMethods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)

					if ("GivenBeta" in label and beta == givenBeta) or \
						(("ALL-Beta" in label or "BM-Beta" in label) and str(beta) in label) or \
						label == "JustBM":

						tempMax = np.max(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:])
						tempMin = np.min(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:])

						# Update max error seen - will use as y axis limit
						if tempMax > maxXskillError[label]:
							maxXskillError[label] = tempMax

						if tempMin < minXskillError[label]:
							minXskillError[label] = tempMin 


						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:

								tempMaxSelected = np.max(mseSelectedAgents[at][names[x]][tt][tempM][beta][1:])
								tempMinSelected = np.min(mseSelectedAgents[at][names[x]][tt][tempM][beta][1:])
							
								if tempMaxSelected > maxXskillErrorSelectedAgents[names[x]][label]:
									maxXskillErrorSelectedAgents[names[x]][label] = tempMaxSelected

								if tempMinSelected < minXskillErrorSelectedAgents[names[x]][label]:
									minXskillErrorSelectedAgents[names[x]][label] = tempMinSelected

				else:


					if "pSkills" in m:
						
						tempMax = np.max(mseAcrossAllAgentsPerAgentType[at][m])
						tempMin = np.min(mseAcrossAllAgentsPerAgentType[at][m])
						
						# Update max error seen - will use as y axis limit
						if tempMax > maxPskillError[label]:
							maxPskillError[label] = tempMax 

						if tempMin < minPskillError[label]:
							minPskillError[label] = tempMin 


						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:

								tempMaxSelected = np.max(mseSelectedAgents[at][names[x]][m])
								tempMinSelected = np.min(mseSelectedAgents[at][names[x]][m])

								if tempMaxSelected > maxPskillErrorSelectedAgents[names[x]][label]:
									maxPskillErrorSelectedAgents[names[x]][label] = tempMaxSelected 

								if tempMinSelected < minPskillErrorSelectedAgents[names[x]][label]:
									minPskillErrorSelectedAgents[names[x]][label] = tempMinSelected 

					else:

						tempMax = np.max(mseAcrossAllAgentsPerAgentType[at][m][1:])
						tempMin = np.min(mseAcrossAllAgentsPerAgentType[at][m][1:])

						# Update max error seen - will use as y axis limit
						if tempMax > maxXskillError[label]:
							maxXskillError[label] = tempMax 

						if tempMin < minXskillError[label]:
							minXskillError[label] = tempMin 


						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:
								tempMaxSelected = np.max(mseSelectedAgents[at][names[x]][m][1:])
								tempMinSelected = np.min(mseSelectedAgents[at][names[x]][m][1:])

								if tempMaxSelected > maxXskillErrorSelectedAgents[names[x]][label]:
									maxXskillErrorSelectedAgents[names[x]][label] = tempMaxSelected 

								if tempMinSelected < minXskillErrorSelectedAgents[names[x]][label]:
									minXskillErrorSelectedAgents[names[x]][label] = tempMinSelected 


	#####################################################################################################

	# '''
	# Add padding
	percent = 0.01


	# TO ADD PADDING
	for i in range(len(methodsLists)):
		
		label = labels[i]

		maxXskillError[label] = maxXskillError[label] + percent*(maxXskillError[label]-minXskillError[label])
		minXskillError[label] = minXskillError[label] - percent*(maxXskillError[label]-minXskillError[label])

		maxPskillError[label] = maxPskillError[label] + percent*(maxPskillError[label]-minPskillError[label])
		minPskillError[label] = minPskillError[label] - percent*(maxPskillError[label]-minPskillError[label])


		for x in range(len(agentSubsets)):

			maxXskillErrorSelectedAgents[names[x]][label] = maxXskillErrorSelectedAgents[names[x]][label] + percent*(maxXskillErrorSelectedAgents[names[x]][label]-minXskillErrorSelectedAgents[names[x]][label])
			minXskillErrorSelectedAgents[names[x]][label] = minXskillErrorSelectedAgents[names[x]][label] - percent*(maxXskillErrorSelectedAgents[names[x]][label]-minXskillErrorSelectedAgents[names[x]][label])			

			maxPskillErrorSelectedAgents[names[x]][label] = maxPskillErrorSelectedAgents[names[x]][label] + percent*(maxPskillErrorSelectedAgents[names[x]][label]-minPskillErrorSelectedAgents[names[x]][label])
			minPskillErrorSelectedAgents[names[x]][label] = minPskillErrorSelectedAgents[names[x]][label] - percent*(maxPskillErrorSelectedAgents[names[x]][label]-minPskillErrorSelectedAgents[names[x]][label])
	# '''			

	# code.interact("...", local=dict(globals(), **locals()))


	#####################################################################################################
	# COMPUTE CONFIDENCE INTERVALS
	#####################################################################################################
	
	if not loaded:
		ci = 0.95
		
		# For 95% interval
		Z = 1.960

		for at in agentTypes:
		
			N = mseAcrossAllAgentsPerAgentType[at]["numAgents"]
			
			for m in methods:

				if m == "tn":
					continue

				if "BM" in m:

					tempM, beta, tt = getInfoBM(m)

					mu = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1]
					sigma = stdPerAgentTypePerMethod[at][tt][tempM][beta]

					confidenceIntervals[at][tt][tempM][beta]["low"], confidenceIntervals[at][tt][tempM][beta]["high"] =\
					stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

					confidenceIntervals[at][tt][tempM][beta]["value"] = Z * (sigma/np.sqrt(N))

				else:

					mu = mseAcrossAllAgentsPerAgentType[at][m][-1]
					sigma = stdPerAgentTypePerMethod[at][m]

					confidenceIntervals[at][m]["low"], confidenceIntervals[at][m]["high"] =\
					stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

					confidenceIntervals[at][m]["value"] = Z * (sigma/np.sqrt(N))
				
		#####################################################################################################

		xskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + "confidenceIntervals-xSkills.txt", "a")
		pskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + "confidenceIntervals-pSkills.txt", "a")

		# Save info to text files
		for at in agentTypes:

			# Gather all MSE's of current method
			allMseCurrentMethod = []
			tempSeenMethods = []

			for m in methods:

				if m == "tn":
					continue

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					allMseCurrentMethod.append(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1])
				else:
					allMseCurrentMethod.append(mseAcrossAllAgentsPerAgentType[at][m][-1])
				
				tempSeenMethods.append(m)

			# order from highest to lowest MSE
			orderedMSE = sorted(allMseCurrentMethod, reverse = True)


			d_x = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}
			d_p = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}

			# Output info to files
			for mseO in range(len(orderedMSE)):

				index = allMseCurrentMethod.index(orderedMSE[mseO])

				m = tempSeenMethods[index]

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					# Translate to paper's name
					mm = methodNamesPaper[m.split("-Beta")[0]]
				else:
					# Translate to paper's name
					mm = methodNamesPaper[m]


				### for pskills 
				if "-pSkills" in m:
					pskillsCI.write("Agent: " + at + " |   Method: " + str(m) + \
							" ->  Low: " + f"{confidenceIntervals[at][m]['low']:.4f}" +\
							" | High: " + f"{confidenceIntervals[at][m]['high']:.4f}" +\
							" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" + \
							" | Value: " + f"{confidenceIntervals[at][m]['value']:.4f}\n")

					d_p["Agents"].append(at)
					d_p["Methods"].append(mm)
					d_p["MSE"].append(round(orderedMSE[mseO],4))
					d_p["Low"].append(round(confidenceIntervals[at][m]["low"],4))
					d_p["High"].append(round(confidenceIntervals[at][m]["high"],4))
					d_p["Values"].append(round(confidenceIntervals[at][m]["value"],4))

				### for xskills
				else:

					if "BM" in m:
						xskillsCI.write("Agent: " + at + " | Method: " + str(m) + "-" + tt + " | Beta: " + str(beta) +\
								" ->  Low: " + f"{confidenceIntervals[at][tt][tempM][beta]['low']:.4f}" +\
								" | High: " + f"{confidenceIntervals[at][tt][tempM][beta]['high']:.4f}" +\
								" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
								" | Value: " + f"{confidenceIntervals[at][tt][tempM][beta]['value']:.4f}" + "\n")

						d_x["Agents"].append(at)
						d_x["Methods"].append(mm)
						d_x["MSE"].append(round(orderedMSE[mseO],4))
						d_x["Low"].append(round(confidenceIntervals[at][tt][tempM][beta]["low"],4))
						d_x["High"].append(round(confidenceIntervals[at][tt][tempM][beta]["high"],4))
						d_x["Values"].append(round(confidenceIntervals[at][tt][tempM][beta]["value"],4))

					else:
						xskillsCI.write("Agent: " + at + " | Method: " + str(m) + \
								" ->  Low: " + f"{confidenceIntervals[at][m]['low']:.4f}" +\
								" | High: " + f"{confidenceIntervals[at][m]['high']:.4f}" +\
								" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
								" | Value: " + f"{confidenceIntervals[at][m]['value']:.4f}" + "\n")

						d_x["Agents"].append(at)
						d_x["Methods"].append(mm)
						d_x["MSE"].append(round(orderedMSE[mseO],4))
						d_x["Low"].append(round(confidenceIntervals[at][m]["low"],4))
						d_x["High"].append(round(confidenceIntervals[at][m]["high"],4))
						d_x["Values"].append(round(confidenceIntervals[at][m]["value"],4))


			pskillsCI.write("\n")
			xskillsCI.write("\n")

			pskillsCI.write("\n")
			xskillsCI.write("\n")

			 # Convert dicts to pandas dataframe
			d_x_pd = pd.DataFrame(d_x, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])
			d_p_pd = pd.DataFrame(d_p, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])

			xskillsCI.write(d_x_pd.style.to_latex())
			pskillsCI.write(d_p_pd.style.to_latex())

			pskillsCI.write("\n")
			xskillsCI.write("\n")

		# code.interact("here...", local=dict(globals(), **locals()))

		stdInfoPerAgentTypePerMethod.clear()
		stdPerAgentTypePerMethod.clear()
		confidenceIntervals.clear()

		del stdInfoPerAgentTypePerMethod
		del stdPerAgentTypePerMethod
		del confidenceIntervals


		# NOW FOR SELECTED AGENTS

		for at in mseSelectedAgents:

			for x in range(len(agentSubsets)):

				sub = agentSubsets[x]

				if totalSelectedAgents[at][names[x]] != 0:

					for i in range(len(methodsLists)):		

						if len(ops[x]) == 1:
							tempFolder = f"MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType{os.sep}{sub}{ops[x][0]}{params[x][0]}{os.sep}"
						else:
							tempFolder = f"MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType{os.sep}{sub}{ops[x][0]}{params[x][0]}and{ops[x][1]}{params[x][1]}{os.sep}"

						makeFolder(resultsFolder,tempFolder)
						makeFolder(resultsFolder,tempFolder+labels[i])

						tempMethods = methodsLists[i]

						allBetas = False

						if labels[i] == "JustBM":
							allBetas = True

						if len(tempMethods) == 0:
							continue

						N = mseSelectedAgents[at][names[x]]["numAgents"]
						
						for m in tempMethods:

							if m == "tn":
								continue

							if "BM" in m:

								tempM, beta, tt = getInfoBM(m)

								mu = mseSelectedAgents[at][names[x]][tt][tempM][beta][-1]
								sigma = stdPerAgentTypePerMethodSelectedAgents[at][names[x]][tt][tempM][beta]

								confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["low"], confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["high"] =\
								stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

								confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["value"] = Z * (sigma/np.sqrt(N))

							else:

								mu = mseSelectedAgents[at][names[x]][m][-1]
								sigma = stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m]

								confidenceIntervalsSelectedAgents[at][names[x]][m]["low"], confidenceIntervalsSelectedAgents[at][names[x]][m]["high"] =\
								stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

								confidenceIntervalsSelectedAgents[at][names[x]][m]["value"] = Z * (sigma/np.sqrt(N))
							

						# Save info to text files
						xskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + tempFolder + os.sep + labels[i] + os.sep + "confidenceIntervals-xSkills.txt", "a")
						pskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + tempFolder + os.sep + labels[i] + os.sep + "confidenceIntervals-pSkills.txt", "a")

						# Gather all MSE's of current method
						allMseCurrentMethod = []
						tempSeenMethods = []

						for m in tempMethods:

							if m == "tn":
								continue

							if "BM" in m:
								tempM, beta, tt = getInfoBM(m)
								allMseCurrentMethod.append(mseSelectedAgents[at][names[x]][tt][tempM][beta][-1])
							else:
								allMseCurrentMethod.append(mseSelectedAgents[at][names[x]][m][-1])
						
							tempSeenMethods.append(m)

						# order from highest to lowest MSE
						orderedMSE = sorted(allMseCurrentMethod, reverse = True)


						d_x = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}
						d_p = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}

						# Output info to files
						for mseO in range(len(orderedMSE)):

							index = allMseCurrentMethod.index(orderedMSE[mseO])
							m = tempSeenMethods[index]

							if m == "tn":
								continue


							if "BM" in m:
								tempM, beta, tt = getInfoBM(m)
								# Translate to paper's name
								mm = methodNamesPaper[m.split("-Beta")[0]]
							else:
								# Translate to paper's name
								mm = methodNamesPaper[m]


							### for pskills 
							if "-pSkills" in m:
								pskillsCI.write("Agent: " + at + " |   Method: " + str(m) + \
										" ->  Low: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["low"],4)) +\
										" | High: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["high"],4)) +\
										" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
										" | Value: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["value"],4)) + "\n")

								d_p["Agents"].append(at)
								d_p["Methods"].append(mm)
								d_p["MSE"].append(round(orderedMSE[mseO],2))
								d_p["Low"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["low"],2))
								d_p["High"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["high"],2))
								d_p["Values"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["value"],2))

							### for xskills
							else:

								if "BM" in m:
									xskillsCI.write("Agent: " + at + " | Method: " + str(m) + "-" + tt + " | Beta: " + str(beta) +\
											" ->  Low: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["low"],4)) +\
											" | High: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["high"],4)) +\
											" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
											" | Value: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["value"],4)) + "\n")

									d_x["Agents"].append(at)
									d_x["Methods"].append(mm)
									d_x["MSE"].append(round(orderedMSE[mseO],2))
									d_x["Low"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["low"],2))
									d_x["High"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["high"],2))
									d_x["Values"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][tt][tempM][beta]["value"],2))

								else:
									xskillsCI.write("Agent: " + at + " | Method: " + str(m) + \
											" ->  Low: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["low"],4)) +\
											" | High: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["high"],4)) +\
											" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
											" | Value: " + str(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["value"],4)) + "\n")

									d_x["Agents"].append(at)
									d_x["Methods"].append(mm)
									d_x["MSE"].append(round(orderedMSE[mseO],2))
									d_x["Low"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["low"],2))
									d_x["High"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["high"],2))
									d_x["Values"].append(round(confidenceIntervalsSelectedAgents[at][names[x]][m]["value"],2))


						pskillsCI.write("\n")
						xskillsCI.write("\n")

						pskillsCI.write("\n")
						xskillsCI.write("\n")

						 # Convert dicts to pandas dataframe
						d_x_pd = pd.DataFrame(d_x, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])
						d_p_pd = pd.DataFrame(d_p, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])

						xskillsCI.write(d_x_pd.style.to_latex())
						pskillsCI.write(d_p_pd.style.to_latex())

						pskillsCI.write("\n")
						xskillsCI.write("\n")

						# code.interact("here...", local=dict(globals(), **locals()))

			stdInfoPerAgentTypePerMethodSelectedAgents[at].clear()
			stdPerAgentTypePerMethodSelectedAgents[at].clear()
			confidenceIntervalsSelectedAgents[at].clear()

		del stdInfoPerAgentTypePerMethodSelectedAgents
		del stdPerAgentTypePerMethodSelectedAgents
		del confidenceIntervalsSelectedAgents



	#####################################################################################################
	# PLOTS!
	#####################################################################################################

	print("Creating plots....")

	for at in agentTypes:

		for i in range(len(methodsLists)):

			makeFolder(resultsFolder,"MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType"+os.sep+labels[i])

			tempMethods = methodsLists[i]

			allBetas = False

			if labels[i] == "JustBM":
				allBetas = True

			if len(tempMethods) == 0:
				continue

			##################################### FOR XSKILLS #####################################
			# create plot for each one of the different agents - estimates vs obs

			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2,1,1)
			# lines = []
			# tempLabels = []

			makePlot = False


			'''
			'upper right'  : 1,
			'upper left'   : 2,
			'lower left'   : 3,
			'lower right'  : 4,
			'right'        : 5,
			'center left'  : 6,
			'center right' : 7,
			'lower center' : 8,
			'upper center' : 9,
			'center'       : 10
			'''

			'''
			if ("ALL" in labels[i] or "OR-BM-Beta" in labels[i] or "JustEES" in labels[i]):

				makePlot = True

				if (domain == "2d" and "Target" not in at) or \
					(domain == "sequentialDarts") or \
					(("Bounded" in at or "Flip" in at) and domain == "1d") or \
					("OR-BM" in labels[i] and "Target" in at):
					makePlot = False

				if makePlot:
					loc = 1

					# x0,y0,w,h.
					# axins = ax1.inset_axes([np.log10(startZoom),0,1000-startZoom,7000])#maxXskillError[labels[i]]])
					axins = zoomed_inset_axes(ax1,zoom=20,loc=loc,borderpad=1)
			'''
			

			# DISABLE CREATION OF INNER PLOTS
			makePlot = False


			maxLast = -9999
			minLast = 9999

			c = 0

			for method in tempMethods:

				# only plotting xskills methods
				if "pSkills" in method or "DomainTargets" in method:
					continue

				c += 1

				# Will just plot given beta
				if "BM" in method:
					tempM, beta, tt = getInfoBM(method)
					#m = methodsDict[method]
					tempM2 = method.split("-Beta")[0]
					tempM3 = methodNamesPaper[method.split("-Beta")[0]]


					label = str(tempM3)+"-"+str(beta)

					if (allBetas or str(beta) in labels[i] or "SelectedBetas" in labels[i]) and "GivenBeta" not in labels[i]:
						if allBetas:
							if "EES" in method:
								line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:], lw='2.0', label= label)
		
						else:

							if "JustEES" in labels[i]:
								label = label.replace("-ES","")
	
							color = None

							if "SelectedBetas" in labels[i]:
								if beta == 0.75:
									cc = "C6"
								elif beta == 0.85:
									cc = "C2"
								else:
									cc = "C9"

								color = cc
								line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:], lw='2.0', label= label,c = cc,ls=lineStylesPaper[tempM2])		
							else:
								color =  methodsColors[tempM2]
								line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:], lw='2.0', label= label,c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])						
							
							if makePlot:
								axins.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][1:], lw='2.0', label= label,c = color,ls=lineStylesPaper[tempM2])
								
								if "Target" in at and "ES" in m: # Only allow TBA for Target
									if mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1] > maxLast:
										maxLast = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1]

									if mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1] < minLast:
										minLast = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1]

						# lines.append(line)
						# tempLabels.append(label)

					else:
						if beta == givenBeta:
							line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta][1:], lw='2.0', label= str(tempM3), c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])
							# lines.append(line)
							# tempLabels.append(label)
							
							if makePlot:
								axins.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta][1:], lw='2.0', label= str(tempM3), c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])

								if "Target" in at and "ES" in m: # Only allow TBA for Target
									if mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1] > maxLast:
										maxLast = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1]

									if mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1] < minLast:
										minLast = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1]

				# other methods (and not other BM methods)         
				elif "BM" not in method:		
					#m = methodsDict[method]
					m = methodNamesPaper[method]
					label = str(m)

					if "JustEES" in labels[i]:
						label = label.replace("-ES","")

					line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method][1:], lw='2.0', label= label, c = methodsColors[method], ls= lineStylesPaper[method])
					
					# lines.append(line)
					# tempLabels.append(label)

					if makePlot:
						axins.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method][1:], lw='2.0', label= label, c = methodsColors[method], ls= lineStylesPaper[method])
						# print(method)
						if "OR" not in method or \
							(("OR" in method and "Target" in at) and domain != "sequentialDarts") or\
							(domain != "1d" and "Bounded" not in at and "Flip" not in at and domain != "sequentialDarts"):
							
							if mseAcrossAllAgentsPerAgentType[at][method][-1] > maxLast:
								maxLast = mseAcrossAllAgentsPerAgentType[at][method][-1]

							if mseAcrossAllAgentsPerAgentType[at][method][-1] < minLast:
								minLast = mseAcrossAllAgentsPerAgentType[at][method][-1]


			if c != 0:
				ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
				ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
				#ax1.set_xscale('symlog')
				plt.ylim(top = maxXskillError[labels[i]],bottom = minXskillError[labels[i]])
				plt.margins(0.05)

					
				if makePlot:
					startZoom = 900

					# subregion of the original image
					axins.set_xlim(startZoom,1000)

					# < 0 to add padding on plot
					# minZoomYLim = -0.90

					if minLast <= 0:
						minLast = -0.80
					else:
						minLast = minLast-(minLast*0.20)
					
					axins.set_ylim(minLast,maxLast+(maxLast*0.20))

					if domain == "sequentialDarts":
						#plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
						axins.locator_params(nbins=2, axis='y')

					axins.set_xticklabels([])
					axins.set_xticklabels([],minor=True)
					#axins.set_yticklabels([])
					plt.xticks(visible=False) 
					# plt.yticks(visible=False)

					#ax1.indicate_inset_zoom(axins,edgecolor="black")
					mark_inset(ax1,axins,loc1=2,loc2=4,fc="none",ec="0.5")


				#plt.title('XSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

				'''
				if "ALL" in labels[i]:
					if at == "Target":
						ax1.legend(loc='best')
				else:
				'''
				
				if makePlot:
					ax1.legend(loc='upper left')
				else:
					ax1.legend(loc='best')

				fig.tight_layout()

				'''
				special = False

				# Special case
				if at == "Target" and len(tempMethods) == 2 and "OR-"+str(numHypsX[0]) in tempMethods and "tn" in tempMethods:
					special = True
					copyAx = plt.gca()
				'''

				# plt.show()
				# code.interact("after...", local=dict(globals(), **locals()))
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight',pad_inches = 0)


				# Saving legend on separate file
				'''
				legendFig = plt.figure(figsize=(2,2))
				legendFig.legend(lines,tempLabels,loc='center')
				legendFig.tight_layout()
				plt.grid(False)
				plt.axis('off')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
				'''


				'''
				if special:
					copyAx.relim()
					copyAx.autoscale()
					plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-Upd-YLim-"+domain+".png", bbox_inches='tight')
					plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-Upd-YLim-"+domain+".pdf", bbox_inches='tight')
				'''

			plt.clf()
			plt.close("all")


			##################################### FOR PSKILLS #####################################
			# create plot for each one of the different agents - estimates vs obs

			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)
			lines = []
			tempLabels = []

			c = 0

			for method in tempMethods:

				# only plotting xskills methods
				if "pSkills" not in method:
					continue

				c += 1

				#m = methodsDict[method]
				m = methodNamesPaper[method]
	  
				# other methods          
				line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method][1:], lw='2.0', label= str(m), c = methodsColors[method], ls=lineStylesPaper[method])
				# plt.plot(range(len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method], lw='2.0', label= str(m))

				# lines.append(line)
				# tempLabels.append(str(m))

			if c != 0:
				ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
				ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
				# ax1.set_xscale('symlog')
				plt.ylim(top = maxPskillError[labels[i]],bottom = minPskillError[labels[i]])
				plt.margins(0.05)

				#plt.title('PSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

				'''
				if "ALL" in labels[i]:
					if at == "Target":
						ax1.legend(loc='best')
				else:
				'''
				ax1.legend(loc='best')
				fig.tight_layout()
				
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight',pad_inches = 0)

				# Saving legend on separate file
				'''
				legendFig = plt.figure(figsize=(2,2))
				legendFig.legend(lines,tempLabels,loc='center')
				legendFig.tight_layout()
				plt.grid(False)
				plt.axis('off')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.sep + labels[i] + os.sep +  "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
				'''

			plt.clf()
			plt.close("all")
			

	#####################################################################################################

	
	if makeOtherPlots:
		plotMSEAllBetasSamePlotPerAgentType(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerAgentType,betas)
		plotLastMSEAllBetasSamePlotPerAgentType(methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,mseAcrossAllAgentsPerAgentType,betas)
		#plotMseDiffBetasPerPskillBucketsPerAgentTypes(actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas)
		#plotMseDiffBetasPerXskillBucketsPerAgentTypes(actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas)
	

	#####################################################################################################
	# Plot MSE for Selected Bounded Agents
	#####################################################################################################
	
	for at in mseSelectedAgents:

		for x in range(len(agentSubsets)):

			sub = agentSubsets[x]

			if totalSelectedAgents[at][names[x]] != 0:

				for i in range(len(methodsLists)):		

					if len(ops[x]) == 1:
						tempFolder = f"MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType{os.sep}{sub}{ops[x][0]}{params[x][0]}{os.sep}"
					else:
						tempFolder = f"MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType{os.sep}{sub}{ops[x][0]}{params[x][0]}and{ops[x][1]}{params[x][1]}{os.sep}"

					makeFolder(resultsFolder,tempFolder)
					makeFolder(resultsFolder,tempFolder+labels[i])

					tempMethods = methodsLists[i]

					allBetas = False

					if labels[i] == "JustBM":
						allBetas = True

					if len(tempMethods) == 0:
						continue

					##################################### FOR XSKILLS #####################################
					# create plot for each one of the different agents - estimates vs obs

					fig = plt.figure(figsize = (10,10))
					ax1 = plt.subplot(2, 1, 1)
					# lines = []
					# tempLabels = []

					makePlot = False


					'''
					'upper right'  : 1,
					'upper left'   : 2,
					'lower left'   : 3,
					'lower right'  : 4,
					'right'        : 5,
					'center left'  : 6,
					'center right' : 7,
					'lower center' : 8,
					'upper center' : 9,
					'center'       : 10
					'''

					if ("ALL" in labels[i] or "OR-BM-Beta" in labels[i] or "JustEES" in labels[i]) and domain == "1d":

						makePlot = True

						if (domain != "1d"):
							makePlot = False
						else:
							if "Target" not in at:
								makePlot = False
							if "Bounded" in at and "NearlyRational" in sub:
								makePlot = True

					if makePlot:
						loc = 1

						# x0,y0,w,h.
						# axins = ax1.inset_axes([np.log10(startZoom),0,1000-startZoom,7000])#maxXskillError[labels[i]]])
						axins = zoomed_inset_axes(ax1,zoom=20,loc=loc,borderpad=1)

					maxLast = -9999
					minLast = 9999


					c = 0

					for method in tempMethods:

						# only plotting xskills methods
						if "pSkills" in method or "DomainTargets" in method:
							continue

						c += 1

						# Will just plot given beta
						if ("BM" in method):
							tempM, beta, tt = getInfoBM(method)
							#m = methodsDict[method]
							tempM2 = method.split("-Beta")[0]
							tempM3 = methodNamesPaper[method.split("-Beta")[0]]

							label = str(tempM3)+"-"+str(beta)

							if (allBetas or str(beta) in labels[i] or "SelectedBetas" in labels[i]) and "GivenBeta" not in labels[i]:
								if allBetas:
									if "EES" in method:
										line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][beta])),mseSelectedAgents[at][names[x]][tt][tempM][beta][1:], lw='2.0', label= label,ls=lineStylesPaper[tempM2])
								else:

									if "JustEES" in labels[i]:
										label = label.replace("-ES","")
	
									color = None

									if "SelectedBetas" in labels[i]:
										if beta == 0.75:
											cc = "C6"
										elif beta == 0.85:
											cc = "C2"
										else:
											cc = "C9"
										color = cc
										line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][beta])),mseSelectedAgents[at][names[x]][tt][tempM][beta][1:], lw='2.0', label= label,c = cc,ls=lineStylesPaper[tempM2])
									else:
										color = methodsColors[tempM2]
										line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][beta])),mseSelectedAgents[at][names[x]][tt][tempM][beta][1:], lw='2.0', label= label,c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])
							
									if makePlot:
										axins.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][beta])),mseSelectedAgents[at][names[x]][tt][tempM][beta][1:], lw='2.0', label= label,c = color,ls=lineStylesPaper[tempM2])
										
										if mseSelectedAgents[at][names[x]][tt][tempM][beta][-1] > maxLast:
											maxLast = mseSelectedAgents[at][names[x]][tt][tempM][beta][-1]

										if mseSelectedAgents[at][names[x]][tt][tempM][beta][-1] < minLast:
											minLast = mseSelectedAgents[at][names[x]][tt][tempM][beta][-1]

								# lines.append(line)
								# tempLabels.append(label)

							else:
								if beta == givenBeta:
									line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][givenBeta])),mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][1:], lw='2.0', label= label, c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])
									# lines.append(line)
									# tempLabels.append(label)

									if makePlot:
										axins.semilogx(range(1,len(mseSelectedAgents[at][names[x]][tt][tempM][givenBeta])),mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][1:], lw='2.0', label= str(tempM3), c = methodsColors[tempM2],ls=lineStylesPaper[tempM2])

										if mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][-1] > maxLast:
											maxLast = mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][-1]

										if mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][-1] < minLast:
											minLast = mseSelectedAgents[at][names[x]][tt][tempM][givenBeta][-1]


						# other methods (and not other BM methods)         
						elif "BM" not in method:		
							#m = methodsDict[method]
							m = methodNamesPaper[method]
							label = str(m)

							if "JustEES" in labels[i]:
								label = label.replace("-ES","")

							line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][method])),mseSelectedAgents[at][names[x]][method][1:], lw='2.0', label= label, c = methodsColors[method],ls=lineStylesPaper[method])
						
							# lines.append(line)
							# tempLabels.append(label)

							if makePlot:
								axins.semilogx(range(1,len(mseSelectedAgents[at][names[x]][method])),mseSelectedAgents[at][names[x]][method][1:], lw='2.0', label= label, c = methodsColors[method], ls= lineStylesPaper[method])
								
								if mseSelectedAgents[at][names[x]][method][-1] > maxLast:
									maxLast = mseSelectedAgents[at][names[x]][method][-1]

								if mseSelectedAgents[at][names[x]][method][-1] < minLast:
									minLast = mseSelectedAgents[at][names[x]][method][-1]


					if c != 0:
						ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
						ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
						# ax1.set_xscale('symlog')
						plt.ylim(top = maxXskillErrorSelectedAgents[names[x]][labels[i]],bottom = minXskillErrorSelectedAgents[names[x]][labels[i]])
						plt.margins(0.05)


						if makePlot:
							startZoom = 900

							# subregion of the original image
							axins.set_xlim(startZoom,1000)

							# < 0 to add padding on plot
							# minZoomYLim = -0.90

							if minLast <= 0:
								minLast = -0.80
							else:
								minLast = minLast-(minLast*0.20)
							
							axins.set_ylim(minLast,maxLast+(maxLast*0.20))

							axins.set_xticklabels([])
							axins.set_xticklabels([],minor=True)
							#axins.set_yticklabels([])
							plt.xticks(visible=False) 
							# plt.yticks(visible=False)

							#ax1.indicate_inset_zoom(axins,edgecolor="black")
							mark_inset(ax1,axins,loc1=2,loc2=4,fc="none",ec="0.5")


						#plt.title('XSKILL | Agent: ' + at + ' | '+str(mseSelectedAgents[at][names[x]]["totalNumExps"]) + ' experiments')

						'''
						if "ALL" in labels[i]:
							if at == "Target":
								ax1.legend(loc='best')
						else:
						'''

						if makePlot:
							ax1.legend(loc='upper left')
						else:
							ax1.legend(loc='best')

						fig.tight_layout()

						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight',pad_inches = 0)

						# Saving legend on separate file
						'''
						legendFig = plt.figure(figsize=(2,2))
						legendFig.legend(lines,tempLabels,loc='center')
						legendFig.tight_layout()
						plt.grid(False)
						plt.axis('off')
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType" + os.sep + names[x] + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
						'''

					plt.clf()
					plt.close("all")


					##################################### FOR PSKILLS #####################################
					# create plot for each one of the different agents - estimates vs obs

					fig = plt.figure(figsize = (10,10))
					ax1 = plt.subplot(2, 1, 1)
					lines = []
					tempLabels = []

					c = 0

					for method in tempMethods:

						# only plotting xskills methods
						if "pSkills" not in method:
							continue

						c += 1

						#m = methodsDict[method]
						m = methodNamesPaper[method]
			  
						# other methods          
						line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][method])),mseSelectedAgents[at][names[x]][method][1:], lw='2.0', label= str(m), c = methodsColors[method],ls=lineStylesPaper[method])
						# plt.plot(range(len(mseSelectedAgents[at][names[x]][method])),mseSelectedAgents[at][names[x]][method], lw='2.0', label= str(m))

						# lines.append(line)
						# tempLabels.append(str(m))


					if c != 0:
						ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
						ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
						# ax1.set_xscale('symlog')
						plt.ylim(top = maxPskillErrorSelectedAgents[names[x]][labels[i]],bottom = minPskillErrorSelectedAgents[names[x]][labels[i]])
						plt.margins(0.05)

						#plt.title('PSKILL | Agent: ' + at + ' | '+str(mseSelectedAgents[at][names[x]]["totalNumExps"]) + ' experiments')

						'''
						if "ALL" in labels[i]:
							if at == "Target":
								ax1.legend(loc='best')
						else:
						'''
						
						ax1.legend(loc='best')
						fig.tight_layout()

						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight',pad_inches = 0)

						# Saving legend on separate file
						'''
						legendFig = plt.figure(figsize=(2,2))
						legendFig.legend(lines,tempLabels,loc='center')
						legendFig.tight_layout()
						plt.grid(False)
						plt.axis('off')
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSE-SelectedAgents-AllMethodsSamePlot-PerAgentType" + os.sep + names[x] + os.sep + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
						'''

					plt.clf()
					plt.close("all")
			

					##################################### SAVE BEST BETA INFO #####################################

					if labels[i] == "JustBM":

						with open(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep + "bestBetaInfo.txt","a") as outfile:
							print(f"Domain: {domain} | Agent: {at}",file=outfile,end="\n\n")
							print(f"Beta\t|\tLastMSE",file=outfile)

							bestMSE = 99999999
							bestBeta = ""

							for method in tempMethods:

								if "pSkills" in method:
									continue

								if "BM" in method:
									tempM, beta, tt = getInfoBM(method)
									#m = methodsDict[method]
									tempM2 = method.split("-Beta")[0]
									tempM3 = methodNamesPaper[method.split("-Beta")[0]]

									label = str(tempM3)+"-"+str(beta)

									lastMSE = mseSelectedAgents[at][names[x]][tt][tempM][beta][-1]

									print(f"{beta}\t|\t{lastMSE}",file=outfile)

									if lastMSE < bestMSE:
										bestMSE = lastMSE
										bestBeta = beta

							print(f"\nBest Beta: {bestBeta} with MSE = {bestMSE}",file=outfile)
							print("\n",file=outfile)


	#####################################################################################################
	
	del mseAcrossAllAgentsPerAgentType
	del mseSelectedAgents


	# code.interact("...", local=dict(globals(), **locals())) 

def plotPercentTimesDistributionPskillBuckets(domain,processedRFsAgentNames,agentTypes,methods,resultsFolder,numStates):

	makeFolder(resultsFolder, "plotPercentTimesDistribution-PskillBuckets")



	if domain == "1d":
		# Buckets (very bad, bad, regular, good, very good)(backwards)--- xskill
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0] 
	elif domain == "2d" or domain == "sequentialDarts":
		# Buckets (very bad, bad, regular, good, very good)(backwards)--- xskill
		buckets = [25, 50, 75, 100, 150]


	percentOfTimeRightBucketPerAgent = {}

	for at in agentTypes:
		percentOfTimeRightBucketPerAgent[at] = {}


		if "Flip" in at or "Tricker" in at:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		for m in methods:
			if "xSkills" in m:
				percentOfTimeRightBucketPerAgent[at][m] = {}

				for b in pskillBuckets:
					percentOfTimeRightBucketPerAgent[at][m][str(b)] = {}

					for bb in buckets:
						percentOfTimeRightBucketPerAgent[at][m][str(b)][str(bb)] = {"dist": [0.0]*len(buckets), "totalNumExps": 0.0}


	resultsDict = {}

	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if "Flip" in aType or "Tricker" in aType:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		# Find pskill bucket
		for b in range(len(pskillBuckets)):
			if p <= pskillBuckets[b]:
				break

		# get actual bucket
		pskillBucket = pskillBuckets[b]



		trueX = x

		# find bucket corresponding to trueX
		for b in range(len(buckets)):
			if trueX <= buckets[b]:
				break

		# get actual bucket
		rightBucket = buckets[b]


		for m in methods:

			if "xSkills" in m:

				estimatedX = resultsDict[a]["estimates"][m][numStates-1]


				# find bucket corresponding to estimatedX
				for b2 in range(len(buckets)):
					if estimatedX <= buckets[b2]:
						break

				# get actual bucket
				estimatedBucket = b2 #buckets[b2]

				# Count on corresponding bucket
				percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["dist"][estimatedBucket] += 1.0
					
				# Count experiment regardless
				percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["totalNumExps"] += 1.0
		

		del resultsDict[a]


	# Compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():


		if "Flip" in at or "Tricker" in at:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		for m in methods:

			if "xSkills" in m:
			
				rects = []

				x = np.arange(len(buckets))  # the label locations
				width = 0.17 # the width of the bars


				## Outer plot
				fig, axes = plt.subplots(len(buckets), 1, figsize=(15,20)) #sharex=True)

				# for true xskills buckets
				for ib in range(len(buckets)):

					## init subplot
					ax = axes[ib]
					ax.set_title('True xskill Bucket: ' + str(buckets[ib]))

					for bii in range(len(pskillBuckets)):

						xb = pskillBuckets[bii]

						percents = []

						for b in range(len(buckets)):

							# in case bucket doesn't contain any info
							try:
								percent = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[ib])]["dist"][b] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[ib])]["totalNumExps"]) * 100.0

							# to avoid error of float division by 0 when no experiments have been seen for a given bucket (for the given agent type)
							except:
								percent = 0.0
							
							percents.append(percent)


						size = width
						op = None

						# bii represents the bucket position
						# To make conditions independent of buckets (in case they change)

						if bii == 0:
							pos = size * 2
							op = "-"
						elif bii == 1:
							pos = size
							op = "-"
						elif bii == 2:
							pos = 0
							op = "+"
						elif bii == 3:
							pos = size
							op = "+"
						else:
							pos = size * 2
							op = "+"


						if op == "+":    
							rect = ax.bar(x + pos, percents, width = width, label = str(xb), align='edge')
						else:
							rect = ax.bar(x - pos, percents, width = width, label = str(xb), align='edge')

						rects.append(rect)


						# Add some text for labels, title and custom x-axis tick labels, etc.
						ax.set_xticks(x)
						ax.set_xticklabels(buckets)

						#'''
						# Shrink current axis by 20%
						box = ax.get_position()
						ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

						# Put a legend to the right of the current axis
						ax.legend(title = "True pSkill Bucket", loc='center left', bbox_to_anchor=(1, 0.5))

						ax.set_xlabel(r'\textbf{Buckets Xskill}')
						ax.set_ylabel(r'\textbf{Percent Times in Bucket}')
						ax.set_ylim(0,100)

				
				fig.tight_layout()

				plt.margins(0.05)
				fig.suptitle('Agent: ' + at + " | Method: " + m, y=1.02)

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "plotPercentTimesDistribution-PskillBuckets" + os.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")

	#code.interact("here", local = locals())

	###############################################################################################################3

def plotPercentTimesDistributionXskillBuckets(domain,processedRFsAgentNames,agentTypes,methods,resultsFolder,numStates):

	makeFolder(resultsFolder, "plotPercentTimesDistribution-XskillBuckets")


	# Buckets (very bad, bad, regular, good, very good) - in terms of percents
	buckets = [0.45, 0.60, 0.75, 0.90, 1.0]


	if domain == "1d":
		# xskillBuckets = [1.0, 2.0, 3.0, 4.0, 5.0]
		xskillBuckets = np.linspace(0.5, 15.0, num = 5)
	elif domain == "2d" or domain == "sequentialDarts":
		xskillBuckets = [25, 50, 75, 100, 150]



	percentOfTimeRightBucketPerAgent = {}

	for at in agentTypes:
		percentOfTimeRightBucketPerAgent[at] = {}
		

		for m in methods:
			if "pSkills" in m:
				percentOfTimeRightBucketPerAgent[at][m] = {}

				for xb in xskillBuckets:
					percentOfTimeRightBucketPerAgent[at][m][str(xb)] = {}


					for b in buckets:
						percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(b)] = {"dist": [0.0]*len(buckets), "totalNumExps": 0.0}


	resultsDict = {}
	
	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# Find true xskill bucket
		for b in range(len(xskillBuckets)):
			if x <= xskillBuckets[b]:
				break

		# get actual bucket
		xskillBucket = xskillBuckets[b]



		trueP = resultsDict[a]["percentTrueP"]

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break

		# get actual bucket
		rightBucket = buckets[b]


		for m in methods:

			if "pSkills" in m:

				estimatedP = resultsDict[a]["percentsEstimatedPs"][m]["averaged"][numStates-1]


				# find bucket corresponding to estimatedP
				for b2 in range(len(buckets)):
					if estimatedP <= buckets[b2]:
						break

				# get actual bucket
				estimatedBucket = b2 #buckets[b2]

				percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["dist"][estimatedBucket] += 1.0
					
				# Count experiment
				percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["totalNumExps"] += 1.0
			

		del resultsDict[a]


	# compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():

		for m in methods:

			if "pSkills" in m:
			
				rects = []

				x = np.arange(len(buckets))  # the label locations
				width = 0.17 # the width of the bars


				## Outer plot
				fig, axes = plt.subplots(len(buckets), 1, figsize=(15,20)) #sharex=True)

				# for true xskills buckets
				for ib in range(len(buckets)):

					## init subplot
					ax = axes[ib]
					ax.set_title('True pskill Bucket: ' + str(buckets[ib]))

					for bii in range(len(xskillBuckets)):

						xb = xskillBuckets[bii]

						percents = []

						for b in range(len(buckets)):

							# in case bucket doesn't contain any info
							try:
								percent = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[ib])]["dist"][b] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[ib])]["totalNumExps"]) * 100.0

							# to avoid error of float division by 0 when no experiments have been seen for a given bucket (for the given agent type)
							except:
								percent = 0.0
							
							percents.append(percent)


						# bii represents the bucket position
						# To make conditions independent of buckets (in case they change)

						size = width
						op = None

						if bii == 0:
							pos = size * 2 
							op = "-"
						elif bii == 1:
							pos = size 
							op = "-"
						elif bii == 2:
							pos = 0 
							op = "+"
						elif bii == 3:
							pos = size
							op = "+"
						else:
							pos = size * 2 
							op = "+"


						if op == "+":    
							rect = ax.bar(x + pos, percents, width = width, label = str(xb), align='edge')
						else:
							rect = ax.bar(x - pos, percents, width = width, label = str(xb), align='edge')

						rects.append(rect)


						# Add some text for labels, title and custom x-axis tick labels, etc.
						ax.set_xticks(x)
						ax.set_xticklabels(buckets)

						# Shrink current axis by 20%
						box = ax.get_position()
						ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

						# Put a legend to the right of the current axis
						ax.legend(title = "True xSkill Bucket", loc='center left', bbox_to_anchor=(1, 0.5))
						
						ax.set_xlabel(r'\textbf{Buckets Percent Reward}')
						ax.set_ylabel(r'\textbf{Percent Times in Bucket}')
						ax.set_ylim(0,100)


				fig.tight_layout(pad = 2.0)

				plt.margins(0.05)
				fig.suptitle('Agent: ' + at + " | Method: " + m, y=1.02)

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "plotPercentTimesDistribution-XskillBuckets" + os.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")

def plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndPskillBuckets(domain,processedRFsAgentNames,agentTypes,methods,resultsFolder,numStates):

	makeFolder(resultsFolder, "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets")

	if domain == "1d":
		# Buckets (very bad, bad, regular, good, very good)--- xskill
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0] 
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [25,50,75,100,150]


	#pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS


	percentOfTimeRightBucketPerAgent = {}

	resultsDict = {}

	for at in agentTypes:
		percentOfTimeRightBucketPerAgent[at] = {}

		if "Flip" in at or "Tricker" in at:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		for m in methods:
			if "xSkills" in m:
				percentOfTimeRightBucketPerAgent[at][m] = {}

				for xb in pskillBuckets:
					percentOfTimeRightBucketPerAgent[at][m][str(xb)] = {}


					for b in buckets:
						percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(b)] = {"totalNumExps": 0.0, "timesInBucket": 0.0, "avgTrueX": 0.0, "avgEstimatedX": 0.0, "maxTrueX": -999.9, "minTrueX": 999.0, "maxEstimatedX": -999.9, "minEstimatedX": 999.0}


	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if "Flip" in aType or "Tricker" in aType:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		# Find pskill bucket
		for b in range(len(pskillBuckets)):
			if p <= pskillBuckets[b]:
				break

		# get actual bucket
		pskillBucket = pskillBuckets[b]



		trueX = x

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueX <= buckets[b]:
				break

		# get actual bucket
		rightBucket = buckets[b]


		for m in methods:

			if "xSkills" in m:

				estimatedX = resultsDict[a]["estimates"][m][numStates-1]


				# find bucket corresponding to estimatedX
				for b2 in range(len(buckets)):
					if estimatedX <= buckets[b2]:
						break

				# get actual bucket
				estimatedBucket = buckets[b2]

				# if the buckets matched, count
				if estimatedBucket == rightBucket:
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["timesInBucket"] += 1.0
					
					# update avg of true  & estimated
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["avgTrueX"] += trueX
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["avgEstimatedX"] += estimatedX


				# Count experiment regardless
				percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["totalNumExps"] += 1.0
				


				##### Update min & maxes bounds

				if trueX < percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["minTrueX"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["minTrueX"] = trueX

				if trueX > percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["maxTrueX"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["maxTrueX"] = trueX


				if estimatedX < percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["minEstimatedX"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["minEstimatedX"] = estimatedX

				if estimatedX > percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["maxEstimatedX"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(pskillBucket)][str(rightBucket)]["maxEstimatedX"] = estimatedX


		del resultsDict[a]


	# compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():

		# RESET FILE
		with open(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.sep  + "BucketBounds-Agent-"+at+".txt", "w") as aFile:
			aFile.truncate()


		if "Flip" in at or "Tricker" in at:
			# Buckets (very bad, bad, regular, good, very good)
			pskillBuckets = [0.45, 0.6, 0.75, 0.9, 1.0] # NOT IN % TERMS
			
		# For the rest of the agents
		else:
			# Buckets (very bad, bad, regular, good, very good)
			if domain == "1d":
				pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS
			elif domain == "2d" or domain == "sequentialDarts":
				pskillBuckets = [5, 15, 20, 25 ,32] # NOT IN % TERMS


		for m in methods:

			if "xSkills" in m:
			
				rects = []
				allAvgEstimatedX = []
				allPercents = []

				x = np.arange(len(buckets))  # the label locations
				width = 0.17 # the width of the bars

				fig, ax = plt.subplots()
				fig.set_size_inches(15,10)


				for bii in range(len(pskillBuckets)):

					xb = pskillBuckets[bii]

					avgTrueX = []
					avgEstimatedX = []
					percents = []


					for b in range(len(buckets)):

						# in case bucket doesn't contain any info
						try:
							percent = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["timesInBucket"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"]) * 100.0
							meanTrueX = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["avgTrueX"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"])
							meanEstimatedX = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["avgEstimatedX"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"])

							percents.append(percent)
							avgTrueX.append(meanTrueX)
							avgEstimatedX.append(meanEstimatedX)

						# to avoid error of float division by 0 when no experiments have been seen for a given bucket (for the given agent type)
						except:
							percent = 0.0
							percents.append(percent)
							avgTrueX.append(0.0)
							avgEstimatedX.append(0.0)


					allAvgEstimatedX.append(avgEstimatedX)
					allPercents.append(percents)


					# bii represents the bucket position
					# To make conditions independent of buckets (in case they change)

					size = width
					op = None

					if bii == 0:
						pos = size * 2
						op = "-"
					elif bii == 1:
						pos = size
						op = "-"
					elif bii == 2:
						pos = 0
						op = "+"
					elif bii == 3:
						pos = size
						op = "+"
					else:
						pos = size * 2
						op = "+"


					if op == "+":    
						rect = ax.bar(x + pos, percents, width = width, label = str(xb), align='edge')
					else:
						rect = ax.bar(x - pos, percents, width = width, label = str(xb), align='edge')

					rects.append(rect)


					# store information about bounds
					with open(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.sep  + "BucketBounds-Agent-"+at+".txt", "a") as aFile:

						aFile.write("\nMethod: "+ str(m)+"\n")
						aFile.write("Bucket\tMinTrueP\tMaxTrueP\tMinEstimatedP\tMaxEstimatedP \n")
						
						for b in range(len(buckets)):
							aFile.write(str(buckets[b])+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["minTrueX"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["maxTrueX"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["minEstimatedX"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["maxEstimatedX"],2))+"\n")



				# Add some text for labels, title and custom x-axis tick labels, etc.
				ax.set_xticks(x)
				ax.set_xticklabels(buckets)


				# Shrink current axis by 20%
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

				# Put a legend to the right of the current axis
				ax.legend(title = "True pSkill Bucket", loc='center left', bbox_to_anchor=(1, 0.5))
				
				fig.tight_layout()


				ax.set_xlabel(r'\textbf{Buckets Xskill}', labelpad = 40)
				ax.set_ylabel(r'\textbf{Percent Times in Right Bucket}')
				plt.margins(0.05)
				ax.set_title('Agent: ' + at + " | Method: " + m)
				ax.set_ylim(0,100)

				'''
				def autolabel(rects,avgEstimatedX):
					"""Attach a text label above each bar in *rects*, displaying its height."""
					for r in range(len(rects)):
						height = rects[r].get_height()
						ax.annotate('{}'.format(round(avgEstimatedX[r],2)),
									xy=(rects[r].get_x() + rects[r].get_width() / 2, height),
									xytext=(0, 3),  # 3 points vertical offset
									textcoords="offset points",
									ha='center', va='bottom')



				for i in range(len(rects)):
					#autolabel(rects[i],allAvgEstimatedP[i])
					autolabel(rects[i],allPercents[i])
				'''

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")


				####### scatter plot of avgTrueP and avgEstimatedP
				fig = plt.figure()
				ax = plt.subplot(111)

				s = ax.scatter(avgTrueX, avgEstimatedX)
				plt.xlabel("AVG True Xskill")
				plt.ylabel("AVG Estimated Xskill")
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.sep  + "scatterPlot-avgTrueVSEstimatedP-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")
  

			###############################################################################################################3

def plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndXskillBuckets(domain,processedRFsAgentNames,agentTypes,methods,resultsFolder,numStates):

	makeFolder(resultsFolder, "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets")


	# Buckets (very bad, bad, regular, good, very good) - in terms of percents
	buckets = [0.45, 0.60, 0.75, 0.90, 1.0]


	if domain == "1d":
		# xskillBuckets = [1.0, 2.0, 3.0, 4.0, 5.0]
		xskillBuckets = np.linspace(0.5, 15.0, num = 100)
	elif domain == "2d" or domain == "sequentialDarts":
		xskillBuckets = [25, 50, 75, 100, 150]


	percentOfTimeRightBucketPerAgent = {}

	resultsDict = {}

	for at in agentTypes:
		percentOfTimeRightBucketPerAgent[at] = {}


		for m in methods:
			if "pSkills" in m:
				percentOfTimeRightBucketPerAgent[at][m] = {}

				for xb in xskillBuckets:
					percentOfTimeRightBucketPerAgent[at][m][str(xb)] = {}


					for b in buckets:
						percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(b)] = {"totalNumExps": 0.0, "timesInBucket": 0.0, "avgTrueP": 0.0, "avgEstimatedP": 0.0, "maxTrueP": -999.9, "minTrueP": 999.0, "maxEstimatedP": -999.9, "minEstimatedP": 999.0}




	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# Find true xskill bucket
		for b in range(len(xskillBuckets)):
			if x <= xskillBuckets[b]:
				break

		# get actual bucket
		xskillBucket = xskillBuckets[b]



		trueP = resultsDict[a]["percentTrueP"]

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break

		# get actual bucket
		rightBucket = buckets[b]


		for m in methods:

			if "pSkills" in m:

				# estimatedP = float(round(resultsDict[a]["estimates"][m][numStates-1],4))
				try:
					estimatedP = resultsDict[a]["percentsEstimatedPs"][m]["averaged"][numStates-1]
				except:
					code.interact("...", local=dict(globals(), **locals()))


				# find bucket corresponding to estimatedP
				for b2 in range(len(buckets)):
					if estimatedP <= buckets[b2]:
						break

				# get actual bucket
				estimatedBucket = buckets[b2]

				# if the buckets matched, count
				if estimatedBucket == rightBucket:
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["timesInBucket"] += 1.0
					
					# update avg of true  & estimated
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["avgTrueP"] += trueP
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["avgEstimatedP"] += estimatedP


				# Count experiment
				percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["totalNumExps"] += 1.0
				


				##### Update min & maxes bounds

				if trueP < percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["minTrueP"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["minTrueP"] = trueP

				if trueP > percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["maxTrueP"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["maxTrueP"] = trueP


				if estimatedP < percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["minEstimatedP"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["minEstimatedP"] = estimatedP

				if estimatedP > percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["maxEstimatedP"]:
					percentOfTimeRightBucketPerAgent[aType][m][str(xskillBucket)][str(rightBucket)]["maxEstimatedP"] = estimatedP


		del resultsDict[a]


	# compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():

		# RESET FILE
		with open(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.sep  + "BucketBounds-Agent-"+at+".txt", "w") as aFile:
			aFile.truncate()


		for m in methods:

			if "pSkills" in m:
			
				rects = []
				allAvgEstimatedP = []
				allPercents = []

				x = np.arange(len(buckets))  # the label locations
				width = 0.17 # the width of the bars

				fig, ax = plt.subplots()
				fig.set_size_inches(15,10)


				for bii in range(len(xskillBuckets)):

					xb = xskillBuckets[bii]

					avgTrueP = []
					avgEstimatedP = []
					percents = []


					for b in range(len(buckets)):

						# in case bucket doesn't contain any info
						try:
							percent = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["timesInBucket"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"]) * 100.0
							meanTrueP = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["avgTrueP"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"])
							meanEstimatedP = (percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["avgEstimatedP"] / percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["totalNumExps"])

							percents.append(percent)
							avgTrueP.append(meanTrueP)
							avgEstimatedP.append(meanEstimatedP)

						# to avoid error of float division by 0 when no experiments have been seen for a given bucket (for the given agent type)
						except:
							percent = 0.0
							percents.append(percent)
							avgTrueP.append(0.0)
							avgEstimatedP.append(0.0)


					allAvgEstimatedP.append(avgEstimatedP)
					allPercents.append(percents)


					# bii represents the bucket position
					# To make conditions independent of buckets (in case they change)

					size = width 
					op = None

					if bii == 0:
						pos = size * 2
						op = "-"
					elif bii == 1:
						pos = size
						op = "-"
					elif bii == 2:
						pos = 0
						op = "+"
					elif bii == 3:
						pos = size 
						op = "+"
					else:
						pos = size * 2
						op = "+"


					if op == "+":    
						rect = ax.bar(x + pos, percents, width = width, label = str(xb), align='edge')
					else:
						rect = ax.bar(x - pos, percents, width = width, label = str(xb), align='edge')

					rects.append(rect)


					# store information about bounds
					with open(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.sep  + "BucketBounds-Agent-"+at+".txt", "a") as aFile:

						aFile.write("\nMethod: "+ str(m)+"\n")
						aFile.write("Bucket\tMinTrueP\tMaxTrueP\tMinEstimatedP\tMaxEstimatedP \n")
						
						for b in range(len(buckets)):
							aFile.write(str(buckets[b])+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["minTrueP"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["maxTrueP"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["minEstimatedP"],2))+"\t"+\
										str(round(percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(buckets[b])]["maxEstimatedP"],2))+"\n")



				# Add some text for labels, title and custom x-axis tick labels, etc.
				ax.set_xticks(x)
				ax.set_xticklabels(buckets)


				# Shrink current axis by 20%
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

				# Put a legend to the right of the current axis
				ax.legend(title = "True xSkill Bucket", loc='center left', bbox_to_anchor=(1, 0.5))
				
				fig.tight_layout()


				ax.set_xlabel(r'\textbf{Buckets Percent Reward}', labelpad = 40)
				ax.set_ylabel(r'\textbf{Percent Times in Right Bucket}')
				plt.margins(0.05)
				ax.set_title('Agent: ' + at + " | Method: " + m)
				ax.set_ylim(0,100)


				'''
				def autolabel(rects,avgEstimatedP):
					"""Attach a text label above each bar in *rects*, displaying its height."""
					for r in range(len(rects)):
						height = rects[r].get_height()
						ax.annotate('{}'.format(round(avgEstimatedP[r],2)),
									xy=(rects[r].get_x() + rects[r].get_width() / 2, height),
									xytext=(0, 3),  # 3 points vertical offset
									textcoords="offset points",
									ha='center', va='bottom')



				for i in range(len(rects)):
					#autolabel(rects[i],allAvgEstimatedP[i])
					autolabel(rects[i],allPercents[i])

				'''


				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")


				####### scatter plot of avgTrueP and avgEstimatedP
				fig = plt.figure()
				ax = plt.subplot(111)

				s = ax.scatter(avgTrueP, avgEstimatedP)
				plt.xlabel("AVG True Percent")
				plt.ylabel("AVG Estimated Percent")
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.sep  + "scatterPlot-avgTrueVSEstimatedP-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close("all")
  

			###############################################################################################################3

			'''
			# Sample code obtained from: https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
			# and modified for this purpose

			centers = []

			# find centers of buckets
			for b in range(len(buckets)):
				if b == 0:
					centers.append((0 + buckets[b])/2)
				else:
					centers.append((buckets[b-1] + buckets[b])/2)



			for i in range(len(centers)):
				ax.annotate(format(avgTrueP[i], ".2f"), xy=(centers[i], 0), xycoords=('data', 'axes fraction'),
					xytext=(0, -24), textcoords='offset points', va='top', ha='center')

				ax.annotate(format(avgEstimatedP[i], ".2f"), xy=(centers[i], 0), xycoords=('data', 'axes fraction'),
					xytext=(0, -44), textcoords='offset points', va='top', ha='center')

				ax.text(x = centers[i], y = percents[i], s = str(round(percents[i],2)) + "%", fontsize = 12)
			'''

			# Give ourselves some more room at the bottom of the plot
			#plt.subplots_adjust(bottom = 0.30)

def plotMSEPercentPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,resultsFolder,domain):

	makeFolder(resultsFolder, "msePercent-PerAgentType-PskillMethods")

	# init dict to store info
	percentRewardObtainedPerAgentType = {}


	for at in seenAgents:
		percentRewardObtainedPerAgentType[at] = {"perMethod": {}, "numAgents": 0.0}

		for m in actualMethodsOnExps:
			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			percentRewardObtainedPerAgentType[at]["perMethod"][m] = [0.0] * numStates # to store per state - across different exps per agent type


	resultsDict = {}

	for a in processedRFsAgentNames:

		aType, x, p = getParamsFromAgentName(a)


		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# Update agent count
		percentRewardObtainedPerAgentType[aType]["numAgents"] += resultsDict[a]["num_exps"]
		
		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			# for each state
			for mxi in range(numStates):

				sq = resultsDict[a]["mse_percent_pskills"][m][mxi]

				# Store squared error
				percentRewardObtainedPerAgentType[aType]["perMethod"][m][mxi] += sq


		del resultsDict[a]


	# Normalize - Find Mean Squared Error

	# For each agent type
	for at in seenAgents:

		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			# for each state
			for mxi in range(numStates):

				#find MSE  = SE / # of agents
				percentRewardObtainedPerAgentType[at]["perMethod"][m][mxi] /= percentRewardObtainedPerAgentType[at]["numAgents"]


	# Plot - for MSE

	# For each agent type
	for at in seenAgents:


		###################################################### Normal Scale ######################################################
		fig = plt.figure()
		ax = plt.subplot(111)


		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			#if "BM" in m: # 'BM-MAP', 'BM-EES' to TBA
			#    m = m.replace("BM","TBA")

			plt.plot(range(len(percentRewardObtainedPerAgentType[at]["perMethod"][m])),percentRewardObtainedPerAgentType[at]["perMethod"][m], lw=2.0, label= str(m))

		plt.xlabel(r'\textbf{Number of observations}',fontsize=18)
		plt.ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)
		plt.title('Agent: ' + at + ' | MSE % of TR')

		# Shrink current axis by 10%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})
		
		plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePercent-PerAgentType-PskillMethods" + os.sep +  "results-Agent"+at+".png", bbox_inches='tight')

		plt.clf()
		plt.close("all")

		############################################################################################################################


		###################################################### LOG Scale ######################################################
		fig = plt.figure()
		ax = plt.subplot(111)


		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			#if "BM" in m: # 'BM-MAP', 'BM-EES' to TBA
			#    m = m.replace("BM","TBA")

			plt.semilogx(range(len(percentRewardObtainedPerAgentType[at]["perMethod"][m])),percentRewardObtainedPerAgentType[at]["perMethod"][m], lw=2.0, label= str(m))

		plt.xlabel(r'\textbf{Number of observations}',fontsize=18)
		plt.ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)
		plt.title('Agent: ' + at + ' | MSE % of TR')

		# Shrink current axis by 10%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})
		
		plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "msePercent-PerAgentType-PskillMethods" + os.sep +  "results-LOG-Agent"+at+".png", bbox_inches='tight')

		plt.clf()
		plt.close("all")

		############################################################################################################################

# Computes MSE for rationality percentages (mse_percent_pskills)
def computeMSEPercentPskillsMethods(processedRFsAgentNames,actualMethodsOnExps,pconfPerXskill,numStates,numHypsX,numHypsP,domain):

	# NOTE: Bounded agent still needs conversion process!
	#		Estimate is already in lambda terms
	#		But multiple lambdas can mean the same rationality percentage.

	resultsDict = {}

	for a in processedRFsAgentNames:

		aType, xStr, p = getParamsFromAgentName(a)
		

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		###################################################################################
		# Find "right" answer - true percent for trueX
		# Bounded agents -> percentTrueP = actual pskill
		# Other agents -> percentTrueP = estimate converted to percentage terms
		percent_trueP = resultsDict[a]["percentTrueP"]
		###################################################################################

		
		# For each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA
			if "pSkills" not in m:
				continue


			# For each exp
			for expNum in range(1,resultsDict[a]["num_exps"]+1):			

				# for each state
				for mxi in range(numStates):

					percent_estimatedP = resultsDict[a]["percentsEstimatedPs"][m][expNum][mxi]

					# Compute squared error =  (estimatedP - trueP) ** 2
					sq = (percent_trueP - percent_estimatedP) ** 2

					# Store squared error
					resultsDict[a]["mse_percent_pskills"][m][mxi] += sq

					resultsDict[a]["percentsEstimatedPs"][m]["averaged"][mxi] += percent_estimatedP
	

			# for each state
			for mxi in range(numStates):

				# Find MSE  = SE / # of exps seen for given agent
				# If exps with random params, will just divide by 1
				# As not very likely to see multiple exps for the same exact agent
				resultsDict[a]["mse_percent_pskills"][m][mxi] /= float(resultsDict[a]["num_exps"]) 
				
				# Find mean estimate
				resultsDict[a]["percentsEstimatedPs"][m]["averaged"][mxi] /= float(resultsDict[a]["num_exps"]) 


			# code.interact("after norm...", local=dict(globals(), **locals()))


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]

# Converting from true pskill to rationality percentage
def plotPercentRandMaxRewardObtainedPerXskillPerAgentType(processedRFsAgentNames,agentTypes,resultsFolder,seenAgents,domain,pconfPerXskill):
	# all xskills included - focus on pskills

	resultsDict = {}

	makeFolder(resultsFolder, "percentRandMaxRewardObtained-PerXskillPerAgentType")

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["axes.titleweight"] = "bold"

	bucketsX = sorted(pconfPerXskill.keys())
	
	minMaxX = [bucketsX[0],bucketsX[-1]]

	percentOfRewardPerAgent = {}

	for at in agentTypes:

		percentOfRewardPerAgent[at] = {}

		percentOfRewardPerAgent[at] = {"allPercents": [], "allXskills": [], "allPskills": []}


	for a in processedRFsAgentNames:

		aType, xStr, p = getParamsFromAgentName(a)


		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# Bounded agent still needs conversion process!
		# Estimate is already in lambda terms
		# But multiple lambdas can mean the same rationality percentage.

		
		# minR = resultsDict[a]["mean_random_reward"]
		minR = resultsDict[a]["mean_random_reward_mean_vs"]

		expectedR = resultsDict[a]["mean_value_intendedAction"]
		maxR = resultsDict[a]["mean_true_reward"]


		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		# percentOfReward = (((expectedR - minR)/(maxR - minR)) * 100.0)
		percentOfReward = (((expectedR - minR)/(maxR - minR)))

		resultsDict[a]["percentTrueP"] = percentOfReward


	

		# if percentOfReward < 0: 
			# code.interact("% of reward is negative...", local=locals())
		
		#code.interact("after % of reward...", local=locals())


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]


		percentOfRewardPerAgent[aType]["allPercents"].append(percentOfReward)
		percentOfRewardPerAgent[aType]["allXskills"].append(float(xStr))
		percentOfRewardPerAgent[aType]["allPskills"].append(float(p)) 

	
	# code.interact("after true % reward...", local=dict(globals(), **locals()))

	for aType in agentTypes:

		fig = plt.figure()


		# If we didn't see any experiments for the given agent type, continue/skip
		if aType not in seenAgents:
			continue

		if aType == "Random":
			continue


		ax = plt.subplot(111)

		cmap = plt.get_cmap("viridis")
		# norm = plt.Normalize(minMaxX[0], minMaxX[1])

		plt.scatter(np.asarray(percentOfRewardPerAgent[aType]["allPskills"]),np.asarray(percentOfRewardPerAgent[aType]["allPercents"]),\
					# c =  cmap(norm(np.asarray(percentOfRewardPerAgent[aType]["allXskills"]))))
					c =  np.asarray(percentOfRewardPerAgent[aType]["allXskills"]))

		
		plt.xlabel(r'\textbf{Rationality Parameter}')
		plt.ylabel(r'\textbf{Rationality Percentage}')
		

		# sm = ScalarMappable(norm = norm, cmap = cmap)
		# sm.set_array([])

		cbar = plt.colorbar()
		cbar.ax.set_title("Execution Noise Level", fontdict = {'verticalalignment': 'center', 'horizontalalignment': "center"},\
						    y = 0.50, rotation = 90)
		# cbar.set_label(r'\textbf{Execution Noise Level}',size=14))

		plt.margins(0.05)

		plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentRandMaxRewardObtained-PerXskillPerAgentType" + os.sep  + "results-Percent-RandMaxRewards-Agent-"+aType+".png", bbox_inches='tight')

		plt.clf()
		plt.close("all")


		#######################################################################################################################################
	

	# TO PLOT PCONF INFO
	ax = plt.subplot(111)
	
	cmap = plt.get_cmap("rainbow")
	norm = plt.Normalize(min(bucketsX),max(bucketsX))

	for i in range(len(bucketsX)):
		b = bucketsX[i]
		if domain == "1d":
			plt.plot(pconfPerXskill[b]["lambdas"], pconfPerXskill[b]["prat"],label=round(b,2), color=cmap(norm(b)))
		else:
			plt.plot(pconfPerXskill[b]["lambdas"], pconfPerXskill[b]["prat"],color=cmap(norm(b)))

	plt.xlabel(r'\textbf{Rationality Parameter}')
	plt.ylabel(r'\textbf{Rationality Percentage}')
	
	if domain == "1d":
		plt.legend()
	else:
		sm = ScalarMappable(norm = norm, cmap = cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm,ax=ax)			
		cbar.set_label(r'\textbf{Execution Skill Level', labelpad=+1)
		cbar.ax.invert_yaxis()

	plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "percentRandMaxRewardObtained-PerXskillPerAgentType" + os.sep  + "results-Percent-RandMaxRewards-FITTED-LINE-Agent-Bounded-PCONF.png", bbox_inches='tight')

	plt.clf()
	plt.close("all")

	# code.interact("after percent rationality...", local=dict(globals(), **locals()))

def computeMeanAVGAndTrueRewardsPerState(processedRFsAgentNames):
	
	resultsDict = {}

	#Compute Mean of Average Reward & True Rewards across experiments 
	for a in processedRFsAgentNames:

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		for mxi in range(numStates):
			# Compute the avg of the avg rewards (across the different experiments)
			resultsDict[a]["avg_rewards"][mxi] /= (1.0 * resultsDict[a]["num_exps"])
			resultsDict[a]["true_rewards"][mxi] /= (1.0 * resultsDict[a]["num_exps"])


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]

def computeMeanAvgRewardsPerExp(processedRFsAgentName):

	resultsDict = {}

	#Compute Mean of mean Observed/True Reward  - per experiments 
	for a in processedRFsAgentNames:

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# Compute the mean of the mean reward estimate
		resultsDict[a]["mean_observed_reward"] /= (1.0 * resultsDict[a]["num_exps"])
		resultsDict[a]["mean_true_reward"] /= (1.0 * resultsDict[a]["num_exps"])

		resultsDict[a]["mean_value_intendedAction"] /= (1.0 * resultsDict[a]["num_exps"])

		resultsDict[a]["mean_random_reward_mean_vs"] /= (1.0 * resultsDict[a]["num_exps"])


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]

def computeMeanEstimates(processedRFsAgentNames):

	resultsDict = {}

	#Compute Mean of XSkill estimate across experiments 
	for a in processedRFsAgentNames:

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		for m in actualMethodsOnExps:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for mxi in range(numStates):
					# compute the avg of the avg skill estimate (across the different experiments)
					resultsDict[a]["estimates"][tt][tempM][beta][mxi] /= (1.0 * resultsDict[a]["num_exps"])

			else:
				for mxi in range(numStates):
					# compute the avg of the avg skill estimate (across the different experiments)
					resultsDict[a]["estimates"][m][mxi] /= (1.0 * resultsDict[a]["num_exps"])


			# code.interact("...", local=dict(globals(), **locals()))

		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]

# Will create initial copy of rdFile into prdFile
def computeMSE(processedRFsAgentNames):

	resultsDict = {}

	# Compute MSE
	for a in processedRFsAgentNames:

		# print("\nAgent: ", a)


		# Load processed info		
		resultsDict[a] = loadProcessedInfo(rdFile,a)


		for m in methods:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for mxi in range(len(resultsDict[a]["plot_y"][tt][tempM][beta])):
					resultsDict[a]["plot_y"][tt][tempM][beta][mxi] = resultsDict[a]["plot_y"][tt][tempM][beta][mxi] / float(resultsDict[a]["num_exps"]) #MSError

				# print('Method : ', tempM, " -", tt, " Beta: ", beta, 'has', len(resultsDict[a]["plot_y"][tt][tempM][beta]), 'data points and ', \
				# resultsDict[a]['num_exps'], ' experiments.')

			else:
			
				for mxi in range(len(resultsDict[a]["plot_y"][m])):
					resultsDict[a]["plot_y"][m][mxi] = resultsDict[a]["plot_y"][m][mxi] / float(resultsDict[a]["num_exps"]) #MSError

				# print('Method : ', m, 'has', len(resultsDict[a]["plot_y"][m]), 'data points and ', \
					# resultsDict[a]['num_exps'], ' experiments.')

		# Do it for the TN as well
		for mxi in range(len(resultsDict[a]["plot_y"][m])):
			resultsDict[a]["plot_y"]["tn"][mxi] /= float(resultsDict[a]["num_exps"]) #MSError


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)
		
		# code.interact("...", local=dict(globals(), **locals()))


		del resultsDict[a]


if __name__ == "__main__":


	# ASSUMES RESULTS OF EXPERIMENTS WERE PROCESSED ALREADY


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "1d")	
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	rdFile = args.resultsFolder + os.sep + "ProcessedResultsFiles" + os.sep + "resultsDictInfo"
	oiFile = args.resultsFolder + os.sep + "otherInfo" 


	with open(oiFile,"rb") as file:
		otherInfo = pickle.load(file)

		actualMethodsNames = otherInfo["actualMethodsNames"]
		actualMethodsOnExps = otherInfo["actualMethodsOnExps"]
		typeTargetsList = otherInfo["typeTargetsList"]
		numHypsX = otherInfo['numHypsX']
		numHypsP = otherInfo['numHypsP']
		numStates = otherInfo["numObservations"]
		seenAgents = otherInfo["seenAgents"]
		methods = otherInfo["methods"]
		domain = otherInfo["domain"]
		mode = otherInfo["mode"]
		betas = otherInfo["betas"]

		processedRFs = otherInfo["processedRFs"]
		processedRFsAgentNames = otherInfo["processedRFsAgentNames"]

		try:
			wrap = otherInfo["wrap"]
		except:
			wrap = True

	try:
		actualProcessedRFs = os.listdir(args.resultsFolder + os.sep + "ProcessedResultsFiles")
	except:
		print("Folder for processed results files not present.\nNeed to process results files first.")
		exit()


	if len(actualProcessedRFs) == 0:
		print("Need to process results files first.")
		exit()


	makeFolder2(args.resultsFolder,"ProcessedResultsFilesForPlots")
	
	prdFile = args.resultsFolder + os.sep + "ProcessedResultsFilesForPlots" + os.sep + "resultsDictInfo"


	#agentTypes = ["Target", "Flip", "Tricker","Bounded", "TargetBelief"]
	agentTypes = ["Target", "Flip", "Tricker","Bounded"]

	domainModule,delta = getDomainInfo(domain,wrap)


	'''
	resultsDict = {}
	
	total_num_exps = 0

	for pf in actualProcessedRFs: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(actualProcessedRFs))+') - Processed File: ', pf)

		aName = pf.split("resultsDictInfo-")[1]

		resultsDict[aName] = {}
	'''

	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args,wrap)

	####################################


	bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		minMaxX = [bucketsX[0],bucketsX[-1]]#[0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]



	##################################################
	# PARAMETERS FOR PLOTS
	##################################################

	matplotlib.rcParams.update({'font.size': 14})
	matplotlib.rcParams.update({'legend.fontsize': 14})
	matplotlib.rcParams["axes.labelweight"] = "bold"
	matplotlib.rcParams["axes.titleweight"] = "bold"

	# To make font of title & labels bold
	matplotlib.rc('text', usetex=True)

	##################################################


	##################################################

	global methodsDictNames
	methodsDictNames = {'OR': 'OR'+"-"+str(numHypsX[0]),\
					"OR-MidGame": 'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame", \
					"OR-FullGame": 'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame", \

					'BM-MAP': 'BM-MAP'+"-"+str(numHypsX[0]),\
					'BM-EES': 'BM-EES'+"-"+str(numHypsX[0]),\

					"JT-QRE-MAP": ["JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"],\
					"JT-QRE-EES": ["JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"],\

					"JT-FLIP-MAP": ["JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"],\
					"JT-FLIP-EES": ["JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"],\

					"NJT-QRE-MAP": ["NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"],\
					"NJT-QRE-EES": ["NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills","NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills"]}

	global methodsDict
	methodsDict = {'OR'+"-"+str(numHypsX[0]): "OR", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "OR-MidGame", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "OR-FullGame", \

					'BM-MAP'+"-"+str(numHypsX[0]): 'BM-MAP', \
					'BM-EES'+"-"+str(numHypsX[0]): 'BM-EES',\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":"JT-QRE-MAP",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JT-QRE-MAP",\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "JT-QRE-EES",\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JT-QRE-EES",\

					 "JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "JT-FLIP-MAP",\
					 "JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "JT-FLIP-MAP",\

					 "JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "JT-FLIP-EES",\
					 "JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "JT-FLIP-EES",\

					 "NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "NJT-QRE-MAP",\
					 "NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "NJT-QRE-MAP",\

					 "NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "NJT-QRE-EES",\
					 "NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "NJT-QRE-EES"}

	global methodsColors
	methodsColors = {"tn":"tab:red",\

					'OR'+"-"+str(numHypsX[0]): "tab:olive",\
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "tab:olive", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "tab:gray", \


					'BM-MAP'+"-"+str(numHypsX[0]): "tab:pink", \
					'BM-EES'+"-"+str(numHypsX[0]): "tab:green",\

					'BM-MAP'+"-"+str(numHypsX[0])+"-OptimalTargets": "tab:pink", \
					'BM-EES'+"-"+str(numHypsX[0])+"-OptimalTargets": "tab:green",\

					'BM-MAP'+"-"+str(numHypsX[0])+"-DomainTargets": "xkcd:fuchsia", \
					'BM-EES'+"-"+str(numHypsX[0])+"-DomainTargets": "xkcd:darkblue",\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:purple",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:purple" ,\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:blue" ,\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:blue" ,\

					# "JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:gray" ,\
					# "JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:gray",\

					# "JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:olive",\
					# "JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:olive",\

					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:orange",\
					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:orange" ,\

					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:brown",\
					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:brown"}

	global methodNamesPaper
	methodNamesPaper = { "tn": "TN",\
					'OR'+"-"+str(numHypsX[0]): "OR", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "OR-MidGame", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "OR-FullGame", \
					
					'BM-EES'+"-"+str(numHypsX[0]): 'AXE-ES',\
					'BM-MAP'+"-"+str(numHypsX[0]): 'AXE-MS',\

					'BM-EES'+"-"+str(numHypsX[0])+"-OptimalTargets": 'AXE-ES',\
					'BM-MAP'+"-"+str(numHypsX[0])+"-OptimalTargets": 'AXE-MS',\
					'BM-EES'+"-"+str(numHypsX[0])+"-DomainTargets": 'AXE-ES-DomainTargets',\
					'BM-MAP'"-"+str(numHypsX[0])+"-DomainTargets": 'AXE-MS-DomainTargets',\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS-ES",\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS-ES",\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS-MS",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS-MS",\

					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "MEEDS-ES",\
					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "MEEDS-ES",\

					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "MEEDS-MS",\
					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "MEEDS-MS"}

	
	global lineStylesPaper
	lineStylesPaper = { "tn": "dashed",\
					'OR'+"-"+str(numHypsX[0]): "solid", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "solid", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "solid", \
					
					'BM-EES'+"-"+str(numHypsX[0]): 'solid',\
					'BM-MAP'+"-"+str(numHypsX[0]): 'dashdot',\

					'BM-EES'+"-"+str(numHypsX[0])+"-OptimalTargets": 'solid',\
					'BM-MAP'+"-"+str(numHypsX[0])+"-OptimalTargets": 'dashdot',\
					'BM-EES'+"-"+str(numHypsX[0])+"-DomainTargets": 'solid',\
					'BM-MAP'"-"+str(numHypsX[0])+"-DomainTargets": 'dashdot',\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "solid",\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "solid",\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "dashdot",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "solid",\

					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "solid",\
					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "solid",\

					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "dashdot",\
					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "solid"}


	# selectedBetas = [0.75,0.85,0.95]
	selectedBetas = [0.75,0.85,0.99]


	# Including OR as a baseline for comparison
	TN_OR = ["tn"] 
	allJTM = []
	JTM_GivenPrior = [] #["OR-"+str(numHypsX[0])]
	JTM_MinLambda = [] #["OR-"+str(numHypsX[0])]
	JTM_GivenPrior_MinLambda = [] #["OR-"+str(numHypsX[0])]
	normalJTM = []
	justBM = []
	BM_OR = []
	noJTM = []

	allSelectedBetas = ["tn"]
	justEES_SelectedBetas = ["tn"]
	BM_OR_SelectedBetas = ["tn"]
	BM_OR_JustEES_SelectedBetas = ["tn"]


	allGivenBeta = {}
	OR_BM_GivenBeta = {}
	OR_BM_GivenBeta_justEES = {}
	justEESGivenBeta = {}



	for b in betas:
		allGivenBeta[b] = ["tn"]
		OR_BM_GivenBeta[b] = ["tn"]
		OR_BM_GivenBeta_justEES[b] = ["tn"]
		justEESGivenBeta[b] = ["tn"]


	for m in methods:

		if "DomainTargets" in m:
			continue

		if "JT" in m:
			allJTM.append(m)

			if "GivenPrior" in m and "MinLambda" in m:
				JTM_GivenPrior_MinLambda.append(m)
			elif "GivenPrior" in m:
				JTM_GivenPrior.append(m)
			elif "MinLambda" in m:
				JTM_MinLambda.append(m)
			else: 
				normalJTM.append(m)

			
			for b in betas:
				allGivenBeta[b].append(m)

				if "EES" in m:
					justEESGivenBeta[b].append(m)

			allSelectedBetas.append(m)

			if "EES" in m:
				justEES_SelectedBetas.append(m)


		elif "BM" in m:
			justBM.append(m)
			BM_OR.append(m)

			tempM, beta, tt = getInfoBM(m)
			OR_BM_GivenBeta[beta].append(m)
			
			allGivenBeta[beta].append(m)

			if "EES" in m:
				justEESGivenBeta[beta].append(m)
				OR_BM_GivenBeta_justEES[beta].append(m)

			if beta in selectedBetas:
				allSelectedBetas.append(m)
				BM_OR_SelectedBetas.append(m)

				if "EES" in m:
					justEES_SelectedBetas.append(m)
					BM_OR_JustEES_SelectedBetas.append(m)					


		if "JT" not in m:
			noJTM.append(m)

		if "OR" in m:
			TN_OR.append(m)
			#allJTM.append(m)
			normalJTM.append(m)
			BM_OR.append(m)

			for b in betas:
				allGivenBeta[b].append(m)
				OR_BM_GivenBeta[b].append(m)
				OR_BM_GivenBeta_justEES[b].append(m)
				justEESGivenBeta[b].append(m)

			allSelectedBetas.append(m)
			justEES_SelectedBetas.append(m)
			BM_OR_SelectedBetas.append(m)
			BM_OR_JustEES_SelectedBetas.append(m)

	
	# code.interact("...", local=dict(globals(), **locals()))

	
	#################################################
	# Assuming same beta for all agent types for now
	# Need to set to best beta per domain according 
	# to plots (MSE-AcrossAllAgentTypes)
	#################################################
	
	makeFolder(args.resultsFolder,"BETAS")

	givenBeta = 0.50

	#################################################


	labels = [f"ALL-GivenBeta{givenBeta}","TN-OR","NormalJTM","AllJTM",
			"JTM-GivenPrior","JTM-MinLambda","JTM-GivenPriorMinLambda",
			"NoJTM","JustBM",f"OR-BM-GivenBeta{givenBeta}",
			"All-SelectedBetas","JustEES-SelectedBetas",
			"OR-BM-SelectedBetas","OR-BM-JustEES-SelectedBetas"]

	for b in betas:
		labels.append(f"ALL-Beta{b}")

	for b in betas:
		labels.append(f"OR-BM-Beta{b}")

	for b in betas:
		labels.append(f"ALL-JustEES-Beta{b}")

	for b in betas:
		labels.append(f"OR-BM-JustEES-Beta{b}")

	
	methodsLists = [methods,TN_OR,normalJTM,allJTM,
			JTM_GivenPrior,JTM_MinLambda,JTM_GivenPrior_MinLambda,
			noJTM,justBM,BM_OR,
			allSelectedBetas, justEES_SelectedBetas,
			BM_OR_SelectedBetas,BM_OR_JustEES_SelectedBetas]

	for b in betas:
	 	methodsLists.append(allGivenBeta[b])

	for b in betas:
		methodsLists.append(OR_BM_GivenBeta[b])

	for b in betas:
		methodsLists.append(justEESGivenBeta[b])

	for b in betas:
		methodsLists.append(OR_BM_GivenBeta_justEES[b])


	##################################################




	size1 = len(os.listdir(args.resultsFolder + os.sep + "ProcessedResultsFiles" + os.sep))
	size2 = len(os.listdir(args.resultsFolder + os.sep + "ProcessedResultsFilesForPlots" + os.sep))



	if size1 == size2:
		print("Folders have similar sizes. Loading already computed info.\n")
	else:
		print("New experiments added. Need to recompute info.\n")

		# Compute Mean Squared Error
		# AND will create initial copy of rdFile into prdFile
		print("\ncomputeMSE()...")
		startTime = time()
		computeMSE(processedRFsAgentNames)
		print("Time: ", time()-startTime)
		print()

		'''
		print("computeMeanEstimates()...")
		startTime = time()
		computeMeanEstimates(processedRFsAgentNames)
		print("Time: ", time()-startTime)
		print()
		

		# Compute Mean of Average Reward & True Rewards across experiments 
		print("computeMeanAVGAndTrueRewardsPerState()...")
		startTime = time()
		computeMeanAVGAndTrueRewardsPerState(processedRFsAgentNames)
		print("Time: ", time()-startTime)
		print()

		print("computeMeanAvgRewardsPerExp()...")
		startTime = time()
		computeMeanAvgRewardsPerExp(processedRFsAgentNames)
		print("Time: ", time()-startTime)

		'''



		'''
		#######################################################################
		# Used to remove any method not to be on final plots
		#######################################################################
		
		actualMethodsOnExpsPrev = deepcopy(actualMethodsOnExps)

		temp = []

		for m in actualMethodsOnExps:
			if ("BM-MAP" not in m) and ("tn" not in m): #and ("MAP" not in m):
				temp.append(m)

		actualMethodsOnExps = temp
		
		#######################################################################
		'''

		print("\n")

		# '''


	#######################################################################
	# Unindented to redo processed result files from JAIR-23 exps
	# Computes percentTrueP
	print("plotPercentRandMaxRewardObtainedPerXskillPerAgentType()...")
	startTime = time()
	plotPercentRandMaxRewardObtainedPerXskillPerAgentType(processedRFsAgentNames, agentTypes, args.resultsFolder, seenAgents, domain, pconfPerXskill)
	print("Time: ", time()-startTime)
	print()
	# '''

	# Converting from estimate to percent RandMax Reward
	print("computeMSEPercentPskillsMethods()...")
	startTime = time()
	computeMSEPercentPskillsMethods(processedRFsAgentNames,actualMethodsOnExps,pconfPerXskill,numStates,numHypsX,numHypsP,domain)
	print("Time: ", time()-startTime)
	print()
	#######################################################################



	# '''
	# Computes percentTrueP
	print("plotPercentRandMaxRewardObtainedPerXskillPerAgentType()...")
	startTime = time()
	plotPercentRandMaxRewardObtainedPerXskillPerAgentType(processedRFsAgentNames, agentTypes, args.resultsFolder, seenAgents, domain, pconfPerXskill)
	print("Time: ", time()-startTime)
	print()
	# '''

	# Plots MSE for pskill methods - in percent terms
	print("plotMSEPercentPerAgentTypes()...")
	startTime = time()
	plotMSEPercentPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,args.resultsFolder,domain)
	print("Time: ", time()-startTime)
	print()
	
	# '''

	# '''
	print("computeAndPlotMSEAcrossAllAgentsTypesAllMethods()...")
	startTime = time()
	computeAndPlotMSEAcrossAllAgentsTypesAllMethods(processedRFsAgentNames,actualMethodsNames,actualMethodsOnExps,args.resultsFolder,seenAgents,numStates,domain,betas,givenBeta,makeOtherPlots=True)
	print("Time: ", time()-startTime)
	# print()
	# '''

	# print("computeAndPlotMSEAcrossAllAgentsPerMethod()...")
	# startTime = time()
	# computeAndPlotMSEAcrossAllAgentsPerMethod(processedRFsAgentNames,actualMethodsNames,actualMethodsOnExps,args.resultsFolder,seenAgents,numStates,domain,betas,givenBeta)
	# print("Time: ", time()-startTime)
	# print()



	# '''
	print("plotMSExSkillsPerBucketsPerAgentTypes()...")
	startTime = time()
	plotMSExSkillsPerBucketsPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,args.resultsFolder,domain,numStates,numHypsX,numHypsP)
	print("Time: ", time()-startTime)
	#print()
	'''

	'''
	print("plotMSEpSkillsPerBucketsPerAgentTypes()...")
	startTime = time()
	plotMSEpSkillsPerBucketsPerAgentTypes(processedRFsAgentNames,actualMethodsOnExps,args.resultsFolder,domain,numStates,numHypsX,numHypsP)
	print("Time: ", time()-startTime)
	print()
	# '''

	'''
	print("plotMeanEstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType()...")
	startTime = time()
	plotMeanEstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType(processedRFsAgentNames,actualMethodsNames,actualMethodsOnExps,numHypsX,numHypsP,args.resultsFolder,agentTypes,seenAgents,givenBeta)
	print("Time: ", time()-startTime)
	print()
	'''

	# '''
	print("plotContourMSE_xSkillpSkillPerAgentTypePerMethod()...")
	startTime = time()
	plotContourMSE_xSkillpSkillPerAgentTypePerMethod(processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates,domain)
	print("Time: ", time()-startTime)
	print()
	# '''

	# '''
	print("plotContourEstimates_xSkillpSkill_PerAgentTypePerMethod()...")
	startTime = time()
	plotContourEstimates_xSkillpSkill_PerAgentTypePerMethod(processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates,domain)
	print("Time: ", time()-startTime)
	print()
	'''

	# '''
	print("plotContourEstimates_xSkillPercent_PerAgentTypePerMethod()...")
	startTime = time()
	plotContourEstimates_xSkillPercent_PerAgentTypePerMethod(processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates,domain)
	print("Time: ", time()-startTime)
	print()
	# '''



	'''
	print("plotRewardsVSagentType()...")
	startTime = time()
	plotRewardsVSagentType(processedRFsAgentNames,numHypsX,numHypsP,args.resultsFolder,domain)
	print("Time: ", time()-startTime)
	print()


	print("plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndXskillBuckets()...")
	startTime = time()
	plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndXskillBuckets(domain,processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates)
	print("Time: ", time()-startTime)
	print()

	print("plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndPskillBuckets()...")
	startTime = time()
	plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndPskillBuckets(domain,processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates)
	print("Time: ", time()-startTime)
	print()


	print("plotPercentTimesDistributionXskillBuckets()...")
	startTime = time()
	plotPercentTimesDistributionXskillBuckets(domain,processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates)
	print("Time: ", time()-startTime)
	print()

	print("plotPercentTimesDistributionPskillBuckets()...")
	startTime = time()
	plotPercentTimesDistributionPskillBuckets(domain,processedRFsAgentNames,seenAgents,actualMethodsOnExps,args.resultsFolder,numStates)
	print("Time: ", time()-startTime)
	print()
	'''


	##################### FOR RATIONALITY PARAMETER - ALL AGENTS ######################
	'''
	
	args.resultsFolder += os.sep + "plots"
	
	for agentType in ["Flip", "Tricker", "Bounded"]:

		# Only continue to create the corresponding plots if we have seen experiments for such agent type
		if agentType in seenAgents:

			print(f"\nAgent Type: {agentType}")
			makeFolder2(args.resultsFolder, agentType)

			print("\tplotMSEAllRationalityParamsAllMethods()...")
			plotMSEAllRationalityParamsAllMethods(processedRFsAgentNames,actualMethodsNames,actualMethodsOnExps,numHypsX,numHypsP,args.resultsFolder,agentType)

			print("\tplotMSEAllRationalityParamsPerMethods()...")
			plotMSEAllRationalityParamsPerMethods(processedRFsAgentNames,actualMethodsNames,actualMethodsOnExps,numHypsX,numHypsP,args.resultsFolder,agentType)

			print("\tplotRationalityParamsVsSkillEstimatePerMethod()...")
			plotRationalityParamsVsSkillEstimatePerMethod(processedRFsAgentNames,numHypsX,numHypsP,actualMethodsOnExps,args.resultsFolder,agentType)
	
	'''
	###################################################################################


	# Close all remaining figures
	plt.close("all")


	# code.interact("End.", local=dict(globals(), **locals()))



