import argparse
import os
import pickle
from time import time

from matplotlib.patches import Ellipse
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors
from matplotlib import rc

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from scipy import stats

from utilsDartsMulti import *

import pandas as pd

from scipy.stats import multivariate_normal


infoIndexes = {}

# Number of sample indexes (for selecting sample of available exps)
amount = 2


# For zooming in at the end on plotObservationsVsMetricPerAgentType plot
startZoom = 50 #6
zoom = 2.0


rc('text', usetex=True)


def plotObservationsVsMetric_AbruptAgents_Centered(metric,maxY,givenAgents,givenAgentsType,subsetLabel):

	label =  metric.replace(" ","")
	labelSave =  "plotsObservationsVs" + metric.replace(" ","") + f"-AbruptAgents-Centered-{subsetLabel}"
	makeFolder3(f"{plotFolder}{labelSave}{os.sep}")

	tempLabel = f"{plotFolder}{labelSave}{os.sep}{givenAgentsType}{os.sep}"
	makeFolder3(tempLabel)

	labelSave2 = f"{tempLabel}AllMethods-SamePlot{os.sep}"
	makeFolder3(labelSave2)


	centerAt = numStates // 2
	print(f"centerAt: {centerAt}")

	step = numStates//3

	shift = abs(centerAt-step)


	metricPerObservation = {}

	for m in methods:

		if "pSkills" in m or "rhos" in m:
			continue

		metricPerObservation[m] = {}	

		for i in range(0-shift,numStates+shift):
			metricPerObservation[m][i] = {"metric":[],"count":0,"Average":0.0,"Standard Deviation":0.0}	

		# print(metricPerObservation[m].keys())


	resultsDict = {}

	# code.interact("...", local=dict(globals(), **locals()))

	for a in givenAgents:

		if "Change" in a:
			agentType = a.split("|")[0].split("Change")[0]
		else:
			agentType = a.split("Agent")[0]


		result = loadProcessedInfo(prdFile,a)

		if result == False:
			continue
		
		# Load processed info		
		resultsDict[a] = result

		
		info = resultsDict[a]["infoMetrics"]

		tempA = a.replace("|","_")
		tempA = "_".join(tempA.split("_")[:-1])

		# Search for respective rfs
		for ib in processedRFs:
			if tempA in ib:
				b = ib
				break

		try:
			with open(args.resultsFolder+os.sep+"results"+os.sep+b,"rb") as infile:
				results = pickle.load(infile)
		except:
			continue

		changeAtExp = results["changeAt"]
		shiftBy = centerAt-changeAtExp


		del results
		del resultsDict[a]


		# For each seen exp for the given agent
		for each in info:

			for m in methods:

				if "pSkills" in m or "rhos" in m:
					continue

				try:
					metric = each["methods"][m][label]
				except:
					# Case method not present on rfs
					continue
					# code.interact("...", local=dict(globals(), **locals()))


				if type(metric[0]) == list:
					metric = np.array(metric)	
					infoMetric = metric[:,0]
				else:
					metric = np.array(metric)
					infoMetric = metric


				for eachObs in range(len(infoMetric)):

					index = eachObs + shiftBy
					mm = infoMetric[eachObs]

					metricPerObservation[m][index]["metric"].append(mm)

					metricPerObservation[m][index]["count"] += 1




	# COMPUTE METRIC
	for m in metricPerObservation:
		for iii in metricPerObservation[m]:
			if metricPerObservation[m][iii]["count"] != 0:

				for toPlot in ["Average","Standard Deviation"]:

					if toPlot == "Average":
						metricPerObservation[m][iii][toPlot] = sum(metricPerObservation[m][iii]["metric"])/metricPerObservation[m][iii]["count"]
					else: # Compute standard deviation
						metricPerObservation[m][iii][toPlot] = np.std(metricPerObservation[m][iii]["metric"])
					


	makeFolder3(f"{labelSave2}All{os.sep}")


	# MAKE PLOT - ALL METHODS SAME PLOT

	for toPlot in ["Average","Standard Deviation"]:

		for tt in range(2):

			fig = plt.figure()
			ax = plt.subplot(1,1,1)
			
			plt.margins(0.05)

			tempLabel = ""

			for m in metricPerObservation:

				xs = []
				ys = []

				for ei in metricPerObservation[m]:
					xs.append(ei)
					ys.append(metricPerObservation[m][ei][toPlot])

				# Plot the data points
				ax.plot(xs,ys,label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])


			ax.set_xlabel(r'\textbf{Number of Observations')
			ax.set_ylabel(r'\textbf{'+toPlot+' '+label+'}')

			# Zoomed in version of plot
			if tt == 1:
				ax.set_xlim(centerAt-shift,centerAt+shift)
				tempLabel = "-Zoom"
				print(centerAt-shift,centerAt+shift)

			ax.set_ylim(0,20)
			ax.legend()

			plt.savefig(f"{labelSave2}All{os.sep}plot-centerAt{centerAt}-{toPlot.replace(' ','')}{tempLabel}.png")
			plt.clf()
			plt.close("all")


	# code.interact("...", local=dict(globals(), **locals()))

	
	# MAKE PLOT - ALL METHODS SAME PLOT - Different methods set
	for ii in range(len(labels)):

		methodLabel = labels[ii]
		tempMethods = methodsLists[ii]

		labelSave3 = f"{labelSave2}{os.sep}{methodLabel}{os.sep}"
		makeFolder3(labelSave3)

		for toPlot in ["Average","Standard Deviation"]:

			for tt in range(2):

				tempLabel = ""

				fig = plt.figure()
				ax = plt.subplot(1,1,1)
				
				plt.margins(0.05)

				for m in tempMethods:

					if "xSkills" not in m:
						continue

					xs = []
					ys = []

					for ei in metricPerObservation[m]:
						xs.append(ei)
						ys.append(metricPerObservation[m][ei][toPlot])


					# Plot the data points
					ax.plot(xs,ys,label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])


				ax.set_xlabel(r'\textbf{Number of Observations')
				ax.set_ylabel(r'\textbf{'+toPlot+' '+label+'}')

				# Zoomed in version of plot
				if tt == 1:
					ax.set_xlim(centerAt-shift,centerAt+shift)
					tempLabel = "-Zoom"

				ax.set_ylim(0,20)
				ax.legend()

				plt.savefig(f"{labelSave3}plot-centerAt{centerAt}-{toPlot.replace(' ','')}{tempLabel}.png")
				plt.clf()
				plt.close("all")


def plotObservationsVsMetricPerAgentType_xbyp(metric,givenAgents,givenAgentsType):

	# BUCKETS PER PERCENTS RAND/MAX REWARD -- SHOWING METRIC FOR XSKILL METHODS

	label =  metric.replace(" ","")
	labelSave =  "plotsObservationsVs" + metric.replace(" ","") + "-AveragedAcrossAgentType-xbyp"
	makeFolder3(f"{plotFolder}{labelSave}{os.sep}")

	tempLabel = f"{plotFolder}{labelSave}{os.sep}{givenAgentsType}{os.sep}"
	makeFolder3(tempLabel)

	labelSave1 = f"{tempLabel}PerMethod{os.sep}"
	makeFolder3(labelSave1)


	
	# Buckets in percents terms - between 0-1
	buckets = [0.25,0.50,0.75,1.0]

	# labelsB = {0.25:"0.00-0.25",0.50:"0.25-0.50",0.75:"0.50-0.75",1.0:"0.75-1.00"}
	labelsB = {0.25:"0%-25%",0.50:"25%-50%",0.75:"50%-75%",1.0:"75%-100%"}


	# init dict to store info
	metricDict = {}


	for at in seenAgents:
		metricDict[at] = {"perMethod": {}, "numAgents": {}}


		for b in buckets:
			metricDict[at]["numAgents"][b] = 0.0


		for m in methods:
			# Skip pskill methods
			if "-pSkills" in m or "rhos" in m:
				continue

			if "tn" in m:
				continue

			metricDict[at]["perMethod"][m] = {}


			for b in buckets:
				metricDict[at]["perMethod"][m][b] = [0.0] * numStates # to store per state - across different exps per agent type



	resultsDict = {}

	for a in givenAgents:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)

		info = resultsDict[a]["infoMetrics"]


		# update agent count
		# metricDict[aType]["numAgents"] += resultsDict[a]["num_exps"]

		#estimatedP = resultsDict[a]["mse_percent_pskills"][method][numStates-1] # #### ESTIMATED %

		trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
		# using true percent and not estimated one


		# Find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break

		# Get actual bucket
		b = buckets[b]

		
		del resultsDict[a]


		# For each seen exp for the given agent
		for each in info:

			for m in each["methods"]:

				if "pSkills" in m or "rhos" in m:
					continue

				metricInfo = each["methods"][m][label]

				if type(metricInfo[0]) == list:
					metricInfo = np.array(metricInfo)	
					metricDict[aType]["perMethod"][m][b] += metricInfo[:,0]
				else:
					metricInfo = np.array(metricInfo)
					metricDict[aType]["perMethod"][m][b] += metricInfo

			metricDict[aType]["numAgents"][b] += 1.0



	# Normalize
	for at in seenAgents:
		for m in methods:

			# Skip pskill methods
			if "-pSkills" in m or "rhos" in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:

				if metricDict[at]["numAgents"][b] != 0:

					# for each state
					for mxi in range(numStates):
						metricDict[at]["perMethod"][m][b][mxi] /= metricDict[at]["numAgents"][b]



	colors = ["red", "green", "blue", "orange"]

	for at in seenAgents:

		for m in methods:

			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)

			
			# Skip pxskill methods
			if "-pSkills" in m or "rhos" in m:
				continue

			if "tn" in m:
				continue

			c = 0
			for b in buckets:
				if np.count_nonzero(metricDict[at]["perMethod"][m][b]) != 0:
					# print "b: ", b, "| color: ", colors[c] 
					# plt.plot(range(numStates),metricDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = colors[c])
					plt.semilogx(range(numStates),metricDict[at]["perMethod"][m][b], lw=2.0, label=labelsB[b], c = colors[c])
				c += 1


			plt.xlabel('Number of Observations',fontsize=18)
			plt.ylabel(f'{metric}', fontsize=18)

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

			plt.savefig(f"{labelSave1}results-Agent{at}-Method{m}.png", bbox_inches='tight')

			plt.clf()
			plt.close("all")


def plotObservationsVsMetricPerAgentType_pbyx(givenAgents,givenAgentsType):

	labelSave =  "plotsObservationsVsMSEPercent" + "-AveragedAcrossAgentType-pbyx"
	makeFolder3(f"{plotFolder}{labelSave}{os.sep}")

	tempLabel = f"{plotFolder}{labelSave}{os.sep}{givenAgentsType}{os.sep}"
	makeFolder3(tempLabel)

	labelSave1 = f"{tempLabel}PerMethod{os.sep}"
	makeFolder3(labelSave1)


	
	if domain == "1d":
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif "2d" in domain or domain == "sequentialDarts":
		buckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]

	cmap = plt.get_cmap("rainbow")
	norm = plt.Normalize(min(buckets),max(buckets))


	# init dict to store info
	metricDict = {}

	for at in seenAgents:
		metricDict[at] = {"perMethod": {}, "numAgents": {}}

		for b in buckets:
			metricDict[at]["numAgents"][b] = 0.0


		for m in methods:
			# Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			metricDict[at]["perMethod"][m] = {}

			for b in buckets:
				metricDict[at]["perMethod"][m][b] = [0.0] * numStates # to store per state - across different exps per agent type


	resultsDict = {}


	for a in givenAgents:

		aType, x, p = getParamsFromAgentName(a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		# update agent count
		# metricDict[aType]["numAgents"] += resultsDict[a]["num_exps"]

		#estimatedP = resultsDict[a]["mse_percent_pskills"][method][numStates-1] # #### ESTIMATED %

		trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
		# using true percent and not estimated one


		# Find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break

		# Get actual bucket
		b = buckets[b]

		
		metricDict[aType]["numAgents"][b] += 1.0


		# for each method
		for m in methods:

			#Skip pskill methods
			if "-pSkills" not in m:
				continue

			for mxi in range(numStates):

				# mse percent
				sq = resultsDict[a]["mse_percent_pskills"][m][mxi]

				# store squared error
				metricDict[aType]["perMethod"][m][b][mxi] += sq


		del resultsDict[a]


	# Normalize
	for at in seenAgents:
		for m in methods:

			# Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:

				if metricDict[at]["numAgents"][b] != 0:

					# for each state
					for mxi in range(numStates):
						metricDict[at]["perMethod"][m][b][mxi] /= metricDict[at]["numAgents"][b]



	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	# colors = ["red", "green", "blue", "orange"]


	for at in seenAgents:

		for m in methods:

			maxLast = -9999
			minLast = 9999
			

			# Skip pxskill methods
			if "-pSkills" not in m:
				continue

			
			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)


			c = 0
			for b in buckets:
				if np.count_nonzero(metricDict[at]["perMethod"][m][b]) != 0:
					# print "b: ", b, "| color: ", colors[c] 
					# color = colors[list(colors.keys())[c]]
					color = cmap(norm(b))
					# plt.plot(range(numStates),metricDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
					plt.semilogx(range(numStates),metricDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
				
				c += 1


			plt.xlabel('Number of Observations',fontsize=18)
			plt.ylabel(f'Mean Squared Error', fontsize=18)

			plt.margins(0.05)

			# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


			fig.subplots_adjust(hspace= 1.0, wspace=1.0)

			elements = []

			for i in range(len(buckets)):
				# elements.append(Line2D([0],[0], color = colors[list(colors.keys())[i]],label = buckets[i]))
				elements.append(Line2D([0],[0], color = cmap(norm(buckets[i])),label = buckets[i]))

				
			# Put a legend to the right of the current axis
			plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))


			plt.savefig(f"{labelSave1}results-Agent{at}-Method{m}.png", bbox_inches='tight', pad_inches = 0)

			plt.clf()
			plt.close("all")


# Computes MSE for rationality percentages (mse_percent_pskills)
def computeMSEPercentPskillsMethods(processedRFsAgentNames):

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
		for m in methods:

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
def plotPercentRandMaxRewardObtainedPerXskillPerAgentType(processedRFsAgentNames):

	resultsDict = {}

	makeFolder(args.resultsFolder, "percentRandMaxRewardObtained-PerXskillPerAgentType")

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["axes.titleweight"] = "bold"


	percentOfRewardPerAgent = {}

	for at in seenAgents:

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
	

		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)


		del resultsDict[a]


		percentOfRewardPerAgent[aType]["allPercents"].append(percentOfReward)
		percentOfRewardPerAgent[aType]["allXskills"].append(xStr)
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

		plt.scatter(np.asarray(percentOfRewardPerAgent[aType]["allPskills"]),np.asarray(percentOfRewardPerAgent[aType]["allPercents"]))
					# ,\
					# c =  cmap(norm(np.asarray(percentOfRewardPerAgent[aType]["allXskills"]))))
					# c =  np.asarray(percentOfRewardPerAgent[aType]["allXskills"]))

		
		plt.xlabel(r'\textbf{Rationality Parameter}')
		plt.ylabel(r'\textbf{Rationality Percentage}')
		

		# sm = ScalarMappable(norm = norm, cmap = cmap)
		# sm.set_array([])

		cbar = plt.colorbar()
		cbar.ax.set_title("Execution Noise Level", fontdict = {'verticalalignment': 'center', 'horizontalalignment': "center"},\
						    y = 0.50, rotation = 90)
		# cbar.set_label(r'\textbf{Execution Noise Level}',size=14))

		plt.margins(0.05)

		plt.savefig(args.resultsFolder + os.sep + "plots" + os.sep + "percentRandMaxRewardObtained-PerXskillPerAgentType" + os.sep  + "results-Percent-RandMaxRewards-Agent-"+aType+".png", bbox_inches='tight')

		plt.clf()
		plt.close("all")


		#######################################################################################################################################
	

	# code.interact("after percent rationality...", local=dict(globals(), **locals()))


def plotDistribution(givenProcessedRFsAgentNames,metric):

	XD = np.arange(-340.0,341.0,delta)
	YD = np.arange(-340.0,341.0,delta)


	XXD,YYD = np.meshgrid(XD,YD,indexing="ij")


	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(0.0,1.0)
	sm = ScalarMappable(cmap=cmap)
	sm.set_array([])


	makeFolder3(f"{plotFolder}plotsDistributions{os.sep}")
	makeFolder3(f"{plotFolder}plotsDistributionsOverlay{os.sep}")

	# Which observations to look at for each sample exp

	if numStates > 5:
		#lookAt = [[0,"Obs-0"],[1,"Obs-1"],[2,"Obs-2"],[3,"Obs-3"],[4,"Obs-4"],[-1,"Obs-End"]]
		lookAt = [[-1,"Obs-End"]]

		# for each in range(5,28):
			# lookAt.append([each,f"Obs-{each}"])
			
	elif numStates > 2:
		lookAt = [[0,"Obs-0"],[1,"Obs-1"],[-1,"Obs-End"]]
	# Case just 1 observation
	else:
		lookAt = [[0,"Obs-0"],[-1,"Obs-End"]]



	for m in methods:

		if "pSkills" in m or "rhos" in m:
			continue

		folder1 = f"{args.resultsFolder}plots{os.sep}plotsDistributions{os.sep}{m}{os.sep}"
		makeFolder3(folder1)

		folder2 = f"{args.resultsFolder}plots{os.sep}plotsDistributionsOverlay{os.sep}{m}{os.sep}"
		makeFolder3(folder2)

		for agentType in seenAgents:
			tempFolder1 = f"{folder1}{agentType}{os.sep}"
			makeFolder3(tempFolder1)

			tempFolder2 = f"{folder2}{agentType}{os.sep}"
			makeFolder3(tempFolder2)

	
	resultsDict = {}

	seenIndexes = []


	for a in givenProcessedRFsAgentNames:

		if "Change" in a:
			agentType = a.split("|")[0].split("Change")[0]
		else:
			agentType = a.split("Agent")[0]

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)

		trueXS,trueR = getXandR_FromAgentName(a)

		agentInfo = {}	
		agentInfo["trueXS"] = trueXS

		print(a)

		tempA = a.replace("|","_")
		tempA = "_".join(tempA.split("_")[:-1])

		# Search for respective rfs
		for ib in processedRFs:
			if tempA in ib:
				b = ib
				break

		if "Abrupt" in a or "Gradual" in a:

			try:
				with open(args.resultsFolder+os.sep+"results"+os.sep+b,"rb") as infile:
					results = pickle.load(infile)
		
				if "Abrupt" in a:
					agentInfo["changeAt"] = results["changeAt"]
					agentInfo["start"] = trueXS[0] 
					agentInfo["end"] = trueXS[1]
					# print(agentInfo["changeAt"])
				elif "Gradual" in a:
					agentInfo["gradualXskills"] = results["gradualXskills"]
					# print(agentInfo["gradualXskills"])

				del results

			except Exception as e:
				# code.interact("...", local=dict(globals(), **locals()))
				continue


		if a not in infoIndexes:
			# Randomly select sample experiments
			try:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),amount,replace=False)
			# Case when not enough samples (select just 1 then)
			except:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),1,replace=False)
			
			infoIndexes[a] = indexes
		else:
			indexes = infoIndexes[a]


		for index in indexes:

			info = resultsDict[a]["infoCovError"][index]
		
			states = info["states"]


			for m in methods:

				if "pSkills" in m or "rhos" in m:
					continue

				folder1 = f"{args.resultsFolder}plots{os.sep}plotsDistributions{os.sep}{m}{os.sep}{agentType}{os.sep}"
				folder2 = f"{args.resultsFolder}plots{os.sep}plotsDistributionsOverlay{os.sep}{m}{os.sep}{agentType}{os.sep}"

				
				estimatedXS = info["methods"][m]["estimatedXS"]

				if "Multi" in m:
					estimatedR = info["methods"][m]["estimatedR"]


				# For the 1st and last observation only
				for when,which in lookAt:

					# Recreate true coMatrix based on type of agent
					if "Change" in a:
						
						if "Abrupt" in a:
							changeAt = agentInfo["changeAt"]

							if when < changeAt:
								trueXS = agentInfo["start"]
							else:
								trueXS = agentInfo["end"]

						elif "Gradual" in a:
							trueXS = agentInfo["gradualXskills"][when]

						trueCovMatrix = domainModule.getCovMatrix(trueXS,trueR)

					else:
						trueXS = agentInfo["trueXS"]
						trueCovMatrix = domainModule.getCovMatrix(trueXS,trueR)


					if "JT-QRE" in m:
						# Normal JEEDS (assuming symmetric agents)
						estimatedCovMatrix = domainModule.getCovMatrix([estimatedXS[when],estimatedXS[when]],0.0)
						estX = np.round(estimatedXS[when],4)
						info2 = f"({estX},{estX},0.00)"
					else:
						estimatedCovMatrix = domainModule.getCovMatrix(estimatedXS[when],estimatedR[when])
						estR = np.round(estimatedR[when],4)
						estX = np.round(estimatedXS[when],4)
						info2 = f"({estX[0]},{estX[1]},{estR})"


					metricNum = resultsDict[a]["infoMetrics"][index]["methods"][m][metric][when]


					rng = np.random.default_rng(np.random.randint(1,1000000000))

					distrTrue = domainModule.draw_noise_sample(rng,mean=[0.0,0.0],covMatrix=trueCovMatrix)
					distrEst = domainModule.draw_noise_sample(rng,mean=[0.0,0.0],covMatrix=estimatedCovMatrix)

					
					XYD,pdfTrue = domainModule.get_symmetric_normal_distribution(rng,[0.0,0.0],trueCovMatrix,delta)
					XYD,pdfEst = domainModule.get_symmetric_normal_distribution(rng,[0.0,0.0],estimatedCovMatrix,delta)

					# Reshape cause pdf array is 1D
					pdfTrue = np.array(pdfTrue).reshape((len(XD),len(YD)))
					pdfEst = np.array(pdfEst).reshape((len(XD),len(YD)))


					fig,axs = plt.subplots(1,2,figsize=(14,8))

					axs[0].contourf(XXD,YYD,pdfTrue,cmap=cmap)

					#plt.xlabel('X axis')
					#plt.ylabel('Y axis')
					axs[0].set_xlim(-170,170)
					axs[0].set_ylim(-170,170)
					#axs[0].axis('equal')


					axs[1].contourf(XXD,YYD,pdfEst,cmap=cmap)				

					#plt.xlabel('X axis')
					#plt.ylabel('Y axis')
					axs[1].set_xlim(-170,170)
					axs[1].set_ylim(-170,170)
					#axs[1].axis('equal')


					cb_ax = fig.add_axes([1.0,0.05,0.02,0.85])
					cbar = fig.colorbar(sm,cax=cb_ax)			


					if "Target" in a:
						typeAgent = "Rational"
					elif "Bounded" in a:
						typeAgent = "Softmax"
					elif "Flip" in a:
						typeAgent = "Flip"
					else:
						typeAgent = "Deceptive"

					import re
					tempMetric = re.sub( r"([A-Z])", r" \1", metric)

					infoTrue = f"({trueXS[0]},{trueXS[1]},{trueR})"

					# title = f"{typeAgent} Agent\n"
					title = f"True Execution Skill: {infoTrue}\n"
					title += f"Estimated Execution Skill: {info2}\n"
					title += f"JD: {metricNum:.4f}"

					fig.suptitle(title)

					plt.tight_layout()

					if "Abrupt" in a:
						fn = f"{folder1}{os.sep}{a}-ChangeAt-{changeAt}-Index{index}-{which}.png"
					else:
						fn = f"{folder1}{os.sep}{a}-Index{index}-{which}.png"

					plt.savefig(fn,bbox_inches='tight')

					plt.clf()
					plt.close("all")



					# OVERLAY VERSION

					fig = plt.figure(figsize=(8,8))

					c1 = plt.contour(XXD,YYD,pdfTrue,colors="b",alpha=0.5)
					c2 = plt.contour(XXD,YYD,pdfEst,colors="r",alpha=0.5)				

					# plt.xlabel(r'\textbf{$\sigma_x$}')
					# plt.ylabel(r'\textbf{$\sigma_y$}')
					# plt.xlim(-170,170)
					# plt.ylim(-170,170)

					h1,_ = c1.legend_elements()
					h2,_ = c2.legend_elements()
					plt.legend([h1[0], h2[0]], ['True','Estimated'])


					# fig.suptitle(f'Distribution: {info2}\n{a}\n{metric}: {metricNum:.4f}')
					fig.suptitle(title)

					plt.tight_layout()	

					if "Abrupt" in a:
						fn = f"{folder2}{os.sep}{a}-ChangeAt-{changeAt}-Index{index}-{which}-Metric{metric}-{metricNum:.4f}.png"
					else:
						fn = f"{folder2}{os.sep}{a}-Index{index}-{which}-Metric{metric}-{metricNum:.4f}.png"
					
					plt.savefig(fn,bbox_inches='tight')

					plt.clf()
					plt.close("all")

					# code.interact("...", local=dict(globals(), **locals()))

			
			del states
		

		del resultsDict[a]


def plotCovErrorElipse(givenProcessedRFsAgentNames):

	makeFolder3(f"{plotFolder}covErrorElipse{os.sep}")

	for m in methods:

		if "pSkills" in m or "rhos" in m:
			continue

		folder = f"{args.resultsFolder}plots{os.sep}covErrorElipse{os.sep}{m}{os.sep}"
		makeFolder3(folder)

		for agentType in seenAgents:
			tempFolder = f"{folder}{agentType}{os.sep}"
			makeFolder3(tempFolder)

	
	resultsDict = {}

	for a in givenProcessedRFsAgentNames:

		if "Change" in a:
			agentType = a.split("|")[0].split("Change")[0]
		else:
			agentType = a.split("Agent")[0]

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if a not in infoIndexes:
			# Randomly select sample experiments
			try:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),amount,replace=False)
			# Case when not enough samples (select just 1 then)
			except:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),1,replace=False)
			
			infoIndexes[a] = indexes
		else:
			indexes = infoIndexes[a]


		for index in indexes:

			info = resultsDict[a]["infoCovError"][index]
	
			states = info["states"]


			for m in methods:

				if "pSkills" in m or "rhos" in m:
					continue

				folder = f"{args.resultsFolder}plots{os.sep}covErrorElipse{os.sep}{m}{os.sep}{agentType}{os.sep}"

				noisyActions = info["noisyActions"]
				intendedActions = info["intendedActions"]
				infoElipse = info["methods"][m]["infoElipse"]

				estimatedXS = info["methods"][m]["estimatedXS"]

				if "Multi" in m:
					estimatedR = info["methods"][m]["estimatedR"]


				if "Multi" in m:
					estX = np.round(estimatedXS[-1],4)
					estR = np.round(estimatedR[-1],4)

					info2 = f"X: {estX} | R: {estR}"

				else:
					estX = np.round(estimatedXS[-1],4)
					info2 = f"X: {estX}"


				fig = plt.figure()
				ax = plt.gca()

				# Plot Dartboard
				domainModule.draw_board(ax)
				domainModule.label_regions(states[-1],color="black")

				# Plot the data points
				plt.scatter(noisyActions[:,0],noisyActions[:,1], label='Noisy Actions')
				plt.scatter(intendedActions[:,0],intendedActions[:,1], label='Intended Actions')

				# Plot the covariance error ellipse
				ellipse = Ellipse(xy=np.mean(noisyActions,axis=0),
								  width=infoElipse["majorAxis"], height=infoElipse["minorAxis"],
								  angle=infoElipse["rotationAngle"],
								  edgecolor='red', fc='None', lw=2, label='Covariance Error Ellipse')
				plt.gca().add_patch(ellipse)

				# Set axis labels and legend
				plt.xlabel('X-axis')
				plt.ylabel('Y-axis')
				plt.legend()

				# Show the plot
				plt.title(f'Covariance Error Ellipse: {info2}\n{a}')

				plt.savefig(f"{folder}{os.sep}{a}-Index{index}.png",bbox_inches='tight')
				plt.clf()
				plt.close("all")

				# code.interact("...", local=dict(globals(), **locals()))


			del states


		del resultsDict[a]


def plotObservationsVsMetricPerAgentType(metric,maxY,givenAgents,givenAgentsType,subsetLabel):

	makeFolder3(f"{args.resultsFolder}metricsAVG{os.sep}")
	makeFolder3(f"{args.resultsFolder}metricsAVG{os.sep}{subsetLabel}")

	label =  metric.replace(" ","")
	labelSave =  "plotsObservationsVs" + metric.replace(" ","") + f"-AveragedAcrossAgentType-{subsetLabel}"
	makeFolder3(f"{plotFolder}{labelSave}{os.sep}")

	tempLabel = f"{plotFolder}{labelSave}{os.sep}{givenAgentsType}{os.sep}"
	makeFolder3(tempLabel)

	labelSave1 = f"{tempLabel}PerMethod{os.sep}"
	makeFolder3(labelSave1)

	labelSave2 = f"{tempLabel}AllMethods-SamePlot{os.sep}"
	makeFolder3(labelSave2)


	metricPerAgentType = {}
	
	for agentType in seenAgents:
		metricPerAgentType[agentType] = {}	


	resultsDict = {}

	for a in givenAgents:

		if "Change" in a:
			agentType = a.split("|")[0].split("Change")[0]
		else:
			agentType = a.split("Agent")[0]
		

		result = loadProcessedInfo(prdFile,a)

		if result == False:
			continue
		
		# Load processed info		
		resultsDict[a] = result

		
		info = resultsDict[a]["infoMetrics"]


		del resultsDict[a]


		# For each seen exp for the given agent
		for each in info:

			for m in methods:

				if "pSkills" in m or "rhos" in m:
					continue

				if m not in metricPerAgentType[agentType]:
					metricPerAgentType[agentType][m] = {"numExps":0, "info":[0.0]*numStates, "avgMetric":None,"infoPerExp":[],"agents":[]}

				try:
					metric = each["methods"][m][label]
				except:
					continue
					#code.interact("...", local=dict(globals(), **locals()))


				if type(metric[0]) == list:
					metric = np.array(metric)	
					metricPerAgentType[agentType][m]["info"] += metric[:,0]
					metricPerAgentType[agentType][m]["infoPerExp"].append(metric[:,0])

				else:
					metric = np.array(metric)
					metricPerAgentType[agentType][m]["info"] += metric
					metricPerAgentType[agentType][m]["infoPerExp"].append(metric)


				metricPerAgentType[agentType][m]["numExps"] += 1
				metricPerAgentType[agentType][m]["agents"].append(a)


	# FIND AVERAGE
	for at in seenAgents:
		for m in metricPerAgentType[at].keys():
			metricPerAgentType[at][m]["avgMetric"] = metricPerAgentType[at][m]["info"]/metricPerAgentType[at][m]["numExps"]


	# MAKE PLOT - PER METHOD
	for at in seenAgents:

		for m in metricPerAgentType[at].keys():

			fig = plt.figure()
			ax = plt.gca()

			# Plot the data points
			plt.plot(range(len(metricPerAgentType[at][m]["avgMetric"])),metricPerAgentType[at][m]["avgMetric"])

			# Set axis labels and legend
			plt.xlabel('Number of Observations')
			plt.ylabel(f"{label}")


			plt.savefig(f"{labelSave1}Agent-{at}-Method-{m}-NumExps{metricPerAgentType[at][m]['numExps']}.png",bbox_inches='tight')
			plt.clf()
			plt.close("all")



	makeFolder3(f"{labelSave2}All{os.sep}")


	# MAKE PLOT - ALL METHODS SAME PLOT
	for at in seenAgents:

		minLast = 999999
		maxLast = -999999

		fig = plt.figure()
		ax = plt.subplot(1,1,1)
		
		plt.margins(0.05)


		# x0,y0,w,h
		# axins = zoomed_inset_axes(ax,zoom=zoom,loc=1,borderpad=1)
		# axins = inset_axes(ax,
		# 		width="40%", # width = 40% of parent_bbox
		# 		height=1., # height : 1 inch
		# 		loc=1)


		for m in metricPerAgentType[at].keys():

			# Plot the data points
			ax.plot(range(1,len(metricPerAgentType[at][m]["avgMetric"])+1),metricPerAgentType[at][m]["avgMetric"],label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])
			# axins.plot(range(1,len(metricPerAgentType[at][m]["avgMetric"])+1),metricPerAgentType[at][m]["avgMetric"],label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])

			# Set axis labels and legend
			ax.set_xlabel(r'\textbf{Number of Observations')
			ax.set_ylabel(r'\textbf{' +label+'}')


			if metricPerAgentType[at][m]["avgMetric"][-1] < minLast:
				minLast = metricPerAgentType[at][m]["avgMetric"][-1]

			if metricPerAgentType[at][m]["avgMetric"][-1] > maxLast:
				maxLast = metricPerAgentType[at][m]["avgMetric"][-1]


		# ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
		ax.legend()

		# subregion of the original image
		# axins.set_xlim(startZoom,numStates)

		minLast = -0.05

		# axins.set_ylim(minLast,maxLast+(maxLast*0.20))

		# axins.set_xticklabels([])
		# axins.set_xticklabels([],minor=True)
		# plt.xticks(visible=False) 

		# mark_inset(ax,axins,loc1=2,loc2=4,fc="none",ec="0.5")

		plt.savefig(f"{labelSave2}All{os.sep}Agent-{at}.png")
		plt.clf()
		plt.close("all")

		# code.interact("...", local=dict(globals(), **locals()))


	
	# MAKE PLOT - ALL METHODS SAME PLOT - Different methods set
	for ii in range(len(labels)):

		methodLabel = labels[ii]
		tempMethods = methodsLists[ii]

		labelSave3 = f"{labelSave2}{os.sep}{methodLabel}{os.sep}"
		makeFolder3(labelSave3)


		for at in seenAgents:

			if len(metricPerAgentType[at]) == 0:
				continue

			minLast = 999999
			maxLast = -999999

			fig = plt.figure()
			ax = plt.subplot(1,1,1)
			
			plt.margins(0.05)


			# x0,y0,w,h
			# axins = zoomed_inset_axes(ax,zoom=zoom,loc=1,borderpad=1)
			# axins = inset_axes(ax,
			# 		width="40%", # width = 40% of parent_bbox
			# 		height=1., # height : 1 inch
			# 		loc=1)

			for m in tempMethods:

				if "xSkills" not in m:
					continue

				endZoom = len(metricPerAgentType[at][m]["avgMetric"])


				# Plot the data points
				ax.plot(range(1,len(metricPerAgentType[at][m]["avgMetric"])+1),metricPerAgentType[at][m]["avgMetric"],label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])
				# axins.plot(range(1,len(metricPerAgentType[at][m]["avgMetric"])+1),metricPerAgentType[at][m]["avgMetric"],label=methodNamesPaper[m],c=methodsColors[m],ls=lineStylesPaper[m])

				# Set axis labels and legend
				ax.set_xlabel(r'\textbf{Number of Observations')
				ax.set_ylabel(r'\textbf{'+label+'}')


				if metricPerAgentType[at][m]["avgMetric"][-1] < minLast:
					minLast = metricPerAgentType[at][m]["avgMetric"][-1]

				if metricPerAgentType[at][m]["avgMetric"][-1] > maxLast:
					maxLast = metricPerAgentType[at][m]["avgMetric"][-1]


			# ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
			ax.legend()

			# subregion of the original image
			# axins.set_xlim(startZoom,endZoom)

			minLast = -0.05

			# axins.set_ylim(minLast,maxLast+(maxLast*0.20))

			# axins.set_xticklabels([])
			# axins.set_xticklabels([],minor=True)
			# plt.xticks(visible=False) 

			# mark_inset(ax,axins,loc1=2,loc2=4,fc="none",ec="0.5")

			ax.set_ylim(0.0,maxY)


			plt.savefig(f"{labelSave3}Agent-{at}.png")
			plt.clf()
			plt.close("all")


	with open(f"{args.resultsFolder}{os.sep}metricsAVG{os.sep}{subsetLabel}{os.sep}AVG-{label}-{givenAgentsType}","wb") as outfile:
		pickle.dump(metricPerAgentType,outfile)


def plotObservationsVsMetric(givenProcessedRFsAgentNames,metric):

	label = metric.replace(" ","")
	labelSave = "plotsObservationsVs" + label

	makeFolder3(f"{plotFolder}{labelSave}{os.sep}")
	
	for m in methods:

		if "pSkills" in m or "rhos" in m:
			continue

		folder = f"{args.resultsFolder}plots{os.sep}{labelSave}{os.sep}{m}{os.sep}"
		makeFolder3(folder)

		for agentType in seenAgents:
			tempFolder = f"{folder}{agentType}{os.sep}"
			makeFolder3(tempFolder)


	resultsDict = {}

	for a in givenProcessedRFsAgentNames:

		if "Change" in a:
			agentType = a.split("|")[0].split("Change")[0]
		else:
			agentType = a.split("Agent")[0]
		
		# Load processed info		
		resultsDict[a] = loadProcessedInfo(prdFile,a)


		if a not in infoIndexes:
			# Randomly select sample experiments
			try:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),amount,replace=False)
			# Case when not enough samples (select just 1 then)
			except:
				indexes = np.random.choice(list(range(0,len(resultsDict[a]["infoMetrics"]))),1,replace=False)
			
			infoIndexes[a] = indexes
		else:
			indexes = infoIndexes[a]


		for index in indexes:
			
			info = resultsDict[a]["infoMetrics"][index]


			# For the selected sample exp for the given agent
			for m in methods:

				if "pSkills" in m or "rhos" in m:
					continue

				folder = f"{args.resultsFolder}plots{os.sep}{labelSave}{os.sep}{m}{os.sep}{agentType}{os.sep}"

				estimatedXS = info["methods"][m]["estimatedXS"]

				if "Multi" in m:
					estimatedR = info["methods"][m]["estimatedR"]
	
				toPlot = info["methods"][m][label]


				if type(toPlot[0]) == list:
					loop = 2
					toPlot = np.array(toPlot)
				else:
					loop = 1


				for ii in range(loop):

					# MAKE PLOT
					fig = plt.figure()
					ax = plt.gca()

					# Plot the data points

					if loop == 2:
						plt.plot(range(len(estimatedXS)),toPlot[:,ii])
					else:
						plt.plot(range(len(estimatedXS)),toPlot)

					# Set axis labels and legend
					plt.xlabel('Number of Observations')

					if ii == 0:
						ylabel = f"{metric} - E0/E1"
						tempA = f"{a}-E0-E1"
					else:
						ylabel = f"{metric} - E1/E0"
						tempA = f"{a}-E1-E0"

					plt.ylabel(ylabel)

					if "Multi" in m:
						info2 = f"X: {estimatedXS[-1]} | R: {estimatedR[-1]:.4f}"
					else:
						info2 = f"X: {estimatedXS[-1]}"
					plt.title(f'Estimates: {info2}\n{a}\n{toPlot[-1]}')

					plt.savefig(f"{folder}{tempA}-Index{index}.png",bbox_inches='tight')
					plt.clf()
					plt.close("all")


		del resultsDict[a]


# MSE ACROSS ALL & SELECTED SUBSETS
def computeAndPlotMSEAcrossAllAgentsTypesAllMethods(givenAgents,givenAgentsType,dimensions,methods,resultsFolder,agentTypes,numStates,domain,betas=None,givenBeta=None,makeOtherPlots=False):

	makeFolder(resultsFolder, f"MSE-AcrossAll")
	makeFolder(resultsFolder, f"MSE-AcrossSelected")
	makeFolder(resultsFolder, f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents")
	makeFolder(resultsFolder, f"MSE-AcrossSelected{os.sep}{givenAgentsType}Agents")


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

				if "Multi" in m and "pSkills" not in m and "rhos" not in m:
					mseAcrossAllAgentsPerAgentType[at][m] = np.zeros((numStates,dimensions))
					stdInfoPerAgentTypePerMethod[at][m] = [[] for _ in range(dimensions)] 
					stdPerAgentTypePerMethod[at][m] = [[] for _ in range(dimensions)]
					confidenceIntervals[at][m] = {"low": [0.0]*dimensions, "high": [0.0]*dimensions, "value": [0.0]*dimensions}

					if at in selectedAgentTypes:
						for x in range(len(agentSubsets)):
							mseSelectedAgents[at][names[x]][m] = np.zeros((numStates,dimensions))		

							stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = [[] for _ in range(dimensions)] 
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = [[] for _ in range(dimensions)]
							confidenceIntervalsSelectedAgents[at][names[x]][m] = {"low": [0.0]*dimensions, "high": [0.0]*dimensions, "value": [0.0]*dimensions}

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
	
	saveTo = resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + "info.pickle"
	loaded = False


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
	for a in givenAgents:

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

			# print(m)

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

						if "Multi" in m and "rhos" not in m:
							for ee in range(dimensions):
								mseAcrossAllAgentsPerAgentType[aType][m][s][ee] += resultsDict[a]["plot_y"][m][s][ee]
								
								for x in range(len(agentSubsets)):
									if selected[x]:
										mseSelectedAgents[aType][names[x]][m][s][ee] += resultsDict[a]["plot_y"][m][s][ee]
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
					if "Multi" in m and "rhos" not in m:
						for ee in range(dimensions):
							stdInfoPerAgentTypePerMethod[aType][m][ee].append(resultsDict[a]["plot_y"][m][-1][ee])
							for x in range(len(agentSubsets)):
								if selected[x]:
									stdInfoPerAgentTypePerMethodSelectedAgents[aType][names[x]][m][ee].append(resultsDict[a]["plot_y"][m][-1][ee])

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

	# code.interact("...", local=dict(globals(), **locals()))
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

				if "-pSkills" in m or "-rhos" in m or "Multi" not in m:
					for s in range(numStates):

						# To avoid 0 division error 
						# Since MSE for percent rational is all 0.0 for now
						try:
							mseAcrossAllAgentsPerAgentType[at][m][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
						except:
							mseAcrossAllAgentsPerAgentType[at][m][s] = mseAcrossAllAgentsPerAgentType[at][m][s]


						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:
								mseSelectedAgents[at][names[x]][m][s] /= (mseSelectedAgents[at][names[x]]["numAgents"] * 1.0)

					stdPerAgentTypePerMethod[at][m] = np.std(stdInfoPerAgentTypePerMethod[at][m])
			
					for x in range(len(agentSubsets)):
						if totalSelectedAgents[at][names[x]] != 0:
							stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m] = np.std(stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][m])
				
				else:
					for s in range(numStates):

						for ee in range(dimensions):
							mseAcrossAllAgentsPerAgentType[at][m][s][ee] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
							
							for x in range(len(agentSubsets)):
								if totalSelectedAgents[at][names[x]] != 0:
									mseSelectedAgents[at][names[x]][m][s][ee] /= (mseSelectedAgents[at][names[x]]["numAgents"] * 1.0)


							stdPerAgentTypePerMethod[at][m][ee] = np.std(stdInfoPerAgentTypePerMethod[at][m][ee])
						
							for x in range(len(agentSubsets)):
								if totalSelectedAgents[at][names[x]] != 0:
									stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m][ee] = np.std(stdInfoPerAgentTypePerMethodSelectedAgents[at][names[x]][m][ee])




	

	# To set y axis limits
	maxXskillError = {}
	maxPskillError = {}
	maxRhoError = {}
	minXskillError = {}
	minPskillError = {}
	minRhoError = {}

	maxXskillErrorSelectedAgents = {}
	maxPskillErrorSelectedAgents = {}
	maxRhoErrorSelectedAgents = {}
	minXskillErrorSelectedAgents = {}
	minPskillErrorSelectedAgents = {}
	minRhoErrorSelectedAgents = {}


	for x in range(len(agentSubsets)):

		maxXskillErrorSelectedAgents[names[x]] = {}
		maxPskillErrorSelectedAgents[names[x]] = {}
		maxRhoErrorSelectedAgents[names[x]] = {}
		
		minXskillErrorSelectedAgents[names[x]] = {}
		minPskillErrorSelectedAgents[names[x]] = {}
		minRhoErrorSelectedAgents[names[x]] = {}


	for i in range(len(methodsLists)):

		tempMethods = methodsLists[i]
		label = labels[i]

		# Plots showing both normal JTM & PFE will not set y axis limit for now
		if "Normal" in label and "Multi" in label:
			continue

		maxXskillError[label] = -999999
		maxPskillError[label] = -999999
		maxRhoError[label] = -999999

		minXskillError[label] = 999999
		minPskillError[label] = 999999
		minRhoError[label] = 999999


		for x in range(len(agentSubsets)):

			maxXskillErrorSelectedAgents[names[x]][label] = -999999
			maxPskillErrorSelectedAgents[names[x]][label] = -999999
			maxRhoErrorSelectedAgents[names[x]][label] = -999999

			minXskillErrorSelectedAgents[names[x]][label] = 999999
			minPskillErrorSelectedAgents[names[x]][label] = 999999
			minRhoErrorSelectedAgents[names[x]][label] = 999999


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


					elif "rho" in m:

						tempMax = np.max(mseAcrossAllAgentsPerAgentType[at][m])
						tempMin = np.min(mseAcrossAllAgentsPerAgentType[at][m])
						
						# Update max error seen - will use as y axis limit
						if tempMax > maxRhoError[label]:
							maxRhoError[label] = tempMax 

						if tempMin < minRhoError[label]:
							minRhoError[label] = tempMin 


						for x in range(len(agentSubsets)):
							if totalSelectedAgents[at][names[x]] != 0:

								tempMaxSelected = np.max(mseSelectedAgents[at][names[x]][m])
								tempMinSelected = np.min(mseSelectedAgents[at][names[x]][m])

								if tempMaxSelected > maxRhoErrorSelectedAgents[names[x]][label]:
									maxRhoErrorSelectedAgents[names[x]][label] = tempMaxSelected 

								if tempMinSelected < minRhoErrorSelectedAgents[names[x]][label]:
									minRhoErrorSelectedAgents[names[x]][label] = tempMinSelected 


					# xskill
					else:

						if "Multi" in m:
														
							if type(maxXskillError[label]) == int:
								maxXskillError[label] = [-999999]*dimensions
								minXskillError[label] = [999999]*dimensions

							if type(maxXskillErrorSelectedAgents[names[x]][label]) == int:
								for x in range(len(agentSubsets)):
									maxXskillErrorSelectedAgents[names[x]][label] = [-999999]*dimensions
									minXskillErrorSelectedAgents[names[x]][label] = [999999]*dimensions


							for ee in range(dimensions):

								tempMax = np.max(mseAcrossAllAgentsPerAgentType[at][m][1:][ee])
								tempMin = np.min(mseAcrossAllAgentsPerAgentType[at][m][1:][ee])

								# Update max error seen - will use as y axis limit
								if tempMax > maxXskillError[label][ee]:
									maxXskillError[label][ee] = tempMax 

								if tempMin < minXskillError[label][ee]:
									minXskillError[label][ee] = tempMin 


								for x in range(len(agentSubsets)):
									if totalSelectedAgents[at][names[x]] != 0:
										tempMaxSelected = np.max(mseSelectedAgents[at][names[x]][m][1:][ee])
										tempMinSelected = np.min(mseSelectedAgents[at][names[x]][m][1:][ee])

										if tempMaxSelected > maxXskillErrorSelectedAgents[names[x]][label][ee]:
											maxXskillErrorSelectedAgents[names[x]][label][ee] = tempMaxSelected 

										if tempMinSelected < minXskillErrorSelectedAgents[names[x]][label][ee]:
											minXskillErrorSelectedAgents[names[x]][label][ee] = tempMinSelected 


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

					if "Multi" in m and "pSkills" not in m and "rho" not in m:

						for ee in range(dimensions):
							mu = mseAcrossAllAgentsPerAgentType[at][m][-1][ee]
							sigma = stdPerAgentTypePerMethod[at][m][ee]

							confidenceIntervals[at][m]["low"][ee], confidenceIntervals[at][m]["high"][ee] =\
							stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

							confidenceIntervals[at][m]["value"][ee] = Z * (sigma/np.sqrt(N))


					else:
						mu = mseAcrossAllAgentsPerAgentType[at][m][-1]
						sigma = stdPerAgentTypePerMethod[at][m]

						confidenceIntervals[at][m]["low"], confidenceIntervals[at][m]["high"] =\
						stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

						confidenceIntervals[at][m]["value"] = Z * (sigma/np.sqrt(N))
					
		#####################################################################################################

		xskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + "confidenceIntervals-xSkills.txt", "a")
		pskillsCI = open(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + "confidenceIntervals-pSkills.txt", "a")

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
					tempSeenMethods.append(m)

				else:

					if "Multi" in m and "pSkills" not in m and "rho" not in m:
						for ee in range(dimensions):
							allMseCurrentMethod.append(mseAcrossAllAgentsPerAgentType[at][m][-1][ee])
							tempSeenMethods.append(f"{m}-{ee}")
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


				# Translate to paper's name
				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					mm = methodNamesPaper[m.split("-Beta")[0]]
				else:
					if "Multi" in m and "pSkills" not in m and "rho" not in m:
						mm = methodNamesPaper[m[:-2]] + m[-2:]
					else:
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
						if "Multi" in m and "pSkills" not in m and "rho" not in m:
							infoM = m
							ee = int(m[-1])
							m = m[:-2]

							xskillsCI.write("Agent: " + at + " | Method: " + str(infoM) + \
									" ->  Low: " + f"{confidenceIntervals[at][m]['low'][ee]:.4f}" +\
									" | High: " + f"{confidenceIntervals[at][m]['high'][ee]:.4f}" +\
									" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
									" | Value: " + f"{confidenceIntervals[at][m]['value'][ee]:.4f}" + "\n")

						else:
							infoM = m 

							xskillsCI.write("Agent: " + at + " | Method: " + str(infoM) + \
								" ->  Low: " + f"{confidenceIntervals[at][m]['low']:.4f}" +\
								" | High: " + f"{confidenceIntervals[at][m]['high']:.4f}" +\
								" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
								" | Value: " + f"{confidenceIntervals[at][m]['value']:.4f}" + "\n")

						d_x["Agents"].append(at)
						d_x["Methods"].append(mm)
						d_x["MSE"].append(np.round(orderedMSE[mseO],4))
						d_x["Low"].append(np.round(confidenceIntervals[at][m]["low"],4))
						d_x["High"].append(np.round(confidenceIntervals[at][m]["high"],4))
						d_x["Values"].append(np.round(confidenceIntervals[at][m]["value"],4))


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
							tempFolder = f"MSE-AcrossSelected{os.sep}{givenAgentsType}Agents{os.sep}{sub}{ops[x][0]}{params[x][0]}{os.sep}"
						else:
							tempFolder = f"MSE-AcrossSelected{os.sep}{givenAgentsType}Agents{os.sep}{sub}{ops[x][0]}{params[x][0]}and{ops[x][1]}{params[x][1]}{os.sep}"

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

								if "Multi" in m and "pSkills" not in m and "rho" not in m:

									for ee in range(dimensions):
										mu = mseSelectedAgents[at][names[x]][m][-1][ee]
										sigma = stdPerAgentTypePerMethodSelectedAgents[at][names[x]][m][ee]

										confidenceIntervalsSelectedAgents[at][names[x]][m]["low"][ee], confidenceIntervalsSelectedAgents[at][names[x]][m]["high"][ee] =\
										stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(N))

										confidenceIntervalsSelectedAgents[at][names[x]][m]["value"][ee] = Z * (sigma/np.sqrt(N))
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
								tempSeenMethods.append(m)
							else:

								if "Multi" in m and "pSkills" not in m and "rho" not in m:
									for ee in range(dimensions):
										allMseCurrentMethod.append(mseSelectedAgents[at][names[x]][m][-1][ee])
										tempSeenMethods.append(f"{m}-{ee}")
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


							# Translate to paper's name
							if "BM" in m:
								tempM, beta, tt = getInfoBM(m)
								mm = methodNamesPaper[m.split("-Beta")[0]]
							else:
								if "Multi" in m and "pSkills" not in m and "rho" not in m:
									mm = methodNamesPaper[m[:-2]] + m[-2:]
								else:
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

									if "Multi" in m and "pSkills" not in m and "rho" not in m:
										infoM = m
										ee = int(m[-1])
										m = m[:-2]

										xskillsCI.write("Agent: " + at + " | Method: " + str(infoM) + \
												" ->  Low: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['low'][ee]:.4f}" +\
												" | High: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['high'][ee]:.4f}" +\
												" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
												" | Value: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['value'][ee]:.4f}" + "\n")

									else:
										infoM = m 

										xskillsCI.write("Agent: " + at + " | Method: " + str(infoM) + \
											" ->  Low: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['low']:.4f}" +\
											" | High: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['high']:.4f}" +\
											" ||| Mean: " + f"{orderedMSE[mseO]:.4f}" +\
											" | Value: " + f"{confidenceIntervalsSelectedAgents[at][names[x]][m]['value']:.4f}" + "\n")

									d_x["Agents"].append(at)
									d_x["Methods"].append(infoM)
									d_x["MSE"].append(np.round(orderedMSE[mseO],4))
									d_x["Low"].append(np.round(confidenceIntervalsSelectedAgents[at][names[x]][m]["low"],4))
									d_x["High"].append(np.round(confidenceIntervalsSelectedAgents[at][names[x]][m]["high"],4))
									d_x["Values"].append(np.round(confidenceIntervalsSelectedAgents[at][names[x]][m]["value"],4))


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

	# print("Creating plots....")

	for at in agentTypes:

		for i in range(len(methodsLists)):

			makeFolder(resultsFolder,f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents"+os.sep+labels[i])

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


			# NEED TO UPDATE AS NEEDED
			# if ("ALL" in labels[i] or "OR-BM-Beta" in labels[i] or "JustEES" in labels[i]):

			# 	makePlot = True

			# 	if (domain == "2d" and "Target" not in at) or \
			# 		(domain == "sequentialDarts") or \
			# 		(("Bounded" in at or "Flip" in at) and domain == "1d") or \
			# 		("OR-BM" in labels[i] and "Target" in at):
			# 		makePlot = False

			# 	if makePlot:
			# 		loc = 1

			# 		# x0,y0,w,h.
			# 		# axins = ax1.inset_axes([np.log10(startZoom),0,1000-startZoom,7000])#maxXskillError[labels[i]]])
			# 		axins = zoomed_inset_axes(ax1,zoom=20,loc=loc,borderpad=1)


			maxLast = -9999
			minLast = 9999

			c = 0

			for method in tempMethods:

				# only plotting xskills methods
				if "pSkills" in method or "DomainTargets" in method or "rho" in method:
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


					if "Multi" in method and "rhos" not in method:
						for ee in range(dimensions):

							# QRE-Multi-Particles-2-Resample75%-NoiseDiv500-JT-EES-xSkills
							splitted = method.split("-")
							numP = int(splitted[3])
							resample = int(splitted[4].split("Resample")[1].split("%")[0])
							
							indexN = 5

							if "ResampleNEFF" in method:
								indexN += 1

							noise = int(splitted[indexN].split("NoiseDiv")[1])

							tempLabel = f"P{numP}-R{resample}-N{noise}-D{ee}"

							line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method][:,ee])),mseAcrossAllAgentsPerAgentType[at][method][1:,ee], lw='2.0', label= tempLabel, c = methodsColors[method], ls= lineStylesPaper[method])

					else:
						line, = ax1.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method][1:], lw='2.0', label= label, c = methodsColors[method], ls= lineStylesPaper[method])
					
					# lines.append(line)
					# tempLabels.append(label)

					if makePlot:
						axins.semilogx(range(1,len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method][1:], lw='2.0', label= label, c = methodsColors[method], ls= lineStylesPaper[method])

						if "OR" not in method or \
							(("OR" in method and "Target" in at) and domain != "sequentialDarts") or\
							(domain != "1d" and "Bounded" not in at and "Flip" not in at and domain != "sequentialDarts"):
							
							if mseAcrossAllAgentsPerAgentType[at][method][-1] > maxLast:
								maxLast = mseAcrossAllAgentsPerAgentType[at][method][-1]

							if mseAcrossAllAgentsPerAgentType[at][method][-1] < minLast:
								minLast = mseAcrossAllAgentsPerAgentType[at][method][-1]


			if c != 0:
				ax1.set_xlabel('Number of observations',fontsize=24)
				ax1.set_ylabel('Mean squared error', fontsize=24)
				#ax1.set_xscale('symlog')


				# if len(maxXskillError[labels[i]]) == 1:
					# plt.ylim(top = maxXskillError[labels[i]],bottom = minXskillError[labels[i]])
				
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

				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)


				# Saving legend on separate file
				'''
				legendFig = plt.figure(figsize=(2,2))
				legendFig.legend(lines,tempLabels,loc='center')
				legendFig.tight_layout()
				plt.grid(False)
				plt.axis('off')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
				'''


				'''
				if special:
					copyAx.relim()
					copyAx.autoscale()
					plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-Upd-YLim-"+domain+".png", bbox_inches='tight')
				'''

			plt.clf()
			plt.close("all")


			##################################### FOR RHOS #####################################
			# create plot for each one of the different agents - estimates vs obs

			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)
			lines = []
			tempLabels = []

			c = 0

			for method in tempMethods:

				# only plotting xskills methods
				if "rhos" not in method:
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
				ax1.set_xlabel('Number of observations',fontsize=24)
				ax1.set_ylabel('Mean squared error', fontsize=24)
				# ax1.set_xscale('symlog')
				# plt.ylim(top = maxRhoError[labels[i]],bottom = minRhoError[labels[i]])
				plt.margins(0.05)

				#plt.title('RHO | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')
				'''
				if "ALL" in labels[i]:
					if at == "Target":
						ax1.legend(loc='best')
				else:
				'''
				ax1.legend(loc='best')
				fig.tight_layout()
		
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep + "results-RHO-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)

				# Saving legend on separate file
				'''
				legendFig = plt.figure(figsize=(2,2))
				legendFig.legend(lines,tempLabels,loc='center')
				legendFig.tight_layout()
				plt.grid(False)
				plt.axis('off')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep +  "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
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
				ax1.set_xlabel('Number of observations',fontsize=24)
				ax1.set_ylabel('Mean squared error', fontsize=24)
				# ax1.set_xscale('symlog')

				# Plots showing both normal JTM & PFE will not set y axis limit for now
				if "Normal" not in labels[i] and "Multi" not in labels[i]:
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
			
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight',pad_inches = 0)

				# Saving legend on separate file
				'''
				legendFig = plt.figure(figsize=(2,2))
				legendFig.legend(lines,tempLabels,loc='center')
				legendFig.tight_layout()
				plt.grid(False)
				plt.axis('off')
				plt.savefig(resultsFolder + os.sep + "plots" + os.sep + f"MSE-AcrossAll{os.sep}{givenAgentsType}Agents" + os.sep + labels[i] + os.sep +  "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
				'''

			plt.clf()
			plt.close("all")
			

	#####################################################################################################

	
	# if makeOtherPlots:
		# plotMSEAllBetasSamePlotPerAgentType(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerAgentType,betas)
		# plotLastMSEAllBetasSamePlotPerAgentType(methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,mseAcrossAllAgentsPerAgentType,betas)
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
						tempFolder = f"MSE-AcrossSelected{os.sep}{givenAgentsType}Agents{os.sep}{sub}{ops[x][0]}{params[x][0]}{os.sep}"
					else:
						tempFolder = f"MSE-AcrossSelected{os.sep}{givenAgentsType}Agents{os.sep}{sub}{ops[x][0]}{params[x][0]}and{ops[x][1]}{params[x][1]}{os.sep}"

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

					# NEED TO UPDATE AS NEEDED
					# if ("ALL" in labels[i] or "OR-BM-Beta" in labels[i] or "JustEES" in labels[i]) and domain == "1d":

					# 	makePlot = True

					# 	if (domain != "1d"):
					# 		makePlot = False
					# 	else:
					# 		if "Target" not in at:
					# 			makePlot = False
					# 		if "Bounded" in at and "NearlyRational" in sub:
					# 			makePlot = True


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


							if "Multi" in method and "rhos" not in method:
								for ee in range(dimensions):

									# QRE-Multi-Particles-2-Resample75%-NoiseDiv500-JT-EES-xSkills
									splitted = method.split("-")
									numP = int(splitted[3])
									resample = int(splitted[4].split("Resample")[1].split("%")[0])
									

									indexN = 5

									if "ResampleNEFF" in method:
										indexN += 1

									noise = int(splitted[indexN].split("NoiseDiv")[1])

									tempLabel = f"P{numP}-R{resample}-N{noise}-D{ee}"

									line, = ax1.semilogx(range(1,len(mseSelectedAgents[at][names[x]][method][:,ee])),mseSelectedAgents[at][names[x]][method][1:,ee], lw='2.0', label= tempLabel, c = methodsColors[method], ls= lineStylesPaper[method])

							else:
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
						ax1.set_xlabel('Number of observations',fontsize=24)
						ax1.set_ylabel('Mean squared error', fontsize=24)
						# ax1.set_xscale('symlog')

						# Plots showing both normal JTM & PFE will not set y axis limit for now
						if "Normal" not in labels[i] and "Multi" not in labels[i]:
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

						# Saving legend on separate file
						'''
						legendFig = plt.figure(figsize=(2,2))
						legendFig.legend(lines,tempLabels,loc='center')
						legendFig.tight_layout()
						plt.grid(False)
						plt.axis('off')
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSE-SelectedAgents" + os.sep + names[x] + os.sep + labels[i] + os.sep +  "results-XSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
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
						ax1.set_xlabel('Number of observations',fontsize=24)
						ax1.set_ylabel('Mean squared error', fontsize=24)
						# ax1.set_xscale('symlog')
						# plt.ylim(top = maxPskillErrorSelectedAgents[names[x]][labels[i]],bottom = minPskillErrorSelectedAgents[names[x]][labels[i]])
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

						# Saving legend on separate file
						'''
						legendFig = plt.figure(figsize=(2,2))
						legendFig.legend(lines,tempLabels,loc='center')
						legendFig.tight_layout()
						plt.grid(False)
						plt.axis('off')
						plt.savefig(resultsFolder + os.sep + "plots" + os.sep + "MSE-SelectedAgents" + os.sep + names[x] + os.sep + labels[i] + os.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseSelectedAgents[at][names[x]]["totalNumExps"]))+"-"+domain+"-LEGEND.png", bbox_inches='tight',dpi=200)
						'''

					plt.clf()
					plt.close("all")
			

					##################################### SAVE BEST BETA INFO #####################################

					if labels[i] == "JustBM":

						with open(resultsFolder + os.sep + "plots" + os.sep + tempFolder + labels[i] + os.sep + "bestBetaInfo.txt","a") as outfile:
							print(f"Domain: {args.domain} | Agent: {at}",file=outfile,end="\n\n")
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
	

def computeMSE(processedRFsAgentNames):

	resultsDict = {}

	# Compute MSE
	for a in processedRFsAgentNames:

		# print("\nAgent: ", a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(rdFile,a)


		try:
			for m in methods:
				# print(m)

				if "Multi" in m and "pSkills" not in m and "rhos" not in m:
					for mxi in range(len(resultsDict[a]["plot_y"][m])):
						for ee in range(len(resultsDict[a]["plot_y"][m][mxi])):
							resultsDict[a]["plot_y"][m][mxi][ee] = resultsDict[a]["plot_y"][m][mxi][ee]/float(resultsDict[a]["num_exps"]) #MSError

				else:		
					for mxi in range(len(resultsDict[a]["plot_y"][m])):
						resultsDict[a]["plot_y"][m][mxi] = resultsDict[a]["plot_y"][m][mxi]/float(resultsDict[a]["num_exps"]) #MSError
		except:
			code.interact("...", local=dict(globals(), **locals()))


		# Update info on file
		updateProcessedInfo(prdFile,a,resultsDict)
		
		# code.interact("...", local=dict(globals(), **locals()))


		del resultsDict[a]


def createCopyRFs():

	resultsDict = {}

	for a in processedRFsAgentNames:

		result = loadProcessedInfo(rdFile,a)

		if result == False:
			# print("skipping...")
			# code.interact("...", local=dict(globals(), **locals())) 
			continue

		else: 

			# Load processed info		
			resultsDict[a] = result

			# Update info on file
			updateProcessedInfo(prdFile,a,resultsDict)

			del resultsDict[a]


if __name__ == '__main__':
	

	# ASSUMES RESULTS OF EXPERIMENTS WERE PROCESSED ALREADY


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	parser.add_argument("-randomAgents", dest = "randomAgents", help = "Flag to indicate processing rfs of exps with random agents.", action = 'store_true')
	parser.add_argument("-dynamic", dest = "dynamic", help = "Flag to indicate processing rfs of exps with dynamic agents.", action = 'store_true')
	args = parser.parse_args()


	if args.resultsFolder[-1] != os.sep:
		args.resultsFolder += os.sep


	rdFile = f"{args.resultsFolder}{os.sep}ProcessedResultsFiles{os.sep}resultsDictInfo"
	oiFile = f"{args.resultsFolder}{os.sep}otherInfo" 

	try:
		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)
	except:
		print("The 'otherInfo' file was not found. Need to process results first.")
		exit()


	typeTargetsList = otherInfo["typeTargetsList"]
	numHypsX = otherInfo['numHypsX']
	numHypsP = otherInfo['numHypsP']
	numStates = otherInfo["numObservations"]
	seenAgents = otherInfo["seenAgents"]
	methods = otherInfo["methods"]
	domain = otherInfo["domain"]
	mode = otherInfo["mode"]
	dimensions = otherInfo["dimensions"]

	processedRFs = otherInfo["processedRFs"]
	processedRFsAgentNames = otherInfo["processedRFsAgentNames"]
	seenAgents = otherInfo["seenAgents"]

	numParticles = otherInfo["numParticles"]
	metrics = otherInfo["metrics"]


	'''
	try:
		actualProcessedRFs = os.listdir(f"{args.resultsFolder}ProcessedResultsFiles")
		# print("actualProcessedRFs: ",actualProcessedRFs)
	except:
		print("Folder for processed results files not present.\nNeed to process results files first.")
		exit()


	if len(actualProcessedRFs) == 0:
		print("Need to process results files first.")
		exit()
	'''


	makeFolder2(args.resultsFolder,"ProcessedResultsFilesForPlots")
	
	prdFile = f"{args.resultsFolder}ProcessedResultsFilesForPlots{os.sep}resultsDictInfo"

	plotFolder = f"{args.resultsFolder}plots{os.sep}"

	# agentTypes = ["Target", "Flip", "Tricker","Bounded"]
	agentTypes = seenAgents

	domainModule,delta = getDomainInfo(domain)

	makeFolder3(plotFolder)


	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	# pconfPerXskill = pconf(rng,args.resultsFolder,domain,domainModule,spacesModule,mode,args,wrap)

	####################################



	# createCopyRFs()


	params = {"numParticles":[],"resamples":[],"noises":[]}


	normalJTM = []
	multiJTM = []
	justEES_Normal = []
	justEES_Multi = []
	justEES_Normal_Multi = []


	# To remove multiple occurrences of JEEDS (if any)
	methods = list(set(methods))


	# Remove other noises (keep selected one)
	'''
	tempList = []

	for m in methods:

		if "Multi" in m:
			if "200" in m:
				tempList.append(m)			
		else:
			tempList.append(m)

	methods = tempList
	'''ond


	for m in methods:
		
		if "JT-QRE" in m:
			normalJTM.append(m)

			if "EES" in m:
				justEES_Normal.append(m)
				justEES_Normal_Multi.append(m)


		if "Multi" in m:
			multiJTM.append(m)

			if "EES" in m:
				justEES_Multi.append(m)
				justEES_Normal_Multi.append(m)


			if "Particles" in m:

				# print(m)

				splitted = m.split("-")

				indexN = 5

				if "ResampleNEFF" in m:
					indexN += 1

				p = int(splitted[3])
				r = int(splitted[4].split("Resample")[1].split("%")[0])
				n = int(splitted[indexN].split("NoiseDiv")[1])

				if p not in params["numParticles"]:
					params["numParticles"].append(p)
	
				if r not in params["resamples"]:
					params["resamples"].append(r)
	
				if n not in params["noises"]:
					params["noises"].append(n)



	methodsLists = [normalJTM,multiJTM,justEES_Normal,justEES_Multi,justEES_Normal_Multi]
	# methodsLists = [normalJTM,multiJTM]

	labels = ["NormalJTM","MultiJTM","JustEES-Normal","JustEES-Multi","JustEES-Normal-Multi"]


	global methodNamesPaper
	methodNamesPaper = { 
						"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS",\
						"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS",\

						"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS-MS",\
						"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS-MS",\

						# f"QRE-Multi-Particles-{numParticles}-JT-MAP-xSkills": "JEEDS-Particles-MS",\
						# f"QRE-Multi-Particles-{numParticles}-JT-MAP-pSkills": "JEEDS-Particles-MS",\
						# f"QRE-Multi-Particles-{numParticles}-JT-MAP-rhos": "JEEDS-Particles-MS",\

						# f"QRE-Multi-Particles-{numParticles}-JT-EES-xSkills": "JEEDS-Particles-ES",\
						# f"QRE-Multi-Particles-{numParticles}-JT-EES-pSkills": "JEEDS-Particles-ES",\
						# f"QRE-Multi-Particles-{numParticles}-JT-EES-rhos": "JEEDS-Particles-ES"}
						}



	global methodsColors
	methodsColors = {
					"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills": "tab:green",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills": "tab:green" ,\

					"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills": "tab:orange" ,\
					"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills": "tab:orange" ,\

					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-xSkills": "tab:orange",\
					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-pSkills": "tab:orange",\
					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-rhos": "tab:orange",\

					# f"QRE-Multi-Particles-{numParticles}-JT-EES-xSkills": "tab:green",\
					# f"QRE-Multi-Particles-{numParticles}-JT-EES-pSkills": "tab:green",\
					# f"QRE-Multi-Particles-{numParticles}-JT-EES-rhos": "tab:green"
					}


	global lineStylesPaper
	lineStylesPaper = { 
					"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills": "dashdot",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills": "solid" ,\

					"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-xSkills": "solid" ,\
					"JT-QRE-EES"+"-"+str(numHypsX[0][0])+"-"+str(numHypsP[0])+"-pSkills": "solid" ,\

					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-xSkills": "dashdot",\
					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-pSkills": "dashdot",\
					# f"QRE-Multi-Particles-{numParticles}-JT-MAP-rhos": "dashdot",\

					# f"QRE-Multi-Particles-{numParticles}-JT-EES-xSkills": "solid",\
					# f"QRE-Multi-Particles-{numParticles}-JT-EES-pSkills": "solid",\
					# f"QRE-Multi-Particles-{numParticles}-JT-EES-rhos": "solid"

					}


	colorsList = ["tab:blue","tab:green","tab:red"]
	lineStylesList = ["dashed","dashdot","dotted"]

	# QRE-Multi-Particles-50-Resample90%-NoiseDiv50-JT-MAP-xSkills

	if params["numParticles"] != []:

		for eachNumP in params["numParticles"]:

			# counter = 0

			for resample in params["resamples"]:

				counter = 0

				for noise in params["noises"]:

					for each in ["xSkills","pSkills","rhos"]:
						tempM1 = f"QRE-Multi-Particles-{eachNumP}-Resample{resample}%-NoiseDiv{noise}-JT-MAP-{each}"
						tempM2 = f"QRE-Multi-Particles-{eachNumP}-Resample{resample}%-NoiseDiv{noise}-JT-EES-{each}"

						tempM3 = f"QRE-Multi-Particles-{eachNumP}-Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-MAP-{each}"
						tempM4 = f"QRE-Multi-Particles-{eachNumP}-Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-EES-{each}"


						w = 148/noise
						wp = w/148

						# Add noise to label since tested multiple
						if args.dynamic:
							tempStr = f" (w\% = {wp})"
							# tempStr = f""
						else:
							tempStr = ""


						methodNamesPaper[tempM1] = f"MCSE-MS{tempStr}"
						methodNamesPaper[tempM2] = f"MCSE{tempStr}"
						
						methodNamesPaper[tempM3] = f"MCSE-MS{tempStr}"
						methodNamesPaper[tempM4] = f"MCSE{tempStr}"

						if args.dynamic:
							methodsColors[tempM1] = colorsList[counter]
							methodsColors[tempM2] = colorsList[counter]
							methodsColors[tempM3] = colorsList[counter]
							methodsColors[tempM4] = colorsList[counter]

						else:
							methodsColors[tempM1] = "tab:purple"
							methodsColors[tempM2] = "tab:red"
							methodsColors[tempM3] = "tab:cyan"
							methodsColors[tempM4] = "tab:blue"

						'''
						if counter == 0: 
							lineType = "solid"
						elif counter == 1:
							lineType = "dashed"
						else:
							lineType = "dashdot"
						'''

						if args.dynamic:
							lineStylesPaper[tempM1] = lineStylesList[counter]
							lineStylesPaper[tempM2] = lineStylesList[counter]
							lineStylesPaper[tempM3] = lineStylesList[counter]
							lineStylesPaper[tempM4] = lineStylesList[counter]

						else:

							lineType = "dashed"

							lineStylesPaper[tempM1] = lineType
							lineStylesPaper[tempM2] = lineType
							lineStylesPaper[tempM3] = lineType
							lineStylesPaper[tempM4] = lineType


					counter += 1
				
				# counter += 1


	# blue,orange,red,green,purple,pink,cyan

	# JEEDS-MS - green
	# JEEDS-ES - orange
	# PFE-MS - purple
	# PFE-ES - red
	# PFE-NEFF-MS - cyan
	# PFE-NEFF-ES - blue


	# if params["numParticles"] != []:

	# 	for eachP in params["numParticles"]:
	# 		for eachR in params["resamples"]:
	# 			for eachN in params["noises"]:

	# 				for t in [("MAP","MS"),("EES","ES")]:
	# 					for e in ["xSkills","pSkills","rhos"]:
							
	# 						name1 = f"QRE-Multi-Particles-{eachP}-Resample{eachR}%-NoiseDiv{eachN}-JT-{t[0]}-{e}"
	# 						methodNamesPaper[name1] = f"JEEDS-Particles-{t[1]}"

	# 						name2 = f"QRE-Multi-Particles-{eachP}-Resample{eachR}%-ResampleNEFF-NoiseDiv{eachN}-JT-{t[0]}-{e}"
	# 						methodNamesPaper[name2] = f"JEEDS-Particles-{t[1]}"

	# 						if t[0] == "MAP":
	# 							methodsColors[name1] = "tab:orange"
	# 							methodsColors[name2] = "tab:orange"

	# 							lineStylesPaper[name1] = "dashdot"
	# 							lineStylesPaper[name2] = "dashdot"

	# 						else:
	# 							methodsColors[name1] = "tab:green"
	# 							methodsColors[name2] = "tab:green"

	# 							lineStylesPaper[name1] = "solid"
	# 							lineStylesPaper[name2] = "solid"

	

	# Exps with random params, make plots for sample of agents
	if args.randomAgents:
		# For now, select some at random
		sample = np.random.randint(0,len(processedRFsAgentNames),20)
		givenProcessedRFsAgentNames = [val for i,val in enumerate(processedRFsAgentNames) if i in sample]

	# Exps with given params, make plots for all agents
	else:
		givenProcessedRFsAgentNames = processedRFsAgentNames


	if args.dynamic:
		maxY = 90.0
	else:
		maxY = 18.0


	plt.rcParams.update({'axes.titlesize': 'large'})
	plt.rcParams.update({'axes.labelsize': 'large'})

	plt.rcParams.update({'xtick.labelsize': 'large'})
	plt.rcParams.update({'ytick.labelsize': 'large'})

	plt.rc('legend',fontsize='large')



	'''
	metric = "JeffreysDivergence"
	print("plotDistribution()...")
	startTime = time()
	plotDistribution(givenProcessedRFsAgentNames,metric)
	print("Time: ",time()-startTime)
	print()
	'''


	#code.interact("...", local=dict(globals(), **locals()))


	'''
	print("plotCovErrorElipse()...")
	startTime = time()
	plotCovErrorElipse(givenProcessedRFsAgentNames)
	print("Time: ",time()-startTime)
	print()
	'''


	sampleGoodXskillRange = [8,15]
	sampleBadXskillsRange = [130,145]



	# from symmetric to asymmetic
	# 3.0 - 150.5
	# diffs = [0.0, 0.50, 1.0, 5.0,15.0,25,50,75,100,150]




	if args.randomAgents:
		#threshold1 = [1.0,2.0,5.0,10.0,20.0,50.0]
		threshold1 = [50.0]

		#threshold2 = [0.05,0.10,0.20,0.25]
		threshold2 = [0.20]
	else:
		threshold1 = [0.05]
		threshold2 = [0.05]


	allSymmetric = []
	allAsymmetric = []

	for eachThres1 in threshold1:

		tempS = []
		tempA = []


		for eachThres2 in threshold2:

			subsetLabel = f"ThresholdX-{eachThres1}-ThresholdR-{eachThres2}"


			symmetricAgents = []
			asymmetricAgents = []

			abruptAgents = []
			gradualAgents = []

			abruptGoodToBad = []
			abruptBadToGood = []

			gradualGoodToBad = []
			gradualBadToGood = []


			for eachAgent in processedRFsAgentNames:

				splitted = eachAgent.split("|")
				# print(splitted)

				if "Change" in eachAgent:
					temp = eval(splitted[1].split("X")[1])
					x1 = temp[0]
					x2 = temp[1]
					rho = float(splitted[3].split("R")[1])
				else: 
					x1 = float(splitted[1].split("X")[1])
					x2 = float(splitted[2].split("X")[1])
					rho = float(splitted[3].split("R")[1])


				# Case: Symmetric Agent
				if x1 == x2 and rho == 0.0:
					symmetricAgents.append(eachAgent)
				# Case: Symmetric Agent for Rand Params (considered symmetric if within threshold)
				elif (abs(x1-x2) <= eachThres1) and (abs(rho-0.0) <= eachThres2):
					symmetricAgents.append(eachAgent)
				# Case: Asymmetric Agent
				else:
					asymmetricAgents.append(eachAgent)


				if "Change" in eachAgent:
					if "Abrupt" in eachAgent:
						abruptAgents.append(eachAgent)

						if x1 >= sampleGoodXskillRange[0] and x2 <= sampleGoodXskillRange[1]:
							abruptGoodToBad.append(eachAgent)
						else:
							abruptBadToGood.append(eachAgent)


					elif "Gradual" in eachAgent:
						gradualAgents.append(eachAgent)

						if x1 >= sampleGoodXskillRange[0] and x2 <= sampleGoodXskillRange[1]:
							gradualGoodToBad.append(eachAgent)
						else:
							gradualBadToGood.append(eachAgent)



			tempS.append(symmetricAgents)
			tempA.append(asymmetricAgents)

			agents = [processedRFsAgentNames,symmetricAgents,asymmetricAgents]
			agentTypesLabels = ["All","Symmetric","Asymmetric"]

			if abruptAgents != []:
				agents += [abruptAgents]
				agentTypesLabels += ["Abrupt"]

				agents += [abruptGoodToBad]
				agentTypesLabels += ["AbruptGoodToBad"]

				agents += [abruptBadToGood]
				agentTypesLabels += ["AbruptBadToGood"]


			if gradualAgents != []:
				agents += [gradualAgents]
				agentTypesLabels += ["Gradual"]

				agents += [gradualGoodToBad]
				agentTypesLabels += ["GradualGoodToBad"]

				agents += [gradualBadToGood]
				agentTypesLabels += ["GradualBadToGood"]




			# FOR TESTING
			# agentTypesLabels = ["All"]
			# agents = [processedRFsAgentNames]


			# code.interact("...", local=dict(globals(), **locals()))	



			#'''
			for eachMetric in metrics:

				# print(f"plotObservationsVsMetric() - {eachMetric}...")
				# startTime = time()
				# plotObservationsVsMetric(givenProcessedRFsAgentNames,eachMetric)
				# print("Time: ",time()-startTime)
				# print()

				for eachAgent in range(len(agents)):

					if agents[eachAgent] != []:
						print(f"plotObservationsVsMetricPerAgentType() - {eachMetric} - {agentTypesLabels[eachAgent]} Agents ...")
						startTime = time()
						plotObservationsVsMetricPerAgentType(eachMetric,maxY,agents[eachAgent],agentTypesLabels[eachAgent],subsetLabel)
						print("Time: ",time()-startTime)
						print()
			#'''



			'''
			print("computeMSE()...")
			startTime = time()
			computeMSE(processedRFsAgentNames)
			print("Time: ",time()-startTime)
			print()
			'''


			'''
			for each in range(len(agents)):

				if agents[each] != []:
					print(f"computeAndPlotMSEAcrossAllAgentsTypesAllMethods() - {agentTypesLabels[each]}...")
					startTime = time()
					computeAndPlotMSEAcrossAllAgentsTypesAllMethods(agents[each],agentTypesLabels[each],dimensions,methods,args.resultsFolder,agentTypes,numStates,domain)
					print("Time: ",time()-startTime)
					print()
			'''


			# Computes true percent rationality
			# plotPercentRandMaxRewardObtainedPerXskillPerAgentType(processedRFsAgentNames)

			# Computes MSE percent
			#computeMSEPercentPskillsMethods(processedRFsAgentNames)

			
			'''
			metric = "JeffreysDivergence"
			for eachAgent in range(len(agents)):

				if agents[eachAgent] != []:
					plotObservationsVsMetricPerAgentType_xbyp(metric,agents[eachAgent],agentTypesLabels[eachAgent])
					plotObservationsVsMetricPerAgentType_pbyx(agents[eachAgent],agentTypesLabels[eachAgent])
			'''


			# metric = "JeffreysDivergence"
			# plotObservationsVsMetric_AbruptAgents_Centered(metric,maxY,abruptAgents,"AbruptAgents",subsetLabel)


			# code.interact("...", local=dict(globals(), **locals()))	


		allSymmetric.append(tempS)
		allAsymmetric.append(tempA)


	with open(f"{args.resultsFolder}{os.sep}infoAgents.txt","w") as outfile:
		for ii in range(len(threshold1)):
			for jj in range(len(threshold2)):
				print(f"ThresX: {threshold1[ii]} | ThresXR: {threshold2[jj]} | Symmetric: {len(allSymmetric[ii][jj])} | Asymmetric: {len(allAsymmetric[ii][jj])}",file=outfile)











