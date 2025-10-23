import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)

# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
# To be used later for finding and properly loading the domains 
# Will look something like: "/home/archibald/skill-estimation/Environments/"
mainFolderName = scriptPath.split("Processing")[0]	 + "Environments" + os.path.sep
spacesModule = SourceFileLoader("spaces",mainFolderName.split("Environments"+os.path.sep)[0] + "setupSpaces.py").load_module()

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

import pickle, json
import argparse

from utilsDarts import *

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors

################################### UTILS ###################################

def interpolate_r_skill(r, rs, xs):   
	if r <= rs[-1]:
		return xs[-1]

	if r >= rs[0]:
		return xs[0]

	for i in range(len(rs)-1):
		if rs[i] >= r and r >= rs[i+1]:
			return xs[i] + (r-rs[i])*(xs[i+1] - xs[i])/(rs[i+1] - rs[i])
		
	return xs[-1]

def sortTwoLists(aList, bList):
	# sorts aList and find parallel on bList to "sort" - put in the same/corresponding position
	# for example: lambas & estimates

	originalList = deepcopy(aList)

	aList.sort()
	
	tempList = []

	for each in range(len(aList)):

		# Find position of element sortedList[each] in original list
		i = originalList.index(aList[each])

		# Get element at position i from bList and store
		tempList.append(bList[i])

	# return new "sorted" lists
	return aList, tempList

def getParamsFromAgentName(a):  

	if "Bounded" in a:
		string = a.split("-L")
		aType = "Bounded"

	elif "Flip" in a:
		string = a.split("-P")
		aType = "Flip"

	elif "Tricker" in a:
		string = a.split("-Eps")
		aType = "Tricker"

	elif "TargetBelief" in a:
		string = a.split("-B")
		aType = "TargetBelief"

	elif "Target" in a:
		string = [a]
		aType = "Target"

	elif "Random" in a:
		string = a.split("-X")
		aType = "Random"

	#print("aType: ", aType)

	# Find pskill
	if aType == "Target":
		p = 100.0
	elif aType == "Random": 
		string2 = string[1].split("-N")[1]
		string3 = string2.split("-K")
		p = string3[0] + "/" + string3[1] 
	else:
		p = round(float(string[1]),4)

		# verify if number in scientific notation
		if "e" in str(p):
			# will truncate rest of decimal places
			# Just keep first 3
			# Splits on "e" to stay with first part
			# Convert back to number
			p =  float('{:0.3e}'.format(p).split("e")[0])



	#print "a: ", a
	#print "p: ", p

	# Find xskill
	string2 = string[0].split("-X")
	x = round(float(string2[1]),4)
	#print "x: ", x

	# Return info
	return aType, x, p

def getBucket(bucketsX,minMaxX,xParam):

	# Find proper bucket for current x
	for b in range(len(bucketsX)):
		if xParam <= bucketsX[b]:
			break

	# Get actual bucket
	bucket1 = bucketsX[b]


	otherBucket = None
	bucket2 = None

	# First bucket
	if b == 0:
		# use left edge/extreme - i.e. 0
		otherBucket = minMaxX[0]
	# If last bucket
	elif b == len(bucketsX)-1:
		# use right edge/extreme - i.e. 5/100 depending on the domain
		otherBucket = minMaxX[1]
	# Somewhere in the middle - consider next bucket
	else:
		bucket2 = bucket1
		bucket1 = bucketsX[b-1]

	return bucket1, bucket2

###################################################################################


##################### FOR RATIONALITY PARAMETER - ALL AGENTS ######################

def plotPercentTimesDistributionPskillBuckets(domain, resultsDict, agentTypes, methods, resultsFolder, numStates):

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




	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


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
				

	# compute percents
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

				
				fig.tight_layout(pad = 2.0)

				plt.margins(0.05)
				fig.suptitle('Agent: ' + at + " | Method: " + m, y=1.02)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "plotPercentTimesDistribution-PskillBuckets" + os.path.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()

	#code.interact("here", local = locals())

	###############################################################################################################3

def plotPercentTimesDistributionXskillBuckets(domain, resultsDict, agentTypes, methods, resultsFolder, numStates):

	makeFolder(resultsFolder, "plotPercentTimesDistribution-XskillBuckets")


	# Buckets (very bad, bad, regular, good, very good) - in terms of percents
	buckets = [0.45, 0.60, 0.75, 0.90, 1.0]


	if domain == "1d":
		xskillBuckets = [1.0, 2.0, 3.0, 4.0, 5.0]
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



	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


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

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "plotPercentTimesDistribution-XskillBuckets" + os.path.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()

	#code.interact("here", local = locals())

def plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndPskillBuckets(domain, resultsDict, agentTypes, methods, resultsFolder, numStates):

	makeFolder(resultsFolder, "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets")

	if domain == "1d":
		# Buckets (very bad, bad, regular, good, very good)--- xskill
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0] 
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [25,50,75,100,150]


	#pskillBuckets = [45, 60, 75, 90, 100] # NOT IN % TERMS


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

				for xb in pskillBuckets:
					percentOfTimeRightBucketPerAgent[at][m][str(xb)] = {}


					for b in buckets:
						percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(b)] = {"totalNumExps": 0.0, "timesInBucket": 0.0, "avgTrueX": 0.0, "avgEstimatedX": 0.0, "maxTrueX": -999.9, "minTrueX": 999.0, "maxEstimatedX": -999.9, "minEstimatedX": 999.0}




	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

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




	# compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():

		# RESET FILE
		with open(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.path.sep  + "BucketBounds-Agent-"+at+".txt", "w") as aFile:
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
					with open(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.path.sep  + "BucketBounds-Agent-"+at+".txt", "a") as aFile:

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

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.path.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()


				####### scatter plot of avgTrueP and avgEstimatedP
				fig = plt.figure()
				ax = plt.subplot(111)

				s = ax.scatter(avgTrueX, avgEstimatedX)
				plt.xlabel("AVG True Xskill")
				plt.ylabel("AVG Estimated Xskill")
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-PskillBuckets" + os.path.sep  + "scatterPlot-avgTrueVSEstimatedP-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()
  

			###############################################################################################################3

def plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndXskillBuckets(domain, resultsDict, agentTypes, methods, resultsFolder, numStates):

	makeFolder(resultsFolder, "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets")


	# Buckets (very bad, bad, regular, good, very good) - in terms of percents
	buckets = [0.45, 0.60, 0.75, 0.90, 1.0]


	if domain == "1d":
		xskillBuckets = [1.0, 2.0, 3.0, 4.0, 5.0]
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
						percentOfTimeRightBucketPerAgent[at][m][str(xb)][str(b)] = {"totalNumExps": 0.0, "timesInBucket": 0.0, "avgTrueP": 0.0, "avgEstimatedP": 0.0, "maxTrueP": -999.9, "minTrueP": 999.0, "maxEstimatedP": -999.9, "minEstimatedP": 999.0}




	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


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
					print("estimatedP: ", estimatedP)
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




	# compute percents
	for at in percentOfTimeRightBucketPerAgent.keys():

		# RESET FILE
		with open(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.path.sep  + "BucketBounds-Agent-"+at+".txt", "w") as aFile:
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
					with open(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.path.sep  + "BucketBounds-Agent-"+at+".txt", "a") as aFile:

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


				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.path.sep  + "results-Percent-TimesRightBucket-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()


				####### scatter plot of avgTrueP and avgEstimatedP
				fig = plt.figure()
				ax = plt.subplot(111)

				s = ax.scatter(avgTrueP, avgEstimatedP)
				plt.xlabel("AVG True Percent")
				plt.ylabel("AVG Estimated Percent")
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTimesOnBucketObtainedPerAgentTypeAndMethod-XskillBuckets" + os.path.sep  + "scatterPlot-avgTrueVSEstimatedP-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.clf()
				plt.close()
  

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



def plotRationalityParamsVSRewardsPerXSkill(resultsDict, numHypsX, numHypsP, resultsFolder, agentType):


	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	for a in resultsDict.keys():

		# only consider specified agent type
		if agentType not in a:
			continue
		else:
			# print a

			aType, x, p = getParamsFromAgentName(a)


			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = { "xSkills": {} }

			# if we haven't seen this xskill for this lambda yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"true": 0.0, "observed": 0.0}

			# counter will be 1 for all since only one agent sample
			# update ifo
			params[p]["xSkills"][x]["observed"] = resultsDict[a]["mean_observed_reward"]
			params[p]["xSkills"][x]["true"] = resultsDict[a]["mean_true_reward"]
			# params[p]["xSkills"][x]["counter"] += 1

	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-rationalityParamsVSRewardsPerXSkill")


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different params - ok for now
	# won't work if rand parameters for agent - most likey only 1 XSkill for a given param
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	#code.interact(local=locals())

	for eachXSkill in sortedXSkills:
		
		fig = plt.figure()
		ax = plt.subplot(111)
		
		for eachParam in sortedParams:
 
			try:
				plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["observed"], color = "blue", marker = ".")
				plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["true"], color = "red", marker = "*")
			except:
				continue

		plt.xlabel(r'\textbf{'+xString+'}')
		plt.ylabel(r'\textbf{Mean Rewards}')
		plt.margins(0.05)
		plt.title(xString + " vs. Rewards | " + agentType + " Agent | xSkill: " + str(eachXSkill))

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		elements = [Line2D([0],[0], color = "blue", marker = ".", label = "mean OR"),
					Line2D([0],[0], color = "red", marker = "*", label = "mean TR")]
  
		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-rationalityParamsVSRewardsPerXSkill" + os.path.sep + "results-" + xString + "VSRewards-xSkill" + str(eachXSkill) + ".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()

def plotXSkillsVSRewardsPerRationalityParams(resultsDict, numHypsX, numHypsP, resultsFolder, agentType):

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	for a in resultsDict.keys():

		# only consider specified type of agent
		if agentType not in a:
			continue
		else:
			# print a

			aType, x, p = getParamsFromAgentName(a)


			# if we haven't seen this param, init it
			if p not in params:
				params[p] = { "xSkills": {} }

			# if we haven't seen this xskill for this lambda yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"true": 0.0, "observed": 0.0}

			# counter will be 1 for all since only one agent sample
			# update ifo
			params[p]["xSkills"][x]["observed"] = resultsDict[a]["mean_observed_reward"]
			params[p]["xSkills"][x]["true"] = resultsDict[a]["mean_true_reward"]
			# lambdas[l]["xSkills"][x]["counter"] += 1

	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-xSkillsVSRewardsPerRationalityParams")


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different lambdas - ok for now
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	for eachParam in sortedParams:

		fig = plt.figure()
		ax = plt.subplot(111)

		for eachXSkill in sortedXSkills:
			try:
				plt.plot(eachXSkill,params[eachParam]["xSkills"][eachXSkill]["observed"], color = "blue", marker = ".")
				plt.plot(eachXSkill,params[eachParam]["xSkills"][eachXSkill]["true"], color = "red", marker = "*")
			except:
				continue

		plt.xlabel(r'\textbf{xSkills}')
		plt.ylabel(r'\textbf{Mean Rewards}')
		plt.margins(0.05)
		plt.title("xSkills vs. Rewards | "+ agentType + " Agent | " + xString + ": " + str(eachParam))

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		elements = [Line2D([0],[0], color = "blue", marker = ".", label = "mean OR"),
					Line2D([0],[0], color = "red", marker = "*", label = "mean TR")]
  
		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType +  "-xSkillsVSRewardsPerRationalityParams" + os.path.sep + "results-xSkillsVSRewards-" + xString + str(eachParam)+".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()

def plotRationalityParamsVsEVIntended(resultsDict, numHypsX, numHypsP, resultsFolder, agentType):

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	for a in resultsDict.keys():
		# only consider specified agent type
		if agentType not in a:
			continue
		else:
			#print a

			aType, x, p = getParamsFromAgentName(a)

			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = {"true": 0.0, "intended": 0.0, "counter": 0.0}

			# update ifo
			# params[p]["intended"] += resultsDict[a]["mean_value_intendedAction"]
			params[p]["true"] += resultsDict[a]["mean_true_reward"]
			params[p]["counter"] += 1

	# after seing them all compute mean of lambas
	for p in params:
		params[p]["intended"] /= params[p]["counter"]
		params[p]["true"] /= params[p]["counter"]


	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-rationalityParamsVsEVIntended")


	fig = plt.figure()
	ax = plt.subplot(111)

	sortedParams = sorted(params.keys())

	for eachParam in sortedParams:
		try:
			plt.plot(eachParam,params[eachParam]["intended"], color = "blue", marker = ".")
			plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*", alpha = 0.3)
		except:
			continue

	# plt.xlabel(r'\textbf{'+xStr+'}')
	plt.ylabel(r'\textbf{Mean EV intended action}')
	plt.margins(0.05)
	plt.title("RationalityParam vs. EV Intended | " + agentType + " Agent")

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	elements = [Line2D([0],[0], color = "blue", marker = ".", label = "mean EV"),
				Line2D([0],[0], color = "red", marker = "*", label = "mean TR")]

	# Put a legend to the right of the current axis
	ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

	plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-rationalityParamsVsEVIntended" + os.path.sep + "results-" + xString + "VsEVIntended.png", bbox_inches = 'tight')
	plt.clf()
	plt.close()

def plotRationalityParamsVSRewards(resultsDict, numHypsX, numHypsP, resultsFolder, agentType):

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	for a in resultsDict.keys():
		# only consider specified agent type
		if agentType not in a:
			continue
		else:
			#print a

			aType, x, p = getParamsFromAgentName(a)

			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = {"true": 0.0, "observed": 0.0, "counter": 0.0}

			# update ifo
			params[p]["observed"] += resultsDict[a]["mean_observed_reward"]
			params[p]["true"] += resultsDict[a]["mean_true_reward"]
			params[p]["counter"] += 1

	# after seing them all compute mean of lambas
	for p in params:
		params[p]["observed"] /= params[p]["counter"]
		params[p]["true"] /= params[p]["counter"]


	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-rationalityParamsVSRewards")


	fig = plt.figure()
	ax = plt.subplot(111)

	sortedParams = sorted(params.keys())

	for eachParam in sortedParams:
		try:
			plt.plot(eachParam,params[eachParam]["observed"], color = "blue", marker = ".")
			plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*")
		except:
			continue

	# plt.xlabel(r'\textbf{'+xStr+'}')
	plt.ylabel(r'\textbf{Mean Rewards}')
	plt.margins(0.05)
	plt.title("RationalityParam vs. Rewards | " + agentType + " Agent")

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	elements = [Line2D([0],[0], color = "blue", marker = ".", label = "mean OR"),
				Line2D([0],[0], color = "red", marker = "*", label = "mean TR")]

	# Put a legend to the right of the current axis
	ax.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

	plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-rationalityParamsVSRewards" + os.path.sep + "results-" + xString + "VSRewards.png", bbox_inches = 'tight')
	plt.clf()
	plt.close()


def plotRationalityParamsVsSkillEstimatePerMethod(resultsDict, numHypsX, numHypsP, methods, resultsFolder, agentType):

	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-rationalityParamsVSxSkillEstimatePerMethod")
	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-rationalityParamsVSpSkillEstimatePerMethod")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

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
				for eachAgent in resultsDict.keys():

					# only look at specified type of agent
					if agentType in eachAgent:

						aType, x, p = getParamsFromAgentName(a)

						if x not in params.keys():
							params[x] = []
							estimates[x] = []

						params[x].append(p)
						estimates[x].append(resultsDict[eachAgent]["estimates"][method][-1])

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

					plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-rationalityParamsVSxSkillEstimatePerMethod" + os.path.sep + "results-"+xString+"VSEstimate-Method-" + method + "-XSkill-" + str(each) + ".png", bbox_inches = 'tight')
					plt.clf()
					plt.close()
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
				for eachAgent in resultsDict.keys():

					# only look at specified type of agent
					if agentType in eachAgent:

						aType, x, p = getParamsFromAgentName(a)

						if x not in params.keys():
							params[x] = []
							estimates[x] = []

						params[x].append(p)
						estimates[x].append(resultsDict[eachAgent]["estimates"][method][-1])

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

					plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-rationalityParamsVSpSkillEstimatePerMethod" + os.path.sep + "results-LambdaVSEstimate-Method-" + method + "-XSkill-" + str(each) + ".png", bbox_inches = 'tight')
					plt.clf()
					plt.close()

	#######################################################################################


def plotMSEAllRationalityParamsPerMethods(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp, agentType):

	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-mseAllRationalityParamsPerXSkillMethodsBoundedAgent")
	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-mseAllRationalityParamsPerPSkillMethodsBoundedAgent")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	numMethods = len(methods)

	for a in resultsDict.keys():

		# only consider specified type of agent
		if agentType not in a:
			continue
		else:
			# print a

			aType, x, p = getParamsFromAgentName(a)

			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = { "xSkills": {}, "numExps": 0 }

			# if we haven't seen this xskill for this lambda yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"mseMethods": {}}

			params[p]["numExps"] = resultsDict[a]["num_exps"]

			# update ifo
			for m in methods:
				params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][m]


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different params - ok for now
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()['color']


	##################################### FOR XSKILLS #####################################
	c = 0
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

				plt.savefig(resultsFolder +  os.path.sep + agentType + os.path.sep + agentType + "-mseAllRationalityParamsPerXSkillMethodsBoundedAgent" + os.path.sep + "results-mseVs"+xString+"-Method-"+str(methods[m])+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
			
			c += 1
	######################################################################################


	##################################### FOR PSKILLS #####################################
	c = 0
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
 
				plt.savefig(resultsFolder +os.path.sep + agentType + os.path.sep + agentType + "-mseAllRationalityParamsPerPSkillMethodsBoundedAgent" + os.path.sep + "results-mseVs"+xString+"-Method-"+str(methods[m])+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
			
			c += 1
	######################################################################################

def plotMSEAllRationalityParamsAllMethods(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp, agentType):

	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-mseAllRationalityParamsAllXSkillMethods")
	makeFolder2(resultsFolder + os.path.sep + agentType, agentType + "-mseAllRationalityParamsAllPSkillMethods")

	if agentType == "Bounded":
		xString = "Lambdas"
	elif agentType == "Flip":
		xString = "Ps"
	elif agentType == "Tricker":
		xString = "Eps"

	params = {}

	numMethods = len(methods)

	for a in resultsDict.keys():

		# only consider specified type of agent
		if agentType not in a:
			continue
		else:
			# print a

			aType, x, p = getParamsFromAgentName(a)

			# if we haven't seen this lambda, init it
			if p not in params:
				params[p] = { "xSkills": {}, "numExps": 0 }

			# if we haven't seen this xskill for this param yet, init it
			if x not in params[p]["xSkills"].keys():
				params[p]["xSkills"][x] = {"mseMethods": {}}

			params[p]["numExps"] = resultsDict[a]["num_exps"]

			# update ifo
			for m in methods:
				params[p]["xSkills"][x]["mseMethods"][m] = resultsDict[a]["plot_y"][m]


	sortedParams = sorted(params.keys())

	# assuming using the same set of xskills for all the different params - ok for now
	sortedXSkills = sorted(params[sortedParams[0]]["xSkills"].keys())

	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()['color']

	# code.interact(local = locals())

	##################################### FOR XSKILLS #####################################
	for eachXSkill in sortedXSkills:
		
		fig = plt.figure()
		ax = plt.subplot(111)
		
		for eachParam in sortedParams:
			c = 0

			for m in range(len(methods)):
				
				if "pSkills" not in methods[m]:
					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o")
						c += 1
					except:
						continue
				else:
					continue

		plt.xlabel(r'\textbf{' + xString + '}', fontsize=18)
		plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)
		plt.margins(0.05)
		plt.title("xSkill: " + str(eachXSkill) + " | Experiments: " + str(params[p]["numExps"]))

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		
		elements = []
		c = 0

		for m in range(len(methods)):
			if "pSkills" not in methods[m]:
				elements.append(Line2D([0],[0], color = colors[c], marker = "o", label = methods[m]))
				c += 1

		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc = 'center left', bbox_to_anchor = (1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-mseAllRationalityParamsAllXSkillMethods" + os.path.sep + "results-mseVs"+xString+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()
	#######################################################################################

	##################################### FOR PSKILLS #####################################
	for eachXSkill in sortedXSkills:
		
		fig = plt.figure()
		ax = plt.subplot(111)
		
		for eachParam in sortedParams:
			c = 0

			for m in range(len(methods)):
				
				if "pSkills" in methods[m]:
					try:
						plt.plot(eachParam,params[eachParam]["xSkills"][eachXSkill]["mseMethods"][methods[m]][-1], color = colors[c], marker = "o")
						c += 1
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
		c = 0

		for m in range(len(methods)):
			if "pSkills" in methods[m]:
				elements.append(Line2D([0],[0], color = colors[c], marker = "o", label = methods[m]))
				c += 1

		# Put a legend to the right of the current axis
		ax.legend(handles = elements, loc = 'center left', bbox_to_anchor = (1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + agentType + os.path.sep + agentType + "-mseAllRationalityParamsAllPSkillMethods" + os.path.sep + "results-mseVs"+xString+"-xSkill"+str(eachXSkill)+".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()
	#######################################################################################

###################################################################################


################################### FOR REWARDS ###################################

def contourPlotPercentRandMaxRewardObtainedPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain):
	# will display info per xskill & per pskill

	makeFolder(resultsFolder, "contourPercentRandMaxRewardObtainedPerAgentType")

	percentOfRewardPerAgent = {}

	for at in seenAgents:
		if "Random" not in at:
			percentOfRewardPerAgent[at] = {"ps":{}}


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		if p not in percentOfRewardPerAgent[aType]["ps"].keys():
			percentOfRewardPerAgent[aType]["ps"][p] = {"xs": {}}

		if x not in percentOfRewardPerAgent[aType]["ps"][p]["xs"].keys():
			percentOfRewardPerAgent[aType]["ps"][p]["xs"][x] = {"percent": 0.0, "numExps": 0.0, "allPercents": []}


		# minR = resultsDict[a]["mean_random_reward"]
		minR = resultsDict[a]["mean_random_reward_mean_vs"]
		
		#expectedR = resultsDict[a]["mean_value_intendedAction"]
		maxR = resultsDict[a]["mean_true_reward"]

		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		percentOfReward = (((expectedR - minR)/(maxR - minR)) * 100.0)

		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		# percentOfTrueReward = (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"]) * 100.0
		
		percentOfRewardPerAgent[aType]["ps"][p]["xs"][x]["percent"] += percentOfReward
		percentOfRewardPerAgent[aType]["ps"][p]["xs"][x]["allPercents"].append(percentOfReward)

		percentOfRewardPerAgent[aType]["ps"][p]["xs"][x]["numExps"] += 1.0


	# normalize % of TR
	for at in seenAgents:
		if "Random" not in at:
			for p in percentOfRewardPerAgent[at]["ps"].keys():
				for x in percentOfRewardPerAgent[at]["ps"][p]["xs"].keys():
					percentOfRewardPerAgent[at]["ps"][p]["xs"][x]["percent"] /= percentOfRewardPerAgent[at]["ps"][p]["xs"][x]["numExps"]


	if domain == "1d":
		# Create different execution skill levels 
		xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
	elif domain == "2d" or domain == "sequentialDarts":
		# Create different execution skill levels 
		xSkills = np.linspace(2.5, 100.5, num = 100) # (start, stop, num samples)
	

	# create contour plot for each agent type
	for at in seenAgents:
		if "Random" not in at:

			xs = []
			ps = []
			percents = []

			# get info out of dictionary
			for p in percentOfRewardPerAgent[at]["ps"].keys():
				for x in percentOfRewardPerAgent[at]["ps"][p]["xs"].keys():
					ps.append(p)
					xs.append(x)
					percents.append(percentOfRewardPerAgent[at]["ps"][p]["xs"][x]["percent"])


			# scatter plot for now

			fig = plt.figure()

			# rows, cols, pos
			ax = fig.add_subplot(2, 1, 1)

			s = ax.scatter(xs, ps, c = percents)
			

			cbar = fig.colorbar(s, ax = ax)
			cbar.set_label("Percent Rand/Max", labelpad=+1)

			ax.set_xlabel("xSkills")
			ax.set_ylabel("pSkills")
			
			plt.margins(0.05)

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourPercentRandMaxRewardObtainedPerAgentType" + os.path.sep + "scatterPlot-percentRandMax-domain-" + domain + "-agent" + at + ".png", bbox_inches = 'tight')
			plt.clf()
			plt.close()

			# do not create the contour plot for target bc causes error since only 1 dim
			if "Target" in at:
				continue


			if "Bounded" in at:
				# Create different probabilities for an agent being rational
				probsRational = np.linspace(0.0, 100.0, num = 100)
			else:
				# Create different probabilities for an agent being rational
				probsRational = np.linspace(0.0, 1.0, num = 100)
			

			# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
			gx, gy = np.meshgrid(xSkills,probsRational)


			POINTS = []
			VALUES = []

			# find points and values

			for p in percentOfRewardPerAgent[at]["ps"].keys():
				for x in percentOfRewardPerAgent[at]["ps"][p]["xs"].keys():
					POINTS.append([x,p])
					VALUES.append(percentOfRewardPerAgent[at]["ps"][p]["xs"][x]["percent"]) # percent


			POINTS = np.asarray(POINTS)
			VALUES = np.asarray(VALUES)

			if "Bounded" in at:
				# Z = griddata(POINTS, VALUES, (gxl, gyl), method = 'linear')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
				Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
			else:
				Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
			# Z = griddata(POINTS, VALUES, (gx, gy), method = 'cubic')

			# remove inf's -> causes surface plot to be all of the same color 
			Z[Z == np.inf] = np.nan

			fig = plt.figure()
			ax = plt.subplot(111)

			if "Bounded" in at:
				# cs = plt.contourf(gxl, gyl, Z)
				# plt.scatter(gxl,gyl)
				# plt.yscale("log")
				cs = plt.contourf(gx, gy, Z)
			else:
				cs = plt.contourf(gx, gy, Z)
				# plt.scatter(gx,gy)

			plt.xlabel("XSkills")
			plt.ylabel("PSkills")
			plt.title("Percent of Rand/Max| Domain: " + domain + " | Agent: " + at)
			fig.colorbar(cs)

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourPercentRandMaxRewardObtainedPerAgentType" +os.path.sep + "contourPercentRandMax-Domain-"+domain+"-Agent-"+at+".png", bbox_inches='tight')
			plt.clf()
			plt.close()

def contourPlotPercentTrueRewardObtainedPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain):
	# will display info per xskill & per pskill

	makeFolder(resultsFolder, "contourPercentTrueRewardObtainedPerAgentType")

	percentOfTrueRewardPerAgent = {}

	for at in seenAgents:
		if "Target" not in at and "Random" not in at:
			percentOfTrueRewardPerAgent[at] = {"ps":{}}


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		string2 = string[0].split("-X")
		x = round(float(string2[1]),4)


		if p not in percentOfTrueRewardPerAgent[aType]["ps"].keys():
			percentOfTrueRewardPerAgent[aType]["ps"][p] = {"xs": {}}

		if x not in percentOfTrueRewardPerAgent[aType]["ps"][p]["xs"].keys():
			percentOfTrueRewardPerAgent[aType]["ps"][p]["xs"][x] = {"percent": 0.0, "numExps": 0.0, "allPercents": []}


		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		percentOfTrueReward = (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"]) * 100.0
		
		percentOfTrueRewardPerAgent[aType]["ps"][p]["xs"][x]["percent"] += percentOfTrueReward
		percentOfTrueRewardPerAgent[aType]["ps"][p]["xs"][x]["allPercents"].append(percentOfTrueReward)

		percentOfTrueRewardPerAgent[aType]["ps"][p]["xs"][x]["numExps"] += 1.0


	# normalize % of TR
	for at in seenAgents:
		if "Target" not in at and "Random" not in at:
			for p in percentOfTrueRewardPerAgent[at]["ps"].keys():
				for x in percentOfTrueRewardPerAgent[at]["ps"][p]["xs"].keys():
					percentOfTrueRewardPerAgent[at]["ps"][p]["xs"][x]["percent"] /= percentOfTrueRewardPerAgent[at]["ps"][p]["xs"][x]["numExps"]


	if domain == "1d":
		# Create different execution skill levels 
		xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
	elif domain == "2d" or domain == "sequentialDarts":
		# Create different execution skill levels 
		xSkills = np.linspace(2.5, 100.5, num = 100) # (start, stop, num samples)
	

	# create contour plot for each agent type
	for at in seenAgents:
		if "Target" not in at and "Random" not in at:

			xs = []
			ps = []
			percents = []

			# get info out of dictionary
			for p in percentOfTrueRewardPerAgent[at]["ps"].keys():
				for x in percentOfTrueRewardPerAgent[at]["ps"][p]["xs"].keys():
					ps.append(p)
					xs.append(x)
					percents.append(percentOfTrueRewardPerAgent[at]["ps"][p]["xs"][x]["percent"])


			# scatter plot for now

			fig = plt.figure()

			# rows, cols, pos
			ax = fig.add_subplot(2, 1, 1)

			s = ax.scatter(xs, ps, c = percents)
			

			cbar = fig.colorbar(s, ax = ax)
			cbar.set_label("Percent TR", labelpad=+1)

			ax.set_xlabel("xSkills")
			ax.set_ylabel("pSkills")
			
			plt.margins(0.05)

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourPercentTrueRewardObtainedPerAgentType" + os.path.sep + "scatterPlot-percentTR-domain-" + domain + "-agent" + at + ".png", bbox_inches = 'tight')
			plt.clf()


			if "Bounded" in at:
				# Create different probabilities for an agent being rational
				probsRational = np.linspace(0.0, 100.0, num = 100)
			else:
				# Create different probabilities for an agent being rational
				probsRational = np.linspace(0.0, 1.0, num = 100)
			

			# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
			gx, gy = np.meshgrid(xSkills,probsRational)


			POINTS = []
			VALUES = []

			# find points and values

			for p in percentOfTrueRewardPerAgent[at]["ps"].keys():
				for x in percentOfTrueRewardPerAgent[at]["ps"][p]["xs"].keys():
					POINTS.append([x,p])
					VALUES.append(percentOfTrueRewardPerAgent[at]["ps"][p]["xs"][x]["percent"]) # percent


			POINTS = np.asarray(POINTS)
			VALUES = np.asarray(VALUES)

			if "Bounded" in at:
				# Z = griddata(POINTS, VALUES, (gxl, gyl), method = 'linear')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
				Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
			else:
				Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
			# Z = griddata(POINTS, VALUES, (gx, gy), method = 'cubic')

			# remove inf's -> causes surface plot to be all of the same color 
			Z[Z == np.inf] = np.nan

			fig = plt.figure()
			ax = plt.subplot(111)

			if "Bounded" in at:
				# cs = plt.contourf(gxl, gyl, Z)
				# plt.scatter(gxl,gyl)
				# plt.yscale("log")
				cs = plt.contourf(gx, gy, Z)
			else:
				cs = plt.contourf(gx, gy, Z)
				# plt.scatter(gx,gy)

			plt.xlabel("XSkills")
			plt.ylabel("PSkills")
			plt.title("Percent of TR | Domain: " + domain + " | Agent: " + at)
			fig.colorbar(cs)

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourPercentTrueRewardObtainedPerAgentType" +os.path.sep + "contourPercentTR-Domain-"+domain+"-Agent-"+at+".png", bbox_inches='tight')
			plt.clf()
			plt.close()


def needToUpdateThis(savingCodeForLater):
	
	# This commented code works but was previously used within another function
	# Since no longer needed there, moving here to make space
	# Not needed for now but storing in case needed later
	# Will need to update if needed

	'''
	fig = plt.figure()
	ax = plt.subplot(111)

	if domain == "1d":
		# windowSize = len(tempPercents) / 2
		windowSize = len(tempPercents)

	elif domain == "2d" or domain == "sequentialDarts":
		windowSize = len(tempPercents)

	# Verify if even and if so, +/- 1 to make it odd since savgol_filter's window size must be odd
	if windowSize % 2 == 0:

		if domain == "1d":
			windowSize += 1
		# 2D, substract since using all info and windowsize must be odd but less than info (x) 
		# Will use all info minus 1
		else:
			windowSize -= 1

	print "windowSize: ", windowSize

	# To smooth out the data (percentages) 
	# As suggested in: https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
	#yhat = savgol_filter(allPercents, len(allPercents)/2, 1) # (window size, polynomial order)
	if domain == "1d":
		yhat = savgol_filter(tempPercents, windowSize, 3) # (window size, polynomial order)
	else:
		yhat = savgol_filter(tempPercents, windowSize, 3) # (window size, polynomial order)


	# plt.bar(ps,percentOfRewardPerAgentAndRationalityParam[aType][ps]["percent"],width = 0.10, color = "blue")
	plt.plot(sortedPs,yhat,lw='2.0', color = "blue")

	plt.xlabel(r'\textbf{Rationality Parameter}')
	plt.ylabel(r'\textbf{Percent of Rand/Max Reward obtained}')
	plt.margins(0.05)
	# plt.title('Avg Rewards | Agent: ' +aType+ " (across " +str(resultsDict[a]["num_exps"])+ " experiments)")

	# Shrink current axis by 20%
	# box = ax.get_position()
	# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentRandMaxRewardObtainedPerAgentType" + os.path.sep  + "results-Percent-RandMax-Rewards-Agent-"+aType+".png", bbox_inches='tight')
	
	# save interactive version

	# ~~~~~ PLOTLY ~~~~~

	# Remove legend
	#ax.get_legend().remove()

	# Re-do axis labels
	ax.set_xlabel('<b>Rationality Parameter</b>',fontsize=18)
	ax.set_ylabel('<b>Percent of Rand/Max Reward obtained</b>', fontsize=18)


	# Create Plotly Plot -- Hosting offline
	plotly_fig =  px.plot_mpl(fig)
	#plotly_fig['layout']['showlegend'] = True   
	#plotly_fig['layout']['autosize'] = True  

	# Save plotly
	unique_url = px.offline.plot(plotly_fig, filename=resultsFolder + os.path.sep + "plots" + \
											os.path.sep + "percentRandMaxRewardObtainedPerAgentType" + os.path.sep + "results-Percent-RandMax-Rewards-Agent-"+aType +".html", auto_open=False)

	plt.clf()
	plt.close()
	'''

def pconf(resultsFolder,domain,domainModule,spaceModule,mode,args):

	print("\n----------------------")
	print("PCONF: ")
	print("Domain: ", domain)
	print("Mode: ", mode)
	print("----------------------\n")


	numSamples = 1000

	mainFolder = "Spaces" + os.path.sep + "ExpectedRewards" + os.path.sep
	fileName = f"ExpectedRewards-{args.domain}-N{numSamples}"
	expectedRFolder = mainFolder + fileName


	pconfPerXskill = {}


	tempName = resultsFolder + os.path.sep + "plots" + os.path.sep + "pconfInfo"

	if domain == "sequentialDarts":
		tempName += "-Values"


	# if file is not present, need to compute info
	if not os.path.exists(tempName):


		if domain == "1d" or domain == "2d":

			if domain == "1d":
				# All the lambdas that we will use to generate the plot
				lambdas = np.logspace(-5,2,100)
				# lambdas = np.linspace(0.001, 100, 100)

				# The xskills we want to have our predictions at
				xskills = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

				args.resolution = 1e-1

				# Get the states to use for evaluation
				states = domainModule.generate_random_states(rng,3,10,numSamples)

			else: # 2D

				# All the lambdas that we will use to generate the plot
				lambdas = np.logspace(-5,1.5,100)

				# The xskills we want to have our predictions at
				# xskills = [5, 10, 30, 50, 70, 90, 110, 130, 150]

				xskills = np.linspace(2.5,150.5,num=33)	

				args.resolution = 5.0

				# Get the states to use for evaluation
				states = domainModule.generate_random_states(rng,1,args.mode)

		elif domain == "sequentialDarts":
			
			lambdas = np.logspace(-5,1.5,100)

			xskills = np.linspace(2.5,150.5,num=33)	

			args.resolution = 5.0
			
			startScore = domainModule.getPlayerStartScore()
			states = list(range(startScore + 1))

			args.N = 1


		print("\nCreating spaces...")

		if domain == "1d" or domain == "2d":
			spaces = spacesModule.SpacesRandomDarts(numSamples,domainModule,args.mode,args.resolution,numSamples,expectedRFolder)
			spaces.updateSpace(xskills,states)

		elif domain == "sequentialDarts":
			spaces = spacesModule.SpacesSequentialDarts(numSamples,domainModule,args.mode,args.resolution,numSamples,expectedRFolder)
			spaces.updateSpace(xskills)

		print("\nDone spaces...")


		# Go through all of the execution skills
		for x in xskills:
			print('Generating data for execution skill level', x)

			prat = [] #This is where the probability of rational reward will be stored
			mins = [] #Store min reward possible
			maxs = [] #Store max reward possible
			means = [] #Store the mean of the possible rewards (this is the uniform random reward)
			evs = [] #Store the ev of the current agent's strategy

			if domain in ["1d","2d"]:
				space = spaces.convolutionsPerXskill[x]
			else:
				space = spaces.spacesPerXskill[x]


			if domain in ["1d","2d"]:
				size = len(states)
				loopInfo = states

			else:
				size = len(states)-2
				loopInfo = states[2:]


			for l in lambdas:     

				# Minus 2 to not include info for state 0 and 1
				# If initialize to 0, causes NANs
				max_rs = np.zeros(size)
				min_rs = np.zeros(size)
				exp_rs = np.zeros(size)
				mean_rs = np.zeros(size)

				si = 0
				
				for s in loopInfo:

					if domain == "1d":
						values = space[str(s)]["all_vs"]
					elif domain == "2d":
						values = space[str(s)]["all_vs"].flatten()
					else:
						values = np.copy(space.flatEVsPerState[s])


					# Get the values from the ev 
					max_rs[si] = np.max(values)
					min_rs[si] = np.min(values) 
					mean_rs[si] = np.mean(values) 

					# Bounded decision-making with lambda = l
					b = np.max(values*l)
					expev = np.exp(values*l-b)
					sumexp = np.sum(expev)
					P = expev/sumexp


					# Store bounded agent's EV
					exp_rs[si] = sum(P*values)
					# code.interact("v...", local=dict(globals(), **locals()))


					si += 1

				
				prat.append(np.mean((exp_rs - mean_rs)/(max_rs - mean_rs)))
				mins.append(np.mean(min_rs))
				means.append(np.mean(mean_rs))
				maxs.append(np.mean(max_rs))
				evs.append(np.mean(exp_rs))

			# plt.plot(lambdas, prat, label='x=' + str(x))

			# store to use later
			pconfPerXskill[x] = {"lambdas":lambdas, "prat": prat}
			#code.interact("...",local=locals())

		# plt.xlabel('Lambda')
		# plt.ylabel('% Rational Reward')
		# plt.legend()
		# plt.show()

		# Save dict containing all info - to be able to rerun it later
		with open(tempName,"wb") as outfile:
			pickle.dump(pconfPerXskill, outfile)

	# file with the info is present, proceed to load
	else:
		print("Loading pconf info...")

		with open(tempName, "rb") as file:
			pconfPerXskill = pickle.load(file)

	# code.interact("pconf end",local=locals())
	print("Finished pconf()")
	return pconfPerXskill


# Converting from true pskill to rationality percentage
def plotPercentRandMaxRewardObtainedPerXskillPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain, pconfPerXskill):
	# all xskills included - focus on pskills

	makeFolder(resultsFolder, "percentRandMaxRewardObtained-PerXskillPerAgentType")

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["axes.titleweight"] = "bold"

	bucketsX = sorted(pconfPerXskill.keys())
	
	if domain == "1d":
		minMaxX = [0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]


	percentOfRewardPerAgent = {}

	for at in agentTypes:

		percentOfRewardPerAgent[at] = {}

		percentOfRewardPerAgent[at] = {"allPercents": [], "allXskills": [], "allPskills": []}

	for a in resultsDict.keys():

		aType, xStr, p = getParamsFromAgentName(a)

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


		'''
		if aType == "Flip" or aType == "Bounded":
			print("\nAgent: ",a)
			print("maxR: ",maxR)
			print("expectedR: ",expectedR)
			print("minR: ",minR)
		'''
		

		# if percentOfReward < 0: 
			# code.interact("% of reward is negative...", local=locals())
		# code.interact(f"% of reward is {percentOfReward}...", local=locals())
		
		#code.interact("after % of reward...", local=locals())

		# if aType == "Flip" or aType == "Bounded": # and float(xStr) >= 100:
			# code.interact("...", local=dict(globals(), **locals()))
		

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

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentRandMaxRewardObtained-PerXskillPerAgentType" + os.path.sep  + "results-Percent-RandMaxRewards-Agent-"+aType+".png", bbox_inches='tight')

		plt.clf()
		plt.close()


		#######################################################################################################################################
	

	# TO PLOT PCONF INFO
	ax = plt.subplot(111)

	for b in bucketsX:
		plt.plot(pconfPerXskill[b]["lambdas"], pconfPerXskill[b]["prat"], label = b)

	plt.xlabel(r'\textbf{Rationality Parameter}')
	plt.ylabel(r'\textbf{Rationality Percentage}')
	
	plt.legend()

	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentRandMaxRewardObtained-PerXskillPerAgentType" + os.path.sep  + "results-Percent-RandMaxRewards-FITTED-LINE-Agent-Bounded-PCONF.png", bbox_inches='tight')

	plt.clf()
	plt.close()

	code.interact("after percent rationality...", local=dict(globals(), **locals()))


# Currently not used
def plotPercentRandMaxRewardObtainedPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain):
	# all xskills included - focus on pskills

	makeFolder(resultsFolder, "percentRandMaxRewardObtainedPerAgentType")

	percentOfRewardPerAgentAndRationalityParam = {}

	for at in agentTypes:
		if "Target" in at:
			percentOfRewardPerAgentAndRationalityParam[at] = {"rationalityParams": [], "allPercents": [], "allXskills": [], "numExps": 0.0}
		else:
			percentOfRewardPerAgentAndRationalityParam[at] = {"rationalityParams": {}}

	for a in resultsDict.keys():

		aType, xStr, p = getParamsFromAgentName(a)


		if aType == "Target":
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"].append(p)
			percentOfRewardPerAgentAndRationalityParam[aType]["allXskills"].append(p)
		elif aType == "Random":
			if N not in percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys(): 
				percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N] = {}
			if K not in percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N].keys():
				percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N][K] = {"percent": 0.0, "numExps": 0.0, "allPercents": [], "allXskills": []}
		else:
			if p not in percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys():
				percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p] = {"percent": 0.0, "numExps": 0.0, "allPercents": [], "allXskills": []}


		# minR = resultsDict[a]["mean_random_reward"]
		minR = resultsDict[a]["mean_random_reward_mean_vs"]

		expectedR = resultsDict[a]["mean_value_intendedAction"]
		maxR = resultsDict[a]["mean_true_reward"]

		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		# percentOfReward = (((expectedR - minR)/(maxR - minR)) * 100.0)
		percentOfReward = (((expectedR - minR)/(maxR - minR)))

		resultsDict[a]["percentTrueP"] = percentOfReward


		if percentOfReward < 0: 
			code.interact("% of reward is negative...", local=locals())
		

		if aType == "Target":
			percentOfRewardPerAgentAndRationalityParam[aType]["allPercents"].append(percentOfReward)

		elif aType == "Random":
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N][K]["percent"] += percentOfReward
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N][K]["allPercents"].append(percentOfReward)
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N][K]["allXskills"].append(float(xStr))
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][N][K]["numExps"] += 1.0

		else:
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["percent"] += percentOfReward
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["allPercents"].append(percentOfReward)
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["allXskills"].append(float(xStr))
			percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["numExps"] += 1.0

		# print "Agent: ", a
		# print "\tminR: ", minR
		# print "\texpectedR: ", expectedR
		# print "\tmaxR: ", maxR  
		# print "\n"  
		# code.interact("", local=locals())


	functionsPerAgentType = {}

	for aType in agentTypes:

		# If we didn't see any experiments for the given agent type, continue/skip
		if aType not in seenAgents:
			continue


		if aType not in functionsPerAgentType.keys():
			functionsPerAgentType[aType + "-" + domain] = {"params":0, "function":0}


		tempPercents = []
		tempXskills = []

		if aType == "Target":
			# Get all the 100's
			sortedPs = percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"]
			tempPercents = percentOfRewardPerAgentAndRationalityParam[aType]["allPercents"]
			tempXskills = percentOfRewardPerAgentAndRationalityParam[aType]["allXskills"]

		elif aType == "Random":

			allNs = []
			allKs = []

			# need to handle xskills for random agent
			tempXskills = []


			for ns in percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys():
				for ks in percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ns].keys():

					allNs.append(float(ns))
					allKs.append(float(ks))

					percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ns][ks]["percent"] /= percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ns][ks]["numExps"]
					tempPercents.append(percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ns][ks]["percent"])

		else:
			sortedPs = sorted(percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys()) 

			for ps in sortedPs:
				percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["percent"] /= percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["numExps"]
				tempPercents.append(percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["percent"])

				tempXskills.append(percentOfRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["allXskills"][0])


		# code.interact(local = locals())

		#######################################################################################################################################
		# Learning the function
		#######################################################################################################################################
		
		# linear regression
		# slope, intercept, r_value, p_value, std_err = stats.linregress(sortedPs, tempPercents)

		if aType == "Bounded":

			# '''
			params, pcov = curve_fit(func,np.asarray(sortedPs),np.asarray(tempPercents))

			tempYs = []

			for i in range(len(sortedPs)):

				# binding curve
				tempYs.append(func(sortedPs[i],params[0],params[1]))
			
			functionsPerAgentType[aType + "-" + domain]["params"] = params
			functionsPerAgentType[aType + "-" + domain]["function"] = func
			# '''

		elif aType == "Random":
			
			params, pcov = curve_fit(funcRandom,[allNs,allKs],tempPercents)

			tempNs = []
			tempKs = []
			tempYs = []

			for i in range(len(allNs)):
				for j in range(len(allKs)):
					tempNs.append(allNs[i])
					tempKs.append(allKs[j])
					
					tempYs.append(funcRandom([allNs[i],allKs[j]],params[0],params[1]))
				
			functionsPerAgentType[aType + "-" + domain]["params"] = params
			functionsPerAgentType[aType + "-" + domain]["function"] = funcRandom

		else:
			degree = 3

			# polynomial fit
			z = np.polyfit(sortedPs,tempPercents,degree)

			poly = np.poly1d(z)

			tempYs = []

			for i in range(len(sortedPs)):

				# linear regression
				# tempYs.append(intercept + slope*sortedPs[i])

				# polynomial fit
				tempYs.append(poly(sortedPs[i]))

			functionsPerAgentType[aType + "-" + domain]["params"] = z
			functionsPerAgentType[aType + "-" + domain]["function"] = poly


		# scatter plot
		fig = plt.figure()

		# skip bounded agent for now since creating the plot above --- for now
		if aType == "Bounded":
			continue
		
		if aType == "Random":
			from mpl_toolkits.mplot3d import Axes3D
			#ax = fig.add_subplot(111, projection='3d')
			ax = plt.axes(projection='3d')
			ax.scatter(allNs, allKs, tempPercents, color = "blue")
			# ax.plot(tempNs, tempKs, tempYs, color = "red") # plot learned function

			ax.set_xlabel("N")
			ax.set_ylabel("K")
			ax.set_zlabel("Percent of Rand/Max Reward obtained")

		else:
			ax = plt.subplot(111)

			plt.scatter(sortedPs,tempPercents, c = tempXskills, cmap = "viridis")
			plt.colorbar()
			plt.plot(sortedPs, tempYs, lw='2.0', color = "red", label = "fitted line")

			plt.xlabel(r'\textbf{Rationality Parameter}')
			plt.ylabel(r'\textbf{Percent of Rand/Max Reward obtained}')
		plt.margins(0.05)

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentRandMaxRewardObtainedPerAgentType" + os.path.sep  + "results-Percent-RandMaxRewards-FITTED-LINE-Agent-"+aType+".png", bbox_inches='tight')
		
		plt.clf()
		plt.close()


		#######################################################################################################################################

	return percentOfRewardPerAgentAndRationalityParam, functionsPerAgentType

# Currently not used
def plotPercentTrueRewardObtainedPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain):
	# all xskills included - focus on pskills

	makeFolder(resultsFolder, "percentTrueRewardObtainedPerAgentType")

	percentOfTrueRewardPerAgentAndRationalityParam = {}

	for at in agentTypes:
		percentOfTrueRewardPerAgentAndRationalityParam[at] = {"rationalityParams": {}}

	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		if p not in percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys():
			percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p] = {"percent": 0.0, "numExps": 0.0, "allPercents": []}

		# percentOfTrueReward = (resultsDict[a]["mean_rs_reward_per_exp"] / resultsDict[a]["mean_true_reward"]) * 100.0
		percentOfTrueReward = (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"]) * 100.0
		
		percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["percent"] += percentOfTrueReward
		percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["allPercents"].append(percentOfTrueReward)

		percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][p]["numExps"] += 1.0

		# code.interact(local=locals())

	functionsPerAgentType = {}

	for aType in agentTypes:

		if aType == "Target" or aType == "Random":
			continue


		# If we didn't see any experiments for the given agent type, continue/skip
		if aType not in seenAgents:
			continue


		if aType not in functionsPerAgentType.keys():
			functionsPerAgentType[aType + "-" + domain] = {"params":0, "function":0}

		tempPercents = []
		sortedPs = sorted(percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"].keys()) 

		for ps in sortedPs:
			percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["percent"] /= percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["numExps"]
			tempPercents.append(percentOfTrueRewardPerAgentAndRationalityParam[aType]["rationalityParams"][ps]["percent"])


		#######################################################################################################################################
		# Learning the function
		#######################################################################################################################################
		
		# linear regression
		# slope, intercept, r_value, p_value, std_err = stats.linregress(sortedPs, tempPercents)

		if aType == "Bounded":
			# degree = 6

			params, pcov = curve_fit(func,sortedPs,tempPercents)

			tempYs = []

			for i in range(len(sortedPs)):

				# binding curve
				tempYs.append(func(sortedPs[i],params[0],params[1]))
			
			functionsPerAgentType[aType + "-" + domain]["params"] = params
			functionsPerAgentType[aType + "-" + domain]["function"] = func

		else:
			degree = 3

			# polynomial fit
			z = np.polyfit(sortedPs,tempPercents,degree)

			poly = np.poly1d(z)

			tempYs = []

			for i in range(len(sortedPs)):

				# linear regression
				# tempYs.append(intercept + slope*sortedPs[i])

				# polynomial fit
				tempYs.append(poly(sortedPs[i]))

			functionsPerAgentType[aType + "-" + domain]["params"] = z
			functionsPerAgentType[aType + "-" + domain]["function"] = poly


		fig = plt.figure()
		ax = plt.subplot(111)
		
		plt.plot(sortedPs,tempPercents, lw='2.0', color = "blue")
		plt.plot(sortedPs, tempYs, lw='2.0', color = "red", label = "fitted line")

		plt.xlabel(r'\textbf{Rationality Parameter}')
		plt.ylabel(r'\textbf{Percent of True Reward obtained}')
		plt.margins(0.05)

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "percentTrueRewardObtainedPerAgentType" + os.path.sep  + "results-Percent-TrueRewards-FITTED-LINE-Agent-"+aType+".png", bbox_inches='tight')
		
		plt.clf()
		plt.close()


		#######################################################################################################################################

	return percentOfTrueRewardPerAgentAndRationalityParam, functionsPerAgentType

def func(x,a,b):
# def func(t, P0, P1, P2, P3):

	# binding curve
	# formula obtained from: https://stackoverflow.com/questions/49944018/fit-a-logarithmic-curve-to-data-points-and-extrapolate-out-in-numpy/49944478
	return (b*x)/((x+a)*1.0)

	# bezier curve
	# y = P0*(1-t)*(1-t)*(1-t) + 3*(1-t)*(1-t)*t*P1 + 3*(1-t)*t*t*P2 + t*t*t*P3
	# return y

def funcRandom(params,a,b):
	x = params[0]
	y = params[1]
	return a*x + b*y



def plotEVintendedVSagentType(resultsDict, numHypsX, numHypsP, resultsFolder, domain):

	makeFolder(resultsFolder, "EVsIntendedPerAgentType")

	rewardsPerAgentType = {}

	for a in resultsDict.keys():

		agentType, x, p = getParamsFromAgentName(a)


		if agentType == "Target":
			continue
		if agentType == "Random":
			continue


		if agentType not in rewardsPerAgentType.keys():
			rewardsPerAgentType[agentType] = {"x": [], "p": [], "evIntended": [], "trueReward": []}

		rewardsPerAgentType[agentType]["x"].append(x)
		rewardsPerAgentType[agentType]["p"].append(p)
		rewardsPerAgentType[agentType]["evIntended"].append(resultsDict[a]["mean_value_intendedAction"])
		rewardsPerAgentType[agentType]["trueReward"].append(resultsDict[a]["mean_true_reward"])


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

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "EVsIntendedPerAgentType" + os.path.sep + "evsIntended-domain-" + domain + "-agent" + at + ".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()

def plotRewardsVSagentType(resultsDict, numHypsX, numHypsP, resultsFolder, domain):

	makeFolder(resultsFolder, "rewardsPerAgentType")

	rewardsPerAgentType = {}

	for a in resultsDict.keys():

		agentType, x, p = getParamsFromAgentName(a)

		if agentType == "Target":
			continue
		if agentType == "Random":
			continue


		if agentType not in rewardsPerAgentType.keys():
			rewardsPerAgentType[agentType] = {"x": [], "p": [], "observedReward": [], "trueReward": []}

		rewardsPerAgentType[agentType]["x"].append(x)
		rewardsPerAgentType[agentType]["p"].append(p)
		rewardsPerAgentType[agentType]["observedReward"].append(resultsDict[a]["mean_observed_reward"])
		rewardsPerAgentType[agentType]["trueReward"].append(resultsDict[a]["mean_true_reward"])


	for at in rewardsPerAgentType.keys():

		fig = plt.figure()

		fig.suptitle("Rewards | Domain: " + domain + " | Agent: " + at)

		# rows, cols, pos
		ax = fig.add_subplot(2, 1, 1)
		ax2 = fig.add_subplot(2, 1, 2)

		#plt.plot(eachParam,params[eachParam]["observed"], color = "blue", marker = ".")
		#plt.plot(eachParam,params[eachParam]["true"], color = "red", marker = "*")

		s = ax.scatter(rewardsPerAgentType[at]["x"],rewardsPerAgentType[at]["p"], c = rewardsPerAgentType[at]["observedReward"])
		cbar = fig.colorbar(s, ax = ax)
		cbar.set_label(" Observed Reward", labelpad=+1)

		s2 = ax2.scatter(rewardsPerAgentType[at]["x"],rewardsPerAgentType[at]["p"], c = rewardsPerAgentType[at]["trueReward"])
		cbar = fig.colorbar(s2, ax = ax2)
		cbar.set_label("True Reward", labelpad=+1)

		
		ax.set_xlabel("xSkills")
		ax.set_ylabel("pSkills")
		
		ax2.set_xlabel("xSkills")
		ax2.set_ylabel("pSkills")
		
		plt.margins(0.05)

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "rewardsPerAgentType" + os.path.sep + "rewards-domain-" + domain + "-agent" + at + ".png", bbox_inches = 'tight')
		plt.clf()
		plt.close()


def plotMeanAVGAndTrueRewardsPerAgent(resultsDict, resultsFolder):

	makeFolder(resultsFolder, "meanAVGAndTrueRewardsPerAgent")

	for a in resultsDict.keys():

		fig = plt.figure()
		ax = plt.subplot(111)

		plt.plot(range(len(resultsDict[a]["avg_rewards"])),resultsDict[a]["avg_rewards"], lw='2.0', label= "Avg" )
		plt.plot(range(len(resultsDict[a]["true_rewards"])),resultsDict[a]["true_rewards"],c = "k", lw='2.0', label= "True" )

		plt.xlabel(r'\textbf{Number of observations}')
		plt.ylabel(r'\textbf{Reward}')
		plt.margins(0.05)
		plt.title('Rewards comparison | Agent: ' +a+ " (across " +str(resultsDict[a]["num_exps"])+ " experiments)")

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanAVGAndTrueRewardsPerAgent" + os.path.sep + "results-AVG&True-Rewards-Agent-"+a+".png", bbox_inches='tight')
		plt.clf()
		plt.close()

def plotMeanAVGRewardsAllAgents(resultsDict, resultsFolder):

	makeFolder(resultsFolder, "meanAVGRewardsAllAgents")

	fig = plt.figure()
	ax = plt.subplot(111)

	# create plot for the avg mean rewards for ALL agents on same plot
	for a in resultsDict.keys():
		plt.plot(range(len(resultsDict[a]["avg_rewards"])),resultsDict[a]["avg_rewards"], lw='2.0', label= str(a))

	plt.xlabel(r'\textbf{Number of observations}')
	plt.ylabel(r'\textbf{Avg Reward}')
	plt.margins(0.05)
	plt.title('Avg Rewards of different agents')

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanAVGRewardsAllAgents" + os.path.sep + "results-AVG-Rewards-ALL-Agents.png", bbox_inches='tight')
	plt.clf()
	plt.close()

def plotMeanAVGRewardsPerAgent(resultsDict, agentTypes, resultsFolder):

	makeFolder(resultsFolder, "meanAVGRewardsPerAgent")

	# create plot for the avg mean rewards for each one of the different agents
	for aType in agentTypes:

		fig = plt.figure()
		ax = plt.subplot(111)

		for a in resultsDict.keys():
			if aType in a:
				plt.plot(range(len(resultsDict[a]["avg_rewards"])),resultsDict[a]["avg_rewards"], lw='2.0', label= str(a))

		plt.xlabel(r'\textbf{Number of observations}')
		plt.ylabel(r'\textbf{Avg Reward}')
		plt.margins(0.05)
		plt.title('Avg Rewards | Agent: ' +aType+ " (across " +str(resultsDict[a]["num_exps"])+ " experiments)")

		# Shrink current axis by 20%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanAVGRewardsPerAgent" + os.path.sep  + "results-AVG-Rewards-Agent-"+aType+".png", bbox_inches='tight')
		plt.clf()
		plt.close()

###################################################################################


################################## FOR ESTIMATES ##################################

def computeAndPlotEstimatesPercentPerXskillBucketsPerMethodsAndAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, domain, seenAgents):


	makeFolder(resultsFolder, "EstimatesPercentScatter-PerXskillBuckets-PerAgentTypeAndPerPskillMethods")


	if domain == "1d":
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]

	bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		minMaxX = [0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]


	# init dict to store info
	infoDict = {}

	for at in seenAgents:
		infoDict[at] = {"perMethod": {}, "numAgents": 0.0}


		for m in actualMethodsOnExps:
			# Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			infoDict[at]["perMethod"][m] = {}

			for b in buckets:
				infoDict[at]["perMethod"][m][b] = {"truePercents": [], "estimatedPercents":[], "pskills": [], "xskills": []}



	# convert trueP & estimatedP (per method & per state) to it's corresponding % of TR

	for a in resultsDict.keys():

		aType, X, p = getParamsFromAgentName(a)

		# trueP = find agent's true pskill
		trueP = p

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if X <= buckets[b]:
				break

		# get actual bucket
		bucket1 = buckets[b]

		percent_trueP = resultsDict[a]["percentTrueP"]


		# for each method
		for m in actualMethodsOnExps:

			if m == "tn" or "xSkills" in m:
				continue

			# Skip OR & TBA
			if "pSkills" not in m:
				continue

			infoDict[aType]["perMethod"][m][bucket1]["pskills"].append(trueP)
			infoDict[aType]["perMethod"][m][bucket1]["xskills"].append(X)



			if "JT-QRE" in m:
				# method = "JT-QRE"
				aFunc = "Bounded"
			elif "JT-FLIP" in m:
				# method = "JT-FLIP"
				aFunc = "Flip"
			elif "NJT-QRE" in m:
				# method = "NJT-QRE"
				aFunc = "Bounded"


			# Get pskill estimate of current method - estimatedP
			estimatedP = resultsDict[a]["estimates"][m][-1]
			#print estimatedP


			###################################################################################
			# convert estimatedP to corresponding % of rand max

			otherBucket = None
			bucket2 = None

			# first bucket
			if b == 0:
				# use left edge/extreme - i.e. 0
				otherBucket = minMaxX[0]
			# if last bucket
			elif b == len(bucketsX)-1:
				# use right edge/extreme - i.e. 5/100 depending on the domain
				otherBucket = minMaxX[1]
			# somewhere in the middle - consider next bucket
			else:
				otherBucket = b + 1
				bucket2 = bucketsX[otherBucket]

			# code.interact(local=locals())

			if bucket2 != None:

				# pconfPerXskill[x] = {"lambdas":lambdas, "prat": prat}
				# evAction = np.interp(action, all_ts, all_vs)
				prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
				prat2 = np.interp(estimatedP,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

				prat = np.interp(estimatedP, [prat1], [prat2])
				percent_estimatedP = prat
			# edges/extremes case
			else:
				# using one of the functions for now
				prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

				percent_estimatedP = prat1
			###################################################################################

			

			infoDict[aType]["perMethod"][m][bucket1]["truePercents"].append(percent_trueP)
			infoDict[aType]["perMethod"][m][bucket1]["estimatedPercents"].append(percent_estimatedP)

			# For debuging
			'''
			if "Target" in a and bucket1 == 5.0:
				print "Agent: ", a
				print "Method: ", m
				code.interact("here, after estimates....", local = dict(globals(), **locals()))
			'''

	   

	#code.interact("EstimatesPercentScatter", local=locals())


	# NEED TO NORMALIZE???

	# normalize - find Mean Squared Error

	'''
	# for each agent type
	for at in seenAgents:
		if "Target" not in at and "Random" not in at:

			# for each method
			for m in actualMethodsOnExps:

				if m == "tn" or "xSkills" in m:
					continue

				# Skip OR & TBA
				if "pSkills" not in m:
					continue

				# for each p
				for ps in percentRewardObtainedPerAgentType[at]["perMethod"][m].keys():
					
					# for each state
					for mxi in range(numStates):

						#find MSE  = SE / # of agents
						percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["se"][mxi] /=  percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["numAgents"] 
	'''

	# code.interact("", local=locals())
	# '''


	# for each agent type
	for at in seenAgents:

		if "Random" not in at:

			# for each method
			for m in actualMethodsOnExps:

				if m == "tn" or "xSkills" in m:
					continue

				# Skip OR & TBA
				if "pSkills" not in m:
					continue

				for b in buckets:

					fig = plt.figure()
					ax = plt.subplot(111)


					# in case bucket is empty
					try:
						cmap = plt.get_cmap("viridis")
						norm = plt.Normalize(min(buckets), max(buckets))

						sm = ScalarMappable(norm = norm, cmap = cmap)
						sm.set_array([])
						cbar = fig.colorbar(sm)

						ax.scatter(infoDict[at]["perMethod"][m][b]["truePercents"], infoDict[at]["perMethod"][m][b]["estimatedPercents"],\
						c = cmap(norm(np.array(infoDict[at]["perMethod"][m][b]["xskills"]))))

						cbar.ax.get_yaxis().labelpad = 15
						cbar.ax.set_ylabel('true xskill', rotation=270)


					except:
						continue


					plt.xlabel('Percents True P',fontsize=18)
					plt.ylabel('Percent Estimated P', fontsize=18)

					plt.margins(0.05)
					plt.title('Agent: ' + at + " | Method: " + m + " | XBucket: " + str(b))

					# Shrink current axis by 10%
					box = ax.get_position()
					ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

				
					plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "EstimatesPercentScatter-PerXskillBuckets-PerAgentTypeAndPerPskillMethods" + os.path.sep +  "results-Agent"+at+"-Method"+m+"-XBucket"+str(b)+".png", bbox_inches='tight')

					plt.clf()
					plt.close(fig)


def plotMeanEstimatesForDiffXskillPskillsPerMethodAndAgentTypeAndGivenStates(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, agentTypes, numStates, domain):
	############################################################################################################################


	makeFolder(resultsFolder, "meanEstimatesForDiffXskillsPskillsPerMethodAndAgentType-GivenStates")

	getStates = 25

	info = {}


	for a in resultsDict.keys():

		at, X, p = getParamsFromAgentName(a)


		if at not in info.keys():
			info[at] = {}


		for m in methods:
			#method += "-" + str(numHyps[0]

			if m not in info[at].keys():
				info[at][m] = {"rationalityParams": [], "xs": [], "estimates": [], "percentsEstimatedPs": []}

			info[at][m]["rationalityParams"].append(p)
			info[at][m]["xs"].append(X)


			info[at][m]["estimates"].append(resultsDict[a]["estimates"][m][0:getStates])

			if "-pSkills" in m:
				info[at][m]["percentsEstimatedPs"].append(resultsDict[a]["percentsEstimatedPs"][m]["averaged"][0:getStates])


		##########################################################################


	#code.interact(local=locals())

	cmap = plt.get_cmap("viridis")


	if domain == "1d":
		# Create different execution skill levels 
		xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
		minX = 0.5
		maxX = 4.5

	elif domain == "2d" or domain == "sequentialDarts":
		# Create different execution skill levels 
		xSkills = np.linspace(2.5, 100.5, num = 100) # (start, stop, num samples)
		minX = 2.5
		maxX = 100.5


	for at in info.keys():

		if "Bounded" in at or "Target" in at:
			minP = 0.0
			maxP = 100.0
		else:
			minP = 0.0
			maxP = 1.0


		for m in info[at].keys():

			if "tn" in m:
				continue


			if "pSkills" in m:
				#norm = plt.Normalize(np.log(minP), np.log10(maxP))
				norm = plt.Normalize(minP, maxP)

			# for xskills - diff range
			else:
				norm = plt.Normalize(minX, maxX)


			for s in range(getStates):

				fig = plt.figure()
				ax = plt.gca()

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm)

				
				#plt.scatter(info[at][m]["xs"],np.log10(info[at][m]["rationalityParams"]), c =  cmap(norm(np.asarray(np.log10(info[at][m]["estimates"][s])))), cmap = "viridis")
				plt.scatter(info[at][m]["xs"],info[at][m]["rationalityParams"], \
					#sm.to_rgba(np.asarray(info[at][m]["estimates"][s])))
					c =  cmap(norm(np.asarray(info[at][m]["estimates"])[:,s])), cmap = "viridis")
				
				plt.xlabel(r'\textbf{Xskills}')
				plt.ylabel(r'\textbf{Pskills}')
				plt.title("Actual Estimates | Agent" + at + " | Method" + m + " | State" + str(s))

				# Save png
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanEstimatesForDiffXskillsPskillsPerMethodAndAgentType-GivenStates" + os.path.sep + "results-AgentType" + at + "-Method" + m + "-State" + str(s) + ".png", bbox_inches='tight')
				

				plt.clf()
				plt.close()


				# create same plot for % of R's
				if "pSkills" in m:

					# Percent between 0 -1 
					norm2 = plt.Normalize(0.0, 1.0)

					fig = plt.figure()
					ax = plt.gca()

					sm = ScalarMappable(norm = norm2, cmap = cmap)
					sm.set_array([])
					cbar = fig.colorbar(sm)

					plt.scatter(info[at][m]["xs"],info[at][m]["rationalityParams"],\
						#sm.to_rgba(np.asarray(info[at][m]["percentsEstimatedPs"][s])))
						c =  cmap(norm2(np.asarray(info[at][m]["percentsEstimatedPs"])[:,s])), cmap = "viridis")
					
					plt.xlabel(r'\textbf{Xskills}')
					plt.ylabel(r'\textbf{Pskills}')
					plt.title("PercentRewards | Agent" + at + " | Method" + m + " | State" + str(s))


					# Save png
					plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanEstimatesForDiffXskillsPskillsPerMethodAndAgentType-GivenStates" + os.path.sep + "results-AgentType" + at + "-Method" + m + "-PercentRewards-" + "-State" + str(s) + ".png", bbox_inches='tight')
					

					plt.clf()
					plt.close()

	# code.interact("",local=locals())
	#######################################################################################

def plotMeanEstimatesForDiffPskillsPerMethodAndAgentType(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, agentTypes, numStates):
	############################################################################################################################

	# for the Bounded & Target agent for now

	# x = different pskills
	# y = avg estimate
	# for the different pskills - (not accounting for xskill)
	# accross the different experiments


	makeFolder(resultsFolder, "meanEstimatesForDiffPskillsPerMethodAndAgentType")

	info = {}


	for a in resultsDict.keys():

		at, x, p = getParamsFromAgentName(a)


		if at not in info.keys():
			info[at] = {}


		for m in methods:
			#method += "-" + str(numHyps[0])

			if "pSkills" not in m:
				continue

			if m not in info[at].keys():
				info[at][m] = {"rationalityParams": [], "estimates": []}

			info[at][m]["rationalityParams"].append(p)

			info[at][m]["estimates"].append(resultsDict[a]["estimates"][m][numStates-1])


		##########################################################################

	for at in info.keys():

		for m in info[at].keys():

			fig = plt.figure()
			ax = plt.gca()

			# ax.scatter(info[at][m]["rationalityParams"], info[at][m]["estimates"], label = m)
			# ax.set_xscale('log')
			# ax.set_yscale('log')
			
			ax.scatter(np.log10(info[at][m]["rationalityParams"]), np.log10(info[at][m]["estimates"]), label = m)
			

			plt.xlabel(r'\textbf{Pskills}')
			plt.ylabel(r'\textbf{Estimate}')

			plt.legend()


			# Save png
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "meanEstimatesForDiffPskillsPerMethodAndAgentType" + os.path.sep + "results-AgentType" + at + "-Method" + m + ".png", bbox_inches='tight')
			

			plt.clf()
			plt.close()

	#code.interact("",local=locals())
	#######################################################################################



def plotMeanEstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, agentTypes, seenAgents,givenBeta):
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


	for a in resultsDict.keys():

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

				for a in resultsDict.keys():
					# code.interact(local = locals())

					if at == "Target": # since target agent doesn't have a p param
						if ("X" + str(x)[:-1]) in a and at in a:
							found = True
					else:
						if ("X" + str(x)[:-1]) in a and  str(p)[:-1] in a and at in a:
							found = True


					if found:

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

						# stop searching - assuming only 1 exp per agent since params assigned at random
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

					plt.plot(range(len(infoPlot[at][x][p][m])),infoPlot[at][x][p][m], lw = 2.0, label = round(p,4), color = color, ls = ls[mi])
				
					plt.plot(range(len(infoPlot[at][x][p][m])),[x]*len(infoPlot[at][x][p][m]), lw = 2.0, color = color)

					mi += 1


			plt.xlabel(r'\textbf{Number of observations}')
			plt.ylabel(r'\textbf{Execution Noise Level Estimate}')
			plt.margins(0.05)
			
			#fig.suptitle(r'\textbf{Method: ' + str(m) + ' | Agent Type: ' + at)

			# Put a legend to the right of the current axis
			plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			# Save png
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "EstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType" + os.path.sep + "estimatesXSKILLS-Method" + m + "-AgentType" + at + ".png", bbox_inches='tight')
			

			plt.clf()
			plt.close()


	##################################### FOR PSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs

	#'''
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

				
					plt.plot(range(len(infoPlot[at][x][p][m]["percentsEstimatedPs"])),infoPlot[at][x][p][m]["percentsEstimatedPs"], lw = 2.0, label = p, color = color, ls = ls[mi])
				
					plt.plot(range(len(infoPlot[at][x][p][m]["percentsEstimatedPs"])),[infoPlot[at][x][p][m]["percentTrueP"]]*len(infoPlot[at][x][p][m]["percentsEstimatedPs"]), lw = 2.0, color = color)

					mi += 1


				plt.xlabel(r'\textbf{Number of observations}')
				plt.ylabel(r'\textbf{Percent Reward}')
				plt.margins(0.05)
				
				fig.suptitle(r'\textbf{Method: ' + str(m) + ' | Agent Type: ' + at + " | Xskill: " + str(x))

				# Put a legend to the right of the current axis
				plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = "pskill")

				# Save png
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "EstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType" + os.path.sep + "estimatesPSKILLS-Method" + m + "-AgentType" + at + "-Xskills" + str(x) + ".png", bbox_inches='tight')

				plt.clf()
				plt.close()
	#'''


	#######################################################################################


def plotMeanEstimatesAllAgentsSamePlotPerMethodAndPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, agentTypes):
	############################################################################################################################
	# plot all agents (of a given type) on the same plot for each one of the methods 
	# will plot mean of estimates
	# y axis = agent's true noise
	############################################################################################################################

	makeFolder(resultsFolder, "estimateAllAgentsSamePlotPerMethodAndPerAgentType")

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	noisesOfAgentTypes = {}

	# get the different noises
	for at in agentTypes:

		noisesOfAgentTypes[at] = []

		for a in resultsDict.keys():

			aType, x, p = getParamsFromAgentName(a)

			# if agent is of given type
			if aType == at:

				noisesOfAgentTypes[at].append(x)

	for method in methodsNames:

		method += "-" + str(numHypsX[0]) + "-" + str(numHypsP[0])

		if "tn" in method:
			continue


		##################################### FOR XSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		for at in agentTypes:

			noises = sorted(noisesOfAgentTypes[at])

			# for each noise
			for ax in noises:

				fig = plt.figure()

				# use as colors list index
				i = 0

				if "OR" in method or "BM" in method:
					ax1 = plt.subplot(label = "xskills")
				else:
					ax1 = plt.subplot(2, 1, 1, label = "xskills")


				# for each agent
				for a in resultsDict.keys():

					aType, x, p = getParamsFromAgentName(a)

					# only consider agent if of current agent type
					if aType == a:

						#and if of the given noise
						if x == ax: 


							# get color
							color = colors[list(colors.keys())[i]]

							# increment index
							i += 1  

							if "OR" in method or "BM" in method:
								methodN = method
							else:
								methodN = method + "-xSkills" 


							# plt.semilogx(range(len(resultsDict[a]["estimates"][methodN])),resultsDict[a]["estimates"][methodN], lw='2.0', label = a, color = color)
							ax1.plot(range(len(resultsDict[a]["estimates"][methodN])),resultsDict[a]["estimates"][methodN], lw='2.0', label = a, color = color)
								
							ax1.plot(range(len(resultsDict[a]["estimates"][methodN])),[x]*len(resultsDict[a]["estimates"][methodN]), lw='2.0', color = color)


							##################################### FOR PSKILLS #####################################
							# JT Method thus create subplot for pskills
							if "OR" not in method and "BM" not in method:
							
								# ADD SUBPLOT FOR PSKILLS
								ax2 = plt.subplot(2, 1, 2, label = "pskills")

								methodN = method + "-pSkills" 

								# plt.semilogx(range(len(resultsDict[a]["estimates"][methodN])),resultsDict[a]["estimates"][methodN], lw='2.0', label = methodN, color = color)
								ax2.plot(range(len(resultsDict[a]["estimates"][methodN])),resultsDict[a]["estimates"][methodN], lw='2.0', color = color)
								
								# needs to be p instead of x - for true planning skill
								# but since different pskills per plot, can't do 
								# ax2.plot(range(len(resultsDict[a]["estimates"][methodN])),[x]*len(resultsDict[a]["estimates"][methodN]), lw='2.0', color = "k")

								ax2.set_xlabel(r'\textbf{Number of observations}')
								ax2.set_ylabel(r'\textbf{Mean PSkill Estimate}')

								# Put a legend to the right of the current axis
								ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))


				ax1.set_xlabel(r'\textbf{Number of observations}')
				ax1.set_ylabel(r'\textbf{Mean XSkill Estimate}')
				plt.margins(0.05)
				
				fig.suptitle(r'\textbf{Method: ' + str(method) + ' | Agent Type: ' + at + " | Noise: " + str(ax))

				# Put a legend to the right of the current axis
				ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

				# Save png
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "estimateAllAgentsSamePlotPerMethodAndPerAgentType" + os.path.sep + "results-Method" + method + "-AgentType" + at + "-Noise" + str(ax) + ".png", bbox_inches='tight')
				

				plt.clf()
				plt.close()
	#######################################################################################


###################################### FOR BETAS #####################################

def plotEstimateAllBetasSamePlotPerAgentBAR(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder,betas):

	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "estimateAllBetasSamePlotPerAgent-Bar" + os.path.sep 
	makeFolder3(saveAt)

	for a in resultsDict.keys():

		for tt in typeTargetsList:

			for plottingMethod in ["BM-MAP","BM-EES"]:

				try:
			
					makeFolder3(saveAt+plottingMethod)

					tempInfo = []

					for beta in betas:
						tempInfo.append(resultsDict[a]["estimates"][tt][plottingMethod][beta][-1])

					numObs = len(resultsDict[a]["estimates"][tt][plottingMethod][beta])

					xskill = float(a.split("-X")[1].split("-")[0])

					fig = plt.figure(figsize = (20,10))
					ax = plt.subplot(111)

					plt.bar(range(len(betas)),tempInfo,width=0.4,tick_label=betas)
					plt.plot(range(len(betas)),[xskill]*len(betas),lw=2.0,c="black")
					
					plt.xlabel(r'\textbf{Betas}')
					plt.ylabel(r'\textbf{Mean xskill estimate after '+str(numObs)+' observations}')
					plt.margins(0.05)
					plt.title(plottingMethod + " - " + tt + '- Agent: ' + a + ' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

					# Put a legend to the right of the current axis
					ax.tick_params(axis='x', labelrotation = 90)

					#matplotlib.rc('xtick', labelsize=20) 
					#matplotlib.rc('ytick', labelsize=20) 

					fileName = "results-"+tt+"-"+plottingMethod+"-Agent"+a
					plt.savefig(saveAt + os.path.sep + plottingMethod + os.path.sep + fileName + ".png", bbox_inches='tight')

					plt.clf()
					plt.close()
				
				except:
					continue
	#code.interact("after...", local=dict(globals(), **locals()))
	#######################################################################################

def plotEstimateAllBetasSamePlotPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder):

	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "estimateAllBetasSamePlotPerAgent" + os.path.sep 
	makeFolder3(saveAt)
	# makeFolder3(saveAt+"Plotly")

	for a in resultsDict.keys():


		for xType in ["Log","NotLog"]:
		
			makeFolder3(saveAt+xType)
			# makeFolder3(saveAt+"Plotly"+os.path.sep+xType)
			
			for plottingMethod in ["BM-MAP","BM-EES"]:

				makeFolder3(saveAt+xType+os.path.sep+plottingMethod)
				# makeFolder3(saveAt+"Plotly"+os.path.sep+xType+os.path.sep+plottingMethod)
				
				for tt in typeTargetsList:

					fig = plt.figure(figsize = (20,30))
					ax = plt.subplot(111)

					for method in methods:

						if plottingMethod not in method:
							continue

						if tt not in method:
							continue

						tempM, beta, ttTemp = getInfoBM(method)

						if xType == "Log":
							plt.semilogx(range(len(resultsDict[a]["estimates"][tt][tempM][beta])),resultsDict[a]["estimates"][tt][tempM][beta], lw='2.0', label = beta)
						else:
							plt.plot(range(len(resultsDict[a]["estimates"][tt][tempM][beta])),resultsDict[a]["estimates"][tt][tempM][beta], lw='2.0', label = beta)
						
					plt.xlabel(r'\textbf{Number of observations}')
					plt.ylabel(r'\textbf{Mean xskill estimate}')
					plt.margins(0.05)
					plt.title(xType + " - " + plottingMethod + " - " + tt + '- Agent: ' + a + ' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

					# Put a legend to the right of the current axis
					ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

					fileName = "results-"+xType+"-"+tt+"-"+plottingMethod+"-Agent"+a
					plt.savefig(saveAt + xType + os.path.sep + plottingMethod + os.path.sep + fileName + ".png", bbox_inches='tight')
					plt.clf()
					plt.close()

				# ~~~~~ PLOTLY ~~~~~

				# # Remove legend
				# ax.get_legend().remove()

				# # Re-do axis labels
				# ax.set_xlabel('<b>Number of observations</b>',fontsize=18)
				# ax.set_ylabel('<b>Mean Xskill estimate</b>', fontsize=18)

				# # Create Plotly Plot -- Hosting offline
				# plotly_fig =  px.plot_mpl(fig,resize=True)
				# plotly_fig['layout']['showlegend'] = True   
				# plotly_fig['layout']['autosize'] = True  
				# plotly_fig['layout']['height'] *= .80
				# plotly_fig['layout']['width'] *= .80
				# plotly_fig['layout']['margin']['t'] = 50
				# plotly_fig['layout']['margin']['l'] = 0
				# plotly_fig['layout']['title'] = "<b>" + xType + " - " + plottingMethod + '- Agent: ' + a + ' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)</b>'

				# # Save plotly
				# unique_url = px.offline.plot(plotly_fig, filename = saveAt + "Plotly" + os.path.sep + xType + os.path.sep + plottingMethod + os.path.sep + fileName + ".html", auto_open=False)

				plt.clf()
				plt.close()
	#######################################################################################

###################################################################################


def plotEstimateAllMethodsSamePlotPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, givenBeta):

	makeFolder(resultsFolder, "estimateAllMethodsSamePlotPerAgent")

	for a in resultsDict.keys():

		fig = plt.figure(figsize = (10,10))

		##################################### FOR XSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		ax1 = plt.subplot(2, 1, 1)

		for method in methods:

			if "tn" in method:
				continue

			# only xskill methods
			if "pSkills" not in method: 

				if "BM" in method and str(givenBeta) in method:  
					tempM, beta, tt = getInfoBM(method)
					plt.semilogx(range(len(resultsDict[a]["estimates"][tt][tempM][givenBeta])),resultsDict[a]["estimates"][tt][tempM][givenBeta], lw='2.0', label = method)
				elif "BM" not in method:
					plt.semilogx(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)
					#plt.plot(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)

		ax1.set_xlabel(r'\textbf{Number of observations}')
		ax1.set_ylabel(r'\textbf{Mean Xskill estimate}')
		plt.margins(0.05)
		
		fig.suptitle(r'\textbf{Agent: ' + a + ' | '+str(resultsDict[a]["num_exps"]) + ' experiments}')

		# Put a legend to the right of the current axis
		ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))


		# to add space between subplots
		plt.subplots_adjust(hspace = 0.3)

		# Adds "nothing" to the plot 
		# Done in order to add an empty label to the legend 
		# So that there can be a space between the xskill elements & the pskill elements
		plt.plot(np.NaN, np.NaN, '-', alpha = 0.0, label=" ")



		##################################### FOR PSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		ax2 = plt.subplot(2, 1, 2)

		for method in methods:

			# skip TN method
			if method == "tn":
				continue

			# only pskill methods
			if "pSkills" in method:      
				plt.semilogx(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)
				#plt.plot(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)

		ax2.set_xlabel(r'\textbf{Number of observations}')
		ax2.set_ylabel(r'\textbf{Mean Pskill estimate}')
		plt.margins(0.05)

		# Put a legend to the right of the current axis
		ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		# Save png
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "estimateAllMethodsSamePlotPerAgent" + os.path.sep + "results-Agent"+a+".png", bbox_inches='tight')
		

		# # ~~~~~ PLOTLY ~~~~~

		# # Remove legend
		# ax1.get_legend().remove()
		# ax2.get_legend().remove()


		# # Re-do axis labels
		# ax1.set_xlabel('<b>Number of observations</b>',fontsize=18)
		# ax1.set_ylabel('<b>Mean Xskill estimate</b>', fontsize=18)
		# ax2.set_xlabel('<b>Number of observations</b>',fontsize=18)
		# ax2.set_ylabel('<b>Mean Pskill estimate</b>', fontsize=18)


		# # Create Plotly Plot -- Hosting offline
		# plotly_fig =  px.plot_mpl(fig)
		# plotly_fig['layout']['showlegend'] = True   
		# plotly_fig['layout']['autosize'] = True  
		# plotly_fig['layout']['title'] = '<b>Agent: ' + a + ' | '+str(resultsDict[a]["num_exps"]) + ' experiments</b>'

		# # Save plotly
		# unique_url = px.offline.plot(plotly_fig, filename=resultsFolder + os.path.sep + "plots" + os.path.sep + \
		# 										 "estimateAllMethodsSamePlotPerAgent" + os.path.sep + "results-Agent"+a+".html", auto_open=False)


		plt.clf()
		plt.close()
	#######################################################################################

def plotEstimateAllMethodsPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, givenBeta):

	makeFolder(resultsFolder, "estimateAllXSkillMethodsPerAgent")
	makeFolder(resultsFolder, "estimateAllPSkillMethodsPerAgent")

	##################################### FOR XSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs
	for a in resultsDict.keys():

		fig = plt.figure()
		ax = plt.subplot(111)

		for method in methods:

			if "tn" in method:
				continue

			if "pSkills" not in method:

				if "BM" in method and str(givenBeta) in method:  
					tempM, beta, tt = getInfoBM(method)
					plt.semilogx(range(len(resultsDict[a]["estimates"][tt][tempM][givenBeta])),resultsDict[a]["estimates"][tt][tempM][givenBeta], lw='2.0', label = tempM + "-" + tt)
				elif "BM" not in method:
					plt.semilogx(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)
					#plt.plot(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)

		plt.xlabel(r'\textbf{Number of observations}')
		plt.ylabel(r'\textbf{Mean xskill estimate}')
		plt.margins(0.05)
		plt.title('Agent: ' + a + '| All Methods | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "estimateAllXSkillMethodsPerAgent" + os.path.sep + "results-Agent"+a+".png", bbox_inches='tight')
		plt.clf()
		plt.close()
	#######################################################################################

	##################################### FOR PSKILLS #####################################
	# create plot for each one of the different agents - estimates vs obs
	for a in resultsDict.keys():
			
		fig = plt.figure()
		ax = plt.subplot(111)

		for method in methods:

			# skip TN method
			if method == "tn":
				continue

			if "pSkills" in method:      
				plt.semilogx(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)
				#plt.plot(range(len(resultsDict[a]["estimates"][method])),resultsDict[a]["estimates"][method], lw='2.0', label = method)

		plt.xlabel(r'\textbf{Number of observations}')
		plt.ylabel(r'\textbf{Mean pskill estimate}')
		plt.margins(0.05)
		plt.title('Agent: ' + a + '| All Methods | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "estimateAllPSkillMethodsPerAgent" + os.path.sep + "results-Agent"+a+".png", bbox_inches='tight')
		plt.clf()
		plt.close()
	#######################################################################################

def computeMeanEstimates(resultsDict):

	#Compute Mean of XSkill estimate across experiments 
	for a in resultsDict.keys():
		#print "\nAgent: ", a
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

###################################################################################


###################################### FOR MSE #####################################

def computeAndPlotMSEPercentPerPsAndMethodsAndAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, functionsPerAgentType, domain):


	makeFolder(resultsFolder, "msePercent-PerPs-PerAgentType-PerPskillMethods")

	# init dict to store info
	percentRewardObtainedPerAgentType = {}

	for at in seenAgents:
		if at != "Target" and at != "Random":
			percentRewardObtainedPerAgentType[at] = {"perMethod": {}}

			for m in actualMethodsOnExps:

				if m == "tn" or "xSkills" in m:
					continue

				percentRewardObtainedPerAgentType[at]["perMethod"][m] = {}

   
	# convert trueP & estimatedP (per method & per state) to it's corresponding % of TR

	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		# trueP = find agent's true pskill
		trueP = p

		percent_trueP = None

		if aType == "Bounded":
			# #func(sortedPs[i],params[0],params[1])
			percent_trueP = functionsPerAgentType[aType + "-" + domain]["function"](trueP,functionsPerAgentType[aType + "-" + domain]["params"][0],functionsPerAgentType[aType + "-" + domain]["params"][1])
		
		else:
			# polynomial fit
			#poly(sortedPs[i])
			percent_trueP = functionsPerAgentType[aType + "-" + domain]["function"](trueP)
		

		# for each method
		for m in actualMethodsOnExps:

			if m == "tn" or "xSkills" in m:
				continue

			# Skip OR & TBA
			if "pSkills" not in m:
				continue

			if trueP not in percentRewardObtainedPerAgentType[aType]["perMethod"][m].keys():
				percentRewardObtainedPerAgentType[aType]["perMethod"][m][trueP] = { "numAgents": 0.0, "se": [0.0] * numStates}


			if "JT-QRE" in m:
				# method = "JT-QRE"
				aFunc = "Bounded"
			elif "JT-FLIP" in m:
				# method = "JT-FLIP"
				aFunc = "Flip"
			elif "NJT-QRE" in m:
				# method = "NJT-QRE"
				aFunc = "Bounded"


			# update agent count
			percentRewardObtainedPerAgentType[aType]["perMethod"][m][trueP]["numAgents"] += resultsDict[a]["num_exps"]


			# for each state
			for mxi in range(numStates):

				# Get pskill estimate of current method - estimatedP
				estimatedP = resultsDict[a]["estimates"][m][mxi]
				#print estimatedP


				# percentTR_estimatedP = computePercentTRGivenP(bucketsPskillsForMethods, bucketsPercentsForMethods, estimatedP)

				if aFunc == "Bounded":
					# #func(sortedPs[i],params[0],params[1])
					percent_estimatedP = functionsPerAgentType[aFunc + "-" + domain]["function"](estimatedP, functionsPerAgentType[aFunc + "-" + domain]["params"][0],functionsPerAgentType[aFunc + "-" + domain]["params"][1])
				
				else:
					# polynomial fit
					#poly(sortedPs[i])
					percent_estimatedP = functionsPerAgentType[aFunc + "-" + domain]["function"](estimatedP)
				   

				#if percent_trueP == None or percentTR_estimatedP == None:
				#    code.interact("none found", local=locals())

				# compute squared error =  (estimatedP - trueP) ** 2
				sq = (percent_trueP - percent_estimatedP) ** 2

				# store squared error
				percentRewardObtainedPerAgentType[aType]["perMethod"][m][trueP]["se"][mxi] += sq
				

	#code.interact("sq", local=locals())

	# normalize - find Mean Squared Error

	# for each agent type
	for at in seenAgents:
		if "Target" not in at and "Random" not in at:

			# for each method
			for m in actualMethodsOnExps:

				if m == "tn" or "xSkills" in m:
					continue

				# Skip OR & TBA
				if "pSkills" not in m:
					continue

				# for each p
				for ps in percentRewardObtainedPerAgentType[at]["perMethod"][m].keys():
					
					# for each state
					for mxi in range(numStates):

						#find MSE  = SE / # of agents
						percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["se"][mxi] /=  percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["numAgents"] 


	#code.interact("mse", local=locals())
	# '''


	# Plot - for MSE

	# for each agent type
	for at in seenAgents:

		if "Target" not in at and "Random" not in at:

			# for each method
			for m in actualMethodsOnExps:

				if m == "tn" or "xSkills" in m:
					continue

				# Skip OR & TBA
				if "pSkills" not in m:
					continue

				fig = plt.figure()
				ax = plt.subplot(111)

				#if "BM" in m: # 'BM-MAP', 'BM-EES' to TBA
				#    m = m.replace("BM","TBA")

				params = sorted(percentRewardObtainedPerAgentType[at]["perMethod"][m].keys())

				for ps in params:

					#plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label= str(m))
					plt.plot(range(len(percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["se"])),percentRewardObtainedPerAgentType[at]["perMethod"][m][ps]["se"], lw=2.0, label= str(ps))

				plt.xlabel(r'\textbf{Number of observations}',fontsize=18)
				plt.ylabel(r'\textbf{Mean squared error}', fontsize=18)
				plt.margins(0.05)
				plt.title('Agent: ' + at + ' | MSE % of TR')

				# Shrink current axis by 10%
				box = ax.get_position()
				ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

				# Put a legend to the right of the current axis
				ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})
				
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePercent-PerPs-PerAgentType-PerPskillMethods" + os.path.sep +  "results-Agent"+at+"-Method"+m+".png", bbox_inches='tight')

				plt.clf()

				plt.close()

# Computes MSE for rationality percentages (mse_percent_pskills)
def computeMSEPercentPskillsMethods(resultsDict, actualMethodsOnExps, pconfPerXskill, numStates, numHypsX, numHypsP, domain):

	# NOTE: Bounded agent still needs conversion process!
	#		Estimate is already in lambda terms
	#		But multiple lambdas can mean the same rationality percentage.

	for a in resultsDict.keys():

		aType, xStr, p = getParamsFromAgentName(a)


		###################################################################################
		# Find "right" answer - true percent for trueX
		# Bounded agents -> percentTrueP = actual pskill
		# Other agents -> percentTrueP = estimate converted to percentage terms
		percent_trueP = resultsDict[a]["percentTrueP"]
		###################################################################################


		###################################################################################
		# ANOTHER WAY - Find "right" answer - true percent for trueX
		###################################################################################

		'''
		xStr = float(xStr)

		# use estimated xskill and not actual true one
		#xStr = float(resultsDict[a]["plot_y"][mm][numStates-1])

		bucket1, bucket2 = getBucket(bucketsX,minMaxX,xStr)


		if bucket2 != None:
			prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
			prat2 = np.interp(p,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

			prat = np.interp(p, [prat1], [prat2])
			percent_trueP_2 = prat
		# edges/extremes case
		else:
			# using one of the functions for now
			prat1 = np.interp(p,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

			percent_trueP_2 = prat1


		resultsDict[a]["percentTrueP_2"] = percent_trueP_2
		'''
		###################################################################################


		# For each method
		for m in actualMethodsOnExps:

			if m == "tn" or "xSkills" in m:
				continue

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


		'''
		if "Tricker" in a and "150" in a:
			code.interact("Tricker X = 150...", local=dict(globals(), **locals()))
		
		if "Bounded" in a and "2.5" in a:
			code.interact("Bounded X = 2.5...", local=dict(globals(), **locals()))
		'''


def plotScatterPercentsPerAgentTypesAndXskillBuckets(resultsDict, actualMethodsOnExps, resultsFolder, domain, numStates, pconfPerXskill, seenAgents):

	makeFolder(resultsFolder, "scatterPercents-PerAgentType-PskillMethodsAndXskillBuckets")


	#bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		bucketsX = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif domain == "2d" or domain == "sequentialDarts":
		bucketsX = [10, 25, 50, 75, 100, 125, 150]



	# init dict to store info
	percentRewardObtainedPerAgentType = {}

	for at in seenAgents:
		percentRewardObtainedPerAgentType[at] = {}


		for m in actualMethodsOnExps:
			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			percentRewardObtainedPerAgentType[at][m] = {}

			for b in bucketsX:
				percentRewardObtainedPerAgentType[at][m][b] = {"truePercents": [], "estimatedPercents": [], "trueXskill": []}


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		# find proper bucket for current x
		for b in range(len(bucketsX)):
			if x <= bucketsX[b]:
				break

		# get actual bucket
		bucket = bucketsX[b]


		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			percentRewardObtainedPerAgentType[aType][m][bucket]["truePercents"].append(resultsDict[a]["percentTrueP"])
			percentRewardObtainedPerAgentType[aType][m][bucket]["estimatedPercents"].append(resultsDict[a]["percentsEstimatedPs"][m]["averaged"][numStates-1])
			percentRewardObtainedPerAgentType[aType][m][bucket]["trueXskill"].append(x)

	# PLOT

	# for each agent type
	for at in seenAgents:

		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			for b in bucketsX:

				fig = plt.figure()
				# ax = plt.subplot(111)


				cmap = plt.get_cmap("viridis")
				norm = plt.Normalize(min(bucketsX), max(bucketsX))

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm)


				plt.scatter(percentRewardObtainedPerAgentType[at][m][b]["truePercents"],\
							percentRewardObtainedPerAgentType[at][m][b]["estimatedPercents"],
							c = sm.to_rgba(np.asarray(percentRewardObtainedPerAgentType[at][m][b]["trueXskill"])))


				plt.xlabel("True Percent")
				plt.ylabel("Estimated Percent")
				plt.title("xSkill Bucket" + str(b))
				plt.margins(0.05)
				
				# plt.show()
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "scatterPercents-PerAgentType-PskillMethodsAndXskillBuckets" + os.path.sep +  "scatter-Agent"+at+"-Method"+m+"-XBucket"+str(b)+".png", bbox_inches='tight')

				plt.clf()
				plt.close()

	# code.interact("", local = locals())

def plotScatterPercentsPerAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, domain, numStates):

	makeFolder(resultsFolder, "scatterPercents-PerAgentType-PskillMethods")

	# init dict to store info
	percentRewardObtainedPerAgentType = {}

	for at in seenAgents:
		percentRewardObtainedPerAgentType[at] = {}


		for m in actualMethodsOnExps:
			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			percentRewardObtainedPerAgentType[at][m] = {"truePercents": [], "estimatedPercents": [], "trueXskill": []}


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			percentRewardObtainedPerAgentType[aType][m]["truePercents"].append(resultsDict[a]["percentTrueP"])
			percentRewardObtainedPerAgentType[aType][m]["estimatedPercents"].append(resultsDict[a]["percentsEstimatedPs"][m]["averaged"][numStates-1])
			percentRewardObtainedPerAgentType[aType][m]["trueXskill"].append(x)

	# code.interact("", local = locals())
	# PLOT

	# for each agent type
	for at in seenAgents:

		# for each method
		for m in actualMethodsOnExps:

			fig = plt.figure()
			ax = plt.subplot(111)
			
			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			plt.scatter(percentRewardObtainedPerAgentType[at][m]["truePercents"],\
						percentRewardObtainedPerAgentType[at][m]["estimatedPercents"],
						c = np.asarray(percentRewardObtainedPerAgentType[at][m]["trueXskill"]))

			plt.colorbar()

			plt.xlabel("True Percent")
			plt.ylabel("Estimated Percent")
			plt.margins(0.05)
			
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "scatterPercents-PerAgentType-PskillMethods" + os.path.sep +  "scatter-Agent"+at+"-Method"+m+".png", bbox_inches='tight')

			plt.clf()
			plt.close(fig)

def plotMSEPercentPerAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, domain):

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


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		# update agent count
		percentRewardObtainedPerAgentType[aType]["numAgents"] += resultsDict[a]["num_exps"]
		
		# for each method
		for m in actualMethodsOnExps:

			# Skip OR & TBA & TN & xskill methods
			if "-pSkills" not in m:
				continue

			# for each state
			for mxi in range(numStates):

				sq = resultsDict[a]["mse_percent_pskills"][m][mxi]

				# store squared error
				percentRewardObtainedPerAgentType[aType]["perMethod"][m][mxi] += sq
				

	# normalize - find Mean Squared Error

	# for each agent type
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

	# for each agent type
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
		
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePercent-PerAgentType-PskillMethods" + os.path.sep +  "results-Agent"+at+".png", bbox_inches='tight')

		plt.clf()
		plt.close()

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
		
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePercent-PerAgentType-PskillMethods" + os.path.sep +  "results-LOG-Agent"+at+".png", bbox_inches='tight')

		plt.clf()
		plt.close()

		############################################################################################################################

def plotMSExSkillsPerBucketsPerAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, domain, numStates, numHypsX, numHypsP):

	# BUCKETS PER PERCENTS RAND/MAX REWARD -- SHOWING MSE FOR XSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-pSkills"

	makeFolder(resultsFolder, "mseXskills-PerBucketsPerAgentType")

	# buckets = [25,50,75,100]

	# Buckets in percents terms - between 0-1
	buckets = [0.25,0.50,0.75,1.0]


	# init dict to store info
	mseDict = {}

	stdInfoPerAgentTypePerMethod = {}
	stdPerAgentTypePerMethod = {}
	confidenceIntervals = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": 0.0}

		stdInfoPerAgentTypePerMethod[at] = {}
		stdPerAgentTypePerMethod[at] = {}
		confidenceIntervals[at] = {}


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


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		# update agent count
		mseDict[aType]["numAgents"] += resultsDict[a]["num_exps"]
		

		#estimatedP = resultsDict[a]["mse_percent_pskills"][method][numStates-1] # #### ESTIMATED %

		trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
		# using true percent and not estimated one

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break


		# get actual bucket
		b = buckets[b]


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


		# if aType == "Target":
			# code.interact(local=locals())


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
					mseDict[at]["perMethod"][m][b][mxi] /= mseDict[at]["numAgents"]


	#####################################################################################################
	# get data for standard deviation across all agents of same type --- for last state
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

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


	# compute actual std
	for at in seenAgents:
		for m in actualMethodsOnExps:

			#Skip pskill methods
			if "-pSkills" in m:
				continue

			if "tn" in m:
				continue

			for b in buckets:
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
				mu = mseDict[at]["perMethod"][m][b][-1]
				sigma = stdPerAgentTypePerMethod[at][m][b]
				N = mseDict[at]["numAgents"]


				confidenceIntervals[at][m][b]["low"], confidenceIntervals[at][m][b]["high"] =\
				stats.norm.interval(ci, loc=mu, scale=sigma/np.sqrt(N))

				# for 95% interval
				Z = 1.960

				confidenceIntervals[at][m][b]["value"] = Z * (sigma/np.sqrt(N))

	#####################################################################################################

	xskillsCI = open(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseXskills-PerBucketsPerAgentType" + os.path.sep + "confidenceIntervals-xSkills.txt", "a")

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
							" ->  Low: " + str(round(confidenceIntervals[at][m][b]["low"],4)) +\
							" | High: " + str(round(confidenceIntervals[at][m][b]["high"],4)) +\
							" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
							" | Value: " + str(round(confidenceIntervals[at][m][b]["value"],4)) + "\n")

					d_x["Agents"].append(at)
					d_x["Bucket"].append(b)
					d_x["MSE"].append(round(orderedMSE[mseO],2))
					d_x["Low"].append(round(confidenceIntervals[at][m][b]["low"],2))
					d_x["High"].append(round(confidenceIntervals[at][m][b]["high"],2))
					d_x["Values"].append(round(confidenceIntervals[at][m][b]["value"],2))


					xskillsCI.write("\n")

				xskillsCI.write("\n")

				 # Convert dicts to pandas dataframe
				d_x_pd = pd.DataFrame(d_x, columns = ["Agents", "Bucket", "Low", "MSE",  "High", "Values"])

				xskillsCI.write(d_x_pd.to_latex(index=False))
				
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
					plt.semilogx(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = colors[c])
				c += 1


			plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
			plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

			plt.margins(0.05)
			# plt.suptitle('Agent: ' + at + ' | MSE of Xskill Methods')

			fig.subplots_adjust(hspace= 1.0, wspace=1.0)

			# Shrink current axis by 10%
			# box = ax.get_position()
			# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

			elements = [Line2D([0],[0], color = colors[0],label = buckets[0]),
						Line2D([0],[0], color = colors[1], label = buckets[1]),
						Line2D([0],[0], color = colors[2], label = buckets[2]),
						Line2D([0],[0], color = colors[3], label = buckets[3])]
			
			# Put a legend to the right of the current axis
			plt.legend(handles = elements)#, loc='center left', bbox_to_anchor=(1, 0.5))

			# plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseXskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".png", bbox_inches='tight')
			# plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseXskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".pdf", bbox_inches='tight')
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseXskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".png", bbox_inches='tight')
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseXskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".pdf", bbox_inches='tight')

			plt.clf()
			plt.close()

def plotMSEpSkillsPerBucketsPerAgentTypes(resultsDict, actualMethodsOnExps, resultsFolder, domain, numStates, numHypsX, numHypsP):

	# BUCKETS PER XSKILLS -- SHOWING MSE FOR PSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0])+ "-xSkills"

	makeFolder(resultsFolder, "msePskills-PerBucketsPerAgentType")

	if domain == "1d":
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]

	# init dict to store info
	mseDict = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": 0.0}


		for m in actualMethodsOnExps:
			# Skip pskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			mseDict[at]["perMethod"][m] = {}

			for b in buckets:
				mseDict[at]["perMethod"][m][b] = [0.0] * numStates # to store per state - across different exps per agent type


	for a in resultsDict.keys():

		aType, X, p = getParamsFromAgentName(a)


		# update agent count
		mseDict[aType]["numAgents"] += resultsDict[a]["num_exps"]
		

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if X <= buckets[b]:
				break

		# get actual bucket
		b = buckets[b]


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
					mseDict[at]["perMethod"][m][b][mxi] /= mseDict[at]["numAgents"]


	# Plot - for MSE

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	# colors = ["red", "green", "blue", "orange"]

	# for each agent type
	for at in seenAgents:


		# for each method
		for m in actualMethodsOnExps:
			

			# Skip pxskill methods
			if "-pSkills" not in m:
				continue

			if "tn" in m:
				continue

			
			fig = plt.figure(figsize = (10,10))
			ax1 = plt.subplot(2, 1, 1)


			# ax = fig.add_subplot(1, 2, i)
			# ax.title.set_text('Method: ' + m)
			
			c = 0
			for b in buckets:
				if np.count_nonzero(mseDict[at]["perMethod"][m][b]) != 0:
					# print "b: ", b, "| color: ", colors[c] 
					color = colors[list(colors.keys())[c]]
					# plt.plot(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
					plt.semilogx(range(numStates),mseDict[at]["perMethod"][m][b], lw=2.0, label= str(b), c = color)
				c += 1

			plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
			plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

			plt.margins(0.05)
			# plt.suptitle('Agent: ' + at + ' | MSE of Pskills Methods')

			fig.subplots_adjust(hspace= 1.0, wspace=1.0)

			# Shrink current axis by 10%
			# box = ax.get_position()
			# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

			elements = []

			for i in range(len(buckets)):
				elements.append(Line2D([0],[0], color = colors[list(colors.keys())[i]] ,label = buckets[i]))

				
			# Put a legend to the right of the current axis
			plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))

			# plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".png", bbox_inches='tight')
			# plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+methodNamesPaper[m]+"-"+domain+".pdf", bbox_inches='tight')
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".png", bbox_inches='tight')
			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePskills-PerBucketsPerAgentType" + os.path.sep +  "results-Agent"+at+"-Method"+m+"-"+domain+".pdf", bbox_inches='tight')

			plt.clf()
			plt.close()


def diffPlotsForMSExSkillpSkillPerAgentTypePerMethod(resultsFolder, agentTypes, methods, domain, numStates, info, xskillBuckets, pskillBuckets):

	# This method gets called from plotContourMSE_xSkillpSkillPerAgentTypePerMethod() to make use of the already computed info

	makeFolder(resultsFolder, "MSExSkillPSkill-PerAgentTypePerMethod")


	prop_cycle = plt.rcParams["axes.prop_cycle"]
	colors = prop_cycle.by_key()['color']


	####################################################################################################
	# find mid point of buckets (in order to better visualize the regions in the plots)
	####################################################################################################
	# find middle point - xskills
	midX = int(len(xskillBuckets)/2)

	# find middle point - pskills
	midP = int(len(pskillBuckets)/2)
	####################################################################################################


	# Plot per method and per agent type
	for at in agentTypes:

		if "Random" in at:
			continue


		for m in methods:

			if "tn" not in m:

				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

				N = len(info[at]["method"][m]["x"])


				####################################################################################################
				# Plot - MSE vs xskills - ALL
				####################################################################################################

				fig = plt.figure()

				
				for i in range(N):
					plt.plot(info[at]["method"][m]["x"][i], info[at]["method"][m]["mse"][i], marker='o', c = colors[info[at]["method"][m]["pBuckets"][i]])


				elements = []

				for i in range(len(info[at]["method"][m]["pskillBuckets"])):
					elements.append(Line2D([0],[0], color = colors[i], label = str(round(info[at]["method"][m]["pskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
	  

				plt.xlabel("xSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsXskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-ALL.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################


				####################################################################################################
				# Plot - MSE vs xskills - First Half
				####################################################################################################

				fig = plt.figure()

				
				for i in range(N):

					# if the data point belongs to the first part of the buckets (pskill located on first region of buckets)
					if info[at]["method"][m]["pBuckets"][i] < midP:
						plt.plot(info[at]["method"][m]["x"][i], info[at]["method"][m]["mse"][i], marker='o', c = colors[info[at]["method"][m]["pBuckets"][i]])


				elements = []

				for i in range(0, midP):
					elements.append(Line2D([0],[0], color = colors[i], label = str(round(info[at]["method"][m]["pskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
				

				plt.xlabel("xSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsXskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-FirstHalf.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################


				####################################################################################################
				# Plot - MSE vs xskills - Second Half
				####################################################################################################

				fig = plt.figure()

				
				for i in range(N):

					# if the data point belongs to the second part of the buckets (pskill located on second region of buckets)
					if info[at]["method"][m]["pBuckets"][i] >= midP:
						plt.plot(info[at]["method"][m]["x"][i], info[at]["method"][m]["mse"][i], marker='o', c = colors[info[at]["method"][m]["pBuckets"][i]])


				elements = []

				for i in range(midP, len(info[at]["method"][m]["pskillBuckets"])):
					elements.append(Line2D([0],[0], color = colors[i], label = str(round(info[at]["method"][m]["pskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
				

				plt.xlabel("xSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m + "")
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsXskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-SecondHalf.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################

				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#




				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

				####################################################################################################
				# Plot - MSE vs pskills - ALL
				####################################################################################################

				fig = plt.figure()


				for i in range(N):
					plt.plot(info[at]["method"][m]["p"][i], info[at]["method"][m]["mse"][i],  marker='o', c = colors[info[at]["method"][m]["xBuckets"][i]])


				elements = []

				for i in range(len(info[at]["method"][m]["xskillBuckets"])):
					elements.append(Line2D([0],[0], color = colors[i],label = str(round(info[at]["method"][m]["xskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
				

				plt.xlabel("pSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsPskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-ALL.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################


				####################################################################################################
				# Plot - MSE vs pskills - First Half
				####################################################################################################

				fig = plt.figure()


				for i in range(N):

					# if the data point belongs to the first part of the buckets (xskill located on first region of buckets)
					if info[at]["method"][m]["xBuckets"][i] < midX:
						plt.plot(info[at]["method"][m]["p"][i], info[at]["method"][m]["mse"][i],  marker='o', c = colors[info[at]["method"][m]["xBuckets"][i]])


				elements = []

				for i in range(0, midX):
					elements.append(Line2D([0],[0], color = colors[i],label = str(round(info[at]["method"][m]["xskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
				

				plt.xlabel("pSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsPskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-FirstHalf.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################


				####################################################################################################
				# Plot - MSE vs pskills - Second Half
				####################################################################################################

				fig = plt.figure()


				for i in range(N):

					# if the data point belongs to the second part of the buckets (xskill located on second region of buckets)
					if info[at]["method"][m]["xBuckets"][i] >= midX:
						plt.plot(info[at]["method"][m]["p"][i], info[at]["method"][m]["mse"][i],  marker='o', c = colors[info[at]["method"][m]["xBuckets"][i]])


				elements = []

				for i in range(midX,len(info[at]["method"][m]["xskillBuckets"])):
					elements.append(Line2D([0],[0], color = colors[i],label = str(round(info[at]["method"][m]["xskillBuckets"][i],2))))


				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))
				

				plt.xlabel("pSkills")
				plt.ylabel("MSE")
				plt.title("Agent: " + at + " | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSExSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "plot-mseVsPskills-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+"-SecondHalf.png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################

				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

def createContourMSECategories_xSkillpSkill(resultsFolder, resultsDict, agentTypes, methods, xSkills, pSkills, info):

	makeFolder(resultsFolder, "MseCategoriesXSkillPSkill-PerAgentTypePerMethod")


	# loop to create buckets
	for at in info.keys():

		for m in info[at]["method"].keys():


			# find min MSE for given method
			minMSE = min(info[at]["method"][m]["mse"])


			# find max MSE for given method
			maxMSE = max(info[at]["method"][m]["mse"])


			# find step size
			step = (maxMSE - minMSE) / 3

			# create 3 buckets
			info[at]["method"][m]["bucketsCategory"].append(minMSE + step)
			info[at]["method"][m]["bucketsCategory"].append(minMSE + (2 * step))
			info[at]["method"][m]["bucketsCategory"].append(maxMSE)



	# loop to create categories/tags
	for at in info.keys():

		for m in info[at]["method"].keys():

			for e in info[at]["method"][m]["mse"]:

				b = None

				# find bucket
				if e <= info[at]["method"][m]["bucketsCategory"][0]:
					b = 0
				elif e > info[at]["method"][m]["bucketsCategory"][0] and e <= info[at]["method"][m]["bucketsCategory"][1]:
					b = 1
				elif e > info[at]["method"][m]["bucketsCategory"][1]: # and e <= info[at]["method"][m]["bucketsCategory"][2]:
					b = 2

				# if b == None:
				# 	code.interact("none...", local=dict(globals(), **locals()))

				# assign category
				info[at]["method"][m]["mseCategory"].append(b)

	#code.interact("",local=locals())


	# code.interact("",local=locals())
	gx, gy = np.meshgrid(xSkills,pSkills)


	# Plot per method and per agent type
	for at in agentTypes:

		if "Random" in at:
			continue

		for m in methods:

			if "tn" not in m:

				# create plot - contour plot

				N = len(info[at]["method"][m]["x"])

				# to store the different xskills & probs rational 
				POINTS = np.zeros((N,2))
				
				# to store the mean of the observed rewards
				VALUES = np.zeros((N,1))

				for i in range(N):

					POINTS[i][:] = [info[at]["method"][m]["x"][i],info[at]["method"][m]["p"][i]] #[x,p]

					# VALUES[i] = info[at]["method"][m]["mseCategory"][i] #mse Category
					VALUES[i] = info[at]["method"][m]["mse"][i] #mse


				Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'cubic')

				Z = Z[:,:,0]

				# remove inf's -> causes surface plot to be all of the same color 
				Z[Z == np.inf] = np.nan


				fig = plt.figure()
				ax = plt.subplot(111)


				# cmapBig = plt.get_cmap('plasma', 512)

				top = plt.get_cmap('plasma', 16)
				# bottom = plt.get_cmap('plasma', 128)

				# newcolors = np.vstack((top(np.linspace(0, 1, 128)),
									   # bottom(np.linspace(0, 1, 128))))

				newcolors = top(np.linspace(0, 1, 16))
				cmap = matplotlib.colors.ListedColormap(newcolors)


				if "Bounded" in at:
					cs = plt.contourf(gx, np.log10(gy), Z, cmap = cmap)
				else:
					cs = plt.contourf(gx, gy, Z, cmap = cmap)

				fig.colorbar(cs)


				plt.xlabel("Execution Skills")
				plt.ylabel("Planning Skills")
				plt.title("MSE | Agent: " + at +" | Method: " + m)
			

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MseCategoriesXSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "contour-mse-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight') 
				plt.clf()
				plt.close()


				####################################################################################################
				# Scatter plot
				####################################################################################################

				fig = plt.figure()

				# rows, cols, pos
				ax = fig.add_subplot(2, 1, 1)

				#s = plt.scatter(info[at]["method"][m]["x"], np.log10(info[at]["method"][m]["p"]), c = info[at]["method"][m]["mseCategory"], s= 3.0)
				
				colors = ["C0", "C1", "C2"]


				for i in range(N):
					if "Bounded" in at:
						plt.plot(info[at]["method"][m]["x"][i], np.log10(info[at]["method"][m]["p"][i]), color = colors[info[at]["method"][m]["mseCategory"][i]], marker='o', ms= 1.0 + 2*info[at]["method"][m]["mseCategory"][i])
					else:
						plt.plot(info[at]["method"][m]["x"][i], info[at]["method"][m]["p"][i], color = colors[info[at]["method"][m]["mseCategory"][i]], marker='o', ms= 1.0 + 2*info[at]["method"][m]["mseCategory"][i])

				#cmapBig = plt.get_cmap('plasma', 512)
				#cmap = matplotlib.colors.ListedColormap(cmapBig(np.linspace(0.25,0.75,256)))

				#cbar = fig.colorbar()
				#cbar.set_label("MSE Category", labelpad=+1)


				elements = [Line2D([0],[0], color = colors[0],label = str(round(info[at]["method"][m]["bucketsCategory"][0],4))),
							Line2D([0],[0], color = colors[1], label = str(round(info[at]["method"][m]["bucketsCategory"][1],4))),
							Line2D([0],[0], color = colors[2], label = str(round(info[at]["method"][m]["bucketsCategory"][2],4)))]
				
				# Put a legend to the right of the current axis
				plt.legend(handles = elements, loc='center left', bbox_to_anchor=(1, 0.5))


				plt.xlabel("xSkills")
				plt.ylabel("pSkills")
				plt.title("MSE Categories | Agent: " + at +" | Method: " + m)
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MseCategoriesXSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "scatterPlot-mse-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################

def plotContourMSE_xSkillpSkillPerAgentTypePerMethod(resultsDict, agentTypesFull, methods, resultsFolder, numStates, domain):

	makeFolder(resultsFolder, "contourMseXSkillPSkill-PerAgentTypePerMethod")

	agentTypes = deepcopy(seenAgents)

	# Remove target agent since no pskill param and thus not going to plot it
	if "Target" in agentTypesFull:
		agentTypes.remove("Target")

	print("agentTypes: ", agentTypes)


	if domain == "1d":
		pskillBuckets = [10, 30, 50, 70, 100]
	elif domain == "2d" or domain == "sequentialDarts":
		pskillBuckets = [5, 10, 15, 20, 32]


	if domain == "1d":
		xskillBuckets = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
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


	probsRational = None

	# Get info 
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


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
					

	#infoAvg = {}
	# compute avg of the data (per xskill and pskill) --- ???

   # code.interact("???", local=dict(globals(), **locals())) 




	if domain == "1d":
		# Create different execution skill levels 
		xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)
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

				if "-pSkills" in m:
					norm = plt.Normalize(minMseP, maxMseP)
				else:
					norm = plt.Normalize(minMseX, maxMseX)


				s = ax.scatter(info[at]["method"][m]["x"], info[at]["method"][m]["p"], c = cmap(norm(info[at]["method"][m]["mse"])))

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm)
				cbar.set_label("MSE", labelpad=+1)

				ax.set_xlabel("xSkills")
				ax.set_ylabel("pSkills")
				
				plt.margins(0.05)

				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "scatterPlot-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
				plt.clf()
				plt.close()
				####################################################################################################


				N = len(info[at]["method"][m]["x"])
				# code.interact(local=locals())

				# to store the different xskills & probs rational 
				POINTS = np.zeros((N,2))
				
				# to store the mean of the observed rewards
				VALUES = np.zeros((N,1))

				for i in range(N):

					POINTS[i][:] = [info[at]["method"][m]["x"][i],info[at]["method"][m]["p"][i]] #[x,p]

					VALUES[i] = info[at]["method"][m]["mse"][i] #mse



				#if "Bounded" in at:
				#    code.interact("Bounded: ", local=locals())
				'''
				try:
					if "Bounded" in at:
						Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
					else:
						Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')
						#Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
				except:
					code.interact("error: why?", local=locals())

				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'nearest')
				# Z = griddata(POINTS, VALUES, (gx, gy), method = 'cubic')
				'''
				Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')


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
				# Not doing for now in order to see errors properly
				# Will only normalize for the one that shows them all on the same plot
				'''
				if "-pSkills" in m:
					norm = plt.Normalize(minPZ, maxPZ)
				else:
					norm = plt.Normalize(minXZ, maxXZ)
				'''


				gx = info[at]["method"][m]["gx"]
				gy = info[at]["method"][m]["gy"]
				Z = info[at]["method"][m]["Z"]

				norm = plt.Normalize(np.nanmin(Z), np.nanmax(Z))


				if "Bounded" in at:
					cs = plt.contourf(gx, np.log10(gy), Z, norm = norm)
				else:
					cs = plt.contourf(gx, gy, Z, norm = norm)


				#plt.xlabel("Execution Skills")
				#plt.ylabel("Planning Skills")
				plt.xlabel(r"\textbf{Execution Noise Levels}")
				plt.ylabel(r"\textbf{Rationality Parameters}")
				#plt.title("MSE | Domain: " + domain + " | Agent: " + at +" | Method: " + m)

				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm)


				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
				plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".pdf", bbox_inches='tight')
				plt.clf()
				plt.close()



	rows = int(len(methods) / 2)

	# if odd, add +
	if rows % 2 != 0:
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


				if "-pSkills" in m:
					norm = plt.Normalize(minPZ, maxPZ)
				else:
					norm = plt.Normalize(minXZ, maxXZ)


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

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourMseXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-mse-Domain-"+domain+"-Agent-"+at+"-AllMethods.png", bbox_inches='tight')
		plt.clf()
		plt.close()



	##############################################################################################################
	# Calling function(s) here in order to make use of the info that was computed in this function
	##############################################################################################################

	# diffPlotsForMSExSkillpSkillPerAgentTypePerMethod(resultsFolder, agentTypes, methods, domain, numStates, info, xskillBuckets, pskillBuckets)

	# createContourMSECategories_xSkillpSkill(resultsFolder, resultsDict, agentTypes, methods, xSkills, probsRational, info)

def plotContourEstimates_xSkillpSkillPerAgentTypePerMethod(resultsDict, agentTypesFull, methods, resultsFolder, numStates, domain):

		makeFolder(resultsFolder, "contourEstimatesXSkillPSkill-PerAgentTypePerMethod")

		agentTypes = deepcopy(seenAgents)

		# Remove target agent since no pskill param and thus not going to plot it
		if "Target" in agentTypesFull:
			agentTypes.remove("Target")

		print("agentTypes: ", agentTypes)


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


		# Get info 
		for a in resultsDict.keys():

			aType, x, p = getParamsFromAgentName(a)


			if aType == "Target":
				continue
			
			for m in methods:

				if "tn" not in m:
					info[aType]["method"][m]["x"].append(x)
					info[aType]["method"][m]["p"].append(p)


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

	
				
		if domain == "1d":
			# Create different execution skill levels 
			xSkills = np.linspace(0.5, 4.5, num = 100) # (start, stop, num samples)

		elif domain == "2d" or domain == "sequentialDarts":
			# Create different execution skill levels 
			xSkills = np.linspace(2.5, 150.5, num = 100) # (start, stop, num samples)

		   
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


			# gx, gy = np.meshgrid(xSkills,probsRational, indexing = "ij")
			gx, gy = np.meshgrid(xSkills,probsRational)


			for m in methods:

				if "tn" not in m:

					####################################################################################################
					# Scatter plot
					####################################################################################################

					fig = plt.figure()

					# rows, cols, pos
					ax = fig.add_subplot(2, 1, 1)


					cmap = plt.get_cmap("viridis")

					if "-pSkills" in m:
						norm = plt.Normalize(minP, maxP)
					else:
						norm = plt.Normalize(minX, maxX)


					s = ax.scatter(info[at]["method"][m]["x"], info[at]["method"][m]["p"], c = cmap(norm(info[at]["method"][m]["estimates"])))

					sm = ScalarMappable(norm = norm, cmap = cmap)
					sm.set_array([])
					cbar = fig.colorbar(sm)
					cbar.set_label("Estimate", labelpad=+1)

					ax.set_xlabel("xSkills")
					ax.set_ylabel("pSkills")
					
					plt.margins(0.05)

					plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" + os.path.sep + "scatterPlot-Estimate-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches = 'tight')
					plt.clf()
					plt.close()
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


					Z = griddata(POINTS, VALUES, (gx, gy), method = 'linear')


					Z = Z[:,:,0]

					# remove inf's -> causes surface plot to be all of the same color 
					Z[Z == np.inf] = np.nan

					# Save info to use in plot next
					info[at]["method"][m]["POINTS"] = POINTS
					info[at]["method"][m]["VALUES"] = VALUES
					info[at]["method"][m]["Z"] = Z
					info[at]["method"][m]["gx"] = gx
					info[at]["method"][m]["gy"] = gy



		#code.interact("before contours", local=locals())

		# Plotting contours separately in order to have access to the Z info before to normalize colorbar
		for at in agentTypes:

			for m in methods:

				if "tn" not in m:
					fig = plt.figure()
					ax = plt.subplot(111)

					cmap = plt.get_cmap("viridis")

					gx = info[at]["method"][m]["gx"]
					gy = info[at]["method"][m]["gy"]
					Z = info[at]["method"][m]["Z"]

					norm = plt.Normalize(np.nanmin(Z), np.nanmax(Z))


					if "Bounded" in at:
						cs = plt.contourf(gx, np.log10(gy), Z, norm = norm)
					else:
						cs = plt.contourf(gx, gy, Z, norm = norm)


					plt.xlabel(r"\textbf{Execution Skill Noise Levels}")
					plt.ylabel(r"\textbf{Rationality Parameters}")

					sm = ScalarMappable(norm = norm, cmap = cmap)
					sm.set_array([])
					cbar = fig.colorbar(sm)


					plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".png", bbox_inches='tight')
					plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-Method"+str(m)+".pdf", bbox_inches='tight')
					plt.clf()
					plt.close()



		rows = int(len(methods) / 2)

		# if odd, add +
		if rows % 2 != 0:
			rows += 1

		cols = 2

		# Plot all methods together (different subplots) per agent type
		for at in agentTypes:

			if at == "Random":
				continue

			if at == "Target":
				continue


			fig = plt.figure(figsize=(10,15))

			fig.subplots_adjust(hspace=0.8, wspace=0.6)

			fig.suptitle("Estimates | Domain: " + domain + " | Agent: " + at)

			cmap = plt.get_cmap("viridis")

			for mi in range(len(methods)):
				
				m = methods[mi]

				if "tn" not in m:


					if "-pSkills" in m:
						norm = plt.Normalize(minP, maxP)
					else:
						norm = plt.Normalize(minX, maxX)


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

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "contourEstimatesXSkillPSkill-PerAgentTypePerMethod" +os.path.sep + "results-Estimates-Domain-"+domain+"-Agent-"+at+"-AllMethods.png", bbox_inches='tight')
			plt.clf()
			plt.close()


# Verify this plot & fix!
def computeAndPlotMSEAcrossAllAgentsTypesAndRationalityParamsAllMethods(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp, agentTypes, numStates):

	makeFolder(resultsFolder, "MSEAcrossAllAgentsTypesAndRationalityParamsAllMethods")

	# print methodsNames

	# using only pskill methods 
	methodsDict = {"JT-QRE-MAP": "JT-QRE-MAP"+"-"+str(hyp)+"-pSkills","JT-QRE-EES": "JT-QRE-EES"+"-"+str(hyp)+"-pSkills",
					 "JT-FLIP-MAP": "JT-FLIP-MAP"+"-"+str(hyp)+"-pSkills","JT-FLIP-EES": "JT-FLIP-EES"+"-"+str(hyp)+"-pSkills",
					 "NJT-QRE-MAP": "NJT-QRE-MAP"+"-"+str(hyp)+"-pSkills","NJT-QRE-EES": "NJT-QRE-EES"+"-"+str(hyp)+"-pSkills"}


	methodsNamesC = deepcopy(methodsNames)

	if "OR" in methodsNamesC:
		methodsNamesC.remove("OR")

	if "BM-MAP" in methodsNamesC:
		methodsNamesC.remove("BM-MAP")
	
	if "BM-EES" in methodsNamesC:
		methodsNamesC.remove("BM-EES")


	mseAcrossAllAgentsPerAgentTypeAndRP = {}

	# set up dict to save info
	for at in agentTypes:
		mseAcrossAllAgentsPerAgentTypeAndRP[at] = {"numAgents": {}, "totalNumExps": 0.0, "msePerMethod": {}}

	# compute mse across all agents of same type
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)


		for m in methodsNamesC:

			if m not in mseAcrossAllAgentsPerAgentTypeAndRP[aType]["msePerMethod"].keys():
				mseAcrossAllAgentsPerAgentTypeAndRP[aType]["msePerMethod"][m] = {}
				mseAcrossAllAgentsPerAgentTypeAndRP[aType]["numAgents"][m] = {}


			if "JT-QRE" in m:
				buckets = [[0,20], [20,40], [40,60], [60,80], [80,100]]
			elif "JT-FLIP" in m:
				buckets = [[0.0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0]]


			bucketFound = False
			b = 0

			# find bucket of p
			while bucketFound != True:
				#print b
				#print trueP

				if p > buckets[b][0] and p <= buckets[b][1]:
					bucketFound = True
					break
				b += 1


			# init info for bucket if needed
			if str(buckets[b]) not in mseAcrossAllAgentsPerAgentTypeAndRP[aType]["msePerMethod"][m].keys():
				mseAcrossAllAgentsPerAgentTypeAndRP[aType]["msePerMethod"][m][str(buckets[b])] = 0.0
				mseAcrossAllAgentsPerAgentTypeAndRP[aType]["numAgents"][m][str(buckets[b])] = 0.0


			mseAcrossAllAgentsPerAgentTypeAndRP[aType]["msePerMethod"][m][str(buckets[b])] += resultsDict[a]["plot_y"][methodsDict[m]][numStates-1]
			mseAcrossAllAgentsPerAgentTypeAndRP[aType]["numAgents"][m][str(buckets[b])] += 1.0
		
		mseAcrossAllAgentsPerAgentTypeAndRP[aType]["totalNumExps"] += resultsDict[a]["num_exps"]

	# Normalize
	for at in agentTypes:
		if "Target" not in at:

			#print methodsNamesC
			
			for m in methodsNamesC:
				#print mseAcrossAllAgentsPerAgentTypeAndRP[at]["msePerMethod"][m].keys()

				for bu in mseAcrossAllAgentsPerAgentTypeAndRP[at]["msePerMethod"][m].keys():
					# code.interact("before norm", local=locals())
					mseAcrossAllAgentsPerAgentTypeAndRP[at]["msePerMethod"][m][bu] /= (mseAcrossAllAgentsPerAgentTypeAndRP[at]["numAgents"][m][bu] * 1.0)
	# code.interact("after norm", local=locals())


	for at in agentTypes:

		if "Target" not in at:

			fig = plt.figure(figsize = (10,10))

			##################################### FOR PSKILLS #####################################

			ax1 = plt.subplot(2, 1, 1)

			for method in methodsNamesC:

				if "JT-QRE" in m:
					buckets = [[0,20], [20,40], [40,60], [60,80], [80,100]]
				elif "JT-FLIP" in m:
					buckets = [[0.0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], [0.8,0.9], [0.9,1.0]]

				xs = []
				ys = []

				if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
					m = method.replace("BM","TBA")
				else:
					m = method

				if "tn" in method: 
					# don't plot the TN since MSE
					continue
				else:   
					# other methods 

					for bu in buckets:
						# try in case bucket doesn't exists
						try: 
							ys.append(mseAcrossAllAgentsPerAgentTypeAndRP[at]["msePerMethod"][method][str(bu)])

							plt.axvline(x = bu[0], lw='2.0', ls = "--", color = "black", alpha = 0.70)
							plt.axvline(x = bu[1], lw='2.0', ls = "--", color = "black", alpha = 0.70)
						except:
							#continue
							ys.append("")
						
						xs.append((bu[0] + bu[1])/2.0)


					plt.plot(xs,ys, label= str(m))

			ax1.set_xlabel(r'\textbf{Rationality Parameter Buckets}',fontsize=18)
			ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
			plt.margins(0.05)

			# fig.suptitle(r'\textbf{Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments}')

			# Put a legend to the right of the current axis
			ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgentsTypesAndRationalityParamsAllMethods" + os.path.sep + "results-Agent"+at+"-Exps"+str(mseAcrossAllAgentsPerAgentTypeAndRP[at]["totalNumExps"])+".png", bbox_inches='tight')

			plt.clf()
			plt.close()

def computeAndPlotMSEAcrossAllAgentsTypesAllMethodsPercentTR(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp, agentTypes, numStates):

	makeFolder(resultsFolder, "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType-PercentTR")

	mseAcrossAllAgentsPerAgentType = {}

	# set up dict to save info
	for at in agentTypes:
		mseAcrossAllAgentsPerAgentType[at] = {"numAgents": 0.0, "totalNumExps": 0.0}

		for m in methods:
			mseAcrossAllAgentsPerAgentType[at][m] = [0.0] * numStates

	#####################################################################################################
	# compute mse across all agents of same type
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		for m in methods:
			for s in range(numStates):
				mseAcrossAllAgentsPerAgentType[aType][m][s] += resultsDict[a]["plot_y"][m][s]

		mseAcrossAllAgentsPerAgentType[aType]["numAgents"] += 1.0
		mseAcrossAllAgentsPerAgentType[aType]["totalNumExps"] += resultsDict[a]["num_exps"]


	# Normalize MSE    
	for at in agentTypes:
		for m in methods:
			for s in range(numStates):
				# code.interact("before norm", local=locals())
				mseAcrossAllAgentsPerAgentType[at][m][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
				# code.interact("after norm", local=locals())
				
	#####################################################################################################


	for at in agentTypes:

		##################################### FOR PSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		fig = plt.figure(figsize = (10,10))

		ax1 = plt.subplot(2, 1, 1)

		for method in methods:

			# only plotting xskills methods
			if "pSkills" not in method  or "tn" in method:
				continue

			if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
				m = methodsDict[method]
				m = m.replace("BM","TBA")
			else:
				m = methodsDict[method]
  
			# other methods          
			plt.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method], lw='2.0', label= str(m))
			# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))
			# print mseAcrossAllAgentsPerAgentType[at][method]

		ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=18)
		ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)

		#plt.title('PSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

		# Put a legend to the right of the current axis
		ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType-PercentTR" + os.path.sep + "results-PSKILL-Agent"+at+"-Exps"+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"])+".png", bbox_inches='tight')

		plt.clf()
		plt.close()


###################################### FOR BETAS #####################################

def plotMseDiffBetasPerXskillBucketsPerAgentTypes(resultsDict,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas):

	# BUCKETS PER PERCENTS RAND/MAX REWARD -- SHOWING MSE FOR XSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-pSkills"


	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "mseDiffBetas-PerXskillBucketsPerAgentType" + os.path.sep
	makeFolder3(saveAt)
	# makeFolder3(saveAt+"Plotly")

	if domain == "1d":
		buckets = [1.0, 2.0, 3.0, 4.0, 5.0]
	elif domain == "2d" or domain == "sequentialDarts":
		buckets = [5, 10, 30, 50, 70, 90, 110, 130, 150]

	# Init dict to store info
	mseDict = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": 0.0}

		for tt in typeTargetsList:
			mseDict[at]["perMethod"][tt] = {}
			
			for m in ["BM-MAP","BM-EES"]:
				mseDict[at]["perMethod"][tt][m] = {}

				for beta in betas:
					mseDict[at]["perMethod"][tt][m][beta] = {}

					for b in buckets:
						mseDict[at]["perMethod"][tt][m][beta][b] = [0.0] * numStates # to store per state - across different exps per agent type


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		# update agent count
		mseDict[aType]["numAgents"] += resultsDict[a]["num_exps"]


		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if x <= buckets[b]:
				break

		# get actual bucket
		b = buckets[b]


		# for each method
		for m in actualMethodsOnExps:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				# for each state
				for mxi in range(numStates):	

					# xskill error
					sq = resultsDict[a]["plot_y"][tt][tempM][beta][mxi]

					# store squared error
					mseDict[aType]["perMethod"][tt][tempM][beta][b][mxi] += sq


	# Normalize - find Mean Squared Error
	for at in seenAgents:
		for tt in typeTargetsList:
			for m in ["BM-MAP","BM-EES"]:
				for beta in betas:
					for b in buckets:

						# for each state
						for mxi in range(numStates):

							#find MSE  = SE / # of agents
							mseDict[at]["perMethod"][tt][m][beta][b][mxi] /= mseDict[at]["numAgents"]


	# Plot - for MSE
	for at in seenAgents:

		for tt in typeTargetsList:

			for m in ["BM-MAP","BM-EES"]:

				makeFolder3(saveAt+m)
				# makeFolder3(saveAt+"Plotly"+os.path.sep+m)

				for b in buckets:
				
					fig = plt.figure(figsize = (20,30))
					ax = plt.subplot(2,1,1)

					for beta in betas:

						if np.count_nonzero(mseDict[at]["perMethod"][tt][m][beta][b]) != 0:
							# plt.plot(range(numStates),mseDict[at]["perMethod"][m][beta][b], lw=2.0, label= str(b), c = colors[c])
							plt.semilogx(range(numStates),mseDict[at]["perMethod"][tt][m][beta][b], lw=2.0, label= str(beta))

					plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
					plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

					plt.margins(0.05)
					plt.suptitle('Agent: ' + at + ' | MSE of Xskill Methods Diff Betas | Xskill Bucket: ' + str(b)+"\n"+m+" - " + tt)

					fig.subplots_adjust(hspace= 1.0, wspace=1.0)

					plt.legend()
					fig.canvas.mpl_connect("scroll_event", func)

					plt.savefig(saveAt + m + os.path.sep + "results-Agent"+at+"-Method"+m+"-"+tt+"-Bucket"+str(b)+"-"+domain+".png", bbox_inches='tight')

					# # ~~~~~ PLOTLY ~~~~~

					# # Remove legend
					# ax.get_legend().remove()

					# # Re-do axis labels
					# ax.set_xlabel('<b>Number of observations</b>',fontsize=18)
					# ax.set_ylabel('<b>Mean squared error</b>', fontsize=18)

					# # Create Plotly Plot -- Hosting offline
					# plotly_fig =  px.plot_mpl(fig,resize=True)
					# plotly_fig['layout']['showlegend'] = True   
					# plotly_fig['layout']['autosize'] = True  
					# plotly_fig['layout']['height'] *= .80
					# plotly_fig['layout']['width'] *= .80
					# plotly_fig['layout']['margin']['t'] = 50
					# plotly_fig['layout']['margin']['l'] = 0
					# plotly_fig['layout']['title'] = 'Agent: ' + at + ' | MSE of Xskill Methods Diff Betas | Xskill Bucket: ' + str(b)+"\n"+m+" - " + tt

					# # Save plotly
					# unique_url = px.offline.plot(plotly_fig, filename = saveAt + "Plotly" + os.path.sep + m + os.path.sep + "results-Agent"+at+"-Method"+m+"-"+tt+"-Bucket"+str(b)+"-"+domain+ ".html", auto_open=False)


					plt.clf()
					plt.close()

def plotMseDiffBetasPerPskillBucketsPerAgentTypes(resultsDict,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas):

	# BUCKETS PER PERCENTS RAND/MAX REWARD -- SHOWING MSE FOR XSKILL METHODS
	
	method = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-pSkills"


	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "mseDiffBetas-PerPskillBucketsPerAgentType" + os.path.sep
	makeFolder3(saveAt)
	# makeFolder3(saveAt+"Plotly")


	# Buckets in percents terms - between 0-1
	buckets = [0.25,0.50,0.75,1.0]


	# Init dict to store info
	mseDict = {}

	for at in seenAgents:
		mseDict[at] = {"perMethod": {}, "numAgents": 0.0}

		for tt in typeTargetsList:
			mseDict[at]["perMethod"][tt] = {}

			for m in ["BM-MAP","BM-EES"]:
				mseDict[at]["perMethod"][tt][m] = {}

				for beta in betas:
					mseDict[at]["perMethod"][tt][m][beta] = {}

					for b in buckets:
						mseDict[at]["perMethod"][tt][m][beta][b] = [0.0] * numStates # to store per state - across different exps per agent type


	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		# update agent count
		mseDict[aType]["numAgents"] += resultsDict[a]["num_exps"]
		

		#estimatedP = resultsDict[a]["mse_percent_pskills"][method][numStates-1] # #### ESTIMATED %

		trueP = resultsDict[a]["percentTrueP"] # #### TRUE %
		# using true percent and not estimated one

		# find bucket corresponding to trueP
		for b in range(len(buckets)):
			if trueP <= buckets[b]:
				break


		# get actual bucket
		b = buckets[b]


		# for each method
		for m in actualMethodsOnExps:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				# for each state
				for mxi in range(numStates):	

					# xskill error
					sq = resultsDict[a]["plot_y"][tt][tempM][beta][mxi]

					# store squared error
					mseDict[aType]["perMethod"][tt][tempM][beta][b][mxi] += sq


	# Normalize - find Mean Squared Error
	for at in seenAgents:
		for ttt in typeTargetsList:
			for m in ["BM-MAP","BM-EES"]:
				for beta in betas:
					for b in buckets:

						# for each state
						for mxi in range(numStates):

							#find MSE  = SE / # of agents
							mseDict[at]["perMethod"][tt][m][beta][b][mxi] /= mseDict[at]["numAgents"]


	# Plot - for MSE
	for at in seenAgents:

		for tt in typeTargetsList:

			for m in ["BM-MAP","BM-EES"]:

				makeFolder3(saveAt+m)
				# makeFolder3(saveAt+"Plotly"+os.path.sep+m)


				for b in buckets:
				
					fig = plt.figure(figsize = (20,30))
					ax = plt.subplot(2, 1, 1)

					for beta in betas:

						if np.count_nonzero(mseDict[at]["perMethod"][tt][m][beta][b]) != 0:
							# plt.plot(range(numStates),mseDict[at]["perMethod"][m][beta][b], lw=2.0, label= str(b), c = colors[c])
							plt.semilogx(range(numStates),mseDict[at]["perMethod"][tt][m][beta][b], lw=2.0, label= str(beta))

					plt.xlabel(r'\textbf{Number of Observations}',fontsize=18)
					plt.ylabel(r'\textbf{Mean Squared Error}', fontsize=18)

					plt.margins(0.05)
					plt.suptitle('Agent: ' + at + ' | MSE of Xskill Methods Diff Betas | Pskill Bucket: ' + str(b) + "\n"+m+" - "+tt)

					fig.subplots_adjust(hspace= 1.0, wspace=1.0)

					plt.legend()
					fig.canvas.mpl_connect("scroll_event", func)

					plt.savefig(saveAt + m + os.path.sep + "results-Agent"+at+"-Method"+m+"-"+tt+"-Bucket"+str(b)+"-"+domain+".png", bbox_inches='tight')

					# ~~~~~ PLOTLY ~~~~~

					# # Remove legend
					# ax.get_legend().remove()

					# # Re-do axis labels
					# ax.set_xlabel('<b>Number of observations</b>',fontsize=18)
					# ax.set_ylabel('<b>Mean squared error</b>', fontsize=18)

					# # Create Plotly Plot -- Hosting offline
					# plotly_fig =  px.plot_mpl(fig,resize=True)
					# plotly_fig['layout']['showlegend'] = True   
					# plotly_fig['layout']['autosize'] = True  
					# plotly_fig['layout']['height'] *= .80
					# plotly_fig['layout']['width'] *= .80
					# plotly_fig['layout']['margin']['t'] = 50
					# plotly_fig['layout']['margin']['l'] = 0
					# plotly_fig['layout']['title'] = 'Agent: ' + at + ' | MSE of Xskill Methods Diff Betas | Pskill Bucket: ' + str(b) + "\n"+m+" - "+tt

					# # Save plotly
					# unique_url = px.offline.plot(plotly_fig, filename = saveAt + "Plotly" + os.path.sep + m + os.path.sep + "results-Agent"+at+"-Method"+m+"-"+tt+"-Bucket"+str(b)+"-"+domain+ ".html", auto_open=False)


					plt.clf()
					plt.close()

def plotLastMSEAllBetasSamePlotPerAgentType(resultsDict,methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,mseAcrossAllAgentsPerAgentType,betas):

	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "LastMSE-AllBetasSamePlotPerAgentType" + os.path.sep 
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
					plt.savefig(saveAt + os.path.sep + plottingMethod + os.path.sep + fileName + ".png", bbox_inches='tight')

					plt.clf()
					plt.close()

				except:
					continue

	#code.interact("after...", local=dict(globals(), **locals()))
	#######################################################################################


def plotMSEAllBetasSamePlotPerAgentType(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerAgentType,betas):

	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "MSEAcrossAllAgents-AllBetasSamePlot-PerAgentType"+os.path.sep
	
	makeFolder3(saveAt)
	# makeFolder3(saveAt+"Plotly")

	for at in agentTypes:

		for xType in ["Log","NotLog"]:
		
			makeFolder3(saveAt+xType)
			# makeFolder3(saveAt+"Plotly"+os.path.sep+xType)

			for tt in typeTargetsList:

				for plottingMethod in ["BM-MAP","BM-EES"]:

					makeFolder3(saveAt+xType+os.path.sep+plottingMethod)
					# makeFolder3(saveAt+"Plotly"+os.path.sep+xType+os.path.sep+plottingMethod)
					
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
					plt.savefig(saveAt + xType + os.path.sep + plottingMethod + os.path.sep + fileName + ".png", bbox_inches='tight')


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
					pio.write_html(fig, file=saveAt + "Plotly" + os.path.sep + xType + os.path.sep + plottingMethod + os.path.sep + fileName + ".html", auto_open=False)
					# plotly_fig['layout']['title'] = xType + '-- XSKILL -' + plottingMethod + ' | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments'
					# Save plotly
					# unique_url = py.offline.plot(plotly_fig, filename = saveAt + "Plotly" + os.path.sep + xType + os.path.sep + plottingMethod + os.path.sep + fileName + ".html", auto_open=False)
					'''
					plt.clf()
					plt.close("all")

###################################################################################


def plotLastMSEAllBetasSamePlotAllAgent(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerMethod,betas):

	# Plot - For BM method
	#	X: Different betas
	# 	Y: MSE across agents last observation 

	saveAt = resultsFolder + os.path.sep + "plots" + os.path.sep + "BETAS" + os.path.sep + "LastMSE-AcrossAllAgents-AllBetasSamePlot" + os.path.sep 
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
				plt.savefig(saveAt + os.path.sep + plottingMethod + os.path.sep + fileName + ".png", bbox_inches='tight')

				plt.clf()
				plt.close()

			except:
				continue

def computeAndPlotMSEAcrossAllAgentsPerMethod(resultsDict, methodsNames, methods, resultsFolder, agentTypes, numStates, domain, betas, givenBeta):

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



	totalAgents = len(resultsDict.keys())

	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

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

	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-PerMethod" + os.path.sep + "results-XSKILL"+domain+".png", bbox_inches='tight')
	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-PerMethod" + os.path.sep + "results-XSKILL"+domain+".pdf", bbox_inches='tight')

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
	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-PerMethod" + os.path.sep + "results-PSKILL-Agent"+"-"+domain+".png", bbox_inches='tight')
	plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-PerMethod" + os.path.sep + "results-PSKILL-Agent"+"-"+domain+".pdf", bbox_inches='tight')

	plt.clf()
	plt.close("all")

	# code.interact("...", local=dict(globals(), **locals()))
	plotLastMSEAllBetasSamePlotAllAgent(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerMethod,betas)
		

def computeAndPlotMSEAcrossAllAgentsTypesAllMethods(resultsDict, methodsNames, methods, resultsFolder, agentTypes, numStates, domain, betas, givenBeta, makeOtherPlots=False):

	makeFolder(resultsFolder, "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType")

	mseAcrossAllAgentsPerAgentType = {}
	stdInfoPerAgentTypePerMethod = {}
	stdPerAgentTypePerMethod = {}
	confidenceIntervals = {}

	# set up dict to save info
	for at in agentTypes:
		mseAcrossAllAgentsPerAgentType[at] = {"numAgents": 0.0, "totalNumExps": 0.0}
		stdInfoPerAgentTypePerMethod[at] = {}
		stdPerAgentTypePerMethod[at] = {}
		confidenceIntervals[at] = {}

		for tt in typeTargetsList:
			mseAcrossAllAgentsPerAgentType[at][tt] = {}
			stdInfoPerAgentTypePerMethod[at][tt] = {}
			stdPerAgentTypePerMethod[at][tt] = {}
			confidenceIntervals[at][tt] = {}


		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				if tempM not in mseAcrossAllAgentsPerAgentType[at][tt]:
					mseAcrossAllAgentsPerAgentType[at][tt][tempM] = {}
					stdInfoPerAgentTypePerMethod[at][tt][tempM] = {}
					stdPerAgentTypePerMethod[at][tt][tempM] = {}
					confidenceIntervals[at][tt][tempM] = {}

				if beta not in mseAcrossAllAgentsPerAgentType[at][tt][tempM]:
					mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta] = [0.0] * numStates
					stdInfoPerAgentTypePerMethod[at][tt][tempM][beta] = [] 
					stdPerAgentTypePerMethod[at][tt][tempM][beta] = 0.0
					confidenceIntervals[at][tt][tempM][beta] = {"low": 0.0, "high": 0.0, "value": 0.0}
				
			else:
				mseAcrossAllAgentsPerAgentType[at][m] = [0.0] * numStates
				stdInfoPerAgentTypePerMethod[at][m] = [] 
				stdPerAgentTypePerMethod[at][m] = 0.0
				confidenceIntervals[at][m] = {"low": 0.0, "high": 0.0, "value": 0.0}


	#####################################################################################################
	# compute mse across all agents of same type
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for s in range(numStates):
					mseAcrossAllAgentsPerAgentType[aType][tt][tempM][beta][s] += resultsDict[a]["plot_y"][tt][tempM][beta][s]

			else:

				for s in range(numStates):
					if "-pSkills" in m:
						mseAcrossAllAgentsPerAgentType[aType][m][s] += resultsDict[a]["mse_percent_pskills"][m][s]
					else:
						mseAcrossAllAgentsPerAgentType[aType][m][s] += resultsDict[a]["plot_y"][m][s]

		mseAcrossAllAgentsPerAgentType[aType]["numAgents"] += 1.0
		mseAcrossAllAgentsPerAgentType[aType]["totalNumExps"] += resultsDict[a]["num_exps"]


	# Will help set y axis limits
	maxXskillError = -99999
	maxPskillError = -99999

	# Normalize
	for at in agentTypes:
		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for s in range(numStates):
					#code.interact("before norm", local=locals())
					mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
					#code.interact("after norm", local=locals())

					# Update max error seen - will use as y axis limit
					if mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][s] > maxXskillError:
						maxXskillError = mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][s] 

			else:
				for s in range(numStates):
					#code.interact("before norm", local=locals())
					mseAcrossAllAgentsPerAgentType[at][m][s] /= (mseAcrossAllAgentsPerAgentType[at]["numAgents"] * 1.0)
					#code.interact("after norm", local=locals())

					if "xSkills" in m or "OR" in m or "BM" in m:
						# Update max error seen - will use as y axis limit
						if mseAcrossAllAgentsPerAgentType[at][m][s] > maxXskillError:
							maxXskillError = mseAcrossAllAgentsPerAgentType[at][m][s] 

					elif "pSkills" in m:
						# Update max error seen - will use as y axis limit
						if mseAcrossAllAgentsPerAgentType[at][m][s] > maxPskillError:
							maxPskillError = mseAcrossAllAgentsPerAgentType[at][m][s] 

	#####################################################################################################



	#####################################################################################################
	# get data for standard deviation across all agents of same type --- for last state
	for a in resultsDict.keys():

		aType, x, p = getParamsFromAgentName(a)

		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				stdInfoPerAgentTypePerMethod[aType][tt][tempM][beta].append(resultsDict[a]["plot_y"][tt][tempM][beta][-1])

			else:

				if "-pSkills" in m:
					stdInfoPerAgentTypePerMethod[aType][m].append(resultsDict[a]["mse_percent_pskills"][m][-1])
				else:
					stdInfoPerAgentTypePerMethod[aType][m].append(resultsDict[a]["plot_y"][m][-1])

	# compute actual std
	for at in agentTypes:
		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)
				stdPerAgentTypePerMethod[at][tt][tempM][beta] = np.std(stdInfoPerAgentTypePerMethod[at][tt][tempM][beta])
			else:
				stdPerAgentTypePerMethod[at][m] = np.std(stdInfoPerAgentTypePerMethod[at][m])

	#####################################################################################################

	# code.interact("MSE...", local=dict(globals(), **locals()))

	#####################################################################################################
	# COMPUTE CONFIDENCE INTERVALS
	#####################################################################################################
	
	ci = 0.95
	
	# for 95% interval
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
				stats.norm.interval(ci, loc=mu, scale=sigma/np.sqrt(N))

				confidenceIntervals[at][tt][tempM][beta]["value"] = Z * (sigma/np.sqrt(N))

			else:

				mu = mseAcrossAllAgentsPerAgentType[at][m][-1]
				sigma = stdPerAgentTypePerMethod[at][m]

				confidenceIntervals[at][m]["low"], confidenceIntervals[at][m]["high"] =\
				stats.norm.interval(ci, loc=mu, scale=sigma/np.sqrt(N))

				confidenceIntervals[at][m]["value"] = Z * (sigma/np.sqrt(N))

	#####################################################################################################
	

	xskillsCI = open(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "confidenceIntervals-xSkills.txt", "a")
	pskillsCI = open(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "confidenceIntervals-pSkills.txt", "a")

	# save info to text files

	for at in agentTypes:

		# Gather all MSE's of current method
		allMseCurrentMethod = []

		for m in methods:

			if m == "tn":
				continue

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)
				allMseCurrentMethod.append(mseAcrossAllAgentsPerAgentType[at][tt][tempM][beta][-1])
			else:
				allMseCurrentMethod.append(mseAcrossAllAgentsPerAgentType[at][m][-1])
		
		# order from highest to lowest MSE
		orderedMSE = sorted(allMseCurrentMethod, reverse = True)


		d_x = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}
		d_p = {"Agents":[], "Methods": [], "MSE":[], "Low": [], "High": [], "Values": []}

		# Output info to files
		for mseO in range(len(orderedMSE)):

			index = allMseCurrentMethod.index(orderedMSE[mseO])

			m = methods[index]

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
						" ->  Low: " + str(round(confidenceIntervals[at][m]["low"],4)) +\
						" | High: " + str(round(confidenceIntervals[at][m]["high"],4)) +\
						" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
						" | Value: " + str(round(confidenceIntervals[at][m]["value"],4)) + "\n")

				d_p["Agents"].append(at)
				d_p["Methods"].append(mm)
				d_p["MSE"].append(round(orderedMSE[mseO],2))
				d_p["Low"].append(round(confidenceIntervals[at][m]["low"],2))
				d_p["High"].append(round(confidenceIntervals[at][m]["high"],2))
				d_p["Values"].append(round(confidenceIntervals[at][m]["value"],2))

			### for xskills
			else:

				if "BM" in m:
					xskillsCI.write("Agent: " + at + " | Method: " + str(m) + "-" + tt + " | Beta: " + str(beta) +\
							" ->  Low: " + str(round(confidenceIntervals[at][tt][tempM][beta]["low"],4)) +\
							" | High: " + str(round(confidenceIntervals[at][tt][tempM][beta]["high"],4)) +\
							" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
							" | Value: " + str(round(confidenceIntervals[at][tt][tempM][beta]["value"],4)) + "\n")

					d_x["Agents"].append(at)
					d_x["Methods"].append(mm)
					d_x["MSE"].append(round(orderedMSE[mseO],2))
					d_x["Low"].append(round(confidenceIntervals[at][tt][tempM][beta]["low"],2))
					d_x["High"].append(round(confidenceIntervals[at][tt][tempM][beta]["high"],2))
					d_x["Values"].append(round(confidenceIntervals[at][tt][tempM][beta]["value"],2))

				else:
					xskillsCI.write("Agent: " + at + " | Method: " + str(m) + \
							" ->  Low: " + str(round(confidenceIntervals[at][m]["low"],4)) +\
							" | High: " + str(round(confidenceIntervals[at][m]["high"],4)) +\
							" ||| Mean: " + str(round(orderedMSE[mseO],4)) +\
							" | Value: " + str(round(confidenceIntervals[at][m]["value"],4)) + "\n")

					d_x["Agents"].append(at)
					d_x["Methods"].append(mm)
					d_x["MSE"].append(round(orderedMSE[mseO],2))
					d_x["Low"].append(round(confidenceIntervals[at][m]["low"],2))
					d_x["High"].append(round(confidenceIntervals[at][m]["high"],2))
					d_x["Values"].append(round(confidenceIntervals[at][m]["value"],2))


		pskillsCI.write("\n")
		xskillsCI.write("\n")

		pskillsCI.write("\n")
		xskillsCI.write("\n")

		 # Convert dicts to pandas dataframe
		d_x_pd = pd.DataFrame(d_x, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])
		d_p_pd = pd.DataFrame(d_p, columns = ["Agents", "Methods", "Low", "MSE",  "High", "Values"])

		xskillsCI.write(d_x_pd.to_latex(index=False))
		pskillsCI.write(d_p_pd.to_latex(index=False))

		pskillsCI.write("\n")
		xskillsCI.write("\n")

	# code.interact("here...", local=dict(globals(), **locals()))


	#####################################################################################################
	# PLOTS!
	#####################################################################################################

	# For padding at the top of the y axis
	maxXskillError += (maxXskillError/10.0) 
	maxPskillError += (maxPskillError/10.0) 

	for at in agentTypes:

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

				plt.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta], lw='2.0', label= str(tempM3), c = methodsColors[tempM2])
				

			# other methods (and not other BM methods)         
			elif "BM" not in method:		
				#m = methodsDict[method]
				m = methodNamesPaper[method]
				plt.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method], lw='2.0', label= str(m), c = methodsColors[method])

		ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
		ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
		plt.margins(0.05)
		plt.ylim(top = maxXskillError)

		#plt.title('XSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

		# Put a legend to the right of the current axis
		ax1.legend(loc='best')

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight')
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "results-XSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight')

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
			plt.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method], lw='2.0', label= str(m), c = methodsColors[method])
			# plt.plot(range(len(mseAcrossAllAgentsPerAgentType[at][method])),mseAcrossAllAgentsPerAgentType[at][method], lw='2.0', label= str(m))

		ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=24)
		ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=24)
		plt.margins(0.05)
		plt.ylim(top = maxPskillError)

		#plt.title('PSKILL | Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

		# Put a legend to the right of the current axis
		ax1.legend(loc='best')
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".png", bbox_inches='tight')
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "results-PSKILL-Agent"+at+"-Exps"+str(int(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]))+"-"+domain+".pdf", bbox_inches='tight')

		plt.clf()
		plt.close("all")
		

		##################################### FOR XSKILLS & PSKILLS - same plot #####################################
		# create plot for each one of the different agents - estimates vs obs

		'''
		fig = plt.figure()

		# rows, cols, pos
		ax1 = plt.subplot(2, 1, 1)
		ax2 = plt.subplot(2, 1, 2)

		fig.subplots_adjust(hspace=1.0, wspace=0.4)

		for method in methods: #methodsDictNames.keys():

			if "tn" in method:
				continue

			# plotting xskills & pskill methods
			if ("JT" in method or "NJT" in method) and method in methodsNames:
				m = method

				# xskill 
				m1 = methodsDictNames[method][0]
				m1N = methodNamesPaper[m1] # m1

				# pskill
				m2 = methodsDictNames[method][1]
				m2N = methodNamesPaper[m2] #m2


				ax1.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][m1])),mseAcrossAllAgentsPerAgentType[at][m1], lw='2.0', label= str(m1N), c = methodsColors[m1])
				
				ax2.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][m2])),mseAcrossAllAgentsPerAgentType[at][m2], lw='2.0', label= str(m2N), c = methodsColors[m2])
			# only plotting xskills methods
			else:

				# Will just plot given beta
				if "BM" in method and str(givenBeta) in method:
					tempM, beta, tt = getInfoBM(method)

					if tempM in methodsNames:
						#m = methodsDict[method]
						tempM2 = method.split("-Beta")[0]
						tempM3 = methodNamesPaper[method.split("-Beta")[0]]
						ax1.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta])),mseAcrossAllAgentsPerAgentType[at][tt][tempM][givenBeta], lw='2.0', label= str(tempM3)+"Beta"+str(givenBeta), c = methodsColors[tempM2])
				
				elif "BM" not in method and method in methodsNames:
					m1 = methodsDictNames[method]
					m1N = methodNamesPaper[m1] #m1
	  
					# other methods 
					ax1.semilogx(range(len(mseAcrossAllAgentsPerAgentType[at][m1])),mseAcrossAllAgentsPerAgentType[at][m1], lw='2.0', label= str(m1N), c = methodsColors[m1])
					# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))
					# print mseAcrossAllAgentsPerAgentType[at][method]


		ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=18)
		ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
		ax1.set_title("Xskills")
		ax1.set_ylim(0,maxXskillError)

		ax2.set_xlabel(r'\textbf{Number of observations}',fontsize=18)
		ax2.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
		ax2.set_title("Pskills")
		ax2.set_ylim(0,maxPskillError)

		plt.margins(0.05)
		fig.suptitle('Agent: ' + at + ' | '+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"]) + ' experiments')

		# Put a legend to the right of the current axis
		ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAcrossAllAgents-AllMethodsSamePlot-PerAgentType" + os.path.sep + "results-BOTH-Agent"+at+"-Exps"+str(mseAcrossAllAgentsPerAgentType[at]["totalNumExps"])+".png", bbox_inches='tight')

		plt.clf()
		plt.close("all")
		'''


	if makeOtherPlots:
		plotMSEAllBetasSamePlotPerAgentType(resultsFolder,agentTypes,methods,mseAcrossAllAgentsPerAgentType,betas)
		plotLastMSEAllBetasSamePlotPerAgentType(resultsDict,methodsNames,methods,numHypsX,numHypsP,resultsFolder,agentTypes,mseAcrossAllAgentsPerAgentType,betas)
		#plotMseDiffBetasPerPskillBucketsPerAgentTypes(resultsDict,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas)
		#plotMseDiffBetasPerXskillBucketsPerAgentTypes(resultsDict,actualMethodsOnExps,resultsFolder,domain,numStates,numHypsX,numHypsP,betas)
	# code.interact("MSE-ALL", local=dict(globals(), **locals())) 


def plotMSEAllMethodsSamePlotPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder):

	makeFolder(resultsFolder, "MSEAllMethodsSamePlotPerAgent")

	for a in resultsDict.keys():

		fig = plt.figure(figsize = (10,10))

		##################################### FOR XSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		ax1 = plt.subplot(2, 1, 1)

		for method in methods:

			# only plotting xskills methods
			if "pSkills" in method:
				continue

			if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
				m = method.replace("BM","TBA")
			else:
				m = method

			if "tn" in method: 
				# to plot the TN
				continue
				# plt.semilogx(range(len(resultsDict[a]["plot_y"]["tn"])),resultsDict[a]["plot_y"]["tn"],lw=2.0, ls='dashed', label= "TN")
				# plt.plot(range(len(resultsDict[a]["plot_y"]["tn"])),resultsDict[a]["plot_y"]["tn"],lw=2.0, ls='dashed', label= "TN")
			else:   
				# other methods          
				plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label= str(m))
				# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))


		ax1.set_xlabel(r'\textbf{Number of observations}',fontsize=18)
		ax1.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)

		fig.suptitle(r'\textbf{Agent: ' + a + ' | '+str(resultsDict[a]["num_exps"]) + ' experiments}')

		# Put a legend to the right of the current axis
		ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))



		# to add space between subplots
		plt.subplots_adjust(hspace = 0.3)

		# Adds "nothing" to the plot 
		# Done in order to add an empty label to the legend 
		# So that there can be a space between the xskill elements & the pskill elements
		plt.plot(np.NaN, np.NaN, '-', alpha = 0.0, label=" ")


	
		##################################### FOR PSKILLS #####################################
		# create plot for each one of the different agents - estimates vs obs

		ax2 = plt.subplot(2, 1, 2)

		for method in methods:

			# only plotting pskills methods
			if "pSkills" not in method:
				continue

			if "tn" in method:
				continue


			if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
				m = method.replace("BM","TBA")
			else:
				m = method

			plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label= str(m))
			# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))

		ax2.set_xlabel(r'\textbf{Number of observations}',fontsize=18)
		ax2.set_ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)

		# Put a legend to the right of the current axis
		ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		# Save png
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "MSEAllMethodsSamePlotPerAgent" + os.path.sep + "results-Agent"+a+".png", bbox_inches='tight')


		# # ~~~~~ PLOTLY ~~~~~

		# # Remove legend
		# ax1.get_legend().remove()
		# ax2.get_legend().remove()


		# # Re-do axis labels
		# ax1.set_xlabel('<b>Number of observations</b>',fontsize=18)
		# ax1.set_ylabel('<b>Mean squared error</b>', fontsize=18)
		# ax2.set_xlabel('<b>Number of observations</b>',fontsize=18)
		# ax2.set_ylabel('<b>Mean squared error</b>', fontsize=18)


		# # Create Plotly Plot -- Hosting offline
		# plotly_fig =  px.plot_mpl(fig)
		# plotly_fig['layout']['showlegend'] = True   
		# plotly_fig['layout']['autosize'] = True  
		# plotly_fig['layout']['title'] = '<b>Agent: ' + a + ' | '+str(resultsDict[a]["num_exps"]) + ' experiments</b>'

		# # Save plotly
		# unique_url = px.offline.plot(plotly_fig, filename=resultsFolder + os.path.sep + "plots" + \
		# 										os.path.sep + "MSEAllMethodsSamePlotPerAgent" + os.path.sep + "results-Agent"+a+".html", auto_open=False)


		plt.clf()
		plt.close()


	#######################################################################################

def plotMSEAllPSkillMethodsPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp):

	makeFolder(resultsFolder, "mseAllPSkillMethodsPerAgent")

	for a in resultsDict.keys():
		fig = plt.figure()
		ax = plt.subplot(111)

		for method in methods:

			# only plotting pskills methods
			if "pSkills" not in method:
				continue

			if "tn" in method:
				continue

			if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
				m = method.replace("BM","TBA")
			else:
				m = method

			plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label= str(m))
			# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))

		plt.xlabel(r'\textbf{Number of observations}',fontsize=18)
		plt.ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)
		plt.title('Agent: ' + a + ' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

		#axes = plt.gca()
		#axes.set_ylim([0,2])

		# Shrink current axis by 10%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})
		
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseAllPSkillMethodsPerAgent" + os.path.sep +  "results-Agent"+a+"-AllMethods-Hyps"+str(hyp)+".png", bbox_inches='tight')
		plt.clf()

		plt.close()

def plotMSEAllXSkillMethodsPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder, hyp):

	makeFolder(resultsFolder, "mseAllXSkillMethodsPerAgent")

	for a in resultsDict.keys():
		fig = plt.figure()
		ax = plt.subplot(111)

		for method in methods:

			# only plotting xskills methods
			if "pSkills" in method:
				continue

			if "BM" in method: # 'BM-MAP', 'BM-EES' to TBA
				m = method.replace("BM","TBA")
			else:
				m = method

			if "tn" in method: 
				# to plot the TN
				plt.semilogx(range(len(resultsDict[a]["plot_y"]["tn"])),resultsDict[a]["plot_y"]["tn"],lw=2.0, ls='dashed', label= "TN")
				# plt.plot(range(len(resultsDict[a]["plot_y"]["tn"])),resultsDict[a]["plot_y"]["tn"],lw=2.0, ls='dashed', label= "TN")
			else:   
				# other methods          
				plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label= str(m))
				# plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw=2.0, label= str(m))



		plt.xlabel(r'\textbf{Number of observations}',fontsize=18)
		plt.ylabel(r'\textbf{Mean squared error}', fontsize=18)
		plt.margins(0.05)
		plt.title('Agent: ' + a + ' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')

		#axes = plt.gca()
		#axes.set_ylim([0,2])

		# Shrink current axis by 10%
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),prop={'size':14})
		
		plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "mseAllXSkillMethodsPerAgent" + os.path.sep +  "results-Agent"+a+"-AllMethods-Hyps"+str(hyp)+".png", bbox_inches='tight')
		plt.clf()

		plt.close()

def plotMSEPerMethodPerAgent(resultsDict, methodsNames, methods, numHypsX, numHypsP, resultsFolder):

	makeFolder(resultsFolder, "msePerMethodPerAgent")

	# create plot for each one of the different methods - MSE
	for a in resultsDict.keys():
		for method in methods:
					
			# plt.semilogx(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label = method)
			plt.plot(range(len(resultsDict[a]["plot_y"][method])),resultsDict[a]["plot_y"][method], lw='2.0', label = method)

			# to plot the TN
			# not plotting TN for pskills info - for now
			if "pSkills" not in method:
				plt.plot(range(len(resultsDict[a]["plot_y"]["tn"])),resultsDict[a]["plot_y"]["tn"], lw='2.0', label= "TN")

			plt.xlabel(r'\textbf{Number of observations}')
			plt.ylabel(r'\textbf{Mean squared error}')
			plt.margins(0.05)
			plt.title('Agent: ' + a + '| Method: '+ method +' | (averaged over '+str(resultsDict[a]["num_exps"]) + ' experiments)')
			plt.legend()

			plt.savefig(resultsFolder + os.path.sep + "plots" + os.path.sep + "msePerMethodPerAgent" + os.path.sep + "results-Agent"+a+"-Method"+method+".png", bbox_inches='tight')
			plt.clf()
			plt.close()

def computeMSE(resultsDict, methodsNames, methods, numHypsX, numHypsP):

	# Compute MSE
	for a in resultsDict.keys():
		print("\nAgent: ", a)

		for m in methods:

			if "BM" in m:
				tempM, beta, tt = getInfoBM(m)

				for mxi in range(len(resultsDict[a]["plot_y"][tt][tempM][beta])):
					resultsDict[a]["plot_y"][tt][tempM][beta][mxi] = resultsDict[a]["plot_y"][tt][tempM][beta][mxi] / float(resultsDict[a]["num_exps"]) #MSError

				print('Method : ', tempM, " -", tt, " Beta: ", beta, 'has', len(resultsDict[a]["plot_y"][tt][tempM][beta]), 'data points and ', \
				resultsDict[a]['num_exps'], ' experiments.')

			else:
			
				for mxi in range(len(resultsDict[a]["plot_y"][m])):
					resultsDict[a]["plot_y"][m][mxi] = resultsDict[a]["plot_y"][m][mxi] / float(resultsDict[a]["num_exps"]) #MSError

				print('Method : ', m, 'has', len(resultsDict[a]["plot_y"][m]), 'data points and ', \
					resultsDict[a]['num_exps'], ' experiments.')

		# Do it for the TN as well
		# for mxi in range(len(resultsDict[a]["plot_y"][m])):
			# resultsDict[a]["plot_y"]["tn"][mxi] /= float(resultsDict[a]["num_exps"]) #MSError


def getAgentInfo(domainName,agentName):

	if domainName in ["1d","2d","sequentialDarts"]:

		if "TargetAgent-BeliefX" in agentName:
			aNameOriginal = str(agentName.split("Agent-BeliefX")[0]) + "Belief"
			params = str(agentName.split("Agent-BeliefX")[1])

		else:		
			aNameOriginal = str(agentName.split("Agent-X")[0])
			params = str(agentName.split("Agent-X")[1])


		aName = aNameOriginal

		if aName == "Flip":
			string = params.split("-P")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-P" + param

		elif aName == "Tricker":
			string = params.split("-Eps")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-Eps" + param

		elif aName == "Bounded":
			string = params.split("-L")
			xStr = string[0]
			param = string[1]
			aName += "-X" + xStr
			aName += "-L" + param

		elif aName == "Target":
			string = params.split("-X")
			xStr = string[0]
			param = "100"
			aName += "-X" + xStr

		elif aName == "TargetBelief":
			string = params.split("-TrueX")
			param = string[0] # beliefX
			xStr = string[1] # trueX
			aName += "-X" + xStr
			aName += "-B" + param

		elif aName == "Random":
			string1 = params.split("-N")
			string2 = string1[1].split("-K")
			xStr = string1[0]
			param1 = string2[0]
			param2 = string2[1]
			aName += "-X" + xStr
			aName += "-N" + param1
			aName += "-K" + param2

			param = {"param1":param1,"param2":param2}

	
	# code.interact("...", local=dict(globals(), **locals()))
	
	if aName == "Random":
		return aNameOriginal,aName,param
	else:
		return aNameOriginal,aName,float(param)

###################################################################################


if __name__ == "__main__":

	# USAGE EXAMPLE:
	#  python Processing/process_results.py -resultsFolder Experiments/testingRandomAgent-AllMethods-1D-rand/ -domain 1d

	# OR 
	# python Processing/process_results.py -resultsFolder Experiments/ExpsAllMethods-AllAgents-Log-200States-2D -processingAgents Target
	# Use 3rd argument to indicate filtering by that agent
	# As in just process files for given agent type (one at a time for now)


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-delta", dest = "delta", help = "Delta = resolution to use when doing the convolution", type = float, default = 5.0)
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "2d")
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	parser.add_argument("-processingAgents", dest = "processingAgents", help = "Focus on the experiments of which agent?", type = str, default = "")
	parser.add_argument("-N", dest = "N", help = "", type = int, default = 1)
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	result_files = os.listdir(args.resultsFolder + os.path.sep + "results")

	# If the plots folder doesn't exist already, create it
	if not os.path.exists(args.resultsFolder + os.path.sep + "plots" + os.path.sep):
		os.mkdir(args.resultsFolder + os.path.sep + "plots" + os.path.sep)


	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework")) + os.path.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	seed = np.random.randint(0,1000000,1)

	rng = np.random.default_rng(seed)


	# methodsNames = ['OR', 'BM-MAP', 'BM-EES', "JT-FLIP-MAP", "JT-FLIP-EES","JT-QRE-MAP", "JT-QRE-EES"]
	methodsNames = ['OR', 'BM-MAP', 'BM-EES',"JT-QRE-MAP", "JT-QRE-EES", "NJT-QRE-MAP", "NJT-QRE-EES"]


	#agentTypes = ["Target", "Flip", "Tricker","Bounded", "TargetBelief"]
	agentTypes = ["Target", "Flip", "Tricker","Bounded"]


	resultsDict = {}

	actualMethodsNames = []
	actualMethodsOnExps = []
	typeTargetsList = []

	numHypsX = []
	numHypsP = []
	numStates = 0
	seenAgents = []


	processingAgents = None

	if args.processingAgents != "":
		processingAgents = [args.processingAgents]
	#processingAgents = ["Target"]

	domainModule,delta = getDomainInfo(args.domain)


	# For 2D will have "normal", "rand_pos" or "rand_v".
	# For 1D it will stay as none.
	mode = None


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		if args.processingAgents != "":
			rdFile = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "resultsDictInfo" + processingAgents[0]
			oiFile = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "otherInfo" + processingAgents[0]
		else:
			rdFile = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "resultsDictInfo"
			oiFile = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "otherInfo"     


		with open(rdFile,"rb") as file:
			resultsDict = pickle.load(file)

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

			resultFilesLoaded = otherInfo["result_files"]
			actualMethodsOnExpsObtainedFlag = otherInfo["actualMethodsOnExpsObtainedFlag"]
	
		loadedInfo = True

	except:
		# Do nothing, just continue to processing files
		loadedInfo = False

		resultFilesLoaded = []


	# If wasn't able to load prev processed info, start from scratch
	if not loadedInfo:
		domain = ""

		# Open the first file to load the different number of hypothesis used for the different estimators
		i = 0
		while result_files[i][-7:] != 'results':
			i += 1

		with open(args.resultsFolder + os.path.sep + "results" + os.path.sep + result_files[i]) as infile:
			results = json.load(infile)
			
			numHypsX = results['numHypsX']
			numHypsP = results['numHypsP']

			numStates = results["numObservations"]

			domain = results["domain"]
			mode = results["mode"]


		betas = []

		for tempKey in results.keys():

			if "BM-EES" in tempKey and "Beta" in tempKey:
				b = float(tempKey.split("Beta-")[1])
				if b not in betas:
					betas.append(b)

			if "BM" in tempKey:
				if "Optimal" in tempKey and "OptimalTargets" not in typeTargetsList:
					typeTargetsList.append("OptimalTargets")
				if "Domain" in tempKey and "DomainTargets" not in typeTargetsList:
					typeTargetsList.append("DomainTargets")


		methods = []
		
		for m in methodsNames:
			if "JT" in m:   
				for nh in range(len(numHypsX)):
					methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-pSkills")
					methods.append(m + "-" + str(numHypsX[nh]) + "-" + str(numHypsP[nh]) + "-xSkills")
			elif "OR" in m:
				for nh in range(len(numHypsX)):
					if domain == "1d" or domain == "2d":
						methods.append(m + "-" + str(numHypsX[nh]))
					else: # For sequential darts & billiards
						methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesMidGame")
						methods.append(m + "-" + str(numHypsX[nh]) + "-estimatesFullGame")
			elif "BM" in m:
				for nh in range(len(numHypsX)):
					for tt in typeTargetsList:
						# 1D & 2D will only have TBA-"OptimalTargets"
						if (domain == "1d" or domain == "2d") and tt == "DomainTargets":
							continue
						for b in betas:
							methods.append(m + "-" + str(numHypsX[nh]) + "-" + tt + "-Beta-" + str(b))

			else:
				for nh in range(len(numHypsX)):
					# Methods left is OR & no pskill hyps is needed
					methods.append(m + "-" + str(numHypsX[nh])) # + "-" + str(numHypsP[nh]))
		
		methods.append("tn")

		actualMethodsOnExpsObtainedFlag = False

		#code.interact("!!!...", local=dict(globals(), **locals()))


	############################################################################################################################
	############################################################################################################################
	# Use when debugging/testing - to speed up - read only specified # of results file (and not however many there are in the folder)
	#result_files = result_files[0:3000]
	############################################################################################################################
	############################################################################################################################


	####################################
	# PCONF
	####################################
	
	# Compute functions - to use for conversion to % of RandMax Reward
	pconfPerXskill = pconf(args.resultsFolder,domain,domainModule,spacesModule,mode,args)

	####################################


	bucketsX = sorted(pconfPerXskill.keys())

	if domain == "1d":
		minMaxX = [0,5]

	elif domain == "2d" or domain == "sequentialDarts":
		minMaxX = [0,150]


	# Start processing results
	total_num_exps = 0

	allExpsCount = 1

	# Collate results for the methods
	for rf in result_files: 
		# For each file, get the information from it
		print ('('+str(allExpsCount)+'/'+str(len(result_files))+') - RF : ', rf)

		allExpsCount += 1

		if rf in resultFilesLoaded:
			print("\tSkipping ", rf, " since already loaded.")
			continue

		param = ""

		with open(args.resultsFolder + os.path.sep + "results" + os.path.sep + rf) as infile:
			results = json.load(infile)


			try:
				tempTime = results["expTotalTime"]
			except: 
				# To skip results file for exps that are still running
				print("\tSkipping ", rf, " since not done yet.")
				continue


			aNameOriginal,aName,param = getAgentInfo(args.domain,results['agent_name'])

			#####################################################################################################
			if args.processingAgents != "":
				# if not a result file of one of the agents we are processing
				if aNameOriginal not in processingAgents:
					print("\tSkipping ", rf, " since not of processing agent(s).")
					continue
			#####################################################################################################


			# Create copy of agent type
			seenAgentType = aNameOriginal


			# if the agent exists already in the results dict, add the info
			if aName in resultsDict.keys():
				resultsDict[aName]["num_exps"] += 1

			# otherwise, initialize corresponding things
			else:

				# avg_rewards - avg of observed rewards per shot - [0:shotNum]    
				resultsDict[aName] = {"plot_y": dict(), "estimates": dict(),\
									"mse_percent_pskills": dict(),\
									"percentsEstimatedPs": dict(),\
									"percentTrueP": 0.0, "avg_rewards": [], "num_exps": 1}

				resultsDict[aName]["true_rewards"]: []

				#for True Noise (will be obtained using std)
				resultsDict[aName]["plot_y"]['tn'] = [0.0]


				# for the rest of the methods
				for m in methods:

					if m == "tn":
						resultsDict[aName]["plot_y"][m] = [0.0] * numStates
						
						resultsDict[aName]["estimates"][m] = [0.0] * numStates

						if actualMethodsOnExpsObtainedFlag == False:
							actualMethodsOnExps.append(m)
					else:

						try:
							# if the method exists on the results file, load
							testLoadMethod = results[m]
						except:
							# print("Skipping:",m)
							# code.interact(local=locals())
							continue

						# If TBA/BM method, need to account for possible different betas
						if "BM" in m:
							tempM, beta, tt = getInfoBM(m)

							# To initialize once
							if tt not in resultsDict[aName]["plot_y"]:
								resultsDict[aName]["plot_y"][tt] = {}
								resultsDict[aName]["estimates"][tt] = {}

							if tempM not in resultsDict[aName]["plot_y"][tt]:
								resultsDict[aName]["plot_y"][tt][tempM] = {}
								resultsDict[aName]["estimates"][tt][tempM] = {}

							if beta not in resultsDict[aName]["plot_y"][tt][tempM]:
								resultsDict[aName]["plot_y"][tt][tempM][beta] = [0.0] * numStates
								resultsDict[aName]["estimates"][tt][tempM][beta] = [0.0] * numStates

						else:
							resultsDict[aName]["plot_y"][m] = [0.0] * numStates
							resultsDict[aName]["estimates"][m] = [0.0] * numStates

			
						# to make it only happen once
						if actualMethodsOnExpsObtainedFlag == False:
								actualMethodsOnExps.append(m)

								if "OR" in m and "-estimatesMidGame" in m:
									actualMethodsNames.append("OR-MidGame")
								elif "OR" in m and "-estimatesFullGame" in m:
									actualMethodsNames.append("OR-FullGame")
								elif "OR" in m:
									actualMethodsNames.append("OR")

								# To add just once as can possibly appear multiple 
								# times because of diff betas
								elif "BM-MAP" in m and "BM-MAP" not in actualMethodsNames:
									actualMethodsNames.append("BM-MAP")
								elif "BM-EES" in m and "BM-EES" not in actualMethodsNames:
									actualMethodsNames.append("BM-EES")

								elif "JT-FLIP-MAP" in m:
									actualMethodsNames.append("JT-FLIP-MAP")
								elif "JT-FLIP-EES" in m:
									actualMethodsNames.append("JT-FLIP-EES")

								# Must come before "JT-QRE" method comparison
								# since ("JT-QRE" in "NJT-QRE") results in true as well 
								elif "NJT-QRE-MAP" in m and "xSkills" in m:
									actualMethodsNames.append("NJT-QRE-MAP")
								elif "NJT-QRE-EES" in m and "xSkills" in m:
									actualMethodsNames.append("NJT-QRE-EES")
								
								elif "JT-QRE-MAP" in m and "xSkills" in m:
									actualMethodsNames.append("JT-QRE-MAP")
								elif "JT-QRE-EES" in m and "xSkills" in m:
									actualMethodsNames.append("JT-QRE-EES")

					

				actualMethodsOnExpsObtainedFlag = True
				
				# to store the avg reward of each experiment - for each one of the agents
				resultsDict[aName]["avg_rewards"] = [0.0] * numStates



				resultsDict[aName]["ev_intendedAction"] = [0.0] * numStates
				resultsDict[aName]["percent_true_reward"] = [0.0] * numStates


				# mean observed reward of experiment
				resultsDict[aName]["mean_observed_reward"] = 0.0

				# mean true reward of experiment
				resultsDict[aName]["mean_true_reward"] = 0.0

				resultsDict[aName]["mean_rs_reward_per_exp"] = 0.0
				resultsDict[aName]["mean_rs_reward_"] = []

				# to store the true reward of each experiment - for each one of the agents
				resultsDict[aName]["true_rewards"] = [0.0] * numStates
				resultsDict[aName]["mean_value_intendedAction"] = 0.0
				resultsDict[aName]["mean_random_reward_mean_vs"] = 0.0
				

				if seenAgentType not in seenAgents:
					seenAgents.append(seenAgentType)

			# mean observed reward of experiment
			resultsDict[aName]["mean_observed_reward"] += np.nanmean(results['observed_rewards'])

			# mean true reward of experiment
			resultsDict[aName]["mean_true_reward"] += np.nanmean(results['true_rewards'])
			resultsDict[aName]["mean_value_intendedAction"]  += np.nanmean(results["valueIntendedActions"])
			resultsDict[aName]["mean_random_reward_mean_vs"]  += np.nanmean(results["meanAllVsPerState"])

			# if "Flip" in aName and str(150.5) in aName:
				# code.interact("main...", local=dict(globals(), **locals()))
			

			#######################################################################################################
			# Computing MSE
			########################################################################################################
			
			x = float(results['xskill'])
			
			if "Random" not in results['agent_name']:
				p = float(param)


			for m in actualMethodsOnExps: 

				# skip TN method since there's not info about it in the results file
				if m == "tn":
					continue

				# Tries to load the results/info for the given method if present in results file. Otherwise, keep going
				try:
					mxs = results[m]
				except:
					continue

				if "-pSkills" in m:
					useP = True
				else:
					useP = False


				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)


					# For each observation/state
					for mxi in range(len(mxs)):	
						merr = (mxs[mxi] - x) ** 2.0  #MSE ERROR

						if len(resultsDict[aName]["plot_y"][tt][tempM][beta]) < mxi + 1: 
							# print "if: mxi: ", mxi
							resultsDict[aName]["plot_y"][tt][tempM][beta].append(merr)
						else:
							# print "else: mxi: ", mxi
							resultsDict[aName]["plot_y"][tt][tempM][beta][mxi] += merr

						# store estimate per method & per obs
						resultsDict[aName]["estimates"][tt][tempM][beta][mxi] += mxs[mxi]

				# Rest of the methods
				else:

					###################################################
					# Find MSE of actual estimate
					###################################################
					
					# For each observation/state
					for mxi in range(len(mxs)):

						if useP and "Random" not in aName:
							merr = (mxs[mxi] - p) ** 2.0  #MSE ERROR

						else:
							merr = (mxs[mxi] - x) ** 2.0  #MSE ERROR

						try:
							if len(resultsDict[aName]["plot_y"][m]) < mxi + 1: 
								# print "if: mxi: ", mxi
								resultsDict[aName]["plot_y"][m].append(merr)
							else:
								# print "else: mxi: ", mxi
								resultsDict[aName]["plot_y"][m][mxi] += merr
						except:
							pass
							#code.interact("...", local=dict(globals(), **locals()))

						# Store estimate per method & per obs
						resultsDict[aName]["estimates"][m][mxi] += mxs[mxi]
					
					###################################################
						

					###################################################
					# Convert from estimate to rationality percentage
					###################################################
					
					if m == "tn" or "xSkills" in m:
						continue

					# Skip OR & TBA
					if "pSkills" not in m:
						continue


					# To determine whether to use JT's or NJT's current xskill estimate 
					if "NJT" in m:
						mm = "NJT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"
					else:
						mm = "JT-QRE-EES-" + str(numHypsX[0]) + "-" + str(numHypsP[0]) + "-xSkills"


					if m not in resultsDict[aName]["percentsEstimatedPs"]:
						resultsDict[aName]["percentsEstimatedPs"][m] = {}


					resultsDict[aName]["percentsEstimatedPs"][m][resultsDict[aName]["num_exps"]] = [0.0] * numStates
					resultsDict[aName]["percentsEstimatedPs"][m]["averaged"] = [0.0] * numStates
					resultsDict[aName]["mse_percent_pskills"][m] = [0.0] * numStates
					
					# For each observation/state
					for mxi in range(len(mxs)):
						
						# Use estimated xskill and not actual true one
						# WHY? estimatedX and not trueX?? because "right" answer is not available
						xStr = float(resultsDict[aName]["plot_y"][mm][mxi])

						# find proper bucket for current x
						bucket1, bucket2 = getBucket(bucketsX,minMaxX,xStr)


						# Get pskill estimate of current method - estimatedP
						estimatedP = mxs[mxi]


						# Convert estimatedP to corresponding % of rand max
						if bucket2 != None:
							prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
							prat2 = np.interp(estimatedP,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

							prat = np.interp(estimatedP, [prat1], [prat2])
							percent_estimatedP = prat
						# edges/extremes case
						else:
							# using one of the functions for now
							prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

							percent_estimatedP = prat1


						# Save info
						resultsDict[aName]["percentsEstimatedPs"][m][resultsDict[aName]["num_exps"]][mxi] = percent_estimatedP

					###################################################
							

			########################################################################################################


			# Compute avg observed reward and resampled rewards
			for mxi in range(numStates):
				# Compute the mean of the rewards received up til this point
				resultsDict[aName]["avg_rewards"][mxi] += sum(results['observed_rewards'][0:mxi]) / (1.0 * (mxi+1)) 
				resultsDict[aName]["true_rewards"][mxi] += sum(results['true_rewards'][0:mxi]) / (1.0 * (mxi+1)) 


				# resultsDict[aName]["ev_intendedAction"][mxi] += sum(results["evIntendedActions"][0:mxi])/(1.0 * (mxi + 1))
				# resultsDict[aName]["percent_true_reward"][mxi] += (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"])
				# percentOfTrueReward = (resultsDict[a]["mean_value_intendedAction"] / resultsDict[a]["mean_true_reward"])

				# resultsDict[aName]["mean_rs_reward_"].append(sum(results["rs_rewards"][mxi]) / (len(results["rs_rewards"][mxi]) * 1.0))

			# get mean rs per exp
			# resultsDict[aName]["mean_rs_reward_per_exp"] = sum( resultsDict[aName]["mean_rs_reward_"])/(len( resultsDict[aName]["mean_rs_reward_"]) * 1.0)

			  
			# compute the mean of the actual estimates - for the different observations - EVERY
			# for m in methods:
			#     for mxi in range(numStates): 
			#         resultsDict[aName]["estimates"][m][mxi] += sum(results[m][0:mxi]) / (1.0 * (mxi+1)) 

				# if m == "OR-17" and aName == "Target":
					# print resultsDict["Target"]["plot_y"]["OR-17"][:10]


			# Compute TN - Plot out the estimate from numpy std on the true difference
			td = results['true_diffs']
			
			for ti in range(len(td)):

				# Get prediction
				est = np.std(td[:ti],ddof=1)

				# Get error 
				er = (est-x) ** 2.0 #MSE ERROR

				if len(resultsDict[aName]["plot_y"]['tn']) < ti + 1:
					resultsDict[aName]["plot_y"]['tn'].append(er)
				else:
					resultsDict[aName]["plot_y"]['tn'][ti] += er  

				resultsDict[aName]["estimates"]["tn"][ti] += est                        


			# resampled_rewards = results['rs_rewards']

			
			'''
			print '*******\n\n'
			print ' Average Observed Reward: ', np.mean(results['observed_rewards'])
			print ' TRUE EXECUTION SKILL LEVEL: ', x
			print '*******\n\n'
			'''
		
			total_num_exps += 1


	print('Compiled results for', total_num_exps, 'experiments')
	

	#############################################################################3
	# Store processed results
	#############################################################################3

	if args.processingAgents != "":
		saveAs = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "resultsDictInfo" + processingAgents[0]
	else:
		saveAs = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "resultsDictInfo"

	# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
	with open(saveAs, "wb") as outfile:
		pickle.dump(resultsDict, outfile)

	otherInfo = {}
	otherInfo["actualMethodsNames"] = actualMethodsNames
	otherInfo["actualMethodsOnExps"] = actualMethodsOnExps
	otherInfo["actualMethodsOnExpsObtainedFlag"] = actualMethodsOnExpsObtainedFlag
	otherInfo["typeTargetsList"] = typeTargetsList

	otherInfo["methods"] = methods
	otherInfo['numHypsX'] = numHypsX
	otherInfo['numHypsP'] = numHypsP
	otherInfo["numObservations"] = numStates
	otherInfo["seenAgents"] = seenAgents
	otherInfo["domain"] = domain
	otherInfo["mode"] = mode
	otherInfo["betas"] = betas

	otherInfo["result_files"] = result_files


	if args.processingAgents != "":
		saveAs2 = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "otherInfo" + processingAgents[0]
	else:
		saveAs2 = args.resultsFolder + os.path.sep + "plots" + os.path.sep + "otherInfo"

	with open(saveAs2, "wb") as outfile:
		pickle.dump(otherInfo, outfile)

	#############################################################################3
		   

	##################################################
	# PARAMETERS FOR PLOTS
	##################################################

	matplotlib.rcParams.update({'font.size': 14})
	matplotlib.rcParams.update({'legend.fontsize': 14})
	matplotlib.rcParams["axes.labelweight"] = "bold"
	matplotlib.rcParams["axes.titleweight"] = "bold"

	##################################################



	# To make font of title & labels bold
	matplotlib.rc('text', usetex=True)

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
	methodsColors = {'OR'+"-"+str(numHypsX[0]): "tab:purple",\
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "tab:gray", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "tab:pink", \


					'BM-MAP'+"-"+str(numHypsX[0]): "tab:brown", \
					'BM-EES'+"-"+str(numHypsX[0]): "tab:cyan",\

					'BM-MAP'+"-"+str(numHypsX[0])+"-OptimalTargets": "xkcd:teal", \
					'BM-EES'+"-"+str(numHypsX[0])+"-OptimalTargets": "xkcd:darkgreen",\

					'BM-MAP'+"-"+str(numHypsX[0])+"-DomainTargets": "xkcd:fuchsia", \
					'BM-EES'+"-"+str(numHypsX[0])+"-DomainTargets": "xkcd:darkblue",\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:red",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:red" ,\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:green" ,\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:green" ,\

					"JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:gray" ,\
					"JT-FLIP-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:gray",\

					"JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:olive",\
					"JT-FLIP-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:olive",\

					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:blue",\
					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:blue" ,\

					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "tab:orange",\
					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "tab:orange"}



	#######################################################################
	#######################################################################
	#######################################################################

	# SIMPLIFY PROCESSING BY LOADING METHODS NAMES FROM EXP!!!

	#######################################################################
	#######################################################################
	#######################################################################



	# Compute Mean Squared Error
	computeMSE(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP)

	computeMeanEstimates(resultsDict)


	# code.interact("...", local=dict(globals(), **locals()))

	#############################################################################################################################################


	# percentOfTrueRewardPerAgentAndRationalityParam, functionsPerAgentType = plotPercentTrueRewardObtainedPerAgentType(resultsDict, seenAgents, resultsFolder,seenAgents, domain)
	# contourPlotPercentTrueRewardObtainedPerAgentType(resultsDict, agentTypes, resultsFolder, seenAgents, domain)

	
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


	#'''
	global methodNamesPaper
	methodNamesPaper = {'OR'+"-"+str(numHypsX[0]): "OR", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesMidGame": "OR-MidGame", \
					'OR'+"-"+str(numHypsX[0])+"-estimatesFullGame": "OR-FullGame", \
					
					'BM-EES'+"-"+str(numHypsX[0]): 'TBA-EES',\
					'BM-MAP'+"-"+str(numHypsX[0]): 'TBA-MAP',\

					'BM-EES'+"-"+str(numHypsX[0])+"-OptimalTargets": 'TBA-EES-OptimalTargets',\
					'BM-MAP'+"-"+str(numHypsX[0])+"-OptimalTargets": 'TBA-MAP-OptimalTargets',\
					'BM-EES'+"-"+str(numHypsX[0])+"-DomainTargets": 'TBA-EES-DomainTargets',\
					'BM-MAP'"-"+str(numHypsX[0])+"-DomainTargets": 'TBA-MAP-DomainTargets',\

					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS-EES",\
					"JT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS-EPS",\

					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills":  "JEEDS-MAP",\
					"JT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills":  "JEEDS-MAP",\

					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "MEEDS-EES",\
					"NJT-QRE-EES"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "MEEDS-EPS",\

					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-xSkills": "MEEDS-MAP",\
					"NJT-QRE-MAP"+"-"+str(numHypsX[0])+"-"+str(numHypsP[0])+"-pSkills": "MEEDS-MAP"}
	#'''



	
	makeFolder(args.resultsFolder,"BETAS")

	# code.interact("main...", local=dict(globals(), **locals()))


	### PLOTS
	
	# plots with random rewards - only for 1d for now
	# percentOfRandMaxRewardPerAgentAndRationalityParam, functionsPerAgentTypeRandMax = plotPercentRandMaxRewardObtainedPerAgentType(resultsDict, seenAgents, args.resultsFolder,seenAgents, domain, pconfPerXskill)
	# contourPlotPercentRandMaxRewardObtainedPerAgentType(resultsDict, agentTypes, args.resultsFolder, seenAgents, domain)


	# computes percentTrueP
	plotPercentRandMaxRewardObtainedPerXskillPerAgentType(resultsDict, agentTypes, args.resultsFolder, seenAgents, domain, pconfPerXskill)


	# Converting from estimate to percent RandMax Reward
	computeMSEPercentPskillsMethods(resultsDict, actualMethodsOnExps, pconfPerXskill, numStates, numHypsX, numHypsP, domain)


	# Plots MSE for pskill methods - in percent terms
	plotMSEPercentPerAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, domain)



	
	# plotScatterPercentsPerAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, domain, numStates)
	# plotScatterPercentsPerAgentTypesAndXskillBuckets(resultsDict, actualMethodsOnExps, args.resultsFolder, domain, numStates, pconfPerXskill, seenAgents)




	# computeAndPlotEstimatesPercentPerXskillBucketsPerMethodsAndAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, domain, seenAgents)
	
	# computeAndPlotMSEPercentPerPsAndMethodsAndAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, functionsPerAgentTypeRandMax, domain)


	# '''
	
	# plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndXskillBuckets(domain, resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates)
	# plotPercentTimesOnBucketObtainedPerAgentTypeAndMethodAndPskillBuckets(domain, resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates)

	# plotPercentTimesDistributionXskillBuckets(domain, resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates)
	# plotPercentTimesDistributionPskillBuckets(domain, resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates)
	# '''

	# code.interact("...", local=dict(globals(), **locals()))

	#################################################
	# Assuming same beta for all agent types for now
	#################################################
	#givenBeta = 0.735
	givenBeta = 0.99
	#################################################
	

	###################################### FOR MSE #####################################

	# create plot for each one of the methods - per agent - against TN 
	#plotMSEPerMethodPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder)

	# create plot for each agent - all methods - MSE and a given hypothesis
	########################
	
	computeAndPlotMSEAcrossAllAgentsTypesAllMethods(resultsDict, actualMethodsNames, actualMethodsOnExps, args.resultsFolder, seenAgents, numStates, domain,betas,givenBeta, makeOtherPlots=True)
	
	computeAndPlotMSEAcrossAllAgentsPerMethod(resultsDict, actualMethodsNames, actualMethodsOnExps, args.resultsFolder, seenAgents, numStates, domain,betas,givenBeta)
	
	########################
	
	

	# '''
	################
	# Pending update
	# plotMSExSkillsPerBucketsPerAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, domain, numStates, numHypsX, numHypsP)
	# plotMSEpSkillsPerBucketsPerAgentTypes(resultsDict, actualMethodsOnExps, args.resultsFolder, domain, numStates, numHypsX, numHypsP)
	################
	# '''


	# plotMSEAllXSkillMethodsPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, hyp)
	
	# plotMSEAllPSkillMethodsPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, hyp)


	# computeAndPlotMSEAcrossAllAgentsTypesAndRationalityParamsAllMethods(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, hyp, seenAgents, numStates)


	plotContourMSE_xSkillpSkillPerAgentTypePerMethod(resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates, domain)

	plotContourEstimates_xSkillpSkillPerAgentTypePerMethod(resultsDict, seenAgents, actualMethodsOnExps, args.resultsFolder, numStates, domain)

	# plotMSEAllMethodsSamePlotPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder)


	###################################################################################



	################################## FOR ESTIMATES ##################################

	# '''
	
	# uncomment
	# plotEstimateAllMethodsPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, givenBeta)
	
	# uncomment
	plotEstimateAllMethodsSamePlotPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, givenBeta)

	# uncomment
	# plotEstimateAllBetasSamePlotPerAgentBAR(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder,betas)
	
	# uncomment
	# plotEstimateAllBetasSamePlotPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder)
	
	# code.interact("...", local=dict(globals(), **locals()))

	# '''


	#'''
	# plotMeanEstimatesAllAgentsSamePlotPerMethodAndPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, agentTypes)
	#


	################################################
	plotMeanEstimatesSelectedAgentsSamePlotPerMethodAndPerAgentType(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder,agentTypes,seenAgents,givenBeta)
	
	# code.interact("main()...", local=dict(globals(), **locals()))
	################################################
	
	###################################################################################


	

	################################### FOR REWARDS ###################################

	# create plot for the avg mean rewards & true rewards for each one of the different agents
	# to compare them
	# plotMeanAVGAndTrueRewardsPerAgent(resultsDict, args.resultsFolder)

	# '''
	# Create plot for the avg mean rewards - for each one of the different agents
	# plotMeanAVGRewardsPerAgent(resultsDict, agentTypes, args.resultsFolder)
	# '''

	# create plot for the avg mean rewards  - all agent same plot
	# plotMeanAVGRewardsAllAgents(resultsDict, args.resultsFolder)

	plotRewardsVSagentType(resultsDict, numHypsX, numHypsP, args.resultsFolder, domain)

	plotEVintendedVSagentType(resultsDict, numHypsX, numHypsP, args.resultsFolder, domain)

	###################################################################################


	# plotMeanEstimatesForDiffPskillsPerMethodAndAgentType(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, agentTypes, numStates)
	
	#plotMeanEstimatesForDiffXskillPskillsPerMethodAndAgentTypeAndGivenStates(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, agentTypes, numStates, domain)


	args.resultsFolder += os.path.sep + "plots"

	##################### FOR RATIONALITY PARAMETER - ALL AGENTS ######################
	'''

	for agentType in ["Flip", "Tricker", "Bounded"]:

		# Only continue to create the corresponding plots if we have seen experiments for such agent type
		if agentType in seenAgents:

			makeFolder2(args.resultsFolder, agentType)

			#plotMSEAllRationalityParamsAllMethods(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, hyp, agentType)

			#plotMSEAllRationalityParamsPerMethods(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder, hyp, agentType)


			#plotRationalityParamsVsSkillEstimatePerMethod(resultsDict, numHypsX, numHypsP, actualMethodsOnExps, args.resultsFolder, agentType)


			plotRationalityParamsVSRewards(resultsDict, numHypsX, numHypsP, args.resultsFolder, agentType)

			plotRationalityParamsVsEVIntended(resultsDict, numHypsX, numHypsP, args.resultsFolder, agentType)


			# plotRationalityParamsVSRewardsPerXSkill(resultsDict, numHypsX, numHypsP, args.resultsFolder, agentType)
			
			# plotXSkillsVSRewardsPerRationalityParams(resultsDict, numHypsX, numHypsP, args.resultsFolder, agentType)

	'''
	###################################################################################


	# Close all remaining figures
	plt.close("all")


	#code.interact("End.", local=dict(globals(), **locals()))



 