from matplotlib import rcParams,rc
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

import argparse,sys,os
import math 
import json,pickle
import numpy as np
import pandas as pd
import code,csv
from copy import deepcopy

import dataframe_image as dfi

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors

from utils import *


from pybaseball import playerid_reverse_lookup, cache

import six

# from arviz import hdi

'''
def makeCSV(resultsDict,methods,label,resultsFolder):
	
	makeFolder(resultsFolder,"CSV")

	finalEstimates = []

	for pitcherID in resultsDict.keys():

		for pitchType in resultsDict[pitcherID]:

			estimatesAcrossMethods = []
			
			numObs = resultsDict[pitcherID][pitchType]["numObservations"]

			for method in methods:

				est = None

				if "BM" in method:  
					tempM, beta, tt = getInfoBM(method)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
						est = resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta][-1]
				elif "BM" not in method:
					if resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
						est = resultsDict[pitcherID][pitchType]["estimates"][method][-1]
						
				if est != None:
					estimatesAcrossMethods.append(est)

			finalEstimates.append([pitcherID,pitcherNames[pitcherID],pitchType,numObs] + estimatesAcrossMethods)


	saveTo = open(f"{args.resultsFolder}{os.sep}plots{os.sep}CSV{os.sep}finalEstimatesInfo-{label}.csv","w")

	csvWriter = csv.writer(saveTo)

	columns = ["ID","Name","PitchType","NumObservations"] + methods

	csvWriter.writerow(columns)
	
	for i in range(len(finalEstimates)):
		csvWriter.writerow(finalEstimates[i])

	saveTo.close()

	# code.interact("...", local=dict(globals(), **locals()))
'''

def topXskillHyps(resultsDict,methodsAllProbs,resultsFolder,topX=5):

	makeFolder(resultsFolder,"XskillMethods-TopHyps")

	# 2.0 inches | 0.17 feet
	startX_Estimator = 0.17
	# 33.72 inches | 2.81 feet
	stopX_Estimator = 2.81

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	xskills = np.array(np.concatenate((np.linspace(startX_Estimator,1.0,num=60),np.linspace(1.00+delta,stopX_Estimator,num=6))))
	

	tops = {}

	for ma in methodsAllProbs:

		if "pSkill" in ma:
			continue

		# To save info only once
		# As BM & NJT allProbs appear twice (for MAP & EES)
		# And probs will be the same 
		if "NJT" in ma or "BM-MAP" in ma:
			continue

		for pitcherID in resultsDict.keys():

			for pitchType in resultsDict[pitcherID]:

				# Skip experiments with prev JT naming convention 
				# and missing all probs for diff bbetas
				# for now
				if resultsDict[pitcherID][pitchType]["allProbs"] == {}:
					continue

				if resultsDict[pitcherID][pitchType]["allProbs"][ma] == {}:
					continue
			
				if ma not in tops:
					tops[ma] = {}

				if pitcherID not in tops[ma]:
					tops[ma][pitcherID] = {}
	
				if "NJT" in ma:
					for e in range(len(resultsDict[pitcherID][pitchType]["allProbs"][ma][-1])):
						resultsDict[pitcherID][pitchType]["allProbs"][ma][-1][e] = resultsDict[pitcherID][pitchType]["allProbs"][ma][-1][e][0]

			
				lastProbs = np.array(resultsDict[pitcherID][pitchType]["allProbs"][ma][-1])
				

				actualMethod = ma.split("-allProbs")[0]

				if actualMethod == "JT-QRE":
					m1 = "JT-QRE-MAP-66-66-xSkills"
					m2 = "JT-QRE-EES-66-66-xSkills"
				elif "BM" not in actualMethod and "NJT" not in actualMethod:
					actualMethod += "-xSkills"
					m1 = actualMethod.replace("-QRE-","-QRE-MAP-66-66-")
					m2 = actualMethod.replace("-QRE-","-QRE-EES-66-66-")
				elif "NJT" in actualMethod:
					m1 = actualMethod.replace("EES","MAP")
					m2 = actualMethod

				
				if "BM" in ma: 
					print(ma)
					tempM, beta, tt = getInfoBM(actualMethod)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
						estimate1 = resultsDict[pitcherID][pitchType]["estimates"][tt]["BM-MAP"][beta][-1]
						estimate2 = resultsDict[pitcherID][pitchType]["estimates"][tt]["BM-EES"][beta][-1]
				elif "BM" not in ma:
					if "NJT" not in ma and "EES" not in actualMethod and "MAP" not in actualMethod:
						if "EES" not in m2:
							toCheck = m2.replace("-QRE-","-QRE-EES-")
						else:
							toCheck = m2
					else:
						toCheck = actualMethod

					if resultsDict[pitcherID][pitchType]["estimates"][toCheck] != {}:
						estimate1 = resultsDict[pitcherID][pitchType]["estimates"][m1][-1]
						estimate2 = resultsDict[pitcherID][pitchType]["estimates"][m2][-1]
					else:
						estimate1 = None
						estimate2 = None
						
				numObs = resultsDict[pitcherID][pitchType]["numObservations"]


				if "JT-QRE" in ma and "allProbs" in ma and "NJT" not in ma:
					topProbsInd = np.argsort(lastProbs,axis=None)[::-1][:topX]
					topProbs = lastProbs.flatten()[topProbsInd]
					iis,jjs = np.unravel_index(topProbsInd,lastProbs.shape)

					if "MinLambda" in ma:
						try:
							minLambda = float(ma.split("MinLambda-")[1].split("-allProbs")[0])
						except:
							minLambda = float(ma.split("MinLambda")[1].split("-allProbs")[0])

						pskills = np.logspace(minLambda,3.6,66)
					else:
						pskills = np.logspace(-3,3.6,66)

					topHypsX = xskills[iis]
					topHypsP = pskills[jjs]
					tops[ma][pitcherID][pitchType] = [topHypsX,topHypsP,topProbs,estimate1,estimate2,numObs]
				
				else:				
					topProbsInd = np.argsort(lastProbs)[::-1][:topX]
					topProbs = lastProbs[topProbsInd]
			
					topHyps = xskills[topProbsInd]
					tops[ma][pitcherID][pitchType] = [topHyps,topProbs,estimate1,estimate2,numObs]


		for m in tops:

			if "NJT" in m:
				tempM = m.replace("-QRE-EES-","-QRE-")
			elif "BM" in m:
				tempM = m.replace("BM-EES-","-BM-")
			else:
				tempM = m

			with open(f"{resultsFolder}{os.sep}plots{os.sep}XskillMethods-TopHyps{os.sep}Top{topX}-AcrossAll-Method{tempM}.txt","w") as outfile:
			
				outfile.write(f"METHOD: {tempM}\n\n")
				
				for pid in tops[m]:
					for pt in tops[m][pid]:
						
						if "JT-QRE" in m and "allProbs" in m and "NJT" not in m:
							outfile.write(f"Pitcher: {pitcherNames[pid]} ({pid}) | PitchType: {pt} | Num Observations: {tops[m][pid][pt][5]} |\nFinal Estimate MAP: {round(tops[m][pid][pt][3],4)} | Final Estimate EES: {round(tops[m][pid][pt][4],4)}\n")
						else:
							outfile.write(f"Pitcher: {pitcherNames[pid]} ({pid}) | PitchType: {pt} | Num Observations: {tops[m][pid][pt][4]} |\nFinal Estimate MAP: {round(tops[m][pid][pt][2],4)} | Final Estimate EES: {round(tops[m][pid][pt][3],4)}\n")

						for i in range(topX):

							if "JT-QRE" in m and "allProbs" in m and "NJT" not in m:
								outfile.write(f"X: {round(tops[m][pid][pt][0][i],4)} - P: {round(tops[m][pid][pt][1][i],4)}-> Prob: {round(tops[m][pid][pt][2][i],4)}\n")
							else:
								outfile.write(f"X: {round(tops[m][pid][pt][0][i],4)}-> Prob: {round(tops[m][pid][pt][1][i],4)}\n")

						outfile.write("\n\n")


	# code.interact("topXskillHyps()...", local=dict(globals(), **locals()))


def compareToGivenMethod(resultsDict,methods,resultsFolder,compareTo="OR",thresObservations=50,thresSame=0.10):

	makeFolder(resultsFolder,"ComparisonXskillMethods")

	countsInfo = {}

	keyCompareTo = ""

	for m in methods:
		if compareTo in m:
			keyCompareTo = m


	for method in methods:

		if keyCompareTo in method:
			continue

		if "BM" in method or "xSkill" in method:

			countsInfo[method] = {}

			total = 0
			countBetter = 0
			countWorse = 0
			countSame = 0

			for pitcherID in resultsDict.keys():

				for pitchType in resultsDict[pitcherID]:

					# If exp has enough observations
					if resultsDict[pitcherID][pitchType]["numObservations"] > thresObservations:

						est = None

						if "BM" in method:  
							tempM, beta, tt = getInfoBM(method)
							if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
								est = resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta][-1]
						elif "BM" not in method:
							if resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
								est = resultsDict[pitcherID][pitchType]["estimates"][method][-1]
									
						if est != None:

							estOR = resultsDict[pitcherID][pitchType]["estimates"][keyCompareTo][-1]
							thres = estOR*0.10

							# Considered same if within threshold
							if est >= estOR-thres and est <= estOR+thres:
								countSame += 1
							# Worse than OR estimate?
							if est > estOR+thres:
								countWorse += 1
							# Better than OR estimate?
							else:
								countBetter += 1

						total += 1

			countsInfo[method]["total"] = total
			countsInfo[method]["countBetter"] = countBetter
			countsInfo[method]["countWorse"] = countWorse
			countsInfo[method]["countSame"] = countSame

	with open(f"{resultsFolder}{os.sep}plots{os.sep}ComparisonXskillMethods{os.sep}XskillMethodsVs{compareTo}-thresObservations{thresObservations}-thresSame{thresSame}-AcrossAll.txt","w") as outfile:

		outfile.write(f"{'METHOD':65s}\t{'SAME':10s}{'BETTER':10s}{'WORSE':10s}{'TOTAL':10s}\n")
		
		for m in countsInfo:
			outfile.write(f"{m:65s}{countsInfo[m]['countSame']:10d}{countsInfo[m]['countBetter']:10d}{countsInfo[m]['countWorse']:10d}{countsInfo[m]['total']:10d}\n")


	# code.interact("...", local=dict(globals(), **locals()))

	# PER PITCH TYPE

	# Load pdfs previously used
	with open(f"{resultsFolder}{os.sep}info.pkl",'rb') as handle:
		info = pickle.load(handle)
	

	allPitchTypes = info["allPitchTypes"]

	countsInfo = {}

	for method in methods:

		if keyCompareTo in method:
			continue

		if "BM" in method or "xSkill" in method:

			countsInfo[method] = {}


			for pitchType in allPitchTypes:

				countsInfo[method][pitchType] = {}

				total = 0
				countBetter = 0
				countWorse = 0
				countSame = 0


				for pitcherID in resultsDict.keys():

					if pitchType not in resultsDict[pitcherID]:
						continue

					# If exp has enough observations
					if resultsDict[pitcherID][pitchType]["numObservations"] > thresObservations:

						est = None

						if "BM" in method:  
							tempM, beta, tt = getInfoBM(method)
							if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
								est = resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta][-1]
						elif "BM" not in method:
							if resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
								est = resultsDict[pitcherID][pitchType]["estimates"][method][-1]
									
						if est != None:

							estOR = resultsDict[pitcherID][pitchType]["estimates"][keyCompareTo][-1]
							thres = estOR*0.10

							# Considered same if within threshold
							if est >= estOR-thres and est <= estOR+thres:
								countSame += 1
							# Worse than OR estimate?
							if est > estOR+thres:
								countWorse += 1
							# Better than OR estimate?
							else:
								countBetter += 1

						total += 1


				countsInfo[method][pitchType]["total"] = total
				countsInfo[method][pitchType]["countBetter"] = countBetter
				countsInfo[method][pitchType]["countWorse"] = countWorse
				countsInfo[method][pitchType]["countSame"] = countSame


	for pitchType in allPitchTypes:
		
		with open(f"{resultsFolder}{os.sep}plots{os.sep}ComparisonXskillMethods{os.sep}XskillMethodsVs{compareTo}-thresObservations{thresObservations}-thresSame{thresSame}-PitchType{pitchType}.txt","w") as outfile:

			outfile.write(f"{'METHOD':65s}\t{'SAME':10s}{'BETTER':10s}{'WORSE':10s}{'TOTAL':10s}\n")
			
			for m in countsInfo:
				outfile.write(f"{m:65s}{countsInfo[m][pitchType]['countSame']:10d}{countsInfo[m][pitchType]['countBetter']:10d}{countsInfo[m][pitchType]['countWorse']:10d}{countsInfo[m][pitchType]['total']:10d}\n")


def computeCredibleIntervals(resultsDict,methodsAllProbs,resultsFolder):

	folder = "PosteriorProbs"
	makeFolder(resultsFolder,folder)

	# 2.0 inches | 0.17 feet
	startX_Estimator = 0.17
	# 33.72 inches | 2.81 feet
	stopX_Estimator = 2.81

	# 0.5 inches | 0.0417 feet
	delta = 0.0417


	xskills = list(np.concatenate((np.linspace(startX_Estimator,1.0,num=60),np.linspace(1.00+delta,stopX_Estimator,num=6))))
	

	# BM = 1D -> xskill probs
	# NJT = 1D -> xskill probs, pskill probs
	# JT = 2D -> [xskill probs X pskill probs]

	for pitcherID in resultsDict.keys():

		makeFolder(resultsFolder,f"{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}")

		for pitchType in resultsDict[pitcherID]:

			makeFolder(resultsFolder,f"{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}")

			for ma in methodsAllProbs:

				# To save info only once
				# As BM & NJT allProbs appear twice (for MAP & EES)
				# And probs will be the same 
				if "NJT-QRE-MAP" in ma or "BM-MAP" in ma:
					continue

				# Skip experiments with prev JT naming convention 
				# and missing all probs for diff bbetas
				# for now
				if resultsDict[pitcherID][pitchType]["allProbs"] == {}:
					continue

				if resultsDict[pitcherID][pitchType]["allProbs"][ma] == {}:
					continue

				fig = plt.figure(figsize = (12,8))

				# BM & NJT = 1D
				if "BM" in ma or "NJT" in ma:

					lastProbs = resultsDict[pitcherID][pitchType]["allProbs"][ma][-1]
					
					# low, high = hdi(lastProbs)

					# p25 = np.percentile(lastProbs,q=[2.5],axis=0)
					# p975 = np.percentile(lastProbs,q=[97.5],axis=0)

					# code.interact("...", local=dict(globals(), **locals()))
					

					if "BM" in ma or "xSkills" in ma:
						lookingAt = xskills
						plt.xlabel("Execution Skill")
					else:
						lookingAt = pskills
						plt.xlabel("Planning Skill")


					# low = np.where(lookingAt == low)
					# high = np.where(lookingAt == high)
					
					plt.plot(lookingAt,lastProbs)
					# plt.vline(low)
					# plt.vline(high)

					plt.ylabel("Posterior Probability")
				
				else:

					if "MinLambda" in ma:
						try:
							minLambda = float(ma.split("MinLambda-")[1].split("-allProbs")[0])
						except:
							minLambda = float(ma.split("MinLambda")[1].split("-allProbs")[0])

						pskills = np.logspace(minLambda,3.6,66)
					else:
						pskills = np.logspace(-3,3.6,66)

					lastProbs = resultsDict[pitcherID][pitchType]["allProbs"][ma][-1]


					xx,yy = np.meshgrid(xskills,pskills,indexing="ij")
					cs = plt.contourf(xx,yy,lastProbs)
					cbar = plt.colorbar(cs)
					plt.xlabel("Execution Skill")
					plt.ylabel("Planning Skill")

				plt.title(f"Final Posterior Probability: Method: {ma.split('-allProbs')[0].replace('-EES-','-')}")
				plt.savefig(f"{resultsFolder}{os.sep}plots{os.sep}{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}{os.sep}pitcherID{pitcherID}-pitchType{pitchType}-{ma.split('-allProbs')[0].replace('-EES-','-')}.png", bbox_inches='tight')
				plt.clf()
				plt.close()

			# code.interact("...", local=dict(globals(), **locals()))


# From: https://stackoverflow.com/questions/26678467/export-a-pandas-dataframe-as-a-table-image
def render_mpl_table(data,saveAt,col_width=3.0,row_height=0.625,font_size=14,
					 header_color='#40466e',row_colors=['#f1f1f2', 'w'],edge_color='w',
					 bbox=[0, 0, 1, 1],header_columns=0,ax=None,**kwargs):
	if ax is None:
		size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
		fig, ax = plt.subplots(figsize=size)
		ax.axis('off')

	mpl_table = ax.table(cellText=data.values,bbox=bbox,colLabels=data.columns,cellLoc='center',**kwargs)

	mpl_table.auto_set_font_size(False)
	mpl_table.set_fontsize(font_size)

	for k, cell in  six.iteritems(mpl_table._cells):
		cell.set_edgecolor(edge_color)
		if k[0] == 0 or k[1] < header_columns:
			cell.set_text_props(weight='bold',color='w')
			cell.set_facecolor(header_color)
		else:
			cell.set_facecolor(row_colors[k[0]%len(row_colors)])
	
	fig.savefig(saveAt)
	plt.clf()
	plt.close("all")


def rankAgents(resultsDict,methods,resultsFolder,givenBeta):

	#######################################################
	# Add number of observations to tablle
	# Save table df to file
	#######################################################


	makeFolder(resultsFolder,"agentRanksByMethods")
	makeFolder(resultsFolder,f"agentRanksByMethods{os.sep}pskills")
	makeFolder(resultsFolder,f"agentRanksByMethods{os.sep}xskills")

	finalEstimates = []

	for pitcherID in resultsDict.keys():

		for pitchType in resultsDict[pitcherID]:

			makeFolder(resultsFolder,f"agentRanksByMethods{os.sep}pskills{os.sep}{pitchType}")
			makeFolder(resultsFolder,f"agentRanksByMethods{os.sep}xskills{os.sep}{pitchType}")

			for method in methods:

				est = None

				if "BM" in method:  
					tempM, beta, tt = getInfoBM(method)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:						
						est = resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta][-1]
				elif "BM" not in method:
					if resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
						est = resultsDict[pitcherID][pitchType]["estimates"][method][-1]
						
				if est != None:
					numObs = resultsDict[pitcherID][pitchType]["numObservations"]
					finalEstimates.append([pitcherID,pitchType,pitcherNames[pitcherID],numObs,method,est])



	df = pd.DataFrame(finalEstimates,columns=["PitcherID","PitchType","PitcherName","NumObs","Method","Estimate"])

	xMethods = []
	pMethods = []

	for method in methods:
		if "OR" in method or "BM" in method or "xSkill" in method:
			xMethods.append(method)

		else:
			pMethods.append(method)


	types = [xMethods,pMethods]
	labels = ["xskills","pskills"]

	for mi in range(len(types)):

		methods = types[mi]

		for method in methods:

			for pitchType in df.PitchType.unique():

				try:
					ranks = df.loc[(df.Method==method) & (df.PitchType==pitchType)].sort_values("Estimate")
					ranks = ranks[["PitcherID","PitcherName","NumObs","Estimate"]]
					ranks = ranks.rename(columns={"PitcherID": "Pitcher ID", "PitcherName": "Pitcher Name"})
					ranks.Estimate = ranks.Estimate.round(4)
					# ranks = ranks.style.set_table_attributes("style='width:500px;'").set_caption(f"Pitch Type: {pitchType} | Method: {method}")
					# dfi.export(ranks,f'{resultsFolder}{os.sep}plots{os.sep}agentRanksByMethods{os.sep}ranks-PitchType{pitchType}-Method-{method}.png')
					saveAt = f"{resultsFolder}{os.sep}plots{os.sep}agentRanksByMethods{os.sep}{labels[mi]}{os.sep}{pitchType}{os.sep}ranks-PitchType{pitchType}-Method-{method}.png"
					render_mpl_table(ranks,saveAt,header_columns=0,col_width=2.0)			
				except:
					print(f"No results for pitchType: {pitchType}")
					# code.interact("...", local=dict(globals(), **locals()))

	# code.interact("rankAgents()...", local=dict(globals(), **locals()))


def plotEstimateAllMethodsSamePlotPerAgent(resultsDict,methods,resultsFolder,givenBeta,label):

	folder = "estimateAllMethodsSamePlotPerAgent"

	makeFolder(resultsFolder, folder)

	for pitcherID in resultsDict.keys():

		makeFolder(resultsFolder,f"{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}")


		for pitchType in resultsDict[pitcherID]:

			makeFolder(resultsFolder,f"{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}")


			fig = plt.figure(figsize = (10,10))

			##################################### FOR XSKILLS #####################################
			# create plot for each one of the different agents - estimates vs obs

			ax1 = plt.subplot(2, 1, 1)

			makePlot = False

			for method in methods:

				# only xskill methods
				if "pSkills" not in method: 

					# if "BM" in method and str(givenBeta) in method:  
					if "BM" in method:  
						tempM, beta, tt = getInfoBM(method)
						if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:
							plt.semilogx(range(1,resultsDict[pitcherID][pitchType]["numObservations"]+1),resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta], lw='2.0', label = method)
							makePlot = True
					elif "BM" not in method and resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
						plt.semilogx(range(1,resultsDict[pitcherID][pitchType]["numObservations"]+1),resultsDict[pitcherID][pitchType]["estimates"][method], lw='2.0', label = method)
						numObs = len(resultsDict[pitcherID][pitchType]["estimates"][method])
						makePlot = True

			if makePlot:
				# Put a legend to the right of the current axis
				ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
				
				ax1.set_xlabel('Number of Observations')
				ax1.set_ylabel('Xskill Estimate')
				plt.margins(0.05)
			
				fig.suptitle(f"Pitcher: {pitcherNames[pitcherID]} ({pitcherID}) | Pitch Type: {pitchType} | Num Observations: {resultsDict[pitcherID][pitchType]['numObservations']}")


			# to add space between subplots
			plt.subplots_adjust(hspace = 0.3)

			# Adds "nothing" to the plot 
			# Done in order to add an empty label to the legend 
			# So that there can be a space between the xskill elements & the pskill elements
			plt.plot(np.NaN, np.NaN, '-', alpha = 0.0, label=" ")


			##################################### FOR PSKILLS #####################################
			# create plot for each one of the different agents - estimates vs obs

			ax2 = plt.subplot(2, 1, 2)

			makePlot = False

			for method in methods:

				# skip TN method
				if method == "tn":
					continue

				if "BM" in method:  
					tempM, beta, tt = getInfoBM(method)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] == {}:
						continue
				else:
					if resultsDict[pitcherID][pitchType]["estimates"][method] == {}:
						continue

				# only pskill methods
				if "pSkills" in method:      
					plt.semilogx(range(resultsDict[pitcherID][pitchType]["numObservations"]),resultsDict[pitcherID][pitchType]["estimates"][method], lw='2.0', label = method)
					makePlot = True

			if makePlot:
				# Put a legend to the right of the current axis
				ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

				ax2.set_xlabel("Number of Observations")
				ax2.set_ylabel("Pskill Estimate")
				plt.margins(0.05)

				# Save png
				plt.savefig(f"{resultsFolder}{os.sep}plots{os.sep}{folder}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}{os.sep}results-PitchType{pitchType}-Pitcher{pitcherID}{label}.png", bbox_inches='tight')
		
			plt.clf()
			plt.close()
	#######################################################################################


def plotEstimateAllMethodsPerAgent(resultsDict,methods,resultsFolder,givenBeta,label=""):

	folderX = "estimateAllXSkillMethodsPerAgent"
	folderP = "estimateAllPSkillMethodsPerAgent"


	makeFolder(resultsFolder,folderX)
	makeFolder(resultsFolder,folderP)


	##################################### FOR XSKILLS #####################################
	# Create plot for each one of the different agents - estimates vs obs
	
	for pitcherID in resultsDict.keys():

		makeFolder(resultsFolder,f"{folderX}{os.sep}{pitcherNames[pitcherID].replace(' ','')}")
		makeFolder(resultsFolder,f"{folderP}{os.sep}{pitcherNames[pitcherID].replace(' ','')}")

		for pitchType in resultsDict[pitcherID]:

			makeFolder(resultsFolder,f"{folderX}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}")
			makeFolder(resultsFolder,f"{folderP}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}")


			# print(f"Pitcher: {pitcherID} | Pitch Type: {pitchType}")

			fig = plt.figure()
			ax = plt.subplot(111)

			if len(methods) > 10:
				colors = iter(plt.cm.rainbow(np.linspace(0,1,len(methods))))
			else:
				colors = iter(["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"])

			makePlot = False

			for method in methods:

				try:

					if "pSkills" not in method:

						color = next(colors)

						# if "BM" in method and str(givenBeta) in method:  
						if "BM" in method:  
							tempM, beta, tt = getInfoBM(method)
							if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] != {}:
								plt.semilogx(range(1,resultsDict[pitcherID][pitchType]["numObservations"]+1),resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta],c=color, lw='2.0', label = tempM + "-" + str(beta))
								makePlot = True
						elif "BM" not in method and resultsDict[pitcherID][pitchType]["estimates"][method] != {}:
							plt.semilogx(range(1,resultsDict[pitcherID][pitchType]["numObservations"]+1),resultsDict[pitcherID][pitchType]["estimates"][method],c=color, lw='2.0', label = method)
							makePlot = True
				except:
					code.interact("plotEstimateAllMethodsPerAgent()...", local=dict(globals(), **locals()))

			if makePlot:
				# Put a legend to the right of the current axis
				ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
				
				plt.xlabel('Number of Observations')
				plt.ylabel('Xskill Estimate')
				plt.margins(0.05)

				plt.title(f"Pitcher: {pitcherNames[pitcherID]} ({pitcherID}) | Pitch Type: {pitchType} | Num Observations: {resultsDict[pitcherID][pitchType]['numObservations']}")
				
				plt.savefig(f"{resultsFolder}{os.sep}plots{os.sep}{folderX}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}{os.sep}results-PitchType{pitchType}-Pitcher{pitcherID}{label}.png", bbox_inches='tight')
				
			plt.clf()
			plt.close()

	#######################################################################################


	##################################### FOR PSKILLS #####################################
	# Create plot for each one of the different agents - estimates vs obs
	
	for pitcherID in resultsDict.keys():

		for pitchType in resultsDict[pitcherID]:

			fig = plt.figure()
			ax = plt.subplot(111)


			makePlot = False

			for method in methods:

				# skip TN method
				if method == "tn":
					continue

				if "BM" in method:  
					tempM, beta, tt = getInfoBM(method)
					if resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] == {}:
						continue
				else:
					if resultsDict[pitcherID][pitchType]["estimates"][method] == {}:
						continue

				if "pSkills" in method:
					makePlot = True 
					plt.semilogx(range(1,resultsDict[pitcherID][pitchType]["numObservations"]+1),resultsDict[pitcherID][pitchType]["estimates"][method], lw='2.0', label = method)

			if makePlot:
				# Put a legend to the right of the current axis
				ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
				
				plt.xlabel('Number of Observations')
				plt.ylabel('Pskill Estimate')
				plt.margins(0.05)
				plt.title(f"Pitcher: {pitcherNames[pitcherID]} ({pitcherID}) | Pitch Type: {pitchType} | Num Observations: {len(resultsDict[pitcherID][pitchType]['estimates'][method])}")
				plt.savefig(f"{resultsFolder}{os.sep}plots{os.sep}{folderP}{os.sep}{pitcherNames[pitcherID].replace(' ','')}{os.sep}{pitchType}{os.sep}results-PitchType{pitchType}-Pitcher{pitcherID}{label}.png", bbox_inches='tight')
				
			plt.clf()
			plt.close()

	#######################################################################################


if __name__ == "__main__":

	# USAGE EXAMPLE:
	#  python Processing/processResultsBaseball.py -resultsFolder Experiments/baseball/testing/ -domain baseball


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-delta", dest = "delta", help = "Delta = resolution to use", type = float, default = 5.0)
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")
	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	result_files = os.listdir(args.resultsFolder + os.sep + "results")


	try:
		result_files.remove(".DS_Store")
	except:
		pass

	try:
		result_files.remove("backup")
	except:
		pass


	if len(result_files) == 0:
		print("No result files present for experiment.")
		exit()


	# If the plots folder doesn't exist already, create it
	if not os.path.exists(args.resultsFolder + os.sep + "plots" + os.sep):
		os.mkdir(args.resultsFolder + os.sep + "plots" + os.sep)


	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework")) + os.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	cache.enable()


	resultsDict = {}

	namesEstimators = []
	typeTargetsList = []

	numHypsX = []
	numHypsP = []
	seenAgents = []

	'''
	methodsNames = ['OR', 'BM-MAP', 'BM-EES',
				"JT-QRE-MAP","JT-QRE-MAP-GivenPrior","JT-QRE-MAP-MinLambda","JT-QRE-MAP-GivenPrior-MinLambda",
				"JT-QRE-EES","JT-QRE-EES-GivenPrior","JT-QRE-EES-MinLambda","JT-QRE-EES-GivenPrior-MinLambda",
				"NJT-QRE-MAP","NJT-QRE-MAP-GivenPrior","NJT-QRE-MAP-MinLambda","NJT-QRE-MAP-GivenPrior-MinLambda",
				"NJT-QRE-EES","NJT-QRE-EES-GivenPrior","NJT-QRE-EES-MinLambda","NJT-QRE-EES-GivenPrior-MinLambda"]
	'''
	methodsNames = ['OR','BM-MAP','BM-EES',
				"JT-QRE-MAP","JT-QRE-EES"]
				#,"NJT-QRE-MAP","NJT-QRE-EES"]

	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Environments/"
	mainFolderName = scriptPath.split("Processing")[0] + "Environments" + os.sep

	domainModule,delta = getDomainInfo(args.domain)


	methodsAllProbs = ["BM-MAP-66-Beta-0.5-allProbs"
			"BM-EES-66-Beta-0.5-allProbs",
			"BM-MAP-66-Beta-0.75-allProbs",
			"BM-EES-66-Beta-0.75-allProbs",
			"BM-MAP-66-Beta-0.85-allProbs",
			"BM-EES-66-Beta-0.85-allProbs",
			"BM-MAP-66-Beta-0.9-allProbs",
			"BM-EES-66-Beta-0.9-allProbs",
			"BM-MAP-66-Beta-0.95-allProbs",
			"BM-EES-66-Beta-0.95-allProbs",
			"BM-MAP-66-Beta-0.99-allProbs",
			"BM-EES-66-Beta-0.99-allProbs",
			"JT-QRE-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-allProbs",
			"JT-QRE-MinLambda-1.3-allProbs",
			"JT-QRE-MinLambda-1.7-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.3-allProbs",
			"JT-QRE-GivenPrior-8-0.4-1.0-MinLambda1.7-allProbs"]
			
			# ,"NJT-QRE-MAP-66-66-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-pSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-pSkills-allProbs",
			# "NJT-QRE-MAP-66-66-MinLambda-1.3-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-MinLambda-1.3-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-MinLambda-1.3-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-MinLambda-1.3-pSkills-allProbs",
			# "NJT-QRE-MAP-66-66-MinLambda-1.7-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-MinLambda-1.7-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-MinLambda-1.7-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-MinLambda-1.7-pSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-MinLambda1.3-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-MinLambda1.3-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-MinLambda1.3-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-MinLambda1.3-pSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-MinLambda1.7-xSkills-allProbs",
			# "NJT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-MinLambda1.7-pSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-MinLambda1.7-xSkills-allProbs",
			# "NJT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-MinLambda1.7-pSkills-allProbs"]


	# Before processing the results, verify if file with information is available to start up with that information
	# In order to not recompute info all over again and only process the new files/experiments
	try:

		rdFile = args.resultsFolder + os.sep + "plots" + os.sep + "resultsDictInfo"
		oiFile = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"     

		with open(rdFile,"rb") as file:
			resultsDict = pickle.load(file)

		with open(oiFile,"rb") as file:
			otherInfo = pickle.load(file)

			namesEstimators = otherInfo["namesEstimators"]
			methods = otherInfo["methods"]
			methodsAllProbs = otherInfo["methodsAllProbs"]
			numHypsX = otherInfo['numHypsX']
			numHypsP = otherInfo['numHypsP']
			seenAgents = otherInfo["seenAgents"]
			domain = otherInfo["domain"]
			typeTargetsList = otherInfo["typeTargetsList"]
			betas = otherInfo["betas"]
			resultFilesLoaded = otherInfo["result_files"]

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

		with open(args.resultsFolder + os.sep + "results" + os.sep + result_files[i]) as infile:
			results = json.load(infile)
			
			namesEstimators = results["namesEstimators"]
			numHypsX = results['numHypsX']
			numHypsP = results['numHypsP']
			domain = results["domain"]


			methods = []
			# methodsAllProbs = []

			for m in results.keys():

				# SKIPPING NJT METHODS
				if "NJT" in m:
					continue

				if (not m.isalpha()) and "-" in m and "allProbs" not in m:
					methods.append(m)

				# if "allProbs" in m:
				# 	methodsAllProbs.append(m)



		betas = []
		getBetas(results,betas,typeTargetsList)

		# methods = getMethods(args.domain,methodsNames,namesEstimators,numHypsX,numHypsP,betas,typeTargetsList)


		# code.interact("1...", local=dict(globals(), **locals()))
		
	############################################################################################################################
	############################################################################################################################
	# Use when debugging/testing - to speed up - read only specified # of results file (and not however many there are in the folder)
	#result_files = result_files[0:30]
	############################################################################################################################
	############################################################################################################################


	# Start processing results
	total_num_exps = 0

	# NOTE: EACH RESULT FILE BELONGS TO A GIVEN PITCHER AND PITCH TYPE
	# EACH COMBINATION WILL BE SEEN ONLY ONCE (MEANING JUST 1 EXP)

	for rf in result_files: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)


		if rf in resultFilesLoaded:
			print(f"\tSkipping {rf} since already loaded.")
			continue

		param = ""

		with open(args.resultsFolder + os.sep + "results" + os.sep + rf) as infile:
			results = json.load(infile)


			agent = results["agent_name"]
			pitcherID = agent[0]
			pitchType = agent[1]
			seenAgents.append(agent)

			numObservations = results["numObservations"]


			if pitcherID not in resultsDict:
				resultsDict[pitcherID] = {}

			if pitchType not in resultsDict[pitcherID]:
				resultsDict[pitcherID][pitchType] = {}


			resultsDict[pitcherID][pitchType] = {"estimates": {},"allProbs":{},"numObservations":numObservations}


			for m in methods:

				try:
					validCount = False

					# if the method exists on the results file, load
					testLoadMethod = results[m]

					if len(testLoadMethod) == numObservations:
						validCount = True

				except:
					print(f"\t\t{m} - not present")
					# code.interact("...", local=dict(globals(), **locals()))
					continue


				# If TBA/BM method, need to account for possible different betas
				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)

					# To initialize once
					if tt not in resultsDict[pitcherID][pitchType]["estimates"]:
						resultsDict[pitcherID][pitchType]["estimates"][tt] = {}

					if tempM not in resultsDict[pitcherID][pitchType]["estimates"][tt]:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM] = {}

					if beta not in resultsDict[pitcherID][pitchType]["estimates"][tt][tempM]:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = [0.0] * numObservations
						
					# Save estimates
					if validCount:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = results[m]
					else:
						resultsDict[pitcherID][pitchType]["estimates"][tt][tempM][beta] = {}

				else:
					# Save estimates
					# Won't add info for method if there's a mismatch between
					# expected # of observations and the number of estimates produced
					if validCount:
						resultsDict[pitcherID][pitchType]["estimates"][m] = results[m]
					else:
						resultsDict[pitcherID][pitchType]["estimates"][m] = {}


			for ma in methodsAllProbs:

				try:

					validCount = False

					# if the method exists on the results file, load
					testLoadMethod = results[ma]

					# Plus 1 since prior probs included in all probs
					if len(testLoadMethod) == numObservations+1:
						validCount = True

				except:
					# print("->",m)
					# code.interact("all probs...", local=dict(globals(), **locals()))
					continue


				# Won't add info for method if there's a mismatch between
				# expected # of observations and the number of estimates produced	
				if validCount:
					resultsDict[pitcherID][pitchType]["allProbs"][ma] = testLoadMethod
				else:
					resultsDict[pitcherID][pitchType]["allProbs"][ma] = {}


			# FOR TESTING
			# if pitcherID == "425794" and pitchType == 'CH':
			# 	code.interact("test...", local=dict(globals(), **locals()))


			if list(resultsDict[pitcherID][pitchType]["estimates"].keys()) == []:
				print(f"\n\t\tNo results seen yet for {pitcherID}-{pitchType}. Only initial exp info present.")
				del resultsDict[pitcherID][pitchType]



	print('\nCompiled results for', total_num_exps, 'experiments')
	# code.interact("...", local=dict(globals(), **locals()))
	

	#############################################################################
	# Store processed results
	#############################################################################

	saveAs = args.resultsFolder + os.sep + "plots" + os.sep + "resultsDictInfo"

	# Save dict containing all info - to be able to rerun it later - for "cosmetic" changes only
	with open(saveAs, "wb") as outfile:
		pickle.dump(resultsDict, outfile)


	otherInfo = {}
	otherInfo["namesEstimators"] = namesEstimators
	otherInfo["methods"] = methods
	otherInfo["methodsAllProbs"] = methodsAllProbs
	otherInfo['numHypsX'] = numHypsX
	otherInfo['numHypsP'] = numHypsP
	otherInfo["seenAgents"] = seenAgents
	otherInfo["domain"] = domain
	otherInfo["typeTargetsList"] = typeTargetsList
	otherInfo["betas"] = betas
	otherInfo["result_files"] = result_files

	saveAs2 = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

	with open(saveAs2, "wb") as outfile:
		pickle.dump(otherInfo, outfile)

	#############################################################################3
		   

	#############################################################################
	# PLOTS
	#############################################################################

	# Parameters for plots
	rcParams.update({'font.size': 14})
	rcParams.update({'legend.fontsize': 14})
	rcParams["axes.labelweight"] = "bold"
	rcParams["axes.titleweight"] = "bold"


	# To make font of title & labels bold
	# rc('text', usetex=True)

	xHypsStr = str(numHypsX[0])
	pHypsStr = str(numHypsP[0])

	global methodsDictNames
	methodsDictNames = {'OR': f"OR-{xHypsStr}",\
					"OR-MidGame": f"OR-{xHypsStr}-estimatesMidGame", \
					"OR-FullGame": f"OR-{xHypsStr}-estimatesFullGame", \

					'BM-MAP': f"BM-MAP-{xHypsStr}",\
					'BM-EES': f"BM-EES-{xHypsStr}",\

					"JT-QRE-MAP": [f"JT-QRE-MAP-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-MAP-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-MAP-GivenPrior": [f"JT-QRE-MAP-GivenPrior-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-MAP-GivenPrior-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-MAP-MinLambda": [f"JT-QRE-MAP-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-MAP-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-MAP-GivenPrior-MinLambda": [f"JT-QRE-MAP-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-MAP-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					
					"JT-QRE-EES": [f"JT-QRE-EES-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-EES-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-EES-GivenPrior": [f"JT-QRE-EES-GivenPrior-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-EES-GivenPrior-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-EES-MinLambda": [f"JT-QRE-EES-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-EES-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					"JT-QRE-EES-GivenPrior-MinLambda": [f"JT-QRE-EES-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"JT-QRE-EES-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\


					"JT-FLIP-MAP": ["JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills","JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills"],\
					"JT-FLIP-EES": ["JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills","JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills"],\


					"NJT-QRE-MAP": [f"NJT-QRE-MAP-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-MAP-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-MAP-GivenPrior": [f"NJT-QRE-MAP-GivenPrior-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-MAP-GivenPrior-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-MAP-MinLambda": [f"NJT-QRE-MAP-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-MAP-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-MAP-GivenPrior-MinLambda": [f"NJT-QRE-MAP-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-MAP-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					
					"NJT-QRE-EES": [f"NJT-QRE-EES-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-EES-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-EES-GivenPrior": [f"NJT-QRE-EES-GivenPrior-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-EES-GivenPrior-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-EES-MinLambda": [f"NJT-QRE-EES-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-EES-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"],\
					"NJT-QRE-EES-GivenPrior-MinLambda": [f"NJT-QRE-EES-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-xSkills",f"NJT-QRE-EES-GivenPrior-MinLambda-{xHypsStr}-{pHypsStr}-pSkills"]}


	global methodsDict
	methodsDict = {'OR'+"-"+xHypsStr: "OR", \
					'OR'+"-"+xHypsStr+"-estimatesMidGame": "OR-MidGame", \
					'OR'+"-"+xHypsStr+"-estimatesFullGame": "OR-FullGame", \

					'BM-MAP'+"-"+xHypsStr: 'BM-MAP', \
					'BM-EES'+"-"+xHypsStr: 'BM-EES',\

					"JT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills":"JT-QRE-MAP",\
					"JT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills":  "JT-QRE-MAP",\

					"JT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills":  "JT-QRE-EES",\
					"JT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills":  "JT-QRE-EES",\

					 "JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "JT-FLIP-MAP",\
					 "JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "JT-FLIP-MAP",\

					 "JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "JT-FLIP-EES",\
					 "JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "JT-FLIP-EES",\

					 "NJT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "NJT-QRE-MAP",\
					 "NJT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "NJT-QRE-MAP",\

					 "NJT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "NJT-QRE-EES",\
					 "NJT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "NJT-QRE-EES"}

	global methodsColors
	methodsColors = {'OR'+"-"+xHypsStr: "tab:purple",\
					'OR'+"-"+xHypsStr+"-estimatesMidGame": "tab:gray", \
					'OR'+"-"+xHypsStr+"-estimatesFullGame": "tab:pink", \


					'BM-MAP'+"-"+xHypsStr: "tab:brown", \
					'BM-EES'+"-"+xHypsStr: "tab:cyan",\

					'BM-MAP'+"-"+xHypsStr+"-OptimalTargets": "xkcd:teal", \
					'BM-EES'+"-"+xHypsStr+"-OptimalTargets": "xkcd:darkgreen",\

					'BM-MAP'+"-"+xHypsStr+"-DomainTargets": "xkcd:fuchsia", \
					'BM-EES'+"-"+xHypsStr+"-DomainTargets": "xkcd:darkblue",\

					"JT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:red",\
					"JT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:red" ,\

					"JT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:green" ,\
					"JT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:green" ,\

					"JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:gray" ,\
					"JT-FLIP-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:gray",\

					"JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:olive",\
					"JT-FLIP-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:olive",\

					"NJT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:blue",\
					"NJT-QRE-MAP"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:blue" ,\

					"NJT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-xSkills": "tab:orange",\
					"NJT-QRE-EES"+"-"+xHypsStr+"-"+pHypsStr+"-pSkills": "tab:orange"}

	global methodNamesPaper
	methodNamesPaper = {'OR'+"-"+xHypsStr: "OR", \
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

	pitcherNames = {}
	
	for pitcherID in resultsDict.keys():

		result = playerid_reverse_lookup([int(pitcherID)])[["name_first","name_last"]]
		pitcherNames[pitcherID] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"


	#################################################
	# Assuming same beta for all agent types for now
	#################################################
	# UPDATE ACCORDINGLY
	givenBeta = 0.99
	#################################################
	
	
	
	################################## FOR ESTIMATES ##################################

	# Including OR as a baseline for comparison
	allJTM = ["OR-66"]
	JTM_GivenPrior = ["OR-66"]
	JTM_MinLambda = ["OR-66"]
	JTM_GivenPrior_MinLambda = ["OR-66"]
	normalJTM = ["OR-66"]
	justBM = []
	BM_OR = ["OR-66"]
	noJTM = []
	minMaxBM = ['OR-66', 'BM-MAP-66-Beta-0.5', 'BM-EES-66-Beta-0.5','BM-MAP-66-Beta-0.99', 'BM-EES-66-Beta-0.99']


	forCSV = ['OR-66','BM-MAP-66-Beta-0.5', 'BM-EES-66-Beta-0.5',
			'BM-MAP-66-Beta-0.95', 'BM-EES-66-Beta-0.95',
			'BM-MAP-66-Beta-0.99', 'BM-EES-66-Beta-0.99',
			'JT-QRE-MAP-66-66-GivenPrior-8-0.4-1.0-xSkills',
			'JT-QRE-EES-66-66-GivenPrior-8-0.4-1.0-xSkills',
			'JT-QRE-MAP-66-66-xSkills','JT-QRE-EES-66-66-xSkills']

	for m in methods:
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

		elif "BM" in m:
			justBM.append(m)
			BM_OR.append(m)

		if "JT" not in m:
			noJTM.append(m)



	# code.interact("...", local=dict(globals(), **locals()))

	labels = ["-ALL","-NormalJTM","-AllJTM",
			"-JTM-GivenPrior","-JTM-MinLambda","-JTM-GivenPrior-MinLambda",
			"-NoJTM","-JustBM","-OR-BM","-MinMaxBM"]

	methodsLists = [methods,normalJTM,allJTM,
			JTM_GivenPrior,JTM_MinLambda,JTM_GivenPrior_MinLambda,
			noJTM,justBM,BM_OR,minMaxBM]



	# plotEstimateAllBetasSamePlotPerAgentBAR(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder,betas)
	# plotEstimateAllBetasSamePlotPerAgent(resultsDict, actualMethodsNames, actualMethodsOnExps, numHypsX, numHypsP, args.resultsFolder)


	# label = "AllMethods"
	# makeCSV(resultsDict,methods,label,args.resultsFolder)

	# label = "SelectedMethods"
	# makeCSV(resultsDict,forCSV,label,args.resultsFolder)


	rankAgents(resultsDict,methods,args.resultsFolder,givenBeta)
 

	computeCredibleIntervals(resultsDict,methodsAllProbs,args.resultsFolder)


	compareToGivenMethod(resultsDict,methods,args.resultsFolder,compareTo="OR",thresObservations=50,thresSame=0.10)


	topXskillHyps(resultsDict,methodsAllProbs,args.resultsFolder,topX=5)



	for i in range(len(labels)):
		plotEstimateAllMethodsPerAgent(resultsDict,methodsLists[i],args.resultsFolder,givenBeta,labels[i])
		plotEstimateAllMethodsSamePlotPerAgent(resultsDict,methodsLists[i],args.resultsFolder,givenBeta,labels[i])



	# Close all remaining figures
	plt.close("all")

	# code.interact("...", local=dict(globals(), **locals()))

