from matplotlib import rcParams,rc
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

import argparse,sys,os
import pickle
import scipy

import numpy as np
import pandas as pd
import code

global methodsDictNames
global methodsDict
global methodNamesPaper
global methodsColors

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

from matplotlib.patches import Ellipse

from importlib.machinery import SourceFileLoader

from pybaseball import playerid_reverse_lookup, cache

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}makePlotsBaseballMulti.py")[0]

module = SourceFileLoader("baseball_multi.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball_multi.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


class HandlerEllipse(HandlerPatch):
	def create_artists(self, legend, orig_handle,
					   xdescent, ydescent, width, height, fontsize, trans):
		center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
		p = mpatches.Ellipse(xy=center, width=orig_handle.width,
										height=orig_handle.height)
		self.update_prop(p, orig_handle, legend)
		p.set_transform(trans)
		return [p]


def plotOtherInfo(titleStr):

	# Overlay strike zone dimensions on plot
	# Plate_x: [-0.71,0.71]
	# Plate_z: [1.546,3.412]
	plt.hlines(y=1.546,xmin=-0.71,xmax=0.71,color="k")
	plt.hlines(y=3.413,xmin=-0.71,xmax=0.71,color="k")
	plt.vlines(x=-0.71, ymin=1.546,ymax=3.412,color="k")
	plt.vlines(x=0.71, ymin=1.546,ymax=3.412,color="k")

	plt.xlabel("Horizontal Location (Pitcher's Perspective)")
	plt.ylabel("Vertical Location")

	plt.title(titleStr)
	plt.tight_layout()


def getAvgEstimatePerDimensionAcrossChunks():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}avgEstimate-PerDimension-AcrossChunks{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	infoForAvg = {}


	for pitcherID in resultsDict:

		if pitcherID not in infoForAvg:
			infoForAvg[pitcherID] = {}

		for pitchType in resultsDict[pitcherID]:

			if pitchType not in infoForAvg[pitcherID]:
				infoForAvg[pitcherID][pitchType] = {}

			for chunk in resultsDict[pitcherID][pitchType]:

				if chunk not in infoForAvg[pitcherID][pitchType]:
					infoForAvg[pitcherID][pitchType][chunk] = {}


				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)

				info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]

				tempMethods = []
				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
					and "all" not in m and "resamplingMethod" not in m:
						tempMethods.append(m)

						if "Multi" in m:

							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = info[m][-1]
							rho = info[tempM][-1]

						# Case: JEEDS
						else:
							estimatedX = info[m][-1]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0

						infoForAvg[pitcherID][pitchType][chunk][m] = [stdDevs,rho]

				del loadedDictInfo



	# code.interact("...", local=dict(globals(), **locals()))


	avgInfo = {}

	for pitcherID in infoForAvg:

		if pitcherID not in avgInfo:
			avgInfo[pitcherID] = {}

		for pitchType in infoForAvg[pitcherID]:

			if pitchType not in avgInfo[pitcherID]:
				avgInfo[pitcherID][pitchType] = {}

			for m in tempMethods:

				if m not in avgInfo[pitcherID][pitchType]:
					avgInfo[pitcherID][pitchType][m] = {"individualAvgsXS":[0.0,0.0],"avgXS":[0.0,0.0],"individualAvgsR":[],"avgR":0.0}


				for chunk in infoForAvg[pitcherID][pitchType]:
					avgInfo[pitcherID][pitchType][m]["individualAvgsXS"][0] += infoForAvg[pitcherID][pitchType][chunk][m][0][0]
					avgInfo[pitcherID][pitchType][m]["individualAvgsXS"][1] += infoForAvg[pitcherID][pitchType][chunk][m][0][1]

					avgInfo[pitcherID][pitchType][m]["individualAvgsR"].append(infoForAvg[pitcherID][pitchType][chunk][m][1])


				# Find avg per method and pitcher and pitch type
				avgInfo[pitcherID][pitchType][m]["avgXS"][0] = avgInfo[pitcherID][pitchType][m]["individualAvgsXS"][0]/len(infoForAvg[pitcherID][pitchType])
				avgInfo[pitcherID][pitchType][m]["avgXS"][1] = avgInfo[pitcherID][pitchType][m]["individualAvgsXS"][1]/len(infoForAvg[pitcherID][pitchType])
				
				avgInfo[pitcherID][pitchType][m]["avgR"] = sum(avgInfo[pitcherID][pitchType][m]["individualAvgsR"])/len(infoForAvg[pitcherID][pitchType])



	with open(saveAt+"pitchersInfo.txt",'w') as handle:

		for pitcherID in avgInfo:
			for pitchType in avgInfo[pitcherID]:
				print(f"{pitcherID} | {pitchType}",file=handle)

				for m in avgInfo[pitcherID][pitchType]:
					print(f"\t{m} | {avgInfo[pitcherID][pitchType][m]['avgXS']} | {avgInfo[pitcherID][pitchType][m]['avgR']}",file=handle)

				print(file=handle)


	dfInfo = []
	for pitcherID in avgInfo:

		for pitchType in avgInfo[pitcherID]:

			temp = [pitcherID,pitchType]
			newTempMethods = []

			for m in methods:

				if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
				and "all" not in m and "resamplingMethod" not in m:
					temp.append(f"{avgInfo[pitcherID][pitchType][m]['avgXS'][0]} | {avgInfo[pitcherID][pitchType][m]['avgXS'][1]}")
					temp.append(avgInfo[pitcherID][pitchType][m]['avgR'])

					newTempMethods.append(m)
					newTempMethods.append(f"{m.split('-xSkill')[0]}-rho")

			dfInfo.append(temp)


	df = pd.DataFrame(np.array(dfInfo),columns = ["PitcherID","PitchType"] + newTempMethods)

	df.to_csv(saveAt+"pitchersInfo-avgEstimate-PerDimension-AcrossChunks.csv",index=False)

	# code.interact("...", local=dict(globals(), **locals()))


def getAvgEstimateAcrossChunks():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}avgEstimate-AcrossChunks{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	avgInfo = {}


	for pitcherID in resultsDict:

		if pitcherID not in avgInfo:
			avgInfo[pitcherID] = {}

		for pitchType in resultsDict[pitcherID]:

			if pitchType not in avgInfo[pitcherID]:
				avgInfo[pitcherID][pitchType] = {}
 
			for chunk in resultsDict[pitcherID][pitchType]:

				if chunk not in avgInfo[pitcherID][pitchType]:
					avgInfo[pitcherID][pitchType][chunk] = {}


				# textFile = open(f"{saveAt}pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}.txt","w")

				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)

				info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]

				# print("Method | Avg Estimate",file=textFile)

				tempMethods = []
				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
					and "all" not in m and "resamplingMethod" not in m:

						if m not in avgInfo[pitcherID][pitchType][chunk]:
							avgInfo[pitcherID][pitchType][chunk][m] = {"xs":0.0,"rho":0.0}

						tempMethods.append(m)

						if "Multi" in m:
							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = info[m][-1]
							rho = info[tempM][-1]

						# Case: JEEDS
						else:
							estimatedX = info[m][-1]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0

						avgXS = sum(stdDevs)/len(stdDevs)

						# print(f"{m} | {avg}",file=textFile)

						avgInfo[pitcherID][pitchType][chunk][m]["xs"] = avgXS
						avgInfo[pitcherID][pitchType][chunk][m]["rho"] = rho


				
				del loadedDictInfo

		# textFile.close()


	# code.interact("...", local=dict(globals(), **locals()))


	info = {}

	for pitcherID in avgInfo:

		if pitcherID not in info:
			info[pitcherID] = {}

		for pitchType in avgInfo[pitcherID]:

			if pitchType not in info[pitcherID]:
				info[pitcherID][pitchType] = {}

			for m in tempMethods:

				if m not in info[pitcherID][pitchType]:
					info[pitcherID][pitchType][m] = {"individualAvgsXS":[],"avgXS":0.0,"individualRhos":[],"avgR":0.0}


				for chunk in avgInfo[pitcherID][pitchType]:
					info[pitcherID][pitchType][m]["individualAvgsXS"].append(avgInfo[pitcherID][pitchType][chunk][m]["xs"])
					info[pitcherID][pitchType][m]["individualRhos"].append(avgInfo[pitcherID][pitchType][chunk][m]["rho"])

				# Find avg per method and pitcher and pitch type
				info[pitcherID][pitchType][m]["avgXS"] = sum(info[pitcherID][pitchType][m]["individualAvgsXS"])/len(info[pitcherID][pitchType][m]["individualAvgsXS"])
				info[pitcherID][pitchType][m]["avgR"] = sum(info[pitcherID][pitchType][m]["individualRhos"])/len(info[pitcherID][pitchType][m]["individualRhos"])



	dfInfo = []
	for pitcherID in info:

		for pitchType in info[pitcherID]:

			temp = [pitcherID,pitchType]
			newTempMethods = []

			for m in methods:

				if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
				and "all" not in m and "resamplingMethod" not in m:
					temp.append(info[pitcherID][pitchType][m]["avgXS"])
					temp.append(info[pitcherID][pitchType][m]["avgR"])

					newTempMethods.append(m)
					newTempMethods.append(f"{m.split('-xSkill')[0]}-rho")

			dfInfo.append(temp)

	df = pd.DataFrame(np.array(dfInfo),columns = ["PitcherID","PitchType"] + newTempMethods)

	df.to_csv(saveAt+"pitchersInfo-avgEstimate-AcrossChunks.csv",index=False)


	# code.interact("...", local=dict(globals(), **locals()))


def getAvgEstimateGivenChunk():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}avgEstimateGivenChunk{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	avgInfo = {}


	for pitcherID in resultsDict:

		if pitcherID not in avgInfo:
			avgInfo[pitcherID] = {}

		for pitchType in resultsDict[pitcherID]:

			if pitchType not in avgInfo[pitcherID]:
				avgInfo[pitcherID][pitchType] = {}

			for chunk in resultsDict[pitcherID][pitchType]:

				if chunk not in avgInfo[pitcherID][pitchType]:
					avgInfo[pitcherID][pitchType][chunk] = {}


				# textFile = open(f"{saveAt}pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}.txt","w")

				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)

				info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]

				# print("Method | Avg Estimate",file=textFile)

				tempMethods = []
				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m \
					and "all" not in m and "resamplingMethod" not in m:
						
						if m not in avgInfo[pitcherID][pitchType][chunk]:
							avgInfo[pitcherID][pitchType][chunk][m] = {"xs":0.0,"rho":0.0}


						tempMethods.append(m)

						if "Multi" in m:

							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = info[m][-1]
							rho = info[tempM][-1]

						# Case: JEEDS
						else:
							estimatedX = info[m][-1]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0

						avgXS = sum(stdDevs)/len(stdDevs)

						# print(f"{m} | {avg}",file=textFile)

						avgInfo[pitcherID][pitchType][chunk][m]["xs"] = avgXS
						avgInfo[pitcherID][pitchType][chunk][m]["rho"] = rho


				# code.interact("...", local=dict(globals(), **locals()))
				
				del loadedDictInfo

		# textFile.close()


	dfInfo = []

	for pitcherID in avgInfo:
		for pitchType in avgInfo[pitcherID]:

			for chunk in avgInfo[pitcherID][pitchType]:

				temp = [pitcherID,pitchType,chunk]
				newTempMethods = []


				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
					and "all" not in m and "resamplingMethod" not in m:

						temp.append(avgInfo[pitcherID][pitchType][chunk][m]["xs"])
						temp.append(avgInfo[pitcherID][pitchType][chunk][m]["rho"])

						newTempMethods.append(m)
						newTempMethods.append(f"{m.split('-xSkill')[0]}-rho")

				dfInfo.append(temp)

	df = pd.DataFrame(np.array(dfInfo),columns = ["PitcherID","PitchType","Chunk"] + newTempMethods)

	df.to_csv(saveAt+"pitchersInfo-avgEstimateGivenChunk.csv",index=False)
	# code.interact("...", local=dict(globals(), **locals()))


def plotCovErrorElipse():

	if not os.path.exists(f"{args.resultsFolder}{os.sep}plots{os.sep}covErrorElipse{os.sep}"):
		os.mkdir(f"{args.resultsFolder}{os.sep}plots{os.sep}covErrorElipse{os.sep}")


	for pitchType in ["FF"]:

		for chunk in ["1"]:

			for m in methods:

				if "xSkill" not in m:
					continue


				folder = f"{args.resultsFolder}{os.sep}plots{os.sep}covErrorElipse{os.sep}{m}{os.sep}"
				
				if not os.path.exists(folder):
					os.mkdir(folder)

				
				fig = plt.figure(figsize=(6,6))
				ax = plt.gca()

				titleStr = ""
				plotOtherInfo(titleStr)

				#plt.axis('equal')
				ax.set_aspect('equal', adjustable='box')
				plt.xlim(-1.5,1.5)
				plt.ylim(1,4)


				for pitcherID in resultsDict:

					if pitcherID == "672851":
						continue

					with open(f"{args.resultsFolder}{os.sep}results{os.sep}OnlineExp_{pitcherID}_{pitchType}_JEEDS_Chunk_{chunk}_PFE_NEFF_Chunk_{chunk}.results",'rb') as handle:
						loadedDictInfo = pickle.load(handle)

					# info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]
					# info = loadedDictInfo[m]

					noisyActions = loadedDictInfo["noisy_actions"]

					# code.interact("...", local=dict(globals(), **locals()))
					estimatedXS = loadedDictInfo[m]

					if "Multi" in m:
						tempM = m.split("-xSkills")[0] +"-rhos"
						estimatedR = loadedDictInfo[tempM]
					else:
						estimatedR = 0.0


					if "Multi" in m:
						estX = np.round(estimatedXS[-1],4)
						estR = np.round(estimatedR[-1],4)

						info2 = f"X{estX}-R{estR}"

					else:
						estX = np.round(estimatedXS[-1],4)
						info2 = f"X{estX}"


					top = ["594798",'455119',"446372"]
					bottom = ["518617",'656945',"656354"]


					if pitcherID in top:
						color = "tab:green"
						label = "Top Pitcher"
					else:
						color = "tab:red"
						label = "Bottom Pitcher"


					k = 2
					covMatrix = np.zeros((k,k))

					# for eachObservation in range(len(estimatedXS)):

					# ASSUMING LAST OBSERVATION

					if "Multi" in m:
						estX = np.round(estimatedXS[-1],4)
						estR = round(estimatedR[-1],4)
					# Normal JEEDS (assuming symmetric agents)
					else:
						estX = np.round([estimatedXS[-1],estimatedXS[-1]],4)
						estR = 0.0


					# CREATE COVARIANCE MATRIX
					np.fill_diagonal(covMatrix,np.square(estX))

					# Fill the upper and lower triangles
					for i in range(k):
						for j in range(i+1,k):
							covMatrix[i,j] = np.prod(estX) * estR
							covMatrix[j,i] = covMatrix[i,j]



					# num = 5.991 # 95%
					# num = 2.278 # 68%
					# num = 1.386 # 50%
					# num = 0.575 # 25%

					# confidenceLevel = 0.997
					# confidenceLevel = 0.95
					# confidenceLevel = 0.50
					# confidenceLevel = 0.75
					confidenceLevel = 0.68

					# Calculate the chi-square critical value for the given confidence level
					chi_square_val = scipy.stats.chi2.ppf(confidenceLevel,2)
					print(confidenceLevel)
					print(chi_square_val)
					print()


					# Calculate ellipse parameters
					eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
					major_axis = 2 * np.sqrt(chi_square_val*eigenvalues[0])
					minor_axis = 2 * np.sqrt(chi_square_val*eigenvalues[1])
					rotation_angle = np.arctan2(eigenvectors[1,0],eigenvectors[0,0])*(180/np.pi)

					# Save info
					infoElipse = {"majorAxis": major_axis,
						   "minorAxis": minor_axis,
						   "rotationAngle": rotation_angle,"num":confidenceLevel,"covMatrix":covMatrix}					



					#'''

					# Plot the covariance error ellipse
					ellipse = Ellipse(xy=np.mean(noisyActions,axis=0),
									  width=infoElipse["majorAxis"], height=infoElipse["minorAxis"],
									  angle=infoElipse["rotationAngle"],
									  edgecolor=color, fc='None', lw=2, label='Covariance Error Ellipse')
					plt.gca().add_patch(ellipse)

					c = [mpatches.Ellipse((),width=5,height=5,color="tab:green"),
						mpatches.Ellipse((),width=5,height=5,color="tab:red")]
					legend = ax.legend(c,['Top Pitcher','Bottom Pitcher'], handler_map={mpatches.Ellipse: HandlerEllipse()})
					
					#'''



					#'''

					# https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

					n_std = 0.75

					covMatrix = infoElipse["covMatrix"]

					pearson = covMatrix[0, 1]/np.sqrt(covMatrix[0, 0] * covMatrix[1, 1])
				   
					## Using a special case to obtain the eigenvalues of this
					## two-dimensional dataset.
					ell_radius_x = np.sqrt(1 + pearson)
					ell_radius_y = np.sqrt(1 - pearson)
					ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
									  facecolor="none",edgecolor=color,lw=2, label=label)

					## Calculating the standard deviation of x from
					## the squareroot of the variance and multiplying
					## with the given number of standard deviations.
					scale_x = np.sqrt(covMatrix[0, 0]) * n_std
					mean_x = np.mean(noisyActions,axis=0)[0]

					## calculating the standard deviation of y ...
					scale_y = np.sqrt(covMatrix[1, 1]) * n_std
					mean_y = np.mean(noisyActions,axis=0)[1]

					import matplotlib
					transf = matplotlib.transforms.Affine2D() \
						.rotate_deg(45) \
						.scale(scale_x, scale_y) \
						.translate(mean_x, mean_y)

					ellipse.set_transform(transf + ax.transData)
					ax.add_patch(ellipse)

					#'''
		
				# plt.savefig(f"{folder}{os.sep}PitchType{pitchType}-Chunk{chunk}-CI{infoElipse['num']*100}%.png",bbox_inches='tight')
				plt.savefig(f"{folder}{os.sep}PitchType{pitchType}-Chunk{chunk}-nstd-{n_std}.png",bbox_inches='tight')
				plt.clf()
				plt.close("all")


def getCovMatrix(stdDevs,rho):

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	return covMatrix


def getDetCovMatrixOfAVG():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}DetCovMatrixAVG{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	detInfo = {}


	for pitcherID in resultsDict:

		if pitcherID not in detInfo:
			detInfo[pitcherID] = {}

		for pitchType in resultsDict[pitcherID]:

			if pitchType not in detInfo[pitcherID]:
				detInfo[pitcherID][pitchType] = {}

			for chunk in resultsDict[pitcherID][pitchType]:


				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)

				info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]



				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m\
					and "all" not in m and "resamplingMethod" not in m:


						if m not in detInfo[pitcherID][pitchType]:
							detInfo[pitcherID][pitchType][m] = {"x1":0.0,"x2":0.0,"r":0.0,"count":0,"avgX1":0.0,"avgX2":0.0,"avgR":0.0,"det":0.0}

						if "Multi" in m:
							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = info[m][-1]
							rho = info[tempM][-1]

						# Case: JEEDS
						else:
							estimatedX = info[m][-1]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0


						detInfo[pitcherID][pitchType][m]["x1"] += stdDevs[0]
						detInfo[pitcherID][pitchType][m]["x2"] += stdDevs[1]
						detInfo[pitcherID][pitchType][m]["r"] += rho
						detInfo[pitcherID][pitchType][m]["count"] += 1
				

				
				del loadedDictInfo


	for pitcherID in detInfo:
		for pitchType in detInfo[pitcherID]:
			for m in detInfo[pitcherID][pitchType]:
				detInfo[pitcherID][pitchType][m]["avgX1"] = detInfo[pitcherID][pitchType][m]["x1"]/detInfo[pitcherID][pitchType][m]["count"]
				detInfo[pitcherID][pitchType][m]["avgX2"] = detInfo[pitcherID][pitchType][m]["x2"]/detInfo[pitcherID][pitchType][m]["count"]
				detInfo[pitcherID][pitchType][m]["avgR"] = detInfo[pitcherID][pitchType][m]["r"]/detInfo[pitcherID][pitchType][m]["count"]


	tempMethods = []

	for pitcherID in detInfo:
		for pitchType in detInfo[pitcherID]:
			for m in detInfo[pitcherID][pitchType]:

				if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
				and "all" not in m and "resamplingMethod" not in m:
					
					if m not in tempMethods:
						tempMethods.append(m)

					if "Multi" in m:
						rho = detInfo[pitcherID][pitchType][m]["avgR"]
					# Case: JEEDS
					else:
						rho = 0.0


					stdDevs = [detInfo[pitcherID][pitchType][m]["avgX1"],detInfo[pitcherID][pitchType][m]["avgX2"]]

					# Get Covariance Matrix
					covMatrix = getCovMatrix(stdDevs,rho)

					# Compute the determinant of the covariance matrix
					detCov = np.linalg.det(covMatrix)

					detInfo[pitcherID][pitchType][m]["det"] = detCov



	for pitchType in ["FF"]:

		dfInfo = [list(detInfo.keys())]

		for m in methods:
	
			if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
			and "all" not in m and "resamplingMethod" not in m:	
		
				temp = []

				for pitcherID in detInfo:
					temp.append(detInfo[pitcherID][pitchType][m]["det"])

				dfInfo.append(temp)

	df = pd.DataFrame(np.array(dfInfo).T,columns = ["PitcherID"] + tempMethods)

	df.to_csv(saveAt+"pitchersInfo.csv",index=False)
	# code.interact("...", local=dict(globals(), **locals()))


def getDetCovMatrixGivenChunk():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}DetCovMatrix{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	detInfo = {}


	for pitcherID in resultsDict:

		if pitcherID not in detInfo:
			detInfo[pitcherID] = {}

		for pitchType in resultsDict[pitcherID]:

			if pitchType not in detInfo[pitcherID]:
				detInfo[pitcherID][pitchType] = {}

			for chunk in resultsDict[pitcherID][pitchType]:

				if chunk not in detInfo[pitcherID][pitchType]:
					detInfo[pitcherID][pitchType][chunk] = {}


				textFile = open(f"{saveAt}pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}.txt","w")

				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)

				info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"]

				print("Method | EstimatedXS | EstimatedR | DetCovMatrix",file=textFile)

				for m in methods:

					if "rho" not in m and "pSkills" not in m and "whenResampled" not in m\
					and "all" not in m and "resamplingMethod" not in m:

						# code.interact("...", local=dict(globals(), **locals()))
	
						if "Multi" in m:

							tempM = m.split("-xSkills")[0] +"-rhos"
							stdDevs = info[m][-1]
							rho = info[tempM][-1]

						# Case: JEEDS
						else:
							estimatedX = info[m][-1]
							stdDevs = [estimatedX,estimatedX]
							rho = 0.0

						# Get Covariance Matrix
						covMatrix = getCovMatrix(stdDevs,rho)

						# Compute the determinant of the covariance matrix
						detCov = np.linalg.det(covMatrix)

						print(f"{m} | {stdDevs} | {rho} | {detCov}",file=textFile)

						detInfo[pitcherID][pitchType][chunk][m] = detCov


				# code.interact("...", local=dict(globals(), **locals()))
				
				del loadedDictInfo

		textFile.close()


	tempMethods = []

	for pitchType in ["FF"]:

		for chunk in ["Chunk"]:

			dfInfo = [list(resultsDict.keys())]

			for m in methods:
				if "rho" not in m and "pSkills" not in m and "whenResampled" not in m and "MAP" not in m\
				and "all" not in m and "resamplingMethod" not in m:
					
					if m not in tempMethods:
						tempMethods.append(m)

					temp = []

					for pitcherID in resultsDict:
						temp.append(detInfo[pitcherID][pitchType][chunk][m])

					dfInfo.append(temp)

	df = pd.DataFrame(np.array(dfInfo).T,columns = ["PitcherID"] + tempMethods)

	df.to_csv(saveAt+"pitchersInfo.csv",index=False)
	# code.interact("...", local=dict(globals(), **locals()))


def plotDistribution():

	lookAt = [[0,"Obs-0"],[-1,"Obs-End"]]

	rng = np.random.default_rng(np.random.randint(1,1000000000))

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	XD,YD = np.meshgrid(spaces.targetsPlateXFeet,spaces.targetsPlateZFeet,indexing="ij")
	tempXYD = np.vstack([XD.ravel(),YD.ravel()])

	XYD = np.dstack(tempXYD)[0]

	metric = "Entropy"

	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(0.0,1.0)
	sm = ScalarMappable(cmap=cmap)


	for pitcherID in resultsDict:
		for pitchType in resultsDict[pitcherID]:

			for chunk in resultsDict[pitcherID][pitchType]:

				agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}-chunk{chunk}"

				with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
					loadedDictInfo = pickle.load(handle)


				for m in methods:

					if "pSkills" in m or "rhos" in m or "whenResampled" in m or "allProbs" in m or "allParticles" in m:
						continue


					folders = [f"{args.resultsFolder}{os.sep}plots{os.sep}plotsDistributions{os.sep}",
								f"{args.resultsFolder}{os.sep}plots{os.sep}plotsDistributions{os.sep}{agentFolder}{os.sep}",
								f"{args.resultsFolder}{os.sep}plots{os.sep}plotsDistributions{os.sep}{agentFolder}{os.sep}{m}{os.sep}"]

					for each in folders:
						if not os.path.exists(each):
							os.mkdir(each)

					folder = folders[-1]


					info = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"][m]


					for when,which in lookAt:

						if "JT-QRE" in m:
							# Normal JEEDS (assuming symmetric agents)
							estimatedCovMatrix = sys.modules["domain"].getCovMatrix([info[when],info[when]],0.0)
							estX = np.round(info[when],4)
							info2 = f"X: {estX}"
						else:

							tempM = m.split("-xSkills")[0] +"-rhos"
							rhos = loadedDictInfo[pitcherID][pitchType][chunk]["estimates"][tempM]

							estimatedCovMatrix = sys.modules["domain"].getCovMatrix(info[when],rhos[when])
							estR = np.round(rhos[when],4)
							estX = np.round(info[when],4)
							info2 = f"X: {estX} | R: {estR}"


						distrEst = sys.modules["domain"].draw_noise_sample(rng,mean=[0.0,0.0],covMatrix=estimatedCovMatrix)

						pdfEst = sys.modules["domain"].getNormalDistribution(rng,estimatedCovMatrix,delta,spaces.targetsPlateXFeet,spaces.targetsPlateZFeet)

						metricNum = loadedDictInfo[pitcherID][pitchType][chunk]["entropy"][m][when]

						fig,axs = plt.subplots(figsize=(8,6))

						axs.contourf(XD,YD,pdfEst,cmap=cmap)				

						# axs.axis('equal')

						# cb_ax = fig.add_axes([1.0,0.05,0.02,0.85])
						cbar = fig.colorbar(sm)			

						fig.suptitle(f'Distribution: {info2}\n{metric}: {metricNum:.4f}')

						plt.tight_layout()
						plt.savefig(f"{folder}{os.sep}Obs-{which}.png",bbox_inches='tight')

						plt.clf()
						plt.close("all")
						
						# code.interact("...", local=dict(globals(), **locals()))


def plotObservationsVsEntropy():

	saveAt = f"{args.resultsFolder}{os.sep}plots{os.sep}plotObservationsVsEntropy{os.sep}"

	folders = [saveAt]

	for each in labels:
		folders.append(saveAt+each+os.sep)


	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)


	for ii in range(len(folders[1:])):

		saveAt = folders[ii+1]

		methods = subsetMethods[ii]
		label = labels[ii]

		textFile = open(f"{saveAt}info.txt","w")

		for pitcherID in resultsDict:
			for pitchType in resultsDict[pitcherID]:

				for chunk in resultsDict[pitcherID][pitchType]:

					with open(f"{rdFile}-pitcherID{pitcherID}-pitchType{pitchType}-chunk{chunk}",'rb') as handle:
						loadedDictInfo = pickle.load(handle)

					info = loadedDictInfo[pitcherID][pitchType][chunk]["entropy"]

					fig = plt.figure()
					ax = plt.gca()

					for m in methods:

						if "xSkills" in m:
							plt.plot(range(len(info[m])),info[m],label=m)
							print(f"Pitcher: {pitcherID} | Pitch Type {pitchType} | Method: {m} | Final Entropy: {info[m][-1]}",file=textFile)

					# Set axis labels and legend
					plt.xlabel('Number of Observations')
					plt.ylabel(f"Entropy")
					plt.legend()

					plt.savefig(f"{saveAt}Pitcher{pitcherID}-PitchType{pitchType}-Chunk{chunk}.png",bbox_inches='tight')
					plt.clf()
					plt.close("all")
					
					# code.interact("...", local=dict(globals(), **locals()))
					
					del loadedDictInfo

		textFile.close()


if __name__ == "__main__":


	# ASSUMES RESULTS OF EXPERIMENTS WERE PROCESSED ALREADY


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball-multi")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	result_files = os.listdir(args.resultsFolder + os.sep + "ResultsDictFiles")


	try:
		result_files.remove(".DS_Store")
	except:
		pass


	if len(result_files) == 0:
		print("No processed result files present.")
		exit()


	# If the plots folder doesn't exist already, create it
	if not os.path.exists(args.resultsFolder + os.sep + "plots" + os.sep):
		os.mkdir(args.resultsFolder + os.sep + "plots" + os.sep)


	homeFolder = os.path.dirname(os.path.realpath("skill-estimation-framework")) + os.sep

	# In order to find the "Domains" folder/module to access its files
	sys.path.append(homeFolder)


	resultsDict = {}


	cache.enable()


	namesEstimators = []
	typeTargetsList = []

	numHypsX = []
	numHypsP = []
	seenAgents = []


	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Environments/"
	mainFolderName = scriptPath.split("Processing")[0] + "Environments" + os.sep


	rdFile = args.resultsFolder + os.sep + "ResultsDictFiles" + os.sep + "resultsDictInfo"

	oiFile = args.resultsFolder + os.sep + "plots" + os.sep + "otherInfo"

	with open(oiFile,"rb") as file:
		otherInfo = pickle.load(file)

		namesEstimators = otherInfo["namesEstimators"]
		methods = otherInfo["methods"]
		# methodsAllProbs = otherInfo["methodsAllProbs"]
		numHypsX = otherInfo['numHypsX']
		numHypsP = otherInfo['numHypsP']
		seenAgents = otherInfo["seenAgents"]
		domain = otherInfo["domain"]
		typeTargetsList = otherInfo["typeTargetsList"]
		# betas = otherInfo["betas"]
		resultFilesLoaded = otherInfo["result_files"]

		loadedInfo = True



	total_num_exps = 0

	for rf in result_files: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)

		temp = rf.split("pitcherID")[1].split("-pitchType")

		pitcherID = temp[0]
		pitchType = temp[1].split(".")[0].split("chunk")[0][:-1]
		chunk = temp[1].split(".")[0].split("chunk")[1]
		# code.interact("...", local=dict(globals(), **locals()))
	

		if pitcherID not in resultsDict:
			resultsDict[pitcherID] = {}

		if pitchType not in resultsDict[pitcherID]:
			resultsDict[pitcherID][pitchType] = {}

		if chunk not in resultsDict[pitcherID][pitchType]:
			resultsDict[pitcherID][pitchType][chunk] = {}


	#############################################################################
	# PLOTS
	#############################################################################

	# Parameters for plots
	rcParams.update({'font.size': 14})
	rcParams.update({'legend.fontsize': 14})
	rcParams["axes.labelweight"] = "bold"
	rcParams["axes.titleweight"] = "bold"



	pitcherNames = {}
	
	for pitcherID in resultsDict.keys():

		try:
			result = playerid_reverse_lookup([int(pitcherID)])[["name_first","name_last"]]
			pitcherNames[pitcherID] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"

		except:
			print("Error in playerid_reverse_lookup(). Loading info from pickle instead...\n")

			with open(f"Experiments{os.sep}baseball{os.sep}pitcherNames","rb") as infile:
				names = pickle.load(infile)

			tempName = names[pitcherID].split(", ")
			pitcherNames[pitcherID] = f"{tempName[1]} {tempName[0]}"

		

	xskills_justEES = []

	for m in methods:
		if "EES" in m and "xSkills" in m:
			xskills_justEES.append(m)

	subsetMethods = [methods,xskills_justEES]
	labels = ["All","JustEES"]


	print()

	# print("plotObservationsVsEntropy()...")
	# plotObservationsVsEntropy()


	# print("plotDistribution()...")
	# plotDistribution()

	# getDetCovMatrixGivenChunk()


	# getDetCovMatrixOfAVG()


	plotCovErrorElipse()

	# getAvgEstimateGivenChunk()
	# print("getAvgEstimateGivenChunk()...")


	# getAvgEstimateAcrossChunks()
	# print("getAvgEstimateAcrossChunks()...")


	# getAvgEstimatePerDimensionAcrossChunks()
	# print("getAvgEstimatePerDimensionAcrossChunks()...")


	# Close all remaining figures
	plt.close("all")

	# code.interact("...", local=dict(globals(), **locals()))

