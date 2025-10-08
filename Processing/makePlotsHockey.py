from matplotlib import rcParams,rc
from matplotlib import pyplot as plt

import argparse,sys,os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
import code

from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

import csv


leftPost = np.array([89,-3])
rightPost = np.array([89,3])


def getAngle(point1,point2):

    x1,y1 = point1
    x2,y2 = point2
    
    angle = np.arctan2(y2-y1,x2-x1)
    
    return angle


def getAngularCoordinates(playerLocation):

	# Generate edges - directions
	dirL = getAngle(playerLocation,leftPost)
	dirR = getAngle(playerLocation,rightPost)

	# Generate edges - elevations
	dist1 = np.linalg.norm(playerLocation-leftPost)
	dist2 = np.linalg.norm(playerLocation-rightPost)

	minDist = min(dist1,dist2)
	elevationTop = np.arctan2(4,minDist)


	dirs = [dirL,dirR]
	elevations = [0,elevationTop]

	center = [(dirL+dirR)/2,elevationTop/2]

	return dirs,elevations,center


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=orig_handle.width,
                                        height=orig_handle.height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def plotOtherInfo(dirs,elevations,titleStr=""):

	# Overlay strike zone dimensions on plot
	plt.hlines(y=elevations[0],xmin=dirs[0],xmax=dirs[1],color="k")
	plt.hlines(y=elevations[1],xmin=dirs[0],xmax=dirs[1],color="k")
	plt.vlines(x=dirs[0], ymin=elevations[0],ymax=elevations[1],color="k")
	plt.vlines(x=dirs[1], ymin=elevations[0],ymax=elevations[1],color="k")

	plt.xlabel("")
	plt.ylabel("")

	plt.title(titleStr)
	plt.tight_layout()


def computeCovErrorElipse(method,estimatedXS,estimatedR,each,percentCI):

	k = 2
	covMatrix = np.zeros((k,k))


	if "Multi" in method:
		estX = np.round(estimatedXS[each],4)
		estR = round(estimatedR[each],4)
	# Normal JEEDS (assuming symmetric agents)
	else:
		estX = np.round([estimatedXS[each],estimatedXS[each]],4)
		estR = 0.0


	#############################################
	# CREATE COVARIANCE MATRIX
	#############################################

	np.fill_diagonal(covMatrix,np.square(estX))

	# Fill the upper and lower triangles
	for i in range(k):
		for j in range(i+1,k):
			covMatrix[i,j] = np.prod(estX) * estR
			covMatrix[j,i] = covMatrix[i,j]

	# code.interact("...", local=dict(globals(), **locals()))

	#############################################


	# Using Z-Distribution
	alpha = 1 - percentCI
	num = stats.norm.ppf(1-alpha/2)


	# Calculate ellipse parameters
	eigenvalues, eigenvectors = np.linalg.eig(covMatrix)
	major_axis = 2 * np.sqrt(num*eigenvalues[0])
	minor_axis = 2 * np.sqrt(num*eigenvalues[1])
	rotation_angle = np.arctan2(eigenvectors[1,0],eigenvectors[0,0])*(180/np.pi)

	# Save info
	return {"majorAxis": major_axis,
		   "minorAxis": minor_axis,
	 	   "rotationAngle": rotation_angle}


def plotCovError(saveAt):

	ids = loadedDictInfo["ids"]

	noisyActions = loadedDictInfo["noisy_actions"]
	noisyActions = np.array(noisyActions)	


	saveAtOriginal = f"{saveAt}CovErrorElipse{os.sep}"
	
	if not os.path.exists(saveAtOriginal):
		os.mkdir(saveAtOriginal)


	playerLocations = [[69,0]]#,[79,0],[59,0],[69,10],[69,30]]

	for infoM in [["JEEDS",'JT-QRE-EES-33-33-xSkills'],["PFE",'QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-xSkills']]:
		
		m = infoM[1]
		#print("\n",m)

		saveAtMain = f"{saveAtOriginal}{infoM[0]}{os.sep}"

		if not os.path.exists(saveAtMain):
			os.mkdir(saveAtMain)


		for playerLocation in playerLocations:

			saveAt = f"{saveAtMain}{playerLocation}{os.sep}"

			if not os.path.exists(saveAt):
				os.mkdir(saveAt)


			estimatedXS = loadedDictInfo[m]

			if "Multi" in m:
				tempM = m.split("-xSkills")[0] +"-rhos"
				estimatedR = loadedDictInfo[tempM]
			else:
				estimatedR = 0.0


			# print(estimatedXS)
			# print(estimatedR)


			for each in range(len(estimatedXS)):

				index = ids[each]

				fig, ax = plt.subplots()


				dirs,elevations,centerNet = getAngularCoordinates(playerLocation)
				plotOtherInfo(dirs,elevations,"")
				

				for percentCI, color in [[0.50,"blue"],[0.90,"black"]]:

					infoElipse = computeCovErrorElipse(m,estimatedXS,estimatedR,each,percentCI)


					if "Multi" in m:
						estX = np.round(estimatedXS[each],4)
						estR = np.round(estimatedR[each],4)

						info2 = f"X{estX}-R{estR}"

					else:
						estX = np.round(estimatedXS[each],4)
						info2 = f"X{estX}"


					# Plot the covariance error ellipse
					ellipse = Ellipse(xy=centerNet,
									  width=infoElipse["majorAxis"], height=infoElipse["minorAxis"],
									  angle=infoElipse["rotationAngle"],
									  edgecolor=color, fc='None', lw=2, label='Covariance Error Ellipse')
					plt.gca().add_patch(ellipse)

				c = [mpatches.Ellipse((),width=5,height=5,color="blue"),
					mpatches.Ellipse((),width=5,height=5,color="black")]
				legend = ax.legend(c,['50%','90%'], handler_map={mpatches.Ellipse: HandlerEllipse()})
									


				plt.savefig(f"{saveAt}{each}-{info2}-{playerLocation}.png",bbox_inches='tight')
				plt.clf()
				plt.close("all")

				# code.interact("...", local=dict(globals(), **locals()))



			# Plot just last estimate on separately

			each = len(estimatedXS)-1
			index = ids[each]

			fig, ax = plt.subplots()


			dirs,elevations,centerNet = getAngularCoordinates(playerLocation)
			plotOtherInfo(dirs,elevations,"")
			

			for percentCI, color in [[0.50,"blue"],[0.90,"black"]]:

				infoElipse = computeCovErrorElipse(m,estimatedXS,estimatedR,each,percentCI)


				if "Multi" in m:
					estX = np.round(estimatedXS[each],4)
					estR = np.round(estimatedR[each],4)

					info2 = f"X{estX}-R{estR}"

				else:
					estX = np.round(estimatedXS[each],4)
					info2 = f"X{estX}"


				# Plot the covariance error ellipse
				ellipse = Ellipse(xy=centerNet,
								  width=infoElipse["majorAxis"], height=infoElipse["minorAxis"],
								  angle=infoElipse["rotationAngle"],
								  edgecolor=color, fc='None', lw=2, label='Covariance Error Ellipse')
				plt.gca().add_patch(ellipse)

			c = [mpatches.Ellipse((),width=5,height=5,color="blue"),
				mpatches.Ellipse((),width=5,height=5,color="black")]
			legend = ax.legend(c,['50%','90%'], handler_map={mpatches.Ellipse: HandlerEllipse()})
								


			plt.savefig(f"{saveAtMain}{each}-{info2}-{playerLocation}.png",bbox_inches='tight')
			plt.clf()
			plt.close("all")



def plotNumObsVsEstimates(saveAt):

	saveAt = f"{saveAt}NumObsVsEstimates{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)

	for infoM in methods:
				
		m = infoM[1]

		estimatedXS = loadedDictInfo[m]
		estimatedXS = np.array(estimatedXS)

		if "Multi" in m:
			tempM = m.split("-xSkills")[0]+"-rhos"
			estimatedR = loadedDictInfo[tempM]

			tempM = m.split("-xSkills")[0]+"-pSkills"
			estimatedPS = loadedDictInfo[tempM]

		else:
			tempM = m.replace("-xSkills","-pSkills")
			estimatedPS = loadedDictInfo[tempM]


		if "JEEDS" in infoM[0]:
			fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10,10))

			ax1.scatter(range(len(estimatedXS)),estimatedXS)
			ax1.plot(range(len(estimatedXS)),estimatedXS)
			ax1.set_title(f'X - Final Estimate: {estimatedXS[-1]}')

			ax2.scatter(range(len(estimatedPS)),estimatedPS)
			ax2.plot(range(len(estimatedPS)),estimatedPS)
			ax2.set_title(f'Lambda - Final Estimate: {estimatedPS[-1]}')

		else:

			fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, figsize=(10,20))

			ax1.scatter(range(len(estimatedXS)),estimatedXS[:,0])
			ax1.plot(range(len(estimatedXS)),estimatedXS[:,0])
			ax1.set_title(f'X1 - Final Estimate: {estimatedXS[:,0][-1]}')


			ax2.scatter(range(len(estimatedXS)),estimatedXS[:,1])
			ax2.plot(range(len(estimatedXS)),estimatedXS[:,1])
			ax2.set_title(f'X2 - Final Estimate: {estimatedXS[:,1][-1]}')

			ax3.scatter(range(len(estimatedR)),estimatedR)
			ax3.plot(range(len(estimatedR)),estimatedR)
			ax3.set_title(f'R - Final Estimate: {estimatedR[-1]}')

			ax4.scatter(range(len(estimatedPS)),estimatedPS)
			ax4.plot(range(len(estimatedPS)),estimatedPS)
			ax4.set_title(f'Lambda - Final Estimate: {estimatedPS[-1]}')


		plt.tight_layout()

		plt.savefig(f"{saveAt}{os.sep}{infoM[0]}.jpg",bbox_inches="tight")		
		plt.close()



def colorCells(val):

	try: 
		color = infoPlayers[val][1]
	except:
		for pid in infoPlayers:
			if infoPlayers[pid][0] == val: 	
				color = infoPlayers[pid][1]
				break

	return 'background-color: %s' % color


def makeCSV(expFolder):

	saveAt = f"{expFolder}plots{os.sep}CSV{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	finalEstimatesList = []

	for playerID in finalEstimates.keys():

		name, color = infoPlayers[playerID]	

		for typeShot in finalEstimates[playerID]:

			estimatesAcrossMethods = []
			numObsAcrossMethods = []

			for m in allMethods:

				est = finalEstimates[playerID][typeShot][m]["estimate"]	
				estimatesAcrossMethods.append(est)

				numObs = finalEstimates[playerID][typeShot][m]["numObservations"]	
				numObsAcrossMethods.append(numObs)

			# ASSUMING SAME NUMBER OF OBSERVATIONS ACROSS METHODS FOR NOW
			finalEstimatesList.append([playerID,name,typeShot,numObsAcrossMethods[0]] + estimatesAcrossMethods)


	tempMethods = []
	for m in allMethods:
		tempMethods.append(methodsRename[m])


	# saveTo = open(f"{saveAt}finalEstimatesInfo.csv","w")

	# csvWriter = csv.writer(saveTo)

	columns = ["ID","Name","Shot Type","NumObservations"] + tempMethods

	# csvWriter.writerow(columns)
	
	# for i in range(len(finalEstimatesList)):
		# csvWriter.writerow(finalEstimatesList[i])

	# saveTo.close()

	df = pd.DataFrame(finalEstimatesList)
	df.columns = columns


	# Group by shot type. Sort by JEEDS.
	df = df.groupby('Shot Type').apply(lambda x: x.sort_values('x-JEEDS-EES')).reset_index(drop=True)

	# print(df)

	try:
		df.style.map(colorCells, subset=["ID","Name"]).to_excel(f"{saveAt}finalEstimatesInfo.xlsx", engine='openpyxl')
	except:
		df.style.applymap(colorCells, subset=["ID","Name"]).to_excel(f"{saveAt}finalEstimatesInfo.xlsx", engine='openpyxl')



def getCovMatrix(stdDevs,rho):
	# print("stdDevs: ",stdDevs)
	# print("rho",rho)

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	# print("covMatrix")
	# print(covMatrix)
	return covMatrix


def getDetCovMatrix(expFolder):

	saveAt = f"{expFolder}plots{os.sep}DetCovMatrix{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	detInfo = {}
	typeShots = []


	for playerID in finalEstimates:

		if playerID not in detInfo:
			detInfo[playerID] = {}


		for typeShot in finalEstimates[playerID]:

			if typeShot not in detInfo[playerID]:
				detInfo[playerID][typeShot] = {}
				typeShots.append(typeShot)
		

			textFile = open(f"{saveAt}playerID_{playerID}-typeShot_{typeShot}.txt","w")


			print("Method | EstimatedXS | EstimatedR | DetCovMatrix",file=textFile)

			for m in allMethods:

				if "rho" not in m and "pSkills" not in m and "whenResampled" not in m\
				and "all" not in m and "resamplingMethod" not in m:

					if "Multi" in m:

						tempM = m.split("-xSkills")[0] +"-rhos"
						stdDevs = finalEstimates[playerID][typeShot][m]["estimate"] 
						rho = finalEstimates[playerID][typeShot][tempM]["estimate"] 


					# Case: JEEDS
					else:
						estimatedX = finalEstimates[playerID][typeShot][m]["estimate"] 
						stdDevs = [estimatedX,estimatedX]
						rho = 0.0

					# Get Covariance Matrix
					covMatrix = getCovMatrix(stdDevs,rho)

					# Compute the determinant of the covariance matrix
					detCov = np.linalg.det(covMatrix)

					print(f"{m} | {stdDevs} | {rho} | {detCov}",file=textFile)

					detInfo[playerID][typeShot][m] = detCov



		textFile.close()




	dfInfo = []
	methodsPresent = []

	for playerID in detInfo:

		name, color = infoPlayers[playerID]	

		for typeShot in detInfo[playerID]:
			
			detsAcrossMethods = []

			for m in detInfo[playerID][typeShot]:

				if m not in methodsPresent:
					methodsPresent.append(m)

				detsAcrossMethods.append(detInfo[playerID][typeShot][m])

			dfInfo.append([playerID,name,typeShot] + detsAcrossMethods)



	tempMethods = []
	for m in methodsPresent:
		tempMethods.append(methodsRename[m])


	columns = ["ID","Name","Shot Type"] + tempMethods

	df = pd.DataFrame(dfInfo)
	df.columns = columns


	for sortBy in ['x-JEEDS-EES','x-PF-EES']:
		# Group by shot type. Sort by JEEDS.
		df = df.groupby('Shot Type').apply(lambda x: x.sort_values(sortBy)).reset_index(drop=True)

		try:
			df.style.map(colorCells, subset=["ID"]).to_excel(f"{saveAt}detCovMatrixInfo-{sortBy}.xlsx", engine='openpyxl')
		except:
			df.style.applymap(colorCells, subset=["ID"]).to_excel(f"{saveAt}detCovMatrixInfo-{sortBy}.xlsx", engine='openpyxl')


	# code.interact("...", local=dict(globals(), **locals()))



if __name__ == '__main__':


	try:
		experimentFolder = sys.argv[1]
	except:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()


	expFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Experiment{os.sep}"


	saveAtMain = f"{expFolder}{os.sep}plots{os.sep}"
	
	if not os.path.exists(saveAtMain):
		os.mkdir(saveAtMain)


	methods = [["JEEDS-EES",'JT-QRE-EES-33-33-xSkills'],
				["JEEDS-MAP",'JT-QRE-MAP-33-33-xSkills'],
				["PFE-EES",'QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-xSkills'],
				["PFE-MAP",'QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-xSkills']]


	allMethods = ["JT-QRE-EES-33-33-xSkills",
				"JT-QRE-MAP-33-33-xSkills",
				"JT-QRE-EES-33-33-pSkills",
				"JT-QRE-MAP-33-33-pSkills",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-xSkills",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-xSkills",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-rhos",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-rhos",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-pSkills",
				"QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-pSkills"]



	methodsRename = {"JT-QRE-EES-33-33-xSkills":"x-JEEDS-EES",
				 "JT-QRE-MAP-33-33-xSkills":"x-JEEDS-MAP",
				 "JT-QRE-EES-33-33-pSkills":"p-JEEDS-EES",
				 "JT-QRE-MAP-33-33-pSkills":"p-JEEDS-MAP",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-xSkills":"x-PF-EES",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-xSkills":"x-PF-MAP",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-rhos":"r-PF-EES",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-rhos":"r-PF-MAP",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES-pSkills":"p-PF-EES",
				 "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP-pSkills":"p-PF-MAP"}



	dataFolder = f"Data{os.sep}Hockey{os.sep}"

	with open(f"{dataFolder}infoPlayers.txt","r") as infile:
		loadedInfo = infile.readlines()


	splittedInfo = list(map(lambda x:x.split(","),loadedInfo))

	temp = map(lambda temp:{temp[0]:[temp[1],temp[2].strip()]},splittedInfo)


	infoPlayers = {}

	for d in temp:
		infoPlayers.update(d)



	finalEstimates = {}


	files = os.listdir(f"{expFolder}results{os.sep}")

	for each in files:

		if "OnlineExp" in each:
			rf = each


			with open(f"{expFolder}results{os.sep}{rf}",'rb') as handle:
				loadedDictInfo = pickle.load(handle)

			splitted = each.split("_")

			playerID = splitted[2]
			typeShot = splitted[4]

			agent = f"{playerID}-{typeShot}"

			saveAt = f"{saveAtMain}{agent}{os.sep}"


			if not os.path.exists(saveAt):
				os.mkdir(saveAt)



			plotNumObsVsEstimates(saveAt)

			plotCovError(saveAt)



			# Grab final estimates for each method
			for infoM in allMethods:

				if playerID not in finalEstimates:
					finalEstimates[playerID] = {}

				if typeShot not in finalEstimates[playerID]:
					finalEstimates[playerID][typeShot] = {}

				if infoM not in finalEstimates[playerID][typeShot]:
					finalEstimates[playerID][typeShot][infoM] = {}

				finalEstimates[playerID][typeShot][infoM]["estimate"] = loadedDictInfo[infoM][-1]
				finalEstimates[playerID][typeShot][infoM]["numObservations"] = len(loadedDictInfo[infoM])


	makeCSV(expFolder)

	getDetCovMatrix(expFolder)

	# code.interact("...", local=dict(globals(), **locals()))


