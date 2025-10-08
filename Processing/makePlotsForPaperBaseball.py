import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys,code
from importlib.machinery import SourceFileLoader
import torch
import torch.nn as nn 
from matplotlib.cm import ScalarMappable
from scipy.signal import convolve2d, fftconvolve
from math import dist
from pybaseball import playerid_reverse_lookup, cache

from matplotlib.colors import ListedColormap

from makePlotsStudy5 import *


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}makePlotsForPaperBaseball.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


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


def getBoardPlots(pitcherID,pitchType,saveAt,cmapStr):

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
			
	rng = np.random.default_rng(np.random.randint(1,1000000000))

	# 2.0 inches | 0.17 feet
	startX_Estimator = 0.17
	# 33.72 inches | 2.81 feet
	stopX_Estimator = 2.81

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	xSkills = list(np.concatenate((np.linspace(startX_Estimator,1.0,num=60),np.linspace(1.00+delta,stopX_Estimator,num=6))))
	print(xSkills)

	# ALL XSKILLS
	# [0.17, 0.1840677966101695, 0.198135593220339, 0.2122033898305085, 0.22627118644067798, 
	# 0.24033898305084744, 0.25440677966101694, 0.2684745762711864, 0.2825423728813559, 0.2966101694915254,
	# 0.3106779661016949, 0.3247457627118644, 0.3388135593220339, 0.3528813559322034, 0.36694915254237287, 
	# 0.38101694915254236, 0.39508474576271185, 0.40915254237288134, 0.4232203389830508, 0.43728813559322033, 
	# 0.45135593220338976, 0.4654237288135593, 0.47949152542372875, 0.4935593220338983, 0.5076271186440677,
	# 0.5216949152542373, 0.5357627118644067, 0.5498305084745763, 0.5638983050847457, 0.5779661016949152,
	# 0.5920338983050847, 0.6061016949152542, 0.6201694915254237, 0.6342372881355932, 0.6483050847457626, 
	# 0.6623728813559322, 0.6764406779661016, 0.6905084745762712, 0.7045762711864406, 0.7186440677966102, 
	# 0.7327118644067796, 0.7467796610169491, 0.7608474576271186, 0.7749152542372881, 0.7889830508474576,
	# 0.8030508474576271, 0.8171186440677966, 0.8311864406779661, 0.8452542372881355, 0.8593220338983051,
	# 0.8733898305084745, 0.8874576271186441, 0.9015254237288135, 0.915593220338983, 0.9296610169491525,
	# 0.943728813559322, 0.9577966101694915, 0.971864406779661, 0.9859322033898305, 1.0, 
	# 1.0417, 1.3953600000000002, 1.74902, 2.10268, 2.45634, 2.81]

	# xSkills = [0.17, 0.2825423728813559, 0.40915254237288134,
	#  		0.5216949152542373, 0.7045762711864406, 1.0417, 2.81]

	xSkills = [0.35,0.50,0.63,0.75,0.88,1.0]
	N = len(xSkills)+1

	cmap = plt.get_cmap(cmapStr)
	colors = [cmap(i / (N-1)) for i in range(N)][::-1]


	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	copyPlateX = spaces.possibleTargetsForModel[:,0]
	copyPlateZ = spaces.possibleTargetsForModel[:,1]
	possibleTargetsLen = len(spaces.possibleTargetsForModel)


	pdfsPerXskill = {}

	for x in xSkills:
		pdfsPerXskill[x] = sys.modules["domain"].getSymmetricNormalDistribution(rng,x,delta,spaces.targetsPlateXFeet,spaces.targetsPlateZFeet)

		fig,ax = plt.subplots()
		cs = ax.scatter(spaces.possibleTargetsFeet[:,0],spaces.possibleTargetsFeet[:,1],c = pdfsPerXskill[x])
		cbar = fig.colorbar(cs,ax=ax)
		plt.legend()
		plt.savefig(f"{saveAt}{os.sep}pdfsPerXskill{x}.jpg",bbox_inches="tight")
		plt.close()
		plt.clf()


	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))

	# code.interact("after...", local=dict(globals(), **locals()))
	# code.interact("after...", local=dict(globals(), **locals()))

	# cmap = plt.cm.jet(np.arange(n))
	# cmap[:,0:3] /= c 
	cmap = ListedColormap(cmap)


	dataBy = "recent"
	toSend = [10]

	agentData = spaces.getAgentData(dataBy,pitcherID,pitchType,toSend)

	saveAtOriginal = saveAt

	# code.interact("after...", local=dict(globals(), **locals()))


	for row in agentData.itertuples():

		index = row.Index

		saveAt = f"{saveAtOriginal}{os.sep}Index-{index}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)


		allTempData = pd.DataFrame([row]*(possibleTargetsLen))

		# Update position of each copy of the row to be that of a given possible action
		allTempData["plate_x"] = copyPlateX
		allTempData["plate_z"] = copyPlateZ


		# Include original 'row' (df with actual pitch info) to get the probabilities 
		# for the different outcomes as well as the utility - for the actual pitch
		allTempData.loc[len(allTempData.index)+1] = row


		########################################
		# NEW MODEL
		########################################

		batch_x = allTempData[sys.modules["modelTake2"].features].values
		batch_y = allTempData.outcome.values.astype(int)

		#Reshape so that each pa is a separate entry in the batch
		batch_x = batch_x.reshape((len(batch_x), 1,len(sys.modules["modelTake2"].features)))
		batch_y = batch_y.reshape((len(batch_y),1))

		torch_batch_x = torch.tensor(batch_x,dtype=torch.float)
		torch_batch_y = torch.tensor(batch_y,dtype=torch.long)
		

		ypred = sys.modules["modelTake2"].prediction_func(spaces.model,torch_batch_x,torch_batch_y)

		allTempData[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = nn.functional.softmax(ypred, dim = 1).detach().cpu().numpy()
		
		# code.interact("after...", local=dict(globals(), **locals()))

		########################################

		
		# Get utilities
		withUtilities = sys.modules["utilsBaseball"].getUtility(allTempData)


		# Get updated info for actual pitch (actual pitch + probs + utility)
		row = withUtilities.iloc[-1].copy()

		# Remove actual pitch from data
		withUtilities = withUtilities.iloc[:-1,:]
		
		minUtility = np.min(withUtilities["utility"].values)


		norm = plt.Normalize(minUtility, max(withUtilities["utility"].values))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])


		###############################################################
		# Strike Zone Board
		###############################################################

		fileName = f"{agentFolder}-index{index}"
		
		
		# PLOT - RAW BOARD

		fig,ax = plt.subplots()

		cbar = fig.colorbar(sm,ax=ax)
		cbar.ax.get_yaxis().labelpad = 15
		cbar.ax.set_ylabel("Expected Utilities",rotation = 270)
		ax.set_aspect('equal')
		plt.xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
		# plt.ylim([min(spaces.possibleTargetsFeet[:,1]),max(spaces.possibleTargetsFeet[:,1])])
		plt.ylim(0.5,4.5)


		# max utility
		maxUtility = np.max(withUtilities["utility"].values)
		iis = np.where(withUtilities["utility"].values == maxUtility)[0][0]
		maxUtilityAction = [spaces.possibleTargetsFeet[:,0][iis],spaces.possibleTargetsFeet[:,1][iis]]


		ax.scatter(spaces.possibleTargetsFeet[:,0],spaces.possibleTargetsFeet[:,1],c = cmap(norm(withUtilities["utility"].values)))

		titleStr = f"{pitcherNames[pitcherID]} - {pitchType}"
		plotOtherInfo("")

		# Plot actual executed action & EV
		ax.scatter(row["plate_x_feet"],row["plate_z_feet"],color="red",marker="X",s=60,edgecolors="black",label="Actual Pitch")
		ax.scatter(maxUtilityAction[0],maxUtilityAction[1],color="black",marker="X",s=60,edgecolors="black",label="Max Expected Utility")
		plt.legend()

		plt.savefig(f"{saveAt}{fileName}.jpg",bbox_inches="tight")
		plt.close()
		plt.clf()


		###############################################################
		

		# Populate Dartboard - Can create once since independent of xskill
		Zs = np.zeros((len(spaces.modelTargetsPlateX), len(spaces.modelTargetsPlateZ)))

		for i in list(range(len(spaces.modelTargetsPlateX))):
			for j in list(range(len(spaces.modelTargetsPlateZ))):
				tempIndex = np.where((withUtilities.plate_x == spaces.modelTargetsPlateX[i]) & (withUtilities.plate_z == spaces.modelTargetsPlateZ[j]))[0][0]
				Zs[i][j] = np.copy(allTempData.iloc[tempIndex]["utility"])

		middle = spaces.focalActionMiddle
		newFocalActions = []


		allActions = []

		for x in xSkills:

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,pdfsPerXskill[x],mode="same",fillvalue=minUtility)
			
			# FOR TESTING
			# EVs = np.ones(Zs.shape)

			maxEV = np.max(EVs)	
			mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
			action = [spaces.targetsPlateXFeet[mx],spaces.targetsPlateZFeet[mz]]
			allActions.append(action)


			# Adding extra focal actions to default set:
			# 	- target for best xskill hyp
			#	- other targets if they are more than 0.16667 feet (or 2 inches) away 
			# 	  from middle target OR last focal target added
			if action not in newFocalActions:
				if (x == np.min(xSkills)) or dist(action,middle) >= 0.16667 or dist(action,newFocalActions[-1]) >= 0.16667:
					newFocalActions.append(action)
			

			# PLOT - BOARD EV's given xskill

			fig,ax = plt.subplots()
		
			#norm = plt.Normalize(np.min(EVs),np.max(EVs))
			norm = plt.Normalize(minUtility, max(withUtilities["utility"].values))
			sm = ScalarMappable(norm=norm,cmap=cmap)
			sm.set_array([])
			cbar = fig.colorbar(sm,ax=ax)
			cbar.ax.get_yaxis().labelpad = 15
			cbar.ax.set_ylabel("Expected Utilities",rotation = 270)

			ax.scatter(spaces.possibleTargetsFeet[:,0],spaces.possibleTargetsFeet[:,1],c = cmap(norm(EVs.flatten())))
			ax.set_xlabel("targetsPlateX")
			ax.set_ylabel("targetsPlateZ")

			ax.set_aspect('equal')
			plt.xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
			# plt.ylim([min(spaces.possibleTargetsFeet[:,1]),max(spaces.possibleTargetsFeet[:,1])])
			plt.ylim(0.5,4.5)

			ax.scatter(action[0],action[1],color="black",marker="X",s=60,edgecolors="black",label="Max Expected Utility")

			titleStr = f"{pitcherNames[pitcherID]} - Pitch: {pitchType} - Execution Skill {format(x,'.4f')}"
			plotOtherInfo("")
			plt.legend()

			# Plot best action & EV
			#ax.scatter(action[0],action[1],color=cmap(norm(maxEV)),marker="*",edgecolors="black")
			plt.savefig(f"{saveAt}{fileName}-xskill{str(round(x,4)).replace('.','-')}.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()


			# code.interact("after...", local=dict(globals(), **locals()))


		# Plot all max ev actions same plot
		numTries = 500
		allNoisyActions,allHits = getPlots(numTries,xSkills,rhos=[0.0])


		fig,ax = plt.subplots()
		
		allActions = np.array(allActions)

		ax.scatter(row["plate_x_feet"],row["plate_z_feet"],color="red",marker="X",s=60,edgecolors="black",label="Actual Pitch")
		ax.scatter(maxUtilityAction[0],maxUtilityAction[1],color="black",marker="X",s=60,edgecolors="black",label="Max Expected Utility")

		for ii,action in enumerate(allActions):
			xs = xSkills[ii]
			key = getKey([xs,xs],r=0.0)
			hits = allHits[key]
			ax.scatter(action[0],action[1],color=colors[ii+1],marker="X",s=60,edgecolors="black",label=f"{(hits/numTries)*100:.2f}%")

		ax.set_aspect('equal')
		plt.xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
		plt.ylim(0.5,4.5)

		plotOtherInfo("")
		plt.legend()
		plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")

		plt.savefig(f"{saveAt}{fileName}-allLocations.jpg",bbox_inches="tight")
		plt.close()
		plt.clf()


def getFocalActionsPlot():

	minPlateX = -2.13
	maxPlateX = 2.13
	xs = [minPlateX,maxPlateX]

	minPlateZ = -2.50
	maxPlateZ = 6.60
	ys = [minPlateZ,maxPlateZ]

	defaultFocalActions = []
	for ii in xs:
		for jj in ys:
			defaultFocalActions.append([ii,jj])

	defaultFocalActions = np.array(defaultFocalActions)


	minX = -0.71
	maxX = 0.71
	minZ = 1.546
	maxZ = 3.412

	# Overlay strike zone dimensions on plot
	plate_x = [minX,maxX]
	plate_z = [minZ,maxZ]

	targets = []

	for xi in plate_x:
		for zi in plate_z:
			targets.append([xi,zi])
			
	# Targets in line with strikezone (midpoint on grid)
	targets.append([plate_x[0],(plate_z[0]+plate_z[1])/2])
	targets.append([plate_x[1],(plate_z[0]+plate_z[1])/2])
	targets.append([(plate_x[0]+plate_x[1])/2,plate_z[0]])
	targets.append([(plate_x[0]+plate_x[1])/2,plate_z[1]])


	# Middle of strike zone
	middle = [(plate_x[0]+plate_x[1])/2,(plate_z[0]+plate_z[1])/2]
	targets.append(middle)
	print("MIDDLE: ",middle)


	# Inner quadrants
	temp1 = (targets[4][1]+plate_z[0])/2
	temp2 = (targets[4][1]+plate_z[1])/2
	targets.append([(targets[4][0]+middle[0])/2,temp1])
	targets.append([(targets[5][0]+middle[0])/2,temp1])
	targets.append([(targets[4][0]+middle[0])/2,temp2])
	targets.append([(targets[5][0]+middle[0])/2,temp2])


	# Outer - Cross
	dist = plate_z[0]-(plate_z[0]+plate_z[1])/4
	targets.append([middle[0],plate_z[0]-dist])
	targets.append([middle[0],plate_z[1]+dist])
	targets.append([plate_x[0]-dist,middle[1]])
	targets.append([plate_x[1]+dist,middle[1]])


	# Outer - Diagonal
	yB = targets[-4][1]
	yT = targets[-3][1]
	xL = targets[-2][0]
	xR = targets[-1][0]
	targets.append([xL,yB])
	targets.append([xL,yT])
	targets.append([xR,yB])
	targets.append([xR,yT])


	targets = np.array(targets)
	# print(targets)

	# Format target array like to use on framework 
	# Also rounds to 4 decimal places
	aStr = "["
	for t in targets:
		aStr += f"[{round(t[0],4)},{round(t[1],4)}],"

	aStr = aStr[:-1] + "]"
	print(aStr)

	fig,ax = plt.subplots()

	plt.scatter(targets[:,0],targets[:,1],marker="x")

	titleStr = "Focal Points for AXE"
	plotOtherInfo(titleStr)

	#plt.axis('equal')
	ax.set_aspect('equal', adjustable='box')
	plt.xlim(-1.5,1.5)
	plt.ylim(1,4)

	plt.savefig(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}FocalPoints.png", bbox_inches='tight')
	plt.clf()
	plt.close()


if __name__ == '__main__':


	try:
		cmapStr = sys.argv[1]
	except:
		print("Need to specify colormap to use as command line argument.")
		exit()


	folders = [f"Data{os.sep}",f"Data{os.sep}Baseball{os.sep}",
				f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"]

	for folder in folders:
		#If the folder doesn't exist already, create it
		if not os.path.exists(folder):
			os.mkdir(folder)


	getFocalActionsPlot()


	# PITCHERS OF INTEREST
	# info = [["547973", "FF"]]
	'''
	info = [["547973", "FF"], ["547973", "CH"], ["547973", "FS"], ["547973", "SL"], ["547973", "SI"],
			["594798", "CH"], ["594798", "SI"], ["594798", "SL"], ["594798", "CU"], ["594798", "FF"],
			["623433", "FF"], ["623433", "CU"], ["623433", "SI"], ["623433", "CH"],
			["445276", "FF"], ["445276", "SI"], ["445276", "SL"], ["445276", "FC"],
			["605483", "CU"], ["605483", "FF"], ["605483", "CH"], ["605483", "SL"]]
	'''
	#pitchers = [547973,594798,623433,445276,605483]
	
	# info = [["547973", "SL"]]
	

	# Mantiply, Joe: 573009 
	#	CU: 517 | CH: 371 | SL: 0 | FF: 78 | FS: 0 | SI: 725 
	# Scott, Tanner: 656945 | 
	# 	CU: 0 | CH: 0 | SL: 2002 | FF: 1945 | FS: 0 | SI: 52 | FC: 0 | KC: 0 | FA: 0 | CS: 0 | nan: 0 | EP: 0 | FO: 0 | KN: 0 | PO: 0 | SC: 0 | FT: 0 | 

	# info = [["573009", "SI"],["573009", "CU"],["573009", "CH"],["573009", "FF"],
			# ["656945","SL"],["656945","FF"],["656945","SI"]]

	# info = [["445276", "SL"], ["445276", "FF"], ["445276", "SI"], ["445276", "FC"], ["623433", "CU"], ["623433", "CH"], ["623433", "FF"], ["623433", "SI"]]
	info = [["573009", "CU"]]


	pitcherNames = {}

	for each in info:

		result = playerid_reverse_lookup([int(each[0])])[["name_first","name_last"]]
		pitcherNames[int(each[0])] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"

		nameNoSpace = pitcherNames[int(each[0])].replace(' ','')
		
		if not os.path.exists(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}"):
			os.mkdir(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}")

		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}{os.sep}{each[1]}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)

		getBoardPlots(int(each[0]),each[1],saveAt,cmapStr)


	
