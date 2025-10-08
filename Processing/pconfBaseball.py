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
import pickle
import time


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}pconfBaseball.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


def pconf(xSkills,numObs,pitcherID,pitchType,saveAt,pconfInfo):

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
	print(f"-> {agentFolder}")

	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	agentData = spaces.getAgentData(pitcherID,pitchType,maxRows=1000)

	agentData =  agentData.sample(min(numObs,len(agentData)))

	# code.interact("after...", local=dict(globals(), **locals()))

	for row in agentData.itertuples():

		index = row.Index

		pconfInfo[index] = {}

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
		batch_x = batch_x.reshape((len(batch_x),1,len(sys.modules["modelTake2"].features)))
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


		Zs = np.zeros((len(spaces.modelTargetsPlateX), len(spaces.modelTargetsPlateZ)))

		for i in list(range(len(spaces.modelTargetsPlateX))):
			for j in list(range(len(spaces.modelTargetsPlateZ))):
				tempIndex = np.where((withUtilities.plate_x == spaces.modelTargetsPlateX[i]) & (withUtilities.plate_z == spaces.modelTargetsPlateZ[j]))[0][0]
				Zs[i][j] = np.copy(allTempData.iloc[tempIndex]["utility"])

		# middle = spaces.focalActionMiddle
		# newFocalActions = []

		# max utility
		# maxUtility = np.max(withUtilities["utility"].values)
		# iis = np.where(withUtilities["utility"].values == maxUtility)[0][0]
		# maxUtilityAction = [spaces.possibleTargetsFeet[:,0][iis],spaces.possibleTargetsFeet[:,1][iis]]


		for xi in range(len(xSkills)):

			x = xSkills[xi]

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,pdfsPerXskill[x],mode="same",fillvalue=minUtility)

			'''
			maxEV = np.max(EVs)	
			mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
			action = [spaces.targetsPlateXFeet[mx],spaces.targetsPlateZFeet[mz]]
			'''

			pconfInfo[index][x] = {"EVs":EVs}

	return len(agentData.index)


if __name__ == '__main__':


	folders = [f"Data{os.sep}",f"Data{os.sep}Baseball{os.sep}",
				f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"]

	for folder in folders:
		#If the folder doesn't exist already, create it
		if not os.path.exists(folder):
			os.mkdir(folder)


	# 2.0 inches | 0.17 feet
	startX_Estimator = 0.17
	# 33.72 inches | 2.81 feet
	stopX_Estimator = 2.81

	# 0.5 inches | 0.0417 feet
	delta = 0.0417


	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	copyPlateX = spaces.possibleTargetsForModel[:,0]
	copyPlateZ = spaces.possibleTargetsForModel[:,1]
	possibleTargetsLen = len(spaces.possibleTargetsForModel)





	#xSkills = list(np.concatenate((np.linspace(startX_Estimator,1.0,num=60),np.linspace(1.00+delta,stopX_Estimator,num=6))))
	# print(xSkills)

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

	xSkills = [0.50,0.63,0.75,0.88,1.0]

	pdfsPerXskill = {}

	for x in xSkills:
		pdfsPerXskill[x] = sys.modules["domain"].getSymmetricNormalDistribution(x,delta,spaces.targetsPlateXFeet,spaces.targetsPlateZFeet)
	

	# PITCHERS OF INTEREST
	# Mantiply, Joe: 573009 
	#	CU: 517 | CH: 371 | SL: 0 | FF: 78 | FS: 0 | SI: 725 
	# Scott, Tanner: 656945 | 
	# 	CU: 0 | CH: 0 | SL: 2002 | FF: 1945 | FS: 0 | SI: 52 | FC: 0 | KC: 0 | FA: 0 | CS: 0 | nan: 0 | EP: 0 | FO: 0 | KN: 0 | PO: 0 | SC: 0 | FT: 0 | 
	# info = [["573009", "CU"],["573009", "CH"],["573009", "FF"],["573009", "SI"],
	# 		["656945","SL"],["656945","FF"],["656945","SI"]]


	# JAIR-23 PITCHERS OF INTEREST
	info = [["594798", "CU"], ["594798", "FF"], ["605400", "FF"], ["670950", "FF"], 
			["592773", "FF"], ["670950", "CH"], ["446372", "FC"], ["663855", "SI"],
			["656945", "FF"], ["518617", "SL"], ["661403", "FC"], ["579328", "FC"],
			["592773", "SL"], ["458681", "CU"], ["656354", "SL"], ["579328", "CH"],
			["573009", "SI"], ["518617", "SI"], ["573009", "CU"], ["455119", "SL"],
			["458681", "CH"], ["455119", "SI"], ["605400", "FC"], ["670950", "SL"],
			["669923", "SL"], ["669923", "CH"], ["656354", "CU"], ["672851", "SL"],
			["656945", "SI"], ["672851", "SI"], ["663855", "FC"], ["656354", "SI"],
			["670950", "CS"], ["594798", "CH"], ["594798", "SI"], ["605400", "KC"],
			["579328", "FF"], ["605400", "CH"], ["458681", "FC"], ["592761", "SL"],
			["605400", "SI"], ["579328", "SL"], ["548389", "SL"], ["548389", "KC"],
			["656354", "FF"], ["643493", "FF"], ["592773", "FS"], ["663855", "SL"],
			["643493", "SL"], ["446372", "CH"], ["656353", "SL"], ["643493", "CU"],
			["656354", "CH"], ["573009", "CH"], ["669923", "SI"], ["643493", "CH"],
			["669923", "FC"], ["592761", "CU"], ["656353", "CU"], ["672851", "CH"],
			["573009", "FF"], ["656353", "SI"], ["656353", "CH"], ["455119", "CU"],
			["643493", "SI"], ["592761", "SI"], ["594798", "SL"], ["458681", "FF"],
			["592761", "FF"], ["548389", "FF"], ["458681", "SI"], ["518617", "FF"],
			["446372", "SI"], ["656945", "SL"], ["446372", "CU"], ["455119", "FF"],
			["548389", "CH"], ["592761", "CH"], ["669923", "FF"], ["672851", "FF"],
			["455119", "FC"], ["661403", "SL"], ["670950", "CU"], ["656353", "FF"],
			["446372", "FF"], ["579328", "CU"], ["455119", "FS"], ["669923", "CU"],
			["672851", "CU"], ["670950", "FC"], ["548389", "SI"], ["458681", "SL"],
			["548389", "FC"], ["663855", "CH"], ["661403", "FF"], ["663855", "FF"], 
			["518617", "CH"], ["458681", "CS"], ["446372", "PO"]]

	pitcherNames = {}
	pconfInfo = {}

	numObs = 10
	p = 0

	for each in info:

		print(f"\n\n({p+1}/{len(info)}) - Looking at: {each}")
		p += 1

		result = playerid_reverse_lookup([int(each[0])])[["name_first","name_last"]]
		pitcherNames[int(each[0])] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"

		nameNoSpace = pitcherNames[int(each[0])].replace(' ','')
		
		if not os.path.exists(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}"):
			os.mkdir(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}")

		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}{os.sep}{each[1]}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)

		s = time.time()
		seenObs = pconf(xSkills,numObs,int(each[0]),each[1],saveAt,pconfInfo)
		e = time.time()

		print(f"Total time: {e-s} (seenObs = {seenObs})")


	lambdas = np.linspace(0.001,1300,100)

	size = len(pconfInfo)
	
	pconfPerXskill = {}

	print("\nPCONF...")

	for x in xSkills:

		print(f"xskill: {x}")

		prat = [] #This is where the probability of rational reward will be stored
		mins = [] #Store min reward possible
		maxs = [] #Store max reward possible
		means = [] #Store the mean of the possible rewards (this is the uniform random reward)
		evs = [] #Store the ev of the current agent's strategy

		for l in lambdas:

			max_rs = np.zeros(size)
			min_rs = np.zeros(size)
			exp_rs = np.zeros(size)
			mean_rs = np.zeros(size)

			si = 0

			for s in pconfInfo:

				values = pconfInfo[s][x]["EVs"].flatten()

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


				si += 1

			prat.append(np.mean((exp_rs-mean_rs)/(max_rs-mean_rs)))
			mins.append(np.mean(min_rs))
			means.append(np.mean(mean_rs))
			maxs.append(np.mean(max_rs))
			evs.append(np.mean(exp_rs))

				
		pconfPerXskill[x] = {"lambdas":lambdas, "prat": prat}


	ax = plt.subplot(111)
	
	for b in pconfPerXskill:
		plt.plot(pconfPerXskill[b]["lambdas"], pconfPerXskill[b]["prat"],label=round(b,2))

	plt.xlabel("Rationality Parameter")
	plt.ylabel("Rationality Percentage")	
	plt.legend()
	plt.savefig(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}pconf-NumObs{size}.jpg",bbox_inches="tight")
	plt.clf()
	plt.close()

	# Save dict containing all info - to be able to rerun it later
	with open(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}pconf-NumObs{size}.pickle","wb") as outfile:
		pickle.dump(pconfPerXskill,outfile)





	
