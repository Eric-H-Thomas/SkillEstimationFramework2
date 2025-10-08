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
mainFolderName = scriptPath.split(f"Processing{os.sep}pconfBaseballTake2.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


def pconf(pitchersInfo,pitcherID,pitchType,numObs,needToHave):

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
	print(f"-> {agentFolder}")

	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	agentData = spaces.getAgentData(pitcherID,pitchType,maxRows=1000)


	if len(agentData) < needToHave:
		print(f"Skipping {agentFolder} as it doesn't have at least {needToHave} obs")
		pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prats"] = []
		pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prat"] = -1
		return


	agentData =  agentData.sample(numObs)

	x = pitchersInfo[pitcherID]["pitchTypes"][pitchType]["xs"]
	l = pitchersInfo[pitcherID]["pitchTypes"][pitchType]["p"]

	# code.interact("after...", local=dict(globals(), **locals()))

	for row in agentData.itertuples():

		index = row.Index

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


		# Convolve to produce the EV and aiming spot
		EVs = convolve2d(Zs,pdfsPerXskill[x],mode="same",fillvalue=minUtility)


		values = EVs.flatten()

		# Get the values from the ev 
		max_rs = np.max(values)
		mean_rs = np.mean(values) 

		# Bounded decision-making with lambda = l
		b = np.max(values*l)
		expev = np.exp(values*l-b)
		sumexp = np.sum(expev)
		P = expev/sumexp

		# Store bounded agent's EV
		exp_rs = sum(P*values)

		prat = np.mean((exp_rs-mean_rs)/(max_rs-mean_rs))

		pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prats"].append(prat)


	# Find avg prats
	pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prat"] = sum(pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prats"])/len(pitchersInfo[pitcherID]["pitchTypes"][pitchType]["prats"])


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


	rng = np.random.default_rng(np.random.randint(1,1000000000))

	try:
		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"
		tempName = "pitchersInfo.txt"

		with open(saveAt+tempName, "r") as file:
			data = file.readlines()
	except:
		print("Pitcher file not present.")
		exit()


	pitchersInfo = {}
	pdfsPerXskill = {}

	for i in range(len(data)):

		splittedInfo = data[i].split(",")
		# print(splittedInfo)

		pid = splittedInfo[0]
		name = splittedInfo[1]
		pitchType = splittedInfo[2]
		# [3] = numObs
		xs = float(splittedInfo[4])
		p = float(splittedInfo[5])
		
		if pid not in pitchersInfo:
			pitchersInfo[pid] = {"name":name,"pitchTypes":{}}

		if pitchType not in pitchersInfo[pid]["pitchTypes"]:
			pitchersInfo[pid]["pitchTypes"][pitchType] = {"xs":xs,"p":p,"prat":None,"prats":[]}


	for pid in pitchersInfo:
		for pt in pitchersInfo[pid]["pitchTypes"]:
			if pitchersInfo[pid]["pitchTypes"][pt]["xs"] not in pdfsPerXskill:
					pdfsPerXskill[pitchersInfo[pid]["pitchTypes"][pt]["xs"]] = sys.modules["domain"].getSymmetricNormalDistribution(rng,pitchersInfo[pid]["pitchTypes"][pt]["xs"],delta,spaces.targetsPlateXFeet,spaces.targetsPlateZFeet)
	


	# PITCHERS OF INTEREST
	# Mantiply, Joe: 573009 
	#	CU: 517 | CH: 371 | SL: 0 | FF: 78 | FS: 0 | SI: 725 
	# Scott, Tanner: 656945 | 
	# 	CU: 0 | CH: 0 | SL: 2002 | FF: 1945 | FS: 0 | SI: 52 | FC: 0 | KC: 0 | FA: 0 | CS: 0 | nan: 0 | EP: 0 | FO: 0 | KN: 0 | PO: 0 | SC: 0 | FT: 0 | 
	# info = [["573009", "CU"],["573009", "CH"],["573009", "FF"],["573009", "SI"],
	# 		["656945","SL"],["656945","FF"],["656945","SI"]]


	# JAIR-23 PITCHERS OF INTEREST
	# info = [["594798", "CU"], ["594798", "FF"], ["605400", "FF"], ["670950", "FF"], 
	# 		["592773", "FF"], ["670950", "CH"], ["446372", "FC"], ["663855", "SI"],
	# 		["656945", "FF"], ["518617", "SL"], ["661403", "FC"], ["579328", "FC"],
	# 		["592773", "SL"], ["458681", "CU"], ["656354", "SL"], ["579328", "CH"],
	# 		["573009", "SI"], ["518617", "SI"], ["573009", "CU"], ["455119", "SL"],
	# 		["458681", "CH"], ["455119", "SI"], ["605400", "FC"], ["670950", "SL"],
	# 		["669923", "SL"], ["669923", "CH"], ["656354", "CU"], ["672851", "SL"],
	# 		["656945", "SI"], ["672851", "SI"], ["663855", "FC"], ["656354", "SI"],
	# 		["670950", "CS"], ["594798", "CH"], ["594798", "SI"], ["605400", "KC"],
	# 		["579328", "FF"], ["605400", "CH"], ["458681", "FC"], ["592761", "SL"],
	# 		["605400", "SI"], ["579328", "SL"], ["548389", "SL"], ["548389", "KC"],
	# 		["656354", "FF"], ["643493", "FF"], ["592773", "FS"], ["663855", "SL"],
	# 		["643493", "SL"], ["446372", "CH"], ["656353", "SL"], ["643493", "CU"],
	# 		["656354", "CH"], ["573009", "CH"], ["669923", "SI"], ["643493", "CH"],
	# 		["669923", "FC"], ["592761", "CU"], ["656353", "CU"], ["672851", "CH"],
	# 		["573009", "FF"], ["656353", "SI"], ["656353", "CH"], ["455119", "CU"],
	# 		["643493", "SI"], ["592761", "SI"], ["594798", "SL"], ["458681", "FF"],
	# 		["592761", "FF"], ["548389", "FF"], ["458681", "SI"], ["518617", "FF"],
	# 		["446372", "SI"], ["656945", "SL"], ["446372", "CU"], ["455119", "FF"],
	# 		["548389", "CH"], ["592761", "CH"], ["669923", "FF"], ["672851", "FF"],
	# 		["455119", "FC"], ["661403", "SL"], ["670950", "CU"], ["656353", "FF"],
	# 		["446372", "FF"], ["579328", "CU"], ["455119", "FS"], ["669923", "CU"],
	# 		["672851", "CU"], ["670950", "FC"], ["548389", "SI"], ["458681", "SL"],
	# 		["548389", "FC"], ["663855", "CH"], ["661403", "FF"], ["663855", "FF"], 
	# 		["518617", "CH"], ["458681", "CS"], ["446372", "PO"]]

	# JQAS PITCHERS OF INTEREST

	info = [["445276", "SL", "554"], ["445276", "SI", "969"], 
			["445276", "FC", "3547"],["623433", "CU", "629"], 
			["623433", "FF", "1050"]]


	# For testing
	# info = [["594798", "CU"], ["594798", "FF"], ["656945","SI"]]


	numObs = 50

	needToHave = 80
	p = 0

	for each in info:

		print(f"\n\n({p+1}/{len(info)}) - Looking at: {each}")
		p += 1

		seenObs = pconf(pitchersInfo,each[0],each[1],numObs,needToHave)

		# code.interact("...", local=dict(globals(), **locals()))


	# Save dict containing all info - to be able to rerun it later
	with open(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}pconfTake2.pickle","wb") as outfile:
		pickle.dump(pitchersInfo,outfile)


	# Save dict containing all info - to be able to rerun it later
	with open(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}pconfTake2.txt","w") as outfile:
		
		for each in info:

			pid = each[0]
			pt = each[1]


			if pitchersInfo[pid]["pitchTypes"][pt]['prat'] != -1:
				# name, id, pitch type, x, p, %
				print(f"{pitchersInfo[pid]['name']},{pid},{pt},{pitchersInfo[pid]['pitchTypes'][pt]['xs']},{pitchersInfo[pid]['pitchTypes'][pt]['p']},{pitchersInfo[pid]['pitchTypes'][pt]['prat']},{(pitchersInfo[pid]['pitchTypes'][pt]['prat'])*100:.2f}",file=outfile)
			else:
				print(f"{pitchersInfo[pid]['name']},{pid},{pt},{pitchersInfo[pid]['pitchTypes'][pt]['xs']},{pitchersInfo[pid]['pitchTypes'][pt]['p']},skipped",file=outfile)


