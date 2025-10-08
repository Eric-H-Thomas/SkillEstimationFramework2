import torch
import torch.nn as nn 

import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split("baseball_multi.py")[0]

#for each in ["data","model","utilsBaseball"]:
for each in ["dataTake2","modelTake2","utilsBaseball"]:
	module = SourceFileLoader(each,f"{mainFolderName}{each}.py").load_module()
	sys.modules[each] = module


import pandas as pd
import numpy as np
import argparse,code

from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from time import perf_counter


def getDomainName():
	return "baseball-multi"


def plotBoard(info):

	fig,ax = plt.subplots()
	
	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(min(info["utility"].values), max(info["utility"].values))

	sm = ScalarMappable(norm = norm, cmap = cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm)
	cbar.ax.get_yaxis().labelpad = 15
	cbar.ax.set_ylabel("Utilities",rotation = 270)

	ax.scatter(info["plate_x"].values,info["plate_z"].values,c = cmap(norm(info["utility"].values)))
	ax.set_xlabel("plate_x")
	ax.set_ylabel("plate_z")

	return ax,cmap,norm


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

def getNoiseModel(rng,mean,covMatrix):
	
	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy

	N = multivariate_normal(mean=mean,cov=covMatrix,seed=seed)
	
	return N


def sample_action(rng,mean,covMatrix,a,noiseModel=None):

	# If noise model was not given, proceed to get it
	if noiseModel == None:
		N = getNoiseModel(rng,mean,covMatrix)
	# Otherwise, use given noise model
	else:
		N = noiseModel

	#Get noise (sample)
	noise = N.rvs()

	# Add noise to planned action (This creates the noisy action)
	na = [a[0]+noise[0],a[1]+noise[1]]

	return na

def getNormalDistribution(rng,covMatrix,resolution,X,Y):

	if "XYD" not in globals():
		global XYD

		XD,YD = np.meshgrid(X,Y,indexing="ij")
		tempXYD = np.vstack([XD.ravel(),YD.ravel()])

		XYD = np.dstack(tempXYD)[0]


	mean = [0.0,Y[int(len(Y)/2)]]
	# Results in mean = [0.0, 2.0870000000000077]
	# For plate_z -> Center of array/targets,
	# not quite center of strikezone


	N = getNoiseModel(rng,mean,covMatrix)
	
	D = N.pdf(XYD)

	
	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  
	D *= np.square(resolution)

	# Reshape back to original dimensions
	D = np.array(D).reshape((len(X),len(Y)))

	# code.interact("get...", local=dict(globals(), **locals()))

	return D


def testHits(args):

	numTries = 10000.0

	minXskill = 170.0 #0.50
	maxXskill = 200.0
	xSkills = np.linspace(minXskill,maxXskill,num=100)

	rawData1 = sys.modules["data"].getData(args.startYear1,args.startMonth1,args.startDay1,args.endYear1,args.endMonth1,args.endDay1)
	rawData2 = sys.modules["data"].getData(args.startYear2,args.startMonth2,args.startDay2,args.endYear2,args.endMonth2,args.endDay2)

	train,test,allData,batterIndices = sys.modules["data"].manageDataForModel(rawData1,rawData2)


	minPlateX = np.min(allData["plate_x_inches"].values)
	maxPlateX = np.max(allData["plate_x_inches"].values)

	minPlateZ = np.min(allData["plate_z_inches"].values)
	maxPlateZ = np.max(allData["plate_z_inches"].values)


	# code.interact("...", local=dict(globals(), **locals()))


	# Select target in the middle of the board
	x = (minPlateX+maxPlateX)/2.0
	z = (minPlateZ+maxPlateZ)/2.0
	action = [x,z]

	saveFolder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}PercentHits{os.sep}"

	folders = [f"..{os.sep}..{os.sep}Data{os.sep}",
			   f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}",
			   saveFolder]

	for folder in folders:
		if not Path(folder).is_dir():
			os.mkdir(folder)

	# Prep file for saving results
	outFile = open(f"{saveFolder}PercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.txt", "w")


	print(f"\n--- Performing testHit experiment... ---")
	allPercentHits = []


	for xs in xSkills:

		xs = round(xs,4)
		print(f"\txskill: {xs}")

		N = getNoiseModel(xs**2)

		hits = 0.0

		for tries in range(int(numTries)):

			# Get noise sample
			noise = N.rvs()

			# Add noise to action
			noisyAction = [action[0]+noise[0],action[1]+noise[1]]

			#print(f"\t\t action: {action}")
			#print(f"\t\t noisyAction: {noisyAction}")

			# Verify if the action hits the board or not
			if (noisyAction[0] >= minPlateX and noisyAction[0] <= maxPlateX) and\
				(noisyAction[1] >= minPlateZ and noisyAction[1] <= maxPlateZ):
				hits += 1.0


			####################################
			# PLOT - Strike Zone Board
			####################################
			'''
			fig,ax = plt.subplots()

			# Plot boundaries
			ax.scatter(minPlateX,minPlateZ,c = "black")
			ax.scatter(maxPlateX,maxPlateZ,c = "black")
			ax.scatter(minPlateX,maxPlateZ,c = "black")
			ax.scatter(maxPlateX,minPlateZ,c = "black")
			
			# Plot actual executed action & EV
			ax.scatter(action[0],action[1],c = "red", marker = "*")
			ax.scatter(noisyAction[0],noisyAction[1],c = "blue", marker = "*")

			ax.set_title(f"xskill: {xs}")
			plt.show()
			plt.clf()
			plt.close()
			code.interact("...", local=dict(globals(), **locals()))
			'''
			####################################
			

		percentHit = (hits/numTries)*100.0
		allPercentHits.append(percentHit)
		
		print(f"\t\txSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%")
		# Save to file
		print(f"xSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%",file=outFile)


	outFile.close()

	plt.plot(xSkills,allPercentHits)
	plt.xlabel('xSkills')
	plt.ylabel('% Hits')
	plt.savefig(f"{saveFolder}xskillsVsPercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.png")
	plt.clf()
	plt.close()


def main(args):

	###########################################################################
	# Data
	###########################################################################

	rawData1 = sys.modules["data"].getData(args.startYear1,args.startMonth1,args.startDay1,args.endYear1,args.endMonth1,args.endDay1)
	rawData2 = sys.modules["data"].getData(args.startYear2,args.startMonth2,args.startDay2,args.endYear2,args.endMonth2,args.endDay2)

	train,test,allData,batterIndices = sys.modules["data"].manageDataForModel(rawData1,rawData2)

	###########################################################################
	


	###########################################################################
	# MODEL
	###########################################################################

	# Set hyperparameters
	learningRate = 1.0 #1e-5
	epochs = 1 #40

	model,trainLosses,testLosses,trainAccs,testAccs,modelFolder = \
					sys.modules["model"].getModel(learningRate,epochs,batterIndices,train,test)

	###########################################################################


	###########################################################################
	###########################################################################
	# TEST BEFORE INCORPORATING INTO SKILL ESTIMATION FRAMEWORK
	###########################################################################
	###########################################################################
	


	# Find the unique IDs for the pitchers
	allPitcherIDs = allData["pitcher"].unique()

	# Subset of the pitchers for testing
	# testingPitchersIDs = np.random.choice(allPitcherIDs,2)
	testingPitchersIDs = [656605]


	# Find the different pitch types
	pitchTypes = allData["pitch_type"].unique()


	minPlateX =  np.min(allData["plate_x"].values)
	maxPlateX =  np.max(allData["plate_x"].values)

	minPlateZ =  np.min(allData["plate_z"].values)
	maxPlateZ =  np.max(allData["plate_z"].values)


	# Resolution needs to be smaller.
	# Leaving like this for testing.
	resolution = 0.10

	targetsPlateX = np.arange(minPlateX,maxPlateX,resolution)
	targetsPlateZ = np.arange(minPlateZ,maxPlateZ,resolution)

	
	minXskill = 0.50
	maxXskill = 10.0
	numHyps = 2
	xSkills = np.linspace(minXskill,maxXskill,num=numHyps)

	mainFolder = f"..{os.sep}..{os.sep}Data{os.sep}"


	dataFolder = f"{mainFolder}BaseballStrikeZoneBoards{os.sep}"
	plotFolder = f"{mainFolder}BaseballStrikeZoneBoards{os.sep}Plots{os.sep}"

	for f in [dataFolder,plotFolder]:
		if not Path(f).is_dir():
			os.mkdir(f)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


	# For a given pitcher and pitch type
	for pitcherID in testingPitchersIDs:

		print(f"\n--- Processing for pitcherID: {pitcherID}... ---")
		
		for pitchType in pitchTypes:

			code.interact("()...", local=dict(globals(), **locals()))

			
			# Grab corresponding data
			data = allData.query(f"pitcher == {pitcherID} and pitch_type == '{pitchType}'" )


			batterIndices = pd.DataFrame({'batter': data.batter.unique()})
			# pitcher_indices just once since data is for given pitcher

			batterIndices['batterIndex'] = batterIndices.index.values

			dataForModel = data.merge(batterIndices, on='batter')

			dataX = dataForModel[sys.modules["data"].xswingFeats].values.astype(float)
			dataY = dataForModel.outcome.values

			code.interact("inside...", local=dict(globals(), **locals()))
			


			###############################################################
			# SAVE PROCESSED DATA??
			# To load & avoid need to recompute each time??
			###############################################################


			'''
			Columns: [game_date, player_name, pitcher, batter, pitch_type, pitch_name,
					 balls, strikes, release_speed, release_spin_rate, 
					 release_extension, release_pos_x, release_pos_z, plate_x, 
					 plate_z, type, events, description, woba_value, mx, mz,
					 pit_handR, bat_handR, outcome, is_swing, is_miss, batterIndex,
					 o0, o1, o2, o3, o4, o5, o6, o7, o8, swing_utility, take_utility]
			'''


			# For each row in the data
			# That is, for a given pitch (observation/state)
			for index, row in dataForModel.iterrows():


				###############################################################
				# Find Outcome Probs & Utilities
				# GOAL: Create "dartboard"
				###############################################################
				
				# Including 'row' (df with actual pitch info) to get the probabilities 
				# for the different outcomes as well as the utility - for the actual pitch
				allTempData = [row.copy()]

				for px in targetsPlateX:
					for pz in targetsPlateZ:
						
						# Copy pitch info
						tempData = row.copy()	

						# Update position
						tempData["plate_x"] = px
						tempData["plate_z"] = pz

						allTempData.append(tempData)

				# Convert to dataframe
				allTempData = pd.DataFrame(allTempData)

				
				# Run model
				results = nn.functional.softmax(model(torch.tensor(allTempData[sys.modules["data"].xswingFeats].values.astype(float), dtype = torch.float32).to(device)), dim = 1).cpu().detach().numpy()

				# Save info
				for i in range(9):
					allTempData[f'o{i}'] = results[:,i]


				# Get utilities
				withUtilities = sys.modules["utilsBaseball"].getUtility(allTempData)
				# withUtilities[['player_name', 'pitch_type', 'balls', 'strikes', 'take_utility', 'swing_utility']].head()

				###############################################################


				# Get updated info for actual pitch (actual pitch + probs + utility)
				row = withUtilities.iloc[0].copy()

				# Remove actual pitch from data
				withUtilities =  withUtilities.iloc[1:,:]

				fileName = f"pitcherID{pitcherID}-pitchType{pitchType}-index{index}"


				####################################
				# PLOT - Strike Zone Board
				####################################
				ax,cmap,norm = plotBoard(withUtilities)
				ax.set_title(f"pitcherID: {pitcherID} | pitchType: {pitchType} | index: {index}")

				# Plot actual executed action & EV
				ax.scatter(row["plate_x"],row["plate_z"],color = cmap(norm(row["utility"])), marker = "*")
				plt.savefig(f"{plotFolder}{fileName}.jpg")
				plt.close()
				plt.clf()
				####################################
			


				###############################################################
				# Save strike zone "dartboards" if not present already
				###############################################################

				if not Path(f"{dataFolder}{fileName}").is_file():
					withUtilities[["plate_x","plate_z","utility"]].to_pickle(f"{dataFolder}{fileName}.pkl")  

				###############################################################



				###############################################################
				# Find Pdfs & EVs
				###############################################################

				pdfsPerXskill = {}
				evsPerXskill = {}

				for xs in xSkills:

					Xn,Yn,Zn = getSymmetricNormalDistribution(xs,resolution,targetsPlateX,targetsPlateZ)
					pdfsPerXskill[xs] = Zn


					Zs = np.zeros((len(targetsPlateX), len(targetsPlateZ)))

					for i in range(len(targetsPlateX)):
						for j in range(len(targetsPlateZ)):
							index = withUtilities.index[(withUtilities["plate_x"] == targetsPlateX[i]) 
									& (withUtilities["plate_z"] == targetsPlateZ[j])].tolist()[0]

							Zs[i][j] = withUtilities.iloc[index]["utility"]

					# Convolve to produce the EV and aiming spot
					EVs = fftconvolve(Zs,Zn,mode="same")

					evsPerXskill[xs] = EVs					
				
				code.interact("end row...", local=dict(globals(), **locals()))

				###############################################################



				###############################################################
				# Skill Estimation Experiment
				###############################################################

				executedAction = [row["plate_x"],row["plate_z"]]

				observedReward = row["utility"]


				# Feed data to estimators



	###########################################################################

	code.interact("end...", local=dict(globals(), **locals()))


if __name__ == '__main__':
	
	# Get arguments from command line
	parser = argparse.ArgumentParser(description="Obtain statcast data for given date range")
	
	parser.add_argument("-startYear1", dest = "startYear1", help = "Desired start year for 1st set of data.", type = str, default = "2021")
	parser.add_argument("-endYear1", dest = "endYear1", help = "Desired end year for 1st set of data.", type = str, default = "2021")
	
	parser.add_argument("-startMonth1", dest = "startMonth1", help = "Desired start month for 1st set of data.", type = str, default = "01")
	parser.add_argument("-endMonth1", dest = "endMonth1", help = "Desired end month for 1st set of data.", type = str, default = "12")
	
	parser.add_argument("-startDay1", dest = "startDay1", help = "Desired start day for 1st set of data.", type = str, default = "01")
	parser.add_argument("-endDay1", dest = "endDay1", help = "Desired end day for 1st set of data.", type = str, default = "31")
	

	parser.add_argument("-startYear2", dest = "startYear2", help = "Desired start year for 2nd set of data.", type = str, default = "2022")
	parser.add_argument("-endYear2", dest = "endYear2", help = "Desired end year for 2nd set of data.", type = str, default = "2022")
	
	parser.add_argument("-startMonth2", dest = "startMonth2", help = "Desired start month for 2nd set of data.", type = str, default = "01")
	parser.add_argument("-endMonth2", dest = "endMonth2", help = "Desired end month for 2nd set of data.", type = str, default = "12")
	
	parser.add_argument("-startDay2", dest = "startDay2", help = "Desired start day for 2nd set of data.", type = str, default = "01")
	parser.add_argument("-endDay2", dest = "endDay2", help = "Desired end day for 2nd set of data.", type = str, default = "31")

	args = parser.parse_args()


	# main(args)


	testHits(args)
