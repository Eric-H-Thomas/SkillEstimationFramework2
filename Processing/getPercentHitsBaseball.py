import argparse
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import code
import os, sys
from importlib.machinery import SourceFileLoader


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}getPercentHitsBaseball.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


def getCovMatrix(stdDevs,rho):

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	return covMatrix


if __name__ == '__main__':


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()
	

	# StrikeZone Info - in feets
	minPlateX = -0.71
	maxPlateX = 0.71
	minPlateZ = 1.546
	maxPlateZ = 3.412


	# Select target in the middle of the board
	x = (minPlateX+maxPlateX)/2.0
	z = (minPlateZ+maxPlateZ)/2.0
	middleAction = [x,z]
	
	mean = [0.0,(minPlateZ+maxPlateZ)/2]

	
	# 0.5 inches | 0.0417 feet
	# delta = 0.0417
	# 1.0 inches | 0.0833333 feet
	delta = 0.0833333



	numTriesOriginal = 500
	

	# pitchers = [594798,455119,656945,446372,518617,656354,672851]
	pitchers = [455119,656945]
	pitchTypes = ["FF"]


	for pitcherID in pitchers:

		for pitchType in pitchTypes:


			spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)
			agentData = spaces.getAgentData(pitcherID,pitchType,maxRows=numTriesOriginal,numObservations=numTriesOriginal,chunkNum=1)


			print(f"Pitcher: {pitcherID} | Pitch Type: {pitchType}")


			hits1 = 0.0

			for row in agentData.itertuples():

				index = row.Index

				action = [row.plate_x_feet,row.plate_z_feet]

				# Verify if the action hits the board or not
				if (action[0] >= minPlateX and action[0] <= maxPlateX) and\
					(action[1] >= minPlateZ and action[1] <= maxPlateZ):
					hits1 += 1.0

				# code.interact("...", local=dict(globals(), **locals()))


			numObservations = numTries = len(agentData)

			percentHit = (hits1/numTries)*100.0

			print(f"\tFrom Data - Total Hits: {hits1} out of {numTries} -> {percentHit:.4f}%")
			


			info = pd.read_csv(f"{args.resultsFolder}{os.sep}plots{os.sep}avgEstimate-PerDimension-AcrossChunks{os.sep}pitchersInfo-avgEstimate-PerDimension-AcrossChunks.csv")

			pitcherInfo = info.loc[(info["PitcherID"]==pitcherID) & (info["PitchType"]==pitchType)]

			methods = pitcherInfo.columns[2:]

			print()

			for mi in range(0,len(methods),2):

				xm = methods[mi]
				rm = methods[mi+1]

				# To remove notation (|)
				temp = pitcherInfo[xm].values[0].split(" ")
				estX = [round(float(temp[0]),4),round(float(temp[2]),4)]

				estR = round(float(pitcherInfo[rm].values[0]),4)

				estimatedCovMatrix = getCovMatrix(estX,estR)


				# Draw sample from distribution
				N = multivariate_normal(mean=mean,cov=estimatedCovMatrix)


				hits2 = 0.0

				for tries in range(int(numTries)):

					# Get noise sample
					noise = N.rvs()

					# Add noise to action
					noisyAction = [middleAction[0]+noise[0],middleAction[1]+noise[1]]

					# Verify if the action hits the board or not
					if (noisyAction[0] >= minPlateX and noisyAction[0] <= maxPlateX) and\
						(noisyAction[1] >= minPlateZ and noisyAction[1] <= maxPlateZ):
						hits2 += 1.0


				percentHit = (hits2/numTries)*100.0
				
				print(f"\tMethod: {xm.split('-xSkills')[0]}")
				print(f"\tEstimatedX: {estX} | EstimatedR: {estR}")
				print(f"\tTotal Hits: {hits2} out of {numTries} -> {percentHit:.4f}%")
				print()

			code.interact("...", local=dict(globals(), **locals()))






















