import code
import pickle
import argparse
import os,sys
import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader
from pybaseball import playerid_reverse_lookup, cache


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}getInfoPitches.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


def findBucketsStats(agentData):

	gameIDs, counts = np.unique(agentData["game_pk"],return_counts=True)



def testGroupByPitchNum(agentData,b1,b2):

	# Design so that it returns the pitches for a given bucket #

	
	# Group by date first
	gameIDs, counts = np.unique(agentData["game_pk"],return_counts=True)

	print(f"\tNumber of Games: {len(counts)}")


	# Need to handle the case where multiple games in a day?? Possible?

	data = pd.DataFrame()

	for gid in gameIDs:

		pitches = agentData[agentData["game_pk"] == gid]

		# Slice pitches by pitchNums of interest (given bucket)
		subPitches = pitches[int(b1):int(b2)]

		# Save just those of interest (given bucket)
		data = pd.concat([data,subPitches])


	# code.interact("...", local=dict(globals(), **locals()))

	return data


if __name__ == '__main__':
	

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use", type = str, default = "baseball")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	parser.add_argument("-file", dest = "file", help = "Name of results file containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# # Prevent error with "/"
	# if args.resultsFolder[-1] == os.sep:
	# 	args.resultsFolder = args.resultsFolder[:-1]
	
	# result_files = os.listdir(args.resultsFolder + os.sep + "results")


	# if len(result_files) == 0:
	# 	print("No result files present for experiment.")
	# 	exit()



	# # NOTE: EACH RESULT FILE BELONGS TO A GIVEN PITCHER AND PITCH TYPE
	# # EACH COMBINATION WILL BE SEEN ONLY ONCE (MEANING JUST 1 EXP)

	# for rf in result_files: 

	# 	with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:
	# 		results = pickle.load(infile)

	# 		agent = results["agent_name"]
	# 		pitcherID = agent[0]
	# 		pitchType = agent[1]
	# 		numObservations = results["numObservations"]


	# 		code.interact("...", local=dict(globals(), **locals()))
	

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	maxRows = 1000


	# JQAS Pitchers of Interest
	info = [["445276", "SL", "554"]]#, ["445276", "SI", "969"], 
			# ["445276", "FC", "3547"],["623433", "CU", "629"], 
			# ["623433", "FF", "1050"]]


	'''
	with open(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}infoPitches.txt","w") as outfile:

		print(f"Max Pitches Used For Exps: {maxRows}",file=outfile)
		print(file=outfile)

		for pitcherID,pitchType,numObs in info:

			print(f"Pitcher: {pitcherID} | Pitch Type: {pitchType} | Total Pitches Available: {numObs}",file=outfile)

			spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

			agentData = spaces.getAgentData("recent",pitcherID,pitchType,[maxRows])

			dist = np.unique(agentData["game_year"],return_counts=True)
			dist = np.dstack(np.vstack(dist))[0]

			# To sort in descending order (by year)
			dist = dist[(-dist[:,0]).argsort()]

			for year, count in dist:
				print(f"\t{year}: {count}",file=outfile)

			print(file=outfile)
	'''



	# Max number of pitches per game = 100 - 110
	# "Beginning 4/8/2024 the maximum number of pitches a player may throw in a game is 85."

	minNum = 1 #20
	maxNum = 2 #100

	numBuckets = 1 #5

	temp = np.linspace(minNum,maxNum+1,numBuckets)

	buckets = []

	# Create bucket pairs
	prev = 0
	for i in range(len(temp)):
		buckets.append([prev,temp[i]])
		prev = temp[i] + 1


	# For Testing
	buckets = [[0,1],[1,3],[6,10]]

	print("Buckets: ", buckets)



	for pitcherID,pitchType,numObs in info:
		

		spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

		agentData = spaces.getAgentData("recent",pitcherID,pitchType,[maxRows])

		
		print(f"\nPitcher: {pitcherID} | Pitch Type: {pitchType} | Num Pitches: {numObs}")

		for b1,b2 in buckets:
			data = testGroupByPitchNum(agentData,b1,b2)

			print(f"\tB1: {b1} | B2: {b2} | Num Pitches: {len(data)}")

			# code.interact("...", local=dict(globals(), **locals()))

















