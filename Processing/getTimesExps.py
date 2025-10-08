import argparse
import os,sys
import code
import pickle
import numpy as np


if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Find experiments/value iteration times')
	parser.add_argument('-resultsFolder','-resultsFolder', help='Name of the folder that contains the results file for a given experiment', default = "testing")	
	parser.add_argument('-max','-max', help='Max number of exps to get info from', type = int, default = -1)
	args = parser.parse_args()


	# To preven error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]


	original_stdout = sys.stdout

	##################################################################
	# Find avg time of experiments for the sequential darts domain
	##################################################################

	print("\nFinding times from result files...")

	resultFiles = os.listdir(args.resultsFolder + os.sep + "results")


	if args.max == -1:
		args.max = len(resultFiles)


	expTimes = []

	for i in range(len(resultFiles)):

		if i+1 > args.max:
			print("\nStopping since reached specified max.")
			break

		eachRF = resultFiles[i]

		with open(args.resultsFolder+os.sep+"results"+os.sep+eachRF,"rb") as infile:
			results = pickle.load(infile)
			# print(results.keys())

			print(f"({i+1}/{len(resultFiles)}) {eachRF}")

			try:
				expTimes.append(results["expTotalTime"])
			except: 
				# To skip results file for exps that are still running
				continue


	avgTime = sum(expTimes)/len(expTimes)
	
	print("\n"+"-"*40)
	print(f"Experiments performed: {len(resultFiles)}")
	print(f"Experiments processed (got time info): {len(expTimes)}")
	print(f"Avg time per experiment: {round(avgTime,4)} seconds")
	print(f"Longest experiment: {round(max(expTimes),4)} seconds")
	print(f"Shortest experiment: {round(min(expTimes),4)} seconds")
	print("-"*40)


	if not os.path.exists(f"{args.resultsFolder}{os.sep}times{os.sep}"):
		os.mkdir(f"{args.resultsFolder}{os.sep}times{os.sep}")


	if not os.path.exists(f"{args.resultsFolder}{os.sep}times{os.sep}experiments{os.sep}"):
		os.mkdir(f"{args.resultsFolder}{os.sep}times{os.sep}experiments{os.sep}")


	outfile = open(f"{args.resultsFolder}{os.sep}times{os.sep}experiments{os.sep}info-expTimes.txt","w")
	sys.stdout = outfile
	print("\n"+"-"*40)
	print(f"Experiments performed: {len(resultFiles)}")
	print(f"Experiments processed (got time info): {len(expTimes)}")
	print(f"Avg time per experiment: {round(avgTime,4)} seconds")
	print(f"Longest experiment: {round(max(expTimes),4)} seconds")
	print(f"Shortest experiment: {round(min(expTimes),4)} seconds")
	print("-"*40)
	sys.stdout = original_stdout
	outfile.close()

	##################################################################


