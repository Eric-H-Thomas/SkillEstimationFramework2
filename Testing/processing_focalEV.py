import numpy as np
import scipy
import scipy.stats as stats
from matplotlib import pyplot as plt
import itertools
import sys
import time 
import json
import os
import code
import argparse


if __name__ == "__main__":

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "", type = str, default = "Results")
	args = parser.parse_args()

	allResults = {}

	resultFiles = os.listdir(args.resultsFolder + os.path.sep + "Results")

	for rf in resultFiles:
		with open(args.resultsFolder+os.path.sep+"Results"+os.path.sep+rf) as infile:
			results = json.load(infile)

			allResults[int(rf.split("Samples")[1].split("-")[0])] = results



	for numSamples in allResults.keys():

		results = allResults[numSamples]

		numExps = results["numExps"]
		xskills = results["xskills"]
		numXskills = len(results["xskills"])
		
		sEV_mse = [0.0]*numXskills
		nEV_mse = [0.0]*numXskills


		for i in range(numExps):

			cEV,sEV,nEV = results[str(i)]["cEV"],results[str(i)]["sEV"],results[str(i)]["nEV"]

			for si in range(numXskills):
				sEV_mse[si] += (cEV[si]-sEV[si])**2.0
				nEV_mse[si] += (cEV[si]-nEV[si])**2.0


		# Normalize
		for si in range(numXskills):
			sEV_mse[si] /= numExps
			nEV_mse[si] /= numExps


		allResults[numSamples]["sEV_mse"] = sEV_mse 
		allResults[numSamples]["nEV_mse"] = nEV_mse 

		# Plot - For a given num samples
		plt.plot(xskills,sEV_mse,label='MSE Sample')
		plt.plot(xskills,nEV_mse,label='MSE New')

		plt.xlabel("Xskills")
		plt.ylabel("MSE")
			
		plt.legend()
		plt.savefig(args.resultsFolder+os.path.sep+"Plots"+os.path.sep+f"MSE-Xskills{numXskills}-Samples{numSamples}-NumExps{numExps}.png", bbox_inches='tight')
		plt.clf()
		plt.close()

		# code.interact("...", local=dict(globals(), **locals()))


	# Plot - all samples same plot

	sortedSamples = sorted(list(allResults.keys()))

	for xi in range(len(xskills)):

		x = round(xskills[xi],4)
		
		tempInfo1 = []
		tempInfo2 = []

		for numSamples in sortedSamples:
			tempInfo1.append(allResults[numSamples]["sEV_mse"][xi])
			tempInfo2.append(allResults[numSamples]["nEV_mse"][xi])
		
		plt.scatter(sortedSamples,tempInfo1,label='MSE Sample')
		plt.scatter(sortedSamples,tempInfo2,label='MSE New')
		plt.plot(sortedSamples,tempInfo1)
		plt.plot(sortedSamples,tempInfo2)

		plt.xlabel("Number of samples")
		plt.ylabel("MSE")

		plt.legend()
		plt.savefig(args.resultsFolder+os.path.sep+"Plots"+os.path.sep+f"MSE-Xskills{numXskills}-AllSamples-NumExps{numExps}-Xskill{x}.png", bbox_inches='tight')
		plt.clf()
		plt.close()

