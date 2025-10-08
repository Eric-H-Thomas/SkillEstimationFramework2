import argparse
import os,sys
import code
import json
import numpy as np


if __name__ == "__main__": 

	
	original_stdout = sys.stdout


	############################################################################
	# Find avg time to perform value iteration for the sequential darts domain
	############################################################################


	print("\nFinding times to perform value iter for different xskills...")

	valueIterFiles = os.listdir("Spaces"+os.sep+"ValueFunctions")


	valueIterTimes = []
	xskills = []
	iters = []

	for eachVIF in valueIterFiles:
		# print(eachVIF)

		if ".pickle" not in eachVIF:

			print(f"  {eachVIF}")

			x = eachVIF.split("Xskill")[1].split(".pickle")[0]
			xskills.append(float(x))

			with open("Spaces"+os.sep+"ValueFunctions"+os.sep+eachVIF) as infile:
				results = json.load(infile)
				# print(results.keys())

				valueIterTimes.append(results["totalTime"])

				iters.append(results["iterations"])


	info = np.column_stack((xskills,valueIterTimes,iters))

	sortedInfo = info[info[:,0].argsort()]

	avgTimeValueIter = sum(valueIterTimes)/len(valueIterTimes)


	print("\n"+"-"*40)
	print(f"Time to perform value iter per xskill:")

	for each in range(len(sortedInfo)):
		print(f"   X: {sortedInfo[each][0]} | Time: {round(sortedInfo[each][1],4)} | Iters: {sortedInfo[each][2]}")
	
	print(f"\nAvg time to perform value iter: {round(avgTimeValueIter,4)} seconds")
	print(f"Longest time: {round(max(sortedInfo[:,1]),4)} seconds | For xskill: {sortedInfo[np.argmax(sortedInfo[:,1]),0]}")
	print(f"Shortest time: {round(min(sortedInfo[:,1]),4)} seconds| For xskill: {sortedInfo[np.argmin(sortedInfo[:,1]),0]}")
	print("-"*40)


	outfile = open(f"Spaces{os.sep}valueIterationTimingInfo.txt","w")
	sys.stdout = outfile

	print("\n"+"-"*40)
	print(f"Time to perform value iter per xskill:")

	for each in range(len(sortedInfo)):
		print(f"   X: {sortedInfo[each][0]} | Time: {round(sortedInfo[each][1],4)} | Iters: {sortedInfo[each][2]}")
	
	print(f"\nAvg time to perform value iter: {round(avgTimeValueIter,4)} seconds")
	print(f"Longest time: {round(max(sortedInfo[:,1]),4)} seconds | For xskill: {sortedInfo[np.argmax(sortedInfo[:,1]),0]}")
	print(f"Shortest time: {round(min(sortedInfo[:,1]),4)} seconds| For xskill: {sortedInfo[np.argmin(sortedInfo[:,1]),0]}")
	print("-"*40)

	sys.stdout = original_stdout
	outfile.close()

	############################################################################


