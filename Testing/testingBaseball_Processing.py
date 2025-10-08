import os,json,code,pickle,sys
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

def getBucket(bucketsX,minMaxX,xParam):

	# Find proper bucket for current x
	for b in range(len(bucketsX)):
		if xParam <= bucketsX[b]:
			break

	# Get actual bucket
	bucket1 = bucketsX[b]


	otherBucket = None
	bucket2 = None

	# First bucket
	if b == 0:
		# use left edge/extreme - i.e. 0
		otherBucket = minMaxX[0]
	# If last bucket
	elif b == len(bucketsX)-1:
		# use right edge/extreme - i.e. 5/100 depending on the domain
		otherBucket = minMaxX[1]
	# Somewhere in the middle - consider next bucket
	else:
		bucket2 = bucket1
		bucket1 = bucketsX[b-1]

	return bucket1, bucket2


if __name__ == '__main__':


	if len(sys.argv) != 3:
		print("Please provide pitcher ID and pitch type.\nAlso make sure results file (from experiments) exist already. \nUsage: testingBaseball.py pitcherID pitchType")
		exit()


	# Testing pitcher IDs: 642232, 621237, 621107, 622250
	pitcherID =  sys.argv[1]

	# Testing pitch types: FF, CU
	pitchType = sys.argv[2]

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
	print(f"\n{agentFolder}\n")

	resultsFolder = f"BaseballTesting{os.sep}Results{os.sep}{agentFolder}{os.sep}"
	plotsFolder = f"BaseballTesting{os.sep}Plots{os.sep}{agentFolder}{os.sep}"

	folders = [f"BaseballTesting{os.sep}Plots{os.sep}",plotsFolder, 
	f"{plotsFolder}Estimates{os.sep}",f"{plotsFolder}PercentRationality{os.sep}"]

	for f in folders:
		if not Path(f).is_dir():
			os.mkdir(f)

	lambdas = np.logspace(-3,3.4,100)

	
	resultFiles = os.listdir(resultsFolder)

	resultFiles.remove("pconf.pkl")


	#############################################################
	# For percent rationality conversion
	#############################################################
	
	with open(f"{resultsFolder}pconf.pkl", "rb") as handle:
		pconfPerXskill = pickle.load(handle)

	bucketsX = sorted(pconfPerXskill.keys())
	minMaxX = [bucketsX[0],bucketsX[-1]]
	mm = None

	#############################################################


	# RF for a given simulated agent
	for rf in resultFiles:	

		with open(f"BaseballTesting{os.sep}Results{os.sep}{agentFolder}{os.sep}{rf}") as infile:			
			results = json.load(infile)

			xskill = results["xskill"]
			pskill = results["pskill"]


			#################################################
			# PLOT - Observations vs Estimates
			#################################################

			types = ["xSkill","pSkill"]

			for t in types:

				fig = plt.figure(figsize=(10,6))
				ax = plt.subplot(111)

				for m in results:

					if m in ["agentFolder","xskill","pskill"]:
						continue

					if (not ("OR" in m and t == "xSkill")) and (t not in m or "allProbs" in m):
						continue

					# To use later - for conversion % rationality & plot
					if "JT-QRE" in m and "xSkill" in m:
						mm = m

					estimates = results[m]
 
					plt.plot(range(len(estimates)),estimates,label=m)
					
				if "x" in t:
					plt.plot(range(len(estimates)),[xskill]*len(estimates),linestyle="dashed",label="True X",c="black")
				else: #pskill
					plt.plot(range(len(estimates)),[pskill]*len(estimates),linestyle="dashed",label="True P",c="black")


				plt.xlabel("Number of Observations")
				plt.ylabel(f"{t} Estimates")
				plt.legend()
				ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
				plt.title(f"Agent - X: {xskill} | P: {pskill}")
				plt.tight_layout()
				plt.savefig(f"{plotsFolder}Estimates{os.sep}ObservationsVsEstimates-{t}-Agent-X{xskill}-P{pskill}.png")
				plt.clf()
				plt.close()

			#################################################


			#################################################
			# PLOT - Observations vs Percent Rationality
			#################################################

			fig = plt.figure(figsize=(10,6))
			ax = plt.subplot(111)

			xskillEstimates = results[mm]

			for m in results:

				if m in ["agentFolder","xskill","pskill"]:
					continue

				if "pSkill" not in m or "allProbs" in m:
					continue

				estimates = results[m]

				percentEstimates = []

				# Convert to percent rat
				for i in range(len(estimates)):

					# Get pskill estimate of current method - estimatedP
					estimatedP = estimates[i]

					# Use estimated xskill and not actual true one
					# WHY? estimatedX and not trueX?? because "right" answer is not available
					estimatedX = xskillEstimates[i]

					# find proper bucket for current x
					bucket1, bucket2 = getBucket(bucketsX,minMaxX,estimatedX)


					# Convert estimatedP to corresponding % of rand max
					if bucket2 != None:

						prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])
						prat2 = np.interp(estimatedP,pconfPerXskill[bucket2]["lambdas"], pconfPerXskill[bucket2]["prat"])

						prat = np.interp(estimatedP, [prat1], [prat2])
						percent_estimatedP = prat
					# edges/extremes case
					else:
						# using one of the functions for now
						prat1 = np.interp(estimatedP,pconfPerXskill[bucket1]["lambdas"], pconfPerXskill[bucket1]["prat"])

						percent_estimatedP = prat1


					percentEstimates.append(percent_estimatedP)


				plt.plot(range(len(percentEstimates)),percentEstimates,label=m)	
				

			pi = np.where(lambdas == pskill)[0][0]
			truePercent = pconfPerXskill[xskill]["prat"][pi]
			plt.plot(range(len(percentEstimates)),[truePercent]*len(percentEstimates),linestyle="dashed",label="True Percent",c="black")

			code.interact("...", local=dict(globals(), **locals()))
			plt.xlabel("Number of Observations")
			plt.ylabel(f"Percent Rationality")
			plt.legend()
			ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
			plt.title(f"Agent - X: {xskill} | P: {pskill}")
			plt.tight_layout()
			plt.savefig(f"{plotsFolder}PercentRationality{os.sep}ObservationsVsPercentRationality-{t}-Agent-X{xskill}-P{pskill}.png")
			plt.clf()
			plt.close()

			#################################################

