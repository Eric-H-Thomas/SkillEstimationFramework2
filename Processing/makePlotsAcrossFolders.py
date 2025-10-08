import code
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from time import time

from utilsDartsMulti import *


def plotAllVsMetric(group,methodName,estimateType,numObs):

	saveAt = f"{plotFolder}AllVsMetric{os.sep}"
	makeFolder3(saveAt)

	saveAt += f"{group}{os.sep}"
	makeFolder3(saveAt)


	# PLOT: all methods vs metric per agent type and method for given number of particles
	for at in agentTypes:
		for metric in metrics:

			tempSaveAt = f"{saveAt}{metric}{os.sep}"
			makeFolder3(tempSaveAt)

			tempLabel = ""

			for ii in range(2):

				if ii == 1:
					tempLabel = "-WithJEEDS"


				fig1 = plt.figure(figsize =(8,10))

				ax1 = plt.gca()

				allMetrics = []
				allLabels = []

				for numP in numParticles:

					for noise in noises:

						for resample in resamples:
	 
							key = f"NumObservations{numObs}-numParticles{numP}"

							'''
							methodName2 = f"Resample{resample}%-NoiseDiv{noise}-JT-{estimateType}-xSkills"

							method = f"{methodName}-{numP}-{methodName2}"

							# AVG metric after all observations
							info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]
							allMetrics.append(info)
							
							ax1.scatter(1.0,info,label=f"P{numP}-R{resample}-N{noise}")

							ax1.annotate(f"P{numP}-R{resample}-N{noise}",(1.0,info))

							allLabels.append(f"P{numP}-R{resample}-N{noise}")
							'''

							if resampleNEFF:

								tempMethodName = f"Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-{estimateType}-xSkills"
								method = f"{methodName}-{numP}-{tempMethodName}"

								# AVG metric after all observations
								info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]
								allMetrics.append(info)

								ax1.scatter(1.0,info,marker="*",label=f"P{numP}-R{resample}-N{noise}")

								ax1.annotate(f"P{numP}-R{resample}-N{noise}",(1.0,info))

								allLabels.append(f"P{numP}-R{resample}-N{noise}-NEFF")



				if ii == 1:
					# Only if JEEDS present on exps
					try:
						for mm in ["JT-QRE-MAP-33-33-xSkills","JT-QRE-EES-33-33-xSkills"]:
							info = infoMetricsAVG[key][group][metric]["info"][at][mm]["avgMetric"][-1]
							ax1.scatter(1.0,info,label=mm)
							ax1.annotate(mm,(1.0,info))
							allMetrics.append(info)
							allLabels.append(mm)
					except:
						continue


				# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

				ax1.set_title(f"Agent: {at} | Number of Observations: {numObs}")

				ax1.set_xlabel('All')
				ax1.set_ylabel(f"{metric}")

				fig1.tight_layout()

				# code.interact("...", local=dict(globals(), **locals()))

				fig1.savefig(f"{tempSaveAt}Agent-{at}-NumObs{numObs}-Type-{estimateType}{tempLabel}.png",bbox_inches='tight')
				plt.clf()
				plt.close("all")

				sortedInfo = sorted(allMetrics)

				with open(f"{tempSaveAt}Agent-{at}-NumObs{numObs}-Type-{estimateType}{tempLabel}.txt","w") as outfile:
					for each in sortedInfo:
						print(f"{allLabels[allMetrics.index(each)]}: {each}",file=outfile)


def plotResampleVsMetric(methodName,group):

	saveAt = f"{plotFolder}ResampleVsMetric{os.sep}"
	makeFolder3(saveAt)

	saveAt += f"{group}{os.sep}"
	makeFolder3(saveAt)


	# PLOT: resample % vs metric per agent type and method for given number of particles
	for at in agentTypes:
		for metric in metrics:

			tempSaveAt2 = f"{saveAt}{metric}{os.sep}"
			makeFolder3(tempSaveAt2)

			for noise in noises:

				tempSaveAt = f"{tempSaveAt2}Noise{noise}{os.sep}"
				makeFolder3(tempSaveAt)


				for estimateType in estimateTypes:

					fig = plt.figure()
					ax = plt.gca()

					for numObs in numObservations:
						for numP in numParticles:
							key = f"NumObservations{numObs}-numParticles{numP}"

							'''
							if resampleNEFF:
								iters = 2
							else:
							'''
							iters = 1

							for eachIter in range(iters):

								xs = []
								ys = []

								for resample in resamples:

									methodName2_Temp = f"Resample{resample}%-NoiseDiv{noise}-JT-{estimateType}-xSkills"
									methodName2_NEFF = f"Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-{estimateType}-xSkills"

									#if eachIter == 0:
									#	methodName2 = methodName2_Temp
									#else:
									methodName2 = methodName2_NEFF

									method = f"{methodName}-{numP}-{methodName2}"

									# AVG metric after all observations
									info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]
									
									xs.append(resample)
									ys.append(info)

								
								aStr = f"NumObservations{numObs}-numParticles{numP}"
								if eachIter == 1:
									aStr += "-NEFF"

								plt.plot(xs,ys,label=aStr)
								plt.scatter(xs,ys)

					plt.legend()

					plt.title(f"Agent: {at} | | Noise: {noise}")

					plt.xlabel('Resample Percentage')
					plt.ylabel(f"{metric}")

					# code.interact("...", local=dict(globals(), **locals()))

					plt.savefig(f"{tempSaveAt}Agent-{at}-Type-{estimateType}.png",bbox_inches='tight')
					plt.clf()
					plt.close("all")


def plotNoiseVsMetric(methodName,group):

	saveAt = f"{plotFolder}NoiseVsMetric{os.sep}"
	makeFolder3(saveAt)

	saveAt += f"{group}{os.sep}"
	makeFolder3(saveAt)


	# PLOT: noise vs metric per agent type and method for given number of particles
	for at in agentTypes:
		for metric in metrics:

			tempSaveAt2 = f"{saveAt}{metric}{os.sep}"
			makeFolder3(tempSaveAt2)

			for resample in resamples:

				tempSaveAt = f"{tempSaveAt2}ResamplePercent{resample}{os.sep}"
				makeFolder3(tempSaveAt)

				for estimateType in estimateTypes:

					fig = plt.figure()
					ax = plt.gca()

					for numObs in numObservations:
						for numP in numParticles:
							key = f"NumObservations{numObs}-numParticles{numP}"

							#if resampleNEFF:
							#	iters = 2
							#else:
							iters = 1

							for eachIter in range(iters):

								xs = []
								ys = []

								for noise in noises:
										
									methodName2_Temp = f"Resample{resample}%-NoiseDiv{noise}-JT-{estimateType}-xSkills"
									methodName2_NEFF = f"Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-{estimateType}-xSkills"

									#if eachIter == 0:
									#	methodName2 = methodName2_Temp
									#else:
									methodName2 = methodName2_NEFF

									method = f"{methodName}-{numP}-{methodName2}"

									# AVG metric after all observations
									info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]
									
									xs.append(noise)
									ys.append(info)

								aStr = f"NumObservations{numObs}-numParticles{numP}"
								if eachIter == 1:
									aStr += "-NEFF"

								plt.plot(xs,ys,label=aStr)
								plt.scatter(xs,ys)

					plt.legend()

					plt.title(f"Agent: {at} | Resample Percent: {resample}")

					plt.xlabel('Noise')
					plt.ylabel(f"{metric}")

					# code.interact("...", local=dict(globals(), **locals()))

					plt.savefig(f"{tempSaveAt}Agent-{at}-Type-{estimateType}.png",bbox_inches='tight')
					plt.clf()
					plt.close("all")


def plotNumObservationsVsMetric(methodName,group):

	saveAt = f"{plotFolder}NumObservationsVsMetric{os.sep}"
	makeFolder3(saveAt)

	saveAt += f"{group}{os.sep}"
	makeFolder3(saveAt)


	# PLOT: # of particles vs metric per agent type and method for given number of particles
	for at in agentTypes:
		for metric in metrics:

			tempSaveAt = f"{saveAt}{metric}{os.sep}"
			makeFolder3(tempSaveAt)

			for numP in numParticles:

				for estimateType in estimateTypes:

					fig = plt.figure()
					ax = plt.gca()

					for resample in resamples:
						for noise in noises:

							methodName2_Temp = f"Resample{resample}%-NoiseDiv{noise}-JT-{estimateType}-xSkills"
							methodName2_NEFF = f"Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-{estimateType}-xSkills"

							'''
							if resampleNEFF:
								iters = 2
							else:
								iters = 1
							'''
							iters = 1


							for eachIter in range(iters):

								'''
								if eachIter == 0:
									methodName2 = methodName2_Temp
								else:
								'''
								methodName2 = methodName2_NEFF

								xs = []
								ys = []

								for numObs in numObservations:

									key = f"NumObservations{numObs}-numParticles{numP}"

									method = f"{methodName}-{numP}-{methodName2}"

									# AVG metric after all observations
									info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]

									xs.append(numObs)
									ys.append(info)

								plt.plot(xs,ys,label=methodName2)
								plt.scatter(xs,ys)
								

					plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')

					plt.title(f"Agent: {at} | Number of Particles: {numP}")

					plt.xlabel('Number of Observations')
					plt.ylabel(f"{metric}")

				# code.interact("...", local=dict(globals(), **locals()))

				plt.savefig(f"{tempSaveAt}Agent-{at}-NumParticles{numP}-JT-{estimateType}.png",bbox_inches='tight')
				plt.clf()
				plt.close("all")


def plotNumParticlesVsMetric(methodName,group):

	saveAt = f"{plotFolder}NumParticlesVsMetric{os.sep}"
	makeFolder3(saveAt)

	saveAt += f"{group}{os.sep}"
	makeFolder3(saveAt)


	# PLOT: # of particles vs metric - per agent type - all methods same plot - for given number of observations
	for at in agentTypes:
		for metric in metrics:

			tempSaveAt = f"{saveAt}{metric}{os.sep}"
			makeFolder3(tempSaveAt)


			for numObs in numObservations:

				for estimateType in estimateTypes:

					fig = plt.figure()
					ax = plt.gca()


					for resample in resamples:
						for noise in noises:

							#methodName2_Temp = f"Resample{resample}%-NoiseDiv{noise}-JT-{estimateType}-xSkills"
							methodName2_NEFF = f"Resample{resample}%-ResampleNEFF-NoiseDiv{noise}-JT-{estimateType}-xSkills"

							'''
							if resampleNEFF:
								iters = 2
							else:
								iters = 1
							'''
							iters = 1

							for eachIter in range(iters):

								#if eachIter == 0:
								#	methodName2 = methodName2_Temp
								#else:
								methodName2 = methodName2_NEFF

								xs = []
								ys = []

								for numP in numParticles:
									key = f"NumObservations{numObs}-numParticles{numP}"

									method = f"{methodName}-{numP}-{methodName2}"

									# AVG metric after all observations
									info = infoMetricsAVG[key][group][metric]["info"][at][method]["avgMetric"][-1]
									xs.append(numP)
									ys.append(info)

								sortedXS = sorted(xs)
								sortedYS = []

								for tempI in range(len(xs)):
									toGet = xs.index(sortedXS[tempI])
									sortedYS.append(ys[toGet])

								plt.plot(sortedXS,sortedYS,label=methodName2)
								plt.scatter(xs,ys)
								print(methodName2,len(xs))
								print(xs,ys)
								print(sortedXS,sortedYS)



					# plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')

					# plt.title(f"Agent: {at} | Number of Observations: {numObs}")

					plt.xlabel('Number of Particles')
					import re
					plt.ylabel(re.sub( r"([A-Z])", r" \1", metric))

					# code.interact("...", local=dict(globals(), **locals()))

					plt.savefig(f"{tempSaveAt}Agent-{at}-NumObs{numObs}-JT-{estimateType}.png",bbox_inches='tight')
					plt.clf()
					plt.close("all")


if __name__ == '__main__':	

	# ASSUMES RESULTS OF EXPERIMENTS WERE PROCESSED ALREADY
	# And plots were created so that metrics avg file is present


	domain = "2d-multi"


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = f"Experiments{os.sep}{domain}")	
	args = parser.parse_args()


	if args.resultsFolder[-1] != os.sep:
		args.resultsFolder += os.sep


	plotFolder = f"{args.resultsFolder}PlotsAcrossFolders{os.sep}"
	makeFolder2(f"{args.resultsFolder}","PlotsAcrossFolders")

	methodName = "QRE-Multi-Particles"


	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# resultFolders = ["Experiments-NumObservations2-NumParticles2","Experiments-NumObservations2-NumParticles5",\
					# "Experiments-NumObservations5-NumParticles2","Experiments-NumObservations5-NumParticles5"]

	resultFoldersAll = os.listdir(args.resultsFolder)

	resultFolders = []
	resamples = []
	noises = []

	for each in resultFoldersAll:
		if "Experiments-NumObservations" in each:
			resultFolders.append(each)



	# Hardcoded params for now
	resamples = [90]
	noises = [200]
	resampleNEFF = True


	estimateTypes = ["MAP","EES"]

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	agentGroups = ["All", "Symmetric", "Asymmetric"]

	infoMetricsAVG = {}

	numObservations = []
	numParticles = []
	metrics = []

	agentTypes = []
	methods = []


	# Grab metric avgs per agent types from each exp folder
	for folder in resultFolders:

		expFolder = f"{args.resultsFolder}{folder}{os.sep}metricsAVG{os.sep}ThresholdX-0.05-ThresholdR-0.05{os.sep}"
		folderFiles = os.listdir(expFolder)


		metricFiles = {"All":[],"Symmetric":[],"Asymmetric":[]}

		# Filter by groups of agents
		for each in folderFiles:
			if "All" in each:
				metricFiles["All"].append(each)
			elif "Asymmetric" in each:
				metricFiles["Asymmetric"].append(each)
			else:
				metricFiles["Symmetric"].append(each)


		if metricFiles["All"] == []:
			print(f"No metricsAVG files found in {folder}. Need to make plots first.")
			continue


		splitted = folder.split("-")
		
		numObs = int(splitted[1].split("NumObservations")[1])
		numP = int(splitted[2].split("NumParticles")[1])


		if numObs not in numObservations:
			numObservations.append(numObs)

		if numP not in numParticles:
			numParticles.append(numP)
		

		key = f"NumObservations{numObs}-numParticles{numP}"

		infoMetricsAVG[key] = {}


		for g in agentGroups:

			infoMetricsAVG[key][g] = {}

			for eachFile in metricFiles[g]:

				with open(f"{expFolder}{eachFile}","rb") as file:
					info = pickle.load(file)


				metric = eachFile.split("-")[1]

				if metric not in metrics:
					metrics.append(metric)


				infoMetricsAVG[key][g][metric] = {"numObservations": numObs, "numParticles": numP, "info": info}


				# To get set of agentTypes and methods used within exps once. Assuming same set of methods used across all exps/folders.
				if agentTypes == []:
					agentTypes = list(info.keys())
					methods = info[agentTypes[0]].keys()



	# code.interact("...", local=dict(globals(), **locals()))	

	
	plt.rcParams.update({'axes.titlesize': 'large'})
	plt.rcParams.update({'axes.labelsize': 'large'})

	plt.rcParams.update({'xtick.labelsize': 'large'})
	plt.rcParams.update({'ytick.labelsize': 'large'})

	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["axes.titleweight"] = "bold"


	plt.rc('legend',fontsize='large')


	# metrics = ["KLD","JeffreysDivergence"]
	
	# '''
	for group in agentGroups:

		print(f"plotNumParticlesVsMetric() - {group} - ...")
		startTime = time()
		plotNumParticlesVsMetric(methodName,group)
		print("Time: ",time()-startTime)
		print()
		

		print(f"plotNumObservationsVsMetric() - {group} - ...")
		startTime = time()
		plotNumObservationsVsMetric(methodName,group)
		print("Time: ",time()-startTime)
		print()

	# '''



	# '''
	for group in agentGroups:

		print(f"plotNoiseVsMetric() - {group} ...")
		startTime = time()
		plotNoiseVsMetric(methodName,group)
		print("Time: ",time()-startTime)
		print()

		
		print(f"plotResampleVsMetric() - {group} ...")
		startTime = time()
		plotResampleVsMetric(methodName,group)
		print("Time: ",time()-startTime)
		print()
	# '''


	# # Plot All Methods
	for group in agentGroups:
		for eachObs in numObservations:
			for estimateType in estimateTypes:
				print(f"plotAllVsMetric() - {group} - NumObs: {eachObs} - {estimateType} ...")
				startTime = time()
				plotAllVsMetric(group,methodName,estimateType,eachObs)
				print("Time: ",time()-startTime)
				print()






















