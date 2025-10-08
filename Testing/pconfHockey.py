from pathlib import Path
from importlib.machinery import SourceFileLoader
import os,sys,code,copy,json

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Testing{os.sep}pconfHockey.py")[0]

module = SourceFileLoader("hockey",f"{mainFolderName}{os.sep}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import random

from scipy.stats import multivariate_normal


def getAngle(point1,point2):

	x1,y1 = point1
	x2,y2 = point2
	
	angle = np.arctan2(y2-y1,x2-x1)
	
	return angle


def getNoiseModel(rng,mean=[0.0,0.0],X=0.0):

	# X is squared already (x**2 = variance)
	
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy


	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	# print(seed)
	N = multivariate_normal(mean=mean,cov=X,seed=seed)
	
	return N


def testHits(folder):

	numTries = 10000.0

	minXskill = 0.004
	maxXskill = np.pi/4
	xSkills = np.linspace(minXskill,maxXskill,num=2)


	seeds = np.random.randint(0,1000000,1)
	rng = np.random.default_rng(seeds[0])


	leftPost = np.array([89,-3])
	rightPost = np.array([89,3])

	leftAugmented = np.array([89,-9])
	rightAugmented = np.array([89,9])

	top = 4
	topAugmented = 8



	# Select target in the middle of the net - [Y,Z]

	# playerLocation = [86,0]
	# playerLocation = [25,0]
	# playerLocation = [0,0]
	playerLocation = [-500,0]


	# Generate edges - directions
	dirL = getAngle(playerLocation,leftAugmented)
	dirR = getAngle(playerLocation,rightAugmented)

	# Generate edges - elevations
	dist1 = np.linalg.norm(playerLocation-leftAugmented)
	dist2 = np.linalg.norm(playerLocation-rightAugmented)

	minDist = min(dist1,dist2)
	elevationTop = np.arctan2(topAugmented,minDist)


	action = [(dirL+dirR)/2,elevationTop/2]


	mean = [0,0]



	# Prep file for saving results
	outFile = open(f"{folder}PercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.txt", "w")


	print(f"\n--- Performing testHit experiment... ---")
	allPercentHits = []


	for xs in xSkills:

		xs = round(xs,4)
		print(f"\txskill: {xs}")

		N = getNoiseModel(rng,mean,xs**2)

		# D = N.pdf(grid)
		# D /= np.sum(D)

		# plt.contourf(grid[:,:,0],grid[:,:,1],D)

		# code.interact("...", local=dict(globals(), **locals()))


		hits = 0.0


		# fig,ax = plt.subplots()

		# # Plot boundaries
		# ax.scatter(dirL,0.0,c = "black")
		# ax.scatter(dirR,elevationTop,c = "black")
		# ax.scatter(dirL,elevationTop,c = "black")
		# ax.scatter(dirR,0.0,c = "black")


		for tries in range(int(numTries)):

			# Get noise sample
			noise = N.rvs()

			# Add noise to action
			noisyAction = [action[0]+noise[0],action[1]+noise[1]]

			# print(f"\t\t action: {action}")
			# print(f"\t\t noisyAction: {noisyAction}")

			# ax.scatter(action[0],action[1],c = "red", marker = "*")


			# Verify if the action hits the board or not
			if (noisyAction[0] >= dirL and noisyAction[0] <= dirR) and\
				(noisyAction[1] >= 0.0 and noisyAction[1] <= elevationTop):
				hits += 1.0

				# ax.scatter(noisyAction[0],noisyAction[1],c = "blue", marker = "*")



			####################################
			# PLOT - Strike Zone Board
			####################################
			'''
			fig,ax = plt.subplots()

			# Plot boundaries
			ax.scatter(dirL,0.0,c = "black")
			ax.scatter(dirR,elevationTop,c = "black")
			ax.scatter(dirL,elevationTop,c = "black")
			ax.scatter(dirR,0.0,c = "black")
			
			# Plot actual executed action & EV
			ax.scatter(action[0],action[1],c = "red", marker = "*")
			ax.scatter(noisyAction[0],noisyAction[1],c = "blue", marker = "*")

			ax.set_title(f"xskill: {xs}")
			plt.show()
			plt.clf()
			plt.close()
			# code.interact("...", local=dict(globals(), **locals()))
			'''
			####################################
			

		# code.interact("...", local=dict(globals(), **locals()))
		# plt.clf()

		percentHit = (hits/numTries)*100.0
		allPercentHits.append(percentHit)
		
		print(f"\t\txSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%")
		# Save to file
		print(f"xSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%",file=outFile)


	outFile.close()

	plt.plot(xSkills,allPercentHits)
	plt.xlabel('xSkills')
	plt.ylabel('% Hits')
	plt.savefig(f"{folder}xskillsVsPercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.png")
	plt.clf()
	plt.close()


def pconf(folder):

	info = {}

	pconfPerXskill = {}


	xskills = np.round(np.linspace(0.004,np.pi/4,num=10),4)
	lambdas = np.logspace(-3,1.6,10)

	seeds = np.random.randint(0,1000000,1)
	rng = np.random.default_rng(seeds[0])


	# Go through all of the execution skills
	for x in xskills:
		print('Generating data for execution skill level', x)

		info[x] = {}

		prat = [] # This is where the probability of rational reward will be stored
		mins = [] # Store min reward possible
		maxs = [] # Store max reward possible
		means = [] # Store the mean of the possible rewards (this is the uniform random reward)
		evs = [] # Store the ev of the current agent's strategy

		key = "|".join(map(str,[x,x]))+f"|0.0"


		for l in lambdas:   

			size = len(data)  

			max_rs = np.zeros(size)
			min_rs = np.zeros(size)
			exp_rs = np.zeros(size)
			mean_rs = np.zeros(size)

			si = 0
			
			for angularHeatmap in data:

				dirs,elevations = data[angularHeatmap]["dirs"],data[angularHeatmap]["elevations"]
				delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

				# Assumming both same size (-1 to ofset for index-based 0)
				middle = int(len(dirs)/2) - 1
				mean = [dirs[middle],elevations[middle]]

				gridTargetsAngular = data[angularHeatmap]["gridTargetsAngular"]

				covMatrix = sys.modules["domain"].getCovMatrix([x,x],0.0)


				Zn = sys.modules["domain"].getNormalDistribution(rng,covMatrix,delta,mean,gridTargetsAngular)

				Zs = data[angularHeatmap]["gridUtilitiesComputed"]

				EVs =  convolve2d(Zs,Zn,mode="same",fillvalue=0.0)


				# Get the values from the ev 
				max_rs[si] = np.max(EVs)
				min_rs[si] = np.min(EVs) 
				mean_rs[si] = np.mean(EVs) 

				# Bounded decision-making with lambda = l
				b = np.max(EVs*l)
				expev = np.exp(EVs*l-b)
				sumexp = np.sum(expev)
				P = expev/sumexp

				# Store bounded agent's EV
				boundedEVs = P*EVs
				exp_rs[si] = np.sum(boundedEVs)


				si += 1

			
			prat.append(np.mean((exp_rs - mean_rs)/(max_rs - mean_rs)))
			mins.append(np.mean(min_rs))
			means.append(np.mean(mean_rs))
			maxs.append(np.mean(max_rs))
			evs.append(np.mean(exp_rs))

		plt.plot(lambdas,prat,label='x='+str(x))
		plt.scatter(lambdas,prat)

		pconfPerXskill[key] = {"lambdas":lambdas, "prat": prat}


	plt.xlabel('Lambda')
	plt.ylabel('% Rational Reward')
	plt.legend()
	plt.savefig(f"{folder}lambdasVSprat-{len(data)}.png")



	# code.interact("pconf()...", local=dict(globals(), **locals()))
	return pconfPerXskill,xskills,lambdas


if __name__ == '__main__':

	if len(sys.argv) != 2:
		print("Please provide experiment folder.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{sys.argv[1]}{os.sep}Data{os.sep}"
	folder = f"{mainFolder}AngularHeatmaps{os.sep}"
	files = os.listdir(folder)
	# print(files)

	data = {}
	N = 10


	# Grab sample shots from all available files
	for eachFile in files:

		if "angular" not in eachFile:
			continue

		with open(folder+eachFile,"rb") as infile:
			info = pickle.load(infile)

			try:
				temp = dict(random.sample(sorted(info.items()),N))
			except:
				temp = dict(random.sample(sorted(info.items()),len(info)))

			data.update(temp)

	
	# '''
	pconfPerXskill,xskills,lambdas = pconf(mainFolder)


	with open(mainFolder+"pconf","wb") as outfile:
		pickle.dump(pconfPerXskill,outfile)

	with open(mainFolder+"pconfData","wb") as outfile:
		pickle.dump(data,outfile)
	# '''

	# testHits(folder)


	# code.interact("...", local=dict(globals(), **locals()))


	
