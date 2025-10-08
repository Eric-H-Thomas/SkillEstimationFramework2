import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader
import pickle

from matplotlib.colors import ListedColormap
from itertools import product


import pandas as pd
import numpy as np
import argparse,code
from random import sample

from scipy.stats import multivariate_normal,multivariate_t

import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from matplotlib.cm import ScalarMappable



# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}hockey.py")[0]


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


sys.modules["domain"] = sys.modules[__name__]



def getDomainName():
	return "hockey-multi"


def getCovMatrix(stdDevs,rho):
	# print("stdDevs: ",stdDevs)
	# print("rho",rho)

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))
	# code.interact("get...", local=dict(globals(), **locals()))
	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	# print(stdDevs,covMatrix)
	return covMatrix


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


def sample_action(rng,mean,L,a,noiseModel=None):

	# If noise model was not given, proceed to get it
	if noiseModel == None:
		N = getNoiseModel(mean,L**2)
	# Otherwise, use given noise model
	else:
		N = noiseModel

	#Get noise (sample)
	noise = N.rvs(random_state=rng)
	# print(noise)

	# Add noise to planned action (This creates the noisy action)
	na = [a[0] + noise[0], a[1] + noise[1]]

	return na


def getNormalDistribution(rng,covMatrix,resolution,mean,grid,saveAt=None,x=None):

	N = getNoiseModel(rng,mean,covMatrix)
	D = N.pdf(grid)

	# df = 3
	# D = multivariate_t.pdf(grid,loc=mean,shape=covMatrix,df=df)


	# print(D)
	# print(np.sum(D))
	# print()

	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  

	# D *= np.square(resolution[0]*resolution[1])
	D /= np.sum(D)


	if saveAt != None:
		plt.contourf(grid[:,:,0],grid[:,:,1],D)
		plt.savefig(f"{saveAt}{os.sep}pdfs{os.sep}xskill{x}.jpg",bbox_inches="tight")
		plt.close()
		plt.clf()

	# code.interact("get...", local=dict(globals(), **locals()))

	return D


def plotEVs():

	try:
		experimentFolder = sys.argv[1]
		playerID = sys.argv[2]
		typeShot = sys.argv[3]
	except:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/'), the ID of the player and type of shot as command line argument.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"


	saveAt = f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}Player{playerID}{os.sep}{typeShot}"

	folders = [f"{mainFolder}Plots{os.sep}",
				f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}",
				f"{mainFolder}Plots{os.sep}AngularHeatmapsPerXskill{os.sep}Player{playerID}{os.sep}",saveAt]

	for folder in folders:
		#If the folder doesn't exist already, create it
		if not os.path.exists(folder):
			os.mkdir(folder)



	folder = f"{mainFolder}AngularHeatmaps{os.sep}"
	fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"

	try:
		with open(folder+fileName,"rb") as infile:
			data = pickle.load(infile)
	except Exception as e:
		print(e)
		print("Can't load data for that player.")
		exit()


	cmapStr = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)


	# Feet
	delta = None

	spaces = sys.modules["spaces"].SpacesHockey([],1,sys.modules["domain"],delta)

	Y = spaces.targetsY
	Z = spaces.targetsZ

	targetsUtilityGridY,targetsUtilityGridZ = np.meshgrid(Y,Z)
	targetsUtilityGridYZ = np.stack((targetsUtilityGridY,targetsUtilityGridZ),axis=-1)

	shape = targetsUtilityGridYZ.shape
	listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0]*shape[1],shape[2]))



	# radians
	minX = 0.004 #0.17
	maxX = np.pi/4
	tempXskills = np.linspace(minX,maxX,10)

	xskills = np.concatenate((np.linspace(minX,tempXskills[1],10), np.array(tempXskills[2:])))

	rhos = [0.0]#,0.75]


	rng = np.random.default_rng(1000)


	allInfo = list(product(xskills,xskills))
	allInfo = list(product(allInfo,rhos))

	for each in range(len(allInfo)):
		allInfo[each] = list(eval(str(allInfo[each]).replace(")","").replace("(","")))


	allInfo = np.round(allInfo,4)


	allInfo = []

	for xi in xskills:
		allInfo.append([xi,xi,0.0])



	# Select sample rows
	rows = list(data.keys())
	try:
		sampleRows = sample(rows,20)
	except:
		sampleRows = rows

	data = {k:data[k] for k in sampleRows}



	saveAtOriginal = saveAt

	for index in data:

		saveAt = saveAtOriginal+os.sep+str(index)

		# If the folder doesn't exist already, create it
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)

		if not os.path.exists(f"{saveAt}{os.sep}pdfs"):
			os.mkdir(f"{saveAt}{os.sep}pdfs")


		heatmap = data[index]["heat_map"]
		shape = heatmap.shape
		listedUtilities = heatmap.reshape((shape[0]*shape[1],1))


		Zs = data[index]["gridUtilitiesComputed"]
		gridTargetsAngular = data[index]["gridTargetsAngular"]
		listedTargetsAngular = data[index]["listedTargetsAngular"]
		executedAction = [data[index]["shot_location"][0],data[index]["shot_location"][1]]
		executedActionAngular = data[index]["executedActionAngular"]


		dirs,elevations = data[index]["dirs"],data[index]["elevations"]

		# Assumming both same size (-1 to ofset for index-based 0)
		middle = int(len(dirs)/2) - 1
		mean = [dirs[middle],elevations[middle]]


		spaces.delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

		playerLocation = [data[index]["start_x"],data[index]["start_y"]]


		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

		norm = plt.Normalize(0.0,np.max(listedUtilities))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])

		ax1.scatter(listedTargetsUtilityGridYZ[:,0],listedTargetsUtilityGridYZ[:,1],c=cmap(norm(listedUtilities)))
		ax1.scatter(executedAction[0],executedAction[1],c="black",marker="x")
		ax1.set_title('Given Heatmap - YZ')
		fig.colorbar(sm,ax=ax1)


		ax2.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=cmap(norm(Zs.flatten())))
		ax2.scatter(executedActionAngular[0],executedActionAngular[1],c="black",marker="x")
		ax2.set_title('Computed Heatmap - Angular')
		fig.colorbar(sm,ax=ax2)

		plt.suptitle(f"Player Location: {playerLocation}")
		plt.tight_layout()

		plt.savefig(f"{saveAt}{os.sep}heatmaps.jpg",bbox_inches="tight")		
		plt.close()


		norm = plt.Normalize(np.min(Zs),np.max(Zs))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])


		for iii,x in enumerate(allInfo):

			covMatrix = getCovMatrix([x[0],x[1]],x[2])

			x = spaces.getKey([x[0],x[1]],x[2])

			spaces.pdfsPerXskill[x] = getNormalDistribution(rng,covMatrix,spaces.delta,mean,gridTargetsAngular,saveAt,x)

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,spaces.pdfsPerXskill[x],mode="same",fillvalue=0.0)
			
			maxEV = np.max(EVs)	
			ii = np.unravel_index(EVs.argmax(),EVs.flatten().shape)

			fig,ax = plt.subplots()

			cbar = fig.colorbar(sm,ax=ax)
			cbar.ax.get_yaxis().labelpad = 15
			cbar.ax.set_ylabel("Expected Utilities",rotation = 270)

			ax.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c = cmap(norm(EVs.flatten())))
			ax.scatter(listedTargetsAngular[:,0][ii],listedTargetsAngular[:,1][ii],color="black",marker="X",s=60,edgecolors="black",label="Max Expected Utility")

			ax.set_xlabel("")
			ax.set_ylabel("")
			ax.set_title(f"xskill: {x}")
			plt.savefig(f"{saveAt}{os.sep}{iii}-xskill{x}.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()


		# code.interact("...", local=dict(globals(), **locals()))




if __name__ == '__main__':
	

	plotEVs()


	# code.interact("...", local=dict(globals(), **locals()))









