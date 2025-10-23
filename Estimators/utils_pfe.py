import numpy as np
import time, code
import pandas as pd
from scipy.signal import convolve2d
from scipy.stats import multivariate_normal

from Estimators.utils import *


def draw_noise_sample(rng,mean=[0.0,0.0],X=0.0):

	# X is squared already (x**2 = variance)
	
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy

	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	N = multivariate_normal(mean=mean,cov=X,seed=seed)
	
	return N


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


	N = draw_noise_sample(rng,mean,covMatrix)
	
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


def getKey(info,r):
	return "|".join(map(str,info))+f"|{r}"


def getCovMatrix(stdDevs,rho):

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	return covMatrix


def workUpdate(task,each):

	pdfsPerXskill = {}
	evsPerXskill = {}

	# Causes pickling error
	# spaces.updateSpaceParticles(task["rng"],each,task["state"],task["otherArgs"],wid)

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	# allTempData = task["allTempData"]
	minUtility = task["minUtility"]
	
	# Assuming method will get called only with multi domain
	covMatrix = getCovMatrix(each[:-2],each[-2])
	key = getKey(each[:-2],each[-2])


	if key not in pdfsPerXskill:
		# print(f"Computing pdfs for {key}... (wid: {task['wid']})")
		pdfsPerXskill[key] = getNormalDistribution(task["rng"],covMatrix,delta,task["targetsPlateXFeet"],task["targetsPlateZFeet"])
	else:
		# print(f"Pdfs info is present for {key}. (wid: {task['wid']})")
		pass


	if key not in evsPerXskill:
		# print(f"Computing EVs for {key}... (wid: {task['wid']})")
		
		Zs = task["Zs"]

		# t1 = time.perf_counter()
		evsPerXskill[key] = convolve2d(Zs,pdfsPerXskill[key],mode="same",fillvalue=minUtility)
		# print(f"Total time for convolve2d: {time.perf_counter()-t1:.4f}")
	else:
		# print(f"EVs info present for {key}... (wid: {task['wid']})")
		pass

	return pdfsPerXskill, evsPerXskill

































