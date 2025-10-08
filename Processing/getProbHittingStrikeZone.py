import numpy as np
import code
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

def getCovMatrix(stdDevs,rho):

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	return covMatrix


def getNoiseModel(covMatrix,mean=[0.0,0.0]):

	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	N = multivariate_normal(mean=mean,cov=covMatrix)
	
	return N


if __name__ == '__main__':
	

	numTries = 1_000_000

	minPlateX = -0.71
	maxPlateX = 0.71

	minPlateZ = 1.546
	maxPlateZ = 3.412


	# Select target in the middle of the board
	x = (minPlateX+maxPlateX)/2.0
	z = (minPlateZ+maxPlateZ)/2.0
	action = [x,z]


	# PFE-NEFF
	'''
	allStdDevs = [[0.320661477744379, 0.42692607500778773], 
				 [0.2906137413356851, 0.5373972397545834],
				 [0.2783565414339406, 0.3734617733500142],
				 [0.34815869607615507, 0.4988160100622659], 
				 [0.6623698903094134, 0.5912169039290699], 
				 [0.3190506309525008, 0.41243913919570835]]

	allRhos = [-0.098021078, -0.135048583, -0.122881, 0.080270147, -0.068913905, -0.089321397]
	'''


	# JEEDS
	#'''
	allStdDevs = [[0.634565705,0.634565705], 
				 [0.879165411,0.879165411],
				 [0.726315819,0.726315819],
				 [0.87472542,0.87472542],
			     [0.811180152,0.811180152],
				 [0.807017557,0.807017557]]

	allRhos = [0.0,0.0,0.0,0.0,0.0,0.0]
	#'''


	print(f"\n--- Performing testHit experiment... ---")

	allPercentHits = []


	for each in range(len(allStdDevs)):

		stdDevs = allStdDevs[each]
		rho = allRhos[each]

		covMatrix = getCovMatrix(stdDevs,rho)



		xs = np.round(stdDevs,4)
		# print(f"\tXskill: ({xs[0]}, {xs[1]}, {rho})")


		N = getNoiseModel(covMatrix)


		hits = 0.0

		for tries in range(int(numTries)):

			# Get noise sample
			noise = N.rvs()

			# Add noise to action
			noisyAction = [action[0]+noise[0],action[1]+noise[1]]

			#print(f"\t\t action: {action}")
			#print(f"\t\t noisyAction: {noisyAction}")

			# Verify if the action hits the board or not
			if (noisyAction[0] >= minPlateX and noisyAction[0] <= maxPlateX) and\
				(noisyAction[1] >= minPlateZ and noisyAction[1] <= maxPlateZ):
				hits += 1.0


			####################################
			# PLOT - Strike Zone Board
			####################################
			'''
			fig,ax = plt.subplots()

			# Plot boundaries
			ax.scatter(minPlateX,minPlateZ,c = "black")
			ax.scatter(maxPlateX,maxPlateZ,c = "black")
			ax.scatter(minPlateX,maxPlateZ,c = "black")
			ax.scatter(maxPlateX,minPlateZ,c = "black")
			
			# Plot actual executed action & EV
			ax.scatter(action[0],action[1],c = "red", marker = "*")
			ax.scatter(noisyAction[0],noisyAction[1],c = "blue", marker = "*")

			ax.set_title(f"xskill: {xs}")
			plt.show()
			plt.clf()
			plt.close()
			code.interact("...", local=dict(globals(), **locals()))
			'''
			####################################
			

		percentHit = (hits/numTries)*100.0
		allPercentHits.append(percentHit)
		
		print(f"xSkill: ({xs[0]}, {xs[1]}, {rho}) | Total Hits: {hits} out of {numTries} -> {percentHit}%")


	code.interact("...", local=dict(globals(), **locals()))