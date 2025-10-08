import os
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


minPlateX = -0.71
maxPlateX = 0.71

minPlateZ = 1.546
maxPlateZ = 3.412


def getKey(info,r):
	return "|".join(map(str,info))+f"|{r}"


def getCovMatrix(stdDevs,rho):
	# print("stdDevs: ",stdDevs)
	# print("rho",rho)

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))

	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	# print("covMatrix")
	# print(covMatrix)
	return covMatrix


def getNoiseModel(rng,mean,covMatrix):
	
	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy

	N = multivariate_normal(mean=mean,cov=covMatrix,seed=seed)
	
	return N



def plotOtherInfo(titleStr):

	# Overlay strike zone dimensions on plot
	# Plate_x: [-0.71,0.71]
	# Plate_z: [1.546,3.412]
	plt.hlines(y=minPlateZ,xmin=minPlateX,xmax=maxPlateX,color="k")
	plt.hlines(y=maxPlateZ,xmin=minPlateX,xmax=maxPlateX,color="k")
	plt.vlines(x=minPlateX, ymin=minPlateZ,ymax=maxPlateZ,color="k")
	plt.vlines(x=maxPlateX, ymin=minPlateZ,ymax=maxPlateZ,color="k")

	plt.xlabel("Horizontal Location (Pitcher's Perspective)")
	plt.ylabel("Vertical Location")

	plt.title(titleStr)
	plt.tight_layout()



def getPlots(numTries,xSkills,rhos):

	rng = np.random.default_rng(np.random.randint(1,1000000000))

	# Select target in the middle of the board
	x = (minPlateX+maxPlateX)/2.0
	z = (minPlateZ+maxPlateZ)/2.0
	action = [x,z]


	# Center of the strike zone = [0.0,2.479]
	# mean = [0.0,(minPlateZ+maxPlateZ)/2]
	mean = [0.0,0.0]


	allNoisyActions = {}
	allHits = {}

	for xs in xSkills:

		xs = round(xs,4)

		for rho in rhos:

			key = getKey([xs,xs],rho)

			# print(f"xskill: {key}")

			allNoisyActions[key] = []
			allHits[key] = 0


			covMatrix = getCovMatrix([xs,xs],rho)
			N = getNoiseModel(rng,mean,covMatrix)


			hits = 0

			for tries in range(int(numTries)):

				# Get noise sample
				noise = N.rvs()

				# Add noise to action
				noisyAction = [action[0]+noise[0],action[1]+noise[1]]

				if (noisyAction[0] >= minPlateX and noisyAction[0] <= maxPlateX) and\
					(noisyAction[1] >= minPlateZ and noisyAction[1] <= maxPlateZ):
					hits += 1


				allNoisyActions[key].append(noisyAction)

	
			allHits[key] = hits


	return allNoisyActions,allHits

			
			
if __name__ == '__main__':

	numTries = 500


	# xSkills = [0.17, 0.2825423728813559, 0.40915254237288134,
	 		# 0.5216949152542373, 0.7045762711864406, 1.0417, 2.81]

	xSkills = [0.17, 0.83, 1.49, 2.15, 2.81]

	xSkills = np.round(np.linspace(0.17,2.81,30),4)

	rhos = [0.0]


	saveAt = f"Experiments{os.sep}baseball{os.sep}Study5{os.sep}"


	if not os.path.exists(saveAt):
		os.mkdir(saveAt)
	

	allNoisyActions,allHits = getPlots(numTries,xSkills,rhos)


	for xs in xSkills:

		xs = round(xs,4)

		for rho in rhos:

			key = getKey([xs,xs],rho)

			print(f"xskill: {key}")


			fig, ax = plt.subplots()

			for ii in range(len(allNoisyActions[key])):
				ax.scatter(allNoisyActions[key][ii][0],allNoisyActions[key][ii][1],c = "blue", marker = "*")


			hits = allHits[key]
			plotOtherInfo(f"Hits: {hits}/{numTries} ({(hits/numTries)*100:.2f}%)")


			ax.set_aspect('equal', adjustable='box')
			plt.xlim(-1.5,1.5)
			plt.ylim(1,4)


			plt.savefig(f"{saveAt}{key}.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()





# How man

