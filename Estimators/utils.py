import numpy as np
import scipy
from copy import deepcopy
import code

# Obtained from:
# http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
# On May/10/2022
def computePDF(x,means,covs):
	return np.exp(multiple_logpdfs(x,means,covs))


# Obtained from:
# http://gregorygundersen.com/blog/2020/12/12/group-multivariate-normal-pdf/
# On May/10/2022
def multiple_logpdfs(x,means,covs):

	# Thankfully, NumPy broadcasts `eigh`.

	vals, vecs = np.linalg.eigh(covs)

	# Compute the log determinants across the second axis.
	logdets = np.sum(np.log(vals), axis=1)

	# Invert the eigenvalues.
	valsinvs = 1./vals
	
	# Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
	Us = vecs * np.sqrt(valsinvs)[:, None]
	devs = x - means

	# Use `einsum` for matrix-vector multiplications across the first dimension.
	devUs = np.einsum('ni,nij->nj', devs, Us)

	# Compute the Mahalanobis distance by squaring each term and summing.
	mahas = np.sum(np.square(devUs), axis=1)
	
	# Compute and broadcast scalar normalizers.
	dim = len(vals[0])
	log2pi = np.log(2 * np.pi)

	return -0.5 * (dim * log2pi + mahas + logdets)


def getPDFsAndEVsBilliardsSampling(spaces,x,action,otherArgs):
	
	agentType = otherArgs["agentType"]
	shot = otherArgs["shot"]
	processedShot = otherArgs["processedShot"]
	
	# [:,0] to only obtain column 0
	# Column 0 = estimated phi, column 1 = method/type used to estimate phi
	possibleShots = np.array(otherArgs["possibleShots"])[:,0]

	sizeActionSpace = 360.0
	numSamples = 10


	space = spaces.spacesPerXskill[agentType][x]


	pdfs = scipy.stats.norm.pdf([action]*len(possibleShots),loc=possibleShots,scale=[x]*len(possibleShots))

	evs = []

	# targets = possibleShots
	for eachShot in possibleShots:

		successfulCount = 0.0

		tempShot = deepcopy(shot)


		# Sample shot N times
		for ni in range(numSamples):

			# Add noise to shot based on current xskills
			noise = np.random.normal(0.0,x)
			tempNoisyPhi = eachShot + noise

			tempShot.phi = tempNoisyPhi 

			# Execute shot - returns boolean
			result, gameShot, gameState = spaces.domain.executeShot(tempNoisyPhi,tempShot)
		
			if result:
				successfulCount += 1


		# Compute success rate = evs
		evs.append(successfulCount/float(numSamples))

	return pdfs,evs


def getPDFsAndEVsBilliardsFocalEV(spaces,xskills,action,otherArgs):
	
	agentType = otherArgs["agentType"]
	shot = otherArgs["shot"]
	processedShot = otherArgs["processedShot"]
	
	# [:,0] to only obtain column 0
	# Column 0 = estimated phi, column 1 = method/type used to estimate phi
	possibleShots = np.array(otherArgs["possibleShots"])[:,0]

	sizeActionSpace = 360.0
	numSamples = 10

	space = spaces.spacesPerXskill[agentType][x]

	nv = []

	As = []
	Vs = []
	Qs = []


	# GET SOME SAMPLES FOR EACH XSKILL
	for x in xskills:

		# Get normal pdf for xskill
		g = stats.norm(scale=x)

		successfulCount = 0.0
		tempShot = deepcopy(shot)

		# Sample shot N times
		for ni in range(numSamples):

			# Add noise to shot based on current xskills
			noise = np.random.normal(0.0,x)
			tempNoisyPhi = eachShot + noise

			tempShot.phi = tempNoisyPhi 

			# Execute shot - returns boolean
			result,gameShot,gameState = spaces.domain.executeShot(tempNoisyPhi,tempShot)
			
			if result:
				successfulCount += 1

			# Save info
			As.append(tempNoisyPhi)
			Vs.append(result)

			# Get and store pdf value for sampled actions with cuRrent xskill
			Qs.append(g.pdf(tempNoisyPhi))



	# COMPUTE EV ESTIMATES FOR ALL XSKILLS
	for x in xskills:
	
		pdfs = []
		evs = []

		# To track the value and weight
		curV = 0.0
		curW = 0.0

		# Get the current pdf
		g = stats.norm(scale=x)

		# Loop over all the samples
		for i, ai in enumerate(As):

			# Get the value
			vi = Vs[i]
			
			# Get the distance of this action from the intended one
			d = spaces.domain.calculate_wrapped_action_difference(action,ai)
			
			# Get the probability of that action under this x 
			W  = g.pdf(d)

			# Importance Weighting to get correct estimate
			curV += vi*W/Qs[i]
			curW += W/Qs[i]


			#############################
			# ????????????????????????? #
			#############################
			pdfs.append(W)
			evs.append(vi*W/Qs[i])
			#############################


		# Final EV estimate is just the weighted average
		nv.append(curV/curW)


	return pdfs,nv