import numpy as np
import code

def getUnifomParticles(tempN):

	dimensions = 4

	# tempN = ceil(N*percent)
	tempParticles = np.empty((tempN,dimensions))

	for d in range(dimensions):
		# Not including endpoint

		# For pskill dimension
		if d == dimensions-1:
			temp = np.random.uniform(ranges["start"][d],ranges["end"][d],size=tempN)
			tempParticles[:,d] = np.power(10,temp) # Exponentiate
		else:
			tempParticles[:,d] = np.random.uniform(ranges["start"][d],ranges["end"][d],size=tempN)

	return tempParticles


s = [0.004,0.004,-0.75,-3]
e = [np.pi/4,np.pi/4,0.75,1.5]


ranges = {}
ranges["start"] = s
ranges["end"] = e

particles = getUnifomParticles(10)


code.interact("...", local=dict(globals(), **locals()))