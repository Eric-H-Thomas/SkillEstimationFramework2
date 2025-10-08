import numpy as np
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt
import code

# Example: 4D space with known bounds
bounds = np.array([
	[0.004, np.pi/4],
	[0.004, np.pi/4],
	[-0.75, 0.75],
	[-3,1.6]   
	])

D = bounds.shape[0]
N = 1024  # number of particles

# 1. Create Sobol sequence sampler
sobol = Sobol(d=D, scramble=True)
samples = sobol.random(n=N)  # samples in [0, 1]^D

# 2. Scale to actual bounds
scaled_particles = bounds[:,0] + samples * (bounds[:,1] - bounds[:,0])
scaled_particles = scaled_particles.T  # shape (N, D)

print(scaled_particles.shape)  # should be (1000, 4)

scaled_particles[-1,:] = np.power(10,scaled_particles[-1,:] )




tempParticles = np.empty((N,D))

for d in range(D):

	tempParticles[:,d] = np.round(np.random.uniform(bounds[d][0],bounds[d][1],size=N),4)

	# For pskill dimension
	if d == D-1:
		tempParticles[:,d] = np.power(10,tempParticles[:,d]) # Exponentiate


# plt.scatter(tempParticles[:,0],tempParticles[:,1])


# code.interact("...", local=dict(globals(), **locals()))





fig = plt.figure(num=0,figsize=(16,9))

plt.subplots_adjust(wspace=0.3,hspace=0.4)

ax1 = plt.subplot2grid((3,2),(0,0))
ax2 = plt.subplot2grid((3,2),(1,0))
ax3 = plt.subplot2grid((3,2),(2,0))
ax4 = plt.subplot2grid((3,2),(0,1))
ax5 = plt.subplot2grid((3,2),(1,1))
ax6 = plt.subplot2grid((3,2),(2,1))


ax1.scatter(tempParticles[:,0],tempParticles[:,1])
ax1.set_title("Execution Skill")

ax2.scatter(range(N),tempParticles[:,2])
ax2.set_title("Rhos")

ax3.scatter(range(N),tempParticles[:,3])
ax3.set_title("Dec-Making Skills")



ax4.scatter(scaled_particles[0,:],scaled_particles[1,:])
ax4.set_title("Execution Skill")

ax5.scatter(range(N),scaled_particles[2,:])
ax5.set_title("Rhos")

ax6.scatter(range(N),scaled_particles[3,:])
ax6.set_title("Dec-Making Skills")

fig.suptitle("Random Uniform | Sobol")

# plt.show()

plt.savefig("comparisonInits.png")


plt.clf()
plt.close()


