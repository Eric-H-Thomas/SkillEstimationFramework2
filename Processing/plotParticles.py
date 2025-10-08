import argparse
import pickle
import code
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from matplotlib.patches import Circle
from math import pi, cos, sin


# For testing in interactive mode
def computeEES(tempN,particles,probs):
	ees = [0.0]*2

	aSum = 0.0
	x1 = 0.0
	x2 = 0.0

	for ii in range(tempN):
		print("-"*15)
		print("ii: ",ii)

		each = particles[ii]

		xs = each

		print("xs: ",xs)
		print("probs: ",probs[ii])

		for d in range(len(xs)):
			print(f"\txs[{d}]: {xs[d]}")
			print(f"\tprobs[{ii}][0]: {probs[ii][0]}")
			print()
			ees[d] += xs[d] * probs[ii][0]

		print(ees)
		print("-"*15)

		aSum += probs[ii][0]
		x1 += xs[0]
		x2 += xs[1]


	print("aSum: ",aSum)
	print("x1: ",x1)
	print("x2: ",x2)

	return ees

def draw_board(ax):

	#Draw the bullseye rings and scoring rings
	radii = [6.35, 15.9, 99, 107, 162, 170]
	for r in radii: 
		circle = Circle((0,0),r,fill=False)
		ax.add_artist(circle)

	#Do the radii 
	start_d = 15.9
	end_d = 170.0
	angle_increment = pi / 10.0
	angle = -angle_increment / 2.0

	for i in range(20):
		sx = start_d * cos(angle)
		sy = start_d * sin(angle)
		dx = end_d * cos(angle)
		dy = end_d * sin(angle)
		plt.plot([sx, dx], [sy, dy], color="Black")
		# print 'Angle = ', 180.0*angle/pi
		angle += angle_increment


def label_regions(slices,color="black"):

	angle_increment = pi / 10.0    
	angle = pi
	r = 130.0

	for i in range(1,21):
		x = r*cos(angle)
		y = r*sin(angle)
		plt.text(x,y,str(slices[i]),fontsize=12,horizontalalignment='center', color = color)
		angle += angle_increment

	# For single Bullseye
	# Value for bullseye will always be at the first index
	plt.text(0,0,str(slices[0]),fontsize=12,horizontalalignment='center', color = color)




def plotStates(saveAt,agent,method,estimates,states,intendedActions,noisyActions):


	folders = [f"{saveAt}{os.sep}States{os.sep}",
			f"{saveAt}{os.sep}States{os.sep}{agent}{os.sep}",
			f"{saveAt}{os.sep}States{os.sep}{agent}{os.sep}{method}{os.sep}"]

	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)


	cmap = plt.get_cmap("viridis")



	saveAt = folders[-1]

	for each in range(len(states)):

		fig = plt.figure(num=0)
		ax = plt.subplot()

		draw_board(ax)
		label_regions(states[each])


		ax.scatter(intendedActions[each][0],intendedActions[each][1],color='tab:blue',label="Intended Action")
		ax.scatter(noisyActions[each][0],noisyActions[each][1],color='tab:orange',label="Noisy Action")


		tt = f"EES:{np.round(estimates[0][each],4)} | ERS:{np.round(estimates[1][each],4)} | EPS:{np.round(estimates[2][each],4)}\n"
		try:
			tt += agent
		except:
			tt += "|".join(agent)
		plt.suptitle(tt)


		plt.legend()

		# cbar = plt.colorbar(sm,ax=ax1)

		fig.savefig(f"{saveAt}pf-{each}.png",bbox_inches = 'tight')
		plt.clf()
		plt.close("all")

		# code.interact("...", local=dict(globals(), **locals()))


def getRMSE(true_states,estimated_states):
    """
    Computes the RMSE between the true states and estimated states.
    
    Parameters:
    - true_states: np.array of shape (time_steps, dimensions)
    - estimated_states: np.array of shape (time_steps, dimensions)
    
    Returns:
    - rmse: float, the RMSE over all dimensions and time steps
    """
    # Ensure inputs are numpy arrays
    true_states = np.array(true_states)
    estimated_states = np.array(estimated_states)
    
    # Compute the squared differences
    squared_errors = (true_states - estimated_states) ** 2
    
    # Compute the mean of the squared errors
    mean_squared_errors = np.mean(squared_errors)
    
    # Compute the RMSE
    rmse = np.sqrt(mean_squared_errors)
    
    return rmse


def computeRMSE(trueXskill,allParticles,allProbs):

	folders = [f"{saveAt}{os.sep}ParticleFilter{os.sep}",
			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}",
			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-RMSE{os.sep}"]
	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)


	true = [trueXskill] * len(allParticles)

	estimated = []
	rmse = []

	for each in range(len(allParticles)):

		particles = allParticles[each][0] + allParticles[each][1]
		probs = allProbs[each]

		particles = np.array(particles)

		# Ignoring pskill
		particles = particles[:,:-1]

		estimated.append(np.average(particles,weights=probs.flatten(),axis=0))

		rmse.append(getRMSE(true[-1],estimated[-1]))


	# code.interact("...", local=dict(globals(), **locals()))


	true = np.array(true)
	estimated = np.array(estimated)

	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(0.0,len(allParticles))
	sm = ScalarMappable(cmap=cmap,norm=norm)
	sm.set_array([])


	fig = plt.figure(num=0)
	ax = plt.subplot()

	for each in range(len(allParticles)):
		plt.scatter(estimated[each,0], estimated[each,1],color=cmap(norm(each)))

	plt.plot(true[:,0], true[:,1],marker="*",c="black",label='True')
	cbar = plt.colorbar(sm,ax=ax)

	fig.savefig(f"{folders[-1]}estimates.png",bbox_inches = 'tight')
	plt.legend()
	plt.clf()
	plt.close("all")


	fig = plt.figure(num=0)
	plt.plot(range(len(rmse)),rmse,label='RMSE')
	fig.savefig(f"{folders[-1]}rmse.png",bbox_inches = 'tight')
	plt.clf()
	plt.close("all")


def bucketProbs(allParticlesNoNoise,allParticles,allProbs,allResampledProbs,numBuckets=100):

	folders = [f"{saveAt}{os.sep}ParticleFilter{os.sep}",
			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}",
			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-ProbsDist{os.sep}",
			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-ProbsDistAvg{os.sep}"]

	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)


	
	totalN = len(allParticlesNoNoise[0])

	# Find number of resampled particles
	N = len(allParticles[1][0])


	for each in range(len(allParticlesNoNoise)):
		print("EACH: ", each)

		# particles = allParticles[each][0] + allParticles[each][1]

		# Just resampled ones
		particles = np.array(allParticlesNoNoise[each][:N])
		print("Number resampled: ",N)


		probs = allResampledProbs[each]
		maxProbResampled = np.max(allResampledProbs[each])

		maxIndexes = np.where(probs==maxProbResampled)[0]
		maxParticlesResampled = particles[maxIndexes]

		
		# Max prob for all particles (+1 to skip initial uniform dist)
		maxProb = np.max(allProbs[each+1])

		maxIndexes = np.where(allProbs[each+1]==maxProb)[0]
		tempParticles = np.array(allParticles[each][0]+allParticles[each][1])
		maxParticles = tempParticles[maxIndexes]

		# code.interact("...", local=dict(globals(), **locals()))


		buckets = np.linspace(0.0,maxProbResampled,numBuckets)

		# b1 = np.linspace(0.0,0.001,numBuckets)
		# b2 = np.linspace(0.002,maxProbResampled,int(numBuckets/2))

		# b1 = np.linspace(0.0,0.0001,numBuckets)
		# b2 = np.linspace(0.0002,maxProbResampled,int(numBuckets/2))

		# b1 = np.linspace(0.0,0.0005,int(numBuckets/2))
		# b2 = np.linspace(0.0005,maxProbResampled,int(numBuckets/2))

		# buckets = np.concatenate((b1,b2))
		# buckets = np.sort(np.unique(buckets))

		
		info = {}

		# Resetting each time since getting info per observation
		for b in buckets:
			info[b] = {"particles":[],"probs":[]}


		# Find corresponding bucket	
		for ii in range(N):

			particle = particles[ii]


			# Find corresponding prob for particle (resample prob)
			prob = probs[ii]

			for b in buckets:

				if prob <= b:
					info[b]["particles"].append(particle.tolist())
					info[b]["probs"].append(prob)
					break


		# MAKE PLOT
		xs = []
		ys = []

		for b in buckets:
			xs.append(b)
			ys.append(len(info[b]["particles"]))


		fig = plt.figure(num=0,figsize=(12,8))
		plt.plot(xs,ys)
		plt.scatter(xs,ys)

		fig.savefig(f"{folders[-2]}pf-{each}.png",bbox_inches = 'tight')
		plt.clf()
		plt.close("all")



		infoParticles = {}
		infoProbs = {}

		# Count how many times particles where selected
		for b in buckets:

			infoParticles[b] = {}
			infoProbs[b] = {}

			particles = info[b]["particles"]

			for p in particles:

				if str(p) not in infoParticles[b]:
					infoParticles[b][str(p)] = 0.0
				
				infoParticles[b][str(p)] += 1.0


			for p in particles:
				infoProbs[b][str(p)] = infoParticles[b][str(p)]/totalN


		if each == 0:
			mode = "w"
		else:
			mode = "a"


		# Save info to file
		with open(f"{folders[-2]}infoProbs.txt",mode) as outfile:

			print(f"Iteration #{each}:",file=outfile)

			for b in buckets:
				particles = info[b]["particles"]

				for p in particles:
					index = info[b]["particles"].index(p)
					pStr = "[" + ",".join([f"{pi:.8f}" for pi in p]) + "]"
					print(f"\tBucket: {b} | P: {pStr} | Prob: {info[b]['probs'][index]} | Count: {infoParticles[b][str(p)]} | Count/{totalN}: {infoProbs[b][str(p)]}",file=outfile)


		# Save info to file
		with open(f"{folders[-2]}maxProbInfo.txt",mode) as outfile:

			# Find particle with max prob
			print("\n"+"-"*60,file=outfile)
			print(f"\nIteration #{each}:",file=outfile)
			print(f"\nMax prob: {maxProb}",file=outfile)

			for eachP in maxParticles:
				print(f"\tMax prob particle: {eachP}",file=outfile)

			print(f"\nMax prob resampled: {maxProbResampled}",file=outfile)

			for eachP in maxParticlesResampled:
				print(f"\tMax prob particle resampled: {eachP}",file=outfile)

			print("\nParticles with max prob: ",file=outfile)

			# for b in buckets:
			# 	particles = info[b]["particles"]
			# 	for j in range(len(info[b]["probs"])):
			# 		# if p == maxParticle:
			# 		if info[b]["probs"][j] == max
			# 			index = info[b]["particles"].index(p)
			# 			pStr = "[" + ",".join([f"{pi:.8f}" for pi in p]) + "]"
			# 			print(f"\tP: {pStr} | Prob: {info[b]['probs'][index]} | B: {b} | Computed prob: {infoProbs[b][str(p)]}",file=outfile)
			# # code.interact("...", local=dict(globals(), **locals()))


			# Find max computed prob and for which particle
			tempAllProbs = []
			tempAllParticles = []

			for b in buckets:
				particles = info[b]["particles"]
				for p in particles:
					tempAllProbs.append(infoProbs[b][str(p)])
					tempAllParticles.append(p)

			tempMax = np.max(tempAllProbs)
			print(f"Max computed prob: {tempMax}",file=outfile)
			print("Particles with max computed prob: ",file=outfile)

			iis = np.where(tempAllProbs == tempMax)[0]

			for ii in iis:
				print(f"P: {tempAllParticles[ii]}",file=outfile)
			print("-"*60,file=outfile)



		# '''
		tempPs, counts = np.unique(allParticlesNoNoise[each],return_counts=True,axis=0)
		tempProbs = counts/N

		# Save info to file
		with open(f"{folders[-2]}infoProbsParticlesNoNoise.txt",mode) as outfile:

			print(f"\nIteration #{each}:",file=outfile)


			for eachP in range(len(tempPs)):
				pStr = "[" + ",".join([f"{pi:.8f}" for pi in tempPs[eachP]]) + "]"

				toPrint = f"P: {pStr} | Count: {counts[eachP]} | Computed Prob: {tempProbs[eachP]}\n"

				cc = 0

				# Find all occurrences of that particle to get probs
				for b in buckets:
					particles = info[b]["particles"]
					for p in particles:
						if (p == tempPs[eachP]).all():
							index = info[b]["particles"].index(p)
							toPrint += f"\t Exp Prob: {info[b]['probs'][index]}"

							cc += 1

							if cc%5 == 0:
								toPrint += "\n"

				print(toPrint,file=outfile)

		# '''

		# code.interact("...", local=dict(globals(), **locals()))

		xs = []
		ys = []

		for b in buckets:

			xs.append(b)

			particles = info[b]["particles"]

			if len(particles) != 0:
				aSum = 0.0

				for p in particles:

					# infoProbs[b][str(p)] = infoParticles[b][str(p)]/N
					aSum += infoProbs[b][str(p)]

				avg = aSum/len(particles)

				ys.append(avg)
			else:
				ys.append(0.0)


		# MAKE PLOT
		fig = plt.figure(num=0,figsize=(12,8))
		plt.plot(xs,ys)
		plt.scatter(xs,ys)

		fig.savefig(f"{folders[-1]}pf-{each}.png",bbox_inches = 'tight')
		plt.clf()
		plt.close("all")


		# code.interact(f"each: {each}...", local=dict(globals(), **locals()))

	# code.interact("end...", local=dict(globals(), **locals()))


# def plotParticles(saveAt,agent,method,allParticles,estimates,resampledInfo,allProbs):
	
# 	N = len(allParticles[0])

# 	alpha = .30

# 	if N > 5000:
# 		alpha *= np.sqrt(5000)/np.sqrt(N) 


# 	folders = [f"{saveAt}{os.sep}ParticleFilter{os.sep}",
# 			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}",
# 			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}{os.sep}",
# 			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-JustResampled{os.sep}",
# 			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-Probs{os.sep}",
# 			f"{saveAt}{os.sep}ParticleFilter{os.sep}{agent}{os.sep}{method}-Probs-JustResampled{os.sep}"]

# 	for each in folders:
# 		if not os.path.exists(each):
# 			os.mkdir(each)


# 	print(resampledInfo)

# 	cmap = plt.get_cmap("viridis")



# 	for ii in range(4):

# 		saveAt = folders[ii+2]


# 		# allParticles = [ [[],initialRand] + [resampled,random] per particle] ]

# 		for each in range(len(allParticles)):

# 			fig = plt.figure(num=0,figsize=(12,8))

# 			ax1 = plt.subplot2grid((2,2),(0,0),colspan=2,rowspan=1)
# 			ax2 = plt.subplot2grid((2,2),(1,0))
# 			ax3 = plt.subplot2grid((2,2),(1,1))

# 			ax1.set_title("xskills")
# 			ax2.set_title("rhos")
# 			ax3.set_title("pskills")



			
# 			# For previous rfs (since they had a different structure)
# 			# resampled = allParticles[each]
# 			# random = []

# 			# particles = allParticles[each]
# 			# resampled = particles[0]
# 			# random = particles[1]

# 			# # skip first iter as shape mismatch
# 			if each == 0:
# 				resampled = allParticles[each][0]
# 				random = allParticles[each][1]
# 			else:
# 				resampled = allParticles[each-1][0]
# 				random = allParticles[each-1][1]


# 			s1 = len(resampled)
# 			s2 = len(random)


# 			norm = plt.Normalize(0.0,max(allProbs[each]))
# 			sm = ScalarMappable(cmap=cmap,norm=norm)
# 			sm.set_array([])

	
# 			if each == 89:
# 				code.interact("...", local=dict(globals(), **locals()))


# 			# For resampled particles
# 			if resampled != []:
# 				resampled = np.asarray(resampled)

# 				if each-1 in resampledInfo:
# 					c = 'tab:green'
# 				# No resampling occured, just noise added
# 				else:
# 					c = 'tab:orange'
				
# 				# PROBS
# 				if ii >= 2:
# 					ax1.scatter(resampled[:,0],resampled[:,1],alpha=alpha,c=cmap(norm(allProbs[each][:s1,:])))
# 					ax2.scatter(resampled[:,-2],[0]*len(resampled),alpha=alpha,c=cmap(norm(allProbs[each][:s1,:])))
# 					ax3.scatter(resampled[:,-1],[0]*len(resampled),alpha=alpha,c=cmap(norm(allProbs[each][:s1,:])))
# 				else:
# 					ax1.scatter(resampled[:,0],resampled[:,1],s=allProbs[each][:s1]*20000,alpha=alpha,color=c)
# 					ax2.scatter(resampled[:,-2],[0]*len(resampled),s=allProbs[each][:s1]*20000,alpha=alpha,color=c)
# 					ax3.scatter(resampled[:,-1],[0]*len(resampled),s=allProbs[each][:s1]*20000,alpha=alpha,color=c)



# 			# If not just resampled plot
# 			if ii != 1 and ii != 3:
	
# 				# For random particles
# 				if random != []:
# 					random = np.asarray(random)
					
# 					# PROBS
# 					if ii == 2:
# 						ax1.scatter(random[:,0],random[:,1],alpha=alpha,c=cmap(norm(allProbs[each][s1:,:])))
# 						ax2.scatter(random[:,-2],[0]*len(random),alpha=alpha,c=cmap(norm(allProbs[each][s1:,:])))
# 						ax3.scatter(random[:,-1],[0]*len(random),alpha=alpha,c=cmap(norm(allProbs[each][s1:,:])))
# 					else:
# 						ax1.scatter(random[:,0],random[:,1],s=allProbs[each][s1:]*20000,alpha=alpha,color='tab:blue')
# 						ax2.scatter(random[:,-2],[0]*len(random),s=allProbs[each][s1:]*20000,alpha=alpha,color='tab:blue')
# 						ax3.scatter(random[:,-1],[0]*len(random),s=allProbs[each][s1:]*20000,alpha=alpha,color='tab:blue')


# 			if each >= 1:
# 				tt = f"EES:{np.round(estimates[0][each-1],4)} | ERS:{np.round(estimates[1][each-1],4)} | EPS:{np.round(estimates[2][each-1],4)}\n"
				
# 				try:
# 					tt += agent
# 				except:
# 					tt += "|".join(agent)
# 				plt.suptitle(tt)


# 			if each == 0:
# 				temp  = "init"
# 			else:
# 				temp = each-1


# 			if ii >= 3:
# 				cbar = plt.colorbar(sm,ax=ax1)

# 			fig.savefig(f"{saveAt}pf-{temp}.png",bbox_inches = 'tight')
# 			plt.clf()
# 			plt.close("all")

# 			# code.interact("...", local=dict(globals(), **locals()))



if __name__ == '__main__':

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	parser.add_argument("-file", dest = "file", help = "Name of results file containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	if args.resultsFolder[-1] != os.sep:
		args.resultsFolder += os.sep

	expFolder = args.resultsFolder + "results" + os.sep + args.file

	saveAt = args.resultsFolder


	with open(expFolder,"rb") as infile:
		info = pickle.load(infile)

	print(info.keys())


	# methods = []

	# for each in info.keys():
	# 	if "-xSkills" in each:
	# 		methods.append(each.split("-xSkills"))


	# JEEDS via PFE
	# method = "QRE-Multi-Particles-33-Resample-100%-NoiseDiv-1-JT-EES"
	# method2 = "QRE-Multi-Particles-33-Resample-100%-NoiseDiv-1"
	

	# method = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv50-JT-EES"
	method = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES"
	# method2 = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv50"
	method2 = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200"
	resampleKey = method2 + "-whenResampled"


	agent = info["agent_name"]
	print(agent)
	agent = agent[1]


	allParticles = info[method2+"-allParticles"]
	# allParticlesNoNoise = info[method2+"-allParticlesNoNoise"]
	# allNoises = info[method2+"-allNoises"]
	allProbs = np.array(info[method2+"-allProbs"])
	estimates = [info[method+"-xSkills"],info[method+"-rhos"],info[method+"-pSkills"]]
	resampledInfo = info[resampleKey]

	# allResampledProbs = np.array(info[method2+"-allResampledProbs"])

	# particles = np.array(allParticles[25])[:,0:2]
	# probs = allProbs[26]
	# computeEES(900,particles,probs)
	# np.average(particles[0:900],weights=probs[0:900].flatten(),axis=0)	

	# code.interact("...", local=dict(globals(), **locals()))

	plotParticles(saveAt,agent,method,allParticles,estimates,resampledInfo,allProbs)
	
	# bucketProbs(allParticlesNoNoise,allParticles,allProbs,allResampledProbs)
	

	# Only works for Target Agent
	# temp = agent.split("|")[1:]
	# trueXskill = [float(each[1:]) for each in temp]

	# computeRMSE(trueXskill,allParticles,allProbs)


	'''
	states = info["states"]["states"]
	intendedActions = info["intended_actions"]
	noisyActions = info["noisy_actions"]
	plotStates(saveAt,agent,method,estimates,states,intendedActions,noisyActions)
	'''

	# code.interact("...", local=dict(globals(), **locals()))


	
