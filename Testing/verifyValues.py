import code
import argparse
import os
import imp
import numpy as np
import scipy
from scipy.signal import convolve2d, convolve, fftconvolve
import math
import json
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable


# For given target, keep track of distribution for other states


# Verify prob of getting to state = 2 if selecting optimal target
# How often do we hit state 2 from state 2?



def valueIteration(domain,x,actions,resolution):

	#################################
	np.random.seed(0)
	#################################


	##############################################################################
	# SET PARAMS FOR VALUE ITERATION
	##############################################################################

	# Discount factor
	gamma = 1.0

	# Error tolerance (when do we stop?)
	tolerance = 0.001 * x

	# How much did it change this iter?
	delta = 10.0

	iterations = 0
		
	startScore = domain.getPlayerStartScore()
	# V = -1 * np.linspace(x,1.5*x,startScore+1)
	V = -1 * np.linspace(1,16,startScore+1)


	# 0.0 since game ends (done)
	V[0] = 0.0
	# 0.0 since never gets here
	V[1] = 0.0


	PI = [[None,None]] * len(states)
	PI_EV = [None] * len(states)


	numSamples = 100_000
	singleProbs, doubleProbs = precalc(domain,actions,x**2,numSamples)


	##############################################################################
	# PERFORM VALUE ITERATION
	##############################################################################


	# Xn,Yn,Zn = getSymmetricNormalDistribution(x,resolution)
	# Xn,Yn,Zn = getSymmetricNormalDistributionAlt(x,args.delta)


	while delta > tolerance:

		# To remember EVs for different scores/states
		# Resets each time as interested in 
		# remembering the EVs once converged only
		allEVs = {}

		# Reset delta
		delta = 0.0
		
		# Skip states 0 & 1
		# for s in states[2:]:
		for s in range(2,len(V)):

			# print(s)
			# code.interact("HERE...", local=dict(globals(), **locals()))

			# To print every 5 iters and 20 states
			if iterations%5 == 0 and s%20 == 0:
				print("Iteration: " + str(iterations) + " | State: " + str(s))
			

			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# Compute EVs - With convolution
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			'''
			# Get values
			Xs,Ys,Zs = get_values(x,resolution,s,V)
			# Xs, Ys, Zs = getValuesGivenTargets(resolution,s,V,Xn,Yn)

			# Convolve to produce the EV and aiming spot
			EV = convolve2d(Zs,Zn,mode="same",fillvalue=V[s])
			# EV = fftconvolve(Zs,Zn,mode="same")
			'''

			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# Compute EVs - Code from Thomas Miller (Dr. Archibald's student)
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			# '''
			if s <= 61:
				score_change = singleProbs[:,:s-1] @ np.flip(V[2:s+1])
				bust = np.sum(singleProbs[:,s-1:],axis=1) * V[s]
				
				doub_change = doubleProbs[:,:s-1] @ np.flip(V[2:s+1])
				doub_bust = (np.sum(doubleProbs[:,s+1:],axis=1) + doubleProbs[:,s-1]) * V[s]

				EV = score_change+bust+doub_change+doub_bust
				
			else:
				score_change = (singleProbs + doubleProbs) @ np.flip(V[s-60:s+1])
				EV = score_change
			# '''
			
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


			# Get max EV
			bestEV = np.max(EV)	


			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# Find best action
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			# PREV - with convolution
			# mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
			# action = [Xn[mxi],Yn[myi]]

			# Nows
			mi = np.unravel_index(EV.argmax(), EV.shape)[0]
			action = [actions[mi][0],actions[mi][1]]


			# if s >= 59 and s <= 65:
				# code.interact(f"State {s}: ", local=dict(globals(), **locals()))


			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			

			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# Save info
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

			#if s not in allEVs:
			#	allEVs[s] = []
			#allEVs[s].append(EV)

			allEVs[s] = EV
			PI_EV[s] = float(bestEV)
			PI[s] = action
		
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			

			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# Update info
			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			
			# How much are we going to change the value?
			currentDelta = abs(V[s] + 1 - gamma*bestEV)

			if currentDelta > delta:
				delta = currentDelta						
			
			# Update value of state with highest value
			# Value of state = direct reward of action + expected value of next states
			V[s] = -1 + gamma*bestEV

			# code.interact(f"after iter {iterations}...", local=dict(globals(), **locals()))


			# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		# print("Current Delta: ", delta)
		#code.interact(f"after iter {iterations}...", local=dict(globals(), **locals()))
		
		iterations += 1

	# code.interact(f"after iter {iterations}...", local=dict(globals(), **locals()))

	# code.interact(f"done with value iter {iterations}...", local=dict(globals(), **locals()))
	return V, PI, PI_EV, allEVs


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Code from: Thomas Miller (Dr. Archibald's student)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# VECTORIZED
def precalc(domain,mus,var,sample_size=100000):

	N = scipy.stats.multivariate_normal([0,0],var)

	
	# Generate noises (sample from noise model)
	ps = N.rvs(size=sample_size)
	
	# To store probs
	# (actions, possible scores(60)) 
	non_doubs = np.zeros((mus.shape[0],61))
	doub_a = np.zeros_like(non_doubs)
	# non_doubs = np.zeros((mus.shape[0],61),dtype=np.longdouble)
	# doub_a = np.zeros_like(non_doubs,dtype=np.longdouble)
	

	# Monte-Carlo sampling

	# For a given target action
	for i,mu in enumerate(mus):
		
		# Add noises to target action 
		# p = an array that contains target action shifted by the different noises
		p = ps+mu

		# Find respective score for each noisy action
		ss,doubs = domain.npscore(p[:,0],p[:,1],return_doub=True)
		
		# Find scores of actions that don't result in doubles 
		# Indices not repeated since using np.unique
		# nonc = counts for each unique score
		nond, nonc = np.unique(ss[~doubs],return_counts=True)
		# x = np.sum(nonc)
		
		# Find scores of actions that result in doubles 
		# Indices not repeated since using np.unique
		# nonc = counts for each unique score
		doub,c = np.unique(ss[doubs],return_counts=True)
		# y = np.sum(c)

		# Save probs
		non_doubs[i,nond] = nonc/float(sample_size)
		doub_a[i,doub] = c/float(sample_size)
		
	
	########################################
	# non_doubs *= np.square(resolution)
	# doub_a *= np.square(resolution)
	########################################
	
	# code.interact("precalc()...", local=dict(globals(), **locals()))
		
	return non_doubs, doub_a

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def score(x,y):
	""" 
	Return the score for location (x,y) on the dartboard. 
	Units are mm    
	"""
	
	#First convert to polar coordinates to get distance from (0,0) and angle from (0,0)
	a = math.atan2(y,x) #angle
	r = math.hypot(x,y) #radius

	scaling = 1.0

	double = False

	#Figure out which distance we fall in 
	if r < 6.35:
		#Double bullseye = 50 points
		return 50.0, True
	if r < 15.9: 
		#Single bullseye = 25 points
		return 25.0, False

	if r > 99 and r < 107:
		# Triple score
		scaling = 3.0

	if r > 162 and r < 170:
		# Double score
		double = True
		scaling = 2.0

	if r > 170:
		# Off the board
		return 0.0, double

	#Figure out which slice we fall in
	angle_increment = math.pi / 10.0
	slice_low = - math.pi - angle_increment / 2.0
	slice_high = slice_low + angle_increment


	slices = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]


	for i in range(21):
		if a > slice_low and a < slice_high:
			return scaling*slices[i], double
		slice_low += angle_increment
		slice_high += angle_increment

	#Check for 11 slice

	#Must have missed the board!
	return 0.0, double


# Get all of the possible actions in domain
def getActions():

	# actions [0] = center of board (inner bull)
	# actions [1] = just 1 target fot the  outer bull 
	#				since same region all around
	actions = [(0,0)]#,(0,10)]

	# Distances are inner-single, triple, outer-single, double
	distances = [66,103,135,166]

	angle_increment = math.pi/10.0
	angle = 0

	for wedge in range(20):
		for d in distances:
			# Get (x,y) location for this target)
			x = d*math.cos(angle)
			y = d*math.sin(angle)
			actions.append((x,y))
		angle += angle_increment

	# print("Number of actions: ", len(actions))
	# code.interact("getActions()...", local=dict(globals(), **locals()))

	return actions


def npscore(x,y,return_doub=False):

	slices = np.array([11,8,8,16,16,7,7,19,19,3,3,17,17,2,2,15,15,10,10,6,6,13,13,4,4,18,18,1,1,20,20,5,5,12,12,9,9,14,14,11,11])

	a = np.arctan2(y,x)
	r = np.hypot(x,y)

	a = ((a + math.pi) * (40/(2*math.pi))).astype(int)
	ans = slices[a]
	
	doubb = r<6.35
	bulls = r<15.9
	on = r<170
	
	trip = (r>99) & (r<107)
	doub = (r<170) & (r>162)

	ans *= np.invert(bulls)*on 
	ans += (25*bulls) + (25*doubb)
	ans += (ans*trip*2) + (ans*doub)
	
	if return_doub:
		return ans, doub+doubb
	return ans 


def getValuesAlt(resolution,curScore,values,actions):

	V = []
	
	actions = np.array(actions)	

	S,D = npscore(actions[:,0],actions[:,1],return_doub=True)

	for a in range(len(actions)):
			
			newScore = curScore - S[a]

			#Did we bust (score too much)?
			# Less than 0 or exactly 1
			if newScore < 0 or newScore == 1:
				newScore = curScore
			
			#Did we double out correctly?
			if newScore == 0:
				if not D[a]:
					newScore = curScore

			V.append(values[int(newScore)])
	
	return V

def getValuesGivenTargets(resolution,curScore,values,X,Y):

	# gridX, gridY = np.meshgrid(X,Y,indexing="ij")
	# S,D = npscore(gridX,gridY,return_doub=True)

	V = np.zeros((len(X), len(Y)))

	# '''
	V = []

	for i in range(len(X)):

		tempV = []

		for j in range(len(Y)):
			
			# newScore = curScore - S[i,j]
			score, double = npscore(X[i],Y[j],True)
			newScore = curScore - score

			#Did we bust (score too much)?
			# Less than 0 or exactly 1
			if newScore < 0 or newScore == 1:
				newScore = curScore
			
			#Did we double out correctly?
			if newScore == 0:
				if not double:
					newScore = curScore

			tempV.append(values[int(newScore)])

		V.append(tempV)
	# '''

	'''
	V = []

	for a in zip(X,Y):

		# newScore = curScore - S[i,j]
		score, double = npscore(a[0],a[1],True)
		newScore = curScore - score

		#Did we bust (score too much)?
		# Less than 0 or exactly 1
		if newScore < 0 or newScore == 1:
			newScore = curScore
		
		#Did we double out correctly?
		if newScore == 0:
			if not double:
				newScore = curScore

		V.append(values[int(newScore)])
	'''

	# code.interact("getValuesGivenTargets()...", local=dict(globals(), **locals()))
	return X,Y,np.array(V)


def get_values(XS,resolution,curScore,values):

	#X = np.arange(-340.0,341.0,resolution)
	#Y = np.arange(-340.0,341.0,resolution)

	X = np.arange(-170.0,171.0,resolution)
	Y = np.arange(-170.0,171.0,resolution)

	# boundary = 4*np.ceil(XS)

	# X = np.arange(-boundary,boundary+resolution,resolution)
	# Y = np.arange(-boundary,boundary+resolution,resolution)

	V = np.zeros((len(X),len(Y)))
	
	# gridX, gridY = np.meshgrid(X,Y,indexing="ij")
	# S,D = npscore(gridX,gridY,return_doub=True)

	for i in range(len(X)):
		for j in range(len(Y)):
			
			# s, double = score(X[i],Y[j])
			s, double = npscore(X[i],Y[j],return_doub=True)
			
			# newScore = curScore - S[i,j]
			newScore = curScore - s

			#Did we bust (score too much)?
			# Less than 0 or exactly 1
			if newScore < 0 or newScore == 1:
				newScore = curScore
			
			#Did we double out correctly?
			if newScore == 0:
				# if not D[i,j]:
				if not double:
					newScore = curScore

			V[i,j] = values[int(newScore)]
	

	# code.interact("get_values()...", local=dict(globals(), **locals()))
	return X,Y,V


def getNoiseModel(X):
	N = scipy.stats.multivariate_normal(mean=[0.0,0.0],cov=X)
	return N


def getSymmetricNormalDistributionAlt(XS,resolution):

	# XS it's the standard deviation (not squared yet)

	actions = np.array(getActions())


	X = actions[:,0]
	Y = actions[:,1]

	# D = np.zeros((len(X),len(Y)))
	D = []

	# XS**2 to get variance
	N = getNoiseModel(XS**2)
	
	for i, a in enumerate(actions):
		D.append(N.pdf([a[0],a[1]]))

	# for xi in X:
	# 	tempD = []
		
	# 	for yi in Y:
	# 		tempD.append(N.pdf([xi,yi]))

	# 	D.append(tempD)

	# X2, Y2 = np.meshgrid(X,Y,indexing="ij")


	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  
	D = np.array(D)
	D *= np.square(resolution)


	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# # ax.plot_surface(X,Y,D,cmap='viridis',linewidth=0)
	# ax.plot(X,Y,D)
	# ax.set_xlabel('X axis')
	# ax.set_ylabel('Y axis')
	# # ax.set_zlabel('Z axis')
	# plt.show()


	code.interact("getSymmetricNormalDistributionAlt()...", local=dict(globals(), **locals()))

	return X,Y,D


def getSymmetricNormalDistribution(XS,resolution):

	# XS it's the standard deviation (not squared yet)


	# DEFAULT SET OF TARGETS	
	# From -340 to 341 in order to consider targets outside of the darts board as well
	# If doing only -170 to 171, we will miss the probabilities of the targets being outside of the board
	defaultX1 = np.arange(-170.0,171.0,resolution)
	defaultY1 = np.arange(-170.0,171.0,resolution)

	# defaultX2 = np.arange(-340.0,341.0,resolution)
	# defaultY2 = np.arange(-340.0,341.0,resolution)
	
	# defaultX1 = defaultX2
	# defaultY1 = defaultY2


	# ACCOUNT FOR POSSIBILITY OF BAD XSKILL AGENTS
	# Determine what 4 standard deviations is
	# when the standard deviation gets too big, 
	# it will fail to capture all of the probability mass.
	# Hence, adapting range of targets
	# boundary = 4*np.ceil(XS)

	# otherX = np.arange(-boundary,boundary+resolution,resolution)
	# otherY = np.arange(-boundary,boundary+resolution,resolution)
	

	# Merge possible set of targets, ensure not repeated
	# defaultX1 = np.unique(np.concatenate((defaultX,otherX)))
	# defaultY1 = np.unique(np.concatenate((defaultY,otherY)))

	# X = defaultX 
	# Y = defaultY 

	# X = otherX 
	# Y = otherY 

	'''
	# Determine which set of targets to use
	X = None
	Y = None

	# If more targets present on other set
	# Use those and not the default set
	if len(otherY) > len(defaultX):
		X = otherX
		Y = otherY
	else:
		X = defaultX
		Y = defaultY
	'''


	D1 = np.zeros((len(defaultX1),len(defaultY1)))
	# D2 = np.zeros((len(defaultX2),len(defaultY2)))

	# XS**2 to get variance
	N = getNoiseModel(XS**2)
	
	for i in range(len(defaultX1)):
		for j in range(len(defaultY1)):
			D1[i,j] = N.pdf([defaultX1[i],defaultY1[j]])

	'''
	for i in range(len(defaultX2)):
		for j in range(len(defaultY2)):
			D2[i,j] = N.pdf([defaultX2[i],defaultY2[j]])
	'''

	
	# code.interact("b...", local=dict(globals(), **locals()))

	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  
	D1 *= np.square(resolution)
	# D2 *= np.square(resolution)


	'''
	X1, Y1 = np.meshgrid(defaultX1,defaultY1,indexing="ij")
	plt.scatter(X1,Y1,c = D1,cmap='viridis',linewidth=0)
	plt.show()
	'''

	# code.interact("a...", local=dict(globals(), **locals()))

	return defaultX1,defaultY1,D1
	#return defaultX2,defaultY2,D2


def simulateGames(domain,convInfo,values,N,startScore,xSkill,resolution,counts,totalTurns,busted,info):

	fullGameThrows = 0
	limitedGames = 0.0


	# Run N games
	for n in range(N):

		throws = 0
		S = startScore
		scoreSequence = [startScore]

		executedActions = []
		scores = []
		
		while S > 0:

			# Now throw at that aiming point
			# xa,ya = np.random.multivariate_normal((convInfo[S]["mx"],convInfo[S]["my"]),(xSkill**2)*np.identity(2))

			N = scipy.stats.multivariate_normal(mean=[0.0,0.0],cov=xSkill**2)
			
			# Generate noises (sample from noise model)
			ps = N.rvs(size=1)


			xa = convInfo[S]["mx"]+ps[0]
			ya = convInfo[S]["my"]+ps[1]

			executedActions.append([xa,ya])

			# curScore2, double2 = score(xa,ya)
			curScore,double = npscore(xa,ya,return_doub=True)

			scores.append([curScore,double])


			nextScore = S - curScore

			# Did we bust (score too much)?
			if nextScore < 0 or nextScore == 1:
				busted[startScore] += 1.0
				nextScore = S 

			#Did we double out correctly?
			if nextScore == 0:
				if not double:
					nextScore = S

			S = nextScore

			scoreSequence.append(S)
			throws += 1 

			
			info[startScore][S] += 1.0


			# To limit possible number of turns as agents with bad xskill can
			# take lots of turns to finish an actual game
			if throws >= 500:
				limitedGames += 1
				break

		print('Done with game', n, ' : (', throws, ') - ', scoreSequence)
		# code.interact("...", local=dict(globals(), **locals()))
		
		fullGameThrows += throws


		# [201, 184, 124, 64, 50, 0]
		# Figure out the scores for all states
		T = 1.0
		for i in range(len(scoreSequence)-2,-1,-1):
			counts[int(scoreSequence[i])] += 1.0 
			totalTurns[int(scoreSequence[i])] += T
			T += 1.0
		# code.interact("...", local=dict(globals(), **locals()))


	return fullGameThrows,limitedGames


def validateValueFunction(domain,convInfo,values,xSkill,startScore,N,resolution,toSave):
	
	counts = [0.0]*(startScore+1)
	totalTurns = [0.0]*(startScore+1)
	busted = [0.0]*(startScore+1) 

	totalNumGames = 0.0


	valuesMC = [0.0]*(startScore+1)

	# [from][to]
	info = np.zeros((startScore+1,startScore+1))


	print(f"Simulating games...")	

	# Simulate full game
	fullGameThrows, limitedGames = simulateGames(domain,convInfo,values,N,startScore,xSkill,resolution,counts,totalTurns,busted,info)
	totalNumGames += N

	# Verify if seen enough samples (number of turns) for the different states/scores
	# Start at state 201 and stop at state 2 (inclusive)
	for s in range(startScore,1,-1):

		# Do at least 1 pass from current start state
		# To have info for probs (info[stsrtState][s])
		fullGameThrowsTemp,limitedGamesTemp = simulateGames(domain,convInfo,values,N,s,xSkill,resolution,counts,totalTurns,busted,info)
		fullGameThrows += fullGameThrowsTemp
		limitedGames += limitedGamesTemp
		totalNumGames += N

		# If saw fewer than X samples for a given state,
		# simulate more games (starting from given state)
		while counts[s] < N:
			print(f"Simulating games again for state {s} since counts = {counts[s]} | totalTurns = {totalTurns[s]}")
			fullGameThrowsTemp,limitedGamesTemp = simulateGames(domain,convInfo,values,N,s,xSkill,resolution,counts,totalTurns,busted,info)
			fullGameThrows += fullGameThrowsTemp
			limitedGames += limitedGamesTemp
			totalNumGames += N

	print('Done with validation.')
	print('\nValue of state ', len(values)-1, '=', values[-1])
	print('Average number of turns =', fullGameThrows/N)


	for s in range(2,startScore+1):
		if counts[s] != 0:
			valuesMC[s] = totalTurns[s]/counts[s]


	toSaveUpd = f"{toSave}N{N}{os.sep}xskill{xSkill}{os.sep}"

	if not os.path.exists(toSaveUpd):
		os.mkdir(toSaveUpd)


	outfile = open(f"{toSaveUpd}valueComparison.txt","w")
	
	print("FULL VALUE COMPARISON: ")
	for s in range(2,startScore+1):
		if counts[s] != 0:
			print(f"State: {s} | Value: {values[s]} | MC-V: {valuesMC[s]}\n\tCounts: {counts[s]} | TotalTurns: {totalTurns[s]} \n\n")
			outfile.write(f"State: {s} | Value: {values[s]} | MC-V: {valuesMC[s]}\n\tCounts: {counts[s]} | TotalTurns: {totalTurns[s]} \n\n")
			

	outfile.write(f"\n\nTotal # of games: {totalNumGames}")
	outfile.write(f"\n\nLimited games: {limitedGames}")
	outfile.close()

	probs = []

	for s in range(2,startScore+1):
		probs.append(info[s]/np.sum(info[s]))


	fig, ax = plt.subplots(1,1)

	cbar = ax.pcolor(probs)
	fig.colorbar(cbar,ax=ax)

	# The grid orientation follows the standard matrix convention:
	# An array C with shape (nrows, ncolumns) is plotted with 
	# the column number as X and the row number as Y.
	ax.set_xlabel("TO")
	ax.set_ylabel("FROM")

	fig.tight_layout()
	plt.savefig(f"{toSaveUpd}probsFromTo.png")

	probs = np.around(probs,4)


	# Row -> From state 2 - 201
	# Column -> To state 0 - 201
	np.savetxt(f"{toSaveUpd}probsFromTo.csv", probs, fmt="%.2f", delimiter=",", header = str(list(range(startScore+1))))

	busted = np.array(busted)
	busted = np.reshape(busted,(1,len(busted)))

	np.savetxt(f"{toSaveUpd}busted.csv", busted, fmt="%.2f", delimiter=",", header = str(list(range(startScore+1))))

	# code.interact("...", local=dict(globals(), **locals()))

	return limitedGames


if __name__ == '__main__':

	np.random.seed(0)


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Testing value iter vs expected rewards')
	parser.add_argument("-delta", dest = "delta", help = "Delta = resolution to use when doing the convolution", type = float, default = 1e-2)
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "sequentialDarts")
	parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")

	args = parser.parse_args()

	toSave = f"ComparingValuesAndSampleRewards{os.path.sep}"

	# Create folder if not present
	if not os.path.exists(toSave):
		os.mkdir(toSave)

	# Create folder if not present
	if not os.path.exists(f"{toSave}PlotsEVs{os.sep}"):
		os.mkdir(f"{toSave}PlotsEVs{os.sep}")


	# Find location of current file
	scriptPath = os.path.realpath(__file__)

	# Find path of "skill-estimation" folder and add "Domains" folder to find such path 
	# To be used later for finding and properly loading the domains 
	# Will look something like: "/home/archibald/skill-estimation/Domains/"
	mainFolderName = scriptPath.split("Testing")[0]


	domainModule = imp.load_source("sequential_darts",mainFolderName+"Environments"+os.path.sep+"Darts"+ os.path.sep + "SequentialDarts" + os.path.sep + "sequential_darts.py")
	spacesModule = imp.load_source("spaces",mainFolderName + "setupSpaces.py")

	args.N = 1
	args.delta = 5.0

	# Assuming normal mode and same dartboard state for now
	boardStates = domainModule.getBoardStates(1,"normal")[0] 


	numObservations = 1

	xskills = np.linspace(2.5,150.5,num = 33)
	# xskills = np.linspace(150.5,160,num = 1)
	# xskills = [2.5,39.5,76.5,113.5,150.5]


	N = 1000 #10_000
	valueIterFolder = ".."+os.path.sep+"Spaces" + os.path.sep + "ValueFunctions" + os.path.sep
	
	mainFolder = ".."+os.path.sep+"Spaces" + os.path.sep + "ExpectedRewards" + os.path.sep
	fileName = f"ExpectedRewards-{args.domain}-N{N}"
	expectedRFolder = mainFolder + fileName

	# Make spaces for initial xskills
	# spaces = spacesModule.SpacesSequentialDarts(numObservations,domainModule,args.mode,args.delta,N,expectedRFolder=expectedRFolder,valueIterFolder=valueIterFolder)
	# spaces.updateSpace(xskills,boardStates)


	startScore = domainModule.getPlayerStartScore()
	states = list(range(startScore+1))


	# Create folder if not present
	if not os.path.exists(toSave+f"N{N}"):
		os.mkdir(toSave+f"N{N}")


	slicesScores = [11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]

	limitedGamesInfo = []

	MAX_EV = {}


	for x in xskills: 

		print(f"\n\nxSkill: {x}")

		# space = spaces.spacesPerXskill[x]


		# PERFORM VALUE ITERATION HERE
		allTargets = []

		defaultX = np.arange(-170.0,171.0,args.delta)
		defaultY = np.arange(-170.0,171.0,args.delta)

		for xi in defaultX:
			for yi in defaultY:
				allTargets.append((xi,yi))
		actions = np.array(allTargets)

		
		V, PI, PI_EV, allEVs = valueIteration(domainModule,x,actions,args.delta)
		


		convInfo = {}

		# print("Doing convolution for different states...")

		###Xn,Yn,Zn = getSymmetricNormalDistribution(x,args.delta)
		# Xn,Yn,Zn = getSymmetricNormalDistributionAlt(x,args.delta)

		###X2,Y2 = np.meshgrid(Xn,Yn,indexing="ij")


		'''
		fig,ax = plt.subplots()
		cmap = plt.get_cmap("viridis")
		norm = plt.Normalize(np.min(Zn),np.max(Zn))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])
		cbar = fig.colorbar(sm,ax=ax)
		ax.scatter(X2,Y2,c=cmap(norm(Zn.flatten())))
		domainModule.drawBoard(ax,slicesScores)
		plt.savefig(f"{toSave}{os.sep}PlotsEVs{os.sep}xskill{x}{os.sep}probs.png")
		plt.clf()
		plt.close()	
		'''	


		# Create folder if not present
		if not os.path.exists(f"{toSave}PlotsEVs{os.sep}xskill{x}{os.sep}"):
			os.mkdir(f"{toSave}PlotsEVs{os.sep}xskill{x}{os.sep}")


		# cmap = plt.get_cmap("viridis")
		# norm = plt.Normalize(np.min(space.V),np.max(space.V))
		# sm = ScalarMappable(norm=norm,cmap=cmap)


		MAX_EV[x] = []

		for state in range(2,startScore+1):

			convInfo[state] = {}

			# value = space.V[state]
			value = V[state]

			'''

			### Xs,Ys,Zs = get_values(x,args.delta,state,space.V)
			# Xs,Ys,Zs = getValuesGivenTargets(args.delta,state,V,Xn,Yn)
			# Zs = getValuesAlt(args.delta,state,space.V,actions)
			
			Xs,Ys,Zs = get_values(x,args.delta,state,V)

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,Zn,mode="same",fillvalue=value)
			# EVs = fftconvolve(Zs,Zn,mode="same")

			convInfo[state]["EVs"] = EVs
			MAX_EV[x].append(np.max(EVs))


			# Get maximum of EV
			mxi, myi = np.unravel_index(EVs.argmax(), EVs.shape)

			# Best aiming point
			mx = Xn[mxi]
			my = Yn[myi]

			# convInfo[state]["mx"] = mx
			# convInfo[state]["my"] = my

			'''

			convInfo[state]["mx"] = PI[state][0]
			convInfo[state]["my"] = PI[state][1]


			'''
			fig,ax = plt.subplots()

			cmap = plt.get_cmap("viridis")
			norm = plt.Normalize(np.min(allEVs[state]),np.max(allEVs[state]))
			sm = ScalarMappable(norm=norm,cmap=cmap)
			sm.set_array([])
			cbar = fig.colorbar(sm,ax=ax)

			ax.scatter(actions[:,0],actions[:,1],c=cmap(norm(allEVs[state])))
			ax.scatter(convInfo[state]["mx"],convInfo[state]["my"],color=cmap(norm(np.max(allEVs[state]))),marker="X",s=60,edgecolors="black")
			
			domainModule.drawBoard(ax,slicesScores)
			plt.savefig(f"{toSave}{os.sep}PlotsEVs{os.sep}xskill{x}{os.sep}state{state}.png")

			plt.clf()
			plt.close()	
			'''

		#code.interact("after conv...", local=dict(globals(), **locals()))


		print(f"Validation in progress...")
		# validateValueFunction(domainModule,convInfo,space.V,x,startScore,N,args.delta,toSave)
		limitedGames = validateValueFunction(domainModule,convInfo,V,x,startScore,N,args.delta,toSave)
		# code.interact("...", local=dict(globals(), **locals()))
		
		limitedGamesInfo.append(limitedGames)		

		#for s in states[2:]:
			#avgTurns = getAvgTurns(space,domainModule,x,s,N)
			# action = space.PI[s]


	fig, ax = plt.subplots(1,1)
	plt.scatter(xskills,limitedGamesInfo)
	ax.set_xlabel("Xskills")
	ax.set_ylabel("Limited Games")

	fig.tight_layout()
	plt.savefig(f"{toSave+f'N{N}'+os.sep}limitedGamesInfo.png")


