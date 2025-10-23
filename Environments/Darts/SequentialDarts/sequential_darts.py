import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable

import numpy as np
import math
import code

from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve

def get_domain_name():
	return "sequentialDarts"

def getPlayerStartScore():
	return 201 # GAME 201 DARTS


def labelRegions(slices):

	angle_increment = math.pi / 10.0    
	angle = math.pi
	r = 130.0

	for i in range(20):
		x = r*math.cos(angle)
		y = r*math.sin(angle)
		plt.text(x,y,str(slices[i]),fontsize=12,horizontalalignment='center')
		angle += angle_increment

def drawBoard(ax,slices):

	#Draw the bullseye rings and scoring rings
	radii = [6.35, 15.9, 99, 107, 162, 170]
	for r in radii: 
		circle = Circle((0,0),r,fill=False)
		ax.add_artist(circle)

	#Do the radii 
	start_d = 15.9
	end_d = 170.0
	angle_increment = math.pi / 10.0
	angle = -angle_increment / 2.0

	for i in range(20):
		sx = start_d * math.cos(angle)
		sy = start_d * math.sin(angle)
		dx = end_d * math.cos(angle)
		dy = end_d * math.sin(angle)
		plt.plot([sx, dx], [sy, dy], color="Black")
		# print 'Angle = ', 180.0*angle/math.pi
		angle += angle_increment
	labelRegions(slices)


def getBoardStates(N,mode="normal"):

	boardStates = []

	# For each state
	for n in range(N):

		slices = []

		if mode == "normal":
			# Including 25 - always in the first position
			slices = [25,11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]
		
		elif mode == "rand_pos":
			# Including 25 - can end up anywhere in the list
			slices = [25,11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]
			
			# Change the order of the values - (order will be different every time)
			np.random.shuffle(slices)
		
		elif mode == "rand_v":

			for i in range(21):

				# Get a random number between 1 and 20 to use as value
				# randint low = inclusive, high = exclusive (hence up to 21)
				randV = np.random.randint(1,21)
				
				# Save on list
				slices.append(randV)

			# Add 25
			slices.append(25)

			# Change the order of the values - (order will be different every time)
			# To ensure 50 & 25 are not always at the end
			np.random.shuffle(slices)


		boardStates.append(slices)

	return boardStates

def score(slices,action):
	""" 
	Return the score for location (x,y) on the dartboard.
	Units are mm.
	"""
	
	x = action[0]
	y = action[1]

	#First convert to polar coordinates to get distance from (0,0) and angle from (0,0)
	a = np.arctan2(y,x) #angle
	r = np.hypot(x,y) #radius

	# print(f"radius: {r} | angle: {a}")

	# print(f"action: {action}\nradius: {r} | angle: {a}")

	scaling = 1.0

	double = False

	#Figure out which distance we fall in 

	# If on the double bullseye region
	if r < 6.35:
		# 2*Slices[0] will be 50 if on normal mode and any other value between 1-20 or 25 if on other modes
		return 2 * slices[0], double

	# If on the single bullseye region
	if r < 15.9: 
		# Slices[0] will be 25 if on normal mode and any other value between 1-20 or 25 if on other modes
		return slices[0], double


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

	# Figure out which slice we fall in
	angle_increment = math.pi / 10.0
	slice_low = - math.pi - angle_increment / 2.0
	slice_high = slice_low + angle_increment

	# Rest (21) of the slices
	for i in range(1,22):
		if a > slice_low and a < slice_high:
			return scaling*slices[i], double 
		slice_low += angle_increment
		slice_high += angle_increment

	# Check for 11 slice

	# Must have missed the board!
	return 0.0, double


def draw_noise_sample(rng,X):

	# X is squared already (x**2 = variance)
	
	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy

	N = multivariate_normal(mean=[0.0,0.0],cov=X,seed=seed)
	
	return N


def getSymmetricNormalDistribution(XS,resolution):

	# XS it's the standard deviation (not squared yet)

	# DEFAULT SET OF TARGETS	
	# From -340 to 341 in order to consider targets outside of the darts board as well
	# If doing only -170 to 171, we will miss the probabilities of the targets being outside of the board
	defaultX = np.arange(-170.0,171.0,resolution)
	defaultY = np.arange(-170.0,171.0,resolution)

	# defaultX = np.arange(-340.0,341.0,resolution)
	# defaultY = np.arange(-340.0,341.0,resolution)


	# ACCOUNT FOR POSSIBILITY OF BAD XSKILL AGENTS
	# Determine what 4 standard deviations is
	# when the standard deviation gets too big, 
	# it will fail to capture all of the probability mass.
	# Hence, adapting range of targets
	# boundary = 4*np.ceil(XS)

	# otherX = np.arange(-boundary,boundary+resolution,resolution)
	# otherY = np.arange(-boundary,boundary+resolution,resolution)
	

	X = defaultX 
	Y = defaultY 

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


	D = np.zeros((len(X),len(Y)))

	# XS**2 to get variance
	N = draw_noise_sample(XS**2)
	
	for i in range(len(X)):
		for j in range(len(Y)):
			D[i,j] = N.pdf([X[i],Y[j]])

	
	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  
	D *= np.square(resolution)
	
	# code.interact("...", local=dict(globals(), **locals()))

	return X,Y,D


def getScores(slices,resolution):
   
	X = np.arange(-170.0, 171.0, resolution)
	Y = np.arange(-170.0, 171.0, resolution)

	S = np.zeros((len(X), len(Y)))

	for i in range(len(X)):
		for j in range(len(Y)):
			action = [X[i],Y[j]]
			S[i,j], double = score(slices,action)

	return X,Y,S

def getValues(slices,XS,resolution,curScore,values):

	# X = np.arange(-340.0, 341.0, resolution)
	# Y = np.arange(-340.0, 341.0, resolution)

	X = np.arange(-170.0, 171.0, resolution)
	Y = np.arange(-170.0, 171.0, resolution)

	# boundary = 4*np.ceil(XS)

	# X = np.arange(-boundary,boundary+resolution,resolution)
	# Y = np.arange(-boundary,boundary+resolution,resolution)


	V = np.zeros((len(X), len(Y)))

	
	# gridX, gridY = np.meshgrid(X,Y,indexing="ij")
	# S,D = npscore(gridX,gridY,return_doub=True)

	for i in range(len(X)):
		for j in range(len(Y)):

			# action = [X[i],Y[j]]

			# s, double = score(slices,action)
			s,double = npscore(X[i],Y[j],return_doub=True)

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

	return X,Y,V


def getInfoOnBoardOnly(Xs,Ys,EV):

	all_vs_copy = np.copy(EV)

	# find targets/EVs that fall on dartboard

	N = len(Xs)

	all_x, all_y = np.meshgrid(Xs, Ys)

	all_x.shape = (N*N,)
	all_y.shape = (N*N,)

	points = (all_x, all_y)

	# For testing - Make plot to verify positions on board
	'''
	fig = plt.figure()
	ax = plt.gca()
	'''

	# to keep track of which position is on the board - will have a 1 if in it
	onBoard = np.zeros((N,N))
	#onBoardEVs = np.zeros((N,N))

	#for each one of the possible targets - dense
	for x,y in zip(points[0], points[1]):
		# convert from cartesian (X,Y) to polar (R, thetha)
		a = np.arctan2(y,x) #angle
		r = np.hypot(x,y) #radius

		xi = np.where(Xs == x)[0][0]
		yi = np.where(Ys == y)[0][0]

		# if within board, update status
		if r <= 170.0:
			onBoard[xi][yi] = 1.0
		#    onBoardEVs[xi][yi] = all_vs_copy[xi][yi]
		#else:
		#   onBoardEVs[xi][yi] = np.nan


	indexPositionsOnBoard = np.where(onBoard == 1.0)

	# Get actual positions/targets on board
	positionsOnBoardX = Xs[indexPositionsOnBoard[0]]
	positionsOnBoardY = Ys[indexPositionsOnBoard[1]]

	# subset EVs - to only EVs corresponding to the targets within board
	evsPositionsOnBoard = all_vs_copy[indexPositionsOnBoard]

	
	# For testing - Make plot to verify positions on board
	''' 
	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(min(evsPositionsOnBoard),max(evsPositionsOnBoard))
	plt.scatter(positionsOnBoardX,positionsOnBoardY, c = cmap(norm(evsPositionsOnBoard)))
	sm = ScalarMappable(norm = norm, cmap = cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm)
	cbar.ax.set_title("EVs")
	plt.title("Positions on board")
	draw_board(ax)
	plt.show()
	'''

	#code.interact("getInfoOnBoardOnly: ", local=dict(globals(), **locals())) 


	return positionsOnBoardX, positionsOnBoardY, evsPositionsOnBoard

def convolveEV(slices,X,resolution,returnZn = False):

	Xn,Yn,Zn = getSymmetricNormalDistribution(X,resolution)

	Xs,Ys,Zs = getScores(slices,resolution)

	# Convolve to produce the EV and aiming spot
	# Output array will have the shape of the first input (Zs)
	# Zn = -340 to 341 | Zs = -170 to 171 | ("zoom in")
	EV = fftconvolve(Zs,Zn,mode="same")

	# Restricting the values of the EVs to be 0 or greater
	# This is done because, for small xskills (1.0 for example), the EVs are very small and some are negative
	# Since we know that all EVs should be positive (non-negative), we can limit their values to be 0.0 or greater
	# Update: After doing the scaling up of pdfs by resolution^2 this may no longer be needed but still leaving it here just in case. 
	EV = np.clip(EV,0.0,None)


	# Return only info about targets & EVs that are actually within the board
	Xs,Ys,EVs = getInfoOnBoardOnly(Xs,Ys,EV)

	#code.interact("compute_expected_value_curve(): ", local=dict(globals(), **locals())) 


	# Returns pdfs as well, used on script for testing
	if returnZn:
		return Xs, Ys, EVs, Zn
	# Returns targets & EVs
	else:
		return Xs, Ys, EVs

		
def sampleAction(XS,action,noiseModel=None):

	# If noise model was not given, proceed to get it
	if noiseModel == None:
		N = draw_noise_sample(XS**2)
	# Otherwise, use given noise model
	else:
		N = noiseModel

	# Get noise (sample)
	noise = N.rvs()

	# Add noise to planned action (This creates the noisy action)
	na = [action[0] + noise[0], action[1] + noise[1]]

	# code.interact("sample_noisy_action", local=locals())

	return na

def calculate_wrapped_action_difference(action1, action2):

	x1 = action1[0]
	y1 = action1[1]

	x2 = action2[0]
	y2 = action2[1]

	# Formula obtained from: https://www.ck12.org/book/CK-12-Trigonometry-Concepts/section/6.2/

	# Convert action 1 (x1 ,y1) to polar coordinates
	a1 = np.arctan2(y1,x1) # angle in radians
	r1 = np.hypot(x1,y1) # radius

	# Convert action 2 (x2 ,y2) to polar coordinates
	a2 = np.arctan2(y2,x2) # angle in radians
	r2 = np.hypot(x2,y2) # radius

	# cos func takes as input radians
	d = np.sqrt(np.power(r1,2.0) + np.power(r2,2.0) - ((2.0 * r1 * r2 )* np.cos(a2-a1)))

	# code.interact(local=locals())

	return d


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Code from Thomas Miller (Dr. Archibald's student)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
	return actions

# VECTORIZED
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

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if __name__ == '__main__':
	

	slices = np.array([11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11])	
	print("Slices: ", slices)

	'''
	x = 2.5
	action = [0.0, 105.0]
	s = score(slices,action)
	print(f"xskill: {x} action: {action} | score: {s}\n")
	'''



	'''
	x = 150.5
	action = [0.0, 0.0]
	s = score(slices,action)
	print(f"xskill: {x} action: {action} | score: {s}\n")
	'''


	'''
	resolution = 5.0
	s = 201
	V = 
	Xz,Yz,Zs = getValues(slices,resolution,s,V)
	'''


	print("Displaying dartboard . . .\n")
	
	plt.figure()
	ax = plt.gca()
	drawBoard(ax,slices)   
	plt.axis('equal')

	# actions = np.array(getActions())
	# plt.scatter(actions[:,0],actions[:,1])

	ax.set_xticks([])
	ax.set_yticks([])

	plt.show()

	# code.interact("...", local=dict(globals(), **locals()))