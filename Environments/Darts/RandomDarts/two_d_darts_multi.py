import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable

import numpy as np
import code

from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve
from itertools import product
from math import dist, cos, sin, pi

from time import perf_counter

def get_domain_name():
	return "2d-multi"

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

def draw_noise_sample(rng,mean,covMatrix):
	
	# Need to use rng.bit_generator._seed_seq.entropy instead of just rng to ensure same noises produced each time for given params 
	if type(rng.bit_generator._seed_seq.entropy) == np.ndarray:
		seed = rng.bit_generator._seed_seq.entropy[0]
	else:
		seed = rng.bit_generator._seed_seq.entropy

	# print(seed)
	N = multivariate_normal(mean,covMatrix,seed=seed)
	
	return N


def get_scores(slices,resolution):

	if 'XYS' not in globals():
		global XYS, sizeXYS

		XS = np.arange(-170.0,171.0,resolution)
		YS = np.arange(-170.0,171.0,resolution)

		XXS,YYS = np.meshgrid(XS,YS,indexing="ij")
		tempXYS = np.vstack([XXS.ravel(),YYS.ravel()])

		XYS = np.dstack(tempXYS)[0]

		sizeXYS = int(np.sqrt(len(XYS)))

		# print("Setting global XYS")


	S = npscore(slices,XYS[:,0],XYS[:,1])

	'''
	S = []
	for each in XYS:
		S.append(get_reward_for_action(slices,[each[0],each[1]]))
	'''

	# code.interact("get_scores(): ", local=dict(globals(), **locals())) 
	return XYS,S

def get_symmetric_normal_distribution(rng,mean,covMatrix,resolution):

	if 'XYD' not in globals():
		global XYD, sizeXYD

		# From -340 to 341 in order to consider targets outside of the darts board as well
		# If doing only -170 to 171, we will miss the probabilities of the targets being outside of the board
		# (for really bad agents, if just normalizing all the time)
		XD = np.arange(-340.0,341.0,resolution)
		YD = np.arange(-340.0,341.0,resolution)

		XXD,YYD = np.meshgrid(XD,YD,indexing="ij")
		tempXYD = np.vstack([XXD.ravel(),YYD.ravel()])

		XYD = np.dstack(tempXYD)[0]

		sizeXYD = int(np.sqrt(len(XYD)))


		# print("Setting global XYD")

	N = draw_noise_sample(rng,mean,covMatrix)
	
	D = N.pdf(XYD)


	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured by the resolution thus avoiding problems.  
	D *= np.square(resolution)
	
	# code.interact("get_symmetric_normal_distribution(): ", local=dict(globals(), **locals()))

	return XYD,D


def getInfoOnBoardOnly(XY,EV):

	all_vs_copy = np.copy(EV)

	# find targets/EVs that fall on dartboard

	# To keep track of which position is on the board - will have a 1 if in it
	onBoard = np.zeros((len(XY),1))


	# For each one of the possible targets - dense
	for ii in range(len(XY)):

		x,y = XY[ii]
		
		# Convert from cartesian (X,Y) to polar (R, thetha)
		a = np.arctan2(y,x) #angle
		r = np.hypot(x,y) #radius

		# xi = np.where(X == x)[0][0]
		# yi = np.where(Y == y)[0][0]

		# if within board, update status
		if r <= 170.0:
			onBoard[ii] = 1.0


	indexPositionsOnBoard = np.where(onBoard == 1.0)


	# Get actual positions/targets on board
	positionsOnBoard = XY[indexPositionsOnBoard[0]]

	# subset EVs - to only EVs corresponding to the targets within board
	evsPositionsOnBoard = all_vs_copy[indexPositionsOnBoard[0]]

	
	# For testing - Make plot to verify positions on board
	''' 
	fig = plt.figure()
	ax = plt.gca()
	cmap = plt.get_cmap("viridis")
	norm = plt.Normalize(min(evsPositionsOnBoard),max(evsPositionsOnBoard))
	plt.scatter(positionsOnBoard[:,0],positionsOnBoard[:,1], c = cmap(norm(evsPositionsOnBoard)))
	sm = ScalarMappable(norm = norm, cmap = cmap)
	sm.set_array([])
	cbar = fig.colorbar(sm,ax=ax)
	cbar.ax.set_title("EVs")
	plt.title("Positions on board")
	draw_board(ax)
	plt.show()
	# plt.savefig(f"onBoard-xskill{X}.png")
	'''

	# code.interact("getInfoOnBoardOnly: ", local=dict(globals(), **locals())) 

	return positionsOnBoard[:,0],positionsOnBoard[:,1],evsPositionsOnBoard


def compute_expected_value_curve(rng,slices,mean,covMatrix,resolution,returnZn=False):

	# tt = perf_counter()
	XYn,Zn = get_symmetric_normal_distribution(rng,mean,covMatrix,resolution)
	# print(f"Time pdfs: {perf_counter()-tt}")


	# tt = perf_counter()
	XYs,Zs = get_scores(slices,resolution)
	# print(f"Time scores: {perf_counter()-tt}")


	# tt = perf_counter()
	# Convolve to produce the EV and aiming spot
	# Output array will have the shape of the first input (Zs)
	# Zn = -340 to 341 | Zs = -170 to 171 | ("zoom in")
	EVs = fftconvolve(np.array(Zs).reshape((sizeXYS,sizeXYS)),np.array(Zn).reshape((sizeXYD,sizeXYD)),mode="same")
	EVs = EVs.flatten()

	# print(f"Time fftconvolve: {perf_counter()-tt}\n")


	# Restricting the values of the EVs to be 0 or greater
	# This is done because, for small xskills (1.0 for example), the EVs are very small and some are negative
	# Since we know that all EVs should be positive (non-negative), we can limit their values to be 0.0 or greater
	# Update: After doing the scaling up of pdfs by resolution^2 this may no longer be needed but still leaving it here just in case. 
	np.clip(EVs,0.0,None,out=EVs)


	# Return only info about targets & EVs that are actually within the board
	onBoardXs,onBoardYs,onBoardEVs = getInfoOnBoardOnly(XYs,EVs)
	

	# plt.scatter(onBoardXs,onBoardYs,c=onBoardEVs)
	# plt.show()

	# code.interact("multi - compute_expected_value_curve(): ", local=dict(globals(), **locals())) 


	# Returns pdfs as well, used on script for testing
	if returnZn:
		return XYs,EVs,onBoardEVs,Zn
	# Returns targets & EVs
	else:
		return XYs,EVs,onBoardEVs

def generate_random_states(rng,N,mode):
	states = []

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
			rng.shuffle(slices)
		
		elif mode == "rand_v":

			for i in range(21):

				# Get a random number between 1 and 20 to use as value
				# randint low = inclusive, high = exclusive (hence up to 21)
				randV = rng.integers(1,21)
				
				# Save on list
				slices.append(randV)

			# Add 25
			slices.append(25)

			# Change the order of the values - (order will be different every time)
			# To ensure 50 & 25 are not always at the end
			rng.shuffle(slices)


		states.append(slices)

	return states


def get_optimal_action_and_value(rng,S,mean,covMatrix,resolution,returnZn=False): 
	''' Get the target for a given xskill level and resolution '''   

	if returnZn:
		Xn, Yn, EV, onBoardEVs, Zn = compute_expected_value_curve(rng,S,mean,covMatrix,resolution,returnZn)
	else:
		Xn, Yn, EV, onBoardEVs = compute_expected_value_curve(rng,S,mean,covMatrix,resolution,returnZn)

	
	#code.interact("get_optimal_action_and_value(): ", local=dict(globals(), **locals())) 


	#Get maximum of EV
	#mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
	# Since now flatten
	mi = np.unravel_index(EV.argmax(), EV.shape)

	# Get action with max EV
	mx = Xn[mi]
	my = Yn[mi]

	# Return target that will give max ev and the actual max ev
	if returnZn:
		return [mx, my], EV[mi], Zn
	else:
		return [mx, my], EV[mi]
 
def get_expected_values_and_optimal_action(rng,S,mean,covMatrix,resolution,returnZn=False): 
	''' Get all the targets for a given xskill level and resolution
		as well as the one with the max EV '''

	if returnZn:
		XY, EV, onBoardEVs, Zn = compute_expected_value_curve(rng,S,mean,covMatrix,resolution,returnZn)
	else:
		XY, EV, onBoardEVs = compute_expected_value_curve(rng,S,mean,covMatrix,resolution,returnZn)


	#Get maximum of EV
	# mxi, myi = np.unravel_index(EV.argmax(), EV.shape)
	# Since now flatten
	mi = np.unravel_index(EV.argmax(),EV.shape)


	# Get action with max EV
	ts = XY[mi].tolist()

	# all_ts = {"all_ts_x": XY[:,0], "all_ts_y": XY[:,1]}
	
	# Return target that will give max ev and the actual max ev
	# as well as all the other targets and evs (all the information from the convolution)
	if returnZn:
		return EV,ts,EV[mi],onBoardEVs,Zn
	else:
		return EV,ts,EV[mi],onBoardEVs


def wrap_angle_360(angle):
	while angle > 2*pi:
		angle -= 2*pi
	while angle < 0.0:
		angle += 2*pi

	return angle

# Previously "score()"
def get_reward_for_action(slices,action,testing=False):
	""" 
	Return the score for location (x,y) on the dartboard. 
	Units are mm    
	"""
	
	x = action[0]
	y = action[1]

	#First convert to polar coordinates to get distance from (0,0) and angle from (0,0)
	a = np.arctan2(y,x) #angle
	r = np.hypot(x,y) #radius

	scaling = 1.0

	#Figure out which distance we fall in 

	# If on the double bullseye region
	if r < 6.35:
		# 2*Slices[0] will be 50 if on normal mode and any other value between 1-20 or 25 if on other modes
		return 2 * slices[0]

	# If on the single bullseye region
	if r < 15.9: 
		# Slices[0] will be 25 if on normal mode and any other value between 1-20 or 25 if on other modes
		return slices[0]


	if r > 99 and r < 107:
		# Triple score
		scaling = 3.0

	if r > 162 and r < 170:
		# Double score
		scaling = 2.0

	if r >= 170:
		# Off the board
		return 0.0

	#Figure out which slice we fall in
	angle_increment = pi / 10.0

	slice_low = - pi - angle_increment / 2.0
	slice_high = slice_low + angle_increment

	# print("getV a: ",a)
	# print("getV r: ",r)

	# Rest (21) of the slices
	for i in range(1,22):
		if testing:
			'''
			ax = plt.gca()
			draw_board(ax)
			label_regions(slices,color="black")
			plt.scatter(slice_low,slice_high)
			plt.show()
			plt.clf()	
			plt.close()
			'''
			print(slice_low,slice_high)


		if a >= slice_low and a < slice_high:
			return scaling*slices[i]

		slice_low += angle_increment
		slice_high += angle_increment

	# code.interact("...", local=dict(globals(), **locals())) 

	#Check for 11 slice

	#Must have missed the board!
	return 0.0


def npscore(slices,x,y,return_doub=False):

	# Get the bullseye value
	be = slices[0]

	# slices = np.array([11,8,8,16,16,7,7,19,19,3,3,17,17,2,2,15,15,10,10,6,6,13,13,4,4,18,18,1,1,20,20,5,5,12,12,9,9,14,14,11,11])   
	slices = np.repeat(slices,2)[3:]

	a = np.arctan2(y,x)
	r = np.hypot(x,y)

	a = ((a + pi) * (40/(2*pi))).astype(int)
	ans = slices[a]
	

	doubb = r<6.35
	bulls = r<15.9
	on = r<170
	
	trip = (r>99) & (r<107)
	doub = (r<170) & (r>162)

	# Did you hit the board and not the bullseye (single/double)?
	ans *= np.invert(bulls)*on 

	# Are you on the bullseye (single/double)?
	ans += (be*bulls) + (be*doubb)
	
	# Did you have a triple/double?
	ans += (ans*trip*2) + (ans*doub)

	# code.interact("npscore(): ", local=dict(globals(), **locals())) 
	
	if return_doub:
		return ans, doub+doubb
	return ans 


def sample_noisy_action(rng,S,mean,covMatrix,a,noiseModel=None):

	# If noise model was not given, proceed to get it
	if noiseModel == None:
		N = draw_noise_sample(rng,mean,covMatrix)
	# Otherwise, use given noise model
	else:
		N = noiseModel

	# Get noise (sample)
	noise = N.rvs()
	# print("NOISE: ", noise)

	# Add noise to planned action (This creates the noisy action)
	na = [a[0]+noise[0],a[1]+noise[1]]

	# code.interact("sample_noisy_action() ", local=dict(globals(), **locals()))

	return na

def calculate_wrapped_action_difference(action1,action2):

	'''
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
	d = np.sqrt(np.power(r1,2.0) + np.power(r2,2.0) - ((2.0*r1*r2)* np.cos(a2-a1)))
	'''

	# Compute euclidean distance (Equivalent to the code above)
	d = dist(action1,action2)

	# code.interact(local=locals())

	return d

def isHit(action):

	x = action[0]
	y = action[1]

	# Convert action (x, y) to polar coordinates
	a = np.arctan2(y,x) # angle in radians
	r = np.hypot(x,y) # radius

	# Off the board
	if r > 170:
		return 0.0
	else:
		return 1.0


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


def simulate_board_hits(rng,covMatrices,numTries):

	# Representation of a state (0 since the state is the same every time)
	S = 0

	resolution = 5.0

	mean = [0.0]*len(nums)


	allPercentHits = []


	for each in range(len(xSkills)):

		xs,rho = xSkills[each]

		hits = 0.0

		key = getKey(xs,rho)

		# print(key)
		# print(covMatrices[key])

		for tries in range(int(numTries)):

			# from all the targets, select one in the middle of the board
			# if r < 6.35 -> Double bullseye = 50 points = middle of board

			# Get random radius
			r = np.random.uniform(low=0.0,high=6.35,size=1)

			# Get random angle - in radians
			a = np.pi * np.random.uniform(0,2)

			# Convert from polar to coordinate
			x = r * np.cos(a)
			y = r * np.sin(a)

			action = [x, y]

			nx, ny = sample_noisy_action(rng,S,mean,covMatrices[key],action)
			noisyAction = [nx, ny]

			'''
                    print("\t\tsample_noisy_action(): ")
			print("\t\tnx: ", nx)
			print("\t\tny: ", ny)
			print("\t\tscore: ", get_reward_for_action(S, noisyAction))
			print("\n")
			'''

			# Verify if the action hits the board or not
			# Returns 1 if hit, 0 if not
			hits += isHit(noisyAction)


		percentHit = (hits/numTries)*100.0
		allPercentHits.append(percentHit)
		
		print(f"xSkills: {key} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%\n")


	'''
	plt.plot(xSkills, allPercentHits)
	plt.xlabel('xSkills')
	plt.ylabel('% Hits')
	#plt.legend()
	plt.show()
	'''

	return xSkills, allPercentHits


def drawBoardWithEVsForDiffXskillsAndResolutions(rng):

	##################################################
	# PARAMETERS FOR PLOTS
	##################################################

	plt.rcParams.update({'font.size': 14})
	plt.rcParams.update({'legend.fontsize': 14})
	plt.rcParams["axes.labelweight"] = "bold"
	plt.rcParams["axes.titleweight"] = "bold"

	##################################################


	# For convolution
	# resolutions = [1.0, 5.0, 10.0]
	resolutions = [5.0]
	mean = [0.0]*len(nums)

	# mode = "normal"
	mode = "rand_pos"
	#mode = "rand_v"
	
	numStates = 3 #20

	# Representation of a state (0 since the state is the same every time)
	S = generate_random_states(rng,numStates,mode)


	targets_x = []
	targets_y = []


	for each in range(len(xSkills)):

		xs,rho = xSkills[each]

		key = getKey(xs,rho)
		
		print()
		print(key)
	
		for res in resolutions:

			string = "\tresolution: " + str(res)

			# for each one of the states
			for s in range(len(S)):

				state = S[s]
				print("slices main: ", state)

				# all_ts, EV, xy, ev, Zn = get_expected_values_and_optimal_action(S,xs,res, returnZn = True)
				all_ts, EV, xy, ev, onBoardEVs = get_expected_values_and_optimal_action(state,mean,covMatrices[key],res)

				'''
                            print "\t\tget_optimal_action_and_value(): "
				print "\t\tx: ", xy[0]
				print "\t\ty: ", xy[1]
				print "\t\tev: ", ev
				print "\t\tscore: ", get_reward_for_action(S, [xy[0],xy[1]])
				print "\n"
				'''

				targets_x.append(xy[0])
				targets_y.append(xy[1])

				nx, ny = sample_noisy_action(rng,S,mean,covMatrices[key],[xy[0],xy[1]])

				'''
                            print "\t\tsample_noisy_action(): "
				print "\t\tnx: ", nx
				print "\t\tny: ", ny
				print "\t\tscore: ", get_reward_for_action(S, [nx,ny])
				print "\n"
				'''

				# calculate_wrapped_action_difference(xy,[nx,ny])

				# code.interact("...", local=dict(globals(), **locals()))


				#'''
				fig = plt.figure()
				ax = plt.gca()

				cmap = plt.get_cmap("viridis")
				
				norm = plt.Normalize(np.min(EV),np.max(EV))

				# xx, yy = np.meshgrid(all_ts["all_ts_x"],all_ts["all_ts_y"],indexing="ij")
				xx, yy = all_ts["all_ts_x"],all_ts["all_ts_y"]
				plt.scatter(xx,yy,c=EV,norm=norm)
				
				
				sm = ScalarMappable(norm = norm, cmap = cmap)
				sm.set_array([])
				cbar = fig.colorbar(sm,ax=ax)
				#cbar.ax.set_title("Expected Score", fontdict = {'verticalalignment': 'center', 'horizontalalignment': "left"},\
				#                     y = 0.5, rotation = 90, pad = 100.0)

				cbar.set_label("Expected Score")

				draw_board(ax)
				label_regions(state, color = "white")

				#plt.plot(xy[0], xy[1], marker = "*", label = "Max EV Action")
				#plt.plot(nx, ny, marker = "*", label = "Max EV Noisy Action")

				plt.title(f"Execution Noise Level: {key}")
				ax.set_xticks([])
				ax.set_yticks([])

				#plt.legend()

				#plt.show()
				plt.savefig(f"xSkill-{key}-mode-{mode}-resolution-{res}-state{str(s)}.png")

				plt.clf()
				plt.close()
				#'''

def getKey(info,rho):

	temp = ""

	for t in info:
		temp += f"{t}|"

	return f"{temp}{rho}"


if __name__ == "__main__":

	rng = np.random.default_rng(1000)

	# num to sample
	nums = [5,5]

	# format: [[startX,stopX] per each dimension]
	# range for each xskill dimension
	info = [[3.0,150.0],[3.0,150.0]]


	# rhos = np.round(np.linspace(-0.99,0.99,num=5),4)
	rhos = np.round(np.linspace(-0.75,0.75,num=5),4)


	temp = []

	for i in range(len(nums)):
		temp.append(np.round(np.linspace(info[i][0],info[i][1],num=nums[i]),4))

	xSkills = list(product(*temp))
	xSkills = list(product(xSkills,rhos))
	# code.interact("...", local=dict(globals(), **locals()))


	covMatrices = {}

	for each in range(len(xSkills)):

		xs,rho = xSkills[each]

		key = getKey(xs,rho)
			
		covMatrices[key] = getCovMatrix(xs,rho)



	numTries = 10_000

	# allPercentHits = simulate_board_hits(rng,covMatrices,numTries)	

	
	drawBoardWithEVsForDiffXskillsAndResolutions(rng)


	# code.interact("...", local=dict(globals(), **locals()))


