import numpy as np
import scipy
import scipy.stats as stats
from matplotlib import pyplot as plt
import itertools
import sys
import time 
import json
import os
import code
import random

#File used to generate data/plots for the extended abstract plots

def get_reward_for_action(S,m,a):
	low = True
	for s in S:
		if a < s:
			break
		low = not low

	if low:
		return 0.0
	else:
		return 1.0

def sample_noisy_action(S,m,L,a):
	#Noisy action
	# print 'Sampling 1 for ', a, L
	na = np.random.normal(a,L)

	if na > m:
		na = na - 2*m
	if na < -m:
		na = na + 2*m

	return na

def action_diff(a1,a2,m):
	d = a1 - a2
	if d > m:
		d -= 2*m
	if d < -m:
		d += 2*m

	return d

def sample_single_rollout(S,m,L,a):
	#See where noisy action lands in S
	return get_reward_for_action(S,m,sample_noisy_action(S,m,L,a))

def estimate_value_with_samples(S,m,L,NS,a):
	# print 'Sampling N for', a, L
	tr = 0.0
	for i in range(NS):
		tr += sample_single_rollout(S,m,L,a)
	return tr / float(NS)

def wrap_action_within_bounds(a,m):
	while a > m:
		a = a - 2*m
	while a < -m:
		a = a + 2*m
	return a

def compute_expected_value_curve(S,m,L,a,delta=1e-3):
	num_points = int(6*m/delta)
	big_grid = np.linspace(-3*m,3*m,num_points)

	state = [get_reward_for_action(S,m,wrap_action_within_bounds(a,m)) for a in big_grid]

	#Get convolver
	err = stats.norm(loc=0,scale=L)
	errpmf = err.pdf(big_grid)*delta

	conv_pmf = np.convolve(state,errpmf,'same')

	left = int(num_points/3)
	right = int(2*left)

	ai = int(np.rint(left + (right-left)*(m-a)/(2*m)))
	# print('This is AI: ', ai)

	return conv_pmf[ai]

def target_strategy(S, m,ps):
	L = ps['noise_level']
	return get_optimal_action_and_value(m, S, L)

def generate_random_states(m,low,high,N):
	states = []

	for n in range(N):
		#Create N regions (where N is even)
		num_r = np.random.randint(low,high)*2

		#Get the N boundary points
		S = np.random.uniform(-m, m, size=num_r)
		S = np.sort(S)
		states.append(S)

	return states

def newEV(s, m, Xs, a, N, expNum):
	#Input arguments: 
	# s - the state
	# m - the radius of the state (usually 10)
	# Xs - list of all the execution skills we are evaluated 
	# a - action that we are computing the EV of for all these execution skills
	# N - number of samples to do per execution skill 
	nv = []

	As = []
	Vs = []
	Qs = []


	#First get some samples for each execution skill
	for x in Xs:
		#Get our normal pdf for this execution skill
		g = stats.norm(scale=x)
		#Sample the action N times: 
		for n in range(N):
			#Get sampled action
			sa = sample_noisy_action(s,m,x,a)
			#Get corresponding value
			v = get_reward_for_action(s,m,sa)
			#Add those to our lists
			As.append(sa)
			Vs.append(v)
			#Compute the probability of that action (from pdf)
			# distance of sampled action from input action
			da = action_diff(a, sa, m)
			# get pdf value for this x and append to our list
			Qs.append(g.pdf(sa))

	#Now we loop to compute EV estimates for all Xs
	for x in Xs:
		#To track the value and weight
		curV = 0.0
		curW = 0.0
		#Get the current pdf
		g = stats.norm(scale=x)
		#Loop over all the samples
		for i, ai in enumerate(As):
			#Get the value
			vi = Vs[i]
			#Get the distance of this action from the intended one
			d = action_diff(ai, a, m)
			#Get the probability of that action under this x 
			W = g.pdf(d)
			#Importance Weighting to get correct estimate
			curV += vi*W/Qs[i]
			curW += W/Qs[i]

		#Final EV estimate is just the weighted average
		nv.append(curV/curW)

	if expNum == 50:
		code.interact("newEV()...", local=dict(globals(), **locals()))

	#Return all estimates
	return nv

def online_experiment(xskills,numSamples,expNum,s):
	
	#For each, select an action
	a = 0 #np.random.uniform(-10,10)

	#Compute EV using 3 different methods
	#   1. Convolve
	#   2. Sample
	cEV = []
	sEV = []

	#   3. Sample all, store, reuse with weights from pdf
	# This method uses the same number of samples as method 2. 
	# Can try reducing (i did one tenth and it still seemed ok)

	nEV = newEV(s,10,xskills,a,numSamples,expNum)

	for x in xskills:
		#Get convolve EV
		cv = compute_expected_value_curve(s,10,x,a)
		cEV.append(cv)
		#Get sample ev
		sv = estimate_value_with_samples(s,10,x,numSamples,a)
		sEV.append(sv)


	plt.plot(xskills,cEV,label='Conv')
	plt.plot(xskills,sEV,label='Samp')
	plt.plot(xskills,nEV,label='New')
		
	plt.legend()
	plt.savefig(expName+os.path.sep+"Plots"+os.path.sep+"EVsDiffMethods"+os.path.sep+f"{numSamples}Samples"+os.path.sep+f"EVs-Xskills{len(xskills)}-Samples{numSamples}-Exp{expNum}.png", bbox_inches='tight')
	plt.clf()
	plt.close()

	# Testing
	diffs = abs(np.array(cEV) - np.array(nEV))

	if (diffs >= 0.15).any():
		code.interact("!!!...", local=dict(globals(), **locals()))

	return cEV,sEV,nEV


if __name__ == "__main__":

	np.random.seed(0)
	random.seed(0)

	expName = 'OnlineExp_'+time.strftime("%y%m%d%M%S")

	for each in [expName,expName+os.path.sep+"Results",
				expName+os.path.sep+"Plots",expName+os.path.sep+"Plots"+os.path.sep+"EVsDiffMethods"]:
		if not os.path.exists(each):
			os.mkdir(each)


	start = time.time()

	num_experiments = 500
	num_x = 10 # Number of execution skill hypotheses
	numSamples = [1000,500] # Number of samples per xskill 

	# Execution skills to compute EV for
	xskills = np.linspace(0.25, 5.0, num_x)

	states = generate_random_states(10,2,4,num_experiments)
	

	results = {}
	results["numExps"] = num_experiments
	results["numSamples"] = numSamples
	results["xskills"]= xskills.tolist()

	for ns in numSamples:

		if not os.path.exists(expName+os.path.sep+"Plots"+os.path.sep+"EVsDiffMethods"+os.path.sep+f"{ns}Samples"):
			os.mkdir(expName+os.path.sep+"Plots"+os.path.sep+"EVsDiffMethods"+os.path.sep+f"{ns}Samples")


		for i in range(num_experiments):
			
			print(f"Samples: {ns} | Experiment #{i}")
			results[i] = {}

			cEV,sEV,nEV = online_experiment(xskills,ns,i,states[i])
			
			results[i]["cEV"] = cEV
			results[i]["sEV"] = sEV
			results[i]["nEV"] = nEV


		# SAVE RESULTS
		with open(expName+os.path.sep+"Results"+os.path.sep+f"results-Xskills{num_x}-Samples{ns}-NumExps{num_experiments}.json", 'w') as outfile:
			json.dump(results, outfile)

	# code.interact("...", local=dict(globals(), **locals()))



