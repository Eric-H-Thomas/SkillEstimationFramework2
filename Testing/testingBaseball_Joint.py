import numpy as np
import scipy

import code
import time
import os
from copy import deepcopy

from scipy.stats import multivariate_normal

# for testing
from matplotlib import pyplot as plt

import numpy as np
import scipy
from copy import deepcopy

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


# Joint estimation of planning and execution skill - bounded agent rationality model
class JointMethodQRE():

	def __init__(self,xskills,numPskills,domainName):
  
		self.xskills = xskills
		self.numXskills = len(xskills)
		self.numPskills = numPskills

		self.domainName = domainName

		self.names = ['JT-QRE-MAP-'+str(self.numXskills)+'-'+str(self.numPskills),'JT-QRE-EES-'+str(self.numXskills)+'-'+str(self.numPskills)]
	   
		self.estimatesXskills = dict()
		self.estimatesPskills = dict()

		for n in self.names:
			self.estimatesXskills[n] = []
			self.estimatesPskills[n] = []


		self.pskills = np.logspace(-3,3.6,self.numPskills)


		self.probs = np.ndarray(shape=(self.numXskills,self.numPskills))

		# Initializing the array
		self.probs.fill(1.0/(self.numXskills*self.numPskills*1.0))

		self.allProbs = []

		# To save the initial probs - uniform distribution for all
		self.allProbs.append(self.probs.tolist())


	def getEstimatorName(self):
		return self.names
	   

	def reset(self):

		for n in self.names:
			self.estimatesPskills[n] = []
			self.estimatesXskills[n] = []

		# reset probs
		self.probs.fill(1.0/(self.numXskills*self.numPskills*1.0))

		self.allProbs = []

		self.allProbs.append(self.probs.tolist())


	def addObservation(self,action,possibleTargets,otherArgs):

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Initialize info
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		delta = otherArgs["delta"]

		action = np.array(action)
		
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		startTimeEst = time.perf_counter()


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Compute PDFs and EVs
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		PDFsPerXskill = {}
		EVsPerXskill = {}


		# For each execution skill hyps
		for xi in range(self.numXskills):

			# Get the corresponding xskill level hypothesis at the given index
			x = self.xskills[xi]

			evs = otherArgs["evsPerXskill"][x].flatten()

			pdfs = computePDF(x=action,means=possibleTargets,covs=np.array([otherArgs["allCovs"][xi]]*len(possibleTargets)))

			# code.interact("JTM...", local=dict(globals(), **locals()))


			# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
			# This is because depending on the xskill/resolution combination, the pdf of
			# a given xskill may not show up in any of the resolution buckets 
			# causing then the pdfs not adding up to 1
			# (example: xskill of 1.0 & resolution > 1.0)
			# If the resolution is less than the xskill, the xskill distribution can be fully captured 
			# by the resolution thus avoiding problems. 
			pdfs = np.multiply(pdfs,np.square(delta))


			# Save info
			PDFsPerXskill[x] = pdfs
			EVsPerXskill[x] = evs

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Perform Update
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		# For each execution skill hyps
		for xi in range(self.numXskills):

			# Get the corresponding xskill level hypothesis at the given index
			x = self.xskills[xi]

			pdfs = PDFsPerXskill[x]
			evs = EVsPerXskill[x]


			# For each planning skill hyp
			for pi in range(self.numPskills):
			   
				# Get the corresponding pskill level at the given index
				p = self.pskills[pi]

				# Create copy of EVs 
				evsC = np.copy(evs)

				# With norm trick
				b = np.max(evsC*p)
				expev = np.exp(evsC*p-b)

				# Without norm trick
				# expev = np.exp(evsC*p)

				sumexp = np.sum(expev)

				# JT Update 
				summultexps = np.sum(np.multiply(expev,np.copy(pdfs)))

				P = summultexps/sumexp

				upd = P

				# Update probs
				self.probs[xi][pi] *= upd


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Normalize
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		self.probs /= np.sum(self.probs)
		self.allProbs.append(self.probs.tolist())
		# code.interact("after norm ", local=dict(globals(), **locals()))

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Get estimates
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		# MAP estimate - Get index of maximum prob - returns flat index
		mi = np.argmax(self.probs)

		# "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
		xmi, pmi = np.unravel_index(mi,self.probs.shape)

		self.estimatesXskills[self.names[0]].append(self.xskills[xmi])
		self.estimatesPskills[self.names[0]].append(self.pskills[pmi])


		# Get EES & EPS Estimate
		ees = 0.0
		eps = 0.0
		
		for xi in range(self.numXskills):
			for pi in range(self.numPskills):
				ees += self.xskills[xi] * self.probs[xi][pi]
				eps += self.pskills[pi] * self.probs[xi][pi]

		self.estimatesXskills[self.names[1]].append(ees)
		self.estimatesPskills[self.names[1]].append(eps) 

		# sprint(f"JT-QRE -> Current estimate: X:{ees} P:{eps}\n")
		# code.interact("...", local=dict(globals(), **locals()))

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	   

		endTimeEst = time.perf_counter()
		totalTimeEst = endTimeEst - startTimeEst

		'''
		print("JT-QRE")
		print("intendedAction: ", intendedAction, "\t noisy action: ", action)
		print("EES: ", ees, "\t\t MAP: ", self.estimatesXskills[self.names[0]][-1])
		print("EPS: ", eps, "\t\t MAP: ", self.estimatesPskills[self.names[0]][-1], "\n")
		#print (self.probs)
		code.interact("after computing MAP & EES", local=locals())
		'''


	def getResults(self):
		results = dict()
		
		for n in self.names:
			results[n + "-pSkills"] = self.estimatesPskills[n]
			results[n + "-xSkills"] = self.estimatesXskills[n]

		results["JT-QRE"+ str(self.numXskills)+"-"+str(self.numPskills)+"-allProbs"] = self.allProbs
		
		return results


# Non-Joint estimation of planning and execution skill - bounded agent rationality model
class NonJointMethodQRE():
   
	def __init__(self,xskills,numPskills,domainName):
  
		self.xskills = xskills
		self.numXskills = len(xskills)
		self.numPskills = numPskills

		self.domainName = domainName

		self.names = ['NJT-QRE-MAP-'+str(self.numXskills)+'-'+str(self.numPskills),'NJT-QRE-EES-'+str(self.numXskills)+'-'+str(self.numPskills)]

		self.estimatesXskills = dict()
		self.estimatesPskills = dict()

		for n in self.names:
			self.estimatesXskills[n] = []
			self.estimatesPskills[n] = []


		self.pskills = np.logspace(-3,3.6,self.numPskills)


		self.probsXskills = np.ndarray(shape=(self.numXskills,1))
		self.probsPskills = np.ndarray(shape=(self.numPskills,1))

		# Initializing the array
		self.probsXskills.fill(1.0/(self.numXskills*1.0))
		self.probsPskills.fill(1.0/(self.numPskills*1.0))

		self.allProbsXskills = []
		self.allProbsPskills = []

		# To save the initial probs - uniform distribution for all
		self.allProbsXskills.append(self.probsXskills.tolist())
		self.allProbsPskills.append(self.probsPskills.tolist())


	def getEstimatorName(self):
		return self.names
	 

	def reset(self):

		for n in self.names:
			self.estimatesPskills[n] = []
			self.estimatesXskills[n] = []

		# Reset probs
		self.probsXskills.fill(1.0/(self.numXskills*1.0))
		self.probsPskills.fill(1.0/(self.numPskills*1.0))

		self.allProbsXskills = []
		self.allProbsPskills = []

		# To save the initial probs - uniform distribution for all
		self.allProbsXskills.append(self.probsXskills.tolist())
		self.allProbsPskills.append(self.probsPskills.tolist())


	def addObservation(self,action,possibleTargets,otherArgs):

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# Initialize info
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		
		delta = otherArgs["delta"]

		action = np.array(action)

		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		PDFsPerXskill = {}
		EVsPerXskill = {}


		startTimeEst = time.perf_counter()

		##################################################################################################################################
		# ESTIMATING EXECUTION SKILL
		##################################################################################################################################

		# For each execution skill hyps
		for xi in range(len(self.xskills)):

			# Get the corresponding xskill level hypothesis at the given index
			x = self.xskills[xi]


			###########################################################################################
			# Compute PDFs and EVs
			###########################################################################################

			evs = otherArgs["evsPerXskill"][x].flatten()

			pdfs = computePDF(x=action,means=possibleTargets,covs=np.array([otherArgs["allCovs"][xi]]*len(possibleTargets)))


			pdfs = np.multiply(pdfs,np.square(delta))


			# Store in order to reuse later on when updating pskills probs
			PDFsPerXskill[x] = pdfs
			EVsPerXskill[x] = evs
			
			###########################################################################################

			v3 = []

			# For each planning skill hyp
			for pi in range(len(self.pskills)):
			   
				# Get the corresponding pskill level at the given index
				p = self.pskills[pi]


				# Create copy of EVs 
				evsCP = np.copy(evs)

				# To be used for exp normalization trick - find maxEV and * by p
				# To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
				# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
				b = np.max(evsCP*p)
				
				# With normalization trick
				expev = np.exp(evsCP*p-b) 

				# exps = V1
				sumexp = np.sum(expev)
				
				V2 = expev/sumexp

				# Non-JT Update 
			
				mult = np.multiply(V2,pdfs)

				# upd = v2
				upd = np.sum(mult)

				# Update probs pskill
				v3.append(self.probsPskills[pi] * upd)


			# Update probs xskill
			self.probsXskills[xi] *= np.sum(v3)

		##################################################################################################################################


		##################################################################################################################################
		# ESTIMATING PLANNING SKILL
		##################################################################################################################################

		# For each pskill hyp
		for pi in range(len(self.pskills)):
		   
			# Get the corresponding pskill level at the given index
			p = self.pskills[pi]


			v3 = []

			# For each xskill hyp
			for xi in range(len(self.xskills)):

				# Get the corresponding xskill level hypothesis at the given index
				x = self.xskills[xi]

				pdfs = PDFsPerXskill[x]
				evs = EVsPerXskill[x]

				# Create copy of EVs 
				evsC = np.copy(evs)

				# To be used for exp normalization trick - find maxEV and * by p
				# To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
				# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
				b = np.max(evsCP*p)

				# With normalization trick
				expev = np.exp(evsCP*p-b) 

				# exps = V1
				sumexp = np.sum(expev)

				V2 = expev/sumexp

				# Non-JT Update 

				mult = np.multiply(V2,pdfs)

				# upd = v2
				upd = np.sum(mult)

				# Update probs pskill
				v3.append(self.probsXskills[xi]*upd)


			# Update probs pskill
			self.probsPskills[pi] *= np.sum(v3)

		##################################################################################################################################
		
		# Once done updating the different probabilities, proceed to get estimates

		# Normalize
		self.probsXskills /= np.sum(self.probsXskills)
		self.probsPskills /= np.sum(self.probsPskills)


		self.allProbsXskills.append(self.probsXskills.tolist())
		self.allProbsPskills.append(self.probsPskills.tolist())


		# Get estimate. Uses MAP estimate
		# Get index of maximum prob
		xmi = np.argmax(self.probsXskills)
		pmi = np.argmax(self.probsPskills)

		# code.interact("...", local=dict(globals(), **locals()))

		self.estimatesXskills[self.names[0]].append(self.xskills[xmi])
		self.estimatesPskills[self.names[0]].append(self.pskills[pmi])


		#Get EES Estimate
		ees = 0.0
		#print "probs xskills: "
		for xi in range(len(self.xskills)):
			# print "x: " + str(self.xskills[xi]) + "->" + str(self.probsXskills[pi])
			# [0] in order to get number out of array
			# To avoid problems when saving results to json file since results in an array within an array
			ees += self.xskills[xi] * self.probsXskills[xi][0]
			
	  
		#Get EPS Estimate
		eps = 0.0
		#print "probs pskills: "
		for pi in range(len(self.pskills)):
			# print "p: " + str(self.pskills[pi]) + "-> " + str(self.probsPskills[pi][0])
			# [0] in order to get number out of array
			# To avoid problems when saving results to json file since results in an array within an array
			eps += self.pskills[pi] * self.probsPskills[pi][0]


		self.estimatesXskills[self.names[1]].append(ees)
		self.estimatesPskills[self.names[1]].append(eps)        

		endTimeEst = time.perf_counter()
		totalTimeEst = endTimeEst-startTimeEst

		# print("NJT-QRE")
		# print("EES: ", ees, "\t\t MAP: ", self.estimatesXskills[self.names[0]][-1])
		# print("EPS: ", eps, "\t\t MAP: ", self.estimatesPskills[self.names[0]][-1], "\n")
		# code.interact("...", local=dict(globals(), **locals()))


	def getResults(self):
		results = dict()
		
		for n in self.names:
			results[n + "-pSkills"] = self.estimatesPskills[n]
			results[n + "-xSkills"] = self.estimatesXskills[n]

		results["NJT-QRE"+ str(self.numXskills)+"-allProbs-xSkills"] = self.allProbsXskills
		results["NJT-QRE"+ str(self.numPskills)+"-allProbs-pSkills"] = self.allProbsPskills
		
		return results

