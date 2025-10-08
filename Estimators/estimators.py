import os,copy

from Estimators.observed_reward import *
from Estimators.bayesian import *
from Estimators.joint import *
from Estimators.joint_pfe import *
from itertools import product


# ToDo: Create an ABC
# class EstimatorsBase():


class Estimators():

	__slots__ = ["num_execution_skill_hypotheses","num_rationality_hypotheses","allXskills","estimators","estimators_list","numHypsR","rhos","numParticles","xskillsNormal"]


	def __init__(self,infoForEstimators,env):

		self.num_execution_skill_hypotheses = infoForEstimators["num_execution_skill_hypotheses"]
		self.num_rationality_hypotheses = infoForEstimators["num_rationality_hypotheses"]

		domainName = env.domainName		

		if domainName in ["2d-multi","baseball-multi","hockey-multi"]:
			self.numHypsR = infoForEstimators["numHypsRhos"]
			self.rhos = np.round(np.linspace(-0.80,0.80,num=self.numHypsR),4)

		if env.particles:
			self.numParticles = infoForEstimators["diffNs"]


		self.allXskills = []

		# Proceed to create the estimators for each one of the different number of hypothesis skills
		self.estimators = []

		# For each one of the different number of hypothesis
		for i in range(len(self.num_execution_skill_hypotheses)):

			numXs = self.num_execution_skill_hypotheses[i]			

			if domainName == "billiards":
				# NOTE: agentTypes[0] for billiards so that it can get the 
				# available xskills based on agent type. [0] bc assuming one type at a time.
				# Need to update later
				xskills = list(env.spaces.xskilltoAgentId[env.agentTypes[0]].keys())
			
			elif domainName == "baseball":
				xskills = list(np.concatenate((np.linspace(infoForEstimators["min_execution_skill_for_domain"],1.0,num=60),np.linspace(1.00+env.delta,infoForEstimators["max_execution_skill_for_domain"],num=6))))
				# FOR TESTING
				# xskills = list(np.linspace([infoForEstimators"min_execution_skill_for_domain"],infoForEstimators["max_execution_skill_for_domain"],num=numXs))
			
			elif domainName in ["2d-multi","baseball-multi","hockey-multi"]:

				# For normal xskill hyps space (focusing on 1 dimension - the 1st one)
				if domainName in ["2d-multi","hockey-multi"]:
					self.xskillsNormal = list(np.round(np.linspace(infoForEstimators["min_execution_skill_for_domain"][0],infoForEstimators["max_execution_skill_for_domain"][0],num=numXs[0]),4))
				else:
					self.xskillsNormal = list(np.concatenate((np.linspace(infoForEstimators["min_execution_skill_for_domain"][0],1.0,num=60),np.linspace(1.00+env.delta,infoForEstimators["max_execution_skill_for_domain"][0],num=6))))
				
					# Create symmetric set
					# self.xskillsNormal = [[each]*len(infoForEstimators["min_execution_skill_for_domain"]) for each in self.xskillsNormal]

				xskills = []
				for j in range(len(numXs)):

					if domainName in ["2d-multi","hockey-multi"]:
						xskills.append(list(np.round(np.linspace(infoForEstimators["min_execution_skill_for_domain"][j],infoForEstimators["max_execution_skill_for_domain"][j],num=numXs[j]),4)))
					else:
						xskills.append(list(np.concatenate((np.linspace(infoForEstimators["min_execution_skill_for_domain"][j],1.0,num=60),np.linspace(1.00+env.delta,infoForEstimators["max_execution_skill_for_domain"][j],num=6)))))

				temp = list(product(*xskills))

			else: # 1d/2d/sequentialDarts
				xskills = list(np.round(np.linspace(infoForEstimators["min_execution_skill_for_domain"],infoForEstimators["max_execution_skill_for_domain"],num=numXs),4))



			if domainName in ["2d-multi","baseball-multi","hockey-multi"]:
				self.allXskills = sorted(list(set(self.allXskills+temp)))
			else:
				self.allXskills = sorted(list(set(self.allXskills+xskills)))
		

			# code.interact("...", local=dict(globals(), **locals()))			



			# Create objects for corresponding estimator
			if "OR" in infoForEstimators["estimators_list"]:
				self.estimators.append(ObservedReward(xskills,domainName))

			if "BM" in infoForEstimators["estimators_list"]:

				for b in infoForEstimators["betas"]:

					if domainName in ["baseball","baseball-multi","hockey-multi"]:
						self.estimators.append(BayesianMethod(xskills,b,domainName))
			
					# For normal xskill hyps space (focusing on 1 dimension - the 1st one)	
					elif domainName == "2d-multi":
						self.estimators.append(BayesianMethod(self.xskillsNormal,b,domainName,typeTargets="OptimalTargets"))
						
					else:
						self.estimators.append(BayesianMethod(xskills,b,domainName,typeTargets="OptimalTargets"))

						if domainName == "sequentialDarts":
							self.estimators.append(BayesianMethod(xskills,b,domainName,typeTargets="DomainTargets"))
					

			for j in range(len(self.num_rationality_hypotheses)):

				numPs = self.num_rationality_hypotheses[j]

				if "JT-FLIP" in infoForEstimators["estimators_list"]:
					self.estimators.append(JointMethodFlip(xskills,numPs,domainName))


				for each in infoForEstimators["estimators_list"]:

					allInfo = []
	
					if "JT-QRE" in each or "NJT-QRE" in each:
						
						if "GivenPrior" in each and "MinLambda" in each:
							givenPrior = True
							minLambda = True

							for each1 in infoForEstimators["otherArgs"]["givenPrior"]:
								for each2 in infoForEstimators["otherArgs"]["minLambda"]:
									toSend = {"delta":env.delta,"givenPrior":each1,"minLambda":each2}
									allInfo.append(toSend)

						elif "GivenPrior" in each:
							givenPrior = True
							minLambda = False

							for each1 in infoForEstimators["otherArgs"]["givenPrior"]:
								toSend = {"delta":env.delta,"givenPrior":each1}
								allInfo.append(toSend)

						elif "MinLambda" in each:
							givenPrior = False
							minLambda = True

							for each1 in infoForEstimators["otherArgs"]["minLambda"]:
								toSend = {"delta":env.delta,"minLambda":each1}
								allInfo.append(toSend)

						else:
							# Default JT & NJT	
							givenPrior = False
							minLambda = False
	
							if "NJT" in each:
								self.estimators.append(NonJointMethodQRE(xskills,numPs,domainName,givenPrior,minLambda))
							
							elif "Multi" in each and "Particles" not in each:
								self.estimators.append(QREMethod_Multi(xskills,numPs,self.rhos,domainName,givenPrior,minLambda))
							
							elif "Multi-Particles" in each:

								for pi in infoForEstimators["diffNs"]:
									for ni in infoForEstimators["noises"]:
										for percent in infoForEstimators["percents"]:

											# For when testing both at the same time (with and without resampleNEFF)
											'''
											self.estimators.append(QREMethod_Multi_Particles(env,pi,ni,percent,False,infoForEstimators["resamplingMethod"],infoForEstimators["ranges"],givenPrior,minLambda))
										
											if infoForEstimators["resampleNEFF"]:
												self.estimators.append(QREMethod_Multi_Particles(env,pi,ni,percent,True,infoForEstimators["resamplingMethod"],infoForEstimators["ranges"],givenPrior,minLambda))
											'''
											self.estimators.append(QREMethod_Multi_Particles(env,pi,ni,percent,infoForEstimators["resampleNEFF"],infoForEstimators["resamplingMethod"],infoForEstimators["ranges"],givenPrior,minLambda))

							else:
								if domainName in ["2d-multi","baseball-multi","hockey-multi"]:
								 	self.estimators.append(JointMethodQRE(self.xskillsNormal,numPs,domainName,givenPrior,minLambda))
								else:
									self.estimators.append(JointMethodQRE(xskills,numPs,domainName,givenPrior,minLambda))
							

					for toSend in allInfo:	
						
						if "NJT-QRE" in each:
							self.estimators.append(NonJointMethodQRE(xskills,numPs,domainName,givenPrior,minLambda,toSend))
						
						elif "Multi" in each:

							xskills = []
							for j in range(len(numXs)):
								xskills.append(list(np.round(np.linspace(infoForEstimators["min_execution_skill_for_domain"][j],infoForEstimators["max_execution_skill_for_domain"][j],num=numXs[j]),4)))

							self.estimators.append(QREMethod_Multi(xskills,numPs,self.rhos,domainName,givenPrior,minLambda))
						
						# elif "Multi-Particles" in each:
						# 	self.estimators.append(QREMethod_Multi_Particles(env,infoForEstimators["diffNs"],infoForEstimators["ranges"],givenPrior,minLambda,toSend))
						
						else:
							if domainName in ["2d-multi","baseball-multi","hockey-multi"]:
								self.estimators.append(JointMethodQRE(self.xskillsNormal,numPs,domainName,givenPrior,minLambda,toSend))
							else:
								self.estimators.append(JointMethodQRE(xskills,numPs,domainName,givenPrior,minLambda,toSend))


		self.estimators_list = []

		for each in self.estimators:

			names = each.getEstimatorName()
			
			if type(names) == str:			
				self.estimators_list.append(names)
				# print(names)
			else:
				for n in names:	
					# print(n)
					self.estimators_list.append(n)	
		
		# code.interact("...", local=dict(globals(), **locals()))			


	def printEstimators(self):
		for e in self.estimators:
			print("Estimator: " + str(e.getEstimatorName()) + "\n" )
			print("\txskills: " + str(e.xskills)  + "\n")

	
	def getCopyOfEstimators(self):

		# Get copy of the estimators
		copyEstimators = copy.deepcopy(self.estimators)

		# Reset each one (will reset the attributes from within the object)
		for e in copyEstimators:
			e.reset()

		# Will return a copy of the list that contains all the estimators
		return copyEstimators


