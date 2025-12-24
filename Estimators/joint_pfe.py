import numpy as np
import scipy
import time
import os,sys
from itertools import product

from scipy.stats import multivariate_t

from filterpy.monte_carlo import residual_resample, stratified_resample, systematic_resample, multinomial_resample


from Estimators.utils_pfe import *
from sklearn.cluster import KMeans


import matplotlib.pyplot as plt


class QREMethod_Multi_Particles():

    slots = ["xskills","numXskills","numPskills","domainName","names",
            "estimatesXskills","estimatesPskills","pskills",
            "probs","allProbs","label","estimatesRhos","indexes","methodType","subTypes",
            "particles","N","dimensions","ranges","alpha","otherArgs","rf","allParticles",
            "toPlot","percent","resampleNEFF","resamplingMethod","whenResampled","allParticlesNoNoise",
            "allNoises","allResampledProbs"]


    def __init__(self,env,N,noise,percent,resampleNEFF,resamplingMethod,ranges,givenPrior=False,minLambda=False,otherArgs=None):

        self.domainName = env.domain_name

        self.rf = env.resultsFolder

        self.seedNum = env.seedNum


        self.N = N
        self.dimensions = len(ranges["start"])
        self.particles = np.empty((N,self.dimensions))


        # Quantal Response (logit) inverse-temperature (lambda) grid.
        # We sample lambda on a log10 scale from 10^-3 to 10^1.6 (~0.001 → ~40).
        # Rationale: this spans very noisy behavior (near-random choice) up to
        # highly rational/peaked responses, which is a standard practical range
        # for inverse temperatures in QRE/softmax models.
        exponents = np.linspace(-3, 1.6, num=self.N)
        self.pskills = np.power(10, exponents)


        self.ranges = ranges

        WS = []

        for ei in range(len(self.ranges["start"])):

            start = self.ranges["start"][ei]
            stop = self.ranges["end"][ei]

            # if pskill dimension
            if ei == len(self.ranges["start"])-1:
                start = np.power(10,float(start))
                stop =  np.power(10,float(stop))

            # Assuming range in order (stop = bigger, start = smaller)
            W = stop-start
            WS.append(W)

        self.ranges["W"] = WS

        # TODO: This patch (next 23 lines) is AI-generated. Make sure it works. ----------------------------------------
        # Normalize noise specification across domains:
        # - If `noise` is a scalar, use the same divisor for all dims (incl. pskill)
        # - If `noise` has length 2, broadcast as [for all non-pskill dims] + [pskill dim]
        # - If `noise` has length == self.dimensions, use it as-is (one per dim)
        # This maintains compatibility for darts (historically scalar) and multi-dim sports.
        noise_list = None
        if isinstance(noise, (int, float, np.floating)):
            noise_list = [float(noise)] * self.dimensions
        elif isinstance(noise, (list, tuple, np.ndarray)):
            if len(noise) == 2:
                noise_list = [float(noise[0])] * (self.dimensions - 1) + [float(noise[1])]
            elif len(noise) == self.dimensions:
                noise_list = [float(x) for x in noise]
            else:
                raise ValueError(
                    f"noise length {len(noise)} incompatible with dimensions {self.dimensions}; expected 2 or {self.dimensions}"
                )
        else:
            raise TypeError(f"Unsupported noise type: {type(noise)}")

        self.ranges["noise"] = noise_list
        print(self.ranges["noise"])  # Debug: confirm per-dimension noise divisors
        # code.interact("...", local=dict(globals(), **locals()))



        # Percent of particles to resample
        self.percent = percent
        self.resampleNEFF = resampleNEFF

        self.whenResampled = []

        self.resamplingMethod = resamplingMethod


        #################################################
        # For plots
        #################################################

        # Initialize in case plotting is enabled
        self.alpha = .30

        if N > 5000:
            self.alpha *= np.sqrt(5000)/np.sqrt(N)

        #################################################


        # Initialize particles
        percent = 1.0 # To get full N initially

        # Running JEEDS via PFE
        if noise == -1 and self.percent == -1:
            self.particles = self.getParticlesJEEDS()
        else:
            self.particles = self.getUnifomParticles(self.N)

        self.allParticles = [[],self.particles.tolist()]
        self.allParticlesNoNoise = []

        self.allResampledProbs = []
        self.allNoises = []


        ll = ""
        if self.resampleNEFF:
            ll = "-ResampleNEFF"


        self.methodType = f"QRE-Multi-Particles-{self.N}-Resample{int(self.percent*100)}%{ll}-NoiseDiv{noise}"
        self.subTypes = ["-JT"]

        baseNames = [f"{self.methodType}{self.subTypes[0]}-MAP",
                    f"{self.methodType}{self.subTypes[0]}-EES"]


        self.label = ""

        if givenPrior and minLambda:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}-MinLambda{otherArgs['minLambda']}"
        elif givenPrior:
            self.label += f"-GivenPrior-{otherArgs['givenPrior'][0]}-{otherArgs['givenPrior'][1]}-{otherArgs['givenPrior'][2]}"
        elif minLambda:
            self.label += f"-MinLambda-{otherArgs['minLambda']}"

        for i in range(len(baseNames)):
            baseNames[i] += self.label

        self.names = baseNames

        self.estimatesXskills = dict()
        self.estimatesRhos = dict()
        self.estimatesPskills = dict()

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesPskills[n] = []


        # print(self.methodType)
        # code.interact("...", local=dict(globals(), **locals()))


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NEED TO UPDATE TO MULTIPLE DIMENSIONS
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # Initializing the array (init prior distribution)
        if givenPrior:
            pass

            # # Get "skewed" dist for the different xskill hyps
            # xProbs = skewnorm.pdf(self.xskills,a=otherArgs["givenPrior"][0],loc=otherArgs["givenPrior"][1],scale=otherArgs["givenPrior"][2])
            # xProbs = xProbs.reshape(xProbs.shape[0],1)

            # self.probs = xProbs.copy()
            # self.probs = self.probs.reshape(self.probs.shape[0],1)

            # # Fill the rest of the columns with the same value
            # # Assumption: All lambdas are equally likely (uniform distribution)
            # for i in range(self.numPskills-1):
            # 	self.probs = np.hstack((self.probs,xProbs))

            # self.probs /= np.sum(self.probs)

        else:
            # Running JEEDS via PFE
            if noise == -1 and self.percent == -1:
                self.probs = np.ndarray(shape=(len(self.particles),1))
                self.probs.fill(1/len(self.particles))
            else:
                self.probs = np.ndarray(shape=(self.N,1))
                self.probs.fill(1/self.N)


        # To save the initial probs
        self.allProbs = [self.probs.tolist()]

        # code.interact("init...", local=dict(globals(), **locals()))


    def getEstimatorName(self):
        return self.names


    def midReset(self):

        for n in self.names:
            self.estimatesXskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesPskills[n] = []

        self.allProbs = []
        self.allParticles = []
        self.allParticlesNoNoise = []
        self.whenResampled = []
        self.allResampledProbs = []
        self.allNoises = []


    def reset(self):

        for n in self.names:
            self.estimatesPskills[n] = []
            self.estimatesRhos[n] = []
            self.estimatesXskills[n] = []


        self.particles = np.empty((self.N,self.dimensions))

        percent = 1.0 # To get full N initially

        # Running JEEDS via PFE
        if self.percent == -1:
            self.particles = self.getParticlesJEEDS()
        else:
            self.particles = self.getUnifomParticles(self.N)

        self.allParticles = []
        self.allParticles.append([[],self.particles.tolist()])
        self.allParticlesNoNoise = []
        self.allResampledProbs = []
        self.allNoises = []


        self.whenResampled = []

        # Reset probs

        # Running JEEDS via PFE
        if self.percent == -1:
            self.probs.fill(1/len(self.particles))
            # code.interact("reset()...", local=dict(globals(), **locals()))
        else:
            self.probs.fill(1/self.N)


        self.allProbs = []
        self.allProbs.append(self.probs.tolist())


    # @profile
    def addObservation(self,rng,spaces,state,action,**otherArgs):

        startTimeEst = time.perf_counter()

        # timesEVs = []

        # t1 = time.perf_counter()

        # Update convolution space after setting new params
        for each in self.particles:
            # tt1 = time.perf_counter()

            # each[0] = np.round(each[0],4)
            # each[1] = np.round(each[1],4)
            # each[2] = np.round(each[2],4)

            spaces.updateSpaceParticles(rng,each,state,otherArgs)
            # timesEVs.append(time.perf_counter()-tt1)

        # print(timesEVs)
        # print(f"AVG time updateSpace - per particle: {sum(timesEVs)/len(timesEVs):.4f}")
        # print(f"Total time for updateSpace: {time.perf_counter()-t1:.4f}")


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Initialize info
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        resultsFolder = otherArgs["resultsFolder"]
        tag = otherArgs["tag"]
        delta = spaces.delta


        action = np.array(action)

        if self.domainName in ["2d-multi"]:
            listedTargets = spaces.listedTargets


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # t1 = time.perf_counter()
        # Plot initial set of particles
        if "plot" in otherArgs:
            # If plotting enabled
            if otherArgs["plot"] and otherArgs["i"] == 0:
                self.plotParticles(self.particles,[],otherArgs["agent"],"init")
        # print(f"Total time for plotParticles initial: {time.perf_counter()-t1:.4f}")


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Compute PDFs and EVs
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # t1 = time.perf_counter()

        PDFsPerXskill = {}
        EVsPerXskill = {}


        for each in self.particles:

            # t22 = time.perf_counter()
            # each[0] = np.round(each[0],4)
            # each[1] = np.round(each[1],4)
            # each[2] = np.round(each[2],4)
            key = "|".join(map(str,each[:-1]))
            # print(f"\tTotal time for creating key: {time.perf_counter()-t22:.4f}")


            if self.domainName in ["2d-multi"]:

                space = spaces.convolutionsPerXskill[key][otherArgs["s"]]
                evs = space["all_vs"].flatten()

                # t22 = time.perf_counter()
                pdfs = computePDF(x=action,means=listedTargets,covs=np.array([spaces.convolutionsPerXskill[key]["cov"]]*len(listedTargets)))
                # print(f"\tTotal time for computePDF: {time.perf_counter()-t22:.4f}")
                # code.interact("...", local=dict(globals(), **locals()))

                del space

            elif self.domainName in ["baseball-multi"]:

                evs = spaces.evsPerXskill[key].flatten()

                covMatrix = spaces.domain.getCovMatrix(each[:-2],each[-2])
                pdfs = computePDF(x=action,means=spaces.possibleTargetsFeet,covs=np.array([covMatrix]*len(spaces.possibleTargetsFeet)))

            elif self.domainName in ["hockey-multi"]:

                evs = spaces.evsPerXskill[key].flatten()
                covMatrix = spaces.domain.getCovMatrix(each[:-2],each[-2])
                pdfs = computePDF(x=action,means=spaces.possibleTargets,covs=np.array([covMatrix]*len(spaces.possibleTargets)))

                '''
                df = 3 

                pdfs = np.array([
                    multivariate_t.pdf(action,loc=mean,shape=covMatrix,df=df)
                    for mean in spaces.possibleTargets])
                '''


            # Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
            # This is because depending on the xskill/resolution combination, the pdf of
            # a given xskill may not show up in any of the resolution buckets
            # causing then the pdfs not adding up to 1
            # (example: xskill of 1.0 & resolution > 1.0)
            # If the resolution is less than the xskill, the xskill distribution can be fully captured
            # by the resolution thus avoiding problems.

            if self.domainName == "hockey-multi":
                #pdfs = np.multiply(pdfs,np.square(delta[0]*delta[1]))
                pdfs /= np.sum(pdfs)
            else:
                pdfs = np.multiply(pdfs,np.square(delta))


            # Save info
            PDFsPerXskill[key] = pdfs
            EVsPerXskill[key] = evs


            # if np.isnan(pdfs).any():
                # code.interact("...", local=dict(globals(), **locals()))


            del pdfs, evs


        # print(f"Total time for computing pdfs and evs: {time.perf_counter()-t1:.4f}")


        # t22 = time.perf_counter()
        # del space
        # del listedTargets
        # del pdfs
        # gc.collect()
        # print(f"Total time memory management: {time.perf_counter()-t22:.4f}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Perform Joint Update
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # t1 = time.perf_counter()

        for ii in range(len(self.particles)):

            each = self.particles[ii]

            p = each[-1]

            key = "|".join(map(str,each[:-1]))


            pdfs = PDFsPerXskill[key]
            evs = EVsPerXskill[key]


            # If resulting posterior distribution for possible targets
            # given xskill hyp & executed action results in all 0's
            # Means there's no way you'll be of this xskill
            # So no need to update probs, can remain 0.0
            if np.sum(pdfs) == 0.0 or np.isnan(np.sum(pdfs)):
                self.probs[xi] = [0.0] * len(each)
                # print(f"skipping (pdfs sum = 0) - x hyp: {x}")
                continue


            # print("\np: ", p)

            # Create copy of EVs
            evsC = np.copy(evs)
            # print("evsC: ",evsC)

            # To be used for exp normalization trick - find maxEV and * by p
            # To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
            # As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
            b = np.max(evsC*p)
            # print("b: ", b)

            # With normalization trick
            expev = np.exp(evsC*p-b)
            # print("expev: ", expev)

            sumexp = np.sum(expev)
            # print("sumexp: ", sumexp)

            # JT Update
            summultexps = np.sum(np.multiply(expev,np.copy(pdfs)))
            # print("summultexps: ", summultexps)

            upd = summultexps/sumexp
            # print("upd: ",upd)

            # Update probs (particle weights)
            self.probs[ii] *= upd


            del pdfs,evs


        # code.interact("...", local=dict(globals(), **locals()))


        PDFsPerXskill.clear()
        EVsPerXskill.clear()

        del PDFsPerXskill
        del EVsPerXskill

        # collected = gc.collect()
        # print(f"Garbage collector: collected {collected} objects.")

        # for item in gc.garbage:
            # print(item)

        # code.interact("...", local=dict(globals(), **locals()))

        # print(f"Total time for joint update: {time.perf_counter()-t1:.4f}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Normalize
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self.probs /= np.sum(self.probs)
        self.allProbs.append(self.probs.tolist())

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Get estimates - For Joint
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # t1 = time.perf_counter()

        # MAP estimate - Get index of maximum prob - returns flat index
        mi = np.argmax(self.probs)

        # "Converts a flat index or array of flat indices into a tuple of coordinate arrays."
        iis = np.unravel_index(mi,self.probs.shape)[0]

        estimate = self.particles[iis]

        self.estimatesXskills[self.names[0]].append(estimate[:-2].tolist())
        self.estimatesRhos[self.names[0]].append(estimate[-2])
        self.estimatesPskills[self.names[0]].append(estimate[-1])


        # Get Expected Estimate
        '''
        ees2 = [0.0]*len(self.estimatesXskills[self.names[0]][-1])
        ers2 = 0.0
        eps2 = 0.0

        for ii in range(self.N):

            each = self.particles[ii]

            xs, r, p = each[:-2],each[-2],each[-1]


            for d in range(len(xs)):
                ees2[d] += xs[d] * self.probs[ii][0]

            ers2 += r * self.probs[ii][0]
            eps2 += p * self.probs[ii][0]
        '''


        expected = np.average(self.particles,weights=self.probs.flatten(),axis=0).tolist()
        ees, ers, eps = expected[0:2], expected[2], expected[3]

        self.estimatesXskills[self.names[1]].append(ees)
        self.estimatesRhos[self.names[1]].append(ers)
        self.estimatesPskills[self.names[1]].append(eps)


        # '''
        print(self.methodType)
        print(f"EES:{ees}  |  MAP: {self.estimatesXskills[self.names[0]][-1]}")
        print(f"ERS:{ers}  |  MAP: {self.estimatesRhos[self.names[0]][-1]}")
        print(f"EPS:{eps}  |  MAP: {self.estimatesPskills[self.names[0]][-1]}")
        # code.interact("...", local=dict(globals(), **locals()))
        # '''

        # code.interact("...", local=dict(globals(), **locals()))

        # print(f"Total time for norm and estimates: {time.perf_counter()-t1:.4f}")

        # t1 = time.perf_counter()


        if self.percent != -1: # If not running JEEDS via PFE

            if self.resampleNEFF:

                thres = self.N/2
                neff = self.neff()

                # print("NEFF: ",neff)

                if neff <= thres:
                    # print("RESAMPLING")
                    label = "-resampled"
                    rr = True

                    temp1,temp2 = self.resample(rng,spaces,state,otherArgs["i"])
                    self.whenResampled.append(otherArgs["i"])
                else:
                    rr = False


            # Always resample
            else:
                # print("RESAMPLING")
                label = "-resampled"
                rr = True
                temp1,temp2 = self.resample(rng,spaces,state,otherArgs["i"])
                self.whenResampled.append(otherArgs["i"])

            # print(f"Total time for resampling: {time.perf_counter()-t1:.4f}")


            # Add noise regardless of resampling step
            # t1 = time.perf_counter()
            self.addNoise(rng)




            ###################################
            # TESTING ENFORCING SAME PSKILLS
            self.particles[:,-1] = self.pskills
            ####################################



            # print(f"Total time for adding noise: {time.perf_counter()-t1:.4f}")

        # Case running JEEDS via PFE
        else:
            rr = False



        # t1 = time.perf_counter()

        if "plot" in otherArgs:
            # If plotting enabled
            if otherArgs["plot"] and rr:
                self.plotParticles(temp1,temp2,otherArgs["agent"],f"{otherArgs['i']}{label}",[[ees,ers,eps],[self.estimatesXskills[self.names[0]][-1],self.estimatesRhos[self.names[0]][-1],self.estimatesPskills[self.names[0]][-1]]])


        if rr:
            # Save actual noisy version (but remember which
            # where resampled and which were randomly added)
            # self.allParticles.append([temp1.tolist(),temp2.tolist()])
            tempN = int(self.N*self.percent)
            self.allParticles.append([self.particles[:tempN,:].tolist(),self.particles[tempN:,:].tolist()])

        else:
            self.allParticles.append([self.particles.tolist(),[]])


        # print("allNoises: ",self.allNoises[-1][:2])
        # print("allParticles: ",self.allParticles[-1][0][:2])
        # code.interact("()...", local=dict(globals(), **locals()))

        # print(f"Total time for plotParticles: {time.perf_counter()-t1:.4f}")

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        endTimeEst = time.perf_counter()
        totalTimeEst = endTimeEst - startTimeEst


        # print(f"Total time addObservation(): {totalTimeEst}")
        # print()


        # collected = gc.collect()
        # print(f"Garbage collector: collected {collected} objects.")

        # for item in gc.garbage:
            # print(item)


        folder = resultsFolder

        #If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep)

        #If the folder doesn't exist already, create it
        if not os.path.exists("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators"):
            os.mkdir("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators")

        with open("Experiments" + os.path.sep + folder + os.path.sep + "times" + os.path.sep + "estimators" + os.path.sep + "JT-QRE-Times-"+ tag, "a") as file:
            file.write(str(totalTimeEst) + "\n")


    def getParticlesJEEDS(self):

        # Assuming only 2 dimensions and that the set of xskills is the same for both dimensions
        xskills = list(np.round(np.linspace(self.ranges["start"][0],self.ranges["end"][0],num=self.N),4))

        pskills = list(np.round(np.logspace(self.ranges["start"][-1],self.ranges["end"][-1],num=self.N),4))


        particles = np.vstack([xskills,xskills]).T.tolist()


        particles = list(product(particles,[0.0]))
        particles = list(product(particles,pskills))

        for ii in range(len(particles)):
            particles[ii] = eval(str(particles[ii]).replace(")","").replace("(","").replace("[","").replace("]",""))

        # code.interact("getParticlesJEEDS()...", local=dict(globals(), **locals()))

        return np.array(particles,dtype="object")


    def resample(self,rng,spaces,state,i):

        # t1 = time.perf_counter()

        # Remove spaces for previous set of particles (for memory management)
        # Removing them all since they'll be different
        # since they'll be resampled and noise will be added
        for each in self.particles:
            spaces.deleteSpaceParticles(each[:-1],state)


        # collected = gc.collect()
        # print(f"Garbage collector: collected {collected} objects.")
        # code.interact("resample()...", local=dict(globals(), **locals()))


        # print(f"Total time for deleteSpace: {time.perf_counter()-t1:.4f}")


        # t1 = time.perf_counter()

        tempN = int(self.N*self.percent)
        temp1,tempProbs = self.resampleFromProbs(rng,tempN,i)

        tempN = int(len(self.particles)-tempN)
        temp2 = self.getUnifomParticles(tempN)

        # New set of particles
        self.particles = np.concatenate((temp1,temp2),axis=0)

        self.allParticlesNoNoise.append(deepcopy(self.particles.tolist()))

        # print("allParticlesNoNoise: ",self.allParticlesNoNoise[-1][:2])

        self.allResampledProbs.append(tempProbs.tolist())

        # code.interact("...", local=dict(globals(), **locals()))

        # Reset weights
        self.probs.fill(1.0/self.N)


        # print(f"Total time for generating new particles: {time.perf_counter()-t1:.4f}")

        return temp1,temp2


    def getUnifomParticles(self,tempN,label=None):

        # tempN = ceil(self.N*percent)
        tempParticles = np.empty((tempN,self.dimensions))

        exponents = np.linspace(-3, 1.6, num=tempN)
        pskills = np.power(10, exponents)

        for d in range(self.dimensions):
            # Not including endpoint

            # For pskill dimension
            if d == self.dimensions-1:
                # temp = np.random.uniform(self.ranges["start"][d],self.ranges["end"][d],size=tempN)
                # tempParticles[:,d] = np.power(10,temp) # Exponentiate

                tempParticles[:,d] = pskills

            else:
                tempParticles[:,d] = np.round(np.random.uniform(self.ranges["start"][d],self.ranges["end"][d],size=tempN),4)

        return tempParticles


    # @profile
    def plotParticles(self,t1,t2,tag,l,info=None):

        toPlot = []
        folders = ["ParticleFilter",f"ParticleFilter{os.sep}{tag}"]

        for each1 in [self.methodType]:
            folders.append(f"ParticleFilter{os.sep}{tag}{os.sep}{each1}{os.sep}")
            for each2 in ["xskills","rhos","pskills"]:
                folders.append(f"ParticleFilter{os.sep}{tag}{os.sep}{each1}{os.sep}{each2}")
                toPlot.append(f"{tag}{os.sep}{each1}{os.sep}{each2}")

        for each in folders:
            if not os.path.exists(f"Experiments{os.sep}{self.rf}{os.sep}{each}"):
                os.mkdir(f"Experiments{os.sep}{self.rf}{os.sep}{each}")


        for each in toPlot:

            fig = plt.figure(num=1,clear=True)
            ax = fig.add_subplot()

            if "xskills" in each:
                ax.scatter(t1[:,0],t1[:,1],alpha=self.alpha,color='g')
                if type(t2) != list:
                    ax.scatter(t2[:,0],t2[:,1],alpha=self.alpha,color='b')
            elif "rhos" in each:
                ax.scatter(t1[:,-2],[0]*len(t1),alpha=self.alpha,color='g')
                if type(t2) != list:
                    ax.scatter(t2[:,-2],[0]*len(t2),alpha=self.alpha,color='b')
            else:
                ax.scatter(t1[:,-1],[0]*len(t1),alpha=self.alpha,color='g')
                if type(t2) != list:
                    ax.scatter(t2[:,-1],[0]*len(t2),alpha=self.alpha,color='b')

            if info != None:
                tt = f"EES:{np.round(info[0][0],4)} | ERS:{np.round(info[0][1],4)} | EPS:{np.round(info[0][2],4)}\n"
                tt += f"MES:{np.round(info[1][0],4)} | MRS:{np.round(info[1][1],4)} | MPS:{np.round(info[1][2],4)}"
                plt.title(tt)

            fig.savefig(f"Experiments{os.sep}{self.rf}{os.sep}ParticleFilter{os.sep}{each}{os.sep}pf-{l}.png",bbox_inches = 'tight')
            plt.clf()
            plt.close("all")

            del fig, ax
            gc.collect()


    def resampleFromProbs(self,rng,tempN,i):

        # '''

        # tempN = int(self.N*percent)
        # tempParticles = rng.choice(self.particles,size=tempN,replace=True,p=self.probs.flatten())

        # multinomial_resample = numpy's choice
        if self.resamplingMethod == "numpy":
            tempIndexes = rng.choice(range(self.N),size=tempN,replace=True,p=self.probs.flatten())

        elif self.resamplingMethod == "systematic":
            tempIndexes = systematic_resample(weights=self.probs.flatten())

        elif self.resamplingMethod == "stratified":
            tempIndexes = stratified_resample(weights=self.probs.flatten())

        elif self.resamplingMethod == "residual":
            tempIndexes = residual_resample(weights=self.probs.flatten())


        # tempIndexes = rng.choice(tempIndexes2,size=tempN)

        tempParticles = self.particles[tempIndexes]
        tempProbs = self.probs[tempIndexes]

        # np.set_printoptions(suppress=True)
        # print(tempParticles)
        # print(tempProbs)

        # code.interact("resample...", local=dict(globals(), **locals()))

        return tempParticles,tempProbs


        # '''



        '''

        temp = np.linspace(-3,1.6,10)
        givenPskills = np.power(10,temp)


        # [0] since all the same
        noise = self.ranges["noise"][0]
        WX = self.ranges["W"][0]
        WR = self.ranges["W"][2]
        WP = self.ranges["W"][3]


        particlesR = self.particles[:tempN,:]
        probsR = self.probs[:tempN]


        K = 7 
        kmeans = KMeans(n_clusters=K, n_init='auto').fit(particlesR[:,:2])
        labels = kmeans.labels_


        new_particles = np.zeros_like(particlesR)
        new_probs = np.zeros_like(probsR)

        for k in range(K):

            idxs = np.where(labels==k)[0]

            if len(idxs) == 0:
                continue

            # Get weights within this cluster
            cluster_weightsPrev = probsR[idxs]
            cluster_weights = cluster_weightsPrev/cluster_weightsPrev.sum()  
            cluster_weights = cluster_weights.flatten()
                
            # Resample from cluster members
            resampled_idxs = np.random.choice(idxs,size=len(idxs),replace=True, p=cluster_weights)
            new_particles[idxs,:2] = particlesR[resampled_idxs,:2] 

            new_particles[idxs,0] = particlesR[resampled_idxs,0] + np.random.normal(0,WX/noise,size=len(idxs))
            new_particles[idxs,1] = particlesR[resampled_idxs,1] + np.random.normal(0,WX/noise,size=len(idxs))


            new_probs[idxs] = probsR[resampled_idxs]


            # Step 3: Cross-dim bias – perturb within cluster
            rhos = particlesR[resampled_idxs,2]
            pskills = particlesR[resampled_idxs,3]


            new_particles[idxs,2] = rhos + np.random.normal(0,WR/noise,size=len(idxs))


            # If they amount to less than 20% of the prob???
            if cluster_weightsPrev.sum() < 0.20:

                print(f"Resampling - w sum: {cluster_weightsPrev.sum()}")

                eps = 1e-3

                # Discard seen pskills
                mask = np.all(np.abs(pskills[:,None]-givenPskills[None,:])>eps, axis=1)

                # Filtered result
                filteredPskills = pskills[mask]
                filteredPskills.reshape((len(filteredPskills),1))

                
                diff = abs(len(filteredPskills)- len(pskills))
                restPskills = np.random.choice(givenPskills,size=diff,replace=True)
                restPskills.reshape((len(restPskills),1))


                new_particles[idxs,3] = np.concatenate((filteredPskills,restPskills),axis=0)
                new_particles[idxs,3] +=  np.random.normal(0,WP/noise,size=len(idxs))
            
            else:
                new_particles[idxs,3] = pskills + np.random.normal(0,WP/noise,size=len(idxs))



        fig = plt.figure(num=0,figsize=(16,9))

        plt.subplots_adjust(wspace=0.3,hspace=0.4)

        ax1 = plt.subplot2grid((3,2),(0,0))
        ax2 = plt.subplot2grid((3,2),(1,0))
        ax3 = plt.subplot2grid((3,2),(2,0))
        ax4 = plt.subplot2grid((3,2),(0,1))
        ax5 = plt.subplot2grid((3,2),(1,1))
        ax6 = plt.subplot2grid((3,2),(2,1))


        N = tempN

        ax1.scatter(self.particles[:tempN,0],self.particles[:tempN,1])
        ax1.set_title("Execution Skill")

        ax2.scatter(range(N),self.particles[:tempN,2])
        ax2.set_title("Rhos")

        ax3.scatter(range(N),self.particles[:tempN,3])
        ax3.set_title("Dec-Making Skills")


        ax4.scatter(new_particles[:,0],new_particles[:,1])
        ax4.set_title("Execution Skill")

        ax5.scatter(range(N),new_particles[:,2])
        ax5.set_title("Rhos")

        ax6.scatter(range(N),new_particles[:,3])
        ax6.set_title("Dec-Making Skills")

        fig.suptitle("Prev | Resampled")


        plt.savefig(f"Experiments{os.sep}{self.rf}{os.sep}{i}-particles.png")
        plt.clf()
        plt.close()


        return new_particles, probsR


        '''


    def addNoise(self,rng):

        mean = [0.0]*(len(self.particles[0])-2)

        tempAllNoises = []


        for ii in range(len(self.particles)):

            each = self.particles[ii]

            stdDevs = each[:-2]
            rho = each[-2]
            ps = each[-1]


            noisy = []
            noises = []

            for ei in range(len(each)):

                start = self.ranges["start"][ei]
                stop = self.ranges["end"][ei]

                # if pskill dimension
                if ei == len(self.ranges["start"])-1:
                    start = np.power(10,float(start))
                    stop =  np.power(10,float(stop))


                if ei == len(self.ranges["start"])-1:
                    noise = 0.0
                else:
                    # Drawn gaussian
                    noise = rng.normal(0.0,(self.ranges["W"][ei]/self.ranges["noise"][ei]))

                # Drawn uniform random
                # number = np.sqrt(3)*(self.ranges["W"][ei]/self.ranges["noise"][ei])
                # noise = each[ei]+rng.uniform(low=-number,high=number,size=1)[0]


                upd = each[ei]+noise

                # Truncate (to ensure within range)
                if upd <= start:
                    upd = start

                if upd >= stop:
                    upd = stop

                noisy.append(upd)
                noises.append(noise)


            # code.interact("addNoise()...", local=dict(globals(), **locals()))

            self.particles[ii] = noisy

            tempAllNoises.append(noises)


        self.allNoises.append(tempAllNoises)

        # self.particles = np.round(self.particles,4)

        # code.interact("addNoise()...", local=dict(globals(), **locals()))


    # Calculate the effective # of particles
    def neff(self):
        return 1./np.sum(np.square(self.probs))


    def getResults(self):

        results = dict()

        for n in self.names:
            results[f"{n}-xSkills"] = self.estimatesXskills[n]
            results[f"{n}-rhos"] = self.estimatesRhos[n]
            results[f"{n}-pSkills"] = self.estimatesPskills[n]

        results[f"{self.methodType}{self.label}-allProbs"] = self.allProbs
        results[f"{self.methodType}{self.label}-allParticles"] = self.allParticles
        results[f"{self.methodType}{self.label}-allParticlesNoNoise"] = self.allParticlesNoNoise
        results[f"{self.methodType}{self.label}-whenResampled"] = self.whenResampled
        results[f"{self.methodType}{self.label}-allResampledProbs"] = self.allResampledProbs
        results[f"{self.methodType}{self.label}-allNoises"] = self.allNoises
        results[f"{self.methodType}{self.label}-resamplingMethod"] = self.resamplingMethod

        return results



