import sys,os
import pickle
import numpy as np
import code 

from scipy.signal import convolve2d

from pathlib import Path
from importlib.machinery import SourceFileLoader

from itertools import chain

import matplotlib.pyplot as plt
import plotly.express as px
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

import matplotlib.patches as patches
import plotly.graph_objects as go

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib


scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Testing{os.sep}testingUpdatePFE.py")[0]

module = SourceFileLoader("hockeyUtils",f"{mainFolderName}{os.sep}Estimators{os.sep}utils.py").load_module()
sys.modules["utils"] = module


module = SourceFileLoader("hockey.py",f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

np.set_printoptions(suppress=True)




def givenState(saveAt,i,row):

	saveAtOriginal = saveAt


	cmapStr = "nipy_spectral"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)

		
	print("\nRow: ",row)


	saveAt = f"{saveAtOriginal}{os.sep}{i}-{row}{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	if not os.path.exists(saveAt+"7-expev-otherPs"):
		os.mkdir(saveAt+"7-expev-otherPs")

	gridTargetsAngular = infoRows[row]["row"]["gridTargetsAngular"]

	dirs,elevations = infoRows[row]["row"]["dirs"],infoRows[row]["row"]["elevations"]
	delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

	middle = int(len(dirs)/2) - 1
	mean = [dirs[middle],elevations[middle]]

	Zs = infoRows[row]["row"]["gridUtilitiesComputed"]

	possibleTargets = listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]
	action = executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
	
	
	tempParticles = list(chain.from_iterable(allParticles[i]))


	N = len(tempParticles)


	PDFsPerXskill = {}
	EVsPerXskill = {}



	# LOAD IF FILE WITH INFO PRESENT
	if Path(f"{saveAt}info.pkl").is_file():
		
		print("Loading info...")

		with open(f"{saveAt}info.pkl","rb") as infile:
			loadedInfo = pickle.load(infile)

		PDFsPerXskill = loadedInfo["pdfs"]
		EVsPerXskill = loadedInfo["evs"]


	# OTHERWISE, COMPUTE
	else:

		print("Computing info...")

		for each in tempParticles:

			key = "|".join(map(str,each[:-1]))
			# print("Key: ", key)


			covMatrix = sys.modules["domain"].getCovMatrix(each[:2],each[2])
			pdfs = sys.modules["domain"].getNormalDistribution(rng,covMatrix,delta,mean,gridTargetsAngular)
			EVsPerXskill[key] = convolve2d(Zs,pdfs,mode="same",fillvalue=0.0)
			

			# FOR METHOD
			EVsPerXskill[key] = EVsPerXskill[key].flatten()
			PDFsPerXskill[key] = sys.modules["utils"].computePDF(x=action,means=possibleTargets,covs=np.array([covMatrix]*len(possibleTargets)))
			
			# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
			# This is because depending on the xskill/resolution combination, the pdf of
			# a given xskill may not show up in any of the resolution buckets 
			# causing then the pdfs not adding up to 1
			# (example: xskill of 1.0 & resolution > 1.0)
			# If the resolution is less than the xskill, the xskill distribution can be fully captured 
			# by the resolution thus avoiding problems.
			PDFsPerXskill[key] /= np.sum(PDFsPerXskill[key])



	temp = np.linspace(-3,1.6,10)
	pSkills = np.power(10,temp)



	for ii in range(len(tempParticles)):


		each = tempParticles[ii]

		key = "|".join(map(str,each[:-1]))
		# print("Key: ", key)

		# UPDATE
		pdfs = PDFsPerXskill[key]
		evs = EVsPerXskill[key]


		for p in pSkills:

			# Create copy of EVs 
			evsC = np.copy(evs)

			# To be used for exp normalization trick - find maxEV and * by p
			# To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
			# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
			b = np.max(evsC*p)

			# With normalization trick
			expev = np.exp(evsC*p-b)

			sumexp = np.sum(expev)


			# infoXY,info,label,cmap,title=""
			plotInfo(possibleTargets,expev,f"{saveAt}7-expev-otherPs{os.sep}{ii}-{each}-{p}.png",cmap,f"Particle: {np.round(each,4)} | P: {p:.4}\nB: {b:.4f} | sumexp: {sumexp}")



	del pdfs,evs
	

	PDFsPerXskill.clear()
	EVsPerXskill.clear()




def plotInfo(infoXY,info,label,cmap,title=""):

	norm = plt.Normalize(np.min(info),max(info))
	sm = ScalarMappable(cmap=cmap,norm=norm)
	sm.set_array([])

	fig,ax = plt.subplots()
	ax.scatter(infoXY[:,0],infoXY[:,1],c=cmap(norm(info)))
	plt.colorbar(sm,ax=ax)
	plt.title(title)
	plt.savefig(label,bbox_inches='tight')
	plt.tight_layout()
	plt.clf()
	plt.close("all")


def updatePlots(saveAt,agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs):


	cmapStr = "nipy_spectral"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)


	cmapStr2 = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap2 = (1. - c) * plt.get_cmap(cmapStr2)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap2 = ListedColormap(cmap2)


	saveAtOriginal = saveAt


	for i, row in enumerate(ids):
		
		print("\nRow: ",row)

		# FOR TESTING
		# i = 44
		# row = 271387021143


		saveAt = f"{saveAtOriginal}{os.sep}{i}-{row}{os.sep}"

		if not os.path.exists(saveAt):
			os.mkdir(saveAt)


		if not os.path.exists(saveAt+"1-expev"):
			os.mkdir(saveAt+"1-expev")

		gridTargetsAngular = infoRows[row]["row"]["gridTargetsAngular"]

		dirs,elevations = infoRows[row]["row"]["dirs"],infoRows[row]["row"]["elevations"]
		delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

		middle = int(len(dirs)/2) - 1
		mean = [dirs[middle],elevations[middle]]

		Zs = infoRows[row]["row"]["gridUtilitiesComputed"]

		possibleTargets = listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]
		action = executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
		
		
		tempParticles = list(chain.from_iterable(allParticles[i]))


		# FOR TESTING
		# tempParticles = tempParticles[:9]


		N = len(tempParticles)
		probs = np.ndarray(shape=(N,1))
		probs.fill(1/N)


		PDFsPerXskill = {}
		EVsPerXskill = {}


		# '''


		# LOAD IF FILE WITH INFO PRESENT
		if Path(f"{saveAt}info.pkl").is_file():
			
			print("Loading info...")

			with open(f"{saveAt}info.pkl","rb") as infile:
				loadedInfo = pickle.load(infile)

			PDFsPerXskill = loadedInfo["pdfs"]
			EVsPerXskill = loadedInfo["evs"]


		# OTHERWISE, COMPUTE
		else:

			print("Computing info...")

			for each in tempParticles:

				key = "|".join(map(str,each[:-1]))
				# print("Key: ", key)


				covMatrix = sys.modules["domain"].getCovMatrix(each[:2],each[2])
				pdfs = sys.modules["domain"].getNormalDistribution(rng,covMatrix,delta,mean,gridTargetsAngular)
				EVsPerXskill[key] = convolve2d(Zs,pdfs,mode="same",fillvalue=0.0)
				

				# FOR METHOD
				EVsPerXskill[key] = EVsPerXskill[key].flatten()
				PDFsPerXskill[key] = sys.modules["utils"].computePDF(x=action,means=possibleTargets,covs=np.array([covMatrix]*len(possibleTargets)))
				
				# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
				# This is because depending on the xskill/resolution combination, the pdf of
				# a given xskill may not show up in any of the resolution buckets 
				# causing then the pdfs not adding up to 1
				# (example: xskill of 1.0 & resolution > 1.0)
				# If the resolution is less than the xskill, the xskill distribution can be fully captured 
				# by the resolution thus avoiding problems.
				PDFsPerXskill[key] /= np.sum(PDFsPerXskill[key])



		# UPDATE STEP
		print("Update...")

		allUpds = []
		allSumexp = []
		allSummultexps = []

		for ii in range(len(tempParticles)):

			each = tempParticles[ii]

			key = "|".join(map(str,each[:-1]))
			# print("Key: ", key)

			# UPDATE
			pdfs = PDFsPerXskill[key]
			evs = EVsPerXskill[key]

			# If resulting posterior distribution for possible targets 
			# given xskill hyp & executed action results in all 0's
			# Means there's no way you'll be of this xskill
			# So no need to update probs, can remain 0.0
			if np.sum(pdfs) == 0.0 or np.isnan(np.sum(pdfs)):
				probs[xi] = [0.0] * len(each)
				# print(f"skipping (pdfs sum = 0) - x hyp: {x}")
				continue


			p = each[-1]

			# Create copy of EVs 
			evsC = np.copy(evs)

			# To be used for exp normalization trick - find maxEV and * by p
			# To avoid "overflow encountered in exp" warning for some p's and x's since numbers become too big for exp func when multiplied by EVs
			# As sugggested in: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
			b = np.max(evsC*p)

			# With normalization trick
			expev = np.exp(evsC*p-b)

			sumexp = np.sum(expev)
			allSumexp.append(sumexp)

			# JT Update 
			summultexps = np.sum(np.multiply(expev,np.copy(pdfs)))
			allSummultexps.append(summultexps)

			upd = summultexps/sumexp
			allUpds.append(upd)

			# Update probs (particle weights)
			probs[ii] *= upd


			# For a given particle, b
			# PLOT: expev
			if i < 50:
				plotInfo(possibleTargets,expev,f"{saveAt}1-expev{os.sep}{ii}-{each}.png",cmap2,f"Particle: {each} | B: {b}")


		del pdfs,evs
		

		# '''


		# Save info to file
		if not Path(f"{saveAt}info.pkl").is_file():

			info = {"pdfs":PDFsPerXskill, "evs":EVsPerXskill}

			with open(f"{saveAt}info.pkl","wb") as outfile:
				pickle.dump(info,outfile)



		PDFsPerXskill.clear()
		EVsPerXskill.clear()


		probs /= np.sum(probs)


		particlesNP = np.array(tempParticles)
		

		# '''

		# # PLOT: sumexp
		plotInfo(particlesNP,allSumexp,f"{saveAt}2-sumexp.png",cmap)

		# # PLOT: summultexps
		plotInfo(particlesNP,allSummultexps,f"{saveAt}3-summultexps.png",cmap)

		# # PLOT: upd
		plotInfo(particlesNP,allUpds,f"{saveAt}4-upd.png",cmap)

		# # PLOT: probs
		plotInfo(particlesNP,probs,f"{saveAt}5-probs.png",cmap)
		
		# '''



		indices = np.lexsort((particlesNP[:,0],particlesNP[:,1],particlesNP[:,2],particlesNP[:,3]))
		
		sortedParticles = particlesNP[indices]

		sortedProbs = probs[indices]
		sortedProbs = sortedProbs.flatten()

		cmapStr = "viridis"

		norm = mcolors.Normalize(vmin=min(sortedProbs), vmax=max(sortedProbs))
		colormap = matplotlib.colormaps[cmapStr]



		size1 = 50
		size2 = 20
		grid_size = size1 * size2
		square_size = 15.0 


		fig = go.Figure(layout=go.Layout(width=1300,height=900))

		x_centers = []
		y_centers = []
		hover_labels = []
		colors = []

		pi = 0

		# Loop to place squares in a grid
		for ii in range(size1):
			for jj in range(int(size2)):

				particle = sortedParticles[pi]
				prob = sortedProbs[pi] 
				color = mcolors.to_hex(colormap(norm(prob))) 
				colors.append(prob)

				pi += 1


				x = jj * square_size
				y = (grid_size - 1 - ii) * square_size  # Invert y-axis to have (0, 0) in bottom-left

				x0 = jj * (2 * square_size)  # Keep square size fixed
				y0 = (grid_size - 1 - ii) * (2 * square_size)

				x1 = x0 + (2 * square_size)
				y1 = y0 + (2 * square_size)



				fig.add_shape(
					type="rect",
					x0=x0, y0=y0, x1=x1, y1=y1,
					line=dict(color="black"),
					fillcolor=color
				)
				

				particleStr = list(map(lambda x: str(round(x,6)),particle))
				# print(particleStr)

				fig.add_trace(go.Scatter(
		            x=[x0, x1, x1, x0, x0],  # Define square corners
		            y=[y0, y0, y1, y1, y0],
		            fill="toself",  # Fully fills the shape
		            text=f"Particle: {','.join(particleStr)} | Probability: {prob:.10f}",
		            mode="lines",
		            hoverinfo="text",
		            opacity=0,  # Invisible, only used for hover
		            hoverlabel=dict(
			        bgcolor="white",  # Background color of the hover box
			        font=dict(color="black", size=14, family="Arial")),
			        showlegend=False

		        ))


				x_centers.append((x0 + x1) / 2)
				y_centers.append((y0 + y1) / 2)
				hover_labels.append(f"({particle})")  


		# To remove the whole axis
		fig.update_layout(
		    xaxis={'visible': False},
		    yaxis={'visible': False}
		)

		# To remove grid lines
		fig.update_layout(
		    xaxis={'showgrid': False},
		    yaxis={'showgrid': False}
		)

		# To remove tick labels
		fig.update_layout(
		    xaxis={'showticklabels': False},
		    yaxis={'showticklabels': False}
		)
				

		fig.update_layout(hovermode='closest')

		fig.add_trace(go.Scatter(
			x=x_centers, 
			y=y_centers, 
			mode="markers",
			marker=dict(
			    size=0.001,  # Small invisible points
			    color=colors,  # Use probability values for color mapping
			    colorscale=cmapStr,  # Same as the colormap
			    showscale=True,  # Show colorbar
			),

			hoverinfo="text",
			showlegend=False
		))



		fig.update_traces(marker=dict(
		    colorbar=dict(
		        title="Probability",
		        tickformat=".10f"  # Forces decimal format (e.g., 0.12 instead of 120µ)
		    )
		))



		# fig.show()
		fig.write_html(f"{saveAt}6-particles-probs.html")


		mi = np.argmax(probs)
		iis = np.unravel_index(mi,probs.shape)[0]
		estimate = tempParticles[iis]

		expected = np.average(tempParticles,weights=probs.flatten(),axis=0).tolist()
		ees, ers, eps = expected[0:2], expected[2], expected[3]

		print("-- COMPUTED --")
		print(f"EES:{ees}  |  MAP: {estimate[:-2]}")
		print(f"ERS:{ers}  |  MAP: {estimate[-2]}")
		print(f"EPS:{eps}  |  MAP: {estimate[-1]}")


		print("-- FROM EXPERIMENT --")
		print(f"EES:{estimatesES[i][0]}  |  MAP: {estimatesMS[i][0]}")
		print(f"ERS:{estimatesES[i][1]}  |  MAP: {estimatesMS[i][1]}")
		print(f"EPS:{estimatesES[i][2]}  |  MAP: {estimatesMS[i][2]}")

	
		# code.interact("...", local=dict(globals(), **locals()))



def update(agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs):


	PDFsPerXskill = {}
	EVsPerXskill = {}


	i = 0




	for info in zip(estimatesMS,estimatesES):
		
		row = ids[i]


		# FOR TESTING
		row = 271387021143
		i = 44

		info = (estimatesMS[i],estimatesES[i])


		print("Row: ",row)

		gridTargetsAngular = infoRows[row]["row"]["gridTargetsAngular"]

		dirs,elevations = infoRows[row]["row"]["dirs"],infoRows[row]["row"]["elevations"]
		delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

		middle = int(len(dirs)/2) - 1
		mean = [dirs[middle],elevations[middle]]

		Zs = infoRows[row]["row"]["gridUtilitiesComputed"]

		possibleTargets = listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]
		action = executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
		
		
		tempParticles = list(chain.from_iterable(allParticles[i]))


		# info += (([0.004,0.004],0.001,10),)


		for each in range(len(info)):
				
			particle = info[each]

			if each == 0:
				label = "MAP"
			elif each == 1:
				label = "EES"
			else:
				label = "Given"


			print("Looking at: ", label)


			# Rand initial prob
			prob = allProbs[0][0][0]

			particle = list(particle)
			particle = list(particle[0]) + particle[1:]
			print("Particle: ", particle)


			key = "|".join(map(str,particle[:2]))
			print("Key: ", key)

 
			covMatrix = sys.modules["domain"].getCovMatrix(particle[:2],particle[2])
			pdfs = sys.modules["domain"].getNormalDistribution(rng,covMatrix,delta,mean,gridTargetsAngular)


			EVsPerXskill[key] = convolve2d(Zs,pdfs,mode="same",fillvalue=0.0)
			


			# FOR METHOD
			EVsPerXskill[key] = EVsPerXskill[key].flatten()
			PDFsPerXskill[key] = sys.modules["utils"].computePDF(x=action,means=possibleTargets,covs=np.array([covMatrix]*len(possibleTargets)))

			
			# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
			# This is because depending on the xskill/resolution combination, the pdf of
			# a given xskill may not show up in any of the resolution buckets 
			# causing then the pdfs not adding up to 1
			# (example: xskill of 1.0 & resolution > 1.0)
			# If the resolution is less than the xskill, the xskill distribution can be fully captured 
			# by the resolution thus avoiding problems.

			#pdfs = np.multiply(pdfs,np.square(delta[0]*delta[1]))
			PDFsPerXskill[key] /= np.sum(PDFsPerXskill[key])



			# UPDATE
			pdfs = PDFsPerXskill[key]
			evs = EVsPerXskill[key]

			# If resulting posterior distribution for possible targets 
			# given xskill hyp & executed action results in all 0's
			# Means there's no way you'll be of this xskill
			# So no need to update probs, can remain 0.0
			if np.sum(pdfs) == 0.0 or np.isnan(np.sum(pdfs)):
				probs[xi] = [0.0] * len(particle)
				# print(f"skipping (pdfs sum = 0) - x hyp: {x}")
				continue


			p = particle[-1]
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
			prob *= upd

			print(f"Prob (computed, before norm): {prob:.20f}")

			if label == "MAP":
				pi = tempParticles.index(particle)

				# probs[i+1] since probs[0] = uniform random
				print("Prob (experiment): ", allProbs[i+1][pi])


			code.interact("...", local=dict(globals(), **locals()))

			del pdfs,evs


			PDFsPerXskill.clear()
			EVsPerXskill.clear()

		

		i += 1



if __name__ == '__main__':
	

	expFolder, rf = sys.argv[1], sys.argv[2]

	if expFolder[-1] != os.sep:
		expFolder += os.sep

	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{expFolder}{os.sep}Experiment{os.sep}"
	expFolder = f"{mainFolder}results{os.sep}"
	infoFolder = f"{mainFolder}info{os.sep}"
	saveAt = f"{mainFolder}plots{os.sep}"

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)



	with open(expFolder+rf,"rb") as infile:
		infoExperiment = pickle.load(infile)

	print(infoExperiment.keys())

	agent = infoExperiment["agent_name"]
	print(agent)


	# Previous version
	# saveAt += "-".join(agent[1]) + os.sep

	saveAt += "-".join(agent) + os.sep

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)


	saveAt += "Update" + os.sep

	if not os.path.exists(saveAt):
		os.mkdir(saveAt)



	rfInfo = rf.replace("OnlineExp_","info-").replace(".results","")
	# rfInfo += "_JEEDS.pkl"
	rfInfo += ".pkl"

	with open(infoFolder+rfInfo,"rb") as infile:
		infoRows = pickle.load(infile)

	print(infoRows.keys())



	# For Cluster exps since id list wasn't reset
	# ids = infoExperiment["ids"][-167:]

	ids = infoExperiment["ids"]
	print(len(ids))


	agent = agent[1]


	# method = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv50-JT-EES"
	method = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-EES"
	# method2 = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv50"
	method2 = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200-JT-MAP"
	method3 = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv200"
	resampleKey = method3 + "-whenResampled"

	noisyActions = infoExperiment["noisy_actions"]


	allParticles = infoExperiment[method3+"-allParticles"]
	# allParticlesNoNoise = infoExperiment[method3+"-allParticlesNoNoise"]
	# allNoises = infoExperiment[method3+"-allNoises"]
	allProbs = np.array(infoExperiment[method3+"-allProbs"])

	
	temp1 = infoExperiment[method+"-xSkills"]
	temp2 = infoExperiment[method+"-rhos"]
	temp3 = infoExperiment[method+"-pSkills"]
	estimatesES = list(zip(temp1,temp2,temp3))
	

	temp1 = infoExperiment[method2+"-xSkills"]
	temp2 = infoExperiment[method2+"-rhos"]
	temp3 = infoExperiment[method2+"-pSkills"]
	estimatesMS = list(zip(temp1,temp2,temp3))



	resampledInfo = infoExperiment[resampleKey]

	seed = infoExperiment["seedNum"]
	np.random.seed(seed)

	seeds = np.random.randint(0,1000000,1)
	rng = np.random.default_rng(seeds[0])


	# update(agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs)


	# updatePlots(saveAt,agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs)


	# i = 44
	# row = 271387021143

	# givenState(saveAt,i,row)


	code.interact("...", local=dict(globals(), **locals()))



