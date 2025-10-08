import sys,os
import code
import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

from scipy.signal import fftconvolve


import urllib.request
from hockey_rink import NHLRink


from pathlib import Path
from importlib.machinery import SourceFileLoader

scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}makePlotAllInfo.py")[0]

module = SourceFileLoader("hockey",f"{mainFolderName}{os.sep}Estimators{os.sep}utils.py").load_module()
sys.modules["utils"] = module



def getCovMatrix(stdDevs,rho):
	# print("stdDevs: ",stdDevs)
	# print("rho",rho)

	covMatrix = np.zeros((len(stdDevs),len(stdDevs)))
	# code.interact("get...", local=dict(globals(), **locals()))
	np.fill_diagonal(covMatrix,np.square(stdDevs))

	# Fill the upper and lower triangles
	for i in range(len(stdDevs)):
		for j in range(i+1,len(stdDevs)):
			covMatrix[i,j] = np.prod(stdDevs) * rho
			covMatrix[j,i] = covMatrix[i,j]


	# print(stdDevs,covMatrix)
	return covMatrix



def getBoardPlots(saveAt1,saveAt2):

	cmapStr = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)


			
	rng = np.random.default_rng(np.random.randint(1,1000000000))

	startX_Estimator = 0.004
	stopX_Estimator = np.pi/4

	xSkills = list(np.round(np.linspace(startX_Estimator,stopX_Estimator,num=33),4))
	print(xSkills)


	allActions = []
	allActionsAngular = []


	for i, row in enumerate(ids):

		print(f"({i+1}/{len(ids)}): Looking at row = {row}")

		listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]

		executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
		allActionsAngular.append(executedActionAngular)

		executedActionOriginal = [infoRows[row]["row"]["shot_location"][0],infoRows[row]["row"]["shot_location"][1]]
		allActions.append(executedActionOriginal)

		dirs,elevations = infoRows[row]["row"]["dirs"],infoRows[row]["row"]["elevations"]
		delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

		Zs = infoRows[row]["row"]["gridUtilitiesComputed"]

		EVs = infoRows[row]["evsPerXskill"]


		pdfsPerXskill = {}


		for xi, x in enumerate(xSkills):

			r = 0.0
			key = "|".join(map(str,[x,x]))+f"|{r}"
			cov = getCovMatrix([x,x],r)

			pdfs = sys.modules["utils"].computePDF(x=executedActionAngular,means=listedTargetsAngular,covs=np.array([cov]*len(listedTargetsAngular)))
			pdfs /= np.sum(pdfs)

			pdfsPerXskill[key] = pdfs

			fig,ax = plt.subplots()
			cs = ax.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c = pdfs)

			ax.set_xlabel("Direction",fontweight='bold')
			ax.set_ylabel("Elevation",fontweight='bold')

			plt.savefig(f"{saveAt1}{os.sep}{i}-{row}-{xi}-{x}.jpg",bbox_inches="tight")
			
			plt.close()
			plt.clf()


		rhos = [0.0]



		maxEV = -99999

		for x in xSkills:
			for r in rhos:
				key = "|".join(map(str,[x,x]))+f"|{r}"

				tempMax = np.max(EVs[key])
				if tempMax > maxEV:
					maxEV = tempMax



		norm = plt.Normalize(0.0,np.max(maxEV))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])


		for xi, x in enumerate(xSkills):

			for r in rhos:

				key = "|".join(map(str,[x,x]))+f"|{r}"


				fig,ax = plt.subplots()

				# x axis = elevation
				# y axis = direction

				cbar = plt.colorbar(sm,ax=ax)
				cbar.ax.get_yaxis().labelpad = 15
				cbar.ax.set_ylabel("Expected Utilities",rotation = 270)

				ax.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c = cmap(norm(EVs[key].flatten())))
				ax.set_xlabel("Direction",fontweight='bold')
				ax.set_ylabel("Elevation",fontweight='bold')

				maxEV = np.max(EVs[key])	
				mi = EVs[key].argmax()
				action = [listedTargetsAngular[:,0][mi],listedTargetsAngular[:,1][mi]]

				ax.scatter(action[0],action[1],color="black",marker="X",s=60,edgecolors="black",label="Max Expected Utility")

				titleStr = f"Execution Skill: {key}"
				plt.legend()

				plt.savefig(f"{saveAt2}{i}-{row}-{xi}-{key}.jpg",bbox_inches="tight")
				plt.close()
				plt.clf()


				# code.interact("after...", local=dict(globals(), **locals()))



def makePlotsFirstColumn(ax1,ax2,ax3,ax4,cmap,playerLocation,action,executedAction,executedActionOriginal,rfunc,rfuncAngular,listedTargetsAngular,dirs,elevations):


	######################################################
	# ax1 = rink
	######################################################

	rink = NHLRink()

	rink.draw(ax=ax1)

	rink.arrow(ax=ax1,
	x=playerLocation[0], y=playerLocation[1],
	x2=executedAction[0], y2=executedAction[1],
	facecolor="lime", edgecolor="black",
	head_width=4, length_includes_head=True)

	######################################################



	######################################################
	# ax2 = shot closer view
	######################################################

	goalW = 6 # feet
	goalX = 89

	leftPost = np.array([goalX,-3])
	rightPost = np.array([goalX,3])


	# Draw goal
	goal = Rectangle((goalX,leftPost[1]),0.5,goalW, edgecolor='black', facecolor='lightgray')
	ax2.add_patch(goal)

	# Draw puck
	ax2.plot(playerLocation[0],playerLocation[1],'ko',label="Player Location")

	ax2.plot([playerLocation[0],leftPost[0]],[playerLocation[1],leftPost[1]], 'r--', label='Left')
	ax2.plot([playerLocation[0],rightPost[0]],[playerLocation[1],rightPost[1]], 'b--', label='Right')

	# Add annotations
	# ax2.text(goalX+1, rightPost[1], 'Goal', verticalalignment='bottom')
	# ax2.text(playerLocation[0],playerLocation[1]+1,'Puck',horizontalalignment='center')

	# Axis limits and labels
	# ax2.set_xlim(playerLocation[0]-10,goalX+5)

	# offset = 5
	# ax2.set_ylim(playerLocation[1]-offset,playerLocation[1]+offset)

	ax2.set_xlabel('X')
	ax2.set_ylabel('Y')
	ax2.legend()

	######################################################



	######################################################
	# ax3 = reward func - regular info
	######################################################

	norm = plt.Normalize(0.0,np.max(rfunc))
	sm = ScalarMappable(norm=norm,cmap=cmap)
	sm.set_array([])

	# x axis = Y
	# y axis = Z

	ax3.scatter(listedTargetsUtilityGridYZ[:,0],listedTargetsUtilityGridYZ[:,1],c=cmap(norm(rfunc)))
	
	# pos = ax3.get_position()

	# Create a new axes for the colorbar
	# [left, bottom, width, height]
	# cbar_ax = fig.add_axes([pos.x0,pos.y0-0.05,pos.width,0.02])
	# code.interact("...", local=dict(globals(), **locals()))

	# plt.colorbar(sm,cax=cbar_ax,orientation='horizontal',shrink=0.75)
	plt.colorbar(sm,ax=ax3)

	minY = np.min(listedTargetsUtilityGridYZ[:,0])
	maxY = np.max(listedTargetsUtilityGridYZ[:,0])
	minZ = np.min(listedTargetsUtilityGridYZ[:,1])
	maxZ = np.max(listedTargetsUtilityGridYZ[:,1])

	ax3.scatter(minY,minZ,c="gray")
	ax3.scatter(minY,maxZ,c="gray")
	ax3.scatter(maxY,minZ,c="gray")
	ax3.scatter(maxY,maxZ,c="gray")

	ax3.scatter(executedActionOriginal[0],executedActionOriginal[1],marker="X",c="black")

	ax3.set_xlabel("Y",fontweight='bold')
	ax3.set_ylabel("Z",fontweight='bold')

	######################################################



	######################################################
	# ax4 = reward func - angular info
	######################################################

	wd = abs(dirs[0]-dirs[-1])
	we = abs(elevations[0]-elevations[-1])

	norm = plt.Normalize(0.0,np.max(rfuncAngular))
	sm = ScalarMappable(norm=norm,cmap=cmap)
	sm.set_array([])

	# x axis = elevation
	# y axis = direction

	ax4.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=cmap(norm(rfuncAngular)))
	plt.colorbar(sm,ax=ax4)

	minD = np.min(listedTargetsAngular[:,0])
	maxD = np.max(listedTargetsAngular[:,0])
	minE = np.min(listedTargetsAngular[:,1])
	maxE = np.max(listedTargetsAngular[:,1])

	ax4.scatter(minD,minE,c="gray")
	ax4.scatter(minD,maxE,c="gray")
	ax4.scatter(maxD,minE,c="gray")
	ax4.scatter(maxD,maxE,c="gray")
	ax4.scatter(action[0],action[1],marker="X",c="black")

	ax4.set_xlabel(f"Direction (w = {wd:.4f} rad)",fontweight='bold')
	ax4.set_ylabel(f"Elevation (w = {we:.4f} rad)",fontweight='bold')

	######################################################



def plotParticles(saveAt,agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs):

	# print(resampledInfo)
	# offset resampled info by 1
	# bc remembering when resampled
	# but particles are resampled & then are used on the next iter
	# (iter num which starts at 1)
	# so if resampling = [3]
	# means that resampling occured after observation 3
	# for particles to be used at next iter 4
	resampledInfo = list(map(lambda x:x+1,resampledInfo))

	print("(+1) resampledInfo: ", resampledInfo)


	cmapStr = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)



	N = len(allParticles[0])

	alpha = .30

	if N > 5000:
		alpha *= np.sqrt(5000)/np.sqrt(N) 

	cmap2 = plt.get_cmap("viridis")




	folders = [f"{saveAt}{os.sep}{os.sep}{method}{os.sep}",
			f"{saveAt}{os.sep}{os.sep}{method}-JustResampled{os.sep}",
			f"{saveAt}{os.sep}{os.sep}{method}-Probs{os.sep}",
			f"{saveAt}{os.sep}{os.sep}{method}-Probs-JustResampled{os.sep}",
			f"{saveAt}{os.sep}{os.sep}{method}-Particles-OnX{os.sep}",
			f"{saveAt}{os.sep}{os.sep}{method}-Particles-OnY{os.sep}"]

	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)



	for i,row in enumerate(ids):

		tempStr = ""

		listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]
		
		rfunc = infoRows[row]["row"]["heat_map"]
		shape = rfunc.shape
		rfunc = rfunc.reshape((shape[0]*shape[1],1))


		rfuncAngular = infoRows[row]["row"]["listedUtilitiesComputed"]

		dirs, elevations = infoRows[row]["row"]["dirs"], infoRows[row]["row"]["elevations"]


		playerLocation = [infoRows[row]["row"]["start_x"],infoRows[row]["row"]["start_y"]]
		executedAction = [89,infoRows[row]["row"]["shot_location"][0]]

		# shot_location = final_y, projected_z, start_x, start_y
		executedActionOriginal = [infoRows[row]["row"]["shot_location"][0],infoRows[row]["row"]["shot_location"][1]]


		action = noisyActions[i]
		executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
		

		d,e = action

		# Step 1
		xp = 89 - playerLocation[0]
		deltaY = xp * np.tan(d)
		D = xp / np.cos(d)

		# Step 2
		deltaZ = D * np.tan(e)

		executedActionComputed = [playerLocation[1]+deltaY,deltaZ]

		print("\ni: ",i)
		print("\nrow: ",row)
		print("playerLocation: ", playerLocation)
		print("executedAction: ", executedActionOriginal)
		print("executedActionAngular (from noisy actions): ", action)
		print("executedActionAngular (from file): ", executedActionAngular)
		print("executedActionComputed: ", executedActionComputed)


		# Not skipping initial info since initial set of particles used for 1st obs
		tempAllParticles = allParticles[i]

		# +1 to use probs from update with current observation (skipping uniform random in the fist position)
		tempAllProbs = allProbs[i+1]

		tempEstimates = [estimatesES[0][i],estimatesES[1][i],estimatesES[2][i]]


		resampled = tempAllParticles[0]
		random = tempAllParticles[1]

		s1 = len(resampled)
		s2 = len(random)



		for ii in range(6):

			saveAt = folders[ii]


			# allParticles = [ [[],initialRand] + [resampled,random] per particle] ]


			fig = plt.figure(num=0,figsize=(30,18))

			# Add a rectangle behind the first column
			row_rect = Rectangle(
			    (0,0),  # x, y (bottom left corner)
			    1/4,         # width (normalized)
			    1,       # height (normalized for row size)
			    color='lightgray',
			    zorder=-1   # Send behind the plots
			)
			fig.add_artist(row_rect)


			# Add a rectangle behind the third column
			row_rect = Rectangle(
			    (2/4,0),  # x, y (bottom left corner)
			    1/4,         # width (normalized)
			    1,       # height (normalized for row size)
			    color='lightgray',
			    zorder=-1   # Send behind the plots
			)
			fig.add_artist(row_rect)


			# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
			plt.subplots_adjust(wspace=0.3,hspace=0.4)

			ax1 = plt.subplot2grid((4,4),(0,0))
			ax2 = plt.subplot2grid((4,4),(1,0))

			ax3 = plt.subplot2grid((4,4),(0,1))
			ax4 = plt.subplot2grid((4,4),(1,1))

			ax5 = plt.subplot2grid((4,4),(0,2))
			ax6 = plt.subplot2grid((4,4),(1,2))
			ax7 = plt.subplot2grid((4,4),(2,2))

			ax8 = plt.subplot2grid((4,4),(0,3))
			ax9 = plt.subplot2grid((4,4),(1,3))
			ax10 = plt.subplot2grid((4,4),(2,3))
			ax11 = plt.subplot2grid((4,4),(3,3))


			ax1.set_title("Rink",fontweight='bold')
			ax2.set_title("Shot",fontweight='bold')

			ax3.set_title("Reward Function",fontweight='bold')
			ax4.set_title("Reward Function - Angular Coordinates",fontweight='bold')
			
			ax5.set_title("Execution Skill",fontweight='bold')
			ax6.set_title("Rhos ",fontweight='bold')
			ax7.set_title("Dec-Making Skill",fontweight='bold')


			makePlotsFirstColumn(ax1,ax2,ax3,ax4,cmap,playerLocation,action,executedAction,executedActionOriginal,rfunc,rfuncAngular,listedTargetsAngular,dirs,elevations)



			######################################################
			# PARTICLES
			######################################################


			if ii in [4,5]:


				if type(resampled) != list:
					resampled = resampled.tolist()

				if type(random) != list:
					random = random.tolist()


				toPlot = np.array(resampled+random)


				if i+1 in resampledInfo:
					c = 'tab:green'
				# No resampling occured, just noise added
				else:
					c = 'tab:orange'

				toPlotColors = [c]*len(resampled) + ["tab:blue"]*len(random)

				ax5.scatter(toPlot[:,0],toPlot[:,1],c=toPlotColors)

				if ii == 4:
					ax6.scatter(list(range(len(toPlot))),toPlot[:,-2],c=toPlotColors)
					ax7.scatter(list(range(len(toPlot))),toPlot[:,-1],c=toPlotColors)
				else:
					ax6.scatter(toPlot[:,-2],list(range(len(toPlot))),c=toPlotColors)
					ax7.scatter(toPlot[:,-1],list(range(len(toPlot))),c=toPlotColors)


				forLegend = []
				labels = []

				if "tab:green" in toPlotColors:
					forLegend.append("tab:green")
					labels.append("Resampled")
			
				if "tab:orange" in toPlotColors:
					forLegend.append("tab:orange")
					labels.append("Noisy")

				if "tab:blue" in toPlotColors:
					forLegend.append("tab:blue")
					labels.append("Random")


				legend_patches = [Patch(color=forLegend[i], label=labels[i]) for i in range(len(forLegend))]
				ax7.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=len(labels))


			else:

				# For the rest of the plots/iters

				colors = []
				labels = []

				norm = plt.Normalize(0.0,max(tempAllProbs))
				sm = ScalarMappable(cmap=cmap2,norm=norm)
				sm.set_array([])


				# For resampled particles
				if len(resampled) > 0:
					resampled = np.asarray(resampled)

					# whenResampled using otherInfo['i'] and i starts at 1!
					# i + 1 cause i is going per observation (starting at 0)
					if i+1 in resampledInfo:
						c = 'tab:green'
					# No resampling occured, just noise added
					else:
						c = 'tab:orange'
					
					# PROBS
					if ii >= 2:
						ax5.scatter(resampled[:,0],resampled[:,1],alpha=alpha,c=cmap2(norm(tempAllProbs[:s1,:])))
						ax6.scatter(resampled[:,-2],[0]*len(resampled),alpha=alpha,c=cmap2(norm(tempAllProbs[:s1,:])))
						ax7.scatter(resampled[:,-1],[0]*len(resampled),alpha=alpha,c=cmap2(norm(tempAllProbs[:s1,:])))
					else:

						# minSize, maxSize = 10, 100
						# sizes = minSize + tempAllProbs[:s1]*(maxSize-minSize)
						sizes = 100 + tempAllProbs[:s1] * 490
						
						ax5.scatter(resampled[:,0],resampled[:,1],s=sizes,alpha=alpha,color=c)
						ax6.scatter(resampled[:,-2],[0]*len(resampled),s=sizes,alpha=alpha,color=c)
						ax7.scatter(resampled[:,-1],[0]*len(resampled),s=sizes,alpha=alpha,color=c)

						if c == 'tab:green':
							tempL = "Resampled"
						else:
							tempL = "Noisy"

						colors.append(c)
						labels.append(tempL)


						#######
						# EES just resampled particles
						#######

						# if i+1 in resampledInfo:
						# 	expected = np.average(resampled,weights=tempAllProbs[:s1,:].flatten(),axis=0).tolist()
						# 	ees, ers, eps = expected[0:2], expected[2], expected[3]

						# 	tempStr = f"EES - Just Resampled: ({ees[0]},{ees[1]},{ers},{eps})\n"


				# If not just resampled plot
				if ii != 1 and ii != 3:

					# code.interact("...", local=dict(globals(), **locals()))

					c = 'tab:blue'

					if type(random) != list:
						random = random.tolist()

					# For random particles
					if random != []:
						random = np.asarray(random)
						
						# PROBS
						if ii == 2:
							ax5.scatter(random[:,0],random[:,1],alpha=alpha,c=cmap2(norm(tempAllProbs[s1:,:])))
							ax6.scatter(random[:,-2],[0]*len(random),alpha=alpha,c=cmap2(norm(tempAllProbs[s1:,:])))
							ax7.scatter(random[:,-1],[0]*len(random),alpha=alpha,c=cmap2(norm(tempAllProbs[s1:,:])))
						else:

							# minSize, maxSize = 10, 100
							# sizes = minSize + tempAllProbs[s1:]*(maxSize-minSize)
							sizes = 100 + tempAllProbs[s1:] * 490

							ax5.scatter(random[:,0],random[:,1],s=sizes,alpha=alpha,color=c)
							ax6.scatter(random[:,-2],[0]*len(random),s=sizes,alpha=alpha,color=c)
							ax7.scatter(random[:,-1],[0]*len(random),s=sizes,alpha=alpha,color=c)

							colors.append(c)
							labels.append("Random")


				if ii < 2:

					if colors:
						legend_patches = [Patch(color=colors[i], label=labels[i]) for i in range(len(colors))]
						ax7.legend(handles=legend_patches, loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=len(labels))



			######################################################


			######################################################
			# Plot estimates over time
			######################################################

			for ai, tempInfo in enumerate([(ax8,"X",0),(ax9,"X",1),(ax10,"R",1),(ax11,"DM",2)]):

				ax, label, mdi = tempInfo

				# code.interact("...", local=dict(globals(), **locals()))

				if "X" in label:
					label = f"{label}-{mdi+1}"
					toPlot = list(estimatesES[0][:,mdi])
				else:
					toPlot = estimatesES[mdi]

				ax.scatter(range(len(toPlot)),toPlot)
				ax.plot(range(len(toPlot)),toPlot)

				ax.axvline(x=i, color='red', linestyle='--', linewidth=2)

				ax.set_xlabel("Number of Observations",fontweight='bold')
				ax.set_ylabel(f"{label} Estimate",fontweight='bold')

			######################################################


			######################################################
			# Other plot info
			######################################################

			action = np.round(action,4)
			executedActionOriginal = np.round(executedActionOriginal,4)
			executedAction = np.round(executedAction,4)
			playerLocation = np.round(playerLocation,4)


			title = f"Player Location: ({playerLocation[0]},{playerLocation[1]})\n"
			title += f"Executed Action: ({executedActionOriginal[0]},{executedActionOriginal[1]}) (Plot: ({executedAction[0]},{executedAction[1]})) | Angular Action: ({action[0]},{action[1]})\n"
			
			title += ("-"*60)+"\n"

			if i > 0:
				title += f"Prev - ES: ({estimatesES[0][i-1][0]},{estimatesES[0][i-1][1]},{estimatesES[1][i-1]},{estimatesES[2][i-1]}) | MS: ({estimatesMS[0][i-1][0]},{estimatesMS[0][i-1][1]},{estimatesMS[1][i-1]},{estimatesMS[2][i-1]})\n"


			info1 = f" ({estimatesES[0][i][0]},{estimatesES[0][i][1]},{estimatesES[1][i]},{estimatesES[2][i]})"
			info2 = f" ({estimatesMS[0][i][0]},{estimatesMS[0][i][1]},{estimatesMS[1][i]},{estimatesMS[2][i]})"
			title += r"$\bf{" + "PFE - ES: " + info1 + " | MS: " + info2 + " }$\n"


			if i+1 < len(estimatesES[0]):
				title += f"Next - ES: ({estimatesES[0][i+1][0]},{estimatesES[0][i+1][1]},{estimatesES[1][i+1]},{estimatesES[2][i+1]})| MS: ({estimatesMS[0][i+1][0]},{estimatesMS[0][i+1][1]},{estimatesMS[1][i+1]},{estimatesMS[2][i+1]})\n"


			if tempStr != "":
				title += tempStr

			plt.suptitle(title)

			plt.tight_layout()

			if ii >= 2:
				cbar = plt.colorbar(sm,ax=ax4)

			fig.savefig(f"{saveAt}pf-{i}-{row}.png",bbox_inches = 'tight')
			plt.clf()
			plt.close("all")

			######################################################

			# code.interact("...", local=dict(globals(), **locals()))



def plotAllInfo(i,row,rfunc,listedTargetsAngular,rfuncAngular,action,executedActionAngular,prevProbs,probs,xsMAP,xsEES,psMAP,psEES,playerLocation,executedAction,executedActionOriginal):
	
	probs = np.array(probs)

	cmapStr = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)


	cmapStr = "nipy_spectral"
	cmap2 = plt.get_cmap(cmapStr)(np.linspace(0., 1., n))# + c * np.ones((n, 4))
	cmap2 = ListedColormap(cmap2)


	fig = plt.figure(num=0,figsize=(28,14))

	# Add a rectangle behind the first column
	row_rect = Rectangle(
	    (0,0),  # x, y (bottom left corner)
	    1/4,         # width (normalized)
	    1,       # height (normalized for row size)
	    color='lightgray',
	    zorder=-1   # Send behind the plots
	)
	fig.add_artist(row_rect)


	# Add a rectangle behind the third column
	row_rect = Rectangle(
	    (2/4,0),  # x, y (bottom left corner)
	    1/4,         # width (normalized)
	    1,       # height (normalized for row size)
	    color='lightgray',
	    zorder=-1   # Send behind the plots
	)
	fig.add_artist(row_rect)


	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	# plt.subplots_adjust(wspace=0.3,hspace=0.4)

	ax1 = plt.subplot2grid((2,4),(0,0))
	ax2 = plt.subplot2grid((2,4),(1,0))

	ax3 = plt.subplot2grid((2,4),(0,1))
	ax4 = plt.subplot2grid((2,4),(1,1))
	
	ax5 = plt.subplot2grid((2,4),(0,2))
	ax6 = plt.subplot2grid((2,4),(1,2))
	
	ax7 = plt.subplot2grid((2,4),(0,3))
	ax8 = plt.subplot2grid((2,4),(1,3))

	ax1.set_title("Rink",fontweight='bold')
	ax2.set_title("Shot",fontweight='bold')
	ax3.set_title("Reward Function",fontweight='bold')
	ax4.set_title("Reward Function - Angular Coordinates",fontweight='bold')
	ax5.set_title("Before Update",fontweight='bold')
	ax6.set_title("After Update",fontweight='bold')


	
	makePlotsFirstColumn(ax1,ax2,ax3,ax4,cmap,playerLocation,action,executedAction,executedActionOriginal,rfunc,rfuncAngular,listedTargetsAngular,dirs,elevations)



	######################################################
	# ax5 = prev probs
	######################################################

	norm = plt.Normalize(0.0,np.max(prevProbs))
	sm = ScalarMappable(norm=norm,cmap=cmap2)
	sm.set_array([])

	ax5.scatter(xg,pg,c=cmap2(norm(prevProbs)).reshape(-1,4))

	ax5cb = plt.colorbar(sm,ax=ax5)
	ax5cb = ax5cb.ax

	ax5.set_xlabel("Execution Skill Hypothesis",fontweight='bold')
	ax5.set_ylabel("Dec-Making Skill Hypothesis",fontweight='bold')

	######################################################



	######################################################
	# ax6 = probs
	######################################################

	norm = plt.Normalize(0.0,np.max(probs))
	sm = ScalarMappable(norm=norm,cmap=cmap2)
	sm.set_array([])

	ax6.scatter(xg,pg,c=cmap2(norm(probs)).reshape(-1,4))
	
	ax6cb = plt.colorbar(sm,ax=ax6)
	ax6cb = ax6cb.ax
	
	ax6.set_xlabel("Execution Skill Hypothesis",fontweight='bold')
	ax6.set_ylabel("Dec-Making Skill Hypothesis",fontweight='bold')

	# mi = np.argmax(probs)
	# xmi, pmi = np.unravel_index(mi,probs.shape)
	# ax6.scatter(xskills[xmi],pskills[pmi],marker="X",color="black")

	######################################################


	######################################################
	# ax7 = estimate JEEDS xskill
	######################################################

	ax7.scatter(range(len(xsEES_all)),xsEES_all)
	ax7.plot(range(len(xsEES_all)),xsEES_all)

	ax7.axvline(x=i, color='red', linestyle='--', linewidth=2)

	ax7.set_xlabel("Number of Observations",fontweight='bold')
	ax7.set_ylabel("Execution Skill Estimate",fontweight='bold')

	######################################################


	######################################################
	# ax8 = estimate JEEDS pskill
	######################################################

	ax8.scatter(range(len(psEES_all)),psEES_all)
	ax8.plot(range(len(psEES_all)),psEES_all)

	ax8.axvline(x=i, color='red', linestyle='--', linewidth=2)

	ax8.set_xlabel("Number of Observations",fontweight='bold')
	ax8.set_ylabel("Dec-Making Skill Estimate",fontweight='bold')

	######################################################


	######################################################
	# Other plot info
	######################################################

	action = np.round(action,4)
	executedActionOriginal = np.round(executedActionOriginal,4)
	executedAction = np.round(executedAction,4)
	playerLocation = np.round(playerLocation,4)
	
	title = f"Player Location: ({playerLocation[0]},{playerLocation[1]})\n"
	title += f"Executed Action: ({executedActionOriginal[0]},{executedActionOriginal[1]}) (Plot: ({executedAction[0]},{executedAction[1]})) | Angular Action: ({action[0]},{action[1]})\n"
	
	title += ("-"*60)+"\n"

	if i > 0:
		title += f"Prev - ES: ({xsEES_all[i-1]},{psEES_all[i-1]}) | MS: ({xsMAP_all[i-1]},{psMAP_all[i-1]})\n"
	

	info1 = f"({xsEES},{psEES})"
	info2 = f"({xsMAP},{psMAP})"
	title += r"$\bf{JTM - ES: " + info1 + " }$" + r"$\bf{ | MS: " + info2 + " }$\n"


	if i+1 < len(xsEES_all):
		title += f"Next - ES: ({xsEES_all[i+1]},{psEES_all[i+1]}) | MS: ({xsMAP_all[i+1]},{psMAP_all[i+1]})\n"


	plt.suptitle(title)

	plt.tight_layout()

	plt.savefig(f"{path}{i}-{row}.png")
	plt.clf()
	plt.close()

	######################################################


	######################################################
	# Test executed action convertion
	######################################################

	d,e = action

	# Step 1
	xp = 89 - playerLocation[0]
	deltaY = xp * np.tan(d)
	D = xp / np.cos(d)

	# Step 2
	deltaZ = D * np.tan(e)

	executedActionComputed = [playerLocation[1]+deltaY,deltaZ]

	print("\ni: ",i)
	print("\nrow: ",row)
	print("playerLocation: ", playerLocation)
	print("executedAction: ", executedActionOriginal)
	print("executedActionAngular (from noisy actions): ", action)
	print("executedActionAngular (from file): ", executedActionAngular)
	print("executedActionComputed: ", executedActionComputed)
	
	######################################################
	

	# code.interact("...", local=dict(globals(), **locals()))



if __name__ == '__main__':


	if len(sys.argv) != 3:
		print("Need to specify the name of the experiment folder (located under 'Experiments/hockey-multi/') and the result file of interest as command line argument.")
		exit()



	expFolder, rf = sys.argv[1], sys.argv[2]

	# if expFolder[-1] != os.sep:
	# 	expFolder += os.sep

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


	saveAtCopy = saveAt
	saveAt += "AllInfo" + os.sep

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


	# ASSUMING HOCKEY DOMAIN


	minY = -3.0
	maxY = 3.0

	minZ = 0.0
	maxZ = 4.0

	targetsY = np.linspace(minY,maxY,60)
	targetsZ = np.linspace(minZ,maxZ,40) 

	Y = targetsY
	Z = targetsZ

	targetsUtilityGridY,targetsUtilityGridZ = np.meshgrid(Y,Z)
	targetsUtilityGridYZ = np.stack((targetsUtilityGridY,targetsUtilityGridZ),axis=-1)
	shape = targetsUtilityGridYZ.shape
	listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0]*shape[1],shape[2]))




	#####################################################################################
	# JEEDS
	#####################################################################################

	'''

	methodType = "JEEDS"

	path = f"{saveAt}{os.sep}{methodType}{os.sep}"
	
	if not os.path.exists(path):
		os.mkdir(path)


	startX_Estimator = 0.004
	stopX_Estimator = np.pi/4

	numXs = infoExperiment["numHypsX"][0][0]
	numPs = infoExperiment["numHypsP"][0]
 
	xskills = list(np.round(np.linspace(startX_Estimator,stopX_Estimator,num=numXs),4))
	pskills = np.round(np.logspace(-3,1.6,numPs),4)

	xg,pg = np.meshgrid(xskills,pskills,indexing="ij")


	# JEEDS via PFE
	methodMAP = f"JT-QRE-MAP-{numXs}-{numPs}"
	methodEES = f"JT-QRE-MAP-{numXs}-{numPs}"


	probs = infoExperiment["JT-QRE-allProbs"]

	noisyActions = infoExperiment["noisy_actions"]

	xsMAP_all = infoExperiment['JT-QRE-MAP-33-33-xSkills']
	xsEES_all = infoExperiment['JT-QRE-EES-33-33-xSkills']

	psMAP_all = infoExperiment['JT-QRE-MAP-33-33-pSkills']
	psEES_all = infoExperiment['JT-QRE-EES-33-33-pSkills']

	xsMAP_all = np.round(xsMAP_all,4)
	xsEES_all = np.round(xsEES_all,4)
	psMAP_all = np.round(psMAP_all,4)
	psEES_all = np.round(psEES_all,4)


	# FOR TESTING
	# xsMAP_all = xsMAP_all[:20]
	# xsEES_all = xsEES_all[:20]
	# psMAP_all = psMAP_all[:20]
	# psEES_all = psEES_all[:20]


	for i, row in enumerate(ids):

		listedTargetsAngular = infoRows[row]["row"]["listedTargetsAngular"]

		rfunc = infoRows[row]["row"]["heat_map"]
		shape = rfunc.shape
		rfunc = rfunc.reshape((shape[0]*shape[1],1))


		rfuncAngular = infoRows[row]["row"]["listedUtilitiesComputed"]


		executedActionAngular = infoRows[row]["row"]["executedActionAngular"]
		

		playerLocation = [infoRows[row]["row"]["start_x"],infoRows[row]["row"]["start_y"]]
		executedAction = [89,infoRows[row]["row"]["shot_location"][0]]

		# shot_location = final_y, projected_z, start_x, start_y
		executedActionOriginal = [infoRows[row]["row"]["shot_location"][0],infoRows[row]["row"]["shot_location"][1]]

		dirs, elevations = infoRows[row]["row"]["dirs"], infoRows[row]["row"]["elevations"]

		plotAllInfo(i,row,rfunc,listedTargetsAngular,rfuncAngular,noisyActions[i],executedActionAngular,probs[i],probs[i+1],xsMAP_all[i],xsEES_all[i],psMAP_all[i],psEES_all[i],playerLocation,executedAction,executedActionOriginal)
		
		
		# code.interact("...", local=dict(globals(), **locals()))


	'''

	#####################################################################################
	


	#####################################################################################
	# PFE
	#####################################################################################

	# '''

	path = f"{saveAt}{os.sep}"

	agent = agent[1]


	# method = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv[200, 200]-JT-EES"
	method = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv[200, 200]-JT-EES"
	# method2 = "QRE-Multi-Particles-1000-Resample90%-NoiseDiv[200, 200]"
	method2 = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv[200, 200]-JT-MAP"
	method3 = "QRE-Multi-Particles-1000-Resample90%-ResampleNEFF-NoiseDiv[200, 200]"
	resampleKey = method3 + "-whenResampled"

	noisyActions = infoExperiment["noisy_actions"]


	allParticles = infoExperiment[method3+"-allParticles"]
	# allParticlesNoNoise = infoExperiment[method3+"-allParticlesNoNoise"]
	# allNoises = infoExperiment[method3+"-allNoises"]
	allProbs = np.array(infoExperiment[method3+"-allProbs"])
	
	estimatesES = [infoExperiment[method+"-xSkills"],infoExperiment[method+"-rhos"],infoExperiment[method+"-pSkills"]]
	estimatesES[0] = np.round(estimatesES[0],4)
	estimatesES[1] = np.round(estimatesES[1],4)
	estimatesES[2] = np.round(estimatesES[2],4)

	estimatesMS = [infoExperiment[method2+"-xSkills"],infoExperiment[method2+"-rhos"],infoExperiment[method2+"-pSkills"]]
	estimatesMS[0] = np.round(estimatesMS[0],4)
	estimatesMS[1] = np.round(estimatesMS[1],4)
	estimatesMS[2] = np.round(estimatesMS[2],4)


	resampledInfo = infoExperiment[resampleKey]


	plotParticles(path,agent,method,allParticles,estimatesES,estimatesMS,resampledInfo,allProbs)

	# '''

	#####################################################################################
	


	#####################################################################################
	# PDFs & EVs
	#####################################################################################

	'''

	saveAt1 = f"{saveAtCopy}{os.sep}pdfsPerXskill{os.sep}"
	saveAt2 = f"{saveAtCopy}{os.sep}evsPerXskill{os.sep}"


	for each in [saveAt1,saveAt2]:
		if not os.path.exists(each):
			os.mkdir(each)

	getBoardPlots(saveAt1,saveAt2)

	'''

	#####################################################################################


	# code.interact("...", local=dict(globals(), **locals()))


	



