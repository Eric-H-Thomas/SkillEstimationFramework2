import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys,code
from importlib.machinery import SourceFileLoader
import torch
import torch.nn as nn 
from matplotlib.cm import ScalarMappable
from scipy.signal import convolve2d, fftconvolve
from math import dist
from pybaseball import playerid_reverse_lookup, cache

from scipy import stats
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Processing{os.sep}makeCIPlotsBaseball.py")[0]

module = SourceFileLoader("baseball.py",f"{mainFolderName}Environments{os.sep}Baseball{os.sep}baseball.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module

# FROM: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, facecolor = "none", **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, facecolor=facecolor, **kwargs)

    ax.add_artist(ellip)
    return ellip


# FROM: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
	"""
	Create a plot of the covariance confidence ellipse of *x* and *y*.

	Parameters
	----------
	x, y : array-like, shape (n, )
		Input data.

	ax : matplotlib.axes.Axes
		The axes object to draw the ellipse into.

	n_std : float
		The number of standard deviations to determine the ellipse's radiuses.

	**kwargs
		Forwarded to `~matplotlib.patches.Ellipse`

	Returns
	-------
	matplotlib.patches.Ellipse
	"""
	if x.size != y.size:
		raise ValueError("x and y must be the same size")

	cov = np.cov(x, y)
	pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
	# Using a special case to obtain the eigenvalues of this
	# two-dimensional dataset.
	ell_radius_x = np.sqrt(1 + pearson)
	ell_radius_y = np.sqrt(1 - pearson)
	ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
					  facecolor=facecolor, **kwargs)

	# Calculating the standard deviation of x from
	# the squareroot of the variance and multiplying
	# with the given number of standard deviations.
	scale_x = np.sqrt(cov[0, 0]) * n_std
	mean_x = np.mean(x)

	# calculating the standard deviation of y ...
	scale_y = np.sqrt(cov[1, 1]) * n_std
	mean_y = np.mean(y)

	transf = transforms.Affine2D() \
		.rotate_deg(45) \
		.scale(scale_x, scale_y) \
		.translate(mean_x, mean_y)

	ellipse.set_transform(transf + ax.transData)
	return ax.add_patch(ellipse)


def plotOtherInfo(titleStr,ax):

	# Overlay strike zone dimensions on plot
	# Plate_x: [-0.71,0.71]
	# Plate_z: [1.546,3.412]
	ax.hlines(y=1.546,xmin=-0.71,xmax=0.71,color="k")
	ax.hlines(y=3.413,xmin=-0.71,xmax=0.71,color="k")
	ax.vlines(x=-0.71, ymin=1.546,ymax=3.412,color="k")
	ax.vlines(x=0.71, ymin=1.546,ymax=3.412,color="k")

	ax.set_xlabel("Horizontal Location (Pitcher's Perspective)")
	ax.set_ylabel("Vertical Location")

	ax.set_title(titleStr)


def getCIsMinMaxXskill(minX,maxX):

	saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"


	xSkills = [minX,maxX]

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)


	# PLOT STRIKEZONE
	fig,ax = plt.subplots()

	ax.set_aspect('equal')
	plt.xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
	plt.ylim(0.5,4.5)

	plotOtherInfo("",ax)

	colors = ["C0","C1"]


	middleAction = spaces.focalActionMiddle
	numSamples = 1000

	for xi in range(len(xSkills)):

		x = xSkills[xi]

		executedActions = []

		# Assuming aiming at best possible action
		# Sample executed actions (intended + sample noise given xskill)
		for s in range(numSamples):

			mean = [0.0,0.0]
			executedActions.append(sys.modules["domain"].sample_action(mean,x,middleAction))

		executedActions = np.asarray(executedActions)


		# 50% CI ?
		nstd = 0.67449
		confidence_ellipse(executedActions[:,0],executedActions[:,1],ax,n_std=nstd,label=x, edgecolor=colors[xi])#'firebrick')

		plot_point_cov(executedActions,nstd=nstd,ax=ax,edgecolor = colors[xi])


	ax.legend()

	fig.savefig(f"{saveAt}CI-minX-maxX.jpg",bbox_inches="tight")

	plt.clf()
	plt.close("all")


def getBoardPlotsAndCI(pitcherID,pitchType,saveAt,numSamples=1000):

	agentFolder = f"pitcherID{pitcherID}-PitchType{pitchType}"
			
	if not os.path.exists(saveAt+os.sep+"ConfidenceIntervalPlots"):
		os.mkdir(saveAt+os.sep+"ConfidenceIntervalPlots")

	# 2.0 inches | 0.17 feet
	startX_Estimator = 0.17
	# 33.72 inches | 2.81 feet
	stopX_Estimator = 2.81

	# 0.5 inches | 0.0417 feet
	delta = 0.0417

	xSkills = list(np.concatenate((np.linspace(startX_Estimator,1.0,num=60),np.linspace(1.00+delta,stopX_Estimator,num=6))))

	# ALL XSKILLS
	# [0.17, 0.1840677966101695, 0.198135593220339, 0.2122033898305085, 0.22627118644067798, 
	# 0.24033898305084744, 0.25440677966101694, 0.2684745762711864, 0.2825423728813559, 0.2966101694915254,
	# 0.3106779661016949, 0.3247457627118644, 0.3388135593220339, 0.3528813559322034, 0.36694915254237287, 
	# 0.38101694915254236, 0.39508474576271185, 0.40915254237288134, 0.4232203389830508, 0.43728813559322033, 
	# 0.45135593220338976, 0.4654237288135593, 0.47949152542372875, 0.4935593220338983, 0.5076271186440677,
	# 0.5216949152542373, 0.5357627118644067, 0.5498305084745763, 0.5638983050847457, 0.5779661016949152,
	# 0.5920338983050847, 0.6061016949152542, 0.6201694915254237, 0.6342372881355932, 0.6483050847457626, 
	# 0.6623728813559322, 0.6764406779661016, 0.6905084745762712, 0.7045762711864406, 0.7186440677966102, 
	# 0.7327118644067796, 0.7467796610169491, 0.7608474576271186, 0.7749152542372881, 0.7889830508474576,
	# 0.8030508474576271, 0.8171186440677966, 0.8311864406779661, 0.8452542372881355, 0.8593220338983051,
	# 0.8733898305084745, 0.8874576271186441, 0.9015254237288135, 0.915593220338983, 0.9296610169491525,
	# 0.943728813559322, 0.9577966101694915, 0.971864406779661, 0.9859322033898305, 1.0, 
	# 1.0417, 1.3953600000000002, 1.74902, 2.10268, 2.45634, 2.81]

	# xSkills = [0.17, 0.2825423728813559, 0.40915254237288134,
	#  		0.5216949152542373, 0.7045762711864406, 1.0417, 2.81]

	xSkills = [0.50,0.63,0.75,0.88,1.0]

	spaces = sys.modules["spaces"].SpacesBaseball([],1,sys.modules["domain"],delta,1,"",0,0)

	copyPlateX = spaces.possibleTargetsForModel[:,0]
	copyPlateZ = spaces.possibleTargetsForModel[:,1]
	possibleTargetsLen = len(spaces.possibleTargetsForModel)


	ci = 0.95

	# For 95% interval
	Z = 1.960
	

	pdfsPerXskill = {}

	for x in xSkills:
		pdfsPerXskill[x] = sys.modules["domain"].getSymmetricNormalDistribution(x,delta,spaces.targetsPlateXFeet,spaces.targetsPlateZFeet)


	agentData = spaces.getAgentData(pitcherID,pitchType,maxRows=5)

	saveAtOriginal = saveAt

	for row in agentData.itertuples():

		index = row.Index

		saveAt = f"{saveAtOriginal}{os.sep}Index-{index}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)


		allTempData = pd.DataFrame([row]*(possibleTargetsLen))

		# Update position of each copy of the row to be that of a given possible action
		allTempData["plate_x"] = copyPlateX
		allTempData["plate_z"] = copyPlateZ


		# Include original 'row' (df with actual pitch info) to get the probabilities 
		# for the different outcomes as well as the utility - for the actual pitch
		allTempData.loc[len(allTempData.index)+1] = row


		########################################
		# NEW MODEL
		########################################

		batch_x = allTempData[sys.modules["modelTake2"].features].values
		batch_y = allTempData.outcome.values.astype(int)

		#Reshape so that each pa is a separate entry in the batch
		batch_x = batch_x.reshape((len(batch_x), 1,len(sys.modules["modelTake2"].features)))
		batch_y = batch_y.reshape((len(batch_y),1))

		torch_batch_x = torch.tensor(batch_x,dtype=torch.float)
		torch_batch_y = torch.tensor(batch_y,dtype=torch.long)
		

		ypred = sys.modules["modelTake2"].prediction_func(spaces.model,torch_batch_x,torch_batch_y)

		allTempData[['o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']] = nn.functional.softmax(ypred, dim = 1).detach().cpu().numpy()
		
		# code.interact("after...", local=dict(globals(), **locals()))

		########################################

		
		# Get utilities
		withUtilities = sys.modules["utilsBaseball"].getUtility(allTempData)


		# Get updated info for actual pitch (actual pitch + probs + utility)
		row = withUtilities.iloc[-1].copy()

		# Remove actual pitch from data
		withUtilities = withUtilities.iloc[:-1,:]
		
		minUtility = np.min(withUtilities["utility"].values)

		
		# Prepare color map
		cmapStr = "viridis" # seismic
		cmap = plt.get_cmap(cmapStr)
		norm = plt.Normalize(minUtility,max(withUtilities["utility"].values))
		sm = ScalarMappable(norm=norm,cmap=cmap)
		sm.set_array([])


		###############################################################
		# Strike Zone Board
		###############################################################

		fileName = f"{agentFolder}-Index{index}-diffXskills"
	

		# Populate Dartboard - Can create once since independent of xskill
		Zs = np.zeros((len(spaces.modelTargetsPlateX), len(spaces.modelTargetsPlateZ)))

		for i in list(range(len(spaces.modelTargetsPlateX))):
			for j in list(range(len(spaces.modelTargetsPlateZ))):
				tempIndex = np.where((withUtilities.plate_x == spaces.modelTargetsPlateX[i]) & (withUtilities.plate_z == spaces.modelTargetsPlateZ[j]))[0][0]
				Zs[i][j] = np.copy(allTempData.iloc[tempIndex]["utility"])

		middle = spaces.focalActionMiddle
		newFocalActions = []


		# PLOT - RAW BOARD

		fig1,ax1 = plt.subplots()
		fig2,ax2 = plt.subplots()
		fig3,ax3 = plt.subplots()

		'''
		cbar = fig.colorbar(sm,ax=ax)
		cbar.ax.get_yaxis().labelpad = 15
		cbar.ax.set_ylabel("Utilities",rotation = 270)
		'''

		cmap2 = plt.get_cmap("rainbow")
		norm2 = plt.Normalize(min(xSkills),max(xSkills))


		ax1.set_aspect('equal')
		ax2.set_aspect('equal')
		ax3.set_aspect('equal')

		ax1.set_xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
		ax2.set_xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
		ax3.set_xlim([min(spaces.possibleTargetsFeet[:,0]),max(spaces.possibleTargetsFeet[:,0])])
		# plt.ylim([min(spaces.possibleTargetsFeet[:,1]),max(spaces.possibleTargetsFeet[:,1])])
		ax1.set_ylim(0.5,4.5)
		ax2.set_ylim(0.5,4.5)
		ax3.set_ylim(0.5,4.5)


		# max utility
		maxUtility = np.max(withUtilities["utility"].values)
		iis = np.where(withUtilities["utility"].values == maxUtility)[0][0]
		maxUtilityAction = [spaces.possibleTargetsFeet[:,0][iis],spaces.possibleTargetsFeet[:,1][iis]]

		# ax.scatter(spaces.possibleTargetsFeet[:,0],spaces.possibleTargetsFeet[:,1],c = cmap(norm(withUtilities["utility"].values)))

		titleStr = f"Max Expected Utilities - Different Execution Skills"
		plotOtherInfo("",ax1)
		plotOtherInfo("",ax2)
		plotOtherInfo(titleStr,ax3)

		# Plot actual executed action & EV
		# ax.scatter(row["plate_x_feet"],row["plate_z_feet"],color=cmap(norm(row["utility"])),marker="X",s=60,edgecolors="black",label="Actual Pitch")
		ax1.scatter(maxUtilityAction[0],maxUtilityAction[1],color=cmap(norm(maxUtility)),marker="X",s=60,edgecolors="black",label="Max Utility")
		ax2.scatter(maxUtilityAction[0],maxUtilityAction[1],color=cmap(norm(maxUtility)),marker="X",s=60,edgecolors="black",label="Max Utility")
		ax3.scatter(maxUtilityAction[0],maxUtilityAction[1],color=cmap(norm(maxUtility)),marker="X",s=60,edgecolors="black",label="Max Utility")
	

		for xi in range(len(xSkills)):

			x = xSkills[xi]

			# Convolve to produce the EV and aiming spot
			EVs = convolve2d(Zs,pdfsPerXskill[x],mode="same",fillvalue=minUtility)

			maxEV = np.max(EVs)	
			mx,mz = np.unravel_index(EVs.argmax(),EVs.shape)
			action = [spaces.targetsPlateXFeet[mx],spaces.targetsPlateZFeet[mz]]

			ax3.scatter(action[0],action[1],color=cmap(norm(maxEV)),marker="X",s=60,edgecolors="black",label=r"$\sigma$: "+f"{x:.4f}")


			executedActions = []

			# Assuming aiming at best possible action
			# Sample executed actions (intended + sample noise given xskill)
			for s in range(numSamples):

				mean = [0.0,0.0]
				# mean = maxUtilityAction
				executedActions.append(sys.modules["domain"].sample_action(mean,x,action))

			executedActions = np.asarray(executedActions)

			'''
			info[x] = {}

			mu = np.mean(executedActions)
			sigma = np.std(executedActions)

			low, high = stats.norm.interval(ci,loc=mu,scale=sigma/np.sqrt(numSamples))
			value = Z * (sigma/np.sqrt(numSamples))
			'''

			# 2 = 95% CI
			confidence_ellipse(executedActions[:,0],executedActions[:,1],ax1,n_std=2,label=x, edgecolor= cmap2(norm2([x])))#'firebrick')

			# plot_point_cov(executedActions,ax=ax2,edgecolor=cmap2(norm2([x])))
	

		ax1.legend()
		ax2.legend()
		ax3.legend()

		cbar = fig3.colorbar(sm,ax=ax3)
		cbar.ax.get_yaxis().labelpad = 15
		cbar.ax.set_ylabel("Utilities",rotation = 270)


		fig1.savefig(f"{saveAtOriginal}{os.sep}ConfidenceIntervalPlots{os.sep}{fileName}-1.jpg",bbox_inches="tight")
		fig2.savefig(f"{saveAtOriginal}{os.sep}ConfidenceIntervalPlots{os.sep}{fileName}-2.jpg",bbox_inches="tight")
		fig3.savefig(f"{saveAtOriginal}{os.sep}ConfidenceIntervalPlots{os.sep}{fileName}-optimalActions.jpg",bbox_inches="tight")
		
		plt.clf()
		plt.close("all")

		# code.interact("after...", local=dict(globals(), **locals()))



if __name__ == '__main__':


	############################################################
	# Plot 50% CI on strikezone of given min and max xskill
	############################################################

	# Joe Mantiply
	minX = 0.637

	# Tanner Scott
	maxX = 0.967
	
	getCIsMinMaxXskill(minX,maxX)

	############################################################


	############################################################

	folders = [f"Data{os.sep}",f"Data{os.sep}Baseball{os.sep}",
				f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}"]

	for folder in folders:
		#If the folder doesn't exist already, create it
		if not os.path.exists(folder):
			os.mkdir(folder)


	# PITCHERS OF INTEREST
	# Mantiply, Joe: 573009 
	#	CU: 517 | CH: 371 | SL: 0 | FF: 78 | FS: 0 | SI: 725 
	# Scott, Tanner: 656945 | 
	# 	CU: 0 | CH: 0 | SL: 2002 | FF: 1945 | FS: 0 | SI: 52 | FC: 0 | KC: 0 | FA: 0 | CS: 0 | nan: 0 | EP: 0 | FO: 0 | KN: 0 | PO: 0 | SC: 0 | FT: 0 | 

	info = [["573009", "CU"],["573009", "CH"],["573009", "FF"],["573009", "SI"],
			["656945","SL"],["656945","FF"],["656945","SI"]]

	pitcherNames = {}
	p = 0

	for each in info:

		print(f"\n\n({p+1}/{len(info)}) - Looking at: {each}")
		p += 1

		result = playerid_reverse_lookup([int(each[0])])[["name_first","name_last"]]
		pitcherNames[int(each[0])] = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"

		nameNoSpace = pitcherNames[int(each[0])].replace(' ','')
		
		if not os.path.exists(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}"):
			os.mkdir(f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}")

		saveAt = f"Data{os.sep}Baseball{os.sep}PlotsForPaper{os.sep}{nameNoSpace}{os.sep}{each[1]}{os.sep}"
		
		if not os.path.exists(saveAt):
			os.mkdir(saveAt)

		getBoardPlotsAndCI(int(each[0]),each[1],saveAt)
	
	############################################################



	
