import code
import numpy as np
import os,sys

import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Testing{os.sep}testPdfs.py")[0]

if "spaces" not in sys.modules:
	module = SourceFileLoader("spaces",f"{mainFolderName}setupSpaces.py").load_module()
	sys.modules["spaces"] = module

if "utils" not in sys.modules:
	module = SourceFileLoader("utils",f"{mainFolderName}Estimators{os.sep}utils.py").load_module()
	sys.modules["utils"] = module

from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle
from math import pi, cos, sin

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
		ax.plot([sx, dx], [sy, dy], color="Black")
		# print 'Angle = ', 180.0*angle/pi
		angle += angle_increment


def label_regions(ax,slices,color="black"):

	angle_increment = pi / 10.0    
	angle = pi
	r = 130.0

	for i in range(1,21):
		x = r*cos(angle)
		y = r*sin(angle)
		ax.text(x,y,str(slices[i]),fontsize=12,horizontalalignment='center', color = color)
		angle += angle_increment

	# For single Bullseye
	# Value for bullseye will always be at the first index
	ax.text(0,0,str(slices[0]),fontsize=12,horizontalalignment='center', color = color)


def getTargets(delta):

	XS = np.arange(-170.0,171.0,delta)
	YS = np.arange(-170.0,171.0,delta)

	XXS,YYS = np.meshgrid(XS,YS,indexing="ij")
	tempXYS = np.vstack([XXS.ravel(),YYS.ravel()])

	listedTargets = np.dstack(tempXYS)[0]
	listedTargets = np.array(listedTargets)

	return listedTargets


def getInfo(delta,targets,action,key):

	print(f"{'-'*15}\nKey: {key}")

	listedTargets = getTargets(delta)
	print(f"Delta: {delta}")
	print(f"len targets: {len(targets)}")

	pdfs = sys.modules["utils"].computePDF(x=action,means=targets,covs=np.array([spaces.convolutionsPerXskill[key]["cov"]]*len(targets)))

	scaledPdfs = np.multiply(pdfs,np.square(delta))

	print(f"Sum pdfs: {np.sum(pdfs)}")
	print(f"Sum scaled pdfs: {np.sum(scaledPdfs)}")

	print(f"Pdf for {targets[0]}: {pdfs[0]}")
	print(f"Scaled Pdf for {targets[0]}: {scaledPdfs[0]}")

	print("-"*15,"\n")

	cmap = plt.get_cmap("viridis")
	
	norm1 = plt.Normalize(0.0,max(pdfs))
	norm2 = plt.Normalize(0.0,max(scaledPdfs))

	sm1 = ScalarMappable(cmap=cmap,norm=norm1)
	sm1.set_array([])

	sm2 = ScalarMappable(cmap=cmap,norm=norm2)
	sm2.set_array([])


	fig = plt.figure(num=0,figsize=(12,8))

	ax1 = plt.subplot2grid((2,1),(0,0))
	ax2 = plt.subplot2grid((2,1),(1,0))

	ax1.scatter(listedTargets[:,0],listedTargets[:,1],c=cmap(norm1(pdfs)))
	ax2.scatter(listedTargets[:,0],listedTargets[:,1],c=cmap(norm2(scaledPdfs)))

	draw_board(ax1)
	label_regions(ax1,state)

	draw_board(ax2)
	label_regions(ax2,state)

	cbar = plt.colorbar(sm1,ax=ax1)
	cbar = plt.colorbar(sm2,ax=ax2)

	fig.savefig(f"Testing/pdfs-delta-{delta}-key-{key}.png",bbox_inches = 'tight')
	plt.clf()
	plt.close("all")



if __name__ == '__main__':

	# Sample state
	state = [25,11,8,16,7,19,3,17,2,15,10,6,13,4,18,1,20,5,12,9,14,11]
	
	# Sample action
	action = [58.81632006, 81.17515193]


	# Sample target action
	targetAction = [[-170.0,-170.0]]


	# [X1, X2, R, P]
	particles = [[10.0,10.0,0.0,100],
				[100.0,100.0,0.0,100]]


	numSamples = 1000

	domain = "2d-multi"
	mainFolder = "Spaces" + os.sep + "ExpectedRewards" + os.sep
	fileName = f"ExpectedRewards-{domain}-N{numSamples}"
	expectedRFolder = mainFolder + fileName

	load = f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
	domainModule = SourceFileLoader("two_d_darts_multi",load+"two_d_darts_multi.py").load_module()
	mode = "normal"


	rng = np.random.default_rng(1000)
	

	for delta in [5,10]:
		targets = getTargets(delta)

		spaces = sys.modules["spaces"].SpacesRandomDarts(numSamples,domainModule,mode,delta,numSamples,expectedRFolder)
		
		for each in particles:
			key = "|".join(map(str,each[:-1]))
			spaces.updateSpaceParticles(rng,each,state,{})

			space = spaces.convolutionsPerXskill[key][str(state)]
			evs = space["all_vs"].flatten()

			getInfo(delta,targets,action,key)

	code.interact("...", local=dict(globals(), **locals()))


















