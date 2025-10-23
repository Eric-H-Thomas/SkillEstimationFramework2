import numpy as np
import sys, os, pickle
import code
from scipy.interpolate import griddata
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}getAngularHeatmapsPerPlayer.py")[0]

module = SourceFileLoader("hockey.py",f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


module = SourceFileLoader("utils.py",f"{mainFolderName}Estimators{os.sep}utils.py").load_module()
sys.modules["utils"] = module


# Feet
delta = 1.0 #0.16

spaces = sys.modules["spaces"].SpacesHockey([],1,sys.modules["domain"],delta)

Y = spaces.targetsY
Z = spaces.targetsZ

targetsUtilityGridY,targetsUtilityGridZ = np.meshgrid(Y,Z)
targetsUtilityGridYZ = np.stack((targetsUtilityGridY,targetsUtilityGridZ),axis=-1)

shape = targetsUtilityGridYZ.shape
listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0]*shape[1],shape[2]))


leftPost = np.array([89,-3])
rightPost = np.array([89,3])


leftAugmented = np.array([89,-9])
rightAugmented = np.array([89,9])

top = 4
topAugmented = 8


def getAngle(point1,point2):

	x1,y1 = point1
	x2,y2 = point2
	
	angle = np.arctan2(y2-y1,x2-x1)
	
	return angle


def getAngularHeatmap(heatmap,playerLocation,executedAction):

	rng = np.random.default_rng(1000)


	shape = heatmap.shape
	listedUtilities = heatmap.reshape((shape[0]*shape[1],1))


	# Generate edges - directions
	dirL = getAngle(playerLocation,leftAugmented)
	dirR = getAngle(playerLocation,rightAugmented)

	# Generate edges - elevations
	dist1 = np.linalg.norm(playerLocation-leftAugmented)
	dist2 = np.linalg.norm(playerLocation-rightAugmented)

	minDist = min(dist1,dist2)
	elevationTop = np.arctan2(topAugmented,minDist)


	# Create grid

	'''
	deltaD = 0.01*(abs(dirL)+abs(dirR))
	deltaE = 0.01*elevationTop

	dirs = np.arange(dirL,dirR,deltaD)

	# In case discretization yield too little points
	if len(dirs) < 10:
		deltaD = 0.001*(abs(dirL)+abs(dirR))
		dirs = np.arange(dirL,dirR,deltaD)


	elevations = np.arange(0.0,elevationTop,deltaE)

	# To account for max endpoint
	if dirR not in dirs:
		dirs = np.append(dirs,dirR)

	if elevationTop not in elevations:
		elevations = np.append(elevations,elevationTop)
	'''

	resolution = 100
	dirs = np.linspace(dirL,dirR,resolution)
	elevations = np.linspace(0,elevationTop,resolution)


	gridTargets = []

	for j in Z:
		for i in Y:
			gridTargets.append([i,j])



	listedTargetsAngular = []

	for e in elevations:
		for d in dirs:
			listedTargetsAngular.append([d,e])



	tempDist = np.linalg.norm(np.array([-3,0])-np.array([3,0]))

	listedTargetsAngular2YZ = []

	# For each target on the grid
	for target in listedTargetsAngular:

		d,e = target

		# Step 1
		xp = 89 - playerLocation[0]
		deltaY = xp * np.tan(d)
		D = xp / np.cos(d)

		# Step 2
		deltaZ = D * np.tan(e)

		listedTargetsAngular2YZ.append([playerLocation[1]+deltaY,deltaZ])


	# gridTargets = np.array(gridTargets)

	listedTargetsAngular = np.array(listedTargetsAngular)
	gridTargetsAngular = np.array(listedTargetsAngular).reshape((len(dirs),len(elevations),2))


	listedTargetsAngular2YZ = np.array(listedTargetsAngular2YZ)
	gridTargetsAngular2YZ = np.array(listedTargetsAngular2YZ).reshape((len(dirs),len(elevations),2))



	# Interpolate to find utility
	listedUtilitiesComputed = griddata(listedTargetsUtilityGridYZ,listedUtilities,listedTargetsAngular2YZ,method='cubic',fill_value=0.0)

	gridUtilitiesComputed = np.array(listedUtilitiesComputed).reshape((len(dirs),len(elevations)))


	##################################################
	# Convert executed action to angular coordinates
	##################################################

	negativeElevation = False
	originalElevation = None


	deltaX = 89-playerLocation[0]
	deltaY = executedAction[0]-playerLocation[1]
	deltaZ = executedAction[1]
	
	# Direction Angle
	d = np.arctan2(deltaY,deltaX)
	
	# Elevation Angle
	D = np.sqrt(deltaX**2+deltaY**2)	
	e = np.arctan2(deltaZ,D)


	# No negative angles possible. Can't miss low. Capping at 0.
	if e < 0:
		originalElevation = e
		e = 0.0
		negativeElevation = True

	executedActionAngular = [d,e]


	##################################################

	##################################################
	# TEST
	##################################################
	# '''
	d,e = executedActionAngular

	# Step 1
	xp = 89 - playerLocation[0]
	deltaY = xp * np.tan(d)
	D = xp / np.cos(d)

	# Step 2
	deltaZ = D * np.tan(e)

	executedActionComputed = [playerLocation[1]+deltaY,deltaZ]

	print("playerLocation: ", playerLocation)
	print("executedAction: ", executedAction)
	print("executedActionAngular: ", executedActionAngular)
	print("executedActionComputed: ", executedActionComputed)

	if negativeElevation:
		print("Negative Elevation found. Elevation was set to 0. Original Elevation: ", originalElevation)

	

	# d = executedActionAngular[0]
	# e = executedActionAngular[1]

	# # Convert to Cartesian coordinates on the unit circle 
	# x = np.cos(d)
	# y = np.sin(d)

	# plt.subplot(1, 2, 2, projection='polar')
	# plt.polar(d, 1, 'ro')
	# plt.show()

	# '''
	##################################################


	# Assumming both same size (-1 to ofset for index-based 0)
	middle = int(len(dirs)/2) - 1
	mean = [dirs[middle],elevations[middle]]



	##################################################
	# Filtering
	##################################################

	# Radians
	minX = 0.004
	maxX = np.pi/4
	rhos = [0.0,-0.75,0.75]
	# rhos = [-0.75]

	allXS = [[minX,minX],[minX,maxX],[maxX,minX],[maxX,maxX]]
	# allXS = [[minX,minX]]
	# allXS = [[minX,maxX]]

	skip = False

	tempPDFs = {}

	for eachX in allXS:
		for rho in rhos:


			# FOR TESTING
			# skip = False


			covMatrix = spaces.domain.getCovMatrix([eachX[0],eachX[1]],rho)	

			x = spaces.getKey([eachX[0],eachX[1]],rho)


			pdfs = sys.modules["utils"].computePDF(x=executedActionAngular,means=listedTargetsAngular,covs=np.array([covMatrix]*len(listedTargetsAngular)))
			prev = np.copy(pdfs)

			pdfs /= np.sum(pdfs)


			N = sys.modules["domain"].draw_noise_sample(rng,mean,covMatrix)
			D = N.pdf(gridTargetsAngular)
			D /= np.sum(D)


			if np.isnan(pdfs).any():
				skip = True


				# UNCOMMENT AFTER TESTING
				break


			# tempPDFs[x] = {"pdfs":pdfs,"prev":prev,"D":D}




			'''
			savePdfs = f"{mainFolder}PDFs{os.sep}"
			
			if not os.path.exists(savePdfs):
				os.mkdir(savePdfs)

			plt.contourf(gridTargetsAngular[:,:,0],gridTargetsAngular[:,:,1],D)
			plt.savefig(f"{savePdfs}{os.sep}pdfs-xskill{eachX}-rho{rho}-1.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()
		
			plt.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=prev)
			plt.savefig(f"{savePdfs}{os.sep}pdfs-xskill{eachX}-rho{rho}-{skip}-2.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()

			plt.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=pdfs)
			plt.savefig(f"{savePdfs}{os.sep}pdfs-xskill{eachX}-rho{rho}-{skip}-3-normalized.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()
			'''

	##################################################


	# code.interact("...", local=dict(globals(), **locals()))

	return dirs,elevations,listedTargetsAngular,gridTargetsAngular,listedTargetsAngular2YZ,gridTargetsAngular2YZ,listedUtilitiesComputed,gridUtilitiesComputed,executedActionAngular,skip,(negativeElevation,originalElevation)



if __name__ == '__main__':


	try:
		experimentFolder = sys.argv[1]
	except Exception as e:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"


	folder = f"{mainFolder}Heatmaps{os.sep}"
	files = os.listdir(folder)


	saveAt = f"{mainFolder}AngularHeatmaps{os.sep}"
	saveAtFiltered = f"{mainFolder}AngularHeatmaps-Filtered{os.sep}"
	saveAtNegativeElevations = f"{mainFolder}AngularHeatmaps-NegativeElevations{os.sep}"


	for each in [f"Data{os.sep}",mainFolder,saveAt,saveAtFiltered,saveAtNegativeElevations]:
		if not os.path.exists(each):
			os.mkdir(each)

	
	statsInfo = {}
	typeShots = []


	for eachFile in files:

		if "player" not in eachFile:
			continue


		splitted = eachFile.split("_")
		playerID = splitted[3]
		typeShot = splitted[-1].split(".")[0]



		# FOR TESTING
		# playerID = 950041
		# playerID = 949936
		playerID = 950148
		# typeShot = "snapshot"
		typeShot = "wristshot"



		if playerID not in statsInfo:
			statsInfo[playerID] = {}

		if typeShot not in statsInfo[playerID]:
			statsInfo[playerID][typeShot] = 0

		if typeShot not in typeShots:
			typeShots.append(typeShot)


		print(f"Creating angular heatmap for player {playerID} - {typeShot} ...")

		fileName = f"heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"

		with open(folder+fileName,"rb") as infile:
			data = pickle.load(infile)



		statsInfo[playerID][typeShot] = len(data)
		statsInfo[playerID][f"filtered-{typeShot}"] = 0
		statsInfo[playerID][f"negativeElevation-{typeShot}"] = []


		playerData = {}
		filtered = {}



		# FOR TESTING
		# data = {270544010806:data[270544010806]}
		# data = {270443030337:data[270443030337]}

		# Get angular heatmaps per player
		for i,row in enumerate(data):

			print("\ni: ",i)
			print("row: ",row)

			# FOR TESTING
			# row = 270544010806


			heatmap = data[row]["heat_map"]
			playerLocation = [data[row]["start_x"],data[row]["start_y"]]
			projectedZ = data[row]["projected_z"]
			# shot_location = final_y, projected_z, start_x, start_y
			executedAction = [data[row]["shot_location"][0],data[row]["shot_location"][1]]

			info = getAngularHeatmap(heatmap,playerLocation,executedAction)

			skip = info[9]
			# print(f"{row} - skip? {skip}")

			# Determine if to save in file or save at file for filtered ones
			if not skip:
				where = playerData				
			else:
				where = filtered
				statsInfo[playerID][f"filtered-{typeShot}"] += 1
				statsInfo[playerID][typeShot] -= 1


			where[row] = data[row]
			where[row]["dirs"] = info[0]
			where[row]["elevations"] = info[1]
			where[row]["listedTargetsAngular"] = info[2]
			where[row]["gridTargetsAngular"] = info[3]
			where[row]["listedTargetsAngular2YZ"] = info[4]
			where[row]["gridTargetsAngular2YZ"] = info[5]
			where[row]["listedUtilitiesComputed"] = info[6]
			where[row]["gridUtilitiesComputed"] = info[7]
			where[row]["executedActionAngular"] = info[8]


			negativeElevation = info[10][0]

			if not skip and negativeElevation:
				statsInfo[playerID][f"negativeElevation-{typeShot}"].append((info[10][1],row))


		with open(f"{saveAt}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl","wb") as outfile:
			pickle.dump(playerData,outfile)


		if len(filtered) > 0:
			with open(f"{saveAtFiltered}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_filtered{len(filtered)}.pkl","wb") as outfile:
				pickle.dump(filtered,outfile)


		infoNegativeElevations = statsInfo[playerID][f"negativeElevation-{typeShot}"]
		
		if len(infoNegativeElevations) > 0:
			with open(f"{saveAtNegativeElevations}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_negativeElevations{len(infoNegativeElevations)}.pkl","wb") as outfile:
				pickle.dump(infoNegativeElevations,outfile)


		code.interact("...", local=dict(globals(), **locals()))	


	with open(f"{mainFolder}statsAfterFiltering.txt","w") as outfile:

		for eachID in statsInfo:
			print(f"Player: {eachID}",file=outfile)

			for st in typeShots:
				if st in statsInfo[eachID]:
					print(f"\t{st}: {statsInfo[eachID][st]}",file=outfile)

			for st in typeShots:
				if st in statsInfo[eachID]:
					if statsInfo[eachID][f"filtered-{st}"] > 0:
						print(f"\tfiltered-{st}: {statsInfo[eachID][f'filtered-{st}']}",file=outfile)
					if len(statsInfo[eachID][f"negativeElevation-{st}"]) > 0:
						print(f"\negativeElevation-{st}: {len(statsInfo[eachID][f'negativeElevation-{st}'])}",file=outfile)



	# code.interact("...", local=dict(globals(), **locals()))	
	print("Done.")


