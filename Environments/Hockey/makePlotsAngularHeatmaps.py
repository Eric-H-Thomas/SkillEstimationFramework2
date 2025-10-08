import matplotlib.pyplot as plt
import numpy as np
import os,sys,code
from importlib.machinery import SourceFileLoader
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import pickle, random

from getAngularHeatmapsPerPlayer import getAngularHeatmap


# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}makePlotsAngularHeatmaps.py")[0]

module = SourceFileLoader("hockey.py",f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module


module = SourceFileLoader("setupSpaces.py",f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


def drawRink():

    fig, ax = plt.subplots()

    # Set the aspect ratio to 1 for equal scaling
    ax.set_aspect(1)

    # Define rink dimensions based on the user's specifications
    rink_length = 89  # The length up to the goal line (x-direction)
    rink_width = 85   # Full width of the rink (y-direction, from -42.5 to +42.5)
    goal_line_x = 89  # Goal line is at 89 feet
    wall_y = 42.5     # Left and right walls are at y = ±42.5

    # Draw rink boundary (only the half where the goal is)
    rink_boundary = plt.Rectangle((0, -wall_y), goal_line_x, 2 * wall_y, ec="blue", fc="none", lw=2)
    ax.add_patch(rink_boundary)

    # Draw the goal line at x = 89 feet
    ax.plot([goal_line_x, goal_line_x], [-wall_y, wall_y], color='red', lw=2)

    # Draw the goal crease (semi-circle in front of the goal)
    crease_radius = 6  # 6 feet radius for crease
    crease = plt.Circle((goal_line_x, 0), crease_radius, color="red", fill=False, lw=2)
    ax.add_patch(crease)

    # Draw the face-off circle at a distance from the goal
    faceoff_circle_radius = 15
    faceoff_circle_x = 69  # Place the faceoff circle at x = 69 feet
    faceoff_circle = plt.Circle((faceoff_circle_x, 0), faceoff_circle_radius, color="red", fill=False, lw=2)
    ax.add_patch(faceoff_circle)

    # Draw the goal net (small rectangle on the goal line)
    net_width = 6  # 6 feet wide net
    net_depth = 4  # 4 feet deep
    goal_net = plt.Rectangle((goal_line_x - net_depth, -net_width / 2), net_depth, net_width, ec="black", fc="none", lw=2)
    ax.add_patch(goal_net)

    # Adjust plot limits to fit the whole rink and net
    ax.set_xlim(0, goal_line_x + 5)

    # ax.set_ylim(-wall_y, wall_y)

    # y+ bottom (lower half of the rink)
    ax.set_ylim(wall_y,-wall_y)

    # Remove axes for a cleaner plot
    ax.axis('off')

    return ax


def makePlots(experimentFolder,playerID,typeShot):

	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"

	saveAt = f"{mainFolder}{os.sep}Plots{os.sep}Heatmaps{os.sep}Player{playerID}{os.sep}{typeShot}{os.sep}"
	saveAt1 = f"{saveAt}{os.sep}Rink{os.sep}" 


	folders = [f"{mainFolder}{os.sep}Plots{os.sep}",f"{mainFolder}{os.sep}Plots{os.sep}Heatmaps{os.sep}",f"{mainFolder}Plots{os.sep}Heatmaps{os.sep}Player{playerID}{os.sep}",
				saveAt,saveAt1]

	for folder in folders:
		#If the folder doesn't exist already, create it
		if not os.path.exists(folder):
			os.mkdir(folder)



	folder = f"{mainFolder}AngularHeatmaps{os.sep}"
	fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"

	try:
		with open(folder+fileName,"rb") as infile:
			data = pickle.load(infile)
	except Exception as e:
		print(e)
		print("Can't load data for that player.")
		exit()


	cmapStr = "gist_rainbow"
	c = 0.4
	n = plt.cm.jet.N
	cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
	cmap = ListedColormap(cmap)



	# Feet
	delta = 1.0

	spaces = sys.modules["spaces"].SpacesHockey([],1,sys.modules["domain"],delta)

	Y = spaces.targetsY
	Z = spaces.targetsZ

	targetsUtilityGridY,targetsUtilityGridZ = np.meshgrid(Y,Z)
	targetsUtilityGridYZ = np.stack((targetsUtilityGridY,targetsUtilityGridZ),axis=-1)

	shape = targetsUtilityGridYZ.shape
	listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0]*shape[1],shape[2]))



	leftPost = np.array([89,-3])
	rightPost = np.array([89,3])


	# n = len(data)
	n = 10

	try:
		randKeys = random.sample(list(data.keys()),n)
	except:
		randKeys = random.sample(list(data.keys()),len(data))


	data = {key: data[key] for key in randKeys}


	for index in data:

		playerLocation = [data[index]["start_x"],data[index]["start_y"]]


		locations = [playerLocation,[88.5,0]]

		for i in range(len(locations)):

			playerLocation = locations[i]

			ax = drawRink()
			ax.scatter(playerLocation[0],playerLocation[1])
			ax.plot([playerLocation[0],leftPost[0]],[playerLocation[1],leftPost[1]], color='blue')
			ax.plot([playerLocation[0],rightPost[0]],[playerLocation[1],rightPost[1]], color='blue')
			ax.set_title(f"Index: {index} | PlayerLocation: {playerLocation}")
			plt.tight_layout()
			plt.savefig(f"{saveAt1}rink-index{index}-location{playerLocation}.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()

			heatmap = data[index]["heat_map"]

			# shot_location = final_y, projected_z, start_x, start_y
			executedAction = [data[index]["shot_location"][0],data[index]["shot_location"][1]]

			shape = heatmap.shape
			listedUtilities = heatmap.reshape((shape[0]*shape[1],1))


			# For all additional locations (not including actual player location)
			if i >= 1:

				dirs,elevations,listedTargetsAngular,gridTargetsAngular,\
				listedTargetsAngular2YZ,gridTargetsAngular2YZ,\
				listedUtilitiesComputed,gridUtilitiesComputed,executedActionAngular,skip \
					= getAngularHeatmap(heatmap,playerLocation,executedAction)
				
				angularHeatmap = listedUtilitiesComputed

			# Angular heatmap for actual player location already in data. Can just load.
			else:
				angularHeatmap = data[index]["listedUtilitiesComputed"]
				listedTargetsAngular = data[index]["listedTargetsAngular"]
				executedActionAngular = data[index]["executedActionAngular"]

			listedTargetsAngular = np.array(listedTargetsAngular)



			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

			norm = plt.Normalize(0.0,np.max(listedUtilities))
			sm = ScalarMappable(norm=norm,cmap=cmap)
			sm.set_array([])

			ax1.scatter(listedTargetsUtilityGridYZ[:,0],listedTargetsUtilityGridYZ[:,1],c=cmap(norm(listedUtilities)))
			ax1.scatter(executedAction[0],executedAction[1],c="black",marker="x")
			ax1.set_title('Given Heatmap')
			ax1.set_xlabel("Y")
			ax1.set_ylabel("Z")
			fig.colorbar(sm,ax=ax1)


			ax2.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=cmap(norm(angularHeatmap)))
			ax2.scatter(executedActionAngular[0],executedActionAngular[1],c="black",marker="x")
			ax2.set_title('Computed Heatmap - Angular')
			ax2.set_xlabel("Direction")
			ax2.set_ylabel("Elevation")
			fig.colorbar(sm,ax=ax2)

			plt.suptitle(f"Player Location: {playerLocation}")
			plt.tight_layout()
			plt.savefig(f"{saveAt}index{index}-location{playerLocation}.jpg",bbox_inches="tight")
			plt.close()
			plt.clf()


			# code.interact("after...", local=dict(globals(), **locals()))

			
if __name__ == '__main__':


	try:
		experimentFolder = sys.argv[1]
		playerID = sys.argv[2]
		typeShot = sys.argv[3]
	except:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/'), the ID of the player and type of shot as command line argument.")
		exit()


	makePlots(experimentFolder,playerID,typeShot)


