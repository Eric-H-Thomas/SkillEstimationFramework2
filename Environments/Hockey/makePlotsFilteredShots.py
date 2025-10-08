import numpy as np
import sys, os, pickle
import code
from scipy.interpolate import griddata
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap


leftPost = np.array([89,-3])
rightPost = np.array([89,3])

cmapStr = "gist_rainbow"
c = 0.4
n = plt.cm.jet.N
cmap = (1. - c) * plt.get_cmap(cmapStr)(np.linspace(0., 1., n)) + c * np.ones((n, 4))
cmap = ListedColormap(cmap)



# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}makePlotsFilteredShots.py")[0]

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
    # net_width = 6  # 6 feet wide net
    # net_depth = 4  # 4 feet deep
    # goal_net = plt.Rectangle((goal_line_x - net_depth, -net_width / 2), net_depth, net_width, ec="black", fc="none", lw=2)
    # ax.add_patch(goal_net)

    # Adjust plot limits to fit the whole rink and net
    ax.set_xlim(0, goal_line_x + 5)

    # ax.set_ylim(-wall_y, wall_y)

    # y+ bottom (lower half of the rink)
    ax.set_ylim(wall_y,-wall_y)

    # Remove axes for a cleaner plot
    # ax.axis('off')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ticks = np.linspace(0,goal_line_x,10,dtype=int,endpoint=True)
    ax.set_xticks(ticks)

    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False)

    return ax



if __name__ == '__main__':


	try:
		experimentFolder = sys.argv[1]
	except:
		print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"


	folder = f"{mainFolder}AngularHeatmaps-Filtered{os.sep}"
	files = os.listdir(folder)

	saveAt = f"{mainFolder}Plots{os.sep}InfoFilteredShots{os.sep}"
	saveAtRink = f"{saveAt}Rinks{os.sep}"


	for each in [f"{mainFolder}Plots{os.sep}",saveAt,saveAtRink]:
		if not os.path.exists(each):
			os.mkdir(each)

	
	saveAtTextFile = open(saveAt+"filtered.txt","w")


	info = {}

	allLocations = []


	for eachFile in files:

		if "player" in eachFile:

			splitted = eachFile.split("_")
			playerID = splitted[4]
			typeShot = splitted[-2]


			if playerID not in info:
				info[playerID] = {}

			if typeShot not in info[playerID]:
				info[playerID][typeShot] = []


			with open(folder+eachFile,"rb") as infile:
				data = pickle.load(infile)



			info[playerID][typeShot] = list(data.keys())


			for index in data:

				heatmap = data[index]["heat_map"]
				playerLocation = [data[index]["start_x"],data[index]["start_y"]]
				projectedZ = data[index]["projected_z"]
				# shot_location = final_y, projected_z, start_x, start_y
				executedAction = [data[index]["shot_location"][0],data[index]["shot_location"][1]]

				allLocations.append(playerLocation)


				ax = drawRink()
				ax.scatter(playerLocation[0],playerLocation[1])
				ax.plot([playerLocation[0],leftPost[0]],[playerLocation[1],leftPost[1]], color='blue')
				ax.plot([playerLocation[0],rightPost[0]],[playerLocation[1],rightPost[1]], color='blue')
				ax.set_title(f"Player Location: {playerLocation}")
				plt.tight_layout()
				plt.savefig(f"{saveAtRink}rink-{playerID}-{typeShot}-{index}.jpg",bbox_inches="tight")
				plt.close()
				plt.clf()


				shape = heatmap.shape
				listedUtilities = heatmap.reshape((shape[0]*shape[1],1))


				Zs = data[index]["gridUtilitiesComputed"]
				gridTargetsAngular = data[index]["gridTargetsAngular"]
				listedTargetsAngular = data[index]["listedTargetsAngular"]
				executedAction = [data[index]["shot_location"][0],data[index]["shot_location"][1]]
				executedActionAngular = data[index]["executedActionAngular"]


				dirs,elevations = data[index]["dirs"],data[index]["elevations"]

				# Assumming both same size (-1 to ofset for index-based 0)
				middle = int(len(dirs)/2) - 1
				mean = [dirs[middle],elevations[middle]]


				spaces.delta = [abs(dirs[0]-dirs[1]),abs(elevations[0]-elevations[1])]

				playerLocation = [data[index]["start_x"],data[index]["start_y"]]


				fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

				norm = plt.Normalize(0.0,np.max(listedUtilities))
				sm = ScalarMappable(norm=norm,cmap=cmap)
				sm.set_array([])

				ax1.scatter(listedTargetsUtilityGridYZ[:,0],listedTargetsUtilityGridYZ[:,1],c=cmap(norm(listedUtilities)))
				ax1.scatter(executedAction[0],executedAction[1],c="black",marker="x",label="(final_y, projected_z)")
				ax1.set_title('Given Heatmap')
				ax1.set_xlabel("Y")
				ax1.set_ylabel("Z")
				ax1.legend()
				fig.colorbar(sm,ax=ax1)


				ax2.scatter(listedTargetsAngular[:,0],listedTargetsAngular[:,1],c=cmap(norm(Zs.flatten())))
				ax2.scatter(executedActionAngular[0],executedActionAngular[1],c="black",marker="x")
				ax2.set_title('Computed Heatmap - Angular')
				ax2.set_xlabel("Direction")
				ax2.set_ylabel("Elevation")
				fig.colorbar(sm,ax=ax2)

				plt.suptitle(f"Player Location: {playerLocation}")
				plt.tight_layout()

				plt.savefig(f"{saveAt}{os.sep}{playerID}-{typeShot}-{index}.jpg",bbox_inches="tight")		
				plt.close()


				print(f"{playerID},{typeShot},{index}",file=saveAtTextFile)


	saveAtTextFile.close()


	allLocations = np.array(allLocations)

	ax = drawRink()
	ax.scatter(allLocations[:,0],allLocations[:,1])
	ax.set_title(f"Player Locations")
	plt.tight_layout()
	plt.savefig(f"{saveAtRink}allLocations.jpg",bbox_inches="tight")
	plt.close()
	plt.clf()



	# code.interact("...", local=dict(globals(), **locals()))	
	print("Done.")


