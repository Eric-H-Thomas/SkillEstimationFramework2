"""Visualization and analysis of filtered hockey shot data.

This module processes filtered angular heatmap data to generate comprehensive
visualizations and metadata for hockey shot attempts that have passed quality
filters. Unlike ``makePlotsAngularHeatmaps.py`` which processes all shots, this
script focuses specifically on filtered shots stored in the AngularHeatmaps-Filtered
directory.

The script performs batch processing across all filtered shot data files, generating:
- Individual rink diagrams showing player positions and shooting angles
- Side-by-side YZ (Cartesian) and angular heatmap comparisons
- A summary plot showing all player locations across all filtered shots
- A text file index of all processed shots for reference

Key Features
------------
- Automated batch processing of all filtered shot data in an experiment
- Organizes output by player ID, shot type, and attempt index
- Creates a comprehensive location map showing where filtered shots originated
- Generates a text manifest (filtered.txt) listing all processed shots

Functions
---------
``drawRink``
    Creates a stylized hockey rink diagram with goal, crease, and face-off
    circle. Unlike the version in makePlotsAngularHeatmaps.py, this includes
    visible axis ticks and labels for precise location reference.

Command-Line Usage
------------------
When run as a script, expects one argument:
    1. experimentFolder: Name of the experiment folder under 'Experiments/hockey-multi/'

Example:
    python makePlotsFilteredShots.py my_experiment

Input Requirements
------------------
Expects filtered angular heatmap data files in:
    Experiments/hockey-multi/{experimentFolder}/Data/AngularHeatmaps-Filtered/
    
Files should follow the naming convention:
    angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_filtered{N}.pkl
    where {N} is the count of filtered shots for that player/shot-type combination

Output Structure
----------------
Plots and data are saved to:
    Experiments/hockey-multi/{experimentFolder}/Data/Plots/InfoFilteredShots/
    - Individual heatmap comparisons: {playerID}-{typeShot}-{index}.jpg
    - Rink diagrams in Rinks/ subdirectory
    - allLocations.jpg: Aggregate plot of all shot origin points
    - filtered.txt: CSV-style manifest of all processed shots (playerID, typeShot, index)

Notes
-----
This script is typically run after shot data has been filtered to remove outliers
or shots that don't meet quality criteria. The filtering process (which generates
the AngularHeatmaps-Filtered directory and its .pkl files) is performed by
``getAngularHeatmapsPerPlayer.py`` before this visualization step. This script
assumes the filtered data already exists and focuses solely on visualization and
analysis of that pre-filtered dataset.
"""

import numpy as np
import sys
import os
import pickle
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
#  Global configuration
# ---------------------------------------------------------------------------

# Define goal post positions (standard NHL net is 6 feet wide)
# Goal line is at x=89, posts at y=±3
leftPost = np.array([89, -3])
rightPost = np.array([89, 3])

# Create a custom colormap for the heatmap visualizations
# This adds a constant brightness offset to make colors more visible
cmapStr = "gist_rainbow"
c = 0.4  # Brightness offset factor
n = plt.cm.jet.N
cmap = (1.0 - c) * plt.get_cmap(cmapStr)(np.linspace(0.0, 1.0, n)) + c * np.ones((n, 4))
cmap = ListedColormap(cmap)

# ---------------------------------------------------------------------------
#  Dynamic module setup
# ---------------------------------------------------------------------------
# The framework dynamically imports the hockey environment, spaces, and
# estimator utility modules to ensure plotting uses the same discretizations
# and utilities as the simulation environment.

scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}makePlotsFilteredShots.py")[0]

module = SourceFileLoader("hockey.py", f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

module = SourceFileLoader("setupSpaces.py", f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module

module = SourceFileLoader("utils.py", f"{mainFolderName}Estimators{os.sep}utils.py").load_module()
sys.modules["utils"] = module

# Build the target grid using the spaces helper
# Delta is the discretization resolution in feet
delta = 1.0  # Originally could be 0.16 for finer resolution

spaces = sys.modules["spaces"].SpacesHockey([], 1, sys.modules["domain"], delta)

# Extract the Y and Z target coordinates from the spaces object
Y = spaces.targetsY
Z = spaces.targetsZ

# Create a meshgrid of all YZ target combinations for plotting
targetsUtilityGridY, targetsUtilityGridZ = np.meshgrid(Y, Z)
targetsUtilityGridYZ = np.stack((targetsUtilityGridY, targetsUtilityGridZ), axis=-1)

# Flatten the grid for scatter plotting
shape = targetsUtilityGridYZ.shape
listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0] * shape[1], shape[2]))


# ---------------------------------------------------------------------------
#  Rink visualization
# ---------------------------------------------------------------------------

def drawRink():
    """Create a stylized hockey rink diagram with key landmarks and axis labels.

    Draws the offensive half of a hockey rink including the goal line, crease,
    and face-off circle. Uses standard NHL dimensions where applicable. Unlike
    the version in makePlotsAngularHeatmaps.py, this keeps axis ticks and labels
    visible for precise location reference, which is useful for filtered shot
    analysis.

    Returns
    -------
    plt.Axes
        Matplotlib axes object with rink diagram drawn, ready for additional
        annotations like player positions and shooting lines.

    Notes
    -----
    The goal net drawing is commented out but can be enabled if needed.
    The spines are hidden while keeping the tick marks and labels visible
    for a cleaner look with positional information.
    """

    fig, ax = plt.subplots()

    # Use equal aspect ratio so the rink proportions appear correct
    ax.set_aspect(1)

    # Define rink dimensions (feet, standard NHL specifications)
    rink_length = 89    # Distance from center ice to goal line (x-direction)
    rink_width = 85     # Full rink width (y-direction, -42.5 to +42.5)
    goal_line_x = 89    # Goal line position
    wall_y = 42.5       # Side boards at y = ±42.5

    # Draw the rink boundary (only the offensive half where the goal is located)
    rink_boundary = plt.Rectangle((0, -wall_y), goal_line_x, 2 * wall_y, ec="blue", fc="none", lw=2)
    ax.add_patch(rink_boundary)

    # Draw the goal line (red line at x = 89 feet)
    ax.plot([goal_line_x, goal_line_x], [-wall_y, wall_y], color='red', lw=2)

    # Draw the goal crease (semi-circle in front of the net)
    crease_radius = 6  # Standard 6-foot radius
    crease = plt.Circle((goal_line_x, 0), crease_radius, color="red", fill=False, lw=2)
    ax.add_patch(crease)

    # Draw the face-off circle in the offensive zone
    faceoff_circle_radius = 15
    faceoff_circle_x = 69  # Position at x = 69 feet
    faceoff_circle = plt.Circle((faceoff_circle_x, 0), faceoff_circle_radius, color="red", fill=False, lw=2)
    ax.add_patch(faceoff_circle)

    # Goal net drawing is disabled but available if needed
    # net_width = 6  # Standard 6-foot wide net
    # net_depth = 4  # 4-foot depth behind goal line
    # goal_net = plt.Rectangle((goal_line_x - net_depth, -net_width / 2), net_depth, net_width, ec="black", fc="none", lw=2)
    # ax.add_patch(goal_net)

    # Set plot limits to show the full offensive zone plus some margin
    ax.set_xlim(0, goal_line_x + 5)

    # Set y-axis with positive values at bottom (inverted for visualization preference)
    # This makes the rink appear with the "natural" orientation for hockey
    ax.set_ylim(wall_y, -wall_y)

    # Hide the axis spines but keep ticks visible for location reference
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Position tick marks on the left and bottom
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Create evenly spaced x-axis ticks for distance reference
    ticks = np.linspace(0, goal_line_x, 10, dtype=int, endpoint=True)
    ax.set_xticks(ticks)

    # Configure tick parameters to show only on appropriate sides
    ax.tick_params(axis='y', which='both', left=True, right=False)
    ax.tick_params(axis='x', which='both', bottom=True, top=False)

    return ax


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line argument for the experiment folder
    try:
        experimentFolder = sys.argv[1]
    except Exception:
        print(
            "Need to specify the name of the folder for the experiment (located under\n"
            "'Experiments/hockey-multi/') as command line argument."
        )
        sys.exit(1)

    # Build paths to input and output directories
    mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"

    # Input: folder containing filtered angular heatmap pickle files
    folder = f"{mainFolder}AngularHeatmaps-Filtered{os.sep}"
    files = os.listdir(folder)

    # Output: directories for plots and rink diagrams
    saveAt = f"{mainFolder}Plots{os.sep}InfoFilteredShots{os.sep}"
    saveAtRink = f"{saveAt}Rinks{os.sep}"

    # Create all necessary output directories
    for each in [f"{mainFolder}Plots{os.sep}", saveAt, saveAtRink]:
        os.makedirs(each, exist_ok=True)

    # Open a text file to log all processed shots
    saveAtTextFile = open(saveAt + "filtered.txt", "w")

    # Dictionary to organize shot indices by player and shot type
    info = {}

    # List to accumulate all player locations for aggregate visualization
    allLocations = []

    # Process each file in the filtered heatmaps directory
    for eachFile in files:
        # Only process files that contain player data
        if "player" in eachFile:
            # Parse the filename to extract player ID and shot type
            # Expected format: angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_filtered.pkl
            splitted = eachFile.split("_")
            playerID = splitted[4]
            typeShot = splitted[-2]

            # Initialize nested dictionary structure if needed
            if playerID not in info:
                info[playerID] = {}

            if typeShot not in info[playerID]:
                info[playerID][typeShot] = []

            # Load the filtered angular heatmap data for this player/shot combination
            with open(folder + eachFile, "rb") as infile:
                data = pickle.load(infile)

            # Store the indices of all shots for this player/shot combination
            info[playerID][typeShot] = list(data.keys())

            # Process each individual shot attempt in the data
            for index in data:
                # Extract basic shot information
                heatmap = data[index]["heat_map"]
                playerLocation = [data[index]["start_x"], data[index]["start_y"]]
                projectedZ = data[index]["projected_z"]
                
                # shot_location = final_y, projected_z, start_x, start_y
                # Extract only the first two elements (final_y, projected_z)
                executedAction = [data[index]["shot_location"][0], data[index]["shot_location"][1]]

                # Accumulate player location for aggregate visualization later
                allLocations.append(playerLocation)

                # Create a rink diagram showing the player position and shooting angles
                ax = drawRink()
                ax.scatter(playerLocation[0], playerLocation[1])  # Mark player position
                # Draw lines from player to both goal posts to show shooting angle
                ax.plot([playerLocation[0], leftPost[0]], [playerLocation[1], leftPost[1]], color='blue')
                ax.plot([playerLocation[0], rightPost[0]], [playerLocation[1], rightPost[1]], color='blue')
                ax.set_title(f"Player Location: {playerLocation}")
                plt.tight_layout()
                plt.savefig(f"{saveAtRink}rink-{playerID}-{typeShot}-{index}.jpg", bbox_inches="tight")
                plt.close()
                plt.clf()

                # Flatten the heatmap for scatter plotting
                shape = heatmap.shape
                listedUtilities = heatmap.reshape((shape[0] * shape[1], 1))

                # Extract angular heatmap data computed during filtering
                Zs = data[index]["gridUtilitiesComputed"]
                gridTargetsAngular = data[index]["gridTargetsAngular"]
                listedTargetsAngular = data[index]["listedTargetsAngular"]
                executedAction = [data[index]["shot_location"][0], data[index]["shot_location"][1]]
                executedActionAngular = data[index]["executedActionAngular"]

                # Get direction and elevation arrays from the angular transformation
                dirs, elevations = data[index]["dirs"], data[index]["elevations"]

                # Find the center point of the angular grid
                # Assuming both arrays have the same size (-1 to offset for 0-based indexing)
                middle = int(len(dirs) / 2) - 1
                mean = [dirs[middle], elevations[middle]]

                # Update spaces delta to match the resolution of this angular grid
                spaces.delta = [abs(dirs[0] - dirs[1]), abs(elevations[0] - elevations[1])]

                # Re-extract player location (already have it, but kept for consistency with original)
                playerLocation = [data[index]["start_x"], data[index]["start_y"]]

                # Create side-by-side comparison of YZ and angular heatmaps
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

                # Set up color normalization based on the utility values
                norm = plt.Normalize(0.0, np.max(listedUtilities))
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])

                # Left subplot: Original YZ (Cartesian) heatmap
                ax1.scatter(
                    listedTargetsUtilityGridYZ[:, 0],
                    listedTargetsUtilityGridYZ[:, 1],
                    c=cmap(norm(listedUtilities)),
                )
                # Mark the executed shot location with a black X
                ax1.scatter(executedAction[0], executedAction[1], c="black", marker="x", label="(final_y, projected_z)")
                ax1.set_title('Given Heatmap')
                ax1.set_xlabel("Y")
                ax1.set_ylabel("Z")
                ax1.legend()
                fig.colorbar(sm, ax=ax1)

                # Right subplot: Angular (direction-elevation) heatmap
                ax2.scatter(
                    listedTargetsAngular[:, 0],
                    listedTargetsAngular[:, 1],
                    c=cmap(norm(Zs.flatten())),
                )
                # Mark the executed shot in angular coordinates
                ax2.scatter(executedActionAngular[0], executedActionAngular[1], c="black", marker="x")
                ax2.set_title('Computed Heatmap - Angular')
                ax2.set_xlabel("Direction")
                ax2.set_ylabel("Elevation")
                fig.colorbar(sm, ax=ax2)

                # Add overall title and save the figure
                plt.suptitle(f"Player Location: {playerLocation}")
                plt.tight_layout()
                plt.savefig(f"{saveAt}{os.sep}{playerID}-{typeShot}-{index}.jpg", bbox_inches="tight")
                plt.close()

                # Log this shot to the manifest file
                print(f"{playerID},{typeShot},{index}", file=saveAtTextFile)

    # Close the manifest file
    saveAtTextFile.close()

    # Create an aggregate visualization showing all player locations
    allLocations = np.array(allLocations)

    ax = drawRink()
    ax.scatter(allLocations[:, 0], allLocations[:, 1])
    ax.set_title(f"Player Locations")
    plt.tight_layout()
    plt.savefig(f"{saveAtRink}allLocations.jpg", bbox_inches="tight")
    plt.close()
    plt.clf()

    print("Done.")


