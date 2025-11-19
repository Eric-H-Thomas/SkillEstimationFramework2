"""Visualization utilities for hockey angular heatmap analysis.

This module generates visual comparisons between Cartesian (YZ) and angular
(direction-elevation) heatmaps for hockey shot analysis. It is invoked after
angular heatmap data has been serialized by the hockey environment experiments,
and produces side-by-side visualizations showing how utility distributions
transform between coordinate systems.

The script dynamically loads the ``hockey.py`` environment module and
``setupSpaces.py`` to ensure consistency with the discretized target grids
used during simulation. It also imports ``getAngularHeatmapsPerPlayer`` to
compute angular heatmaps for hypothetical player locations.

Key Features
------------
- Generates rink diagrams showing player location and shooting angles
- Produces dual-plot visualizations comparing YZ and angular heatmaps
- Supports analysis from both actual and hypothetical player positions
- Uses consistent color mapping across all plots for easy comparison

Functions
---------
``drawRink``
    Creates a stylized hockey rink diagram with goal, crease, and face-off
    circle marked. Used as a base layer for plotting player positions and
    shooting angles.

``makePlots``
    Main plotting function that loads serialized angular heatmap data for a
    specific player and shot type, then generates comparison visualizations.
    For each shot attempt, it creates:
    - A rink diagram showing player position and shooting angles to goal posts
    - Side-by-side heatmaps in YZ (Cartesian) and angular coordinate systems
    - Optional visualizations from hypothetical positions for comparison

Command-Line Usage
------------------
When run as a script, expects three arguments:
    1. experimentFolder: Name of the experiment folder under 'Experiments/hockey-multi/'
    2. playerID: Identifier for the specific player to analyze
    3. typeShot: Type of shot to visualize (e.g., 'wristshot', 'snapshot')

Example:
    python makePlotsAngularHeatmaps.py my_experiment player_001 wristshot

Output Structure
----------------
Plots are saved to:
    Experiments/hockey-multi/{experimentFolder}/Data/Plots/Heatmaps/Player{playerID}/{typeShot}/
    - Main heatmap comparisons in the base directory
    - Rink diagrams in the Rink/ subdirectory
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from importlib.machinery import SourceFileLoader
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import pickle
import random

from getAngularHeatmapsPerPlayer import getAngularHeatmap

# ---------------------------------------------------------------------------
#  Dynamic module setup
# ---------------------------------------------------------------------------
# The framework dynamically imports the hockey environment and spaces modules
# to ensure plotting uses the same discretizations and utilities as the
# simulation environment.
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}makePlotsAngularHeatmaps.py")[0]

module = SourceFileLoader("hockey.py", f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

module = SourceFileLoader("setupSpaces.py", f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module


# ---------------------------------------------------------------------------
#  Rink visualization
# ---------------------------------------------------------------------------

def drawRink() -> plt.Axes:
    """Create a stylized hockey rink diagram with key landmarks.

    Draws the offensive half of a hockey rink including the goal line, crease,
    face-off circle, and net. Uses standard NHL dimensions where applicable.
    The coordinate system places the goal at x=89 feet with y=0 at center ice.

    Returns
    -------
    plt.Axes
        Matplotlib axes object with rink diagram drawn, ready for additional
        annotations like player positions and shooting lines.
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

    # Draw the goal net (small rectangle on the goal line)
    net_width = 6   # Standard 6-foot wide net
    net_depth = 4   # 4-foot depth behind goal line
    goal_net = plt.Rectangle((goal_line_x - net_depth, -net_width / 2), net_depth, net_width, ec="black", fc="none", lw=2)
    ax.add_patch(goal_net)

    # Set plot limits to show the full offensive zone plus some margin
    ax.set_xlim(0, goal_line_x + 5)

    # Set y-axis with positive values at bottom (inverted for visualization preference)
    # This makes the rink appear with the "natural" orientation for hockey
    ax.set_ylim(wall_y, -wall_y)

    # Remove axis labels and ticks for a cleaner diagram
    ax.axis('off')

    return ax


# ---------------------------------------------------------------------------
#  Heatmap plotting
# ---------------------------------------------------------------------------

def makePlots(experimentFolder: str, playerID: str, typeShot: str) -> None:
    """Generate heatmap comparison plots for a specific player and shot type.

    Loads serialized angular heatmap data for the specified player and shot
    type, then generates visualizations comparing the original YZ (Cartesian)
    heatmaps with their angular (direction-elevation) representations. Also
    creates rink diagrams showing player position and shooting angles.

    Parameters
    ----------
    experimentFolder : str
        Name of the experiment folder under 'Experiments/hockey-multi/'.
    playerID : str
        Identifier for the player whose shots should be visualized.
    typeShot : str
        Type of shot to analyze (e.g., 'wristshot', 'snapshot', 'slapshot').

    Notes
    -----
    The function processes a random sample of up to 10 shot attempts from the
    player's data to avoid generating excessive plots. For each attempt, it
    creates:
    - A rink diagram showing the actual player position
    - Optional rink diagrams for hypothetical comparison positions
    - Dual heatmap plots (YZ vs angular) for each position

    All plots are saved to the experiment's Data/Plots/Heatmaps/ directory.
    """

    # Build the directory structure for this experiment's outputs
    mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"
    saveAt = f"{mainFolder}{os.sep}Plots{os.sep}Heatmaps{os.sep}Player{playerID}{os.sep}{typeShot}{os.sep}"
    saveAt1 = f"{saveAt}{os.sep}Rink{os.sep}"

    # Create all necessary subdirectories if they don't exist
    folders = [
        f"{mainFolder}{os.sep}Plots{os.sep}",
        f"{mainFolder}{os.sep}Plots{os.sep}Heatmaps{os.sep}",
        f"{mainFolder}Plots{os.sep}Heatmaps{os.sep}Player{playerID}{os.sep}",
        saveAt,
        saveAt1,
    ]

    for folder in folders:
        # Use os.makedirs for cleaner directory creation
        os.makedirs(folder, exist_ok=True)

    # Load the pickled angular heatmap data for this player and shot type
    folder = f"{mainFolder}AngularHeatmaps{os.sep}"
    fileName = f"angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl"

    try:
        with open(folder + fileName, "rb") as infile:
            data = pickle.load(infile)
    except Exception as exc:
        print(exc)
        print("Can't load data for that player.")
        sys.exit(1)

    # Create a custom colormap for the heatmap visualizations
    # This adds a constant brightness offset to make colors more visible
    cmapStr = "gist_rainbow"
    c = 0.4  # Brightness offset factor
    n = plt.cm.jet.N
    cmap = (1.0 - c) * plt.get_cmap(cmapStr)(np.linspace(0.0, 1.0, n)) + c * np.ones((n, 4))
    cmap = ListedColormap(cmap)

    # Build the target grid using the spaces helper
    # Delta is the discretization resolution in feet
    delta = 1.0
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

    # Define goal post positions (standard NHL net is 6 feet wide)
    # Goal line is at x=89, posts at y=±3
    leftPost = np.array([89, -3])
    rightPost = np.array([89, 3])

    # Randomly sample up to 10 shot attempts to avoid generating too many plots
    # To process all attempts, change to: n = len(data)
    # Note: If n = len(data), the sampling below becomes unnecessary (can just iterate over data directly)
    n = 10

    try:
        randKeys = random.sample(list(data.keys()), n)
    except Exception:
        # If there are fewer than n attempts, use all available
        randKeys = random.sample(list(data.keys()), len(data))

    # Filter the data dictionary to only the sampled keys
    data = {key: data[key] for key in randKeys}

    # Process each shot attempt in the sampled data
    for index in data:
        # Extract the actual player position for this shot attempt
        playerLocation = [data[index]["start_x"], data[index]["start_y"]]

        # Generate visualizations for both the actual player location and
        # a hypothetical location near the goal for comparison
        locations = [playerLocation, [88.5, 0]]

        for i in range(len(locations)):
            playerLocation = locations[i]

            # Create a rink diagram showing the player position and shooting angles
            ax = drawRink()
            ax.scatter(playerLocation[0], playerLocation[1])  # Mark player position
            # Draw lines from player to both goal posts to show shooting angle
            ax.plot([playerLocation[0], leftPost[0]], [playerLocation[1], leftPost[1]], color='blue')
            ax.plot([playerLocation[0], rightPost[0]], [playerLocation[1], rightPost[1]], color='blue')
            ax.set_title(f"Index: {index} | PlayerLocation: {playerLocation}")
            plt.tight_layout()
            plt.savefig(f"{saveAt1}rink-index{index}-location{playerLocation}.jpg", bbox_inches="tight")
            plt.close()
            plt.clf()

            # Extract the utility heatmap for this shot attempt
            heatmap = data[index]["heat_map"]

            # shot_location = final_y, projected_z, start_x, start_y
            # Extract only the first two elements (final_y, projected_z)
            executedAction = [data[index]["shot_location"][0], data[index]["shot_location"][1]]

            # Flatten the heatmap for scatter plotting
            shape = heatmap.shape
            listedUtilities = heatmap.reshape((shape[0] * shape[1], 1))

            # For hypothetical locations (i >= 1), compute the angular heatmap on-the-fly
            # For the actual player location (i == 0), load pre-computed angular data
            if i >= 1:
                # Compute angular heatmap transformation for this hypothetical position
                (
                    dirs,
                    elevations,
                    listedTargetsAngular,
                    gridTargetsAngular,
                    listedTargetsAngular2YZ,
                    gridTargetsAngular2YZ,
                    listedUtilitiesComputed,
                    gridUtilitiesComputed,
                    executedActionAngular,
                    skip,
                ) = getAngularHeatmap(heatmap, playerLocation, executedAction)

                angularHeatmap = listedUtilitiesComputed

            else:
                # For the actual player location, the angular heatmap was pre-computed
                # and stored in the data file, so just load it
                angularHeatmap = data[index]["listedUtilitiesComputed"]
                listedTargetsAngular = data[index]["listedTargetsAngular"]
                executedActionAngular = data[index]["executedActionAngular"]

            # Ensure targets are in array format for plotting
            listedTargetsAngular = np.array(listedTargetsAngular)

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
            ax1.scatter(executedAction[0], executedAction[1], c="black", marker="x")
            ax1.set_title('Given Heatmap')
            ax1.set_xlabel("Y")
            ax1.set_ylabel("Z")
            fig.colorbar(sm, ax=ax1)

            # Right subplot: Angular (direction-elevation) heatmap
            ax2.scatter(
                listedTargetsAngular[:, 0],
                listedTargetsAngular[:, 1],
                c=cmap(norm(angularHeatmap)),
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
            plt.savefig(f"{saveAt}index{index}-location{playerLocation}.jpg", bbox_inches="tight")
            plt.close()
            plt.clf()


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Parse command-line arguments for experiment folder, player ID, and shot type
    try:
        experimentFolder = sys.argv[1]
        playerID = sys.argv[2]
        typeShot = sys.argv[3]
    except Exception:
        print(
            "Need to specify the name of the folder for the experiment (located under\n"
            "'Experiments/hockey-multi/'), the ID of the player and type of shot as\n"
            "command line argument."
        )
        sys.exit(1)

    # Generate the heatmap comparison plots
    makePlots(experimentFolder, playerID, typeShot)


