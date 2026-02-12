"""
Generate angular heatmaps for hockey player shot data.

This script converts Cartesian heatmap data (Y/Z utility values for a player
shooting from a given position) into angular coordinates (direction/elevation)
so that utilities can be analyzed relative to the net. It loads supporting
modules dynamically, computes angular grids, interpolates utilities, and saves
per-player/per-shot-type results.

Functions
---------
getAngle(point1, point2)
    Compute the polar angle (in radians) from point1 to point2.
getAngularHeatmap(heatmap, playerLocation, executedAction)
    Convert a player's Cartesian heatmap and executed shot into angular space,
    returning both the angular grid and the interpolated utility values.

How it fits
-----------
This file is part of the Hockey environment preprocessing pipeline. It takes
heatmaps produced elsewhere in the experiment folder structure and transforms
them into an angular representation used by evaluators in the Estimators
package. The generated pickle files are consumed by downstream analysis tools
to study player shot tendencies and performance in angular space.
"""

import code
import os
import pickle
import sys
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Net geometry constants (in feet)
NET_GOAL_LINE_X = 89  # Distance from blueline to goal line
NET_POST_Y = 3  # Half-width of actual net
NET_POST_Z = 0  # Posts at ice level
NET_HEIGHT = 4  # Actual net height

# Augmented geometry (extended range to capture misses)
AUGMENTED_POST_Y = 9  # Extended Y range for missed shots
AUGMENTED_HEIGHT_Z = 8  # Extended height for missed shots

# Angular grid resolution (number of samples in each dimension)
ANGULAR_GRID_RESOLUTION = 100

# Find location of current file to resolve project-root-relative imports.
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split(f"Environments{os.sep}Hockey{os.sep}getAngularHeatmapsPerPlayer.py")[0]

# Dynamically load project modules without altering import paths.
module = SourceFileLoader("hockey.py", f"{mainFolderName}Environments{os.sep}Hockey{os.sep}hockey.py").load_module()
sys.modules["domain"] = module

module = SourceFileLoader("setupSpaces.py", f"{mainFolderName}setupSpaces.py").load_module()
sys.modules["spaces"] = module

module = SourceFileLoader("utils.py", f"{mainFolderName}Estimators{os.sep}utils.py").load_module()
sys.modules["utils"] = module

# Heatmap grid resolution (in feet).
delta = 1.0  # 0.16

# Initialize hockey-specific space definitions for target coordinates.
spaces = sys.modules["spaces"].SpacesHockey([], 1, sys.modules["domain"], delta)

Y = spaces.targetsY
Z = spaces.targetsZ

# Pre-compute target grid in Cartesian coordinates (Y, Z).
targetsUtilityGridY, targetsUtilityGridZ = np.meshgrid(Y, Z)
targetsUtilityGridYZ = np.stack((targetsUtilityGridY, targetsUtilityGridZ), axis=-1)

shape = targetsUtilityGridYZ.shape
listedTargetsUtilityGridYZ = targetsUtilityGridYZ.reshape((shape[0] * shape[1], shape[2]))

# Net geometry reference points (feet).
leftPost = np.array([NET_GOAL_LINE_X, -NET_POST_Y])
rightPost = np.array([NET_GOAL_LINE_X, NET_POST_Y])

# Augmented goalposts and crossbar heights for larger angular span.
leftAugmented = np.array([NET_GOAL_LINE_X, -AUGMENTED_POST_Y])
rightAugmented = np.array([NET_GOAL_LINE_X, AUGMENTED_POST_Y])

top = NET_HEIGHT
topAugmented = AUGMENTED_HEIGHT_Z


def getAngle(point1, point2):
    """Return the polar angle from point1 to point2 (radians)."""

    x1, y1 = point1
    x2, y2 = point2

    angle = np.arctan2(y2 - y1, x2 - x1)

    return angle


def getAngularHeatmap(heatmap, playerLocation, executedAction):
    """Convert a Cartesian heatmap for one shot into angular space."""

    rng = np.random.default_rng(1000)

    # Flatten utilities for interpolation convenience.
    shape = heatmap.shape
    listedUtilities = heatmap.reshape((shape[0] * shape[1], 1))

    # Generate angular bounds for direction based on augmented posts.
    dir_left = getAngle(playerLocation, leftAugmented)
    dir_right = getAngle(playerLocation, rightAugmented)

    # Compute maximum elevation angle based on top of net.
    dist1 = np.linalg.norm(playerLocation - leftAugmented)
    dist2 = np.linalg.norm(playerLocation - rightAugmented)

    min_dist = min(dist1, dist2)
    elevation_top = np.arctan2(topAugmented, min_dist)

    # Create grid of direction/elevation samples (uniform resolution).
    resolution = ANGULAR_GRID_RESOLUTION
    dirs = np.linspace(dir_left, dir_right, resolution)
    elevations = np.linspace(0, elevation_top, resolution)

    # Cartesian target coordinates (unused, kept for reference/testing).
    gridTargets = []

    for j in Z:
        for i in Y:
            gridTargets.append([i, j])

    # Build angular grid list [direction, elevation].
    listedTargetsAngular = []

    for elevation_angle in elevations:
        for direction_angle in dirs:
            listedTargetsAngular.append([direction_angle, elevation_angle])

    listedTargetsAngular2YZ = []

    # For each target on the angular grid, convert to Cartesian (Y/Z).
    for target in listedTargetsAngular:

        direction_angle, elevation_angle = target

        # Step 1: rotate from player's X/Y to Y direction along net line.
        x_to_goal_line = 89 - playerLocation[0]
        delta_y = x_to_goal_line * np.tan(direction_angle)
        horizontal_distance_to_target = x_to_goal_line / np.cos(direction_angle)

        # Step 2: project elevation given distance to target point.
        z_coordinate_at_net = horizontal_distance_to_target * np.tan(elevation_angle)

        listedTargetsAngular2YZ.append([playerLocation[1] + delta_y, z_coordinate_at_net])

    # gridTargets = np.array(gridTargets)

    listedTargetsAngular = np.array(listedTargetsAngular)
    gridTargetsAngular = np.array(listedTargetsAngular).reshape((len(dirs), len(elevations), 2))

    listedTargetsAngular2YZ = np.array(listedTargetsAngular2YZ)
    gridTargetsAngular2YZ = np.array(listedTargetsAngular2YZ).reshape((len(dirs), len(elevations), 2))

    # Interpolate utilities from Y/Z grid into angular grid.
    listedUtilitiesComputed = griddata(
        listedTargetsUtilityGridYZ,
        listedUtilities,
        listedTargetsAngular2YZ,
        method='cubic',
        fill_value=0.0,
    )

    gridUtilitiesComputed = np.array(listedUtilitiesComputed).reshape((len(dirs), len(elevations)))

    ##################################################
    # Convert executed action to angular coordinates
    ##################################################

    negativeElevation = False
    originalElevation = None

    deltaX = NET_GOAL_LINE_X - playerLocation[0]
    deltaY = executedAction[0] - playerLocation[1]
    deltaZ = executedAction[1]

    # Direction Angle
    direction_angle = np.arctan2(deltaY, deltaX)

    # Elevation Angle
    horizontal_distance = np.sqrt(deltaX**2 + deltaY**2)
    elevation_angle = np.arctan2(deltaZ, horizontal_distance)

    # No negative angles possible. Can't miss low. Capping at 0.
    if elevation_angle < 0:
        originalElevation = elevation_angle
        elevation_angle = 0.0
        negativeElevation = True

    executedActionAngular = [direction_angle, elevation_angle]

    ##################################################

    # Assumming both same size (-1 to ofset for index-based 0)
    middle = int(len(dirs) / 2) - 1
    mean = [dirs[middle], elevations[middle]]

    ##################################################
    # Filtering
    ##################################################

    # Skill range for testing different execution skill hypotheses (in radians)
    min_skill = 0.004
    max_skill = np.pi / 4
    rhos = [0.0, -0.75, 0.75]  # Correlation coefficients

    allXS = [[min_skill, min_skill], [min_skill, max_skill], [max_skill, min_skill], [max_skill, max_skill]]

    skip = False

    tempPDFs = {}

    for eachX in allXS:
        for rho in rhos:

            # FOR TESTING
            # skip = False

            covMatrix = spaces.domain.getCovMatrix([eachX[0], eachX[1]], rho)

            x = spaces.get_key([eachX[0], eachX[1]], rho)

            pdfs = sys.modules["utils"].computePDF(
                x=executedActionAngular,
                means=listedTargetsAngular,
                covs=np.array([covMatrix] * len(listedTargetsAngular)),
            )
            prev = np.copy(pdfs)

            with np.errstate(invalid="ignore"):
                pdfs /= np.sum(pdfs)

            noise_distribution = sys.modules["domain"].draw_noise_sample(rng, mean, covMatrix)
            probability_density = noise_distribution.pdf(gridTargetsAngular)
            probability_density /= np.sum(probability_density)

            if np.isnan(pdfs).any():
                skip = True

                # UNCOMMENT AFTER TESTING
                break

    ##################################################

    return (
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
        (negativeElevation, originalElevation),
    )


if __name__ == '__main__':

    try:
        experimentFolder = sys.argv[1]
    except Exception as e:
        print("Need to specify the name of the folder for the experiment (located under 'Experiments/hockey-multi/') as command line argument.")
        exit()

    # Locate hockey experiment data for the provided experiment name.
    mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}{experimentFolder}{os.sep}Data{os.sep}"

    folder = f"{mainFolder}Heatmaps{os.sep}"
    files = os.listdir(folder)

    # Destination folders for generated outputs.
    saveAt = f"{mainFolder}AngularHeatmaps{os.sep}"
    saveAtFiltered = f"{mainFolder}AngularHeatmaps-Filtered{os.sep}"
    saveAtNegativeElevations = f"{mainFolder}AngularHeatmaps-NegativeElevations{os.sep}"

    # Ensure required directories exist.
    for each in [f"Data{os.sep}", mainFolder, saveAt, saveAtFiltered, saveAtNegativeElevations]:
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

        with open(folder + fileName, "rb") as infile:
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
        for i, row in enumerate(data):

            print("\ni: ", i)
            print("row: ", row)

            # FOR TESTING
            # row = 270544010806

            heatmap = data[row]["heat_map"]
            playerLocation = [data[row]["start_x"], data[row]["start_y"]]
            projectedZ = data[row]["projected_z"]
            # shot_location = final_y, projected_z, start_x, start_y
            executedAction = [data[row]["shot_location"][0], data[row]["shot_location"][1]]

            info = getAngularHeatmap(heatmap, playerLocation, executedAction)

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
                statsInfo[playerID][f"negativeElevation-{typeShot}"].append((info[10][1], row))

        with open(f"{saveAt}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}.pkl", "wb") as outfile:
            pickle.dump(playerData, outfile)

        if len(filtered) > 0:
            with open(f"{saveAtFiltered}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_filtered{len(filtered)}.pkl", "wb") as outfile:
                pickle.dump(filtered, outfile)

        infoNegativeElevations = statsInfo[playerID][f"negativeElevation-{typeShot}"]
        if len(infoNegativeElevations) > 0:
            with open(f"{saveAtNegativeElevations}angular_heatmap_data_player_{playerID}_type_shot_{typeShot}_negativeElevations{len(infoNegativeElevations)}.pkl", "wb") as outfile:
                pickle.dump(infoNegativeElevations, outfile)

        code.interact("...", local=dict(globals(), **locals()))

    with open(f"{mainFolder}statsAfterFiltering.txt", "w") as outfile:

        for eachID in statsInfo:
            print(f"Player: {eachID}", file=outfile)

            for st in typeShots:
                if st in statsInfo[eachID]:
                    print(f"\t{st}: {statsInfo[eachID][st]}", file=outfile)

            for st in typeShots:
                if st in statsInfo[eachID]:
                    if statsInfo[eachID][f"filtered-{st}"] > 0:
                        print(f"\tfiltered-{st}: {statsInfo[eachID][f'filtered-{st}']}", file=outfile)
                    if len(statsInfo[eachID][f"negativeElevation-{st}"]) > 0:
                        print(f"\negativeElevation-{st}: {len(statsInfo[eachID][f'negativeElevation-{st}'])}", file=outfile)

    # code.interact("...", local=dict(globals(), **locals()))
    print("Done.")
