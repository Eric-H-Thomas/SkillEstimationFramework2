import os,sys
from pathlib import Path
from importlib.machinery import SourceFileLoader

# Find location of current file
scriptPath = os.path.realpath(__file__)
mainFolderName = scriptPath.split("baseball.py")[0]

for each in []:
	module = SourceFileLoader(each,f"{mainFolderName}{each}.py").load_module()
	sys.modules[each] = module


import pandas as pd
import numpy as np
import json
import argparse,code

import socceraction.vaep.features as fs
from rich.progress import track
from socceraction.spadl import config as spadl
from socceraction.spadl.utils import add_names
from socceraction.vaep.features import gamestates as to_gamestates
from unxpass.features import all_features, _spadl_cfg
from unxpass.labels import all_labels
from unxpass.databases import Database
import xgboost as xgb


from typing import Callable, List, Dict, Optional, Tuple
from functools import reduce
from rich.progress import track

from scipy.stats import multivariate_normal
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable



# VARIATION FROM simulate_features()
# FROM unxpass' features.py
def simulate_features(
	db: Database,
	gamestate: List,
	game_id: int,
	xfns: List[Callable] = all_features,
	actionfilter: Optional[Callable] = None,
	nb_prev_actions: int = 3,
	xy: Optional[List[pd.DataFrame]] = None,
	x_bins: int = 104,
	y_bins: int = 68,
	result: Optional[str] = None,
):
	"""Apply a list of feature generators.

	Parameters
	----------
	db : Database
		The database with raw data.
	game_id : int
		The ID of the game for which features should be computed.
	xfns : List[Callable], optional
		The feature generators.
	actionfilter : Callable, optional
		A function that filters the actions to be used.
	nb_prev_actions : int, optional
		The number of previous actions to be included in a game state.
	xy: list(pd.DataFrame), optional
		The x and y coordinates of simulated end location.
	x_bins : int, optional
		The number of bins to simulated for the end location along the x-axis.
	y_bins : int, optional
		The number of bins to simulated for the end location along the y-axis.
	result : Optional[str], optional
		Sets the action result to be used for the simulation. If None, the
		actual result of the actions is used.

	Returns
	-------
	pd.DataFrame
		A dataframe with the features.
	"""
	# retrieve actions from database
	# actions = add_names(action)

	# # filter actions of interest
	# if actionfilter is None:
	#     idx = pd.Series([True] * len(actions), index=actions.index)
	# else:
	#     idx = actionfilter(actions)
	# # check if we have to return an empty dataframe
	# if idx.sum() < 1:
	#     column_names = feature_column_names(xfns, nb_prev_actions)
	#     return pd.DataFrame(columns=column_names)
	# if len(xfns) < 1:
	#     return pd.DataFrame(index=actions.index.values[idx])
	# # convert actions to gamestates
	# home_team_id, _ = db.get_home_away_team_id(game_id)

	# import code
	# code.interact("...", local=dict(globals(), **locals()))

	# gamestates = play_left_to_right(to_gamestates(actions, nb_prev_actions), home_team_id)

	gamestates = gamestate

	# simulate end location
	if xy is None:
		# - create bin centers
		yy, xx = np.ogrid[0.5:y_bins, 0.5:x_bins]
		# - map to spadl coordinates
		x_coo = np.clip(xx / x_bins * _spadl_cfg["length"], 0, _spadl_cfg["length"])
		y_coo = np.clip(yy / y_bins * _spadl_cfg["width"], 0, _spadl_cfg["width"])
	
	# simulate action result
	if result is not None:
		if result not in spadl.results:
			raise ValueError(f"Invalid result: {result}. Valid results are: {spadl.results}")
		gamestates[0].loc[:, ["result_id", "result_name"]] = (spadl.results.index(result), result)
	
	# compute fixed features
	xfns_fixed = [
		fn
		for fn in xfns
		if "end_x" not in fn.required_fields and "end_y" not in fn.required_fields
	]


	df_fixed_features = reduce(
		lambda left, right: pd.merge(left, right, how="outer", left_index=True, right_index=True),
		(fn(gamestates) for fn in xfns_fixed),
	)

	# simulate other features
	xfns_to_simulates = [
		fn for fn in xfns if "end_x" in fn.required_fields or "end_y" in fn.required_fields
	]

	df_simulated_features = []
	
	if xy is None:
		for end_x, end_y in track(
			np.array(np.meshgrid(x_coo, y_coo)).T.reshape(-1, 2),
			description=f"Simulating features for game {game_id}",
		):
			# code.interact("()...", local=dict(globals(), **locals()))

			gamestates[0].loc[:, ["end_x", "end_y"]] = (end_x, end_y)
			df_simulated_features.append(
				reduce(
					lambda left, right: pd.merge(
						left, right, how="outer", left_index=True, right_index=True
					),
					(fn(gamestates) for fn in xfns_to_simulates),
				)
			)
	else:
		for end in xy:
			gamestates[0].loc[end.index, ["end_x", "end_y"]] = end.values
			df_simulated_features.append(
				reduce(
					lambda left, right: pd.merge(
						left, right, how="outer", left_index=True, right_index=True
					),
					(fn(gamestates) for fn in xfns_to_simulates),
				)
			)

	df_features = pd.concat(df_simulated_features, axis=0).join(df_fixed_features, how="left")
	
	# if we generated features for pass options instead of pass actions,
	# we need to add the pass option's ID to the index
	if "pass_option_id" in df_features.columns:
		df_features["pass_option_id"] = df_features["pass_option_id"].fillna(0).astype(int)
		df_features = df_features.set_index("pass_option_id", append=True)
	
	return df_features

# VARIATION FROM predict_surface()
# FROM unxpass' pass_value.py
def predict_surface(model, dataset, gamestate, game_id, db=None, x_bins=104, y_bins=68, result=None) -> Dict:
	data = model.offensive_model.initialize_dataset(dataset)
	games = data.features.index.unique(level=0)
	assert game_id in games, "Game ID not found in dataset!"
	sim_features = simulate_features(
		db,
		gamestate,
		game_id,
		xfns=list(data.xfns.keys()),
		actionfilter=data.actionfilter,
		x_bins=x_bins,
		y_bins=y_bins,
		result=result,
	)

	out = {}
	cols = [item for sublist in data.xfns.values() for item in sublist]
	for action_id in sim_features.index.unique(level=1):
		if isinstance(model.offensive_model.model, xgb.XGBClassifier):
			out[f"action_{action_id}"] = (
				model.offensive_model.model.predict_proba(
					sim_features.loc[(game_id, action_id), cols]
				)[:, 1]
				.reshape(x_bins, y_bins)
				.T
			) - (
				model.defensive_model.model.predict_proba(
					sim_features.loc[(game_id, action_id), cols]
				)[:, 1]
				.reshape(x_bins, y_bins)
				.T
			)
		elif isinstance(model.offensive_model.model, xgb.XGBRegressor):
			# code.interact("()...", local=dict(globals(), **locals()))
			
			out[f"action_{action_id}"] = (
				model.offensive_model.model.predict(
					sim_features.loc[(game_id, action_id), cols]
				)
				.reshape(x_bins, y_bins)
				.T
			) - (
				model.defensive_model.model.predict(
					sim_features.loc[(game_id, action_id), cols]
				)
				.reshape(x_bins, y_bins)
				.T
			)
		else:
			raise AttributeError(
				f"Unsupported xgboost model: {type(self.offensive_model.model)}"
			)
	return out

def getDomainName():
	return "soccer"


def plotPitch(info):

	fig,ax = plt.subplots()
	
	return ax,cmap,norm


def getNoiseModel(mean=[0.0,0.0],X=0.0):
	# X is squared already (x**2 = variance)
	N = multivariate_normal(mean=mean,cov=X)
	return N

def sample_action(mean,L,a,noiseModel=None):

	# If noise model was not given, proceed to get it
	if noiseModel == None:
		N = getNoiseModel(mean,L**2)
	# Otherwise, use given noise model
	else:
		N = noiseModel

	#Get noise (sample)
	noise = N.rvs()

	# Add noise to planned action (This creates the noisy action)
	na = [a[0] + noise[0], a[1] + noise[1]]

	return na

def getSymmetricNormalDistribution(XS,resolution,X,Y):

	# XS it's the standard deviation (not squared yet)
	D = np.zeros((len(X),len(Y)))

	# TODO : ??????
	mean = [0.0,0.0]


	# XS**2 to get variance
	N = getNoiseModel(mean,XS**2)
	
	for i in range(len(X)):
		for j in range(len(Y)):
			D[i,j] = N.pdf([X[i],Y[j]])

	
	# Scale up probs by resolution^2 to avoid having very small probs (not adding up to 1)
	# This is because depending on the xskill/resolution combination, the pdf of
	# a given xskill may not show up in any of the resolution buckets 
	# causing then the pdfs not adding up to 1
	# (example: xskill of 1.0 & resolution > 1.0)
	# If the resolution is less than the xskill, the xskill distribution can be fully captured 
	# by the resolution thus avoiding problems.  
	D *= np.square(resolution)

	return D


def testHits():

	numTries = 10000.0

	minXskill = 170.0 #0.50
	maxXskill = 200.0
	xSkills = np.linspace(minXskill,maxXskill,num=100)


	# code.interact("...", local=dict(globals(), **locals()))


	# Select target in the middle of the board
	x = (minPlateX+maxPlateX)/2.0
	z = (minPlateZ+maxPlateZ)/2.0

	# Assume aiming at the middle
	action = [x,z]

	saveFolder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}PercentHits{os.sep}"

	folders = [f"..{os.sep}..{os.sep}Data{os.sep}",
			   f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}",
			   saveFolder]

	for folder in folders:
		if not Path(folder).is_dir():
			os.mkdir(folder)

	# Prep file for saving results
	outFile = open(f"{saveFolder}PercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.txt", "w")


	print(f"\n--- Performing testHit experiment... ---")
	allPercentHits = []


	for xs in xSkills:

		xs = round(xs,4)
		print(f"\txskill: {xs}")

		N = getNoiseModel(xs**2)

		hits = 0.0

		for tries in range(int(numTries)):

			# Get noise sample
			noise = N.rvs()

			# Add noise to action
			noisyAction = [action[0]+noise[0],action[1]+noise[1]]

			#print(f"\t\t action: {action}")
			#print(f"\t\t noisyAction: {noisyAction}")

			# Verify if the action hits the board or not
			if (noisyAction[0] >= minPlateX and noisyAction[0] <= maxPlateX) and\
				(noisyAction[1] >= minPlateZ and noisyAction[1] <= maxPlateZ):
				hits += 1.0


			####################################
			# PLOT - Strike Zone Board
			####################################
			'''
			fig,ax = plt.subplots()

			# Plot boundaries
			ax.scatter(minPlateX,minPlateZ,c = "black")
			ax.scatter(maxPlateX,maxPlateZ,c = "black")
			ax.scatter(minPlateX,maxPlateZ,c = "black")
			ax.scatter(maxPlateX,minPlateZ,c = "black")
			
			# Plot actual executed action & EV
			ax.scatter(action[0],action[1],c = "red", marker = "*")
			ax.scatter(noisyAction[0],noisyAction[1],c = "blue", marker = "*")

			ax.set_title(f"xskill: {xs}")
			plt.show()
			plt.clf()
			plt.close()
			code.interact("...", local=dict(globals(), **locals()))
			'''
			####################################
			

		percentHit = (hits/numTries)*100.0
		allPercentHits.append(percentHit)
		
		print(f"\t\txSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%")
		# Save to file
		print(f"xSkill: {xs} | \tTotal Hits: {hits} out of {numTries} -> {percentHit}%",file=outFile)


	outFile.close()

	plt.plot(xSkills,allPercentHits)
	plt.xlabel('xSkills')
	plt.ylabel('% Hits')
	plt.savefig(f"{saveFolder}xskillsVsPercentHits-minXskill{minXskill}-maxXskill{maxXskill}-numTries{numTries}.png")
	plt.clf()
	plt.close()


if __name__ == '__main__':
	
	testHits()
