import os,sys,code
sys.path.insert(0, '../un-xPass')

from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import itertools

# Disable private API warnings
import warnings
from statsbombpy.api_client import NoAuthWarning
warnings.filterwarnings(action="ignore", category=NoAuthWarning, module='statsbombpy')

from unxpass.databases import Database, connect
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset

from unxpass import features as fs
from unxpass import labels as ls
from unxpass.utils import play_left_to_right
from unxpass.visualization import plot_action

from socceraction.spadl.utils import add_names
from socceraction.vaep.features import gamestates as to_gamestates

import pickle


def getPlayersInfo(path):

	# Where the processed data should be stored
	DATA_DIR = Path("../un-xPass/stores")
	
	# Configure database
	DB_PATH = Path("../un-xPass/stores/database.sqlite")
	db = SQLiteDatabase(DB_PATH)

	allPlayers = pd.read_sql_query("SELECT * FROM players", db.conn)

	allFiles = os.listdir(path)

	ids = []
	numObs = []

	for eachFile in allFiles:
		ids.append(int(eachFile.split(".csv")[0].split("Player")[1]))

		with open(path+eachFile) as infile:
			numObs.append(len(pd.read_csv(infile)))


	names = []

	for eachID in ids:
		names.append(allPlayers[allPlayers["player_id"] == eachID].player_name.values[0])


	path = path.split("PerPlayer")[0]



	labels = ["All","AboveThres"]
	thres = 50

	for label in labels:

		idToName = {}
		nameToID = {}


		# Readable text file
		with open(path+f"{label}Players.txt","w") as outfile:
			print(f"ID - PLAYER NAME - # EVENTS",file=outfile)

			count = 0

			for each in range(len(ids)):

				pid = ids[each]
				name = names[each]
				num = numObs[each]

				if label == "All" or num >= thres:
					print(f"{pid} - {name} - {num}",file=outfile)

					if pid not in idToName:
						idToName[pid] = {"name":name, "numObs":num}

					if name not in nameToID:
						nameToID[name] = {"id":pid, "numObs":num}

					count += 1

			print(f"\nTOTAL # OF PLAYERS: {count}",file=outfile)


		info = {}
		info["idToName"] = idToName
		info["nameToID"] = nameToID

		# For exps
		with open(path+f"info{label}Players.pickle", "wb") as outfile:
			pickle.dump(info,outfile)


	code.interact("...", local=dict(globals(), **locals()))

	db.close()


def interactDB():

	# Configure leagues and seasons to download and convert

	datasets = [
	# Full EURO 2020 dataset
	# { "getter": "remote", "competition_id":  55, "season_id": 43 }
	# For EURO 2022 - Women's
	# { "getter": "remote", "competition_id":  53, "season_id": 106 }
	# { "getter": "local", "competition_id":  53, "season_id": 106 }

	# For Womens World Cup 2023
	# { "getter": "remote", "competition_id":  72, "season_id": 107 }

	# For Men's FIFA World Cup 2022
	{ "getter": "remote", "competition_id":  43, "season_id": 106 }

	# BEL v ITA at EURO2020 (enable for a quick test run)
	# { "getter": "remote", "competition_id":  55, "season_id": 43, "game_id": 3795107 }
	# You can also import a local dataset
	#{ "getter": "local", "root": "../raw_data", "competition_id":  55, "season_id": 43 }
	]


	# Where the processed data should be stored
	DATA_DIR = Path("../un-xPass/stores")

	
	# Configure database
	DB_PATH = Path("../un-xPass/stores/database.sqlite")
	db = SQLiteDatabase(DB_PATH)


	# Import data
	# for dataset in datasets:
		# db.import_data(**dataset,root=Path("/Users/delmairis_22/Desktop/Research/skill-estimation-framework/Data/Soccer/Statsbomb/Data/"))



	'''	
	# Access data
	# List of games included in the database
	df_games = db.games()
	print(df_games.head())


	# Dataframe with all SPADL actions + 360 snapshots for a particular game
	# df_actions = db.actions(game_id=3795107)
	# print(df_actions.head())


	# Sample plot 
	# sample = (3795107,2)
	# plot_action(df_actions.loc[sample])
	# plt.show()



	game_id = 3795107

	# load SPADL actions
	df_actions = add_names(db.actions(game_id))
	df_actions.head()

	# Select passes
	# Only use passes that are
	# - performed by foot
	# - part of open play
	# - for which the start and end location are included in the 360 snapshot
	passes_idx = PassesDataset.actionfilter(df_actions)
	df_actions.loc[passes_idx].head()



	# List of available features
	print("Features:", [f.__name__ for f in fs.all_features])

	# List of available labels
	print("Labels:", [f.__name__ for f in ls.all_labels])


	# convert actions to gamestates
	home_team_id, _ = db.get_home_away_team_id(game_id)
	gamestates = play_left_to_right(to_gamestates(df_actions, nb_prev_actions=3), home_team_id)

	# compute features and labels
	pd.concat([
		fs.actiontype(gamestates),
		ls.success(df_actions)
	], axis=1).loc[passes_idx]


	# or, as a shorthand to the above
	pd.concat([
		fs.get_features(db, game_id, xfns=[fs.actiontype], actionfilter=PassesDataset.actionfilter),
		ls.get_labels(db, game_id, yfns=[ls.success], actionfilter=PassesDataset.actionfilter)
	], axis=1)
	'''


	'''
	# The "PassesDataset" interface

	dataset = PassesDataset(
	path=DATA_DIR / "datasets" / "euro2020",
	xfns=["actiontype"],
	yfns=["success"]
	)
	
	dataset.create(db)

	print(dataset.features)
	print(dataset.labels)

	print(dataset[0])
	'''

	df_competitions = pd.read_sql_query("SELECT * FROM competitions", db.conn)

	df_actions = pd.read_sql_query("SELECT * FROM actions", db.conn)




	# dataset_train = partial(PassesDataset, path=DATA_DIR / "datasets" / "euro2020" / "train")
	# dataset_train = partial(PassesDataset, path=DATA_DIR / "datasets" / "euro2022" / "train")
	# dataset_train = partial(PassesDataset, path=DATA_DIR / "datasets" / "worldcup2023" / "train")
	dataset_train = partial(PassesDataset, path=DATA_DIR / "datasets" / "FifaWorldCup2022" / "train")

	# dataset_test = partial(PassesDataset, path=DATA_DIR / "datasets" / "euro2020" / "test")
	# dataset_test = partial(PassesDataset, path=DATA_DIR / "datasets" / "euro2022" / "test")
	# dataset_test = partial(PassesDataset, path=DATA_DIR / "datasets" / "worldcup2023" / "test")



	infoTrain = dataset_train(xfns=["actiontype"],yfns=["success"])
	# infoTest = dataset_test(xfns=["actiontype"],yfns=["success"])


	code.interact("...", local=dict(globals(), **locals()))

	
	db.close()



def getDataPerPlayer(path,infoTypes,cid,sid):

	# WILL GET ACTIONS & FEATURES/LABELS


	# Features to use for value model (vaep_xg_360)
	subsetFeatures = [fs.actiontype_onehot,fs.result_onehot,fs.actiontype_result_onehot,
					fs.bodypart_onehot,fs.time,fs.startlocation,fs.endlocation,fs.startpolar,
					fs.endpolar,fs.movement,fs.team,fs.time_delta,fs.space_delta,fs.goalscore,
					fs.packing_rate,fs.defenders_in_3m_radius,fs.defenders_in_5m_radius]

	# Labels to use for value model (vaep_xg_360)
	subsetLabels = [ls.scores,ls.scores_xg,ls.concedes,ls.concedes_xg]



	database = f"sqlite://{os.path.expanduser('~')}/Desktop/Research/un-xPass/stores/database.sql"

	db = connect(database, mode="r")


	# Men's FIFA World Cup 2022
	filters = [{"competition_id": cid, "season_id": sid}]

	nb_prev_actions = 3

	games = list(
			itertools.chain.from_iterable(
				[x["game_id"]] if "game_id" in x else db.games(**x).index.tolist() for x in filters
			)
			if filters is not None
			else db.games().index
		)

	# FOR TESTING
	# games = [games[0],games[1]]

	allActions = {}
	allGameStates = {}
	allFeatures = {}
	allLabels = {}


	# For each available game
	for gid in games:

		# Select all available actions 
		actions = add_names(db.actions(gid))

		# Apply filters
		idx = PassesDataset.actionfilter(actions)


		# Convert actions to gamestates
		home_team_id, _ = db.get_home_away_team_id(gid)
		# Can't filter actions yet bc to_gamestates() assumes actions in order for a given game
		gamestates = play_left_to_right(to_gamestates(actions,nb_prev_actions),home_team_id)
		
		# Only select passes that meet criteria (remove filtered ones)
		allActions[gid] = pd.DataFrame(actions).loc[idx]

		# Manually filter gamestate (can't do .loc[idx] bc of shape mismatch)
		# gamestate = list of dataframes with length = nb_prev_actions
		filteredGameStates = []

		for each in gamestates:
			filteredGameStates.append(each.loc[idx])


		allGameStates[gid] = filteredGameStates

		allFeatures[gid] = fs.get_features(db, gid, xfns=subsetFeatures, actionfilter=PassesDataset.actionfilter)
		allLabels[gid] = ls.get_labels(db, gid, yfns=subsetLabels, actionfilter=PassesDataset.actionfilter)
		
		# code.interact("...", local=dict(globals(), **locals()))



	players = {}

	# Filter data by player
	for gid in games:

		tempActions = allActions[gid]
		tempGameStates = allGameStates[gid]
		tempFeatures = allFeatures[gid]
		tempLabels = allLabels[gid]

		for index, event in tempActions.iterrows():

			pid = event.player_id

			if not pd.isna(pid):

				if pid not in players:
					players[pid] = {"actions":pd.DataFrame(),"gamestates":[],"features":pd.DataFrame(),"labels":pd.DataFrame()}

				event["game_id"] = gid
				
				players[pid]["actions"] = pd.concat([players[pid]["actions"],event.to_frame().T],ignore_index=True)
				players[pid]["gamestates"].append(tempGameStates)
				players[pid]["features"] = pd.concat([players[pid]["features"],tempFeatures.loc[index].to_frame().T])
				players[pid]["labels"] = pd.concat([players[pid]["labels"],tempLabels.loc[index].to_frame().T])

	
	total = 0

	# Save
	for each in players:
		players[each]["actions"].to_pickle(f"{path}PerPlayer{infoTypes[0]}{os.sep}data-Player{each}.pkl")
		players[each]["features"].to_pickle(f"{path}PerPlayer{infoTypes[2]}{os.sep}data-Player{each}.pkl")
		players[each]["labels"].to_pickle(f"{path}PerPlayer{infoTypes[3]}{os.sep}data-Player{each}.pkl")
	
		with open(f"{path}PerPlayer{infoTypes[1]}{os.sep}data-Player{each}.pkl","wb") as outfile:
			pickle.dump(players[each]["gamestates"],outfile)


		total += len(players[each]["actions"])	
	
	print(f"Total number of passes across all players: {total}")

	code.interact("...", local=dict(globals(), **locals()))

	db.close()



def getDataPerPlayer_GameStates(path,infoTypes,cid,sid):

	database = f"sqlite://{os.path.expanduser('~')}/Desktop/Research/un-xPass/stores/database.sql"

	db = connect(database, mode="r")


	# Men's FIFA World Cup 2022
	filters = [{"competition_id": cid, "season_id": sid}]

	nb_prev_actions = 1

	games = list(
			itertools.chain.from_iterable(
				[x["game_id"]] if "game_id" in x else db.games(**x).index.tolist() for x in filters
			)
			if filters is not None
			else db.games().index
		)

	# FOR TESTING
	# games = [games[0],games[1]]

	allGameStates = {}


	# For each available game
	for gid in games:

		# Select all available actions 
		actions = add_names(db.actions(gid))

		# Apply filters
		idx = PassesDataset.actionfilter(actions)

		# Convert actions to gamestates
		home_team_id, _ = db.get_home_away_team_id(gid)
		gamestates = play_left_to_right(to_gamestates(actions,nb_prev_actions),home_team_id)
		
		# Only select passes that meet criteria (remove filtered ones)
		gamestates = pd.DataFrame(gamestates[0]).loc[idx]

		allGameStates[gid] = gamestates


	players = {}

	# Filter data by player
	for gid in games:


		gamestates = allGameStates[gid]

		for index, event in gamestates.iterrows():

			pid = event.player_id

			if not pd.isna(pid):

				if pid not in players:
					players[pid] = {"events":pd.DataFrame()}

				players[pid]["events"] = pd.concat([players[pid]["events"],event.to_frame().T],ignore_index=True)


	total = 0

	# Save
	for each in players:
		players[each]["events"].to_csv(f"{path}PerPlayer{infoTypes[0]}{os.sep}data-Player{each}.csv",mode="w")
	
		total += len(players[each]["events"])	
	
	print(f"Total number of passes across all players: {total}")


	code.interact("...", local=dict(globals(), **locals()))

	db.close()



if __name__ == '__main__':


	# Men's FIFA World Cup 2022
	cid = 43
	sid = 106


	########################################

	# interactDB()

	########################################
	


	########################################
	
	# '''

	path = f"Data{os.sep}Soccer{os.sep}"
	# infoTypes = ["-GameStates"]
	infoTypes = ["-Actions","-GameStates","-Features","-Labels"]


	folders = ["Data",path,path+"Unxpass"]

	for each in infoTypes:
		folders.append(f"{path}Unxpass{os.sep}PerPlayer{each}")


	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)

	path += f"Unxpass{os.sep}"

	# getDataPerPlayer_GameStates(path,infoTypes,cid,sid)

	getDataPerPlayer(path,infoTypes,cid,sid)

	# '''

	########################################



	########################################

	# ASSUMES FOLDER WITH "PERPLAYER" DATA ALREADY EXISTS
	# path = f"Data{os.sep}Soccer{os.sep}Unxpass{os.sep}PerPlayer{os.sep}"

	# getPlayersInfo(path)
	
	########################################
