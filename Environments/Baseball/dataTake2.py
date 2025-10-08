###########################################################################
# Code by: Will Melville
# (Minor changes made for incorporation into the skill estimation framework)
###########################################################################

from pybaseball import statcast,cache,playerid_reverse_lookup, cache

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

from pathlib import Path
import os,argparse,code,json,argparse

import concurrent.futures
import matplotlib.pyplot as plt


# Function to clean and organize data from statcast 
def organizeData(df):

	df = df.loc[df.balls < 4]
	df = df.loc[df.strikes < 3]

	df.dropna(subset = ['release_extension', 
					  'release_speed','release_spin_rate', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z'], axis = 0,
			inplace = True)

	
	# Convert movement to inches instead of feet 
	df[['mx', 'mz']] = df[['pfx_x', 'pfx_z']].values * 12


	df[['plate_x_feet', 'plate_z_feet']] = df[['plate_x', 'plate_z']].values
	df[['plate_x_inches', 'plate_z_inches']] = df[['plate_x', 'plate_z']].values * 12

	# code.interact("...", local=dict(globals(), **locals()))
	
	
	# One hot encode handedness
	pit_hand = pd.get_dummies(df['p_throws'], drop_first = False)
	bat_hand = pd.get_dummies(df['stand'], drop_first = False)
	df['pit_handR'] = pit_hand['R']
	df['bat_handR'] = bat_hand['R']
	df = df.drop(['p_throws', 'stand', 'pfx_x', 'pfx_z'], axis = 1)
	
	# Remove bunts 
	df = df.loc[df.description.isin(['foul_bunt', 'bunt_foul_tip', 'missed_bunt']) == False]
	df = df.loc[df.events != 'sac_bunt']

	# Define the pitch outcome 
	df['outcome'] = -1
	df.loc[df.type == 'B', 'outcome'] = 0 # called ball 
	df.loc[df.description == 'called_strike', 'outcome'] = 1 # called strike 
	df.loc[df.description.isin(['swinging_strike', 'swinging_strike_blocked']), 'outcome'] = 2 # swm 
	df.loc[df.description.isin(['foul', 'foul_tip']), 'outcome'] = 3 # foul ball 

	# The other outcomes are all batted balls, 
	# which should either be outs or singles, doubles, triples, or home runs 
	df.loc[(df.type == 'X') & (df.events.isin(['field_out', 'force_out', 'field_error', 'grounded_into_double_play', 'sac_fly', 'fielders_choice', 
											   'fielders_choice_out', 'double_play', 'other_out', 'triple_play', 
											   'sac_fly_double_play'])), 'outcome'] = 4 # in play out 
	df.loc[(df.type == 'X') & (df.events == 'single'), 'outcome'] = 5 # single 
	df.loc[(df.type == 'X') & (df.events == 'double'), 'outcome'] = 6 # double 
	df.loc[(df.type == 'X') & (df.events == 'triple'), 'outcome'] = 7 # triple 
	df.loc[(df.type == 'X') & (df.events == 'home_run'), 'outcome'] = 8 # hr 

	# If outcome is still -1, drop it 
	df = df.loc[df.outcome != -1]

	# Define an is_swing column 
	df['is_swing'] = -1 
	df.loc[df.description.isin(['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 'foul_tip']), 'is_swing'] = 1
	df.loc[df.description.isin(['called_strike', 'ball', 'blocked_ball', 'hit_by_pitch', 'pitchout']), 'is_swing'] = 0

	# Define an is_miss column 
	df['is_miss'] = -1 
	df.loc[df.is_swing == 0 , 'is_miss'] = 0
	df.loc[df.description.isin(['swinging_strike', 'swinging_strike_blocked']), 'is_miss'] = 1 
	df.loc[df.description.isin(['hit_into_play', 'foul', 'foul_tip']), 'is_miss'] = 0
	
	return df


def manageData(csvFiles):

	#columns needed from statcast 
	needed_columns = ['game_date', 'game_year', 'game_pk', 'player_name', 'pitcher', 'batter', 'pitch_type', 'pitch_name', 'stand', 'p_throws', 'balls', 'strikes', 'release_speed', 
	                  'release_spin_rate', 'release_extension', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
	                  'plate_x', 'plate_z',  'type', 'events', 'description', 'woba_value', 'at_bat_number', 'pitch_number']


	# Read in the data
	raw22 = pd.read_csv(csvFiles[0])
	raw21 = pd.read_csv(csvFiles[1])
	raw19 = pd.read_csv(csvFiles[2])

	# Drop unneeded columns 
	raw21 = raw21[needed_columns]
	raw22 = raw22[needed_columns]
	raw19 = raw19[needed_columns]

	# Clean the data
	df21 = organizeData(raw21)
	df22 = organizeData(raw22)
	df19 = organizeData(raw19)


	all_data = df22.append(df21, ignore_index = True)
	all_data = all_data.append(df19, ignore_index = True)


	#min max scale variables 
	standardizer = StandardScaler().fit(all_data[['release_speed', 'mx', 'mz', 
	     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
	     'release_pos_x', 'release_pos_z']].values)

	all_data[['release_speed', 'mx', 'mz', 
	     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
	     'release_pos_x', 'release_pos_z']] = standardizer.transform(all_data[['release_speed', 'mx', 'mz', 
	     'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
	     'release_pos_x', 'release_pos_z']].values)


	# Get the batter index
	batter_indices = pd.DataFrame({'batter': all_data.batter.unique()})
	batter_indices['batter_index'] = batter_indices.index.values

	# Merge 
	all_data = all_data.merge(batter_indices, on='batter')

	# code.interact("manageData()...", local=dict(globals(), **locals()))

	return all_data,batter_indices,standardizer


def main():


	# Get arguments from command line
	parser = argparse.ArgumentParser(description="Obtain statcast data for given date range")
	
	parser.add_argument("-splitInto", dest = "splitInto", help = "Number of files to split exp info (ids/types) into.", type = int, default = 1)
	
	args = parser.parse_args()


	###############################################################
	# GET DATA
	###############################################################

	folder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	csvFiles = [f"{folder}raw22.csv",f"{folder}raw21.csv",f"{folder}raw18_19_20.csv"]

	print("Loading & managing data...")

	data,batterIndices,standardizer = manageData(csvFiles)

	###############################################################


	###############################################################
	# For all the available pitchers on the data, 
	# Find how many observations are available for each
	# (grouped by with the different pitch types)
	###############################################################

	pitchers = data.pitcher.unique()
	pitchTypes = data.pitch_type.unique()


	IdToName = {}	


	fileName = f"PITCHERS-INFO-Data-From-GivenFiles.txt"	
	saveTo = open(f"{folder}{fileName}","w") 


	'''
	print("Finding number of observations for all pitchers available on data")

	notFound = []

	for pitcherID in pitchers:
		try:
			result = playerid_reverse_lookup([int(pitcherID)])[["name_first","name_last"]]
			name = f"{result.name_first[0].capitalize()} {result.name_last[0].capitalize()}"
		except:
			# Load name from actual data instead
			# In case results is [] (meaning it couldn't load info for ID)
			# Example: 642601 = 'De La Cruz, Oscar'
			name = data.query(f"pitcher == {pitcherID}",engine="python").player_name.values[0]
			notFound.append(name)

			splittedN = name.split(", ")
			name = f"{splittedN[1].capitalize()} {splittedN[0].capitalize()}"


		IdToName[pitcherID] = name

		saveTo.write(f"Pitcher: {name} | ID: {pitcherID} | ")

		for pitchType in pitchTypes:
			count = len(data.query(f"pitcher == {pitcherID} and pitch_type == '{pitchType}'",engine="python"))
			saveTo.write(f"{pitchType}: {count} | ")
		
		saveTo.write("\n")

	saveTo.close()
	'''
	###############################################################



	###############################################################
	# For the given pitchers, find their respective IDs and
	# how many observations are available for each
	# (grouped by with the different pitch types)
	###############################################################
	
	# Input format: Name MiddleName(optional) LastName
	# Format needed for statcasts: LastName, FirstName MiddleName(if any)
	

	if not Path("pitchers.txt").is_file():
		print("Text file named 'pitchers.txt' is not present.\nPlease create such file and list there the names of the pitchers to search for.")
		exit()


	print("\nFinding IDs & number of observations for given pitchers")

	with open("pitchers.txt","r") as inFile:
		allNames = inFile.readlines()


	saveTo = open(f"{folder}pitchersInfo-GivenFiles.txt","w") 
	saveTo2 = open(f"{folder}pitchersIDs-GivenFiles.txt","w") 
	saveTo3 = open(f"{folder}pitchersNotFound-Data-From-GivenFiles.txt","w") 

	seenIDs = []

	expInfo = []


	##############################
	# FOR TESTING
	##############################
	# allNames = allNames[0:2]


	for name in allNames:

		# Format name accordingly
		name = name.strip().split()

		# Case: Name MiddleName LastName
		if len(name) == 3:
			name = f"{name[2]}, {name[0]} {name[1]}"

		# Case: Name LastName
		elif len(name) == 2:
			name = f"{name[1]}, {name[0]}"

		# Case: Blank line
		else:
			continue


		# Will get & save info only if pitcher is present on
		# the data for the given date range. Will skip otherwise.
		try:
			pid = data.query(f"player_name == '{name}'",engine="python").pitcher.unique()[0]
			
			if pid not in seenIDs:
				print(f"Added pitcher {name} - ID: {pid}")
				saveTo.write(f"{name}: {pid} | \n\t")

				temp = []

				for pitchType in pitchTypes:
					count = len(data.query(f"pitcher == {pid} and pitch_type == '{pitchType}'" ,engine="python"))
					temp.append([pitchType,count])
					saveTo.write(f"{pitchType}: {count} | ")

				saveTo.write("\n")

				# Saving just IDs to pitcherIDs file
				saveTo2.write(f"{pid}\n")

				seenIDs.append(pid)

				expInfo.append([pid,temp])

			else:
				print(f"Not adding pitcher {name}-{pid} as it was already seen.")

		except Exception as e:
			print(f"Pitcher {name} not found.")
			# print(e)
			saveTo3.write(f"{name}\n")
			continue

	saveTo.close()
	saveTo2.close()
	saveTo3.close()

	##############################################################
	



	##############################################################
	# 
	##############################################################
	
	expanded = []

	for row in expInfo:
		pid, temp = row[0], row[1]

		for each in temp:
			# To filter combinations with 0 info
			if int(each[1]) != 0:
				# id, pitchType, count
				expanded.append([pid,each[0],int(each[1])])

	expanded = np.array(expanded)
	

	'''
	stayPitchers = ["547973","594798","623433","445276","605483"]
	stayPitchers = []
	allIs = np.array(range(len(expanded)))

	keep = []

	for pidStay in stayPitchers:
		iis = np.where(expanded[:,0] == pidStay)[0]
		keep = np.concatenate([keep,iis])


	first = expanded[keep.astype("int")]

	rest = np.setdiff1d(allIs,keep)

	second = expanded[rest]

	# Sort from max to min the rest of the info
	second = second[second[:,2].astype("int").argsort()[::-1]]
	
	# Merge info
	combined = np.concatenate([first,second])
	'''

	combined = expanded


	# Create json file - ALL
	with open(f"{folder}AllInfoForExps-Total-{len(combined)}.json","w") as outfile:
		json.dump(combined.tolist(),outfile)


	# Keep just pitcher & pitch type
	combined = combined[:,:-1]


	# Split info across files
	numFiles = args.splitInto

	
	splittedInfo = []
	for x in range(numFiles):
		splittedInfo.append([])

	saveAt = 0


	# Assign round-robin like
	for each in range(len(combined)):
		splittedInfo[saveAt].append(combined[each])
		saveAt += 1

		# Reset
		if saveAt == numFiles:
			saveAt = 0


	# Create files with splitted info
	for f in range(numFiles):
		with open(f"{folder}ExpsInfo-{f+1}-Total-{len(splittedInfo[f])}.json","w") as outfile:
			toSave = {"missingExps":np.array(splittedInfo[f]).tolist()}
			json.dump(toSave,outfile)

	# code.interact("...", local=dict(globals(), **locals()))

	##############################################################


if __name__ == '__main__':
	
	main()

