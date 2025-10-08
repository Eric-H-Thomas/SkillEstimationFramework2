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
import os,argparse,code

import concurrent.futures
import matplotlib.pyplot as plt

# Note if you end up doing a batter and pitcher embedding, remove pitch hand
# from the features, but if you just have a batter embedding then keep pitch hand
xswingFeats = ['release_speed', 'mx',
 'mz', 'release_spin_rate',
 'plate_x',
 'plate_z', 'bat_handR', 'pit_handR',
 'balls',
 'strikes', 'batterIndex']


# Data set class 
class DataSet(Dataset):
	
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype = torch.float32)
		self.y = torch.tensor(y, dtype = torch.long)
		self.length = self.x.shape[0]
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
	
	def __len__(self):
		return self.length


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


def convertToTorch(train,test):

	train_x = train[xswingFeats].values.astype(float)
	train_y = train.outcome.values
	test_x = test[xswingFeats].values.astype(float)
	test_y = test.outcome.values

	trainset = DataSet(train_x, train_y)
	testset = DataSet(test_x, test_y)

	# Data loaders 
	trainloader = DataLoader(trainset, batch_size = 128, shuffle = False)
	testloader = DataLoader(testset, batch_size = 128, shuffle = False)

	return trainloader,testloader


def manageData(rawData):

	# Columns needed from statcast 
	needed_columns = ['game_date', 'player_name', 'pitcher', 'batter', 'pitch_type', 'pitch_name', 'stand', 'p_throws', 'balls', 'strikes', 'release_speed', 
					  'release_spin_rate', 'release_extension', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
					  'plate_x', 'plate_z',  'type', 'events', 'description', 'woba_value']

	# Drop unneeded columns 
	raw = rawData[needed_columns]

	# Clean the data and define the train and test sets 
	df = organizeData(raw)


	# Z-score continuous variables 
	standardizer = StandardScaler().fit(df[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)

	df[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']] = standardizer.transform(df[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)


	# Get the batter and pitcher index 
	batterIndices = pd.DataFrame({'batter': df.batter.unique()})
	pitcherIndices = pd.DataFrame({'pitcher': df.pitcher.unique()})

	batterIndices['batterIndex'] = batterIndices.index.values
	pitcherIndices['pitcherIndex'] = pitcherIndices.index.values

	# Merge  
	df = df.merge(batterIndices,on='batter')
	df = df.merge(pitcherIndices,on='pitcher')

	# code.interact("manageData()...", local=dict(globals(), **locals()))

	return df,batterIndices,standardizer


def manageDataForModel(rawData1,rawData2):

	# Columns needed from statcast 
	needed_columns = ['game_date', 'player_name', 'pitcher', 'batter', 'pitch_type', 'pitch_name', 'stand', 'p_throws', 'balls', 'strikes', 'release_speed', 
					  'release_spin_rate', 'release_extension', 'release_pos_x', 'release_pos_z', 'pfx_x', 'pfx_z',
					  'plate_x', 'plate_z',  'type', 'events', 'description', 'woba_value']

	# Drop unneeded columns 
	raw1 = rawData1[needed_columns]
	raw2 = rawData2[needed_columns]


	# Clean the data and define the train and test sets 
	df1 = organizeData(raw1)
	df2 = organizeData(raw2)

	train = df1.copy()
	test = df2.copy()
	allData = train.append(test, ignore_index = True)


	# Z-score continuous variables 
	standardizer = StandardScaler().fit(allData[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)

	allData[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']] = standardizer.transform(allData[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)

	train[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']] = standardizer.transform(train[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)

	test[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']] = standardizer.transform(test[['release_speed', 'mx', 'mz', 
		 'plate_x', 'plate_z', 'release_extension', 'release_spin_rate', 
		 'release_pos_x', 'release_pos_z']].values)


	# Get the batter and pitcher index 
	batterIndices = pd.DataFrame({'batter': train.batter.unique()})
	pitcher_indices = pd.DataFrame({'pitcher': train.pitcher.unique()})
	test = test.loc[test.batter.isin(train.batter.unique())]
	test = test.loc[test.pitcher.isin(train.pitcher.unique())]

	batterIndices['batterIndex'] = batterIndices.index.values
	pitcher_indices['pitcher_index'] = pitcher_indices.index.values

	# Merge 
	train = train.merge(batterIndices,on='batter')
	train = train.merge(pitcher_indices,on='pitcher')
	test = test.merge(batterIndices,on='batter')
	test = test.merge(pitcher_indices,on='pitcher')

	return train,test,allData,batterIndices


def getStatcastData(startDate,endDate):
	
	cache.enable()

	return statcast(start_dt = startDate, end_dt = endDate)


def getData(startYear,startMonth,startDay,endYear,endMonth,endDay,withinFramework=False):

	startDate = f"{startYear}-{startMonth}-{startDay}"
	endDate = f"{endYear}-{endMonth}-{endDay}"

	if withinFramework:
		dataFolder = f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	elif withinFramework == "processing":
		dataFolder = f"..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	else:
		dataFolder = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	
	fileName = f"Data-From-{startDate}-To-{endDate}.csv"	

	# If the folder doesn't exist already, create it
	if not Path(dataFolder).is_dir():
		os.mkdir(dataFolder)


	# If the data for the given date range is not already present, get it from statcast
	if not Path(f"{dataFolder}{fileName}").is_file():
		print(f"Getting data for the date range ({startDate} -> {endDate})")

		# Using thread to avoid error that occurs when performing
		# query (call to statcast()) due to multithreading issues
		# ERROR: "concurrent.futures.process.BrokenProcessPool: 
		# A process in the process pool was terminated abruptly 
		# while the future was running or pending."
		with concurrent.futures.ThreadPoolExecutor() as executor:
			future = executor.submit(getStatcastData,startDate,endDate)
			# print("Waiting for the thread...")
			rawData = future.result()

		rawData.to_csv(f"{dataFolder}{fileName}")
		print(f"Data was obtained successfully")
		
	# Otherwise, proceed load the data
	else:
		print(f"Loading CSV file with the data for the date range ({startDate} -> {endDate})")		
		rawData = pd.read_csv(f"{dataFolder}{fileName}",engine="python")
		print(f"Data was loaded successfully")


	return rawData


def main():

	# Get arguments from command line
	parser = argparse.ArgumentParser(description="Obtain statcast data for given date range")
	
	parser.add_argument("-startYear", dest = "startYear", help = "Desired start year.", type = str, default = "2021")
	parser.add_argument("-endYear", dest = "endYear", help = "Desired end year.", type = str, default = "2021")
	
	parser.add_argument("-startMonth", dest = "startMonth", help = "Desired start month.", type = str, default = "01")
	parser.add_argument("-endMonth", dest = "endMonth", help = "Desired end month.", type = str, default = "12")
	
	parser.add_argument("-startDay", dest = "startDay", help = "Desired start day.", type = str, default = "01")
	parser.add_argument("-endDay", dest = "endDay", help = "Desired end day.", type = str, default = "31")
	
	args = parser.parse_args()
	


	###########################################################################
	# Initial Setup
	###########################################################################

	saveAt = f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}StatcastData{os.sep}"
	
	folders = [f"..{os.sep}..{os.sep}Data{os.sep}",
			  f"..{os.sep}..{os.sep}Data{os.sep}Baseball{os.sep}",
			  saveAt,
			  f"{saveAt}imgs"]
	
	# If the folder doesn't exist already, create it
	for folder in folders:
		if not Path(folder).is_dir():
			os.mkdir(folder)

	###########################################################################


	rawData = getData(args.startYear,args.startMonth,args.startDay,args.endYear,args.endMonth,args.endDay)

	data,batterIndices,standardizer = manageData(rawData)


	pitchers = data.pitcher.unique()
	pitchTypes = data.pitch_type.unique()

	IdToName = {}
	

	###############################################################
	# For all the available pitchers on the data, 
	# Find how many observations are available for each
	# (grouped by with the different pitch types)
	###############################################################
	# '''
	startDate = f"{args.startYear}-{args.startMonth}-{args.startDay}"
	endDate = f"{args.endYear}-{args.endMonth}-{args.endDay}"
	fileName = f"PITCHERS-INFO-Data-From-{startDate}-To-{endDate}.txt"	

	# if not Path(f"{saveAt}{fileName}").is_file():

	print("Finding number of observations for all pitchers available on data")

	saveTo = open(f"{saveAt}{fileName}","w") 

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

	# code.interact("...", local=dict(globals(), **locals()))

	# '''
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


	saveTo = open(f"{saveAt}pitchersInfo.txt","w") 
	saveTo2 = open(f"{saveAt}pitchersIDs.txt","w") 
	saveTo3 = open(f"{saveAt}pitchersNotFound-Data-From-{startDate}-To-{endDate}.txt","w") 

	seenIDs = []

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

				for pitchType in pitchTypes:
					count = len(data.query(f"pitcher == {pid} and pitch_type == '{pitchType}'" ,engine="python"))
					saveTo.write(f"{pitchType}: {count} | ")

				saveTo.write("\n")

				# Saving just IDs to pitcherIDs file
				saveTo2.write(f"{pid}\n")

				seenIDs.append(pid)
			else:
				print(f"Not adding pitcher {name}-{pid} as it was already seen.")

		except:
			print(f"Pitcher {name} not found.")
			saveTo3.write(f"{name}\n")
			continue

	saveTo.close()
	saveTo2.close()
	saveTo3.close()

	##############################################################



	###############################################################
	# Find percentiles info from data
	###############################################################


	dataType = [["plate_x","plate_z"],["plate_x_inches","plate_z_inches"]]
	labels = ["feets-standardized","inches"]

	percentiles = [75,90,95,96,97,98,99]

	for i in range(2):

		# For all the available data, get executed actions
		# coming from loaded data = in feet & normalized
		actions = data[[dataType[i][0],dataType[i][1]]].values

		actions[:,0] = abs(actions[:,0])

		actions[:,1] = abs(actions[:,1] - 2.479)


		'''
		colors = []
		
		for a in actions:

			if a[0] < qx or a[1] < qz:
				colors.append("r")
			else:
				colors.append("b")

		# Make plot
		fig,ax = plt.subplots()
		plt.scatter(actions[:,0],actions[:,1],color=colors,label="Executed Actions")
		plt.legend()
		plt.show()
		plt.close()
		plt.clf()

		'''


		f, (ax1, ax2) = plt.subplots(2,1)
		ax1.scatter(actions[:,0],[0]*len(actions))
		ax1.set_xlabel("Plate X")
		
		ax2.scatter(actions[:,1],[0]*len(actions))
		ax2.set_xlabel("Plate Z")


		# Find percentiles (per column)
		qs = np.percentile(actions,q=percentiles,axis=0)

		for ip in range(len(percentiles)):
			ax1.scatter(qs[ip][0],0,label=f"{percentiles[ip]}->{round(qs[ip][0],4)}")
			ax2.scatter(qs[ip][1],0,label=f"{percentiles[ip]}->{round(qs[ip][1],4)}")

		plt.suptitle(f"From data - {labels[i]}")
		ax1.legend(loc="upper right")
		ax2.legend(loc="upper right")
		plt.tight_layout()
		plt.savefig(f"{saveAt}imgs{os.sep}percentiles-{labels[i]}")
		plt.close()
		plt.clf()


	###############################################################


	# code.interact("...", local=dict(globals(), **locals()))


if __name__ == '__main__':
	
	main()

