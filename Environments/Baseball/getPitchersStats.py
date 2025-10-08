import argparse
import os
from pathlib import Path
import concurrent.futures
import pandas as pd

from pybaseball import statcast,cache,playerid_reverse_lookup, cache
from sklearn.preprocessing import StandardScaler
import code

import pickle

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


def getStatcastData(startDate,endDate):
	
	cache.enable()

	return statcast(start_dt = startDate, end_dt = endDate)


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


	if Path(f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}countsInfo.pkl").is_file():
		
		print("Loading counts...")
		
		with open(f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}countsInfo.pkl","rb") as infile:
			info = pickle.load(infile)

		pitchers = info["pitchers"]
		pitchTypes = info["pitchTypes"]
		IdToName = info["IdToName"]
		counts = info["counts"]

	else:

		print("Computing counts...")

		rawData = getData(args.startYear,args.startMonth,args.startDay,args.endYear,args.endMonth,args.endDay,True)

		data,batterIndices,standardizer = manageData(rawData)


		pitchers = data.pitcher.unique()
		pitchTypes = data.pitch_type.unique()

		IdToName = {}
		counts = {}

		# code.interact("...", local=dict(globals(), **locals()))
		

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

			if pitcherID not in counts:
				counts[pitcherID] = {}

			for pitchType in pitchTypes:
				count = len(data.query(f"pitcher == {pitcherID} and pitch_type == '{pitchType}'",engine="python"))

				if pitchType not in counts[pitcherID]:
					counts[pitcherID][pitchType] = 0

				counts[pitcherID][pitchType] = count

		info = {}
		info["pitchers"] = pitchers
		info["pitchTypes"] = pitchTypes
		info["IdToName"] = IdToName
		info["counts"] = counts

		with open(f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}countsInfo.pkl","wb") as outfile:
			pickle.dump(info,outfile)




	with open(f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}countsPerPitchType-AllPitchers.txt","w") as outfile:

		# For a given pitch type
		for pitchType in pitchTypes:

			tempPitchers = []
			tempCounts = []

			# Find pitcher with max # of pitches (out of all pitchers)
			for pitcherID in pitchers:
				tempPitchers.append(pitcherID)
				tempCounts.append(counts[pitcherID][pitchType])

			maxCount = max(tempCounts)

			if maxCount != 0:
				maxPitcher = tempPitchers[tempCounts.index(maxCount)]

				print(f"Pitch Type: {pitchType} | ID: {maxPitcher} | Pitcher: {IdToName[maxPitcher]} | Count: {maxCount}")
				print(f"Pitch Type: {pitchType} | ID: {maxPitcher} | Pitcher: {IdToName[maxPitcher]} | Count: {maxCount}",file=outfile)


	print()


	pitchersJAIR = [573009,455119,594798,446372,661403,\
					605400,548389,643493,669923,458681,\
					656945,518617,656353,672851,656354,\
					579328,663855,592773,592761,670950]

	with open(f"Data{os.sep}Baseball{os.sep}StatcastData{os.sep}countsPerPitchType-JAIR-Pitchers.txt","w") as outfile:

		# For a given pitch type
		for pitchType in pitchTypes:

			tempPitchers = []
			tempCounts = []

			# Find pitcher with max # of pitches (out of pitchers used for JAIR paper)
			for pitcherID in pitchers:

				if pitcherID not in pitchersJAIR:
					continue

				tempPitchers.append(pitcherID)
				tempCounts.append(counts[pitcherID][pitchType])

			maxCount = max(tempCounts)

			if maxCount != 0:
				maxPitcher = tempPitchers[tempCounts.index(maxCount)]

				print(f"Pitch Type: {pitchType} | ID: {maxPitcher} | Pitcher: {IdToName[maxPitcher]} | Count: {maxCount}")
				print(f"Pitch Type: {pitchType} | ID: {maxPitcher} | Pitcher: {IdToName[maxPitcher]} | Count: {maxCount}",file=outfile)


	# code.interact("...", local=dict(globals(), **locals()))



main()




