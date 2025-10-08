###########################################################################
# Code by: Will Melville
# (Minor changes made for incorporation into the skill estimation framework)
###########################################################################

import pandas as pd
import code


#####################################################################################
# PREVIOUS UTILITY FUNCTION
#####################################################################################

'''
# Define the run values by count and by event 
count_runs = pd.DataFrame({'balls_pre_event': [2, 2, 0, 0, 1, 2, 1, 0, 1, 3, 3, 3], 
						   'strikes_pre_event': [1, 0, 2, 0, 2, 2, 0, 1, 1, 2, 1, 0],
						   'run_expectancy': [0.5268, 0.604, 0.3858, 0.494, 0.4081, 0.4492,
											 0.5312, 0.4434, 0.4737, 0.5537, 0.6406, 0.7179]})

event_runs = pd.DataFrame({'run_so': -0.2745, 'run_bb':  0.3399, 'run_bip_out': -0.2741, 'run_1b':  0.4685, 'run_2b':  0.7607, 'run_3b':  1.0476,'run_hr':  1.3749}, index = [0])
'''

def getUtilitiesPrev(pitch_df):
	'''
	pitch_df is a pandas dataframe with a column for balls, strikes, and the 
	probabilities of each outcome, named o1, o2, o3, o4, o5, o6, o7, and o8

	'''

	pitch_df['swing_utility'] = -1 
	pitch_df['take_utility'] = -1


	for balls in pitch_df.balls.unique():
		for strikes in pitch_df.strikes.unique():
			d = pitch_df.loc[(pitch_df.balls == balls) & (pitch_df.strikes == strikes)]

			# Get the pre pitch run expectancy 
			count_pre_expt = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes), 'run_expectancy'].values[0]

			# Get the value of a called ball 
			if balls == 3:
				# Walk 
				val0 = 0.3399 
			else:
				ball_val = count_runs.loc[(count_runs.balls_pre_event == balls + 1) & (count_runs.strikes_pre_event == strikes), 'run_expectancy'].values[0]
				val0 = ball_val - count_pre_expt 

			# Get the value of a called or swinging strike 
			if strikes == 2:
				# Strikeout 
				val12 = -0.2745 
			else:
				strike_val = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes+1), 'run_expectancy'].values[0]
				val12 = strike_val - count_pre_expt

			# Value of a foul ball 
			if strikes == 2:
				# No change 
				val3 = 0 
			else:
				val3 = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes+1), 'run_expectancy'].values[0] - count_pre_expt 

			# Value of ball in play out 
			val4 = -0.2741 
	  
			# Single 
			val5 = 0.4685
	  
			# Double 
			val6 = 0.7607 
	  
			# Triple 
			val7 = 1.0476
	  
			# Home Run 
			val8 = 1.3749

			# Calculate utilities 
			no_swing = d.o0.values + d.o1.values 
			swing = d[['o2','o3','o4','o5','o6','o7','o8']].values.sum(axis = 1)
			take_utility = (d.o0.values / no_swing) * val0 + (d.o1.values/no_swing) * val12
			swing_utility = (d.o2.values/swing) * val12 + (d.o3.values/swing)*val3 + (d.o4.values/swing)*val4 + (d.o5.values/swing)*val5 + (d.o6.values/swing)*val6 + (d.o7.values/swing)*val7 + (d.o8.values/swing) * val8
			d.swing_utility = swing_utility 
			d.take_utility = take_utility  
			pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls), ['swing_utility', 'take_utility']] = d[['swing_utility', 'take_utility']].values


	return pitch_df       


def getUtilityPrev(pitch_df):
	'''
	pitch_df is a pandas dataframe with a column for balls, strikes, and the 
	probabilities of each outcome, named o1, o2, o3, o4, o5, o6, o7, and o8
	'''

	pitch_df['strikes_pre_event'] = pitch_df.strikes
	pitch_df['balls_pre_event'] = pitch_df.balls

	pitch_df['utility'] = -1 
	pitch_df["actualPitchUtility"] = -1 

	for balls in pitch_df.balls.unique():
		for strikes in pitch_df.strikes.unique():
			d = pitch_df.loc[(pitch_df.balls == balls) & (pitch_df.strikes == strikes)]

			# Get the pre pitch run expectancy 
			count_pre_expt = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes), 'run_expectancy'].values[0]

			# Get the value of a called ball 
			if balls == 3:
				# Walk 
				val0 = 0.3399 
			else:
				ball_val = count_runs.loc[(count_runs.balls_pre_event == balls + 1) & (count_runs.strikes_pre_event == strikes), 'run_expectancy'].values[0]
				val0 = ball_val - count_pre_expt 

			# Get the value of a called or swinging strike 
			if strikes == 2:
				# Strikeout 
				val12 = -0.2745 
			else:
				strike_val = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes+1), 'run_expectancy'].values[0]
				val12 = strike_val - count_pre_expt

			# Value of a foul ball 
			if strikes == 2:
				# No change 
				val3 = 0 
			else:
				val3 = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes+1), 'run_expectancy'].values[0] - count_pre_expt 

			# Value of ball in play out 
			val4 = -0.2741 
	  
			# Single 
			val5 = 0.4685
	  
			# Double 
			val6 = 0.7607 
	  
			# Triple 
			val7 = 1.0476
	  
			# Home Run 
			val8 = 1.3749

			# Calculate utility
			utility =  (d.o0.values*val0) + (d.o1.values*val12) +\
						(d.o2.values*val12) + (d.o3.values*val3) +\
						(d.o4.values*val4) + (d.o5.values*val5) +\
						(d.o6.values*val6) + (d.o7.values*val7) +\
						(d.o8.values*val8)

			# Multiply by -1 to invert utility because we need
			# the utility from the pitcher's perspective 
			# and not the batter's perspective
			# batter = maximizing | pitcher = minimizing
			d.utility = utility*-1
			#pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls),['utility']] = d[['utility']].values


			# Also get the actual pitch utility 
			d.loc[d.outcome == 0,'actualPitchUtility'] = val0 
			d.loc[d.outcome.isin([1,2]),'actualPitchUtility'] = val12 
			d.loc[d.outcome == 3,'actualPitchUtility'] = val3 
			d.loc[d.outcome == 4,'actualPitchUtility'] = val4 
			d.loc[d.outcome == 5,'actualPitchUtility'] = val5 
			d.loc[d.outcome == 6,'actualPitchUtility'] = val6 
			d.loc[d.outcome == 7,'actualPitchUtility'] = val7 
			d.loc[d.outcome == 8,'actualPitchUtility'] = val8

			# d.actualPitchUtility = d.actualPitchUtility*-1

			pitch_df.loc[(pitch_df.strikes_pre_event == strikes) & (pitch_df.balls_pre_event == balls), ['utility','actualPitchUtility']] = d[['utility','actualPitchUtility']].values


	# code.interact("getUtility()...", local=dict(globals(), **locals()))
	return pitch_df       


#####################################################################################


#####################################################################################
# NEW UTILITY FUNCTION
#####################################################################################


count_runs = pd.DataFrame({'balls_pre_event': [3,3,2,3,2,1,0,1,2,0,1,0], 'strikes_pre_event': [0,1,0,2,1,0,0,1,2,1,2,2],
                           'val_ball': [0.131,0.201,0.110,0.276,0.103,0.063,0.034,0.050,0.098,0.027,0.046,0.022],
                           'val_strike': [-0.070,-0.076,-0.062,-0.351,-0.071,-0.050,-0.043,-0.067,-0.252,-0.062,-0.206,-0.184],
                           'val_out': [-0.496,-0.426,-0.385,-0.350,-0.323,-0.323, -0.289,-0.273,-0.252,-0.246,-0.206,-0.184],
                           'val_single': [0.287,0.356,0.397,0.432,0.459,0.460,0.494,0.510,0.530,0.537,0.577,0.598],
                           'val_double': [0.583,0.652,0.693,0.728,0.755,0.756,0.790,0.805,0.826,0.832,0.872,0.894],
                           'val_triple': [0.861,0.930,0.971,1.006,1.033,1.034,1.068,1.083,1.104,1.110,1.150,1.172],
                           'val_hr': [1.2,1.269,1.31,1.345,1.372,1.373,1.407,1.423,1.443,1.45,1.490,1.511]})


def getUtilities(pitch_df):
	'''
	pitch_df is a pandas dataframe with a column for balls, strikes, and the 
	probabilities of each outcome, named o1, o2, o3, o4, o5, o6, o7, and o8

	Returns the expected utility of a swing and of a take for that pitch.
	The batter's optimal utility for the pitch is the larger of those two values
	'''

	pitch_df['swing_utility'] = -1 
	pitch_df['take_utility'] = -1


	for balls in pitch_df.balls.unique():
		for strikes in pitch_df.strikes.unique():
			d = pitch_df.loc[(pitch_df.balls == balls) & (pitch_df.strikes == strikes)]

			#get the corresponding row from count_runs table
			count_pre = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes)]

			#get the value of a called ball 
			val0 = count_pre.val_ball.values[0]

			#get the value of a called or swinging strike 
			val12 = count_pre.val_strike.values[0]

			#value of a foul ball 
			if strikes == 2:
				#no change 
				val3 = 0 
			else:
				#value of foul is just value of a strike
				val3 = count_pre.val_strike.values[0] 

		 	#value of ball in play out 
			val4 = count_pre.val_out.values[0]
			#single 
			val5 = count_pre.val_single.values[0]
			#double 
			val6 = count_pre.val_double.values[0]
			#triple 
			val7 = count_pre.val_triple.values[0]
			#hr 
			val8 = count_pre.val_hr.values[0]

			#calculate utilities
			no_swing = d.o0.values + d.o1.values 
			swing = d[['o2','o3','o4','o5','o6','o7','o8']].values.sum(axis = 1)
			take_utility = (d.o0.values / no_swing) * val0 + (d.o1.values/no_swing) * val12
			swing_utility = (d.o2.values/swing) * val12 + (d.o3.values/swing)*val3 + (d.o4.values/swing)*val4 + (d.o5.values/swing)*val5 + (d.o6.values/swing)*val6 + (d.o7.values/swing)*val7 + (d.o8.values/swing) * val8
			d.swing_utility = swing_utility 
			d.take_utility = take_utility  
			pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls), ['swing_utility', 'take_utility']] = d[['swing_utility', 'take_utility']].values


	return pitch_df       


def getUtility(pitch_df):
	'''
	pitch_df is a pandas dataframe with a column for balls, strikes, and the 
	probabilities of each outcome, named o1, o2, o3, o4, o5, o6, o7, and o8

	Returns the expected utility of a swing and of a take for that pitch.
	The batter's optimal utility for the pitch is the larger of those two values
	'''

	pitch_df['swing_utility'] = -1 
	pitch_df['take_utility'] = -1

	pitch_df['utility'] = -1

	for balls in pitch_df.balls.unique():
		for strikes in pitch_df.strikes.unique():
			d = pitch_df.loc[(pitch_df.balls == balls) & (pitch_df.strikes == strikes)]

			#get the corresponding row from count_runs table
			count_pre = count_runs.loc[(count_runs.balls_pre_event == balls) & (count_runs.strikes_pre_event == strikes)]

			#get the value of a called ball 
			val0 = count_pre.val_ball.values[0]

			#get the value of a called or swinging strike 
			val12 = count_pre.val_strike.values[0]

			#value of a foul ball 
			if strikes == 2:
				#no change 
				val3 = 0 
			else:
				#value of foul is just value of a strike
				val3 = count_pre.val_strike.values[0] 

		 	#value of ball in play out 
			val4 = count_pre.val_out.values[0]
			#single 
			val5 = count_pre.val_single.values[0]
			#double 
			val6 = count_pre.val_double.values[0]
			#triple 
			val7 = count_pre.val_triple.values[0]
			#hr 
			val8 = count_pre.val_hr.values[0]

			#calculate utilities
			''' 
			no_swing = d.o0.values + d.o1.values 
			swing = d[['o2','o3','o4','o5','o6','o7','o8']].values.sum(axis = 1)
			take_utility = (d.o0.values / no_swing) * val0 + (d.o1.values/no_swing) * val12
			swing_utility = (d.o2.values/swing) * val12 + (d.o3.values/swing)*val3 + (d.o4.values/swing)*val4 + (d.o5.values/swing)*val5 + (d.o6.values/swing)*val6 + (d.o7.values/swing)*val7 + (d.o8.values/swing) * val8
			d.swing_utility = swing_utility 
			d.take_utility = take_utility  
			'''


			# Calculate utility
			utility =  (d.o0.values*val0) + (d.o1.values*val12) +\
						(d.o2.values*val12) + (d.o3.values*val3) +\
						(d.o4.values*val4) + (d.o5.values*val5) +\
						(d.o6.values*val6) + (d.o7.values*val7) +\
						(d.o8.values*val8)

			# Multiply by -1 to invert utility because we need
			# the utility from the pitcher's perspective 
			# and not the batter's perspective
			# batter = maximizing | pitcher = minimizing
			d.utility = utility*-1

			# pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls), ['swing_utility', 'take_utility','utility']] = d[['swing_utility', 'take_utility','utility']].values
			pitch_df.loc[(pitch_df.strikes == strikes) & (pitch_df.balls == balls), ['utility']] = d[['utility']].values

	return pitch_df       


#####################################################################################


if __name__ == '__main__':

	test = pd.DataFrame({'o0':0.5, 'o1': 0, 'o2':0, 'o3':0,'o4':0, 'o5':0,'o6':0,'o7':0, 'o8':0.5, 'balls':0,'strikes':0}, index = [0])
	# r = getUtilities(test)
	r = getUtility(test)
	print(r)

