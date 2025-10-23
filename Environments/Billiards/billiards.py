import psycopg2
import numpy as np
import math
import os, sys, subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

import datetime, time
import copy, code, imp
import json

from Environments.Billiards.utilsBilliards import *


def get_domain_name():
	return "billiards"
	

class Shot:
	"""Object to represent tuples from 'shots' table"""

	shot_list = []	# Class attribute; stores all shot instances

	def __init__(self,row):
		self.shotid = row[0]
		self.gameid = row[1]
		self.agentid = row[2]
		self.prev_state = row[3]
		self.next_state = row[4]
		
		self.a = row[5]
		self.b = row[6]
		self.theta = row[7]
		self.phi = row[8]
		self.v = row[9]
		
		self.cue_x = row[10]
		self.cue_y = row[11]
		self.ball = row[12]
		self.pocket = row[13]

		self.nl_a = row[14]
		self.nl_b = row[15]
		self.nl_theta = row[16]
		self.nl_phi = row[17]
		self.nl_v = row[18]
		
		self.decision = row[19]
		self.timespent = row[20]
		self.timedone = row[21]
		self.duration = row[22]
		self.remote_ip = row[23]

	# Must go inside of the class since it uses information from the calling instance
	def getCueAndBallPosFromDb(self):

		if(self.ball is not None):
			ball = self.ball
			pocket = self.pocket
			state = self.prev_state
			
			# Establishing connection to database
			newconn = connectToDb()

			cur = newconn.cursor()
			
			# var = stateid from prev state, var2 = ball from tablestates from stateID)
			query = """Select * from tablestates where stateid = '%s' and ball = '%s'""" % (state, ball)

			cur.execute(query)

			# Populates 'rows' with all shots as tuples
			row = cur.fetchone()

			status = row[2]

			#ball_x =row[3]
			#ball_y =row[4]
			ballPos = Point(row[3],row[4])

			if (self.cue_x is not None and self.cue_y is not None):
				#print 'CUE IF'
				cuePos = Point(self.cue_x, self.cue_y)	
			else:
				#print 'CUE ELSE'
				cue_cur = newconn.cursor()
				cue_query = """Select * from tablestates where stateid = '%s' and ball = '0'""" % state
				cue_cur.execute(cue_query)
				newRow = cue_cur.fetchone()
				cuePos = Point(newRow[3], newRow[4])

			return cuePos, ballPos, pocket


class ProcessedShot:
	"""Object to save all the computed information (including EN & TN) \
		for a shot after been procesed/executed"""

	def __init__(self,gameID,shotID,nlPhi,phi,estimatedNLPhi,tn,en,successfulOrNot,\
					estimatedMethod,estimatedNLPhiClosestToNLPhi,estimatedNLPhiClosestToNLPhiMethod,estimatedNLPhisAll):

		self.gameID = gameID
		self.shotID = shotID
		self.nlPhi = nlPhi
		self.phi = phi
		self.estimatedNLPhi = estimatedNLPhi

		self.trueNoise = tn
		self.estimatedNoise = en

		# To save if the phi (executed) was successful or not
		self.successfulOrNot = successfulOrNot

		self.estimatedMethod = estimatedMethod
		self.estimatedNLPhiClosestToNLPhi = estimatedNLPhiClosestToNLPhi
		self.estimatedNLPhiClosestToNLPhiMethod = estimatedNLPhiClosestToNLPhiMethod
		
		# A list containing all the possible shots for a given executed shot
		self.estimatedNLPhisAll = estimatedNLPhisAll

	def toString(self):

		string = ""
		string += "\n******************************************************************************\n"
		string += "ShotID: " + str(self.shotID) + "\n"
		string += "\tnlPhi: " + str(self.nlPhi) + "\n"
		string += "\tphi: " + str(self.phi) + "\n"
		string += "\tmethod: " + str(self.method) + "\n"
		string += "\testimatedNLPhi: " + str(self.estimatedNLPhi) + "\n"
		string += "\tTrue Noise: " + str(self.trueNoise) + "\n"
		string += "\tEstimated  Noise: " + str(self.estimatedNoise) + "\n"
		string += "\tsuccessful? " + str(self.successfulOrNot) + "\n"
		string += "\testimatedMethod:  " + str(self.estimatedMethod) + "\n"
		string += "\testimatedNLPhiClosestToNLPhi: " + str(self.estimatedNLPhiClosestToNLPhi) + "\n"
		string += "\testimatedNLPhiClosestToNLPhiMethod: " + str(self.estimatedNLPhiClosestToNLPhiMethod) + "\n"
		string += "\testimatedNLPhisAll: \n"

		for each in self.estimatedNLPhisAll:
			string += "\t\t" + str(each) + "\n"

		string += "\n******************************************************************************\n"

		return string 


def getAndProcessShots(numObservations,agent,seedNum=None,rerun=False):

	# Get the desired shots & Initialize Shot.shot_list with them - random
	xskill, seedNum = getShots(numObservations,agent,seedNum)

	# If don't have the info already, proceed to process the shots
	if rerun == False:
		# Compute all the necesary information for all the shots using the specified method
		processed, processedShotsList = processAllShots(numObservations,agent,xskill,seedNum)

		# If not able to process shots bc file for test exist, read - rerun
		if not processed:
			#print "exist. going to read file"
			processedShotsList = readProcessedShotsInfo(numObservations,agent,seedNum)

	# Else, the information has already been saved to a file, read from it and use that for predictions
	else:
		processedShotsList = readProcessedShotsInfo(numObservations,agent,seedNum)

	return processedShotsList


def getShots(numObservations,agent,seedNum=None):
	
	# Selects shots randomly

	# Establishing connection to database
	conn = connectToDb()

	# Open cursor to perform database operations
	cur = conn.cursor()

	# Execute a command
	cur.execute("""SELECT gameid, agentid1, agentid2, noiseid1, noiseid2 from games
						where (agentid1 ='%s'or agentid2 ='%s') """%(agent,agent))

	# Populates 'rows' with all the ids of the available games as tuples
	rows = cur.fetchall()

	allGameIDsList = []
	allAgentIds1 = []
	allAgentIds2 = []
	allNoiseIds1 = []
	allNoiseIds2 = []

	gamesDict = {}

	# Save all the agent ID's on a list (allGameIDsList)
	for row in rows:
		allGameIDsList.append(row[0])
		allAgentIds1.append(row[1])
		allAgentIds2.append(row[2])
		allNoiseIds1.append(row[3])
		allNoiseIds2.append(row[4])

	# Add game to dictionary -- key = game & init to 0 for count
	for game in allGameIDsList:
		gamesDict[game] = 0

	# print("allGameIDsList len:", len(allGameIDsList))#," ",len(allAgentIds1)," ", len(allAgentIds2)," ", len(allNoiseIds1)," ",len(allNoiseIds2))
	#print(len(gamesDict))
	#print(gamesDict)

	# for each in range(len(allGameIDsList)):
		# print("gameID: ",allGameIDsList[each],"agent1: ",allAgentIds1[each],"agent2: ",allAgentIds2[each],"noise1: ",allNoiseIds1[each],"noise2: ",allNoiseIds2[each])

	# To save the id's of the games to select the shots from
	gameIDList = []
	noiseIDList = []


	#If no seed num specified, generate a random one
	if seedNum == None:
		seedNum = random.randint(0,np.iinfo(np.int32).max)

	random.seed(seedNum)


	# While we don't have all the required number of shot on the list
	while len(Shot.shot_list) != numObservations:

		# Pick a random int between 0 and len of list containing the game id's
		# Will represent the index of the game id selected
		pos = random.randint(0,len(allGameIDsList)-1)

		# Get the id of the selected game
		gameID = allGameIDsList[pos]
		# gameID = 3853 #3456
		# print("GameID: ", gameID)
		# print("AgentID: ", agent)
		# print(allNoiseIds1[pos])
		# print(allNoiseIds2[pos])

		# print("agentsNoisesIDs[agentID]: ", agentsNoisesIDs[agentID])

		# Verify if given agent matches any of the one in the game (as well as if the noise matches the one it should be). 
		# If so, consider game. 
		# Else, continue.
		noiseID = 0
		if allAgentIds1[pos] == agent and allNoiseIds1[pos] == agentsNoisesIDs[agent]:
			noiseID = allNoiseIds1[pos]
			# print("noiseID", noiseID)
			#print("if - noiseID: ",noiseID)
			#print("allAgentIds1[pos]: ",allAgentIds1[pos])
			#print("if - allAgentIds1[pos]: ",allAgentIds1[pos])
		elif allAgentIds2[pos] == agent and allAgentIds2[pos] == agentsNoisesIDs[agent]:
			noiseID = allNoiseIds2[pos]
			#print("noiseID", noiseID)
			#print("else - noiseID: ",noiseID)
			#print("allAgentIds1[pos]: ",allAgentIds2[pos])
			#print("else - allAgentIds2[pos]: ",allAgentIds2[pos])
		#Agents don't match, continue (don't consider game)
		else:
			#print("Continue - not same agent")
			#print("allAgentIds1[pos]: ",allAgentIds1[pos], "	allAgentIds2[pos]: ",allAgentIds2[pos])
			continue

		# If we haven't selected the shots for this game previously 
		# or if we have tried to visit it many times before, the allow it to be added again
		
		# print "gamesDict[gameID]: ", gamesDict[gameID]
		if gameID in gamesDict and (gamesDict[gameID] == 0 or gamesDict[gameID]%10 == 0):

			if gamesDict[gameID]%10 == 0 and gamesDict[gameID] != 0:
				visitedTimes = (gamesDict[gameID]/10)
				# print("Game ", gameID," has been visited ",visitedTimes," time(s) already. Going to visit it again.")


			# Select all the shots that were executed by the specified agent during the specified game
			cur.execute("""SELECT * from shots where agentid ='%s' and gameid = '%s'""" % (agent, gameID))

			# Populates 'rows' with all shots as tuples
			rows = cur.fetchall()

			# print('Fetched ', len(rows), ' shots for agent ', agent, ' on game ', gameID)

			# To Remove the null nl_phi & initialize Shot.shot_list
			tempShotList = createShotObjects(rows,agent)

			# print('Actual num of shots: ', len(tempShotList))

			shotsAdded = False

			# Append all shots for given agentID & gameID on class' list if they fit
			if (len(Shot.shot_list) + len(tempShotList)) < numObservations:
				# Save all the shot for given agentID & gameID on a list (shots)
				for row in tempShotList:
					#print(row.shotid)
					#print(row.cue_x,row.cue_y)
					Shot.shot_list.append(row)
				# print("Shots added to list: ", len(tempShotList))
				shotsAdded = True

			# They don't fit all, so append until full
			else:
				spaceLeft = numObservations - len(Shot.shot_list)
				for x in range(0,spaceLeft):
					#print(rows[x])
					Shot.shot_list.append(tempShotList[x])
				# print("Shots added to list: ",spaceLeft)
				shotsAdded = True


			if shotsAdded:
				noiseIDList.append(noiseID)

				# Add game to list of selected games
				# print("Selecting shots from game ", gameID, " and agent ", agent)
				gameIDList.append(gameID)


			# Game was visited once before and have tried to access it again 15 times, allow it to do so then
			# If going to allow game = duplicated data -> inform
			if gamesDict[gameID]%10 == 0 and gamesDict[gameID] != 0:
				with open("Data"+os.path.sep+"BilliardsProcessedShots"+os.path.sep+"shotSelectionDuplicatedData"+str(seedNum)+".txt", 'a') as textfile:
					textfile.write("Game "+str(gameID)+" has been visited "+str(visitedTimes)+" time(s) already. Going to visit it again.\n")
					textfile.write("Shot IDs: \n")
					for eachShot in tempShotList:
						textfile.write(str(eachShot.shotid)+" | ")
					textfile.write("\n\n")

			# Count game as seen once 
			gamesDict[gameID] += 1

			# print("Saw game for the ",(gamesDict[gameID]/10) + 1, " time(s)\n")

		else:
			gamesDict[gameID] += 1
			# print("Incrementing count for game since we tried to access it again...")
			# print("gamesDict[gameID]: ", gamesDict[gameID])
	
		# print()
		
	# print("Shot.shot_list has been initialized with ", len(Shot.shot_list), " shots.")

	#print(noiseIDList)
	# Execute a command
	cur.execute("""SELECT n_phi from noise where noiseid ='%s'"""%(noiseIDList[0]))

	# Populates 'rows' with the corresponding noise for the given noiseID
	rows = cur.fetchone()
	xskill = rows[0]

	# code.interact("...", local=dict(globals(), **locals()))

	return xskill, seedNum

def createShotObjects(shots,agent):

	'''
	This function creates the corresponding instances for each one of the valid shots.
	It filters out the shots having a none nl_phi or a none pockets.
	It also filters the break shots.\
	It saves them all on a temp list. 
	'''

	tempShotList = []

	# Will keep track of how many shots there are for the DESIRED agent
	agent1 = 0
	# Will keep track of how many shots there are for the OTHER agent
	agent2 = 0

	#For each one of the shots on the list
	for shot in shots:
		# Create a new instance of the Shot's class given the information on row (shot)
		new_shot = Shot(shot)

		#print("Shot decision: ", new_shot.decision)
		#print("Shot timespent: ", new_shot.timespent)

		#print("cue_x: ", new_shot.cue_x)
		#print("cue_y: ", new_shot.cue_y)
		#print("nl_phi: ", new_shot.nl_phi)

		#print("Shot: ", shot)

		'''
		if new_shot.nl_phi == 275.0:
			print('275 pocket and ball: ', new_shot.pocket, ' and ', new_shot.ball)
		else:
			print('Normal pocket and ball: ', new_shot.pocket, ' and ', new_shot.ball)
		'''

		# To filter out break shots
		if new_shot.nl_phi is None or new_shot.pocket is None:
			# print('We found a None nl-phi!')
			pass
		else:

			# To keep track of how many shots belong to which agent
			# If the shot belongs to the desired agent, add 1 to desired agent counter
			if new_shot.agentid == agent:
				agent1 += 1
			# Otherwise add 1 to the other counter
			else:
				agent2 += 1

			# Add shot to temp list
			tempShotList.append(new_shot)
			#print("Shot: ", shot)
			#print("New Shot: ", new_shot.cue_x, new_shot.cue_y)

	# Return the list as it is
	return tempShotList

def processAllShots(numObservations,agent,xskill,seedNum):
	'''
	This function processes all the shots. It computes the TN and EN for each shot along with other info and saves it. 
	It first verifies whether a file with the given seed number exists. 
		If it does, it stops and sends a flag back to proceed to read them rather than to process them again.
		If it doesn't, then it proceeds to process them and store the info on a file.
	All of the files are saved within the folder named "processedShotsInfoForDifferentTests" 
	'''

	fileNameProcessedShots = "shotsInfo-Agent"+str(agent)+"-Shots"+str(numObservations)\
								+"-Seed"+str(seedNum)+".json"

	saveAt = "Data"+os.path.sep+"BilliardsProcessedShots"+os.path.sep+fileNameProcessedShots

	# Verify if tests with given seed num already exist
	if os.path.isfile(saveAt):
		# test exits, won't process
		#print("test exists, won't process")
		return False, []

	processedShotsList = []

	gameIDList = []
	shotIDList = []
	nlPhiList = []
	phiList = []
	estNLPhiList = []
	tnList = []
	enList = []
	successfulList = []
	estMethodList = []
	estNLPhiClosestList = []
	estClosestMethodList = []
	estimatedNLPhisAllList = []


	#For each one of the shots
	for each in range(numObservations):

		# Compute difference for (noisy phi - nl phi)
		trueNoiseNoWrap = (Shot.shot_list[each].phi - Shot.shot_list[each].nl_phi)
		trueNoise = wrap_angle_180(trueNoiseNoWrap)

		#print 'Noisy Phi:' +str(Shot.shot_list[each].phi)+ ' - NL Phi: ' +str(Shot.shot_list[each].nl_phi)+ " = "+str(trueNoiseNoWrap)
		#print "After wrapping: " + str(trueNoise)

		# Execute shot with its phi to see if it's successful or not
		result, gameShot, gameState = executeShot(Shot.shot_list[each].phi,Shot.shot_list[each],False)
		
		if result == True:
			successfulOrNot = "Yes"
		else:
			successfulOrNot = "No"		

		# Estimate/predict the nl phi  for the given method -ALL CLOSEST
		estimatedNLPhi,estimatedMethod,estimatedNLPhiClosestToNLPhi,estimatedNLPhiClosestToNLPhiMethod,estimatedNLPhisAll = \
			getEstimatedNLPhiAll(Shot.shot_list[each])

		'''
		print("Noisy Phi: " + str(Shot.shot_list[each].phi))
		print("Estimated NL Phi: "+str(estimatedNLPhi))
		#print("Estimated Method: "+str(estimatedMethod))
		print("NL Phi: " + str(Shot.shot_list[each].nl_phi))
		#print("State: " + str(Shot.shot_list[each].prev_state))
		print("\n")
		'''

		# Compute difference for (noisy phi - estimated nl phi)
		estimatedNoiseNoWrap =  (Shot.shot_list[each].phi - estimatedNLPhi)
		estimatedNoise = wrap_angle_180(estimatedNoiseNoWrap)

		# print('Noisy Phi:' +str(Shot.shot_list[each].phi)+ ' - Estimated NL Phi: ' +str(estimatedNLPhi)+ " = "+str(estimatedNoiseNoWrap))
		# print("After wrapping: " + str(estimatedNoise))
		

		# Create instance of Processed Shot Method 5 class to save all the computed information
		# and save on the list
		processedShotsList.append(ProcessedShot(Shot.shot_list[each].gameid,Shot.shot_list[each].shotid,Shot.shot_list[each].nl_phi,Shot.shot_list[each].phi,\
			estimatedNLPhi,trueNoise,estimatedNoise,successfulOrNot,\
			estimatedMethod,estimatedNLPhiClosestToNLPhi,estimatedNLPhiClosestToNLPhiMethod,estimatedNLPhisAll))


		gameIDList.append(Shot.shot_list[each].gameid)
		shotIDList.append(Shot.shot_list[each].shotid)
		nlPhiList.append(Shot.shot_list[each].nl_phi)
		phiList.append(Shot.shot_list[each].phi)
		estNLPhiList.append(estimatedNLPhi)
		tnList.append(trueNoise)
		enList.append(estimatedNoise)
		successfulList.append(successfulOrNot)
		estMethodList.append(estimatedMethod)
		estNLPhiClosestList.append(estimatedNLPhiClosestToNLPhi)
		estClosestMethodList.append(estimatedNLPhiClosestToNLPhiMethod)
		estimatedNLPhisAllList.append(estimatedNLPhisAll)


	processedShotsDict = {}
	processedShotsDict["gameIDList"] = gameIDList
	processedShotsDict["shotIDList"] = shotIDList
	processedShotsDict["nlPhiList"] = nlPhiList
	processedShotsDict["phiList"] = phiList
	processedShotsDict["estNLPhiList"] = estNLPhiList
	processedShotsDict["tnList"] = tnList
	processedShotsDict["enList"] = enList
	processedShotsDict["successfulList"] = successfulList
	processedShotsDict["estMethodList"] = estMethodList
	processedShotsDict["estNLPhiClosestList"] = estNLPhiClosestList
	processedShotsDict["estClosestMethodList"] = estClosestMethodList
	processedShotsDict["estimatedNLPhisAllList"] = estimatedNLPhisAllList

	processedShotsDict["noise"] = xskill

	# Save to json file
	with open(saveAt, 'w') as outfile:
		json.dump(processedShotsDict, outfile)



	# Shots were processed
	return True, processedShotsList

def getEstimatedNLPhiAll(shot):
	''' 
	Returns the expected phi from -180 to 180
	'''

	noisyPhi = shot.phi
	state = shot.prev_state
	ball = shot.ball
	nlPhi = shot.nl_phi
	
	# print("\nGETTING ESTIMATED NL PHI * * * ")

	verbose = False

	execSkillClosest = []

	# Compute estimated nl phi for straight-in shots
	estNLPhi = estimatedNLPhiForStraightInShots(shot)
	execSkillClosest.append([estNLPhi,1])

	# method == 2, compute estimated nl phi for kick shots
	estimatedNLPhiForDiffRails = estimatedNLPhiForKickShots(shot)
	#print("Estimated nl phi list for kick shots" )
	#print(estimatedNLPhiForDiffRails)
	for each in estimatedNLPhiForDiffRails:
		execSkillClosest.append([each,2])

	# method == 3, compute estimated nl phi for bank shots
	estimatedNLPhiForDiffRails2 = estimatedNLPhiForBankShots(shot)
	#print("Estimated nl phi list for bank shots")
	#print(estimatedNLPhiForDiffRails)
	for each in estimatedNLPhiForDiffRails2:
		execSkillClosest.append([each,3])

	# method == 4, compute estimated nl phi for combo shots
	legalTargets = getLegalTargets(state)
	estimatedNLPhiForDiffLegalTargets = estimatedNLPhiForComboShots(shot,legalTargets,ball)
	#print("Estimated nl phi list for combo shots")
	#print(estimatedNLPhiForDiffLegalTargets)
	for each in estimatedNLPhiForDiffLegalTargets:
		execSkillClosest.append([each,4])

	# To wrap all the computed angles on list
	for each in range(len(execSkillClosest)):
		execSkillClosest[each][0] = wrap_angle_360(execSkillClosest[each][0]) 

	# To save the estimatedNLPhi on a list - without the method
	closestList = []
	for each in execSkillClosest:
		#print each
		closestList.append(each[0])

	# Get the closest estimatedNLPhi to the actual noisyPHi
	closestToNoisyPhi = getClosest(closestList,noisyPhi)
	#print("Noisy Phi: ", noisyPhi)
	#print("Closest to Noisy Phi: ", closestToNoisyPhi)

	# Get the closest estimatedNLPhi to the NL Phi
	closestToNLPhi = getClosest(closestList,nlPhi)
	#print("NL Phi: ", nlPhi)
	#print("Closest to NL Phi: ", closestToNLPhi)

	indexClosestNoisyPhi = closestList.index(closestToNoisyPhi)
	methodClosestNoisyPhi = execSkillClosest[indexClosestNoisyPhi][1]
	#print("indexClosestNoisyPhi: ",indexClosestNoisyPhi)
	#print("methodClosestNoisyPhi: ", methodClosestNoisyPhi)

	indexClosestNLPhi = closestList.index(closestToNLPhi)
	methodClosestNLPhi = execSkillClosest[indexClosestNLPhi][1]
	#print("indexClosestNLPhi: ",indexClosestNLPhi)
	#print("methodClosestNLPhi: ", methodClosestNLPhi)

	# execSkillClosest = list containing all of the possible shots for a given executed shot
	# [0] = shot (phi)
	# [1] = method

	return closestList[indexClosestNoisyPhi],int(methodClosestNoisyPhi),\
			closestList[indexClosestNLPhi],int(methodClosestNLPhi),execSkillClosest

def readProcessedShotsInfo(numObservations,agent,seedNum):
	'''
	This function will read a file (given the seed number) 
	containing the already processed info of the shots.
	It returns the noise of the agent.
	'''

	print("Reading processed shot's info...")

	fileName = "shotsInfo-Agent"+str(agent)+"-Shots"+str(numObservations)+"-Seed"+str(seedNum)+".json"

	# Open the file with the shot's info
	toLoad = "Data"+os.path.sep+"BilliardsProcessedShots"+os.path.sep+fileName


	with open(toLoad, "r") as infile:
		results = json.load(infile)

		gameIDList = results["gameIDList"]
		shotIDList = results["shotIDList"]
		nlPhiList = results["nlPhiList"]
		phiList = results["phiList"]
		estNLPhiList = results["estNLPhiList"]
		tnList = results["tnList"]
		enList = results["enList"]
		successfulList = results["successfulList"]
		estMethodList = results["estMethodList"]
		estNLPhiClosestList = results["estNLPhiClosestList"]
		estClosestMethodList = results["estClosestMethodList"]
		estimatedNLPhisAllList = results["estimatedNLPhisAllList"]

		xskill = results["noise"]


	processedShotsList = []

	for i in range(len(shotIDList)):

		tempProcessedShot = ProcessedShot(gameIDList[i],shotIDList[i],nlPhiList[i],phiList[i],
				estNLPhiList[i],tnList[i],enList[i],successfulList[i],estMethodList[i],
				estNLPhiClosestList[i],estClosestMethodList[i],estimatedNLPhisAllList[i])

		processedShotsList.append(tempProcessedShot)

	return processedShotsList