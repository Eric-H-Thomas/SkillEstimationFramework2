import psycopg2
import random
import time, datetime
import math, numpy
import sys, os

import Environments.Billiards.FastFiz
import Environments.Billiards.Rules

#############################
# CUE CARD
#############################
# agent |  noise  | noise id
#   5 	|  0.125  |    14
#   7 	|  0.25   |    16
#   9 	|  0.0625 |    17
#   10	|  0.1875 |    18
#   11	|  0.375  |    19
#   19	|  0.1    |    20
#   20	|  0.2    |    21
#   21	|  0.3    |    22
#   23	|  0.025  |    23
#   24	|  0.5    |    24
#   25	|  0.4    |    26
#   26	|  0.05   |    25
#   27	|  0.625  |    28
#   28	|  0.75   |    29
#   --	|  0.005  |    27
#############################

#############################
# MACHINE GUNNER
#############################
# agent |  noise  | noise id
# 	6 	|  0.125  |	   14
#   12	|  0.25	  |    16
#   13	|  0.0625 |    17
#   15	|  0.1875 |    18
#   16	|  0.375  |    19
# 	29	|  0.025  |	   23
#	30	|  0.05   |    25
#	31	|  0.5    |    24
# 	32	|  0.75   |    29
#############################

agentsNoisesIDs = {5:14, 7:16, 9:17, 10:18, 11:19, 19:20, 20:21, 21:22, 23:23, \
					24:24, 25:26, 26:25, 27:28, 28:29, 6:14, 12:16, 13:17, 15:18, 16:19,\
					29:23, 30:25, 31:24, 32:29}


############### CONSTANTS ###############
g = 9.81;					 #< Gravitational constant.*/
MU_SLIDING = 0.2;			 #< coefficient of sliding friction */
MU_ROLLING = 0.015;		   #< coefficient of rolling friction */
MU_SPINNING = 0.044;		  #< coefficient of spinning friction */

TABLE_LENGTH = 2.236;		 # table length [m] */
TABLE_WIDTH = 1.116;		  #< table width [m] */
CORNER_POCKET_WIDTH = 0.11;   #< corner pocket width [m] */
SIDE_POCKET_WIDTH = 0.12;	 #< side pocket width [m] */
RAIL_HEIGHT = 0.040005;	   #/**< rail height [m] */
CUE_LENGTH = 1.45;			#/**< cue length [m] */
RAIL_VEL_DAMPING_X = 0.6;	 #< damping factor for velocity component parallel to rail */
RAIL_VEL_DAMPING_Y = 0.9;	 #< damping factor for velocity component perpendicular to rail */
RAIL_SPIN_DAMPING = 0.1;	  #< damping factor for angular velocity component */
RAIL_VEL_ANGLE_ADJ = 0.0;	 #< angle adjustment factor for velocity vector */
RAIL_ZSPIN_ANGLE_ADJ = 0.0;   #< angle adjustment factor for vertical component of angular velocity vector */

CUE_MASS = 160 / .302631579
I = 0.4*160.0*0.028575*0.028575*0.995473;   #< moment of inertia; to match poolfiz */
BALL_RADIUS = 0.028575

CARDINAL_RAIL = ["CR_W","CR_N","CR_E","CR_S","CR_UNKNOWN"]

MAX_CUT_ANGLE_DEGREES = 90.0

##########################################


class Point:
	def __init__(self,x=0,y=0):
		self.x = x
		self.y = y


############### Establishing table parameters ###############
length = TABLE_LENGTH;
width = TABLE_WIDTH;

diagonalWidth = CORNER_POCKET_WIDTH / math.sqrt(2.0);
straightWidth = SIDE_POCKET_WIDTH / 2.0;

_SWpocketLeft = Point( diagonalWidth, 0.0 );
_SWpocketRight = Point( 0.0, diagonalWidth );
_SWpocketCenter = Point( diagonalWidth/2.0, diagonalWidth/2.0 );
_WpocketLeft = Point( 0.0, length/2.0 - straightWidth );
_WpocketRight = Point( 0.0, length/2.0 + straightWidth );
_WpocketCenter = Point( 0.0, length/2.0 );
_NWpocketLeft = Point( 0.0, length - diagonalWidth );
_NWpocketRight = Point( diagonalWidth, length );
_NWpocketCenter = Point( diagonalWidth/2.0, length - diagonalWidth/2.0 );
_NEpocketLeft = Point( width - diagonalWidth, length );
_NEpocketRight = Point( width, length - diagonalWidth );
_NEpocketCenter = Point( width - diagonalWidth/2.0, length - diagonalWidth/2.0 );
_EpocketLeft = Point( width, length/2.0 + straightWidth );
_EpocketRight = Point( width, length/2.0 - straightWidth );
_EpocketCenter = Point( width, length/2.0 );
_SEpocketLeft = Point( width, diagonalWidth );
_SEpocketRight = Point( width - diagonalWidth, 0.0 );
_SEpocketCenter = Point( width - diagonalWidth/2.0, diagonalWidth/2.0 );	

_headString = length * 3/4;
_footSpot = Point(width/2, length/4);
###############################################################



class State:
	"""Object to represent tuples from 'states' table"""

	def __init__(self,row):
		self.stateID = row[0]
		self.turnType = row[1]
		self.curPlayerStarted = row[2] 
		self.playingSolids = row[3]
		self.timeLeft = row[4]
		self.timeLeftOpp = row[5]
		self.gameType = row[6]

	def toString(self):
		'''Function to save all the information on a 'State' instance to a string.'''

		string = str(self.stateID) +" "+ str(self.turnType) +" "+ str(self.curPlayerStarted) +" "+ \
				str(self.playingSolids) +" "+ str(self.timeLeft) +" "+ str(self.timeLeftOpp) +" "+ str(self.gameType)
		return string


class TableState:
	"""Object to represent tuples from 'TableStates' table"""

	def __init__(self,row):
		self.stateID = row[0]
		self.ball = row[1]
		self.status = row[2]
		self.x = row[3]
		self.y = row[4]

	def toString(self):
		'''Function to save all the information on a 'TableState' instance to a string.'''

		string = str(self.stateID) +" "+ str(self.ball) +" "+ str(self.status) +" "+ str(self.x)+" "+ str(self.y)
		return string



def connectToDb():
	'''Establish a connection with the database.'''
	
	try:
		newconn = psycopg2.connect("dbname='pool' user ='pool' host='localhost' password='FastFiz'")
		return newconn
	except psycopg2.Error as e:
		print("Connection error db")
		print(e.pgcode)
		print(e.pgerror)

def wrap_angle_180(angle):
	while(angle < -180):
		angle += 360
	while(angle > 180):
		angle -= 360
	return angle

def wrap_angle_360(angle):
	while(angle < 0):
		angle += 360
	while(angle > 360):
		angle -= 360
	return angle

def calculate_wrapped_action_difference(a1,a2):
	return abs(wrap_angle_180(a1-a2))


def getClosest(listP,p):
	'''
	This function returns the estimated nl phi that is closest to \
	the executed phi out of all the possible ones. Always works with angles. 
	'''

	diffList = []
	
	for each in listP:
		diff = calculate_wrapped_action_difference(p,each)
		diffList.append(diff)

	#Find the min on the list, get the index its position and return the value at that position
	return listP[diffList.index(min(diffList))]

def findClosest(aList,expectedNoises):
	'''Given a noise (or multiple noises), this function returns the noise (or a list of noises)
		from the discretized noises list (also given) that is the closest to the desired one(s).
		Only works with noises. '''

	closestList = []
	for eachEstimated in aList:
		diffs = []

		for eachExpected in expectedNoises:
			diffs.append(abs(eachExpected-eachEstimated))

		closest = min(diffs)
		closestIndex = diffs.index(closest)
		closestList.append(expectedNoises[closestIndex])

		# print "closest: ", closestList[-1],

	return closestList


def getPossibleBallsForState(state,ballsType):
	# 1 - 7 = solids
	# 9 - 15 = stripes
	# not including -> 8 = eight ball & 0 = cue ball

	ballsInfo = {"solid": [1,2,3,4,5,6,7], "stripe": [9,10,11,12,13,14,15]}

	balls = [] 

	# Get legal targets will return a list containing the balls that are currently
	# in play on the given state (balls will have a status of stationary|spinning|rolling|sliding|)
	ballsInPlay = getLegalTargets(state)

	# For each one of the balls in play on the given state
	for eachBall in ballsInPlay:
		# Verify if the ball belongs to the player (otherwise, ignore)
		if eachBall in ballsInfo[ballsType]:
			balls.append(eachBall)


	# Verify if the balls can be used for different types of shots

	return balls

######################################### Functions used within "ShotTypesPrediction.py" #########################################

def angle(A,B):		# takes member of point class (defined above)
	"""
	Return the angle from point A to point B and the positive x-axis.
	Values go from 0 to pi in the upper half-plane, and from 
	0 to -pi in the lower half-plane.
	"""
	return math.atan2(B.y-A.y, B.x-A.x)	#radians

def angle2(A,B,C,degrees):
	# Internal angle of p1,p2,p3.  This is always on [0,pi)

	# Use the cosine rule, solve for C
	a = dist(A,B)
	b = dist(B,C)
	c = dist(A,C)

	denom = 2*a*b

	result = 0
	if denom == 0:
		result = 0

	else:
		numerator = pow(a,2)+pow(b,2)-pow(c,2)
		
		#result = math.acos(numerator/denom) #radians
		result = numpy.arccos(numerator/denom) #radians

	if degrees:
		result = toDegrees(result)

	return result

def dist(A,B):
	'''
	Returns the distance between two points.
	'''
	return math.hypot(B.x-A.x, B.y-A.y)

def toDegrees(angleRad):
	return angleRad * 180 / math.pi

def get_pocket_center(index):

	if index == 0:
		return _SWpocketCenter
	elif index == 1:
		return _WpocketCenter
	elif index == 2:
		return _NWpocketCenter
	elif index == 3:
		return _NEpocketCenter
	elif index == 4:
		return _EpocketCenter
	elif index == 5:
		return _SEpocketCenter
	elif index == 6:
		return "Unknown Pocket"

def distFrom(point,rail):

	if rail == "CR_W":
		return point.x
	elif rail == "CR_N":
		return length - point.y
	elif rail == "CR_E":
		return width - point.x
	elif rail == "CR_S":
		return point.y
	else:
		#print("Not a rail "+str(rail))
		return 0

def getLegalTargets(state,allBallsFlag = False):

	# Establishing connection to database
	newconn = connectToDb()

	cur = newconn.cursor()
	
	query = """Select * from tablestates where stateid = '%s'""" % (state)

	cur.execute(query)

	# Populates 'rows' with all shots as tuples
	rows = cur.fetchall()

	ballAndStatus = []
	for each in rows:
		# Append ball and ball status to list
		ballAndStatus.append([each[1],each[2],each[3],each[4]])

	# If the allBalls flag is enabled, return the info of all the balls in the given state
	if allBallsFlag:
		return ballAndStatus

	# Otherwise, just return the balls in play
	else:
		ballsInPlay = inPlay(ballAndStatus)
		return ballsInPlay

def inPlay(ballAndStatus):
	# All of the possible ball status 
	# 	NOTINPLAY | STATIONARY | SPINNING | SLIDING | ROLLING | POCKETED_SW | POCKETED_W |
	#	POCKETED_NW | POCKETED_NE | POCKETED_E | POCKETED_SE | SLIDING_SPINNING | ROLLING_SPINNING | UNKNOWN_STATE

	# Stationary = 1   #Spinning = 2   #Sliding = 3   #Rolling = 4
	balls = []
	for each in ballAndStatus:
		# If status equal any of the given above
		if each[1] == 1 or each[1] == 2 or each[1] == 3 or each[1] == 4:
			# Append the ball, x  & y pos
			balls.append([each[0],each[2],each[3]])

	return balls

####################################################################################################################################



############################################### Helper functions for "executeShot()" ###############################################

def getResultString(result):
	'''
	This function returns the equivalent string for the given result (ShotResult)
	'''

	if result == 0:
		return "SR_OK"
	elif result == 1:
		return "SR_OK_LOST_TURN"
	elif result == 2:
		return "SR_BAD_PARAMS"
	elif result == 3:
		return "SR_SHOT_IMPOSSIBLE"
	elif result == 4:
		return "SR_TIMEOUT"
	else:
		return "Unknown Result"

def getCueXAndCueYPos(shot):

	# Establishing connection to database
	newconn = connectToDb()

	cue_cur = newconn.cursor()
	cue_query = """Select * from tablestates where stateid = '%s' and ball = '0'""" % shot.prev_state
	cue_cur.execute(cue_query)
	newRow = cue_cur.fetchone()

	# Return cue_x & cue_y
	return newRow[3], newRow[4]

def getStateInfo(state):
	'''
	This function will get the state information for the given state and return it
	'''

	# Establishing connection to database
	newconn = connectToDb()

	cue_cur = newconn.cursor()
	cue_query = """Select * from states where stateid = '%s'""" % state
	cue_cur.execute(cue_query)
	newRow = cue_cur.fetchone()

	state = State(newRow)
	#print(state.toString())

	return state

def getTableStateInfo(state):
	'''
	This function will get the table state information for the given state and return it
	'''

	# Establishing connection to database
	newconn = connectToDb()

	cue_cur = newconn.cursor()
	cue_query = """Select * from tablestates where stateid = '%s'""" % state
	cue_cur.execute(cue_query)
	newRow = cue_cur.fetchall()

	tableStateList = []
	for eachRow in newRow:
		tableStateList.append(TableState(eachRow))

	#print(tableState.toString())

	return tableStateList

def getState(state):

	# Establishing connection to database
	newconn = connectToDb()

	cue_cur = newconn.cursor()
	cue_query = """Select * from ballstates where status = '%s'""" % state.status
	cue_cur.execute(cue_query)
	newRow = cue_cur.fetchone()

	#print("Ball: ",state.ball)
	#print(newRow)

	# Description = state
	state = newRow[3]

	#print("State: ", state)

	return state

def getStateNumber(state):
	'''
	Function to return the equivalent number for the state description. It takes the description of a ball state obtained \
	from the database and look for its equivalent number on the 'State' enum type (found on FastFiz.h)
	'''

	stateNum = 99

	if state == "Not in Play":
		stateNum = 0
	elif state == "Stationary":
		stateNum = 1
	elif state == "Spinning":
		stateNum = 2
	elif state == "Sliding":
		stateNum = 3
	elif state == "Rolling":
		stateNum = 4
	elif state == "Pocketed SW":
		stateNum = 5
	elif state == "Pocketed W":
		stateNum = 6
	elif state == "Pocketed NW":
		stateNum = 7
	elif state == "Pocketed NE":
		stateNum = 8
	elif state == "Pocketed E":
		stateNum = 9
	elif state == "Pocketed SE":
		stateNum = 9
	elif state == "Sliding Spinning":
		stateNum = 10
	elif state == "Rolling Spinning":
		stateNum = 11
	elif state == "Unknown State":
		stateNum = 12

	return stateNum

def createStateString(state,tableStateList):

	numBalls = len(tableStateList)
	#print ("numBalls: ", numBalls)
	
	radius = 0.028575

	if state.curPlayerStarted == True: 
		state.curPlayerStarted = 1 # True
	else:
		state.curPlayerStarted = 0 # False

	# Conversion code from interval to seconds obtained from: http://stackoverflow.com/questions/10663720/converting-a-time-string-to-seconds-in-python

	# Convert time left - from interval to seconds
	#print("timeLeft: ", str(state.timeLeft))
	x = time.strptime(str(state.timeLeft).split('.')[0],'%H:%M:%S')
	state.timeLeft = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
	#print("Converted timeLeft: ", state.timeLeft)

	# Convert time left opponent - from interval to seconds
	#print("timeLeftOpp: ", str(state.timeLeftOpp))
	x = time.strptime(str(state.timeLeftOpp).split('.')[0],'%H:%M:%S')
	state.timeLeftOpp = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
	#print("Converted timeLeftOpp: ", state.timeLeftOpp)


	string = str(state.gameType) +" "+ str(state.turnType) +" "+ str(state.timeLeft)\
			 +" "+ str(state.timeLeftOpp) +" "+ str(state.curPlayerStarted) +" "+ str(numBalls) 

	# For each one of the balls, add info to string 
	for each in tableStateList:
		
		# Get state of the ball from db - description
		stateEach = getState(each)
		
		# Convert state to equivalent int on state enum in fastfiz.h
		
		stateNum = getStateNumber(stateEach)
		#print("StateNum: ", stateNum)

		string +=  " "+str(radius) +" "+ str(stateNum) +" "+ str(each.ball) +" "+ str(each.x) +" "+ str(each.y)


	openTable = 2 # Just an initial value

	if state.playingSolids == None:	# Null if the table is open or solids/stripes are irrelevant for the game.
		openTable = 1 # True  
	else:
		openTable = 0 # Close
		if state.playingSolids == True:
			state.playingSolids = 1 # True ################
		else:
			state.playingSolids = 0 # False

	string += " "+ str(openTable) +" "+ str(state.playingSolids)	

	return string

def executeShot(givenPhi,shot,debugMode=False):

	cue_x = 0
	cue_y = 0

	# Assign cue position params
	if shot.cue_x == None and shot.cue_y == None: 
		# Get the correspnding cue_x & cue_y pos from db
		cue_x, cue_y = getCueXAndCueYPos(shot)

	else:
		cue_x = shot.cue_x
		cue_y = shot.cue_y

	#print("After Shot cue_x: ",cue_x)
	#print("After Shot cue_y: ",cue_y)

	# Assign shot's parameters - Creates instance of ShotParams and save the corresponding shot's params
	shotParams = Environments.Billiards.FastFiz.ShotParams()
	
	#print("shot.v b4: ", shot.v)
	#print("ShotParams.v b4: ", shotParams.v)
	shotParams.v = shot.v
	#print("shot.v: ", shot.v)
	#print("ShotParams.v: ", shotParams.v)
	
	shotParams.a = shot.a
	shotParams.b = shot.b
	shotParams.theta = shot.theta
	shotParams.phi = givenPhi

	#print("ShotParams.phi: ", shotParams.phi)

	# Creates instance of shot to be executed
	gameShot = Environments.Billiards.Rules.GameShot()

	# Assign parameters to shot's instance
	gameShot.params = shotParams

	#Initialize parameters
	gameShot.cue_x = cue_x
	gameShot.cue_y = cue_y
	gameShot.ball = shot.ball
	gameShot.pocket = shot.pocket

	# Get state info from db to create str to use in Factory func
	state = getStateInfo(shot.prev_state)

	# Get table state info from db to create str to use in Factory func
	tableStateList = getTableStateInfo(shot.prev_state)

	# Generate str to use in the Factory func given the info on state and table state
	stateString = createStateString(state,tableStateList)

	# Calling Factory function - will create the state of the game given the information on the string
	gameState = Environments.Billiards.Rules.GameState.Factory(stateString)

	#string = gameState.toString()
	#print("\nStateString after toString(): ", string)

	result = gameState.executeShot(gameShot)
	print("\n")

	# ShotResults Enum
	# SR_OK = 0	|	SR_OK_LOST_TURN = 1	|	SR_BAD_PARAMS = 2	|	SR_SHOT_IMPOSSIBLE = 3	|	SR_TIMEOUT = 4

	if debugMode:
		print("Result on executeShot function: " +str(result)+ " = "+ str(getResultString(result)))

	# If result from executed shot is SR_OK or SR_OK_LOST_TURN, return True = was successful
	if result == 0: #or result == 1:
		#print("Result inside executeShot function: ",result)
		if debugMode:
			print("Valid shot, returning true...")
		return True, gameShot, gameState 
	# Otherwise, return false = was unsuccessful
	else:
		#print("Result inside executeShot function: ",result)
		if debugMode:
			print("Invalid shot, returning false...")
		return False, 0, 0

####################################################################################################################################


def setBouncePosition(targetPos,railIndex,ballPos,distRatio):
	#These equations were based on the observation that the tangent of the angle of
	#reflection is 3/2 times the tangent of the angle of incidence. Start by assuming
	#distRatio is 1 and draw the triangle that results; the base has a 3:2 ratio

	bouncePos = Point()

	if railIndex == "CR_W":
		bouncePos.y = ballPos.y + (2.0/(2 + 3*distRatio))*(targetPos.y - ballPos.y)
		bouncePos.x = BALL_RADIUS
	elif railIndex == "CR_E":
		bouncePos.y = ballPos.y + (2.0/(2 + 3*distRatio))*(targetPos.y - ballPos.y)
		bouncePos.x = width - BALL_RADIUS
	elif railIndex == "CR_N":
		bouncePos.x = ballPos.x + (2.0/(2 + 3*distRatio))*(targetPos.x - ballPos.x)
		bouncePos.y = length - BALL_RADIUS
	elif railIndex == "CR_S":
		bouncePos.x = ballPos.x + (2.0/(2 + 3*distRatio))*(targetPos.x - ballPos.x)
		bouncePos.y = BALL_RADIUS
	else:
		bouncePos.y = 0
		bouncePos.x = 0

	return bouncePos

def bankShotPossible(pocketIndex,rail):

	# See if we can bounce off of this rail into this pocket
	if rail == "CR_W":
		if pocketIndex == "NW" or pocketIndex == "W" or pocketIndex == "SW":
			return False
		else:
			return True
	elif rail == "CR_N":
		if pocketIndex == "NW" or pocketIndex == "NE":
			return False
		else:
			return True
	elif rail == "CR_S":
		if pocketIndex == "SW" or pocketIndex == "SE":
			return False
		else:
			return True
	elif rail == "CR_E":
		if pocketIndex == "NE" or pocketIndex == "E" or pocketIndex == "SE":
			return False
		else:
			return True
	else:
		#print("Not a rail "+str(rail))
		return False

def estimatedNLPhiForStraightInShots(shot):

	cuePos, ballPos, pocket = shot.getCueAndBallPosFromDb()

	#dist1_2 = dist(get_pocket_center(pocket), ballPos)	# Pocket to ball segment
	#dist1_3 = dist(get_pocket_center(pocket), cuePos)	# Pocket to cue segment
	#dist2_3 = dist(ballPos, cuePos) #Ball to Cue segment
	#init_angle = acos((pow(dist1_2,2) + pow(dist1_3,2) - pow(dist2_3,2)) / (2 * dist1_2 * dist1_3))
	# Law of Cosines; is it necessary or is zero reference point appropriate?
	
	'''print 'SHOT ID:', str(shot.shotid)
	print 'POCKET: ', str(pocket)
	print 'POCKET POSITION: ', get_pocket_center(pocket).x, ',', get_pocket_center(pocket).y
	print 'BALL POSITION  : ', ballPos.x, ',', ballPos.y
	print 'CUE POSITION   : ', cuePos.x, ',', cuePos.y'''
	
	init_angle = angle(get_pocket_center(pocket),ballPos)
	#print 'INIT ANGLE :', math.degrees(init_angle)
	
	# cos & sin receives param as radians
	xComp = (math.cos(init_angle)*2*BALL_RADIUS) + ballPos.x
	yComp = (math.sin(init_angle)*2*BALL_RADIUS) + ballPos.y
	desiredPos = Point(xComp, yComp)
	#print 'DESIRED POSITION: ', desiredPos.x, ',', desiredPos.y
	
	estimated_phiNotCorrected = angle(cuePos, desiredPos) # DEGREES, START_ZERO
	estimated_phiNotCorrected = math.degrees(estimated_phiNotCorrected)
	#print 'ESTIMATED PHI Before Correction: ', expected_phiNotCorrected

	if(estimated_phiNotCorrected is not None):
		estimated_phi = wrap_angle_180(estimated_phiNotCorrected)
		#print 'ESTIMATED PHI Corrected: ', estimated_phi

		return estimated_phi

	else:
		return 10000

def estimatedNLPhiForKickShots(shot):

	cuePos, ballPos, pocket = shot.getCueAndBallPosFromDb()

	init_angle = angle(get_pocket_center(pocket),ballPos)	
	#print 'INIT ANGLE :', math.degrees(init_angle)

	# cos & sin receives param as radians
	xComp = (math.cos(init_angle)*2*BALL_RADIUS) + ballPos.x
	yComp = (math.sin(init_angle)*2*BALL_RADIUS) + ballPos.y
	desiredPos = Point(xComp, yComp)
	#print 'DESIRED POSITION: ', desiredPos.x, ',', desiredPos.y
	
	estimatedNLPhiForDiffRails = []

	#For each one of the rails
	for rail in CARDINAL_RAIL:      
		#print "RAIL: " + rail    

		targetDistance = distFrom(desiredPos,rail) - BALL_RADIUS
		cueDistance = distFrom(cuePos,rail) - BALL_RADIUS

		#print "TARGET DISTANCE: " + str(targetDistance)
		#print "CUE DISTANCE B4: " + str(cueDistance)

		if cueDistance == 0:
			cueDistance = 1e-10

		distRatio = targetDistance / cueDistance
		#print "DIST RATIO: " + str(distRatio)

		# Calculate bounce position, set ShotParams values
		bouncePos = setBouncePosition(desiredPos,rail,cuePos,distRatio)
		#print "BOUNCE POS: X: " + str(bouncePos.x) + " Y: " + str(bouncePos.y)

		estimated_phiNotCorrected = angle(cuePos,bouncePos) # DEGREES, START_ZERO
		estimated_phiNotCorrected = math.degrees(estimated_phiNotCorrected)
		#print 'ESTIMATED PHI Before Correction: ', estimated_phiNotCorrected

		if estimated_phiNotCorrected is not None:
			estimated_phi = wrap_angle_180(estimated_phiNotCorrected)
			#print 'ESTIMATED PHI Corrected: ', estimated_phi

			estimatedNLPhiForDiffRails.append(estimated_phi)

	return estimatedNLPhiForDiffRails

def estimatedNLPhiForBankShots(shot):

	cuePos, ballPos, pocket = shot.getCueAndBallPosFromDb()

	pocketPos = get_pocket_center(pocket)

	estimatedNLPhiForDiffRails = []

	# For each one of the rails
	for rail in CARDINAL_RAIL:      
		#print "RAIL: " + rail   

		# If bank shot possible
		if bankShotPossible(pocket,rail):

			pocketDistance = distFrom(pocketPos,rail) - BALL_RADIUS
			ballDistance = distFrom(ballPos,rail) - BALL_RADIUS

			#print "TARGET DISTANCE: " + str(targetDistance)
			#print "CUE DISTANCE B4: " + str(cueDistance)

			if ballDistance == 0:
				ballDistance = 1e-10

			distRatio = pocketDistance / ballDistance
			#print "DIST RATIO: " + str(distRatio)

			# Calculate bounce position, set ShotParams values
			bouncePos = setBouncePosition(pocketPos,rail,ballPos,distRatio)
			#print "BOUNCE POS: X: " + str(bouncePos.x) + " Y: " + str(bouncePos.y)

			# Calculate desired position
			init_angle = angle(bouncePos,ballPos)
			#print 'INIT ANGLE :', math.degrees(init_angle)

			#cos & sin receives param as radians
			xComp = (math.cos(init_angle)*2*BALL_RADIUS) + ballPos.x
			yComp = (math.sin(init_angle)*2*BALL_RADIUS) + ballPos.y
			desiredPos = Point(xComp, yComp)
			#print 'DESIRED POSITION: ', desiredPos.x, ',', desiredPos.y

			estimated_phiNotCorrected = angle(cuePos,desiredPos) # DEGREES, START_ZERO
			estimated_phiNotCorrected = math.degrees(estimated_phiNotCorrected)
			#print 'ESTIMATED PHI Before Correction: ', estimated_phiNotCorrected

			if estimated_phiNotCorrected is not None:
				estimated_phi = wrap_angle_180(estimated_phiNotCorrected)
				#print 'ESTIMATED PHI Corrected: ', estimated_phi

				estimatedNLPhiForDiffRails.append(estimated_phi)

	return estimatedNLPhiForDiffRails

def estimatedNLPhiForComboShots(shot,legalTargets,ball):

	cuePos, ballPos, pocket = shot.getCueAndBallPosFromDb()

	pocketPos = get_pocket_center(pocket)

	estimatedNLPhiForDiffLegalTargets = []

	# For each one of the legal targets
	for legalTarget in legalTargets:      
		#print "legalTarget: " + str(legalTarget)  

		# If bank shot possible
		if legalTarget != ball:

			ball2Pos = Point(float(legalTarget[1]),float(legalTarget[2]))


			# Calculate desired position for ball 2
			init_angle = angle(ball2Pos,pocketPos)
			#print 'INIT ANGLE :', math.degrees(init_angle)

			#cos & sin receives param as radians
			xComp = (math.cos(init_angle)*2*BALL_RADIUS) + ball2Pos.x
			yComp = (math.sin(init_angle)*2*BALL_RADIUS) + ball2Pos.y
			desiredPos2 = Point(xComp, yComp)
			#print 'DESIRED POSITION 2: ', desiredPos2.x, ',', desiredPos2.y

			cut2 = 180 - angle2(ballPos,desiredPos2,pocketPos,True) #degrees = true
			#print cut2

			#if cut2 <= MAX_CUT_ANGLE_DEGREES:

			# Calculate desired position for ball
			init_angle = angle(desiredPos2,ballPos)
			#print 'INIT ANGLE :', math.degrees(init_angle)

			# cos & sin receives param as radians
			xComp = (math.cos(init_angle)*2*BALL_RADIUS) + ballPos.x
			yComp = (math.sin(init_angle)*2*BALL_RADIUS) + ballPos.y
			desiredPos = Point(xComp, yComp)
			#print 'DESIRED  POSITION: ', desiredPos.x, ',', desiredPos.y

			estimated_phiNotCorrected = angle(cuePos,desiredPos) # DEGREES, START_ZERO
			estimated_phiNotCorrected = math.degrees(estimated_phiNotCorrected)
			#print 'ESTIMATED PHI Before Correction: ', estimated_phiNotCorrected

			if estimated_phiNotCorrected is not None:
				estimated_phi = wrap_angle_180(estimated_phiNotCorrected)
				#print 'ESTIMATED PHI Corrected: ', estimated_phi

				estimatedNLPhiForDiffLegalTargets.append(estimated_phi)

	return estimatedNLPhiForDiffLegalTargets
