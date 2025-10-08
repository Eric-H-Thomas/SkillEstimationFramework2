import argparse
import subprocess 
import time
import os 

if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Make instructions to run experiments.')
# 	parser.add_argument('-ids','-ids', nargs='+', help='List of pitcher IDs to use',default = [])
# 	parser.add_argument('-types','-types', nargs='+', help='List of pitch types to use',default = [])
# 	parser.add_argument('-N','-N', help='Number of processes to spawn simultaneously', default=25)
	parser.add_argument('-file','-file', help='Name of the file that contains the IDs of the pitchers to use',default = "")
	args = parser.parse_args()


	if args.file == "":
		pitcher_ids = [547943, 621107, 628711, 425794, 605400, 642232, 527048, 547973, 642701, 621237, 670950, 622250, 605483, 675921, 641540, 657248, 621368, 656685, 622786, 592351, 663465, 519043, 667463, 669618, 594798, 543037, 453286, 477132, 669203, 669456, 621111, 623352, 521230, 621242, 607074, 592662]
	else:

		try:
			with open(args.file,"r") as inFile:
				pitcher_ids = inFile.readlines()
		except:
			print("Couldn't open file. Please make sure file exists.")
			exit()


	pitch_types = ['FF', 'SL', 'CU', 'SI', 'CH', 'KC', 'FC', 'FS', 'CS', 'KN', 'FA', 'EP', 'SC']
	
	spawn_list = []

	#Generate the list of all the spawn parameters
	for pitcherID in pitcher_ids:

		# Case reading from a file
		if type(pitcherID) != int:
			pitcherID = int(pitcherID.strip())

		arguments = []
		arguments.append('python')
		arguments.append('runExp.py')
		# arguments.append('testspawn.py')
		# arguments.append('-rerun') #remove if not in rerun mode
		# arguments.append('-domain')
		# arguments.append('baseball')
		arguments.append('-ids')
		arguments.append(f'{pitcherID}')
		arguments.append('-types')
		for pitchType in pitch_types:
			expFolder = f"Experiment-PitcherID-{pitcherID}-PitchType-{pitchType}"
			targ = arguments[:]
			targ.append(f'{pitchType}')
			targ.append('-resultsFolder')
			targ.append(f'{expFolder}')
	
			# Verify if experiment successfully finishes already
			# Inside a try/except block to manage the case when the "status" folder doesn't exist yet
			try:
				# Existence of the file indicates that the experiment finished successfully 
				statusFile = f"Experiments{os.sep}baseball{os.sep}{expFolder}{os.sep}status{os.sep}OnlineExp_{pitcherID}_{pitchType}-DONE.txt"
				print(f"Checking to see if {statusFile} exists.")
				if os.path.isfile(statusFile):
					print(f"Experiment for PitcherID-{pitcherID}-PitchType{pitchType} was already performed and it finished successfully.")
					continue
			except:
				pass

			#Add this to our list of processes to spawn
			spawn_list.append(targ)

	print(f'Created a spawn list with {len(spawn_list)} processes in it.')

	#Now we spawn the processes from the list
	current_spawns = []
	N = 25 #int(args.N)
	spawn_index = 0

	while spawn_index < len(spawn_list):

		#Check on our spawns, to see if any have finished			
		for s in current_spawns: 
			if s.poll() is not None:
				#If it finished, then remove it
				current_spawns.remove(s)
				print(f'Spawn finished. {len(current_spawns)} still running. {len(spawn_list) - spawn_index} left to spawn.')


		#Spawn enough to fill all our N slots
		while len(current_spawns) < N:
			#Get the next process to spawn
			next_spawn = spawn_list[spawn_index]
			spawn_index += 1
			print('Spawning process: ', next_spawn)
			spawn = subprocess.Popen(next_spawn)
			current_spawns.append(spawn)
			if spawn_index == len(spawn_list):
				print(f'Done spawning all {len(spawn_list)} subprocesses')

		#Rest until we check everything again
		time.sleep(120)


	print('All spawns finished.')
