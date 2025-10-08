import argparse
import subprocess 
import time
import os 

if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Make instructions to run experiments.')
	args = parser.parse_args()


	numObservations = [100,250,500]
	numParticles = [200,1000,2000]
	# noises = [5,50,500]
	# percents = [0.75,0.75,0.80,0.90]

	seeds = [10]

	
	spawn_list = []

	#Generate the list of all the spawn parameters

	arguments = []
	arguments.append('python')
	arguments.append('runExp.py')

	arguments.append('-domain')
	arguments.append('2d-multi')

	arguments.append('-mode')
	arguments.append('rand_pos')

	arguments.append('-seed')
	arguments.append(f"{seeds[0]}")


	arguments.append('-iters')
	arguments.append('1')

	arguments.append('-particles')

	arguments.append('-resampleNEFF')


	for numObs in numObservations:
		targ1 = arguments[:]
		targ1.append('-numObservations')
		targ1.append(f'{numObs}')


		for p in numParticles:
			targ2 = targ1[:]
			targ2.append('-numParticles')
			targ2.append(f'{p}')


		# 	for w in noises:
		# 		targ3 = targ2[:]
		# 		targ3.append('-noise')
		# 		targ3.append(f'{w} {w} {w} {w}')


		# 		for pi in range(len(percents)):

		# 			targ4 = targ3[:]
					
		# 			# Case Resample NEFF
		# 			if pi == 0:
		# 				targ4.append('-resample')
		# 				targ4.append(f'{percents[pi]}')
		# 				targ4.append('-resampleNEFF')
		# 			else:
		# 				targ4.append('-resample')
		# 				targ4.append(f'{percents[pi]}')


		# 			expFolder = f'Testing-2D-RandPos-States{numObs}-JustTargetAgent-JTM-PFE-N{p}-{percents[pi]*100:.2f}%Resample-NoiseDiv{w}-Resolution-5'
		# 			targ4.append('-resultsFolder')
		# 			targ4.append(f'{expFolder}')
			
		
		# 			#Add this to our list of processes to spawn
		# 			spawn_list.append(targ4)


			expFolder = f'Testing-2D-RandPos-States{numObs}-JustTargetAgent-Resolution-5'
			targ2.append('-resultsFolder')
			targ2.append(f'{expFolder}')
			
			
			#Add this to our list of processes to spawn
			spawn_list.append(targ2)


	print(f'Created a spawn list with {len(spawn_list)} processes in it.')

	for each in spawn_list:
		print(" ".join(each))


	'''
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

	'''

	print('All spawns finished.')


