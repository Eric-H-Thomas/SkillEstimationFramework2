import os
import glob


if __name__ == "__main__":

	#List all of the folders in the results folder
	expFolder = f"Experiments{os.sep}baseball"

	subfolders = [f.path for f in os.scandir(expFolder) if f.is_dir()]
				

	print(f"Found {len(subfolders)} subfolders")

	finished = 0
	more = 0
	#Let's see if they are done

	for sf in subfolders:
		# Existence of the file indicates that the experiment finished successfully 
		
		statusFiles = glob.glob(f"{sf}{os.sep}status{os.sep}*DONE.txt")

		if len(statusFiles) > 0:
			if len(statusFiles) == 1:
				finished += 1
			else:
				print('More than one done file in ', sf)
				more += 1


	print(f"We found {finished} finished experiments. (And {more} with more files)")