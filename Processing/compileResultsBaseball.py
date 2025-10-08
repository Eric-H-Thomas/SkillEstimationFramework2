import os, argparse
from pathlib import Path
import shutil


if __name__ == '__main__':

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Compile the result files for the given experiments.')
	parser.add_argument("-folderContains", dest = "folderContains", help = "Name of folder containing the results of the experiments", type = str, default = "ForPaper-Experiment")
	args = parser.parse_args()


	mainFolder = f"Experiments{os.sep}baseball-multi{os.sep}"

	saveAtFolder = f"{mainFolder}CompiledResults{os.sep}results{os.sep}"

	for f in [f"{mainFolder}CompiledResults{os.sep}",saveAtFolder]:
		if not Path(f).is_dir():
			os.mkdir(f)

	existingResultFiles = os.listdir(saveAtFolder)
	

	# From Experiments folder
	folders = os.listdir(mainFolder)


	expFolders = []

	# Filter experiments
	# Assumes folders with actual experiments have "Experiment" in their name
	# To ignore testing folders
	for f in folders:
		if args.folderContains in f:
			expFolders.append(f)


	print("Copying files...")
	
	for f in expFolders:

		# Load info from folder
		subfolders = os.listdir(f"{mainFolder}{f}{os.sep}results")

		# Copy each result file to CompiledResults folder (ensure no duplicates)
		for sf in subfolders:

			# if result file not already present on folder with compiled result files
			# And it's an actual results file (to skip backup folder if present)
			if sf not in existingResultFiles and ".results" in sf:
				
				print(f"\tCopying file: {sf}")
				
				fromFile = f"{mainFolder}{f}{os.sep}results{os.sep}{sf}" 
				toFile = f"{saveAtFolder}{sf}" 
				
				shutil.copy(fromFile,toFile)


	print("Done copying files.")




