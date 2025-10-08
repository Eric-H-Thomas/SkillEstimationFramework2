import os
import argparse
import shutil
import json
from pathlib import Path

if __name__ == '__main__':

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Compile the result files for the given experiments.')
	parser.add_argument('-file','-file', help='Name of the file that contains the info of the experiments to compile results files of',default = "")
	parser.add_argument("-folderContains", dest = "folderContains", help = "Name of folder containing the results of the experiments", type = str, default = "ForPaper-Experiment")
	args = parser.parse_args()

	if args.file == "":
	    print("Need to provide file with info about the experiments to be compiled.")
	    exit()
	else:
	    with open(args.file,"r") as infile:
	        info = json.load(infile)
	        expsInfo = info['missingExps']


	mainFolder = f"Experiments{os.sep}baseball{os.sep}"

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
	for each in expsInfo:
		pid = each[0]
		pt = each[1]

		found = False
		
		for f in folders:
			if args.folderContains in f and str(pid) in f and pt in f:
				expFolders.append(f)
				found = True

		if not found:
			print(f"Didn't find an experiment folder for pitcherID {pid} and pitchType {pt}")


	print("Copying files...")

	total = 0

	newFiles = []
	
	for f in expFolders:

		# Load info from folder
		subfolders = os.listdir(f"{mainFolder}{f}{os.sep}results")

		# Copy each result file to CompiledResults folder (ensure no duplicates)
		for sf in subfolders:

			# if result file not already present on folder with compiled result files
			# And it's an actual results file (to skip backup folder if present)
			if sf not in existingResultFiles and sf not in newFiles and ".results" in sf:
				
				print(f"\tCopying file: {sf}")
				
				fromFile = f"{mainFolder}{f}{os.sep}results{os.sep}{sf}" 
				toFile = f"{saveAtFolder}{sf}" 
				
				shutil.copy(fromFile,toFile)

				newFiles.append(sf)

				total += 1


	print(f"Done. Copied a total of {total} results files.")




