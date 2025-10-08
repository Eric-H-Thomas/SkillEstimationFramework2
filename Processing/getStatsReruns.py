import os,sys

import numpy as np
import code

import pickle
import argparse, datetime

from utilsDarts import *


if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='')
	# parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "2d")
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments)", type = str, default = "testing")
	# parser.add_argument("-rerunStart", dest = "rerunStart", help = "When did we started the rerun? (Format YYYY-M-D)", type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date())
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	
	statusFiles = os.listdir(args.resultsFolder + os.path.sep + "status")	
	resultFiles = os.listdir(args.resultsFolder + os.path.sep + "results")

	try:
		statusFiles.remove(".DS_Store")
	except:
		pass


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass

	'''
	filesByType = {}
	filesByType["fullExp"] = []
	filesByType["rerun"] = []


	print("Filtering status files (DONE vs RERUN)...")

	# Divide between done and rerun status files
	for sf in statusFiles: 

		# print ('StatusFile: ', sf)

		# Sample Done File: OnlineExp_70-438_23_05_02_11_27_0_188317-29-AgentBoundedAgent-X70.4377-L0.002910744216446768-DONE
		# Sample Status file: OnlineExp_70-438_23_05_02_11_27_0_188317-29-AgentBoundedAgent-X70.4377-L0.002910744216446768-DONE-RERUN-59
	
		splitted = sf.split("-DONE")
		tag = splitted[0]

		if "-RERUN" in splitted[1]:
			typeFile = "rerun"
		else:
			typeFile = "fullExp"


		filesByType[typeFile].append(tag)
	
	
	# Verify that: 
	# Each experiment have a respective rerun file 
	# OR
	# Full experiment conducted on rerun mode since prev exp not found (no rerun file needed)

	expsWithNoRerunFiles = []
	fullExpsAfter = []

	print("Verification proces...")

	for eachTag in filesByType["fullExp"]:

		print(eachTag)

		splitted = eachTag.split("-Agent")

		agent = splitted[1]

		# Find date of experiment
		splitted2 = splitted[0].split("_")
		# [0] = OnlineExp
		# [1] = xskill (#-##)
		# [2] = Y
		# [3] = M
		# [4] = D
		# [5] = M
		# [6] = S
		# [7] = iter #
		# [-1] / [8] = seedNum-counter
		#seedNumStr = splitted2[-1].split("-")[0]

		# Ignoring time since file name only has minutes & secs (not hour)
		dateInfo = splitted2[2:5]
		expDate = datetime.datetime(*map(int,dateInfo)).date()

		###############################################################		
		### CASE: Each experiment have a respective rerun file 
		###############################################################

		found = False

		# Check if exp has corresponding rerun file
		for otherTagRerun in filesByType["rerun"]:

			# Rerun keeps same tags as before - so can just compare
			if eachTag == otherTagRerun:
				found = True
				break
		###############################################################


		###############################################################
		### CASE: Full experiment conducted on rerun mode since 
		# 		  prev exp not found (no rerun file needed)
		###############################################################

		if not found:
			# Experiment conducted afterwards?
			if expDate >= args.rerunStart:
				fullExpsAfter.append(eachTag)
				found = True

		###############################################################


		if not found:
			expsWithNoRerunFiles.append(eachTag)


	print(f"There are {len(expsWithNoRerunFiles)} experiments without their corresponding rerun file.")
	print(f"There are {len(fullExpsAfter)} experiments conducted afterwards (full exps while on rerun mode).")


	###############################################################
	# SAVE INFO TO FILE
	###############################################################

	outfile = open(f"{args.resultsFolder}{os.sep}statusFilesInfo.txt","w")
	
	print(f"There are {len(expsWithNoRerunFiles)} experiments without their corresponding rerun file.",file=outfile)
	print(f"There are {len(fullExpsAfter)} experiments conducted afterwards (full exps while on rerun mode).",file=outfile)
	print("\n\n",file=outfile)

	print("Experiments without respective rerun file: ",file=outfile)
	for each in expsWithNoRerunFiles:
		print(f"\t{each}",file=outfile)
	print(file=outfile)

	print("Experiments conducted afterwards (full exps while on rerun mode): ",file=outfile)
	for each in fullExpsAfter:
		print(f"\t{each}",file=outfile)
	print(file=outfile)

	###############################################################

	'''

	allFiles = os.listdir(args.resultsFolder)

	rerunStartTimes = []

	try:

		for f in allFiles:

			if "rerunStartTime" in f:
				try:
					infile = open(f"{args.resultsFolder}{os.sep}{f}",'r')
					fileInfo = infile.readline()
					infile.close()
					fileInfo = fileInfo.strip()
					rerunStartTimes.append(fileInfo)
				except Exception as e:
					# print(e)
					continue

		rerunStart = datetime.datetime.strptime(min(rerunStartTimes), '%Y-%m-%d %H:%M:%S.%f')

	except Exception as e:
		# print(e)
		print("Rerun mode has not been performed yet. Hence no rerunStart info.")


	allExpsCount = 1
	skippedError = []

	filesWithoutLastEdited = []
	expsPendingRerun = []

	print("Going through result files...")

	# Collate results for the methods
	for rf in resultFiles: 

		# For each file, get the information from it
		# print('\n('+str(allExpsCount)+'/'+str(len(resultFiles))+') - RF : ', rf)

		allExpsCount += 1

		try:

			with open(args.resultsFolder+os.sep+"results"+os.sep+rf,"rb") as infile:
				results = pickle.load(infile)
		except:
			skippedError.append(rf)
			continue


		try:
			lastEdited = results["lastEdited"]
			lastEditedDate = datetime.datetime.strptime(lastEdited, '%Y-%m-%d %H:%M:%S.%f')
		except: 
			filesWithoutLastEdited.append(rf)


		try:
			if lastEditedDate < rerunStart:
				expsPendingRerun.append(rf)
		except:
			pass


	print(f"\nThere are {len(filesWithoutLastEdited)} experiments without the 'lastEdited' info.\n   Either experiment in process or need to rerun.")
	print(f"\nThere are {len(expsPendingRerun)} experiments pending to rerun.\n   Based on lastEditedDate < rerunStart.")
	print(f"\nThere are {len(skippedError)} experiments that were skipped bc error occured when loading file.\n")


	###############################################################
	# SAVE INFO TO FILE
	###############################################################

	outfile = open(f"{args.resultsFolder}{os.sep}rerunInfo.txt","w")
	
	print(f"There are {len(filesWithoutLastEdited)} experiments without the 'lastEdited' info.\n   Either experiment in process or need to rerun.",file=outfile)
	print(f"\nThere are {len(expsPendingRerun)} experiments pending to rerun.\n   Based on lastEditedDate < rerunStart.",file=outfile)
	print(f"\nThere are {len(skippedError)} experiments that were skipped bc error occured when loading file.\n",file=outfile)
	print("\n\n",file=outfile)

	lists = [filesWithoutLastEdited,expsPendingRerun,skippedError]
	texts = ["Experiments without the 'lastEdited' info: ",
			"Experiments pending to rerun: ",
			"Experiments skipped bc of error when loading file: "]

	for each in range(len(lists)):
		print(texts[each],file=outfile)
		for each in lists[each]:
			print(f"\t{each}",file=outfile)
		print(file=outfile)
	
	###############################################################


	# code.interact("...", local=dict(globals(), **locals()))


