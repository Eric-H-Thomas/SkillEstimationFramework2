import os,sys
import numpy as np
import code
import pickle
import argparse

from utilsDarts import *


if __name__ == "__main__":

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Merge otherInfo files')
	parser.add_argument("-allFolder", dest = "allFolder", help = "Name of folder containing all rfs.", type = str, default = "testing")
	parser.add_argument("-someFolder", dest = "someFolder", help = "Name of folder containing the subset fo rfs.", type = str, default = "testing")
	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.allFolder[-1] == os.path.sep:
		args.allFolder = args.allFolder[:-1]

	# Prevent error with "/"
	if args.someFolder[-1] == os.path.sep:
		args.someFolder = args.someFolder[:-1]

	getFrom = f"{args.allFolder}{os.sep}ProcessedResultsFiles{os.sep}"
	prfs = os.listdir(getFrom)


	oiFileAll = args.allFolder + os.path.sep + "otherInfo" 
	oiFileSome = args.someFolder + os.path.sep + "otherInfo" 

	with open(oiFileAll,"rb") as file:
		oiFileAllInfo = pickle.load(file)

	with open(oiFileSome,"rb") as file:
		oiFileSomeInfo = pickle.load(file)

	resultFilesAll = oiFileAllInfo["resultFiles"] 
	processedRFsAll = oiFileAllInfo["processedRFs"]
	processedRFsAgentNamesAll = oiFileAllInfo["processedRFsAgentNames"]

	resultFilesSome = oiFileSomeInfo["resultFiles"] 
	processedRFsSome  = oiFileSomeInfo["processedRFs"]
	processedRFsAgentNamesSome  = oiFileSomeInfo["processedRFsAgentNames"]


	# Update oiFileAll - since containing rest of info
	oiFileAllInfo["resultFiles"] = resultFilesAll + resultFilesSome
	oiFileAllInfo["processedRFs"] = processedRFsAll + processedRFsSome
	oiFileAllInfo["processedRFsAgentNames"] = processedRFsAgentNamesAll + processedRFsAgentNamesSome


	print("Going through files...")

	total = 0

	copyProcessedRFsAgentNames = [] + oiFileAllInfo["processedRFsAgentNames"]
	copyProcessedRFs = [] + oiFileAllInfo["processedRFs"]

	# To remove possible duplicates
	copyProcessedRFsAgentNames = list(set(copyProcessedRFsAgentNames))
	copyProcessedRFs = list(set(copyProcessedRFs))

	oiFileAllInfo["processedRFsAgentNames"] = []
	oiFileAllInfo["processedRFs"] = [] 


	# Confirm rf exist (to remove rf prev removed from list)
	for i in range(len(copyProcessedRFsAgentNames)):
		rf = copyProcessedRFsAgentNames[i]

		found = False

		for each in prfs:

			# Remove if prf not found
			if rf in each:
				found = True

		'''
		if not found:
			if rf in oiFileAllInfo["processedRFsAgentNames"]:
				oiFileAllInfo["processedRFsAgentNames"].remove(rf)
				total += 1

			if rf in oiFileAllInfo["processedRFs"]:
				oiFileAllInfo["processedRFs"].remove(rf)
		'''

		if found:
			oiFileAllInfo["processedRFsAgentNames"].append(rf)
			oiFileAllInfo["processedRFs"].append(rf)


	print(f"\nRemoved a total of {total} prfs from list.")
	print(f"DONE.")


	with open(oiFileAll,"wb") as outfile:
		pickle.dump(oiFileAllInfo,outfile)

	code.interact("...", local=dict(globals(), **locals()))
