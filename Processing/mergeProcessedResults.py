import os,sys

import numpy as np
import re
import code

import pickle, json
import argparse, datetime

from utilsDarts import *


if __name__ == "__main__":

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("-prfsFolderAll", dest = "prfsFolderAll", help = "Name of folder containing the processed results of the experiments)", type = str, default = "testing")
	parser.add_argument("-prfsFolderSome", dest = "prfsFolderSome", help = "Name of folder containing the processed results of the experiments)", type = str, default = "testing")
	parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d/sequentialDarts/etc)", type = str, default = "2d")
	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.prfsFolderAll[-1] == os.path.sep:
		args.prfsFolderAll = args.prfsFolderAll[:-1]

	# Prevent error with "/"
	if args.prfsFolderSome[-1] == os.path.sep:
		args.prfsFolderSome = args.prfsFolderSome[:-1]


	saveAt = f"Experiments{os.sep}{args.domain}{os.sep}MergedResults{os.sep}"
	makeFolder3(saveAt)

	saveAt = f"{saveAt}{os.sep}ProcessedResultsFiles{os.sep}"
	makeFolder3(saveAt)

	
	resultFilesAll = os.listdir(args.prfsFolderAll + os.path.sep + "ProcessedResultsFiles")
	resultFilesSome = os.listdir(args.prfsFolderSome + os.path.sep + "ProcessedResultsFiles")


	try:
		resultFilesAll.remove(".DS_Store")
	except:
		pass

	try:
		resultFilesSome.remove(".DS_Store")
	except:
		pass


	print("Going through result files...")

	notFound = []

	# Collate results for the methods
	for rf in resultFilesAll:

		# Load rf all estimators info
		with open(args.prfsFolderAll+os.sep+"ProcessedResultsFiles"+os.sep+rf,"rb") as infile:
			resultsAll = pickle.load(infile)

		# Filename for processed result files is the same across exps
		# Example: resultsDictInfo-Flip-X2.6377-P0.43126786861752375

		# Determine if corresponding file present in resultFilesSome
		try:
			index = resultFilesSome.index(rf)
			found = True
		except:
			found = False


		if found:
			print(f"Found respective file for: {rf}")


			# Load rf some estimators info
			with open(args.prfsFolderSome+os.sep+"ProcessedResultsFiles"+os.sep+rf,"rb") as infile:
				resultsSome = pickle.load(infile)
			
			# code.interact("...", local=dict(globals(), **locals()))


			key = list(resultsAll.keys())[0]
			
			# Merge info
			resultsAll[key]["plot_y"]["OptimalTargets"] = resultsSome[key]["plot_y"]["OptimalTargets"] 
			resultsAll[key]["estimates"]["OptimalTargets"] = resultsSome[key]["estimates"]["OptimalTargets"] 


			# Save to new file/folder
			with open(f"{saveAt}{rf}",'wb') as outfile:
				pickle.dump(resultsAll,outfile)

		
		else:
			# code.interact("...", local=dict(globals(), **locals()))
			notFound.append(rf)


	print(f"\nDONE.\nDidn't find respective results file for {len(notFound)} exps.")

	for each in notFound:
		print(f"\t{each}")

	code.interact("...", local=dict(globals(), **locals()))


