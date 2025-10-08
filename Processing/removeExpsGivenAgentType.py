import os,sys

import numpy as np
import re
import code

import pickle, json
import argparse, datetime

from utilsDarts import *

import shutil


if __name__ == "__main__":

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder that contains rfs", type = str, default = "testing")
	parser.add_argument("-type", dest = "type", help = "Specify which agent type to remove exps of.", type = str, default = "Bounded")
	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]

	getFrom = f"{args.resultsFolder}{os.sep}ProcessedResultsFiles{os.sep}"
	resultFiles = os.listdir(getFrom)

	try:
		resultFiles.remove(".DS_Store")
	except:
		pass

	saveTo = f"{args.resultsFolder}{os.sep}RemovedExps{os.sep}"
	makeFolder3(saveTo)


	print("Going through result files...")


	total = 0

	# Collate results for the methods
	for i in range(len(resultFiles)):

		rf = resultFiles[i]

		if args.type in rf:

			print(f"Removing -> {rf}")
			total += 1
		
			# Backup exp
			shutil.copy(f"{getFrom}{rf}",saveTo)

			# Remove exp
			os.remove(f"{getFrom}{rf}")


	print(f"\nRemoved a total of {total} exps ({args.type} agents).")
	print(f"DONE.")
	# code.interact("...", local=dict(globals(), **locals()))


