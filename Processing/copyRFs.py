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
	parser.add_argument("-fromFolder", dest = "fromFolder", help = "Name of folder to copy rfs from.", type = str, default = "testing")
	parser.add_argument("-toFolder", dest = "toFolder", help = "Name of folder to copy files to.", type = str, default = "testing")
	#parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d/sequentialDarts/etc)", type = str, default = "2d")
	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.fromFolder[-1] == os.path.sep:
		args.fromFolder = args.fromFolder[:-1]

	# Prevent error with "/"
	if args.toFolder[-1] == os.path.sep:
		args.toFolder = args.toFolder[:-1]


	saveTo = f"{args.toFolder}{os.sep}"
	makeFolder3(saveTo)

	saveTo = f"{saveTo}{os.sep}results{os.sep}"
	makeFolder3(saveTo)

	getFrom = f"{args.fromFolder}{os.sep}results{os.sep}"
	resultFiles = os.listdir(getFrom)


	try:
		resultFiles.remove(".DS_Store")
	except:
		pass



	print("Going through result files...")

	notFound = []

	# Collate results for the methods
	for i in range(len(resultFiles)):

		rf = resultFiles[i]
		print(f"{i+1}/{len(resultFiles)} -> {rf}")
		
		shutil.copy(f"{getFrom}{rf}",saveTo)


	print(f"\nDONE.")
	# code.interact("...", local=dict(globals(), **locals()))


