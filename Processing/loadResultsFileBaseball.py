import argparse
import os,sys
import code
import json


if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Validate experiments.')
	parser.add_argument('-file','-file', help='Name of the file that contains the IDs of the pitchers to use',default = "ids.txt")	
	args = parser.parse_args()

	with open(args.file) as infile:
		results = json.load(infile)

	print(results.keys())
	print(results["numObservations"])
	print(len(results["OR-66"]))

	code.interact("...", local=dict(globals(), **locals()))
