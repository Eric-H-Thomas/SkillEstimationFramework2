import argparse

if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Make instructions to run experiments.')
	parser.add_argument('-ids','-ids', nargs='+', help='List of pitcher IDs to use',default = [])
	parser.add_argument('-types','-types', nargs='+', help='List of pitch types to use',default = [])
	args = parser.parse_args()


	instructions = []

	for pitcherID in args.ids:
		typesLabel = "-" + "-".join(args.types)
		typesInst = " ".join(args.types)
		expFolder = f"Experiment-PitcherID-{pitcherID}-PitchTypes{typesLabel}"
		instruction = f"python3 runExp.py -domain baseball -ids {pitcherID} -types {typesInst} -resultsFolder {expFolder}"
		instructions.append(instruction)


	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	print("INSTRUCTIONS")
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	for instruction in instructions:
		print(f"\n{instruction}")
	print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")