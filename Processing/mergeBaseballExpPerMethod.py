import argparse
import os
import code
import pickle

if __name__ == "__main__":


	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.sep:
		args.resultsFolder = args.resultsFolder[:-1]
	

	result_files = os.listdir(args.resultsFolder+os.sep+"results")


	if len(result_files) == 0:
		print("No result files present for experiment.")
		exit()


	try:
		result_files.remove("backup")
	except:
		pass

	try:
		result_files.remove(".DS_Store")
	except:
		pass


	saveAt = args.resultsFolder+os.sep+"MergedResults"+os.sep

	#If the folder for the plot(s) doesn't exist already, create it
	if not os.path.exists(saveAt):
		os.mkdir(saveAt)




	info = {}

	methods = ["JEEDS","PFE","PFE_NEFF"]


	# Start processing results
	total_num_exps = 0

	# NOTE: EACH RESULT FILE BELONGS TO A GIVEN PITCHER AND PITCH TYPE
	# EACH COMBINATION WILL BE SEEN ONLY ONCE (MEANING JUST 1 EXP)

	for rf in result_files: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)

		trf = rf.split(".")[0]
		splitted = trf.split("_")

		pitcherID = splitted[1]
		pitchType = splitted[2]

		if len(splitted) == 7:
			method = splitted[3] + "_" + splitted[4]
			chunk = splitted[6]
		else:
			method = splitted[3]
			chunk = splitted[5]

		# print(pitcherID)
		# print(pitchType)
		# print(method)
		# print(chunk)

		if pitcherID not in info:
			info[pitcherID] = {}

		if pitchType not in info[pitcherID]:
			info[pitcherID][pitchType] = {}

		if chunk not in info[pitcherID][pitchType]:
			info[pitcherID][pitchType][chunk] = {}



		with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:
			results = pickle.load(infile)

		# First file seen, just copy everything over
		# if info[pitcherID][pitchType][chunk] == {}:
		
		# Focusing on method keys
		info[pitcherID][pitchType][chunk].update(results)


	# Save to new/merged rf
	for pitcherID in info:
		for pitchType in info[pitcherID]:
			for chunk in info[pitcherID][pitchType]:
				fileName = f"OnlineExp_{pitcherID}_{pitchType}_Chunk_{chunk}.results"
				
				with open(saveAt+fileName,"wb") as outfile:
					pickle.dump(info[pitcherID][pitchType][chunk],outfile)

				# code.interact("...", local=dict(globals(), **locals()))




		