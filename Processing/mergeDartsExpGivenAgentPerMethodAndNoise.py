import argparse
import os
import code
import json, pickle
import time

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


	times = []

	for rf in result_files: 

		total_num_exps += 1

		# For each file, get the information from it
		print ('('+str(total_num_exps)+'/'+str(len(result_files))+') - RF :', rf)

		t1 = time.perf_counter()

		trf = rf.split(".results")[0]
		splitted = trf.split("_")
		# print(splitted)

		#i = splitted[-3]
		method = ""

		if len(splitted) == 14:
			seedNum = splitted[-6]
			method += splitted[-3]
		else:
			seedNum = splitted[-5]


		splittedAgain = splitted[-1].split("-")
		# print(splittedAgain)

		if method != "":
			method += "_" + splitted[-2]
		else:
			method = splittedAgain[0]

		counter = splittedAgain[1]
		#print(splittedAgain)

		# Case negative rho
		if len(splittedAgain) == 4:
			temp = splittedAgain[-2].split("|")
			# times -1 to get back negative
			removed = temp.pop(-1)

			try:
				temp.append(removed + str(-1*float(splittedAgain[-1])))
			except:
				temp.append(removed + str(-1*float(splittedAgain[-1].split("|")[0])))
			splittedAgent = temp
		else:
			splittedAgent = splittedAgain[-1].split("|")

		# print(splittedAgent)

		try:
			agentType, x1, x2, rho = splittedAgent
		except: 
			agentType, x1, x2, rho, p = splittedAgent


		'''
		print(seedNum)
		print(method)
		print(counter)
		print(agentType)
		print(x1)
		print(x2)
		print(rho)
		code.interact("...", local=dict(globals(), **locals()))
		'''


		# if agentType not in info: 
		# 	info[agentType] = {}


		# if x1 not in info[agentType]:
		# 	info[agentType][x1] = {}

		# if x2 not in info[agentType][x1]:
		# 	info[agentType][x1][x2] = {}

		# if rho not in info[agentType][x1][x2]:
		# 	info[agentType][x1][x2][rho] = {}


		# if seedNum not in info[agentType][x1][x2][rho]:
		# 	info[agentType][x1][x2][rho][seedNum] = {}

		# if counter not in info[agentType][x1][x2][rho][seedNum]:
		# 	info[agentType][x1][x2][rho][seedNum][counter] = {}


		# with open(args.resultsFolder + os.sep + "results" + os.sep + rf) as infile:
		# 	results = json.load(infile)

		with open(args.resultsFolder + os.sep + "results" + os.sep + rf,"rb") as infile:
			results = pickle.load(infile)


		# Focusing on method keys
		# info[agentType][x1][x2][rho][seedNum][counter].update(results)


		# if x1 == "X100.0" and x2 == "X100.0" and rho == "R0.75" and seedNum == "73349" and counter == "14":
		# 	code.interact("...", local=dict(globals(), **locals()))


		fileName = f"OnlineExp_{seedNum}_{counter}_{agentType}_{x1}_{x2}_{rho}.results"

		# Load existing info (if any)
		try:
			# with open(saveAt+fileName,"r") as infile:
			# 	temp = json.load(infile)
			with open(saveAt+fileName,"rb") as infile:
				temp = pickle.load(infile)
		except:
			temp = {}

		
		# Update
		if temp != {}:
			results["namesEstimators"] += temp["namesEstimators"]

		temp.update(results)


		# Save updated info to file
		# with open(saveAt+fileName,"w") as outfile:
		# 	json.dump(temp,outfile)
		with open(saveAt+fileName,"wb") as outfile:
			pickle.dump(temp,outfile)


		# Delete from memory
		# del info[agentType][x1][x2][rho][seedNum][counter]
		del results
		del temp


		totalTime = time.perf_counter() - t1
		print(f"{totalTime:.2f}")
		times.append(totalTime)


	print(f"AVG time per file: {sum(times)/len(times):.2f}")

	'''
	# # Save to new/merged rf
	for agentType in info:
		for x1 in info[agentType]:
			for x2 in info[agentType][x1]:
				for rho in info[agentType][x1][x2]:
					for seedNum in info[agentType][x1][x2][rho]:
						for counter in info[agentType][x1][x2][rho][seedNum]:
							fileName = f"OnlineExp_{seedNum}_{counter}_{agentType}_{x1}_{x2}_{rho}.results"
				
							with open(saveAt+fileName,"w") as outfile:
								json.dump(info[agentType][x1][x2][rho][seedNum][counter],outfile)
	'''

	# code.interact("...", local=dict(globals(), **locals()))




		