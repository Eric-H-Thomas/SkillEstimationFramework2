
import argparse, os, pickle, code
from utilsDarts import *

if __name__ == "__main__":
	
	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Processing results')
	parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Name of folder containing the results of the experiments", type = str, default = "testing")	
	args = parser.parse_args()


	# Prevent error with "/"
	if args.resultsFolder[-1] == os.path.sep:
		args.resultsFolder = args.resultsFolder[:-1]

	rdFile = args.resultsFolder + os.sep + "ProcessedResultsFiles" + os.sep + "resultsDictInfo"
	oiFile = args.resultsFolder + os.sep + "otherInfo" 


	# FOR TESTING
	# processedRFs = ["OnlineExp_3-314_23_05_01_08_16_8_349052-243-AgentFlipAgent-X4.3093-P0.5473776942981432.results"]
	# processedRFsAgentNames = ["Flip-X4.3093-P0.5473776942981432"]


	'''
	with open(oiFile,"rb") as infile:
		otherInfo = pickle.load(infile)


	remove = ["Tricker-X18.3947-Eps0.05714832854963151", "Flip-X4.3093-P0.5473776942981432"]

	print(len(otherInfo["processedRFsAgentNames"]))

	for each in remove:

		#otherInfo["processedRFsAgentNames"].remove(each)
		#print("\tRemoved RF agent name from list.")

		splitted = each.split("-")

		for rf in otherInfo["processedRFsAgentNames"]:
			if splitted[0] in rf and splitted[1] in rf and splitted[2] in rf:
				print("\tRemoved RF name from list.")
				otherInfo["processedRFsAgentNames"].remove(rf)


	with open(oiFile,"wb") as outfile:
		pickle.dump(otherInfo,outfile)	

	exit()
	'''


	with open(oiFile,"rb") as file:
		otherInfo = pickle.load(file)

		processedRFs = otherInfo["processedRFs"]
		processedRFsAgentNames = otherInfo["processedRFsAgentNames"]

		methods = otherInfo["methods"]


	badExps = []

	resultsDict = {}

	for i in range(len(processedRFsAgentNames)):

		a = processedRFsAgentNames[i]
		print(f"\n({i}/{len(processedRFsAgentNames)}) - Agent: ", a)

		# Load processed info		
		resultsDict[a] = loadProcessedInfo(rdFile,a)

		mxi = 0

		try:

			for m in methods:

				if "BM" in m:
					tempM, beta, tt = getInfoBM(m)
					resultsDict[a]["plot_y"][tt][tempM][beta][mxi] = resultsDict[a]["plot_y"][tt][tempM][beta][mxi] / float(resultsDict[a]["num_exps"]) #MSError

				else:
					resultsDict[a]["plot_y"][m][mxi] = resultsDict[a]["plot_y"][m][mxi] / float(resultsDict[a]["num_exps"]) #MSError
		except:
			badExps.append(a)

		del resultsDict[a]

	code.interact(f"After checking exps...total of {len(badExps)} bad exps...", local=dict(globals(), **locals()))


	removeFlag = input("Proceed to remove??? (y/n)")
	print("\n\n")

	if removeFlag == "y":

		for eachBadExp in badExps:

			print(eachBadExp)
			
			processedRFsAgentNames.remove(eachBadExp)
			print("\tRemoved RF agent name from list.")

			splitted = eachBadExp.split("-")

			for each in processedRFs:

				if "Target" in eachBadExp:
					if splitted[0] in each and splitted[1] in each:
						print("\tRemoved RF name from list.")
						processedRFs.remove(each)
				else:
					if splitted[0] in each and splitted[1] in each and splitted[2] in each:
						print("\tRemoved RF name from list.")
						processedRFs.remove(each)

			# Delete file
			toRemove = f"{args.resultsFolder}{os.sep}ProcessedResultsFiles{os.sep}resultsDictInfo-{eachBadExp}"
			os.remove(toRemove)
			print("\tRemoved RF from folder.")


		print("\nUpdating otherInfo pickle file...")
		
		# Update other info file
		with open(oiFile,"rb") as infile:
			otherInfo = pickle.load(infile)

		otherInfo["processedRFs"] = processedRFs
		otherInfo["processedRFsAgentNames"] = processedRFsAgentNames
		
		with open(oiFile,"wb") as outfile:
			pickle.dump(otherInfo,outfile)	

		print("Done.")

