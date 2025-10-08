import sys,os
import pickle


if __name__ == '__main__':


	try:
		folder1 = sys.argv[1]
		folder2 = sys.argv[2]
	except Exception as e:
		print("Need to specify the name of the experiments folder to compare data from (2 needed) (located under 'Experiments/hockey-multi/') as command line argument.")
		exit()


	mainFolder = f"Experiments{os.sep}hockey-multi{os.sep}"
	mainFolder1 = f"Experiments{os.sep}hockey-multi{os.sep}{folder1}{os.sep}Data{os.sep}Heatmaps{os.sep}"
	mainFolder2 = f"Experiments{os.sep}hockey-multi{os.sep}{folder2}{os.sep}Data{os.sep}Heatmaps{os.sep}"



	files1 = os.listdir(f"{mainFolder1}")
	files2 = os.listdir(f"{mainFolder2}")

	try:
		files1.remove(".DS_Store")
		files2.remove(".DS_Store")
	except:
		pass


	writeTo = open(f"{mainFolder}dataComparison-folders-{folder1}-{folder2}.txt","w")
	print(f"{'Agent':20}\t{folder1:^15}\t{folder2:^15}\t{'Difference':^15}",file=writeTo)


	seenFiles = []
	notFoundFiles = []


	for prevFile in files1:


		splitted = prevFile.split("_")
		# print(splitted)

		playerID = splitted[3]
		typeShot = splitted[-1].split(".")[0]

		agent = f"{playerID}-{typeShot}"


	
		# Find respective file in other folder, if present

		found = False

		for eachFile in files2:
			if playerID in eachFile and typeShot in eachFile:
				found = True
				break


		if found:
			seenFiles.append(eachFile)

			# print(agent)
			# print(prevFile)
			# print(eachFile)

			with open(f"{mainFolder1}{os.sep}{prevFile}",'rb') as handle:
				loadedInfo1 = pickle.load(handle)

			with open(f"{mainFolder2}{os.sep}{eachFile}",'rb') as handle:
				loadedInfo2 = pickle.load(handle)


			prevLen = len(loadedInfo1)
			newLen = len(loadedInfo2)

			diff = newLen - prevLen

			temp = ""

			if diff > 0:
				temp = f"+{diff}"
			elif diff < 0:
				temp = f"{diff}"


			print(f"{agent:20}\t{prevLen:^15}\t{newLen:^15}\t{temp:^15}",file=writeTo)
		else:
			notFoundFiles.append(prevFile)



	for each in notFoundFiles:

		splitted = each.split("_")

		playerID = splitted[3]
		typeShot = splitted[-1].split(".")[0]

		agent = f"{playerID}-{typeShot}"

		with open(f"{mainFolder1}{each}",'rb') as handle:
			loadedInfo = pickle.load(handle)
			prevLen = len(loadedInfo)


		print(f"{agent:20}\t{prevLen:^15}\t{'':^15}\t{'':^15}",file=writeTo)



	for each in files2:

		if each not in seenFiles:

			splitted = each.split("_")

			playerID = splitted[3]
			typeShot = splitted[-1].split(".")[0]

			agent = f"{playerID}-{typeShot}"

			with open(f"{mainFolder2}{os.sep}{each}",'rb') as handle:
				loadedInfo2 = pickle.load(handle)
				newLen = len(loadedInfo2)


			print(f"{agent:20}\t{'':^15}\t{newLen:^15}\t{'':^15}",file=writeTo)


	writeTo.close()

