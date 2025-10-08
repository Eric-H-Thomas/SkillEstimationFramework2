import os,sys
from datetime import datetime

if __name__ == '__main__':
	

	domain = "hockey-multi"

	tempInfo = datetime.now()


	folder = f"Experiments{os.sep}{domain}{os.sep}{tempInfo.year}-{tempInfo.month}-{tempInfo.day}{os.sep}Data{os.sep}"

	
	folders = ["Experiments",f"Experiments{os.sep}{domain}{os.sep}",f"Experiments{os.sep}{domain}{os.sep}{tempInfo.year}-{tempInfo.month}-{tempInfo.day}",folder,f"{folder}{os.sep}Plots{os.sep}"]

	for each in folders:
		if not os.path.exists(each):
			os.mkdir(each)
