
import os, argparse, time

import threading 
lock = threading.Lock()


if __name__ == "__main__": 

	# Get arguments from command line
	parser = argparse.ArgumentParser(description='Make instructions to run experiments.')
	parser.add_argument('-i','-i', help='', default=1)
	args = parser.parse_args()


	# print("TESTING")


	ok = False

	# while not ok:

	# lock.acquire()


	print('waiting for a lock, i:', args.i)

	with lock:
		print ("Lock Acquired")
		time.sleep(10)  

		

	# try:
	# 	print('Acquired a lock, i:', args.i)
	# 	time.sleep(5)
	# 	ok = True
	# 	print('Released a lock, i:', args.i)
	# 	lock.release()

	# except:
	# 	print("Waiting for a lock")




	'''
	for i in range(20):
		
		folder = f"This-{i}"

		if not os.path.exists(folder):

			# lock.acquire()
			
			print(f"Making folder {folder}")

			try:
				os.mkdir(folder)	
			except FileExistsError as e:
				# print("!!! Folder already present. -- ", e)
				pass


			time.sleep(3)

			# lock.release()	

		else:
			print("Folder already present.")
	'''



	# ok = False

	# while not ok:

	# 	try: 

	# 		if not os.path.exists(folder):

	# 			# lock.acquire()
				
	# 			print("Making folder.")
	# 			os.mkdir(folder)	

	# 			# lock.release()

	# 		else:
	# 			print("Folder already present.")

	# 		ok = True

	# 	except:
	# 		print("Waiting...")

	# 	finally:
	# 		print("Done with initial setup.")



