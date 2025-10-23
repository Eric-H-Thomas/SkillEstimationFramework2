import os
import code
import numpy as np
import matplotlib.pyplot as plt

from importlib.machinery import SourceFileLoader


scriptPath = os.path.realpath(__file__)

mainFolderName = scriptPath.split("Testing")[0]+f"Environments{os.sep}Darts{os.sep}RandomDarts{os.sep}"
module2D = SourceFileLoader("2d",mainFolderName+"two_d_darts.py").load_module()
module2D_Multi = SourceFileLoader("2d_multi",mainFolderName+"two_d_darts_multi.py").load_module()


folder =  f"Testing{os.sep}TestingScores"
if not os.path.isdir(folder):
	os.mkdir(folder)


def main():

	rng = np.random.default_rng(1000)

	mode = "rand_pos"
	
	numStates = 20


	module = module2D_Multi

	states = module.generate_random_states(rng,numStates,mode)


	resolution = 5.0

	XS = np.arange(-170.0,171.0,resolution)
	YS = np.arange(-170.0,171.0,resolution)

	XXS,YYS = np.meshgrid(XS,YS,indexing="ij")
	tempXYS = np.vstack([XXS.ravel(),YYS.ravel()])

	XYS = np.dstack(tempXYS)[0]

	sizeXYS = int(np.sqrt(len(XYS)))



	for s in states:
		print("State: ",s)

		for target in XYS:

			# fig = plt.figure()
			# ax = plt.gca()

			label = ""


			v1 = module.get_reward_for_action(s,target)
			v2 = module.npscore(s,target[0],target[1])

			print(f"\tTarget: {target} | V1: {v1} | V2: {v2}")


			if v1 != v2:
				label = "-NotMatching"
				# module.get_reward_for_action(s,target,True)
				code.interact("...", local=dict(globals(), **locals())) 


			'''
			module.draw_board(ax)
			module.label_regions(s,color="black")

			plt.scatter(target[0],target[1],label="Target")

			plt.title(f"getV(): {v1} | npscore(): {v2}")

			ax.set_xticks([])
			ax.set_yticks([])

			plt.legend()

			# plt.show()
			plt.savefig(f"{folder}{os.sep}{mode}-resolution-{resolution}-target-{target}{label}.png")


			plt.clf()	
			plt.close()
			'''

			# code.interact("...", local=dict(globals(), **locals())) 



if __name__ == '__main__':
	main()