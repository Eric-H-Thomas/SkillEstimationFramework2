import subprocess
import time
import argparse


parser = argparse.ArgumentParser(description='C')
parser.add_argument("-resultsFolder", dest = "resultsFolder", help = "Enter the name of the folder to store the experiments in", type = str, default = "Results")
parser.add_argument("-domain", dest = "domain", help = "Specify which domain to use (1d/2d)", type = str, default = "1d")
parser.add_argument("-mode", dest = "mode", help = "Specify in which mode to use domain 2d (normal|rand_pos|rand_v)", type = str, default = "normal")
parser.add_argument("-iters", dest = "iters", help = "Enter the number of iterations to perform", type = int, default = 200)
parser.add_argument("-numObservations", dest = "numObservations", help = "Enter the number of observations to use for each experiment.", type = int, default = 100)
parser.add_argument("-seeds", dest = "seeds", help = "Main seeds for exps", nargs='+', default = [10])

args = parser.parse_args()


# USAGE EXAMPLE: python testMaster.py -resultsFolder tttt -domain 2d -mode rand_pos -iters 1 -numObservations 2 -seeds 10 20 30


arguments = []

# arguments.append('python')
# arguments.append('testing.py')


# python runExp.py -resultsFolder Experiments-RandomAgents-States1k-2D -domain 2d -iters 150 -numObservations 1000 -seed 10

arguments.append('python')
arguments.append('runExp.py')

arguments.append('-resultsFolder')
arguments.append(args.resultsFolder)


arguments.append('-domain')
arguments.append(args.domain)

arguments.append('-mode')
arguments.append(args.mode)


arguments.append('-iters')
arguments.append(f"{args.iters}")

arguments.append('-numObservations')
arguments.append(f"{args.numObservations}")

arguments.append('-seed')



for i in args.seeds:
	print(f"Process: {i}")

	targ = arguments[:]
	# targ.append('-i')
	targ.append(f"{i}")
	spawn = subprocess.Popen(targ)

# time.sleep(5)

