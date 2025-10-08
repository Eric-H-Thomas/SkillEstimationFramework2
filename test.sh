#!/bin/bash

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10240M   # memory per CPU core
#SBATCH -J "mpiTest"   # job name
#SBATCH --mail-user=archibald@cs.byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
mpirun -n 2 python3 runExpBaseball.py -ids 547973 -types SL
