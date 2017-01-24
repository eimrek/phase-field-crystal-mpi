#!/bin/bash


#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH --mem=20000
#SBATCH --time=00:30:00

module purge
module load gcc-4.8.1
module load fftw-3.3.4
module load openmpi-1.8.4

mpirun ./bin/pfc

