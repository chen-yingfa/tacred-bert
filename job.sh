#!/bin/sh
#
#SBATCH -G 1
#SBATCH --mem-per-cpu=16g
#

srun python3 train.py