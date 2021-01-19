#!/bin/sh
#
#SBATCH --cpu-per-mem=16g
#

srun python3 train.py