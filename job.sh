#!/bin/sh
#
#SBATCH --mem-per-cpu=16g
#SBTACH --gpus=1
#

srun python3 train.py