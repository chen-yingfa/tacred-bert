#!/bin/sh
#
#SBATCH --cpu-per-mem=16g
#SBTACH --gpus=1
#

srun python3 train.py