#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 90
#SBATCH -t 24:00:00
#SBATCH -o test.out
#SBATCH -e test.error
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ssem@wpi.edu
#SBATCH --gres=gpu

python3 nvt_aimd.py
