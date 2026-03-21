#!/bin/bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -o test.out
#SBATCH -e test.error
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=ssem@wpi.edu
#SBATCH --mem=128G
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -C A100|V100
export OMP_NUM_THREADS=8
mpirun -np 2 gpaw python defect_tio2.py
