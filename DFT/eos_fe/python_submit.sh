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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun --mca pml ob1 --mca mtl ^ofi --mca btl self,sm,tcp -np $SLURM_NTASKS gpaw python eos_gpaw.py
