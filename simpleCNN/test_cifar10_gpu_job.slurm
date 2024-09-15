#!/bin/bash
#SBATCH --job-name=cifar10_test
#SBATCH --output=cifar10_test.out
#SBATCH --error=cifar10_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1               # Request 1 GPU
#SBATCH --time=00:10:00        # Set an appropriate time limit
#SBATCH --partition=gpu        # Adjust to the correct partition with GPU access

# Load modules
module load python pytorch  # Adjust module names as needed on Leonardo

# Run the PyTorch script
srun python cifar10_cnn.py
