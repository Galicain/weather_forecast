#!/bin/bash
#SBATCH --job-name=lstm_benchmark_10_days
#SBATCH --output=lstm_10days%j.out
#SBATCH --error=lstm_10days%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

module load cuda

# Activer ton environnement Python

python download_era5__data.py
