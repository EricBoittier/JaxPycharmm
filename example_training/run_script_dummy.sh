#!/bin/bash

#SBATCH --mail-user=ericdavid.boittier@unibas.ch
#SBATCH --job-name=run
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=3000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu23

hostname
python spice.py
