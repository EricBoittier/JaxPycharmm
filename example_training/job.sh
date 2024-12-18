#!/bin/bash

# SBATCH directives
#SBATCH --mail-user=ericdavid.boittier@unibas.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=abl         # Default job name
#SBATCH --ntasks=3              # Default number of tasks
#SBATCH --mem-per-cpu=3400    # Default memory per CPU in MB
#SBATCH --partition=gpu         # Default partition
#SBATCH --gres=gpu:1                 # Default GPU resources

hostname
export POLARS_SKIP_CPU_CHECK="true"
python test_ala_fullbatch_efa.py 
