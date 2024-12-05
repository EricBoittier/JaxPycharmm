#!/bin/bash

# SBATCH directives
#SBATCH --mail-user=ericdavid.boittier@unibas.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=${job_name:-run}           # Default job name
#SBATCH --ntasks=4              # Default number of tasks
#SBATCH --mem-per-cpu=4000    # Default memory per CPU in MB
#SBATCH --partition=gpu         # Default partition
#SBATCH --gres=gpu:1                 # Default GPU resources

# Load the hostname for debugging purposes
hostname

# Environment Variables with Defaults
PY=${PY:-/pchem-data/meuwly/boittier/home/miniforge3/envs/jaxphyscharmm/bin/python}
DATA=${data:-/pchem-data/meuwly/boittier/home/ini.to.dioxi.npz}
NAME=${name:-test1}
NTRAIN=${ntrain:-6500}
NVALID=${nvalid:-1000}
RESTART=${restart:-}  # Leave blank if no restart file is provided
BATCH_SIZE=${batch_size:-50}
NEPOCHS=${nepochs:-1000000}
FORCES_W=${forces_w:-52.91}
CHARGE_W=${charge_w:-27.21}
TOTAL_CHARGE=${total_chg:-0.0}

# Additional arguments with defaults
FEATURES=${features:-128}
MAX_DEGREE=${max_degree:-1}
NUM_ITERATIONS=${num_iterations:-2}
NUM_BASIS_FUNCTIONS=${num_basis_functions:-32}
CUTOFF=${cutoff:-10.0}
MAX_ATOMIC_NUMBER=${max_atomic_number:-9}
N_RES=${n_res:-1}
DEBUG=${debug:-False}
natoms=${natoms:-8}

SCHEDULE=${schedule:-warmup}

# Construct conditional arguments
RESTART_ARG=""
if [ -n "$RESTART" ]; then
    RESTART_ARG="--restart $RESTART"
fi

DEBUG_ARG=""
if [ "$DEBUG" = "True" ]; then
    DEBUG_ARG="--debug"
fi

ADDITIONAL_ARGS="--features $FEATURES --max_degree $MAX_DEGREE --num_iterations $NUM_ITERATIONS \
--num_basis_functions $NUM_BASIS_FUNCTIONS --cutoff $CUTOFF --max_atomic_number $MAX_ATOMIC_NUMBER \
--n_res $N_RES $DEBUG_ARG"

# Logging parameters for debugging
echo "Job Name: $NAME"
echo "Python Path: $PY"
echo "Data Path: $DATA"
echo "Restart File: $RESTART"
echo "Batch Size: $BATCH_SIZE"
echo "Number of Epochs: $NEPOCHS"
echo "Forces Weight: $FORCES_W"
echo "Charge Weight: $CHARGE_W"
echo "Features: $FEATURES"
echo "Max Degree: $MAX_DEGREE"
echo "Number of Iterations: $NUM_ITERATIONS"
echo "Number of Basis Functions: $NUM_BASIS_FUNCTIONS"
echo "Cutoff: $CUTOFF"
echo "Max Atomic Number: $MAX_ATOMIC_NUMBER"
echo "Number of Residuals: $N_RES"
echo "Debug Mode: $DEBUG"
echo "N atoms: $natoms"
echo "Total charge: $TOTAL_CHARGE"
# Construct and run the command
COMMAND="$PY ../physnetjax/api.py \
    --data $DATA \
    --name $NAME \
    --ntrain $NTRAIN \
    --nvalid $NVALID \
    $RESTART_ARG \
    --batch_size $BATCH_SIZE \
    --nepochs $NEPOCHS \
    --natoms $natoms \
    --total_charge $TOTAL_CHARGE \
    --forces_w $FORCES_W \
    --charges_w $CHARGE_W \
    --schedule $SCHEDULE \
    $ADDITIONAL_ARGS"

echo "Running Command: $COMMAND"
eval $COMMAND
