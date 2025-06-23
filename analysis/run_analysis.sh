#!/bin/bash

# Usage: ./run_analysis.sh <restart_path> <files> <num_train> <num_valid> <natoms>

# Validate input arguments
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <restart_path> <files> <num_train> <num_valid> <natoms>"
    exit 1
fi

# Assign positional arguments to variables
RES=$1
FILES=$2
NUM_TRAIN=$3
NUM_VALID=$4
NATOMS=$5

# Execute the Python script
python ../physnetjax/analysis/model_analysis_utils.py \
  --restart "$RES" \
  --files "$FILES" \
  --num_train "$NUM_TRAIN" \
  --num_valid "$NUM_VALID" \
  --natoms "$NATOMS" \
  --load_test \
  --do_plot \
  --save_results