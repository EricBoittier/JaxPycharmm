#!/bin/bash

python model_analysis_utils.py \
  --restart /pchem-data/meuwly/boittier/home/pycharmm_test/ckpts/cf3all-e94f959c-5f22-4b8a-800e-dd6197c07b20 \
  --files /pchem-data/meuwly/boittier/home/jaxeq/notebooks/ala-esp-dip-0.npz \
  --num_train 8000 \
  --num_valid 1786 \
  --natoms 37 \
  --load_test \
  --do_plot \
  --save_results