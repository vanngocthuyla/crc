#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/submit_mcmc_fluorescence.py"

export DATA_DIR="/home/vla/python/bdrc/fluorescence/input"

export OUT_DIR="/home/vla/python/bdrc/fluorescence/1c.unnorm_control_outlier"

python $SCRIPT --data_dir $DATA_DIR --out_dir $OUT_DIR --fitting_with_control --outlier_detection
