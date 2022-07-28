#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/submit_mcmc_ebselen.py"

export DATA_DIR="/home/vla/python/bdrc/ebselen/input"

export OUT_DIR="/home/vla/python/bdrc/ebselen/1h.norm_no_control_outlier"

python $SCRIPT --data_dir $DATA_DIR --out_dir $OUT_DIR --normalized_data --outlier_detection