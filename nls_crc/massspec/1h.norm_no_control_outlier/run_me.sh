#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/submit_nls_massspec.py"

export DATA_DIR="/home/vla/python/nls_drc/massspec/input"

export OUT_DIR="/home/vla/python/nls_drc/massspec/1h.norm_no_control_outlier"

python $SCRIPT --data_dir $DATA_DIR --out_dir $OUT_DIR --normalized_data --outlier_detection