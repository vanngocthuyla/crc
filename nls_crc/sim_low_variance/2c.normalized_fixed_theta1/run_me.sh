#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/submit_nls_sim.py"

export DATA_DIR="/home/vla/python/nls_drc/sim/1.generate_curves"

export DATA="drc_sim_norm" #"nls_drc_sim_unnorm"

export OUT_DIR="/home/vla/python/nls_drc/sim/2c.normalized_fixed_theta1"

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR --fixed_R_b