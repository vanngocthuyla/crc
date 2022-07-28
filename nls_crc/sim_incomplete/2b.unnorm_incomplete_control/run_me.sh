#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/submit_nls_sim.py"

export DATA_DIR="/home/vla/python/nls_drc/sim_incomplete/1.generate_curves"

export DATA="drc_sim_unnorm_incomplete"

export OUT_DIR="/home/vla/python/nls_drc/sim_incomplete/2b.unnorm_incomplete_control"

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR --fitting_with_control