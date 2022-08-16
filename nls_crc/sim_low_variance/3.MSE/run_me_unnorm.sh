#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/run_mse_sim.py"

export DATA_DIR="/home/vla/python/nls_drc/sim/"

export DATA="2a.unnormalized_control/Theta"

export OUT_DIR="/home/vla/python/nls_drc/sim/3.MSE"

export OUT='2a_unnorm_ctrl'

export CONTROL='/home/vla/python/nls_drc/sim/1.generate_curves/drc_sim_unnorm'

export N_BATCH=5

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR --out_files "$OUT" --control_file $CONTROL --n_batch $N_BATCH
