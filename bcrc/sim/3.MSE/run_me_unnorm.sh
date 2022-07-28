#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/run_mse_sim.py"

export DATA_DIR="/home/vla/python/bdrc/sim/"

export DATA="2a.unnormalized_control/Theta"

export OUT="2a_unnorm_ctrl" 

export OUT_DIR="/home/vla/python/bdrc/sim/3.MSE"

export N_BATCH=5

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR --out_files "$OUT" --n_batch $N_BATCH