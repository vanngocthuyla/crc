#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/run_mse_sim.py"

export DATA_DIR="/home/vla/python/nls_drc/sim_incomplete/"

export DATA="2d.norm_incomplete_control/Theta 2e.norm_incomplete/Theta"

export OUT_DIR="/home/vla/python/nls_drc/sim_incomplete/3.MSE"

export OUT="2d_norm_inc_ctrl 2e_norm_inc"

export N_BATCH=5

python $SCRIPT --data_dir $DATA_DIR --data_file "$DATA" --out_dir $OUT_DIR --normalized_data --out_files "$OUT" --n_batch $N_BATCH
