#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/run_mse_sim_incomplete.py"

export DATA_DIR="/home/vla/python/nls_drc/sim_incomplete/"

export DATA="2a.unnorm_complete_control/Theta 2b.unnorm_incomplete_control/Theta 2c.unnorm_incomplete/Theta"

export OUT_DIR="/home/vla/python/nls_drc/sim_incomplete/3.MSE"

export OUT="2a_unnorm_com_ctrl 2b_unnorm_inc_ctrl 2c_unnorm_inc"

export N_BATCH=5

python $SCRIPT --data_dir $DATA_DIR --data_file "$DATA" --out_dir $OUT_DIR --out_files "$OUT" --n_batch $N_BATCH
