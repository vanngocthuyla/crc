#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/run_mse_sim.py"

export DATA_DIR="/home/vla/python/nls_drc/sim/"

export DATA="2b.normalized/Theta 2c.normalized_fixed_theta1/Theta 2d.normalized_fixed_theta2/Theta 2e.normalized_fixed_both/Theta 2f.normalized_control/Theta"

export OUT_DIR="/home/vla/python/nls_drc/sim/3.MSE"

export OUT="2b_norm 2c_norm_fixed_1 2d_norm_fixed_2 2e_norm_fixed_both 2f_norm_ctrl"

export N_BATCH=5

python $SCRIPT --data_dir $DATA_DIR --data_file "$DATA" --out_dir $OUT_DIR --normalized_data --out_files "$OUT" --n_batch $N_BATCH
