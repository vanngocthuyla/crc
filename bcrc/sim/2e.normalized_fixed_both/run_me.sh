#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/submit_mcmc_sim.py"

export DATA_DIR="/home/vla/python/bdrc/sim/1.generate_curves"

export DATA="bdrc_sim_norm" #"bdrc_sim_unnorm"

export OUT_DIR="/home/vla/python/bdrc/sim/2e.normalized_fixed_both"

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR --fixed_Rb --fixed_Rt