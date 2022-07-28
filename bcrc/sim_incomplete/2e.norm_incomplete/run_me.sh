#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/submit_mcmc_sim.py"

export DATA_DIR="/home/vla/python/bdrc/sim_incomplete/1.generate_curves"

export DATA="bdrc_sim_norm_incomplete"

export OUT_DIR="/home/vla/python/bdrc/sim_incomplete/2e.norm_incomplete"

python $SCRIPT --data_dir $DATA_DIR --data_file $DATA --out_dir $OUT_DIR