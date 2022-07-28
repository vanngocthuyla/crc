#!/bin/bash

export SCRIPT="/home/vla/python/bdrc/scripts/generate_incomplete_curves.py"

export OUT_DIR="/home/vla/python/bdrc/sim_incomplete/1.generate_curves"

export N_SIM=50

python $SCRIPT --out_dir $OUT_DIR --nsim $N_SIM
