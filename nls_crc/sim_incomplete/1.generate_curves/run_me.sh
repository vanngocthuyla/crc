#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/generate_incomplete_curves.py"

export OUT_DIR="/home/vla/python/nls_drc/sim_incomplete/1.generate_curves"

export N_SIM=1000

python $SCRIPT --out_dir $OUT_DIR --nsim $N_SIM
