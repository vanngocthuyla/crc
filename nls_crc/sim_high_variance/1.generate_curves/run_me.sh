#!/bin/bash

export SCRIPT="/home/vla/python/nls_drc/scripts/generate_curves.py"

export OUT_DIR="/home/vla/python/nls_drc/sim/1.generate_curves"

export N_SIM=5000

export VAR_NOISE=6

export VAR_CONTROL=12

python $SCRIPT --out_dir $OUT_DIR --nsim $N_SIM
