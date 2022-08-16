import argparse
import pandas as pd
import warnings
import numpy as np
import sys
import os

import matplotlib
import matplotlib.pyplot as plt

from _mse import plot_MSE, calculating_MSE

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str,               default="")
parser.add_argument( "--data_files",            type=str,               default="")
parser.add_argument( "--out_dir",               type=str,               default="")
parser.add_argument( "--out_files",             type=str,               default="")
parser.add_argument( "--control_file",          type=str, 				default="")
parser.add_argument( "--normalized_data",       action="store_true",    default=False)
parser.add_argument( "--n_batch",       		type=int,    			default=1)

args = parser.parse_args()

theta_t = np.array([0, 100, -6.0759, 1.000])

data_files = args.data_files.split()
if len(args.out_files) > 0: 
	out_files = args.out_files.split()
else:
	out_files = data_files

for i, data in enumerate(data_files):
	print("Calculating and plotting MSE for", data)
	plot_MSE(theta_t, input_file=os.path.join(args.data_dir, data+'.csv'), 
			 control_file=args.control_file, normalized_data=args.normalized_data,
			 plot_name=out_files[i], OUT_DIR=args.out_dir)
	calculating_MSE(theta_t, input_file=os.path.join(args.data_dir, data+'.csv'), 
					control_file=args.control_file, normalized_data=args.normalized_data,
					n_batch=args.n_batch, output_file=out_files[i], OUT_DIR=args.out_dir)