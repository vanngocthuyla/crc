import pandas as pd
import numpy as np
import sys
import os
import argparse

import pickle

from _simulation import generate_drc_dataset

parser = argparse.ArgumentParser()
parser.add_argument( "--out_dir",                    type=str, 		default="")

parser.add_argument( "--nsim",                       type=int, 		default=100)
parser.add_argument( "--no_replication",             type=int, 		default=4)
parser.add_argument( "--var_noise",                  type=float, 	default=6)
parser.add_argument( "--var_control",                type=float, 	default=12)
parser.add_argument( "--size_of_control",            type=int, 		default=16)
parser.add_argument( "--seed",                       type=int, 		default=0)

args = parser.parse_args()

theta_t = np.array([0.227, 54.074, -6.0759, 1.000])
conc = np.array([5.12e-11, 2.56e-10, 1.28e-09, 6.40e-09, 3.20e-08, 1.60e-07, 8.00e-07, 4.00e-06, 2.00e-05, 1.00e-04])
logConc = np.log10(conc) 

pos_incomplete = [32, 33, 34, 35, 36, 37, 38, 39]

B = args.nsim
no_replication = args.no_replication
var_noise = args.var_noise
var_control = args.var_control
size_of_control = args.size_of_control
seed = args.seed

print("Number of experiments:", B)
print("Number of replication at each concentration:", no_replication)
print("Variance of data in curves:", var_noise)
print("Variance of control data:", var_control)
print("Number of controls:", size_of_control)
print("Seed of simulation:", seed)
print("Positions of missing data points:", pos_incomplete)

print("Generating complete curves:")
expt = generate_drc_dataset(B, logConc, theta_t, no_replication, var_noise, 
                            var_control, size_of_control, 
                            seed, scaling=False)


expt_incomplete = generate_drc_dataset(B, logConc, theta_t, no_replication, var_noise, 
                                       var_control, size_of_control,
                                       seed, scaling=False, pos_incomplete=pos_incomplete)


pickle.dump(expt.to_dict(), open(os.path.join(args.out_dir, 'drc_sim_unnorm_complete.pickle'), "wb"))
expt.to_csv(os.path.join(args.out_dir, 'drc_sim_unnorm_complete.csv'))

pickle.dump(expt_incomplete.to_dict(), open(os.path.join(args.out_dir, 'drc_sim_unnorm_incomplete.pickle'), "wb"))
expt_incomplete.to_csv(os.path.join(args.out_dir, 'drc_sim_unnorm_incomplete.csv'))

print("Generating incomplete curves:")
expt_1 = generate_drc_dataset(B, logConc, theta_t, no_replication, var_noise, 
                            var_control, size_of_control, 
                            seed, scaling=True)


expt_incomplete_1 = generate_drc_dataset(B, logConc, theta_t, no_replication, var_noise, 
                                       var_control, size_of_control,
                                       seed, scaling=True, pos_incomplete=pos_incomplete)


pickle.dump(expt_1.to_dict(), open(os.path.join(args.out_dir, 'drc_sim_norm_complete.pickle'), "wb"))
expt_1.to_csv(os.path.join(args.out_dir, 'drc_sim_norm_complete.csv'))

pickle.dump(expt_incomplete_1.to_dict(), open(os.path.join(args.out_dir, 'drc_sim_norm_incomplete.pickle'), "wb"))
expt_incomplete_1.to_csv(os.path.join(args.out_dir, 'drc_sim_norm_incomplete.csv'))

print("DONE")