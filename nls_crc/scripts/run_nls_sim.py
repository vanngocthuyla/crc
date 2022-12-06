import sys
import os
import pandas as pd
import warnings
import itertools
import numpy as np
import argparse
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _nls_model import parameter_estimation, parameter_estimation_control, parameter_estimation_fixed_thetas

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str, 				default="")
parser.add_argument( "--data_file",             type=str, 				default="")
parser.add_argument( "--out_dir",               type=str, 				default="")

parser.add_argument( "--fitting_with_control",  action="store_true", 	default=False)
parser.add_argument( "--fixed_R_b", 			action="store_true", 	default=False)
parser.add_argument( "--fixed_R_t", 			action="store_true", 	default=False)

# parser.add_argument( "--fitting_replication",   type=int,               default=1)

args = parser.parse_args()

# assert args.fitting_replication > 0, print("Number of replication for fitting need to be larger than 0.")

print("Fitting model with control data:", args.fitting_with_control)
print("Fixing theta1:", args.fixed_R_b)
print("Fixing theta2:", args.fixed_R_t)

print("Data file:", args.data_file)

try: 
	expt_dict = pickle.load( open(os.path.join(args.data_dir, args.data_file+'.pickle'), "rb") )
	expt = pd.DataFrame.from_dict(expt_dict)
except:
	expt = pd.read_csv(os.path.join(args.data_dir, args.data_file+'.csv'), index_col=0)

exper_names = list(expt.ID)
os.chdir(args.out_dir)

B = len(exper_names)

if args.fixed_R_b and args.fixed_R_t:
	params = ["theta3", "theta4"]
elif args.fixed_R_b and (not args.fixed_R_t): 
	params = ["theta2", "theta3", "theta4"]
elif (not args.fixed_R_b) and args.fixed_R_t: 
	params = ["theta1", "theta3", "theta4"]
else:
	params = ["theta1", "theta2", "theta3", "theta4"]

# for count in range(args.fitting_replication):

Theta_matrix = np.zeros(len(params)*B).reshape((len(params), B))

for i, name in enumerate(exper_names):
	if args.fitting_with_control: 
		Theta_matrix[:,i] = parameter_estimation_control(expt.iloc[i])[0]
	else:
		Theta_matrix[:,i] = parameter_estimation_fixed_thetas(expt.iloc[i], fixed_R_b=args.fixed_R_b, fixed_R_t=args.fixed_R_t)[0]

pd.DataFrame(Theta_matrix, index=params).T.to_csv("Theta.csv")

# pd.DataFrame(Theta_matrix, index=params).T.to_csv("Theta_"+str(count)+".csv")

print("DONE")
