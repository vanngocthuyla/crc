import pandas as pd
import warnings
import numpy as np
import sys
import os
import argparse

import pickle
import arviz as az

import jax
import jax.numpy as jnp
from jax import random
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import LogNormal, Normal, Uniform
from numpyro.infer import MCMC, NUTS

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

from _bdrc_model import bdrc_numpyro_fitting

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str, 				default="")
parser.add_argument( "--data_file",             type=str, 				default="")
parser.add_argument( "--out_dir",               type=str, 				default="")

parser.add_argument( "--fitting_with_control",  action="store_true", 	default=False)
parser.add_argument( "--fixed_Rb", 				action="store_true", 	default=False)
parser.add_argument( "--fixed_Rt", 				action="store_true", 	default=False)

parser.add_argument( "--niters",				type=int, 				default=20000)
parser.add_argument( "--nburn",                 type=int, 				default=5000)
parser.add_argument( "--nthin",                 type=int, 				default=1)
parser.add_argument( "--nchain",                type=int, 				default=4)
parser.add_argument( "--random_key",            type=int, 				default=0)

args = parser.parse_args()

from jax.config import config
config.update("jax_enable_x64", True)
numpyro.set_host_device_count(args.nchain)

print("Fitting model with control data:", args.fitting_with_control)
print("Fixing R_b:", args.fixed_Rb)
print("Fixing R_t:", args.fixed_Rt)
print("ninter:", args.niters)
print("nburn:", args.nburn)
print("nchain:", args.nchain)
print("nthin:", args.nthin)

print("Data file:", args.data_file)
try: 
	expt_dict = pickle.load( open(os.path.join(args.data_dir, args.data_file+'.pickle'), "rb") )
	expt = pd.DataFrame.from_dict(expt_dict)
except:
	expt = pd.read_csv(os.path.join(args.data_dir, args.data_file+'.csv'), index_col=0)

exper_names = list(expt.ID)
rng_key, rng_key_ = random.split(random.PRNGKey(args.random_key))

os.chdir(args.out_dir)

for i, name in enumerate(exper_names):
	if not os.path.exists(name): 
		os.mkdir(name)
	if not os.path.exists(os.path.join(args.out_dir, name, 'traces.pickle')):
		print("Running", name)
		if args.fitting_with_control: 
			bdrc_numpyro_fitting(rng_key_, expt.iloc[i], fitting_by_control=True,
		                         fixed_Rb=args.fixed_Rb, fixed_Rt=args.fixed_Rt,
		                         niters=args.niters, nburn=args.nburn, nchain=args.nchain, nthin=args.nthin,
		                         name=name, OUT_DIR=os.path.join(args.out_dir, name))
		else:
			bdrc_numpyro_fitting(rng_key_, expt.iloc[i], fitting_by_control=False, 
					     		 fixed_Rb=args.fixed_Rb, fixed_Rt=args.fixed_Rt, 
					     		 niters=args.niters, nburn=args.nburn, nchain=args.nchain, nthin=args.nthin,
					     		 name=name, OUT_DIR=os.path.join(args.out_dir, name))


if args.fixed_Rb and args.fixed_Rt:
	params = ["x_50", "H"]
elif args.fixed_Rb and (not args.fixed_Rt): 
	params = ["R_t", "x_50", "H"]
elif (not args.fixed_Rb) and args.fixed_Rt: 
	params = ["R_b", "x_50", "H"]
else:
	params = ["R_b", "R_t", "x_50", "H"]

TRACES_FILE = "traces.pickle"
mcmc_trace_files = [os.path.join(args.out_dir, name, TRACES_FILE) for name in exper_names]

os.chdir(args.out_dir)
theta_list = {}
for i, trace_file in enumerate(mcmc_trace_files):
	traces = pickle.load(open(trace_file, "rb"))
	theta_list[exper_names[i]] = np.array([np.mean(traces[x]) for x in params])

data = pd.DataFrame.from_dict(theta_list).T
data.columns = params
data.to_csv("Theta.csv")

print("DONE")
