import os
import glob
import argparse

import numpy as np

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

running_script = "/home/vla/python/nls_drc/scripts/run_nls_sim.py"

file_name = "drc"

if args.fitting_with_control:
    fitting_with_control = ''' --fitting_with_control '''
    file_name = file_name + "_ctrol"
else:
    fitting_with_control = ''' '''

if args.fixed_R_b:
    fixed_R_b = ''' --fixed_R_b '''
    file_name = file_name + "_t1"
else:
    fixed_R_b = ''' '''

if args.fixed_R_t:
    fixed_R_t = ''' --fixed_R_t '''
    file_name = file_name + "_t2"
else:
    fixed_R_t = ''' '''

qsub_file = os.path.join(args.out_dir, file_name+".job")
log_file  = os.path.join(args.out_dir, file_name+".log")

qsub_script = '''#!/bin/bash
#PBS -S /bin/bash
#PBS -o %s '''%log_file + '''
#PBS -j oe
#PBS -l nodes=1:ppn=1,mem=2048mb,walltime=72:00:00

module load miniconda/3
source activate bitc_race
cd ''' + args.out_dir + '''\n''' + \
    '''date\n''' + \
    '''python ''' + running_script + \
    ''' --data_dir ''' + args.data_dir + \
    ''' --data_file ''' + args.data_file + \
    ''' --out_dir ''' + args.out_dir + \
    fitting_with_control + fixed_R_b + fixed_R_t + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
os.system("qsub %s"%qsub_file)