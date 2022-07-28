import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--data_dir",              type=str, 				default="")
parser.add_argument( "--out_dir",               type=str, 				default="")

parser.add_argument( "--normalized_data",       action="store_true",    default=False)
parser.add_argument( "--fitting_with_control",  action="store_true", 	default=False)
parser.add_argument( "--outlier_detection",     action="store_true",    default=False)

args = parser.parse_args()

running_script = "/home/vla/python/nls_drc/scripts/run_nls_massspec.py"

file_name = "ms"

if args.normalized_data:
    normalized_data = ''' --normalized_data '''
    file_name = file_name + "_norm"
else:
    normalized_data = ''' '''
    file_name = file_name + "_unnorm"

if args.fitting_with_control:
    fitting_with_control = ''' --fitting_with_control '''
    file_name = file_name + "_ctrol"
else:
    fitting_with_control = ''' '''

if args.outlier_detection:
    outlier_detection = ''' --outlier_detection '''
    file_name = file_name + "_outlier"
else:
    outlier_detection = ''' '''


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
    ''' --out_dir ''' + args.out_dir + \
    normalized_data + fitting_with_control + outlier_detection + \
    '''\ndate \n''' 

print("Submitting " + qsub_file)
open(qsub_file, "w").write(qsub_script)
os.system("qsub %s"%qsub_file)